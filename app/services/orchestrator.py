"""
app/services/orchestrator.py

The Orchestrator is the brain of VoiceReach AI. Every incoming message
passes through exactly one Orchestrator method, which coordinates every
other service in the pipeline described in the architecture diagram:

    Incoming Text
      -> Conversation Store   (load)
      -> Orchestrator          <-- this module
           -> Situation Assessment (LLM)
           -> Dialogue Manager      (inlined here -- see note below)
           -> Knowledge Retrieval   (RAG)
           -> Safety Validation
           -> Response Generator    (LLM)
      -> Structured JSON Response

Dialogue Manager note:
    A standalone DialogueManager module has not been built yet. The
    branching logic it would own (emergency vs. clarify vs. retrieve) is
    currently inlined in `_decide_next_step`. This method is intentionally
    small and isolated so that logic can be lifted into a real
    DialogueManager class later without changing the Orchestrator's
    public contract (`handle_message`) at all.

Dependency inversion:
    The Orchestrator depends on abstractions, not concrete classes:
      - ConversationStore (already defined in app.services.conversation_store)
      - LLMService (already defined in app.services.llm_service)
      - KnowledgeRetriever / SafetyValidator (defined as typing.Protocol
        below, since app/services/knowledge.py and app/services/safety.py
        have not been generated yet). Once those modules exist, their
        concrete classes only need to implement the same method
        signatures -- Python's structural typing means no changes to this
        file will be required, and no import of those modules is needed
        here either.

Error handling:
    LLMService already translates all OpenAI failures into
    LLMServiceError subclasses. The Orchestrator catches those at the two
    points where the LLM is called and degrades gracefully to a safe,
    generic fallback response rather than letting an exception surface to
    the API layer and drop the caller's phone call.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.core.models import (
    AIResponse,
    ConversationStage,
    ConversationState,
    Intent,
    KnowledgeChunk,
    Message,
    RiskLevel,
    Sender,
    SituationAssessment,
)
from app.services.conversation_store import ConversationStore
from app.services.llm_service import LLMService, LLMServiceError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for collaborators that don't have concrete implementations yet.
#
# These describe exactly the method shape the Orchestrator needs.
# app/services/knowledge.py's KnowledgeRetriever and app/services/safety.py's
# SafetyValidator should each implement the corresponding protocol below;
# neither needs to inherit from it explicitly (structural typing).
# ---------------------------------------------------------------------------
class KnowledgeRetriever(Protocol):
    """Contract for the RAG layer's retrieval interface."""

    def retrieve(self, query: str, top_k: int = 3) -> list[KnowledgeChunk]:
        """Return up to `top_k` KnowledgeChunks relevant to `query`."""
        ...


class SafetyValidator(Protocol):
    """Contract for the safety-validation layer."""

    def validate(
        self,
        response_text: str,
        state: ConversationState,
        assessment: SituationAssessment,
        knowledge_chunks: list[KnowledgeChunk],
    ) -> str:
        """Validate/sanitize a generated reply before it reaches the user.

        Implementations may adjust wording (e.g. strip an unsupported
        claim) but must return a string safe to speak to the caller as-is.
        """
        ...


# ---------------------------------------------------------------------------
# Internal result of dialogue branching (steps 5/6/7 in the spec).
# Bundling this into one small class -- rather than returning a bare tuple
# or duplicating if/else branches inline in handle_message -- makes the
# branching decision itself a single, independently testable unit.
# ---------------------------------------------------------------------------
class _DialogueDecision:
    """The outcome of the minimal inlined Dialogue Manager step.

    Bundles the new conversation stage together with whether knowledge
    retrieval should run this turn, so `handle_message` has one clear
    branch point instead of duplicating this logic.
    """

    __slots__ = ("stage", "should_retrieve_knowledge")

    def __init__(self, stage: ConversationStage, should_retrieve_knowledge: bool) -> None:
        self.stage = stage
        self.should_retrieve_knowledge = should_retrieve_knowledge


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """Coordinates every service to turn raw user text into an AIResponse.

    This class owns no business logic of its own beyond sequencing and
    branching -- classification lives in LLMService/prompts, retrieval
    lives in KnowledgeRetriever, safety lives in SafetyValidator, and
    persistence lives in ConversationStore. The Orchestrator's job is
    purely to call each of those, in the right order, with the right
    data, and to handle the failure modes at the two points that touch
    the network (the two LLM calls).
    """

    def __init__(
        self,
        conversation_store: ConversationStore,
        llm_service: LLMService,
        knowledge_retriever: KnowledgeRetriever,
        safety_validator: SafetyValidator,
        default_retrieval_top_k: int = 3,
    ) -> None:
        """
        Args:
            conversation_store: Loads/saves ConversationState.
            llm_service: The sole gateway to the OpenAI API.
            knowledge_retriever: RAG layer used to ground non-emergency,
                non-clarifying replies.
            safety_validator: Final safety/grounding check run on every
                generated reply before it is returned to the caller.
            default_retrieval_top_k: How many KnowledgeChunks to request
                per retrieval call.
        """
        self.conversation_store = conversation_store
        self.llm_service = llm_service
        self.knowledge_retriever = knowledge_retriever
        self.safety_validator = safety_validator
        self.default_retrieval_top_k = default_retrieval_top_k

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------
    def handle_message(self, session_id: str, user_message: str) -> AIResponse:
        """Run the full pipeline for one incoming user message.

        Steps (matching the architecture spec exactly):
            1. Load conversation state.
            2. Store the user's message.
            3. Call assess_situation().
            4. Update conversation state with the assessment.
            5. If risk is HIGH/CRITICAL -> emergency path.
            6. If information is missing -> clarifying question path.
            7. Otherwise -> retrieve knowledge.
            8. Call generate_response().
            9. Run the safety validator.
            10. Save the updated state.
            11. Return the final AIResponse.

        Args:
            session_id: Identifies which ConversationState to load/save.
            user_message: Raw text for this turn (already transcribed
                upstream by the STT service -- out of scope here).

        Returns:
            A fully populated AIResponse. This method never raises for
            LLM-related failures -- it degrades to a safe fallback reply
            instead, so a single bad turn never drops the caller's call.
        """
        # 1. Load conversation state.
        state = self.conversation_store.get_or_create(session_id)

        # 2. Store the user's message.
        state = self._record_message(state, Sender.USER, user_message)

        # 3. Call assess_situation().
        try:
            assessment = self.llm_service.assess_situation(state, user_message)
        except LLMServiceError as exc:
            logger.error(
                "Situation Assessment failed for session_id=%s: %s", session_id, exc
            )
            return self._fallback_response(state)

        # 4. Update conversation state with the assessment.
        state = self._merge_assessment(state, assessment)

        # 5/6/7. Decide the dialogue branch and retrieve knowledge if needed.
        decision = self._decide_next_step(assessment)
        state.stage = decision.stage

        knowledge_chunks: list[KnowledgeChunk] = []
        if decision.should_retrieve_knowledge:
            knowledge_chunks = self._retrieve_knowledge(user_message, assessment)
            if knowledge_chunks:
                known_ids = set(state.retrieved_document_ids)
                known_ids.update(chunk.id for chunk in knowledge_chunks)
                state.retrieved_document_ids = sorted(known_ids)
            state.stage = ConversationStage.PROVIDING_ANSWER

        # 8. Call generate_response().
        try:
            raw_reply = self.llm_service.generate_response(
                state, knowledge_chunks, assessment
            )
        except LLMServiceError as exc:
            logger.error(
                "Response Generation failed for session_id=%s: %s", session_id, exc
            )
            return self._fallback_response(state)

        # 9. Run the safety validator.
        safe_reply = self.safety_validator.validate(
            response_text=raw_reply,
            state=state,
            assessment=assessment,
            knowledge_chunks=knowledge_chunks,
        )

        # Close out the conversation stage/continuation flag for GOODBYE
        # turns before persisting, so state and response agree.
        continue_conversation = True
        if assessment.intent == Intent.GOODBYE:
            state.stage = ConversationStage.CLOSING
            continue_conversation = False

        # Record the assistant's reply in history for audit/context.
        state = self._record_message(state, Sender.ASSISTANT, safe_reply)

        # 10. Save the updated state.
        self.conversation_store.save(state)

        # 11. Return the final AIResponse.
        return AIResponse(
            message=safe_reply,
            intent=assessment.intent,
            stage=state.stage,
            risk_level=assessment.risk_level,
            continue_conversation=continue_conversation,
            confidence=assessment.confidence,
            sources=[chunk.id for chunk in knowledge_chunks],
        )

    # -----------------------------------------------------------------
    # Step helpers (each isolated so it's independently testable)
    # -----------------------------------------------------------------
    @staticmethod
    def _record_message(
        state: ConversationState, sender: Sender, text: str
    ) -> ConversationState:
        """Append a Message to state.history. Pure state mutation, no I/O."""
        state.history.append(Message(sender=sender, text=text))
        return state

    @staticmethod
    def _merge_assessment(
        state: ConversationState, assessment: SituationAssessment
    ) -> ConversationState:
        """Fold a SituationAssessment into the running ConversationState.

        - Language is only overwritten once the assessment is no longer
          "unknown", so an ambiguous turn doesn't erase a previously
          established language.
        - Entities are merged (not replaced), so facts learned in earlier
          turns are preserved unless this turn's assessment overwrites a
          specific key.
        - missing_entities reflects this turn's assessment as-is -- it is
          meant to describe what's still needed *right now*.
        """
        if assessment.language.value != "unknown":
            state.language = assessment.language
        state.intent = assessment.intent
        state.confidence = assessment.confidence
        state.risk_level = assessment.risk_level
        state.entities.update(assessment.entities)
        state.missing_entities = list(assessment.missing_entities)
        return state

    @staticmethod
    def _decide_next_step(assessment: SituationAssessment) -> _DialogueDecision:
        """Branch on risk/missing-entities to pick the next stage.

        This is the minimal stand-in for a DialogueManager described in
        the module docstring. Priority order matters: emergency always
        wins over "missing information", since a caller in a medical or
        financial emergency should never be stuck answering follow-up
        questions.
        """
        if assessment.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            # 5. Emergency path: skip retrieval entirely. The Response
            # Generation prompt's EMERGENCY RULE handles the wording; no
            # knowledge grounding is needed (or safe to wait for) here.
            return _DialogueDecision(
                stage=ConversationStage.EMERGENCY_ESCALATION,
                should_retrieve_knowledge=False,
            )

        if assessment.missing_entities:
            # 6. Clarifying-question path: also skip retrieval, since we
            # don't yet have enough information to know what to retrieve.
            return _DialogueDecision(
                stage=ConversationStage.COLLECT_INFORMATION,
                should_retrieve_knowledge=False,
            )

        # 7. Otherwise: retrieve knowledge before generating the answer.
        return _DialogueDecision(
            stage=ConversationStage.RETRIEVING_KNOWLEDGE,
            should_retrieve_knowledge=True,
        )

    def _retrieve_knowledge(
        self, user_message: str, assessment: SituationAssessment
    ) -> list[KnowledgeChunk]:
        """Query the RAG layer, degrading to no results on failure.

        Retrieval failures must never take down the whole turn -- the
        Response Generation prompt already knows how to handle an empty
        knowledge block ("say so clearly rather than guessing"), so an
        empty list here is a safe, well-defined fallback rather than a
        special case the caller needs to worry about.
        """
        try:
            return self.knowledge_retriever.retrieve(
                query=user_message, top_k=self.default_retrieval_top_k
            )
        except Exception as exc:  # noqa: BLE001 -- retrieval backends can
            # raise many different error types (Chroma, embedding model,
            # I/O); all of them degrade to "no knowledge" rather than
            # failing the turn.
            logger.warning(
                "Knowledge retrieval failed for intent=%s: %s",
                assessment.intent,
                exc,
            )
            return []

    def _fallback_response(self, state: ConversationState) -> AIResponse:
        """Build a safe, generic AIResponse when an LLM call fails outright.

        Used when either LLM call raises LLMServiceError. Keeps the
        conversation open (continue_conversation=True) so the caller can
        simply try again, and persists the state as-is (including the
        user's message that was already recorded) so context isn't lost
        for the retry.
        """
        fallback_message = (
            "Sorry, something went wrong on our end. Please say that again."
        )
        state = self._record_message(state, Sender.ASSISTANT, fallback_message)
        self.conversation_store.save(state)

        return AIResponse(
            message=fallback_message,
            intent=state.intent,
            stage=state.stage,
            risk_level=state.risk_level,
            continue_conversation=True,
            confidence=0.0,
            sources=[],
        )