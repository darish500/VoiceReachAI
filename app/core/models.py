"""
app/core/models.py

Core domain models for VoiceReach AI.

This module defines every data structure that flows through the AI Brain:
- Enums that constrain the vocabulary the LLM and the orchestrator use
  (language, intent, risk level, conversation stage).
- Message and context primitives that describe a single turn and the caller.
- Knowledge primitives used by the RAG layer.
- The SituationAssessment model, which is the strict JSON contract returned
  by the first ("assessment") LLM call.
- ConversationState, the single source of truth for a session's memory.
  This is maintained entirely in Python -- the LLM is never trusted to
  remember anything across turns.
- AIResponse, the structured JSON contract returned to the calling service
  (telephony / STT / TTS layer) at the end of the pipeline.

All models use Pydantic v2. Enums are used everywhere a field has a closed
set of valid values so that invalid LLM output fails validation loudly
instead of silently propagating through the system.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Helper for consistent UTC timestamps across all models.
# ---------------------------------------------------------------------------
def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Centralizing this avoids naive-vs-aware datetime bugs and makes every
    model's default timestamp behavior consistent and testable.
    """
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Language(str, Enum):
    """Languages supported by VoiceReach AI.

    Kept as an explicit closed set (rather than a free-text string) so that
    downstream services (TTS, telephony routing) can safely switch on this
    value without needing to validate arbitrary strings.
    """

    ENGLISH = "english"
    HAUSA = "hausa"
    YORUBA = "yoruba"
    IGBO = "igbo"
    PIDGIN = "pidgin"
    UNKNOWN = "unknown"


class Intent(str, Enum):
    """The set of user intents the Situation Assessment step can classify.

    This list is intentionally scoped to the kinds of requests VoiceReach AI
    is designed to handle (health/community information use case). New
    intents should be added here deliberately -- the orchestrator and
    dialogue manager both branch on this enum.
    """

    GREETING = "greeting"
    SYMPTOM_CHECK = "symptom_check"
    HEALTH_INFORMATION = "health_information"
    APPOINTMENT_REQUEST = "appointment_request"
    MEDICATION_QUESTION = "medication_question"
    EMERGENCY = "emergency"
    FEEDBACK = "feedback"
    GENERAL_QUESTION = "general_question"
    CLARIFICATION = "clarification"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Assessed risk level of the current conversational turn.

    Drives safety-gating behavior in the SafetyValidator: HIGH/CRITICAL
    turns bypass normal response generation and are routed to an emergency
    response path.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConversationStage(str, Enum):
    """Where the conversation currently sits in the overall dialogue flow.

    The DialogueManager uses this, together with Intent and missing
    entities, to decide the next_action (e.g. keep collecting information
    vs. move to retrieval vs. close out the conversation).
    """

    GREETING = "greeting"
    COLLECT_INFORMATION = "collect_information"
    CLARIFYING = "clarifying"
    RETRIEVING_KNOWLEDGE = "retrieving_knowledge"
    PROVIDING_ANSWER = "providing_answer"
    EMERGENCY_ESCALATION = "emergency_escalation"
    CLOSING = "closing"


class NextAction(str, Enum):
    """Directive produced by Situation Assessment telling the
    DialogueManager what to do next.

    This decouples "what did the LLM understand" (SituationAssessment)
    from "what should the system do about it" (DialogueManager), which is
    resolved deterministically in Python using this enum plus the current
    ConversationState.
    """

    ASK_CLARIFYING_QUESTION = "ask_clarifying_question"
    COLLECT_MISSING_ENTITY = "collect_missing_entity"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    ESCALATE_EMERGENCY = "escalate_emergency"
    END_CONVERSATION = "end_conversation"
    CONTINUE = "continue"


class Sender(str, Enum):
    """Who authored a given Message in the conversation history."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Message / context primitives
# ---------------------------------------------------------------------------
class Message(BaseModel):
    """A single turn in the conversation history.

    Stored verbatim in ConversationState.history so the system has a full,
    auditable transcript independent of anything the LLM claims to
    remember.
    """

    sender: Sender = Field(..., description="Who produced this message.")
    text: str = Field(..., min_length=1, description="Raw message text.")
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC time the message was recorded.",
    )
    language: Optional[Language] = Field(
        default=None, description="Detected/assumed language of this message."
    )


class UserContext(BaseModel):
    """Caller-supplied or inferred context about the user.

    This is distinct from conversation history: it holds durable facts
    about the person (or the person they are calling about) that persist
    and get enriched turn over turn, e.g. age of a sick child, location.
    """

    caller_id: Optional[str] = Field(
        default=None, description="Opaque identifier for the caller, e.g. phone hash."
    )
    location: Optional[str] = Field(
        default=None, description="Self-reported or inferred location (state/LGA)."
    )
    age: Optional[int] = Field(
        default=None, ge=0, le=130, description="Age of the subject, if known."
    )
    is_subject_self: Optional[bool] = Field(
        default=None,
        description="True if the caller is describing their own situation, "
        "False if describing someone else (e.g. a child).",
    )
    preferred_language: Optional[Language] = Field(
        default=None, description="User's preferred language, if established."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form bag for additional context fields that don't "
        "warrant a first-class field yet.",
    )


# ---------------------------------------------------------------------------
# Knowledge / RAG primitives
# ---------------------------------------------------------------------------
class KnowledgeChunk(BaseModel):
    """A single retrievable unit of knowledge stored in ChromaDB.

    Produced by the ingestion pipeline and returned by KnowledgeRetriever
    queries. `content` is the only field ever allowed to be surfaced to the
    user (via the Response Generator) -- SafetyValidator enforces that
    responses are grounded in the `content` of retrieved chunks.
    """

    id: str = Field(..., description="Stable unique identifier for this chunk.")
    title: str = Field(..., description="Short human-readable title.")
    category: str = Field(
        ..., description="Topical category, e.g. 'maternal_health', 'malaria'."
    )
    content: str = Field(..., min_length=1, description="The actual knowledge text.")
    source: str = Field(
        ..., description="Provenance of this knowledge, e.g. WHO/NCDC guideline name."
    )
    tags: list[str] = Field(
        default_factory=list, description="Free-text tags for filtering/search."
    )
    score: Optional[float] = Field(
        default=None,
        description="Similarity score assigned by the retriever at query time "
        "(not persisted, populated only on retrieval).",
    )


# ---------------------------------------------------------------------------
# Situation Assessment -- strict contract for the first LLM call
# ---------------------------------------------------------------------------
class SituationAssessment(BaseModel):
    """Structured output of the first ("assessment") LLM call.

    This is deliberately the ONLY thing the assessment LLM call is allowed
    to produce. It never generates user-facing text -- it classifies the
    turn so the DialogueManager can decide what happens next in Python.
    The LLM is instructed to return JSON matching exactly this shape; this
    model is used to validate that output before it is trusted.
    """

    language: Language = Field(..., description="Detected language of the user's message.")
    intent: Intent = Field(..., description="Classified intent of the user's message.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model's confidence in the intent classification."
    )
    risk_level: RiskLevel = Field(
        ..., description="Assessed risk level based on the message content."
    )
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Entities extracted from the message, e.g. {'symptom': 'fever'}.",
    )
    missing_entities: list[str] = Field(
        default_factory=list,
        description="Entities the dialogue still needs before it can proceed, "
        "e.g. ['age', 'duration'].",
    )
    next_action: NextAction = Field(
        ..., description="Recommended next step for the DialogueManager."
    )
    reasoning: str = Field(
        ..., description="Brief model-provided rationale, retained for logging/audit only. "
        "Never shown to the end user."
    )

    @field_validator("confidence")
    @classmethod
    def _round_confidence(cls, value: float) -> float:
        """Normalize confidence to 4 decimal places for stable logging/storage."""
        return round(value, 4)


# ---------------------------------------------------------------------------
# Conversation state -- the system's memory, owned entirely by Python
# ---------------------------------------------------------------------------
class ConversationState(BaseModel):
    """Full memory of a single session.

    This is the single source of truth for "what do we know about this
    conversation so far." It is loaded by the ConversationStore at the
    start of every request, mutated by the Orchestrator as the pipeline
    runs, and persisted back at the end of the request. The LLM never
    owns this state -- it only ever sees a serialized view of relevant
    parts of it as prompt context.
    """

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this conversation session.",
    )
    language: Language = Field(
        default=Language.UNKNOWN, description="Current established language of the session."
    )
    stage: ConversationStage = Field(
        default=ConversationStage.GREETING, description="Current stage of the dialogue."
    )
    intent: Intent = Field(
        default=Intent.UNKNOWN, description="Most recently classified intent."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence of the most recent assessment."
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.NONE, description="Most recently assessed risk level."
    )
    goal: Optional[str] = Field(
        default=None,
        description="Short description of what the user is trying to accomplish, "
        "set once intent stabilizes.",
    )
    user_context: UserContext = Field(
        default_factory=UserContext, description="Durable facts known about the user."
    )
    entities: dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated entities collected across the whole conversation "
        "(merged turn over turn, not just the latest turn).",
    )
    missing_entities: list[str] = Field(
        default_factory=list,
        description="Entities still required before the dialogue can move to "
        "knowledge retrieval / final answer.",
    )
    retrieved_document_ids: list[str] = Field(
        default_factory=list,
        description="IDs of KnowledgeChunks retrieved and used so far this session, "
        "kept for grounding/audit and to avoid redundant retrieval.",
    )
    history: list[Message] = Field(
        default_factory=list, description="Full transcript of the conversation so far."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form operational metadata, e.g. turn count, timing info, "
        "feature flags active for this session.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="When this session was first created."
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="When this session was last updated."
    )

    def touch(self) -> None:
        """Update the `updated_at` timestamp.

        Called by the ConversationStore/Orchestrator whenever the state is
        mutated and persisted, so callers don't have to remember to do it
        inline at every mutation site.
        """
        self.updated_at = _utcnow()


# ---------------------------------------------------------------------------
# Final structured response returned by the API
# ---------------------------------------------------------------------------
class AIResponse(BaseModel):
    """The structured JSON contract returned by POST /chat.

    This is the only thing the calling service (telephony/STT/TTS layer)
    ever sees. It intentionally exposes just enough state (intent, stage,
    risk_level, confidence) for the caller to make routing decisions,
    without leaking internal reasoning or raw retrieved knowledge content.
    """

    message: str = Field(..., description="The natural-language text to speak/display to the user.")
    intent: Intent = Field(..., description="Final classified intent for this turn.")
    stage: ConversationStage = Field(..., description="Conversation stage after this turn.")
    risk_level: RiskLevel = Field(..., description="Risk level associated with this turn.")
    continue_conversation: bool = Field(
        ..., description="Whether the calling service should keep the call/session open."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence associated with the assessment this turn."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="IDs (not full content) of KnowledgeChunks that grounded this response, "
        "for audit/transparency purposes.",
    )