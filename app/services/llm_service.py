"""
app/services/llm_service.py

LLMService is the sole gateway to the LLM provider. Every other module
(Orchestrator, and transitively the whole pipeline) talks to the LLM only
through this class's two public methods:

    assess_situation()   -> SituationAssessment (Call 1, strict JSON)
    generate_response()  -> str                 (Call 2, free-text reply)

Provider note:
    This uses the `openai` Python SDK as a thin transport client, but is
    pointed at Google Gemini's OpenAI-compatible endpoint via `base_url`
    (see app/core/config.py -- `llm_base_url`). Because Gemini implements
    the same `/chat/completions` wire protocol, no other code in this
    module (or anywhere downstream) needs to know or care which provider
    is actually being called -- swapping `base_url` and `api_key` is
    enough to point this at a different OpenAI-compatible provider later.

Design notes:
    - Prompt construction is entirely delegated to app.prompts.prompts.
      This module never builds prompt strings itself -- it only sends
      whatever message list the prompt builders produce and parses the
      result.
    - All SDK / network / validation failures are translated into
      LLMServiceError (or a subclass) so the Orchestrator never has to
      know about the underlying SDK's own exception types.
    - Call 1 is requested with a JSON object response format and the raw
      text is validated against the SituationAssessment Pydantic model,
      so malformed or hallucinated LLM output fails loudly here rather
      than propagating downstream.
"""

from __future__ import annotations

import json
import logging

from openai import APIError, APITimeoutError, OpenAI
from pydantic import ValidationError

from app.core.models import ConversationState, KnowledgeChunk, SituationAssessment
from app.prompts.prompts import (
    build_response_generation_prompt,
    build_situation_assessment_prompt,
)

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Base error for any failure while talking to the LLM.

    The Orchestrator catches this exact type (and its subclasses) at both
    call sites and degrades to a safe fallback response instead of
    letting the exception surface to the API layer.
    """


class LLMTimeoutError(LLMServiceError):
    """Raised when the LLM API call exceeds the configured timeout."""


class LLMResponseParsingError(LLMServiceError):
    """Raised when Call 1's output cannot be parsed/validated as JSON
    matching the SituationAssessment schema."""


class LLMService:
    """Thin, typed wrapper around an OpenAI-compatible chat completions API.

    Owns the API client and every model/temperature/timeout setting
    needed to make the two LLM calls in the pipeline. Holds no
    conversation state of its own -- state always flows in as an
    explicit argument (ConversationState) and is never mutated here.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        assessment_model: str = "gemini-2.5-flash",
        response_model: str = "gemini-2.5-flash",
        assessment_temperature: float = 0.0,
        response_temperature: float = 0.4,
        timeout_seconds: float = 20.0,
    ) -> None:
        """
        Args:
            api_key: API key for the configured provider (currently a
                Google Gemini API key from Google AI Studio). May be
                empty in local/dev environments that don't intend to
                make real calls; the client is still constructed
                lazily-safe, but a real call will raise LLMServiceError
                if the key is invalid.
            base_url: Base URL of the OpenAI-compatible endpoint to call.
                Defaults to Gemini's OpenAI-compatibility endpoint.
            assessment_model: Model identifier used for Call 1
                (Situation Assessment).
            response_model: Model identifier used for Call 2
                (Response Generation).
            assessment_temperature: Sampling temperature for Call 1. Kept
                low/zero by default since this call must be consistent
                and machine-parseable.
            response_temperature: Sampling temperature for Call 2, where
                some natural variation in phrasing is acceptable.
            timeout_seconds: Per-request timeout passed to the SDK.
        """
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
        self.assessment_model = assessment_model
        self.response_model = response_model
        self.assessment_temperature = assessment_temperature
        self.response_temperature = response_temperature
        self.timeout_seconds = timeout_seconds

    # -----------------------------------------------------------------
    # Call 1: Situation Assessment
    # -----------------------------------------------------------------
    def assess_situation(
        self, state: ConversationState, user_message: str
    ) -> SituationAssessment:
        """Classify the latest user turn into a strict SituationAssessment.

        Args:
            state: Current ConversationState, used by the prompt builder
                for context (language, entities, history).
            user_message: Raw text of the latest user turn.

        Returns:
            A validated SituationAssessment.

        Raises:
            LLMServiceError: on any network/API failure.
            LLMResponseParsingError: if the model's output is not valid
                JSON matching the SituationAssessment schema.
        """
        messages = build_situation_assessment_prompt(state, user_message)

        try:
            completion = self._client.chat.completions.create(
                model=self.assessment_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.assessment_temperature,
                response_format={"type": "json_object"},
                timeout=self.timeout_seconds,
            )
        except APITimeoutError as exc:
            logger.error("Situation Assessment call timed out: %s", exc)
            raise LLMTimeoutError("Situation Assessment call timed out") from exc
        except APIError as exc:
            logger.error("Situation Assessment call failed: %s", exc)
            raise LLMServiceError(f"Situation Assessment call failed: {exc}") from exc

        raw_content = completion.choices[0].message.content or ""

        try:
            payload = json.loads(raw_content)
            return SituationAssessment.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error(
                "Situation Assessment returned unparseable output: %s | raw=%s",
                exc,
                raw_content,
            )
            raise LLMResponseParsingError(
                "Situation Assessment output failed schema validation"
            ) from exc

    # -----------------------------------------------------------------
    # Call 2: Response Generation
    # -----------------------------------------------------------------
    def generate_response(
        self,
        state: ConversationState,
        knowledge_chunks: list[KnowledgeChunk],
        assessment: SituationAssessment,
    ) -> str:
        """Generate the final natural-language reply for this turn.

        Args:
            state: Current ConversationState (already merged with this
                turn's assessment by the Orchestrator).
            knowledge_chunks: Retrieved grounding knowledge, possibly
                empty.
            assessment: This turn's SituationAssessment, used for
                language/risk/confidence framing.

        Returns:
            Raw reply text. The caller (Orchestrator) is responsible for
            passing this through SafetyValidator before it reaches the
            user -- this method performs no safety filtering itself.

        Raises:
            LLMServiceError: on any network/API failure.
        """
        messages = build_response_generation_prompt(state, knowledge_chunks, assessment)

        try:
            completion = self._client.chat.completions.create(
                model=self.response_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.response_temperature,
                timeout=self.timeout_seconds,
            )
        except APITimeoutError as exc:
            logger.error("Response Generation call timed out: %s", exc)
            raise LLMTimeoutError("Response Generation call timed out") from exc
        except APIError as exc:
            logger.error("Response Generation call failed: %s", exc)
            raise LLMServiceError(f"Response Generation call failed: {exc}") from exc

        reply = (completion.choices[0].message.content or "").strip()
        if not reply:
            logger.error("Response Generation returned empty content")
            raise LLMServiceError("Response Generation returned empty content")

        return reply