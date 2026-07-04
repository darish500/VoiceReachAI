"""
app/services/safety.py
 
SafetyValidator: the final, deterministic safety net between the
Response Generation LLM call and the caller's ear.
 
Why this module exists:
    app/prompts/prompts.py already gives the Response Generation call an
    extremely strict system prompt (grounding rules, emergency rules,
    word limits, persona, output-format constraints). LLMs are
    probabilistic, though, and can still occasionally drift: run long,
    leak a fragment of their instructions, emit markdown, soften an
    emergency instruction, or state something confidently when no
    knowledge was actually retrieved to support it. SafetyValidator is
    the deterministic, non-LLM layer that catches those failure modes
    before a reply reaches a real caller on a real phone call.
 
What this module is NOT:
    - It is not another LLM call. It never calls OpenAI, ChromaDB, or
      any embedding model. It never touches the network.
    - It does not perform semantic fact-checking. It cannot verify
      whether a claim is *true* -- only whether the reply looks like an
      unsupported, overly-confident factual claim made without any
      retrieved knowledge behind it. That is a conservative heuristic,
      not a guarantee.
    - It does not rewrite good responses. If a reply already satisfies
      every rule below, it is returned essentially unchanged (only
      harmless whitespace cleanup is applied).
 
Contract:
    This module implements the exact `SafetyValidator` Protocol that
    app/services/orchestrator.py already depends on:
 
        def validate(
            self,
            response_text: str,
            state: ConversationState,
            assessment: SituationAssessment,
            knowledge_chunks: list[KnowledgeChunk],
        ) -> str: ...
 
    That signature is fixed by the Orchestrator and must not change.
    `validate()` itself never raises during normal operation -- any
    unexpected internal failure is logged and swallowed in favor of
    returning a safe fallback message, since a raised exception here
    would otherwise drop the caller's turn (or the whole call) at the
    very last step of the pipeline.
"""
 
from __future__ import annotations
 
import logging
import re
from typing import Optional
 
from app.core.models import ConversationState, KnowledgeChunk, RiskLevel, SituationAssessment
from app.prompts.prompts import MAX_EMERGENCY_RESPONSE_WORDS, MAX_RESPONSE_WORDS
 
logger = logging.getLogger(__name__)
 
 
# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class SafetyValidationError(Exception):
    """Base class for all errors raised by this module.
 
    Note that `SafetyValidator.validate()` itself never raises this (or
    anything else) during normal operation -- see the module docstring.
    This hierarchy exists for configuration-time failures and for
    internal helpers that choose to raise rather than silently
    misbehave; `validate()` catches and converts those into a safe
    fallback response.
    """
 
 
class SafetyConfigurationError(SafetyValidationError):
    """Raised at construction time if SafetyValidator is misconfigured.
 
    Kept distinct from runtime validation failures: a bad word-limit
    configuration is a deploy-time bug that should fail loudly and
    immediately, not be silently absorbed the way a runtime validation
    hiccup is.
    """
 
 
# ---------------------------------------------------------------------------
# Default fallback messages
#
# These are intentionally short, plain, and domain-neutral (they never
# assume "health" vs. "financial" context) since they may be shown when
# the validator has just discarded a reply it could not trust.
# ---------------------------------------------------------------------------
DEFAULT_GENERIC_FALLBACK_MESSAGE = (
    "Sorry, I could not prepare a safe answer for that. Please try again, "
    "or speak with a local health worker or financial officer."
)
 
DEFAULT_INSUFFICIENT_KNOWLEDGE_FALLBACK_MESSAGE = (
    "I don't have enough information to answer safely. Please speak with "
    "a healthcare worker or local support service."
)
 
DEFAULT_EMERGENCY_FALLBACK_MESSAGE = (
    "This may be an emergency. Please go to the nearest hospital or call "
    "emergency services right now."
)
 
 
# ---------------------------------------------------------------------------
# SafetyValidator
# ---------------------------------------------------------------------------
class SafetyValidator:
    """Deterministic post-processing guardrail applied to every generated reply.
 
    Instances are stateless with respect to any single call to
    `validate()` -- all configuration (word limits, fallback message
    text) is fixed at construction time, and `validate()` is a pure
    function of its four arguments. This makes SafetyValidator trivial
    to unit test and safe to share across concurrent requests.
    """
 
    # Phrases that indicate the model has leaked a fragment of its own
    # instructions, internal reasoning, or the raw prompt scaffolding
    # (e.g. "Retrieved Knowledge", "System Prompt") rather than producing
    # a clean, user-facing reply. Multi-word phrases are matched with
    # flexible whitespace; single, more generic words are matched with
    # word boundaries to reduce (though not eliminate) false positives.
    #
    # Note: "assistant" is deliberately included per spec even though it
    # can occasionally false-positive on legitimate phrases (e.g. a
    # caller asking about a "financial assistant" program). Given the
    # severity of leaking internal scaffolding to a live caller, this
    # trade-off is intentional -- false positives fall back to a safe
    # generic message, which is a low-cost failure mode.
    _LEAK_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"retrieved\s+knowledge", re.IGNORECASE),
        re.compile(r"system\s+prompt", re.IGNORECASE),
        re.compile(r"chain\s+of\s+thought", re.IGNORECASE),
        re.compile(r"\breasoning\b", re.IGNORECASE),
        re.compile(r"\binternal\b", re.IGNORECASE),
        re.compile(r"\bjson\b", re.IGNORECASE),
        re.compile(r"\bassistant\b", re.IGNORECASE),
    ]
 
    # Conservative keyword heuristics used to decide whether a reply
    # "sounds like" a confident, specific factual claim (the kind that
    # should only ever be made when grounded in retrieved knowledge).
    # This is intentionally crude -- see the module docstring's note on
    # NOT performing semantic fact-checking.
    _CONFIDENT_CLAIM_MARKERS: tuple[str, ...] = (
        "always",
        "definitely",
        "certainly",
        "guaranteed",
        "will cure",
        "the treatment is",
        "you should take",
        "the dosage is",
        "the clinic is located",
        "the address is",
        "the interest rate is",
        "the fee is",
    )
 
    # Phrases that indicate a reply is already appropriately hedged (i.e.
    # it is doing exactly what an ungrounded reply should do -- admitting
    # uncertainty and redirecting to a human), and therefore should NOT
    # be treated as an unsupported claim even though it may otherwise
    # match nothing else. This keeps requirement 7 ("preserve valid
    # responses") from being undermined by requirement 4.
    _HEDGE_PHRASES: tuple[str, ...] = (
        "don't have enough information",
        "do not have enough information",
        "not sure",
        "please consult",
        "please speak with",
        "speak with a",
        "unable to confirm",
        "cannot confirm",
        "i can't confirm",
    )
 
    # Keyword heuristics used to check whether a reply already contains
    # a clear emergency instruction. Deliberately broad enough to cover
    # both medical emergencies (hospital, ambulance) and financial ones
    # (stop sharing money/details, contact your bank), since VoiceReach
    # AI covers both domains.
    _EMERGENCY_INDICATOR_PHRASES: tuple[str, ...] = (
        "emergency",
        "immediately",
        "right now",
        "hospital",
        "ambulance",
        "urgent care",
        "call for help",
        "seek care now",
        "seek help now",
        "do not share",
        "stop sending money",
        "contact your bank",
    )
 
    # Markdown constructs stripped so the final output is plain spoken
    # text. Order matters: bold (**/__) must be handled before italic
    # (*/_)  or the italic patterns would partially consume bold markers
    # first and leave stray asterisks/underscores behind.
    _MARKDOWN_SUBSTITUTIONS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),        # headers
        (re.compile(r"```.*?```", re.DOTALL), " "),                    # fenced code blocks
        (re.compile(r"`([^`]*)`"), r"\1"),                             # inline code
        (re.compile(r"\*\*([^*]*)\*\*"), r"\1"),                       # **bold**
        (re.compile(r"__([^_]*)__"), r"\1"),                           # __bold__
        (re.compile(r"\*([^*]*)\*"), r"\1"),                           # *italic*
        (re.compile(r"(?<!\w)_([^_]*)_(?!\w)"), r"\1"),                # _italic_
        (re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE), ""),           # bullet markers
        (re.compile(r"^\s{0,3}\d+\.\s+", re.MULTILINE), ""),           # numbered list markers
    ]
 
    def __init__(
        self,
        max_response_words: Optional[int] = None,
        max_emergency_response_words: Optional[int] = None,
        generic_fallback_message: str = DEFAULT_GENERIC_FALLBACK_MESSAGE,
        insufficient_knowledge_fallback_message: str = DEFAULT_INSUFFICIENT_KNOWLEDGE_FALLBACK_MESSAGE,
        emergency_fallback_message: str = DEFAULT_EMERGENCY_FALLBACK_MESSAGE,
    ) -> None:
        """
        Args:
            max_response_words: Maximum word count for a normal (non-
                emergency) reply. Defaults to the same
                `MAX_RESPONSE_WORDS` constant the Response Generation
                prompt itself is instructed with, so the validator's
                enforced limit never silently drifts from the prompt's
                stated limit.
            max_emergency_response_words: Maximum word count for a reply
                when `assessment.risk_level` is HIGH or CRITICAL.
                Defaults to `MAX_EMERGENCY_RESPONSE_WORDS` from the
                prompts module for the same reason.
            generic_fallback_message: Returned when a reply is empty,
                leaks prompt internals, or becomes empty after markdown
                stripping.
            insufficient_knowledge_fallback_message: Returned when a
                reply appears to make a confident factual claim with no
                retrieved knowledge to support it.
            emergency_fallback_message: Returned when `assessment.risk_level`
                is HIGH/CRITICAL but the reply does not clearly instruct
                the caller to seek emergency help.
 
        Raises:
            SafetyConfigurationError: if the configured word limits are
                not positive, or if the emergency limit exceeds the
                normal limit (which would defeat the point of having a
                stricter emergency limit at all).
        """
        self.max_response_words = (
            max_response_words if max_response_words is not None else MAX_RESPONSE_WORDS
        )
        self.max_emergency_response_words = (
            max_emergency_response_words
            if max_emergency_response_words is not None
            else MAX_EMERGENCY_RESPONSE_WORDS
        )
 
        if self.max_response_words <= 0 or self.max_emergency_response_words <= 0:
            raise SafetyConfigurationError(
                "max_response_words and max_emergency_response_words must "
                "both be positive integers."
            )
        if self.max_emergency_response_words > self.max_response_words:
            raise SafetyConfigurationError(
                "max_emergency_response_words must not exceed "
                "max_response_words -- the emergency limit is supposed to "
                "be the stricter of the two."
            )
 
        self.generic_fallback_message = generic_fallback_message
        self.insufficient_knowledge_fallback_message = insufficient_knowledge_fallback_message
        self.emergency_fallback_message = emergency_fallback_message
 
    # -----------------------------------------------------------------
    # Public API (matches the Orchestrator's SafetyValidator Protocol)
    # -----------------------------------------------------------------
    def validate(
        self,
        response_text: str,
        state: ConversationState,
        assessment: SituationAssessment,
        knowledge_chunks: list[KnowledgeChunk],
    ) -> str:
        """Run every safety check and return the text safe to speak to the caller.
 
        This method deliberately never raises. Any unexpected failure in
        an internal helper is caught, logged, and converted into
        `self.generic_fallback_message` -- a bug in this validator must
        never be the reason a caller's turn (or call) drops.
 
        Checks are applied in an order chosen so that weaker,
        more-easily-triggered problems (empty text, leaked internals,
        markdown noise) are resolved first, and the emergency-instruction
        check runs LAST, as an authoritative override that can replace
        the output of every earlier step. This guarantees the literal
        requirement "never weaken emergency guidance": no matter what
        happened earlier in this method, a HIGH/CRITICAL-risk turn can
        never leave this method without a clear emergency instruction in
        the final text.
 
        Args:
            response_text: The raw text returned by
                LLMService.generate_response().
            state: Current ConversationState (available to helpers for
                context; not currently required by any check, but part
                of the fixed Protocol signature).
            assessment: This turn's SituationAssessment -- specifically
                `risk_level` drives the emergency checks.
            knowledge_chunks: The KnowledgeChunks retrieved for this
                turn. An empty list is what triggers the grounding
                heuristic in `_contains_unsupported_claims`.
 
        Returns:
            The final, safe text to return to the caller. Never empty,
            never raises.
        """
        try:
            return self._run_pipeline(response_text, assessment, knowledge_chunks)
        except Exception as exc:  # noqa: BLE001 -- this is the last line
            # of defense in the whole pipeline; nothing here is allowed
            # to propagate up and drop the caller's turn.
            logger.exception(
                "SafetyValidator.validate failed unexpectedly; returning "
                "generic fallback. error=%s",
                exc,
            )
            return self.generic_fallback_message
 
    # -----------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------
    def _run_pipeline(
        self,
        response_text: str,
        assessment: SituationAssessment,
        knowledge_chunks: list[KnowledgeChunk],
    ) -> str:
        """The actual validation pipeline, factored out of `validate()`.
 
        Split from `validate()` purely so the try/except in the public
        method stays a single, clearly-scoped safety net around the
        whole pipeline rather than being interleaved with the pipeline
        logic itself.
        """
        text = self._clean_whitespace(response_text or "")
 
        if not self._validate_not_empty(text):
            text = self._apply_fallback(
                reason="empty_response",
                original=response_text,
                fallback=self.generic_fallback_message,
            )
        elif self._contains_prompt_leak(text):
            text = self._apply_fallback(
                reason="prompt_leak_detected",
                original=text,
                fallback=self.generic_fallback_message,
            )
        else:
            stripped = self._clean_whitespace(self._strip_markdown(text))
            if stripped != text:
                logger.info("SafetyValidator: stripped markdown artifacts from response.")
            text = stripped
 
            if not self._validate_not_empty(text):
                text = self._apply_fallback(
                    reason="empty_after_markdown_strip",
                    original=response_text,
                    fallback=self.generic_fallback_message,
                )
            elif self._contains_unsupported_claims(text, knowledge_chunks):
                text = self._apply_fallback(
                    reason="unsupported_claim_without_grounding",
                    original=text,
                    fallback=self.insufficient_knowledge_fallback_message,
                )
            else:
                word_limit = (
                    self.max_emergency_response_words
                    if self._requires_emergency_override(assessment)
                    else self.max_response_words
                )
                trimmed = self._enforce_max_words(text, word_limit)
                if trimmed != text:
                    logger.info(
                        "SafetyValidator: trimmed response from %d to %d word(s).",
                        len(text.split()),
                        word_limit,
                    )
                text = trimmed
 
        # --- Authoritative, final emergency safety net ---
        # Runs regardless of which branch above produced `text`, and can
        # override any of them. This is what makes the emergency
        # guarantee unconditional rather than best-effort.
        if self._requires_emergency_override(assessment):
            if not self._has_emergency_instruction(text):
                text = self._apply_fallback(
                    reason="missing_emergency_instruction",
                    original=text,
                    fallback=self.emergency_fallback_message,
                )
            # Re-enforce the stricter emergency word cap as the very
            # last step, whether `text` is the original grounded reply,
            # an earlier fallback, or the emergency fallback itself.
            text = self._enforce_max_words(text, self.max_emergency_response_words)
 
        return text
 
    # -----------------------------------------------------------------
    # Individual checks (kept small and independently testable)
    # -----------------------------------------------------------------
    @staticmethod
    def _validate_not_empty(text: str) -> bool:
        """Return True if `text` contains any non-whitespace content."""
        return bool(text and text.strip())
 
    def _contains_prompt_leak(self, text: str) -> bool:
        """Return True if `text` looks like it leaked internal prompt scaffolding."""
        return any(pattern.search(text) for pattern in self._LEAK_PATTERNS)
 
    def _contains_unsupported_claims(
        self, text: str, knowledge_chunks: list[KnowledgeChunk]
    ) -> bool:
        """Conservative heuristic for an ungrounded, overly-confident factual claim.
 
        Deliberately NOT semantic fact-checking (the module docstring
        explains why that's out of scope). The logic is:
 
        1. If any KnowledgeChunks were actually retrieved, this check
           never triggers -- verifying whether the model used them
           correctly is the Response Generation prompt's job, not this
           validator's. This check exists specifically for the case
           where NO knowledge was retrieved at all.
        2. If the reply already contains hedging language (e.g. "please
           consult", "not sure"), it is already behaving exactly as
           instructed for an ungrounded situation, so it is left alone.
        3. Otherwise, the presence of a digit (a classic marker of a
           specific, checkable fact -- a dosage, an age, a fee, a
           distance) or a confident assertion phrase (e.g. "the
           treatment is", "guaranteed") is treated as a signal that the
           model stated something specific without grounding, and the
           reply is replaced with the insufficient-knowledge fallback.
        """
        if knowledge_chunks:
            return False
 
        lowered = text.lower()
        if any(hedge in lowered for hedge in self._HEDGE_PHRASES):
            return False
 
        has_digit = any(character.isdigit() for character in text)
        has_confident_marker = any(
            marker in lowered for marker in self._CONFIDENT_CLAIM_MARKERS
        )
        return has_digit or has_confident_marker
 
    def _requires_emergency_override(self, assessment: SituationAssessment) -> bool:
        """Return True if this turn's risk level mandates emergency wording."""
        return assessment.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
 
    def _has_emergency_instruction(self, text: str) -> bool:
        """Return True if `text` already contains a clear emergency instruction.
 
        Keyword-heuristic only (see class docstring/comments on
        `_EMERGENCY_INDICATOR_PHRASES`) -- covers both medical emergency
        phrasing ("go to the hospital now") and financial emergency
        phrasing ("do not share your PIN", "contact your bank
        immediately").
        """
        lowered = text.lower()
        return any(phrase in lowered for phrase in self._EMERGENCY_INDICATOR_PHRASES)
 
    # -----------------------------------------------------------------
    # Transformations
    # -----------------------------------------------------------------
    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Collapse repeated whitespace/newlines and trim leading/trailing space.
 
        Applied liberally throughout the pipeline since intermediate
        transformations (markdown stripping in particular) can leave
        behind irregular spacing that would sound unnatural if spoken
        aloud verbatim.
        """
        return re.sub(r"\s+", " ", text).strip()
 
    def _strip_markdown(self, text: str) -> str:
        """Remove common Markdown constructs, leaving their inner text intact.
 
        The final reply must be plain spoken text -- a caller on a phone
        call has no way to "see" a bolded word or a bullet point. Each
        substitution keeps the human-readable content (e.g. `**take
        this**` becomes `take this`) rather than deleting it outright.
        """
        result = text
        for pattern, replacement in self._MARKDOWN_SUBSTITUTIONS:
            result = pattern.sub(replacement, result)
        return result
 
    @staticmethod
    def _enforce_max_words(text: str, limit: int) -> str:
        """Trim `text` to at most `limit` words, always on a word boundary.
 
        Splitting on whitespace and rejoining the first `limit` tokens
        guarantees a word is never cut in half. A trailing period is
        added when the trim lands mid-sentence (i.e. the last kept word
        doesn't already end in sentence punctuation), so a trimmed reply
        still sounds like a complete thought when spoken aloud.
        """
        words = text.split()
        if len(words) <= limit:
            return text
 
        trimmed_words = words[:limit]
        trimmed_text = " ".join(trimmed_words)
        if trimmed_text and trimmed_text[-1] not in ".!?":
            trimmed_text += "."
        return trimmed_text
 
    # -----------------------------------------------------------------
    # Logging helper
    # -----------------------------------------------------------------
    def _apply_fallback(self, reason: str, original: str, fallback: str) -> str:
        """Log that a fallback was applied, and return the fallback text.
 
        Centralizing this (rather than calling `logger.warning` inline
        at each call site) keeps the log format consistent and makes it
        trivial to find every place a fallback can be triggered by
        searching for calls to this method.
 
        Args:
            reason: Short machine-readable reason code (e.g.
                "empty_response", "missing_emergency_instruction") for
                log filtering/alerting.
            original: The text that was discarded. Truncated in the log
                line to keep log volume reasonable.
            fallback: The replacement text being returned.
        """
        truncated_original = (original or "")[:200]
        logger.warning(
            "SafetyValidator: applying fallback (reason=%s). original=%r replacement=%r",
            reason,
            truncated_original,
            fallback,
        )
        return fallback
 