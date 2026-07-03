"""
app/api/routes/chat.py

The single public entry point into VoiceReach AI's conversational
pipeline: POST /chat.

This route contains no business logic of its own -- it validates the
incoming request shape, delegates entirely to
`Orchestrator.handle_message`, and returns whatever AIResponse comes
back. All classification, retrieval, generation, and safety-validation
logic lives in the Orchestrator and its collaborators.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import OrchestratorDep
from app.core.models import AIResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    session_id: str = Field(
        ..., min_length=1, description="Identifies which conversation this message belongs to."
    )
    message: str = Field(
        ..., min_length=1, description="Raw user text for this turn (already transcribed)."
    )


@router.post("/chat", response_model=AIResponse)
def chat(request: ChatRequest, orchestrator: OrchestratorDep) -> AIResponse:
    """Run one turn of the VoiceReach AI pipeline.

    Args:
        request: The session id and raw user message for this turn.
        orchestrator: Injected singleton Orchestrator.

    Returns:
        The structured AIResponse produced by the pipeline.

    Raises:
        HTTPException(500): only for truly unexpected failures.
            `Orchestrator.handle_message` already degrades gracefully for
            every LLM-related failure mode, so this should be rare.
    """
    try:
        return orchestrator.handle_message(
            session_id=request.session_id, user_message=request.message
        )
    except Exception as exc:  # noqa: BLE001 -- last-resort safety net
        logger.exception(
            "Unexpected error handling chat message for session_id=%s: %s",
            request.session_id,
            exc,
        )
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while processing your message."
        ) from exc
