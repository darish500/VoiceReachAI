"""
app/main.py

FastAPI application entry point for VoiceReach AI.

Run locally with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.core.config import get_settings
from app.utils.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoiceReach AI",
    description=(
        "Retrieval-Augmented Voice Assistant backend for underserved "
        "Nigerian communities."
    ),
    version="0.1.0",
)

app.include_router(chat_router)


@app.on_event("startup")
async def on_startup() -> None:
    """Log confirmation that the service is up and configured."""
    logger.info(
        "VoiceReach AI starting up | environment=%s | assessment_model=%s | response_model=%s",
        settings.environment,
        settings.assessment_model,
        settings.response_model,
    )


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Lightweight liveness probe used by orchestration/monitoring tools."""
    return {"status": "ok", "environment": settings.environment}
