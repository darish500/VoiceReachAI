"""
app/api/dependencies.py

FastAPI dependency providers.

Every collaborator the Orchestrator needs (ConversationStore, LLMService,
KnowledgeRetriever, SafetyValidator, and the Orchestrator itself) is
expensive or stateful enough that it should be constructed once per
process and reused across requests -- not rebuilt on every call. Each
provider below is wrapped in `functools.lru_cache(maxsize=1)` so FastAPI's
dependency injection resolves to the same singleton instance every time,
without needing app.state or a custom container.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from typing_extensions import Annotated

from app.core.config import Settings, get_settings
from app.services.conversation_store import ConversationStore
from app.services.knowledge import KnowledgeRetriever
from app.services.llm_service import LLMService
from app.services.orchestrator import Orchestrator
from app.services.safety import SafetyValidator


@lru_cache(maxsize=1)
def get_conversation_store() -> ConversationStore:
    """Return the process-wide singleton ConversationStore."""
    return ConversationStore()


@lru_cache(maxsize=1)
def get_knowledge_retriever() -> KnowledgeRetriever:
    """Return the process-wide singleton KnowledgeRetriever.

    Initializing this opens/creates the persistent Chroma collection on
    disk and loads the sentence-transformers embedding model, so it is
    especially important this only happens once per process.
    """
    settings = get_settings()
    return KnowledgeRetriever(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model,
    )


@lru_cache(maxsize=1)
def get_safety_validator() -> SafetyValidator:
    """Return the process-wide singleton SafetyValidator."""
    return SafetyValidator()


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Return the process-wide singleton LLMService."""
    settings = get_settings()
    return LLMService(
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
        assessment_model=settings.assessment_model,
        response_model=settings.response_model,
        assessment_temperature=settings.assessment_temperature,
        response_temperature=settings.response_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
    )


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    """Return the process-wide singleton Orchestrator.

    Wires together every other singleton collaborator. This is the only
    object the API layer (app/api/routes/chat.py) is allowed to depend on
    for business logic.
    """
    settings = get_settings()
    return Orchestrator(
        conversation_store=get_conversation_store(),
        llm_service=get_llm_service(),
        knowledge_retriever=get_knowledge_retriever(),
        safety_validator=get_safety_validator(),
        default_retrieval_top_k=settings.retrieval_top_k,
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
ConversationStoreDep = Annotated[ConversationStore, Depends(get_conversation_store)]
KnowledgeRetrieverDep = Annotated[KnowledgeRetriever, Depends(get_knowledge_retriever)]
SafetyValidatorDep = Annotated[SafetyValidator, Depends(get_safety_validator)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
OrchestratorDep = Annotated[Orchestrator, Depends(get_orchestrator)]