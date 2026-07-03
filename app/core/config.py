"""
app/core/config.py

Centralized application configuration for VoiceReach AI, loaded from
environment variables (and a local .env file in development) using
pydantic-settings.

Every other module that needs a configurable value (API keys, model
names, temperatures, retrieval top-k, Chroma paths, logging level) should
import the shared `settings` singleton from here rather than reading
`os.environ` directly. This keeps configuration centralized, typed, and
validated at process startup instead of failing deep inside a request.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, sourced from environment / .env.

    Field names map to UPPER_SNAKE_CASE environment variables (e.g.
    `openai_api_key` <- `OPENAI_API_KEY`) via pydantic-settings' default
    case-insensitive env matching.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM configuration -------------------------------------------
    # Field is still named "openai_api_key" for historical reasons, but
    # it holds whatever key belongs to the provider set in llm_base_url
    # (currently Google Gemini's OpenAI-compatible endpoint).
    openai_api_key: str = ""
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    assessment_model: str = "gemini-2.5-flash"
    response_model: str = "gemini-2.5-flash"
    assessment_temperature: float = 0.0
    response_temperature: float = 0.4
    llm_timeout_seconds: float = 20.0

    # --- Retrieval / Knowledge base --------------------------------------
    retrieval_top_k: int = 3
    chroma_persist_directory: str = "data/chroma"
    chroma_collection_name: str = "voice_reach_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Application / environment ----------------------------------------
    environment: str = "development"
    log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance.

    Cached with lru_cache so environment variables are parsed once per
    process and every caller (dependencies, scripts) shares the same
    validated configuration object.
    """
    return Settings()


settings = get_settings()