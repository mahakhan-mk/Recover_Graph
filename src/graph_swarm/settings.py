"""Application settings loaded from environment variables."""

from collections.abc import Callable
from functools import lru_cache
from typing import cast

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the research prototype."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    hf_token: str | None = None
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openrouter_api_key: str | None = None
    # Kept only for non-experimental legacy configuration inspection. Runtime
    # model selection must use one of the role-specific settings below.
    openrouter_model: str | None = None
    openrouter_coding_model: str | None = None
    openrouter_abstraction_model: str | None = None
    model_provider: str = "openrouter"
    agent_request_limit: int = Field(default=10, gt=0)
    agent_command_timeout_seconds: float = Field(default=30.0, gt=0)
    agent_tests_timeout_seconds: float = Field(default=120.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    settings_factory = cast(Callable[..., Settings], Settings)
    return settings_factory()
