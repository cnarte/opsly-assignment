"""Application settings loaded from environment variables via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration for all services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM / OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "qwen/qwen3.6-plus:free"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Repository defaults
    DEFAULT_REPO_URL: str = "https://github.com/fastapi/fastapi.git"
    DEFAULT_REPO_REF: str = ""

    # Service ports
    ORCHESTRATOR_PORT: int = 8010
    INDEXER_PORT: int = 8011
    GRAPH_QUERY_PORT: int = 8012
    CODE_ANALYST_PORT: int = 8013
    MEMORY_PORT: int = 8014
    GATEWAY_PORT: int = 8000

    # General
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
