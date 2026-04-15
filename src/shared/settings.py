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
    # Llama 3.1 8B is ~15x faster than Nemotron 120B on OpenRouter free tier
    # (3-8s vs 60-90s per call). Override via OPENROUTER_MODEL env var or UI dropdown.
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"

    # LLM / LM Studio (local models via OpenAI-compatible API, no rate limiting)
    LMSTUDIO_BASE_URL: str = "http://host.docker.internal:1234/v1"

    # OpenAI Embeddings
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

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
    ORCHESTRATOR_STREAM_PORT: int = 8016
    INDEXER_PORT: int = 8011
    GRAPH_QUERY_PORT: int = 8012
    CODE_ANALYST_PORT: int = 8013
    MEMORY_PORT: int = 8014
    GITNEXUS_PORT: int = 8015
    GATEWAY_PORT: int = 8000

    # MCP client behaviour
    MCP_CALL_TIMEOUT_S: int = 120
    MCP_CALL_RETRIES: int = 1

    # Langfuse tracing (optional — leave blank to disable)
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    # Accept both LANGFUSE_BASE_URL (Langfuse's own name) and LANGFUSE_HOST
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"
    LANGFUSE_HOST: str = ""  # alias — if set, overrides LANGFUSE_BASE_URL

    # General
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
