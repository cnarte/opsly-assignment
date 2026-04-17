"""Shared Pydantic models used across services."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from a user."""

    message: str
    session_id: str | None = None
    stream: bool = False
    repo_id: str | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    """Response returned to the user after agent processing."""

    response: str
    session_id: str
    agents_used: list[str] = Field(default_factory=list)
    agent_results: dict = Field(default_factory=dict)
    tool_plan: list = Field(default_factory=list)
    tool_calls: list = Field(default_factory=list)


class IndexRequest(BaseModel):
    """Request to index (or re-index) a repository."""

    repo_url: str
    ref: str = ""


class IndexResponse(BaseModel):
    """Acknowledgement of an indexing job."""

    job_id: str
    status: str


class HealthResponse(BaseModel):
    """Aggregate health status of the system."""

    status: str
    agents: dict[str, str] = Field(default_factory=dict)


class GraphStats(BaseModel):
    """Summary statistics for the Neo4j knowledge graph."""

    nodes: int
    relationships: int
    labels: dict[str, int] = Field(default_factory=dict)
