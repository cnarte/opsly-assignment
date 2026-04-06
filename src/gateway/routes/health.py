"""Health check router."""

from __future__ import annotations

from fastapi import APIRouter

from src.shared.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/agents/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return aggregate health status of the system."""
    return HealthResponse(
        status="ok",
        agents={
            "orchestrator": "unknown",
            "indexer": "unknown",
            "graph_query": "unknown",
            "code_analyst": "unknown",
            "memory": "unknown",
        },
    )
