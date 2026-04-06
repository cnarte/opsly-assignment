"""Health check router."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from src.shared.schemas import HealthResponse
from src.shared.settings import Settings
from src.gateway.mcp_client import check_agent_health

router = APIRouter(tags=["health"])

settings = Settings()

_AGENTS = {
    "orchestrator": settings.ORCHESTRATOR_PORT,
    "indexer": settings.INDEXER_PORT,
    "graph_query": settings.GRAPH_QUERY_PORT,
    "code_analyst": settings.CODE_ANALYST_PORT,
    "memory": settings.MEMORY_PORT,
}


@router.get("/api/agents/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return aggregate health status by probing each agent MCP server."""
    tasks = {
        name: check_agent_health(name, port) for name, port in _AGENTS.items()
    }
    results = await asyncio.gather(*tasks.values())
    agents = dict(zip(tasks.keys(), results))

    all_healthy = all(s == "healthy" for s in agents.values())
    status = "ok" if all_healthy else "degraded"

    return HealthResponse(status=status, agents=agents)
