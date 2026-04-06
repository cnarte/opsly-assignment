"""Index endpoints — trigger and check repository indexing."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.shared.schemas import IndexRequest, IndexResponse
from src.gateway.mcp_client import call_orchestrator_tool, call_agent_tool
from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

router = APIRouter(tags=["index"])


@router.post("/api/index", response_model=IndexResponse)
async def trigger_index(request: IndexRequest) -> IndexResponse:
    """Trigger repository indexing via the orchestrator."""
    result = await call_orchestrator_tool(
        "handle_index_request",
        {"repo_url": request.repo_url, "ref": request.ref},
    )

    job_id = result.get("job_id", "unknown")
    status = result.get("status", result.get("error", "submitted"))

    return IndexResponse(job_id=job_id, status=status)


@router.get("/api/index/status/{job_id}")
async def index_status(job_id: str) -> dict:
    """Get the status of an indexing job from the Indexer agent."""
    result = await call_agent_tool(
        settings.INDEXER_PORT,
        "get_index_status",
        {"job_id": job_id},
    )
    return result
