"""Tests for the gitnexus-backed indexer server."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_index_repository_returns_job_id():
    """index_repository should return a job_id immediately."""
    with patch("src.indexer.server._call_gitnexus", AsyncMock(return_value={"status": "indexed"})), \
         patch("src.indexer.server._clone_repo", AsyncMock(return_value="/workspace/repos/fastapi")):
        from src.indexer.server import index_repository
        result = await index_repository("https://github.com/fastapi/fastapi.git")
        assert "job_id" in result
        assert result["status"] == "started"


@pytest.mark.asyncio
async def test_get_index_status_known_job():
    """get_index_status should return state for a known job_id."""
    from src.indexer import server as srv_mod
    srv_mod._jobs["test-job-123"] = {"status": "completed", "repo": "fastapi"}
    from src.indexer.server import get_index_status
    result = await get_index_status("test-job-123")
    assert result["status"] == "completed"
    assert result["job_id"] == "test-job-123"


@pytest.mark.asyncio
async def test_get_index_status_unknown_job():
    """get_index_status should return error for unknown job_id."""
    from src.indexer.server import get_index_status
    result = await get_index_status("nonexistent-job")
    assert "error" in result
