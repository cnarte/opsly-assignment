"""Tests for gitnexus-agent MCP server tool shapes."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.call_tool = AsyncMock(return_value={"results": [], "count": 0})
    client.analyze_repo = AsyncMock(return_value={"status": "indexed", "repo": "test", "path": "/tmp/test"})
    return client


@pytest.mark.asyncio
async def test_query_passes_repo_param(mock_client):
    """query() should pass repo param to gitnexus when provided."""
    with patch("src.gitnexus_agent.server._get_client", AsyncMock(return_value=mock_client)):
        from src.gitnexus_agent.server import query as _query
        await _query("FastAPI class", repo="fastapi")
        mock_client.call_tool.assert_awaited_once_with("query", {"query": "FastAPI class", "repo": "fastapi"})


@pytest.mark.asyncio
async def test_query_omits_empty_repo(mock_client):
    """query() should not pass repo key when repo is empty string."""
    with patch("src.gitnexus_agent.server._get_client", AsyncMock(return_value=mock_client)):
        from src.gitnexus_agent.server import query as _query
        await _query("some search", repo="")
        mock_client.call_tool.assert_awaited_once_with("query", {"query": "some search"})


@pytest.mark.asyncio
async def test_analyze_repo_delegates_to_client(mock_client):
    """analyze_repo() should call client.analyze_repo with path and name."""
    with patch("src.gitnexus_agent.server._get_client", AsyncMock(return_value=mock_client)):
        from src.gitnexus_agent.server import analyze_repo
        result = await analyze_repo("/workspace/repos/fastapi", "fastapi")
        mock_client.analyze_repo.assert_awaited_once_with("/workspace/repos/fastapi", "fastapi")
        assert result["status"] == "indexed"
