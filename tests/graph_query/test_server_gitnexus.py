"""Tests for gitnexus-backed graph_query server."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


def _mock_call(return_value: dict):
    return AsyncMock(return_value=return_value)


@pytest.mark.asyncio
async def test_find_entity_calls_query_tool():
    """find_entity() should delegate to gitnexus query tool."""
    with patch("src.graph_query.server._call_gitnexus", _mock_call({"results": []})) as mock:
        from src.graph_query import server as srv_mod
        await srv_mod.find_entity("FastAPI", entity_type="Class", repo_id="fastapi")
        mock.assert_awaited_once_with("query", {"query": "FastAPI", "repo": "fastapi"})


@pytest.mark.asyncio
async def test_get_symbol_context_calls_context_tool():
    """get_symbol_context() should delegate to gitnexus context tool."""
    with patch("src.graph_query.server._call_gitnexus", _mock_call({"symbol": "FastAPI", "outgoing": []})) as mock:
        from src.graph_query import server as srv_mod
        await srv_mod.get_symbol_context("FastAPI", repo_id="fastapi")
        mock.assert_awaited_once_with("context", {"symbol": "FastAPI", "repo": "fastapi"})


@pytest.mark.asyncio
async def test_analyze_impact_calls_impact_tool():
    """analyze_impact() should delegate to gitnexus impact tool."""
    with patch("src.graph_query.server._call_gitnexus", _mock_call({"symbol": "FastAPI", "levels": {}})) as mock:
        from src.graph_query import server as srv_mod
        await srv_mod.analyze_impact("FastAPI", depth=2, repo_id="fastapi")
        mock.assert_awaited_once_with("impact", {"symbol": "FastAPI", "repo": "fastapi"})


@pytest.mark.asyncio
async def test_execute_query_calls_cypher_tool():
    """execute_query() should pass the raw cypher to gitnexus cypher tool."""
    with patch("src.graph_query.server._call_gitnexus", _mock_call({"results": []})) as mock:
        from src.graph_query import server as srv_mod
        await srv_mod.execute_query("MATCH (n:Function) RETURN n LIMIT 5")
        mock.assert_awaited_once_with("cypher", {"query": "MATCH (n:Function) RETURN n LIMIT 5"})
