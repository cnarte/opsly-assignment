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


@pytest.mark.asyncio
async def test_list_entities_tree_groups_by_folder():
    """list_entities_tree should group results by folder prefix."""
    markdown = (
        "| name | file_path |\n"
        "| --- | --- |\n"
        "| get_item | fastapi/routing.py |\n"
        "| add_route | fastapi/routing.py |\n"
        "| test_get | tests/test_routing.py |\n"
    )
    mock_result = {"markdown": markdown, "row_count": 3}

    with patch("src.graph_query.server._call_gitnexus", _mock_call(mock_result)):
        from src.graph_query import server as srv
        result = await srv.list_entities_tree("Function")

    assert result["total"] == 3
    tree = result["tree"]
    assert "fastapi" in tree
    assert tree["fastapi"]["count"] == 2
    assert tree["fastapi"]["files"]["routing.py"] == 2
    assert "tests" in tree
    assert tree["tests"]["count"] == 1


@pytest.mark.asyncio
async def test_list_entities_tree_handles_empty():
    """list_entities_tree should return zero totals on empty gitnexus result."""
    with patch("src.graph_query.server._call_gitnexus", _mock_call({})):
        from src.graph_query import server as srv
        result = await srv.list_entities_tree("Function")

    assert result["total"] == 0
    assert result["tree"] == {}


@pytest.mark.asyncio
async def test_list_entities_tree_rootlevel_file():
    """Files with no folder prefix (e.g. 'main.py') should go under '_root'."""
    markdown = (
        "| name | file_path |\n"
        "| --- | --- |\n"
        "| main_func | main.py |\n"
    )
    with patch("src.graph_query.server._call_gitnexus", _mock_call({"markdown": markdown, "row_count": 1})):
        from src.graph_query import server as srv
        result = await srv.list_entities_tree("Function")

    assert "_root" in result["tree"]
    assert result["tree"]["_root"]["files"]["main.py"] == 1
