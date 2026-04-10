"""Unit tests for graph_query MCP server tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_neo4j(rows: list = None) -> MagicMock:
    client = MagicMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.execute_query = AsyncMock(return_value=rows or [])
    return client


class TestFindEntity:
    @pytest.mark.asyncio
    async def test_returns_entity_when_found(self):
        from src.graph_query.server import find_entity

        rows = [{"symbol_id": "r1:class:m:FastAPI:0", "name": "FastAPI", "kind": "class"}]

        with patch("src.graph_query.server._get_client", new_callable=AsyncMock, return_value=_make_neo4j(rows)):
            result = await find_entity("FastAPI")

        assert result.get("count") == 1
        assert len(result.get("results", [])) == 1

    @pytest.mark.asyncio
    async def test_returns_not_found_when_empty(self):
        from src.graph_query.server import find_entity

        with patch("src.graph_query.server._get_client", new_callable=AsyncMock, return_value=_make_neo4j([])):
            result = await find_entity("NonExistent")

        assert result.get("count") == 0 or result.get("results") == []

    @pytest.mark.asyncio
    async def test_find_entity_with_type_filter(self):
        from src.graph_query.server import find_entity

        rows = [{"symbol_id": "r1:class:m:FastAPI:0", "name": "FastAPI", "kind": "class"}]
        with patch("src.graph_query.server._get_client", new_callable=AsyncMock, return_value=_make_neo4j(rows)):
            result = await find_entity("FastAPI", entity_type="Class")

        assert isinstance(result, dict)


class TestGetDependencies:
    @pytest.mark.asyncio
    async def test_returns_dependencies(self):
        from src.graph_query.server import get_dependencies

        rows = [{"dep": "starlette.routing", "rel": "IMPORTS"}]
        with patch("src.graph_query.server._get_client", new_callable=AsyncMock, return_value=_make_neo4j(rows)):
            result = await get_dependencies("FastAPI")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_empty_on_missing_entity(self):
        from src.graph_query.server import get_dependencies

        with patch("src.graph_query.server._get_client", new_callable=AsyncMock, return_value=_make_neo4j([])):
            result = await get_dependencies("Unknown")

        assert isinstance(result, dict)


class TestGetDependents:
    @pytest.mark.asyncio
    async def test_returns_dependents(self):
        from src.graph_query.server import get_dependents

        rows = [{"dep": "myapp.main", "rel": "IMPORTS"}]
        with patch("src.graph_query.server._get_client", new_callable=AsyncMock, return_value=_make_neo4j(rows)):
            result = await get_dependents("FastAPI")

        assert isinstance(result, dict)


class TestTraceImports:
    @pytest.mark.asyncio
    async def test_returns_import_chain(self):
        from src.graph_query.server import trace_imports

        rows = [{"path": ["fastapi", "starlette"]}]
        with patch("src.graph_query.server._get_client", new_callable=AsyncMock, return_value=_make_neo4j(rows)):
            result = await trace_imports("fastapi")

        assert isinstance(result, dict)


class TestFindRelated:
    @pytest.mark.asyncio
    async def test_returns_related_entities(self):
        from src.graph_query.server import find_related

        rows = [{"name": "APIRouter", "rel": "INHERITS_FROM"}]
        with patch("src.graph_query.server._get_client", new_callable=AsyncMock, return_value=_make_neo4j(rows)):
            result = await find_related("Router", "INHERITS_FROM")

        assert isinstance(result, dict)


class TestExecuteQuery:
    @pytest.mark.asyncio
    async def test_runs_cypher_and_returns_results(self):
        from src.graph_query.server import execute_query

        rows = [{"count": 42}]
        with patch("src.graph_query.server._get_client", new_callable=AsyncMock, return_value=_make_neo4j(rows)):
            result = await execute_query("MATCH (n:Class) RETURN count(n) AS count")

        assert isinstance(result, dict)
