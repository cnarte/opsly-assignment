"""Unit tests for FastAPI gateway routes — MCP client mocked throughout."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.gateway.app import app


# ---------------------------------------------------------------------------
# Shared async client fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# /api/chat
# ---------------------------------------------------------------------------


ORCHESTRATOR_RESPONSE = {
    "query_classification": {"intent": "code_explanation", "entities": ["FastAPI"]},
    "agent_plan": ["memory", "graph_query", "code_analyst"],
    "tool_plan": [],
    "agent_results": {"graph_query": {"entity_FastAPI": {"name": "FastAPI"}}},
    "final_response": "FastAPI is a web framework for Python.",
}


class TestChatRoutes:
    @pytest.mark.asyncio
    async def test_chat_returns_200_and_response(self, client):
        with patch("src.gateway.routes.chat.call_orchestrator_tool", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ORCHESTRATOR_RESPONSE
            resp = await client.post(
                "/api/chat",
                json={"message": "What is FastAPI?"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "session_id" in data
        assert data["response"] == "FastAPI is a web framework for Python."

    @pytest.mark.asyncio
    async def test_chat_passes_session_id(self, client):
        with patch("src.gateway.routes.chat.call_orchestrator_tool", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ORCHESTRATOR_RESPONSE
            resp = await client.post(
                "/api/chat",
                json={"message": "Tell me more", "session_id": "my-session"},
            )

        assert resp.status_code == 200
        # Verify session_id was forwarded to orchestrator
        call_args = mock_call.call_args
        assert call_args.args[1]["session_id"] == "my-session"

    @pytest.mark.asyncio
    async def test_chat_empty_message_passes_to_orchestrator(self, client):
        """Empty message string is forwarded to orchestrator (no gateway-level rejection)."""
        with patch("src.gateway.routes.chat.call_orchestrator_tool", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ORCHESTRATOR_RESPONSE
            resp = await client.post("/api/chat", json={"message": ""})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_rejects_missing_message(self, client):
        resp = await client.post("/api/chat", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_handles_orchestrator_error(self, client):
        with patch("src.gateway.routes.chat.call_orchestrator_tool", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"error": "orchestrator unavailable", "final_response": ""}
            resp = await client.post(
                "/api/chat",
                json={"message": "What is FastAPI?"},
            )

        # Should still return 200 (graceful degradation), not 500
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/agents/health
# ---------------------------------------------------------------------------


class TestHealthRoute:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        with patch("src.gateway.routes.health.check_agent_health", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = "healthy"
            resp = await client.get("/api/agents/health")

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "agents" in data

    @pytest.mark.asyncio
    async def test_health_degraded_when_agent_unhealthy(self, client):
        async def _maybe_unhealthy(name, port):
            return "unhealthy" if name == "memory" else "healthy"

        with patch("src.gateway.routes.health.check_agent_health", side_effect=_maybe_unhealthy):
            resp = await client.get("/api/agents/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["agents"]["memory"] == "unhealthy"


# ---------------------------------------------------------------------------
# /api/index
# ---------------------------------------------------------------------------


class TestIndexRoutes:
    @pytest.mark.asyncio
    async def test_trigger_index_returns_job_id(self, client):
        with patch("src.gateway.routes.index.call_orchestrator_tool", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"job_id": "job-xyz", "status": "started"}
            resp = await client.post(
                "/api/index",
                json={"repo_url": "https://github.com/fastapi/fastapi.git"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-xyz"
        assert data["status"] == "started"

    @pytest.mark.asyncio
    async def test_index_status_calls_indexer(self, client):
        with patch("src.gateway.routes.index.call_agent_tool", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"job_id": "job-xyz", "status": "completed"}
            resp = await client.get("/api/index/status/job-xyz")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert mock_call.called
        call_args = mock_call.call_args
        assert call_args.args[1] == "get_index_status"


# ---------------------------------------------------------------------------
# /api/graph
# ---------------------------------------------------------------------------


class TestGraphRoutes:
    @pytest.mark.asyncio
    async def test_graph_statistics_returns_200_on_neo4j_error(self, client):
        """graph_statistics returns zero-count GraphStats on Neo4j failure (graceful)."""
        with patch("src.gateway.routes.graph.Neo4jClient") as mock_neo4j_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.connect = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client.execute_query = AsyncMock(side_effect=Exception("Neo4j down"))
            mock_neo4j_cls.return_value = mock_client

            resp = await client.get("/api/graph/statistics")

        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == 0
