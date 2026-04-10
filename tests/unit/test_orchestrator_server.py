"""Unit tests for orchestrator MCP server tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# analyze_query tool
# ---------------------------------------------------------------------------


class TestAnalyzeQuery:
    @pytest.mark.asyncio
    async def test_returns_classification(self):
        """analyze_query returns a classification dict."""
        from src.orchestrator.server import analyze_query
        import json

        fake_cls = {
            "intent": "code_explanation",
            "entities": ["FastAPI"],
            "complexity": "simple",
            "tool_plan": [],
        }
        mock_response = AsyncMock()
        mock_response.content = json.dumps(fake_cls)

        with patch("src.orchestrator.nodes._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            result = await analyze_query("What is FastAPI?")

        assert result["intent"] == "code_explanation"
        assert "FastAPI" in result.get("entities", [])

    @pytest.mark.asyncio
    async def test_returns_general_on_empty(self):
        """analyze_query with empty message returns general fallback."""
        from src.orchestrator.server import analyze_query
        import json

        mock_response = AsyncMock()
        mock_response.content = json.dumps({"intent": "general", "entities": [], "complexity": "simple", "tool_plan": []})

        with patch("src.orchestrator.nodes._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            result = await analyze_query("")

        assert "intent" in result


# ---------------------------------------------------------------------------
# route_to_agents tool
# ---------------------------------------------------------------------------


class TestRouteToAgents:
    def _graph_result(self, final_response: str = "test response") -> dict:
        return {
            "query_classification": {"intent": "general", "entities": []},
            "agent_plan": ["memory", "graph_query"],
            "tool_plan": [],
            "agent_results": {"graph_query": {}},
            "conversation_context": [],
            "final_response": final_response,
            "session_id": "",
            "cache_key": "abc",
        }

    @pytest.mark.asyncio
    async def test_returns_final_response(self):
        """route_to_agents runs the full pipeline and returns final_response."""
        from src.orchestrator.server import route_to_agents

        with patch("src.orchestrator.server._graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value=self._graph_result("FastAPI is great."))
            result = await route_to_agents("What is FastAPI?")

        assert result["final_response"] == "FastAPI is great."
        assert "agent_plan" in result
        assert "tool_plan" in result
        assert "conversation_context" in result

    @pytest.mark.asyncio
    async def test_handles_pipeline_error(self):
        """route_to_agents returns error dict on pipeline exception."""
        from src.orchestrator.server import route_to_agents

        with patch("src.orchestrator.server._graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph failed"))
            result = await route_to_agents("What is FastAPI?")

        assert "error" in result
        assert "final_response" in result


# ---------------------------------------------------------------------------
# get_conversation_context tool
# ---------------------------------------------------------------------------


class TestGetConversationContext:
    @pytest.mark.asyncio
    async def test_calls_memory_with_correct_tool_name(self):
        """get_conversation_context calls memory MCP with get_conversation_context."""
        from src.orchestrator.server import get_conversation_context

        with patch("src.orchestrator.server._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"status": "ok", "messages": []}
            result = await get_conversation_context("sess-1")

        called_tool = mock_call.call_args.args[2]
        assert called_tool == "get_conversation_context"

    @pytest.mark.asyncio
    async def test_defaults_session_to_default(self):
        """get_conversation_context uses 'default' when session_id is empty."""
        from src.orchestrator.server import get_conversation_context

        with patch("src.orchestrator.server._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {}
            await get_conversation_context("")

        called_args = mock_call.call_args.args[3]
        assert called_args["session_id"] == "default"


# ---------------------------------------------------------------------------
# handle_index_status tool
# ---------------------------------------------------------------------------


class TestHandleIndexStatus:
    @pytest.mark.asyncio
    async def test_proxies_to_indexer(self):
        """handle_index_status calls get_index_status on the indexer agent."""
        from src.orchestrator.server import handle_index_status

        with patch("src.orchestrator.server._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"job_id": "j1", "status": "completed"}
            result = await handle_index_status("j1")

        assert result["status"] == "completed"
        called_tool = mock_call.call_args.args[2]
        assert called_tool == "get_index_status"


# ---------------------------------------------------------------------------
# synthesize_response tool
# ---------------------------------------------------------------------------


class TestSynthesizeResponse:
    @pytest.mark.asyncio
    async def test_calls_llm_and_returns_response(self):
        """synthesize_response uses the LLM to combine agent results."""
        from src.orchestrator.server import synthesize_response

        mock_response = AsyncMock()
        mock_response.content = "A synthesised answer."

        with patch("src.orchestrator.server._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            result = await synthesize_response(
                {"graph_query": {"entity_FastAPI": {"name": "FastAPI"}}},
                "What is FastAPI?",
            )

        assert result["response"] == "A synthesised answer."

    @pytest.mark.asyncio
    async def test_returns_raw_on_llm_error(self):
        """synthesize_response falls back to raw results when LLM errors."""
        from src.orchestrator.server import synthesize_response

        with patch("src.orchestrator.server._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM error"))
            mock_llm_fn.return_value = mock_llm

            result = await synthesize_response({"agent": "data"}, "query")

        assert "response" in result
        assert "error" in result
