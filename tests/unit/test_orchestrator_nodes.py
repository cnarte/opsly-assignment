"""Unit tests for orchestrator nodes — all MCP calls mocked."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_STATE: dict[str, Any] = {
    "messages": [HumanMessage(content="What is FastAPI?")],
    "query_classification": {"intent": "code_explanation", "entities": ["FastAPI"], "complexity": "simple"},
    "agent_plan": ["graph_query", "code_analyst"],
    "tool_plan": [],
    "agent_results": {},
    "conversation_context": [],
    "final_response": "",
    "session_id": "sess-123",
    "cache_key": "",
}


def _state(**overrides: Any) -> dict[str, Any]:
    s = dict(_BASE_STATE)
    s.update(overrides)
    return s


# ---------------------------------------------------------------------------
# classify_query
# ---------------------------------------------------------------------------


class TestClassifyQuery:
    @pytest.mark.asyncio
    async def test_classifies_with_llm(self):
        """classify_query parses the LLM response into classification + tool_plan."""
        import json
        from src.orchestrator.nodes import classify_query

        fake_classification = {
            "intent": "code_explanation",
            "entities": ["FastAPI"],
            "complexity": "medium",
            "tool_plan": [
                {"agent": "graph_query", "tool": "find_entity", "args": {"name": "FastAPI"}},
                {"agent": "code_analyst", "tool": "explain_implementation", "args": {"entity_name": "FastAPI"}},
            ],
        }

        mock_response = AsyncMock()
        mock_response.content = json.dumps(fake_classification)

        with patch("src.orchestrator.nodes._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            result = await classify_query(_state())

        assert result["query_classification"]["intent"] == "code_explanation"
        assert "FastAPI" in result["query_classification"]["entities"]
        assert len(result["tool_plan"]) == 2
        assert result["tool_plan"][0]["tool"] == "find_entity"

    @pytest.mark.asyncio
    async def test_fallback_on_bad_json(self):
        """classify_query falls back to general when LLM returns invalid JSON."""
        from src.orchestrator.nodes import classify_query

        mock_response = AsyncMock()
        mock_response.content = "not json at all"

        with patch("src.orchestrator.nodes._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            result = await classify_query(_state())

        assert result["query_classification"]["intent"] == "general"
        assert result["tool_plan"] == []

    @pytest.mark.asyncio
    async def test_filters_unknown_agents_from_tool_plan(self):
        """classify_query strips tool_plan entries with unknown agent names."""
        import json
        from src.orchestrator.nodes import classify_query

        bad_plan = {
            "intent": "general",
            "entities": [],
            "complexity": "simple",
            "tool_plan": [
                {"agent": "evil_agent", "tool": "rm_rf", "args": {}},
                {"agent": "graph_query", "tool": "find_entity", "args": {"name": "X"}},
            ],
        }

        mock_response = AsyncMock()
        mock_response.content = json.dumps(bad_plan)

        with patch("src.orchestrator.nodes._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            result = await classify_query(_state())

        assert len(result["tool_plan"]) == 1
        assert result["tool_plan"][0]["agent"] == "graph_query"

    @pytest.mark.asyncio
    async def test_empty_query_returns_general(self):
        """Empty message → general intent, no LLM call."""
        from src.orchestrator.nodes import classify_query

        empty_state = _state(messages=[])

        with patch("src.orchestrator.nodes._get_llm") as mock_llm_fn:
            result = await classify_query(empty_state)
            mock_llm_fn.assert_not_called()

        assert result["query_classification"]["intent"] == "general"


# ---------------------------------------------------------------------------
# plan_agents
# ---------------------------------------------------------------------------


class TestPlanAgents:
    @pytest.mark.asyncio
    async def test_uses_tool_plan_to_derive_agents(self):
        """plan_agents derives agent list from tool_plan order."""
        from src.orchestrator.nodes import plan_agents

        s = _state(
            tool_plan=[
                {"agent": "graph_query", "tool": "find_entity", "args": {}},
                {"agent": "code_analyst", "tool": "explain_implementation", "args": {}},
            ]
        )
        result = await plan_agents(s)
        # memory is always prepended
        assert result["agent_plan"][0] == "memory"
        assert "graph_query" in result["agent_plan"]
        assert "code_analyst" in result["agent_plan"]

    @pytest.mark.asyncio
    async def test_fallback_to_selection_map_when_no_tool_plan(self):
        """plan_agents uses AGENT_SELECTION_MAP when tool_plan is empty."""
        from src.orchestrator.nodes import plan_agents

        s = _state(
            tool_plan=[],
            query_classification={"intent": "comparison", "entities": [], "complexity": "simple"},
        )
        result = await plan_agents(s)
        assert "graph_query" in result["agent_plan"]

    @pytest.mark.asyncio
    async def test_memory_always_included(self):
        """plan_agents always prepends memory to the agent plan."""
        from src.orchestrator.nodes import plan_agents

        result = await plan_agents(_state(tool_plan=[]))
        assert "memory" in result["agent_plan"]


# ---------------------------------------------------------------------------
# call_graph_query
# ---------------------------------------------------------------------------


class TestCallGraphQuery:
    @pytest.mark.asyncio
    async def test_executes_tool_plan_steps(self):
        """call_graph_query executes all graph_query steps in tool_plan."""
        from src.orchestrator.nodes import call_graph_query

        s = _state(
            tool_plan=[
                {"agent": "graph_query", "tool": "find_entity", "args": {"name": "FastAPI"}},
                {"agent": "graph_query", "tool": "get_dependents", "args": {"entity_name": "FastAPI"}},
                {"agent": "code_analyst", "tool": "explain_implementation", "args": {"entity_name": "FastAPI"}},  # filtered out
            ]
        )

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"result": "ok"}
            result = await call_graph_query(s)

        # only the 2 graph_query steps should have been called
        assert mock_call.call_count == 2
        calls = [(c.args[2]) for c in mock_call.call_args_list]
        assert "find_entity" in calls
        assert "get_dependents" in calls
        assert result["agent_results"]["graph_query"]["find_entity"] == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_heuristic_fallback_without_tool_plan(self):
        """call_graph_query falls back to find_entity when no tool_plan."""
        from src.orchestrator.nodes import call_graph_query

        s = _state(
            tool_plan=[],
            messages=[HumanMessage(content="Tell me about APIRouter")],
            query_classification={"intent": "general", "entities": ["APIRouter"], "complexity": "simple"},
        )

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"name": "APIRouter"}
            result = await call_graph_query(s)

        mock_call.assert_called_once()
        assert mock_call.call_args.args[2] == "find_entity"

    @pytest.mark.asyncio
    async def test_fallback_on_tool_error(self):
        """call_graph_query falls back to find_entity when a planned tool returns an error."""
        from src.orchestrator.nodes import call_graph_query

        s = _state(
            tool_plan=[
                {"agent": "graph_query", "tool": "get_dependents", "args": {"entity_name": "FastAPI"}},
            ],
            query_classification={"intent": "general", "entities": ["FastAPI"], "complexity": "simple"},
        )

        async def _fake_call(agent, port, tool, args):
            if tool == "get_dependents":
                return {"error": "tool failed"}
            return {"name": "FastAPI"}

        with patch("src.orchestrator.nodes._call_mcp_agent", side_effect=_fake_call):
            result = await call_graph_query(s)

        gq = result["agent_results"]["graph_query"]
        # Original failed call recorded
        assert gq["get_dependents"] == {"error": "tool failed"}
        # Fallback also recorded
        assert "get_dependents_fallback" in gq

    @pytest.mark.asyncio
    async def test_skips_unknown_tools(self):
        """call_graph_query skips tools not in the allowed set."""
        from src.orchestrator.nodes import call_graph_query

        s = _state(
            tool_plan=[
                {"agent": "graph_query", "tool": "secret_tool", "args": {}},
            ]
        )

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            await call_graph_query(s)

        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# call_code_analyst
# ---------------------------------------------------------------------------


class TestCallCodeAnalyst:
    @pytest.mark.asyncio
    async def test_executes_tool_plan_steps(self):
        """call_code_analyst executes all code_analyst steps in tool_plan."""
        from src.orchestrator.nodes import call_code_analyst

        s = _state(
            tool_plan=[
                {"agent": "graph_query", "tool": "find_entity", "args": {}},  # filtered
                {"agent": "code_analyst", "tool": "explain_implementation", "args": {"entity_name": "FastAPI"}},
                {"agent": "code_analyst", "tool": "get_code_snippet", "args": {"entity_name": "FastAPI"}},
            ]
        )

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"ok": True}
            result = await call_code_analyst(s)

        assert mock_call.call_count == 2
        assert "explain_implementation" in result["agent_results"]["code_analyst"]

    @pytest.mark.asyncio
    async def test_comparison_fallback(self):
        """call_code_analyst calls compare_implementations for comparison intent without tool_plan."""
        from src.orchestrator.nodes import call_code_analyst

        s = _state(
            tool_plan=[],
            query_classification={"intent": "comparison", "entities": ["Path", "Query"], "complexity": "medium"},
        )

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"diff": "..."}
            result = await call_code_analyst(s)

        assert mock_call.call_args.args[2] == "compare_implementations"
        assert result["agent_results"]["code_analyst"]["comparison"] == {"diff": "..."}

    @pytest.mark.asyncio
    async def test_skips_when_no_entities_and_no_plan(self):
        """call_code_analyst skips gracefully when no entities and no tool_plan."""
        from src.orchestrator.nodes import call_code_analyst

        s = _state(
            tool_plan=[],
            query_classification={"intent": "general", "entities": [], "complexity": "simple"},
        )

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            result = await call_code_analyst(s)

        mock_call.assert_not_called()
        assert "skipped" in result["agent_results"]["code_analyst"]


# ---------------------------------------------------------------------------
# call_memory
# ---------------------------------------------------------------------------


class TestCallMemory:
    @pytest.mark.asyncio
    async def test_calls_correct_tool_name(self):
        """call_memory must call get_conversation_context first, then search_memory + get_user_preferences."""
        from src.orchestrator.nodes import call_memory

        ctx_response = {"status": "ok", "messages": [{"role": "user", "content": "hi"}]}
        search_response = {"results": []}
        prefs_response = {"preferences": []}

        call_responses = [ctx_response, search_response, prefs_response]

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = call_responses
            result = await call_memory(_state())

        # First call must be get_conversation_context
        first_tool = mock_call.call_args_list[0].args[2]
        assert first_tool == "get_conversation_context", (
            f"Expected get_conversation_context first but got {first_tool}"
        )
        assert result["conversation_context"] == [{"role": "user", "content": "hi"}]
        assert "memory" in result["agent_results"]

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        """call_memory returns empty context when memory agent is unavailable."""
        from src.orchestrator.nodes import call_memory

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"error": "unavailable"}
            result = await call_memory(_state())

        assert result["conversation_context"] == []


# ---------------------------------------------------------------------------
# persist_interaction
# ---------------------------------------------------------------------------


class TestPersistInteraction:
    @pytest.mark.asyncio
    async def test_calls_store_interaction(self):
        """persist_interaction calls store_interaction on memory agent."""
        from src.orchestrator.nodes import persist_interaction

        s = _state(
            final_response="FastAPI is a web framework.",
            agent_plan=["memory", "graph_query"],
            cache_key="abc123",
        )

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"status": "ok"}
            await persist_interaction(s)

        called_tools = [c.args[2] for c in mock_call.call_args_list]
        assert "store_interaction" in called_tools
        assert "cache_response" in called_tools

    @pytest.mark.asyncio
    async def test_skips_when_no_response(self):
        """persist_interaction is a no-op when final_response is empty."""
        from src.orchestrator.nodes import persist_interaction

        s = _state(final_response="")

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            await persist_interaction(s)

        mock_call.assert_not_called()


# ---------------------------------------------------------------------------
# check_cache
# ---------------------------------------------------------------------------


class TestCheckCache:
    @pytest.mark.asyncio
    async def test_cache_hit_sets_final_response(self):
        """check_cache returns final_response from cache on hit."""
        from src.orchestrator.nodes import check_cache

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"status": "hit", "response": "cached answer"}
            result = await check_cache(_state())

        assert result["final_response"] == "cached answer"
        assert result["cache_key"]  # key was set

    @pytest.mark.asyncio
    async def test_cache_miss_sets_key_only(self):
        """check_cache only sets cache_key on miss (no final_response)."""
        from src.orchestrator.nodes import check_cache

        with patch("src.orchestrator.nodes._call_mcp_agent", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"status": "miss", "response": None}
            result = await check_cache(_state())

        assert result.get("final_response", "") == ""
        assert result["cache_key"]


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_calls_llm_and_returns_response(self):
        """synthesize invokes the LLM and returns final_response."""
        from src.orchestrator.nodes import synthesize

        s = _state(
            agent_results={"graph_query": {"entity_FastAPI": {"name": "FastAPI"}}},
        )

        mock_response = AsyncMock()
        mock_response.content = "FastAPI is a modern Python web framework."

        with patch("src.orchestrator.nodes._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_fn.return_value = mock_llm

            result = await synthesize(s)

        assert result["final_response"] == "FastAPI is a modern Python web framework."

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """synthesize falls back to raw results when LLM errors."""
        from src.orchestrator.nodes import synthesize

        s = _state(
            agent_results={"graph_query": {"entity_FastAPI": {"name": "FastAPI"}}},
        )

        with patch("src.orchestrator.nodes._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
            mock_llm_fn.return_value = mock_llm

            result = await synthesize(s)

        assert "FastAPI" in result["final_response"]

    @pytest.mark.asyncio
    async def test_includes_conversation_context(self):
        """synthesize includes prior conversation in the LLM prompt."""
        from src.orchestrator.nodes import synthesize

        s = _state(
            conversation_context=[{"role": "user", "content": "What is Starlette?"}],
        )

        captured_prompt = {}

        async def _fake_invoke(msgs):
            captured_prompt["content"] = msgs[-1].content
            r = AsyncMock()
            r.content = "ok"
            return r

        with patch("src.orchestrator.nodes._get_llm") as mock_llm_fn:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = _fake_invoke
            mock_llm_fn.return_value = mock_llm
            await synthesize(s)

        assert "Starlette" in captured_prompt["content"]
