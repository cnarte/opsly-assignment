"""Tests for the ReAct orchestrator loop."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


@pytest.mark.asyncio
async def test_inject_history_adds_messages():
    """inject_history should prepend prior conversation messages to state."""
    from src.orchestrator.nodes import inject_history

    mock_history = {
        "context": [
            {"role": "user", "content": "What is FastAPI?"},
            {"role": "assistant", "content": "FastAPI is a web framework."},
        ]
    }

    with patch("src.orchestrator.nodes._call_mcp_agent", AsyncMock(return_value=mock_history)):
        state = {
            "messages": [HumanMessage(content="How does routing work?")],
            "session_id": "test-session",
            "repo_id": "",
            "model": "",
            "final_response": "",
        }
        result = await inject_history(state)

    msgs = result["messages"]
    assert len(msgs) == 3
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "What is FastAPI?"
    assert isinstance(msgs[1], AIMessage)


@pytest.mark.asyncio
async def test_inject_history_handles_empty_history():
    """inject_history should work fine when memory agent returns no history."""
    from src.orchestrator.nodes import inject_history

    with patch("src.orchestrator.nodes._call_mcp_agent", AsyncMock(return_value={"context": []})):
        state = {
            "messages": [HumanMessage(content="Hello")],
            "session_id": "new-session",
            "repo_id": "",
            "model": "",
            "final_response": "",
        }
        result = await inject_history(state)

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Hello"


def test_get_llm_uses_override_model():
    """_get_llm should use the provided model string over the settings default."""
    from src.orchestrator.nodes import _get_llm

    with patch("src.orchestrator.nodes.ChatOpenRouter") as mock_cls:
        mock_cls.return_value = MagicMock()
        _get_llm("qwen/qwen-2.5-72b-instruct:free")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "qwen/qwen-2.5-72b-instruct:free"


def test_get_llm_uses_settings_default_when_empty():
    """_get_llm with empty string should fall back to settings.OPENROUTER_MODEL."""
    from src.orchestrator.nodes import _get_llm, settings

    with patch("src.orchestrator.nodes.ChatOpenRouter") as mock_cls:
        mock_cls.return_value = MagicMock()
        _get_llm("")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == settings.OPENROUTER_MODEL


def test_collect_partial_results_extracts_tool_messages():
    """_collect_partial_results should join ToolMessage content strings."""
    from src.orchestrator.server import _collect_partial_results

    msgs = [
        HumanMessage(content="how does routing work?"),
        AIMessage(content="", tool_calls=[{"id": "t1", "name": "get_symbol_context", "args": {}}]),
        ToolMessage(content='{"symbol": "APIRouter", "outgoing": []}', tool_call_id="t1"),
        AIMessage(content="", tool_calls=[{"id": "t2", "name": "find_entity", "args": {}}]),
        ToolMessage(content='{"results": [{"name": "add_api_route"}]}', tool_call_id="t2"),
    ]
    result = _collect_partial_results(msgs)
    assert '{"symbol": "APIRouter"' in result
    assert '{"results":' in result


def test_collect_partial_results_empty():
    """_collect_partial_results on a message list with no ToolMessages returns empty string."""
    from src.orchestrator.server import _collect_partial_results

    msgs = [HumanMessage(content="hi")]
    assert _collect_partial_results(msgs) == ""


@pytest.mark.asyncio
async def test_route_to_agents_handles_recursion_error():
    """route_to_agents should return a partial answer when GraphRecursionError is raised."""
    from langgraph.errors import GraphRecursionError
    from langchain_core.messages import ToolMessage

    mock_checkpoint = MagicMock()
    mock_checkpoint.values = {
        "messages": [
            HumanMessage(content="how does routing work?"),
            ToolMessage(content='{"symbol": "APIRouter"}', tool_call_id="t1"),
        ]
    }

    with patch("src.orchestrator.server._graph") as mock_graph, \
         patch("src.orchestrator.server._synthesize_partial", AsyncMock(return_value="Partial answer")) as mock_synth:

        mock_graph.ainvoke = AsyncMock(side_effect=GraphRecursionError("limit reached"))
        mock_graph.aget_state = AsyncMock(return_value=mock_checkpoint)

        from src.orchestrator import server as srv
        result = await srv.route_to_agents("how does routing work?", session_id="s1")

    assert "Partial answer" in result["final_response"]
    assert "partial exploration" in result["final_response"].lower()
    mock_synth.assert_awaited_once()
    # Verify the ToolMessage content was passed to synthesize
    call_args = mock_synth.call_args[0]
    assert '{"symbol": "APIRouter"}' in call_args[0]
