"""Tests for the ReAct orchestrator loop."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage


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
