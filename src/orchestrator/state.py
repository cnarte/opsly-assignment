"""Orchestrator state schema for LangGraph."""

from __future__ import annotations

from typing import Any, Annotated

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge two dicts — used as a LangGraph reducer for agent_results."""
    merged = dict(left)
    merged.update(right)
    return merged


class OrchestratorState(TypedDict):
    """Typed state flowing through the orchestrator graph."""

    messages: Annotated[list, add_messages]
    # Classification output
    query_classification: dict
    # High-level list of agent names to invoke
    agent_plan: list[str]
    # Detailed tool-call plan produced by the classifier LLM
    tool_plan: list[dict]
    # Accumulated results keyed by agent name
    agent_results: Annotated[dict[str, Any], _merge_dicts]
    # Conversation history fetched from Memory agent
    conversation_context: list
    # Final synthesised answer
    final_response: str
    # Session identifier for memory / cache
    session_id: str
    # SHA-256 cache key for the current query+session
    cache_key: str
    # Optional repository scope (e.g. "fastapi/fastapi") — empty = all repos
    repo_id: str
