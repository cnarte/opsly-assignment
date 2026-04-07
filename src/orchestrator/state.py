"""Orchestrator state schema for LangGraph."""

from __future__ import annotations

import operator
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
    query_classification: dict
    agent_plan: list[str]
    agent_results: Annotated[dict[str, Any], _merge_dicts]
    conversation_context: list
    final_response: str
    session_id: str
