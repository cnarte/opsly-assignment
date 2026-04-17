"""Orchestrator state for the ReAct LangGraph agent."""

from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class OrchestratorState(TypedDict):
    """State flowing through the ReAct orchestrator loop."""

    messages: Annotated[list, add_messages]

    session_id: str
    repo_id: str
    model: str

    final_response: str

    tool_calls: list
