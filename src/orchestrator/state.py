"""Orchestrator state for the ReAct LangGraph agent."""
from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class OrchestratorState(TypedDict):
    """State flowing through the ReAct orchestrator loop."""

    # Conversation messages (human + AI + tool calls/results)
    messages: Annotated[list, add_messages]

    # Session info
    session_id: str
    repo_id: str   # empty = all repos
    model: str     # empty = use settings.OPENROUTER_MODEL

    # Final synthesised answer (set when ReAct loop ends)
    final_response: str
