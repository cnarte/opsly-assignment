"""Orchestrator state schema for LangGraph."""

from __future__ import annotations

from typing import Any, Annotated

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict):
    """Typed state flowing through the orchestrator graph.

    Attributes:
        messages: Chat history managed by LangGraph's ``add_messages`` reducer.
        query_classification: LLM-produced dict with keys ``intent``, ``entities``,
            and ``complexity``.
        agent_plan: Ordered list of agent names to invoke (e.g.
            ``["graph_query", "code_analyst"]``).
        agent_results: Mapping of ``agent_name -> result`` collected during
            execution.
        conversation_context: Prior conversation turns retrieved from the
            Memory Agent.
        final_response: Synthesised natural-language answer returned to the
            caller.
    """

    messages: Annotated[list, add_messages]
    query_classification: dict
    agent_plan: list[str]
    agent_results: dict[str, Any]
    conversation_context: list
    final_response: str
