"""ReAct LangGraph for the orchestrator agent."""
from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from src.orchestrator.nodes import (
    build_tools,
    inject_history,
    persist_turn,
    react_agent,
)
from src.orchestrator.state import OrchestratorState


def _should_continue(state: OrchestratorState) -> Literal["tools", "persist"]:
    """Route to tool executor if LLM made tool calls, else end the loop."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if last and getattr(last, "tool_calls", None):
        return "tools"
    return "persist"


def build_orchestrator_graph() -> Any:
    """Build and compile the ReAct orchestrator graph."""
    tools = build_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(OrchestratorState)

    graph.add_node("inject_history", inject_history)
    graph.add_node("agent", react_agent)
    graph.add_node("tools", tool_node)
    graph.add_node("persist", persist_turn)

    graph.add_edge(START, "inject_history")
    graph.add_edge("inject_history", "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "persist": "persist"})
    graph.add_edge("tools", "agent")
    graph.add_edge("persist", END)

    # TODO: replace MemorySaver with a persistent checkpointer (e.g. PostgresSaver)
    # before production — MemorySaver is in-process only and grows unboundedly.
    return graph.compile(checkpointer=MemorySaver())
