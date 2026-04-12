"""LangGraph StateGraph for the Orchestrator Agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.orchestrator.nodes import (
    call_code_analyst,
    call_graph_query,
    call_indexer,
    call_memory,
    check_cache,
    classify_query,
    persist_interaction,
    plan_agents,
    rewrite_query,
    synthesize,
)
from src.orchestrator.state import OrchestratorState

# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

_AGENT_NODE_MAP: dict[str, str] = {
    "graph_query": "call_graph_query",
    "code_analyst": "call_code_analyst",
    "indexer":      "call_indexer",
    "memory":       "call_memory",
}


def _route_after_cache(state: OrchestratorState) -> str:
    """Skip the full pipeline on a cache hit — go straight to END (don't re-persist)."""
    if state.get("final_response"):
        return "__end__"
    return "classify"


def _route_from_plan(state: OrchestratorState) -> list[str]:
    """Return the graph node names for every agent in the plan.

    LangGraph will fan-out to all returned nodes in parallel.  If the plan is
    empty we skip straight to synthesis.
    """
    plan = state.get("agent_plan", [])
    nodes = [_AGENT_NODE_MAP[a] for a in plan if a in _AGENT_NODE_MAP]
    return nodes if nodes else ["synthesize"]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_orchestrator_graph() -> Any:
    """Build and compile the orchestrator LangGraph."""

    graph = StateGraph(OrchestratorState)

    # -- nodes ---------------------------------------------------------------
    graph.add_node("check_cache",        check_cache)
    graph.add_node("rewrite",            rewrite_query)
    graph.add_node("classify",           classify_query)
    graph.add_node("plan",               plan_agents)
    graph.add_node("call_graph_query",   call_graph_query)
    graph.add_node("call_code_analyst",  call_code_analyst)
    graph.add_node("call_indexer",       call_indexer)
    graph.add_node("call_memory",        call_memory)
    graph.add_node("synthesize",         synthesize)
    graph.add_node("persist_interaction", persist_interaction)

    # -- edges ---------------------------------------------------------------
    graph.set_entry_point("check_cache")

    # If cache hit → go straight to persist (no-op) → END
    # If cache miss → rewrite query → classify
    graph.add_conditional_edges(
        "check_cache",
        _route_after_cache,
        {
            "classify":  "rewrite",
            "__end__":   END,
        },
    )

    graph.add_edge("rewrite", "classify")
    graph.add_edge("classify", "plan")

    # Conditional fan-out from plan to agent nodes (or straight to synthesize)
    graph.add_conditional_edges(
        "plan",
        _route_from_plan,
        {
            "call_graph_query":  "call_graph_query",
            "call_code_analyst": "call_code_analyst",
            "call_indexer":      "call_indexer",
            "call_memory":       "call_memory",
            "synthesize":        "synthesize",
        },
    )

    # All agent nodes converge on synthesize
    graph.add_edge("call_graph_query",  "synthesize")
    graph.add_edge("call_code_analyst", "synthesize")
    graph.add_edge("call_indexer",      "synthesize")
    graph.add_edge("call_memory",       "synthesize")

    # Synthesize → persist → END
    graph.add_edge("synthesize",          "persist_interaction")
    graph.add_edge("persist_interaction", END)

    return graph.compile()
