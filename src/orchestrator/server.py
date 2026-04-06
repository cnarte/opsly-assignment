"""Orchestrator Agent MCP server -- LangGraph supervisor routing."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from mcp.server.fastmcp import FastMCP

from src.orchestrator.graph import build_orchestrator_graph
from src.orchestrator.nodes import (
    _call_mcp_agent,
    _get_llm,
    classify_query as _classify_node,
)
from src.orchestrator.prompts import QUERY_CLASSIFICATION_PROMPT, RESPONSE_SYNTHESIS_PROMPT
from src.orchestrator.state import OrchestratorState
from src.shared.settings import Settings

logger = logging.getLogger(__name__)

mcp = FastMCP("orchestrator-agent")

settings = Settings()

# Compile the graph once at module level
_graph = build_orchestrator_graph()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _initial_state(message: str, session_id: str = "") -> dict[str, Any]:
    """Build a fresh OrchestratorState dict for a new query."""
    return {
        "messages": [HumanMessage(content=message)],
        "query_classification": {},
        "agent_plan": [],
        "agent_results": {},
        "conversation_context": [],
        "final_response": "",
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def analyze_query(message: str, session_id: str = "") -> dict:
    """Classify query intent and extract key entities."""
    state = _initial_state(message, session_id)
    result = await _classify_node(state)
    return result.get("query_classification", {})


@mcp.tool()
async def route_to_agents(message: str, session_id: str = "") -> dict:
    """Determine which agents should handle the query and execute the full pipeline.

    This is the main entry point -- it runs the complete LangGraph pipeline
    (classify -> plan -> agent calls -> synthesize) and returns the final
    response together with intermediate results.
    """
    state = _initial_state(message, session_id)

    try:
        final_state = await _graph.ainvoke(state)
    except Exception as exc:
        logger.error("Orchestrator pipeline failed: %s", exc)
        return {
            "error": str(exc),
            "final_response": f"Pipeline error: {exc}",
        }

    return {
        "query_classification": final_state.get("query_classification", {}),
        "agent_plan": final_state.get("agent_plan", []),
        "agent_results": _safe_serialise(final_state.get("agent_results", {})),
        "final_response": final_state.get("final_response", ""),
    }


@mcp.tool()
async def get_conversation_context(session_id: str) -> dict:
    """Retrieve relevant conversation history via Memory Agent."""
    result = await _call_mcp_agent(
        "memory",
        settings.MEMORY_PORT,
        "get_session_history",
        {"session_id": session_id or "default"},
    )
    return result


@mcp.tool()
async def synthesize_response(agent_results: dict, query: str) -> dict:
    """Combine agent outputs into coherent response."""
    from langchain_core.messages import SystemMessage

    agent_summary_parts = []
    for name, data in agent_results.items():
        agent_summary_parts.append(
            f"--- {name} ---\n{json.dumps(data, indent=2, default=str)}"
        )
    agent_summary = "\n\n".join(agent_summary_parts) if agent_summary_parts else "(none)"

    llm = _get_llm()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=RESPONSE_SYNTHESIS_PROMPT),
            HumanMessage(
                content=f"User query: {query}\n\nAgent results:\n{agent_summary}\n\n"
                "Synthesise a clear, helpful answer."
            ),
        ])
        return {"response": response.content}
    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        return {"response": agent_summary, "error": str(exc)}


@mcp.tool()
async def handle_index_request(repo_url: str, ref: str = "") -> dict:
    """Route indexing request directly to Indexer Agent (no LLM needed)."""
    result = await _call_mcp_agent(
        "indexer",
        settings.INDEXER_PORT,
        "index_repository",
        {"repo_url": repo_url, "ref": ref},
    )
    return result


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def _safe_serialise(obj: Any) -> Any:
    """Make the object JSON-safe by converting non-serialisable types."""
    if isinstance(obj, dict):
        return {k: _safe_serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialise(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    mcp.run(transport="streamable-http", port=settings.ORCHESTRATOR_PORT)
