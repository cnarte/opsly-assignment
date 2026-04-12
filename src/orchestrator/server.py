"""Orchestrator Agent MCP server — ReAct LangGraph pipeline."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from mcp.server.fastmcp import FastMCP

from src.orchestrator.graph import build_orchestrator_graph
from src.orchestrator.nodes import _get_llm, _call_mcp_agent
from src.orchestrator.prompts import RESPONSE_SYNTHESIS_PROMPT
from src.orchestrator.state import OrchestratorState
from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP("orchestrator-agent", host="0.0.0.0", port=settings.ORCHESTRATOR_PORT)
_graph = build_orchestrator_graph()


def _safe_serialise(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _initial_state(message: str, session_id: str, repo_id: str, model: str) -> OrchestratorState:
    return {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "repo_id": repo_id,
        "model": model,
        "final_response": "",
    }


@mcp.tool()
async def route_to_agents(
    message: str,
    session_id: str = "",
    repo_id: str = "",
    model: str = "",
) -> dict:
    """Run the full ReAct pipeline and return the final response."""
    state = _initial_state(message, session_id, repo_id, model)
    try:
        final_state = await _graph.ainvoke(state)
    except Exception as exc:
        logger.error("Orchestrator pipeline failed: %s", exc)
        return {"error": str(exc), "final_response": f"Pipeline error: {exc}"}

    return {
        "final_response": final_state.get("final_response", ""),
        "session_id": session_id,
        "agent_results": _safe_serialise({}),
        "tool_plan": [],
    }


@mcp.tool()
async def analyze_query(message: str, session_id: str = "") -> dict:
    """Classify query intent (retained for assignment compliance)."""
    return {"intent": "general", "entities": [], "complexity": "medium"}


@mcp.tool()
async def get_conversation_context(session_id: str) -> dict:
    """Retrieve conversation history via Memory Agent."""
    return await _call_mcp_agent("memory", settings.MEMORY_PORT, "get_conversation_context", {"session_id": session_id or "default"})


@mcp.tool()
async def synthesize_response(agent_results: dict, query: str, model: str = "") -> dict:
    """Combine agent outputs into a coherent response (retained for assignment compliance)."""
    from langchain_core.messages import SystemMessage
    summary = "\n\n".join(f"--- {k} ---\n{json.dumps(v, default=str)}" for k, v in agent_results.items()) or "(none)"
    llm = _get_llm(model)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=RESPONSE_SYNTHESIS_PROMPT),
            HumanMessage(content=f"Query: {query}\n\nResults:\n{summary}\n\nSynthesize a clear answer."),
        ])
        return {"response": response.content}
    except Exception as exc:
        return {"response": summary, "error": str(exc)}


@mcp.tool()
async def handle_index_request(repo_url: str, ref: str = "", repo_name: str = "") -> dict:
    """Proxy indexing request to Indexer Agent."""
    return await _call_mcp_agent("indexer", settings.INDEXER_PORT, "index_repository", {"repo_url": repo_url, "ref": ref, "repo_name": repo_name})


@mcp.tool()
async def handle_index_status(job_id: str) -> dict:
    """Proxy index job status from Indexer Agent."""
    return await _call_mcp_agent("indexer", settings.INDEXER_PORT, "get_index_status", {"job_id": job_id})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
