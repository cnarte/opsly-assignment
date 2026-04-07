"""LangGraph node functions for the Orchestrator Agent."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from src.orchestrator.prompts import (
    AGENT_SELECTION_MAP,
    QUERY_CLASSIFICATION_PROMPT,
    RESPONSE_SYNTHESIS_PROMPT,
)
from src.orchestrator.state import OrchestratorState
from src.shared.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()


def _get_llm() -> ChatOpenRouter:
    """Return a ChatOpenRouter instance configured from settings."""
    return ChatOpenRouter(
        model=settings.OPENROUTER_MODEL,
        openrouter_api_key=settings.OPENROUTER_API_KEY,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Node: classify_query
# ---------------------------------------------------------------------------


async def classify_query(state: OrchestratorState) -> dict[str, Any]:
    """Use the LLM to classify the user query by intent, entities, and complexity."""

    # Grab the latest human message as the query
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            query = msg.content if hasattr(msg, "content") else msg.get("content", "")
            break

    if not query:
        return {
            "query_classification": {
                "intent": "general",
                "entities": [],
                "complexity": "simple",
            }
        }

    llm = _get_llm()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=QUERY_CLASSIFICATION_PROMPT),
            HumanMessage(content=query),
        ])
        classification = json.loads(response.content)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Classification failed (%s), falling back to general.", exc)
        classification = {
            "intent": "general",
            "entities": [],
            "complexity": "simple",
        }

    # Ensure required keys exist
    classification.setdefault("intent", "general")
    classification.setdefault("entities", [])
    classification.setdefault("complexity", "simple")

    return {"query_classification": classification}


# ---------------------------------------------------------------------------
# Node: plan_agents
# ---------------------------------------------------------------------------


async def plan_agents(state: OrchestratorState) -> dict[str, Any]:
    """Decide which child agents to invoke based on the classification."""

    intent = state.get("query_classification", {}).get("intent", "general")
    plan = AGENT_SELECTION_MAP.get(intent, ["code_analyst"])
    return {"agent_plan": plan}


# ---------------------------------------------------------------------------
# MCP call helpers
# ---------------------------------------------------------------------------


async def _call_mcp_agent(
    agent_name: str,
    port: int,
    tool_name: str,
    tool_args: dict[str, Any],
) -> dict[str, Any]:
    """Call an MCP tool on a child agent via the raw MCP SDK client.

    Uses Docker service names when running inside Docker, else localhost.
    """
    import asyncio
    import os
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    _PORT_TO_SERVICE: dict[int, str] = {
        settings.ORCHESTRATOR_PORT: "orchestrator",
        settings.INDEXER_PORT: "indexer",
        settings.GRAPH_QUERY_PORT: "graph-query",
        settings.CODE_ANALYST_PORT: "code-analyst",
        settings.MEMORY_PORT: "memory",
    }

    if os.path.exists("/.dockerenv"):
        host = _PORT_TO_SERVICE.get(port, "localhost")
    else:
        host = "localhost"
    url = f"http://{host}:{port}/mcp"

    try:
        async with asyncio.timeout(120):
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, tool_args)

                    if result.content:
                        text = result.content[0].text
                        try:
                            return json.loads(text)
                        except (json.JSONDecodeError, TypeError):
                            return {"result": text}
                    return {}
    except asyncio.TimeoutError:
        logger.error("MCP call to %s/%s timed out", agent_name, tool_name)
        return {"error": f"{agent_name}/{tool_name} timed out"}
    except Exception as exc:
        logger.warning("MCP call to %s/%s failed: %s", agent_name, tool_name, exc)
        return {"error": f"{agent_name} unavailable: {exc}"}


def _extract_query(state: OrchestratorState) -> str:
    """Extract the user query text from state messages."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            return msg.content if hasattr(msg, "content") else msg.get("content", "")
    return ""


def _extract_entities(state: OrchestratorState) -> list[str]:
    """Return the entity list from classification."""
    return state.get("query_classification", {}).get("entities", [])


# ---------------------------------------------------------------------------
# Node: call_graph_query
# ---------------------------------------------------------------------------


async def call_graph_query(state: OrchestratorState) -> dict[str, Any]:
    """Invoke the Graph Query Agent MCP server."""

    query = _extract_query(state)
    entities = _extract_entities(state)

    results: dict[str, Any] = {}

    # Build lookup terms: prefer extracted entities, else pull capitalized
    # words / identifier-like tokens from the query as a heuristic fallback.
    lookup_terms = list(entities[:3])
    if not lookup_terms:
        import re
        candidates = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", query)
        stop = {"the", "what", "how", "does", "show", "find", "all", "and",
                "from", "for", "are", "with", "this", "that", "which",
                "explain", "compare", "used", "into", "inherits", "inherit",
                "module", "function", "class", "classes", "functions"}
        lookup_terms = [c for c in candidates if c.lower() not in stop][:3]

    for term in lookup_terms:
        entity_result = await _call_mcp_agent(
            "graph_query",
            settings.GRAPH_QUERY_PORT,
            "find_entity",
            {"name": term},
        )
        results[f"entity_{term}"] = entity_result

    current_results = dict(state.get("agent_results", {}))
    current_results["graph_query"] = results
    return {"agent_results": current_results}


# ---------------------------------------------------------------------------
# Node: call_code_analyst
# ---------------------------------------------------------------------------


async def call_code_analyst(state: OrchestratorState) -> dict[str, Any]:
    """Invoke the Code Analyst Agent MCP server."""

    query = _extract_query(state)
    entities = _extract_entities(state)
    intent = state.get("query_classification", {}).get("intent", "general")

    results: dict[str, Any] = {}

    if intent == "comparison" and len(entities) >= 2:
        cmp = await _call_mcp_agent(
            "code_analyst",
            settings.CODE_ANALYST_PORT,
            "compare_implementations",
            {"entity_a": entities[0], "entity_b": entities[1]},
        )
        results["comparison"] = cmp
    elif intent == "pattern_analysis":
        pat = await _call_mcp_agent(
            "code_analyst",
            settings.CODE_ANALYST_PORT,
            "find_patterns",
            {"code_path": entities[0] if entities else "fastapi", "pattern_type": ""},
        )
        results["patterns"] = pat
    elif entities:
        # Default: explain the first entity
        expl = await _call_mcp_agent(
            "code_analyst",
            settings.CODE_ANALYST_PORT,
            "explain_implementation",
            {"entity_name": entities[0]},
        )
        results["explanation"] = expl
    else:
        # No entities extracted — skip code_analyst rather than pass the full
        # query as an entity_name (which just produces "entity not found").
        results["skipped"] = {
            "reason": "no entities extracted from query",
            "query": query,
        }

    current_results = dict(state.get("agent_results", {}))
    current_results["code_analyst"] = results
    return {"agent_results": current_results}


# ---------------------------------------------------------------------------
# Node: call_indexer
# ---------------------------------------------------------------------------


async def call_indexer(state: OrchestratorState) -> dict[str, Any]:
    """Invoke the Indexer Agent MCP server (for index requests)."""

    query = _extract_query(state)

    result = await _call_mcp_agent(
        "indexer",
        settings.INDEXER_PORT,
        "index_repository",
        {"repo_url": settings.DEFAULT_REPO_URL, "ref": settings.DEFAULT_REPO_REF},
    )

    current_results = dict(state.get("agent_results", {}))
    current_results["indexer"] = result
    return {"agent_results": current_results}


# ---------------------------------------------------------------------------
# Node: call_memory
# ---------------------------------------------------------------------------


async def call_memory(state: OrchestratorState) -> dict[str, Any]:
    """Invoke the Memory Agent MCP server to fetch conversation context."""

    # Memory agent tools are expected to be defined in Phase 5.
    # For now, attempt the call; if the agent is unavailable, return empty.
    session_id = state.get("session_id", "") or ""

    result = await _call_mcp_agent(
        "memory",
        settings.MEMORY_PORT,
        "get_session_history",
        {"session_id": session_id or "default"},
    )

    context = result.get("messages", []) if isinstance(result, dict) else []
    return {"conversation_context": context}


# ---------------------------------------------------------------------------
# Node: synthesize
# ---------------------------------------------------------------------------


async def synthesize(state: OrchestratorState) -> dict[str, Any]:
    """Use the LLM to combine all agent results into a final response."""

    query = _extract_query(state)
    agent_results = state.get("agent_results", {})

    # Build a summary of agent outputs for the LLM
    parts: list[str] = []
    for agent_name, result in agent_results.items():
        parts.append(f"--- {agent_name} ---\n{json.dumps(result, indent=2, default=str)}")
    agent_summary = "\n\n".join(parts) if parts else "(no agent results)"

    context_msgs = state.get("conversation_context", [])
    context_text = ""
    if context_msgs:
        context_text = "\n\nPrior conversation context:\n" + json.dumps(
            context_msgs[-5:], indent=2, default=str
        )

    user_prompt = (
        f"User query: {query}\n\n"
        f"Agent results:\n{agent_summary}"
        f"{context_text}\n\n"
        "Synthesise a clear, helpful answer."
    )

    llm = _get_llm()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=RESPONSE_SYNTHESIS_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        final = response.content
    except Exception as exc:
        logger.error("Synthesis LLM call failed: %s", exc)
        # Fallback: just dump the raw results
        final = f"I found the following results but could not synthesise them:\n\n{agent_summary}"

    return {"final_response": final}
