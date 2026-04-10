"""LangGraph node functions for the Orchestrator Agent."""

from __future__ import annotations

import hashlib
import json
import logging
import re
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

# Tools that are allowed on each agent (used for fallback validation)
_GRAPH_QUERY_TOOLS = frozenset({
    "find_entity", "get_dependencies", "get_dependents",
    "trace_imports", "find_related", "execute_query",
    "get_symbol_context", "analyze_impact",
})
_CODE_ANALYST_TOOLS = frozenset({
    "explain_implementation", "analyze_function", "analyze_class",
    "get_code_snippet", "find_patterns", "compare_implementations",
})


def _get_llm() -> ChatOpenRouter:
    """Return a ChatOpenRouter instance configured from settings."""
    return ChatOpenRouter(
        model=settings.OPENROUTER_MODEL,
        openrouter_api_key=settings.OPENROUTER_API_KEY,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# MCP call helper
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

    timeout_s = getattr(settings, "MCP_CALL_TIMEOUT_S", 120)
    retries = getattr(settings, "MCP_CALL_RETRIES", 0)

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with asyncio.timeout(timeout_s):
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
        except asyncio.TimeoutError as exc:
            last_error = exc
            logger.warning(
                "MCP call to %s/%s timed out (attempt %d/%d)",
                agent_name, tool_name, attempt + 1, retries + 1,
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "MCP call to %s/%s failed (attempt %d/%d): %s",
                agent_name, tool_name, attempt + 1, retries + 1, exc,
            )
            break  # non-timeout errors are not retried

    if isinstance(last_error, asyncio.TimeoutError):
        return {"error": f"{agent_name}/{tool_name} timed out"}
    return {"error": f"{agent_name} unavailable: {last_error}"}


# ---------------------------------------------------------------------------
# State extraction helpers
# ---------------------------------------------------------------------------


def _extract_query(state: OrchestratorState) -> str:
    """Extract the user query text from state messages."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (
            isinstance(msg, dict) and msg.get("role") == "user"
        ):
            return msg.content if hasattr(msg, "content") else msg.get("content", "")
    return ""


def _extract_entities(state: OrchestratorState) -> list[str]:
    """Return the entity list from classification."""
    return state.get("query_classification", {}).get("entities", [])


def _heuristic_terms(query: str, n: int = 3) -> list[str]:
    """Extract identifier-like tokens from a free-text query as a fallback."""
    stop = {
        "the", "what", "how", "does", "show", "find", "all", "and",
        "from", "for", "are", "with", "this", "that", "which",
        "explain", "compare", "used", "into", "inherits", "inherit",
        "module", "function", "class", "classes", "functions",
        "does", "use", "where", "about", "give", "list", "get",
    }
    candidates = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", query)
    return [c for c in candidates if c.lower() not in stop][:n]


# ---------------------------------------------------------------------------
# Node: check_cache
# ---------------------------------------------------------------------------


async def check_cache(state: OrchestratorState) -> dict[str, Any]:
    """Check Redis cache for this query before running the full pipeline."""
    query = _extract_query(state)
    session_id = state.get("session_id", "") or ""
    repo_id = state.get("repo_id", "") or ""
    raw = (query[:200] + session_id + repo_id).encode()
    cache_key = hashlib.sha256(raw).hexdigest()[:16]

    result = await _call_mcp_agent(
        "memory",
        settings.MEMORY_PORT,
        "get_cached_response",
        {"cache_key": cache_key},
    )

    if isinstance(result, dict) and result.get("status") == "hit":
        logger.info("Cache hit for key %s", cache_key)
        return {
            "cache_key": cache_key,
            "final_response": result.get("response", ""),
        }

    return {"cache_key": cache_key}


# ---------------------------------------------------------------------------
# Node: classify_query
# ---------------------------------------------------------------------------


_LIFECYCLE_KEYWORDS = frozenset({
    "lifecycle", "life cycle", "life-cycle", "request flow", "request lifecycle",
    "how does", "360", "symbol context", "what calls", "who calls",
    "full flow", "end to end", "end-to-end",
})

_ENTITY_HINTS = ["fastapi", "apirouter", "request", "response", "starlette"]


def _fast_entity_extract(query: str) -> str:
    """Pull the most likely entity name from a query without an LLM call."""
    q = query.lower()
    for hint in _ENTITY_HINTS:
        if hint in q:
            # Return capitalised form from original query if possible
            import re
            m = re.search(re.escape(hint), query, re.IGNORECASE)
            return m.group(0) if m else hint.capitalize()
    # Fallback: first capitalised word
    import re
    words = re.findall(r"[A-Z][a-zA-Z0-9_]+", query)
    return words[0] if words else "FastAPI"


async def classify_query(state: OrchestratorState) -> dict[str, Any]:
    """Use the LLM to classify the user query by intent, entities, and tool plan."""

    query = _extract_query(state)
    if not query:
        return {
            "query_classification": {
                "intent": "general",
                "entities": [],
                "complexity": "simple",
            },
            "tool_plan": [],
        }

    # Fast-path: skip the LLM classifier entirely for lifecycle/flow queries.
    # These queries reliably need get_symbol_context, and the classifier LLM
    # call wastes ~30-60s we can't afford with a free model.
    q_lower = query.lower()
    if any(kw in q_lower for kw in _LIFECYCLE_KEYWORDS):
        primary = _fast_entity_extract(query)
        logger.info("Fast-path lifecycle: get_symbol_context(%s), skipping classifier LLM", primary)
        return {
            "query_classification": {
                "intent": "relationship_query",
                "entities": [primary],
                "complexity": "medium",
            },
            "tool_plan": [
                {"agent": "graph_query", "tool": "get_symbol_context",
                 "args": {"symbol_name": primary}},
            ],
        }

    llm = _get_llm()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=QUERY_CLASSIFICATION_PROMPT),
            HumanMessage(content=query),
        ])
        raw = json.loads(response.content)
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Classification failed (%s), falling back to general.", exc)
        raw = {}

    intent = raw.get("intent", "general")
    entities = raw.get("entities", [])
    complexity = raw.get("complexity", "simple")

    # Validate and sanitise tool_plan
    tool_plan: list[dict] = []
    for entry in raw.get("tool_plan", []):
        if not isinstance(entry, dict):
            continue
        agent = entry.get("agent", "")
        tool = entry.get("tool", "")
        args = entry.get("args", {})
        if (
            agent in {"graph_query", "code_analyst"}
            and isinstance(tool, str) and tool
            and isinstance(args, dict)
        ):
            tool_plan.append({"agent": agent, "tool": tool, "args": args})

    return {
        "query_classification": {
            "intent": intent,
            "entities": entities if isinstance(entities, list) else [],
            "complexity": complexity,
        },
        "tool_plan": tool_plan,
    }


# ---------------------------------------------------------------------------
# Node: plan_agents
# ---------------------------------------------------------------------------


async def plan_agents(state: OrchestratorState) -> dict[str, Any]:
    """Decide which child agents to invoke based on the tool plan / classification."""

    tool_plan = state.get("tool_plan", [])
    if tool_plan:
        # Derive the unique set of agents from the tool plan, preserving order
        seen: set[str] = set()
        agents: list[str] = []
        for entry in tool_plan:
            a = entry.get("agent", "")
            if a and a not in seen:
                seen.add(a)
                agents.append(a)
    else:
        intent = state.get("query_classification", {}).get("intent", "general")
        agents = list(AGENT_SELECTION_MAP.get(intent, ["graph_query", "code_analyst"]))

    # Always fetch conversation context from memory
    if "memory" not in agents:
        agents = ["memory"] + agents

    return {"agent_plan": agents}


# ---------------------------------------------------------------------------
# Node: call_graph_query
# ---------------------------------------------------------------------------


async def call_graph_query(state: OrchestratorState) -> dict[str, Any]:
    """Invoke the Graph Query Agent MCP server."""

    query = _extract_query(state)
    entities = _extract_entities(state)
    tool_plan = state.get("tool_plan", [])
    repo_id = state.get("repo_id", "") or ""

    results: dict[str, Any] = {}

    # Filter steps for this agent
    steps = [s for s in tool_plan if s.get("agent") == "graph_query"]

    if steps:
        for step in steps:
            tool = step.get("tool", "find_entity")
            args = {**step.get("args", {}), "repo_id": repo_id}

            # Only call known tools
            if tool not in _GRAPH_QUERY_TOOLS:
                logger.warning("Unknown graph_query tool in plan: %s — skipping", tool)
                continue

            result = await _call_mcp_agent(
                "graph_query", settings.GRAPH_QUERY_PORT, tool, args,
            )

            results[tool] = result
            # If the planned tool fails, also try find_entity as a fallback
            if "error" in result and tool != "find_entity" and entities:
                logger.info("graph_query/%s failed, falling back to find_entity", tool)
                fallback = await _call_mcp_agent(
                    "graph_query", settings.GRAPH_QUERY_PORT,
                    "find_entity", {"name": entities[0], "repo_id": repo_id},
                )
                results[f"{tool}_fallback"] = fallback
    else:
        # No tool plan — heuristic fallback: find_entity for each term
        lookup_terms = entities[:3] or _heuristic_terms(query)
        for term in lookup_terms:
            results[f"entity_{term}"] = await _call_mcp_agent(
                "graph_query", settings.GRAPH_QUERY_PORT,
                "find_entity", {"name": term, "repo_id": repo_id},
            )

    current_results = dict(state.get("agent_results", {}))
    current_results["graph_query"] = results
    return {"agent_results": current_results}


# ---------------------------------------------------------------------------
# Node: call_code_analyst
# ---------------------------------------------------------------------------


async def call_code_analyst(state: OrchestratorState) -> dict[str, Any]:
    """Invoke the Code Analyst Agent MCP server."""

    entities = _extract_entities(state)
    intent = state.get("query_classification", {}).get("intent", "general")
    tool_plan = state.get("tool_plan", [])

    results: dict[str, Any] = {}

    # Filter steps for this agent
    steps = [s for s in tool_plan if s.get("agent") == "code_analyst"]

    if steps:
        for step in steps:
            tool = step.get("tool", "explain_implementation")
            args = step.get("args", {})

            if tool not in _CODE_ANALYST_TOOLS:
                logger.warning("Unknown code_analyst tool in plan: %s — skipping", tool)
                continue

            result = await _call_mcp_agent(
                "code_analyst", settings.CODE_ANALYST_PORT, tool, args,
            )

            results[tool] = result
            # If the planned tool fails, also try explain_implementation as fallback
            if "error" in result and tool != "explain_implementation" and entities:
                logger.info("code_analyst/%s failed, falling back to explain_implementation", tool)
                fallback = await _call_mcp_agent(
                    "code_analyst", settings.CODE_ANALYST_PORT,
                    "explain_implementation", {"entity_name": entities[0]},
                )
                results[f"{tool}_fallback"] = fallback
    else:
        # No tool plan — intent-based fallback (original behaviour)
        if intent == "comparison" and len(entities) >= 2:
            results["comparison"] = await _call_mcp_agent(
                "code_analyst", settings.CODE_ANALYST_PORT,
                "compare_implementations",
                {"entity_a": entities[0], "entity_b": entities[1]},
            )
        elif intent == "pattern_analysis":
            results["patterns"] = await _call_mcp_agent(
                "code_analyst", settings.CODE_ANALYST_PORT,
                "find_patterns",
                {"code_path": entities[0] if entities else "fastapi", "pattern_type": ""},
            )
        elif entities:
            results["explanation"] = await _call_mcp_agent(
                "code_analyst", settings.CODE_ANALYST_PORT,
                "explain_implementation",
                {"entity_name": entities[0]},
            )
        else:
            results["skipped"] = {
                "reason": "no entities extracted from query",
            }

    current_results = dict(state.get("agent_results", {}))
    current_results["code_analyst"] = results
    return {"agent_results": current_results}


# ---------------------------------------------------------------------------
# Node: call_indexer
# ---------------------------------------------------------------------------


async def call_indexer(state: OrchestratorState) -> dict[str, Any]:
    """Invoke the Indexer Agent MCP server (for index requests)."""

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
    """Invoke the Memory Agent MCP server to fetch conversation context, preferences, and facts."""

    session_id = state.get("session_id", "") or "default"
    query = _extract_query(state)

    # 1. Short-term: recent conversation turns from Redis
    ctx_result = await _call_mcp_agent(
        "memory",
        settings.MEMORY_PORT,
        "get_conversation_context",
        {"session_id": session_id},
    )
    context = ctx_result.get("messages", []) if isinstance(ctx_result, dict) else []

    # 2. Long-term: semantic search over past interactions in Neo4j
    search_result = await _call_mcp_agent(
        "memory",
        settings.MEMORY_PORT,
        "search_memory",
        {"query": query, "session_id": session_id, "limit": 5},
    )

    # 3. User preferences / facts stored in Neo4j
    prefs_result = await _call_mcp_agent(
        "memory",
        settings.MEMORY_PORT,
        "get_user_preferences",
        {"session_id": session_id},
    )

    current_results = dict(state.get("agent_results", {}))
    current_results["memory"] = {
        "long_term_facts": search_result.get("results", []) if isinstance(search_result, dict) else [],
        "preferences": prefs_result.get("preferences", []) if isinstance(prefs_result, dict) else [],
    }
    return {"conversation_context": context, "agent_results": current_results}


# ---------------------------------------------------------------------------
# Node: synthesize
# ---------------------------------------------------------------------------


def _summarise_agent_result(agent_name: str, result: dict) -> str:
    """Convert an agent result dict into a compact, token-efficient summary."""
    lines: list[str] = []

    for tool_name, tool_result in result.items():
        if not isinstance(tool_result, dict):
            lines.append(f"{tool_name}: {str(tool_result)[:200]}")
            continue

        if "error" in tool_result:
            lines.append(f"{tool_name}: error — {tool_result['error']}")
            continue

        # get_symbol_context → compact relationship summary
        if "symbol" in tool_result and ("outgoing" in tool_result or "incoming" in tool_result):
            symbol = tool_result["symbol"]
            out = tool_result.get("outgoing", [])
            inc = tool_result.get("incoming", [])
            out_str = ", ".join(
                f"{r.get('related_name','')} ({r.get('relationship') or r.get('rel','')})"
                for r in out[:10]
            )
            inc_str = ", ".join(
                f"{r.get('related_name','')} ({r.get('relationship') or r.get('rel','')})"
                for r in inc[:15]
            )
            lines.append(
                f"{tool_name}: symbol={symbol}\n"
                f"  outgoing ({len(out)}): {out_str or 'none'}\n"
                f"  incoming ({len(inc)} total, showing 15): {inc_str or 'none'}"
            )
            continue

        # find_entity / find_related → list names only
        if "results" in tool_result:
            results = tool_result["results"]
            entity = tool_result.get("entity", "")
            rel = tool_result.get("relationship", "")
            names = []
            for r in results[:20]:
                name = (r.get("n") or r.get("node") or r.get("source") or r.get("target") or {})
                if isinstance(name, dict):
                    name = name.get("name", "?")
                names.append(str(name))
            header = f"{tool_name}"
            if entity:
                header += f" entity={entity}"
            if rel:
                header += f" rel={rel}"
            lines.append(f"{header}: {len(results)} results — {', '.join(names)}")
            continue

        # get_dependencies / get_dependents
        if "dependencies" in tool_result or "dependents" in tool_result:
            items = tool_result.get("dependencies") or tool_result.get("dependents") or []
            names = [i.get("name", str(i)) for i in items[:20]]
            key = "dependencies" if "dependencies" in tool_result else "dependents"
            lines.append(f"{tool_name} {key} ({len(items)}): {', '.join(names)}")
            continue

        # code_analyst tools — keep snippet or explanation, capped at 600 chars
        raw = json.dumps(tool_result, default=str)
        if len(raw) > 600:
            raw = raw[:600] + "…"
        lines.append(f"{tool_name}: {raw}")

    return "\n".join(lines) if lines else json.dumps(result, default=str)[:800]


async def synthesize(state: OrchestratorState) -> dict[str, Any]:
    """Use the LLM to combine all agent results into a final response."""

    query = _extract_query(state)
    agent_results = state.get("agent_results", {})

    parts: list[str] = []
    for agent_name, result in agent_results.items():
        if agent_name == "memory":
            continue
        summary = _summarise_agent_result(agent_name, result)
        parts.append(f"--- {agent_name} ---\n{summary}")
    agent_summary = "\n\n".join(parts) if parts else "(no agent results)"

    # Include at most last 2 conversation turns, capped at 400 chars each
    context_msgs = state.get("conversation_context", [])
    context_text = ""
    if context_msgs:
        recent = context_msgs[-2:]
        trimmed = []
        for m in recent:
            s = json.dumps(m, default=str)
            trimmed.append(s[:400] + ("…" if len(s) > 400 else ""))
        context_text = "\n\nRecent context:\n" + "\n".join(trimmed)

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
        final = f"I found the following results but could not synthesise them:\n\n{agent_summary}"

    return {"final_response": final}


# ---------------------------------------------------------------------------
# Node: persist_interaction
# ---------------------------------------------------------------------------


async def persist_interaction(state: OrchestratorState) -> dict[str, Any]:
    """Persist the Q&A pair to Memory and cache the response for future hits."""

    query = _extract_query(state)
    final_response = state.get("final_response", "")
    session_id = state.get("session_id", "") or "default"
    cache_key = state.get("cache_key", "")
    agent_plan = state.get("agent_plan", [])

    if not query or not final_response:
        return {}

    agents_used = [a for a in agent_plan if a != "memory"]

    # Store Q&A in Redis (short-term) and Neo4j (long-term)
    try:
        await _call_mcp_agent(
            "memory",
            settings.MEMORY_PORT,
            "store_interaction",
            {
                "session_id": session_id,
                "query": query,
                "response": final_response[:2000],
                "agents_used": agents_used,
            },
        )
    except Exception as exc:
        logger.warning("persist_interaction: store_interaction failed: %s", exc)

    # Cache the response so repeated identical queries are fast
    if cache_key:
        try:
            await _call_mcp_agent(
                "memory",
                settings.MEMORY_PORT,
                "cache_response",
                {
                    "cache_key": cache_key,
                    "response": final_response[:2000],
                    "ttl": 3600,
                },
            )
        except Exception as exc:
            logger.warning("persist_interaction: cache_response failed: %s", exc)

    return {}
