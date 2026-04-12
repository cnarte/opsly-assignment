"""ReAct node functions for the orchestrator LangGraph."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from src.orchestrator.prompts import REACT_SYSTEM_PROMPT, RESPONSE_SYNTHESIS_PROMPT
from src.orchestrator.state import OrchestratorState
from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _get_llm(model: str = "") -> ChatOpenRouter:
    """Return a ChatOpenRouter, using model override when provided."""
    return ChatOpenRouter(
        model=model or settings.OPENROUTER_MODEL,
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
    timeout: int | None = None,
) -> dict[str, Any]:
    """Call an MCP tool via streamable-http transport."""
    import asyncio
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    _PORT_TO_SERVICE: dict[int, str] = {
        settings.ORCHESTRATOR_PORT: "orchestrator",
        settings.INDEXER_PORT: "indexer",
        settings.GRAPH_QUERY_PORT: "graph-query",
        settings.CODE_ANALYST_PORT: "code-analyst",
        settings.MEMORY_PORT: "memory",
        settings.GITNEXUS_PORT: "gitnexus-agent",
    }

    host = _PORT_TO_SERVICE.get(port, "localhost") if os.path.exists("/.dockerenv") else "localhost"
    url = f"http://{host}:{port}/mcp"
    timeout_s = timeout or getattr(settings, "MCP_CALL_TIMEOUT_S", 30)

    try:
        async with asyncio.timeout(timeout_s):
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, tool_args)
                    if result.content:
                        try:
                            return json.loads(result.content[0].text)
                        except (json.JSONDecodeError, TypeError):
                            return {"result": result.content[0].text}
                    return {}
    except Exception as exc:
        logger.warning("MCP call %s/%s failed: %s", agent_name, tool_name, exc)
        return {"error": f"{agent_name}/{tool_name} failed: {exc}"}


# ---------------------------------------------------------------------------
# Build LangChain tools from MCP agents
# ---------------------------------------------------------------------------


def _make_mcp_tool(agent: str, port: int, name: str, description: str, schema: dict):
    """Create a LangChain StructuredTool that calls an MCP agent."""
    from langchain_core.tools import StructuredTool
    import pydantic

    fields = {}
    for k, v in schema.items():
        if v.get("type") == "string":
            fields[k] = (str, pydantic.Field(default="", description=v.get("description", "")))
        elif v.get("type") == "integer":
            fields[k] = (int, pydantic.Field(default=v.get("default", 0), description=v.get("description", "")))
        else:
            fields[k] = (Any, pydantic.Field(default=None))
    ArgsModel = pydantic.create_model(f"{name}_args", **fields)

    async def _run(**kwargs: Any) -> str:
        result = await _call_mcp_agent(agent, port, name, {k: v for k, v in kwargs.items() if v not in (None, "")})
        return json.dumps(result, default=str)

    return StructuredTool(
        name=name,
        description=description,
        args_schema=ArgsModel,
        coroutine=_run,
    )


def build_tools(repo_id: str = "", model: str = "") -> list:
    """Build the full tool list for the ReAct agent."""
    tools = []

    gq_port = settings.GRAPH_QUERY_PORT

    for name, desc, schema in [
        ("find_entity", "Locate a class, function, or module by name using hybrid search.",
         {"name": {"type": "string", "description": "Entity name"}, "entity_type": {"type": "string", "description": "Optional: Class, Function, Method"}, "repo_id": {"type": "string", "description": "Repo scope"}}),
        ("get_dependencies", "Find what an entity depends on (outgoing relationships).",
         {"entity_name": {"type": "string", "description": "Entity name"}, "repo_id": {"type": "string"}}),
        ("get_dependents", "Find what depends on an entity (incoming relationships).",
         {"entity_name": {"type": "string", "description": "Entity name"}, "repo_id": {"type": "string"}}),
        ("trace_imports", "Follow the import chain for a module.",
         {"module_name": {"type": "string"}, "repo_id": {"type": "string"}}),
        ("find_related", "Get entities related by a specific relationship type (CALLS, INHERITS_FROM, IMPORTS, DECORATED_BY).",
         {"entity_name": {"type": "string"}, "relationship_type": {"type": "string"}, "repo_id": {"type": "string"}}),
        ("execute_query", "Run a raw Cypher query against LadybugDB (read-only).",
         {"cypher": {"type": "string", "description": "Cypher query string"}}),
        ("get_symbol_context", "360-degree view of a symbol: callers, callees, imports, process participation. BEST tool for lifecycle/flow questions.",
         {"symbol_name": {"type": "string"}, "repo_id": {"type": "string"}}),
        ("analyze_impact", "Blast-radius: what breaks if this symbol changes.",
         {"symbol_name": {"type": "string"}, "depth": {"type": "integer", "default": 2}, "repo_id": {"type": "string"}}),
        ("list_entities", "List all entities of a type (Function, Class, Method, Module, File).",
         {"entity_type": {"type": "string"}, "limit": {"type": "integer", "default": 50}, "repo_id": {"type": "string"}}),
    ]:
        tools.append(_make_mcp_tool("graph_query", gq_port, name, desc, schema))

    ca_port = settings.CODE_ANALYST_PORT

    for name, desc, schema in [
        ("explain_implementation", "LLM-generated plain-English explanation of how an entity works.",
         {"entity_name": {"type": "string"}, "model": {"type": "string"}}),
        ("analyze_function", "Deep analysis of a function's logic.",
         {"function_name": {"type": "string"}, "repo_path": {"type": "string"}, "model": {"type": "string"}}),
        ("analyze_class", "Comprehensive analysis of a class and its methods.",
         {"class_name": {"type": "string"}, "repo_path": {"type": "string"}, "model": {"type": "string"}}),
        ("get_code_snippet", "Raw source code with surrounding context lines.",
         {"entity_name": {"type": "string"}, "context_lines": {"type": "integer", "default": 5}, "repo_id": {"type": "string"}}),
        ("find_patterns", "Detect design patterns in a module or entity.",
         {"code_path": {"type": "string"}, "pattern_type": {"type": "string"}}),
        ("compare_implementations", "LLM comparison of two code entities side-by-side.",
         {"entity_a": {"type": "string"}, "entity_b": {"type": "string"}, "model": {"type": "string"}}),
    ]:
        tools.append(_make_mcp_tool("code_analyst", ca_port, name, desc, schema))

    return tools


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def inject_history(state: OrchestratorState) -> dict:
    """Prepend conversation history from memory agent to state messages."""
    session_id = state.get("session_id", "")
    current = list(state.get("messages", []))

    if not session_id:
        return {"messages": current}

    history_result = await _call_mcp_agent(
        "memory", settings.MEMORY_PORT,
        "get_conversation_context", {"session_id": session_id},
        timeout=10,
    )

    prior: list = history_result.get("context", []) or []
    history_messages = []
    for turn in prior[-10:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    if not history_messages:
        return {"messages": current}

    return {"messages": history_messages + current}


async def react_agent(state: OrchestratorState) -> dict:
    """Single ReAct step: LLM decides next tool call or final answer."""
    model = state.get("model", "")
    repo_id = state.get("repo_id", "")

    tools = build_tools(repo_id=repo_id, model=model)
    llm = _get_llm(model).bind_tools(tools)

    messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + list(state.get("messages", []))
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


async def persist_turn(state: OrchestratorState) -> dict:
    """Store this conversation turn in the memory agent."""
    session_id = state.get("session_id", "")
    if not session_id:
        return {}

    messages = state.get("messages", [])
    user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    ai_msg = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)), "")

    if user_msg and ai_msg:
        await _call_mcp_agent(
            "memory", settings.MEMORY_PORT,
            "store_interaction",
            {"session_id": session_id, "user_message": user_msg, "assistant_response": ai_msg},
            timeout=10,
        )
    return {"final_response": ai_msg}
