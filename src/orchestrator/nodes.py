"""ReAct node functions for the orchestrator LangGraph."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from src.orchestrator.prompts import REACT_SYSTEM_PROMPT, RESPONSE_SYNTHESIS_PROMPT
from src.orchestrator.state import OrchestratorState
from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


# ---------------------------------------------------------------------------
# Langfuse tracing helper
# ---------------------------------------------------------------------------


def get_langfuse_callback(session_id: str = "", user_id: str = ""):
    """Return a Langfuse LangChain callback handler, or None if not configured."""
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse.callback import CallbackHandler  # langfuse >= 3.x
        host = settings.LANGFUSE_HOST or settings.LANGFUSE_BASE_URL
        return CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=host,
            session_id=session_id or None,
            user_id=user_id or None,
        )
    except Exception:
        logger.warning("Langfuse callback unavailable — tracing disabled", exc_info=False)
        return None


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _get_llm(model: str = ""):
    """Return a ChatOpenRouter or ChatOpenAI (LM Studio), using model override when provided.

    Model IDs starting with 'lmstudio:' are routed to LM Studio's local server
    via its OpenAI-compatible API. All others use ChatOpenRouter.
    """
    resolved = model or settings.OPENROUTER_MODEL
    if resolved.startswith("lmstudio:"):
        lms_model = resolved.removeprefix("lmstudio:")
        return ChatOpenAI(
            model=lms_model,
            base_url=settings.LMSTUDIO_BASE_URL,
            api_key="lm-studio",
            temperature=0,
        )
    return ChatOpenRouter(
        model=resolved,
        openrouter_api_key=settings.OPENROUTER_API_KEY,
        temperature=0,
        streaming=True,
    )


async def _compress_tool_result(result: str, tool_name: str = "", max_chars: int = 8000) -> str:
    """LLM-summarise tool results that are too large for the agent context window.

    Results under max_chars pass through unchanged (no LLM call, no latency).
    On LLM failure, falls back to the first max_chars characters of the raw result.
    """
    if len(result) <= max_chars:
        return result

    prompt = (
        f"The following is the raw output of the '{tool_name}' tool. "
        "Summarise it concisely so the key information is preserved in under 500 words. "
        "Preserve file paths, counts, and names. Do not add commentary.\n\n"
        f"{result[:4000]}"
    )
    try:
        llm = _get_llm()
        response = await llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content="Summarise the above.")])
        return response.content
    except Exception as exc:
        logger.warning("Tool result compression failed for %s: %s", tool_name, exc)
        hint = (
            "\n\n[Result truncated — too large to summarise. "
            "Call list_entities_tree instead for a compact folder-grouped view.]"
            if "list_entities" in tool_name
            else "\n\n[Result truncated — too large to summarise.]"
        )
        return result[:max_chars] + hint


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
    import httpx
    from mcp.client.streamable_http import streamable_http_client
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
    # Default 90s — graph-query calls gitnexus-agent with a 60s read timeout, so
    # we need at least 90s here to avoid closing the connection while graph-query
    # is still waiting for gitnexus (which causes the ASGI ClosedResourceError).
    timeout_s = timeout or getattr(settings, "MCP_CALL_TIMEOUT_S", 90)

    try:
        async with asyncio.timeout(timeout_s):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, read=timeout_s),
                follow_redirects=True,
            ) as http_client:
                async with streamable_http_client(url, http_client=http_client) as (read, write, _):
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
        raw = json.dumps(result, default=str)
        return await _compress_tool_result(raw, tool_name=name)

    return StructuredTool(
        name=name,
        description=description,
        args_schema=ArgsModel,
        coroutine=_run,
    )


_tools_cache: dict[str, list] = {}


def build_tools(repo_id: str = "", model: str = "") -> list:
    """Build the full tool list for the ReAct agent (cached per repo_id)."""
    cache_key = repo_id or "__default__"
    if cache_key in _tools_cache:
        return _tools_cache[cache_key]
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
        ("list_entities_tree",
         "List all entities of a type grouped into a folder/file tree. "
         "Use this instead of list_entities for 'get all X' queries — returns a compact summary safe for large repos.",
         {"entity_type": {"type": "string", "description": "Function, Class, File, Folder"},
          "repo_id": {"type": "string"}}),
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

    _tools_cache[cache_key] = tools
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

    prior: list = history_result.get("messages", history_result.get("context", [])) or []
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
    import asyncio as _asyncio
    model = state.get("model", "")
    repo_id = state.get("repo_id", "")

    tools = build_tools(repo_id=repo_id, model=model)
    llm = _get_llm(model).bind_tools(tools)

    messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + list(state.get("messages", []))

    # Retry up to 4 times on transient errors from OpenRouter free models:
    #   524/timeout → short backoff (1s, 2s, 4s)
    #   429 rate-limit → longer backoff (10s, 20s, 40s)
    last_exc = None
    for attempt in range(4):
        try:
            response = await llm.ainvoke(messages)
            break
        except Exception as exc:
            err = str(exc)
            is_timeout = "524" in err or "timeout" in err.lower() or "timed out" in err.lower()
            is_rate_limit = "429" in err or "rate limit" in err.lower() or "too many requests" in err.lower()
            if is_rate_limit:
                last_exc = exc
                wait = 10 * (2 ** attempt)  # 10s, 20s, 40s, 80s
                logger.warning("LLM rate-limited (429), retrying in %ds [attempt %d/4]: %s", wait, attempt + 1, err[:120])
                try:
                    writer = get_stream_writer()
                    writer({"type": "retry", "reason": "rate_limit", "wait": wait, "attempt": attempt + 1})
                except Exception:
                    pass
                await _asyncio.sleep(wait)
                continue
            elif is_timeout:
                last_exc = exc
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                logger.warning("LLM timeout, retrying in %ds [attempt %d/4]: %s", wait, attempt + 1, err[:120])
                try:
                    writer = get_stream_writer()
                    writer({"type": "retry", "reason": "timeout", "wait": wait, "attempt": attempt + 1})
                except Exception:
                    pass
                await _asyncio.sleep(wait)
                continue
            raise
    else:
        raise last_exc
    return {"messages": [response]}


async def persist_turn(state: OrchestratorState) -> dict:
    """Store this conversation turn in the memory agent."""
    session_id = state.get("session_id", "")
    messages = state.get("messages", [])

    logger.info("persist_turn: %d messages in state", len(messages))
    for i, m in enumerate(messages):
        msg_type = type(m).__name__
        has_tool_calls = getattr(m, "tool_calls", None) is not None
        content_preview = str(m.content)[:60] if hasattr(m, "content") else "N/A"
        logger.info("  [%d] %s (tool_calls=%s): %s", i, msg_type, has_tool_calls, content_preview)

    if not session_id:
        logger.info("persist_turn: no session_id, returning empty")
        return {}

    user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    ai_msg = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)), "")

    logger.info("persist_turn RESULT: user_msg=%.50s ai_msg=%.50s", user_msg, ai_msg)

    if user_msg and ai_msg:
        await _call_mcp_agent(
            "memory", settings.MEMORY_PORT,
            "store_interaction",
            {"session_id": session_id, "query": user_msg, "response": ai_msg},
            timeout=10,
        )
    return {"final_response": ai_msg}
