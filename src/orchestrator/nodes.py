"""ReAct node functions for the orchestrator LangGraph."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _msg_text(msg: AIMessage) -> str:
    """Extract plain text from an AIMessage.content.

    ChatAnthropic returns content as a list of content blocks
    [{'type': 'text', 'text': '...', 'index': 0}], while ChatOpenRouter
    and ChatOpenAI return a plain string. Normalise to string.
    """
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


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
    """Return a configured Langfuse client, or None if keys are not set.

    Uses Langfuse v3 direct SDK so it works with graph.astream_events().
    LangChain CallbackHandler conflicts with astream_events streaming in LangGraph.
    Reads from Settings (not os.environ) so .env file loading works correctly.
    """
    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        return None

    try:
        from langfuse import Langfuse

        base_url = settings.LANGFUSE_HOST or settings.LANGFUSE_BASE_URL or None
        client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=base_url,
        )
        client.auth_check()
        logger.info("Langfuse tracing enabled (session=%s)", session_id or "default")
        return client
    except Exception as exc:
        logger.warning("Langfuse client init failed: %s — tracing disabled", exc)
        return None


class _NoOpContext:
    """No-op context manager used when Langfuse is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _observe(session_id: str, name: str, input_data: dict, user_id: str = ""):
    """Return a Langfuse observe context manager, or a no-op if not configured."""
    client = get_langfuse_callback(session_id=session_id)
    if client is None:
        return _NoOpContext()

    metadata = {}
    if session_id:
        metadata["session_id"] = session_id
    if user_id:
        metadata["user_id"] = user_id

    try:
        return client.start_as_current_observation(
            name=name,
            input=input_data,
            metadata=metadata or None,
        )
    except Exception as exc:
        logger.warning("Langfuse observe failed: %s — continuing without tracing", exc)
        return _NoOpContext()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _get_llm(model: str = ""):
    from langchain_anthropic import ChatAnthropic

    if model:
        provider, _, model_name = model.partition(":")
        resolved_provider = provider or settings.ORCHESTRATOR_PROVIDER
        resolved_model = model_name or settings.ORCHESTRATOR_MODEL
    else:
        resolved_provider = settings.ORCHESTRATOR_PROVIDER
        resolved_model = settings.ORCHESTRATOR_MODEL

    if resolved_provider == "anthropic":
        return ChatAnthropic(
            model=resolved_model,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            temperature=0,
            streaming=True,
        )
    if resolved_provider == "lmstudio":
        return ChatOpenAI(
            model=resolved_model,
            base_url=settings.LMSTUDIO_BASE_URL,
            api_key="lm-studio",
            temperature=0,
        )
    return ChatOpenRouter(
        model=settings.OPENROUTER_MODEL,
        openrouter_api_key=settings.OPENROUTER_API_KEY,
        temperature=0,
        streaming=True,
    )


async def _compress_tool_result(
    result: str, tool_name: str = "", max_chars: int = 8000
) -> str:
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
        response = await llm.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content="Summarise the above."),
            ]
        )
        return _msg_text(response)
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

    start_time = time.time()

    _PORT_TO_SERVICE: dict[int, str] = {
        settings.ORCHESTRATOR_PORT: "orchestrator",
        settings.INDEXER_PORT: "indexer",
        settings.GRAPH_QUERY_PORT: "graph-query",
        settings.CODE_ANALYST_PORT: "code-analyst",
        settings.MEMORY_PORT: "memory",
        settings.GITNEXUS_PORT: "gitnexus-agent",
    }

    host = (
        _PORT_TO_SERVICE.get(port, "localhost")
        if os.path.exists("/.dockerenv")
        else "localhost"
    )
    url = f"http://{host}:{port}/mcp"
    # Default 90s — graph-query calls gitnexus-agent with a 60s read timeout, so
    # we need at least 90s here to avoid closing the connection while graph-query
    # is still waiting for gitnexus (which causes the ASGI ClosedResourceError).
    timeout_s = timeout or getattr(settings, "MCP_CALL_TIMEOUT_S", 300)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout_s, read=timeout_s, pool=timeout_s, write=timeout_s
            ),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        ) as http_client:
            async with streamable_http_client(url, http_client=http_client) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, tool_args)
                    duration_ms = int((time.time() - start_time) * 1000)
                    if result.content:
                        try:
                            return {
                                "_duration_ms": duration_ms,
                                **json.loads(result.content[0].text),
                            }
                        except (json.JSONDecodeError, TypeError):
                            return {
                                "_duration_ms": duration_ms,
                                "result": result.content[0].text,
                            }
                    return {"_duration_ms": duration_ms}
    except Exception as exc:
        logger.warning("MCP call %s/%s failed: %s", agent_name, tool_name, exc)
        return {
            "error": f"{agent_name}/{tool_name} failed: {exc}",
            "_duration_ms": int((time.time() - start_time) * 1000),
        }


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
            fields[k] = (
                str,
                pydantic.Field(default="", description=v.get("description", "")),
            )
        elif v.get("type") == "integer":
            fields[k] = (
                int,
                pydantic.Field(
                    default=v.get("default", 0), description=v.get("description", "")
                ),
            )
        else:
            fields[k] = (Any, pydantic.Field(default=None))
    ArgsModel = pydantic.create_model(f"{name}_args", **fields)

    async def _run(**kwargs: Any) -> str:
        result = await _call_mcp_agent(
            agent, port, name, {k: v for k, v in kwargs.items() if v not in (None, "")}
        )
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
        (
            "find_entity",
            "Locate a class, function, or module by name using hybrid search.",
            {
                "name": {"type": "string", "description": "Entity name"},
                "entity_type": {
                    "type": "string",
                    "description": "Optional: Class, Function, Method",
                },
                "repo_id": {"type": "string", "description": "Repo scope"},
            },
        ),
        (
            "get_dependencies",
            "Find what an entity depends on (outgoing relationships).",
            {
                "entity_name": {"type": "string", "description": "Entity name"},
                "repo_id": {"type": "string"},
            },
        ),
        (
            "get_dependents",
            "Find what depends on an entity (incoming relationships).",
            {
                "entity_name": {"type": "string", "description": "Entity name"},
                "repo_id": {"type": "string"},
            },
        ),
        (
            "trace_imports",
            "Find what a module/file calls or imports. Accepts a symbol name (e.g. 'APIRouter') or file path fragment (e.g. 'routing' or 'fastapi/routing.py').",
            {
                "module_name": {
                    "type": "string",
                    "description": "Symbol name or file path fragment",
                },
                "repo_id": {"type": "string"},
            },
        ),
        (
            "find_related",
            "Get entities related by a specific relationship type (CALLS, MEMBER_OF, STEP_IN_PROCESS, ACCESSES, DEFINES, HAS_METHOD).",
            {
                "entity_name": {"type": "string"},
                "relationship_type": {"type": "string"},
                "repo_id": {"type": "string"},
            },
        ),
        (
            "execute_query",
            "Run a raw Cypher query against LadybugDB (read-only).",
            {"cypher": {"type": "string", "description": "Cypher query string"}},
        ),
        (
            "get_symbol_context",
            "360-degree view of a symbol: callers, callees, imports, process participation. BEST tool for lifecycle/flow questions.",
            {"symbol_name": {"type": "string"}, "repo_id": {"type": "string"}},
        ),
        (
            "analyze_impact",
            "Blast-radius: what breaks if this symbol changes.",
            {
                "symbol_name": {"type": "string"},
                "depth": {"type": "integer", "default": 2},
                "repo_id": {"type": "string"},
            },
        ),
        (
            "list_entities",
            "List all entities of a type (Function, Class, Method, Module, File).",
            {
                "entity_type": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
                "repo_id": {"type": "string"},
            },
        ),
        (
            "list_entities_tree",
            "List all entities of a type grouped into a folder/file tree. "
            "Use this instead of list_entities for 'get all X' queries — returns a compact summary safe for large repos.",
            {
                "entity_type": {
                    "type": "string",
                    "description": "Function, Class, File, Folder",
                },
                "repo_id": {"type": "string"},
            },
        ),
        (
            "analyze_file",
            "Extract classes, functions, and decorators from a file via the graph. "
            "Accepts full path ('fastapi/routing.py') OR partial name ('routing') — CONTAINS matching resolves the real path. "
            "Always use this for 'find decorators/imports in <module>' queries.",
            {
                "file_path": {
                    "type": "string",
                    "description": "Full or partial path, e.g. 'routing' or 'fastapi/routing.py'",
                },
                "repo_id": {"type": "string", "description": "Optional repo filter"},
            },
        ),
    ]:
        tools.append(_make_mcp_tool("graph_query", gq_port, name, desc, schema))

    ca_port = settings.CODE_ANALYST_PORT

    for name, desc, schema in [
        (
            "explain_implementation",
            "LLM-generated plain-English explanation of how an entity works.",
            {"entity_name": {"type": "string"}, "model": {"type": "string"}},
        ),
        (
            "analyze_function",
            "Deep analysis of a function's logic.",
            {
                "function_name": {"type": "string"},
                "repo_path": {"type": "string"},
                "model": {"type": "string"},
            },
        ),
        (
            "analyze_class",
            "Comprehensive analysis of a class and its methods.",
            {
                "class_name": {"type": "string"},
                "repo_path": {"type": "string"},
                "model": {"type": "string"},
            },
        ),
        (
            "find_patterns",
            "Detect design patterns in a module or entity.",
            {"code_path": {"type": "string"}, "pattern_type": {"type": "string"}},
        ),
    ]:
        tools.append(_make_mcp_tool("code_analyst", ca_port, name, desc, schema))

    _tools_cache[cache_key] = tools
    return tools


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def inject_history(state: OrchestratorState) -> dict:
    """Prepare messages for the ReAct agent.

    If the MemorySaver checkpoint has already restored prior turn messages, skip
    Redis history injection to avoid duplicating them. The checkpoint is the
    authoritative source for conversation history within a session.
    """
    session_id = state.get("session_id", "")
    current = list(state.get("messages", []))
    current_user_msg = ""

    # Identify the current turn's HumanMessage before we prepend history.
    # persist_turn will use _current_user_msg instead of searching reversed(messages),
    # which avoids accidentally picking up an injected-history HumanMessage.
    if current and isinstance(current[0], HumanMessage):
        current_user_msg = current[0].content
        current[0].content = f"[CURRENT_TURN] {current[0].content}"

    # Skip Redis history if checkpoint already populated the state with prior turns.
    # MemorySaver restores ALL previous messages at the start of each turn, so
    # injecting Redis history would duplicate them and grow the message list
    # unboundedly (50 → 53 → 56 → 60 → ...), exhausting the context window.
    if len(current) > 1:
        logger.info(
            "inject_history: checkpoint has %d messages, skipping Redis history",
            len(current),
        )
        return {"messages": current, "_current_user_msg": current_user_msg}

    if not session_id:
        return {"messages": current}

    history_result = await _call_mcp_agent(
        "memory",
        settings.MEMORY_PORT,
        "get_conversation_context",
        {"session_id": session_id},
        timeout=10,
    )

    prior: list = history_result.get(
        "messages", history_result.get("context", []) or []
    )
    history_messages = []
    _MAX_TURNS = 6
    _MAX_AI_CHARS = 600
    for turn in prior[-_MAX_TURNS:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            if len(content) > _MAX_AI_CHARS:
                content = content[:_MAX_AI_CHARS] + "\n... [truncated for context]"
            history_messages.append(AIMessage(content=content))

    if not history_messages:
        return {"messages": current, "_current_user_msg": current_user_msg}

    logger.info(
        "inject_history: injecting %d history messages for session %s",
        len(history_messages),
        session_id,
    )
    return {
        "messages": history_messages + current,
        "_current_user_msg": current_user_msg,
    }


async def react_agent(state: OrchestratorState) -> dict:
    """Single ReAct step: LLM decides next tool call or final answer."""
    import asyncio as _asyncio

    model = state.get("model", "")
    repo_id = state.get("repo_id", "")

    tools = build_tools(repo_id=repo_id, model=model)
    llm = _get_llm(model).bind_tools(tools)

    # Build system prompt with context-specific info
    system_prompt = REACT_SYSTEM_PROMPT
    if repo_id:
        logger.info("react_agent: Adding repo_id context to prompt: %s", repo_id)
        system_prompt += f"\n\n**Repository Context**: You are analyzing repository '{repo_id}'. ALWAYS pass repo_id='{repo_id}' to EVERY graph-query tool call like find_entity(name='...', repo_id='{repo_id}')."
    else:
        logger.info("react_agent: No repo_id in state")

    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))

    logger.info("react_agent: %d messages to LLM", len(messages))
    for i, m in enumerate(messages):
        m_type = type(m).__name__
        content_len = len(str(getattr(m, "content", "")))
        if i == 0:  # System prompt
            logger.info("  [sys] SystemMessage: %d chars", content_len)
        elif m_type == "HumanMessage":
            logger.info("  [%d] %s: %s...", i, m_type, str(m.content)[:100])
        elif m_type == "AIMessage":
            tool_calls = len(getattr(m, "tool_calls", []))
            logger.info(
                "  [%d] %s: %d tool_calls, content=%d chars",
                i,
                m_type,
                tool_calls,
                content_len,
            )
        elif m_type == "ToolMessage":
            tool_name = getattr(m, "name", "?")
            logger.info(
                "  [%d] ToolMessage (%s): %d chars, preview: %s",
                i,
                tool_name,
                content_len,
                str(m.content)[:300],
            )
        else:
            logger.info("  [%d] %s: %d chars", i, m_type, content_len)

    # Retry up to 4 times on transient errors from OpenRouter free models:
    #   524/timeout → short backoff (1s, 2s, 4s)
    #   429 rate-limit → longer backoff (10s, 20s, 40s)
    last_exc = None
    for attempt in range(4):
        try:
            response = await llm.ainvoke(messages)
            logger.info(
                "react_agent LLM response: type=%s, content_len=%d, tool_calls=%s",
                type(response).__name__,
                len(str(getattr(response, "content", ""))),
                len(getattr(response, "tool_calls", []))
                if hasattr(response, "tool_calls")
                else "N/A",
            )
            logger.info(
                "  content preview: %s", str(getattr(response, "content", ""))[:200]
            )
            break
        except Exception as exc:
            err = str(exc)
            is_timeout = (
                "524" in err or "timeout" in err.lower() or "timed out" in err.lower()
            )
            is_rate_limit = (
                "429" in err
                or "rate limit" in err.lower()
                or "too many requests" in err.lower()
            )
            if is_rate_limit:
                last_exc = exc
                wait = 10 * (2**attempt)  # 10s, 20s, 40s, 80s
                logger.warning(
                    "LLM rate-limited (429), retrying in %ds [attempt %d/4]: %s",
                    wait,
                    attempt + 1,
                    err[:120],
                )
                try:
                    writer = get_stream_writer()
                    writer(
                        {
                            "type": "retry",
                            "reason": "rate_limit",
                            "wait": wait,
                            "attempt": attempt + 1,
                        }
                    )
                except Exception:
                    pass
                await _asyncio.sleep(wait)
                continue
            elif is_timeout:
                last_exc = exc
                wait = 2**attempt  # 1s, 2s, 4s, 8s
                logger.warning(
                    "LLM timeout, retrying in %ds [attempt %d/4]: %s",
                    wait,
                    attempt + 1,
                    err[:120],
                )
                try:
                    writer = get_stream_writer()
                    writer(
                        {
                            "type": "retry",
                            "reason": "timeout",
                            "wait": wait,
                            "attempt": attempt + 1,
                        }
                    )
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

    # Collect tool call timing data
    tool_calls_data: list[dict] = []
    tool_start_times: dict[str, float] = {}
    tool_args_map: dict[str, dict] = {}

    logger.info("persist_turn: %d messages in state", len(messages))
    for i, m in enumerate(messages):
        msg_type = type(m).__name__
        has_tool_calls = getattr(m, "tool_calls", None) is not None

        content = getattr(m, "content", "N/A")
        content_len = len(str(content))
        content_preview = str(content)[:300]

        if msg_type == "AIMessage" and has_tool_calls:
            tool_calls = getattr(m, "tool_calls", [])
            logger.info(
                "  [%d] AIMessage: %d tool_calls, content=%s",
                i,
                len(tool_calls),
                content_preview,
            )
            for j, tc in enumerate(tool_calls):
                tool_name = tc.get("name", "?")
                tool_id = tc.get("id", f"call_{j}")
                tool_args = tc.get("args", {})
                tool_start_times[tool_id] = time.time()
                tool_args_map[tool_id] = tool_args
                logger.info(
                    "       tool[%d]: id=%s name=%s args=%s",
                    j,
                    tool_id,
                    tool_name,
                    str(tool_args)[:100],
                )
        elif msg_type == "ToolMessage":
            tool_name = getattr(m, "name", "?")
            tool_id = getattr(m, "tool_call_id", "")
            content_str = str(content) if content else ""
            duration_ms = 0
            result_for_ui = content_str
            try:
                if content_str.startswith("{"):
                    result_data = json.loads(content_str)
                    duration_ms = result_data.get("_duration_ms", 0)
                    clean_result = {
                        k: v for k, v in result_data.items() if k != "_duration_ms"
                    }
                    result_for_ui = json.dumps(clean_result)[:300]
            except Exception:
                result_for_ui = content_str[:300]
            logger.info(
                "  [%d] ToolMessage (%s): %d chars, duration_ms=%d, preview: %s",
                i,
                tool_name,
                content_len,
                duration_ms,
                content_preview,
            )
            tool_calls_data.append(
                {
                    "tool": tool_name,
                    "args": tool_args_map.get(tool_id, {}),
                    "result": result_for_ui,
                    "duration_ms": duration_ms,
                    "ts": time.strftime("%H:%M:%S"),
                }
            )
        elif msg_type == "AIMessage":
            logger.info(
                "  [%d] AIMessage (final): content_len=%d, preview=%s",
                i,
                len(str(content)),
                content_preview,
            )
            logger.info("       Full content: %s", repr(content)[:500])
        else:
            logger.info("  [%d] %s: %s", i, msg_type, content_preview)

    if not session_id:
        return {"final_response": "", "tool_calls": tool_calls_data}

    current_input = state.get("_current_user_msg", "")
    if not current_input:
        for m in reversed(messages):
            content = str(getattr(m, "content", "") or "")
            if isinstance(m, HumanMessage) and not content.startswith("[CURRENT_TURN]"):
                current_input = content
                break
    user_msg = current_input
    ai_msg = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            ai_msg = _msg_text(m)
            break

    logger.info("persist_turn RESULT: user_msg=%.50s ai_msg=%.50s", user_msg, ai_msg)

    if user_msg and ai_msg:
        await _call_mcp_agent(
            "memory",
            settings.MEMORY_PORT,
            "store_interaction",
            {"session_id": session_id, "query": user_msg, "response": ai_msg},
            timeout=10,
        )
    return {"final_response": ai_msg, "tool_calls": tool_calls_data}
