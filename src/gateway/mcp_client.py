"""MCP client helper for gateway routes to call agent MCP servers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

from src.shared.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Default timeout for MCP calls (seconds).
# Raised to 300 to allow orchestrator to complete complex multi-step queries
# (recursion_limit=50 with LLM tool calls can legitimately take 2-3 minutes).
_DEFAULT_TIMEOUT = 300

# Map ports to Docker Compose service names for inter-container communication
_PORT_TO_SERVICE: dict[int, str] = {
    settings.ORCHESTRATOR_PORT: "orchestrator",
    settings.INDEXER_PORT: "indexer",
    settings.GRAPH_QUERY_PORT: "graph-query",
    settings.CODE_ANALYST_PORT: "code-analyst",
    settings.MEMORY_PORT: "memory",
    settings.GITNEXUS_PORT: "gitnexus-agent",
}


def _agent_url(port: int) -> str:
    """Build MCP URL using Docker service name if in Docker, else localhost."""
    import os
    # Inside Docker, /.dockerenv exists
    if os.path.exists("/.dockerenv"):
        service = _PORT_TO_SERVICE.get(port, "localhost")
        return f"http://{service}:{port}/mcp"
    return f"http://localhost:{port}/mcp"


async def call_orchestrator_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call a tool on the Orchestrator MCP server."""
    url = _agent_url(settings.ORCHESTRATOR_PORT)
    return await _call_mcp_tool(url, tool_name, arguments, timeout=timeout)


async def call_agent_tool(
    port: int,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call a tool on an arbitrary agent MCP server by port."""
    url = _agent_url(port)
    return await _call_mcp_tool(url, tool_name, arguments, timeout=timeout)


async def check_agent_health(name: str, port: int) -> str:
    """Attempt to connect to an agent MCP server and return 'healthy' or 'unhealthy'."""
    import httpx
    url = _agent_url(port)
    try:
        async with asyncio.timeout(5):
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as http_client:
                async with streamable_http_client(url, http_client=http_client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        return "healthy"
    except Exception as exc:
        logger.warning("Agent %s health check failed: %s", name, exc)
        return "unhealthy"


async def _call_mcp_tool(
    url: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Low-level helper: connect, call tool, parse result.

    Uses a custom httpx client with a long read timeout so the inline POST
    response from long-running tools (e.g. route_to_agents, which runs a
    multi-step LangGraph pipeline) isn't cut short by httpx's default 30-second
    read timeout.
    """
    import json
    import httpx

    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, read=timeout),
                follow_redirects=True,
            ) as http_client:
                async with streamable_http_client(url, http_client=http_client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)

                        # result.content is a list of content blocks
                        if result.content:
                            text = result.content[0].text
                            try:
                                return json.loads(text)
                            except (json.JSONDecodeError, TypeError):
                                return {"result": text}
                        return {}
    except asyncio.TimeoutError:
        logger.error("MCP call to %s/%s timed out after %ss", url, tool_name, timeout)
        return {"error": f"Timeout calling {tool_name}"}
    except Exception as exc:
        import traceback
        logger.error("MCP call to %s/%s failed: %s\n%s", url, tool_name, exc, traceback.format_exc())
        return {"error": str(exc)}
