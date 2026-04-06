"""MCP client helper for gateway routes to call agent MCP servers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from src.shared.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Default timeout for MCP calls (seconds)
_DEFAULT_TIMEOUT = 120


async def call_orchestrator_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call a tool on the Orchestrator MCP server.

    Establishes a short-lived connection, invokes the tool, and returns
    the parsed result dict.
    """
    url = f"http://localhost:{settings.ORCHESTRATOR_PORT}/mcp/"
    return await _call_mcp_tool(url, tool_name, arguments, timeout=timeout)


async def call_agent_tool(
    port: int,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Call a tool on an arbitrary agent MCP server by port."""
    url = f"http://localhost:{port}/mcp/"
    return await _call_mcp_tool(url, tool_name, arguments, timeout=timeout)


async def check_agent_health(name: str, port: int) -> str:
    """Attempt to connect to an agent MCP server and return 'healthy' or 'unhealthy'."""
    url = f"http://localhost:{port}/mcp/"
    try:
        async with asyncio.timeout(5):
            async with streamablehttp_client(url) as (read, write, _):
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
    """Low-level helper: connect, call tool, parse result."""
    import json

    try:
        async with asyncio.timeout(timeout):
            async with streamablehttp_client(url) as (read, write, _):
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
        logger.error("MCP call to %s/%s failed: %s", url, tool_name, exc)
        return {"error": str(exc)}
