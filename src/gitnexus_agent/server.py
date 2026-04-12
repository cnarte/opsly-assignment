"""GitNexus Agent — MCP server that proxies to the gitnexus CLI."""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from src.shared.gitnexus_client import GitNexusClient
from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP(
    "gitnexus-agent",
    host="0.0.0.0",
    port=settings.GITNEXUS_PORT,
)

_client: GitNexusClient | None = None


async def _get_client() -> GitNexusClient:
    """Return the singleton GitNexusClient, connecting on first call."""
    global _client
    if _client is None:
        _client = GitNexusClient()
        await _client.connect()
    return _client


def _args(base: dict, repo: str) -> dict:
    """Add repo key only when non-empty."""
    return {**base, "repo": repo} if repo else base


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@mcp.tool()
async def analyze_repo(path: str, repo_name: str) -> dict:
    """Index a repository at `path` using `gitnexus analyze`.

    Args:
        path: Absolute path to the cloned repository on disk.
        repo_name: Short identifier used in subsequent queries (e.g. 'fastapi').

    Returns:
        {"status": "indexed", "repo": repo_name, "path": path}
    """
    client = await _get_client()
    return await client.analyze_repo(path, repo_name)


# ---------------------------------------------------------------------------
# Query tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def query(q: str, repo: str = "") -> dict:
    """Hybrid BM25 + semantic search across the indexed codebase.

    Args:
        q: Natural language or keyword query.
        repo: Repo name to scope search. Empty = all repos.
    """
    client = await _get_client()
    return await client.call_tool("query", _args({"query": q}, repo))


@mcp.tool()
async def context(symbol: str, repo: str = "") -> dict:
    """360-degree view of a symbol: callers, callees, imports, process participation.

    Args:
        symbol: Exact or partial symbol name (class, function, method).
        repo: Repo name to scope. Empty = all repos.
    """
    client = await _get_client()
    return await client.call_tool("context", _args({"symbol": symbol}, repo))


@mcp.tool()
async def impact(symbol: str, repo: str = "") -> dict:
    """Blast-radius analysis: what breaks if this symbol changes.

    Args:
        symbol: Symbol name to analyse.
        repo: Repo name to scope. Empty = all repos.
    """
    client = await _get_client()
    return await client.call_tool("impact", _args({"symbol": symbol}, repo))


@mcp.tool()
async def cypher(query_str: str, repo: str = "") -> dict:
    """Run a raw Cypher query against the LadybugDB graph (read-only).

    Args:
        query_str: Cypher query string.
        repo: Repo name to scope. Empty = all repos.
    """
    client = await _get_client()
    return await client.call_tool("cypher", _args({"query": query_str}, repo))


@mcp.tool()
async def list_repos() -> dict:
    """Return all repositories currently indexed in LadybugDB."""
    client = await _get_client()
    return await client.call_tool("list_repos", {})


@mcp.tool()
async def group_query(q: str) -> dict:
    """Search for execution flows and contracts across ALL indexed repos.

    Args:
        q: Natural language query spanning multiple repos.
    """
    client = await _get_client()
    return await client.call_tool("group_query", {"query": q})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
