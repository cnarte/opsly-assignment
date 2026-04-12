"""Graph Query Agent MCP server — backed by gitnexus-agent."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP("graph-query-agent", host="0.0.0.0", port=settings.GRAPH_QUERY_PORT)


# ---------------------------------------------------------------------------
# Internal helper — call gitnexus-agent
# ---------------------------------------------------------------------------


async def _call_gitnexus(tool: str, args: dict) -> dict:
    """Call a tool on the gitnexus-agent MCP server."""
    host = "gitnexus-agent" if os.path.exists("/.dockerenv") else "localhost"
    url = f"http://{host}:{settings.GITNEXUS_PORT}/mcp"
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            if result.content:
                try:
                    return json.loads(result.content[0].text)
                except (json.JSONDecodeError, TypeError):
                    return {"result": result.content[0].text}
            return {}


def _repo_args(base: dict, repo_id: str) -> dict:
    return {**base, "repo": repo_id} if repo_id else base


# ---------------------------------------------------------------------------
# MCP tools (assignment-required names)
# ---------------------------------------------------------------------------


@mcp.tool()
async def find_entity(name: str, entity_type: str = "", repo_id: str = "") -> dict:
    """Locate a class, function, or module by name using hybrid search."""
    return await _call_gitnexus("query", _repo_args({"query": name}, repo_id))


@mcp.tool()
async def get_dependencies(entity_name: str, repo_id: str = "") -> dict:
    """Find what an entity depends on (outgoing relationships)."""
    result = await _call_gitnexus("context", _repo_args({"symbol": entity_name}, repo_id))
    return {
        "entity": entity_name,
        "dependencies": result.get("outgoing", result.get("refs", [])),
        "count": len(result.get("outgoing", result.get("refs", []))),
    }


@mcp.tool()
async def get_dependents(entity_name: str, repo_id: str = "") -> dict:
    """Find what depends on an entity (incoming relationships)."""
    result = await _call_gitnexus("context", _repo_args({"symbol": entity_name}, repo_id))
    return {
        "entity": entity_name,
        "dependents": result.get("incoming", []),
        "count": len(result.get("incoming", [])),
    }


@mcp.tool()
async def trace_imports(module_name: str, repo_id: str = "") -> dict:
    """Follow the import chain for a module."""
    cypher = (
        "MATCH path = (m {name: $name})-[:IMPORTS*1..5]->(t) "
        "RETURN [node IN nodes(path) | node.name] AS chain LIMIT 20"
    )
    args = _repo_args({"query": cypher.replace("$name", f'"{module_name}"')}, repo_id)
    result = await _call_gitnexus("cypher", args)
    return {"module": module_name, "import_chains": result.get("results", []), "count": len(result.get("results", []))}


@mcp.tool()
async def find_related(entity_name: str, relationship_type: str, repo_id: str = "") -> dict:
    """Get entities related by a specific relationship type."""
    cypher = f'MATCH (n {{name: "{entity_name}"}})-[r:{relationship_type}]->(t) RETURN n, r, t LIMIT 50'
    result = await _call_gitnexus("cypher", _repo_args({"query": cypher}, repo_id))
    return {"entity": entity_name, "relationship": relationship_type, "results": result.get("results", []), "count": len(result.get("results", []))}


@mcp.tool()
async def execute_query(cypher: str) -> dict:
    """Run a raw Cypher query (read-only, LadybugDB-sandboxed)."""
    result = await _call_gitnexus("cypher", {"query": cypher})
    return {"results": result.get("results", result), "count": len(result.get("results", []))}


@mcp.tool()
async def get_symbol_context(symbol_name: str, repo_id: str = "") -> dict:
    """360-degree view of a symbol: callers, callees, imports, process participation."""
    return await _call_gitnexus("context", _repo_args({"symbol": symbol_name}, repo_id))


@mcp.tool()
async def analyze_impact(symbol_name: str, depth: int = 2, repo_id: str = "") -> dict:
    """Blast-radius analysis: what would break if this symbol changed."""
    return await _call_gitnexus("impact", _repo_args({"symbol": symbol_name}, repo_id))


@mcp.tool()
async def list_entities(entity_type: str, limit: int = 50, repo_id: str = "") -> dict:
    """List all entities of a given type via Cypher."""
    label = entity_type.capitalize()
    cypher = f"MATCH (n:{label}) WHERE n.name IS NOT NULL RETURN n.name AS name, n.file AS file_path LIMIT {min(limit, 200)}"
    result = await _call_gitnexus("cypher", _repo_args({"query": cypher}, repo_id))
    return {"entity_type": label, "entities": result.get("results", []), "count": len(result.get("results", []))}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
