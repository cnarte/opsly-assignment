"""Graph Query Agent MCP server — backed by gitnexus-agent."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP("graph-query-agent", host="0.0.0.0", port=settings.GRAPH_QUERY_PORT)

# Safety guard for Cypher injection prevention
_SAFE_PREFIX = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')

# Entity type mapping for LadybugDB id-prefix resolution
_ENTITY_TYPE_MAP = {
    "function": "Function", "functions": "Function",
    "class": "Class", "classes": "Class",
    "file": "File", "files": "File",
    "folder": "Folder", "module": "File",
}


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


def _parse_result(raw: dict) -> list[dict]:
    """Parse LadybugDB cypher result (markdown table format) into list of dicts."""
    if "error" in raw:
        return []
    markdown = raw.get("markdown", "")
    if not markdown:
        return []
    lines = [l.strip() for l in markdown.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return []
    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
    rows = []
    for line in lines[2:]:  # skip header and separator
        cells = [c.strip() for c in line.split("|") if c.strip() != ""]
        if len(cells) == len(headers):
            row: dict = {}
            for h, c in zip(headers, cells):
                try:
                    row[h] = int(c)
                except ValueError:
                    row[h] = c
            rows.append(row)
    return rows


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
    # LadybugDB: use n.name property match, not Cypher $params
    cypher = (
        f'MATCH (m) WHERE m.name = "{module_name}" '
        f'MATCH (m)-[*1..5]->(t) WHERE t.name IS NOT NULL '
        f'RETURN t.name AS name, t.filePath AS file_path LIMIT 20'
    )
    result = await _call_gitnexus("cypher", _repo_args({"query_str": cypher}, repo_id))
    rows = _parse_result(result)
    return {"module": module_name, "imports": rows, "count": len(rows)}


@mcp.tool()
async def find_related(entity_name: str, relationship_type: str, repo_id: str = "") -> dict:
    """Get entities related by a specific relationship type."""
    cypher = (
        f'MATCH (n) WHERE n.name = "{entity_name}" '
        f'MATCH (n)-[r]->(t) WHERE t.name IS NOT NULL '
        f'RETURN n.name AS source, t.name AS target, t.filePath AS file_path LIMIT 50'
    )
    result = await _call_gitnexus("cypher", _repo_args({"query_str": cypher}, repo_id))
    rows = _parse_result(result)
    return {"entity": entity_name, "relationship": relationship_type, "results": rows, "count": len(rows)}


@mcp.tool()
async def execute_query(cypher: str) -> dict:
    """Run a raw Cypher query (read-only, LadybugDB-sandboxed)."""
    result = await _call_gitnexus("cypher", {"query_str": cypher})
    rows = _parse_result(result)
    return {"results": rows, "count": len(rows), "raw": result}


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
    """List all entities of a given type.

    entity_type: Function, Class, File, Folder (LadybugDB id-prefix based).
    """
    # LadybugDB stores node types as id prefixes (e.g. "Function:my_func")
    # Map common aliases to their prefix
    prefix_raw = _ENTITY_TYPE_MAP.get(entity_type.lower(), entity_type.capitalize())
    prefix = prefix_raw if _SAFE_PREFIX.match(prefix_raw) else "Function"
    cap = min(limit, 200)
    cypher = (
        f'MATCH (n) WHERE n.id STARTS WITH "{prefix}:" AND n.name IS NOT NULL '
        f'RETURN n.name AS name, n.filePath AS file_path LIMIT {cap}'
    )
    result = await _call_gitnexus("cypher", _repo_args({"query_str": cypher}, repo_id))
    rows = _parse_result(result)
    return {"entity_type": prefix, "entities": rows, "count": len(rows)}


@mcp.tool()
async def list_entities_tree(entity_type: str, repo_id: str = "") -> dict:
    """List all entities of a type grouped into a folder/file tree.

    Returns a compact tree rather than a flat list — safe for large codebases.
    entity_type: Function, Class, File, Folder (LadybugDB id-prefix based).
    """
    prefix_raw = _ENTITY_TYPE_MAP.get(entity_type.lower(), entity_type.capitalize())
    prefix = prefix_raw if _SAFE_PREFIX.match(prefix_raw) else "Function"
    cypher = (
        f'MATCH (n) WHERE n.id STARTS WITH "{prefix}:" AND n.name IS NOT NULL '
        f'RETURN n.name AS name, n.filePath AS file_path LIMIT 5000'
    )
    result = await _call_gitnexus("cypher", _repo_args({"query_str": cypher}, repo_id))
    rows = _parse_result(result)

    tree: dict[str, dict] = {}
    for row in rows:
        fp: str = row.get("file_path") or ""
        parts = fp.split("/")
        folder = parts[0] if len(parts) > 1 else "_root"
        filename = parts[-1] or "_unknown"

        if folder not in tree:
            tree[folder] = {"count": 0, "files": {}}
        tree[folder]["count"] += 1
        tree[folder]["files"][filename] = tree[folder]["files"].get(filename, 0) + 1

    return {
        "entity_type": prefix,
        "total": len(rows),
        "tree": tree,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
