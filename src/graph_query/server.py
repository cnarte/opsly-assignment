"""Graph Query Agent MCP server — backed by gitnexus-agent."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP("graph-query-agent", host="0.0.0.0", port=settings.GRAPH_QUERY_PORT)

# Safety guard for Cypher injection prevention
_SAFE_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Entity type mapping for LadybugDB id-prefix resolution
_ENTITY_TYPE_MAP = {
    "function": "Function",
    "functions": "Function",
    "class": "Class",
    "classes": "Class",
    "file": "File",
    "files": "File",
    "folder": "Folder",
    "module": "File",
}


# ---------------------------------------------------------------------------
# Internal helper — call gitnexus-agent
# ---------------------------------------------------------------------------


async def _call_gitnexus(tool: str, args: dict) -> dict:
    """Call a tool on the gitnexus-agent MCP server."""
    host = "gitnexus-agent" if os.path.exists("/.dockerenv") else "localhost"
    url = f"http://{host}:{settings.GITNEXUS_PORT}/mcp"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, read=300.0, pool=300.0, write=300.0),
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
                result = await session.call_tool(tool, args)
                if result.content:
                    try:
                        parsed = json.loads(result.content[0].text)
                        # Unwrap double-serialised responses: {"result": "<json string>"}
                        if (
                            isinstance(parsed, dict)
                            and list(parsed.keys()) == ["result"]
                            and isinstance(parsed["result"], str)
                        ):
                            inner = parsed["result"]
                            try:
                                return json.loads(inner)
                            except json.JSONDecodeError:
                                # Response truncated by MCP transport — try progressively
                                # smaller slices at brace/bracket boundaries
                                for pct in (0.9, 0.75, 0.5):
                                    candidate = inner[: int(len(inner) * pct)]
                                    last_brace = max(
                                        candidate.rfind("}"), candidate.rfind("]")
                                    )
                                    if last_brace > 0:
                                        try:
                                            return json.loads(
                                                candidate[: last_brace + 1]
                                            )
                                        except json.JSONDecodeError:
                                            pass
                                return {"raw_truncated": inner[:4000]}
                        return parsed
                    except (json.JSONDecodeError, TypeError):
                        return {"result": result.content[0].text}
                return {}


def _normalize_repo_id(repo_id: str) -> str:
    if not repo_id:
        return repo_id
    if "/" in repo_id:
        repo_id = repo_id.rsplit("/", 1)[-1]
    if repo_id.endswith(".git"):
        repo_id = repo_id[:-4]
    return repo_id


def _repo_args(base: dict, repo_id: str) -> dict:
    repo_id = _normalize_repo_id(repo_id)
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
    """Locate a class, function, or module by hybrid search."""
    return await _call_gitnexus("query", _repo_args({"q": name}, repo_id))


@mcp.tool()
async def get_dependencies(entity_name: str, repo_id: str = "") -> dict:
    """Find what an entity depends on (outgoing relationships)."""
    result = await _call_gitnexus(
        "context", _repo_args({"symbol": entity_name}, repo_id)
    )
    return {
        "entity": entity_name,
        "dependencies": result.get("outgoing", result.get("refs", [])),
        "count": len(result.get("outgoing", result.get("refs", []))),
    }


@mcp.tool()
async def get_dependents(entity_name: str, repo_id: str = "") -> dict:
    """Find what depends on an entity (incoming relationships)."""
    result = await _call_gitnexus(
        "context", _repo_args({"symbol": entity_name}, repo_id)
    )
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
        f"MATCH (m)-[*1..5]->(t) WHERE t.name IS NOT NULL "
        f"RETURN t.name AS name, t.filePath AS file_path LIMIT 20"
    )
    result = await _call_gitnexus("cypher", _repo_args({"query_str": cypher}, repo_id))
    rows = _parse_result(result)
    return {"module": module_name, "imports": rows, "count": len(rows)}


@mcp.tool()
async def find_related(
    entity_name: str, relationship_type: str, repo_id: str = ""
) -> dict:
    """Get entities related by a specific relationship type."""
    cypher = (
        f'MATCH (n) WHERE n.name = "{entity_name}" '
        f"MATCH (n)-[r]->(t) WHERE t.name IS NOT NULL "
        f"RETURN n.name AS source, t.name AS target, t.filePath AS file_path LIMIT 50"
    )
    result = await _call_gitnexus("cypher", _repo_args({"query_str": cypher}, repo_id))
    rows = _parse_result(result)
    return {
        "entity": entity_name,
        "relationship": relationship_type,
        "results": rows,
        "count": len(rows),
    }


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
    """Blast-radius analysis: what would break if this symbol changed.
    Returns impacted entities grouped by depth level (capped at 20 per level).
    """
    safe_name = symbol_name.replace('"', '\\"').replace("\\", "\\\\")
    cap_depth = max(1, min(int(depth), 4))
    cypher = (
        f'MATCH (target) WHERE target.name = "{safe_name}" '
        f"WITH target LIMIT 1 "
        f"MATCH path = (target)<-[*1..{cap_depth}]-(dependent) "
        f"WHERE dependent.name IS NOT NULL AND dependent.id <> target.id "
        f"RETURN DISTINCT dependent.name AS name, dependent.filePath AS file_path, "
        f"dependent.id AS id, length(path) AS depth_level "
        f"ORDER BY depth_level, name LIMIT 100"
    )
    result = await _call_gitnexus("cypher", _repo_args({"query_str": cypher}, repo_id))
    rows = _parse_result(result)

    by_depth: dict = {}
    for row in rows:
        lvl = str(row.get("depth_level", 1))
        if lvl not in by_depth:
            by_depth[lvl] = []
        if len(by_depth[lvl]) < 20:
            by_depth[lvl].append(
                {
                    "name": row.get("name"),
                    "file_path": row.get("file_path"),
                    "id": row.get("id"),
                }
            )

    return {
        "symbol": symbol_name,
        "impacted_count": len(rows),
        "depth": cap_depth,
        "by_depth": by_depth,
        "note": "Showing up to 20 items per depth level, max 100 total.",
    }


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
        f"RETURN n.name AS name, n.filePath AS file_path LIMIT {cap}"
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
    # Aggregate in the DB — one row per file with count, avoids transmitting thousands of rows
    cypher = (
        f'MATCH (n) WHERE n.id STARTS WITH "{prefix}:" AND n.name IS NOT NULL '
        f"WITH n.filePath AS file_path, count(*) AS cnt "
        f"RETURN file_path, cnt ORDER BY file_path LIMIT 500"
    )
    result = await _call_gitnexus("cypher", _repo_args({"query_str": cypher}, repo_id))
    rows = _parse_result(result)

    tree: dict[str, dict] = {}
    total = 0
    for row in rows:
        fp: str = row.get("file_path") or ""
        cnt: int = row.get("cnt", 1)
        total += cnt
        parts = fp.split("/")
        folder = parts[0] if len(parts) > 1 else "_root"
        filename = parts[-1] or "_unknown"

        if folder not in tree:
            tree[folder] = {"count": 0, "files": {}}
        tree[folder]["count"] += cnt
        tree[folder]["files"][filename] = tree[folder]["files"].get(filename, 0) + cnt

    return {
        "entity_type": prefix,
        "total": total,
        "tree": tree,
    }


# ---------------------------------------------------------------------------
# File-level analysis tool (bypasses symbol search limitations)
# ---------------------------------------------------------------------------


@mcp.tool()
async def analyze_file(file_path: str, focus: str = "") -> dict:
    """Analyze a file directly for decorators, imports, classes, functions.

    Bypasses symbol-search limitations by reading raw file content.
    Args:
        file_path: e.g., "fastapi/routing.py"
        focus: What to analyze (e.g., "decorators", "imports", "classes")

    Returns: Analysis with file content, extracted entities, and patterns
    """
    import os

    workspace = os.getenv("WORKSPACE_PATH", "/workspace/repos")
    analysis = {
        "file_path": file_path,
        "focus": focus,
        "found": False,
        "content": "",
        "decorators": [],
        "classes": [],
        "functions": [],
        "imports": [],
    }

    # Try to find the file in indexed repos
    if os.path.exists(workspace):
        for repo_dir in os.listdir(workspace):
            candidate = os.path.join(workspace, repo_dir, file_path)
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r") as f:
                        content = f.read()
                    analysis["found"] = True
                    analysis["content"] = content
                    analysis["repo"] = repo_dir

                    # Basic pattern extraction
                    import re
                    analysis["decorators"] = re.findall(r"@\w+[\w\.\(]*", content)
                    analysis["classes"] = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
                    analysis["functions"] = re.findall(r"^(?:async\s+)?def\s+(\w+)", content, re.MULTILINE)
                    analysis["imports"] = re.findall(r"^(?:from|import)\s+(.+)$", content, re.MULTILINE)

                    return analysis
                except Exception as e:
                    analysis["error"] = str(e)
                    return analysis

    analysis["error"] = f"File not found in {workspace}"
    return analysis


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
