"""Graph statistics endpoints — backed by gitnexus-agent (LadybugDB)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter

from src.shared.schemas import GraphStats
from src.shared.settings import Settings
from src.gateway.mcp_client import call_agent_tool

logger = logging.getLogger(__name__)
settings = Settings()

router = APIRouter(tags=["graph"])


def _parse_markdown_table(markdown: str) -> list[dict]:
    """Parse a markdown table returned by gitnexus cypher into list of dicts."""
    lines = [l.strip() for l in markdown.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return []
    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
    rows = []
    for line in lines[2:]:  # skip header and separator
        cells = [c.strip() for c in line.split("|") if c.strip() != ""]
        if len(cells) == len(headers):
            row = {}
            for h, c in zip(headers, cells):
                try:
                    row[h] = int(c)
                except ValueError:
                    row[h] = c
            rows.append(row)
    return rows


async def _gitnexus_cypher(query: str) -> list[dict]:
    """Run a Cypher query via gitnexus-agent and return parsed result rows."""
    raw = await call_agent_tool(settings.GITNEXUS_PORT, "cypher", {"query_str": query})
    if "error" in raw:
        logger.warning("gitnexus cypher error: %s", raw["error"])
        return []
    markdown = raw.get("markdown", "")
    if markdown:
        return _parse_markdown_table(markdown)
    return []


# Node types known to LadybugDB (id prefix → display label)
_NODE_TYPES = [
    ("Function", "Function"),
    ("Class", "Class"),
    ("Method", "Method"),
    ("File", "File"),
    ("Folder", "Folder"),
    ("Process", "Process"),
    ("Cluster", "Cluster"),
]


@router.get("/api/graph/statistics", response_model=GraphStats)
async def graph_statistics() -> GraphStats:
    """Return knowledge-graph statistics from LadybugDB via gitnexus-agent."""
    try:
        node_rows = await _gitnexus_cypher("MATCH (n) RETURN count(n) AS cnt")
        total_nodes = node_rows[0].get("cnt", 0) if node_rows else 0

        rel_rows = await _gitnexus_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")
        total_rels = rel_rows[0].get("cnt", 0) if rel_rows else 0

        # Count each node type using id-prefix filter (LadybugDB lacks split/type functions)
        import asyncio
        async def _count(prefix: str) -> int:
            rows = await _gitnexus_cypher(
                f'MATCH (n) WHERE n.id STARTS WITH "{prefix}:" RETURN count(n) AS cnt'
            )
            return rows[0].get("cnt", 0) if rows else 0

        counts = await asyncio.gather(*[_count(p) for p, _ in _NODE_TYPES])
        labels = {label: cnt for (_, label), cnt in zip(_NODE_TYPES, counts) if cnt > 0}

        return GraphStats(nodes=total_nodes, relationships=total_rels, labels=labels)
    except Exception as exc:
        logger.error("Failed to query gitnexus statistics: %s", exc)
        return GraphStats(nodes=0, relationships=0, labels={})


@router.get("/api/graph/repos")
async def list_repos() -> dict:
    """Return all repos indexed in LadybugDB (via gitnexus-agent)."""
    raw = await call_agent_tool(settings.GITNEXUS_PORT, "list_repos", {})

    if "error" in raw:
        logger.error("list_repos tool error: %s", raw["error"])
        return {"repos": []}

    # New format: {"repos": ["fastapi", ...]}
    if "repos" in raw:
        return {"repos": raw["repos"]}

    # Legacy/fallback: {"result": "<text>"} or {"result": "[...]"}
    result_text: str = raw.get("result", "") or ""
    text_before_separator = result_text.split("---")[0].strip()

    repos: list[str] = []
    try:
        parsed = json.loads(text_before_separator)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    repos.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("repo") or item.get("id")
                    if name:
                        repos.append(str(name))
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse list_repos result: %r", text_before_separator)

    return {"repos": repos}
