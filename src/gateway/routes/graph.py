"""Graph statistics endpoints — direct Neo4j queries."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter

from src.shared.schemas import GraphStats
from src.shared.neo4j_client import Neo4jClient
from src.shared.settings import Settings
from src.gateway.mcp_client import call_agent_tool

logger = logging.getLogger(__name__)
settings = Settings()

router = APIRouter(tags=["graph"])

_MEMORY_LABELS = {"MemoryInteraction", "MemoryFact"}


@router.get("/api/graph/statistics", response_model=GraphStats)
async def graph_statistics() -> GraphStats:
    """Return code knowledge-graph statistics (excludes memory labels)."""
    client = Neo4jClient(settings)
    try:
        await client.connect()

        # Total node count — exclude memory-layer labels
        node_result = await client.execute_query(
            "MATCH (n) WHERE NOT n:MemoryInteraction AND NOT n:MemoryFact "
            "RETURN count(n) AS cnt"
        )
        total_nodes = node_result[0]["cnt"] if node_result else 0

        # Total relationship count — exclude memory-layer nodes at either end
        rel_result = await client.execute_query(
            "MATCH ()-[r]->() "
            "WHERE NOT startNode(r):MemoryInteraction AND NOT startNode(r):MemoryFact "
            "  AND NOT endNode(r):MemoryInteraction AND NOT endNode(r):MemoryFact "
            "RETURN count(r) AS cnt"
        )
        total_rels = rel_result[0]["cnt"] if rel_result else 0

        # Per-label counts — try APOC first, fall back to pure Cypher
        label_result: list[dict] = []
        try:
            label_result = await client.execute_query(
                "CALL db.labels() YIELD label "
                "CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS cnt', {}) "
                "YIELD value "
                "RETURN label, value.cnt AS cnt"
            )
        except Exception:
            label_result = []

        if not label_result:
            label_result = await client.execute_query(
                "MATCH (n) WITH labels(n) AS lbls UNWIND lbls AS label "
                "RETURN label, count(*) AS cnt"
            )

        labels = {
            row["label"]: row["cnt"]
            for row in label_result
            if row["label"] not in _MEMORY_LABELS
        }

        return GraphStats(nodes=total_nodes, relationships=total_rels, labels=labels)
    except Exception as exc:
        logger.error("Failed to query graph statistics: %s", exc)
        return GraphStats(nodes=0, relationships=0, labels={})
    finally:
        await client.close()


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
