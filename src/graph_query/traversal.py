"""Graph traversal helpers for multi-hop queries."""

from __future__ import annotations

from typing import Any

from src.shared.neo4j_client import Neo4jClient
from src.graph_query import queries


async def build_impact_graph(
    client: Neo4jClient,
    symbol_name: str,
    depth: int = 2,
    repo_id: str = "",
) -> dict[str, Any]:
    """BFS from a symbol, collecting dependents at each depth level.

    Returns a dict with ``symbol``, ``depth``, and ``levels`` (a list of
    lists, one per depth level).
    """
    query = queries.impact_at_depth(symbol_name, depth, repo_id)
    records = await client.execute_query(query.cypher, query.parameters)

    levels: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        d = rec.get("depth", 1)
        entry = {"name": rec.get("name"), "labels": rec.get("labels", [])}
        levels.setdefault(d, []).append(entry)

    return {
        "symbol": symbol_name,
        "max_depth": depth,
        "total_affected": len(records),
        "levels": {str(k): v for k, v in sorted(levels.items())},
    }


async def build_context_view(
    client: Neo4jClient,
    symbol_name: str,
    repo_id: str = "",
) -> dict[str, Any]:
    """Collect all direct relationships for a symbol (360-degree view)."""
    query = queries.get_symbol_context(symbol_name, repo_id)
    records = await client.execute_query(query.cypher, query.parameters)

    outgoing: list[dict[str, Any]] = []
    incoming: list[dict[str, Any]] = []

    for rec in records:
        entry = {
            "relationship": rec.get("rel"),
            "related_name": rec.get("related_name"),
            "related_labels": rec.get("related_labels", []),
        }
        if rec.get("direction") == "outgoing":
            outgoing.append(entry)
        else:
            incoming.append(entry)

    return {
        "symbol": symbol_name,
        "outgoing": outgoing,
        "incoming": incoming,
        "total_relationships": len(outgoing) + len(incoming),
    }
