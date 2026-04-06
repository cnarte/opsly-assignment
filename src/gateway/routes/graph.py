"""Graph statistics endpoints — direct Neo4j queries."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.shared.schemas import GraphStats
from src.shared.neo4j_client import Neo4jClient
from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

router = APIRouter(tags=["graph"])


@router.get("/api/graph/statistics", response_model=GraphStats)
async def graph_statistics() -> GraphStats:
    """Return knowledge graph statistics (node/relationship counts per label)."""
    client = Neo4jClient(settings)
    try:
        await client.connect()

        # Total node count
        node_result = await client.execute_query("MATCH (n) RETURN count(n) AS cnt")
        total_nodes = node_result[0]["cnt"] if node_result else 0

        # Total relationship count
        rel_result = await client.execute_query(
            "MATCH ()-[r]->() RETURN count(r) AS cnt"
        )
        total_rels = rel_result[0]["cnt"] if rel_result else 0

        # Node counts per label
        label_result = await client.execute_query(
            "CALL db.labels() YIELD label "
            "CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS cnt', {}) "
            "YIELD value "
            "RETURN label, value.cnt AS cnt"
        )

        # Fallback: if APOC is not available, use a simpler approach
        if not label_result:
            label_result = await client.execute_query(
                "MATCH (n) "
                "WITH labels(n) AS lbls "
                "UNWIND lbls AS label "
                "RETURN label, count(*) AS cnt"
            )

        labels: dict[str, int] = {}
        for row in label_result:
            labels[row["label"]] = row["cnt"]

        return GraphStats(nodes=total_nodes, relationships=total_rels, labels=labels)
    except Exception as exc:
        logger.error("Failed to query graph statistics: %s", exc)
        return GraphStats(nodes=0, relationships=0, labels={})
    finally:
        await client.close()
