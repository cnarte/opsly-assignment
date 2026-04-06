"""Long-term memory backed by Neo4j (simplified Graphiti-style storage).

Uses dedicated ``MemoryFact`` and ``MemoryInteraction`` labels so that
the code-graph data managed by the Indexer / Graph Query agents is never
touched.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.shared.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphitiMemory:
    """Simplified long-term memory using Neo4j nodes."""

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j

    # ------------------------------------------------------------------
    # Episodes (interactions)
    # ------------------------------------------------------------------

    async def add_episode(self, session_id: str, content: str, query: str = "", response: str = "") -> str:
        """Store an interaction episode. Returns the created node ID."""
        node_id = str(uuid4())
        await self._neo4j.execute_write(
            """
            CREATE (n:MemoryInteraction {
                id: $id,
                session_id: $session_id,
                content: $content,
                query: $query,
                response: $response,
                created_at: $created_at
            })
            RETURN n.id AS id
            """,
            {
                "id": node_id,
                "session_id": session_id,
                "content": content,
                "query": query,
                "response": response,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return node_id

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    async def add_fact(self, session_id: str, fact: str, category: str = "general") -> str:
        """Store a durable fact. Returns the created node ID."""
        node_id = str(uuid4())
        await self._neo4j.execute_write(
            """
            CREATE (n:MemoryFact {
                id: $id,
                session_id: $session_id,
                fact: $fact,
                category: $category,
                created_at: $created_at
            })
            RETURN n.id AS id
            """,
            {
                "id": node_id,
                "session_id": session_id,
                "fact": fact,
                "category": category,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return node_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Basic text search across facts and interactions."""
        results = await self._neo4j.execute_query(
            """
            CALL {
                MATCH (n:MemoryFact)
                WHERE toLower(n.fact) CONTAINS toLower($query)
                RETURN n.id AS id, n.fact AS text, n.category AS category,
                       n.session_id AS session_id, n.created_at AS created_at,
                       'fact' AS type
                UNION ALL
                MATCH (n:MemoryInteraction)
                WHERE toLower(n.content) CONTAINS toLower($query)
                   OR toLower(n.query) CONTAINS toLower($query)
                RETURN n.id AS id, n.content AS text, 'interaction' AS category,
                       n.session_id AS session_id, n.created_at AS created_at,
                       'interaction' AS type
            }
            RETURN id, text, category, session_id, created_at, type
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            {"query": query, "limit": limit},
        )
        return results

    async def get_preferences(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve facts categorised as preferences for a session."""
        return await self._neo4j.execute_query(
            """
            MATCH (n:MemoryFact {session_id: $session_id, category: 'preference'})
            RETURN n.id AS id, n.fact AS fact, n.category AS category,
                   n.created_at AS created_at
            ORDER BY n.created_at DESC
            """,
            {"session_id": session_id},
        )
