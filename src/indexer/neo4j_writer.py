"""Batched Neo4j graph writer with idempotent MERGE operations."""

from __future__ import annotations

import datetime
import json
from typing import Any

from src.shared.neo4j_client import Neo4jClient


_BATCH_SIZE = 500

# Labels that carry a symbol_id
_SYMBOL_LABELS = [
    "File", "Module", "Class", "Function", "Method",
    "Parameter", "Decorator", "Import", "Docstring",
]


# Keys that are internal to AST parsing and should not be stored in Neo4j
_INTERNAL_KEYS = {"_raw_name"}


def _sanitize_props(node: dict[str, Any]) -> dict[str, Any]:
    """Convert nested structures to Neo4j-compatible property values.

    Neo4j properties must be primitives or arrays of primitives.
    Nested dicts and lists-of-dicts are JSON-serialized to strings.
    """
    clean: dict[str, Any] = {}
    for k, v in node.items():
        if k in _INTERNAL_KEYS:
            continue
        if isinstance(v, dict):
            clean[k] = json.dumps(v)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            clean[k] = json.dumps(v)
        elif isinstance(v, list):
            # arrays of primitives are fine
            clean[k] = v
        else:
            clean[k] = v
    return clean


class Neo4jWriter:
    """Write nodes and relationships to Neo4j in batches."""

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    # -- schema setup ----------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create constraints and indexes (idempotent)."""
        for label in _SYMBOL_LABELS:
            await self._client.execute_write(
                f"CREATE CONSTRAINT IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.symbol_id IS UNIQUE"
            )

        # Full-text index on name across all symbol labels
        # Neo4j requires explicit label list for full-text indexes
        label_list = ", ".join(_SYMBOL_LABELS)
        try:
            await self._client.execute_write(
                f"CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS "
                f"FOR (n:{label_list}) ON EACH [n.name]"
            )
        except Exception:
            pass  # full-text syntax varies across Neo4j versions

        # Secondary indexes
        for prop in ("qualified_name", "path"):
            for label in _SYMBOL_LABELS:
                try:
                    await self._client.execute_write(
                        f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
                    )
                except Exception:
                    pass

    # -- node writing ----------------------------------------------------------

    async def merge_nodes(
        self,
        label: str,
        nodes: list[dict[str, Any]],
        *,
        repo_id: str = "",
        commit_sha: str = "",
    ) -> int:
        """MERGE a batch of nodes by symbol_id. Returns count of nodes written."""
        now = datetime.datetime.utcnow().isoformat()
        total = 0
        for i in range(0, len(nodes), _BATCH_SIZE):
            batch = nodes[i : i + _BATCH_SIZE]
            sanitized = []
            for node in batch:
                clean = _sanitize_props(node)
                clean.setdefault("repo_id", repo_id)
                clean.setdefault("commit_sha", commit_sha)
                clean["indexed_at"] = now
                sanitized.append(clean)
            await self._client.execute_write(
                f"UNWIND $batch AS props "
                f"MERGE (n:{label} {{symbol_id: props.symbol_id}}) "
                f"SET n += props",
                {"batch": sanitized},
            )
            total += len(batch)
        return total

    # -- relationship writing --------------------------------------------------

    async def merge_relationships(
        self,
        rel_type: str,
        rels: list[dict[str, Any]],
    ) -> int:
        """MERGE relationships between nodes identified by symbol_id.

        Each dict in *rels* must contain ``source_id`` and ``target_id``
        (symbol_id values).  Any extra keys become relationship properties.
        """
        total = 0
        for i in range(0, len(rels), _BATCH_SIZE):
            batch = rels[i : i + _BATCH_SIZE]
            await self._client.execute_write(
                f"UNWIND $batch AS rel "
                f"MATCH (a {{symbol_id: rel.source_id}}) "
                f"MATCH (b {{symbol_id: rel.target_id}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                f"SET r += rel.props",
                {
                    "batch": [
                        {
                            "source_id": r["source_id"],
                            "target_id": r["target_id"],
                            "props": {
                                k: v
                                for k, v in r.items()
                                if k not in ("source_id", "target_id")
                            },
                        }
                        for r in batch
                    ]
                },
            )
            total += len(batch)
        return total
