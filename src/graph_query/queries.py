"""Parameterized Cypher query builders for the graph query agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.graph_query.safety import ALLOWED_LABELS


@dataclass(frozen=True)
class CypherQuery:
    """A parameterized Cypher query with its parameters."""

    cypher: str
    parameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

_ALL_ENTITY_LABELS = ["Class", "Function", "Method", "Module", "File"]

_DEPENDENCY_RELS = "CALLS|IMPORTS|DEPENDS_ON|INHERITS_FROM"


def _entity_label(entity_type: str) -> str:
    """Return a validated label or raise ValueError."""
    if entity_type and entity_type in ALLOWED_LABELS:
        return entity_type
    if entity_type:
        raise ValueError(
            f"Unknown entity type '{entity_type}'. "
            f"Allowed: {sorted(ALLOWED_LABELS)}"
        )
    return ""


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def find_entity(name: str, entity_type: str = "") -> CypherQuery:
    """Build a query to locate entities by name."""
    label = _entity_label(entity_type)
    if label:
        cypher = (
            f"MATCH (n:{label}) "
            "WHERE n.name = $name OR n.qualified_name CONTAINS $name "
            "RETURN n LIMIT 20"
        )
    else:
        # Union across all entity labels to avoid bare MATCH (n)
        parts = []
        for lbl in _ALL_ENTITY_LABELS:
            parts.append(
                f"MATCH (n:{lbl}) "
                "WHERE n.name = $name OR n.qualified_name CONTAINS $name "
                "RETURN n"
            )
        cypher = " UNION ".join(parts) + " LIMIT 20"
    return CypherQuery(cypher=cypher, parameters={"name": name})


def get_dependencies(entity_name: str) -> CypherQuery:
    """Build a query to find what an entity depends on."""
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        parts.append(
            f"MATCH (n:{lbl})-[r:CALLS|IMPORTS|DEPENDS_ON|INHERITS_FROM]->(dep) "
            "WHERE n.name = $name "
            "RETURN n.name AS source, type(r) AS rel, dep.name AS target, labels(dep) AS target_labels"
        )
    cypher = " UNION ".join(parts) + " LIMIT 50"
    return CypherQuery(cypher=cypher, parameters={"name": entity_name})


def get_dependents(entity_name: str) -> CypherQuery:
    """Build a query to find what depends on an entity."""
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        for dep_lbl in _ALL_ENTITY_LABELS:
            parts.append(
                f"MATCH (dep:{dep_lbl})-[r:CALLS|IMPORTS|DEPENDS_ON|INHERITS_FROM]->(n:{lbl}) "
                "WHERE n.name = $name "
                "RETURN dep.name AS source, labels(dep) AS source_labels, type(r) AS rel, n.name AS target"
            )
    cypher = " UNION ".join(parts) + " LIMIT 50"
    return CypherQuery(cypher=cypher, parameters={"name": entity_name})


def trace_imports(module_name: str) -> CypherQuery:
    """Build a query to follow import chains from a module."""
    cypher = (
        "MATCH path = (m:Module)-[:IMPORTS*1..5]->(target:Module) "
        "WHERE m.name = $name "
        "RETURN [node IN nodes(path) | node.name] AS chain "
        "LIMIT 50"
    )
    return CypherQuery(cypher=cypher, parameters={"name": module_name})


def find_related(entity_name: str, relationship_type: str) -> CypherQuery:
    """Build a query to get entities related by a specific relationship type."""
    # Sanitize relationship type to prevent injection
    rel_type = relationship_type.replace("`", "").replace("\\", "").strip()
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        parts.append(
            f"MATCH (n:{lbl})-[r:`{rel_type}`]->(related) "
            "WHERE n.name = $name "
            "RETURN n.name AS source, type(r) AS rel, related.name AS target, labels(related) AS target_labels"
        )
    cypher = " UNION ".join(parts) + " LIMIT 50"
    return CypherQuery(cypher=cypher, parameters={"name": entity_name})


def get_symbol_context(symbol_name: str) -> CypherQuery:
    """Build a query for a 360-degree view of a symbol."""
    all_labels = list(ALLOWED_LABELS)
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        for rel_lbl in all_labels:
            parts.append(
                f"MATCH (n:{lbl})-[r]->(related:{rel_lbl}) "
                "WHERE n.name = $name "
                "RETURN n.name AS symbol, 'outgoing' AS direction, type(r) AS rel, "
                "related.name AS related_name, labels(related) AS related_labels"
            )
            parts.append(
                f"MATCH (related:{rel_lbl})-[r]->(n:{lbl}) "
                "WHERE n.name = $name "
                "RETURN n.name AS symbol, 'incoming' AS direction, type(r) AS rel, "
                "related.name AS related_name, labels(related) AS related_labels"
            )
    cypher = " UNION ".join(parts) + " LIMIT 100"
    return CypherQuery(cypher=cypher, parameters={"name": symbol_name})


def impact_at_depth(symbol_name: str, depth: int) -> CypherQuery:
    """Build a query for dependents at a specific depth."""
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        parts.append(
            f"MATCH path = (n:{lbl})<-[:CALLS|IMPORTS|DEPENDS_ON|INHERITS_FROM*1..{depth}]-(dependent) "
            "WHERE n.name = $name "
            "RETURN DISTINCT dependent.name AS name, labels(dependent) AS labels, "
            "length(path) AS depth"
        )
    cypher = " UNION ".join(parts) + " LIMIT 100"
    return CypherQuery(cypher=cypher, parameters={"name": symbol_name})
