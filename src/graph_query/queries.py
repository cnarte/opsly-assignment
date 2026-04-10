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


def _repo_filter(alias: str = "n") -> str:
    """Return the repo_id WHERE clause fragment (appended after existing conditions)."""
    return f"AND ($repo_id = '' OR {alias}.repo_id = $repo_id)"


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def find_entity(name: str, entity_type: str = "", repo_id: str = "") -> CypherQuery:
    """Build a query to locate entities by name."""
    label = _entity_label(entity_type)
    rf = _repo_filter("n")
    if label:
        cypher = (
            f"MATCH (n:{label}) "
            f"WHERE (n.name = $name OR n.qualified_name CONTAINS $name) {rf} "
            "RETURN n LIMIT 20"
        )
    else:
        parts = []
        for lbl in _ALL_ENTITY_LABELS:
            parts.append(
                f"MATCH (n:{lbl}) "
                f"WHERE (n.name = $name OR n.qualified_name CONTAINS $name) {rf} "
                "RETURN n"
            )
        cypher = " UNION ".join(parts) + " LIMIT 20"
    return CypherQuery(cypher=cypher, parameters={"name": name, "repo_id": repo_id})


def get_dependencies(entity_name: str, repo_id: str = "") -> CypherQuery:
    """Build a query to find what an entity depends on."""
    rf = _repo_filter("n")
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        parts.append(
            f"MATCH (n:{lbl})-[r:CALLS|IMPORTS|DEPENDS_ON|INHERITS_FROM]->(dep) "
            f"WHERE n.name = $name {rf} "
            "RETURN n.name AS source, type(r) AS rel, dep.name AS target, labels(dep) AS target_labels"
        )
    cypher = " UNION ".join(parts) + " LIMIT 50"
    return CypherQuery(cypher=cypher, parameters={"name": entity_name, "repo_id": repo_id})


def get_dependents(entity_name: str, repo_id: str = "") -> CypherQuery:
    """Build a query to find what depends on an entity."""
    rf = _repo_filter("n")
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        for dep_lbl in _ALL_ENTITY_LABELS:
            parts.append(
                f"MATCH (dep:{dep_lbl})-[r:CALLS|IMPORTS|DEPENDS_ON|INHERITS_FROM]->(n:{lbl}) "
                f"WHERE n.name = $name {rf} "
                "RETURN dep.name AS source, labels(dep) AS source_labels, type(r) AS rel, n.name AS target"
            )
    cypher = " UNION ".join(parts) + " LIMIT 50"
    return CypherQuery(cypher=cypher, parameters={"name": entity_name, "repo_id": repo_id})


def trace_imports(module_name: str, repo_id: str = "") -> CypherQuery:
    """Build a query to follow import chains from a module."""
    rf = _repo_filter("m")
    cypher = (
        "MATCH path = (m:Module)-[:IMPORTS*1..5]->(target:Module) "
        f"WHERE m.name = $name {rf} "
        "RETURN [node IN nodes(path) | node.name] AS chain "
        "LIMIT 50"
    )
    return CypherQuery(cypher=cypher, parameters={"name": module_name, "repo_id": repo_id})


def find_related(entity_name: str, relationship_type: str, repo_id: str = "") -> CypherQuery:
    """Build a query to get entities related by a specific relationship type."""
    rel_type = relationship_type.replace("`", "").replace("\\", "").strip()
    rf = _repo_filter("n")
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        parts.append(
            f"MATCH (n:{lbl})-[r:`{rel_type}`]->(related) "
            f"WHERE n.name = $name {rf} "
            "RETURN n.name AS source, type(r) AS rel, related.name AS target, labels(related) AS target_labels"
        )
    cypher = " UNION ".join(parts) + " LIMIT 50"
    return CypherQuery(cypher=cypher, parameters={"name": entity_name, "repo_id": repo_id})


def get_symbol_context(symbol_name: str, repo_id: str = "") -> CypherQuery:
    """Build a query for a 360-degree view of a symbol."""
    all_labels = list(ALLOWED_LABELS)
    rf = _repo_filter("n")
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        for rel_lbl in all_labels:
            parts.append(
                f"MATCH (n:{lbl})-[r]->(related:{rel_lbl}) "
                f"WHERE n.name = $name {rf} "
                "RETURN n.name AS symbol, 'outgoing' AS direction, type(r) AS rel, "
                "related.name AS related_name, labels(related) AS related_labels"
            )
            parts.append(
                f"MATCH (related:{rel_lbl})-[r]->(n:{lbl}) "
                f"WHERE n.name = $name {rf} "
                "RETURN n.name AS symbol, 'incoming' AS direction, type(r) AS rel, "
                "related.name AS related_name, labels(related) AS related_labels"
            )
    cypher = " UNION ".join(parts) + " LIMIT 100"
    return CypherQuery(cypher=cypher, parameters={"name": symbol_name, "repo_id": repo_id})


def impact_at_depth(symbol_name: str, depth: int, repo_id: str = "") -> CypherQuery:
    """Build a query for dependents at a specific depth."""
    rf = _repo_filter("n")
    parts = []
    for lbl in _ALL_ENTITY_LABELS:
        parts.append(
            f"MATCH path = (n:{lbl})<-[:CALLS|IMPORTS|DEPENDS_ON|INHERITS_FROM*1..{depth}]-(dependent) "
            f"WHERE n.name = $name {rf} "
            "RETURN DISTINCT dependent.name AS name, labels(dependent) AS labels, "
            "length(path) AS depth"
        )
    cypher = " UNION ".join(parts) + " LIMIT 100"
    return CypherQuery(cypher=cypher, parameters={"name": symbol_name, "repo_id": repo_id})
