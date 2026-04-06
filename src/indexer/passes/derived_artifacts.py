"""Pass 6: Compute DEPENDS_ON between modules and context neighbourhoods."""

from __future__ import annotations

from typing import Any

from src.indexer.neo4j_writer import Neo4jWriter


async def run(
    writer: Neo4jWriter,
    repo_path: str,
    repo_id: str,
    commit_sha: str = "",
) -> dict[str, Any]:
    """Create DEPENDS_ON edges between Module nodes and INHERITS_FROM edges."""
    depends = await _compute_module_deps(writer, repo_id)
    inherits = await _compute_inheritance(writer, repo_id)
    return {"depends_on": depends, "inherits_from": inherits}


async def _compute_module_deps(writer: Neo4jWriter, repo_id: str) -> int:
    """Create DEPENDS_ON edges by tracing imports back to their owning modules."""
    # Find file -> import -> target-module chains
    records = await writer._client.execute_query(
        "MATCH (m:Module {repo_id: $repo_id})-[:CONTAINS]->(f:File)"
        "-[:CONTAINS]->(i:Import)-[:IMPORTS]->(target) "
        "WITH m, target "
        "OPTIONAL MATCH (tm:Module {repo_id: $repo_id})-[:CONTAINS]->(target) "
        "WHERE tm IS NOT NULL AND tm <> m "
        "RETURN DISTINCT m.symbol_id AS src, tm.symbol_id AS tgt",
        {"repo_id": repo_id},
    )

    # Also try: Module contains file contains import that targets a Module directly
    records2 = await writer._client.execute_query(
        "MATCH (m:Module {repo_id: $repo_id})-[:CONTAINS]->(f:File)"
        "-[:CONTAINS]->(i:Import)-[:IMPORTS]->(tm:Module) "
        "WHERE tm <> m "
        "RETURN DISTINCT m.symbol_id AS src, tm.symbol_id AS tgt",
        {"repo_id": repo_id},
    )

    seen: set[tuple[str, str]] = set()
    rels: list[dict[str, Any]] = []
    for r in records + records2:
        if r["src"] and r["tgt"]:
            key = (r["src"], r["tgt"])
            if key not in seen:
                seen.add(key)
                rels.append({"source_id": r["src"], "target_id": r["tgt"]})

    await writer.merge_relationships("DEPENDS_ON", rels)
    return len(rels)


async def _compute_inheritance(writer: Neo4jWriter, repo_id: str) -> int:
    """Create INHERITS_FROM edges between Class nodes."""
    classes = await writer._client.execute_query(
        "MATCH (c:Class {repo_id: $repo_id}) "
        "RETURN c.symbol_id AS sid, c.name AS name, c.bases AS bases",
        {"repo_id": repo_id},
    )

    # Build name lookup
    name_to_sid: dict[str, list[str]] = {}
    for c in classes:
        name_to_sid.setdefault(c["name"], []).append(c["sid"])

    rels: list[dict[str, Any]] = []
    for c in classes:
        bases = c.get("bases") or []
        if isinstance(bases, str):
            bases = [bases]
        for base in bases:
            simple_base = base.split(".")[-1]
            targets = name_to_sid.get(simple_base, [])
            for tgt in targets:
                if tgt != c["sid"]:
                    rels.append({"source_id": c["sid"], "target_id": tgt})

    await writer.merge_relationships("INHERITS_FROM", rels)
    return len(rels)
