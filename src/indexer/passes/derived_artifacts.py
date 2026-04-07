"""Pass 6: Compute DEPENDS_ON between modules and context neighbourhoods."""

from __future__ import annotations

from typing import Any

from src.indexer.neo4j_writer import Neo4jWriter
from src.indexer.passes._resolution import (
    load_imports_by_file,
    path_to_module,
    resolve_dotted,
)


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
    """Create INHERITS_FROM edges between Class nodes.

    Resolution strategy:
    1. Load per-file import table.
    2. For each Class, resolve each base string to a fully qualified name
       via the file's imports.
    3. Prefer matching by ``qualified_name`` on in-repo Class nodes.
    4. Fall back to matching by simple ``name`` (unambiguous only).
    5. For unresolved external bases, MERGE a stub Class node with
       ``external=true`` so the inheritance chain is still visible.
    """
    imports_by_file = await load_imports_by_file(writer, repo_id)

    classes = await writer._client.execute_query(
        "MATCH (c:Class {repo_id: $repo_id}) "
        "RETURN c.symbol_id AS sid, c.name AS name, c.bases AS bases, "
        "       c.file_path AS fp, c.qualified_name AS qn",
        {"repo_id": repo_id},
    )

    # Lookup tables for in-repo classes
    qn_to_sid: dict[str, str] = {}
    name_to_sid: dict[str, list[str]] = {}
    for c in classes:
        qn = c.get("qn") or ""
        if qn:
            qn_to_sid[qn] = c["sid"]
        name_to_sid.setdefault(c["name"], []).append(c["sid"])

    rels: list[dict[str, Any]] = []
    external_stubs: dict[str, dict[str, Any]] = {}

    for c in classes:
        bases = c.get("bases") or []
        if isinstance(bases, str):
            bases = [bases]
        fp = c.get("fp") or ""
        file_mod = path_to_module(fp)

        for base in bases:
            # Strip generic args like 'Generic[T]' -> 'Generic'
            clean_base = base.split("[", 1)[0].strip()
            if not clean_base or clean_base in ("object", "type", "ABC", "Protocol"):
                continue

            resolved = resolve_dotted(clean_base, fp, file_mod, imports_by_file)
            target_sid: str | None = None

            # 1. exact qualified-name match against in-repo classes
            if resolved in qn_to_sid:
                target_sid = qn_to_sid[resolved]
            else:
                # 2. try the module-local form (file's own top-level)
                local_qn = f"{file_mod}.{clean_base}" if "." not in clean_base else None
                if local_qn and local_qn in qn_to_sid:
                    target_sid = qn_to_sid[local_qn]
                else:
                    # 3. unambiguous simple-name match
                    simple = clean_base.rsplit(".", 1)[-1]
                    cands = name_to_sid.get(simple, [])
                    if len(cands) == 1:
                        target_sid = cands[0]

            if target_sid is None:
                # 4. External stub node
                simple = resolved.rsplit(".", 1)[-1]
                stub_sid = f"{repo_id}:external_class:{resolved}"
                if stub_sid not in external_stubs:
                    external_stubs[stub_sid] = {
                        "symbol_id": stub_sid,
                        "name": simple,
                        "qualified_name": resolved,
                        "kind": "class",
                        "external": True,
                        "file_path": "",
                        "scope_chain": resolved.rsplit(".", 1)[0] if "." in resolved else "",
                    }
                target_sid = stub_sid

            if target_sid != c["sid"]:
                rels.append({
                    "source_id": c["sid"],
                    "target_id": target_sid,
                    "base_expr": base,
                    "resolution_confidence": 1.0 if target_sid in qn_to_sid.values() else 0.5,
                })

    if external_stubs:
        await writer.merge_nodes(
            "Class", list(external_stubs.values()), repo_id=repo_id,
        )
    await writer.merge_relationships("INHERITS_FROM", rels)
    return len(rels)
