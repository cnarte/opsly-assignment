"""Pass 3: Resolve import targets to actual modules/symbols in the repo."""

from __future__ import annotations

from typing import Any

from src.indexer.ast_parser import ASTParser
from src.indexer.neo4j_writer import Neo4jWriter


async def run(
    writer: Neo4jWriter,
    repo_path: str,
    repo_id: str,
    commit_sha: str = "",
) -> dict[str, Any]:
    """Create IMPORTS relationships between files/modules and their targets."""
    # Fetch all Import nodes for this repo
    imports = await writer._client.execute_query(
        "MATCH (i:Import {repo_id: $repo_id}) RETURN i",
        {"repo_id": repo_id},
    )
    # Fetch all Module nodes for resolution lookup
    modules = await writer._client.execute_query(
        "MATCH (m:Module {repo_id: $repo_id}) RETURN m.name AS name, m.symbol_id AS sid",
        {"repo_id": repo_id},
    )
    mod_lookup: dict[str, str] = {m["name"]: m["sid"] for m in modules}

    # Fetch all Class/Function/Method nodes for symbol resolution
    symbols = await writer._client.execute_query(
        "MATCH (s {repo_id: $repo_id}) WHERE s:Class OR s:Function OR s:Method "
        "RETURN s.name AS name, s.symbol_id AS sid, s.scope_chain AS scope",
        {"repo_id": repo_id},
    )
    sym_by_name: dict[str, list[str]] = {}
    for s in symbols:
        sym_by_name.setdefault(s["name"], []).append(s["sid"])

    rels: list[dict[str, Any]] = []
    resolved = 0

    for record in imports:
        imp = record["i"]
        imp_sid = imp.get("symbol_id", "")
        module_name = imp.get("module", "")
        import_name = imp.get("name", "")
        imp_type = imp.get("type", "")

        target_sid: str | None = None

        if imp_type == "from_import":
            # Try module.name first (e.g., from foo.bar import Baz -> foo.bar.Baz)
            full_qual = f"{module_name}.{import_name}" if module_name else import_name
            target_sid = mod_lookup.get(full_qual)
            if not target_sid:
                target_sid = mod_lookup.get(module_name)
            if not target_sid and import_name in sym_by_name:
                target_sid = sym_by_name[import_name][0]
        else:
            # plain import
            target_sid = mod_lookup.get(import_name)

        if target_sid:
            rels.append({
                "source_id": imp_sid,
                "target_id": target_sid,
                "resolved": True,
            })
            resolved += 1

    await writer.merge_relationships("IMPORTS", rels)
    return {"total_imports": len(imports), "resolved": resolved}
