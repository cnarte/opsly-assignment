"""Pass 1: Walk cloned repo and create File / Module nodes for every .py file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.indexer.neo4j_writer import Neo4jWriter
from src.indexer.ast_parser import ASTParser


async def run(
    writer: Neo4jWriter,
    repo_path: str,
    repo_id: str,
    commit_sha: str = "",
) -> dict[str, Any]:
    """Create File and Module nodes. Returns stats dict."""
    root = Path(repo_path)
    file_nodes: list[dict[str, Any]] = []
    module_nodes: list[dict[str, Any]] = []

    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full = Path(dirpath) / fname
            rel = str(full.relative_to(root))

            file_sid = ASTParser.build_symbol_id(repo_id, "file", "<repo>", rel, 0)
            file_nodes.append({
                "symbol_id": file_sid,
                "name": fname,
                "path": rel,
                "kind": "file",
            })

            # Derive module dotted path (e.g. src/indexer/server.py -> src.indexer.server)
            mod_path = rel.replace(os.sep, ".").removesuffix(".py")
            if mod_path.endswith(".__init__"):
                mod_path = mod_path.removesuffix(".__init__")
            mod_sid = ASTParser.build_symbol_id(repo_id, "module", "<repo>", mod_path, 0)
            module_nodes.append({
                "symbol_id": mod_sid,
                "name": mod_path,
                "path": rel,
                "kind": "module",
            })

    file_count = await writer.merge_nodes("File", file_nodes, repo_id=repo_id, commit_sha=commit_sha)
    mod_count = await writer.merge_nodes("Module", module_nodes, repo_id=repo_id, commit_sha=commit_sha)

    # CONTAINS relationships: Module -[CONTAINS]-> File
    rels: list[dict[str, Any]] = []
    for fn, mn in zip(file_nodes, module_nodes):
        rels.append({"source_id": mn["symbol_id"], "target_id": fn["symbol_id"]})
    await writer.merge_relationships("CONTAINS", rels)

    return {"files": file_count, "modules": mod_count}
