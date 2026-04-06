"""Pass 2: Parse each .py file and create Class/Function/Method/Parameter/Decorator/Docstring/Import nodes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.indexer.ast_parser import ASTParser
from src.indexer.neo4j_writer import Neo4jWriter


async def run(
    writer: Neo4jWriter,
    repo_path: str,
    repo_id: str,
    commit_sha: str = "",
) -> dict[str, Any]:
    """Extract symbols from every .py file and write to Neo4j."""
    parser = ASTParser(repo_id=repo_id)
    root = Path(repo_path)
    stats: dict[str, int] = {
        "classes": 0, "functions": 0, "methods": 0,
        "parameters": 0, "decorators": 0, "docstrings": 0, "imports": 0,
    }
    all_classes: list[dict[str, Any]] = []
    all_functions: list[dict[str, Any]] = []
    all_methods: list[dict[str, Any]] = []
    all_params: list[dict[str, Any]] = []
    all_decorators: list[dict[str, Any]] = []
    all_docstrings: list[dict[str, Any]] = []
    all_imports: list[dict[str, Any]] = []
    contains_rels: list[dict[str, Any]] = []
    has_param_rels: list[dict[str, Any]] = []
    decorated_by_rels: list[dict[str, Any]] = []
    documented_by_rels: list[dict[str, Any]] = []

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full = Path(dirpath) / fname
            rel = str(full.relative_to(root))
            try:
                source = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            result = parser.parse(source, file_path=rel)
            if "error" in result:
                continue

            file_sid = ASTParser.build_symbol_id(repo_id, "file", "<repo>", rel, 0)

            for cls in result["classes"]:
                cls.pop("_raw_name", None)
                all_classes.append(cls)
                contains_rels.append({"source_id": file_sid, "target_id": cls["symbol_id"]})
                _collect_decorators(cls, all_decorators, decorated_by_rels, repo_id)
                _collect_docstring(cls, all_docstrings, documented_by_rels, repo_id)

            for fn in result["functions"]:
                if fn["kind"] == "method":
                    all_methods.append(fn)
                else:
                    all_functions.append(fn)
                contains_rels.append({"source_id": file_sid, "target_id": fn["symbol_id"]})
                _collect_args(fn, all_params, has_param_rels, repo_id)
                _collect_decorators(fn, all_decorators, decorated_by_rels, repo_id)
                _collect_docstring(fn, all_docstrings, documented_by_rels, repo_id)

            for imp in result["imports"]:
                imp_sid = ASTParser.build_symbol_id(
                    repo_id, "import", rel, imp["name"], imp.get("line", 0),
                )
                imp["symbol_id"] = imp_sid
                imp["file_path"] = rel
                all_imports.append(imp)
                contains_rels.append({"source_id": file_sid, "target_id": imp_sid})

    # Write nodes
    stats["classes"] = await writer.merge_nodes("Class", all_classes, repo_id=repo_id, commit_sha=commit_sha)
    stats["functions"] = await writer.merge_nodes("Function", all_functions, repo_id=repo_id, commit_sha=commit_sha)
    stats["methods"] = await writer.merge_nodes("Method", all_methods, repo_id=repo_id, commit_sha=commit_sha)
    stats["parameters"] = await writer.merge_nodes("Parameter", all_params, repo_id=repo_id, commit_sha=commit_sha)
    stats["decorators"] = await writer.merge_nodes("Decorator", all_decorators, repo_id=repo_id, commit_sha=commit_sha)
    stats["docstrings"] = await writer.merge_nodes("Docstring", all_docstrings, repo_id=repo_id, commit_sha=commit_sha)
    stats["imports"] = await writer.merge_nodes("Import", all_imports, repo_id=repo_id, commit_sha=commit_sha)

    # Write relationships
    await writer.merge_relationships("CONTAINS", contains_rels)
    await writer.merge_relationships("HAS_PARAMETER", has_param_rels)
    await writer.merge_relationships("DECORATED_BY", decorated_by_rels)
    await writer.merge_relationships("DOCUMENTED_BY", documented_by_rels)

    return stats


# -- helpers ----------------------------------------------------------------

def _collect_args(
    entity: dict, params: list, rels: list, repo_id: str,
) -> None:
    for i, arg in enumerate(entity.get("args", [])):
        sid = f"{entity['symbol_id']}:param:{arg['name']}:{i}"
        params.append({
            "symbol_id": sid,
            "name": arg["name"],
            "annotation": arg.get("annotation"),
            "kind": arg.get("kind", "positional"),
        })
        rels.append({"source_id": entity["symbol_id"], "target_id": sid})


def _collect_decorators(
    entity: dict, decorators: list, rels: list, repo_id: str,
) -> None:
    for i, dec_name in enumerate(entity.get("decorators", [])):
        sid = f"{entity['symbol_id']}:decorator:{dec_name}:{i}"
        decorators.append({"symbol_id": sid, "name": dec_name, "kind": "decorator"})
        rels.append({"source_id": entity["symbol_id"], "target_id": sid})


def _collect_docstring(
    entity: dict, docstrings: list, rels: list, repo_id: str,
) -> None:
    ds = entity.get("docstring")
    if not ds:
        return
    sid = f"{entity['symbol_id']}:docstring:0"
    docstrings.append({"symbol_id": sid, "text": ds, "kind": "docstring", "name": "docstring"})
    rels.append({"source_id": entity["symbol_id"], "target_id": sid})
