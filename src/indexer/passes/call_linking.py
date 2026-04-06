"""Pass 5: Analyze function bodies to create CALLS edges with confidence scoring."""

from __future__ import annotations

import ast
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
    """Scan function/method bodies for ast.Call nodes and create CALLS edges."""
    root = Path(repo_path)

    # Build lookup: name -> list of symbol_ids for callables
    callables = await writer._client.execute_query(
        "MATCH (e {repo_id: $repo_id}) "
        "WHERE e:Function OR e:Method OR e:Class "
        "RETURN e.symbol_id AS sid, e.name AS name, e.qualified_name AS qn",
        {"repo_id": repo_id},
    )
    by_name: dict[str, list[dict[str, str]]] = {}
    for c in callables:
        by_name.setdefault(c["name"], []).append(
            {"sid": c["sid"], "qn": c.get("qn", "")}
        )

    # Also index by qualified_name for higher-confidence matching
    by_qn: dict[str, str] = {}
    for c in callables:
        qn = c.get("qn", "")
        if qn:
            by_qn[qn] = c["sid"]

    # Scan .py files for Call nodes
    rels: list[dict[str, Any]] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            full = Path(dirpath) / fname
            rel = str(full.relative_to(root))
            try:
                source = full.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=rel)
            except (OSError, SyntaxError):
                continue

            _collect_calls(tree, rel, repo_id, by_name, by_qn, rels)

    await writer.merge_relationships("CALLS", rels)
    return {"calls_linked": len(rels)}


def _collect_calls(
    tree: ast.AST,
    file_path: str,
    repo_id: str,
    by_name: dict[str, list[dict[str, str]]],
    by_qn: dict[str, str],
    rels: list[dict[str, Any]],
) -> None:
    """Walk the AST and find Call nodes inside functions/methods."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Determine caller symbol_id
        scope = _get_scope(node, tree)
        caller_sid = ASTParser.build_symbol_id(
            repo_id,
            "method" if scope else "function",
            scope or "<module>",
            node.name,
            0,
        )

        # Find all Call nodes in this function body
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            callee_name = _call_name(child)
            if not callee_name:
                continue

            # Try qualified match first (confidence 1.0)
            if callee_name in by_qn:
                rels.append({
                    "source_id": caller_sid,
                    "target_id": by_qn[callee_name],
                    "confidence": 1.0,
                    "line": getattr(child, "lineno", 0),
                })
                continue

            # Heuristic: match by simple name
            simple = callee_name.split(".")[-1]
            candidates = by_name.get(simple, [])
            if len(candidates) == 1:
                conf = 0.8
            elif len(candidates) > 1:
                conf = 0.5
            else:
                continue  # no match

            rels.append({
                "source_id": caller_sid,
                "target_id": candidates[0]["sid"],
                "confidence": conf,
                "line": getattr(child, "lineno", 0),
            })


def _call_name(node: ast.Call) -> str:
    """Extract a string name from an ast.Call."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        cur: ast.expr = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return ""


def _get_scope(node: ast.AST, tree: ast.AST) -> str:
    """Find the enclosing class name for a function (simple parent check)."""
    for parent in ast.walk(tree):
        if isinstance(parent, ast.ClassDef):
            for child in ast.iter_child_nodes(parent):
                if child is node:
                    return parent.name
    return ""
