"""Pass 5: Analyze function bodies to create CALLS edges with confidence scoring."""

from __future__ import annotations

import ast
import os
from pathlib import Path
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
    """Scan function/method bodies for ast.Call nodes and create CALLS edges.

    Caller sids are reconstructed by walking the AST with a scope stack
    that mirrors the one used during symbol extraction, so the resulting
    sid matches nodes already in the graph.
    """
    root = Path(repo_path)

    # Load all callables with their qualified_name and scope
    callables = await writer._client.execute_query(
        "MATCH (e {repo_id: $repo_id}) "
        "WHERE e:Function OR e:Method OR e:Class "
        "RETURN e.symbol_id AS sid, e.name AS name, "
        "       e.qualified_name AS qn, e.scope_chain AS scope, "
        "       labels(e) AS labels",
        {"repo_id": repo_id},
    )

    by_qn: dict[str, str] = {}
    by_name: dict[str, list[str]] = {}
    for c in callables:
        qn = c.get("qn") or ""
        if qn:
            by_qn[qn] = c["sid"]
        by_name.setdefault(c["name"], []).append(c["sid"])

    imports_by_file = await load_imports_by_file(writer, repo_id)

    rels: list[dict[str, Any]] = []
    skipped = 0
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
                skipped += 1
                continue

            file_mod = path_to_module(rel)
            _walk_for_calls(
                tree,
                scope_stack=[],
                file_path=rel,
                file_mod=file_mod,
                repo_id=repo_id,
                by_qn=by_qn,
                by_name=by_name,
                imports_by_file=imports_by_file,
                rels=rels,
                class_stack=[],
            )

    await writer.merge_relationships("CALLS", rels)
    return {"calls_linked": len(rels), "skipped_files": skipped}


# ---------------------------------------------------------------------------
# AST walking with scope-aware caller sid reconstruction
# ---------------------------------------------------------------------------


def _build_sid(
    repo_id: str, kind: str, scope_chain: str, name: str, index: int = 0
) -> str:
    return f"{repo_id}:{kind}:{scope_chain}:{name}:{index}"


def _walk_for_calls(
    node: ast.AST,
    *,
    scope_stack: list[str],
    file_path: str,
    file_mod: str,
    repo_id: str,
    by_qn: dict[str, str],
    by_name: dict[str, list[str]],
    imports_by_file: dict[str, dict[str, str]],
    rels: list[dict[str, Any]],
    class_stack: list[str],
) -> None:
    """Mirror _walk() in ast_parser.py so caller sids match stored nodes."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            scope_stack.append(child.name)
            class_stack.append(child.name)
            _walk_for_calls(
                child,
                scope_stack=scope_stack,
                file_path=file_path,
                file_mod=file_mod,
                repo_id=repo_id,
                by_qn=by_qn,
                by_name=by_name,
                imports_by_file=imports_by_file,
                rels=rels,
                class_stack=class_stack,
            )
            class_stack.pop()
            scope_stack.pop()

        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Scope chain as ast_parser.py builds it
            scope_chain = file_mod
            if scope_stack:
                scope_chain = f"{file_mod}.{'.'.join(scope_stack)}"
            is_method = bool(class_stack) and scope_stack and scope_stack[-1] == class_stack[-1]
            kind = "method" if is_method else "function"
            caller_sid = _build_sid(repo_id, kind, scope_chain, child.name, 0)

            # Collect calls inside this function body
            _collect_function_calls(
                child,
                caller_sid=caller_sid,
                file_path=file_path,
                file_mod=file_mod,
                by_qn=by_qn,
                by_name=by_name,
                imports_by_file=imports_by_file,
                rels=rels,
            )

            # Recurse for nested defs
            scope_stack.append(child.name)
            _walk_for_calls(
                child,
                scope_stack=scope_stack,
                file_path=file_path,
                file_mod=file_mod,
                repo_id=repo_id,
                by_qn=by_qn,
                by_name=by_name,
                imports_by_file=imports_by_file,
                rels=rels,
                class_stack=class_stack,
            )
            scope_stack.pop()


def _collect_function_calls(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    caller_sid: str,
    file_path: str,
    file_mod: str,
    by_qn: dict[str, str],
    by_name: dict[str, list[str]],
    imports_by_file: dict[str, dict[str, str]],
    rels: list[dict[str, Any]],
) -> None:
    seen: set[tuple[str, int]] = set()
    for sub in ast.walk(func):
        if not isinstance(sub, ast.Call):
            continue
        callee = _call_name(sub)
        if not callee:
            continue

        # Attempt resolution
        resolved = resolve_dotted(callee, file_path, file_mod, imports_by_file)
        target_sid: str | None = None
        method = "qn"
        confidence = 1.0

        if resolved in by_qn:
            target_sid = by_qn[resolved]
        else:
            simple = callee.rsplit(".", 1)[-1]
            cands = by_name.get(simple, [])
            if len(cands) == 1:
                target_sid = cands[0]
                method = "name"
                confidence = 0.8
            elif len(cands) > 1:
                target_sid = cands[0]
                method = "ambiguous"
                confidence = 0.5

        if not target_sid or target_sid == caller_sid:
            continue

        line = getattr(sub, "lineno", 0)
        key = (target_sid, line)
        if key in seen:
            continue
        seen.add(key)

        rels.append({
            "source_id": caller_sid,
            "target_id": target_sid,
            "confidence": confidence,
            "resolution_method": method,
            "line": line,
        })


def _call_name(node: ast.Call) -> str:
    """Extract a dotted-name string from an ast.Call."""
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
