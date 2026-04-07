"""Shared symbol-resolution utilities used by inheritance + call linking.

Builds a per-file import table and resolves dotted names to qualified
targets via that table, the module scope, and the global symbol index.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.indexer.neo4j_writer import Neo4jWriter


def path_to_module(file_path: str) -> str:
    p = file_path.replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    if p.endswith("/__init__"):
        p = p[:-9]
    return p.replace("/", ".") if p else "<module>"


async def load_imports_by_file(
    writer: Neo4jWriter, repo_id: str
) -> dict[str, dict[str, str]]:
    """Return ``file_path -> {local_alias: fully_qualified_name}`` mapping.

    Handles:
    - ``import X``              -> {'X': 'X'}
    - ``import X.Y as Z``       -> {'Z': 'X.Y'}
    - ``from X import Y``       -> {'Y': 'X.Y'}
    - ``from X import Y as Z``  -> {'Z': 'X.Y'}
    - ``from . import Y``       -> relative: resolved against file's package
    """
    rows = await writer._client.execute_query(
        "MATCH (i:Import {repo_id: $repo_id}) "
        "RETURN i.file_path AS fp, i.type AS t, i.module AS module, "
        "       i.name AS name, i.alias AS alias, i.level AS level",
        {"repo_id": repo_id},
    )

    table: dict[str, dict[str, str]] = {}
    for r in rows:
        fp = r.get("fp") or ""
        if not fp:
            continue
        t = r.get("t") or ""
        module = r.get("module") or ""
        name = r.get("name") or ""
        alias = r.get("alias")
        level = r.get("level") or 0

        # Handle relative imports: resolve level dots against file's package
        if level:
            file_mod = path_to_module(fp)
            pkg_parts = file_mod.split(".")[:-level] if level <= len(file_mod.split(".")) else []
            base = ".".join(pkg_parts)
            module = f"{base}.{module}" if module else base

        if t == "from_import":
            local = alias or name
            fq = f"{module}.{name}" if module else name
        else:  # plain import
            # 'import foo.bar' exposes 'foo' unless aliased
            local = alias or name.split(".")[0]
            fq = name

        if not local:
            continue
        table.setdefault(fp, {})[local] = fq

    return table


def resolve_dotted(
    ref: str,
    file_path: str,
    file_module: str,
    imports: dict[str, dict[str, str]],
) -> str:
    """Resolve a dotted reference (e.g. 'routing.Router') to its fully
    qualified name using the file's import table + module scope.

    Returns the best-guess fully-qualified name. If unresolved, returns the
    input unchanged.
    """
    if not ref:
        return ref
    file_imports = imports.get(file_path, {})
    head, _, tail = ref.partition(".")

    # 1. head is a local import alias
    if head in file_imports:
        mapped = file_imports[head]
        return f"{mapped}.{tail}" if tail else mapped

    # 2. head is the current module's own top-level symbol
    #    -> prepend the file's module path
    #    (the caller can decide if this hit an in-repo node)
    return f"{file_module}.{ref}"
