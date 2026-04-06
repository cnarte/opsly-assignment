"""Python AST extraction for code entity discovery."""

from __future__ import annotations

import ast
import hashlib
from typing import Any


class ASTParser:
    """Parse Python source code and extract structured entities using stdlib ast."""

    def __init__(self, repo_id: str = "") -> None:
        self.repo_id = repo_id

    # -- public API ------------------------------------------------------------

    def parse(self, source: str, file_path: str = "") -> dict[str, Any]:
        """Parse *source* and return a structured dict of all entities."""
        try:
            tree = ast.parse(source, filename=file_path or "<string>")
        except SyntaxError as exc:
            return {"error": str(exc), "file_path": file_path, "entities": {}}

        ctx = _ExtractionContext(
            repo_id=self.repo_id,
            file_path=file_path,
            source_lines=source.splitlines(),
        )
        _walk(tree, ctx)

        return {
            "file_path": file_path,
            "module_docstring": ast.get_docstring(tree),
            "all_exports": ctx.all_exports,
            "classes": ctx.classes,
            "functions": ctx.functions,
            "imports": ctx.imports,
        }

    # -- convenience -----------------------------------------------------------

    @staticmethod
    def build_symbol_id(
        repo_id: str, kind: str, scope_chain: str, name: str, index: int = 0,
    ) -> str:
        """Build a deterministic symbol_id.

        Format: ``{repo_id}:{kind}:{scope_chain}:{name}:{index}``
        """
        return f"{repo_id}:{kind}:{scope_chain}:{name}:{index}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ExtractionContext:
    """Mutable accumulator carried through the AST walk."""

    def __init__(
        self, repo_id: str, file_path: str, source_lines: list[str],
    ) -> None:
        self.repo_id = repo_id
        self.file_path = file_path
        self.source_lines = source_lines
        self.scope_stack: list[str] = []
        self.classes: list[dict[str, Any]] = []
        self.functions: list[dict[str, Any]] = []
        self.imports: list[dict[str, Any]] = []
        self.all_exports: list[str] | None = None
        # Track occurrence counts per (scope, kind, name) for index assignment
        self._occurrence: dict[tuple[str, str, str], int] = {}
        # Derive module base from file_path for scope chain root
        self._module_base = self._path_to_module(file_path)

    @staticmethod
    def _path_to_module(file_path: str) -> str:
        """Convert a file path like 'fastapi/routing.py' to 'fastapi.routing'."""
        if not file_path:
            return "<module>"
        # Strip .py extension and convert / to .
        p = file_path.replace("\\", "/")
        if p.endswith(".py"):
            p = p[:-3]
        if p.endswith("/__init__"):
            p = p[:-9]
        return p.replace("/", ".") if p else "<module>"

    @property
    def scope_chain(self) -> str:
        if self.scope_stack:
            return f"{self._module_base}.{'.'.join(self.scope_stack)}"
        return self._module_base

    def next_index(self, kind: str, name: str) -> int:
        key = (self.scope_chain, kind, name)
        idx = self._occurrence.get(key, 0)
        self._occurrence[key] = idx + 1
        return idx


def _decorator_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    names: list[str] = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(ast.dump(dec))  # fallback
            # try reconstructing dotted name
            parts: list[str] = []
            cur: ast.expr = dec
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                names[-1] = ".".join(reversed(parts))
        elif isinstance(dec, ast.Call):
            # decorator with arguments – extract the callable name
            inner = dec.func
            if isinstance(inner, ast.Name):
                names.append(inner.id)
            elif isinstance(inner, ast.Attribute):
                parts = []
                cur = inner
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                    names.append(".".join(reversed(parts)))
                else:
                    names.append(ast.dump(inner))
            else:
                names.append(ast.dump(inner))
        else:
            names.append(ast.dump(dec))
    return names


def _annotation_str(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _extract_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict[str, Any]]:
    """Extract function arguments with annotations."""
    args_info: list[dict[str, Any]] = []
    all_args = node.args

    # positional args (includes 'self'/'cls')
    for arg in all_args.args:
        args_info.append({
            "name": arg.arg,
            "annotation": _annotation_str(arg.annotation),
            "kind": "positional",
        })

    for arg in all_args.posonlyargs:
        args_info.append({
            "name": arg.arg,
            "annotation": _annotation_str(arg.annotation),
            "kind": "positional_only",
        })

    if all_args.vararg:
        args_info.append({
            "name": all_args.vararg.arg,
            "annotation": _annotation_str(all_args.vararg.annotation),
            "kind": "var_positional",
        })

    for arg in all_args.kwonlyargs:
        args_info.append({
            "name": arg.arg,
            "annotation": _annotation_str(arg.annotation),
            "kind": "keyword_only",
        })

    if all_args.kwarg:
        args_info.append({
            "name": all_args.kwarg.arg,
            "annotation": _annotation_str(all_args.kwarg.annotation),
            "kind": "var_keyword",
        })

    return args_info


def _walk(node: ast.AST, ctx: _ExtractionContext) -> None:
    """Recursively walk the AST and extract entities."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            _handle_class(child, ctx)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _handle_function(child, ctx, is_method=len(ctx.scope_stack) > 0 and ctx.classes and ctx.scope_stack[-1] == ctx.classes[-1].get("_raw_name"))
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            _handle_import(child, ctx)
        elif isinstance(child, ast.Assign):
            _handle_assign(child, ctx)


def _handle_class(node: ast.ClassDef, ctx: _ExtractionContext) -> None:
    kind = "class"
    idx = ctx.next_index(kind, node.name)
    symbol_id = ASTParser.build_symbol_id(
        ctx.repo_id, kind, ctx.scope_chain, node.name, idx,
    )
    bases = [ast.unparse(b) for b in node.bases]
    info: dict[str, Any] = {
        "symbol_id": symbol_id,
        "name": node.name,
        "kind": kind,
        "bases": bases,
        "decorators": _decorator_names(node),
        "docstring": ast.get_docstring(node),
        "start_line": node.lineno,
        "end_line": node.end_lineno or node.lineno,
        "scope_chain": ctx.scope_chain,
        "file_path": ctx.file_path,
        "_raw_name": node.name,
    }
    ctx.classes.append(info)

    # Recurse into class body
    ctx.scope_stack.append(node.name)
    _walk(node, ctx)
    ctx.scope_stack.pop()


def _handle_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    ctx: _ExtractionContext,
    *,
    is_method: bool = False,
) -> None:
    kind = "method" if is_method else "function"
    idx = ctx.next_index(kind, node.name)
    symbol_id = ASTParser.build_symbol_id(
        ctx.repo_id, kind, ctx.scope_chain, node.name, idx,
    )
    info: dict[str, Any] = {
        "symbol_id": symbol_id,
        "name": node.name,
        "kind": kind,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "args": _extract_args(node),
        "decorators": _decorator_names(node),
        "docstring": ast.get_docstring(node),
        "return_type": _annotation_str(node.returns),
        "start_line": node.lineno,
        "end_line": node.end_lineno or node.lineno,
        "scope_chain": ctx.scope_chain,
        "file_path": ctx.file_path,
    }
    ctx.functions.append(info)

    # Recurse for nested functions
    ctx.scope_stack.append(node.name)
    _walk(node, ctx)
    ctx.scope_stack.pop()


def _handle_import(node: ast.Import | ast.ImportFrom, ctx: _ExtractionContext) -> None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            ctx.imports.append({
                "type": "import",
                "module": alias.name,
                "name": alias.name,
                "alias": alias.asname,
                "line": node.lineno,
            })
    else:
        module = node.module or ""
        level = node.level or 0
        for alias in node.names:
            ctx.imports.append({
                "type": "from_import",
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "level": level,
                "line": node.lineno,
            })


def _handle_assign(node: ast.Assign, ctx: _ExtractionContext) -> None:
    """Detect ``__all__ = [...]`` at module level."""
    if ctx.scope_stack:
        return
    for target in node.targets:
        if isinstance(target, ast.Name) and target.id == "__all__":
            if isinstance(node.value, (ast.List, ast.Tuple)):
                ctx.all_exports = [
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
