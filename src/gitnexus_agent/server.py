"""GitNexus Agent — MCP server that wraps the gitnexus CLI directly."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import asyncio
import tempfile
import anyio
from mcp.server.fastmcp import FastMCP

from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP(
    "gitnexus-agent",
    host="0.0.0.0",
    port=settings.GITNEXUS_PORT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_gitnexus(*args: str) -> dict:
    """Run `gitnexus <args>` and return a result dict.

    Routes stdout to a temp file to bypass the 64 KB pipe-buffer limit that
    causes Node.js (gitnexus) to truncate output when connected to a pipe.
    """
    cmd = ["gitnexus", *args]
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        with open(tmp_path, "wb") as out_fh:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=out_fh,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr_bytes = await proc.communicate()

        with open(tmp_path, "rb") as f:
            stdout_bytes = f.read()

        import os as _os
        _os.unlink(tmp_path)
    except Exception as exc:
        return {"error": str(exc)}

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip() if stderr_bytes else ""

    if proc.returncode != 0:
        return {"error": stderr or stdout or f"gitnexus exited {proc.returncode}"}

    # Try to parse JSON output; fall back to raw text wrapper
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return {"result": stdout}


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@mcp.tool()
async def analyze_repo(path: str, repo_name: str) -> dict:
    """Index a repository at `path` using `gitnexus analyze` (or `index` if already indexed).

    Args:
        path: Absolute path to the cloned repository on disk.
        repo_name: Short identifier used in subsequent queries (e.g. 'fastapi').

    Returns:
        {"status": "indexed", "repo": repo_name, "path": path}
    """
    logger.info("Indexing repo '%s' at %s", repo_name, path)

    # Use `gitnexus index` if .gitnexus already exists (fast, no re-analysis)
    gitnexus_dir = Path(path) / ".gitnexus"
    if gitnexus_dir.exists():
        result = await anyio.run_process(["gitnexus", "index", path], check=False)
    else:
        result = await anyio.run_process(["gitnexus", "analyze", path], check=False)

    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()

    if result.returncode != 0:
        err = stderr or stdout
        raise RuntimeError(f"gitnexus failed for '{repo_name}': {err}")

    logger.info("Indexed '%s' successfully", repo_name)
    return {"status": "indexed", "repo": repo_name, "path": path}


# ---------------------------------------------------------------------------
# Query tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def query(q: str, repo: str = "") -> dict:
    """Hybrid BM25 + semantic search across the indexed codebase.

    Args:
        q: Natural language or keyword query.
        repo: Repo name to scope search. Empty = all repos.
    """
    args = ["query", q]
    if repo:
        args += ["--repo", repo]
    return await _run_gitnexus(*args)


@mcp.tool()
async def context(symbol: str, repo: str = "") -> dict:
    """360-degree view of a symbol: callers, callees, imports, process participation.

    Args:
        symbol: Exact or partial symbol name (class, function, method).
        repo: Repo name to scope. Empty = all repos.
    """
    args = ["context", symbol]
    if repo:
        args += ["--repo", repo]
    return await _run_gitnexus(*args)


@mcp.tool()
async def impact(symbol: str, repo: str = "") -> dict:
    """Blast-radius analysis: what breaks if this symbol changes.

    Args:
        symbol: Symbol name to analyse.
        repo: Repo name to scope. Empty = all repos.
    """
    args = ["impact", symbol]
    if repo:
        args += ["--repo", repo]
    return await _run_gitnexus(*args)


@mcp.tool()
async def cypher(query_str: str, repo: str = "") -> dict:
    """Run a raw Cypher query against the LadybugDB graph (read-only).

    Args:
        query_str: Cypher query string.
        repo: Repo name to scope. Empty = all repos.
    """
    args = ["cypher", query_str]
    if repo:
        args += ["--repo", repo]
    return await _run_gitnexus(*args)


@mcp.tool()
async def list_repos() -> dict:
    """Return all repositories currently indexed in LadybugDB.

    Returns a JSON object with a "repos" list of repo name strings.
    """
    result = await _run_gitnexus("list")
    if "error" in result:
        return result

    text: str = result.get("result", "")

    # Parse the text output of `gitnexus list`:
    # Lines like "  fastapi" after the header line
    repos: list[str] = []
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        # Skip header and blank lines
        if not stripped or stripped.startswith("Indexed Repositories"):
            continue
        # Lines with repo names start at indent level 2 with no leading key
        # Detail lines have "Path:", "Indexed:", "Commit:", "Stats:", "Clusters:", "Processes:"
        if not any(stripped.startswith(k) for k in ("Path:", "Indexed:", "Commit:", "Stats:", "Clusters:", "Processes:")):
            # Likely a repo name
            if " " not in stripped or (not stripped[0].isupper() and ":" not in stripped):
                repos.append(stripped)

    return {"repos": repos}


@mcp.tool()
async def group_query(q: str) -> dict:
    """Search for execution flows and contracts across ALL indexed repos.

    Args:
        q: Natural language query spanning multiple repos.
    """
    return await _run_gitnexus("query", q)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
