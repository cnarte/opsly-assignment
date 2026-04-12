"""Indexer Agent MCP server — delegates to gitnexus-agent for indexing."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP("indexer-agent", host="0.0.0.0", port=settings.INDEXER_PORT)

# In-memory job tracking
_jobs: dict[str, dict[str, Any]] = {}

WORKSPACE = Path(os.getenv("WORKSPACE_PATH", "/workspace/repos"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _clone_repo(repo_url: str, ref: str, dest: Path) -> str:
    """Clone `repo_url` into `dest` and return the commit SHA.

    If `dest` already contains a git repository, skip cloning and just
    return the current HEAD commit SHA.
    """
    git_dir = dest / ".git"
    if git_dir.exists():
        logger.info("Repo already exists at %s, skipping clone", dest)
    else:
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [repo_url, str(dest)]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"git clone failed: {stderr.decode().strip()}")
    sha_proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(dest), "rev-parse", "HEAD",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    sha_out, _ = await sha_proc.communicate()
    return sha_out.decode().strip() if sha_proc.returncode == 0 else ""


async def _call_gitnexus(path: str, repo_name: str) -> dict:
    """Call gitnexus-agent.analyze_repo via HTTP MCP."""
    host = "gitnexus-agent" if os.path.exists("/.dockerenv") else "localhost"
    url = f"http://{host}:{settings.GITNEXUS_PORT}/mcp"

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "analyze_repo", {"path": path, "repo_name": repo_name}
            )
            if result.content:
                return json.loads(result.content[0].text)
            return {}


async def _analyze_repo_direct(path: str, repo_name: str) -> dict:
    """Run `gitnexus analyze <path>` directly (no MCP hop) and register with graph-query."""
    logger.info("Indexing repo '%s' at %s directly", repo_name, path)

    # Check if already indexed — try `gitnexus index` first (fast path)
    gitnexus_dir = Path(path) / ".gitnexus"
    if gitnexus_dir.exists():
        cmd = ["gitnexus", "index", path]
    else:
        cmd = ["gitnexus", "analyze", path]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        out = stderr.decode().strip() or stdout.decode().strip()
        raise RuntimeError(f"gitnexus analyze/index failed for '{repo_name}': {out}")

    logger.info("Indexed '%s' successfully", repo_name)
    return {"status": "indexed", "repo": repo_name, "path": path}


# ---------------------------------------------------------------------------
# MCP tools (assignment-required names)
# ---------------------------------------------------------------------------


@mcp.tool()
async def index_repository(repo_url: str, ref: str = "", repo_name: str = "") -> dict:
    """Clone a repository and index it with GitNexus.

    Returns immediately with a job_id. Poll get_index_status for progress.
    """
    job_id = str(uuid.uuid4())
    name = repo_name or repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    _jobs[job_id] = {"status": "started", "repo_url": repo_url, "repo_name": name}

    async def _run() -> None:
        try:
            dest = WORKSPACE / name
            dest.mkdir(parents=True, exist_ok=True)
            _jobs[job_id]["status"] = "cloning"
            commit_sha = await _clone_repo(repo_url, ref, dest)
            _jobs[job_id].update({"status": "indexing", "commit_sha": commit_sha})
            result = await _call_gitnexus(str(dest), name)
            _jobs[job_id].update({"status": "completed", **result})
            logger.info("Job %s completed: %s", job_id, result)
        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, exc)
            _jobs[job_id] = {"status": "failed", "error": str(exc)}

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "started"}


@mcp.tool()
async def index_file(repo_path: str, file_path: str) -> dict:
    """Parse a single Python file and return its entities (no graph write)."""
    import ast as _ast

    full = Path(repo_path) / file_path
    if not full.exists():
        return {"error": f"File not found: {full}"}
    source = full.read_text(encoding="utf-8", errors="replace")
    try:
        tree = _ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        return {"error": f"SyntaxError: {exc}"}
    classes = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)]
    functions = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef | _ast.AsyncFunctionDef)]
    return {"file_path": file_path, "classes": classes, "functions": functions}


@mcp.tool()
async def parse_python_ast(source_code: str, file_path: str = "") -> dict:
    """Extract AST summary from Python source code."""
    import ast as _ast

    try:
        tree = _ast.parse(source_code, filename=file_path or "<string>")
    except SyntaxError as exc:
        return {"error": f"SyntaxError: {exc}"}
    classes = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)]
    functions = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef | _ast.AsyncFunctionDef)]
    imports = [
        _ast.unparse(n) for n in _ast.walk(tree)
        if isinstance(n, _ast.Import | _ast.ImportFrom)
    ]
    return {"file_path": file_path, "classes": classes, "functions": functions, "imports": imports}


@mcp.tool()
async def extract_entities(source_code: str, file_path: str = "") -> dict:
    """Identify code entities from source code."""
    parsed = await parse_python_ast(source_code, file_path)
    if "error" in parsed:
        return parsed
    entities = (
        [{"kind": "class", "name": n} for n in parsed["classes"]]
        + [{"kind": "function", "name": n} for n in parsed["functions"]]
    )
    return {"file_path": file_path, "entity_count": len(entities), "entities": entities}


@mcp.tool()
async def get_index_status(job_id: str = "") -> dict:
    """Return status of an indexing job, or all jobs if job_id is empty."""
    if job_id:
        job = _jobs.get(job_id)
        if not job:
            return {"error": f"Unknown job_id: {job_id}"}
        return {"job_id": job_id, **job}
    return {"total_jobs": len(_jobs), "jobs": _jobs}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
