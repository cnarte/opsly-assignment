"""Indexer Agent MCP server — parses repositories and populates a Neo4j knowledge graph."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.indexer.ast_parser import ASTParser
from src.indexer.neo4j_writer import Neo4jWriter
from src.indexer.pipeline import IndexingPipeline
from src.shared.exceptions import IndexingError
from src.shared.neo4j_client import Neo4jClient
from src.shared.settings import Settings

logger = logging.getLogger(__name__)

mcp = FastMCP("indexer-agent")

# In-memory job tracking (lightweight; swap for Redis in production)
_jobs: dict[str, dict[str, Any]] = {}

settings = Settings()


# -- helpers ----------------------------------------------------------------

def _get_neo4j_client() -> Neo4jClient:
    return Neo4jClient(settings)


# -- tools ------------------------------------------------------------------


@mcp.tool()
async def index_repository(repo_url: str, ref: str = "") -> dict:
    """Full repository indexing - clone repo and run 6-pass pipeline."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "repo_url": repo_url, "ref": ref}

    try:
        # Clone to temp directory
        tmp_dir = tempfile.mkdtemp(prefix="indexer_")
        clone_cmd = ["git", "clone", "--depth", "1"]
        if ref:
            clone_cmd += ["--branch", ref]
        clone_cmd += [repo_url, tmp_dir]

        proc = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise IndexingError(f"git clone failed: {proc.stderr.strip()}")

        # Determine commit SHA
        sha_proc = subprocess.run(
            ["git", "-C", tmp_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        commit_sha = sha_proc.stdout.strip() if sha_proc.returncode == 0 else ""

        # Run pipeline
        client = _get_neo4j_client()
        async with client:
            writer = Neo4jWriter(client)
            pipeline = IndexingPipeline(writer)
            result = await pipeline.run(tmp_dir, repo_url=repo_url, commit_sha=commit_sha)

        _jobs[job_id] = {"status": "completed", **result}
        return {"job_id": job_id, **result}

    except Exception as exc:
        _jobs[job_id] = {"status": "failed", "error": str(exc)}
        raise


@mcp.tool()
async def index_file(repo_path: str, file_path: str) -> dict:
    """Single file indexing."""
    full = Path(repo_path) / file_path
    if not full.exists():
        raise IndexingError(f"File not found: {full}")

    source = full.read_text(encoding="utf-8", errors="replace")
    parser = ASTParser(repo_id="local")
    result = parser.parse(source, file_path=file_path)
    return result


@mcp.tool()
async def parse_python_ast(source_code: str, file_path: str = "") -> dict:
    """Extract AST from Python source code."""
    parser = ASTParser(repo_id="")
    return parser.parse(source_code, file_path=file_path)


@mcp.tool()
async def extract_entities(source_code: str, file_path: str = "") -> dict:
    """Identify code entities and relationships from source."""
    parser = ASTParser(repo_id="analysis")
    result = parser.parse(source_code, file_path=file_path)
    # Flatten into a summary
    entities: list[dict[str, Any]] = []
    for cls in result.get("classes", []):
        entities.append({
            "kind": "class",
            "name": cls["name"],
            "symbol_id": cls["symbol_id"],
            "bases": cls.get("bases", []),
            "start_line": cls["start_line"],
            "end_line": cls["end_line"],
        })
    for fn in result.get("functions", []):
        entities.append({
            "kind": fn["kind"],
            "name": fn["name"],
            "symbol_id": fn["symbol_id"],
            "args": [a["name"] for a in fn.get("args", [])],
            "start_line": fn["start_line"],
            "end_line": fn["end_line"],
        })
    return {"file_path": file_path, "entity_count": len(entities), "entities": entities}


@mcp.tool()
async def get_index_status(job_id: str = "") -> dict:
    """Report indexing progress and statistics."""
    if job_id:
        job = _jobs.get(job_id)
        if not job:
            return {"error": f"Unknown job_id: {job_id}"}
        return {"job_id": job_id, **job}
    return {"total_jobs": len(_jobs), "jobs": _jobs}


# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
