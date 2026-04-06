"""Six-pass indexing pipeline for Python repositories."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from src.indexer.neo4j_writer import Neo4jWriter
from src.indexer.passes import (
    repo_structure,
    symbol_extraction,
    import_resolution,
    canonical_identity,
    call_linking,
    derived_artifacts,
)
from src.shared.exceptions import IndexingError

logger = logging.getLogger(__name__)

_PASSES = [
    ("repo_structure", repo_structure),
    ("symbol_extraction", symbol_extraction),
    ("import_resolution", import_resolution),
    ("canonical_identity", canonical_identity),
    ("call_linking", call_linking),
    ("derived_artifacts", derived_artifacts),
]


class IndexingPipeline:
    """Orchestrate the 6-pass indexing pipeline."""

    def __init__(self, writer: Neo4jWriter) -> None:
        self._writer = writer

    @staticmethod
    def repo_id_from_url(url: str) -> str:
        """Derive a stable repo_id from a URL."""
        sanitized = url.rstrip("/").removesuffix(".git")
        # Use last two path segments (org/repo) if possible
        parts = sanitized.split("/")
        if len(parts) >= 2:
            short = f"{parts[-2]}/{parts[-1]}"
        else:
            short = sanitized
        return short

    async def run(
        self,
        repo_path: str,
        repo_url: str = "",
        commit_sha: str = "",
    ) -> dict[str, Any]:
        """Execute all 6 passes sequentially. Returns aggregated stats."""
        repo_id = self.repo_id_from_url(repo_url) if repo_url else "local"

        await self._writer.ensure_schema()

        results: dict[str, Any] = {
            "repo_id": repo_id,
            "commit_sha": commit_sha,
            "passes": {},
        }
        overall_start = time.monotonic()

        for name, module in _PASSES:
            t0 = time.monotonic()
            try:
                stats = await module.run(
                    self._writer, repo_path, repo_id, commit_sha,
                )
                elapsed = time.monotonic() - t0
                results["passes"][name] = {"status": "ok", "stats": stats, "elapsed_s": round(elapsed, 2)}
                logger.info("Pass %s completed in %.2fs: %s", name, elapsed, stats)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                results["passes"][name] = {"status": "error", "error": str(exc), "elapsed_s": round(elapsed, 2)}
                logger.error("Pass %s failed after %.2fs: %s", name, elapsed, exc)
                raise IndexingError(f"Pass {name} failed: {exc}") from exc

        results["total_elapsed_s"] = round(time.monotonic() - overall_start, 2)
        return results
