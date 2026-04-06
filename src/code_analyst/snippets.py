"""Source code extraction and entity location utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.shared.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class SnippetExtractor:
    """Extract source code snippets from files, optionally using Neo4j for entity lookup."""

    def __init__(self, neo4j: Neo4jClient | None = None) -> None:
        self._neo4j = neo4j

    def extract_snippet(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        context_lines: int = 5,
    ) -> dict[str, Any]:
        """Read a file and return the requested line range with surrounding context."""
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:
            return {"error": f"Failed to read file: {exc}"}

        total_lines = len(lines)
        ctx_start = max(0, start_line - 1 - context_lines)
        ctx_end = min(total_lines, end_line + context_lines)

        snippet_lines = lines[ctx_start:ctx_end]
        snippet = "\n".join(snippet_lines)

        return {
            "file_path": file_path,
            "start_line": ctx_start + 1,
            "end_line": ctx_end,
            "entity_start_line": start_line,
            "entity_end_line": end_line,
            "total_lines": total_lines,
            "snippet": snippet,
        }

    async def find_entity_source(
        self, entity_name: str, repo_path: str = ""
    ) -> dict[str, Any]:
        """Look up an entity's source location from the Neo4j graph."""
        if self._neo4j is None:
            return {"error": "No Neo4j client available for entity lookup."}

        try:
            # Try function first, then class
            records = await self._neo4j.execute_query(
                """
                MATCH (e)
                WHERE (e:Function OR e:Class) AND e.name = $name
                RETURN e.name AS name,
                       e.path AS path,
                       e.start_line AS start_line,
                       e.end_line AS end_line,
                       labels(e) AS labels
                LIMIT 1
                """,
                {"name": entity_name},
            )
        except Exception as exc:
            return {"error": f"Graph query failed: {exc}"}

        if not records:
            return {"error": f"Entity '{entity_name}' not found in graph."}

        record = records[0]
        file_path = record.get("path", "")

        # If repo_path is provided, make the path absolute
        if repo_path and file_path and not Path(file_path).is_absolute():
            file_path = str(Path(repo_path) / file_path)

        return {
            "name": record["name"],
            "path": file_path,
            "start_line": record.get("start_line", 1),
            "end_line": record.get("end_line", 1),
            "labels": record.get("labels", []),
        }

    async def get_entity_source_code(
        self, entity_name: str, repo_path: str = "", context_lines: int = 5
    ) -> dict[str, Any]:
        """Convenience: find entity in graph and extract its source code."""
        location = await self.find_entity_source(entity_name, repo_path)
        if "error" in location:
            return location

        file_path = location["path"]
        start_line = location["start_line"]
        end_line = location["end_line"]

        snippet_result = self.extract_snippet(file_path, start_line, end_line, context_lines)
        snippet_result["entity_name"] = entity_name
        return snippet_result
