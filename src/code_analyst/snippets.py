"""Source code extraction and entity location utilities — graph-only, no filesystem access."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.shared.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class SnippetExtractor:
    """Extract source code context from the Neo4j knowledge graph."""

    def __init__(self, neo4j: Neo4jClient | None = None) -> None:
        self._neo4j = neo4j

    async def find_entity_source(
        self, entity_name: str, repo_path: str = ""
    ) -> dict[str, Any]:
        """Look up an entity's source location and context from the Neo4j graph."""
        if self._neo4j is None:
            return {"error": "No Neo4j client available for entity lookup."}

        try:
            # Search for the entity across code labels
            records = await self._neo4j.execute_query(
                """
                MATCH (e)
                WHERE (e:Function OR e:Class OR e:Method) AND e.name = $name
                OPTIONAL MATCH (e)-[:DOCUMENTED_BY]->(doc:Docstring)
                OPTIONAL MATCH (e)-[:HAS_PARAMETER]->(p:Parameter)
                OPTIONAL MATCH (e)-[:DECORATED_BY]->(d:Decorator)
                RETURN e.name AS name,
                       e.file_path AS path,
                       e.start_line AS start_line,
                       e.end_line AS end_line,
                       e.qualified_name AS qualified_name,
                       e.bases AS bases,
                       labels(e) AS labels,
                       doc.text AS docstring,
                       collect(DISTINCT {name: p.name, type: p.type_annotation, position: p.position}) AS params,
                       collect(DISTINCT d.name) AS decorators
                LIMIT 5
                """,
                {"name": entity_name},
            )
        except Exception as exc:
            return {"error": f"Graph query failed: {exc}"}

        if not records:
            # Try fuzzy/partial match
            try:
                records = await self._neo4j.execute_query(
                    """
                    MATCH (e)
                    WHERE (e:Function OR e:Class OR e:Method)
                      AND toLower(e.name) CONTAINS toLower($name)
                    RETURN e.name AS name,
                           e.file_path AS path,
                           e.start_line AS start_line,
                           e.end_line AS end_line,
                           e.qualified_name AS qualified_name,
                           e.bases AS bases,
                           labels(e) AS labels
                    LIMIT 5
                    """,
                    {"name": entity_name},
                )
            except Exception:
                pass

        if not records:
            return {"error": f"Entity '{entity_name}' not found in graph."}

        record = records[0]
        return {
            "name": record.get("name", ""),
            "path": record.get("path", ""),
            "start_line": record.get("start_line", 1),
            "end_line": record.get("end_line", 1),
            "qualified_name": record.get("qualified_name", ""),
            "bases": record.get("bases", ""),
            "labels": record.get("labels", []),
            "docstring": record.get("docstring", ""),
            "params": record.get("params", []),
            "decorators": record.get("decorators", []),
        }

    async def get_entity_source_code(
        self, entity_name: str, repo_path: str = "", context_lines: int = 5
    ) -> dict[str, Any]:
        """Find entity in graph and build a rich context snippet from graph data.

        Since source files are not available at runtime, we reconstruct
        a representation from graph metadata (signature, docstring, params,
        decorators, relationships).
        """
        location = await self.find_entity_source(entity_name, repo_path)
        if "error" in location:
            return location

        # Build a synthetic snippet from graph metadata
        snippet_parts: list[str] = []

        # Decorators
        decorators = location.get("decorators", [])
        if decorators:
            for dec in decorators:
                if dec:
                    snippet_parts.append(f"@{dec}")

        # Determine kind
        labels = location.get("labels", [])
        kind = "class" if "Class" in labels else "def"

        # Signature
        name = location.get("name", entity_name)
        params = location.get("params", [])
        # Filter out None entries and sort by position
        params = [p for p in params if p and p.get("name")]
        params.sort(key=lambda p: p.get("position", 0) or 0)

        if kind == "class":
            bases_raw = location.get("bases", "")
            try:
                bases = json.loads(bases_raw) if isinstance(bases_raw, str) and bases_raw else []
            except Exception:
                bases = [bases_raw] if bases_raw else []
            base_str = f"({', '.join(str(b) for b in bases)})" if bases else ""
            snippet_parts.append(f"class {name}{base_str}:")
        else:
            param_strs = []
            for p in params:
                pname = p.get("name", "")
                ptype = p.get("type", "")
                if ptype:
                    param_strs.append(f"{pname}: {ptype}")
                else:
                    param_strs.append(pname)
            snippet_parts.append(f"def {name}({', '.join(param_strs)}):")

        # Docstring
        docstring = location.get("docstring", "")
        if docstring:
            snippet_parts.append(f'    """{docstring}"""')

        snippet = "\n".join(snippet_parts)

        # Also get relationships for richer context
        related = await self._get_related_entities(name)

        return {
            "entity_name": entity_name,
            "file_path": location.get("path", ""),
            "entity_start_line": location.get("start_line"),
            "entity_end_line": location.get("end_line"),
            "qualified_name": location.get("qualified_name", ""),
            "snippet": snippet,
            "related_entities": related,
        }

    async def _get_related_entities(self, entity_name: str) -> list[dict[str, Any]]:
        """Get entities related to the given entity from the graph."""
        if self._neo4j is None:
            return []
        try:
            records = await self._neo4j.execute_query(
                """
                MATCH (e)-[r]-(other)
                WHERE (e:Function OR e:Class OR e:Method) AND e.name = $name
                  AND (other:Function OR other:Class OR other:Method OR other:Module)
                RETURN type(r) AS rel_type,
                       other.name AS name,
                       labels(other)[0] AS kind,
                       other.file_path AS file
                LIMIT 20
                """,
                {"name": entity_name},
            )
            return [
                {
                    "relationship": r.get("rel_type", ""),
                    "name": r.get("name", ""),
                    "kind": r.get("kind", ""),
                    "file": r.get("file", ""),
                }
                for r in records
            ]
        except Exception:
            return []
