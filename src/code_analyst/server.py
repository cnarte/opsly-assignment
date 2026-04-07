"""Code Analyst Agent MCP server -- LLM-powered code analysis."""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.code_analyst.analyzer import CodeAnalyzer
from src.code_analyst.patterns import PatternDetector
from src.code_analyst.snippets import SnippetExtractor
from src.shared.neo4j_client import Neo4jClient
from src.shared.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

mcp = FastMCP(
    "code-analyst-agent",
    host="0.0.0.0",
    port=settings.CODE_ANALYST_PORT,
)


# -- helpers -----------------------------------------------------------------

def _get_neo4j_client() -> Neo4jClient:
    return Neo4jClient(settings)


# -- tools -------------------------------------------------------------------


@mcp.tool()
async def analyze_function(function_name: str, repo_path: str = "") -> dict:
    """Deep analysis of a function's logic using LLM."""
    client = _get_neo4j_client()
    try:
        await client.connect()
        analyzer = CodeAnalyzer(settings=settings, neo4j=client)
        extractor = SnippetExtractor(neo4j=client)

        # Get source code
        source_info = await extractor.get_entity_source_code(function_name, repo_path)
        source = source_info.get("snippet", "")
        if not source and "error" in source_info:
            return {"error": source_info["error"], "function_name": function_name}

        result = await analyzer.analyze_function(source, function_name)
        result["source_location"] = {
            k: source_info.get(k)
            for k in ("file_path", "entity_start_line", "entity_end_line")
        }
        return result
    finally:
        await client.close()


@mcp.tool()
async def analyze_class(class_name: str, repo_path: str = "") -> dict:
    """Comprehensive class analysis using LLM."""
    client = _get_neo4j_client()
    try:
        await client.connect()
        analyzer = CodeAnalyzer(settings=settings, neo4j=client)
        extractor = SnippetExtractor(neo4j=client)

        source_info = await extractor.get_entity_source_code(class_name, repo_path)
        source = source_info.get("snippet", "")
        if not source and "error" in source_info:
            return {"error": source_info["error"], "class_name": class_name}

        result = await analyzer.analyze_class(source, class_name)
        result["source_location"] = {
            k: source_info.get(k)
            for k in ("file_path", "entity_start_line", "entity_end_line")
        }
        return result
    finally:
        await client.close()


@mcp.tool()
async def find_patterns(code_path: str = "", pattern_type: str = "") -> dict:
    """Detect design patterns in the indexed codebase by querying the Neo4j graph.

    code_path is a name or keyword to search for (e.g. 'FastAPI', 'routing').
    All pattern detection is done via graph structure queries.
    """
    detector = PatternDetector()

    if not code_path:
        return {"error": "code_path is required", "patterns": []}

    all_patterns = await _find_patterns_from_graph(code_path, detector)

    # Filter by pattern_type if specified
    if pattern_type:
        all_patterns = [p for p in all_patterns if p["pattern"].lower() == pattern_type.lower()]

    return {
        "code_path": code_path,
        "pattern_type": pattern_type or "all",
        "count": len(all_patterns),
        "patterns": all_patterns,
    }


async def _find_patterns_from_graph(
    search_term: str, detector: PatternDetector,
) -> list[dict[str, Any]]:
    """Detect design patterns by querying the Neo4j knowledge graph structure."""
    import json as _json

    client = _get_neo4j_client()
    all_patterns: list[dict[str, Any]] = []
    try:
        await client.connect()

        # 1. Decorator pattern — classes/functions with decorators
        dec_records = await client.execute_query(
            """
            MATCH (e)-[:DECORATED_BY]->(d:Decorator)
            WHERE (e:Class OR e:Function OR e:Method)
            RETURN e.name AS name, e.file_path AS file, e.start_line AS line,
                   labels(e)[0] AS kind, collect(d.name) AS decorators
            LIMIT 100
            """,
            {},
        )
        for rec in dec_records:
            decs = rec.get("decorators", [])
            name = rec.get("name", "")
            # Python decorator pattern (route decorators, abstractmethod, etc.)
            for d in decs:
                if d and d in ("abstractmethod", "property", "staticmethod", "classmethod"):
                    continue
                all_patterns.append({
                    "pattern": "Decorator",
                    "confidence": 0.7,
                    "location": f"{rec.get('kind', '')} {name} (line {rec.get('line', '?')})",
                    "file": rec.get("file", ""),
                    "description": f"{name} is decorated by @{d}",
                })

        # 2. Dependency Injection — classes whose __init__ accepts 2+ typed params
        di_records = await client.execute_query(
            """
            MATCH (c:Class)-[:CONTAINS]->(m:Method {name: '__init__'})-[:HAS_PARAMETER]->(p:Parameter)
            WHERE p.name <> 'self'
              AND p.type_annotation IS NOT NULL AND p.type_annotation <> ''
            WITH c, collect(p.name) AS params, collect(p.type_annotation) AS types
            WHERE size(params) >= 2
            RETURN c.name AS name, c.file_path AS file, c.start_line AS line,
                   params, types
            LIMIT 50
            """,
            {},
        )
        for rec in di_records:
            all_patterns.append({
                "pattern": "Dependency Injection",
                "confidence": 0.7,
                "location": f"class {rec['name']} (line {rec.get('line', '?')})",
                "file": rec.get("file", ""),
                "description": (
                    f"Class '{rec['name']}' accepts {len(rec['params'])} typed dependencies "
                    f"in __init__: {rec['params']}"
                ),
            })

        # 3. Factory pattern — functions/methods named create_*/build_* or classes ending in Factory
        factory_records = await client.execute_query(
            """
            MATCH (e)
            WHERE (e:Function OR e:Method) AND (e.name STARTS WITH 'create_' OR e.name STARTS WITH 'build_')
            RETURN e.name AS name, e.file_path AS file, e.start_line AS line, 'function' AS kind
            LIMIT 50
            UNION
            MATCH (c:Class)
            WHERE c.name ENDS WITH 'Factory'
            RETURN c.name AS name, c.file_path AS file, c.start_line AS line, 'class' AS kind
            LIMIT 20
            """,
            {},
        )
        for rec in factory_records:
            all_patterns.append({
                "pattern": "Factory",
                "confidence": 0.7,
                "location": f"{rec['kind']} {rec['name']} (line {rec.get('line', '?')})",
                "file": rec.get("file", ""),
                "description": f"'{rec['name']}' follows the Factory naming pattern.",
            })

        # 4. Inheritance / Strategy — classes with base classes
        inh_records = await client.execute_query(
            """
            MATCH (c:Class)
            WHERE c.bases IS NOT NULL AND c.bases <> '[]' AND c.bases <> ''
            RETURN c.name AS name, c.file_path AS file, c.start_line AS line, c.bases AS bases
            LIMIT 80
            """,
            {},
        )
        for rec in inh_records:
            bases_raw = rec.get("bases", "[]")
            try:
                bases = _json.loads(bases_raw) if isinstance(bases_raw, str) else bases_raw
            except Exception:
                bases = [bases_raw]
            if bases:
                all_patterns.append({
                    "pattern": "Inheritance",
                    "confidence": 0.8,
                    "location": f"class {rec['name']} (line {rec.get('line', '?')})",
                    "file": rec.get("file", ""),
                    "description": f"Class '{rec['name']}' inherits from {bases}",
                })

        # 5. Observer — classes with on_event/add_listener/emit/notify methods
        obs_records = await client.execute_query(
            """
            MATCH (c:Class)-[:CONTAINS]->(m:Method)
            WHERE m.name IN ['on_event', 'emit', 'notify', 'subscribe', 'add_listener',
                             'remove_listener', 'add_observer', 'remove_observer']
            WITH c, collect(m.name) AS methods
            WHERE size(methods) >= 2
            RETURN c.name AS name, c.file_path AS file, c.start_line AS line, methods
            LIMIT 20
            """,
            {},
        )
        for rec in obs_records:
            all_patterns.append({
                "pattern": "Observer",
                "confidence": 0.75,
                "location": f"class {rec['name']} (line {rec.get('line', '?')})",
                "file": rec.get("file", ""),
                "description": f"Class '{rec['name']}' has observer methods: {rec['methods']}",
            })

        # 6. Middleware pattern — classes/functions with "middleware" in name
        mid_records = await client.execute_query(
            """
            MATCH (e)
            WHERE (e:Class OR e:Function OR e:Method)
              AND toLower(e.name) CONTAINS 'middleware'
            RETURN e.name AS name, e.file_path AS file, e.start_line AS line,
                   labels(e)[0] AS kind
            LIMIT 30
            """,
            {},
        )
        for rec in mid_records:
            all_patterns.append({
                "pattern": "Middleware",
                "confidence": 0.8,
                "location": f"{rec.get('kind', '')} {rec['name']} (line {rec.get('line', '?')})",
                "file": rec.get("file", ""),
                "description": f"'{rec['name']}' implements the Middleware pattern.",
            })

        # 7. Router/Dispatcher pattern — APIRouter usage
        router_records = await client.execute_query(
            """
            MATCH (c:Class)
            WHERE c.name CONTAINS 'Router' OR c.name CONTAINS 'Dispatcher'
            RETURN c.name AS name, c.file_path AS file, c.start_line AS line
            LIMIT 20
            """,
            {},
        )
        for rec in router_records:
            all_patterns.append({
                "pattern": "Router/Dispatcher",
                "confidence": 0.85,
                "location": f"class {rec['name']} (line {rec.get('line', '?')})",
                "file": rec.get("file", ""),
                "description": f"Class '{rec['name']}' implements the Router/Dispatcher pattern.",
            })

        return all_patterns
    except Exception as exc:
        logger.warning("Graph pattern search failed: %s", exc)
        return []
    finally:
        await client.close()


@mcp.tool()
async def get_code_snippet(entity_name: str, context_lines: int = 5) -> dict:
    """Extract code with surrounding context."""
    client = _get_neo4j_client()
    try:
        await client.connect()
        extractor = SnippetExtractor(neo4j=client)
        return await extractor.get_entity_source_code(entity_name, context_lines=context_lines)
    finally:
        await client.close()


@mcp.tool()
async def explain_implementation(entity_name: str) -> dict:
    """Generate explanation of how code works using LLM."""
    client = _get_neo4j_client()
    try:
        await client.connect()
        analyzer = CodeAnalyzer(settings=settings, neo4j=client)
        extractor = SnippetExtractor(neo4j=client)

        source_info = await extractor.get_entity_source_code(entity_name)
        source = source_info.get("snippet", "")
        if not source and "error" in source_info:
            return {"error": source_info["error"], "entity_name": entity_name}

        return await analyzer.explain_implementation(source, entity_name)
    finally:
        await client.close()


@mcp.tool()
async def compare_implementations(entity_a: str, entity_b: str) -> dict:
    """Compare two code entities using LLM."""
    client = _get_neo4j_client()
    try:
        await client.connect()
        analyzer = CodeAnalyzer(settings=settings, neo4j=client)
        extractor = SnippetExtractor(neo4j=client)

        source_a_info = await extractor.get_entity_source_code(entity_a)
        source_b_info = await extractor.get_entity_source_code(entity_b)

        source_a = source_a_info.get("snippet", "")
        source_b = source_b_info.get("snippet", "")

        if not source_a and "error" in source_a_info:
            return {"error": f"Could not find source for '{entity_a}': {source_a_info['error']}"}
        if not source_b and "error" in source_b_info:
            return {"error": f"Could not find source for '{entity_b}': {source_b_info['error']}"}

        return await analyzer.compare_implementations(source_a, source_b, entity_a, entity_b)
    finally:
        await client.close()


# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
