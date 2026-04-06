"""Code Analyst Agent MCP server -- LLM-powered code analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.code_analyst.analyzer import CodeAnalyzer
from src.code_analyst.patterns import PatternDetector
from src.code_analyst.snippets import SnippetExtractor
from src.shared.neo4j_client import Neo4jClient
from src.shared.settings import Settings

logger = logging.getLogger(__name__)

mcp = FastMCP("code-analyst-agent")

settings = Settings()


# -- helpers -----------------------------------------------------------------

def _get_neo4j_client() -> Neo4jClient:
    return Neo4jClient(settings)


async def _read_source_file(file_path: str) -> str:
    """Read a source file and return its content."""
    p = Path(file_path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


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
    """Detect design patterns in code."""
    detector = PatternDetector()

    if not code_path:
        return {"error": "code_path is required", "patterns": []}

    path = Path(code_path)
    if not path.exists():
        return {"error": f"Path not found: {code_path}", "patterns": []}

    # Collect Python files
    files: list[Path] = []
    if path.is_file():
        files = [path]
    else:
        files = list(path.rglob("*.py"))

    all_patterns: list[dict[str, Any]] = []
    for f in files:
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        detected = detector.detect_patterns(source)
        for pat in detected:
            pat["file"] = str(f)
        all_patterns.extend(detected)

    # Filter by pattern_type if specified
    if pattern_type:
        all_patterns = [p for p in all_patterns if p["pattern"].lower() == pattern_type.lower()]

    return {
        "code_path": code_path,
        "pattern_type": pattern_type or "all",
        "count": len(all_patterns),
        "patterns": all_patterns,
    }


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
