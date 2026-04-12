"""Code Analyst Agent MCP server -- LLM-powered code analysis."""

from __future__ import annotations

import logging
import os as _os
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.code_analyst.analyzer import CodeAnalyzer, _locate_symbol
from src.code_analyst.patterns import PatternDetector
from src.shared.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

mcp = FastMCP(
    "code-analyst-agent",
    host="0.0.0.0",
    port=settings.CODE_ANALYST_PORT,
)


# -- tools -------------------------------------------------------------------


async def _read_source_from_workspace(file_path: str, start_line: int, num_lines: int = 100) -> str:
    """Read source lines from workspace repos given a relative file_path."""
    workspace = _os.getenv("WORKSPACE_PATH", "/workspace/repos")
    if not _os.path.exists(workspace):
        return ""
    for repo_dir in _os.listdir(workspace):
        candidate = _os.path.join(workspace, repo_dir, file_path)
        if _os.path.exists(candidate):
            lines = open(candidate).readlines()
            start = max(0, start_line - 1)
            end = min(len(lines), start + num_lines)
            return "".join(lines[start:end])
    return ""


@mcp.tool()
async def analyze_function(function_name: str, repo_path: str = "") -> dict:
    """Deep analysis of a function's logic using LLM."""
    location = await _locate_symbol(function_name, repo_id=repo_path)
    source = ""
    if location["file_path"]:
        source = await _read_source_from_workspace(location["file_path"], location["start_line"])
    if not source:
        return {"error": f"Source not found for '{function_name}'", "function_name": function_name}

    analyzer = CodeAnalyzer(settings=settings)
    result = await analyzer.analyze_function(source, function_name)
    result["source_location"] = location
    return result


@mcp.tool()
async def analyze_class(class_name: str, repo_path: str = "") -> dict:
    """Comprehensive class analysis using LLM."""
    location = await _locate_symbol(class_name, repo_id=repo_path)
    source = ""
    if location["file_path"]:
        source = await _read_source_from_workspace(location["file_path"], location["start_line"])
    if not source:
        return {"error": f"Source not found for '{class_name}'", "class_name": class_name}

    analyzer = CodeAnalyzer(settings=settings)
    result = await analyzer.analyze_class(source, class_name)
    result["source_location"] = location
    return result


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
    """Pattern detection is no longer graph-backed; returns empty list."""
    logger.info("Pattern graph detection disabled (Neo4j removed); search_term=%s", search_term)
    return []


@mcp.tool()
async def get_code_snippet(entity_name: str, context_lines: int = 5) -> dict:
    """Extract code with surrounding context via gitnexus-agent."""
    location = await _locate_symbol(entity_name)
    if not location["file_path"]:
        return {"error": f"Entity '{entity_name}' not found via gitnexus-agent."}
    source = await _read_source_from_workspace(
        location["file_path"], location["start_line"], num_lines=context_lines * 2 + 20
    )
    return {
        "entity_name": entity_name,
        "file_path": location["file_path"],
        "start_line": location["start_line"],
        "snippet": source,
    }


@mcp.tool()
async def explain_implementation(entity_name: str, model: str = "") -> str:
    """LLM-generated plain-English explanation of how an entity works."""
    from src.code_analyst.analyzer import explain_entity
    return await explain_entity(entity_name, model=model)


@mcp.tool()
async def compare_implementations(entity_a: str, entity_b: str) -> dict:
    """Compare two code entities using LLM."""
    loc_a = await _locate_symbol(entity_a)
    loc_b = await _locate_symbol(entity_b)

    source_a = ""
    source_b = ""
    if loc_a["file_path"]:
        source_a = await _read_source_from_workspace(loc_a["file_path"], loc_a["start_line"])
    if loc_b["file_path"]:
        source_b = await _read_source_from_workspace(loc_b["file_path"], loc_b["start_line"])

    if not source_a:
        return {"error": f"Could not find source for '{entity_a}'"}
    if not source_b:
        return {"error": f"Could not find source for '{entity_b}'"}

    analyzer = CodeAnalyzer(settings=settings)
    return await analyzer.compare_implementations(source_a, source_b, entity_a, entity_b)


# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
