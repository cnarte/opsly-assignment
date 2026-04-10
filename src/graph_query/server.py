"""MCP server for the Graph Query Agent."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.shared.neo4j_client import Neo4jClient
from src.shared.settings import Settings
from src.graph_query import queries
from src.graph_query.safety import CypherSafetyChecker
from src.graph_query.traversal import build_context_view, build_impact_graph

settings = Settings()
mcp = FastMCP(
    "graph-query-agent",
    host="0.0.0.0",
    port=settings.GRAPH_QUERY_PORT,
)

_client: Neo4jClient | None = None
_safety = CypherSafetyChecker()


async def _get_client() -> Neo4jClient:
    """Lazy-initialise and return the Neo4j client singleton."""
    global _client
    if _client is None:
        _client = Neo4jClient(settings)
        await _client.connect()
    return _client


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def find_entity(name: str, entity_type: str = "", repo_id: str = "") -> dict:
    """Locate a class, function, or module by name. Code-graph labels only."""
    client = await _get_client()
    query = queries.find_entity(name, entity_type, repo_id)
    records = await client.execute_query(query.cypher, query.parameters)
    return {"results": records, "count": len(records)}


@mcp.tool()
async def get_dependencies(entity_name: str, repo_id: str = "") -> dict:
    """Find what an entity depends on. Code-graph labels only."""
    client = await _get_client()
    query = queries.get_dependencies(entity_name, repo_id)
    records = await client.execute_query(query.cypher, query.parameters)
    return {"entity": entity_name, "dependencies": records, "count": len(records)}


@mcp.tool()
async def get_dependents(entity_name: str, repo_id: str = "") -> dict:
    """Find what depends on an entity. Code-graph labels only."""
    client = await _get_client()
    query = queries.get_dependents(entity_name, repo_id)
    records = await client.execute_query(query.cypher, query.parameters)
    return {"entity": entity_name, "dependents": records, "count": len(records)}


@mcp.tool()
async def trace_imports(module_name: str, repo_id: str = "") -> dict:
    """Follow import chain for a module. Code-graph labels only."""
    client = await _get_client()
    query = queries.trace_imports(module_name, repo_id)
    records = await client.execute_query(query.cypher, query.parameters)
    return {"module": module_name, "import_chains": records, "count": len(records)}


@mcp.tool()
async def find_related(entity_name: str, relationship_type: str, repo_id: str = "") -> dict:
    """Get entities related by specified relationship type. Code-graph labels only."""
    client = await _get_client()
    query = queries.find_related(entity_name, relationship_type, repo_id)
    records = await client.execute_query(query.cypher, query.parameters)
    return {"entity": entity_name, "relationship": relationship_type, "results": records, "count": len(records)}


@mcp.tool()
async def execute_query(cypher: str) -> dict:
    """Run custom Cypher query with safety constraints. Read-only, label allowlist enforced."""
    _safety.validate_or_raise(cypher)
    client = await _get_client()
    records = await client.execute_query(cypher)
    return {"results": records, "count": len(records)}


@mcp.tool()
async def get_symbol_context(symbol_name: str, repo_id: str = "") -> dict:
    """360-degree view of a symbol: callers, callees, imports, parameters, decorators."""
    client = await _get_client()
    return await build_context_view(client, symbol_name, repo_id=repo_id)


@mcp.tool()
async def analyze_impact(symbol_name: str, depth: int = 2, repo_id: str = "") -> dict:
    """Blast radius analysis for a symbol change."""
    client = await _get_client()
    return await build_impact_graph(client, symbol_name, depth, repo_id=repo_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
