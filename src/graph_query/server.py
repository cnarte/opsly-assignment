"""MCP server for the Graph Query Agent."""

from __future__ import annotations

from typing import Any

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
async def list_entities(
    entity_type: str,
    limit: int = 50,
    repo_id: str = "",
    path_prefix: str = "",
    exclude_paths: list[str] = [],
) -> dict:
    """List all entities of a given type (Function, Class, Method, Module, File).

    Returns name, file_path, start_line, end_line for each entity, sorted by file.
    Use this whenever the user asks to 'get/list/show all X' or 'what functions/classes exist'.

    Args:
        entity_type: Type of entity to list
        limit: Max results to return (min 1, max 200)
        repo_id: Filter by repository ID
        path_prefix: Only include entities with file_path starting with this prefix
        exclude_paths: Exclude entities with file_path starting with any of these prefixes
    """
    _VALID = {"Function", "Class", "Method", "Module", "File", "Decorator", "Parameter"}
    label = entity_type.capitalize()
    if label not in _VALID:
        return {"error": f"Unknown entity_type '{entity_type}'. Choose from: {sorted(_VALID)}"}

    cap = min(max(1, limit), 200)
    rf = ""
    if repo_id:
        rf = "AND (n.repo_id = $repo_id OR n.qualified_name STARTS WITH $repo_id)"

    # Build path filters
    path_filters = ""
    params: dict[str, Any] = {"repo_id": repo_id}
    
    if path_prefix:
        path_filters += "AND n.file_path STARTS WITH $path_prefix "
        params["path_prefix"] = path_prefix
    
    if exclude_paths:
        exclude_conditions = " AND ".join([
            f"NOT n.file_path STARTS WITH $exclude_path_{i}"
            for i in range(len(exclude_paths))
        ])
        path_filters += f"AND ({exclude_conditions}) "
        for i, exclude_path in enumerate(exclude_paths):
            params[f"exclude_path_{i}"] = exclude_path

    cypher = (
        f"MATCH (n:{label}) WHERE n.name IS NOT NULL {rf} {path_filters} "
        "RETURN n.name AS name, n.file_path AS file_path, "
        "n.start_line AS start_line, n.end_line AS end_line "
        "ORDER BY n.file_path, n.start_line "
        f"LIMIT {cap}"
    )

    # Count separately so the LLM knows the real total
    count_cypher = f"MATCH (n:{label}) RETURN count(n) AS total"

    client = await _get_client()
    records = await client.execute_query(cypher, params)
    count_records = await client.execute_query(count_cypher)
    total = count_records[0]["total"] if count_records else len(records)

    return {
        "entity_type": label,
        "total_in_graph": total,
        "showing": len(records),
        "limit": cap,
        "path_prefix": path_prefix,
        "exclude_paths": exclude_paths,
        "entities": records,
    }


@mcp.tool()
async def semantic_search(
    query: str,
    entity_type: str = "Function",
    path_prefix: str = "",
    exclude_paths: list[str] = [],
    limit: int = 20,
    min_similarity: float = 0.7,
    repo_id: str = "",
) -> dict:
    """Search for code entities using semantic similarity on embeddings.

    Finds functions, classes, and methods based on semantic meaning of the query.
    Works by embedding the query and finding similar entity embeddings.

    Args:
        query: Natural language search query (e.g., "authentication", "error handling")
        entity_type: Type of entity to search (Function, Class, Method)
        path_prefix: Only include entities with file_path starting with this prefix
        exclude_paths: Exclude entities with file_path starting with any of these
        limit: Max results to return (max 50)
        min_similarity: Minimum cosine similarity score (0-1)
        repo_id: Filter by repository ID

    Returns:
        List of matching entities with similarity scores, sorted by relevance.
    """
    from src.shared.embeddings import EmbeddingService
    from src.shared.settings import Settings

    _VALID = {"Function", "Class", "Method"}
    label = entity_type.capitalize()
    if label not in _VALID:
        return {"error": f"semantic_search only supports: {sorted(_VALID)}. Got {entity_type}"}

    if not query or not query.strip():
        return {"error": "query cannot be empty"}

    cap = min(max(1, limit), 50)

    # Generate embedding for the query
    settings = Settings()
    embedding_service = EmbeddingService(settings)
    query_embedding = await embedding_service.embed_text(query.strip())
    await embedding_service.close()

    if not query_embedding or sum(query_embedding) == 0:
        return {"error": "Failed to generate embedding for query"}

    # Build path filters
    path_filters = ""
    params: dict[str, Any] = {
        "repo_id": repo_id,
        "query_embedding": query_embedding,
        "min_similarity": min_similarity,
    }

    rf = ""
    if repo_id:
        rf = "AND (n.repo_id = $repo_id OR n.qualified_name STARTS WITH $repo_id)"

    if path_prefix:
        path_filters += "AND n.file_path STARTS WITH $path_prefix "
        params["path_prefix"] = path_prefix

    if exclude_paths:
        exclude_conditions = " AND ".join([
            f"NOT n.file_path STARTS WITH $exclude_path_{i}"
            for i in range(len(exclude_paths))
        ])
        path_filters += f"AND ({exclude_conditions}) "
        for i, exclude_path in enumerate(exclude_paths):
            params[f"exclude_path_{i}"] = exclude_path

    # Vector similarity search (requires Neo4j 5.11+)
    cypher = (
        f"MATCH (n:{label}) "
        f"WHERE n.embedding IS NOT NULL {rf} {path_filters} "
        "WITH n, vector.similarity.cosine(n.embedding, $query_embedding) AS similarity "
        "WHERE similarity >= $min_similarity "
        "RETURN n.name AS name, n.file_path AS file_path, "
        "n.start_line AS start_line, n.end_line AS end_line, "
        "similarity AS score "
        "ORDER BY score DESC "
        f"LIMIT {cap}"
    )

    client = await _get_client()
    try:
        records = await client.execute_query(cypher, params)
        return {
            "query": query,
            "entity_type": label,
            "results_count": len(records),
            "limit": cap,
            "path_prefix": path_prefix,
            "exclude_paths": exclude_paths,
            "results": records,
        }
    except Exception as e:
        # Fallback if vector search fails (older Neo4j or no embeddings)
        return {
            "error": f"Semantic search not available: {str(e)}",
            "hint": "Ensure Neo4j 5.11+ and embeddings have been generated",
        }


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
