"""Memory Agent MCP server.

Provides tools for short-term (Redis) and long-term (Neo4j) memory
management exposed over Streamable HTTP transport.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

from src.shared.exceptions import AgentMemoryError
from src.shared.redis_client import RedisClient
from src.shared.neo4j_client import Neo4jClient
from src.shared.settings import Settings
from src.memory.session import SessionManager
from src.memory.cache import ResponseCache
from src.memory.graphiti_memory import GraphitiMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — initialise and tear down backing stores
# ---------------------------------------------------------------------------

settings = Settings()
redis_client = RedisClient(settings)
neo4j_client = Neo4jClient(settings)

session_mgr: SessionManager | None = None
response_cache: ResponseCache | None = None
graphiti: GraphitiMemory | None = None


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[None]:
    global session_mgr, response_cache, graphiti
    await redis_client.connect()
    await neo4j_client.connect()
    session_mgr = SessionManager(redis_client)
    response_cache = ResponseCache(redis_client)
    graphiti = GraphitiMemory(neo4j_client)
    logger.info("Memory agent backing stores connected")
    try:
        yield
    finally:
        await redis_client.close()
        await neo4j_client.close()
        logger.info("Memory agent backing stores closed")


mcp = FastMCP(
    "memory-agent",
    host="0.0.0.0",
    port=settings.MEMORY_PORT,
    lifespan=_lifespan,
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_conversation_context(session_id: str, limit: int = 10) -> dict:
    """Get recent conversation history from Redis."""
    try:
        assert session_mgr is not None
        messages = await session_mgr.get_history(session_id, limit)
        return {"status": "ok", "session_id": session_id, "messages": messages}
    except Exception as exc:
        logger.exception("get_conversation_context failed")
        raise AgentMemoryError(str(exc)) from exc


@mcp.tool()
async def store_interaction(
    session_id: str,
    query: str,
    response: str,
    agents_used: list[str] | None = None,
) -> dict:
    """Save Q&A pair to Redis (short-term) and Neo4j (long-term)."""
    try:
        assert session_mgr is not None and graphiti is not None
        agents_used = agents_used or []
        # Short-term: push both messages to session list
        await session_mgr.add_message(session_id, "user", query, {"agents_used": agents_used})
        await session_mgr.add_message(session_id, "assistant", response, {"agents_used": agents_used})
        # Long-term: store as an episode
        episode_id = await graphiti.add_episode(
            session_id, content=f"Q: {query}\nA: {response}", query=query, response=response
        )
        return {"status": "ok", "session_id": session_id, "episode_id": episode_id}
    except Exception as exc:
        logger.exception("store_interaction failed")
        raise AgentMemoryError(str(exc)) from exc


@mcp.tool()
async def cache_response(cache_key: str, response: str, ttl: int = 3600) -> dict:
    """Cache an agent response in Redis with TTL."""
    try:
        assert response_cache is not None
        await response_cache.set(cache_key, response, ttl)
        return {"status": "ok", "cache_key": cache_key, "ttl": ttl}
    except Exception as exc:
        logger.exception("cache_response failed")
        raise AgentMemoryError(str(exc)) from exc


@mcp.tool()
async def get_cached_response(cache_key: str) -> dict:
    """Retrieve cached response from Redis."""
    try:
        assert response_cache is not None
        value = await response_cache.get(cache_key)
        if value is None:
            return {"status": "miss", "cache_key": cache_key, "response": None}
        return {"status": "hit", "cache_key": cache_key, "response": value}
    except Exception as exc:
        logger.exception("get_cached_response failed")
        raise AgentMemoryError(str(exc)) from exc


@mcp.tool()
async def search_memory(query: str, session_id: str = "", limit: int = 5) -> dict:
    """Semantic search across long-term memory via Neo4j."""
    try:
        assert graphiti is not None
        results = await graphiti.search(query, limit)
        # Optionally filter by session
        if session_id:
            results = [r for r in results if r.get("session_id") == session_id]
        return {"status": "ok", "query": query, "results": results}
    except Exception as exc:
        logger.exception("search_memory failed")
        raise AgentMemoryError(str(exc)) from exc


@mcp.tool()
async def remember_fact(session_id: str, fact: str, category: str = "general") -> dict:
    """Store a durable fact/preference via Neo4j."""
    try:
        assert graphiti is not None
        fact_id = await graphiti.add_fact(session_id, fact, category)
        return {"status": "ok", "fact_id": fact_id, "category": category}
    except Exception as exc:
        logger.exception("remember_fact failed")
        raise AgentMemoryError(str(exc)) from exc


@mcp.tool()
async def get_user_preferences(session_id: str) -> dict:
    """Retrieve user preferences from Neo4j."""
    try:
        assert graphiti is not None
        prefs = await graphiti.get_preferences(session_id)
        return {"status": "ok", "session_id": session_id, "preferences": prefs}
    except Exception as exc:
        logger.exception("get_user_preferences failed")
        raise AgentMemoryError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
