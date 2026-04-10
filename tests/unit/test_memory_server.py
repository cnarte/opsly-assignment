"""Unit tests for the Memory Agent MCP server tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to inject mocked global state into memory server
# ---------------------------------------------------------------------------


def _make_session_mgr(messages: list | None = None) -> MagicMock:
    mgr = MagicMock()
    mgr.get_history = AsyncMock(return_value=messages or [])
    mgr.add_message = AsyncMock()
    mgr.clear_session = AsyncMock()
    return mgr


def _make_response_cache(value: str | None = None) -> MagicMock:
    cache = MagicMock()
    cache.get = AsyncMock(return_value=value)
    cache.set = AsyncMock()
    return cache


def _make_graphiti(episode_id: str = "ep-1") -> MagicMock:
    g = MagicMock()
    g.add_episode = AsyncMock(return_value=episode_id)
    g.add_fact = AsyncMock(return_value="fact-1")
    g.search = AsyncMock(return_value=[])
    g.get_preferences = AsyncMock(return_value=[])
    return g


# ---------------------------------------------------------------------------
# get_conversation_context
# ---------------------------------------------------------------------------


class TestGetConversationContext:
    @pytest.mark.asyncio
    async def test_returns_messages(self):
        import src.memory.server as ms
        messages = [{"role": "user", "content": "hi"}]
        ms.session_mgr = _make_session_mgr(messages)

        from src.memory.server import get_conversation_context
        result = await get_conversation_context("sess-1")

        assert result["status"] == "ok"
        assert result["messages"] == messages
        assert result["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_raises_on_session_mgr_error(self):
        import src.memory.server as ms
        mgr = MagicMock()
        mgr.get_history = AsyncMock(side_effect=RuntimeError("Redis down"))
        ms.session_mgr = mgr

        from src.memory.server import get_conversation_context
        from src.shared.exceptions import AgentMemoryError
        with pytest.raises(AgentMemoryError):
            await get_conversation_context("sess-1")


# ---------------------------------------------------------------------------
# store_interaction
# ---------------------------------------------------------------------------


class TestStoreInteraction:
    @pytest.mark.asyncio
    async def test_stores_in_redis_and_neo4j(self):
        import src.memory.server as ms
        ms.session_mgr = _make_session_mgr()
        ms.graphiti = _make_graphiti("ep-42")

        from src.memory.server import store_interaction
        result = await store_interaction("sess-1", "What is FastAPI?", "It's a framework.")

        assert result["status"] == "ok"
        assert result["episode_id"] == "ep-42"
        ms.session_mgr.add_message.assert_awaited()

    @pytest.mark.asyncio
    async def test_accepts_agents_used(self):
        import src.memory.server as ms
        ms.session_mgr = _make_session_mgr()
        ms.graphiti = _make_graphiti()

        from src.memory.server import store_interaction
        result = await store_interaction(
            "sess-1", "Q", "A", agents_used=["graph_query", "code_analyst"]
        )

        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# cache_response
# ---------------------------------------------------------------------------


class TestCacheResponse:
    @pytest.mark.asyncio
    async def test_caches_with_ttl(self):
        import src.memory.server as ms
        ms.response_cache = _make_response_cache()

        from src.memory.server import cache_response
        result = await cache_response("key-1", "response text", ttl=600)

        assert result["status"] == "ok"
        assert result["ttl"] == 600
        ms.response_cache.set.assert_awaited_once_with("key-1", "response text", 600)


# ---------------------------------------------------------------------------
# get_cached_response
# ---------------------------------------------------------------------------


class TestGetCachedResponse:
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        import src.memory.server as ms
        ms.response_cache = _make_response_cache("cached answer")

        from src.memory.server import get_cached_response
        result = await get_cached_response("key-1")

        assert result["status"] == "hit"
        assert result["response"] == "cached answer"

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        import src.memory.server as ms
        ms.response_cache = _make_response_cache(None)

        from src.memory.server import get_cached_response
        result = await get_cached_response("key-1")

        assert result["status"] == "miss"
        assert result["response"] is None


# ---------------------------------------------------------------------------
# search_memory
# ---------------------------------------------------------------------------


class TestSearchMemory:
    @pytest.mark.asyncio
    async def test_returns_results(self):
        import src.memory.server as ms
        g = _make_graphiti()
        g.search = AsyncMock(return_value=[{"content": "match"}])
        ms.graphiti = g

        from src.memory.server import search_memory
        result = await search_memory("FastAPI")

        assert result["status"] == "ok"
        assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_filters_by_session_id(self):
        import src.memory.server as ms
        g = _make_graphiti()
        g.search = AsyncMock(return_value=[
            {"content": "match", "session_id": "sess-1"},
            {"content": "other", "session_id": "sess-2"},
        ])
        ms.graphiti = g

        from src.memory.server import search_memory
        result = await search_memory("FastAPI", session_id="sess-1")

        assert result["status"] == "ok"
        assert all(r["session_id"] == "sess-1" for r in result["results"])


# ---------------------------------------------------------------------------
# remember_fact
# ---------------------------------------------------------------------------


class TestRememberFact:
    @pytest.mark.asyncio
    async def test_stores_fact(self):
        import src.memory.server as ms
        ms.graphiti = _make_graphiti()

        from src.memory.server import remember_fact
        result = await remember_fact("sess-1", "FastAPI uses Starlette", "framework")

        assert result["status"] == "ok"
        assert result["category"] == "framework"


# ---------------------------------------------------------------------------
# get_user_preferences
# ---------------------------------------------------------------------------


class TestGetUserPreferences:
    @pytest.mark.asyncio
    async def test_returns_preferences(self):
        import src.memory.server as ms
        g = _make_graphiti()
        g.get_preferences = AsyncMock(return_value=[{"pref": "dark_mode"}])
        ms.graphiti = g

        from src.memory.server import get_user_preferences
        result = await get_user_preferences("sess-1")

        assert result["status"] == "ok"
        assert len(result["preferences"]) == 1
