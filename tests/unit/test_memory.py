"""Unit tests for Memory Agent components (session, cache, graceful degradation)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.memory.session import SessionManager
from src.memory.cache import ResponseCache


# ---------------------------------------------------------------------------
# Helpers — build a mock RedisClient whose .client returns an async-capable
# Redis stub.
# ---------------------------------------------------------------------------


def _make_mock_redis() -> MagicMock:
    """Return a mock RedisClient with an async .client stub."""
    redis_mock = MagicMock()
    inner = AsyncMock()
    # Make .client a property that returns the inner mock
    type(redis_mock).client = PropertyMock(return_value=inner)

    # Provide async versions of the thin-wrapper helpers too
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock()
    redis_mock.delete = AsyncMock()

    return redis_mock


# ===================================================================
# SessionManager tests
# ===================================================================


class TestSessionManager:
    """Tests for SessionManager add/get/clear."""

    @pytest.fixture()
    def redis(self):
        return _make_mock_redis()

    @pytest.fixture()
    def mgr(self, redis):
        return SessionManager(redis, ttl=3600)

    @pytest.mark.asyncio
    async def test_add_message(self, mgr, redis):
        """add_message should rpush a JSON blob and set expire."""
        await mgr.add_message("s1", "user", "hello", {"extra": 1})

        inner = redis.client
        inner.rpush.assert_awaited_once()
        args = inner.rpush.call_args
        assert args[0][0] == "session:s1:messages"
        payload = json.loads(args[0][1])
        assert payload["role"] == "user"
        assert payload["content"] == "hello"
        assert payload["metadata"] == {"extra": 1}
        assert "timestamp" in payload

        inner.expire.assert_awaited_once_with("session:s1:messages", 3600)

    @pytest.mark.asyncio
    async def test_get_history(self, mgr, redis):
        """get_history should return parsed JSON from the Redis list."""
        msg = json.dumps({"role": "user", "content": "hi", "metadata": {}, "timestamp": "t"})
        redis.client.lrange = AsyncMock(return_value=[msg, msg])

        history = await mgr.get_history("s1", limit=5)

        redis.client.lrange.assert_awaited_once_with("session:s1:messages", -5, -1)
        assert len(history) == 2
        assert history[0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_get_history_empty(self, mgr, redis):
        """get_history returns [] when no messages exist."""
        redis.client.lrange = AsyncMock(return_value=[])
        assert await mgr.get_history("s1") == []

    @pytest.mark.asyncio
    async def test_clear_session(self, mgr, redis):
        """clear_session delegates to redis.delete."""
        await mgr.clear_session("s1")
        redis.delete.assert_awaited_once_with("session:s1:messages")


# ===================================================================
# ResponseCache tests
# ===================================================================


class TestResponseCache:
    """Tests for ResponseCache get/set/invalidate."""

    @pytest.fixture()
    def redis(self):
        return _make_mock_redis()

    @pytest.fixture()
    def cache(self, redis):
        return ResponseCache(redis)

    @pytest.mark.asyncio
    async def test_set(self, cache, redis):
        await cache.set("k1", "data", ttl=120)
        redis.set.assert_awaited_once_with("cache:k1", "data", ex=120)

    @pytest.mark.asyncio
    async def test_get_hit(self, cache, redis):
        redis.get = AsyncMock(return_value="data")
        result = await cache.get("k1")
        redis.get.assert_awaited_once_with("cache:k1")
        assert result == "data"

    @pytest.mark.asyncio
    async def test_get_miss(self, cache, redis):
        redis.get = AsyncMock(return_value=None)
        assert await cache.get("k1") is None

    @pytest.mark.asyncio
    async def test_invalidate(self, cache, redis):
        """invalidate should SCAN and delete matching keys."""
        inner = redis.client
        # Simulate one scan iteration returning keys, then cursor 0
        inner.scan = AsyncMock(return_value=(0, ["cache:repo:a", "cache:repo:b"]))
        inner.delete = AsyncMock()

        await cache.invalidate("repo:*")

        inner.scan.assert_awaited()
        inner.delete.assert_awaited_once_with("cache:repo:a", "cache:repo:b")


# ===================================================================
# Graceful degradation tests
# ===================================================================


class TestGracefulDegradation:
    """Verify that components don't crash when Redis is unavailable."""

    @pytest.mark.asyncio
    async def test_session_get_history_redis_down(self):
        """get_history should propagate errors (caller handles)."""
        redis = _make_mock_redis()
        redis.client.lrange = AsyncMock(side_effect=ConnectionError("Redis down"))
        mgr = SessionManager(redis)

        with pytest.raises(ConnectionError):
            await mgr.get_history("s1")

    @pytest.mark.asyncio
    async def test_cache_get_redis_down(self):
        """cache.get should propagate errors."""
        redis = _make_mock_redis()
        redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        cache = ResponseCache(redis)

        with pytest.raises(ConnectionError):
            await cache.get("k")

    @pytest.mark.asyncio
    async def test_session_add_message_redis_down(self):
        """add_message should propagate errors."""
        redis = _make_mock_redis()
        redis.client.rpush = AsyncMock(side_effect=ConnectionError("Redis down"))
        mgr = SessionManager(redis)

        with pytest.raises(ConnectionError):
            await mgr.add_message("s1", "user", "hello")
