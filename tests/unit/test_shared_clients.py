"""Unit tests for shared Neo4j and Redis client wrappers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.shared.settings import Settings


# ---------------------------------------------------------------------------
# Neo4jClient
# ---------------------------------------------------------------------------


class TestNeo4jClient:
    def _make_client(self) -> "Neo4jClient":  # noqa: F821
        from src.shared.neo4j_client import Neo4jClient
        settings = Settings(
            NEO4J_URI="bolt://localhost:7687",
            NEO4J_USER="neo4j",
            NEO4J_PASSWORD="test",
            NEO4J_DATABASE="neo4j",
        )
        return Neo4jClient(settings)

    @pytest.mark.asyncio
    async def test_connect_creates_driver(self):
        """connect() creates an AsyncGraphDatabase.driver."""
        client = self._make_client()
        mock_driver = AsyncMock()
        mock_driver.verify_connectivity = AsyncMock()

        with patch("src.shared.neo4j_client.AsyncGraphDatabase") as mock_db:
            mock_db.driver.return_value = mock_driver
            await client.connect()

        mock_db.driver.assert_called_once()
        assert client._driver is mock_driver

    @pytest.mark.asyncio
    async def test_close_closes_driver(self):
        """close() closes the driver."""
        client = self._make_client()
        mock_driver = AsyncMock()
        client._driver = mock_driver

        await client.close()

        mock_driver.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_noop_when_not_connected(self):
        """close() does nothing if connect() was never called."""
        client = self._make_client()
        # Should not raise
        await client.close()

    @pytest.mark.asyncio
    async def test_execute_query_delegates_to_session(self):
        """execute_query opens a session and runs a read query."""
        client = self._make_client()

        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"key": "value"}])

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.run = AsyncMock(return_value=mock_result)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        client._driver = mock_driver

        result = await client.execute_query("RETURN 1", {})

        assert result == [{"key": "value"}]

    @pytest.mark.asyncio
    async def test_execute_write_delegates_to_session(self):
        """execute_write opens a session and calls execute_write."""
        client = self._make_client()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute_write = AsyncMock(return_value=[])

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        client._driver = mock_driver

        await client.execute_write("CREATE (n:Test)", {})

        mock_session.execute_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Neo4jClient works as an async context manager."""
        client = self._make_client()

        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect, \
             patch.object(client, "close", new_callable=AsyncMock) as mock_close:
            async with client:
                pass

        mock_connect.assert_awaited_once()
        mock_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# RedisClient
# ---------------------------------------------------------------------------


class TestRedisClient:
    def _make_client(self) -> "RedisClient":  # noqa: F821
        from src.shared.redis_client import RedisClient
        settings = Settings(REDIS_URL="redis://localhost:6379/0")
        return RedisClient(settings)

    @pytest.mark.asyncio
    async def test_connect_creates_redis_instance(self):
        """connect() creates a Redis instance."""
        client = self._make_client()
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        with patch("src.shared.redis_client.aioredis.from_url", return_value=mock_redis) as mock_fn:
            await client.connect()

        mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_closes_connection(self):
        """close() closes the Redis connection."""
        client = self._make_client()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        client._client = mock_redis

        await client.close()

        mock_redis.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_delegates_to_redis(self):
        """get() calls redis.get()."""
        client = self._make_client()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=b"value")
        client._client = mock_redis

        result = await client.get("mykey")

        mock_redis.get.assert_awaited_once_with("mykey")

    @pytest.mark.asyncio
    async def test_set_delegates_to_redis(self):
        """set() calls redis.set()."""
        client = self._make_client()
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()
        client._client = mock_redis

        await client.set("mykey", "myvalue", ex=60)

        mock_redis.set.assert_awaited_once_with("mykey", "myvalue", ex=60)

    @pytest.mark.asyncio
    async def test_delete_delegates_to_redis(self):
        """delete() calls redis.delete()."""
        client = self._make_client()
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock()
        client._client = mock_redis

        await client.delete("mykey")

        mock_redis.delete.assert_awaited_once_with("mykey")
