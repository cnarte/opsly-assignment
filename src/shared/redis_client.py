"""Async Redis client wrapper."""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis

from src.shared.settings import Settings


class RedisClient:
    """Thin async wrapper around redis.asyncio."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: aioredis.Redis | None = None

    # -- lifecycle ------------------------------------------------------------

    async def connect(self) -> None:
        """Create the Redis connection pool."""
        self._client = aioredis.from_url(
            self._settings.REDIS_URL,
            decode_responses=True,
        )
        # Quick ping to verify connectivity.
        await self._client.ping()

    async def close(self) -> None:
        """Close the underlying connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- context manager ------------------------------------------------------

    async def __aenter__(self) -> "RedisClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- helpers --------------------------------------------------------------

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisClient is not connected. Call connect() first.")
        return self._client

    async def get(self, key: str) -> str | None:
        """Get a value by key."""
        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
    ) -> None:
        """Set a value with optional expiry in seconds."""
        await self.client.set(key, value, ex=ex)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        await self.client.delete(key)
