"""Response caching backed by Redis."""

from __future__ import annotations

import logging

from src.shared.redis_client import RedisClient

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 3600  # 1 hour


class ResponseCache:
    """Simple key-value cache with TTL, prefixed under ``cache:``."""

    def __init__(self, redis: RedisClient) -> None:
        self._redis = redis

    @staticmethod
    def _key(cache_key: str) -> str:
        return f"cache:{cache_key}"

    async def get(self, cache_key: str) -> str | None:
        """Fetch a cached response, or ``None`` if missing/expired."""
        return await self._redis.get(self._key(cache_key))

    async def set(self, cache_key: str, response: str, ttl: int = _DEFAULT_TTL) -> None:
        """Store *response* under *cache_key* with a TTL (seconds)."""
        await self._redis.set(self._key(cache_key), response, ex=ttl)

    async def invalidate(self, pattern: str) -> None:
        """Delete all keys matching *pattern* (glob-style, e.g. ``repo:*``)."""
        full_pattern = self._key(pattern)
        client = self._redis.client
        cursor: int | bytes = 0
        while True:
            cursor, keys = await client.scan(cursor=cursor, match=full_pattern, count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break
