"""Redis-backed session management for conversation history."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.shared.redis_client import RedisClient

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 86400  # 24 hours


class SessionManager:
    """Manages per-session conversation history in Redis lists."""

    def __init__(self, redis: RedisClient, ttl: int = _DEFAULT_TTL) -> None:
        self._redis = redis
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a message to the session history."""
        msg = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        key = self._key(session_id)
        client = self._redis.client
        await client.rpush(key, json.dumps(msg))
        await client.expire(key, self._ttl)

    async def get_history(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return the last *limit* messages for a session."""
        key = self._key(session_id)
        client = self._redis.client
        # Fetch last N items from the list
        items = await client.lrange(key, -limit, -1)
        return [json.loads(item) for item in items]

    async def clear_session(self, session_id: str) -> None:
        """Delete all messages for a session."""
        await self._redis.delete(self._key(session_id))
