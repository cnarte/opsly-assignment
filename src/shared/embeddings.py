"""Embedding generation service for semantic code search.

Uses OpenAI's text-embedding-3-small model (1536 dimensions) by default.
Supports batching for efficiency and caching to avoid re-embedding.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

import httpx

from src.shared.settings import Settings

logger = logging.getLogger(__name__)

# Default embedding model configuration
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSIONS = 1536
BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request


class EmbeddingService:
    """Generate embeddings for text using OpenAI's embedding API."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._model = model or getattr(self._settings, "EMBEDDING_MODEL", DEFAULT_MODEL)
        self._dimensions = dimensions or getattr(self._settings, "EMBEDDING_DIMENSIONS", DEFAULT_DIMENSIONS)
        self._api_key = getattr(self._settings, "OPENAI_API_KEY", "")
        self._cache: dict[str, list[float]] = {}
        self._client: httpx.AsyncClient | None = None

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        return self._dimensions

    @property
    def model(self) -> str:
        """Return the embedding model name."""
        return self._model

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    def _cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.sha256(f"{self._model}:{text}".encode()).hexdigest()

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text or not text.strip():
            return [0.0] * self._dimensions

        # Check cache
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate embedding
        embeddings = await self._call_api([text])
        if embeddings:
            self._cache[cache_key] = embeddings[0]
            return embeddings[0]

        return [0.0] * self._dimensions

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as input texts.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        # Check cache first
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = [0.0] * self._dimensions
            else:
                cache_key = self._cache_key(text)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                else:
                    to_embed.append((i, text))

        # Batch embed uncached texts
        if to_embed:
            for batch_start in range(0, len(to_embed), BATCH_SIZE):
                batch = to_embed[batch_start : batch_start + BATCH_SIZE]
                batch_texts = [t for _, t in batch]
                embeddings = await self._call_api(batch_texts)

                for j, (orig_idx, text) in enumerate(batch):
                    if j < len(embeddings):
                        embedding = embeddings[j]
                        self._cache[self._cache_key(text)] = embedding
                        results[orig_idx] = embedding
                    else:
                        results[orig_idx] = [0.0] * self._dimensions

        # Fill any remaining None values
        return [r if r is not None else [0.0] * self._dimensions for r in results]

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not self._api_key:
            logger.warning("No OPENAI_API_KEY configured, returning zero vectors")
            return [[0.0] * self._dimensions for _ in texts]

        try:
            client = await self._get_client()
            response = await client.post(
                "/embeddings",
                json={
                    "model": self._model,
                    "input": texts,
                    "dimensions": self._dimensions,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index to maintain order
            embeddings_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings_data]

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            return [[0.0] * self._dimensions for _ in texts]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [[0.0] * self._dimensions for _ in texts]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def build_embedding_text(
    name: str,
    docstring: str | None = None,
    signature: str | None = None,
    kind: str = "function",
) -> str:
    """Build a text representation for embedding a code entity.

    Combines name, docstring, and signature into a single string
    optimized for semantic search.

    Args:
        name: The entity name (function, class, method name).
        docstring: Optional docstring text.
        signature: Optional function/method signature.
        kind: The entity type (function, class, method).

    Returns:
        A combined text string for embedding.
    """
    parts = [f"{kind}: {name}"]

    if signature:
        parts.append(f"signature: {signature}")

    if docstring:
        # Truncate very long docstrings
        doc = docstring[:1000] if len(docstring) > 1000 else docstring
        parts.append(f"description: {doc}")

    return " | ".join(parts)


# Singleton instance for convenience
_embedding_service: EmbeddingService | None = None


async def get_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    """Get or create the singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(settings)
    return _embedding_service
