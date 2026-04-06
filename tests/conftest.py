"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.shared.settings import Settings
from src.gateway.app import app


@pytest.fixture()
def settings() -> Settings:
    """Return a Settings instance with default values."""
    return Settings()


@pytest.fixture()
async def async_client() -> AsyncClient:
    """Provide an httpx AsyncClient wired to the FastAPI gateway."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
