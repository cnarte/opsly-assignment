"""Async Neo4j connection pool wrapper."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from src.shared.settings import Settings


class Neo4jClient:
    """Thin async wrapper around the official Neo4j async driver."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._driver: AsyncDriver | None = None

    # -- lifecycle ------------------------------------------------------------

    async def connect(self) -> None:
        """Create the driver and verify connectivity."""
        self._driver = AsyncGraphDatabase.driver(
            self._settings.NEO4J_URI,
            auth=(self._settings.NEO4J_USER, self._settings.NEO4J_PASSWORD),
        )
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        """Close the driver and release resources."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    # -- context manager ------------------------------------------------------

    async def __aenter__(self) -> "Neo4jClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- helpers --------------------------------------------------------------

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4jClient is not connected. Call connect() first.")
        return self._driver

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read query and return a list of record dicts."""
        db = database or self._settings.NEO4J_DATABASE
        async with self.driver.session(database=db) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a write query inside an implicit transaction."""
        db = database or self._settings.NEO4J_DATABASE
        async with self.driver.session(database=db) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records
