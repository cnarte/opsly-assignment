"""Async client for the GitNexus MCP stdio server."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


class GitNexusClient:
    """Manages a persistent `gitnexus mcp` subprocess and exposes its tools."""

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._stack = AsyncExitStack()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Spawn `gitnexus mcp` and initialise the MCP session."""
        params = StdioServerParameters(command="gitnexus", args=["mcp"])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = ClientSession(read, write)
        await self._stack.enter_async_context(session)
        await session.initialize()
        self._session = session
        logger.info("GitNexusClient connected to gitnexus mcp subprocess")

    async def close(self) -> None:
        """Shut down the MCP session and terminate the subprocess."""
        await self._stack.aclose()
        self._session = None

    async def __aenter__(self) -> "GitNexusClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Tool calls
    # ------------------------------------------------------------------

    async def call_tool(self, name: str, args: dict) -> dict:
        """Call a gitnexus MCP tool and return the parsed result."""
        if self._session is None:
            raise RuntimeError("GitNexusClient is not connected. Call connect() first.")
        result = await self._session.call_tool(name, args)
        if result.content:
            text = result.content[0].text
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return {"result": text}
        return {}

    # ------------------------------------------------------------------
    # Indexing (CLI — not an MCP tool)
    # ------------------------------------------------------------------

    async def analyze_repo(self, path: str, repo_name: str) -> dict:
        """Run `gitnexus analyze <path>` (or `index` if already analyzed) to register the repo."""
        import os
        from pathlib import Path

        logger.info("Indexing repo '%s' at %s", repo_name, path)

        # Use `gitnexus index` if .gitnexus already exists (fast, no re-analysis)
        gitnexus_dir = Path(path) / ".gitnexus"
        if gitnexus_dir.exists():
            cmd = ["gitnexus", "index", path]
        else:
            cmd = ["gitnexus", "analyze", path]

        # Use anyio.run_process to stay within anyio's task group context
        result = await anyio.run_process(cmd, check=False)
        if result.returncode != 0:
            err = result.stderr.decode().strip() or result.stdout.decode().strip()
            raise RuntimeError(
                f"gitnexus failed for '{repo_name}': {err}"
            )
        logger.info("Indexed '%s' successfully", repo_name)
        return {"status": "indexed", "repo": repo_name, "path": path}
