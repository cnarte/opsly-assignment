# GitNexus Branch — Plan 1: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `feature/gitnexus-backend` branch, stop old containers, build the `gitnexus-agent` MCP service (port 8015), and wire it into docker-compose — so all other agents have a live GitNexus backend to call.

**Architecture:** A new Python + Node.js container (`gitnexus-agent`) spawns `gitnexus mcp` as a persistent stdio subprocess on startup. It exposes seven MCP tools over HTTP (streamable-http transport) at port 8015. Other agents call these tools instead of writing Cypher against Neo4j. A shared Docker volume (`gitnexus_workspace`) holds cloned repos and LadybugDB indexes.

**Tech Stack:** Python 3.12, Node.js 20, gitnexus CLI (npm), FastMCP (mcp>=1.25), mcp.client.stdio, asyncio, Docker Compose

---

### Task 1: Create branch and stop old containers

**Files:** none (git + docker commands only)

- [ ] **Step 1: Check out the new branch**

```bash
git checkout main
git checkout -b feature/gitnexus-backend
```

Expected: `Switched to a new branch 'feature/gitnexus-backend'`

- [ ] **Step 2: Stop all running containers from the old branch**

```bash
docker compose down
```

Expected output ends with: `Network opsly-assignment_default  Removed`

- [ ] **Step 3: Confirm nothing is running**

```bash
docker ps --filter "name=opsly" --format "table {{.Names}}\t{{.Status}}"
```

Expected: empty table (no containers listed)

- [ ] **Step 4: Commit branch marker**

```bash
git commit --allow-empty -m "chore: start feature/gitnexus-backend branch"
```

---

### Task 2: Add GITNEXUS_PORT to Settings

**Files:**
- Modify: `src/shared/settings.py`

- [ ] **Step 1: Add the port constant**

In `src/shared/settings.py`, add one line after `MEMORY_PORT`:

```python
    MEMORY_PORT: int = 8014
    GITNEXUS_PORT: int = 8015       # ← add this line
    GATEWAY_PORT: int = 8000
```

- [ ] **Step 2: Add to .env.example**

Add to `.env.example` (or create it if absent):

```bash
GITNEXUS_PORT=8015
```

- [ ] **Step 3: Verify import works**

```bash
python -c "from src.shared.settings import Settings; s = Settings(); print(s.GITNEXUS_PORT)"
```

Expected: `8015`

- [ ] **Step 4: Commit**

```bash
git add src/shared/settings.py .env.example
git commit -m "feat: add GITNEXUS_PORT to Settings"
```

---

### Task 3: Create GitNexusClient (async stdio MCP wrapper)

**Files:**
- Create: `src/shared/gitnexus_client.py`
- Create: `tests/shared/test_gitnexus_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/shared/test_gitnexus_client.py`:

```python
"""Unit tests for GitNexusClient — mock the stdio transport."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_connect_initialises_session():
    """connect() should enter stdio_client context and call session.initialize()."""
    from src.shared.gitnexus_client import GitNexusClient

    mock_session = AsyncMock()
    mock_read = AsyncMock()
    mock_write = AsyncMock()

    with patch("src.shared.gitnexus_client.stdio_client") as mock_stdio, \
         patch("src.shared.gitnexus_client.ClientSession") as mock_cls:

        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        client = GitNexusClient()
        await client.connect()

        mock_session.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_tool_parses_json():
    """call_tool() should parse JSON from the first content item."""
    from src.shared.gitnexus_client import GitNexusClient

    mock_session = AsyncMock()
    content_item = MagicMock()
    content_item.text = json.dumps({"results": [{"name": "FastAPI"}]})
    mock_result = MagicMock()
    mock_result.content = [content_item]
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    client = GitNexusClient()
    client._session = mock_session

    result = await client.call_tool("query", {"query": "FastAPI"})
    assert result == {"results": [{"name": "FastAPI"}]}
    mock_session.call_tool.assert_awaited_once_with("query", {"query": "FastAPI"})


@pytest.mark.asyncio
async def test_call_tool_returns_raw_on_non_json():
    """call_tool() should wrap non-JSON text in {"result": ...}."""
    from src.shared.gitnexus_client import GitNexusClient

    mock_session = AsyncMock()
    content_item = MagicMock()
    content_item.text = "not json"
    mock_result = MagicMock()
    mock_result.content = [content_item]
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    client = GitNexusClient()
    client._session = mock_session

    result = await client.call_tool("cypher", {"query": "MATCH (n) RETURN n"})
    assert result == {"result": "not json"}


@pytest.mark.asyncio
async def test_call_tool_raises_when_not_connected():
    """call_tool() should raise RuntimeError if connect() was never called."""
    from src.shared.gitnexus_client import GitNexusClient

    client = GitNexusClient()
    with pytest.raises(RuntimeError, match="not connected"):
        await client.call_tool("query", {})
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/shared/test_gitnexus_client.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError` or `ImportError` (file doesn't exist yet)

- [ ] **Step 3: Create `src/shared/gitnexus_client.py`**

```python
"""Async client for the GitNexus MCP stdio server."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack

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
        """Run `gitnexus analyze <path>` to index a repo into LadybugDB."""
        logger.info("Indexing repo '%s' at %s", repo_name, path)
        proc = await asyncio.create_subprocess_exec(
            "gitnexus", "analyze", path, "--skip-embeddings",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"gitnexus analyze failed for '{repo_name}': {stderr.decode().strip()}"
            )
        logger.info("Indexed '%s' successfully", repo_name)
        return {"status": "indexed", "repo": repo_name, "path": path}
```

- [ ] **Step 4: Run tests — all should pass**

```bash
pytest tests/shared/test_gitnexus_client.py -v
```

Expected:
```
test_connect_initialises_session PASSED
test_call_tool_parses_json PASSED
test_call_tool_returns_raw_on_non_json PASSED
test_call_tool_raises_when_not_connected PASSED
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/shared/gitnexus_client.py tests/shared/test_gitnexus_client.py
git commit -m "feat: add GitNexusClient async stdio MCP wrapper"
```

---

### Task 4: Create gitnexus-agent MCP server

**Files:**
- Create: `src/gitnexus_agent/__init__.py`
- Create: `src/gitnexus_agent/server.py`
- Create: `tests/gitnexus_agent/test_server.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/gitnexus_agent/test_server.py`:

```python
"""Tests for gitnexus-agent MCP server tool shapes."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.call_tool = AsyncMock(return_value={"results": [], "count": 0})
    client.analyze_repo = AsyncMock(return_value={"status": "indexed", "repo": "test", "path": "/tmp/test"})
    return client


@pytest.mark.asyncio
async def test_query_passes_repo_param(mock_client):
    """query() should pass repo param to gitnexus when provided."""
    with patch("src.gitnexus_agent.server._get_client", AsyncMock(return_value=mock_client)):
        from src.gitnexus_agent.server import query as _query
        await _query("FastAPI class", repo="fastapi")
        mock_client.call_tool.assert_awaited_once_with("query", {"query": "FastAPI class", "repo": "fastapi"})


@pytest.mark.asyncio
async def test_query_omits_empty_repo(mock_client):
    """query() should not pass repo key when repo is empty string."""
    with patch("src.gitnexus_agent.server._get_client", AsyncMock(return_value=mock_client)):
        from src.gitnexus_agent.server import query as _query
        await _query("some search", repo="")
        mock_client.call_tool.assert_awaited_once_with("query", {"query": "some search"})


@pytest.mark.asyncio
async def test_analyze_repo_delegates_to_client(mock_client):
    """analyze_repo() should call client.analyze_repo with path and name."""
    with patch("src.gitnexus_agent.server._get_client", AsyncMock(return_value=mock_client)):
        from src.gitnexus_agent.server import analyze_repo
        result = await analyze_repo("/workspace/repos/fastapi", "fastapi")
        mock_client.analyze_repo.assert_awaited_once_with("/workspace/repos/fastapi", "fastapi")
        assert result["status"] == "indexed"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/gitnexus_agent/test_server.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError` (module doesn't exist yet)

- [ ] **Step 3: Create `src/gitnexus_agent/__init__.py`**

```python
"""GitNexus agent — MCP server wrapping the gitnexus CLI."""
```

- [ ] **Step 4: Create `src/gitnexus_agent/server.py`**

```python
"""GitNexus Agent — MCP server that proxies to the gitnexus CLI."""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from src.shared.gitnexus_client import GitNexusClient
from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP(
    "gitnexus-agent",
    host="0.0.0.0",
    port=settings.GITNEXUS_PORT,
)

_client: GitNexusClient | None = None


async def _get_client() -> GitNexusClient:
    """Return the singleton GitNexusClient, connecting on first call."""
    global _client
    if _client is None:
        _client = GitNexusClient()
        await _client.connect()
    return _client


def _args(base: dict, repo: str) -> dict:
    """Add repo key only when non-empty."""
    return {**base, "repo": repo} if repo else base


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@mcp.tool()
async def analyze_repo(path: str, repo_name: str) -> dict:
    """Index a repository at `path` using `gitnexus analyze`.

    Args:
        path: Absolute path to the cloned repository on disk.
        repo_name: Short identifier used in subsequent queries (e.g. 'fastapi').

    Returns:
        {"status": "indexed", "repo": repo_name, "path": path}
    """
    client = await _get_client()
    return await client.analyze_repo(path, repo_name)


# ---------------------------------------------------------------------------
# Query tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def query(q: str, repo: str = "") -> dict:
    """Hybrid BM25 + semantic search across the indexed codebase.

    Args:
        q: Natural language or keyword query.
        repo: Repo name to scope search. Empty = all repos.
    """
    client = await _get_client()
    return await client.call_tool("query", _args({"query": q}, repo))


@mcp.tool()
async def context(symbol: str, repo: str = "") -> dict:
    """360-degree view of a symbol: callers, callees, imports, process participation.

    Args:
        symbol: Exact or partial symbol name (class, function, method).
        repo: Repo name to scope. Empty = all repos.
    """
    client = await _get_client()
    return await client.call_tool("context", _args({"symbol": symbol}, repo))


@mcp.tool()
async def impact(symbol: str, repo: str = "") -> dict:
    """Blast-radius analysis: what breaks if this symbol changes.

    Args:
        symbol: Symbol name to analyse.
        repo: Repo name to scope. Empty = all repos.
    """
    client = await _get_client()
    return await client.call_tool("impact", _args({"symbol": symbol}, repo))


@mcp.tool()
async def cypher(query_str: str, repo: str = "") -> dict:
    """Run a raw Cypher query against the LadybugDB graph (read-only).

    Args:
        query_str: Cypher query string.
        repo: Repo name to scope. Empty = all repos.
    """
    client = await _get_client()
    return await client.call_tool("cypher", _args({"query": query_str}, repo))


@mcp.tool()
async def list_repos() -> dict:
    """Return all repositories currently indexed in LadybugDB."""
    client = await _get_client()
    return await client.call_tool("list_repos", {})


@mcp.tool()
async def group_query(q: str) -> dict:
    """Search for execution flows and contracts across ALL indexed repos.

    Args:
        q: Natural language query spanning multiple repos.
    """
    client = await _get_client()
    return await client.call_tool("group_query", {"query": q})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

- [ ] **Step 5: Run tests — all should pass**

```bash
pytest tests/gitnexus_agent/test_server.py -v
```

Expected:
```
test_query_passes_repo_param PASSED
test_query_omits_empty_repo PASSED
test_analyze_repo_delegates_to_client PASSED
3 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/gitnexus_agent/ tests/gitnexus_agent/
git commit -m "feat: add gitnexus-agent MCP server"
```

---

### Task 5: Create Dockerfile.gitnexus-agent

**Files:**
- Create: `docker/Dockerfile.gitnexus-agent`

- [ ] **Step 1: Create the Dockerfile**

```dockerfile
# docker/Dockerfile.gitnexus-agent
# Python 3.12 + Node.js 20 + gitnexus CLI
FROM python:3.12-slim

# Install Node.js 20 and git (needed for gitnexus analyze)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install gitnexus CLI globally
RUN npm install -g gitnexus

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Workspace volume mount point
RUN mkdir -p /workspace/repos

EXPOSE 8015

CMD ["python", "-m", "src.gitnexus_agent.server"]
```

- [ ] **Step 2: Test the build locally (outside compose)**

```bash
docker build -f docker/Dockerfile.gitnexus-agent -t gitnexus-agent-test . 2>&1 | tail -20
```

Expected: `Successfully built <hash>` with no errors.

If `nodesource` setup script fails on your Docker platform, replace the Node install section with:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*
```

- [ ] **Step 3: Verify gitnexus is installed in the image**

```bash
docker run --rm gitnexus-agent-test gitnexus --version
```

Expected: prints the gitnexus version string, no error.

- [ ] **Step 4: Commit**

```bash
git add docker/Dockerfile.gitnexus-agent
git commit -m "feat: add Dockerfile.gitnexus-agent (Python 3.12 + Node.js 20 + gitnexus)"
```

---

### Task 6: Update docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add gitnexus-agent service and shared volume**

Replace the `volumes:` section at the bottom and add the new service. The full changed sections are:

```yaml
# Add this service block after the `memory:` service
  gitnexus-agent:
    build:
      context: .
      dockerfile: docker/Dockerfile.gitnexus-agent
    ports:
      - "8015:8015"
    volumes:
      - gitnexus_workspace:/workspace/repos
    env_file: .env
    healthcheck:
      test: ["CMD", "python", "-c", "import socket; s=socket.create_connection(('localhost',8015),2); s.close()"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 30s
```

- [ ] **Step 2: Add workspace volume to indexer service**

In the `indexer:` service block, add:

```yaml
  indexer:
    # ... existing config ...
    volumes:
      - gitnexus_workspace:/workspace/repos
    depends_on:
      neo4j:
        condition: service_healthy
      gitnexus-agent:
        condition: service_healthy
```

- [ ] **Step 3: Update graph-query to depend on gitnexus-agent (remove neo4j dependency)**

```yaml
  graph-query:
    # ... existing config ...
    depends_on:
      gitnexus-agent:
        condition: service_healthy
    # remove: neo4j dependency
```

- [ ] **Step 4: Add new volume to volumes section**

```yaml
volumes:
  neo4j_data:
  redis_data:
  gitnexus_workspace:    # ← add this
```

- [ ] **Step 5: Bring up just the foundation services to verify**

```bash
docker compose up --build gitnexus-agent -d
```

Wait ~30 seconds for startup, then:

```bash
docker compose logs gitnexus-agent --tail 20
```

Expected logs include:
```
StreamableHTTP session manager started
Uvicorn running on http://0.0.0.0:8015
```

- [ ] **Step 6: Verify MCP tools are discoverable**

```bash
curl -s -X POST http://localhost:8015/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python3 -m json.tool | grep '"name"'
```

Expected (7 tool names):
```
"name": "analyze_repo"
"name": "query"
"name": "context"
"name": "impact"
"name": "cypher"
"name": "list_repos"
"name": "group_query"
```

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add gitnexus-agent to docker-compose with shared workspace volume"
```

---

### Task 7: Smoke-test the full foundation

- [ ] **Step 1: Bring up all services**

```bash
docker compose up --build -d
docker compose ps
```

Expected: all services `Up` (neo4j, redis, gitnexus-agent, memory, indexer, graph-query, orchestrator, code-analyst, gateway, ui)

- [ ] **Step 2: Health check**

```bash
curl -s http://localhost:8000/api/agents/health | python3 -m json.tool
```

Expected: all agents listed, status `ok` (some may be `unhealthy` — graph-query and indexer will degrade gracefully until Plan 2 rewrites them)

- [ ] **Step 3: Index FastAPI repo via gitnexus-agent directly**

```bash
# Shell into the gitnexus-agent container
docker compose exec gitnexus-agent bash -c "
  git clone --depth 1 https://github.com/fastapi/fastapi.git /workspace/repos/fastapi 2>&1 | tail -3
  gitnexus analyze /workspace/repos/fastapi --skip-embeddings 2>&1 | tail -5
"
```

Expected: git clone succeeds, gitnexus outputs progress and ends with `Analysis complete` or similar.

- [ ] **Step 4: Test a live query**

```bash
curl -s -X POST http://localhost:8015/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"query","arguments":{"query":"FastAPI class","repo":"fastapi"}}}' \
  | python3 -m json.tool | head -30
```

Expected: JSON result with at least one matching symbol for `FastAPI`.

- [ ] **Step 5: Final commit for Plan 1**

```bash
git add -A
git commit -m "feat: Plan 1 complete — gitnexus-agent foundation ready"
```

---

## Plan 1 Complete

Proceed to **Plan 2: Agent rewrites** (`docs/superpowers/plans/2026-04-12-gitnexus-plan-2-agents.md`) once all services are healthy and the smoke test passes.
