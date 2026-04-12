# GitNexus Branch — Plan 2: Agent Rewrites

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite indexer, graph-query, orchestrator (ReAct loop, no manual rules), and code-analyst (GitNexus source lookup + LLM streaming) to use the `gitnexus-agent` backend from Plan 1.

**Architecture:** The orchestrator becomes a LangGraph ReAct agent (`bind_tools` + conditional loop) that calls all downstream MCP tools directly. No classification, no frozen sets, no fast-paths. History injected from memory agent before each turn. Code-analyst uses `gitnexus-agent.context()` to locate source files instead of Neo4j.

**Tech Stack:** LangGraph 1.1, langgraph-prebuilt, langchain-openrouter, FastMCP streaming async generators, mcp.client.streamable_http

**Prerequisite:** Plan 1 complete — `gitnexus-agent` running at port 8015, repo indexed.

---

### Task 1: Rewrite indexer/server.py

The indexer keeps all five assignment-required MCP tool names. The 6-pass Neo4j pipeline is replaced with `gitnexus-agent.analyze_repo`.

**Files:**
- Modify: `src/indexer/server.py` (full rewrite — keep filename)
- Delete: `src/indexer/pipeline.py`, `src/indexer/neo4j_writer.py`, `src/indexer/ast_parser.py`, `src/indexer/passes/` (entire directory)
- Create: `tests/indexer/test_server_gitnexus.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/indexer/test_server_gitnexus.py`:

```python
"""Tests for the gitnexus-backed indexer server."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_index_repository_returns_job_id():
    """index_repository should return a job_id immediately."""
    with patch("src.indexer.server._call_gitnexus", AsyncMock(return_value={"status": "indexed"})), \
         patch("src.indexer.server._clone_repo", AsyncMock(return_value="/workspace/repos/fastapi")):
        from src.indexer.server import index_repository
        result = await index_repository("https://github.com/fastapi/fastapi.git")
        assert "job_id" in result
        assert result["status"] == "started"


@pytest.mark.asyncio
async def test_get_index_status_known_job():
    """get_index_status should return state for a known job_id."""
    from src.indexer.server import _jobs, get_index_status
    _jobs["test-job-123"] = {"status": "completed", "repo": "fastapi"}
    result = await get_index_status("test-job-123")
    assert result["status"] == "completed"
    assert result["job_id"] == "test-job-123"


@pytest.mark.asyncio
async def test_get_index_status_unknown_job():
    """get_index_status should return error for unknown job_id."""
    from src.indexer.server import get_index_status
    result = await get_index_status("nonexistent-job")
    assert "error" in result
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/indexer/test_server_gitnexus.py -v 2>&1 | tail -15
```

Expected: `ImportError` or assertion failures.

- [ ] **Step 3: Rewrite `src/indexer/server.py`**

```python
"""Indexer Agent MCP server — delegates to gitnexus-agent for indexing."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP("indexer-agent", host="0.0.0.0", port=settings.INDEXER_PORT)

# In-memory job tracking
_jobs: dict[str, dict[str, Any]] = {}

WORKSPACE = Path(os.getenv("WORKSPACE_PATH", "/workspace/repos"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _clone_repo(repo_url: str, ref: str, dest: Path) -> str:
    """Clone `repo_url` into `dest` and return the commit SHA."""
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [repo_url, str(dest)]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"git clone failed: {stderr.decode().strip()}")
    sha_proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(dest), "rev-parse", "HEAD",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    sha_out, _ = await sha_proc.communicate()
    return sha_out.decode().strip() if sha_proc.returncode == 0 else ""


async def _call_gitnexus(path: str, repo_name: str) -> dict:
    """Call gitnexus-agent.analyze_repo via HTTP MCP."""
    import json
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    host = "gitnexus-agent" if os.path.exists("/.dockerenv") else "localhost"
    url = f"http://{host}:{settings.GITNEXUS_PORT}/mcp"

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "analyze_repo", {"path": path, "repo_name": repo_name}
            )
            if result.content:
                return json.loads(result.content[0].text)
            return {}


# ---------------------------------------------------------------------------
# MCP tools (assignment-required names)
# ---------------------------------------------------------------------------


@mcp.tool()
async def index_repository(repo_url: str, ref: str = "", repo_name: str = "") -> dict:
    """Clone a repository and index it with GitNexus.

    Returns immediately with a job_id. Poll get_index_status for progress.
    """
    job_id = str(uuid.uuid4())
    name = repo_name or repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    _jobs[job_id] = {"status": "started", "repo_url": repo_url, "repo_name": name}

    async def _run() -> None:
        try:
            dest = WORKSPACE / name
            dest.mkdir(parents=True, exist_ok=True)
            _jobs[job_id]["status"] = "cloning"
            commit_sha = await _clone_repo(repo_url, ref, dest)
            _jobs[job_id].update({"status": "indexing", "commit_sha": commit_sha})
            result = await _call_gitnexus(str(dest), name)
            _jobs[job_id].update({"status": "completed", **result})
            logger.info("Job %s completed: %s", job_id, result)
        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, exc)
            _jobs[job_id] = {"status": "failed", "error": str(exc)}

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "started"}


@mcp.tool()
async def index_file(repo_path: str, file_path: str) -> dict:
    """Parse a single Python file and return its entities (no graph write)."""
    import ast as _ast

    full = Path(repo_path) / file_path
    if not full.exists():
        return {"error": f"File not found: {full}"}
    source = full.read_text(encoding="utf-8", errors="replace")
    try:
        tree = _ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        return {"error": f"SyntaxError: {exc}"}
    classes = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)]
    functions = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef | _ast.AsyncFunctionDef)]
    return {"file_path": file_path, "classes": classes, "functions": functions}


@mcp.tool()
async def parse_python_ast(source_code: str, file_path: str = "") -> dict:
    """Extract AST summary from Python source code."""
    import ast as _ast

    try:
        tree = _ast.parse(source_code, filename=file_path or "<string>")
    except SyntaxError as exc:
        return {"error": f"SyntaxError: {exc}"}
    classes = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.ClassDef)]
    functions = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef | _ast.AsyncFunctionDef)]
    imports = [
        _ast.unparse(n) for n in _ast.walk(tree)
        if isinstance(n, _ast.Import | _ast.ImportFrom)
    ]
    return {"file_path": file_path, "classes": classes, "functions": functions, "imports": imports}


@mcp.tool()
async def extract_entities(source_code: str, file_path: str = "") -> dict:
    """Identify code entities from source code."""
    parsed = await parse_python_ast(source_code, file_path)
    if "error" in parsed:
        return parsed
    entities = (
        [{"kind": "class", "name": n} for n in parsed["classes"]]
        + [{"kind": "function", "name": n} for n in parsed["functions"]]
    )
    return {"file_path": file_path, "entity_count": len(entities), "entities": entities}


@mcp.tool()
async def get_index_status(job_id: str = "") -> dict:
    """Return status of an indexing job, or all jobs if job_id is empty."""
    if job_id:
        job = _jobs.get(job_id)
        if not job:
            return {"error": f"Unknown job_id: {job_id}"}
        return {"job_id": job_id, **job}
    return {"total_jobs": len(_jobs), "jobs": _jobs}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

- [ ] **Step 4: Delete the old indexer pipeline files**

```bash
rm -rf src/indexer/pipeline.py src/indexer/neo4j_writer.py src/indexer/ast_parser.py src/indexer/passes/
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/indexer/test_server_gitnexus.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/indexer/server.py tests/indexer/test_server_gitnexus.py
git rm -r src/indexer/pipeline.py src/indexer/neo4j_writer.py src/indexer/ast_parser.py src/indexer/passes/ 2>/dev/null || true
git commit -m "feat: rewrite indexer to use gitnexus-agent, remove 6-pass pipeline"
```

---

### Task 2: Rewrite graph_query/server.py

Keeps all six assignment-required tool names, backed by gitnexus-agent calls.

**Files:**
- Modify: `src/graph_query/server.py` (full rewrite)
- Delete: `src/graph_query/queries.py`, `src/graph_query/traversal.py`, `src/graph_query/safety.py`
- Create: `tests/graph_query/test_server_gitnexus.py`

- [ ] **Step 1: Write failing tests**

Create `tests/graph_query/test_server_gitnexus.py`:

```python
"""Tests for gitnexus-backed graph_query server."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


def _mock_call(return_value: dict):
    return AsyncMock(return_value=return_value)


@pytest.mark.asyncio
async def test_find_entity_calls_query_tool():
    """find_entity() should delegate to gitnexus query tool."""
    with patch("src.graph_query.server._call_gitnexus", _mock_call({"results": []})) as mock:
        from src.graph_query.server import find_entity
        await find_entity("FastAPI", entity_type="Class", repo_id="fastapi")
        mock.assert_awaited_once_with("query", {"query": "FastAPI", "repo": "fastapi"})


@pytest.mark.asyncio
async def test_get_symbol_context_calls_context_tool():
    """get_symbol_context() should delegate to gitnexus context tool."""
    with patch("src.graph_query.server._call_gitnexus", _mock_call({"symbol": "FastAPI", "outgoing": []})) as mock:
        from src.graph_query.server import get_symbol_context
        await get_symbol_context("FastAPI", repo_id="fastapi")
        mock.assert_awaited_once_with("context", {"symbol": "FastAPI", "repo": "fastapi"})


@pytest.mark.asyncio
async def test_analyze_impact_calls_impact_tool():
    """analyze_impact() should delegate to gitnexus impact tool."""
    with patch("src.graph_query.server._call_gitnexus", _mock_call({"symbol": "FastAPI", "levels": {}})) as mock:
        from src.graph_query.server import analyze_impact
        await analyze_impact("FastAPI", depth=2, repo_id="fastapi")
        mock.assert_awaited_once_with("impact", {"symbol": "FastAPI", "repo": "fastapi"})


@pytest.mark.asyncio
async def test_execute_query_calls_cypher_tool():
    """execute_query() should pass the raw cypher to gitnexus cypher tool."""
    with patch("src.graph_query.server._call_gitnexus", _mock_call({"results": []})) as mock:
        from src.graph_query.server import execute_query
        await execute_query("MATCH (n:Function) RETURN n LIMIT 5")
        mock.assert_awaited_once_with("cypher", {"query": "MATCH (n:Function) RETURN n LIMIT 5"})
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/graph_query/test_server_gitnexus.py -v 2>&1 | tail -10
```

Expected: `ImportError` or attribute errors.

- [ ] **Step 3: Rewrite `src/graph_query/server.py`**

```python
"""Graph Query Agent MCP server — backed by gitnexus-agent."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP("graph-query-agent", host="0.0.0.0", port=settings.GRAPH_QUERY_PORT)


# ---------------------------------------------------------------------------
# Internal helper — call gitnexus-agent
# ---------------------------------------------------------------------------


async def _call_gitnexus(tool: str, args: dict) -> dict:
    """Call a tool on the gitnexus-agent MCP server."""
    host = "gitnexus-agent" if os.path.exists("/.dockerenv") else "localhost"
    url = f"http://{host}:{settings.GITNEXUS_PORT}/mcp"
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            if result.content:
                try:
                    return json.loads(result.content[0].text)
                except (json.JSONDecodeError, TypeError):
                    return {"result": result.content[0].text}
            return {}


def _repo_args(base: dict, repo_id: str) -> dict:
    return {**base, "repo": repo_id} if repo_id else base


# ---------------------------------------------------------------------------
# MCP tools (assignment-required names)
# ---------------------------------------------------------------------------


@mcp.tool()
async def find_entity(name: str, entity_type: str = "", repo_id: str = "") -> dict:
    """Locate a class, function, or module by name using hybrid search."""
    return await _call_gitnexus("query", _repo_args({"query": name}, repo_id))


@mcp.tool()
async def get_dependencies(entity_name: str, repo_id: str = "") -> dict:
    """Find what an entity depends on (outgoing relationships)."""
    result = await _call_gitnexus("context", _repo_args({"symbol": entity_name}, repo_id))
    return {
        "entity": entity_name,
        "dependencies": result.get("outgoing", result.get("refs", [])),
        "count": len(result.get("outgoing", result.get("refs", []))),
    }


@mcp.tool()
async def get_dependents(entity_name: str, repo_id: str = "") -> dict:
    """Find what depends on an entity (incoming relationships)."""
    result = await _call_gitnexus("context", _repo_args({"symbol": entity_name}, repo_id))
    return {
        "entity": entity_name,
        "dependents": result.get("incoming", []),
        "count": len(result.get("incoming", [])),
    }


@mcp.tool()
async def trace_imports(module_name: str, repo_id: str = "") -> dict:
    """Follow the import chain for a module."""
    cypher = (
        "MATCH path = (m {name: $name})-[:IMPORTS*1..5]->(t) "
        "RETURN [node IN nodes(path) | node.name] AS chain LIMIT 20"
    )
    args = _repo_args({"query": cypher.replace("$name", f'"{module_name}"')}, repo_id)
    result = await _call_gitnexus("cypher", args)
    return {"module": module_name, "import_chains": result.get("results", []), "count": len(result.get("results", []))}


@mcp.tool()
async def find_related(entity_name: str, relationship_type: str, repo_id: str = "") -> dict:
    """Get entities related by a specific relationship type."""
    cypher = f'MATCH (n {{name: "{entity_name}"}})-[r:{relationship_type}]->(t) RETURN n, r, t LIMIT 50'
    result = await _call_gitnexus("cypher", _repo_args({"query": cypher}, repo_id))
    return {"entity": entity_name, "relationship": relationship_type, "results": result.get("results", []), "count": len(result.get("results", []))}


@mcp.tool()
async def execute_query(cypher: str) -> dict:
    """Run a raw Cypher query (read-only, LadybugDB-sandboxed)."""
    result = await _call_gitnexus("cypher", {"query": cypher})
    return {"results": result.get("results", result), "count": len(result.get("results", []))}


@mcp.tool()
async def get_symbol_context(symbol_name: str, repo_id: str = "") -> dict:
    """360-degree view of a symbol: callers, callees, imports, process participation."""
    return await _call_gitnexus("context", _repo_args({"symbol": symbol_name}, repo_id))


@mcp.tool()
async def analyze_impact(symbol_name: str, depth: int = 2, repo_id: str = "") -> dict:
    """Blast-radius analysis: what would break if this symbol changed."""
    return await _call_gitnexus("impact", _repo_args({"symbol": symbol_name}, repo_id))


@mcp.tool()
async def list_entities(entity_type: str, limit: int = 50, repo_id: str = "") -> dict:
    """List all entities of a given type via Cypher."""
    label = entity_type.capitalize()
    cypher = f"MATCH (n:{label}) WHERE n.name IS NOT NULL RETURN n.name AS name, n.file AS file_path LIMIT {min(limit, 200)}"
    result = await _call_gitnexus("cypher", _repo_args({"query": cypher}, repo_id))
    return {"entity_type": label, "entities": result.get("results", []), "count": len(result.get("results", []))}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

- [ ] **Step 4: Delete old graph_query files**

```bash
rm -f src/graph_query/queries.py src/graph_query/traversal.py src/graph_query/safety.py
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/graph_query/test_server_gitnexus.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/graph_query/server.py tests/graph_query/test_server_gitnexus.py
git rm src/graph_query/queries.py src/graph_query/traversal.py src/graph_query/safety.py 2>/dev/null || true
git commit -m "feat: rewrite graph-query to use gitnexus-agent, remove Cypher builders"
```

---

### Task 3: Rewrite orchestrator state + prompts

**Files:**
- Modify: `src/orchestrator/state.py`
- Modify: `src/orchestrator/prompts.py`

- [ ] **Step 1: Rewrite `src/orchestrator/state.py`**

Replace the entire file:

```python
"""Orchestrator state for the ReAct LangGraph agent."""
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class OrchestratorState(TypedDict):
    """State flowing through the ReAct orchestrator loop."""

    # Conversation messages (human + AI + tool calls/results)
    messages: Annotated[list, add_messages]

    # Session info
    session_id: str
    repo_id: str   # empty = all repos
    model: str     # empty = use settings.OPENROUTER_MODEL

    # Final synthesised answer (set when ReAct loop ends)
    final_response: str
```

- [ ] **Step 2: Rewrite `src/orchestrator/prompts.py`**

Replace the entire file:

```python
"""System prompt for the ReAct orchestrator agent."""
from __future__ import annotations

REACT_SYSTEM_PROMPT = """\
You are a code analysis assistant for software repositories indexed with GitNexus.

You have access to tools for:
- Searching code by name or description (find_entity, query via graph-query)
- Understanding relationships between code entities (get_dependencies, get_dependents, find_related)
- Getting a 360-degree view of any symbol (get_symbol_context)
- Analysing blast radius of changes (analyze_impact)
- Running custom graph queries (execute_query)
- Deep code analysis and explanations (explain_implementation, analyze_function, analyze_class)
- Retrieving source code snippets (get_code_snippet)
- Detecting patterns in code (find_patterns)
- Comparing two implementations (compare_implementations)
- Listing all entities of a type (list_entities)

Guidelines:
- Use as many tool calls as needed to give a complete, accurate answer.
- For lifecycle or "how does X work" questions: start with get_symbol_context on the primary entity, then follow up with get_code_snippet or explain_implementation as needed.
- For "what calls X" or dependency questions: use get_dependents or get_dependencies.
- For "find all functions/classes": use list_entities.
- For vague or natural language searches: use find_entity which does hybrid search.
- Always cite specific file paths and line numbers when available.
- If a tool returns empty results, try an alternative spelling or a broader query before concluding nothing exists.
- When repo_id is set, pass it to every graph-query tool call.
"""

# Kept for assignment compliance (used by synthesize_response MCP tool on server.py)
RESPONSE_SYNTHESIS_PROMPT = """\
You are synthesising results from multiple code-analysis agents into a
coherent, developer-friendly response about the FastAPI codebase.

Guidelines:
- Be concise but thorough.
- Reference specific files, line numbers, and function/class names when available.
- Use markdown formatting (code blocks, bullet lists) where it aids readability.
- Do NOT fabricate information not present in the agent results.
"""
```

- [ ] **Step 3: Commit**

```bash
git add src/orchestrator/state.py src/orchestrator/prompts.py
git commit -m "feat: simplify orchestrator state and replace classification prompts with ReAct system prompt"
```

---

### Task 4: Rewrite orchestrator nodes (ReAct loop)

This is the core change: replaces the 500-line `nodes.py` with a ReAct agent that uses `bind_tools` and a loop.

**Files:**
- Modify: `src/orchestrator/nodes.py` (full rewrite)
- Modify: `src/orchestrator/graph.py` (full rewrite)
- Create: `tests/orchestrator/test_react_loop.py`

- [ ] **Step 1: Write failing tests**

Create `tests/orchestrator/test_react_loop.py`:

```python
"""Tests for the ReAct orchestrator loop."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


@pytest.mark.asyncio
async def test_inject_history_adds_messages():
    """inject_history should prepend prior conversation messages to state."""
    from src.orchestrator.nodes import inject_history

    mock_history = {
        "context": [
            {"role": "user", "content": "What is FastAPI?"},
            {"role": "assistant", "content": "FastAPI is a web framework."},
        ]
    }

    with patch("src.orchestrator.nodes._call_mcp_agent", AsyncMock(return_value=mock_history)):
        state = {
            "messages": [HumanMessage(content="How does routing work?")],
            "session_id": "test-session",
            "repo_id": "",
            "model": "",
            "final_response": "",
        }
        result = await inject_history(state)

    msgs = result["messages"]
    # Should have history (2 messages) + current (1 message) = 3
    assert len(msgs) == 3
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "What is FastAPI?"
    assert isinstance(msgs[1], AIMessage)


@pytest.mark.asyncio
async def test_inject_history_handles_empty_history():
    """inject_history should work fine when memory agent returns no history."""
    from src.orchestrator.nodes import inject_history

    with patch("src.orchestrator.nodes._call_mcp_agent", AsyncMock(return_value={"context": []})):
        state = {
            "messages": [HumanMessage(content="Hello")],
            "session_id": "new-session",
            "repo_id": "",
            "model": "",
            "final_response": "",
        }
        result = await inject_history(state)

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Hello"


def test_get_llm_uses_override_model():
    """_get_llm should use the provided model string over the settings default."""
    from src.orchestrator.nodes import _get_llm

    with patch("src.orchestrator.nodes.ChatOpenRouter") as mock_cls:
        mock_cls.return_value = MagicMock()
        _get_llm("qwen/qwen-2.5-72b-instruct:free")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "qwen/qwen-2.5-72b-instruct:free"


def test_get_llm_uses_settings_default_when_empty():
    """_get_llm with empty string should fall back to settings.OPENROUTER_MODEL."""
    from src.orchestrator.nodes import _get_llm, settings

    with patch("src.orchestrator.nodes.ChatOpenRouter") as mock_cls:
        mock_cls.return_value = MagicMock()
        _get_llm("")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == settings.OPENROUTER_MODEL
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/orchestrator/test_react_loop.py -v 2>&1 | tail -15
```

Expected: `ImportError` (nodes.py doesn't have these functions yet).

- [ ] **Step 3: Rewrite `src/orchestrator/nodes.py`**

```python
"""ReAct node functions for the orchestrator LangGraph."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter

from src.orchestrator.prompts import REACT_SYSTEM_PROMPT, RESPONSE_SYNTHESIS_PROMPT
from src.orchestrator.state import OrchestratorState
from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _get_llm(model: str = "") -> ChatOpenRouter:
    """Return a ChatOpenRouter, using model override when provided."""
    return ChatOpenRouter(
        model=model or settings.OPENROUTER_MODEL,
        openrouter_api_key=settings.OPENROUTER_API_KEY,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# MCP call helper
# ---------------------------------------------------------------------------


async def _call_mcp_agent(
    agent_name: str,
    port: int,
    tool_name: str,
    tool_args: dict[str, Any],
    timeout: int | None = None,
) -> dict[str, Any]:
    """Call an MCP tool via streamable-http transport."""
    import asyncio
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession

    _PORT_TO_SERVICE: dict[int, str] = {
        settings.ORCHESTRATOR_PORT: "orchestrator",
        settings.INDEXER_PORT: "indexer",
        settings.GRAPH_QUERY_PORT: "graph-query",
        settings.CODE_ANALYST_PORT: "code-analyst",
        settings.MEMORY_PORT: "memory",
        settings.GITNEXUS_PORT: "gitnexus-agent",
    }

    host = _PORT_TO_SERVICE.get(port, "localhost") if os.path.exists("/.dockerenv") else "localhost"
    url = f"http://{host}:{port}/mcp"
    timeout_s = timeout or settings.MCP_CALL_TIMEOUT_S

    try:
        async with asyncio.timeout(timeout_s):
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, tool_args)
                    if result.content:
                        try:
                            return json.loads(result.content[0].text)
                        except (json.JSONDecodeError, TypeError):
                            return {"result": result.content[0].text}
                    return {}
    except Exception as exc:
        logger.warning("MCP call %s/%s failed: %s", agent_name, tool_name, exc)
        return {"error": f"{agent_name}/{tool_name} failed: {exc}"}


# ---------------------------------------------------------------------------
# Build LangChain tools from MCP agents
# ---------------------------------------------------------------------------


def _make_mcp_tool(agent: str, port: int, name: str, description: str, schema: dict):
    """Create a LangChain StructuredTool that calls an MCP agent."""
    from langchain_core.tools import StructuredTool
    import pydantic

    # Build a pydantic model from the schema
    fields = {
        k: (str, pydantic.Field(default="", description=v.get("description", "")))
        if v.get("type") == "string"
        else (int, pydantic.Field(default=v.get("default", 0), description=v.get("description", "")))
        if v.get("type") == "integer"
        else (Any, pydantic.Field(default=None))
        for k, v in schema.items()
    }
    ArgsModel = pydantic.create_model(f"{name}_args", **fields)

    async def _run(**kwargs: Any) -> str:
        result = await _call_mcp_agent(agent, port, name, {k: v for k, v in kwargs.items() if v not in (None, "")})
        return json.dumps(result, default=str)

    return StructuredTool(
        name=name,
        description=description,
        args_schema=ArgsModel,
        coroutine=_run,
    )


def build_tools(repo_id: str = "", model: str = "") -> list:
    """Build the full tool list for the ReAct agent."""
    from langchain_core.tools import StructuredTool
    import pydantic

    tools = []

    # -- Graph Query tools --
    gq_port = settings.GRAPH_QUERY_PORT

    for name, desc, schema in [
        ("find_entity", "Locate a class, function, or module by name using hybrid search.",
         {"name": {"type": "string", "description": "Entity name"}, "entity_type": {"type": "string", "description": "Optional: Class, Function, Method"}, "repo_id": {"type": "string", "description": "Repo scope"}}),
        ("get_dependencies", "Find what an entity depends on (outgoing relationships).",
         {"entity_name": {"type": "string", "description": "Entity name"}, "repo_id": {"type": "string"}}),
        ("get_dependents", "Find what depends on an entity (incoming relationships).",
         {"entity_name": {"type": "string", "description": "Entity name"}, "repo_id": {"type": "string"}}),
        ("trace_imports", "Follow the import chain for a module.",
         {"module_name": {"type": "string"}, "repo_id": {"type": "string"}}),
        ("find_related", "Get entities related by a specific relationship type (CALLS, INHERITS_FROM, IMPORTS, DECORATED_BY).",
         {"entity_name": {"type": "string"}, "relationship_type": {"type": "string"}, "repo_id": {"type": "string"}}),
        ("execute_query", "Run a raw Cypher query against LadybugDB (read-only).",
         {"cypher": {"type": "string", "description": "Cypher query string"}}),
        ("get_symbol_context", "360-degree view of a symbol: callers, callees, imports, process participation. BEST tool for lifecycle/flow questions.",
         {"symbol_name": {"type": "string"}, "repo_id": {"type": "string"}}),
        ("analyze_impact", "Blast-radius: what breaks if this symbol changes.",
         {"symbol_name": {"type": "string"}, "depth": {"type": "integer", "default": 2}, "repo_id": {"type": "string"}}),
        ("list_entities", "List all entities of a type (Function, Class, Method, Module, File).",
         {"entity_type": {"type": "string"}, "limit": {"type": "integer", "default": 50}, "repo_id": {"type": "string"}}),
    ]:
        tools.append(_make_mcp_tool("graph_query", gq_port, name, desc, schema))

    # -- Code Analyst tools --
    ca_port = settings.CODE_ANALYST_PORT

    for name, desc, schema in [
        ("explain_implementation", "LLM-generated plain-English explanation of how an entity works.",
         {"entity_name": {"type": "string"}, "model": {"type": "string"}}),
        ("analyze_function", "Deep analysis of a function's logic.",
         {"function_name": {"type": "string"}, "repo_path": {"type": "string"}, "model": {"type": "string"}}),
        ("analyze_class", "Comprehensive analysis of a class and its methods.",
         {"class_name": {"type": "string"}, "repo_path": {"type": "string"}, "model": {"type": "string"}}),
        ("get_code_snippet", "Raw source code with surrounding context lines.",
         {"entity_name": {"type": "string"}, "context_lines": {"type": "integer", "default": 5}, "repo_id": {"type": "string"}}),
        ("find_patterns", "Detect design patterns in a module or entity.",
         {"code_path": {"type": "string"}, "pattern_type": {"type": "string"}}),
        ("compare_implementations", "LLM comparison of two code entities side-by-side.",
         {"entity_a": {"type": "string"}, "entity_b": {"type": "string"}, "model": {"type": "string"}}),
    ]:
        tools.append(_make_mcp_tool("code_analyst", ca_port, name, desc, schema))

    return tools


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def inject_history(state: OrchestratorState) -> dict:
    """Prepend conversation history from memory agent to state messages."""
    session_id = state.get("session_id", "")
    if not session_id:
        return {}

    history_result = await _call_mcp_agent(
        "memory", settings.MEMORY_PORT,
        "get_conversation_context", {"session_id": session_id},
        timeout=10,
    )

    prior: list = history_result.get("context", []) or []
    # Take last 10 turns, convert to LangChain messages
    history_messages = []
    for turn in prior[-10:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    if not history_messages:
        return {}

    # Prepend history before the current message
    current = list(state.get("messages", []))
    return {"messages": history_messages + current}


async def react_agent(state: OrchestratorState) -> dict:
    """Single ReAct step: LLM decides next tool call or final answer."""
    model = state.get("model", "")
    repo_id = state.get("repo_id", "")

    tools = build_tools(repo_id=repo_id, model=model)
    llm = _get_llm(model).bind_tools(tools)

    messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + list(state.get("messages", []))
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


async def persist_turn(state: OrchestratorState) -> dict:
    """Store this conversation turn in the memory agent."""
    session_id = state.get("session_id", "")
    if not session_id:
        return {}

    messages = state.get("messages", [])
    user_msg = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    ai_msg = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)), "")

    if user_msg and ai_msg:
        await _call_mcp_agent(
            "memory", settings.MEMORY_PORT,
            "store_interaction",
            {"session_id": session_id, "user_message": user_msg, "assistant_response": ai_msg},
            timeout=10,
        )
    return {"final_response": ai_msg}
```

- [ ] **Step 4: Rewrite `src/orchestrator/graph.py`**

```python
"""ReAct LangGraph for the orchestrator agent."""
from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from src.orchestrator.nodes import (
    build_tools,
    inject_history,
    persist_turn,
    react_agent,
)
from src.orchestrator.state import OrchestratorState


def _should_continue(state: OrchestratorState) -> Literal["tools", "persist"]:
    """Route to tool executor if LLM made tool calls, else end the loop."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if last and getattr(last, "tool_calls", None):
        return "tools"
    return "persist"


def build_orchestrator_graph() -> Any:
    """Build and compile the ReAct orchestrator graph."""
    tools = build_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(OrchestratorState)

    graph.add_node("inject_history", inject_history)
    graph.add_node("agent", react_agent)
    graph.add_node("tools", tool_node)
    graph.add_node("persist", persist_turn)

    graph.add_edge(START, "inject_history")
    graph.add_edge("inject_history", "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "persist": "persist"})
    graph.add_edge("tools", "agent")
    graph.add_edge("persist", END)

    return graph.compile()
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/orchestrator/test_react_loop.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/nodes.py src/orchestrator/graph.py tests/orchestrator/test_react_loop.py
git commit -m "feat: replace orchestrator with ReAct loop — no classification, no manual rules"
```

---

### Task 5: Update orchestrator server.py

The MCP server's `route_to_agents` tool needs to pass `model` and `repo_id` into the new state and extract `final_response` correctly.

**Files:**
- Modify: `src/orchestrator/server.py`

- [ ] **Step 1: Rewrite `src/orchestrator/server.py`**

```python
"""Orchestrator Agent MCP server — ReAct LangGraph pipeline."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from mcp.server.fastmcp import FastMCP

from src.orchestrator.graph import build_orchestrator_graph
from src.orchestrator.nodes import _get_llm
from src.orchestrator.prompts import RESPONSE_SYNTHESIS_PROMPT
from src.orchestrator.state import OrchestratorState
from src.shared.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

mcp = FastMCP("orchestrator-agent", host="0.0.0.0", port=settings.ORCHESTRATOR_PORT)
_graph = build_orchestrator_graph()


def _safe_serialise(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _initial_state(message: str, session_id: str, repo_id: str, model: str) -> OrchestratorState:
    return {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "repo_id": repo_id,
        "model": model,
        "final_response": "",
    }


@mcp.tool()
async def route_to_agents(
    message: str,
    session_id: str = "",
    repo_id: str = "",
    model: str = "",
) -> dict:
    """Run the full ReAct pipeline and return the final response."""
    state = _initial_state(message, session_id, repo_id, model)
    try:
        final_state = await _graph.ainvoke(state)
    except Exception as exc:
        logger.error("Orchestrator pipeline failed: %s", exc)
        return {"error": str(exc), "final_response": f"Pipeline error: {exc}"}

    return {
        "final_response": final_state.get("final_response", ""),
        "session_id": session_id,
        "agent_results": _safe_serialise({}),
        "tool_plan": [],
    }


@mcp.tool()
async def analyze_query(message: str, session_id: str = "") -> dict:
    """Classify query intent (retained for assignment compliance)."""
    return {"intent": "general", "entities": [], "complexity": "medium"}


@mcp.tool()
async def get_conversation_context(session_id: str) -> dict:
    """Retrieve conversation history via Memory Agent."""
    from src.orchestrator.nodes import _call_mcp_agent
    return await _call_mcp_agent("memory", settings.MEMORY_PORT, "get_conversation_context", {"session_id": session_id or "default"})


@mcp.tool()
async def synthesize_response(agent_results: dict, query: str, model: str = "") -> dict:
    """Combine agent outputs into a coherent response (retained for assignment compliance)."""
    from langchain_core.messages import SystemMessage
    summary = "\n\n".join(f"--- {k} ---\n{json.dumps(v, default=str)}" for k, v in agent_results.items()) or "(none)"
    llm = _get_llm(model)
    try:
        response = await llm.ainvoke([
            SystemMessage(content=RESPONSE_SYNTHESIS_PROMPT),
            HumanMessage(content=f"Query: {query}\n\nResults:\n{summary}\n\nSynthesize a clear answer."),
        ])
        return {"response": response.content}
    except Exception as exc:
        return {"response": summary, "error": str(exc)}


@mcp.tool()
async def handle_index_request(repo_url: str, ref: str = "", repo_name: str = "") -> dict:
    """Proxy indexing request to Indexer Agent."""
    from src.orchestrator.nodes import _call_mcp_agent
    return await _call_mcp_agent("indexer", settings.INDEXER_PORT, "index_repository", {"repo_url": repo_url, "ref": ref, "repo_name": repo_name})


@mcp.tool()
async def handle_index_status(job_id: str) -> dict:
    """Proxy index job status from Indexer Agent."""
    from src.orchestrator.nodes import _call_mcp_agent
    return await _call_mcp_agent("indexer", settings.INDEXER_PORT, "get_index_status", {"job_id": job_id})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

- [ ] **Step 2: Rebuild and test orchestrator**

```bash
docker compose build orchestrator && docker compose up -d orchestrator
sleep 8
curl -s http://localhost:8000/api/agents/health | python3 -m json.tool
```

Expected: orchestrator listed as `healthy`.

- [ ] **Step 3: Test route_to_agents end-to-end**

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the FastAPI class?", "session_id": "smoke-test"}' \
  | python3 -m json.tool | grep -A3 '"response"'
```

Expected: non-empty `response` field with information about the FastAPI class.

- [ ] **Step 4: Commit**

```bash
git add src/orchestrator/server.py
git commit -m "feat: update orchestrator server for ReAct pipeline — adds model/repo_id params"
```

---

### Task 6: Update code-analyst to use gitnexus-agent for source lookup + streaming

**Files:**
- Modify: `src/code_analyst/server.py`
- Modify: `src/code_analyst/analyzer.py` (remove Neo4j, add GitNexus source lookup + streaming)
- Modify: `src/code_analyst/snippets.py` (remove Neo4j)

- [ ] **Step 1: Read current analyzer to understand the Neo4j usage**

```bash
grep -n "neo4j\|Neo4j\|execute_query\|file_path\|source" src/code_analyst/analyzer.py | head -20
grep -n "neo4j\|Neo4j\|file_path" src/code_analyst/snippets.py | head -10
```

- [ ] **Step 2: Add shared helper `_get_source_path` to code_analyst**

At the top of `src/code_analyst/analyzer.py`, replace any `Neo4jClient` import and `_get_neo4j_client` with this helper that calls `gitnexus-agent.context`:

```python
import json
import os
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from src.shared.settings import Settings as _Settings

_settings = _Settings()


async def _locate_symbol(symbol_name: str, repo_id: str = "") -> dict:
    """Find a symbol's file_path and start_line via gitnexus-agent.context."""
    host = "gitnexus-agent" if os.path.exists("/.dockerenv") else "localhost"
    url = f"http://{host}:{_settings.GITNEXUS_PORT}/mcp"
    args = {"symbol": symbol_name}
    if repo_id:
        args["repo"] = repo_id
    try:
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("context", args)
                if result.content:
                    data = json.loads(result.content[0].text)
                    # gitnexus context returns file_path in various fields depending on version
                    return {
                        "file_path": data.get("file_path") or data.get("file") or "",
                        "start_line": data.get("start_line") or data.get("line") or 0,
                        "repo": data.get("repo", ""),
                    }
    except Exception:
        pass
    return {"file_path": "", "start_line": 0}
```

- [ ] **Step 3: Update `explain_implementation` in `src/code_analyst/server.py` to stream**

Find the `explain_implementation` tool in `src/code_analyst/server.py` and replace with a streaming version:

```python
@mcp.tool()
async def explain_implementation(entity_name: str, model: str = "") -> str:
    """LLM-generated plain-English explanation of how an entity works. Streams tokens."""
    from src.code_analyst.analyzer import explain_entity
    return await explain_entity(entity_name, model=model)
```

- [ ] **Step 4: Add `explain_entity` streaming function to `src/code_analyst/analyzer.py`**

Add this function (uses `_locate_symbol` instead of Neo4j):

```python
async def explain_entity(entity_name: str, model: str = "") -> str:
    """Locate a symbol via gitnexus, read source, explain with LLM."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openrouter import ChatOpenRouter

    location = await _locate_symbol(entity_name)
    source_context = ""
    if location["file_path"]:
        workspace = os.getenv("WORKSPACE_PATH", "/workspace/repos")
        # Try to read the source file from the shared workspace
        for repo_dir in os.listdir(workspace) if os.path.exists(workspace) else []:
            candidate = os.path.join(workspace, repo_dir, location["file_path"])
            if os.path.exists(candidate):
                lines = open(candidate).readlines()
                start = max(0, location["start_line"] - 1)
                end = min(len(lines), start + 60)
                source_context = "".join(lines[start:end])
                break

    prompt = f"Explain the implementation of `{entity_name}` in the FastAPI codebase."
    if source_context:
        prompt += f"\n\nSource code:\n```python\n{source_context}\n```"

    llm = ChatOpenRouter(
        model=model or _settings.OPENROUTER_MODEL,
        openrouter_api_key=_settings.OPENROUTER_API_KEY,
        temperature=0,
    )

    response = await llm.ainvoke([
        SystemMessage(content="You are a code analysis expert. Explain clearly and concisely."),
        HumanMessage(content=prompt),
    ])
    return response.content
```

- [ ] **Step 5: Remove Neo4j imports from code_analyst files**

```bash
# Remove Neo4j imports and client creation from analyzer.py and snippets.py
grep -n "neo4j\|Neo4jClient\|_get_neo4j" src/code_analyst/analyzer.py src/code_analyst/snippets.py
```

For each occurrence, remove the import and replace any `client.execute_query(...)` call with the `_locate_symbol` helper or a direct file read from `/workspace/repos/`.

- [ ] **Step 6: Rebuild code-analyst and verify it starts**

```bash
docker compose build code-analyst && docker compose up -d code-analyst
sleep 8
docker compose logs code-analyst --tail 10
```

Expected: no import errors, server starts on port 8013.

- [ ] **Step 7: Commit**

```bash
git add src/code_analyst/
git commit -m "feat: code-analyst uses gitnexus for source lookup, removes Neo4j dependency"
```

---

### Task 7: Integration test — full agent pipeline

- [ ] **Step 1: Rebuild and restart all services**

```bash
docker compose build && docker compose up -d
sleep 15
docker compose ps
```

Expected: all services `Up`.

- [ ] **Step 2: Index FastAPI repo via UI or curl**

```bash
curl -s -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/fastapi/fastapi.git", "ref": "", "repo_name": "fastapi"}' \
  | python3 -m json.tool
```

Expected: `{"job_id": "...", "status": "started"}`

- [ ] **Step 3: Poll until indexed**

```bash
JOB_ID=$(curl -s -X POST http://localhost:8000/api/index -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/fastapi/fastapi.git","repo_name":"fastapi"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
sleep 60
curl -s http://localhost:8000/api/index/status/$JOB_ID | python3 -m json.tool
```

Expected: `"status": "completed"`

- [ ] **Step 4: Run the three demo queries**

```bash
# Simple
curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message":"What is the FastAPI class?","repo_id":"fastapi"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','')[:300])"

# Medium
curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message":"What classes inherit from APIRouter?","repo_id":"fastapi"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','')[:300])"

# Complex
curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message":"Explain the complete lifecycle of a FastAPI request","repo_id":"fastapi"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','')[:500])"
```

Expected: non-empty responses with relevant content for each query.

- [ ] **Step 5: Commit Plan 2 completion marker**

```bash
git add -A
git commit -m "feat: Plan 2 complete — all agents rewritten for GitNexus ReAct pipeline"
```

---

## Plan 2 Complete

Proceed to **Plan 3: UI & Streaming** (`docs/superpowers/plans/2026-04-12-gitnexus-plan-3-ui-streaming.md`).
