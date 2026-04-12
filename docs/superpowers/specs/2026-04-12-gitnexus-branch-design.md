# GitNexus Branch Design

**Branch:** `feature/gitnexus-backend`  
**Date:** 2026-04-12  
**Status:** Approved

## Context

The current `main` branch uses a hand-rolled 6-pass AST indexer that writes to Neo4j, a Cypher-based graph query agent with hardcoded routing rules, and a LangGraph orchestrator that classifies queries through frozen-set keyword matching before dispatching tools. This produces brittle behaviour: the orchestrator decides routing before seeing results, fast-paths for lifecycle/list queries bake in assumptions, and the code analyst uses Neo4j for source file lookup.

This branch replaces that approach with GitNexus as the indexing and query backend, a clean ReAct orchestrator loop with no manual rules, end-to-end streaming, proper conversation history, multi-repo support, and a UI model selector.

The assignment requires five specific MCP servers with specific tool names. All five are preserved — the tool names stay the same on the outside; GitNexus powers them on the inside.

---

## Architecture

```
User types natural language
        │
        ▼
  Streamlit UI  (unchanged except model selector + multi-repo)
        │  POST /api/chat { message, session_id, repo_id, model, stream: true }
        ▼
  FastAPI Gateway  (unchanged — adds model/repo_id forwarding)
        │  MCP → route_to_agents(message, session_id, repo_id, model)
        ▼
  Orchestrator — ReAct loop  ◄──────────────────────────────┐
        │  .astream_events() → SSE → Gateway → UI            │
        │  Events: 🧠 thinking · 🔧 tool_call · 📊 result     │
        │  Loops until LLM decides answer is complete ────────┘
        │
        ├──► indexer-agent  (port 8011)
        │       index_repository / index_file / parse_python_ast
        │       extract_entities / get_index_status
        │       → delegates to gitnexus-agent.analyze_repo
        │
        ├──► graph-query-agent  (port 8012)
        │       find_entity / get_dependencies / get_dependents
        │       trace_imports / find_related / execute_query
        │       → all backed by gitnexus-agent tools
        │
        ├──► code-analyst-agent  (port 8013)
        │       analyze_function / analyze_class / get_code_snippet
        │       explain_implementation / find_patterns / compare_implementations
        │       → uses gitnexus-agent for source file location (replaces Neo4j)
        │       → LLM tools stream tokens via FastMCP async generator
        │
        ├──► memory-agent  (port 8014, unchanged)
        │       store_interaction / get_conversation_context
        │
        └──► gitnexus-agent  (port 8015, NEW)
                query / context / impact / cypher / analyze_repo
                group_query (cross-repo search)
                → Node.js 20 + gitnexus CLI + LadybugDB
                → spawns `gitnexus mcp` as a PERSISTENT stdio subprocess on startup
                  (one subprocess shared across all requests, not re-spawned per call)
                → shared workspace volume at /workspace/repos/
```

---

## Services

### gitnexus-agent (NEW — port 8015)

**Dockerfile:** Python 3.12 + Node.js 20 + `npm install -g gitnexus`

**Purpose:** Sole container that owns Node.js and LadybugDB. All other agents call it via MCP — they stay pure Python.

**MCP tools exposed:**

| Tool | Parameters | Behaviour |
|---|---|---|
| `analyze_repo` | `path, repo_name` | Runs `gitnexus analyze <path> --skip-embeddings`, registers in gitnexus registry |
| `query` | `q, repo=""` | Hybrid BM25 + semantic search; `repo=""` = all repos |
| `context` | `symbol, repo=""` | 360° view: callers, callees, imports, process participation |
| `impact` | `symbol, repo=""` | Blast-radius with depth groups and confidence scores |
| `cypher` | `query, repo=""` | Raw LadybugDB Cypher (read-only, sandboxed by gitnexus) |
| `group_query` | `q` | Cross-repo execution flow search |

**Startup:** On container start, check if `/workspace/repos/` contains any already-indexed repos (`.gitnexus/` present) and register them. No blocking wait — lazy init on first tool call.

**Volume:** `gitnexus_workspace:/workspace/repos/` — shared with indexer so cloned repos are accessible to both.

---

### indexer-agent (port 8011 — rewritten)

Keeps all five assignment-required tool names. The 6-pass pipeline and `Neo4jWriter` are removed.

**`index_repository(repo_url, ref="", repo_name="")`**
1. `git clone --depth 1 <repo_url> /workspace/repos/<repo_name>/`
2. Calls `gitnexus-agent.analyze_repo(path, repo_name)` via MCP
3. Returns `{job_id, status: "started"}` immediately; background task updates `_jobs`

**`get_index_status(job_id="")`** — unchanged interface, now backed by `_jobs` dict

**`index_file`, `parse_python_ast`, `extract_entities`** — kept for assignment compliance; use Python `ast` stdlib, return same shape as before (no Neo4j write)

**Removed:** `src/indexer/pipeline.py`, `src/indexer/passes/`, `src/indexer/neo4j_writer.py`, `src/indexer/ast_parser.py` (the 6-pass indexer code)

---

### graph-query-agent (port 8012 — rewritten)

Keeps all six assignment-required tool names. All Cypher-against-Neo4j replaced with calls to gitnexus-agent.

**Tool mapping:**

| Assignment tool | GitNexus backing |
|---|---|
| `find_entity(name, entity_type, repo_id)` | `gitnexus-agent.query(name, repo_id)` — top result |
| `get_dependencies(entity, repo_id)` | `gitnexus-agent.context(entity, repo_id)` → extract outgoing |
| `get_dependents(entity, repo_id)` | `gitnexus-agent.context(entity, repo_id)` → extract incoming |
| `trace_imports(module, repo_id)` | `gitnexus-agent.cypher("MATCH path=(m {name:$n})-[:IMPORTS*1..5]->(t)...", repo_id)` |
| `find_related(entity, rel_type, repo_id)` | `gitnexus-agent.cypher("MATCH (n {name:$n})-[r:<rel>]->(t)...", repo_id)` |
| `execute_query(cypher)` | `gitnexus-agent.cypher(cypher)` — safety enforced by LadybugDB (read-only) |

**Removed:** `src/graph_query/queries.py`, `src/graph_query/traversal.py`, `src/graph_query/safety.py`

---

### code-analyst-agent (port 8013 — updated)

Tool names and LLM analysis logic unchanged. Source file lookup migrated from Neo4j to gitnexus-agent.

**Change:** Replace `Neo4jClient` import and usage with calls to `gitnexus-agent.context(symbol)` to get `file_path` and `start_line`. Then read source from `/workspace/repos/<repo>/` shared volume.

**Streaming:** `explain_implementation`, `analyze_function`, `analyze_class` stream LLM tokens using FastMCP async generator:
```python
@mcp.tool()
async def explain_implementation(entity_name: str):
    # locate via gitnexus-agent
    # stream LLM response
    async for chunk in llm.astream([...]):
        yield chunk.content
```

**Model override:** Each tool accepts optional `model: str = ""` and passes to `ChatOpenRouter(model=model or settings.OPENROUTER_MODEL)`.

---

### orchestrator-agent (port 8010 — rewritten)

**ReAct loop** replaces the classify → plan → dispatch → synthesize pipeline.

**LangGraph graph:**
```
START → inject_history → react_agent ←──┐
                              │          │
                    tool_call? ──yes──► tool_executor
                              │                │
                           no/done          result back
                              │
                           END → stream final answer
```

**`inject_history` node:**
- Calls `memory-agent.get_conversation_context(session_id)` 
- Gets last 10 turns as `HumanMessage`/`AIMessage` pairs
- Turns older than 10: replaced with a single summary message (LLM-generated, stored in memory)
- Injects into state messages before ReAct starts

**`react_agent` node:**
- LLM receives: system prompt + full history + available tools + current user message
- Uses LangChain `create_react_agent` or manual `bind_tools` + loop
- Model: `model or settings.OPENROUTER_MODEL`
- Loops until LLM emits a final answer (no tool call)

**System prompt** (replaces all classification prompts):
> You are a code analysis assistant for software repositories indexed with GitNexus. You have access to tools for searching code, understanding relationships, analysing implementations, and retrieving conversation history. Use as many tool calls as needed to give a complete, accurate answer. Always check context before answering relationship or lifecycle questions.

**Tool catalogue visible to ReAct LLM:**
- All `graph-query-agent` tools (find_entity, get_dependencies, etc.)
- All `code-analyst-agent` tools (explain_implementation, get_code_snippet, etc.)
- `memory-agent.get_conversation_context`

**Streaming:** `graph.astream_events()` — gateway SSE endpoint forwards each event type:
- `on_chat_model_stream` → `{"type": "token", "content": "..."}`
- `on_tool_start` → `{"type": "tool_call", "tool": "...", "args": {...}}`
- `on_tool_end` → `{"type": "tool_result", "tool": "...", "summary": "..."}`

**After loop:** Store turn in memory: `memory-agent.store_interaction(session_id, user_msg, final_answer, tools_used)`

**Removed:** `classify_query`, `rewrite_query`, `plan_tools`, `dispatch_agents` nodes; `QUERY_CLASSIFICATION_PROMPT`, `QUERY_REWRITE_PROMPT`; `_LIFECYCLE_KEYWORDS`, `_LIST_TRIGGERS`, `_LIST_ENTITY_MAP`, `_GRAPH_QUERY_TOOLS`, `_CODE_ANALYST_TOOLS` frozen sets; `_summarise_agent_result` (no longer needed — ReAct LLM reads raw tool results)

**External MCP access:** Orchestrator remains a FastMCP server at `http://localhost:8010/mcp`. `route_to_agents` and `analyze_query` tools stay, now backed by the ReAct loop. Claude Desktop, Cursor, or any MCP client can connect.

---

### memory-agent (port 8014 — unchanged)

No changes. Redis for session cache, Neo4j for long-term memory storage. Memory is the only remaining reason Neo4j stays in docker-compose.

---

## UI Changes (src/ui/app.py)

### Model selector (sidebar)
```python
OPENROUTER_MODELS = [
    ("Server default",                    ""),
    ("Nemotron 3 Super 120B (free)",      "nvidia/nemotron-3-super-120b-a12b:free"),
    ("Qwen 2.5 72B Instruct (free)",      "qwen/qwen-2.5-72b-instruct:free"),
    ("Llama 3.3 70B Instruct (free)",     "meta-llama/llama-3.3-70b-instruct:free"),
    ("DeepSeek Chat v3 (free)",           "deepseek/deepseek-chat:free"),
    ("Gemini 2.0 Flash Exp (free)",       "google/gemini-2.0-flash-exp:free"),
    ("Other…",                            "__custom__"),
]
```
Stored in `st.session_state.selected_model`. Passed as `model` in chat request body.

### Multi-repo support
- Indexing form gains `repo_name` field (defaults to last path segment of URL)
- Repo selector shows all indexed repos by name (from gitnexus registry via `/api/graph/repos` endpoint)
- `repo_id=""` in chat request = search across all repos
- Sidebar shows per-repo stats from gitnexus

### Streaming chat bubbles
- Replace `api_post("/api/chat")` with SSE fetch
- Each `token` event appends to the current assistant bubble
- Each `tool_call` event shows a small "🔧 calling find_entity..." indicator below the bubble
- `tool_result` events dismissed after 2s (transient)

---

## Gateway Changes

**New endpoint:** `GET /api/graph/repos` in `src/gateway/routes/graph.py` — returns list of indexed repos from gitnexus-agent (`list_repos` tool).

**`POST /api/chat`** — forwards `model` and `repo_id` fields from request to orchestrator.

**SSE forwarding** — existing SSE handler extended to forward structured events (token, tool_call, tool_result, done) not just text chunks.

---

## Data Flow — Example Query

> "How does FastAPI handle dependency injection? Show me examples."

```
1. UI → POST /api/chat { message: "How does FastAPI...", repo_id: "fastapi", model: "" }
2. Gateway → orchestrator.route_to_agents(...)
3. inject_history: retrieves last 3 turns (user was asking about routing before)
4. ReAct LLM thinks:
     → tool_call: graph_query.find_entity("Depends")
     → result: Depends is in fastapi/params.py:27
     → tool_call: graph_query.get_dependents("Depends")
     → result: 47 functions use Depends
     → tool_call: code_analyst.explain_implementation("Depends")
     → streams: "Depends is a marker class that signals FastAPI's..."  ← UI sees tokens live
     → tool_call: code_analyst.get_code_snippet("solve_dependencies")
     → result: source code snippet
     → LLM decides: enough info, compose answer
5. Final answer streamed token by token to UI
6. memory.store_interaction(session_id, user_msg, answer, tools_used=["find_entity","get_dependents","explain_implementation","get_code_snippet"])
```

---

## docker-compose Changes

```yaml
# Added
gitnexus-agent:
  build: docker/Dockerfile.gitnexus-agent
  ports: ["8015:8015"]
  volumes:
    - gitnexus_workspace:/workspace/repos
  healthcheck: curl -f http://localhost:8015/health

# Modified
graph-query:
  depends_on: [gitnexus-agent]   # neo4j dependency removed
  
indexer:
  depends_on: [gitnexus-agent]   # neo4j dependency removed
  volumes:
    - gitnexus_workspace:/workspace/repos

# Kept (memory still needs neo4j)
neo4j: unchanged
memory: unchanged

# New volume
volumes:
  gitnexus_workspace:
```

---

## Files Changed on This Branch

### New
- `docker/Dockerfile.gitnexus-agent` — Python 3.12 + Node.js 20 + gitnexus CLI
- `src/gitnexus_agent/__init__.py` — new top-level module (alongside src/indexer, src/graph_query, etc.)
- `src/gitnexus_agent/server.py` — FastMCP server exposing gitnexus tools
- `src/shared/gitnexus_client.py` — async stdio MCP client wrapper (manages persistent subprocess)

### Rewritten
- `src/indexer/server.py` — delegates to gitnexus-agent; removes 6-pass pipeline
- `src/graph_query/server.py` — maps assignment tools to gitnexus-agent calls
- `src/orchestrator/nodes.py` — ReAct loop; all manual rules removed
- `src/orchestrator/prompts.py` — single system prompt replaces all classification prompts
- `src/orchestrator/state.py` — add `model: str`, `repo_id: str`; remove `tool_plan`, `query_classification`, `agent_plan`, `search_params`
- `src/code_analyst/server.py` — source lookup via gitnexus-agent; streaming added; model override
- `src/ui/app.py` — model selector, multi-repo, streaming SSE chat
- `src/gateway/routes/chat.py` — forward model/repo_id; full SSE event forwarding
- `docker-compose.yml` — add gitnexus-agent service and volume

### Deleted
- `src/indexer/pipeline.py`, `src/indexer/passes/`, `src/indexer/neo4j_writer.py`, `src/indexer/ast_parser.py`
- `src/graph_query/queries.py`, `src/graph_query/traversal.py`, `src/graph_query/safety.py`
- `src/orchestrator/nodes.py` fast-path logic (frozen sets, keyword maps, classify/rewrite nodes)

### Unchanged
- `src/memory/` — all files
- `src/shared/neo4j_client.py`, `src/shared/settings.py`, `src/shared/schemas.py`*
- `docker/Dockerfile.memory`, `docker/Dockerfile.gateway`, `docker/Dockerfile.ui`

`src/shared/schemas.py` — `ChatRequest` gains `model: str | None = None` and `repo_id: str | None = None` (already exists but type clarified)

---

## Verification

1. `git checkout -b feature/gitnexus-backend`
2. `docker compose build && docker compose up -d` — all 8 services healthy
3. `docker compose logs gitnexus-agent` — confirm `gitnexus mcp` subprocess started
4. Index FastAPI: click "Index Repository" in UI → watch status events stream live → "completed"
5. Ask **"What is the FastAPI class?"** — confirm ReAct calls `find_entity` then streams answer; no hardcoded routing
6. Ask **"Explain how dependency injection works and show examples"** — ReAct makes 3-5 tool calls; UI shows tool indicators live; code_analyst streams explanation tokens
7. Ask **"Now how does error handling relate to that?"** — confirm history is injected; LLM knows "that" = dependency injection from previous turn
8. Index a second repo (e.g. Starlette). Ask **"Compare how FastAPI and Starlette handle middleware"** — `repo_id=""` cross-repo query via `group_query`
9. Switch model to Qwen 2.5 72B in sidebar → ask same question → `docker compose logs orchestrator` shows new model id
10. Connect Claude Desktop to `http://localhost:8010/mcp` — confirm `route_to_agents` tool is discoverable and callable
11. `git diff main..feature/gitnexus-backend --stat` — confirm diff scoped to files listed above
