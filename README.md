# FastAPI Repository Chat Agent — MCP Multi-Agent System

A production-ready multi-agent system that answers questions about the FastAPI codebase using real data from a knowledge graph. Built on the Model Context Protocol (MCP) with five specialised agents coordinated by a LangGraph orchestrator and exposed via a FastAPI gateway.

## Demo

<video src="docs/demo.mp4" controls width="100%" poster="docs/demo-preview.png">
  <a href="docs/demo.mp4">Watch the demo walkthrough (MP4, ~8 MB)</a>
</video>

> Can't see the video? [Download docs/demo.mp4](docs/demo.mp4) or watch it locally.

## Architecture Overview

```
User Query (HTTP / WebSocket)
        │
        ▼
┌─────────────────────┐
│  FastAPI Gateway    │  :8000  — HTTP REST + WebSocket
│  (gateway/)         │
└──────────┬──────────┘
           │ MCP (Streamable HTTP)
           ▼
┌─────────────────────┐
│  Orchestrator Agent │  :8010  — LangGraph StateGraph supervisor
│  (orchestrator/)    │
│                     │
│  check_cache ──────────────────────────────────────────┐
│       │                                                 │
│  classify_query  (LLM generates tool_plan)             │
│       │                                                 │
│  plan_agents                                            │
│       │                                                 │
│  ┌────┴──────────────────────────┐                     │
│  │  memory  │ graph_query │ code │  (MCP fan-out)      │
│  └────┬──────────────────────────┘                     │
│       │                                                 │
│  synthesize                                             │
│       │                                                 │
│  persist_interaction ◄──────────────────────────────────┘
└──────────┬──────────────────────────────┬──────────────┘
           │ MCP                          │ MCP
    ┌──────┴──────┐               ┌───────┴──────┐
    │   Indexer   │ :8011         │  Graph Query │ :8012
    │  (indexer/) │               │(graph_query/)│
    └──────┬──────┘               └──────┬───────┘
           │                             │
           │      ┌──────────────────┐   │
           └──────►    Neo4j :7687   ◄───┘
                  │  (19k+ nodes,    │
                  │   28k+ edges)    │
                  └──────────────────┘

    ┌──────────────────┐         ┌──────────────────────┐
    │  Code Analyst    │ :8013   │   Memory Agent       │ :8014
    │(code_analyst/)   │         │   (memory/)          │
    │  LLM analysis    │         │  Redis (session)     │
    └──────────────────┘         │  Neo4j (long-term)   │
                                 └──────────────────────┘
```

For a visual representation of the knowledge graph schema, see [`code graph visualisation.png`](code%20graph%20visualisation.png).

### Query flow

1. **Gateway** receives `POST /api/chat` and forwards to orchestrator via MCP.
2. **check_cache** — computes SHA-256 key; returns cached response on hit (skips pipeline).
3. **classify_query** — LLM reads the query and the tool catalogue, outputs a `tool_plan` listing the exact tools to call (e.g. `graph_query.find_entity → graph_query.get_dependents → code_analyst.explain_implementation`).
4. **plan_agents** — derives agent list from the plan; always prepends `memory`.
5. **Agent fan-out** — `call_memory`, `call_graph_query`, `call_code_analyst` run sequentially, each dispatching its tool_plan steps. Errors trigger automatic fallbacks (e.g. `get_dependents` → `find_entity`).
6. **synthesize** — LLM combines all agent results and conversation context into a final answer.
7. **persist_interaction** — stores the Q&A to Redis (short-term cache) and Neo4j via the Memory agent.

---

## Quick Start

### Prerequisites

- Docker and Docker Compose v2
- An [OpenRouter](https://openrouter.ai) API key (free-tier models are supported)

### Setup

```bash
# 1. Clone and enter directory
git clone <repo-url> && cd opsly-assignment

# 2. Copy env file and set your API key
cp .env.example .env
# edit .env: set OPENROUTER_API_KEY=sk-or-...

# 3. Start all services
docker compose up --build

# 4. Verify all services are healthy
curl http://localhost:8000/api/agents/health
```

### First Use

```bash
# Index the FastAPI repository (~5 min the first time)
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/fastapi/fastapi.git"}'

# Check indexing progress (use job_id from the response above)
curl http://localhost:8000/api/index/status/<job_id>

# Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What classes inherit from APIRouter?"}'

# Multi-turn: reuse session_id from the previous response
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me the source for one of them", "session_id": "<id>"}'
```

### Running without Docker (development)

```bash
pip install -e ".[dev]"

# Start infrastructure only
docker compose up neo4j redis -d

# Start agents individually (each in its own terminal)
python -m src.indexer.server
python -m src.graph_query.server
python -m src.code_analyst.server
python -m src.memory.server
python -m src.orchestrator.server

# Start gateway
uvicorn src.gateway.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a message; returns `response`, `session_id`, `agents_used`, `tool_plan`, `agent_results`. |
| `POST` | `/api/index` | Trigger repository indexing. Returns `job_id`. |
| `GET` | `/api/index/status/{job_id}` | Poll indexing job progress (backed by Indexer agent). |
| `GET` | `/api/agents/health` | Health check for all downstream agents. |
| `GET` | `/api/graph/statistics` | Knowledge graph node/edge counts. |
| `WS`  | `/ws/chat` | WebSocket streaming chat. |

Full interactive docs: `http://localhost:8000/docs`

### Chat request/response

```json
// POST /api/chat
{
  "message": "What classes inherit from APIRouter?",
  "session_id": ""            // optional; omit to start a new session
}

// 200 OK
{
  "response": "Three classes inherit from APIRouter: ...",
  "session_id": "abc-123",
  "agents_used": ["memory", "graph_query", "code_analyst"],
  "tool_plan": [
    {"agent": "graph_query", "tool": "find_related", "args": {"entity_name": "APIRouter", "relationship_type": "INHERITS_FROM"}},
    {"agent": "code_analyst", "tool": "explain_implementation", "args": {"entity_name": "FastAPI"}}
  ],
  "agent_results": { "graph_query": {...}, "code_analyst": {...} }
}
```

---

## Agent Documentation

### Orchestrator Agent (port 8010)

Central coordinator built on LangGraph `StateGraph`. Classifies queries using an LLM, generates a tool plan from a static catalogue, dispatches to specialist agents, and synthesizes responses.

**MCP Tools exposed to the gateway:**

| Tool | Description |
|------|-------------|
| `route_to_agents` | Full pipeline: classify → plan → fan-out → synthesize. Main entry point. |
| `analyze_query` | Classify query intent only (no agent dispatch). |
| `get_conversation_context` | Retrieve session history via Memory agent. |
| `handle_index_status` | Proxy index job status from the Indexer agent. |
| `synthesize_response` | Standalone synthesis from pre-collected agent results. |

**Key state fields:** `messages`, `query_classification`, `agent_plan`, `tool_plan`, `agent_results`, `conversation_context`, `final_response`, `session_id`, `cache_key`

---

### Indexer Agent (port 8011)

Parses Python repositories and writes a knowledge graph to Neo4j using a 6-pass AST pipeline.

**Indexing passes:**
1. **File discovery** — walk repo, identify `.py` files
2. **Module extraction** — module names, docstrings
3. **Entity extraction** — classes, functions, methods, decorators, parameters
4. **Relationship linking** — IMPORTS, INHERITS_FROM, CONTAINS, DECORATED_BY
5. **Call linking** — CALLS edges resolved via import tables
6. **Derived artifacts** — external stub nodes for third-party bases (e.g. `starlette.routing.Router`)

**MCP Tools:**

| Tool | Description |
|------|-------------|
| `index_repository` | Full repository indexing (clone → parse → write to Neo4j). |
| `index_file` | Single-file indexing (incremental update). |
| `parse_python_ast` | Return raw AST entities from source text without writing. |
| `extract_entities` | Extract classes/functions/imports from source text. |
| `get_index_status` | Report job progress and graph statistics. |

After indexing the FastAPI repo, the graph contains approximately:
- **~19,000 nodes** (Module, Class, Function, Method, Decorator, Import, Parameter)
- **~28,000 relationships** (including 6,000+ CALLS and 600+ INHERITS_FROM)

---

### Graph Query Agent (port 8012)

Executes Cypher queries against the Neo4j knowledge graph. Provides a safe, validated interface — destructive operations are blocked.

**MCP Tools:**

| Tool | Description |
|------|-------------|
| `find_entity` | Look up a class, function, or module by name. Returns node properties, file path, line range, docstring. |
| `get_dependencies` | What does an entity import or call? (outgoing edges) |
| `get_dependents` | What calls or imports this entity? (incoming edges) |
| `trace_imports` | Follow import chains for a module. |
| `find_related` | Find entities connected by a specific relationship type (e.g. INHERITS_FROM, DECORATED_BY). |
| `execute_query` | Run a custom read-only Cypher query (validated against an allowlist). |

---

### Code Analyst Agent (port 8013)

LLM-powered code understanding. All tools call the graph first to retrieve source and context, then pass it to the LLM for analysis.

**MCP Tools:**

| Tool | Description |
|------|-------------|
| `analyze_function` | Deep LLM analysis of a function's logic, args, and return value. |
| `analyze_class` | Comprehensive class analysis including methods and inheritance. |
| `find_patterns` | Detect design patterns (Singleton, Factory, Observer, Strategy, Decorator, DI) via AST + graph. |
| `get_code_snippet` | Extract source code for an entity by name. |
| `explain_implementation` | Plain-English explanation of how an entity is implemented. |
| `compare_implementations` | Side-by-side comparison of two code entities. |

---

### Memory Agent (port 8014)

Manages conversation history and context. Redis handles short-term session caching; Neo4j (via Graphiti) handles long-term episodic memory.

**MCP Tools:**

| Tool | Description |
|------|-------------|
| `get_conversation_context` | Retrieve recent messages for a session. |
| `store_interaction` | Persist a Q&A pair with metadata (agents used, timestamp). |
| `cache_response` | Cache a synthesized response with TTL (default 1 hour). |
| `get_cached_response` | Retrieve cached response for a cache key. |
| `search_memory` | Semantic search over past interactions. |
| `remember_fact` | Store a named fact for a session (user preferences, domain facts). |
| `get_user_preferences` | Retrieve stored preferences for a session. |

---

## Knowledge Graph Schema

**Node labels:**

| Label | Properties |
|-------|------------|
| `Module` | name, path, docstring, indexed_at |
| `Class` | name, path, start_line, end_line, bases, docstring, symbol_id |
| `Function` | name, path, start_line, end_line, args, return_type, docstring, symbol_id |
| `Method` | same as Function + class_name |
| `Decorator` | name, path, line |
| `Import` | name, path, alias, module |
| `Parameter` | name, annotation, default |
| `File` | path, size, indexed_at |

**Relationship types:**

| Type | Meaning |
|------|---------|
| `CONTAINS` | Module/Class contains a Function/Method/Class |
| `IMPORTS` | Module imports another module or symbol |
| `INHERITS_FROM` | Class inherits from another class |
| `CALLS` | Function/Method calls another function |
| `DECORATED_BY` | Function/Class is decorated |
| `HAS_PARAMETER` | Function has a parameter |
| `DOCUMENTED_BY` | Node has an associated Docstring node |
| `DEPENDS_ON` | Derived dependency relationship |

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and fill in secrets.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | API key for LLM access (required) | — |
| `OPENROUTER_MODEL` | LLM model identifier | `nvidia/nemotron-3-super-120b-a12b:free` |
| `NEO4J_URI` | Neo4j Bolt URI | `bolt://neo4j:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379/0` |
| `MCP_CALL_TIMEOUT_S` | Per-MCP-call timeout in seconds | `120` |
| `MCP_CALL_RETRIES` | Retry count on timeout | `1` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENVIRONMENT` | `development` / `testing` / `production` | `development` |

**Environment-specific templates:**
- `.env.development` — debug logging, 120s timeout
- `.env.testing` — warning logging, 10s timeout, 0 retries
- `.env.production` — info logging, 60s timeout, 2 retries (secrets via deployment env)

---

## Testing

```bash
# Unit tests (mocked, no infrastructure required)
uv run pytest tests/unit/ -v

# Integration tests (requires docker compose up + indexed repo)
uv run pytest tests/integration/ -v

# Full suite with coverage report
uv run pytest --cov=src --cov-report=term-missing
```

Coverage gate: `--cov-fail-under=70` (currently ~71%).

---

## Design Decisions and Trade-offs

**MCP over direct function calls.** Each agent is a standalone MCP server communicating via Streamable HTTP. This adds a network hop per agent call but provides complete service isolation, independent deployability, and protocol-level interoperability. Any agent can be replaced or versioned independently.

**LangGraph StateGraph for orchestration.** The supervisor pattern gives explicit control over which agents run and in what order, supports conditional routing (cache hit → skip pipeline), and provides a clean state machine model. The alternative (LangChain `create_react_agent` free tool loop) was considered but rejected for this submission because its non-deterministic tool selection is harder to audit; the current design uses LLM-generated `tool_plan` that is validated against a static catalogue before execution.

**Tool catalogue approach.** The orchestrator embeds a static `TOOL_CATALOGUE` documenting every tool across all agents. The classifier LLM picks from this catalogue to produce a concrete `tool_plan`. Unknown tool names are filtered out before dispatch. This means the LLM guides tool selection but cannot call arbitrary code.

**Neo4j for the knowledge graph.** A labeled property graph maps directly to code structure: nodes are entities (class, function, module), edges are relationships (inherits, calls, imports). Cypher traversals like `MATCH (c:Class)-[:INHERITS_FROM]->(b)` are far more natural than multi-join SQL.

**6-pass AST indexer with cross-file resolution.** A single-pass parser cannot resolve cross-file inheritance (e.g. `class FastAPI(Starlette)` where `Starlette` is in a different file). The resolution pass builds an import table per file and links calls and inheritance to their definition nodes, even across module boundaries.

**Redis for session cache.** Conversation history is stored in Redis (TTL-based). Neo4j via Graphiti handles longer-term memory (facts, preferences). The cache layer in `check_cache` uses a 16-character SHA-256 prefix as the key, giving near-zero collision risk for a single session.

**Safety-constrained Cypher.** The `execute_query` tool validates queries against an allowlist of safe read patterns before execution, preventing destructive writes through the chat interface.

**OpenRouter for LLM access.** Model selection is config-only — no code changes needed to switch between providers or models. Free-tier models (Nemotron, Qwen) work for development. Note: free models are rate-limited and may return provider errors under load (see Known Limitations).

---


## Future Improvements

- **Incremental indexing** — skip files unchanged since last index (commit-sha or file-hash comparison).
- **Deep agent mode** — replace the tool_plan dispatch with a `create_react_agent` loop for fully autonomous multi-step reasoning (prototyped on `feature/skill-based` branch).
- **Multi-repo support** — namespace graph nodes by `repo_id` to answer cross-repository questions.
- **Streaming synthesis** — token-by-token streaming via Server-Sent Events on `POST /api/chat`.
- **Auth layer** — API key or OAuth2 middleware on the gateway.
- **Langfuse tracing** — structured span tracing already integrated; needs `LANGFUSE_*` env vars to enable.

---

## Project Structure

```
src/
  shared/          Settings, Neo4j client, Redis client, logging, exceptions, schemas
  gateway/         FastAPI HTTP/WS gateway, routes, MCP client, middleware
  orchestrator/    LangGraph StateGraph, nodes, prompts, MCP server
  indexer/         6-pass AST pipeline, Neo4j writer, repo manager, MCP server
  graph_query/     Cypher query engine, safety filter, MCP server
  code_analyst/    LLM analyzer, pattern detector, snippet extractor, MCP server
  memory/          Redis session, Graphiti long-term memory, MCP server
docker/            Per-service Dockerfiles (one per agent + gateway)
tests/
  unit/            Unit tests with mocked LLM and graph (~240 tests, ~71% coverage)
  integration/     Live-stack integration tests (skipped if gateway not reachable)
```
