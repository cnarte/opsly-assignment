# FastAPI Repository Chat Agent - MCP Multi-Agent System

A production-ready multi-agent system that answers questions about the FastAPI codebase. Built on the Model Context Protocol (MCP) with specialized agents handling repository indexing, graph querying, code analysis, and conversation memory, all coordinated by a LangGraph-based orchestrator and exposed through a FastAPI gateway.

## Architecture

```
User Query
     |
     v
[FastAPI Gateway :8000]
     |
     v
[Orchestrator Agent :8010]  <-- LangGraph supervisor
     |
     +---> [Indexer Agent :8011]      ----+
     |                                     |
     +---> [Graph Query Agent :8012]  ----+--> [Neo4j]
     |                                     |
     +---> [Code Analyst Agent :8013] ----+
     |
     +---> [Memory Agent :8014]  -----------> [Redis]
     |
     v
[Response Synthesis]
     |
     v
User Response
```

Each agent is an independent MCP server. The Orchestrator connects to them as MCP clients, classifies the user query, routes to the appropriate agents, and synthesizes the final response.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- An OpenRouter API key (free-tier models are supported)

### Setup

1. Clone the repository and copy the environment file:

```bash
git clone <repo-url> && cd opsly-assignment
cp .env.example .env
```

2. Set your OpenRouter API key in `.env`:

```
OPENROUTER_API_KEY=your-key-here
```

3. Start all services:

```bash
docker compose up --build
```

4. Verify health:

```bash
curl http://localhost:8000/api/agents/health
```

### First Use

Index the FastAPI repository, then start chatting:

```bash
# Trigger indexing
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/fastapi/fastapi.git"}'

# Check indexing status (use the job_id from the response above)
curl http://localhost:8000/api/index/status/<job_id>

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the FastAPI class?"}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a message and receive a response. Supports `session_id` for multi-turn conversations. |
| `POST` | `/api/index` | Trigger repository indexing. Returns a `job_id` for tracking. |
| `GET` | `/api/index/status/{job_id}` | Check indexing job progress. |
| `GET` | `/api/agents/health` | Health check for all agents. |
| `GET` | `/api/graph/statistics` | Knowledge graph statistics (node/relationship counts). |
| `WS` | `/ws/chat` | WebSocket endpoint for streaming chat. |

## Agents

### Orchestrator Agent (port 8010)

Central coordinator built on LangGraph. Classifies incoming queries, routes them to the appropriate specialist agents, and synthesizes coherent responses from multiple agent outputs.

**MCP Tools:**
- `analyze_query` -- Classify query intent and extract key entities
- `route_to_agents` -- Determine which agents to invoke
- `get_conversation_context` -- Retrieve relevant conversation history
- `synthesize_response` -- Combine agent outputs into a final answer

### Indexer Agent (port 8011)

Parses Python repositories and populates the Neo4j knowledge graph. Uses a 6-pass AST pipeline to extract modules, classes, functions, decorators, imports, and call relationships.

**MCP Tools:**
- `index_repository` -- Full repository indexing (clones, parses, writes to Neo4j)
- `index_file` -- Single file indexing
- `parse_python_ast` -- Extract AST from Python source code
- `extract_entities` -- Identify code entities and relationships
- `get_index_status` -- Report indexing progress and statistics

### Graph Query Agent (port 8012)

Specializes in Neo4j knowledge graph traversal. Executes Cypher queries to find entities, trace dependencies, and discover relationships between code components.

**MCP Tools:**
- `find_entity` -- Locate a class, function, or module by name
- `get_dependencies` -- Find what an entity depends on
- `get_dependents` -- Find what depends on an entity
- `trace_imports` -- Follow import chains for a module
- `find_related` -- Get entities related by a specified relationship type
- `execute_query` -- Run custom Cypher queries (with safety constraints)

### Code Analyst Agent (port 8013)

Provides deep code understanding powered by LLM analysis. Analyzes function implementations, detects design patterns, extracts code snippets with context, and explains complex logic.

**MCP Tools:**
- `analyze_function` -- Deep analysis of a function's logic
- `analyze_class` -- Comprehensive class analysis
- `find_patterns` -- Detect design patterns in code
- `get_code_snippet` -- Extract code with surrounding context
- `explain_implementation` -- Generate explanation of how code works
- `compare_implementations` -- Compare two code entities

### Memory Agent (port 8014)

Manages conversation history and context using Redis for session caching and Neo4j (via Graphiti) for long-term episodic memory.

**MCP Tools:**
- `store_message` -- Store a conversation message
- `get_history` -- Retrieve conversation history for a session
- `search_memory` -- Search past conversations semantically
- `clear_session` -- Clear a session's conversation history

## Knowledge Graph Schema

**Node types:** Module, Class, Function, Method, Parameter, Decorator, Import, Docstring, File

**Relationship types:** CONTAINS, IMPORTS, INHERITS_FROM, CALLS, DECORATED_BY, HAS_PARAMETER, DOCUMENTED_BY, DEPENDS_ON

## Project Structure

```
src/
  shared/          Shared infrastructure (settings, Neo4j client, Redis, logging, exceptions)
  gateway/         FastAPI HTTP gateway
  orchestrator/    LangGraph-based orchestrator MCP server
  indexer/         Repository parser and Neo4j writer
  graph_query/     Cypher query engine
  code_analyst/    LLM-powered code analysis
  memory/          Session and episodic memory
docker/            Per-service Dockerfiles
tests/             Test suite
```

## Design Decisions and Trade-offs

**MCP over direct function calls.** Each agent is a standalone MCP server communicating via HTTP/SSE. This adds network overhead but provides clear boundaries, independent scaling, and protocol-level interoperability.

**LangGraph for orchestration.** The orchestrator uses LangGraph's StateGraph to model the query-classify-route-synthesize workflow as a directed graph. This gives explicit control over agent invocation order and supports conditional routing based on query classification.

**Neo4j for the knowledge graph.** A property graph database naturally models code relationships (inheritance, imports, calls). Cypher queries make traversal intuitive compared to relational joins.

**Redis for session cache.** Conversation history is stored in Redis with TTL-based expiry. This keeps the hot path fast while Neo4j (via Graphiti) handles long-term memory.

**AST-based indexing with LibCST.** We use LibCST (not the stdlib `ast` module) for concrete syntax tree parsing, which preserves formatting and comments. The 6-pass pipeline (files, modules, classes, functions, relationships, cross-references) ensures all entities are created before relationships are linked.

**OpenRouter for LLM access.** Allows switching models without code changes. Free-tier models (e.g. `qwen/qwen3.6-plus:free`) work for development; production deployments can use stronger models.

**Safety-constrained Cypher.** The `execute_query` tool validates queries against an allowlist of operations, preventing destructive mutations through the chat interface.

## Configuration

All configuration is via environment variables (see `.env.example`). Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | API key for LLM access | (required) |
| `OPENROUTER_MODEL` | Model identifier | `qwen/qwen3.6-plus:free` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://neo4j:7687` |
| `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j credentials | `neo4j` / `password` |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379/0` |
| `LOG_LEVEL` | Logging level | `INFO` |

When running via Docker Compose, service names (`neo4j`, `redis`) are used as hostnames automatically.

## Known Limitations

- **No incremental re-indexing.** The indexer performs full repository scans. Re-indexing the same repo duplicates data unless the graph is cleared first.
- **Single-repo scope.** The system is designed to index one repository at a time. Multi-repo support would require namespace isolation in the graph.
- **No authentication.** The gateway has no auth layer. Add API key or OAuth middleware for production use.
- **In-memory job tracking.** Indexing job status is stored in-process memory. Restarting the indexer loses job history. A Redis-backed job store would improve durability.
- **LLM dependency for code analysis.** The Code Analyst agent requires a working LLM connection. Without it, analysis tools return errors rather than degraded results.
- **No streaming synthesis.** The chat endpoint returns complete responses. WebSocket streaming is defined but the full pipeline does not yet stream tokens end-to-end.

## Sample Queries

**Simple (single agent):**
- "What is the FastAPI class?"
- "Show me the docstring for the Depends function"

**Medium (2-3 agents):**
- "How does FastAPI handle request validation?"
- "What classes inherit from APIRouter?"
- "Find all decorators used in the routing module"

**Complex (multiple agents + synthesis):**
- "Explain the complete lifecycle of a FastAPI request"
- "How does dependency injection work and show me examples from the codebase"
- "Compare how Path and Query parameters are implemented"
- "What design patterns are used in FastAPI's core and why?"

## Development

### Running locally (without Docker)

```bash
# Install dependencies
pip install -e ".[dev]"

# Start Neo4j and Redis
docker compose up neo4j redis -d

# Run individual agents
python -m src.orchestrator.server
python -m src.indexer.server
python -m src.graph_query.server
python -m src.code_analyst.server
python -m src.memory.server

# Run the gateway
uvicorn src.gateway.app:app --host 0.0.0.0 --port 8000 --reload
```

### Running tests

```bash
pytest tests/ -v --cov=src
```

## License

This project was created as an assignment submission.
