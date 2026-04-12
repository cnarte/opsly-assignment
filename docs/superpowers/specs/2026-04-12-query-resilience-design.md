# Query Resilience & Tree Listing Design

## Goal

Fix two confirmed production failures — recursion-limit crashes on complex queries and LLM context overflow on large entity listings — and add a folder-tree view of functions in both chat and the Graph View panel.

## Architecture

Three independent changes that each stand alone:

1. **Recursion guard** — raise the LangGraph limit, catch the error gracefully, synthesize partial results
2. **Result size cap** — truncate oversized tool results before they enter the LLM context
3. **Tree listing tool** — group flat entity lists by folder/file, render in chat as markdown and in the Graph View as a folder→file→count graph

## Tech Stack

LangGraph `StateGraph`, FastMCP, LadybugDB Cypher via gitnexus-agent, Streamlit `agraph`

---

## Change 1 — Recursion guard

**Root cause:** `graph.compile()` has no `recursion_limit` — LangGraph defaults to 25. Complex queries like "lifecycle of a request" need 30–40 agent/tool cycles.

**Fix in `src/orchestrator/graph.py`:**

```python
return graph.compile({"recursion_limit": 50})
```

**Graceful degradation in `src/orchestrator/server.py` `route_to_agents`:**

When `ainvoke` raises `GraphRecursionError`, collect all `ToolMessage` content from the accumulated state messages, feed them to the LLM with a synthesis prompt, return the answer with a note. The user sees a real answer instead of a crash.

```python
from langgraph.errors import GraphRecursionError

try:
    final_state = await _graph.ainvoke(state)
except GraphRecursionError:
    # Collect partial tool results from message history
    partial = _collect_partial_results(state["messages"])
    summary = await _synthesize_partial(partial, message, model)
    return {
        "final_response": summary + "\n\n*(Based on partial exploration — ask a narrower question for more detail.)*",
        "session_id": session_id,
        "agent_results": {},
        "tool_plan": [],
    }
```

`_collect_partial_results(messages)` walks the message list, extracts the `content` of every `ToolMessage`, and joins them. `_synthesize_partial(content, query, model)` calls `_get_llm(model).ainvoke(...)` with a synthesis prompt.

**Streaming endpoint** (`_stream_app` `/stream`) gets the same try/except — on `GraphRecursionError` emit a `{"type": "partial", "content": summary}` SSE event instead of `{"type": "error"}`.

---

## Change 2 — Result size cap

**Root cause:** `list_entities` returns up to 200 results; for "all functions" that's 4,406 names compressed into one `ToolMessage`. The LLM then tries to fit all of them into a completion → 524 timeout from OpenRouter.

**Fix in `src/orchestrator/nodes.py`:**

Add a `_truncate_tool_result(result: str, max_chars: int = 3000) -> str` helper. Wrap the result of every MCP tool call in `_make_mcp_tool` with this truncation before returning it to the ToolNode. If truncated, append: `"... (truncated, {N} total items — use list_entities_tree for a structured view)"`.

This keeps the LLM context within budget on any tool, not just entity listings.

---

## Change 3 — Tree listing tool

**New tool `list_entities_tree` in `src/graph_query/server.py`:**

Takes the same `entity_type` and `repo_id` arguments as `list_entities`. Calls `_call_gitnexus("cypher", ...)` with the same `n.id STARTS WITH "{prefix}:"` query, but fetches up to 5,000 results (all of them). Groups by `file_path` prefix using Python string splitting on `/`. Returns:

```json
{
  "entity_type": "Function",
  "total": 4406,
  "tree": {
    "fastapi": { "count": 120, "files": { "routing.py": 45, "applications.py": 18, ... } },
    "tests":   { "count": 3800, "files": { "test_routing.py": 42, ... } },
    "docs":    { "count": 486,  "files": { "conf.py": 12, ... } }
  }
}
```

**Chat rendering:** The orchestrator's system prompt (`src/orchestrator/prompts.py`) gets a note: *"For any query asking to list all entities of a type, prefer `list_entities_tree` over `list_entities` — it returns a compact folder summary instead of a flat list."* The LLM naturally renders the tree dict as an indented markdown list.

**Graph View rendering in `src/ui/app.py`:**

The `_build_graph` helper already handles multiple result shapes. Add a branch for `"tree"` key:

- One root node per top-level folder (e.g. `fastapi/`, `tests/`, `docs/`)
- Each folder node connects to file nodes (e.g. `routing.py`)
- File node label shows `routing.py (45)`
- Folder node size scales with count
- No individual function nodes (avoids 4,406-node graph freeze)

---

## Files Changed

| File | Change |
|---|---|
| `src/orchestrator/graph.py` | Add `recursion_limit: 50` to `compile()` |
| `src/orchestrator/server.py` | Catch `GraphRecursionError`, synthesize partial results |
| `src/orchestrator/nodes.py` | Add `_truncate_tool_result()`, apply in `_make_mcp_tool` |
| `src/orchestrator/prompts.py` | Add note to prefer `list_entities_tree` for listing queries |
| `src/graph_query/server.py` | Add `list_entities_tree` tool |
| `src/ui/app.py` | Add `"tree"` branch in `_build_graph` |

No new dependencies. No Dockerfile changes. No docker-compose changes.

## Verification

1. `docker compose build orchestrator graph-query ui && docker compose up -d orchestrator graph-query ui`
2. Ask **"how does a request lifecycle look in fastapi server"** — should get a real answer (possibly with partial-exploration note) instead of a timeout
3. Ask **"get me all the functions"** — should get a folder-tree markdown list, not a flat dump; graph view shows folder→file nodes
4. Ask **"get me all the classes"** — same tree behaviour
5. Inspect orchestrator logs — confirm `recursion_limit=50` is in play and no 524 errors
6. Deliberately trigger partial: ask a very broad query like "explain every single file in fastapi" — should get a partial answer with the note, not a crash
