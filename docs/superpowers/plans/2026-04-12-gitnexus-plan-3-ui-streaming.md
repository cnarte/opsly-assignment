# GitNexus Branch — Plan 3: UI & Streaming

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model selector + multi-repo support to the UI; wire end-to-end SSE streaming so every ReAct step (tool calls, LLM tokens, final answer) appears live in the chat bubble; add `/api/graph/repos` gateway endpoint.

**Architecture:** `ChatRequest` gains `model` and keeps `repo_id`. The gateway SSE handler forwards structured events (`token`, `tool_call`, `tool_result`, `done`) from the orchestrator's `astream_events()`. The Streamlit UI replaces the blocking `api_post` with a streaming SSE fetch, renders tool-call badges live, and has a sidebar model dropdown plus multi-repo selector.

**Tech Stack:** Streamlit, `requests` SSE (via `sseclient-py`), FastAPI `StreamingResponse`, LangGraph `.astream_events(version="v2")`

**Prerequisite:** Plan 2 complete — all agents running, demo queries returning correct answers.

---

### Task 1: Update schemas + gateway chat route for model forwarding

**Files:**
- Modify: `src/shared/schemas.py`
- Modify: `src/gateway/routes/chat.py`

- [ ] **Step 1: Add `model` field to `ChatRequest`**

In `src/shared/schemas.py`, update `ChatRequest`:

```python
class ChatRequest(BaseModel):
    """Incoming chat message from a user."""

    message: str
    session_id: str | None = None
    stream: bool = False
    repo_id: str | None = None
    model: str | None = None          # ← add this line
```

- [ ] **Step 2: Forward `model` in `chat.py` HTTP handler**

In `src/gateway/routes/chat.py`, update the `call_orchestrator_tool` call inside `async def chat(...)`:

```python
    result = await call_orchestrator_tool(
        "route_to_agents",
        {
            "message": request.message,
            "session_id": session_id,
            "repo_id": repo_id,
            "model": request.model or "",     # ← add this
        },
        timeout=300,
    )
```

- [ ] **Step 3: Forward `model` in WebSocket handler**

In `src/gateway/routes/chat.py`, in `ws_chat`, add model extraction:

```python
                model = payload.get("model", "")
                # ... then in the call_orchestrator_tool call:
            result = await call_orchestrator_tool(
                "route_to_agents",
                {"message": message, "session_id": session_id, "repo_id": repo_id, "model": model},
            )
```

- [ ] **Step 4: Commit**

```bash
git add src/shared/schemas.py src/gateway/routes/chat.py
git commit -m "feat: forward model field from ChatRequest through to orchestrator"
```

---

### Task 2: Add /api/graph/repos endpoint

**Files:**
- Modify: `src/gateway/routes/graph.py`

- [ ] **Step 1: Add `list_repos` endpoint**

Add to `src/gateway/routes/graph.py` after the existing imports:

```python
@router.get("/api/graph/repos")
async def list_repos():
    """Return all repos indexed in LadybugDB (via gitnexus-agent)."""
    from src.gateway.mcp_client import call_agent_tool
    from src.shared.settings import Settings
    settings = Settings()
    result = await call_agent_tool(settings.GITNEXUS_PORT, "list_repos", {})
    repos = result.get("repos", [])
    # Normalise to list of name strings
    if repos and isinstance(repos[0], dict):
        repos = [r.get("name", str(r)) for r in repos]
    return {"repos": repos}
```

- [ ] **Step 2: Rebuild gateway and verify**

```bash
docker compose build gateway && docker compose up -d gateway
sleep 5
curl -s http://localhost:8000/api/graph/repos | python3 -m json.tool
```

Expected: `{"repos": ["fastapi"]}` (or empty list if not indexed yet)

- [ ] **Step 3: Commit**

```bash
git add src/gateway/routes/graph.py
git commit -m "feat: add GET /api/graph/repos endpoint"
```

---

### Task 3: Wire streaming in gateway chat route

The gateway needs to call the orchestrator graph directly (not via MCP) to stream events, **or** forward the orchestrator's SSE stream. The simplest approach: the gateway calls the orchestrator MCP tool but adds a streaming path that uses the orchestrator's `/stream` endpoint if available. Since the orchestrator is also a FastMCP server, the simplest production approach is to run the graph inside the gateway process or use HTTP streaming.

**Chosen approach:** Extend the orchestrator server with a streaming endpoint, then have the gateway proxy it.

**Files:**
- Modify: `src/orchestrator/server.py` (add streaming tool)
- Modify: `src/gateway/routes/chat.py` (proxy SSE from orchestrator)

- [ ] **Step 1: Add `stream_route_to_agents` to orchestrator server**

Add to `src/orchestrator/server.py` after the existing imports:

```python
from fastapi import FastAPI as _FastAPI
from fastapi.responses import StreamingResponse as _StreamingResponse
import asyncio as _asyncio

# Add a raw FastAPI app alongside the MCP server for streaming
_app = _FastAPI()


@_app.post("/stream")
async def stream_chat(body: dict):
    """Stream ReAct events as Server-Sent Events."""
    message = body.get("message", "")
    session_id = body.get("session_id", "")
    repo_id = body.get("repo_id", "")
    model = body.get("model", "")

    state = _initial_state(message, session_id, repo_id, model)

    async def _event_generator():
        import json as _json
        try:
            async for event in _graph.astream_events(state, version="v2"):
                kind = event.get("event", "")
                data: dict | None = None

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        data = {"type": "token", "content": chunk.content}

                elif kind == "on_tool_start":
                    data = {
                        "type": "tool_call",
                        "tool": event.get("name", ""),
                        "args": _json.dumps(event.get("data", {}).get("input", {}), default=str)[:200],
                    }

                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output", "")
                    data = {
                        "type": "tool_result",
                        "tool": event.get("name", ""),
                        "summary": str(output)[:200],
                    }

                if data:
                    yield f"data: {_json.dumps(data)}\n\n"

            yield f"data: {_json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            import json as _json
            yield f"data: {_json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return _StreamingResponse(_event_generator(), media_type="text/event-stream")
```

Then, in the `if __name__ == "__main__":` block of `src/orchestrator/server.py`, mount the FastAPI app alongside the MCP server. Or simply run both via uvicorn on the same port using the FastMCP's built-in app. Since FastMCP uses Starlette internally, add the route to the underlying ASGI app:

```python
# At module level, after mcp is created:
mcp.app.add_route("/stream", _app.routes[-1].endpoint, methods=["POST"])
```

Actually the simplest approach is to expose streaming through the existing MCP mechanism. Let's use a dedicated HTTP endpoint on a second port. Add to settings:

```python
ORCHESTRATOR_STREAM_PORT: int = 8016
```

Then in `src/orchestrator/server.py` main:

```python
if __name__ == "__main__":
    import uvicorn
    import threading

    def _run_stream():
        uvicorn.run(_app, host="0.0.0.0", port=settings.ORCHESTRATOR_STREAM_PORT, log_level="warning")

    t = threading.Thread(target=_run_stream, daemon=True)
    t.start()
    mcp.run(transport="streamable-http")
```

- [ ] **Step 2: Update gateway to proxy SSE stream from orchestrator**

In `src/gateway/routes/chat.py`, update the `stream=True` branch in `async def chat(...)`:

```python
    if request.stream:
        import httpx
        import os

        orch_host = "orchestrator" if os.path.exists("/.dockerenv") else "localhost"
        stream_url = f"http://{orch_host}:{settings.ORCHESTRATOR_STREAM_PORT}/stream"

        async def _proxy_sse():
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST", stream_url,
                    json={
                        "message": request.message,
                        "session_id": session_id,
                        "repo_id": repo_id,
                        "model": request.model or "",
                    }
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"{line}\n\n"

        return StreamingResponse(_proxy_sse(), media_type="text/event-stream")
```

Add `httpx` to `pyproject.toml` dependencies if not present:

```bash
grep "httpx" pyproject.toml || echo "Need to add httpx"
```

If missing, add `"httpx>=0.27"` to the dependencies list in `pyproject.toml`.

- [ ] **Step 3: Add `ORCHESTRATOR_STREAM_PORT` to Settings**

In `src/shared/settings.py`:

```python
    ORCHESTRATOR_STREAM_PORT: int = 8016
```

- [ ] **Step 4: Expose port 8016 in docker-compose orchestrator service**

```yaml
  orchestrator:
    ports:
      - "8010:8010"
      - "8016:8016"    # ← add streaming port
```

- [ ] **Step 5: Rebuild and test streaming**

```bash
docker compose build orchestrator gateway && docker compose up -d orchestrator gateway
sleep 10
curl -s -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is FastAPI?","stream":true,"repo_id":"fastapi"}' | head -20
```

Expected: SSE lines like:
```
data: {"type": "tool_call", "tool": "find_entity", "args": "..."}
data: {"type": "token", "content": "FastAPI is"}
data: {"type": "token", "content": " a modern"}
...
data: {"type": "done"}
```

- [ ] **Step 6: Commit**

```bash
git add src/orchestrator/server.py src/gateway/routes/chat.py src/shared/settings.py docker-compose.yml pyproject.toml
git commit -m "feat: add SSE streaming — orchestrator streams ReAct events, gateway proxies to UI"
```

---

### Task 4: Update Streamlit UI

**Files:**
- Modify: `src/ui/app.py`

- [ ] **Step 1: Add `sseclient-py` dependency**

```bash
grep "sseclient" pyproject.toml || echo "need to add"
```

Add `"sseclient-py>=1.8"` to `pyproject.toml` dependencies.

- [ ] **Step 2: Add model list constant near top of app.py (after LABEL_COLORS ~line 39)**

```python
OPENROUTER_MODELS = [
    ("Server default",                         ""),
    ("Nemotron 3 Super 120B (free)",           "nvidia/nemotron-3-super-120b-a12b:free"),
    ("Qwen 2.5 72B Instruct (free)",           "qwen/qwen-2.5-72b-instruct:free"),
    ("Llama 3.3 70B Instruct (free)",          "meta-llama/llama-3.3-70b-instruct:free"),
    ("DeepSeek Chat v3 (free)",                "deepseek/deepseek-chat:free"),
    ("Gemini 2.0 Flash Exp (free)",            "google/gemini-2.0-flash-exp:free"),
    ("Mistral 7B Instruct (free)",             "mistralai/mistral-7b-instruct:free"),
    ("Other…",                                 "__custom__"),
]
```

- [ ] **Step 3: Add session state keys (after existing session_state init block ~line 213)**

```python
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "available_repos" not in st.session_state:
    st.session_state.available_repos = []
```

- [ ] **Step 4: Add model selector to sidebar (insert before Repository Indexing section ~line 289)**

```python
    # -- Model selector --
    st.markdown('<div class="section-header">LLM Model</div>', unsafe_allow_html=True)
    model_labels = [label for label, _ in OPENROUTER_MODELS]
    current_model_id = st.session_state.selected_model
    current_idx = next(
        (i for i, (_, mid) in enumerate(OPENROUTER_MODELS) if mid == current_model_id), 0
    )
    picked_label = st.selectbox("OpenRouter model", model_labels, index=current_idx, key="model_picker")
    picked_id = dict(OPENROUTER_MODELS)[picked_label]
    if picked_id == "__custom__":
        picked_id = st.text_input(
            "Custom model ID",
            value=current_model_id if current_model_id not in ("", "__custom__") else "",
            placeholder="org/model-name:tag",
            key="custom_model_input",
        )
    st.session_state.selected_model = picked_id or ""
    if st.session_state.selected_model:
        st.caption(f"Active: `{st.session_state.selected_model}`")
    else:
        st.caption("Using server default")

    st.divider()
```

- [ ] **Step 5: Add repo refresh helper and multi-repo selector**

Add a helper function near other API helpers (~line 70):

```python
def api_get_repos() -> list[str]:
    """Fetch all indexed repos from /api/graph/repos."""
    try:
        data = api_get("/api/graph/repos")
        return data.get("repos", [])
    except Exception:
        return []
```

In the sidebar, after the existing repo selector (`active_repo_id` selectbox), add a refresh button and pull the repos list from the API:

```python
    # Refresh repos from gitnexus
    if st.button("🔄 Refresh repos", key="refresh_repos"):
        st.session_state.available_repos = api_get_repos()

    all_repos = list(set(st.session_state.available_repos + st.session_state.indexed_repos))
    if all_repos:
        st.session_state.active_repo_id = st.selectbox(
            "Active Repository",
            options=[""] + all_repos,
            index=([""] + all_repos).index(st.session_state.active_repo_id)
            if st.session_state.active_repo_id in all_repos else 0,
            format_func=lambda x: "All repos" if x == "" else x,
        )
```

- [ ] **Step 6: Replace blocking chat call with streaming SSE fetch**

Find the section in the main chat loop where `api_post("/api/chat", ...)` is called (~line 414) and replace it with an SSE-based streaming version:

```python
                import json as _json
                import requests as _requests

                payload = {
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "repo_id": st.session_state.active_repo_id,
                    "stream": True,
                    "model": st.session_state.selected_model or None,
                }

                gateway_url = os.getenv("GATEWAY_URL", "http://localhost:8000")
                
                # Streaming response container
                with st.chat_message("assistant"):
                    msg_placeholder = st.empty()
                    tool_placeholder = st.empty()
                    accumulated = ""
                    active_tools: list[str] = []

                    try:
                        with _requests.post(
                            f"{gateway_url}/api/chat",
                            json=payload,
                            stream=True,
                            timeout=300,
                        ) as resp:
                            for raw_line in resp.iter_lines():
                                if not raw_line:
                                    continue
                                line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
                                if not line.startswith("data:"):
                                    continue
                                try:
                                    event = _json.loads(line[5:].strip())
                                except _json.JSONDecodeError:
                                    continue

                                etype = event.get("type", "")
                                if etype == "token":
                                    accumulated += event.get("content", "")
                                    msg_placeholder.markdown(accumulated + "▌")
                                elif etype == "tool_call":
                                    tool_name = event.get("tool", "")
                                    active_tools.append(tool_name)
                                    tool_placeholder.caption(
                                        " · ".join(f"🔧 {t}" for t in active_tools[-3:])
                                    )
                                elif etype == "tool_result":
                                    pass  # badge fades naturally
                                elif etype == "done":
                                    break
                                elif etype == "error":
                                    accumulated = f"Error: {event.get('error', 'unknown')}"
                                    break

                        msg_placeholder.markdown(accumulated)
                        tool_placeholder.empty()
                        result = {"response": accumulated, "session_id": st.session_state.session_id}

                    except Exception as exc:
                        accumulated = f"Connection error: {exc}"
                        msg_placeholder.markdown(accumulated)
                        result = {"response": accumulated}
```

- [ ] **Step 7: Make sure `import os` is at the top of app.py**

```bash
grep "^import os" src/ui/app.py || echo "Need to add 'import os'"
```

Add `import os` if missing.

- [ ] **Step 8: Rebuild UI and verify**

```bash
docker compose build ui && docker compose up -d ui
sleep 10
```

Open `http://localhost:8501` in a browser. Confirm:
- Sidebar shows "LLM Model" section with dropdown
- Sidebar shows "🔄 Refresh repos" button
- Send a message — chat bubble updates progressively with tokens
- Tool badges appear under the bubble while agent is working

- [ ] **Step 9: Commit**

```bash
git add src/ui/app.py pyproject.toml
git commit -m "feat: UI streaming chat, model selector, multi-repo support"
```

---

### Task 5: Final integration test

- [ ] **Step 1: Full rebuild**

```bash
docker compose build && docker compose up -d
sleep 20
docker compose ps
```

Expected: all services `Up`.

- [ ] **Step 2: Health check**

```bash
curl -s http://localhost:8000/api/agents/health | python3 -m json.tool
```

Expected: `"status": "ok"`, all agents `"healthy"`.

- [ ] **Step 3: Index FastAPI + Starlette repos**

```bash
curl -s -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/fastapi/fastapi.git","repo_name":"fastapi"}' \
  | python3 -m json.tool

curl -s -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/encode/starlette.git","repo_name":"starlette"}' \
  | python3 -m json.tool
```

- [ ] **Step 4: Verify repos are listed**

```bash
curl -s http://localhost:8000/api/graph/repos | python3 -m json.tool
```

Expected: `{"repos": ["fastapi", "starlette"]}`

- [ ] **Step 5: Multi-repo cross-query**

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Compare how FastAPI and Starlette handle middleware","repo_id":""}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','')[:500])"
```

Expected: answer that references both repos.

- [ ] **Step 6: Model selector end-to-end**

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the FastAPI class?","model":"deepseek/deepseek-chat:free","repo_id":"fastapi"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','')[:300])"

docker compose logs orchestrator --tail 5 | grep -i "deepseek\|model"
```

Expected: response returned; orchestrator logs reference deepseek model.

- [ ] **Step 7: Streaming SSE test**

```bash
curl -s -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What classes inherit from APIRouter?","stream":true,"repo_id":"fastapi"}' \
  | head -30
```

Expected: SSE event lines including `tool_call` and `token` events before `done`.

- [ ] **Step 8: Connect an external MCP client**

Configure Claude Desktop or any MCP client to connect to `http://localhost:8010/mcp`. Verify `route_to_agents` appears in the tool list and can be called with a message.

- [ ] **Step 9: Final commit**

```bash
git add -A
git commit -m "feat: Plan 3 complete — streaming UI, model selector, multi-repo, external MCP access"
```

- [ ] **Step 10: Create a PR summary commit on the branch**

```bash
git log main..HEAD --oneline
```

Verify the diff contains only the files listed in the spec (no unintended changes to memory agent, gateway mcp_client, etc.).

---

## All 3 Plans Complete

The `feature/gitnexus-backend` branch is fully implemented. Key differences from `main`:

| | main | feature/gitnexus-backend |
|---|---|---|
| Graph backend | Neo4j + 6-pass AST indexer | LadybugDB via gitnexus CLI |
| Orchestrator | classify → plan → dispatch (hardcoded rules) | ReAct loop, LLM decides |
| Streaming | Final answer only | Live tokens + tool badges in UI |
| Multi-repo | Partial | Full, cross-repo queries |
| Model selector | .env only | UI dropdown, per-request |
| External MCP | Not accessible | Port 8010 open |
