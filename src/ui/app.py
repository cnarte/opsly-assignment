"""
FastAPI Repo Chat Agent — Streamlit UI
Premium dark-themed interface with agent activity tracking and graph visualization.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import httpx
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

import os
API_BASE = os.getenv("GATEWAY_URL", "http://localhost:8000")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:1234")
REQUEST_TIMEOUT = 300

AGENT_COLORS = {
    "orchestrator": "#8B5CF6",
    "indexer": "#F59E0B",
    "graph_query": "#3B82F6",
    "code_analyst": "#10B981",
    "memory": "#EC4899",
}

AGENT_ICONS = {
    "orchestrator": "🎯",
    "indexer": "📦",
    "graph_query": "🔍",
    "code_analyst": "🧠",
    "memory": "💾",
}

LABEL_COLORS = {
    "File": "#6366F1",
    "Module": "#8B5CF6",
    "Class": "#3B82F6",
    "Function": "#10B981",
    "Method": "#14B8A6",
    "Parameter": "#F59E0B",
    "Decorator": "#F97316",
    "Import": "#EF4444",
    "Docstring": "#EC4899",
}

LMSTUDIO_DEFAULT_MODELS = [
    ("── LM Studio (local) ──",                None),
    ("LM Studio: gemma-4-e4b (default)",       "lmstudio:google/gemma-4-e4b"),
    ("LM Studio: nemotron-cascade-2",          "lmstudio:nemotron-cascade-2"),
    ("LM Studio: devstral (23.6B)",            "lmstudio:devstral"),
    ("LM Studio: qwen3:14b",                   "lmstudio:qwen3:14b"),
    ("LM Studio: qwen3:8b",                    "lmstudio:qwen3:8b"),
    ("LM Studio: deepseek-r1:8b",             "lmstudio:deepseek-r1:8b"),
    ("LM Studio: llama3.2 (3B, fast)",        "lmstudio:llama3.2"),
    ("LM Studio: custom…",                    "__lmstudio_custom__"),
]

OPENROUTER_MODELS = [
    # OpenRouter models (paid, requires credits)
    ("── OpenRouter (paid) ──",                 None),
    ("Claude Sonnet 4.6 (fast, paid)",          "anthropic/claude-sonnet-4-6"),
    ("Claude Haiku 4.5 (fastest, paid)",        "anthropic/claude-haiku-4-5-20251001"),
    # OpenRouter models (free, rate-limited)
    ("── OpenRouter (free) ──",                 None),
    ("GPT-OSS 20B (fast, free)",                "openai/gpt-oss-20b:free"),
    ("Nemotron Nano 9B (fastest, free)",        "nvidia/nemotron-nano-9b-v2:free"),
    ("Nemotron Nano 12B (fast, free)",          "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("Nemotron Nano 30B (medium, free)",        "nvidia/nemotron-3-nano-30b-a3b:free"),
    ("Llama 3.3 70B (slow, free)",              "meta-llama/llama-3.3-70b-instruct:free"),
    ("GPT-OSS 120B (slow, free)",               "openai/gpt-oss-120b:free"),
    ("Nemotron 3 120B (very slow, free)",       "nvidia/nemotron-3-super-120b-a12b:free"),
    ("Other…",                                  "__custom__"),
]


def fetch_lmstudio_models() -> list[tuple[str, str | None]]:
    """Call LM Studio /v1/models and return list of (label, lmstudio:id) tuples."""
    try:
        r = httpx.get(f"{LMSTUDIO_BASE_URL}/v1/models", timeout=5)
        models = r.json().get("data", [])
        if not models:
            return []
        return [("── LM Studio (local) ──", None)] + [
            (f"LM Studio: {m['id']}", f"lmstudio:{m['id']}")
            for m in models
        ] + [("LM Studio: custom…", "__lmstudio_custom__")]
    except Exception:
        return []


def _build_available_models() -> list[tuple[str, str | None]]:
    """Combine server-default, LM Studio, and OpenRouter entries."""
    lms = st.session_state.get("lmstudio_models") or LMSTUDIO_DEFAULT_MODELS
    return [("Server default (env OPENROUTER_MODEL)", "")] + lms + OPENROUTER_MODELS


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def api_get(path: str, timeout: int = 30) -> dict[str, Any]:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(path: str, data: dict, timeout: int = REQUEST_TIMEOUT) -> dict[str, Any]:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=data, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_get_repos() -> list[str]:
    """Fetch all indexed repos from /api/graph/repos."""
    try:
        data = api_get("/api/graph/repos")
        return data.get("repos", [])
    except Exception:
        return []


def _repo_id_from_url(url: str) -> str:
    """Derive a short repo identifier from a git URL, e.g. 'fastapi/fastapi'."""
    parts = url.rstrip("/").removesuffix(".git").split("/")
    return f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else (parts[-1] if parts else "local")


# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FastAPI Repo Chat Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Light theme */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Agent badges */
    .agent-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin: 2px 4px;
    }

    /* Status indicators */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .status-healthy { background-color: #10B981; }
    .status-unhealthy { background-color: #EF4444; }

    /* Chat messages */
    .chat-user {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #0F172A;
    }
    .chat-assistant {
        background: #F8F7FF;
        border: 1px solid #DDD6FE;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #0F172A;
    }

    /* Agent activity panel */
    .agent-activity {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
    }

    /* Metric cards */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #8B5CF6;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748B;
        font-weight: 600;
        margin: 16px 0 8px 0;
    }

    /* Graph container */
    .graph-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 8px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "agent_activities" not in st.session_state:
    st.session_state.agent_activities = []
if "last_graph_data" not in st.session_state:
    st.session_state.last_graph_data = None
if "graph_query_text" not in st.session_state:
    st.session_state.graph_query_text = ""
if "graph_is_current" not in st.session_state:
    st.session_state.graph_is_current = True
if "indexing_job" not in st.session_state:
    st.session_state.indexing_job = None
if "indexed_repos" not in st.session_state:
    st.session_state.indexed_repos = []
if "active_repo_id" not in st.session_state:
    st.session_state.active_repo_id = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "available_repos" not in st.session_state:
    st.session_state.available_repos = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🤖 Repo Chat Agent")
    st.caption("MCP Multi-Agent System")

    st.divider()

    # -- Health dashboard --
    st.markdown('<div class="section-header">System Health</div>', unsafe_allow_html=True)

    health = api_get("/api/agents/health")
    if "error" not in health:
        status = health.get("status", "unknown")
        status_color = "#10B981" if status == "ok" else "#F59E0B"
        st.markdown(
            f'<span style="color:{status_color};font-weight:600;">'
            f'{"🟢" if status == "ok" else "🟡"} System: {status.upper()}</span>',
            unsafe_allow_html=True,
        )

        for name, state in health.get("agents", {}).items():
            color = AGENT_COLORS.get(name, "#64748B")
            icon = AGENT_ICONS.get(name, "⚙️")
            dot_class = "status-healthy" if state == "healthy" else "status-unhealthy"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
                f'<span class="status-dot {dot_class}"></span>'
                f'<span style="color:{color};font-weight:500;">{icon} {name}</span>'
                f'<span style="color:#64748B;font-size:0.8rem;margin-left:auto;">{state}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.error(f"API unreachable: {health['error']}")

    st.divider()

    # -- Graph stats --
    st.markdown('<div class="section-header">Knowledge Graph</div>', unsafe_allow_html=True)

    stats = api_get("/api/graph/statistics")
    if "error" not in stats:
        col1, col2 = st.columns(2)
        col1.metric("Nodes", f"{stats.get('nodes', 0):,}")
        col2.metric("Edges", f"{stats.get('relationships', 0):,}")

        labels = stats.get("labels", {})
        if any(v > 0 for v in labels.values()):
            for label, count in sorted(labels.items(), key=lambda x: -x[1]):
                if count > 0:
                    color = LABEL_COLORS.get(label, "#64748B")
                    pct = count / max(stats.get("nodes", 1), 1) * 100
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;padding:2px 0;">'
                        f'<span style="width:10px;height:10px;border-radius:3px;'
                        f'background:{color};display:inline-block;"></span>'
                        f'<span style="color:#CBD5E1;font-size:0.85rem;flex:1;">{label}</span>'
                        f'<span style="color:#94A3B8;font-size:0.8rem;">{count:,}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Graph is empty. Index a repository first.")

    st.divider()

    # -- Model selector --
    st.markdown('<div class="section-header">LLM Model</div>', unsafe_allow_html=True)

    # Refresh button — fetches live model list from LM Studio
    if st.button("↻ Refresh LM Studio models", key="refresh_lms"):
        fetched = fetch_lmstudio_models()
        if fetched:
            st.session_state.lmstudio_models = fetched
            st.success(f"Found {len(fetched) - 2} model(s) from LM Studio")
        else:
            st.session_state.lmstudio_models = None
            st.warning("LM Studio unreachable or no models loaded")

    available = _build_available_models()
    model_labels = [label for label, _ in available]
    current_model_id = st.session_state.selected_model
    current_idx = next(
        (i for i, (_, mid) in enumerate(available) if mid == current_model_id), 0
    )
    picked_label = st.selectbox("Model", model_labels, index=current_idx, key="model_picker")
    picked_id = dict(available)[picked_label]

    # Handle dividers and custom inputs
    if picked_id is None:
        st.session_state.selected_model = ""
    elif picked_id == "__custom__":
        picked_id = st.text_input(
            "Custom OpenRouter model ID",
            value=current_model_id if current_model_id not in ("", "__custom__") else "",
            placeholder="org/model-name:tag",
            key="custom_model_input",
        )
        st.session_state.selected_model = picked_id or ""
    elif picked_id == "__lmstudio_custom__":
        picked_id = st.text_input(
            "Custom LM Studio model ID",
            value=current_model_id.removeprefix("lmstudio:") if current_model_id.startswith("lmstudio:") else "",
            placeholder="model-id (as shown in LM Studio)",
            key="lmstudio_custom_input",
        )
        st.session_state.selected_model = f"lmstudio:{picked_id}" if picked_id else ""
    else:
        st.session_state.selected_model = picked_id or ""

    if st.session_state.selected_model:
        st.caption(f"Active: `{st.session_state.selected_model}`")
    else:
        st.caption("Using server default")
    st.divider()

    # -- Indexing controls --
    st.markdown('<div class="section-header">Repository Indexing</div>', unsafe_allow_html=True)

    with st.form("index_form", clear_on_submit=False):
        repo_url = st.text_input(
            "Repository URL",
            value="https://github.com/fastapi/fastapi.git",
            placeholder="https://github.com/org/repo.git",
        )
        ref = st.text_input("Branch / Tag (optional)", value="", placeholder="main")
        submitted = st.form_submit_button("🚀 Start Indexing", use_container_width=True)

        if submitted:
            with st.spinner("Submitting..."):
                result = api_post("/api/index", {"repo_url": repo_url, "ref": ref})
                if "error" not in result:
                    st.session_state.indexing_job = result.get("job_id")
                    rid = _repo_id_from_url(repo_url)
                    if rid not in st.session_state.indexed_repos:
                        st.session_state.indexed_repos.append(rid)
                    st.session_state.active_repo_id = rid
                    st.success(f"Job started: `{result.get('job_id', 'unknown')[:8]}...`")
                else:
                    st.error(result["error"])

    # Repository scope selector
    if st.session_state.indexed_repos:
        st.session_state.active_repo_id = st.selectbox(
            "Active Repository",
            options=[""] + st.session_state.indexed_repos,
            index=([""] + st.session_state.indexed_repos).index(
                st.session_state.active_repo_id
            ) if st.session_state.active_repo_id in st.session_state.indexed_repos else 0,
            format_func=lambda x: "All repos" if x == "" else x,
        )

    if st.button("🔄 Refresh repos", key="refresh_repos"):
        st.session_state.available_repos = api_get_repos()

    # Index status polling
    if st.session_state.indexing_job:
        job_id = st.session_state.indexing_job
        status_data = api_get(f"/api/index/status/{job_id}")
        job_status = status_data.get("status", "unknown")

        status_colors = {
            "started": "🔵", "cloning": "🔵", "indexing": "🟡",
            "completed": "🟢", "failed": "🔴",
        }
        st.markdown(
            f'{status_colors.get(job_status, "⚪")} '
            f'**Job:** `{job_id[:8]}...` — **{job_status}**'
        )
        if job_status in ("started", "cloning", "indexing"):
            st.button("🔄 Refresh Status", key="refresh_idx")

    st.divider()

    # -- Session controls --
    st.markdown('<div class="section-header">Session</div>', unsafe_allow_html=True)
    st.caption(f"ID: `{st.session_state.session_id}`")
    load_id = st.text_input("Load session ID", placeholder="paste a session ID…", label_visibility="collapsed")
    if st.button("Load", use_container_width=True, disabled=not load_id.strip()):
        sid = load_id.strip()
        history = api_get(f"/api/history/{sid}")
        turns = history.get("turns", [])
        loaded_messages = []
        for turn in turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                loaded_messages.append({"role": role, "content": content})
        st.session_state.messages = loaded_messages
        st.session_state.session_id = sid
        st.session_state.agent_activities = []
        st.session_state.last_graph_data = None
        st.session_state.graph_query_text = ""
        st.session_state.graph_is_current = True
        st.rerun()
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.agent_activities = []
        st.session_state.last_graph_data = None
        st.session_state.graph_query_text = ""
        st.session_state.graph_is_current = True
        st.rerun()


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Two-column layout: chat + agent panel
chat_col, panel_col = st.columns([3, 2])


# -- Chat column --
with chat_col:
    st.markdown("### 💬 Chat with the Codebase")

    # Message history
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])

                # Show agent badges for assistant messages
                if role == "assistant" and msg.get("agents_used"):
                    badges_html = ""
                    for agent in msg["agents_used"]:
                        color = AGENT_COLORS.get(agent, "#64748B")
                        icon = AGENT_ICONS.get(agent, "⚙️")
                        badges_html += (
                            f'<span class="agent-badge" '
                            f'style="background:{color};">'
                            f'{icon} {agent}</span>'
                        )
                    st.markdown(badges_html, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask about the codebase..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show thinking state
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Call API
        activity = {
            "query": prompt,
            "timestamp": time.strftime("%H:%M:%S"),
            "status": "running",
            "agents": [],
            "tool_calls": [],   # list of {tool, args, result, duration_ms}
        }
        st.session_state.agent_activities.insert(0, activity)

        with chat_container:
            import json as _json
            import os as _os

            payload = {
                "message": prompt,
                "session_id": st.session_state.session_id,
                "repo_id": st.session_state.active_repo_id,
                "stream": True,
                "model": st.session_state.selected_model or None,
            }

            gateway_url = _os.getenv("GATEWAY_URL", "http://localhost:8000")

            with st.chat_message("assistant"):
                msg_placeholder = st.empty()
                tool_placeholder = st.empty()
                accumulated = ""
                active_tools: list[str] = []
                _pending_tool: dict | None = None   # tool_call waiting for its result

                try:
                    with httpx.stream(
                        "POST",
                        f"{gateway_url}/api/chat",
                        json=payload,
                        timeout=300,
                    ) as resp:
                        for raw_line in resp.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.strip()
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
                                _pending_tool = {
                                    "tool": tool_name,
                                    "args": event.get("args", ""),
                                    "result": None,
                                    "ts": time.strftime("%H:%M:%S"),
                                }
                                activity["tool_calls"].append(_pending_tool)
                                tool_placeholder.caption(
                                    " · ".join(f"🔧 {t}" for t in active_tools[-3:])
                                )
                            elif etype == "tool_result":
                                # Attach result to the last pending tool call
                                if _pending_tool is not None:
                                    _pending_tool["result"] = event.get("result", "")
                                    _pending_tool = None
                            elif etype == "retry":
                                reason = event.get("reason", "")
                                wait = event.get("wait", 0)
                                attempt = event.get("attempt", 1)
                                label = "Rate limited" if reason == "rate_limit" else "Timeout"
                                tool_placeholder.caption(
                                    f"⏳ {label} — retrying in {wait}s (attempt {attempt}/4)…"
                                )
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

                response = accumulated
                agents_used = result.get("agents_used", [])
                st.session_state.session_id = result.get(
                    "session_id", st.session_state.session_id
                )

                if agents_used:
                    badges_html = ""
                    for agent in agents_used:
                        color = AGENT_COLORS.get(agent, "#64748B")
                        icon = AGENT_ICONS.get(agent, "⚙️")
                        badges_html += (
                            f'<span class="agent-badge" '
                            f'style="background:{color};">'
                            f'{icon} {agent}</span>'
                        )
                    st.markdown(badges_html, unsafe_allow_html=True)

        # Update activity and message history
        activity["status"] = "done"
        activity["agents"] = agents_used
        activity["classification"] = result.get("query_classification", {})

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "agents_used": agents_used,
            "raw_result": result,
        })

        # Check if graph data is present and track freshness
        agent_results = result.get("agent_results", {})
        if "graph_query" in agent_results:
            st.session_state.last_graph_data = agent_results["graph_query"]
            st.session_state.graph_query_text = prompt
            st.session_state.graph_is_current = True
        else:
            # Keep existing graph but mark as stale (from previous query)
            st.session_state.graph_is_current = False

        st.rerun()


# -- Right panel: Agent Activity + Graph Viz --
with panel_col:
    # Tab layout for agent activity and graph
    tab_agents, tab_graph = st.tabs(["🎯 Agent Activity", "🕸️ Graph View"])

    with tab_agents:
        if not st.session_state.agent_activities:
            st.markdown(
                '<div style="text-align:center;padding:40px;color:#64748B;">'
                '<div style="font-size:2rem;margin-bottom:8px;">🎯</div>'
                'Agent activity will appear here<br>when you start chatting.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            for i, act in enumerate(st.session_state.agent_activities[:10]):
                status_icon = "🟢" if act["status"] == "done" else "🔄"
                query_preview = act["query"][:40] + ("…" if len(act["query"]) > 40 else "")
                tool_count = len(act.get("tool_calls", []))
                label = f'{status_icon} {act["timestamp"]} — {query_preview}'
                if tool_count:
                    label += f'  `{tool_count} tools`'
                with st.expander(label, expanded=(i == 0)):
                    # Agent badges
                    if act.get("agents"):
                        st.markdown("**Agents involved:**")
                        for agent in act["agents"]:
                            color = AGENT_COLORS.get(agent, "#64748B")
                            icon = AGENT_ICONS.get(agent, "⚙️")
                            st.markdown(
                                f'<span class="agent-badge" '
                                f'style="background:{color};">'
                                f'{icon} {agent}</span>',
                                unsafe_allow_html=True,
                            )

                    # Classification
                    cls = act.get("classification", {})
                    if cls:
                        col_a, col_b = st.columns(2)
                        col_a.markdown(f"**Intent:** `{cls.get('intent', '—')}`")
                        col_b.markdown(f"**Complexity:** `{cls.get('complexity', '—')}`")
                        entities = cls.get("entities", [])
                        if entities:
                            st.markdown(
                                f"**Entities:** {', '.join(f'`{e}`' for e in entities)}"
                            )

                    # Tool call timeline
                    tool_calls = act.get("tool_calls", [])
                    if tool_calls:
                        st.markdown("---")
                        st.markdown("**Tool calls:**")
                        for j, tc in enumerate(tool_calls):
                            tool_icon = "🔧"
                            result_icon = "✅" if tc.get("result") is not None else "⏳"
                            with st.expander(
                                f"{result_icon} {tool_icon} `{tc['tool']}` — {tc.get('ts', '')}",
                                expanded=False,
                            ):
                                if tc.get("args"):
                                    st.markdown("**Input:**")
                                    try:
                                        import json as _j
                                        args_parsed = _j.loads(tc["args"]) if isinstance(tc["args"], str) else tc["args"]
                                        st.json(args_parsed)
                                    except Exception:
                                        st.code(str(tc["args"])[:400], language=None)
                                if tc.get("result") is not None:
                                    st.markdown("**Output:**")
                                    result_str = str(tc["result"])
                                    try:
                                        import json as _j
                                        result_parsed = _j.loads(result_str) if result_str.startswith(("{", "[")) else None
                                        if result_parsed:
                                            st.json(result_parsed)
                                        else:
                                            st.code(result_str[:600], language=None)
                                    except Exception:
                                        st.code(result_str[:600], language=None)

    with tab_graph:
        graph_data = st.session_state.last_graph_data

        # Show stale indicator if graph is from a previous query
        if graph_data and not st.session_state.graph_is_current:
            query_text = st.session_state.graph_query_text
            truncated = query_text[:50] + "..." if len(query_text) > 50 else query_text
            st.info(f"📌 From previous query: *\"{truncated}\"*")

        def _build_graph(gd: dict) -> tuple[list, list]:
            """Convert graph_query agent results into agraph Node/Edge lists."""
            g_nodes: list[Node] = []
            g_edges: list[Edge] = []
            seen: set[str] = set()

            def _node(name: str, labels: list | None = None, size: int = 18) -> str:
                if name and name not in seen:
                    seen.add(name)
                    kind = (labels or [""])[0]
                    color = LABEL_COLORS.get(kind, "#64748B")
                    g_nodes.append(Node(id=name, label=name, size=size, color=color,
                                        font={"color": "#E2E8F0"}))
                return name

            def _edge(src: str, tgt: str, label: str = "", color: str = "#475569") -> None:
                if src and tgt:
                    g_edges.append(Edge(source=src, target=tgt, label=label, color=color))

            for tool_result in gd.values():
                if not isinstance(tool_result, dict):
                    continue

                # find_entity → {"results": [{"n": {...}}], "count": N}  (no "entity" key)
                if "results" in tool_result and "entity" not in tool_result:
                    for item in tool_result.get("results", []):
                        if not isinstance(item, dict):
                            continue
                        props = item.get("n", item)
                        if isinstance(props, dict):
                            name = props.get("name") or props.get("qualified_name", "")
                            kind = props.get("kind", "")
                            _node(name, [kind.capitalize()] if kind else None, 22)

                # get_dependencies → {"entity": "...", "dependencies": [...]}
                for r in tool_result.get("dependencies", []):
                    src = r.get("source", tool_result.get("entity", ""))
                    tgt = r.get("target", "")
                    _node(src, None, 24)
                    _node(tgt, r.get("target_labels", []), 18)
                    _edge(src, tgt, r.get("rel", ""), "#3B82F6")

                # get_dependents → {"entity": "...", "dependents": [...]}
                for r in tool_result.get("dependents", []):
                    src = r.get("source", "")
                    tgt = r.get("target", tool_result.get("entity", ""))
                    _node(src, r.get("source_labels", []), 18)
                    _node(tgt, None, 24)
                    _edge(src, tgt, r.get("rel", ""), "#10B981")

                # find_related → {"entity": "...", "results": [...], "relationship": "..."}
                if "results" in tool_result and "entity" in tool_result:
                    for r in tool_result.get("results", []):
                        src = r.get("source", tool_result.get("entity", ""))
                        tgt = r.get("target", "")
                        _node(src, None, 24)
                        _node(tgt, r.get("target_labels", []), 18)
                        _edge(src, tgt, r.get("rel", ""), "#8B5CF6")

                # get_symbol_context → {"symbol": "...", "outgoing": [...], "incoming": [...]}
                symbol = tool_result.get("symbol", "")
                if symbol and ("outgoing" in tool_result or "incoming" in tool_result):
                    _node(symbol, None, 28)
                    for r in tool_result.get("outgoing", []):
                        related = r.get("related_name", "")
                        _node(related, r.get("related_labels", []), 18)
                        _edge(symbol, related, r.get("relationship") or r.get("rel", ""), "#3B82F6")
                    for r in tool_result.get("incoming", []):
                        related = r.get("related_name", "")
                        _node(related, r.get("related_labels", []), 18)
                        _edge(related, symbol, r.get("relationship") or r.get("rel", ""), "#10B981")

                # analyze_impact → {"symbol": "...", "levels": {"1": [...], "2": [...]}}
                symbol = tool_result.get("symbol", "")
                levels = tool_result.get("levels", {})
                if symbol and levels:
                    _node(symbol, None, 28)
                    for depth_str, lvl in levels.items():
                        for n_info in lvl:
                            name = n_info.get("name", "")
                            _node(name, n_info.get("labels", []), 18)
                            _edge(name, symbol, f"depth {depth_str}", "#F59E0B")

                # trace_imports → {"import_chains": [{"chain": ["mod_a", "mod_b", ...]}]}
                for chain_rec in tool_result.get("import_chains", []):
                    chain = chain_rec.get("chain", [])
                    for i, mod in enumerate(chain):
                        _node(mod, ["Module"], 18)
                        if i > 0:
                            _edge(chain[i - 1], mod, "IMPORTS", "#EF4444")

                # list_entities_tree → {"entity_type": "...", "total": N, "tree": {"folder": {"count": N, "files": {...}}}}
                tree = tool_result.get("tree")
                entity_type_label = tool_result.get("entity_type", "Entity")
                if isinstance(tree, dict) and tree:
                    root_id = f"[{entity_type_label}s]"
                    _node(root_id, [entity_type_label], 32)
                    for folder, folder_data in sorted(tree.items(), key=lambda x: -x[1]["count"]):
                        display_folder = "(root files)" if folder == "_root" else f"{folder}/"
                        folder_label = "Root" if folder == "_root" else "Folder"
                        folder_id = display_folder
                        _node(folder_id, [folder_label], 24)
                        _edge(root_id, folder_id, str(folder_data["count"]), "#F59E0B")
                        for filename, count in sorted(folder_data["files"].items(), key=lambda x: -x[1])[:10]:
                            file_id = f"{folder}/{filename}"
                            _node(file_id, ["File"], 18)
                            _edge(folder_id, file_id, str(count), "#64748B")

            return g_nodes, g_edges

        if not graph_data:
            st.markdown(
                '<div style="text-align:center;padding:40px;color:#64748B;">'
                '<div style="font-size:2rem;margin-bottom:8px;">🕸️</div>'
                'Graph visualizations appear here when<br>'
                'the graph_query agent finds relationships.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            nodes, edges = _build_graph(graph_data)
            if nodes:
                st.markdown(f"**{len(nodes)} nodes, {len(edges)} edges**")
                config = Config(
                    width=500,
                    height=400,
                    directed=True,
                    physics=True,
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#8B5CF6",
                    collapsible=False,
                    node={"renderLabel": True},
                    link={"renderLabel": True},
                    backgroundColor="#0F172A",
                )
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.info("Graph query returned results but no relationships to visualize yet.")


# ---------------------------------------------------------------------------
# Bottom bar: quick stats
# ---------------------------------------------------------------------------

st.divider()
stats_data = api_get("/api/graph/statistics")
if "error" not in stats_data:
    labels = stats_data.get("labels", {})
    # Icon map for known label types; unlisted types get a generic icon
    _LABEL_ICONS = {
        "Function": "⚡", "Method": "🔧", "Class": "🏗️",
        "File": "📄", "Folder": "📁", "Process": "🔄", "Cluster": "🔗",
        "Import": "📥", "Decorator": "🎨",
    }
    # Only show labels that exist in the index (count > 0)
    active = [(label, cnt) for label, cnt in sorted(labels.items(), key=lambda x: -x[1]) if cnt > 0]
    foot_cols = st.columns(max(len(active), 1))
    for col, (label, cnt) in zip(foot_cols, active):
        icon = _LABEL_ICONS.get(label, "🔵")
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{cnt:,}</div>'
            f'<div class="metric-label">{icon} {label}s</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
