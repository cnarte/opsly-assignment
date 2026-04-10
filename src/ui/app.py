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
if "indexing_job" not in st.session_state:
    st.session_state.indexing_job = None
if "indexed_repos" not in st.session_state:
    st.session_state.indexed_repos = []
if "active_repo_id" not in st.session_state:
    st.session_state.active_repo_id = ""


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
    st.caption(f"ID: `{st.session_state.session_id[:8]}...`")
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.agent_activities = []
        st.session_state.last_graph_data = None
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
        }
        st.session_state.agent_activities.insert(0, activity)

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("🔄 Agents are analyzing your query..."):
                    result = api_post(
                        "/api/chat",
                        {
                            "message": prompt,
                            "session_id": st.session_state.session_id,
                            "repo_id": st.session_state.active_repo_id,
                        },
                    )

                response = result.get("response", result.get("error", "No response"))
                agents_used = result.get("agents_used", [])
                st.session_state.session_id = result.get(
                    "session_id", st.session_state.session_id
                )

                st.markdown(response)

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

        # Check if graph data is present
        agent_results = result.get("agent_results", {})
        if "graph_query" in agent_results:
            st.session_state.last_graph_data = agent_results["graph_query"]

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
                with st.expander(
                    f'{status_icon} {act["timestamp"]} — {act["query"][:45]}...',
                    expanded=(i == 0),
                ):
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

    with tab_graph:
        graph_data = st.session_state.last_graph_data

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
foot_cols = st.columns(5)
stats_data = api_get("/api/graph/statistics")
if "error" not in stats_data:
    labels = stats_data.get("labels", {})
    metrics = [
        ("Classes", labels.get("Class", 0), "🏗️"),
        ("Functions", labels.get("Function", 0), "⚡"),
        ("Methods", labels.get("Method", 0), "🔧"),
        ("Imports", labels.get("Import", 0), "📥"),
        ("Decorators", labels.get("Decorator", 0), "🎨"),
    ]
    for col, (name, val, icon) in zip(foot_cols, metrics):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val:,}</div>'
            f'<div class="metric-label">{icon} {name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
