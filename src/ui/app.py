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

API_BASE = "http://localhost:8000"
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
    /* Dark premium theme overrides */
    .stApp {
        background-color: #0F1117;
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
    .status-healthy { background-color: #10B981; box-shadow: 0 0 6px #10B981; }
    .status-unhealthy { background-color: #EF4444; box-shadow: 0 0 6px #EF4444; }

    /* Chat messages */
    .chat-user {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #F1F5F9;
    }
    .chat-assistant {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        border: 1px solid #312E81;
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #E2E8F0;
    }

    /* Agent activity panel */
    .agent-activity {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
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
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 8px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
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
                    st.success(f"Job started: `{result.get('job_id', 'unknown')[:8]}...`")
                else:
                    st.error(result["error"])

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
                        {"message": prompt, "session_id": st.session_state.session_id},
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
            # Build nodes and edges from graph query results
            nodes: list[Node] = []
            edges: list[Edge] = []
            seen_nodes: set[str] = set()

            def _add_entity(data: dict, prefix: str = "") -> None:
                """Extract entities from graph query result and build graph nodes."""
                if isinstance(data, dict):
                    # Handle search results
                    results_list = data.get("results", data.get("result", []))
                    if isinstance(results_list, list):
                        for item in results_list:
                            if isinstance(item, dict):
                                _add_entity(item)
                        return

                    name = data.get("name", data.get("entity_name", ""))
                    kind = data.get("kind", data.get("type", "Entity"))
                    sid = data.get("symbol_id", name)

                    if name and sid not in seen_nodes:
                        seen_nodes.add(sid)
                        color = LABEL_COLORS.get(kind, "#64748B")
                        size = 25 if kind == "Class" else 18
                        nodes.append(Node(
                            id=sid, label=name, size=size, color=color,
                            font={"color": "#E2E8F0"},
                        ))

                    # Handle relationships
                    for rel_key in ("bases", "inherits_from", "depends_on",
                                    "calls", "imports"):
                        rels = data.get(rel_key, [])
                        if isinstance(rels, list):
                            for rel in rels:
                                rel_name = rel if isinstance(rel, str) else rel.get("name", "")
                                if rel_name:
                                    rel_sid = f"{prefix}:{rel_name}"
                                    if rel_sid not in seen_nodes:
                                        seen_nodes.add(rel_sid)
                                        nodes.append(Node(
                                            id=rel_sid, label=rel_name,
                                            size=18, color="#475569",
                                            font={"color": "#94A3B8"},
                                        ))
                                    edges.append(Edge(
                                        source=sid, target=rel_sid,
                                        label=rel_key, color="#475569",
                                    ))

            for key, value in graph_data.items():
                if isinstance(value, dict):
                    _add_entity(value, prefix=key)
                elif isinstance(value, list):
                    for v in value:
                        _add_entity(v, prefix=key)

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
                st.info("Graph query returned results but no visual relationships to display.")
                st.json(graph_data)


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
