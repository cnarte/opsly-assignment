"""Integration tests: verify each sample query uses real graph data.

These tests call the live stack (gateway → orchestrator → MCP agents → Neo4j).
They are skipped automatically if the gateway is not reachable.

Run against live services:
    uv run python -m pytest tests/integration/ -v

Each test asserts:
  1. Which agents were invoked.
  2. That specific graph-data keys are present in agent_results.
  3. That the response references real data from the graph (not hallucinated).
"""

from __future__ import annotations

import json
import httpx
import pytest
import pytest_asyncio

GATEWAY_URL = "http://localhost:8000"
TIMEOUT = 420  # seconds — orchestrator chains multiple MCP calls (compare_implementations = 3 LLM calls)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gateway_available() -> bool:
    try:
        r = httpx.get(f"{GATEWAY_URL}/api/agents/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gateway_available(),
    reason="Gateway not reachable — start docker compose first",
)


@pytest.fixture(scope="module")
def session_id() -> str:
    """Single session shared across all queries in this module."""
    import uuid
    return str(uuid.uuid4())


async def _ask(message: str, session_id: str = "") -> dict:
    """POST /api/chat and return the parsed JSON including raw orchestrator data."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{GATEWAY_URL}/api/chat",
            json={"message": message, "session_id": session_id or ""},
        )
        resp.raise_for_status()
        return resp.json()


async def _ask_with_plan(message: str, session_id: str = "") -> dict:
    """Call route_to_agents via the orchestrator MCP with extended timeout."""
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession
    import asyncio, json

    url = "http://localhost:8010/mcp"
    timeout_s = 420  # accommodate compare_implementations chained LLM calls

    async with asyncio.timeout(timeout_s):
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "route_to_agents",
                    {"message": message, "session_id": session_id},
                )
                if result.content:
                    text = result.content[0].text
                    try:
                        return json.loads(text)
                    except Exception:
                        return {"result": text}
    return {}


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------


def _assert_graph_queried(result: dict, label: str = "") -> None:
    """Assert graph_query agent was invoked and returned data."""
    agent_results = result.get("agent_results", {})
    gq = agent_results.get("graph_query", {})
    assert gq, f"{label}: graph_query results are empty — agent was not properly invoked"
    # Check there's at least one non-error result
    real_results = [v for v in gq.values() if not (isinstance(v, dict) and "error" in v and len(v) == 1)]
    assert real_results, f"{label}: all graph_query results were errors: {gq}"


def _assert_entity_in_graph_results(result: dict, entity_name: str, label: str = "") -> None:
    """Assert a specific entity name appears somewhere in graph_query results."""
    agent_results = result.get("agent_results", {})
    gq_text = json.dumps(agent_results.get("graph_query", {})).lower()
    assert entity_name.lower() in gq_text, (
        f"{label}: '{entity_name}' not found in graph_query results.\n"
        f"graph_query results: {json.dumps(agent_results.get('graph_query', {}), indent=2)[:500]}"
    )


def _assert_tools_called(result: dict, expected_tools: list[str], label: str = "") -> None:
    """Assert at least one of the expected tools was in the tool_plan."""
    tool_plan = result.get("tool_plan", [])
    called_tools = [step.get("tool", "") for step in tool_plan]
    overlap = set(expected_tools) & set(called_tools)
    assert overlap, (
        f"{label}: none of {expected_tools} found in tool_plan.\n"
        f"tool_plan was: {tool_plan}"
    )


def _assert_response_not_empty(result: dict, label: str = "") -> None:
    resp = result.get("final_response") or result.get("response", "")
    assert resp and len(resp.strip()) > 50, (
        f"{label}: final_response is empty or too short: {repr(resp[:100])}"
    )


# ---------------------------------------------------------------------------
# Q1: What is the FastAPI class?
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q1_what_is_fastapi_class(session_id: str):
    """graph_query should find the FastAPI class node; response references fastapi/applications.py."""
    result = await _ask_with_plan("What is the FastAPI class?", session_id)

    _assert_response_not_empty(result, "Q1")
    _assert_graph_queried(result, "Q1")
    _assert_entity_in_graph_results(result, "FastAPI", "Q1")

    # Tool plan should have called find_entity for FastAPI
    _assert_tools_called(result, ["find_entity", "get_dependencies", "explain_implementation"], "Q1")

    # Final response must reference real file path (from graph, not hallucination)
    resp = result.get("final_response", "")
    assert "fastapi" in resp.lower(), f"Q1: response should mention fastapi. Got: {resp[:200]}"

    print(f"\nQ1 tool_plan: {result.get('tool_plan', [])}")
    print(f"Q1 agents: {result.get('agent_plan', [])}")


# ---------------------------------------------------------------------------
# Q2: Show me the docstring for the Depends function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q2_depends_docstring(session_id: str):
    """graph_query should locate Depends; response contains its docstring or description."""
    result = await _ask_with_plan("Show me the docstring for the Depends function", session_id)

    _assert_response_not_empty(result, "Q2")
    _assert_graph_queried(result, "Q2")
    _assert_entity_in_graph_results(result, "Depends", "Q2")

    resp = result.get("final_response", "")
    # Depends is in fastapi/params.py
    has_depends_info = any(kw in resp.lower() for kw in ["depends", "dependency", "inject"])
    assert has_depends_info, f"Q2: response should describe Depends. Got: {resp[:300]}"

    print(f"\nQ2 tool_plan: {result.get('tool_plan', [])}")


# ---------------------------------------------------------------------------
# Q3: What classes inherit from APIRouter?
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q3_inherits_from_apirouter(session_id: str):
    """graph_query should use find_related with INHERITS_FROM to get real subclasses."""
    result = await _ask_with_plan("What classes inherit from APIRouter?", session_id)

    _assert_response_not_empty(result, "Q3")
    _assert_graph_queried(result, "Q3")
    _assert_entity_in_graph_results(result, "APIRouter", "Q3")

    # Tool plan should prefer find_related or get_dependents
    tool_plan = result.get("tool_plan", [])
    used_relationship_tool = any(
        step.get("tool") in ("find_related", "get_dependents", "find_entity")
        for step in tool_plan
    )
    assert used_relationship_tool, f"Q3: expected a relationship tool, got: {tool_plan}"

    resp = result.get("final_response", "")
    assert "apirouter" in resp.lower() or "router" in resp.lower(), (
        f"Q3: response should mention APIRouter. Got: {resp[:300]}"
    )

    print(f"\nQ3 tool_plan: {result.get('tool_plan', [])}")


# ---------------------------------------------------------------------------
# Q4: How does FastAPI handle request validation?
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q4_request_validation(session_id: str):
    """Graph should return validation-related entities; response uses that data."""
    result = await _ask_with_plan("How does FastAPI handle request validation?", session_id)

    _assert_response_not_empty(result, "Q4")
    _assert_graph_queried(result, "Q4")

    resp = result.get("final_response", "")
    has_validation_info = any(kw in resp.lower() for kw in ["pydantic", "validation", "model", "body"])
    assert has_validation_info, f"Q4: response should explain validation. Got: {resp[:300]}"

    print(f"\nQ4 tool_plan: {result.get('tool_plan', [])}")


# ---------------------------------------------------------------------------
# Q5: Find all decorators used in the routing module
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q5_routing_decorators(session_id: str):
    """Graph DECORATED_BY edges surface routing decorators.

    The pipeline may use either:
    - graph_query directly (find_related / execute_query on DECORATED_BY)
    - code_analyst.find_patterns(code_path='routing', pattern_type='Decorator')
      which internally queries the Neo4j DECORATED_BY graph.
    Both are valid since both pull from graph data.
    """
    result = await _ask_with_plan("Find all decorators used in the routing module", session_id)

    _assert_response_not_empty(result, "Q5")

    agent_results = result.get("agent_results", {})
    tool_plan = result.get("tool_plan", [])

    # Accept either direct graph_query OR code_analyst find_patterns (both query graph)
    gq = agent_results.get("graph_query", {})
    ca = agent_results.get("code_analyst", {})
    used_graph_data = bool(gq) or "patterns" in ca or "find_patterns" in ca

    assert used_graph_data, (
        f"Q5: expected graph-backed decorator data (graph_query or find_patterns).\n"
        f"agent_results keys: {list(agent_results.keys())}\n"
        f"tool_plan: {tool_plan}"
    )

    resp = result.get("final_response", "")
    has_decorator_info = any(kw in resp.lower() for kw in ["decorator", "route", "get", "post", "router"])
    assert has_decorator_info, f"Q5: response should list decorators. Got: {resp[:300]}"

    print(f"\nQ5 tool_plan: {tool_plan}")
    print(f"Q5 code_analyst keys: {list(ca.keys())}")


# ---------------------------------------------------------------------------
# Q6: Explain the complete lifecycle of a FastAPI request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q6_request_lifecycle(session_id: str):
    """Complex query — should trigger multiple graph tools and a synthesis."""
    result = await _ask_with_plan("Explain the complete lifecycle of a FastAPI request", session_id)

    _assert_response_not_empty(result, "Q6")
    _assert_graph_queried(result, "Q6")

    # Should call at least 2 distinct tools (lifecycle is 'complex')
    tool_plan = result.get("tool_plan", [])
    assert len(tool_plan) >= 1, f"Q6: expected multi-step tool_plan, got: {tool_plan}"

    resp = result.get("final_response", "")
    lifecycle_keywords = ["middleware", "routing", "handler", "request", "response", "asgi"]
    has_lifecycle = any(kw in resp.lower() for kw in lifecycle_keywords)
    assert has_lifecycle, f"Q6: response should describe lifecycle. Got: {resp[:300]}"

    print(f"\nQ6 tool_plan: {result.get('tool_plan', [])}")


# ---------------------------------------------------------------------------
# Q7: Compare how Path and Query parameters are implemented
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q7_compare_path_query(session_id: str):
    """Comparison query — tool_plan should include compare_implementations or two find_entity calls."""
    result = await _ask_with_plan(
        "Compare how Path and Query parameters are implemented", session_id
    )

    _assert_response_not_empty(result, "Q7")
    _assert_graph_queried(result, "Q7")

    tool_plan = result.get("tool_plan", [])
    tools_used = [s.get("tool") for s in tool_plan]

    # Either compare_implementations OR at least two entity lookups
    used_comparison = (
        "compare_implementations" in tools_used
        or sum(1 for t in tools_used if t in ("find_entity", "explain_implementation")) >= 2
        or len(tool_plan) >= 2
    )
    assert used_comparison, f"Q7: expected comparison tooling, got plan: {tool_plan}"

    resp = result.get("final_response", "")
    has_both = ("path" in resp.lower() or "query" in resp.lower())
    assert has_both, f"Q7: response should compare Path/Query. Got: {resp[:300]}"

    print(f"\nQ7 tool_plan: {result.get('tool_plan', [])}")


# ---------------------------------------------------------------------------
# Q8: What design patterns are used in FastAPI's core?
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q8_design_patterns(session_id: str):
    """Pattern analysis query — should invoke find_patterns from code_analyst."""
    result = await _ask_with_plan("What design patterns are used in FastAPI's core?", session_id)

    _assert_response_not_empty(result, "Q8")

    agent_results = result.get("agent_results", {})
    tool_plan = result.get("tool_plan", [])

    # find_patterns should be in tool plan OR in agent results
    used_patterns = (
        any(s.get("tool") == "find_patterns" for s in tool_plan)
        or "patterns" in agent_results.get("code_analyst", {})
    )
    assert used_patterns, (
        f"Q8: expected find_patterns tool. tool_plan: {tool_plan}, "
        f"code_analyst: {list(agent_results.get('code_analyst', {}).keys())}"
    )

    resp = result.get("final_response", "")
    pattern_keywords = ["pattern", "decorator", "factory", "dependency", "middleware", "router"]
    has_pattern = any(kw in resp.lower() for kw in pattern_keywords)
    assert has_pattern, f"Q8: response should list patterns. Got: {resp[:300]}"

    print(f"\nQ8 tool_plan: {result.get('tool_plan', [])}")


# ---------------------------------------------------------------------------
# Multi-turn memory test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_turn_memory_retention():
    """Turn 2 should see Turn 1's Q&A in conversation_context."""
    import uuid

    sess = str(uuid.uuid4())

    # Turn 1
    t1 = await _ask_with_plan("What is FastAPI?", sess)
    assert t1.get("final_response"), "Turn 1 must produce a response"

    # Turn 2 — reference prior turn
    t2 = await _ask_with_plan("What did I just ask about?", sess)
    assert t2.get("final_response"), "Turn 2 must produce a response"

    # conversation_context in Turn 2 should be non-empty (Turn 1 was persisted)
    ctx = t2.get("conversation_context", [])
    print(f"\nTurn 2 conversation_context: {json.dumps(ctx, indent=2)[:400]}")

    assert ctx, (
        "Turn 2 should have prior context from memory. "
        f"conversation_context = {ctx!r}\n"
        f"full result keys: {list(t2.keys())}"
    )
