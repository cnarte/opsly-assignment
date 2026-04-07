"""Assignment coverage test — runs all sample queries from the assignment doc."""
import json
import time
import requests

BASE = "http://localhost:8000"
TIMEOUT = 180  # 3 minutes per chat request

results = []

def test(label, method, path, body=None, checks=None):
    """Run a single test and record result."""
    url = f"{BASE}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=TIMEOUT)
        else:
            r = requests.post(url, json=body, timeout=TIMEOUT)

        data = None
        try:
            data = r.json()
        except Exception:
            pass

        passed = True
        details = []

        if checks:
            for check_name, check_fn in checks.items():
                try:
                    ok = check_fn(r, data)
                    if not ok:
                        passed = False
                        details.append(f"FAIL: {check_name}")
                    else:
                        details.append(f"OK: {check_name}")
                except Exception as e:
                    passed = False
                    details.append(f"FAIL: {check_name} ({e})")

        status = "PASS" if passed else "FAIL"
        results.append((label, status, r.status_code, details, data))

        agents = data.get("agents_used", []) if data else []
        resp_len = len(data.get("response", "")) if data else 0
        print(f"  {'✅' if passed else '❌'} {label}")
        print(f"     HTTP {r.status_code} | Agents: {agents} | Response: {resp_len} chars")
        if details:
            for d in details:
                print(f"     {d}")

    except Exception as e:
        results.append((label, "ERROR", 0, [str(e)], None))
        print(f"  ❌ {label} — ERROR: {e}")


print("=" * 60)
print("FASTAPI REPO CHAT AGENT — ASSIGNMENT COVERAGE TEST")
print("=" * 60)

# ──────────────────────────────────────────────
print("\n📋 Section 1: API Endpoints")
# ──────────────────────────────────────────────

test("GET /api/agents/health", "GET", "/api/agents/health", checks={
    "status_200": lambda r, d: r.status_code == 200,
    "has_status": lambda r, d: "status" in d,
    "has_agents": lambda r, d: "agents" in d,
    "5_agents": lambda r, d: len(d["agents"]) == 5,
    "all_healthy": lambda r, d: all(v == "healthy" for v in d["agents"].values()),
})

test("GET /api/graph/statistics", "GET", "/api/graph/statistics", checks={
    "status_200": lambda r, d: r.status_code == 200,
    "has_nodes": lambda r, d: d["nodes"] > 0,
    "has_relationships": lambda r, d: d["relationships"] > 0,
    "has_labels": lambda r, d: len(d["labels"]) >= 5,
    "has_Class_label": lambda r, d: d["labels"].get("Class", 0) > 0,
    "has_Function_label": lambda r, d: d["labels"].get("Function", 0) > 0,
})

test("POST /api/index", "POST", "/api/index",
    body={"repo_url": "https://github.com/fastapi/fastapi.git", "ref": ""},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_job_id": lambda r, d: "job_id" in d,
        "has_status": lambda r, d: "status" in d,
    })

test("GET /api/index/status/{job_id}", "GET", "/api/index/status/test-123", checks={
    "status_200": lambda r, d: r.status_code == 200,
    "returns_object": lambda r, d: isinstance(d, dict),
})

test("GET /openapi.json", "GET", "/openapi.json", checks={
    "status_200": lambda r, d: r.status_code == 200,
    "has_openapi_version": lambda r, d: "openapi" in d,
    "has_chat_path": lambda r, d: "/api/chat" in d["paths"],
    "has_index_path": lambda r, d: "/api/index" in d["paths"],
    "has_health_path": lambda r, d: "/api/agents/health" in d["paths"],
    "has_graph_path": lambda r, d: "/api/graph/statistics" in d["paths"],
})

test("GET /docs (Swagger)", "GET", "/docs", checks={
    "status_200": lambda r, d: r.status_code == 200,
    "returns_html": lambda r, d: "text/html" in r.headers.get("content-type", ""),
})

# ──────────────────────────────────────────────
print("\n📋 Section 2: Edge Cases")
# ──────────────────────────────────────────────

test("POST /api/chat — missing body field", "POST", "/api/chat",
    body={},
    checks={"status_422": lambda r, d: r.status_code == 422})

test("POST /api/chat — empty message", "POST", "/api/chat",
    body={"message": ""},
    checks={"returns_response": lambda r, d: r.status_code in (200, 422)})

# ──────────────────────────────────────────────
print("\n📋 Section 3: Simple Queries (single agent)")
# ──────────────────────────────────────────────

test("What is the FastAPI class?", "POST", "/api/chat",
    body={"message": "What is the FastAPI class?", "session_id": None},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_response": lambda r, d: len(d.get("response", "")) > 50,
        "has_session_id": lambda r, d: len(d.get("session_id", "")) > 0,
        "mentions_fastapi": lambda r, d: "fastapi" in d["response"].lower(),
    })
time.sleep(2)

test("Show me the docstring for the Depends function", "POST", "/api/chat",
    body={"message": "Show me the docstring for the Depends function", "session_id": None},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_response": lambda r, d: len(d.get("response", "")) > 50,
        "mentions_depends": lambda r, d: "depends" in d["response"].lower(),
    })
time.sleep(2)

# ──────────────────────────────────────────────
print("\n📋 Section 4: Medium Queries (2-3 agents)")
# ──────────────────────────────────────────────

test("How does FastAPI handle request validation?", "POST", "/api/chat",
    body={"message": "How does FastAPI handle request validation?", "session_id": None},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_response": lambda r, d: len(d.get("response", "")) > 50,
        "has_agents_used": lambda r, d: len(d.get("agents_used", [])) >= 1,
    })
time.sleep(2)

test("What classes inherit from APIRouter?", "POST", "/api/chat",
    body={"message": "What classes inherit from APIRouter?", "session_id": None},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_response": lambda r, d: len(d.get("response", "")) > 50,
        "mentions_apirouter": lambda r, d: "apirouter" in d["response"].lower(),
    })
time.sleep(2)

test("Find all decorators used in the routing module", "POST", "/api/chat",
    body={"message": "Find all decorators used in the routing module", "session_id": None},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_response": lambda r, d: len(d.get("response", "")) > 50,
    })
time.sleep(2)

# ──────────────────────────────────────────────
print("\n📋 Section 5: Complex Queries (multi-agent + synthesis)")
# ──────────────────────────────────────────────

test("Explain the complete lifecycle of a FastAPI request", "POST", "/api/chat",
    body={"message": "Explain the complete lifecycle of a FastAPI request", "session_id": None},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_response": lambda r, d: len(d.get("response", "")) > 100,
        "has_agents_used": lambda r, d: len(d.get("agents_used", [])) >= 1,
    })
time.sleep(2)

test("How does dependency injection work?", "POST", "/api/chat",
    body={"message": "How does dependency injection work and show me examples from the codebase", "session_id": None},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_response": lambda r, d: len(d.get("response", "")) > 100,
    })
time.sleep(2)

test("Compare Path vs Query parameters", "POST", "/api/chat",
    body={"message": "Compare how Path and Query parameters are implemented", "session_id": None},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_response": lambda r, d: len(d.get("response", "")) > 100,
        "mentions_path": lambda r, d: "path" in d["response"].lower(),
        "mentions_query": lambda r, d: "query" in d["response"].lower(),
    })
time.sleep(2)

test("Design patterns in FastAPI", "POST", "/api/chat",
    body={"message": "What design patterns are used in FastAPI's core and why?", "session_id": None},
    checks={
        "status_200": lambda r, d: r.status_code == 200,
        "has_response": lambda r, d: len(d.get("response", "")) > 100,
    })
time.sleep(2)

# ──────────────────────────────────────────────
print("\n📋 Section 6: Session & Memory (multi-turn)")
# ──────────────────────────────────────────────

# Turn 1
r1 = requests.post(f"{BASE}/api/chat", json={
    "message": "Tell me about the FastAPI class and its main methods",
    "session_id": None
}, timeout=TIMEOUT)
d1 = r1.json()
sid = d1.get("session_id", "")
print(f"  Turn 1: session={sid[:20]}...")

time.sleep(2)

# Turn 2 — same session
r2 = requests.post(f"{BASE}/api/chat", json={
    "message": "What about its parent class? What does it inherit from?",
    "session_id": sid
}, timeout=TIMEOUT)
d2 = r2.json()
sid2 = d2.get("session_id", "")
session_match = sid == sid2 and len(sid) > 0
print(f"  Turn 2: session={sid2[:20]}... | Match: {'✅' if session_match else '❌'}")

results.append(("Multi-turn session continuity", "PASS" if session_match else "FAIL", 200,
    [f"session preserved: {session_match}"], None))

# ──────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for _, s, _, _, _ in results if s == "PASS")
failed = sum(1 for _, s, _, _, _ in results if s in ("FAIL", "ERROR"))
total = len(results)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 60)

if failed:
    print("\nFailed tests:")
    for label, status, code, details, _ in results:
        if status in ("FAIL", "ERROR"):
            print(f"  ❌ {label}: {details}")
