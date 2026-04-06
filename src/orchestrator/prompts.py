"""System prompts used by the Orchestrator Agent."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

QUERY_CLASSIFICATION_PROMPT = """\
You are a query classifier for a code-analysis assistant that answers questions
about the FastAPI repository.

Given the user's query, produce a JSON object with exactly three keys:

1. "intent" -- one of:
   code_lookup, relationship_query, code_explanation, pattern_analysis,
   comparison, general

2. "entities" -- a list of entity names extracted from the query (function
   names, class names, module names, decorator names, etc.).  Return an empty
   list if none are found.

3. "complexity" -- one of: simple, medium, complex

Rules:
- code_lookup: the user wants to find where something is defined or used.
- relationship_query: the user asks how entities relate (imports, calls,
  inheritance, dependency).
- code_explanation: the user wants a plain-English explanation of how code
  works.
- pattern_analysis: the user asks about design patterns, architectural
  conventions, or recurring idioms.
- comparison: the user explicitly compares two or more entities.
- general: anything else or general questions about the codebase.

Respond ONLY with valid JSON -- no markdown fences, no commentary.
"""

# ---------------------------------------------------------------------------
# Response synthesis
# ---------------------------------------------------------------------------

RESPONSE_SYNTHESIS_PROMPT = """\
You are synthesising results from multiple code-analysis agents into a
coherent, developer-friendly response about the FastAPI codebase.

You will receive:
- The original user query
- Results from one or more agents (graph_query, code_analyst, indexer, memory)

Guidelines:
- Be concise but thorough.
- Reference specific files, line numbers, and function/class names when
  available.
- If an agent returned an error or empty result, acknowledge it gracefully
  and work with whatever data is available.
- Use markdown formatting (code blocks, bullet lists) where it aids
  readability.
- Do NOT fabricate information that is not present in the agent results.
"""

# ---------------------------------------------------------------------------
# Agent selection rules
# ---------------------------------------------------------------------------

# Maps query intent to the list of agents that should be invoked.
AGENT_SELECTION_MAP: dict[str, list[str]] = {
    "code_lookup": ["graph_query"],
    "relationship_query": ["graph_query"],
    "code_explanation": ["code_analyst", "graph_query"],
    "pattern_analysis": ["code_analyst"],
    "comparison": ["code_analyst", "graph_query"],
    "general": ["code_analyst"],
}
