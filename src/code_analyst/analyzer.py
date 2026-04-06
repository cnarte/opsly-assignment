"""LLM-powered code analysis using OpenRouter via LangChain."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.shared.neo4j_client import Neo4jClient
from src.shared.settings import Settings

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Sends source code + graph context to an LLM for structured analysis."""

    def __init__(self, settings: Settings | None = None, neo4j: Neo4jClient | None = None) -> None:
        self._settings = settings or Settings()
        self._neo4j = neo4j
        self._llm: Any = None

    # -- LLM setup -----------------------------------------------------------

    def _get_llm(self) -> Any:
        if self._llm is None:
            from langchain_openrouter import ChatOpenRouter

            self._llm = ChatOpenRouter(
                model=self._settings.OPENROUTER_MODEL,
                openrouter_api_key=self._settings.OPENROUTER_API_KEY,
            )
        return self._llm

    async def _invoke_llm(self, prompt: str) -> str:
        """Invoke the LLM and return its text response, handling errors."""
        try:
            llm = self._get_llm()
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            if not text or not text.strip():
                return "The LLM returned an empty response."
            return text.strip()
        except TimeoutError:
            logger.warning("LLM call timed out")
            return "Analysis unavailable: LLM request timed out."
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return f"Analysis unavailable: {exc}"

    # -- graph helpers -------------------------------------------------------

    async def _get_function_context(self, function_name: str) -> dict[str, Any]:
        """Fetch graph context for a function from Neo4j."""
        if self._neo4j is None:
            return {}
        try:
            records = await self._neo4j.execute_query(
                """
                MATCH (f:Function {name: $name})
                OPTIONAL MATCH (f)-[:CALLS]->(callee:Function)
                OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
                RETURN f.name AS name,
                       f.path AS path,
                       f.start_line AS start_line,
                       f.end_line AS end_line,
                       f.args AS args,
                       f.return_type AS return_type,
                       f.docstring AS docstring,
                       collect(DISTINCT callee.name) AS callees,
                       collect(DISTINCT caller.name) AS callers
                """,
                {"name": function_name},
            )
            return records[0] if records else {}
        except Exception as exc:
            logger.warning("Graph query failed for function %s: %s", function_name, exc)
            return {}

    async def _get_class_context(self, class_name: str) -> dict[str, Any]:
        """Fetch graph context for a class from Neo4j."""
        if self._neo4j is None:
            return {}
        try:
            records = await self._neo4j.execute_query(
                """
                MATCH (c:Class {name: $name})
                OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Function)
                OPTIONAL MATCH (c)-[:INHERITS]->(base:Class)
                RETURN c.name AS name,
                       c.path AS path,
                       c.start_line AS start_line,
                       c.end_line AS end_line,
                       c.docstring AS docstring,
                       collect(DISTINCT m.name) AS methods,
                       collect(DISTINCT base.name) AS bases
                """,
                {"name": class_name},
            )
            return records[0] if records else {}
        except Exception as exc:
            logger.warning("Graph query failed for class %s: %s", class_name, exc)
            return {}

    async def _get_entity_context(self, entity_name: str) -> dict[str, Any]:
        """Fetch graph context for any entity (function or class)."""
        ctx = await self._get_function_context(entity_name)
        if ctx:
            ctx["kind"] = "function"
            return ctx
        ctx = await self._get_class_context(entity_name)
        if ctx:
            ctx["kind"] = "class"
            return ctx
        return {}

    # -- public analysis methods ---------------------------------------------

    def _build_function_prompt(self, source: str, context: dict[str, Any]) -> str:
        parts = [
            "Analyze the following Python function in detail.",
            "Cover: purpose, algorithm/logic, parameters, return value, side effects, complexity, and potential issues.",
        ]
        if context:
            parts.append(f"\nGraph context:\n- Callers: {context.get('callers', [])}")
            parts.append(f"- Callees: {context.get('callees', [])}")
            parts.append(f"- Parameters: {context.get('args', '')}")
            parts.append(f"- Return type: {context.get('return_type', 'N/A')}")
        parts.append(f"\nSource code:\n```python\n{source}\n```")
        return "\n".join(parts)

    def _build_class_prompt(self, source: str, context: dict[str, Any]) -> str:
        parts = [
            "Analyze the following Python class comprehensively.",
            "Cover: purpose, design pattern usage, methods summary, inheritance, key attributes, and potential improvements.",
        ]
        if context:
            parts.append(f"\nGraph context:\n- Methods: {context.get('methods', [])}")
            parts.append(f"- Base classes: {context.get('bases', [])}")
            parts.append(f"- Docstring: {context.get('docstring', 'N/A')}")
        parts.append(f"\nSource code:\n```python\n{source}\n```")
        return "\n".join(parts)

    def _build_explain_prompt(self, source: str, context: dict[str, Any]) -> str:
        kind = context.get("kind", "code entity")
        name = context.get("name", "unknown")
        return (
            f"Explain how the following Python {kind} '{name}' works in plain language.\n"
            f"Provide a clear, concise explanation suitable for a developer unfamiliar with this codebase.\n\n"
            f"Source code:\n```python\n{source}\n```"
        )

    def _build_compare_prompt(self, source_a: str, source_b: str, name_a: str, name_b: str) -> str:
        return (
            f"Compare the following two Python code entities: '{name_a}' and '{name_b}'.\n"
            f"Highlight similarities, differences, strengths, and weaknesses of each.\n\n"
            f"--- {name_a} ---\n```python\n{source_a}\n```\n\n"
            f"--- {name_b} ---\n```python\n{source_b}\n```"
        )

    async def analyze_function(self, source: str, function_name: str) -> dict[str, Any]:
        """Analyze a function using LLM with graph context."""
        context = await self._get_function_context(function_name)
        prompt = self._build_function_prompt(source, context)
        analysis = await self._invoke_llm(prompt)
        return {
            "function_name": function_name,
            "analysis": analysis,
            "graph_context": context,
        }

    async def analyze_class(self, source: str, class_name: str) -> dict[str, Any]:
        """Analyze a class using LLM with graph context."""
        context = await self._get_class_context(class_name)
        prompt = self._build_class_prompt(source, context)
        analysis = await self._invoke_llm(prompt)
        return {
            "class_name": class_name,
            "analysis": analysis,
            "graph_context": context,
        }

    async def explain_implementation(self, source: str, entity_name: str) -> dict[str, Any]:
        """Generate natural language explanation of code."""
        context = await self._get_entity_context(entity_name)
        prompt = self._build_explain_prompt(source, context)
        explanation = await self._invoke_llm(prompt)
        return {
            "entity_name": entity_name,
            "explanation": explanation,
        }

    async def compare_implementations(
        self, source_a: str, source_b: str, name_a: str, name_b: str
    ) -> dict[str, Any]:
        """Compare two code entities using LLM."""
        prompt = self._build_compare_prompt(source_a, source_b, name_a, name_b)
        comparison = await self._invoke_llm(prompt)
        return {
            "entity_a": name_a,
            "entity_b": name_b,
            "comparison": comparison,
        }
