"""LLM-powered code analysis using OpenRouter via LangChain."""

from __future__ import annotations

import logging
import os as _os
import json as _json
from typing import Any

from langchain_core.messages import HumanMessage

from src.shared.settings import Settings

logger = logging.getLogger(__name__)


async def _locate_symbol(symbol_name: str, repo_id: str = "") -> dict:
    """Find a symbol's file_path and start_line via gitnexus-agent.context."""
    from mcp.client.streamable_http import streamablehttp_client
    from mcp import ClientSession
    from src.shared.settings import Settings as _Settings

    _settings = _Settings()

    host = "gitnexus-agent" if _os.path.exists("/.dockerenv") else "localhost"
    url = f"http://{host}:{_settings.GITNEXUS_PORT}/mcp"
    args = {"symbol": symbol_name}
    if repo_id:
        args["repo"] = repo_id
    try:
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("context", args)
                if result.content:
                    data = _json.loads(result.content[0].text)
                    return {
                        "file_path": data.get("file_path") or data.get("file") or "",
                        "start_line": data.get("start_line") or data.get("line") or 0,
                        "repo": data.get("repo", ""),
                    }
    except Exception:
        pass
    return {"file_path": "", "start_line": 0}


async def explain_entity(entity_name: str, model: str = "") -> str:
    """Locate a symbol via gitnexus, read source, explain with LLM."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openrouter import ChatOpenRouter
    from langchain_openai import ChatOpenAI
    from src.shared.settings import Settings as _Settings

    _settings = _Settings()

    location = await _locate_symbol(entity_name)
    source_context = ""
    if location["file_path"]:
        workspace = _os.getenv("WORKSPACE_PATH", "/workspace/repos")
        if _os.path.exists(workspace):
            for repo_dir in _os.listdir(workspace):
                candidate = _os.path.join(workspace, repo_dir, location["file_path"])
                if _os.path.exists(candidate):
                    lines = open(candidate).readlines()
                    start = max(0, location["start_line"] - 1)
                    end = min(len(lines), start + 60)
                    source_context = "".join(lines[start:end])
                    break

    prompt = f"Explain the implementation of `{entity_name}` in this codebase."
    if source_context:
        prompt += f"\n\nSource code:\n```python\n{source_context}\n```"

    resolved = model or _settings.OPENROUTER_MODEL
    # Skip invalid model values like "default"
    if resolved in ("", "default", "none"):
        resolved = _settings.OPENROUTER_MODEL
    if resolved.startswith("lmstudio:"):
        lms_model = resolved.removeprefix("lmstudio:")
        llm = ChatOpenAI(
            model=lms_model,
            base_url=_settings.LMSTUDIO_BASE_URL,
            api_key="lm-studio",
            temperature=0,
        )
    else:
        llm = ChatOpenRouter(
            model=resolved,
            openrouter_api_key=_settings.OPENROUTER_API_KEY,
            temperature=0,
        )

    response = await llm.ainvoke(
        [
            SystemMessage(
                content="You are a code analysis expert. Explain clearly and concisely."
            ),
            HumanMessage(content=prompt),
        ]
    )
    return response.content


class CodeAnalyzer:
    """Sends source code + graph context to an LLM for structured analysis."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._llm: Any = None

    # -- LLM setup -----------------------------------------------------------

    def _get_llm(self) -> Any:
        if self._llm is None:
            from langchain_openrouter import ChatOpenRouter
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic

            provider = self._settings.ORCHESTRATOR_PROVIDER
            model = self._settings.ORCHESTRATOR_MODEL
            if provider == "anthropic":
                self._llm = ChatAnthropic(
                    model=model,
                    anthropic_api_key=self._settings.ANTHROPIC_API_KEY,
                    temperature=0,
                )
            elif provider == "lmstudio":
                self._llm = ChatOpenAI(
                    model=model,
                    base_url=self._settings.LMSTUDIO_BASE_URL,
                    api_key="lm-studio",
                    temperature=0,
                )
            else:
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

    def _build_compare_prompt(
        self, source_a: str, source_b: str, name_a: str, name_b: str
    ) -> str:
        return (
            f"Compare the following two Python code entities: '{name_a}' and '{name_b}'.\n"
            f"Highlight similarities, differences, strengths, and weaknesses of each.\n\n"
            f"--- {name_a} ---\n```python\n{source_a}\n```\n\n"
            f"--- {name_b} ---\n```python\n{source_b}\n```"
        )

    async def analyze_function(self, source: str, function_name: str) -> dict[str, Any]:
        """Analyze a function using LLM."""
        prompt = self._build_function_prompt(source, {})
        analysis = await self._invoke_llm(prompt)
        return {
            "function_name": function_name,
            "analysis": analysis,
        }

    async def analyze_class(self, source: str, class_name: str) -> dict[str, Any]:
        """Analyze a class using LLM."""
        prompt = self._build_class_prompt(source, {})
        analysis = await self._invoke_llm(prompt)
        return {
            "class_name": class_name,
            "analysis": analysis,
        }

    async def explain_implementation(
        self, source: str, entity_name: str
    ) -> dict[str, Any]:
        """Generate natural language explanation of code."""
        prompt = self._build_explain_prompt(source, {"name": entity_name})
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
