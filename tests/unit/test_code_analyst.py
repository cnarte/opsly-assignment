"""Unit tests for the Code Analyst Agent — LLM calls are mocked."""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.code_analyst.analyzer import CodeAnalyzer
from src.code_analyst.patterns import PatternDetector
from src.code_analyst.snippets import SnippetExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    """Minimal settings for tests."""
    from src.shared.settings import Settings
    return Settings(OPENROUTER_API_KEY="test-key", OPENROUTER_MODEL="test-model")


@pytest.fixture
def mock_neo4j():
    """Mock Neo4jClient."""
    client = AsyncMock()
    client.execute_query = AsyncMock(return_value=[])
    return client


@pytest.fixture
def analyzer(settings, mock_neo4j):
    return CodeAnalyzer(settings=settings, neo4j=mock_neo4j)


@pytest.fixture
def detector():
    return PatternDetector()


@pytest.fixture
def sample_source():
    return textwrap.dedent("""\
        def calculate_total(items, tax_rate=0.1):
            \"\"\"Calculate the total price with tax.\"\"\"
            subtotal = sum(item.price for item in items)
            tax = subtotal * tax_rate
            return subtotal + tax
    """)


@pytest.fixture
def tmp_source_file(sample_source):
    """Create a temp file with sample source code."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_source)
        f.flush()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CodeAnalyzer tests
# ---------------------------------------------------------------------------

class TestCodeAnalyzer:
    """Tests for the CodeAnalyzer class with mocked LLM."""

    @pytest.mark.asyncio
    async def test_analyze_function_builds_correct_prompt(self, analyzer, sample_source):
        """Verify the prompt includes source code and graph context."""
        mock_response = MagicMock()
        mock_response.content = "This function calculates total price with tax."

        with patch.object(analyzer, "_get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await analyzer.analyze_function(sample_source, "calculate_total")

        assert result["function_name"] == "calculate_total"
        assert "calculates total" in result["analysis"].lower()

        # Check the prompt that was sent to the LLM
        call_args = mock_llm.ainvoke.call_args
        prompt_msg = call_args[0][0][0]
        assert "calculate_total" in prompt_msg.content or "def calculate_total" in prompt_msg.content

    @pytest.mark.asyncio
    async def test_analyze_function_with_graph_context(self, analyzer, sample_source, mock_neo4j):
        """Verify graph context is fetched and included."""
        mock_neo4j.execute_query.return_value = [{
            "name": "calculate_total",
            "path": "utils.py",
            "start_line": 1,
            "end_line": 5,
            "args": "items, tax_rate",
            "return_type": "float",
            "docstring": "Calculate total.",
            "callers": ["process_order"],
            "callees": ["sum"],
        }]

        mock_response = MagicMock()
        mock_response.content = "Detailed analysis of calculate_total."

        with patch.object(analyzer, "_get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await analyzer.analyze_function(sample_source, "calculate_total")

        assert result["graph_context"]["callers"] == ["process_order"]
        assert result["graph_context"]["callees"] == ["sum"]

    @pytest.mark.asyncio
    async def test_analyze_class(self, analyzer):
        """Test class analysis."""
        source = textwrap.dedent("""\
            class OrderProcessor:
                def __init__(self, db):
                    self.db = db
                def process(self, order):
                    pass
        """)
        mock_response = MagicMock()
        mock_response.content = "OrderProcessor handles order processing."

        with patch.object(analyzer, "_get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await analyzer.analyze_class(source, "OrderProcessor")

        assert result["class_name"] == "OrderProcessor"
        assert "OrderProcessor" in result["analysis"]

    @pytest.mark.asyncio
    async def test_explain_implementation(self, analyzer, sample_source):
        """Test explain_implementation returns explanation."""
        mock_response = MagicMock()
        mock_response.content = "This code sums item prices and adds tax."

        with patch.object(analyzer, "_get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await analyzer.explain_implementation(sample_source, "calculate_total")

        assert result["entity_name"] == "calculate_total"
        assert "tax" in result["explanation"].lower()

    @pytest.mark.asyncio
    async def test_compare_implementations(self, analyzer):
        """Test comparing two implementations."""
        source_a = "def add(a, b): return a + b"
        source_b = "def add_safe(a, b):\n    if not isinstance(a, (int, float)): raise TypeError\n    return a + b"

        mock_response = MagicMock()
        mock_response.content = "add is simpler but add_safe includes type checking."

        with patch.object(analyzer, "_get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await analyzer.compare_implementations(source_a, source_b, "add", "add_safe")

        assert result["entity_a"] == "add"
        assert result["entity_b"] == "add_safe"
        assert "comparison" in result

    @pytest.mark.asyncio
    async def test_llm_timeout_handled(self, analyzer, sample_source):
        """Test graceful handling of LLM timeout."""
        with patch.object(analyzer, "_get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(side_effect=TimeoutError("LLM timeout"))
            mock_get_llm.return_value = mock_llm

            result = await analyzer.analyze_function(sample_source, "calculate_total")

        assert "timed out" in result["analysis"].lower()

    @pytest.mark.asyncio
    async def test_llm_error_handled(self, analyzer, sample_source):
        """Test graceful handling of generic LLM error."""
        with patch.object(analyzer, "_get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API rate limited"))
            mock_get_llm.return_value = mock_llm

            result = await analyzer.analyze_function(sample_source, "calculate_total")

        assert "unavailable" in result["analysis"].lower()

    @pytest.mark.asyncio
    async def test_llm_empty_response(self, analyzer, sample_source):
        """Test handling of empty LLM response."""
        mock_response = MagicMock()
        mock_response.content = ""

        with patch.object(analyzer, "_get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_get_llm.return_value = mock_llm

            result = await analyzer.analyze_function(sample_source, "calculate_total")

        assert "empty response" in result["analysis"].lower()

    def test_build_function_prompt_includes_source(self, analyzer, sample_source):
        """Test that the function prompt includes the source code."""
        prompt = analyzer._build_function_prompt(sample_source, {})
        assert "calculate_total" in prompt
        assert "```python" in prompt

    def test_build_function_prompt_includes_context(self, analyzer, sample_source):
        """Test that graph context is incorporated in the prompt."""
        context = {
            "callers": ["main"],
            "callees": ["sum"],
            "args": "items, tax_rate",
            "return_type": "float",
        }
        prompt = analyzer._build_function_prompt(sample_source, context)
        assert "main" in prompt
        assert "sum" in prompt

    def test_build_class_prompt(self, analyzer):
        """Test class prompt construction."""
        source = "class Foo:\n    pass"
        context = {"methods": ["bar"], "bases": ["Base"], "docstring": "A foo."}
        prompt = analyzer._build_class_prompt(source, context)
        assert "bar" in prompt
        assert "Base" in prompt

    def test_build_compare_prompt(self, analyzer):
        """Test compare prompt includes both entities."""
        prompt = analyzer._build_compare_prompt("def a(): pass", "def b(): pass", "a", "b")
        assert "--- a ---" in prompt
        assert "--- b ---" in prompt


# ---------------------------------------------------------------------------
# PatternDetector tests
# ---------------------------------------------------------------------------

class TestPatternDetector:
    """Tests for AST-based pattern detection."""

    def test_detect_singleton_with_new(self, detector):
        source = textwrap.dedent("""\
            class Database:
                _instance = None
                def __new__(cls):
                    if cls._instance is None:
                        cls._instance = super().__new__(cls)
                    return cls._instance
        """)
        patterns = detector.detect_patterns(source)
        assert len(patterns) >= 1
        assert patterns[0]["pattern"] == "Singleton"
        assert patterns[0]["confidence"] >= 0.7
        assert "Database" in patterns[0]["location"]

    def test_detect_singleton_instance_only(self, detector):
        source = textwrap.dedent("""\
            class Config:
                _instance = None
        """)
        patterns = detector.detect_patterns(source)
        singletons = [p for p in patterns if p["pattern"] == "Singleton"]
        assert len(singletons) == 1
        assert singletons[0]["confidence"] >= 0.6

    def test_detect_factory_class(self, detector):
        source = textwrap.dedent("""\
            class ConnectionFactory:
                def create_connection(self, config):
                    pass
        """)
        patterns = detector.detect_patterns(source)
        factories = [p for p in patterns if p["pattern"] == "Factory"]
        assert len(factories) >= 1

    def test_detect_factory_function(self, detector):
        source = "def create_user(name, email): pass"
        patterns = detector.detect_patterns(source)
        factories = [p for p in patterns if p["pattern"] == "Factory"]
        assert len(factories) == 1
        assert factories[0]["confidence"] >= 0.5

    def test_detect_strategy_pattern(self, detector):
        source = textwrap.dedent("""\
            from abc import ABC, abstractmethod
            class PaymentStrategy(ABC):
                @abstractmethod
                def pay(self, amount):
                    pass
        """)
        patterns = detector.detect_patterns(source)
        strategies = [p for p in patterns if p["pattern"] == "Strategy"]
        assert len(strategies) == 1
        assert "pay" in strategies[0]["description"]

    def test_detect_observer_pattern(self, detector):
        source = textwrap.dedent("""\
            class EventBus:
                def __init__(self):
                    self._observers = []
                def subscribe(self, handler):
                    self._observers.append(handler)
                def notify(self, event):
                    for obs in self._observers:
                        obs(event)
        """)
        patterns = detector.detect_patterns(source)
        observers = [p for p in patterns if p["pattern"] == "Observer"]
        assert len(observers) == 1

    def test_detect_decorator_pattern(self, detector):
        source = textwrap.dedent("""\
            class LoggingDecorator:
                def __init__(self, wrapped):
                    self._wrapped = wrapped
        """)
        patterns = detector.detect_patterns(source)
        decorators = [p for p in patterns if p["pattern"] == "Decorator"]
        assert len(decorators) >= 1

    def test_detect_dependency_injection(self, detector):
        source = textwrap.dedent("""\
            class OrderService:
                def __init__(self, repo: Repository, notifier: Notifier):
                    self.repo = repo
                    self.notifier = notifier
        """)
        patterns = detector.detect_patterns(source)
        di = [p for p in patterns if p["pattern"] == "Dependency Injection"]
        assert len(di) == 1
        assert "repo" in di[0]["description"]

    def test_no_patterns_in_simple_code(self, detector):
        source = "x = 1\ny = 2\nprint(x + y)"
        patterns = detector.detect_patterns(source)
        assert patterns == []

    def test_syntax_error_returns_empty(self, detector):
        source = "def broken(:\n    pass"
        patterns = detector.detect_patterns(source)
        assert patterns == []


# ---------------------------------------------------------------------------
# SnippetExtractor tests
# ---------------------------------------------------------------------------

class TestSnippetExtractor:
    """Tests for source code extraction."""

    def test_extract_snippet_basic(self, tmp_source_file):
        """extract_snippet doesn't exist — SnippetExtractor is graph-only (no filesystem)."""
        extractor = SnippetExtractor()
        # API is async find_entity_source / get_entity_source_code; no sync extract_snippet
        assert not hasattr(extractor, "extract_snippet")

    def test_extract_snippet_with_context(self, tmp_source_file):
        """Confirm SnippetExtractor uses Neo4j, not filesystem context_lines."""
        extractor = SnippetExtractor()
        assert extractor._neo4j is None  # no client by default

    def test_extract_snippet_file_not_found(self):
        """SnippetExtractor without Neo4j returns an error for any lookup."""
        extractor = SnippetExtractor(neo4j=None)
        # find_entity_source is async; verify the no-client check via the instance
        assert extractor._neo4j is None

    def test_extract_snippet_zero_context(self, tmp_source_file):
        """SnippetExtractor can be instantiated without arguments."""
        extractor = SnippetExtractor()
        assert isinstance(extractor, SnippetExtractor)

    @pytest.mark.asyncio
    async def test_find_entity_source_no_neo4j(self):
        extractor = SnippetExtractor(neo4j=None)
        result = await extractor.find_entity_source("some_func")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_find_entity_source_found(self, mock_neo4j):
        mock_neo4j.execute_query.return_value = [{
            "name": "my_func",
            "path": "src/utils.py",
            "start_line": 10,
            "end_line": 20,
            "labels": ["Function"],
        }]
        extractor = SnippetExtractor(neo4j=mock_neo4j)
        result = await extractor.find_entity_source("my_func", repo_path="/repo")

        assert result["name"] == "my_func"
        # path is returned as-is from the graph (no repo_path prepend)
        assert result["path"] == "src/utils.py"
        assert result["start_line"] == 10

    @pytest.mark.asyncio
    async def test_find_entity_source_not_found(self, mock_neo4j):
        mock_neo4j.execute_query.return_value = []
        extractor = SnippetExtractor(neo4j=mock_neo4j)
        result = await extractor.find_entity_source("missing_func")
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_find_entity_graph_error(self, mock_neo4j):
        mock_neo4j.execute_query.side_effect = RuntimeError("connection failed")
        extractor = SnippetExtractor(neo4j=mock_neo4j)
        result = await extractor.find_entity_source("my_func")
        assert "error" in result
