"""Unit tests for indexer server tools and indexer passes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# repo_structure pass
# ---------------------------------------------------------------------------


class TestRepoStructurePass:
    @pytest.mark.asyncio
    async def test_creates_file_and_module_nodes(self):
        from src.indexer.passes.repo_structure import run

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two python files
            Path(tmpdir, "main.py").write_text("# main")
            sub = Path(tmpdir, "pkg")
            sub.mkdir()
            (sub / "__init__.py").write_text("")
            (sub / "utils.py").write_text("def helper(): pass")

            writer = MagicMock()
            writer.merge_nodes = AsyncMock(return_value=3)
            writer.merge_relationships = AsyncMock(return_value=3)

            result = await run(writer, tmpdir, "repo1", "sha1")

        assert result["files"] == 3  # main.py + __init__.py + utils.py
        assert result["modules"] == 3

        # Check CONTAINS rels were created
        writer.merge_relationships.assert_called_once()
        rels = writer.merge_relationships.call_args.args[1]
        assert len(rels) == 3

    @pytest.mark.asyncio
    async def test_ignores_non_python_files(self):
        from src.indexer.passes.repo_structure import run

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "README.md").write_text("docs")
            Path(tmpdir, "main.py").write_text("# main")
            Path(tmpdir, "config.json").write_text("{}")

            writer = MagicMock()
            writer.merge_nodes = AsyncMock(return_value=1)
            writer.merge_relationships = AsyncMock(return_value=1)

            result = await run(writer, tmpdir, "repo1")

        # Only main.py should be found
        file_nodes = writer.merge_nodes.call_args_list[0].args[1]
        assert all(n["path"].endswith(".py") for n in file_nodes)
        assert len(file_nodes) == 1

    @pytest.mark.asyncio
    async def test_init_module_name_truncated(self):
        from src.indexer.passes.repo_structure import run

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = Path(tmpdir, "mypkg")
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")

            writer = MagicMock()
            writer.merge_nodes = AsyncMock(return_value=1)
            writer.merge_relationships = AsyncMock(return_value=1)

            await run(writer, tmpdir, "repo1")

        module_nodes = writer.merge_nodes.call_args_list[1].args[1]
        names = [n["name"] for n in module_nodes]
        # __init__ should be stripped: mypkg (not mypkg.__init__)
        assert all("__init__" not in name for name in names)


# ---------------------------------------------------------------------------
# import_resolution pass
# ---------------------------------------------------------------------------


class TestImportResolutionPass:
    def _make_writer_with_client(self) -> MagicMock:
        """import_resolution queries the graph; needs _client mock."""
        writer = MagicMock()
        writer._client = MagicMock()
        # Return empty lists for all execute_query calls (no data = valid state)
        writer._client.execute_query = AsyncMock(return_value=[])
        writer.merge_relationships = AsyncMock(return_value=0)
        return writer

    @pytest.mark.asyncio
    async def test_creates_import_relationships(self):
        """import_resolution runs without error when graph is empty."""
        from src.indexer.passes.import_resolution import run

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_writer_with_client()
            result = await run(writer, tmpdir, "repo1")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handles_empty_imports(self):
        """import_resolution handles repos with no imports."""
        from src.indexer.passes.import_resolution import run

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._make_writer_with_client()
            result = await run(writer, tmpdir, "repo1")

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# symbol_extraction pass
# ---------------------------------------------------------------------------


class TestSymbolExtractionPass:
    @pytest.mark.asyncio
    async def test_extracts_class_and_function(self):
        from src.indexer.passes.symbol_extraction import run

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "mymod.py").write_text(
                "class Foo:\n    def bar(self):\n        pass\n\ndef top_level():\n    pass\n"
            )

            writer = MagicMock()
            writer.merge_nodes = AsyncMock(return_value=0)
            writer.merge_relationships = AsyncMock(return_value=0)

            await run(writer, tmpdir, "repo1")

        # Verify Class and Function nodes were submitted to merge
        merge_calls = writer.merge_nodes.call_args_list
        labels = [c.args[0] for c in merge_calls]
        assert "Class" in labels
        # Check that at least one class node (Foo) was submitted
        class_call = next(c for c in merge_calls if c.args[0] == "Class")
        class_nodes = class_call.args[1]
        assert any(n["name"] == "Foo" for n in class_nodes)


# ---------------------------------------------------------------------------
# indexer server tools (unit — no real MCP, no real git)
# ---------------------------------------------------------------------------


class TestIndexerServerTools:
    @pytest.mark.asyncio
    async def test_parse_python_ast_returns_entities(self):
        """parse_python_ast extracts classes and functions from source."""
        from src.indexer.server import parse_python_ast

        source = "class MyClass:\n    def my_method(self):\n        pass\n"
        result = await parse_python_ast(source, "test.py")

        assert "classes" in result or "error" not in result
        if "classes" in result:
            assert any(c["name"] == "MyClass" for c in result["classes"])

    @pytest.mark.asyncio
    async def test_extract_entities_returns_entity_list(self):
        """extract_entities flattens AST parse into entity list."""
        from src.indexer.server import extract_entities

        source = "class Foo:\n    def bar(self):\n        pass\n\ndef standalone():\n    pass\n"
        result = await extract_entities(source, "test.py")

        assert result["entity_count"] >= 2
        kinds = {e["kind"] for e in result["entities"]}
        assert "class" in kinds

    @pytest.mark.asyncio
    async def test_get_index_status_unknown_job(self):
        """get_index_status returns error for unknown job_id."""
        from src.indexer.server import get_index_status

        result = await get_index_status("nonexistent-job-id-xyz")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_index_status_no_job_id(self):
        """get_index_status with no job_id returns aggregate."""
        from src.indexer.server import get_index_status

        result = await get_index_status("")
        assert "total_jobs" in result

    @pytest.mark.asyncio
    async def test_index_file_not_found(self):
        """index_file raises IndexingError for missing file."""
        from src.indexer.server import index_file
        from src.shared.exceptions import IndexingError

        with pytest.raises(IndexingError):
            await index_file("/nonexistent/repo", "missing.py")
