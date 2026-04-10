"""Unit tests for src/indexer/passes/_resolution.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.indexer.passes._resolution import (
    load_imports_by_file,
    path_to_module,
    resolve_dotted,
)


# ---------------------------------------------------------------------------
# path_to_module
# ---------------------------------------------------------------------------


class TestPathToModule:
    def test_simple_file(self):
        assert path_to_module("src/foo/bar.py") == "src.foo.bar"

    def test_init_file(self):
        assert path_to_module("src/foo/__init__.py") == "src.foo"

    def test_windows_separators(self):
        assert path_to_module("src\\foo\\bar.py") == "src.foo.bar"

    def test_empty_string(self):
        assert path_to_module("") == "<module>"

    def test_top_level_file(self):
        assert path_to_module("main.py") == "main"

    def test_no_extension(self):
        # Non-.py paths returned as-is with / → .
        assert path_to_module("src/foo") == "src.foo"


# ---------------------------------------------------------------------------
# load_imports_by_file
# ---------------------------------------------------------------------------


def _make_writer(rows: list[dict]) -> MagicMock:
    """Build a mock Neo4jWriter that returns `rows` from execute_query."""
    writer = MagicMock()
    writer._client = MagicMock()
    writer._client.execute_query = AsyncMock(return_value=rows)
    return writer


class TestLoadImportsByFile:
    @pytest.mark.asyncio
    async def test_plain_import(self):
        rows = [{"fp": "a.py", "t": "import", "module": "os", "name": "os", "alias": None, "level": 0}]
        result = await load_imports_by_file(_make_writer(rows), "repo1")
        assert result["a.py"]["os"] == "os"

    @pytest.mark.asyncio
    async def test_from_import(self):
        rows = [{"fp": "a.py", "t": "from_import", "module": "fastapi", "name": "FastAPI", "alias": None, "level": 0}]
        result = await load_imports_by_file(_make_writer(rows), "repo1")
        assert result["a.py"]["FastAPI"] == "fastapi.FastAPI"

    @pytest.mark.asyncio
    async def test_aliased_from_import(self):
        rows = [{"fp": "a.py", "t": "from_import", "module": "fastapi", "name": "FastAPI", "alias": "App", "level": 0}]
        result = await load_imports_by_file(_make_writer(rows), "repo1")
        assert result["a.py"]["App"] == "fastapi.FastAPI"

    @pytest.mark.asyncio
    async def test_aliased_plain_import(self):
        rows = [{"fp": "a.py", "t": "import", "module": "", "name": "os.path", "alias": "osp", "level": 0}]
        result = await load_imports_by_file(_make_writer(rows), "repo1")
        assert result["a.py"]["osp"] == "os.path"

    @pytest.mark.asyncio
    async def test_relative_import_resolved(self):
        # 'from . import utils' in file 'pkg/sub/module.py'
        rows = [{"fp": "pkg/sub/module.py", "t": "from_import", "module": "", "name": "utils", "alias": None, "level": 1}]
        result = await load_imports_by_file(_make_writer(rows), "repo1")
        # pkg/sub/module → file_mod = pkg.sub.module → drop 1 level → pkg.sub
        assert result["pkg/sub/module.py"]["utils"] == "pkg.sub.utils"

    @pytest.mark.asyncio
    async def test_skips_rows_without_file_path(self):
        rows = [{"fp": "", "t": "import", "module": "os", "name": "os", "alias": None, "level": 0}]
        result = await load_imports_by_file(_make_writer(rows), "repo1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_multiple_files(self):
        rows = [
            {"fp": "a.py", "t": "from_import", "module": "x", "name": "Y", "alias": None, "level": 0},
            {"fp": "b.py", "t": "from_import", "module": "z", "name": "W", "alias": None, "level": 0},
        ]
        result = await load_imports_by_file(_make_writer(rows), "repo1")
        assert "Y" in result["a.py"]
        assert "W" in result["b.py"]


# ---------------------------------------------------------------------------
# resolve_dotted
# ---------------------------------------------------------------------------


class TestResolveDotted:
    def _imports(self) -> dict[str, dict[str, str]]:
        return {
            "fastapi/routing.py": {
                "Router": "starlette.routing.Router",
                "routing": "fastapi.routing",
            }
        }

    def test_resolves_head_via_import_table(self):
        result = resolve_dotted(
            "Router",
            "fastapi/routing.py",
            "fastapi.routing",
            self._imports(),
        )
        assert result == "starlette.routing.Router"

    def test_resolves_dotted_head_via_import_table(self):
        result = resolve_dotted(
            "routing.APIRouter",
            "fastapi/routing.py",
            "fastapi.routing",
            self._imports(),
        )
        assert result == "fastapi.routing.APIRouter"

    def test_prepends_module_when_not_in_imports(self):
        result = resolve_dotted(
            "MyClass",
            "fastapi/routing.py",
            "fastapi.routing",
            self._imports(),
        )
        assert result == "fastapi.routing.MyClass"

    def test_file_not_in_imports(self):
        result = resolve_dotted(
            "Foo",
            "unknown/file.py",
            "unknown.file",
            {},
        )
        assert result == "unknown.file.Foo"

    def test_empty_ref(self):
        result = resolve_dotted("", "a.py", "a", {})
        assert result == ""
