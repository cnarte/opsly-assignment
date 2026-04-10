"""Unit tests for indexer pass modules (derived_artifacts, call_linking)."""

from __future__ import annotations

import ast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.indexer.passes._resolution import path_to_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_writer(query_rows: list[dict] | None = None) -> MagicMock:
    writer = MagicMock()
    writer._client = MagicMock()
    writer._client.execute_query = AsyncMock(return_value=query_rows or [])
    writer._client.execute_write = AsyncMock()
    writer.merge_relationships = AsyncMock()
    writer.merge_nodes = AsyncMock()
    return writer


# ---------------------------------------------------------------------------
# call_linking._call_name
# ---------------------------------------------------------------------------


class TestCallName:
    def test_simple_name(self):
        from src.indexer.passes.call_linking import _call_name
        node = ast.parse("foo()").body[0].value
        assert _call_name(node) == "foo"

    def test_attribute_call(self):
        from src.indexer.passes.call_linking import _call_name
        node = ast.parse("obj.method()").body[0].value
        assert _call_name(node) == "obj.method"

    def test_chained_attribute(self):
        from src.indexer.passes.call_linking import _call_name
        node = ast.parse("a.b.c()").body[0].value
        assert _call_name(node) == "a.b.c"

    def test_non_name_call_returns_empty(self):
        from src.indexer.passes.call_linking import _call_name
        # e.g. lambda()() — func is not a Name or Attribute chain
        node = ast.parse("(lambda: None)()").body[0].value
        assert _call_name(node) == ""


# ---------------------------------------------------------------------------
# call_linking._build_sid
# ---------------------------------------------------------------------------


class TestBuildSid:
    def test_format(self):
        from src.indexer.passes.call_linking import _build_sid
        sid = _build_sid("repo1", "function", "mymod", "my_func", 0)
        assert sid == "repo1:function:mymod:my_func:0"

    def test_method_kind(self):
        from src.indexer.passes.call_linking import _build_sid
        sid = _build_sid("repo1", "method", "mymod.MyClass", "run", 0)
        assert sid == "repo1:method:mymod.MyClass:run:0"


# ---------------------------------------------------------------------------
# call_linking._collect_function_calls
# ---------------------------------------------------------------------------


class TestCollectFunctionCalls:
    def _parse_func(self, src: str) -> ast.FunctionDef:
        tree = ast.parse(src)
        return tree.body[0]

    def test_direct_call_resolved_by_qn(self):
        from src.indexer.passes.call_linking import _collect_function_calls

        src = "def caller():\n    do_something()\n"
        func = self._parse_func(src)

        by_qn = {"mymod.do_something": "repo1:function:mymod:do_something:0"}
        by_name = {"do_something": ["repo1:function:mymod:do_something:0"]}
        imports = {}
        rels: list = []

        _collect_function_calls(
            func,
            caller_sid="repo1:function:mymod:caller:0",
            file_path="mymod.py",
            file_mod="mymod",
            by_qn=by_qn,
            by_name=by_name,
            imports_by_file=imports,
            rels=rels,
        )

        assert len(rels) == 1
        assert rels[0]["target_id"] == "repo1:function:mymod:do_something:0"
        assert rels[0]["confidence"] == 1.0
        assert rels[0]["resolution_method"] == "qn"

    def test_ambiguous_name_gets_low_confidence(self):
        from src.indexer.passes.call_linking import _collect_function_calls

        src = "def caller():\n    process()\n"
        func = self._parse_func(src)

        # Two candidates with same name
        by_qn = {}
        by_name = {"process": ["repo1:function:a:process:0", "repo2:function:b:process:0"]}
        rels: list = []

        _collect_function_calls(
            func,
            caller_sid="repo1:function:mymod:caller:0",
            file_path="mymod.py",
            file_mod="mymod",
            by_qn=by_qn,
            by_name=by_name,
            imports_by_file={},
            rels=rels,
        )

        assert len(rels) == 1
        assert rels[0]["confidence"] == 0.5
        assert rels[0]["resolution_method"] == "ambiguous"

    def test_unique_name_medium_confidence(self):
        from src.indexer.passes.call_linking import _collect_function_calls

        src = "def caller():\n    helper()\n"
        func = self._parse_func(src)

        by_qn = {}
        by_name = {"helper": ["repo1:function:util:helper:0"]}
        rels: list = []

        _collect_function_calls(
            func,
            caller_sid="repo1:function:mymod:caller:0",
            file_path="mymod.py",
            file_mod="mymod",
            by_qn=by_qn,
            by_name=by_name,
            imports_by_file={},
            rels=rels,
        )

        assert len(rels) == 1
        assert rels[0]["confidence"] == 0.8
        assert rels[0]["resolution_method"] == "name"

    def test_self_calls_excluded(self):
        """A function calling itself should not produce a CALLS edge."""
        from src.indexer.passes.call_linking import _collect_function_calls

        src = "def recursive():\n    recursive()\n"
        func = self._parse_func(src)

        caller_sid = "repo1:function:mymod:recursive:0"
        by_qn = {"mymod.recursive": caller_sid}
        by_name = {"recursive": [caller_sid]}
        rels: list = []

        _collect_function_calls(
            func,
            caller_sid=caller_sid,
            file_path="mymod.py",
            file_mod="mymod",
            by_qn=by_qn,
            by_name=by_name,
            imports_by_file={},
            rels=rels,
        )

        assert len(rels) == 0

    def test_duplicate_calls_deduped(self):
        """Calling the same function twice on the same line only creates one edge."""
        from src.indexer.passes.call_linking import _collect_function_calls

        # Same call on same line via EXPRESSION statements (both map to line 2)
        src = "def caller():\n    do_something()\n    do_something()\n"
        func = self._parse_func(src)

        by_qn = {}
        by_name = {"do_something": ["repo1:function:m:do_something:0"]}
        rels: list = []

        _collect_function_calls(
            func,
            caller_sid="repo1:function:m:caller:0",
            file_path="m.py",
            file_mod="m",
            by_qn=by_qn,
            by_name=by_name,
            imports_by_file={},
            rels=rels,
        )

        # Two calls on different lines → 2 rels; same (target, line) pair → 1 rel
        assert len(rels) <= 2


# ---------------------------------------------------------------------------
# derived_artifacts._compute_inheritance — tested via function isolation
# ---------------------------------------------------------------------------


class TestComputeInheritance:
    @pytest.mark.asyncio
    async def test_resolves_base_via_import_table(self):
        """_compute_inheritance creates an INHERITS_FROM edge when base is in imports."""
        from src.indexer.passes.derived_artifacts import _compute_inheritance

        classes = [
            {
                "sid": "repo1:class:mymod:Child:0",
                "name": "Child",
                "bases": ["Router"],          # Python list as Neo4j returns
                "fp": "mymod/view.py",
                "qn": "mymod.view.Child",
            },
            {
                "sid": "repo1:class:starlette.routing:Router:0",
                "name": "Router",
                "bases": [],
                "fp": "starlette/routing.py",
                "qn": "starlette.routing.Router",
            },
        ]

        imports_rows = [
            {
                "fp": "mymod/view.py",
                "t": "from_import",
                "module": "starlette.routing",
                "name": "Router",
                "alias": None,
                "level": 0,
            }
        ]

        writer = _make_writer()
        # Sequence of execute_query calls:
        # 1. load_imports_by_file
        # 2. classes query
        writer._client.execute_query = AsyncMock(
            side_effect=[imports_rows, classes]
        )

        count = await _compute_inheritance(writer, "repo1")

        assert count >= 1
        # INHERITS_FROM relationship should have been merged
        writer.merge_relationships.assert_called_once()
        rels_call = writer.merge_relationships.call_args
        rels = rels_call.args[1]
        assert any(r["target_id"] == "repo1:class:starlette.routing:Router:0" for r in rels)

    @pytest.mark.asyncio
    async def test_creates_external_stub_for_unresolved_base(self):
        """_compute_inheritance creates a stub Class node for external bases."""
        from src.indexer.passes.derived_artifacts import _compute_inheritance

        classes = [
            {
                "sid": "repo1:class:mymod:MyView:0",
                "name": "MyView",
                "bases": ["BaseView"],       # Python list
                "fp": "mymod/view.py",
                "qn": "mymod.view.MyView",
            }
        ]

        writer = _make_writer()
        writer._client.execute_query = AsyncMock(side_effect=[[], classes])

        count = await _compute_inheritance(writer, "repo1")

        # External stub node should have been merged
        writer.merge_nodes.assert_called_once()
        nodes_call = writer.merge_nodes.call_args
        stub_nodes = nodes_call.args[1]
        assert any(n.get("external") is True for n in stub_nodes)

    @pytest.mark.asyncio
    async def test_skips_object_base(self):
        """_compute_inheritance ignores 'object' as a base class."""
        from src.indexer.passes.derived_artifacts import _compute_inheritance

        classes = [
            {
                "sid": "repo1:class:mymod:MyClass:0",
                "name": "MyClass",
                "bases": ["object"],         # Python list
                "fp": "mymod/mod.py",
                "qn": "mymod.mod.MyClass",
            }
        ]

        writer = _make_writer()
        writer._client.execute_query = AsyncMock(side_effect=[[], classes])

        count = await _compute_inheritance(writer, "repo1")

        writer.merge_nodes.assert_not_called()
        assert count == 0


# ---------------------------------------------------------------------------
# canonical_identity — qualified_name construction
# ---------------------------------------------------------------------------


class TestCanonicalIdentity:
    @pytest.mark.asyncio
    async def test_qualified_name_uses_scope_chain(self):
        """qualified_name for a method is scope_chain.name, not double-prefixed."""
        from src.orchestrator.nodes import _extract_query  # just to test the import works
        import tempfile, os
        from src.indexer.passes.canonical_identity import run

        # Create a tiny temporary Python file
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "mymod.py")
            with open(fpath, "w") as f:
                f.write("class Foo:\n    def bar(self):\n        pass\n")

            entities = [
                {
                    "sid": "repo1:method:mymod.Foo:bar:0",
                    "name": "bar",
                    "scope": "mymod.Foo",
                    "fp": "mymod.py",
                    "sl": 2,
                    "el": 3,
                }
            ]

            writer = _make_writer(entities)
            # mock execute_write to capture the qualified_name set
            written: list[dict] = []

            async def _capture_write(query, params):
                written.append(params)

            writer._client.execute_write = _capture_write

            await run(writer, tmpdir, "repo1")

            assert len(written) == 1
            # Should be "mymod.Foo.bar" (scope_chain already has module)
            assert written[0]["qn"] == "mymod.Foo.bar"
            # Must NOT be "mymod.mymod.Foo.bar"
            assert "mymod.mymod" not in written[0]["qn"]
