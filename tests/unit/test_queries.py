"""Tests for the Cypher query builders."""

from __future__ import annotations

import pytest

from src.graph_query.queries import (
    CypherQuery,
    find_entity,
    find_related,
    get_dependencies,
    get_dependents,
    get_symbol_context,
    impact_at_depth,
    trace_imports,
)


class TestFindEntity:
    """Tests for the find_entity query builder."""

    def test_with_entity_type(self) -> None:
        q = find_entity("MyClass", "Class")
        assert "MATCH (n:Class)" in q.cypher
        assert q.parameters == {"name": "MyClass"}
        assert "$name" in q.cypher

    def test_without_entity_type(self) -> None:
        q = find_entity("some_func")
        # Should produce UNION across entity labels, no bare MATCH (n)
        assert "UNION" in q.cypher
        assert "MATCH (n:" in q.cypher
        assert q.parameters == {"name": "some_func"}

    def test_invalid_entity_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown entity type"):
            find_entity("x", "InvalidLabel")


class TestGetDependencies:
    def test_produces_parameterized_query(self) -> None:
        q = get_dependencies("MyClass")
        assert q.parameters == {"name": "MyClass"}
        assert "CALLS|IMPORTS|DEPENDS_ON|INHERITS_FROM" in q.cypher
        assert "$name" in q.cypher
        # Must use explicit labels
        assert "MATCH (n:" in q.cypher


class TestGetDependents:
    def test_produces_parameterized_query(self) -> None:
        q = get_dependents("my_func")
        assert q.parameters == {"name": "my_func"}
        assert "$name" in q.cypher
        assert "MATCH" in q.cypher


class TestTraceImports:
    def test_produces_import_chain_query(self) -> None:
        q = trace_imports("utils")
        assert q.parameters == {"name": "utils"}
        assert "Module" in q.cypher
        assert "IMPORTS" in q.cypher
        assert "*1..5" in q.cypher


class TestFindRelated:
    def test_produces_relationship_query(self) -> None:
        q = find_related("MyClass", "INHERITS_FROM")
        assert q.parameters == {"name": "MyClass"}
        assert "INHERITS_FROM" in q.cypher


class TestGetSymbolContext:
    def test_produces_bidirectional_query(self) -> None:
        q = get_symbol_context("my_func")
        assert q.parameters == {"name": "my_func"}
        assert "outgoing" in q.cypher
        assert "incoming" in q.cypher


class TestImpactAtDepth:
    def test_produces_depth_query(self) -> None:
        q = impact_at_depth("my_func", 3)
        assert q.parameters == {"name": "my_func"}
        assert "*1..3" in q.cypher
        assert "depth" in q.cypher


class TestAllQueriesUseExplicitLabels:
    """Every query builder must use explicit label filters (no bare MATCH (n))."""

    @pytest.mark.parametrize(
        "query_fn,args",
        [
            (find_entity, ("x",)),
            (find_entity, ("x", "Class")),
            (get_dependencies, ("x",)),
            (get_dependents, ("x",)),
            (trace_imports, ("x",)),
            (find_related, ("x", "CALLS")),
            (get_symbol_context, ("x",)),
            (impact_at_depth, ("x", 2)),
        ],
    )
    def test_no_bare_match(self, query_fn, args) -> None:
        q = query_fn(*args)
        # Ensure every MATCH has a label
        import re
        bare = re.findall(r"MATCH\s*\(\s*\w+\s*\)", q.cypher)
        assert not bare, f"Found bare MATCH pattern(s): {bare}"

    @pytest.mark.parametrize(
        "query_fn,args",
        [
            (find_entity, ("x",)),
            (find_entity, ("x", "Class")),
            (get_dependencies, ("x",)),
            (get_dependents, ("x",)),
            (trace_imports, ("x",)),
            (find_related, ("x", "CALLS")),
            (get_symbol_context, ("x",)),
            (impact_at_depth, ("x", 2)),
        ],
    )
    def test_queries_are_parameterized(self, query_fn, args) -> None:
        q = query_fn(*args)
        assert isinstance(q, CypherQuery)
        assert isinstance(q.parameters, dict)
        assert len(q.parameters) > 0
