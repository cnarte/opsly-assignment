"""Tests for the Cypher safety checker."""

from __future__ import annotations

import pytest

from src.graph_query.safety import CypherSafetyChecker, CypherSafetyResult
from src.shared.exceptions import CypherSafetyError


@pytest.fixture()
def checker() -> CypherSafetyChecker:
    return CypherSafetyChecker()


# -- Write rejection --------------------------------------------------------


class TestWriteRejection:
    """Safety checker must reject all write operations."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "CREATE (n:Class {name: 'Foo'})",
            "MERGE (n:Class {name: 'Foo'})",
            "MATCH (n:Class) SET n.name = 'Bar'",
            "MATCH (n:Class) DELETE n",
            "MATCH (n:Class) DETACH DELETE n",
            "MATCH (n:Class) REMOVE n.name",
            "DROP CONSTRAINT ON (n:Class) ASSERT n.name IS UNIQUE",
            "CALL db.index.fulltext.createNodeIndex('idx', ['Class'], ['name'])",
        ],
    )
    def test_rejects_write_queries(self, checker: CypherSafetyChecker, cypher: str) -> None:
        result = checker.check(cypher)
        assert not result.is_safe
        assert "not allowed" in result.reason.lower() or "write" in result.reason.lower()

    def test_validate_or_raise_raises(self, checker: CypherSafetyChecker) -> None:
        with pytest.raises(CypherSafetyError):
            checker.validate_or_raise("CREATE (n:Class {name: 'Foo'})")


# -- Label allowlist --------------------------------------------------------


class TestLabelAllowlist:
    """Safety checker must reject queries with non-allowlisted labels."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n:EpisodicNode) RETURN n",
            "MATCH (n:EntityNode) RETURN n",
            "MATCH (n:CommunityNode) RETURN n",
            "MATCH (n:User) RETURN n",
        ],
    )
    def test_rejects_non_allowlisted_labels(self, checker: CypherSafetyChecker, cypher: str) -> None:
        result = checker.check(cypher)
        assert not result.is_safe
        assert "not in the allowed" in result.reason.lower()

    @pytest.mark.parametrize(
        "label",
        [
            "File", "Module", "Class", "Function", "Method",
            "Parameter", "Decorator", "Import", "Docstring",
        ],
    )
    def test_accepts_allowlisted_labels(self, checker: CypherSafetyChecker, label: str) -> None:
        cypher = f"MATCH (n:{label}) RETURN n LIMIT 10"
        result = checker.check(cypher)
        assert result.is_safe


# -- Bare node pattern ------------------------------------------------------


class TestBareNodePattern:
    """Safety checker must reject bare MATCH (n) without a label."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n) RETURN n",
            "MATCH (x) RETURN x LIMIT 10",
            "MATCH (n {name: 'foo'}) RETURN n",
        ],
    )
    def test_rejects_bare_match(self, checker: CypherSafetyChecker, cypher: str) -> None:
        result = checker.check(cypher)
        assert not result.is_safe
        assert "label" in result.reason.lower()


# -- Valid queries -----------------------------------------------------------


class TestValidQueries:
    """Safety checker must accept valid read-only queries with code-graph labels."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n:Class) WHERE n.name = 'Foo' RETURN n",
            "MATCH (n:Function)-[:CALLS]->(m:Function) RETURN n, m",
            "MATCH (m:Module)-[:IMPORTS*1..3]->(t:Module) RETURN m, t",
            "MATCH (c:Class)-[:INHERITS_FROM]->(p:Class) RETURN c.name, p.name",
            "MATCH (f:Function)-[:HAS_PARAMETER]->(p:Parameter) RETURN f.name, p.name",
            "MATCH (f:File)-[:DEFINES]->(c:Class) RETURN f.name, c.name LIMIT 10",
        ],
    )
    def test_accepts_valid_queries(self, checker: CypherSafetyChecker, cypher: str) -> None:
        result = checker.check(cypher)
        assert result.is_safe
        assert result.reason == "Query is safe."
