"""Cypher safety guards for the execute_query tool."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.shared.exceptions import CypherSafetyError

# Labels that belong to the code-graph schema.
ALLOWED_LABELS: frozenset[str] = frozenset(
    {
        "File",
        "Module",
        "Class",
        "Function",
        "Method",
        "Parameter",
        "Decorator",
        "Import",
        "Docstring",
    }
)

# Cypher keywords that mutate the database.
_WRITE_KEYWORDS: re.Pattern[str] = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH\s+DELETE|REMOVE|DROP|CALL)\b",
    re.IGNORECASE,
)

# Pattern to find labels in Cypher — matches (n:Label), (:Label), (n:Label1:Label2)
_LABEL_PATTERN: re.Pattern[str] = re.compile(
    r"(?::\s*)([A-Z][A-Za-z_0-9]+)",
    re.IGNORECASE,
)

# Pattern to detect bare MATCH (n) without a label — e.g. MATCH (n) or MATCH (n {prop: val})
_BARE_NODE_PATTERN: re.Pattern[str] = re.compile(
    r"\bMATCH\b\s*\(\s*[a-zA-Z_]\w*\s*(?:\{[^}]*\})?\s*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CypherSafetyResult:
    """Result of a Cypher safety check."""

    is_safe: bool
    reason: str


class CypherSafetyChecker:
    """Validates user-supplied Cypher queries for safety."""

    def __init__(self, allowed_labels: frozenset[str] | None = None) -> None:
        self.allowed_labels = allowed_labels or ALLOWED_LABELS

    def check(self, cypher: str) -> CypherSafetyResult:
        """Return a CypherSafetyResult indicating whether the query is safe."""

        # 1. Reject write operations
        match = _WRITE_KEYWORDS.search(cypher)
        if match:
            keyword = match.group(1).upper()
            return CypherSafetyResult(
                is_safe=False,
                reason=f"Write operation '{keyword}' is not allowed. Only read queries are permitted.",
            )

        # 2. Reject bare MATCH (n) without label
        # We strip out string literals first to avoid false positives
        stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "", cypher)
        # Find all MATCH clauses and check each node pattern for a label
        # Pattern: MATCH followed by node patterns like (var), (var {props})
        # A node pattern without a colon before the closing paren/brace is bare
        match_clauses = re.finditer(
            r"\bMATCH\b\s*(.*?)(?=\bWHERE\b|\bRETURN\b|\bWITH\b|\bORDER\b|\bLIMIT\b|\bUNION\b|$)",
            stripped,
            re.IGNORECASE,
        )
        for mc in match_clauses:
            clause = mc.group(1)
            # Find node patterns: (identifier) or (identifier {props})
            # A valid labeled pattern has a colon: (n:Label) or (n:Label {props})
            node_patterns = re.finditer(r"\(\s*([a-zA-Z_]\w*)\s*(\{[^}]*\})?\s*\)", clause)
            for np in node_patterns:
                var_and_maybe_label = np.group(0)
                inner = var_and_maybe_label[1:-1]  # strip parens
                # Check for a label colon BEFORE any property block {…}
                brace_pos = inner.find("{")
                label_part = inner[:brace_pos] if brace_pos != -1 else inner
                if ":" not in label_part:
                    return CypherSafetyResult(
                        is_safe=False,
                        reason="Query contains bare node pattern without a label filter. "
                        "All node patterns must specify a label.",
                    )

        # 3. Check all labels used in the query are in the allowlist
        labels_found = _LABEL_PATTERN.findall(stripped)
        # Filter out relationship types and Cypher built-ins
        _cypher_builtins = {
            "MATCH", "WHERE", "RETURN", "WITH", "ORDER", "LIMIT", "SKIP",
            "OPTIONAL", "UNWIND", "AS", "AND", "OR", "NOT", "IN", "IS",
            "NULL", "TRUE", "FALSE", "CONTAINS", "STARTS", "ENDS",
            "COUNT", "COLLECT", "SUM", "AVG", "MIN", "MAX",
            "DISTINCT", "CASE", "WHEN", "THEN", "ELSE", "END",
            "EXISTS", "ALL", "ANY", "NONE", "SINGLE", "BY", "DESC", "ASC",
            "CALLS", "IMPORTS", "DEPENDS_ON", "INHERITS_FROM",
            "HAS_PARAMETER", "HAS_DECORATOR", "HAS_DOCSTRING",
            "DEFINES", "CONTAINS_CLASS", "CONTAINS_FUNCTION",
        }
        node_labels = [
            label for label in labels_found
            if label.upper() not in _cypher_builtins
        ]

        for label in node_labels:
            if label not in self.allowed_labels:
                return CypherSafetyResult(
                    is_safe=False,
                    reason=f"Label '{label}' is not in the allowed code-graph labels: "
                    f"{sorted(self.allowed_labels)}",
                )

        return CypherSafetyResult(is_safe=True, reason="Query is safe.")

    def validate_or_raise(self, cypher: str) -> None:
        """Raise CypherSafetyError if the query is not safe."""
        result = self.check(cypher)
        if not result.is_safe:
            raise CypherSafetyError(result.reason)
