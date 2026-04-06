"""Custom exception hierarchy for the multi-agent system."""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for all agent-related errors."""


class AgentTimeoutError(AgentError):
    """Raised when an agent operation exceeds its timeout."""


class AgentConnectionError(AgentError):
    """Raised when a connection to an agent or backing service fails."""


class GraphQueryError(AgentError):
    """Raised when a graph database query fails."""


class IndexingError(AgentError):
    """Raised when repository indexing fails."""


class MemoryError(AgentError):
    """Raised when memory storage or retrieval fails."""


class CypherSafetyError(GraphQueryError):
    """Raised when a Cypher query is rejected by the safety checker."""
