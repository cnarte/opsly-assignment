"""Indexing pipeline passes."""

from src.indexer.passes import (  # noqa: F401
    repo_structure,
    symbol_extraction,
    import_resolution,
    canonical_identity,
    call_linking,
    derived_artifacts,
)
