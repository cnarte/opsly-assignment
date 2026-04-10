"""Unit tests for Neo4jWriter batched graph writer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.indexer.neo4j_writer import Neo4jWriter


def _make_client() -> MagicMock:
    client = MagicMock()
    client.execute_query = AsyncMock(return_value=[])
    client.execute_write = AsyncMock(return_value=[])
    return client


class TestMergeNodes:
    @pytest.mark.asyncio
    async def test_merge_nodes_single_batch(self):
        """merge_nodes submits a single batch when items fit in one batch."""
        client = _make_client()
        writer = Neo4jWriter(client)

        nodes = [
            {"symbol_id": "r1:class:m:Foo:0", "name": "Foo", "kind": "class"},
            {"symbol_id": "r1:class:m:Bar:0", "name": "Bar", "kind": "class"},
        ]
        count = await writer.merge_nodes("Class", nodes, repo_id="r1")

        client.execute_write.assert_called_once()
        assert count == 2

    @pytest.mark.asyncio
    async def test_merge_nodes_empty(self):
        """merge_nodes with empty list does nothing."""
        client = _make_client()
        writer = Neo4jWriter(client)

        count = await writer.merge_nodes("Class", [], repo_id="r1")

        client.execute_write.assert_not_called()
        assert count == 0

    @pytest.mark.asyncio
    async def test_merge_nodes_sanitises_dict_values(self):
        """merge_nodes serialises nested dict values to JSON strings."""
        client = _make_client()
        writer = Neo4jWriter(client)

        nodes = [
            {
                "symbol_id": "r1:class:m:Foo:0",
                "name": "Foo",
                "metadata": {"nested": "value"},  # dict → JSON string
            }
        ]
        await writer.merge_nodes("Class", nodes, repo_id="r1")

        call_params = client.execute_write.call_args.args[1]
        batch = call_params["batch"]
        import json
        assert isinstance(batch[0]["metadata"], str)
        assert json.loads(batch[0]["metadata"]) == {"nested": "value"}

    @pytest.mark.asyncio
    async def test_merge_nodes_adds_repo_id_and_timestamps(self):
        """merge_nodes injects repo_id and updated_at into every node."""
        client = _make_client()
        writer = Neo4jWriter(client)

        nodes = [{"symbol_id": "r1:class:m:Foo:0", "name": "Foo"}]
        await writer.merge_nodes("Class", nodes, repo_id="r1")

        call_params = client.execute_write.call_args.args[1]
        batch = call_params["batch"]
        assert batch[0]["repo_id"] == "r1"
        assert "indexed_at" in batch[0]


class TestMergeRelationships:
    @pytest.mark.asyncio
    async def test_merge_relationships_single_batch(self):
        """merge_relationships submits a batch of CALLS rels."""
        client = _make_client()
        writer = Neo4jWriter(client)

        rels = [
            {"source_id": "sid1", "target_id": "sid2", "confidence": 1.0},
            {"source_id": "sid3", "target_id": "sid4", "confidence": 0.8},
        ]
        count = await writer.merge_relationships("CALLS", rels)

        client.execute_write.assert_called_once()
        assert count == 2

    @pytest.mark.asyncio
    async def test_merge_relationships_empty(self):
        """merge_relationships with empty list does nothing."""
        client = _make_client()
        writer = Neo4jWriter(client)

        count = await writer.merge_relationships("CALLS", [])

        client.execute_write.assert_not_called()
        assert count == 0

    @pytest.mark.asyncio
    async def test_merge_relationships_includes_extra_props(self):
        """merge_relationships attaches extra properties to the rel."""
        client = _make_client()
        writer = Neo4jWriter(client)

        rels = [
            {"source_id": "sid1", "target_id": "sid2", "confidence": 0.9, "line": 42},
        ]
        count = await writer.merge_relationships("CALLS", rels)

        assert count == 1
        call_params = client.execute_write.call_args.args[1]
        batch = call_params["batch"]
        assert batch[0]["props"]["confidence"] == 0.9
        assert batch[0]["props"]["line"] == 42
