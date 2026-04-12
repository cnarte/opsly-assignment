"""Unit tests for GitNexusClient — mock the stdio transport."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_connect_initialises_session():
    """connect() should enter stdio_client context and call session.initialize()."""
    from src.shared.gitnexus_client import GitNexusClient

    mock_session = AsyncMock()
    mock_read = AsyncMock()
    mock_write = AsyncMock()

    # The implementation uses AsyncExitStack.enter_async_context(), so we patch
    # stdio_client to return an async context manager yielding (read, write),
    # and ClientSession to return an async context manager yielding the mock session.
    mock_stdio_cm = AsyncMock()
    mock_stdio_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
    mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)
    mock_session_cm.initialize = mock_session.initialize

    with patch("src.shared.gitnexus_client.stdio_client", return_value=mock_stdio_cm) as mock_stdio, \
         patch("src.shared.gitnexus_client.ClientSession", return_value=mock_session_cm) as mock_cls:

        client = GitNexusClient()
        await client.connect()

        # Verify initialize() was called and _session is set
        mock_session_cm.initialize.assert_awaited_once()
        assert client._session is mock_session_cm


@pytest.mark.asyncio
async def test_call_tool_parses_json():
    """call_tool() should parse JSON from the first content item."""
    from src.shared.gitnexus_client import GitNexusClient

    mock_session = AsyncMock()
    content_item = MagicMock()
    content_item.text = json.dumps({"results": [{"name": "FastAPI"}]})
    mock_result = MagicMock()
    mock_result.content = [content_item]
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    client = GitNexusClient()
    client._session = mock_session

    result = await client.call_tool("query", {"query": "FastAPI"})
    assert result == {"results": [{"name": "FastAPI"}]}
    mock_session.call_tool.assert_awaited_once_with("query", {"query": "FastAPI"})


@pytest.mark.asyncio
async def test_call_tool_returns_raw_on_non_json():
    """call_tool() should wrap non-JSON text in {"result": ...}."""
    from src.shared.gitnexus_client import GitNexusClient

    mock_session = AsyncMock()
    content_item = MagicMock()
    content_item.text = "not json"
    mock_result = MagicMock()
    mock_result.content = [content_item]
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    client = GitNexusClient()
    client._session = mock_session

    result = await client.call_tool("cypher", {"query": "MATCH (n) RETURN n"})
    assert result == {"result": "not json"}


@pytest.mark.asyncio
async def test_call_tool_raises_when_not_connected():
    """call_tool() should raise RuntimeError if connect() was never called."""
    from src.shared.gitnexus_client import GitNexusClient

    client = GitNexusClient()
    with pytest.raises(RuntimeError, match="not connected"):
        await client.call_tool("query", {})
