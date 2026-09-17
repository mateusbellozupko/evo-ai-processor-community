import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.adk.tools.preload_memory_tool import create_preload_memory_tool
from src.services.memory_service import SearchMemoryResponse, MemoryEntry
from google.genai import types


def _tool_context(app_name="agent-1", user_id="user-1"):
    ctx = MagicMock()
    ctx.session.app_name = app_name
    ctx.session.user_id = user_id
    return ctx


@pytest.mark.asyncio
async def test_preload_memory_calls_search_memory_with_empty_query():
    tool = await create_preload_memory_tool(memory_base_config_id="config-1", default_max_results=10)
    fn = tool.func

    # NOTE: google.adk's MemoryEntry/SearchMemoryResponse are pydantic models with
    # extra="ignore" (unset), so unknown kwargs (e.g. "metadata", "score", "total",
    # "query") are silently dropped rather than raising. The brief's test snippet
    # used `metadata=` and `score=` on MemoryEntry and `total=`/`query=` on
    # SearchMemoryResponse, but the real fields are:
    #   MemoryEntry: content, custom_metadata, id, author, timestamp (no "score")
    #   SearchMemoryResponse: memories only (no "total"/"query")
    # Constructed here with the real field names/shape confirmed against
    # src/services/memory_service.py + the installed google-adk package.
    response = SearchMemoryResponse(
        memories=[
            MemoryEntry(
                content=types.Content(role="user", parts=[types.Part(text="Prior summary.")]),
                custom_metadata={},
                timestamp=None,
            )
        ],
    )

    with patch("src.services.adk.tools.preload_memory_tool.memory_service.search_memory", new=AsyncMock(return_value=response)) as mock_search:
        result = await fn(tool_context=_tool_context())

    mock_search.assert_awaited_once_with(
        app_name="agent-1", user_id="user-1", query="", max_results=10, memory_base_config_id="config-1"
    )
    assert result["status"] == "success"
    assert result["total"] == 1


@pytest.mark.asyncio
async def test_preload_memory_reports_no_memories_when_search_returns_empty():
    tool = await create_preload_memory_tool()
    fn = tool.func

    empty_response = SearchMemoryResponse(memories=[])

    with patch("src.services.adk.tools.preload_memory_tool.memory_service.search_memory", new=AsyncMock(return_value=empty_response)):
        result = await fn(tool_context=_tool_context())

    assert result["status"] == "no_memories"
    assert result["memories"] == []
