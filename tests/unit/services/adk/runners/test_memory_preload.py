import pytest
from unittest.mock import AsyncMock, patch

from src.services.adk.runners.memory_preload import preload_memory
from src.services.memory_service import SearchMemoryResponse, MemoryEntry
from google.genai import types


@pytest.mark.asyncio
async def test_preload_memory_returns_none_when_disabled():
    result = await preload_memory(
        agent_config={"preload_memory": False, "load_memory": True},
        agent_id="agent-1",
        effective_user_id="user-1",
        session=None,
    )
    assert result is None


@pytest.mark.asyncio
async def test_preload_memory_returns_none_when_no_summaries_found():
    # NOTE: google.adk's MemoryEntry/SearchMemoryResponse are pydantic models
    # whose real fields differ from the plan brief's snippet. The real fields are:
    #   MemoryEntry: content, custom_metadata, id, author, timestamp (no "score")
    #   SearchMemoryResponse: memories only (no "total"/"query")
    # Confirmed against src/services/memory_service.py and the installed
    # google-adk package (same correction already applied in sibling tasks
    # 5/6's tests, e.g. test_preload_memory_tool.py).
    empty_response = SearchMemoryResponse(memories=[])

    with patch("src.services.adk.runners.memory_preload.memory_service.search_memory", new=AsyncMock(return_value=empty_response)):
        result = await preload_memory(
            agent_config={"preload_memory": True, "load_memory": True},
            agent_id="agent-1",
            effective_user_id="user-1",
            session=None,
        )

    assert result is None


@pytest.mark.asyncio
async def test_preload_memory_builds_system_event_from_summaries():
    response = SearchMemoryResponse(
        memories=[
            MemoryEntry(
                content=types.Content(role="user", parts=[types.Part(text="User likes dark mode.")]),
                custom_metadata={},
                timestamp=None,
            ),
            MemoryEntry(
                content=types.Content(role="user", parts=[types.Part(text="User is based in Brazil.")]),
                custom_metadata={},
                timestamp=None,
            ),
        ],
    )

    with patch("src.services.adk.runners.memory_preload.memory_service.search_memory", new=AsyncMock(return_value=response)) as mock_search:
        event = await preload_memory(
            agent_config={"preload_memory": True, "load_memory": True, "memory_base_config_id": "config-1"},
            agent_id="agent-1",
            effective_user_id="user-1",
            session=None,
        )

    mock_search.assert_awaited_once_with(app_name="agent-1", user_id="user-1", query="", max_results=10, memory_base_config_id="config-1")
    assert event is not None
    assert "User likes dark mode." in event.content.parts[0].text
    assert "User is based in Brazil." in event.content.parts[0].text
