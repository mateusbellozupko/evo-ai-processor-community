import pytest
from unittest.mock import AsyncMock, patch

from src.services.adk.runners.memory_preload import preload_memory


def _load_response(*contents):
    """Build a GET /memory/load response body (plain dicts, not pydantic models)."""
    return {
        "memories": [{"content": c, "timestamp": None, "metadata": {}} for c in contents],
        "total": len(contents),
        "query": "",
    }


@pytest.mark.asyncio
async def test_preload_memory_returns_none_when_disabled():
    result = await preload_memory(
        agent_config={"preload_memory": False, "load_memory": True},
        agent_id="agent-1",
        effective_user_id="user-1",
    )
    assert result is None


@pytest.mark.asyncio
async def test_preload_memory_returns_none_when_no_summaries_found():
    with patch(
        "src.services.adk.runners.memory_preload.memory_service.load_memory",
        new=AsyncMock(return_value=_load_response()),
    ):
        result = await preload_memory(
            agent_config={"preload_memory": True, "load_memory": True},
            agent_id="agent-1",
            effective_user_id="user-1",
        )

    assert result is None


@pytest.mark.asyncio
async def test_preload_memory_builds_system_event_from_summaries():
    response = _load_response("User likes dark mode.", "User is based in Brazil.")

    with patch(
        "src.services.adk.runners.memory_preload.memory_service.load_memory",
        new=AsyncMock(return_value=response),
    ) as mock_load:
        event = await preload_memory(
            agent_config={"preload_memory": True, "load_memory": True, "memory_base_config_id": "config-1"},
            agent_id="agent-1",
            effective_user_id="user-1",
        )

    mock_load.assert_awaited_once_with(
        app_name="agent-1", user_id="user-1", max_results=10, memory_base_config_id="config-1"
    )
    assert event is not None
    assert "User likes dark mode." in event.content.parts[0].text
    assert "User is based in Brazil." in event.content.parts[0].text


@pytest.mark.asyncio
async def test_preload_memory_uses_load_memory_not_search_memory():
    """Regression: preload must hit GET /memory/load (summaries only), not
    /memory/search with a blank query (which also returns raw short-term events)."""
    with patch(
        "src.services.adk.runners.memory_preload.memory_service.load_memory",
        new=AsyncMock(return_value=_load_response("Prior summary.")),
    ) as mock_load, patch(
        "src.services.adk.runners.memory_preload.memory_service.search_memory",
        new=AsyncMock(),
    ) as mock_search:
        event = await preload_memory(
            agent_config={"preload_memory": True, "load_memory": True},
            agent_id="agent-1",
            effective_user_id="user-1",
        )

    mock_load.assert_awaited_once()
    mock_search.assert_not_awaited()
    assert event is not None
