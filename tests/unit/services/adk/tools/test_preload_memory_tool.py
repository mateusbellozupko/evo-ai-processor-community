import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.adk.tools.preload_memory_tool import create_preload_memory_tool


def _tool_context(app_name="agent-1", user_id="user-1"):
    ctx = MagicMock()
    ctx.session.app_name = app_name
    ctx.session.user_id = user_id
    return ctx


def _load_response(*contents):
    """Build a GET /memory/load response body (plain dicts, not pydantic models)."""
    return {
        "memories": [{"content": c, "timestamp": None, "metadata": {}} for c in contents],
        "total": len(contents),
        "query": "",
    }


@pytest.mark.asyncio
async def test_preload_memory_calls_load_memory_with_max_results():
    tool = await create_preload_memory_tool(memory_base_config_id="config-1", default_max_results=10)
    fn = tool.func

    with patch(
        "src.services.adk.tools.preload_memory_tool.memory_service.load_memory",
        new=AsyncMock(return_value=_load_response("Prior summary.")),
    ) as mock_load:
        result = await fn(tool_context=_tool_context())

    mock_load.assert_awaited_once_with(
        app_name="agent-1", user_id="user-1", max_results=10, memory_base_config_id="config-1"
    )
    assert result["status"] == "success"
    assert result["total"] == 1
    assert result["memories"][0]["content"] == "Prior summary."


@pytest.mark.asyncio
async def test_preload_memory_reports_no_memories_when_load_returns_empty():
    tool = await create_preload_memory_tool()
    fn = tool.func

    with patch(
        "src.services.adk.tools.preload_memory_tool.memory_service.load_memory",
        new=AsyncMock(return_value=_load_response()),
    ):
        result = await fn(tool_context=_tool_context())

    assert result["status"] == "no_memories"
    assert result["memories"] == []


@pytest.mark.asyncio
async def test_preload_memory_tool_uses_load_memory_not_search_memory():
    """Regression: the tool must hit GET /memory/load (summaries only), not
    /memory/search with a blank query (which also returns raw short-term events)."""
    tool = await create_preload_memory_tool()
    fn = tool.func

    with patch(
        "src.services.adk.tools.preload_memory_tool.memory_service.load_memory",
        new=AsyncMock(return_value=_load_response("Prior summary.")),
    ) as mock_load, patch(
        "src.services.adk.tools.preload_memory_tool.memory_service.search_memory",
        new=AsyncMock(),
    ) as mock_search:
        result = await fn(tool_context=_tool_context())

    mock_load.assert_awaited_once()
    mock_search.assert_not_awaited()
    assert result["status"] == "success"
