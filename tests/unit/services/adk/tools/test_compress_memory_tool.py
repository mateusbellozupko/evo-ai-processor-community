import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.adk.tools.compress_memory_tool import create_compress_memory_tool


COMPRESS_TARGET = "src.services.adk.tools.compress_memory_tool.memory_service.compress_memory"


def _patch_compress(result):
    """Patch the tool's memory_service.compress_memory to return `result`."""
    return patch(COMPRESS_TARGET, new=AsyncMock(return_value=result))


def _tool_context(app_name="agent-1", user_id="user-1"):
    ctx = MagicMock()
    ctx.session.app_name = app_name
    ctx.session.user_id = user_id
    return ctx


@pytest.mark.asyncio
async def test_compress_memory_calls_memory_service_compress_memory():
    tool = await create_compress_memory_tool(memory_base_config_id="config-1")
    fn = tool.func

    ok_result = {"success": True, "messages_compressed": 10, "summary_content": "Summary.", "summary_id": "abc"}

    with _patch_compress(ok_result) as mock_compress:
        result = await fn(force=True, tool_context=_tool_context())

    mock_compress.assert_awaited_once_with(
        app_name="agent-1", user_id="user-1", force=True, compression_interval=None, memory_base_config_id="config-1"
    )
    assert result["status"] == "success"
    assert result["messages_compressed"] == 10
    assert result["summary_id"] == "abc"


@pytest.mark.asyncio
async def test_compress_memory_reports_error_status_when_service_reports_failure():
    tool = await create_compress_memory_tool()
    fn = tool.func

    failure_result = {"success": False, "messages_compressed": 0, "message": "Not enough events to compress"}

    with _patch_compress(failure_result):
        result = await fn(tool_context=_tool_context())

    assert result["status"] == "error"
    assert result["messages_compressed"] == 0
    # "error" also covers the not-enough-events case; "message" carries the
    # distinction because the endpoint's failure dicts are shape-identical.
    assert result["message"] == "Not enough events to compress"


@pytest.mark.asyncio
async def test_compress_memory_passes_per_agent_compression_interval():
    tool = await create_compress_memory_tool(memory_base_config_id="config-1", compression_interval=25)
    fn = tool.func

    ok_result = {"success": True, "messages_compressed": 25}

    with _patch_compress(ok_result) as mock_compress:
        await fn(tool_context=_tool_context())

    assert mock_compress.call_args.kwargs["compression_interval"] == 25
