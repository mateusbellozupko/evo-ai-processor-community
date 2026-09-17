import pytest
from unittest.mock import AsyncMock, patch

from src.services.memory_service import HttpMemoryService
from src.utils.http import HttpError


@pytest.mark.asyncio
async def test_compress_memory_posts_to_memory_compress_endpoint():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    with patch("src.services.memory_service.http_client.do_post_json", new=AsyncMock(
        return_value={"success": True, "messages_compressed": 10, "summary_content": "Summary."}
    )) as mock_post:
        result = await service.compress_memory(app_name="agent-1", user_id="user-1", force=True, compression_interval=5)

    mock_post.assert_awaited_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == "http://crm.test/api/v1/memory/compress"
    assert call_kwargs["payload"] == {"app_name": "agent-1", "user_id": "user-1", "force": True, "compression_interval": 5}
    assert result == {"success": True, "messages_compressed": 10, "summary_content": "Summary."}


@pytest.mark.asyncio
async def test_compress_memory_returns_failure_dict_on_http_error_without_raising():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    with patch("src.services.memory_service.http_client.do_post_json", new=AsyncMock(
        side_effect=HttpError("boom", status_code=502)
    )):
        result = await service.compress_memory(app_name="agent-1", user_id="user-1")

    assert result == {"success": False, "messages_compressed": 0, "message": "boom"}
