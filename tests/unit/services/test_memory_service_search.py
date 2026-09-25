import pytest
from unittest.mock import AsyncMock, patch

from src.services.memory_service import HttpMemoryService, set_memory_min_timestamp


@pytest.mark.asyncio
async def test_search_memory_sends_explicit_min_timestamp():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    with patch("src.services.memory_service.http_client.do_post_json", new=AsyncMock(
        return_value={"memories": [], "total": 0, "query": "q"}
    )) as mock_post:
        await service.search_memory(
            app_name="agent-1", user_id="user-1", query="q", min_timestamp="2026-09-25T12:00:00Z"
        )

    assert mock_post.call_args.kwargs["payload"]["min_timestamp"] == "2026-09-25T12:00:00Z"


@pytest.mark.asyncio
async def test_search_memory_falls_back_to_context_scoped_min_timestamp():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")
    set_memory_min_timestamp("2026-09-25T12:00:00Z")

    try:
        with patch("src.services.memory_service.http_client.do_post_json", new=AsyncMock(
            return_value={"memories": [], "total": 0, "query": "q"}
        )) as mock_post:
            await service.search_memory(app_name="agent-1", user_id="user-1", query="q")

        assert mock_post.call_args.kwargs["payload"]["min_timestamp"] == "2026-09-25T12:00:00Z"
    finally:
        set_memory_min_timestamp(None)


@pytest.mark.asyncio
async def test_search_memory_omits_min_timestamp_when_never_scoped():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    with patch("src.services.memory_service.http_client.do_post_json", new=AsyncMock(
        return_value={"memories": [], "total": 0, "query": "q"}
    )) as mock_post:
        await service.search_memory(app_name="agent-1", user_id="user-1", query="q")

    assert "min_timestamp" not in mock_post.call_args.kwargs["payload"]
