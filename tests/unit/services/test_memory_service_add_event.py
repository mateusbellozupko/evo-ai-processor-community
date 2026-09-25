import pytest
from unittest.mock import AsyncMock, patch

from src.services.memory_service import HttpMemoryService, set_memory_min_timestamp


@pytest.mark.asyncio
async def test_add_event_to_memory_posts_to_memory_event_endpoint():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    with patch("src.services.memory_service.http_client.do_post_json", new=AsyncMock(
        return_value={"id": 1}
    )) as mock_post:
        await service.add_event_to_memory(
            app_name="agent-1", user_id="user-1", role="user", content="hi"
        )

    assert mock_post.call_args.kwargs["url"] == "http://crm.test/api/v1/memory/event"
    assert mock_post.call_args.kwargs["payload"] == {
        "app_name": "agent-1", "user_id": "user-1", "role": "user", "content": "hi"
    }


@pytest.mark.asyncio
async def test_add_event_to_memory_falls_back_to_context_scoped_min_timestamp():
    """EVO-2241: the write path must see the same reopen boundary as the read
    path, so the controller's auto-compression trigger can restrict its
    source events to this epoch rather than folding in pre-reset events."""
    service = HttpMemoryService(base_url="http://crm.test/api/v1")
    set_memory_min_timestamp("2026-09-25T12:00:00Z")

    try:
        with patch("src.services.memory_service.http_client.do_post_json", new=AsyncMock(
            return_value={"id": 1}
        )) as mock_post:
            await service.add_event_to_memory(
                app_name="agent-1", user_id="user-1", role="user", content="hi",
                compression_interval=5,
            )

        assert mock_post.call_args.kwargs["payload"]["min_timestamp"] == "2026-09-25T12:00:00Z"
    finally:
        set_memory_min_timestamp(None)


@pytest.mark.asyncio
async def test_add_event_to_memory_omits_min_timestamp_when_never_scoped():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    with patch("src.services.memory_service.http_client.do_post_json", new=AsyncMock(
        return_value={"id": 1}
    )) as mock_post:
        await service.add_event_to_memory(
            app_name="agent-1", user_id="user-1", role="user", content="hi"
        )

    assert "min_timestamp" not in mock_post.call_args.kwargs["payload"]
