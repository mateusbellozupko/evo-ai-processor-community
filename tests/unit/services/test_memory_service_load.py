import pytest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

from src.services.memory_service import HttpMemoryService
from src.utils.http import HttpError


@pytest.mark.asyncio
async def test_load_memory_gets_memory_load_endpoint():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    payload = {
        "memories": [{"content": "Prior summary.", "timestamp": "2026-09-17T00:00:00Z", "metadata": {}}],
        "total": 1,
        "query": "",
    }

    with patch("src.services.memory_service.http_client.do_get_json", new=AsyncMock(
        return_value=payload
    )) as mock_get:
        result = await service.load_memory(
            app_name="agent-1", user_id="user-1", max_results=5, memory_base_config_id="config-1"
        )

    mock_get.assert_awaited_once()
    call_kwargs = mock_get.call_args.kwargs
    # do_get_json takes no params argument, so the query string is in the URL.
    parsed = urlsplit(call_kwargs["url"])
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "http://crm.test/api/v1/memory/load"
    assert parse_qs(parsed.query) == {
        "app_name": ["agent-1"],
        "user_id": ["user-1"],
        "max_results": ["5"],
    }
    assert call_kwargs["expected_status"] == 200
    assert call_kwargs["headers"]["x-memory-base-config-id"] == "config-1"
    assert result == payload


@pytest.mark.asyncio
async def test_load_memory_returns_empty_result_on_http_error_without_raising():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    with patch("src.services.memory_service.http_client.do_get_json", new=AsyncMock(
        side_effect=HttpError("boom", status_code=502)
    )):
        result = await service.load_memory(app_name="agent-1", user_id="user-1")

    assert result == {"memories": [], "total": 0}


@pytest.mark.asyncio
async def test_load_memory_sends_explicit_min_timestamp():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    with patch("src.services.memory_service.http_client.do_get_json", new=AsyncMock(
        return_value={"memories": [], "total": 0, "query": ""}
    )) as mock_get:
        await service.load_memory(
            app_name="agent-1", user_id="user-1", min_timestamp="2026-09-25T12:00:00Z"
        )

    parsed = urlsplit(mock_get.call_args.kwargs["url"])
    assert parse_qs(parsed.query)["min_timestamp"] == ["2026-09-25T12:00:00Z"]


@pytest.mark.asyncio
async def test_load_memory_falls_back_to_context_scoped_min_timestamp():
    from src.services.memory_service import set_memory_min_timestamp

    service = HttpMemoryService(base_url="http://crm.test/api/v1")
    set_memory_min_timestamp("2026-09-25T12:00:00Z")

    try:
        with patch("src.services.memory_service.http_client.do_get_json", new=AsyncMock(
            return_value={"memories": [], "total": 0, "query": ""}
        )) as mock_get:
            await service.load_memory(app_name="agent-1", user_id="user-1")

        parsed = urlsplit(mock_get.call_args.kwargs["url"])
        assert parse_qs(parsed.query)["min_timestamp"] == ["2026-09-25T12:00:00Z"]
    finally:
        set_memory_min_timestamp(None)


@pytest.mark.asyncio
async def test_load_memory_omits_min_timestamp_when_never_scoped():
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    with patch("src.services.memory_service.http_client.do_get_json", new=AsyncMock(
        return_value={"memories": [], "total": 0, "query": ""}
    )) as mock_get:
        await service.load_memory(app_name="agent-1", user_id="user-1")

    parsed = urlsplit(mock_get.call_args.kwargs["url"])
    assert "min_timestamp" not in parse_qs(parsed.query)
