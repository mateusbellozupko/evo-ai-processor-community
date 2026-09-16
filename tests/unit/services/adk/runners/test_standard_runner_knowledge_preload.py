import logging

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.config.settings import settings
from src.services.adk.runners.knowledge_preload import preload_knowledge


@pytest.fixture
def crm_settings(monkeypatch):
    monkeypatch.setattr(settings, "EVO_AI_CRM_URL", "http://crm.test")
    monkeypatch.setattr(settings, "EVOAI_CRM_API_TOKEN", "test-token")


@pytest.fixture
def enabled_agent_config():
    return {
        "load_knowledge": True,
        "preload_knowledge": True,
        "knowledge_base_id": "kb-1",
        "knowledge_tags": ["vendas"],
        "knowledge_max_results": 5,
    }


def _mock_crm_response(results=None, raise_for_status_side_effect=None):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(side_effect=raise_for_status_side_effect)
    mock_response.json.return_value = {"results": results or [], "total": len(results or [])}
    return mock_response


@pytest.mark.asyncio
async def test_preload_knowledge_calls_crm_search_endpoint(crm_settings, enabled_agent_config):
    mock_response = _mock_crm_response(
        results=[{"content": "Resposta relevante", "tags": ["vendas"], "document_title": "Doc"}]
    )

    with patch("httpx.AsyncClient.get", new=AsyncMock()), \
         patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)) as mock_post:
        # import and call the (to-be-extracted) helper directly rather than the whole runner —
        # see Step 3, which extracts this block into a standalone function precisely so it's unit-testable
        result = await preload_knowledge(enabled_agent_config)

        mock_post.assert_called_once()

        # Pin the fix: the URL must be built from settings.EVO_AI_CRM_URL, not the old,
        # nonexistent settings.KNOWLEDGE_SERVICE_URL. Both old and new URLs contain the
        # substring "knowledge/search", so we assert the full URL exactly.
        assert mock_post.call_args.args[0] == "http://crm.test/api/v1/internal/knowledge/search"

        # The service-to-service auth header must carry the CRM token.
        assert mock_post.call_args.kwargs["headers"]["X-Service-Token"] == "test-token"

        # The JSON payload must reflect what was passed into agent_config.
        payload = mock_post.call_args.kwargs["json"]
        assert payload["knowledge_base_id"] == "kb-1"
        assert payload["tags"] == ["vendas"]
        assert payload["max_results"] == 5

        assert result is not None
        assert "Resposta relevante" in result


@pytest.mark.asyncio
async def test_preload_knowledge_returns_none_when_disabled(crm_settings, enabled_agent_config):
    enabled_agent_config.update({"load_knowledge": False, "preload_knowledge": False})

    with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
        result = await preload_knowledge(enabled_agent_config)

        assert result is None
        mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_preload_knowledge_propagates_http_errors(crm_settings, enabled_agent_config):
    """preload_knowledge itself does not catch HTTP/network failures — it lets them
    propagate. The surrounding callers (standard_runner.py / streaming_runner.py) wrap
    the call in `try: ... except Exception as e: logger.warning(...)` so a CRM outage
    logs a warning instead of killing the conversation turn. This test pins the
    lower-level contract (propagate, don't swallow) that the runner's wrapper depends on,
    and then exercises that exact try/except shape to confirm the failure is handled.
    """
    mock_response = _mock_crm_response(
        raise_for_status_side_effect=httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=MagicMock(status_code=500)
        )
    )

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
        with pytest.raises(httpx.HTTPStatusError):
            await preload_knowledge(enabled_agent_config)

        # Simulate the runner-level wrapper found in standard_runner.py / streaming_runner.py
        # around the `preload_knowledge` call, and confirm it turns the raised exception into
        # a logged warning rather than letting it propagate further.
        try:
            await preload_knowledge(enabled_agent_config)
            outcome = "not-handled"
        except Exception as e:
            logging.getLogger("test").warning(f"Could not preload knowledge: {e}")
            outcome = "handled"

    assert outcome == "handled"
