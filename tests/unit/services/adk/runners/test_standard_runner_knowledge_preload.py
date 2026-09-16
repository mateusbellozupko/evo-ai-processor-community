import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx


@pytest.mark.asyncio
async def test_preload_knowledge_calls_crm_search_endpoint(monkeypatch):
    from src.config.settings import settings

    monkeypatch.setattr(settings, "EVO_AI_CRM_URL", "http://crm.test")
    monkeypatch.setattr(settings, "EVOAI_CRM_API_TOKEN", "test-token")

    agent_config = {
        "load_knowledge": True,
        "preload_knowledge": True,
        "knowledge_base_id": "kb-1",
        "knowledge_tags": ["vendas"],
        "knowledge_max_results": 5,
    }

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "results": [{"content": "Resposta relevante", "tags": ["vendas"], "document_title": "Doc"}],
        "total": 1,
    }

    with patch("httpx.AsyncClient.get", new=AsyncMock()) as mock_get, \
         patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)) as mock_post:
        # import and call the (to-be-extracted) helper directly rather than the whole runner —
        # see Step 3, which extracts this block into a standalone function precisely so it's unit-testable
        from src.services.adk.runners.knowledge_preload import preload_knowledge

        result = await preload_knowledge(agent_config)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "knowledge/search" in call_kwargs.args[0] or "knowledge/search" in str(call_kwargs)
        assert result is not None
        assert "Resposta relevante" in result
