"""Regression test for `_find_pipeline_item_id`'s response-shape parsing.

2026-09-22 incident: the function read `items_response.get("payload", [])`,
but `GET /pipelines/{id}/pipeline_items` actually wraps the list under
`"data"` (`{"success": true, "data": [...], "meta": {...}}`) — the "payload"
key belongs to a different endpoint shape (e.g. conversation labels). This
silently returned an empty list every time, so `_find_pipeline_item_id`
always returned `None` even when the item existed, breaking every
custom_fields-on-move and create_task call. The existing
`test_pipeline_tool_context_ids.py` suite never caught this because it
mocks `_move_to_stage` itself, never exercising the real HTTP-response
parsing in `_find_pipeline_item_id`.
"""

import pytest

from src.services.adk.tools.evo_crm.pipeline_manipulation import (
    _find_pipeline_item_id,
)

PIPELINE_ID = "53de6cd9-c550-4ca3-bec6-14550eae4b80"
CONV_ID = "f869c393-b62c-45d0-86e1-9f1b4f15875b"
ITEM_ID = "a08094ef-1e42-43ba-a19d-1171524179fd"


class _FakeClient:
    """Minimal stand-in for EvoCrmClient — only `.get()` is exercised here."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    async def get(self, endpoint, **kwargs):
        self.calls.append(endpoint)
        return self._response


@pytest.mark.asyncio
async def test_finds_the_item_in_the_real_api_response_shape():
    """The actual shape returned by GET /pipelines/{id}/pipeline_items."""
    client = _FakeClient({
        "success": True,
        "data": [
            {"id": "other-item", "conversation_id": "some-other-conversation"},
            {"id": ITEM_ID, "conversation_id": CONV_ID},
        ],
        "meta": {"timestamp": "2026-09-22T08:19:00Z"},
    })

    result = await _find_pipeline_item_id(client, PIPELINE_ID, CONV_ID)

    assert result == ITEM_ID
    assert client.calls == [f"/pipelines/{PIPELINE_ID}/pipeline_items"]


@pytest.mark.asyncio
async def test_returns_none_when_conversation_is_not_in_the_pipeline():
    client = _FakeClient({
        "success": True,
        "data": [{"id": "other-item", "conversation_id": "some-other-conversation"}],
    })

    assert await _find_pipeline_item_id(client, PIPELINE_ID, CONV_ID) is None


@pytest.mark.asyncio
async def test_does_not_match_on_the_wrong_key():
    """Guards the actual regression: a 'payload'-shaped response must not be
    silently treated as a match source (nor crash) — 'data' is the only key
    this function should read.
    """
    client = _FakeClient({
        "payload": [{"id": ITEM_ID, "conversation_id": CONV_ID}],
    })

    assert await _find_pipeline_item_id(client, PIPELINE_ID, CONV_ID) is None
