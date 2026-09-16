import time
import httpx
from src.config.settings import settings
from google.adk.events import Event
from google.genai.types import Content, Part


async def preload_knowledge(agent_config: dict) -> str | None:
    """Fetch knowledge entries relevant to preload for this agent and return
    a formatted context string, or None if nothing was found/configured."""
    if not (agent_config.get("preload_knowledge") and agent_config.get("load_knowledge")):
        return None

    knowledge_base_id = agent_config.get("knowledge_base_id")
    if not knowledge_base_id:
        return None

    knowledge_tags = agent_config.get("knowledge_tags") or []
    max_results = agent_config.get("knowledge_max_results", 5)

    base_url = settings.EVO_AI_CRM_URL.rstrip("/")
    url = f"{base_url}/api/v1/internal/knowledge/search"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.EVOAI_CRM_API_TOKEN:
        headers["X-Service-Token"] = settings.EVOAI_CRM_API_TOKEN

    payload = {
        "knowledge_base_id": knowledge_base_id,
        "query": "",  # empty query: caller decides whether this should preload "everything recent"
        # or be skipped entirely for preload — matches the pre-existing preload_memory contract
        # of "empty query loads a default summary set". If the CRM's search endpoint requires a
        # non-empty query, change this to a fixed prompt like "general overview" instead — decide
        # this in the same PR that flips EVO_AI_CRM_URL live in production, not before.
        "tags": knowledge_tags,
        "max_results": max_results,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    if not results:
        return None

    parts = ["Preloaded knowledge base context:\n"]
    for idx, result in enumerate(results, 1):
        parts.append(f"--- Knowledge Entry {idx} ---")
        if result.get("document_title"):
            parts.append(f"Source: {result['document_title']}")
        parts.append(f"Content: {result.get('content', '')}\n")

    return "\n".join(parts).strip()


def build_knowledge_event(context_text: str) -> Event:
    return Event(
        invocation_id=f"preload_knowledge_{int(time.time())}",
        author="system",
        content=Content(role="system", parts=[Part(text=context_text)]),
        timestamp=time.time(),
    )
