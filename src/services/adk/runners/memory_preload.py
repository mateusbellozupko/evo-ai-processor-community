"""
┌──────────────────────────────────────────────────────────────────────────────┐
│ @file: memory_preload.py                                                     │
│ Shared memory-preload logic for standard and streaming runners.              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Extracted so both StandardRunner and StreamingRunner call one tested         │
│ implementation instead of duplicating the inline preload block, which        │
│ previously referenced a non-existent settings.KNOWLEDGE_SERVICE_URL and      │
│ made a raw httpx call instead of going through the working memory_service    │
│ singleton.                                                                    │
└──────────────────────────────────────────────────────────────────────────────┘
"""

import time
from typing import Any, Optional

from google.adk.events import Event
from google.genai import types

from src.services.memory_service import memory_service
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_MAX_RESULTS = 10


async def preload_memory(
    agent_config: Any,
    agent_id: str,
    effective_user_id: str,
    session: Any,
) -> Optional[Event]:
    """Preload medium-term memory summaries before processing a user message.

    Calls the memory_service singleton (HttpMemoryService.search_memory) with
    an empty query, which returns medium-term summaries for the given
    app/user. Returns an Event to append to the session if summaries were
    found, None if memory preload is disabled or nothing was found.

    Args:
        agent_config: The agent's config dict (must have "preload_memory" and
            "load_memory" both truthy for preload to run).
        agent_id: Application name used as app_name in the memory search.
        effective_user_id: The user id to search memories for.
        session: The ADK session (unused directly here; kept in the
            signature since the caller has it in scope and may need it for
            future preload variants).

    Returns:
        An Event with a "system" role summarizing prior memories, or None.
    """
    if (
        not isinstance(agent_config, dict)
        or not agent_config.get("preload_memory")
        or not agent_config.get("load_memory")
    ):
        return None

    logger.info(f"Preloading memory for agent {agent_id}, user {effective_user_id}")

    memory_base_config_id = agent_config.get("memory_base_config_id")

    response = await memory_service.search_memory(
        app_name=agent_id,
        user_id=effective_user_id,
        query="",
        max_results=DEFAULT_MAX_RESULTS,
        memory_base_config_id=memory_base_config_id,
    )

    if not response.memories:
        logger.debug(
            f"No memory summaries found for preload (agent {agent_id}, user {effective_user_id})"
        )
        return None

    memory_context_parts = ["Previous conversation context:\n\n"]
    for idx, entry in enumerate(response.memories, 1):
        if entry.content and entry.content.parts:
            memory_context_parts.append(f"--- Summary {idx} ---\n")
            memory_context_parts.append(f"{entry.content.parts[0].text}\n")
            if entry.timestamp:
                memory_context_parts.append(f"(Date: {entry.timestamp})\n")
            memory_context_parts.append("\n")

    if len(memory_context_parts) <= 1:
        return None

    memory_context_text = "".join(memory_context_parts).strip()

    logger.info(f"Preloaded {len(response.memories)} memory summaries for agent {agent_id}")

    return Event(
        invocation_id=f"preload_memory_{int(time.time())}",
        author="system",
        content=types.Content(
            role="system",
            parts=[types.Part(text=memory_context_text)],
        ),
        timestamp=time.time(),
    )
