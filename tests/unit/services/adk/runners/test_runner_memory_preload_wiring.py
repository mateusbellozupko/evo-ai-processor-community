"""Runner-level wiring tests for the shared memory preload.

Covers the integration point that neither runner had a test for: that
run_agent / run_agent_stream actually call preload_memory() with the agent
config, agent id and effective user id, append the returned Event to the
session, and survive a preload failure without aborting the rest of the turn
(the knowledge preload below it must still run).

Both runners are driven only as far as the preload block: utils.create_content
is stubbed to return None, which makes each runner take its documented
"no meaningful content" early exit right after preload.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.adk.runners.standard_runner import StandardRunner
from src.services.adk.runners.streaming_runner import StreamingRunner

AGENT_CONFIG = {"preload_memory": True, "load_memory": True}


def _mock_utils():
    utils = MagicMock()
    utils.get_and_build_agent = AsyncMock(return_value=(MagicMock(), {}))
    utils.get_or_create_session = AsyncMock(return_value=MagicMock(name="session"))
    utils.setup_session_state = AsyncMock()
    utils.process_files = AsyncMock(return_value=([], []))
    utils.create_session_id = MagicMock(return_value="adk-session-1")
    utils.create_runner = MagicMock()
    # Force the runner's "no meaningful content" early return right after preload.
    utils.create_content = MagicMock(return_value=None)
    return utils


def _mock_memory_service():
    svc = MagicMock()
    svc.add_event_to_memory = AsyncMock()
    return svc


async def _drive(runner_cls, session_service, memory_service):
    """Run either runner up to its early exit, ignoring the produced output."""
    runner = runner_cls(db=MagicMock())
    runner.utils = _mock_utils()
    session = runner.utils.get_or_create_session.return_value

    kwargs = dict(
        agent_id="agent-1",
        external_id="external-1",
        message="hello",
        session_service=session_service,
        artifacts_service=MagicMock(),
        memory_service=memory_service,
        user_id="user-1",
    )

    if runner_cls is StandardRunner:
        await runner.run_agent(**kwargs)
    else:
        async for _ in runner.run_agent_stream(**kwargs):
            pass

    return session


def _patch_agent(agent_config=AGENT_CONFIG):
    agent = MagicMock()
    agent.config = agent_config
    return patch("src.services.agent_service.get_agent", new=AsyncMock(return_value=agent))


@pytest.mark.parametrize("runner_cls", [StandardRunner, StreamingRunner])
@pytest.mark.asyncio
async def test_runner_calls_preload_memory_and_appends_returned_event(runner_cls):
    session_service = MagicMock()
    session_service.append_event = AsyncMock()

    memory_event = MagicMock(name="memory_event")

    with _patch_agent(), patch(
        "src.services.adk.runners.memory_preload.preload_memory",
        new=AsyncMock(return_value=memory_event),
    ) as mock_preload:
        session = await _drive(runner_cls, session_service, _mock_memory_service())

    mock_preload.assert_awaited_once_with(
        agent_config=AGENT_CONFIG,
        agent_id="agent-1",
        effective_user_id="user-1",
    )
    session_service.append_event.assert_any_await(session, memory_event)


@pytest.mark.parametrize("runner_cls", [StandardRunner, StreamingRunner])
@pytest.mark.asyncio
async def test_runner_does_not_append_event_when_preload_returns_none(runner_cls):
    session_service = MagicMock()
    session_service.append_event = AsyncMock()

    with _patch_agent(), patch(
        "src.services.adk.runners.memory_preload.preload_memory",
        new=AsyncMock(return_value=None),
    ):
        await _drive(runner_cls, session_service, _mock_memory_service())

    session_service.append_event.assert_not_awaited()


@pytest.mark.parametrize("runner_cls", [StandardRunner, StreamingRunner])
@pytest.mark.asyncio
async def test_preload_memory_failure_is_isolated_from_knowledge_preload(runner_cls):
    """A memory preload failure must not propagate, must be logged at warning
    (not debug), and must not skip the knowledge preload that runs afterwards
    inside the same outer try."""
    session_service = MagicMock()
    session_service.append_event = AsyncMock()

    knowledge_config = dict(
        AGENT_CONFIG, preload_knowledge=True, load_knowledge=True, knowledge_max_results=1
    )

    # The project's setup_logger does not propagate to the root logger, so
    # caplog sees nothing; patch the runner module's logger instead.
    logger_target = f"{runner_cls.__module__}.logger"

    with _patch_agent(knowledge_config), patch(logger_target) as mock_logger, patch(
        "src.services.adk.runners.memory_preload.preload_memory",
        new=AsyncMock(side_effect=RuntimeError("memory service down")),
    ) as mock_preload:
        # Must not raise out of the runner.
        await _drive(runner_cls, session_service, _mock_memory_service())

    mock_preload.assert_awaited_once()

    warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
    infos = [c.args[0] for c in mock_logger.info.call_args_list]

    # Logged at warning, not swallowed by the outer debug-level handler.
    assert any("Could not preload memory: memory service down" in m for m in warnings)
    assert not any("Could not check preload config" in str(c) for c in mock_logger.debug.call_args_list)
    # Knowledge preload was still entered despite the memory preload blowing up.
    # (Knowledge preload has its own pre-existing breakage - settings has no
    # KNOWLEDGE_SERVICE_URL - so we assert it was reached, not that it succeeded.)
    assert any("Preloading knowledge for agent agent-1" in m for m in infos)
