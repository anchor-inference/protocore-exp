"""A host's turn context rides after the cache-stable prefix and nowhere else.

The clock, a board, a mode switched a minute ago: what changes between
requests used to go into the system prompt, where it moved the first bytes of
every request and cost the cache of the whole history behind them, or into a
stored message, where it stayed for good. The core now appends it as the last
message of each request and keeps it out of the history, the snapshot and the
cache breakpoints.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.observability import model_visible
from protocore.contracts.turn_context import (
    TURN_CONTEXT_METADATA_KEY,
    VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY,
    ITurnContextProvider,
    TurnContextRequest,
)
from protocore.contracts.types import MessageRole, TextBlock

from .conftest import ScenarioFactory, ScriptedTool

pytestmark = pytest.mark.asyncio


class _Clock:
    """A turn context that changes on every request, like a clock."""

    def __init__(self) -> None:
        self.asked: list[TurnContextRequest] = []

    async def turn_context(self, request: TurnContextRequest) -> str | None:
        self.asked.append(request)
        return f"<turn-context>request {len(self.asked)}</turn-context>"


class _Broken:
    async def turn_context(self, request: TurnContextRequest) -> str | None:
        raise RuntimeError("the board is unreachable")


class _Silent:
    async def turn_context(self, request: TurnContextRequest) -> str | None:
        return None


def _last_text(request: Any) -> str:
    last = request.messages[-1]
    return "".join(block.text for block in last.content_blocks if isinstance(block, TextBlock))


async def _two_requests(scenario: ScenarioFactory, provider: object):  # type: ignore[no-untyped-def]
    run = scenario(tools=[ScriptedTool(tool_name="Note")], turn_context_provider=provider)
    run.llm.queue_tool_call_response(tool_call_id="c1", tool_name="Note", tool_input={"v": "x"})
    run.llm.queue_response(text="done")
    await run.run("what time is it")
    return run


async def test_the_provider_protocol_is_checkable() -> None:
    assert isinstance(_Clock(), ITurnContextProvider)


async def test_each_request_ends_with_the_turn_context(scenario: ScenarioFactory) -> None:
    clock = _Clock()
    run = await _two_requests(scenario, clock)

    first, second = run.requests
    assert _last_text(first) == "<turn-context>request 1</turn-context>"
    assert _last_text(second) == "<turn-context>request 2</turn-context>"
    assert first.messages[-1].role is MessageRole.user
    assert first.messages[-1].metadata[TURN_CONTEXT_METADATA_KEY] is True
    assert first.extra[VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY] == 1
    assert [asked.run_id for asked in clock.asked] == ["run-scenario", "run-scenario"]
    assert clock.asked[0].model == "scenario-model"


async def test_the_prefix_in_front_of_it_is_unchanged_between_requests(
    scenario: ScenarioFactory,
) -> None:
    run = await _two_requests(scenario, _Clock())

    first, second = run.requests
    stable_first = [model_visible(message) for message in first.messages[:-1]]
    stable_second = [model_visible(message) for message in second.messages[:-1]]
    assert stable_second[: len(stable_first)] == stable_first


async def test_it_never_enters_the_history(scenario: ScenarioFactory) -> None:
    run = await _two_requests(scenario, _Clock())

    assert not any("turn-context" in text for text in run.history_texts())
    assert not any(
        message.metadata.get(TURN_CONTEXT_METADATA_KEY) for message in run.engine.history_snapshot()
    )
    assert "turn-context" not in str(run.engine.snapshot())


async def test_no_cache_breakpoint_is_placed_on_it(scenario: ScenarioFactory) -> None:
    run = await _two_requests(scenario, _Clock())

    for request in run.requests:
        last = len(request.messages) - 1
        breakpoints = request.extra["cache_breakpoints"]
        assert breakpoints
        assert all(point.message_index < last for point in breakpoints)


async def test_a_provider_that_fails_costs_the_request_nothing(scenario: ScenarioFactory) -> None:
    run = await _two_requests(scenario, _Broken())

    assert run.engine.state.value == "completed"
    for request in run.requests:
        assert VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY not in request.extra
        assert not request.messages[-1].metadata.get(TURN_CONTEXT_METADATA_KEY)


async def test_nothing_is_appended_when_there_is_nothing_to_say(scenario: ScenarioFactory) -> None:
    run = await _two_requests(scenario, _Silent())

    for request in run.requests:
        assert VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY not in request.extra


async def test_without_a_provider_requests_are_as_before(scenario: ScenarioFactory) -> None:
    run = scenario(tools=[ScriptedTool(tool_name="Note")])
    run.llm.queue_response(text="done")
    await run.run("hello")

    [request] = run.requests
    assert VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY not in request.extra
    assert _last_text(request) == "hello"
