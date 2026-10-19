"""A host can choose the reasoning effort of each turn from the message that opens it.

"Thanks" does not need the effort a refactoring does. A host that sorted each
message into a level had to rebuild the engine around the answer; the core now
asks an :class:`IReasoningEffortClassifier` once per turn, before the first
request, and runs the turn at what it chose. Without one, nothing changes.
"""
from __future__ import annotations

import pytest

from protocore.contracts.reasoning_effort import (
    EffortRequest,
    IReasoningEffortClassifier,
)
from protocore.runtime.events import EventType

from .conftest import ScenarioFactory, ScriptedTool

pytestmark = pytest.mark.asyncio


class _Sorter:
    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.asked: list[EffortRequest] = []

    async def choose_effort(self, request: EffortRequest) -> str | None:
        self.asked.append(request)
        return self.answer


class _Broken:
    async def choose_effort(self, request: EffortRequest) -> str | None:
        raise TimeoutError("the small model did not answer")


def _efforts(run: object) -> list[object]:
    return [request.extra.get("reasoning_effort") for request in run.requests]  # type: ignore[attr-defined]


async def _turn(scenario: ScenarioFactory, classifier: object, **config: object):  # type: ignore[no-untyped-def]
    run = scenario(
        tools=[ScriptedTool(tool_name="Note")],
        reasoning_effort_classifier=classifier,
        reasoning_effort="medium",
        **config,
    )
    run.llm.queue_tool_call_response(tool_call_id="c1", tool_name="Note", tool_input={"v": "x"})
    run.llm.queue_response(text="done")
    await run.run("refactor the whole module")
    return run


async def test_the_protocol_is_checkable() -> None:
    assert isinstance(_Sorter("low"), IReasoningEffortClassifier)


async def test_the_turn_runs_at_the_chosen_effort(scenario: ScenarioFactory) -> None:
    sorter = _Sorter("high")
    run = await _turn(scenario, sorter)

    assert _efforts(run) == ["high", "high"]
    [asked] = sorter.asked
    assert asked.base_effort == "medium"
    assert asked.message is not None and asked.message.text == "refactor the whole module"
    assert asked.run_id == "run-scenario"


async def test_the_choice_is_announced(scenario: ScenarioFactory) -> None:
    run = await _turn(scenario, _Sorter("high"))

    chosen = [
        event
        for event in run.events_of(EventType.STATE_CHANGED)
        if event.payload.get("reason") == "reasoning_effort_chosen"
    ]
    assert len(chosen) == 1
    assert chosen[0].payload["reasoning_effort"] == "high"
    assert chosen[0].payload["base_reasoning_effort"] == "medium"


async def test_no_opinion_keeps_the_configured_effort(scenario: ScenarioFactory) -> None:
    run = await _turn(scenario, _Sorter(None))

    assert _efforts(run) == ["medium", "medium"]


async def test_an_effort_the_core_does_not_know_is_ignored(scenario: ScenarioFactory) -> None:
    run = await _turn(scenario, _Sorter("turbo"))

    assert run.engine.state.value == "completed"
    assert _efforts(run) == ["medium", "medium"]


async def test_a_classifier_that_fails_costs_the_turn_nothing(scenario: ScenarioFactory) -> None:
    run = await _turn(scenario, _Broken())

    assert run.engine.state.value == "completed"
    assert _efforts(run) == ["medium", "medium"]


async def test_a_live_choice_of_effort_wins_and_the_classifier_is_not_asked(
    scenario: ScenarioFactory,
) -> None:
    sorter = _Sorter("low")
    run = scenario(
        tools=[ScriptedTool(tool_name="Note")],
        reasoning_effort_classifier=sorter,
        reasoning_effort="medium",
    )
    run.engine.apply_live_controls(reasoning_effort="xhigh")
    run.llm.queue_response(text="done")
    await run.run("hello")

    assert sorter.asked == []
    assert _efforts(run) == ["xhigh"]


async def test_the_choice_travels_with_the_snapshot(scenario: ScenarioFactory) -> None:
    run = await _turn(scenario, _Sorter("high"))

    assert run.engine.snapshot()["classified_reasoning_effort"] == "high"


async def test_without_a_classifier_nothing_changes(scenario: ScenarioFactory) -> None:
    run = scenario(tools=[ScriptedTool(tool_name="Note")], reasoning_effort="medium")
    run.llm.queue_response(text="done")
    await run.run("hello")

    assert _efforts(run) == ["medium"]
    assert not [
        event
        for event in run.events_of(EventType.STATE_CHANGED)
        if event.payload.get("reason") == "reasoning_effort_chosen"
    ]


async def test_a_re_armed_engine_sorts_its_next_turn_again(scenario: ScenarioFactory) -> None:
    sorter = _Sorter("high")
    run = scenario(reasoning_effort_classifier=sorter, reasoning_effort="medium")
    run.llm.queue_response(text="done")
    await run.run("refactor the module")

    run.engine.rearm()
    sorter.answer = "minimal"
    run.llm.queue_response(text="you are welcome")
    await run.run("thanks")

    assert [asked.message.text for asked in sorter.asked if asked.message] == [
        "refactor the module",
        "thanks",
    ]
    assert _efforts(run) == ["high", "minimal"]
