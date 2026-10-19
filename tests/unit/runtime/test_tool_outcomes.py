"""The rules a run applies with an outcome store, one at a time.

The crash scenarios in ``tests/scenarios/test_tool_outcome_replay.py`` drive
these end to end; here each rule is pinned on its own, so a regression names
the rule it broke.
"""
from __future__ import annotations

from protocore.contracts.tool_outcome import ToolInvocationRecord
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.events import EventType
from protocore.runtime.intent import (
    DISPATCHED,
    PAUSED_ASK_USER,
    commit_intent,
    find_intent,
    fingerprint_arguments,
)
from protocore.runtime.query import (
    _discard_answered_tool_outcomes,
    _replay_recorded_tool_outcomes,
)
from protocore.runtime.tool_outcomes import (
    InMemoryToolOutcomeStore,
    new_invocation_id,
    place_result_in_source_order,
    unanswered_tool_uses,
)


def _batch(*call_ids: str) -> list[Message]:
    return [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")]),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id=call_id, name="Send", arguments_json="{}")
                for call_id in call_ids
            ],
        ),
    ]


def _answered_ids(history: list[Message]) -> list[str]:
    return [
        block.tool_call_id
        for message in history
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]


def test_results_replayed_out_of_order_land_in_the_order_they_were_asked() -> None:
    history = _batch("a", "b", "c")
    for call_id in ("c", "a", "b"):
        assert place_result_in_source_order(
            history, ToolResultBlock(tool_call_id=call_id, content=call_id)
        )
    assert _answered_ids(history) == ["a", "b", "c"]


def test_a_result_for_a_call_history_does_not_hold_is_not_placed() -> None:
    history = _batch("a")
    assert not place_result_in_source_order(
        history, ToolResultBlock(tool_call_id="zz", content="x")
    )
    assert _answered_ids(history) == []


def test_only_calls_without_a_result_are_unanswered() -> None:
    history = _batch("a", "b")
    history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="a", content="done")],
        )
    )
    assert sorted(unanswered_tool_uses(history)) == ["b"]


def test_invocation_ids_do_not_repeat() -> None:
    assert new_invocation_id() != new_invocation_id()


def _started(call_id: str, **overrides: object) -> ToolInvocationRecord:
    values: dict[str, object] = {
        "invocation_id": f"inv-{call_id}",
        "tenant_id": "tenant-test",
        "run_id": "run-test",
        "tool_call_id": call_id,
        "tool_name": "Send",
        "arguments_digest": fingerprint_arguments("Send", {}),
    }
    values.update(overrides)
    return ToolInvocationRecord(**values)  # type: ignore[arg-type]


async def test_a_started_call_with_no_intent_is_given_one_in_flight(engine_factory) -> None:
    """A snapshot older than the start still gets the honest unknown outcome."""
    store = InMemoryToolOutcomeStore()
    await store.record_started(_started("a"))
    engine = engine_factory()
    engine.tool_outcome_store = store
    engine.history.extend(_batch("a"))

    events = [evt async for evt in _replay_recorded_tool_outcomes(engine)]

    assert events == []
    intent = find_intent(engine.open_intents, "a")
    assert intent is not None and intent.state == DISPATCHED


async def test_a_call_waiting_on_a_question_is_left_waiting(engine_factory) -> None:
    store = InMemoryToolOutcomeStore()
    await store.record_started(_started("q"))
    engine = engine_factory()
    engine.tool_outcome_store = store
    engine.history.extend(_batch("q"))
    intent = commit_intent(
        tool_name="Send", tool_call_id="q", rc=engine.config.rc, state=PAUSED_ASK_USER
    )
    engine.open_intents.append(intent)

    _ = [evt async for evt in _replay_recorded_tool_outcomes(engine)]
    await _discard_answered_tool_outcomes(engine)

    assert intent.state == PAUSED_ASK_USER
    # Still unanswered and still the same call, so its record is kept.
    assert len(await store.list_run("tenant-test", "run-test")) == 1


async def test_a_finished_record_settles_the_intent_it_answers(engine_factory) -> None:
    store = InMemoryToolOutcomeStore()
    await store.record_finished(
        _started(
            "a",
            state="finished",
            outcome={"kind": "tool_result", "tool_call_id": "a", "content": "sent"},
        )
    )
    engine = engine_factory()
    engine.tool_outcome_store = store
    engine.history.extend(_batch("a"))
    engine.open_intents.append(
        commit_intent(tool_name="Send", tool_call_id="a", rc=engine.config.rc)
    )

    events = [evt async for evt in _replay_recorded_tool_outcomes(engine)]

    assert [evt.type for evt in events] == [EventType.TOOL_RESULT]
    assert events[0].payload["replayed"] is True
    assert find_intent(engine.open_intents, "a") is None
    assert _answered_ids(engine.history) == ["a"]


async def test_nothing_happens_without_a_store_or_without_records(engine_factory) -> None:
    engine = engine_factory()
    engine.history.extend(_batch("a"))
    assert [evt async for evt in _replay_recorded_tool_outcomes(engine)] == []
    await _discard_answered_tool_outcomes(engine)

    engine.tool_outcome_store = InMemoryToolOutcomeStore()
    assert [evt async for evt in _replay_recorded_tool_outcomes(engine)] == []
    await _discard_answered_tool_outcomes(engine)
    assert _answered_ids(engine.history) == []
