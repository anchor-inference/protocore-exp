"""A host controls the steer and follow-up queues through the engine, not its fields.

A host that let a person take a queued message back, or turn a waiting
follow-up into a steer of the run under way, wrote the engine's private queue
lists itself, and kept its own record of which ids its last store read had
handed the engine so that writing the lists back would not destroy a message
that arrived in between. Each of those is the engine's to answer: what it holds,
what it was handed, and what it placed.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.llm import LLMProviderError, ProviderFailure, ProviderFailureReason
from protocore.contracts.outcome import LEFTOVER_QUEUED_FIELD
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.live_control import (
    QueuedPrompt,
    merge_queue,
    new_queued_prompt,
)
from protocore.runtime.query import _query as query
from protocore.tests_support.adapters import InMemoryLLMProvider
from tests.unit.runtime.test_provider_failure_outcome import _ScriptedLLM


def _enabled(**overrides: object) -> LoopConstants:
    values: dict[str, object] = {
        "model_context_window": 4_096,
        "steer_follow_up_enabled": True,
        "llm_transient_error_retry_max_attempts": 0,
    }
    values.update(overrides)
    return LoopConstants(**values)  # type: ignore[arg-type]


def _item(kind: str, text: str, item_id: str) -> dict[str, Any]:
    return new_queued_prompt(kind, text, item_id=item_id).to_dict()  # type: ignore[arg-type]


def _user(text: str = "go") -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _terminal_stop(events: list[TurnEvent]) -> dict[str, Any]:
    stops = [
        e.payload
        for e in events
        if e.type is EventType.MESSAGE_STOP and e.payload.get("stop_reason") != "tool_use"
    ]
    assert stops
    return stops[-1]


class _Store:
    """A host's durable queue row: two lists, read and written whole."""

    def __init__(self) -> None:
        self.steer: list[dict[str, Any]] = []
        self.follow_up: list[dict[str, Any]] = []
        self.consumed: list[str] = []


def _bind(engine: Any, store: _Store) -> None:
    async def reload(eng: Any) -> None:
        eng.load_queued(list(store.steer), list(store.follow_up))

    async def persist(eng: Any) -> None:
        merged = eng.merge_queued(list(store.steer), list(store.follow_up))
        store.steer = merged.steer
        store.follow_up = merged.follow_up
        store.consumed.extend(merged.consumed)

    engine.reload_live_control = reload
    engine.persist_live_control = persist


# -- what the engine holds ------------------------------------------------


def test_queued_lists_steers_first_then_follow_ups(engine_factory) -> None:
    engine = engine_factory(rc=_enabled())
    later = engine.queue_input("follow_up", "later", item_id="f1")
    now = engine.queue_input("steer", "now", item_id="s1")

    assert engine.queued() == [now, later]
    assert isinstance(engine.queued()[0], QueuedPrompt)


def test_queue_input_keeps_the_caps(engine_factory) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    with pytest.raises(ValueError, match="steer_follow_up_disabled"):
        engine.queue_input("steer", "now")

    capped = engine_factory(rc=_enabled(max_queued_items=1))
    capped.queue_input("steer", "one")
    with pytest.raises(ValueError, match="queue_full"):
        capped.queue_input("steer", "two")


# -- withdraw -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_withdrawn_steer_never_reaches_the_model(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_tool_call_response(tool_call_id="c1", tool_name="Read", tool_input={"path": "x"})
    llm.queue_response(text="done")
    engine = engine_factory(rc=_enabled())
    engine.history.append(_user())
    engine.queue_input("steer", "never mind this", item_id="s1")

    assert engine.withdraw_queued("s1") is True
    assert engine.queued() == []
    [_ async for _ in query(engine)]

    assert "never mind this" not in " ".join(m.text for m in engine.history)


@pytest.mark.asyncio
async def test_withdrawing_an_item_already_placed_answers_false(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_tool_call_response(tool_call_id="c1", tool_name="Read", tool_input={"path": "x"})
    llm.queue_response(text="done")
    engine = engine_factory(rc=_enabled())
    engine.history.append(_user())
    store = _Store()
    store.steer = [_item("steer", "look left", "s1")]
    _bind(engine, store)

    [_ async for _ in query(engine)]

    assert "look left" in " ".join(m.text for m in engine.history)
    assert engine.withdraw_queued("s1") is False
    assert store.consumed == ["s1"]
    assert store.steer == []


def test_a_withdrawal_outlives_a_store_read_that_predates_it(engine_factory) -> None:
    """A reload already waiting on the store returns the row as it stood before
    the withdrawal; the engine must not put the message back into its queue."""
    engine = engine_factory(rc=_enabled())
    stale_row = [_item("steer", "take this back", "s1"), _item("steer", "keep", "s2")]
    engine.load_queued(stale_row, [])

    assert engine.withdraw_queued("s1") is True
    engine.load_queued(stale_row, [])

    assert [item.id for item in engine.queued()] == ["s2"]
    merged = engine.merge_queued(stale_row, [])
    assert [item["id"] for item in merged.steer] == ["s2"]
    # Taken back, not read: the store must not be told the model consumed it.
    assert merged.consumed == []


def test_withdrawing_an_item_that_arrived_after_the_last_read(engine_factory) -> None:
    engine = engine_factory(rc=_enabled())
    engine.load_queued([], [])

    # The engine never held it, so it answers False and the host removes it
    # from its store; a read in flight must still not bring it back.
    assert engine.withdraw_queued("late") is False
    engine.load_queued([_item("steer", "late", "late")], [])
    assert engine.queued() == []


# -- promote --------------------------------------------------------------


def test_promoting_a_follow_up_makes_it_the_last_steer(engine_factory) -> None:
    engine = engine_factory(rc=_enabled())
    engine.load_queued([_item("steer", "first", "s1")], [_item("follow_up", "then", "f1")])

    assert engine.promote_queued("f1") is True

    assert [(item.id, item.kind) for item in engine.queued()] == [("s1", "steer"), ("f1", "steer")]
    assert engine.promote_queued("f1") is False
    assert engine.promote_queued("unknown") is False


def test_a_promotion_survives_a_store_that_has_not_moved_the_item_yet(engine_factory) -> None:
    engine = engine_factory(rc=_enabled())
    row_steer = [_item("steer", "first", "s1")]
    row_follow = [_item("follow_up", "then", "f1")]
    engine.load_queued(row_steer, row_follow)
    engine.promote_queued("f1")

    engine.load_queued(row_steer, row_follow)

    assert [(item.id, item.kind) for item in engine.queued()] == [("s1", "steer"), ("f1", "steer")]


@pytest.mark.asyncio
async def test_a_promoted_follow_up_is_placed_once_and_reported_consumed(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_tool_call_response(tool_call_id="c1", tool_name="Read", tool_input={"path": "x"})
    llm.queue_response(text="done")
    engine = engine_factory(rc=_enabled())
    engine.history.append(_user())
    store = _Store()
    store.follow_up = [_item("follow_up", "do this now", "f1")]
    engine.load_queued([], list(store.follow_up))
    assert engine.promote_queued("f1") is True
    # The host's store moved it too.
    store.steer, store.follow_up = [dict(store.follow_up[0], kind="steer")], []

    async def reload(eng: Any) -> None:
        eng.load_queued(list(store.steer), list(store.follow_up))

    async def persist(eng: Any) -> None:
        merged = eng.merge_queued(list(store.steer), list(store.follow_up))
        store.steer, store.follow_up = merged.steer, merged.follow_up
        store.consumed.extend(merged.consumed)

    engine.reload_live_control = reload
    engine.persist_live_control = persist
    [_ async for _ in query(engine)]

    placed = [m for m in engine.history if m.text == "do this now"]
    assert len(placed) == 1
    assert store.consumed == ["f1"]
    assert store.steer == [] and store.follow_up == []


# -- merge ----------------------------------------------------------------


def test_merge_queue_appends_what_arrived_after_the_read() -> None:
    kept = [_item("steer", "kept", "a")]
    stored = [_item("steer", "placed", "p"), _item("steer", "kept", "a"), _item("steer", "new", "n")]

    merged = merge_queue(kept, stored, seen=["p", "a"])

    assert [item["id"] for item in merged] == ["a", "n"]


@pytest.mark.asyncio
async def test_a_steer_arriving_between_reload_and_persist_survives(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_tool_call_response(tool_call_id="c1", tool_name="Read", tool_input={"path": "x"})
    llm.queue_response(text="done")
    engine = engine_factory(rc=_enabled())
    engine.history.append(_user())
    store = _Store()
    store.steer = [_item("steer", "first steer", "s1")]
    persisted: list[list[str]] = []
    reads = {"n": 0}

    async def reload(eng: Any) -> None:
        reads["n"] += 1
        eng.load_queued(list(store.steer), list(store.follow_up))
        if reads["n"] == 2:
            # The read before the tool round's placement; this steer lands in
            # the store after it and before the write-back.
            store.steer.append(_item("steer", "second steer", "s2"))

    async def persist(eng: Any) -> None:
        merged = eng.merge_queued(list(store.steer), list(store.follow_up))
        store.steer, store.follow_up = merged.steer, merged.follow_up
        store.consumed.extend(merged.consumed)
        persisted.append([item["id"] for item in merged.steer])

    engine.reload_live_control = reload
    engine.persist_live_control = persist
    events = [event async for event in engine.run(None)]

    # The blind write-back of the engine's empty list would have dropped s2.
    assert persisted[0] == ["s2"]
    assert store.consumed[0] == "s1"
    assert "first steer" in " ".join(m.text for m in engine.history)
    # s2 never reached the model (the run ended without another tool round),
    # so the run lists it as left over, and the store still holds it.
    assert _terminal_stop(events)[LEFTOVER_QUEUED_FIELD] == ["s2"]
    assert [item["id"] for item in store.steer] == ["s2"]


# -- leftovers on the terminal stop ---------------------------------------


@pytest.mark.asyncio
async def test_a_failed_run_lists_the_items_it_did_not_place(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_enabled())
    refused = LLMProviderError(
        "billing",
        classified=ProviderFailure(reason=ProviderFailureReason.billing, retryable=False),
    )
    engine.llm = _ScriptedLLM([refused])  # type: ignore[assignment]
    engine.queue_input("steer", "also check the logs", item_id="s1")
    engine.queue_input("follow_up", "then summarise", item_id="f1")

    events = [event async for event in engine.run(_user())]

    stop = _terminal_stop(events)
    assert stop["outcome"] == "failed"
    assert stop[LEFTOVER_QUEUED_FIELD] == ["s1", "f1"]
    # Listed, not drained: the items stay with the engine for the next run.
    assert [item.id for item in engine.queued()] == ["s1", "f1"]


@pytest.mark.asyncio
async def test_a_completed_run_with_nothing_queued_lists_nothing(
    engine_factory, in_memory_runtime
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="done")
    engine = engine_factory(rc=_enabled())

    events = [event async for event in engine.run(_user())]

    assert _terminal_stop(events)[LEFTOVER_QUEUED_FIELD] == []


@pytest.mark.asyncio
async def test_a_steer_queued_during_the_last_turn_is_left_over(
    engine_factory, in_memory_runtime
) -> None:
    """A steer is placed at a tool round; one the engine took in for a turn that
    then answers without tools is never read, and the run says so."""
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="done")
    engine = engine_factory(rc=_enabled())
    store = _Store()
    reads = {"n": 0}

    async def reload(eng: Any) -> None:
        reads["n"] += 1
        if reads["n"] == 1:
            store.steer.append(_item("steer", "one more thing", "s9"))
        eng.load_queued(list(store.steer), list(store.follow_up))

    engine.reload_live_control = reload
    events = [event async for event in engine.run(_user())]

    assert _terminal_stop(events)[LEFTOVER_QUEUED_FIELD] == ["s9"]
