"""A history rewritten outside the loop retires everything derived from the history it replaced.

A host that compacts between runs, reverts a turn, clears a session or imports
one assigned ``engine.history`` and then reset the checkpoint, the compaction
state and the prompt measurement by hand. Missing one of them cost a real
session: the trigger still read the size of the history it had just shrunk, and
nearly half the compactions in its log re-summarised a tail that the previous
one had already made small. ``replace_history`` does all of it in one place.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.context.compaction import CompactionState
from protocore.runtime.events import EventType
from protocore.runtime.history_persist import HistoryDelta, persist_history
from protocore.runtime.query_engine import QueryEngine


def _say(role: MessageRole, text: str) -> Message:
    return Message(role=role, content_blocks=[TextBlock(text=text)])


def _long_history(turns: int, size: int) -> list[Message]:
    history: list[Message] = []
    for n in range(turns):
        history.append(_say(MessageRole.user, f"q{n} " + "x" * size))
        history.append(_say(MessageRole.assistant, f"a{n} " + "y" * size))
    return history


class _Store:
    def __init__(self) -> None:
        self.deltas: list[HistoryDelta] = []

    def persist_session_history(self, engine: QueryEngine) -> None:
        raise AssertionError("the delta path is attached")

    def persist_history_delta(self, engine: QueryEngine, delta: HistoryDelta) -> None:
        self.deltas.append(delta)


def _measured(engine: QueryEngine, prompt_tokens: int) -> None:
    engine.note_dispatched_history()
    engine.note_observed_prompt(engine.effective_model_name, prompt_tokens)


async def test_the_trigger_reads_the_new_history_not_the_old_measurement(engine_factory: Any) -> None:
    rc = LoopConstants(model_context_window=100_000, compaction_trigger_ratio=0.8, token_estimate_calibration=2.0)
    engine = engine_factory(rc=rc)
    engine.history = _long_history(20, 8_000)
    _measured(engine, 90_000)
    assert engine.needs_compaction(), "the provider counted 90k of a 100k window"

    survivors = [_say(MessageRole.user, "summary " + "s" * 4_000), *engine.history[-4:]]
    await engine.replace_history(survivors, reason="compacted")

    assert engine.history == survivors
    assert not engine.needs_compaction(), "the new history is a fraction of the old one"
    # Scaled from the count, so it is not the over-calibrated estimate either.
    assert engine.prompt_tokens_now() <= engine.context_manager.current_prompt_tokens(survivors) + engine.request_overhead_tokens()


async def test_the_estimate_is_scaled_from_the_count_and_never_grows(engine_factory: Any) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=1_000_000))
    engine.history = _long_history(10, 4_000)
    _measured(engine, 50_000)
    half = engine.history[len(engine.history) // 2 :]
    await engine.replace_history(half, reason="revert")
    scaled = engine.prompt_tokens_now()
    assert 0 < scaled < 50_000
    assert abs(scaled - 25_000) < 2_500, scaled

    # A rewrite that does not shrink the history says nothing about the prompt growing.
    grown = [*engine.history, *_long_history(10, 4_000)]
    before = engine.prompt_tokens_now()
    await engine.replace_history(grown, reason="import")
    assert engine.prompt_tokens_now() <= max(before, engine.context_manager.current_prompt_tokens(grown) + engine.request_overhead_tokens())


async def test_messages_added_after_the_replace_are_estimated_on_top(engine_factory: Any) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=1_000_000))
    engine.history = _long_history(10, 4_000)
    _measured(engine, 50_000)
    await engine.replace_history(engine.history[-4:], reason="compacted")
    base = engine.prompt_tokens_now()
    engine.history.append(_say(MessageRole.user, "z" * 4_000))
    added = engine.context_manager.current_prompt_tokens(engine.history[-1:])
    assert engine.prompt_tokens_now() == base + added


async def test_without_a_count_the_estimate_stays_the_authority(engine_factory: Any) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=1_000_000))
    engine.history = _long_history(4, 4_000)
    kept = engine.history[-2:]
    await engine.replace_history(kept, reason="revert")
    assert engine.prompt_tokens_now() == engine.context_manager.current_prompt_tokens(kept) + engine.request_overhead_tokens()


async def test_derived_state_is_retired(engine_factory: Any) -> None:
    engine = engine_factory()
    engine.history = _long_history(3, 100)
    engine.compact_checkpoint = object()
    engine.compaction_state = CompactionState(retry_count=2, summarised_turn_ids={"t1"})
    engine.compaction_backoff_left = 3
    engine.compaction_backoff_prompt_tokens = 900
    engine._proactive_suspension_gates_left = 4
    engine.last_observed_prompt_tokens = 77_000

    await engine.replace_history([_say(MessageRole.user, "fresh")], reason="clear")

    assert engine.compact_checkpoint is None
    assert engine.compaction_state.retry_count == 0
    assert not engine.compaction_state.summarised_turn_ids
    assert engine.compaction_backoff_left == 0
    assert engine.compaction_backoff_prompt_tokens == 0
    assert engine._proactive_suspension_gates_left == 0
    assert engine.last_observed_prompt_tokens == 0


async def test_the_store_is_handed_a_rewrite(engine_factory: Any) -> None:
    engine = engine_factory()
    store = _Store()
    engine.persist_session_history = store.persist_session_history  # type: ignore[attr-defined]
    engine.persist_history_delta = store.persist_history_delta  # type: ignore[attr-defined]
    engine.history = _long_history(3, 10)
    persist_history(engine)
    engine.history.append(_say(MessageRole.user, "more"))
    persist_history(engine)
    assert [d.rewritten for d in store.deltas] == [True, False]

    # A prefix of the old history: the identity comparison alone would read
    # this as nothing new, and the store would keep the rows that are gone.
    await engine.replace_history(engine.history[:2], reason="revert")
    persist_history(engine)
    assert store.deltas[-1].rewritten
    assert [m.text for m in store.deltas[-1].history] == [m.text for m in engine.history]


async def test_it_emits_history_replaced_and_returns_the_event(engine_factory: Any, in_memory_runtime: dict[str, Any]) -> None:
    engine = engine_factory()
    engine.history = _long_history(3, 10)
    event = await engine.replace_history(engine.history[-2:], reason="compacted")

    assert event.type is EventType.HISTORY_REPLACED
    assert event.payload["reason"] == "compacted"
    assert event.payload["messages_before"] == 6
    assert event.payload["messages_after"] == 2
    assert "prompt_tokens_after" in event.payload
    streams = in_memory_runtime["events"]._streams
    emitted = [e for events in streams.values() for e in events if e.name == EventType.HISTORY_REPLACED.value]
    assert len(emitted) == 1 and emitted[0].payload["reason"] == "compacted"


async def test_a_reason_is_required(engine_factory: Any) -> None:
    engine = engine_factory()
    with pytest.raises(ValueError):
        await engine.replace_history([], reason="  ")


async def test_it_refuses_while_a_turn_is_driving(engine_factory: Any) -> None:
    engine = engine_factory()
    engine.history = _long_history(2, 10)
    async with engine.driving_turn():
        with pytest.raises(RuntimeError):
            await engine.replace_history([], reason="clear")
    assert len(engine.history) == 4


async def test_the_scaled_measurement_survives_a_snapshot(engine_factory: Any) -> None:
    rc = LoopConstants(model_context_window=100_000, compaction_trigger_ratio=0.8, token_estimate_calibration=2.0)
    engine = engine_factory(rc=rc)
    engine.history = _long_history(20, 8_000)
    _measured(engine, 90_000)
    await engine.replace_history(engine.history[-6:], reason="compacted")
    kept = engine.prompt_tokens_now()
    snapshot = engine.snapshot()
    assert snapshot["prompt_anchor"]["prefix_length"] == 6

    resumed = engine_factory(rc=rc)
    await resumed.resume_from_snapshot(snapshot)
    assert resumed.prompt_tokens_now() == kept
    resumed.history.append(_say(MessageRole.user, "next"))
    assert resumed.prompt_tokens_now() > kept


async def test_a_snapshot_without_the_anchor_resumes_on_the_estimate(engine_factory: Any) -> None:
    engine = engine_factory()
    engine.history = _long_history(2, 100)
    snapshot = engine.snapshot()
    assert snapshot["prompt_anchor"] is None
    snapshot.pop("prompt_anchor")
    resumed = engine_factory()
    await resumed.resume_from_snapshot(snapshot)
    expected = resumed.context_manager.current_prompt_tokens(resumed.history) + resumed.request_overhead_tokens()
    assert resumed.prompt_tokens_now() == expected


async def test_an_anchor_longer_than_the_history_is_dropped_on_resume(engine_factory: Any) -> None:
    engine = engine_factory()
    engine.history = _long_history(2, 100)
    snapshot = engine.snapshot()
    snapshot["prompt_anchor"] = {"model": engine.effective_model_name, "prompt_tokens": 5, "prefix_length": 99}
    resumed = engine_factory()
    await resumed.resume_from_snapshot(snapshot)
    assert resumed.prompt_tokens_now() != 5


def test_replace_history_is_a_coroutine(engine_factory: Any) -> None:
    engine = engine_factory()
    assert asyncio.iscoroutinefunction(engine.replace_history)
