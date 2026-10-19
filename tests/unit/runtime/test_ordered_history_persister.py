"""A store's writes taken off the loop's stack, in order, with a stated durability.

A store that writes inside the hand-over makes every round wait for its
database. One that turns the write into a task returns at once, but then the
writes of one session are only in order if they are made to be, and a write
that fails after the engine was told it landed leaves a hole every later
append builds on. These tests pin the helper that does both halves: one write
at a time, in hand-over order, the engine told at once when a write is lost,
the history written whole by whatever comes next, and a flush that says
whether the store holds what it was handed.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.history_persist import (
    HistoryDelta,
    HistoryPersistError,
    OrderedHistoryPersister,
    persist_history,
)


def _msg(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


class _Store:
    """Rows as a store holding the session's messages would have them."""

    def __init__(self) -> None:
        self.rows: list[str] = []
        self.calls: list[HistoryDelta] = []
        self.failures = 0
        self.gates: list[asyncio.Event] = []

    async def write(self, delta: HistoryDelta) -> None:
        self.calls.append(delta)
        if self.gates:
            await self.gates.pop(0).wait()
        if self.failures:
            self.failures -= 1
            raise RuntimeError("the store is down")
        if delta.rewritten:
            self.rows = [m.text for m in delta.history]
        else:
            self.rows.extend(m.text for m in delta.appended)


def _wired(engine_factory: Any) -> tuple[Any, _Store, OrderedHistoryPersister]:
    engine = engine_factory()
    store = _Store()
    persister = OrderedHistoryPersister(store.write)
    persister.attach(engine)
    return engine, store, persister


async def test_the_hand_over_returns_before_the_write_runs(engine_factory: Any) -> None:
    engine, store, persister = _wired(engine_factory)
    engine.history.append(_msg("one"))
    persist_history(engine)
    assert store.calls == []
    assert persister.pending == 1
    await persister.flush()
    assert store.rows == ["one"]
    assert persister.pending == 0


async def test_writes_land_in_hand_over_order_even_when_the_first_is_slow(engine_factory: Any) -> None:
    engine, store, persister = _wired(engine_factory)
    slow = asyncio.Event()
    store.gates.append(slow)
    engine.history.append(_msg("one"))
    persist_history(engine)
    engine.history.append(_msg("two"))
    persist_history(engine)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # The second write has not started: it waits for its turn, not for the clock.
    assert len(store.calls) == 1
    slow.set()
    await persister.flush()
    assert store.rows == ["one", "two"]
    assert [call.rewritten for call in store.calls] == [True, False]


async def test_a_lost_write_is_disowned_at_once_and_the_next_write_is_whole(engine_factory: Any) -> None:
    engine, store, persister = _wired(engine_factory)
    engine.history.append(_msg("one"))
    persist_history(engine)
    await persister.flush()

    store.failures = 1
    engine.history.append(_msg("two"))
    persist_history(engine)
    # Handed over before the failure is known: queued as an append.
    engine.history.append(_msg("three"))
    persist_history(engine)
    await persister.flush()

    assert store.rows == ["one", "two", "three"]
    # The append queued behind the failure was written whole instead.
    assert store.calls[2].rewritten
    assert not persister.behind


async def test_the_engine_hears_of_the_loss_before_its_next_hand_over(engine_factory: Any) -> None:
    engine, store, persister = _wired(engine_factory)
    store.failures = 1
    engine.history.append(_msg("one"))
    persist_history(engine)
    assert engine.persisted_history_marker is not None
    while not store.calls:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert engine.persisted_history_marker is None
    await persister.flush()


async def test_a_last_write_that_failed_is_repaired_without_further_traffic(engine_factory: Any) -> None:
    engine, store, persister = _wired(engine_factory)
    engine.history.append(_msg("one"))
    persist_history(engine)
    await persister.flush()
    store.failures = 1
    engine.history.append(_msg("two"))
    persist_history(engine)
    await persister.flush()
    assert store.rows == ["one", "two"]
    assert store.calls[-1].rewritten
    assert not persister.behind


async def test_a_store_refusing_every_write_is_not_retried_forever(engine_factory: Any) -> None:
    engine, store, persister = _wired(engine_factory)
    store.failures = 100
    engine.history.append(_msg("one"))
    persist_history(engine)
    with pytest.raises(HistoryPersistError) as raised:
        await persister.flush()
    assert isinstance(raised.value.__cause__, RuntimeError)
    # The write and one repair, then nothing until the engine hands over again.
    assert len(store.calls) == 2
    assert persister.behind

    store.failures = 0
    engine.history.append(_msg("two"))
    persist_history(engine)
    await persister.flush()
    assert store.rows == ["one", "two"]
    assert not persister.behind


async def test_flush_with_nothing_pending_returns_at_once(engine_factory: Any) -> None:
    _, store, persister = _wired(engine_factory)
    await persister.flush()
    assert store.calls == []


async def test_a_session_state_change_travels_with_the_write(engine_factory: Any) -> None:
    engine, store, persister = _wired(engine_factory)
    engine.history.append(_msg("one"))
    persist_history(engine)
    await persister.flush()
    engine.note_session_state_changed()
    persist_history(engine)
    await persister.flush()
    assert store.calls[-1].session_state_changed
    assert store.calls[-1].appended == ()


async def test_the_full_write_is_queued_as_a_rewrite(engine_factory: Any) -> None:
    engine, store, persister = _wired(engine_factory)
    engine.history.extend([_msg("one"), _msg("two")])
    persister.persist_session_history(engine)
    await persister.flush()
    assert store.rows == ["one", "two"]
    assert store.calls[0].rewritten


async def test_a_cancelled_write_leaves_the_store_behind(engine_factory: Any) -> None:
    engine, store, persister = _wired(engine_factory)
    store.gates.append(asyncio.Event())
    engine.history.append(_msg("one"))
    persist_history(engine)
    while not store.calls:
        await asyncio.sleep(0)
    persister.cancel()
    with pytest.raises(HistoryPersistError):
        await persister.flush()
    assert engine.persisted_history_marker is None
    assert persister.behind


def test_a_hand_over_outside_a_running_loop_is_refused(engine_factory: Any) -> None:
    engine, _, persister = _wired(engine_factory)
    engine.history.append(_msg("one"))
    with pytest.raises(RuntimeError):
        persister.persist_history_delta(
            engine,
            HistoryDelta(history=tuple(engine.history), appended=(), persisted_through=0, rewritten=True),
        )
