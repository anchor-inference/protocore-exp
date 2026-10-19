"""A run whose background pool cannot speak for its session says so.

Background commands outlive the run that spawned them, so the pool is a
collaborator the host injects rather than state the run carries. That makes a
cold start — a fresh process picking up a session whose commands were spawned
by a process that is gone — the case the contract exists for: until the host
re-attaches the session's still-running commands, nobody can say what they are
doing.

That case used to be indistinguishable from a quiet session. The wake check
answered with an empty list either way, and an agent that had started a command
and asked to be told when it finished was never told: the command finished
before the run was resumed, and the run it was resumed into had no pool bound
to the session at all. These tests hold the loop to reporting the difference.
"""
from __future__ import annotations

import builtins
from typing import Any

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.events import EventType
from protocore.runtime.query import (
    BACKGROUND_DETACHED_NO_POOL,
    BACKGROUND_DETACHED_SESSION_NOT_REATTACHED,
    BACKGROUND_DETACHED_TASKS_NOT_READOPTED,
)
from protocore.runtime.query import _query as query
from protocore.runtime.query_engine import QueryEngine
from protocore.tests_support.adapters import InMemoryLLMProvider
from tests.unit.runtime.fake_background_pool import FakeBackgroundPool


def _rc(**overrides: object) -> LoopConstants:
    values: dict[str, object] = {
        "model_context_window": 4096,
        "background_tasks_enabled": True,
    }
    values.update(overrides)
    return LoopConstants(**values)  # type: ignore[arg-type]


class _DetachedPool:
    """A pool that holds the session's records but not its handles.

    The shape a fresh process is in before the host re-adopts anything: it can
    name the session's commands and cannot say a thing about them.
    """

    def __init__(self, task_ids: tuple[str, ...] = ("bg-1",)) -> None:
        self._task_ids = task_ids
        self.refreshed: list[str] = []
        self.drained: list[str] = []

    def mark_session_attached(self, session_id: str) -> None:
        return None

    async def ensure_session_attached(self, session_id: str) -> bool:
        return False

    def list(self, session_id: str) -> builtins.list[Any]:
        return []

    def get(self, task_id: str) -> Any:
        return None

    async def refresh(self, task_id: str) -> object:
        self.refreshed.append(task_id)
        return None

    def drain_wakes(self, session_id: str) -> builtins.list[str]:
        self.drained.append(session_id)
        return []


def _detach_events(events: list[Any]) -> list[Any]:
    return [
        evt
        for evt in events
        if evt.type == EventType.STATE_CHANGED
        and evt.payload.get("reason") == "background_tasks_detached"
    ]


async def _drive(engine: QueryEngine) -> list[Any]:
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])
    )
    return [evt async for evt in query(engine)]


def _engine_with_answer(engine_factory: Any, in_memory_runtime: dict[str, object], rc: LoopConstants) -> QueryEngine:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_response(text="answered")
    engine: QueryEngine = engine_factory(rc=rc)
    return engine


async def test_pool_is_an_injected_collaborator(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    pool = FakeBackgroundPool()
    engine = QueryEngine(
        config=engine_factory(rc=_rc()).config,
        llm_provider=in_memory_runtime["llm"],  # type: ignore[arg-type]
        tool_registry=in_memory_runtime["tools"],  # type: ignore[arg-type]
        event_stream=in_memory_runtime["events"],  # type: ignore[arg-type]
        hook_manager=in_memory_runtime["hooks"],  # type: ignore[arg-type]
        skill_store=in_memory_runtime["skills"],  # type: ignore[arg-type]
        blob_store=in_memory_runtime["blobs"],  # type: ignore[arg-type]
        background_pool=pool,
    )
    assert engine.background_pool is pool


async def test_a_run_that_never_had_a_pool_is_quiet(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    """A host that runs no pool is not missing one.

    Every run of such a host used to be reported as detached with
    ``no_pool_bound``, and the report read as the reason background work never
    woke the agent when it was not; the only way to silence it was to switch
    the feature off. Nothing was ever started in a pool, so there is nothing
    nobody can answer for.
    """
    engine = _engine_with_answer(engine_factory, in_memory_runtime, _rc())
    assert engine.background_pool is None

    events = await _drive(engine)

    assert _detach_events(events) == []
    assert not [evt for evt in events if evt.type == EventType.BACKGROUND_WAKE]


async def test_a_resumed_run_whose_commands_have_no_pool_says_so(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    """The case the report exists for: commands were running, and no pool is bound.

    The snapshot names the commands this run started, and the process it was
    resumed into bound no pool to answer for them.
    """
    rc = _rc()
    source = _engine_with_answer(engine_factory, in_memory_runtime, rc)
    source_pool = FakeBackgroundPool()
    source.background_pool = source_pool
    source_pool.start(source.config.session_id, notify_on_finish=True)
    snapshot = source.snapshot()

    resumed = _engine_with_answer(engine_factory, in_memory_runtime, rc)
    resumed._helpers = {}
    await resumed.resume_from_snapshot(snapshot)
    assert resumed.background_pool is None

    events = await _drive(resumed)

    detached = _detach_events(events)
    assert len(detached) == 1, "reported once per turn, not once per assistant message"
    assert detached[0].payload["detached_reason"] == BACKGROUND_DETACHED_NO_POOL
    assert detached[0].payload["session_id"] == resumed.config.session_id
    assert not [evt for evt in events if evt.type == EventType.BACKGROUND_WAKE]


async def test_a_pool_that_was_not_reattached_says_so_instead_of_reporting_quiet(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    engine = _engine_with_answer(engine_factory, in_memory_runtime, _rc())
    pool = _DetachedPool()
    engine.background_pool = pool

    events = await _drive(engine)

    detached = _detach_events(events)
    assert len(detached) == 1
    assert (
        detached[0].payload["detached_reason"]
        == BACKGROUND_DETACHED_SESSION_NOT_REATTACHED
    )
    assert pool.drained == [engine.config.session_id], (
        "a detached report is not a reason to stop asking what the pool can "
        "still answer"
    )


async def test_an_attached_pool_is_quiet_and_still_delivers_wakes(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    rc = _rc()
    engine = _engine_with_answer(engine_factory, in_memory_runtime, rc)
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    task = pool.start(engine.config.session_id, notify_on_finish=True)
    pool.finish(task.id)

    events = await _drive(engine)

    assert _detach_events(events) == []
    wakes = [evt for evt in events if evt.type == EventType.BACKGROUND_WAKE]
    assert len(wakes) == 1
    assert wakes[0].payload["task_ids"] == [task.id]


async def test_background_tasks_disabled_reports_nothing(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    """The feature is off; a missing pool is the configuration, not a fault."""
    engine = _engine_with_answer(
        engine_factory, in_memory_runtime, _rc(background_tasks_enabled=False)
    )

    events = await _drive(engine)

    assert _detach_events(events) == []


@pytest.mark.parametrize(
    "reason",
    [
        BACKGROUND_DETACHED_NO_POOL,
        BACKGROUND_DETACHED_SESSION_NOT_REATTACHED,
        BACKGROUND_DETACHED_TASKS_NOT_READOPTED,
    ],
)
def test_the_detach_reasons_are_distinct(reason: str) -> None:
    assert reason
    assert (
        len(
            {
                BACKGROUND_DETACHED_NO_POOL,
                BACKGROUND_DETACHED_SESSION_NOT_REATTACHED,
                BACKGROUND_DETACHED_TASKS_NOT_READOPTED,
            }
        )
        == 3
    )


async def test_an_attached_pool_with_nothing_finished_is_simply_quiet(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    """The true negative the whole mechanism exists to tell apart from a fault.

    A pool that speaks for the session and holds a command that is still
    running reports nothing at all — no wake, and no detach.
    """
    rc = _rc()
    engine = _engine_with_answer(engine_factory, in_memory_runtime, rc)
    pool = FakeBackgroundPool()
    engine.background_pool = pool
    pool.start(engine.config.session_id, notify_on_finish=True)

    events = await _drive(engine)

    assert _detach_events(events) == []
    assert [evt for evt in events if evt.type == EventType.BACKGROUND_WAKE] == []


async def test_a_cold_resume_against_a_fresh_pool_reports_the_lost_commands(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    """The run's own durable state is the witness, not the pool.

    The pool a new process mints is empty and vouches for the session it was
    handed, so from inside it there is nothing to notice. What breaks the
    silence is the snapshot: it recorded the ids that were running, and they
    are not there.
    """
    rc = _rc()
    source = _engine_with_answer(engine_factory, in_memory_runtime, rc)
    source_pool = FakeBackgroundPool()
    source.background_pool = source_pool
    running = source_pool.start(source.config.session_id, notify_on_finish=True)
    snapshot = source.snapshot()
    assert snapshot["background_task_ids"] == [running.id]

    resumed = _engine_with_answer(engine_factory, in_memory_runtime, rc)
    fresh_pool = FakeBackgroundPool()
    fresh_pool.mark_session_attached(resumed.config.session_id)
    resumed.background_pool = fresh_pool
    resumed._helpers = {}
    await resumed.resume_from_snapshot(snapshot)

    events = await _drive(resumed)

    detached = _detach_events(events)
    assert len(detached) == 1
    assert (
        detached[0].payload["detached_reason"]
        == BACKGROUND_DETACHED_TASKS_NOT_READOPTED
    )
