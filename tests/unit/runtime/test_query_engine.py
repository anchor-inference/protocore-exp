"""Tests for :mod:`protocore.runtime.query_engine`."""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.loop_state import (
    InvalidStateTransitionError,
    LoopState,
)
from protocore.runtime.query_engine import QueryEngineConfig
from protocore.runtime.usage import TokenUsage


def test_engine_constructed_in_pending_state(engine_factory) -> None:
    engine = engine_factory()
    assert engine.state is LoopState.PENDING
    assert engine.turn_count == 0
    assert engine.history == []


@pytest.mark.parametrize(
    ("parent_run_id", "subagent_id"),
    (("parent-run", None), (None, "worker")),
)
def test_config_rejects_partial_child_topology(
    parent_run_id: str | None,
    subagent_id: str | None,
) -> None:
    with pytest.raises(ValueError, match="parent_run_id and subagent_id"):
        QueryEngineConfig(
            run_id="child-run",
            root_run_id="root-run",
            tenant_id="tenant-test",
            session_id="session-test",
            model_name="model-test",
            parent_run_id=parent_run_id,
            subagent_id=subagent_id,
        )


def test_config_rejects_child_topology_with_root_identity() -> None:
    with pytest.raises(ValueError, match="root run config"):
        QueryEngineConfig(
            run_id="root-run",
            root_run_id="root-run",
            tenant_id="tenant-test",
            session_id="session-test",
            model_name="model-test",
            parent_run_id="parent-run",
            subagent_id="worker",
        )


def test_config_rejects_child_root_without_parent_binding() -> None:
    with pytest.raises(ValueError, match="child run config requires both"):
        QueryEngineConfig(
            run_id="child-run",
            root_run_id="root-run",
            tenant_id="tenant-test",
            session_id="session-test",
            model_name="model-test",
        )


@pytest.mark.parametrize(
    ("run_id", "root_run_id", "parent_run_id", "subagent_id", "run_depth"),
    (
        ("root-run", "", None, None, 0),
        ("child-run", "root-run", "parent-run", "worker", 1),
        # A complete topology names a position, and a grandchild's ids are
        # indistinguishable from a child's; only the depth separates them.
        ("grandchild-run", "root-run", "child-run", "worker", 2),
    ),
)
def test_config_accepts_complete_root_and_child_topologies(
    run_id: str,
    root_run_id: str,
    parent_run_id: str | None,
    subagent_id: str | None,
    run_depth: int,
) -> None:
    config = QueryEngineConfig(
        run_id=run_id,
        root_run_id=root_run_id,
        tenant_id="tenant-test",
        session_id="session-test",
        model_name="model-test",
        parent_run_id=parent_run_id,
        subagent_id=subagent_id,
        run_depth=run_depth,
    )

    assert config.root_run_id == (root_run_id or run_id)
    assert config.run_depth == run_depth


@pytest.mark.parametrize(
    ("run_id", "root_run_id", "parent_run_id", "subagent_id", "run_depth"),
    (
        ("root-run", "", None, None, -1),
        # A root that claims to sit below itself.
        ("root-run", "", None, None, 1),
        # A child that claims the root's own position.
        ("child-run", "root-run", "parent-run", "worker", 0),
    ),
)
def test_config_rejects_a_depth_that_contradicts_the_run_tree_ids(
    run_id: str,
    root_run_id: str,
    parent_run_id: str | None,
    subagent_id: str | None,
    run_depth: int,
) -> None:
    with pytest.raises(ValueError, match="run_depth"):
        QueryEngineConfig(
            run_id=run_id,
            root_run_id=root_run_id,
            tenant_id="tenant-test",
            session_id="session-test",
            model_name="model-test",
            parent_run_id=parent_run_id,
            subagent_id=subagent_id,
            run_depth=run_depth,
        )


@pytest.mark.parametrize(
    ("run_id", "root_run_id", "parent_run_id", "subagent_id"),
    (
        ("", "", None, None),
        (" root-run", "", None, None),
        ("root-run ", "", None, None),
        ("root-run", " root-run", None, None),
        ("root-run", "root-run ", None, None),
        ("child-run", "root-run", "", "worker"),
        ("child-run", "root-run", "parent-run", ""),
        ("child-run", "root-run", " parent-run", "worker"),
        ("child-run", "root-run", "parent-run", "worker "),
        ("child-run", "root-run", "child-run", "worker"),
    ),
)
def test_config_rejects_blank_padded_and_cyclic_run_tree_identifiers(
    run_id: str,
    root_run_id: str,
    parent_run_id: str | None,
    subagent_id: str | None,
) -> None:
    with pytest.raises(ValueError):
        QueryEngineConfig(
            run_id=run_id,
            root_run_id=root_run_id,
            tenant_id="tenant-test",
            session_id="session-test",
            model_name="model-test",
            parent_run_id=parent_run_id,
            subagent_id=subagent_id,
        )


def test_engine_state_transition_legal(engine_factory) -> None:
    engine = engine_factory()
    engine.transition_to(LoopState.RUNNING)
    assert engine.state is LoopState.RUNNING


def test_engine_state_transition_illegal_raises(engine_factory) -> None:
    engine = engine_factory()
    with pytest.raises(InvalidStateTransitionError):
        engine.transition_to(LoopState.COMPACTING)


def test_stop_request_visible(engine_factory) -> None:
    engine = engine_factory()
    assert engine.stop_requested is False
    engine.stop()
    assert engine.stop_requested is True


def test_block_idx_allocation(engine_factory) -> None:
    engine = engine_factory()
    assert engine.next_block_idx() == 0
    assert engine.next_block_idx() == 1
    engine.reset_block_idx()
    assert engine.next_block_idx() == 0


def test_turn_id_pre_loop_uses_legacy_shape(engine_factory) -> None:
    """#1/#4 — before any round (``_wire_round_seq == 0``), ``turn_id`` keeps the
    legacy ``turn-{run}-{turn_count}`` shape for pre-loop terminals."""
    engine = engine_factory(run_id="r1")
    engine.turn_count = 1
    assert engine._wire_round_seq == 0
    assert engine.turn_id() == "turn-r1-1"


def test_begin_wire_round_makes_turn_id_distinct_per_round(engine_factory) -> None:
    """#1/#4 — each ``begin_wire_round`` mints a DISTINCT round-local turn_id and
    restarts block_idx so it is unique within that round's turn."""
    engine = engine_factory(run_id="r1")
    engine.turn_count = 1
    engine.begin_wire_round()
    assert engine.turn_id() == "turn-r1-1-1"
    assert engine.next_block_idx() == 0
    assert engine.next_block_idx() == 1
    engine.begin_wire_round()
    assert engine.turn_id() == "turn-r1-1-2"
    # block_idx restarted at the round boundary (unique-within-turn contract).
    assert engine.next_block_idx() == 0


def test_wire_round_seq_not_persisted(engine_factory) -> None:
    """#1/#4 — the per-round wire counter is intentionally NOT snapshot-persisted
    (a resumed run re-streams fresh rounds; turn_count carries durable identity)."""
    engine = engine_factory()
    engine.begin_wire_round()
    engine.begin_wire_round()
    assert engine._wire_round_seq == 2
    assert "_wire_round_seq" not in engine.snapshot()
    assert "wire_round_seq" not in engine.snapshot()


async def test_stop_self_cancel_safe_inside_run_task(engine_factory) -> None:
    """#6 — ``stop()`` called from WITHIN the run task only sets the cooperative
    flag; it must NOT self-cancel the current frame mid-iteration."""
    import asyncio

    from protocore.contracts.types import Message, MessageRole, TextBlock

    engine = engine_factory()
    observed: dict[str, object] = {}

    async def _drive() -> None:
        # ``run()`` records ``_current_turn_task`` = this task. Calling stop()
        # here must not raise CancelledError into us.
        async for _evt in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
        ):
            observed["current_task_is_self"] = (
                engine._current_turn_task is asyncio.current_task()
            )
            engine.stop()  # self-call: flag only, no hard cancel
            assert engine.stop_requested is True

    # Must complete WITHOUT a CancelledError escaping (self-cancel was skipped).
    await asyncio.wait_for(_drive(), timeout=5.0)
    assert observed.get("current_task_is_self") is True
    # Cleared in run()'s finally.
    assert engine._current_turn_task is None


async def test_stop_from_other_task_hard_cancels_blocking_run(engine_factory) -> None:
    """#6 — ``stop()`` from a DIFFERENT task hard-cancels the run task so a
    blocking ``await`` is interrupted (the previously-dead ``_current_turn_task``
    arm is now live). A controlled blocking task stands in for the run task."""
    import asyncio

    engine = engine_factory()
    blocked = asyncio.Event()

    async def _stuck_run() -> None:
        # Emulate ``run()``: register self as the current turn task, then block
        # on a long await (a stuck tool/subagent dispatch).
        engine._current_turn_task = asyncio.current_task()
        try:
            blocked.set()
            await asyncio.sleep(30)
        finally:
            engine._current_turn_task = None

    task = asyncio.ensure_future(_stuck_run())
    await asyncio.wait_for(blocked.wait(), timeout=2.0)
    assert engine._current_turn_task is task
    # External stop() (this task != the run task) → hard cancel.
    engine.stop()
    assert engine.stop_requested is True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)


def test_remember_and_forget_tool_name(engine_factory) -> None:
    engine = engine_factory()
    engine.remember_tool_name("toolu_1", "Read")
    assert engine.tool_name_for("toolu_1") == "Read"
    engine.forget_tool_name("toolu_1")
    assert engine.tool_name_for("toolu_1") == ""


def test_snapshot_roundtrip_preserves_state(engine_factory) -> None:
    engine = engine_factory()
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    )
    engine.turn_count = 3
    engine.total_usage.add(input_tokens=100, output_tokens=20)
    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(LoopState.COMPACTING)

    snapshot = engine.snapshot()
    assert snapshot["state"] == LoopState.COMPACTING.value
    assert snapshot["turn_count"] == 3
    assert snapshot["usage"]["input_tokens"] == 100


@pytest.mark.asyncio
async def test_resume_from_snapshot_round_trip(engine_factory) -> None:
    original = engine_factory()
    original.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello")])
    )
    original.turn_count = 5
    original.transition_to(LoopState.RUNNING)
    original.total_usage.add(input_tokens=42, output_tokens=7)
    snap = original.snapshot()

    other = engine_factory()
    await other.resume_from_snapshot(snap)
    assert other.state is LoopState.RUNNING
    assert other.turn_count == 5
    assert len(other.history) == 1
    assert other.history[0].text == "hello"
    assert other.total_usage.input_tokens == 42
    assert other.total_usage.output_tokens == 7


@pytest.mark.asyncio
async def test_finalize_prose_gate_latch_survives_snapshot_roundtrip(
    engine_factory,
) -> None:
    """the one-shot prose-gate latch is snapshot-persisted
    so a cross-pod re-drive cannot grant a second prose-gate."""
    original = engine_factory()
    original._finalize_prose_gate_used = True
    snap = original.snapshot()
    assert snap["finalize_prose_gate_used"] is True

    resumed = engine_factory()
    await resumed.resume_from_snapshot(snap)
    assert resumed._finalize_prose_gate_used is True

    # A snapshot predating the field resumes as un-fired (default False).
    legacy = engine_factory()
    snap.pop("finalize_prose_gate_used")
    await legacy.resume_from_snapshot(snap)
    assert legacy._finalize_prose_gate_used is False


@pytest.mark.asyncio
async def test_pointer_answer_repair_budget_survives_snapshot_roundtrip(
    engine_factory,
) -> None:
    """The pointer refusal's attempt budget is snapshot-persisted, so a run
    re-driven on another pod cannot be handed a second full budget of repair
    turns — nor announce a second time that it gave up."""
    original = engine_factory()
    original._pointer_answer_repair_attempts = 2
    original._pointer_answer_repair_released = True
    snap = original.snapshot()
    assert snap["pointer_answer_repair_attempts"] == 2
    assert snap["pointer_answer_repair_released"] is True

    resumed = engine_factory()
    await resumed.resume_from_snapshot(snap)
    assert resumed._pointer_answer_repair_attempts == 2
    assert resumed._pointer_answer_repair_released is True

    # A snapshot predating the fields resumes with the budget untouched.
    legacy = engine_factory()
    snap.pop("pointer_answer_repair_attempts")
    snap.pop("pointer_answer_repair_released")
    await legacy.resume_from_snapshot(snap)
    assert legacy._pointer_answer_repair_attempts == 0
    assert legacy._pointer_answer_repair_released is False


def test_token_usage_reset_turn_does_not_clear_totals() -> None:
    usage = TokenUsage()
    usage.add(input_tokens=10, output_tokens=5)
    usage.reset_turn()
    assert usage.input_tokens == 10
    assert usage.this_turn_input == 0


def test_the_cache_hit_rate_is_the_cached_share_of_the_whole_prompt() -> None:
    usage = TokenUsage()
    # The session that reported 0.47: 2.97 M input of which 2.68 M came from the cache.
    usage.add(input_tokens=2_970_000, cache_read_tokens=2_680_000, output_tokens=11_700)
    assert usage.this_turn_cache_hit_rate() == pytest.approx(2_680_000 / 2_970_000)
    assert TokenUsage().this_turn_cache_hit_rate() == 0.0
    odd = TokenUsage()
    odd.add(input_tokens=10, cache_read_tokens=12)
    assert odd.this_turn_cache_hit_rate() == 1.0


def test_engine_is_terminal_predicate(engine_factory) -> None:
    engine = engine_factory()
    assert not engine.is_terminal
    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(LoopState.COMPLETED)
    assert engine.is_terminal


def test_engine_config_immutable() -> None:
    cfg = QueryEngineConfig(
        run_id="r1",
        tenant_id="t1",
        session_id="s1",
        model_name="qwen3.6-35b-a3b",
        rc=LoopConstants(),
    )
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.run_id = "other"  # type: ignore[misc]


def test_engine_turn_id_changes_with_turn_count(engine_factory) -> None:
    engine = engine_factory()
    first = engine.turn_id()
    engine.turn_count += 1
    second = engine.turn_id()
    assert first != second


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _assistant(text: str) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])


async def test_run_continues_against_user_terminated_history(
    engine_factory, in_memory_runtime
) -> None:
    """``run()`` with no initial message generates an assistant turn against the
    existing history as-is, appending nothing of its own first."""
    in_memory_runtime["llm"].queue_response(text="continuation answer")
    engine = engine_factory()
    engine.history.append(_user("prior question"))

    async for _ in engine.run():
        pass

    # Exactly one message added — the model's answer — with no user message
    # injected ahead of it. The pre-existing user turn is untouched.
    assert [m.role for m in engine.history] == [
        MessageRole.user,
        MessageRole.assistant,
    ]
    assert engine.history[0].text == "prior question"
    assert engine.history[-1].text == "continuation answer"
    # The model was actually driven (one provider call), not short-circuited.
    assert len(in_memory_runtime["llm"].calls) == 1
    assert engine.turn_count == 1


async def test_run_continuation_rejects_non_user_tail(
    engine_factory, in_memory_runtime
) -> None:
    """A continuation run needs something to answer; a plain assistant answer
    is not one, and is rejected before the model is ever called."""
    engine = engine_factory()
    engine.history.append(_user("question"))
    engine.history.append(_assistant("prior answer"))

    with pytest.raises(ValueError, match="must end with"):
        async for _ in engine.run():
            pass

    # No empty/blank message was fabricated to stand in for the missing input,
    # and the model was never driven against one.
    assert [m.role for m in engine.history] == [
        MessageRole.user,
        MessageRole.assistant,
    ]
    assert len(in_memory_runtime["llm"].calls) == 0


async def test_run_continuation_rejects_empty_history(
    engine_factory, in_memory_runtime
) -> None:
    """A continuation run needs a non-empty history to continue from."""
    engine = engine_factory()

    with pytest.raises(ValueError, match="non-empty history"):
        async for _ in engine.run():
            pass

    assert engine.history == []
    assert len(in_memory_runtime["llm"].calls) == 0


async def test_run_explicit_message_behaviour_unchanged(
    engine_factory, in_memory_runtime
) -> None:
    """The default path (an explicit initial message) still appends that message
    before driving the loop — byte-for-byte the prior contract."""
    in_memory_runtime["llm"].queue_response(text="answer")
    engine = engine_factory()

    async for _ in engine.run(_user("hello")):
        pass

    assert [m.role for m in engine.history] == [
        MessageRole.user,
        MessageRole.assistant,
    ]
    assert engine.history[0].text == "hello"
    assert engine.history[-1].text == "answer"
    assert engine.turn_count == 1


async def test_continuation_run_snapshot_round_trip(
    engine_factory, in_memory_runtime
) -> None:
    """A continuation run snapshots and resumes as a flat history, indistinct
    from any other run (the append path it skipped leaves no trace)."""
    in_memory_runtime["llm"].queue_response(text="continuation answer")
    source = engine_factory()
    source.history.append(_user("prior question"))

    async for _ in source.run():
        pass

    snap = source.snapshot()
    resumed = engine_factory()
    await resumed.resume_from_snapshot(snap)

    assert [m.role for m in resumed.history] == [
        MessageRole.user,
        MessageRole.assistant,
    ]
    assert resumed.history[0].text == "prior question"
    assert resumed.history[-1].text == "continuation answer"
    assert resumed.turn_count == source.turn_count
    assert resumed.state is source.state


# --- Continuous run: an agent that lives instead of answering once ----------


def _spend_the_run(engine) -> None:
    """Put the engine where a working run leaves it: budgets used, turn done."""
    engine.transition_to(LoopState.RUNNING)
    engine.turn_count = 7
    engine.total_usage = TokenUsage(input_tokens=90_000, output_tokens=40_000)
    for _ in range(120):
        engine.record_tool_call("mine", ok=True)
    engine._soft_stop_cause = "tool_call_budget"
    engine._soft_stop_stage = "withdrawn"
    engine._terminal_only_active = True
    engine._run_started_monotonic = 1234.5
    engine._run_started_epoch = 1_700_000_000.0
    engine._run_settled_emitted = True
    engine.transition_to(LoopState.COMPLETED)


def test_rearm_returns_a_settled_engine_to_pending(engine_factory) -> None:
    engine = engine_factory()
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="осмотрись")])
    )
    _spend_the_run(engine)

    engine.rearm()

    assert engine.state is LoopState.PENDING
    assert engine.turn_count == 0


def test_rearm_clears_the_bounds_that_would_silently_end_the_agent(
    engine_factory,
) -> None:
    """A bare ``state`` reset leaves these behind; the agent then only finalises."""
    engine = engine_factory()
    _spend_the_run(engine)

    engine.rearm()

    assert engine.tool_call_ledger == [], "cumulative tool-call budget must restart"
    assert engine.total_usage.output_tokens == 0, "output-token budget must restart"
    assert engine._soft_stop_cause is None, "the wind-down latch must be released"
    assert engine._soft_stop_stage == ""
    assert engine._terminal_only_active is False
    assert engine._run_settled_emitted is False
    assert engine._run_started_monotonic == 0.0, "the run clock must restamp"
    assert engine._run_started_epoch == 0.0


def test_rearm_keeps_the_continuity_that_makes_it_the_same_agent(
    engine_factory,
) -> None:
    engine = engine_factory()
    engine.history.append(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="я Кайра")])
    )
    engine.history.append(
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="помню")])
    )
    _spend_the_run(engine)
    before = list(engine.history)

    engine.rearm()

    assert engine.history == before, "history is the agent's life, not a run artifact"


@pytest.mark.parametrize(
    "state", (LoopState.PENDING, LoopState.RUNNING, LoopState.AWAITING)
)
def test_rearm_refuses_a_turn_still_in_flight(engine_factory, state) -> None:
    engine = engine_factory()
    if state is not LoopState.PENDING:
        engine.transition_to(LoopState.RUNNING)
    if state is LoopState.AWAITING:
        # AWAITING is only a legal state when something is recorded that could
        # end the wait.
        engine.mark_pending_approval("call-in-flight")
        engine.transition_to(LoopState.AWAITING)

    with pytest.raises(InvalidStateTransitionError):
        engine.rearm()


@pytest.mark.parametrize("state", (LoopState.FAILED, LoopState.CANCELLED))
def test_rearm_accepts_any_terminal_state(engine_factory, state) -> None:
    """A unit that died of a failed compaction can be revived by its driver."""
    engine = engine_factory()
    engine.transition_to(LoopState.RUNNING)
    engine.transition_to(state)

    engine.rearm()

    assert engine.state is LoopState.PENDING


def test_many_turns_in_a_row_never_arm_the_wind_down(engine_factory) -> None:
    """The regression this method exists for: turn 400 is as free as turn 1."""
    engine = engine_factory()
    for _ in range(400):
        engine.transition_to(LoopState.RUNNING)
        engine.record_tool_call("gather", ok=True)
        engine.transition_to(LoopState.COMPLETED)
        engine.rearm()
        assert engine._soft_stop_cause is None
        assert len(engine.tool_call_ledger) == 0
