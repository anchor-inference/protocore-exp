"""One recovery budget that every retry layer draws from, and when to re-drive.

The loop retried a blip in place, stepped down the provider chain, re-asked
after an empty round and shrank a request the window refused, each against its
own counter; a host then re-drove the whole turn after an outage or an
overflow against a counter of its own. The layers multiply: two in-place
retries on each of three rungs, re-driven five times, is thirty requests for
one outage. ``RecoveryBudget`` is the one allowance they all draw from, and
``recoverable_after`` on the terminal stop tells a host whether, and when, a
re-drive is worth it.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from protocore.contracts.llm import (
    LLMContextWindowExceeded,
    LLMProviderError,
    ProviderFailure,
    ProviderFailureReason,
)
from protocore.contracts.outcome import RECOVERABLE_AFTER_FIELD, RunCause, RunOutcome
from protocore.contracts.recovery import (
    RecoveryBudget,
    RecoveryKind,
    recoverable_after,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.snapshot import (
    RUN_SCOPED_STATE_SNAPSHOT_KEY,
    SNAPSHOT_SCHEMA_KEY,
    migrate_snapshot,
)
from protocore.runtime.events import EventType, TurnEvent
from tests.unit.runtime.test_provider_failure_outcome import (
    _Chain,
    _connection_refused,
    _rc,
    _ScriptedLLM,
    _user,
)
from tests.unit.runtime.test_query_hardening import _ScriptedFailureLLM, _ThinkingOnlyLLM

Kind = RecoveryKind
Reason = ProviderFailureReason


def _terminal_stop(events: list[TurnEvent]) -> dict[str, Any]:
    stops = [
        e.payload
        for e in events
        if e.type is EventType.MESSAGE_STOP and e.payload.get("stop_reason") != "tool_use"
    ]
    assert stops
    return stops[-1]


def _reasons(events: list[TurnEvent]) -> list[str]:
    return [str(e.payload.get("reason")) for e in events if e.type is EventType.STATE_CHANGED]


# -- the budget itself ----------------------------------------------------


def test_an_unbounded_budget_counts_every_draw_and_refuses_none() -> None:
    budget = RecoveryBudget()
    for kind in (Kind.transient_retry, Kind.chain_advance, Kind.transient_retry):
        assert budget.draw(kind, now=100.0) is True

    assert budget.attempts == 3
    assert budget.by_kind == {"transient_retry": 2, "chain_advance": 1}
    assert budget.remaining_attempts is None
    assert budget.started_at == 100.0


def test_a_bounded_budget_refuses_past_its_attempts_without_charging() -> None:
    budget = RecoveryBudget(max_attempts=2)
    assert budget.draw(Kind.chain_advance, now=0.0)
    assert budget.draw(Kind.transient_retry, now=0.0)

    assert budget.draw(Kind.redrive, now=0.0) is False
    assert budget.attempts == 2
    assert budget.remaining_attempts == 0
    assert budget.permits(now=0.0) is False


def test_a_bounded_budget_refuses_past_its_wall_clock() -> None:
    budget = RecoveryBudget(max_seconds=60.0)
    assert budget.draw(Kind.transient_retry, now=1_000.0)
    assert budget.draw(Kind.transient_retry, now=1_059.0)

    assert budget.draw(Kind.transient_retry, now=1_061.0) is False
    assert budget.elapsed(now=1_061.0) == pytest.approx(61.0)


def test_a_clean_answer_settles_the_stretch() -> None:
    budget = RecoveryBudget(max_attempts=1)
    budget.draw(Kind.transient_retry, now=5.0)

    budget.settle()

    assert budget.attempts == 0
    assert budget.by_kind == {}
    assert budget.started_at is None
    assert budget.permits(now=500.0)


def test_the_budget_round_trips_through_json_and_shrugs_off_garbage() -> None:
    budget = RecoveryBudget(max_attempts=4, max_seconds=90.0)
    budget.draw(Kind.chain_advance, now=12.5)

    again = RecoveryBudget.from_dict(json.loads(json.dumps(budget.to_dict())))

    assert again == budget
    assert RecoveryBudget.from_dict({"attempts": "many", "by_kind": [1]}) == RecoveryBudget()
    assert RecoveryBudget.from_dict(None) is None


# -- when a re-drive is worth it ------------------------------------------


def _cause(reason: Reason, *, retryable: bool = True, retry_after: float | None = None) -> RunCause:
    return RunCause(
        kind="llm_provider_error",
        detail="upstream said no",
        failure=ProviderFailure(
            reason=reason, retryable=retryable, retry_after_seconds=retry_after
        ),
    )


def test_an_outage_is_recoverable_after_the_backoff() -> None:
    rc = LoopConstants(model_context_window=4_096)
    budget = RecoveryBudget()
    first = recoverable_after(RunOutcome.failed, _cause(Reason.overloaded), budget, rc, now=0.0)
    budget.draw(Kind.redrive, now=0.0)
    second = recoverable_after(RunOutcome.failed, _cause(Reason.overloaded), budget, rc, now=0.0)

    assert first == rc.recovery_redrive_backoff_base_seconds
    assert second == 2 * rc.recovery_redrive_backoff_base_seconds


def test_the_backoff_stops_at_its_ceiling_and_the_server_can_raise_it() -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        recovery_redrive_backoff_base_seconds=10.0,
        recovery_redrive_backoff_max_seconds=40.0,
    )
    budget = RecoveryBudget(by_kind={"redrive": 5}, attempts=5)

    assert recoverable_after(RunOutcome.failed, _cause(Reason.server_error), budget, rc) == 40.0
    told = _cause(Reason.rate_limit, retry_after=300.0)
    assert recoverable_after(RunOutcome.failed, told, RecoveryBudget(), rc) == 300.0


def test_a_permanent_refusal_or_a_spend_cap_is_not_recoverable() -> None:
    rc = LoopConstants(model_context_window=4_096)
    for reason in (Reason.billing, Reason.auth_permanent, Reason.budget):
        cause = _cause(reason, retryable=False)
        assert recoverable_after(RunOutcome.failed, cause, RecoveryBudget(), rc) is None


def test_only_a_failed_run_is_recoverable() -> None:
    rc = LoopConstants(model_context_window=4_096)
    for outcome in (RunOutcome.completed, RunOutcome.stopped, RunOutcome.wound_down):
        assert recoverable_after(outcome, _cause(Reason.overloaded), RecoveryBudget(), rc) is None
    crash = RunCause(kind="internal_error", detail="boom")
    assert recoverable_after(RunOutcome.failed, crash, RecoveryBudget(), rc) is None


def test_a_timeout_without_a_verdict_is_still_an_outage() -> None:
    rc = LoopConstants(model_context_window=4_096)
    cause = RunCause(kind="llm_timeout", detail="read timed out")
    assert recoverable_after(RunOutcome.failed, cause, RecoveryBudget(), rc) is not None


def test_an_exhausted_budget_leaves_nothing_to_re_drive() -> None:
    rc = LoopConstants(model_context_window=4_096)
    spent = RecoveryBudget(max_attempts=1, attempts=1, by_kind={"chain_advance": 1})
    assert recoverable_after(RunOutcome.failed, _cause(Reason.overloaded), spent, rc) is None

    late = RecoveryBudget(max_seconds=100.0, started_at=0.0, attempts=1)
    # The wait would run past the wall clock the budget allows.
    assert recoverable_after(RunOutcome.failed, _cause(Reason.overloaded), late, rc, now=80.0) is None


def test_an_overflow_is_recoverable_at_once_a_bounded_number_of_times() -> None:
    rc = LoopConstants(model_context_window=4_096, recovery_overflow_redrive_max_attempts=1)
    overflow = RunCause(kind="llm_context_window_exceeded", detail="too long")
    budget = RecoveryBudget()

    assert recoverable_after(RunOutcome.failed, overflow, budget, rc) == 0.0
    budget.draw(Kind.overflow_redrive)
    assert recoverable_after(RunOutcome.failed, overflow, budget, rc) is None


# -- the loop draws from it ---------------------------------------------


@pytest.mark.asyncio
async def test_retries_and_chain_advances_share_one_bound(engine_factory, in_memory_runtime) -> None:
    """Two rungs and two in-place retries: four requests on their own counters,
    three when they draw from a budget of two."""
    engine = engine_factory(rc=_rc(recovery_budget_max_attempts=2))
    llm = _ScriptedLLM([_connection_refused()] * 10)
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary"])

    events = [event async for event in engine.run(_user())]

    assert len(llm.calls) == 3
    assert _reasons(events).count("model_fallback_triggered") == 1
    assert _reasons(events).count("transient_llm_error_retry") == 1
    assert engine.recovery_budget.by_kind == {"chain_advance": 1, "transient_retry": 1}
    stop = _terminal_stop(events)
    assert stop["outcome"] == RunOutcome.failed
    # The budget is spent, so a re-drive would only spend more.
    assert stop[RECOVERABLE_AFTER_FIELD] is None


@pytest.mark.asyncio
async def test_without_a_bound_the_layers_keep_their_own_counts(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([_connection_refused()] * 10)
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary"])

    events = [event async for event in engine.run(_user())]

    assert len(llm.calls) == 4
    assert engine.recovery_budget.attempts == 3
    stop = _terminal_stop(events)
    assert stop[RECOVERABLE_AFTER_FIELD] == engine.config.rc.recovery_redrive_backoff_base_seconds


@pytest.mark.asyncio
async def test_a_permanent_refusal_says_not_to_re_drive(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=_rc())
    refused = LLMProviderError(
        "no balance", classified=ProviderFailure(reason=Reason.billing, retryable=False)
    )
    engine.llm = _ScriptedLLM([refused])  # type: ignore[assignment]

    events = [event async for event in engine.run(_user())]

    assert _terminal_stop(events)[RECOVERABLE_AFTER_FIELD] is None


@pytest.mark.asyncio
async def test_a_completed_run_reports_no_re_drive_and_settles(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([_connection_refused()])
    engine.llm = llm  # type: ignore[assignment]

    events = [event async for event in engine.run(_user())]

    stop = _terminal_stop(events)
    assert stop["outcome"] == RunOutcome.completed
    assert stop[RECOVERABLE_AFTER_FIELD] is None
    # The answer that came back ended the stretch the retry was drawn in.
    assert engine.recovery_budget.attempts == 0


@pytest.mark.asyncio
async def test_an_empty_round_draws_from_the_budget(engine_factory, in_memory_runtime) -> None:
    rc = LoopConstants(
        model_context_window=4_096,
        max_consecutive_empty_responses=3,
        soft_stop_enabled=False,
        recovery_budget_max_attempts=1,
    )
    engine = engine_factory(rc=rc)
    llm = _ThinkingOnlyLLM(empty_rounds=10)
    engine.llm = llm  # type: ignore[assignment]

    events = [event async for event in engine.run(_user())]

    # One re-ask on the budget instead of three on the empty-round count.
    assert len(llm.calls) == 2
    assert _reasons(events).count("continue_prompt_injected") == 1
    assert engine.recovery_budget.by_kind == {"empty_response": 1}


@pytest.mark.asyncio
async def test_an_overflow_draws_from_the_budget(engine_factory, in_memory_runtime) -> None:
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
        context_overflow_retry_max_attempts=1,
        recovery_budget_max_attempts=0,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(exceptions=[LLMContextWindowExceeded("PTL #1")])
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]

    events = [event async for event in engine.run(_user())]

    # Nothing left to spend on shrinking the request: no compaction, no retry.
    assert len(llm.calls) == 1
    assert not any(e.type is EventType.COMPACTION_STARTED for e in events)
    stop = _terminal_stop(events)
    assert RunCause.model_validate(stop["cause"]).kind == "llm_context_window_exceeded"
    assert stop[RECOVERABLE_AFTER_FIELD] is None


@pytest.mark.asyncio
async def test_a_host_re_drive_after_an_overflow_draws_from_the_same_budget(
    engine_factory, in_memory_runtime
) -> None:
    """The host keeps the re-drive (it is a new run) and reads the one budget."""
    rc = LoopConstants(
        model_context_window=65_536,
        llm_output_max_tokens_ratio=0.125,
        compaction_keep_recent_turns=1,
        context_overflow_retry_max_attempts=1,
        recovery_overflow_redrive_max_attempts=1,
    )
    engine = engine_factory(rc=rc)
    llm = _ScriptedFailureLLM(exceptions=[LLMContextWindowExceeded("PTL")] * 20)
    engine.llm = llm  # type: ignore[assignment]
    engine.context_manager._compaction_llm = llm  # type: ignore[attr-defined]
    engine.compaction_llm = llm  # type: ignore[assignment]

    first = [event async for event in engine.run(_user())]
    assert _terminal_stop(first)[RECOVERABLE_AFTER_FIELD] == 0.0

    budget = engine.recovery_budget
    assert budget.draw(Kind.overflow_redrive)
    engine.rearm(recovery_budget=budget)
    second = [event async for event in engine.run(_user("continue"))]

    # Recoverable once: the second overflow in a row is the end of it.
    assert _terminal_stop(second)[RECOVERABLE_AFTER_FIELD] is None
    assert engine.recovery_budget.by_kind["overflow_redrive"] == 1


@pytest.mark.asyncio
async def test_a_re_arm_starts_a_fresh_budget_unless_given_one(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc(recovery_budget_max_attempts=5))
    llm = _ScriptedLLM([_connection_refused()] * 10)
    engine.llm = llm  # type: ignore[assignment]
    [_ async for _ in engine.run(_user())]
    assert engine.recovery_budget.attempts > 0

    engine.rearm()

    assert engine.recovery_budget.attempts == 0
    assert engine.recovery_budget.max_attempts == 5


# -- it travels with the run ----------------------------------------------


@pytest.mark.asyncio
async def test_the_budget_survives_a_snapshot(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=_rc(recovery_budget_max_attempts=6))
    engine.recovery_budget.draw(Kind.chain_advance, now=50.0)
    engine.recovery_budget.draw(Kind.transient_retry, now=51.0)

    stored = json.loads(json.dumps(engine.snapshot(), default=str))
    successor = engine_factory(rc=_rc(recovery_budget_max_attempts=6))
    await successor.resume_from_snapshot(stored)

    assert successor.recovery_budget == engine.recovery_budget


def test_a_version_8_snapshot_lifts_to_one_with_no_budget() -> None:
    payload: dict[str, Any] = {
        SNAPSHOT_SCHEMA_KEY: 8,
        RUN_SCOPED_STATE_SNAPSHOT_KEY: {"run_work_ledger": None, "subagent_tree_budget": None},
    }

    lifted = migrate_snapshot(payload)

    assert lifted[RUN_SCOPED_STATE_SNAPSHOT_KEY]["recovery_budget"] is None
    assert lifted[RUN_SCOPED_STATE_SNAPSHOT_KEY]["run_work_ledger"] is None
