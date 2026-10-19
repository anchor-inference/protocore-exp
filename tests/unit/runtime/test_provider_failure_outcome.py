"""A typed provider verdict drives the loop, and the run says how it ended.

Two consequences of the verdict being a published type rather than a shape the
loop reads by attribute name:

* a connection the endpoint refused is a failure a sibling provider can serve,
  so the chain moves; before, it arrived with no verdict and ended the run;
* a server-stated ``Retry-After`` sets the backoff, within the configured cap.

And the terminal ``message_stop`` carries the run's ``outcome`` with a typed
``cause``: a run wound down because the provider failed, or because the model
made no progress, ended ``failed`` even when the closing turn wrote prose, and
a consumer had to infer that from the wind-down cause and match strings in the
detail to tell its own spend cap from a provider refusal.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRequest,
    LLMStreamEvent,
    ProviderFailure,
    ProviderFailureReason,
)
from protocore.contracts.outcome import RunCause, RunOutcome
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock
from protocore.runtime import soft_stop as _soft_stop
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _transient_retry_backoff_seconds
from tests.unit.runtime.test_soft_stop import (
    _build_engine,
    _FailsOnLLM,
    _FinalizeTool,
    _NamedTool,
)
from tests.unit.runtime.test_soft_stop import _user as _soft_stop_user

Reason = ProviderFailureReason


def _connection_refused() -> LLMProviderError:
    return LLMProviderError(
        "transport error: connection refused",
        classified=ProviderFailure(
            reason=Reason.connection,
            retryable=True,
            provider_message="[Errno 111] Connection refused",
        ),
    )


class _ScriptedLLM:
    """Raise ``failures[i]`` on call ``i``; a call past the script answers."""

    def __init__(self, failures: list[BaseException | None]) -> None:
        self._failures = failures
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        index = len(self.calls)
        self.calls.append(request)
        if index < len(self._failures) and self._failures[index] is not None:
            raise self._failures[index]  # type: ignore[misc]
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        yield LLMStreamEvent(name="content_block_delta", payload={"text": "the answer"})
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        raise AssertionError("not used")

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


class _Chain:
    """A one-way chain over one scripted provider under several model names."""

    def __init__(self, provider: object, names: list[str]) -> None:
        self._provider = provider
        self._names = names
        self._index = 0

    def current(self) -> object:
        return self._provider

    def current_model_name(self) -> str:
        return self._names[self._index]

    async def advance(self, *, reason: str) -> bool:
        if self._index + 1 >= len(self._names):
            return False
        self._index += 1
        return True

    def attempted(self) -> list[tuple[str, str]]:
        return []


def _rc(**overrides: object) -> LoopConstants:
    base: dict[str, object] = {
        "model_context_window": 4096,
        "llm_transient_error_retry_backoff_base_seconds": 0.0,
        "llm_transient_error_retry_backoff_max_seconds": 0.0,
        "llm_transient_error_retry_max_attempts": 2,
    }
    base.update(overrides)
    return LoopConstants(**base)  # type: ignore[arg-type]


def _user(text: str = "what changed overnight?") -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _reasons(events: list[TurnEvent]) -> list[str]:
    return [str(e.payload.get("reason")) for e in events if e.type is EventType.STATE_CHANGED]


def _terminal_stop(events: list[TurnEvent]) -> dict[str, Any]:
    stops = [
        e.payload
        for e in events
        if e.type is EventType.MESSAGE_STOP and e.payload.get("stop_reason") != "tool_use"
    ]
    assert stops, "the run never said it stopped"
    return stops[-1]


def _cause(payload: dict[str, Any]) -> RunCause:
    raw = payload.get("cause")
    assert raw is not None, "a run that did not simply complete must say why"
    return RunCause.model_validate(raw)


# -- the chain moves past a dead endpoint ---------------------------------


@pytest.mark.asyncio
async def test_a_refused_connection_moves_the_chain_to_a_rung_that_answers(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([_connection_refused(), _connection_refused()])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary", "tertiary"])

    events = [event async for event in engine.run(_user())]

    assert _reasons(events).count("model_fallback_triggered") == 2
    assert "transient_llm_error_retry" not in _reasons(events)
    assert engine.config.model_name == "tertiary"
    assert engine.state is LoopState.COMPLETED
    stop = _terminal_stop(events)
    assert stop["outcome"] == RunOutcome.completed
    assert stop["cause"] is None


@pytest.mark.asyncio
async def test_an_unclassified_provider_error_still_does_not_move_the_chain(
    engine_factory, in_memory_runtime
) -> None:
    """The principle stands: a failure nobody characterised is not spent on N
    providers. Only the verdict moves the chain."""
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([LLMProviderError("transport error: connection refused")])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary"])
    started_on = engine.config.model_name

    events = [event async for event in engine.run(_user())]

    assert "model_fallback_triggered" not in _reasons(events)
    assert engine.config.model_name == started_on


# -- the server says when to come back ------------------------------------


def _limited(seconds: float) -> LLMProviderError:
    return LLMProviderError(
        "HTTP 429",
        classified=ProviderFailure(
            reason=Reason.rate_limit, retryable=True, retry_after_seconds=seconds
        ),
    )


def test_retry_after_sets_the_backoff_within_the_ceiling() -> None:
    roomy = _rc(
        llm_transient_error_retry_backoff_base_seconds=1.0,
        llm_transient_error_retry_backoff_max_seconds=60.0,
    )
    tight = _rc(
        llm_transient_error_retry_backoff_base_seconds=1.0,
        llm_transient_error_retry_backoff_max_seconds=8.0,
    )

    assert _transient_retry_backoff_seconds(roomy, 1, _limited(30.0)) == 30.0
    assert _transient_retry_backoff_seconds(tight, 1, _limited(30.0)) == 8.0
    # A hint shorter than the schedule does not shorten it.
    assert _transient_retry_backoff_seconds(roomy, 3, _limited(0.5)) == 4.0


@pytest.mark.asyncio
async def test_the_retry_announced_on_the_wire_waits_as_long_as_the_server_asked(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(
        rc=_rc(
            llm_transient_error_retry_backoff_base_seconds=0.001,
            llm_transient_error_retry_backoff_max_seconds=0.5,
        )
    )
    llm = _ScriptedLLM([_limited(0.05)])
    engine.llm = llm  # type: ignore[assignment]

    events = [event async for event in engine.run(_user())]

    retries = [
        e.payload
        for e in events
        if e.type is EventType.STATE_CHANGED
        and e.payload.get("reason") == "transient_llm_error_retry"
    ]
    assert [r["backoff_seconds"] for r in retries] == [0.05]
    assert engine.state is LoopState.COMPLETED


# -- how the run ended -----------------------------------------------------


def _provider_down() -> LLMProviderError:
    return LLMProviderError(
        "HTTP 503: upstream unavailable",
        classified=ProviderFailure(
            reason=Reason.server_error,
            retryable=True,
            provider_message="upstream unavailable",
        ),
    )


@pytest.mark.asyncio
async def test_a_provider_wind_down_ends_failed_with_the_verdict_as_its_cause() -> None:
    rc = LoopConstants(model_context_window=4_096, llm_transient_error_retry_max_attempts=0)
    llm = _FailsOnLLM(
        [
            {"tool": "Read", "args": {"x": "a"}},
            {"text": "Here is what I found before the provider went away."},
        ],
        _provider_down(),
        fail_on={2},
    )
    engine = _build_engine(rc=rc, llm=llm, tools=[_NamedTool("Read"), _FinalizeTool()])

    events = [evt async for evt in engine.run(_soft_stop_user())]

    assert "soft_stop_notified" in _reasons(events)
    stop = _terminal_stop(events)
    # The closing turn did write prose; that is why the outcome is a field of
    # its own rather than a reading of ``has_final_answer``.
    assert stop["has_final_answer"] is True
    assert stop["outcome"] == RunOutcome.failed
    cause = _cause(stop)
    assert cause.kind == _soft_stop.CAUSE_PROVIDER_ERROR
    assert "upstream unavailable" in cause.detail
    assert cause.failure == ProviderFailure(
        reason=Reason.server_error,
        retryable=True,
        provider_message="upstream unavailable",
    )


@pytest.mark.asyncio
async def test_a_spend_cap_fails_the_run_at_once_and_says_it_was_the_cap(
    engine_factory, in_memory_runtime
) -> None:
    """The host's own budget, told apart from a provider refusal by the reason
    rather than by the wording of the detail."""
    capped = LLMProviderError(
        "spend cap reached for this scope",
        classified=ProviderFailure(reason=Reason.budget, retryable=False),
    )
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([capped, capped, capped])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary"])

    events = [event async for event in engine.run(_user())]

    assert len(llm.calls) == 1
    assert "model_fallback_triggered" not in _reasons(events)
    assert "soft_stop_notified" not in _reasons(events)
    stop = _terminal_stop(events)
    assert stop["stop_reason"] == StopReason.error.value
    assert stop["outcome"] == RunOutcome.failed
    cause = _cause(stop)
    assert cause.failure is not None
    assert cause.failure.reason is Reason.budget


@pytest.mark.asyncio
async def test_a_transient_failure_that_recovered_still_completes(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    engine.llm = _ScriptedLLM([_provider_down()])  # type: ignore[assignment]

    events = [event async for event in engine.run(_user())]

    stop = _terminal_stop(events)
    assert stop["outcome"] == RunOutcome.completed
    assert stop["cause"] is None


@pytest.mark.asyncio
async def test_a_crash_ends_failed_with_no_provider_verdict(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    engine.llm = _ScriptedLLM([RuntimeError("parser bug")])  # type: ignore[assignment]

    events = [event async for event in engine.run(_user())]

    stop = _terminal_stop(events)
    assert stop["outcome"] == RunOutcome.failed
    cause = _cause(stop)
    assert cause.kind == "internal_error"
    assert cause.failure is None


def _stamp(engine: Any, stop_reason: str = "end_turn") -> dict[str, Any]:
    event = TurnEvent(
        type=EventType.MESSAGE_STOP,
        run_id=engine.config.run_id,
        payload={"turn_id": "t", "stop_reason": stop_reason},
    )
    return dict(engine._stamp_run_outcome(event).payload)


@pytest.mark.parametrize(
    ("cause_name", "outcome"),
    [
        (_soft_stop.CAUSE_MODEL_NO_PROGRESS, RunOutcome.failed),
        (_soft_stop.CAUSE_PROVIDER_ERROR, RunOutcome.failed),
        (_soft_stop.CAUSE_MAX_TURNS, RunOutcome.wound_down),
        (_soft_stop.CAUSE_TOOL_CALL_BUDGET, RunOutcome.wound_down),
        (_soft_stop.CAUSE_DEADLINE, RunOutcome.wound_down),
        (_soft_stop.CAUSE_OUTPUT_TOKEN_BUDGET, RunOutcome.wound_down),
    ],
)
def test_a_wind_down_reads_as_failed_only_when_nothing_was_spent(
    engine_factory, cause_name: str, outcome: RunOutcome
) -> None:
    """A budget the run used up is a bound it hit; a provider that refused it
    or a model that went nowhere is a failure, whatever the closing turn said."""
    engine = engine_factory()
    _soft_stop.enter(engine, cause_name=cause_name, detail="the detail")

    payload = _stamp(engine)

    assert payload["outcome"] == outcome
    cause = _cause(payload)
    assert cause.kind == cause_name
    assert cause.detail == "the detail"
    assert cause.failure is None


def test_a_cancelled_run_reads_as_stopped(engine_factory) -> None:
    engine = engine_factory()

    payload = _stamp(engine, StopReason.cancelled.value)

    assert payload["outcome"] == RunOutcome.stopped


def test_a_round_boundary_carries_no_outcome(engine_factory) -> None:
    engine = engine_factory()

    payload = _stamp(engine, "tool_use")

    assert "outcome" not in payload
    assert "cause" not in payload


@pytest.mark.asyncio
async def test_the_wind_down_cause_survives_a_move_to_another_process(engine_factory) -> None:
    source = engine_factory()
    failure = ProviderFailure(reason=Reason.connection, retryable=True)
    _soft_stop.enter(
        source,
        cause_name=_soft_stop.CAUSE_PROVIDER_ERROR,
        detail="LLMProviderError: refused",
        failure=failure,
    )
    payload = source.snapshot()

    resumed = engine_factory()
    await resumed.resume_from_snapshot(payload)

    stop = _stamp(resumed)
    assert stop["outcome"] == RunOutcome.failed
    assert _cause(stop).failure == failure
