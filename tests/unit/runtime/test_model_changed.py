"""The core says when the model answering a run changes, and why.

A fallback retry reopens the stream inside the same assistant message, so no
second ``message_start`` marks the switch, and a snapshot can put a run back on
a later rung of its chain with nothing said at all. A host that wanted to show
which model was answering rebuilt the answer from a ``state_changed`` reason
and the next ``message_start``, and missed the resume case entirely.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRequest,
    LLMStreamEvent,
    ProviderFailure,
    ProviderFailureReason,
)
from protocore.runtime.events import EventType, TurnEvent
from tests.unit.runtime.test_provider_failure_outcome import (
    _Chain,
    _connection_refused,
    _rc,
    _ScriptedLLM,
    _user,
)


class _BreaksMidStream:
    """Call 0 streams a few words and then loses the connection; later calls answer."""

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    async def stream_with_tools(  # type: ignore[no-untyped-def]
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        index = len(self.calls)
        self.calls.append(request)
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
        if index == 0:
            yield LLMStreamEvent(name="content_block_delta", payload={"text": "partial "})
            raise _connection_refused()
        yield LLMStreamEvent(name="content_block_delta", payload={"text": "the answer"})
        yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})

    async def complete_structured(self, request, schema):  # type: ignore[no-untyped-def]
        raise AssertionError("not used")

    def count_tokens(self, text, model=None) -> int:  # type: ignore[no-untyped-def]
        return max(1, len(text) // 4)


def _changes(events: list[TurnEvent]) -> list[dict[str, Any]]:
    return [e.payload for e in events if e.type is EventType.MODEL_CHANGED]


def _delta_index(events: list[TurnEvent], text: str) -> int:
    return next(
        i
        for i, e in enumerate(events)
        if e.type is EventType.CONTENT_BLOCK_DELTA
        and text in str((e.payload.get("delta") or {}).get("text", ""))
    )


async def test_a_chain_advance_mid_stream_says_so_before_the_new_models_words(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _BreaksMidStream()
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary"])

    events = [event async for event in engine.run(_user())]

    [change] = _changes(events)
    assert change["from"] == "primary"
    assert change["to"] == "secondary"
    assert change["reason"] == "provider_chain_advance"
    assert change["error_class"] == "llm_provider_error"
    failure = ProviderFailure.model_validate(change["failure"])
    assert failure.reason is ProviderFailureReason.connection
    position = next(i for i, e in enumerate(events) if e.type is EventType.MODEL_CHANGED)
    assert _delta_index(events, "partial") < position < _delta_index(events, "the answer")


async def test_every_advance_is_announced_from_the_rung_it_left(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([_connection_refused(), _connection_refused()])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary", "tertiary"])

    events = [event async for event in engine.run(_user())]

    assert [(c["from"], c["to"]) for c in _changes(events)] == [
        ("primary", "secondary"),
        ("secondary", "tertiary"),
    ]


async def test_an_error_without_a_verdict_carries_none(engine_factory, in_memory_runtime) -> None:
    engine = engine_factory(rc=_rc(llm_transient_error_retry_max_attempts=0))
    from protocore.contracts.llm import LLMTimeoutError

    llm = _ScriptedLLM([LLMTimeoutError("read timed out")])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary"])

    events = [event async for event in engine.run(_user())]

    [change] = _changes(events)
    assert change["error_class"] == "llm_timeout"
    assert change["failure"] is None or change["failure"]["reason"] == "timeout"


async def test_a_run_that_stays_on_its_provider_announces_nothing(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary"])

    events = [event async for event in engine.run(_user())]

    assert _changes(events) == []


async def test_a_refusal_that_does_not_move_the_chain_announces_nothing(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([LLMProviderError("refused")])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary"])

    events = [event async for event in engine.run(_user())]

    assert _changes(events) == []


async def test_a_run_resumed_on_a_later_rung_says_so_once(engine_factory, in_memory_runtime) -> None:
    first = engine_factory(rc=_rc())
    llm = _ScriptedLLM([_connection_refused()])
    first.llm = llm  # type: ignore[assignment]
    first.provider_chain = _Chain(llm, ["primary", "secondary"])
    [_ async for _ in first.run(_user())]
    assert first.config.model_name == "secondary"
    stored = first.snapshot()

    successor = engine_factory(rc=_rc())
    successor.llm = llm  # type: ignore[assignment]
    successor.provider_chain = _Chain(llm, ["primary", "secondary"])
    await successor.resume_from_snapshot(stored)
    successor.rearm()
    events = [event async for event in successor.run(_user("and then?"))]

    [change] = _changes(events)
    assert change == {
        "from": "primary",
        "to": "secondary",
        "reason": "snapshot_restore",
        "error_class": None,
        "failure": None,
    }
    # Said once: the next run on the same engine is already on that rung.
    successor.rearm()
    later = [event async for event in successor.run(_user("more"))]
    assert _changes(later) == []


async def test_a_live_engine_resumed_from_its_own_snapshot_announces_nothing(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory(rc=_rc())
    llm = _ScriptedLLM([_connection_refused()])
    engine.llm = llm  # type: ignore[assignment]
    engine.provider_chain = _Chain(llm, ["primary", "secondary"])
    [_ async for _ in engine.run(_user())]

    await engine.resume_from_snapshot(engine.snapshot())
    engine.rearm()
    events = [event async for event in engine.run(_user("again"))]

    assert _changes(events) == []
