"""The stream idle watchdog against a clock that only moves when the loop is idle.

Every scenario here is about minutes of silence, so it runs on an event loop
whose clock jumps straight to the next timer instead of waiting for it: a
"120 s think" costs microseconds, and the watchdog's own ``wait_for`` is what
decides when the jump stops. Nothing sleeps for real and nothing reaches the
network.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from protocore.contracts.llm import (
    LLMRequest,
    LLMStreamIdleError,
    ProviderDelta,
    ProviderDeltaKind,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.query import _first_delta_budget, _iter_with_idle_watchdog
from tests._fixtures.virtual_clock import run_on_virtual_clock as _run

BASELINE = 90.0
REASONING = 300.0


def _text(content: str = "answer") -> ProviderDelta:
    return ProviderDelta(kind=ProviderDeltaKind.text, content=content)


def _ping() -> ProviderDelta:
    return ProviderDelta(kind=ProviderDeltaKind.progress)


async def _script(steps: list[float | ProviderDelta]) -> AsyncIterator[ProviderDelta]:
    """A provider stream: a float is that many seconds of silence, a delta is emitted."""
    for step in steps:
        if isinstance(step, ProviderDelta):
            yield step
        else:
            await asyncio.sleep(step)
    yield ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop")


def _drain(
    steps: list[float | ProviderDelta], *, first_delta_timeout: float | None = None
) -> Callable[[], Awaitable[list[ProviderDelta]]]:
    async def go() -> list[ProviderDelta]:
        seen: list[ProviderDelta] = []
        async for delta in _iter_with_idle_watchdog(
            _script(steps),
            idle_timeout=BASELINE,
            stall_threshold=5.0,
            reasoning_idle_timeout=REASONING,
            first_delta_timeout=first_delta_timeout,
        ):
            seen.append(delta)
        return seen

    return go


def test_a_silent_two_minute_think_with_pings_survives_the_baseline() -> None:
    """Heartbeats are liveness: eight pings across 120 s keep a 90 s budget open."""
    steps: list[float | ProviderDelta] = []
    for _ in range(8):
        steps += [15.0, _ping()]
    steps += [_text()]

    seen, elapsed = _run(_drain(steps))

    assert elapsed == pytest.approx(120.0)
    assert [d.kind for d in seen][-2:] == [ProviderDeltaKind.text, ProviderDeltaKind.finish]


def test_a_reasoning_model_may_stay_silent_until_the_reasoning_budget_before_its_first_token() -> None:
    seen, elapsed = _run(_drain([250.0, _text()], first_delta_timeout=REASONING))

    assert elapsed == pytest.approx(250.0)
    assert seen[0].kind is ProviderDeltaKind.text


def test_pings_before_the_first_token_do_not_close_the_first_token_window() -> None:
    seen, _ = _run(
        _drain([5.0, _ping(), 200.0, _ping(), 200.0, _text()], first_delta_timeout=REASONING)
    )

    assert seen[-2].kind is ProviderDeltaKind.text


def test_a_reasoning_model_that_says_nothing_past_the_budget_still_dies() -> None:
    with pytest.raises(LLMStreamIdleError, match=r">300\.0s \(window=first_token"):
        _run(_drain([400.0, _text()], first_delta_timeout=REASONING))


def test_a_request_without_thinking_keeps_the_baseline_before_its_first_token() -> None:
    with pytest.raises(LLMStreamIdleError, match=r">90\.0s \(window=baseline"):
        _run(_drain([100.0, _text()]))


def test_after_the_first_token_a_true_hang_dies_on_the_baseline() -> None:
    """The larger window is for the think before the answer, not for a stream that stopped."""
    with pytest.raises(LLMStreamIdleError, match=r">90\.0s \(window=baseline"):
        _run(_drain([_text("half"), 100.0, _text("rest")], first_delta_timeout=REASONING))


def test_a_hang_that_only_ever_pinged_once_still_dies() -> None:
    with pytest.raises(LLMStreamIdleError, match=r">90\.0s \(window=baseline"):
        _run(_drain([_ping(), _text("half"), 95.0, _text("rest")]))


def test_time_the_consumer_spends_between_deltas_is_not_charged_to_the_stream() -> None:
    """The budget covers waiting on the provider, never the work done with what it sent."""

    async def go() -> int:
        count = 0
        async for _delta in _iter_with_idle_watchdog(
            _script([_text("a"), _text("b")]),
            idle_timeout=BASELINE,
            stall_threshold=5.0,
        ):
            count += 1
            await asyncio.sleep(500.0)
        return count

    count, elapsed = _run(go)

    assert count == 3
    assert elapsed == pytest.approx(1500.0)


def _request(extra: dict[str, Any]) -> LLMRequest:
    return LLMRequest(model="m", messages=[], extra=extra)


def test_the_first_token_budget_follows_what_the_request_asked_for() -> None:
    rc = LoopConstants(llm_stream_reasoning_idle_timeout_seconds=REASONING)

    thinking = _request({"enable_thinking": True, "reasoning_effort": "medium"})
    plain = _request({"enable_thinking": False, "reasoning_effort": "medium"})
    unsaid = _request({})

    assert _first_delta_budget(thinking, rc) == REASONING
    assert _first_delta_budget(plain, rc) is None
    assert _first_delta_budget(unsaid, rc) is None
