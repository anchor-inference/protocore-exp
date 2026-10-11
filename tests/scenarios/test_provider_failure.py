"""A provider that fails mid-stream, and the run that has to survive it.

The load-bearing scenario in this file is
:func:`test_the_replacement_provider_is_shown_the_partial_the_reader_saw`.
Stepping down the chain and rebuilding the context are one operation, and
separating them has no symptom at the seam: the run still completes, the
events still look right, and the only witness is the request the SECOND
provider receives. If the rebuilt context is not the one sent on, the
replacement answers a conversation that does not contain the characters the
reader is looking at — it repeats them, or re-issues a tool call whose
``tool_use`` block is already in history.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamIdleError,
)
from protocore.contracts.turn_policy import TurnContext, TurnCoordinate
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.turn_policies import TurnPolicyRegistry

from .conftest import (
    FailingProvider,
    ProviderChainDouble,
    ScenarioFactory,
    classified,
    default_rc,
)


def _idle() -> BaseException:
    return classified(LLMStreamIdleError("went quiet"), "timeout")


def _chain_scenario(
    scenario: ScenarioFactory,
    *,
    failure: BaseException,
    partial_text: str = "half an answer",
    rc: object | None = None,
    failures: dict[int, BaseException] | None = None,
):
    primary = FailingProvider(
        failures=failures if failures is not None else {0: failure},
        partial_text=partial_text,
    )
    fallback = FailingProvider(failures={}, text="the replacement answer")
    chain = ProviderChainDouble([("primary-model", primary), ("fallback-model", fallback)])
    run = scenario(
        rc=rc or default_rc(llm_transient_error_retry_backoff_base_seconds=0.0),
        llm_provider=primary,
        provider_chain=chain,
        model_name="primary-model",
    )
    return run, primary, fallback, chain


async def test_the_replacement_provider_is_shown_the_partial_the_reader_saw(
    scenario: ScenarioFactory,
) -> None:
    """The context sent to the new rung is the one rebuilt after the demotion."""
    run, _primary, fallback, _chain = _chain_scenario(
        scenario,
        failure=_idle(),
        failures={0: _idle(), 1: _idle()},
        partial_text="the first half",
    )

    await run.run("ask")

    assert len(fallback.calls) == 1
    replacement_texts = [
        block.text
        for message in fallback.calls[0].messages
        for block in message.content_blocks
        if getattr(block, "text", None) is not None
    ]
    assert any("the first half" in text for text in replacement_texts)


async def test_one_silent_stream_is_tried_again_on_the_same_provider(
    scenario: ScenarioFactory,
) -> None:
    """A single silence is a slow think as often as an outage: no demotion.

    A large model reasoning without streaming its reasoning went past the idle
    budget once, and the run was moved to a weaker provider — and from there to
    one out of quota — for a think the next attempt finished.
    """
    run, primary, fallback, chain = _chain_scenario(
        scenario, failure=_idle(), partial_text=""
    )

    produced = await run.run("ask")

    assert chain.advance_reasons == []
    assert len(primary.calls) == 2
    assert fallback.calls == ()
    retries = [
        evt
        for evt in produced
        if evt.type is EventType.STATE_CHANGED
        and evt.payload.get("reason") == "transient_llm_error_retry"
    ]
    assert [evt.payload.get("error_class") for evt in retries] == ["llm_stream_idle"]
    assert run.engine.snapshot()["model_name"] == "primary-model"


async def test_a_silence_with_no_retry_budget_moves_the_run_at_once(
    scenario: ScenarioFactory,
) -> None:
    run, primary, fallback, chain = _chain_scenario(
        scenario,
        failure=_idle(),
        partial_text="",
        rc=default_rc(llm_transient_error_retry_max_attempts=0),
    )

    await run.run("ask")

    assert chain.advance_reasons == ["llm_stream_idle"]
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


async def test_a_silent_stream_that_repeats_moves_the_run_to_the_next_provider(
    scenario: ScenarioFactory,
) -> None:
    run, primary, fallback, chain = _chain_scenario(
        scenario, failure=_idle(), failures={0: _idle(), 1: _idle()}
    )

    produced = await run.run("ask")

    assert chain.advance_reasons == ["llm_stream_idle"]
    assert len(primary.calls) == 2
    assert len(fallback.calls) == 1
    assert "model_fallback_triggered" in [
        str(evt.payload.get("reason", "")) for evt in produced
        if evt.type is EventType.STATE_CHANGED
    ]


async def test_the_demotion_is_announced_with_the_model_it_moved_to(
    scenario: ScenarioFactory,
) -> None:
    run, _primary, _fallback, _chain = _chain_scenario(
        scenario, failure=_idle(), failures={0: _idle(), 1: _idle()}
    )

    produced = await run.run("ask")

    fallbacks = [
        evt
        for evt in produced
        if evt.type is EventType.STATE_CHANGED
        and evt.payload.get("reason") == "model_fallback_triggered"
    ]
    assert fallbacks
    assert fallbacks[0].payload["fallback_model_id"] == "fallback-model"
    assert fallbacks[0].payload["error_class"] == "llm_stream_idle"


async def test_the_run_ends_on_the_rung_it_was_demoted_to(
    scenario: ScenarioFactory,
) -> None:
    """The cursor is one-way: the snapshot records where the run actually is."""
    run, _primary, _fallback, _chain = _chain_scenario(
        scenario, failure=classified(LLMProviderError("503"), "server_error")
    )

    await run.run("ask")

    assert run.engine.snapshot()["model_name"] == "fallback-model"
    assert run.engine.snapshot()["provider_chain_advances"] == 1


async def test_a_provider_error_carries_the_partial_across_too(
    scenario: ScenarioFactory,
) -> None:
    run, _primary, fallback, _chain = _chain_scenario(
        scenario,
        failure=classified(LLMProviderError("503"), "server_error"),
        partial_text="what the reader already has",
    )

    await run.run("ask")

    replacement_texts = [
        block.text
        for message in fallback.calls[0].messages
        for block in message.content_blocks
        if getattr(block, "text", None) is not None
    ]
    assert any("what the reader already has" in text for text in replacement_texts)


async def test_an_unclassified_provider_error_does_not_move_the_run(
    scenario: ScenarioFactory,
) -> None:
    """A policy refusal and a 503 arrive as one type; only a verdict advances."""
    run, _primary, fallback, chain = _chain_scenario(
        scenario, failure=LLMProviderError("refused, and it will refuse again")
    )

    await run.run("ask")

    assert chain.advance_reasons == []
    assert fallback.calls == ()


async def test_a_rate_limit_is_retried_in_place_when_there_is_nowhere_to_go(
    scenario: ScenarioFactory,
) -> None:
    primary = FailingProvider(failures={0: LLMRateLimitError("429")}, text="second try")
    run = scenario(
        rc=default_rc(
            llm_transient_error_retry_max_attempts=2,
            llm_transient_error_retry_backoff_base_seconds=0.001,
        ),
        llm_provider=primary,
        model_name="only-model",
    )

    produced = await run.run("ask")

    assert len(primary.calls) == 2
    assert "transient_llm_error_retry" in [
        str(evt.payload.get("reason", ""))
        for evt in produced
        if evt.type is EventType.STATE_CHANGED
    ]


async def test_a_rate_limit_prefers_a_healthy_sibling_over_its_own_backoff(
    scenario: ScenarioFactory,
) -> None:
    run, primary, fallback, chain = _chain_scenario(
        scenario, failure=LLMRateLimitError("429"), partial_text=""
    )

    await run.run("ask")

    assert chain.advance_reasons == ["llm_rate_limit"]
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


async def test_a_chain_with_nowhere_left_winds_the_run_down_rather_than_dropping_it(
    scenario: ScenarioFactory,
) -> None:
    primary = FailingProvider(
        failures={0: classified(LLMProviderError("503"), "server_error")},
        partial_text="all the reader got",
    )
    chain = ProviderChainDouble([("only-model", primary)])
    run = scenario(
        rc=default_rc(soft_stop_enabled=True),
        llm_provider=primary,
        provider_chain=chain,
        model_name="only-model",
    )

    await run.run("ask")

    assert chain.advance_reasons == ["llm_provider_error"]
    assert any("all the reader got" in text for text in run.history_texts())


async def test_the_advance_budget_bounds_how_far_a_run_walks_down_the_chain(
    scenario: ScenarioFactory,
) -> None:
    failure = classified(LLMProviderError("503"), "server_error")
    first = FailingProvider(failures={0: failure}, partial_text="a")
    second = FailingProvider(failures={0: failure}, partial_text="b")
    third = FailingProvider(failures={}, text="never reached")
    chain = ProviderChainDouble(
        [("m1", first), ("m2", second), ("m3", third)]
    )
    run = scenario(
        rc=default_rc(llm_provider_chain_max_advances=1, soft_stop_enabled=True),
        llm_provider=first,
        provider_chain=chain,
        model_name="m1",
    )

    await run.run("ask")

    assert len(chain.advance_reasons) == 1
    assert third.calls == ()


class _ShrugsAtTheFailure:
    """A host's provider-failure policy that declines to claim this class."""

    name = "provider_failure"
    coordinates = frozenset({TurnCoordinate.stream_failed})

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        # The outcome is left at ``proceed``: this policy has nothing to say
        # about the failure it was shown.
        return
        yield  # pragma: no cover - the generator never yields


async def test_a_failure_no_policy_claims_reaches_the_caller(
    scenario: ScenarioFactory,
) -> None:
    """A stream failure nobody answered is not an ending.

    ``proceed`` and ``end_turn`` used to be indistinguishable at this seam:
    a policy that declined the failure closed the run silently — no terminal,
    no state change, and the exception dropped — which is the one outcome a
    host can neither see nor act on.
    """
    primary = FailingProvider(
        failures={0: classified(LLMProviderError("the endpoint is gone"), "provider")},
        partial_text="",
    )
    run = scenario(
        llm_provider=primary,
        turn_policies=TurnPolicyRegistry([_ShrugsAtTheFailure()]),
    )

    with pytest.raises(LLMProviderError, match="the endpoint is gone"):
        await run.run("ask")
