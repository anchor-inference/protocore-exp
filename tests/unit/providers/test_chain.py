"""``ListProviderChain``: the concrete one-way cursor over a run's providers."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from protocore.conformance import ProviderChainConformance
from protocore.contracts.llm import ILLMProvider, IProviderChain
from protocore.providers.chain import ChainRung, ListProviderChain
from protocore.tests_support.adapters import InMemoryLLMProvider


def _rung(name: str, **kwargs: Any) -> ChainRung:
    kwargs.setdefault("provider", InMemoryLLMProvider())
    return ChainRung(name=name, model=f"model-{name}", **kwargs)


class TestListProviderChainConformance(ProviderChainConformance):
    @pytest.fixture
    def subject_factory(self) -> Callable[[], Any]:
        return lambda: ListProviderChain([_rung("a"), _rung("b")])


def test_is_the_core_contract() -> None:
    assert isinstance(ListProviderChain([_rung("a")]), IProviderChain)


def test_a_chain_needs_a_built_first_rung() -> None:
    with pytest.raises(ValueError, match="at least one rung"):
        ListProviderChain([])

    async def factory() -> ILLMProvider:
        return InMemoryLLMProvider()

    # current() is synchronous, so the provider serving first must exist
    # before the run asks for it.
    with pytest.raises(ValueError, match="first rung"):
        ListProviderChain([ChainRung(name="a", model="m", factory=factory)])


def test_a_rung_names_exactly_one_way_to_get_its_provider() -> None:
    async def factory() -> ILLMProvider:
        return InMemoryLLMProvider()

    with pytest.raises(ValueError, match="exactly one"):
        ChainRung(name="a", model="m")
    with pytest.raises(ValueError, match="exactly one"):
        ChainRung(name="a", model="m", provider=InMemoryLLMProvider(), factory=factory)


async def test_advance_moves_the_whole_provider_and_records_why() -> None:
    first, second = _rung("a"), _rung("b")
    chain = ListProviderChain([first, second])
    assert chain.current() is first.provider
    assert chain.current_model_name() == "model-a"

    assert await chain.advance(reason="rate_limit") is True

    assert chain.current() is second.provider
    assert chain.current_model_name() == "model-b"
    assert chain.position == 1
    assert list(chain.attempted()) == [("a", "rate_limit")]


async def test_exhausted_chain_stays_where_it_is_and_records_nothing() -> None:
    chain = ListProviderChain([_rung("a")])

    assert await chain.advance(reason="overloaded") is False

    assert chain.current_model_name() == "model-a"
    assert list(chain.attempted()) == []


async def test_a_rung_that_cannot_hold_the_prompt_is_skipped_with_its_reason() -> None:
    chain = ListProviderChain(
        [
            _rung("a", context_window=200_000, max_output_tokens=8_000),
            _rung("small", context_window=32_000, max_output_tokens=4_000),
            _rung("large", context_window=128_000, max_output_tokens=8_000),
        ],
        prompt_tokens=lambda: 60_000,
    )

    assert await chain.advance(reason="timeout") is True

    assert chain.current_model_name() == "model-large"
    attempted = list(chain.attempted())
    assert attempted[0] == ("a", "timeout")
    assert attempted[1][0] == "small"
    assert "32000" in attempted[1][1] and "60000" in attempted[1][1]


async def test_prompt_plus_reply_is_what_has_to_fit() -> None:
    chain = ListProviderChain(
        [_rung("a"), _rung("tight", context_window=64_000, max_output_tokens=8_000)],
        prompt_tokens=lambda: 60_000,
    )

    # 60k fits in 64k, but not with the 8k reply the rung asks for: a demotion
    # onto it would trade a stalled stream for a context-window refusal.
    assert await chain.advance(reason="timeout") is False
    assert chain.current_model_name() == "model-a"


async def test_unknown_window_or_unknown_prompt_never_skips() -> None:
    no_window = ListProviderChain([_rung("a"), _rung("b")], prompt_tokens=lambda: 10**9)
    assert await no_window.advance(reason="timeout") is True

    unmeasured = ListProviderChain([_rung("a"), _rung("b", context_window=1_000)])
    assert await unmeasured.advance(reason="timeout") is True


async def test_prompt_size_can_be_bound_after_construction() -> None:
    chain = ListProviderChain([_rung("a"), _rung("b", context_window=1_000, max_output_tokens=100)])
    chain.bind_prompt_size(lambda: 5_000)

    assert await chain.advance(reason="timeout") is False


async def test_a_resting_fallback_is_skipped_but_the_primary_is_never_consulted() -> None:
    asked: list[str] = []

    def resting(rung: ChainRung) -> str | None:
        asked.append(rung.name)
        return "quota spent until tomorrow" if rung.name == "b" else None

    chain = ListProviderChain([_rung("a"), _rung("b"), _rung("c")], resting=resting)

    assert await chain.advance(reason="rate_limit") is True

    assert chain.current_model_name() == "model-c"
    assert ("b", "quota spent until tomorrow") in list(chain.attempted())
    assert "a" not in asked


async def test_a_lazy_rung_is_built_only_when_the_run_reaches_it() -> None:
    built: list[str] = []
    late = InMemoryLLMProvider()

    async def factory() -> ILLMProvider:
        built.append("b")
        return late

    chain = ListProviderChain([_rung("a"), ChainRung(name="b", model="model-b", factory=factory)])
    assert built == []

    assert await chain.advance(reason="timeout") is True

    assert built == ["b"]
    assert chain.current() is late


async def test_a_rung_that_fails_to_build_is_ruled_out_and_the_walk_goes_on() -> None:
    async def broken() -> ILLMProvider:
        raise RuntimeError("no credentials for this vendor")

    chain = ListProviderChain(
        [_rung("a"), ChainRung(name="b", model="model-b", factory=broken), _rung("c")]
    )

    assert await chain.advance(reason="timeout") is True

    assert chain.current_model_name() == "model-c"
    reasons = dict(chain.attempted())
    assert "no credentials" in reasons["b"]


async def test_concurrent_failures_do_not_demote_past_each_other() -> None:
    gate = asyncio.Event()

    async def slow() -> ILLMProvider:
        await gate.wait()
        return InMemoryLLMProvider()

    chain = ListProviderChain(
        [_rung("a"), ChainRung(name="b", model="model-b", factory=slow), _rung("c")]
    )
    first = asyncio.create_task(chain.advance(reason="timeout"))
    second = asyncio.create_task(chain.advance(reason="timeout"))
    await asyncio.sleep(0)
    gate.set()
    results = await asyncio.gather(first, second)

    # One transition at a time: the second advance starts from where the first
    # one landed instead of both deciding from position zero.
    assert results == [True, True]
    assert chain.current_model_name() == "model-c"
    assert [name for name, _ in chain.attempted()] == ["a", "b"]


def test_rungs_are_exposed_read_only() -> None:
    rungs = [_rung("a"), _rung("b")]
    chain = ListProviderChain(rungs)
    assert chain.rungs == tuple(rungs)
