"""Prices, and the spend admission a host hangs a budget on."""
from __future__ import annotations

import math
from typing import Any

import httpx
import pytest

from protocore.conformance import assert_adapter_failure
from protocore.contracts.llm import LLMError, LLMProviderError, ProviderDeltaKind, ProviderFailureReason
from protocore.providers.openai_compat import OpenAICompatibleProvider, normalize_usage
from protocore.providers.spend import (
    ISpendAdmission,
    ModelPricing,
    SpendEnd,
    SpendEstimate,
    SpendRefused,
    SpendSettlement,
    budget_error,
)

from ._openai_support import FakeServer, chunk_stream, collect, endpoint, sse_response, user_request

K = ProviderDeltaKind


# ModelPricing -----------------------------------------------------------------


def test_cost_prices_each_kind_of_token_at_its_own_rate() -> None:
    pricing = ModelPricing(input=2.0, output=8.0, cached_input=0.5, cache_write=2.5, reasoning=10.0)
    usage = {
        "input_tokens": 1_000_000,
        "cache_read_tokens": 400_000,
        "cache_creation_input_tokens": 100_000,
        "output_tokens": 300_000,
        "reasoning_tokens": 100_000,
    }

    # fresh 500k * 2 + read 400k * 0.5 + write 100k * 2.5 + visible 200k * 8 + reasoning 100k * 10
    assert pricing.cost(usage) == pytest.approx(1.0 + 0.2 + 0.25 + 1.6 + 1.0)


def test_unpriced_kinds_fall_back_to_the_rate_that_cannot_undercharge() -> None:
    pricing = ModelPricing(input=1.0, output=4.0)
    usage = {"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000, "output_tokens": 1_000_000,
             "reasoning_tokens": 500_000}

    # An omitted cache rate is the input rate and an omitted reasoning rate the
    # output rate: a missing number must not make tokens free.
    assert pricing.cost(usage) == pytest.approx(1.0 + 4.0)


def test_cost_reads_what_the_adapter_normalised() -> None:
    raw = {"prompt_tokens": 2_000, "completion_tokens": 500, "prompt_cache_hit_tokens": 1_500,
           "completion_tokens_details": {"reasoning_tokens": 100}}
    pricing = ModelPricing(input=1.0, output=2.0, cached_input=0.1)

    expected = (500 * 1.0 + 1_500 * 0.1 + 500 * 2.0) / 1_000_000
    assert pricing.cost(normalize_usage(raw)) == pytest.approx(expected)


def test_cost_tolerates_missing_and_odd_counts() -> None:
    pricing = ModelPricing(input=1.0, output=1.0)
    assert pricing.cost({}) == 0.0
    assert pricing.cost({"input_tokens": "x", "output_tokens": True, "reasoning_tokens": 5}) == 0.0


def test_the_estimate_is_an_upper_bound() -> None:
    pricing = ModelPricing(input=1.0, output=4.0, cached_input=0.1, reasoning=6.0)

    assert pricing.estimate(prompt_tokens=1_000_000, max_output_tokens=1_000_000) == pytest.approx(1.0 + 6.0)


@pytest.mark.parametrize("field", ["input", "output", "cached_input", "cache_write", "reasoning"])
def test_a_price_must_be_a_finite_non_negative_number(field: str) -> None:
    values: dict[str, Any] = {"input": 1.0, "output": 1.0, field: -1.0}
    with pytest.raises(ValueError, match=field):
        ModelPricing(**values)
    values[field] = math.inf
    with pytest.raises(ValueError, match=field):
        ModelPricing(**values)


def test_a_refusal_becomes_a_budget_failure_nothing_retries() -> None:
    error = budget_error("endpoint-a", SpendRefused("daily cap reached", retry_after_seconds=3600))

    failure = assert_adapter_failure(error)
    assert isinstance(error, LLMProviderError)
    assert failure.reason is ProviderFailureReason.budget
    assert failure.retryable is False
    assert failure.retry_after_seconds == 3600
    assert "daily cap reached" in str(error)


# Admission through the adapter ------------------------------------------------


class RecordingAdmission:
    def __init__(self, *, refuse: SpendRefused | None = None, reservation: str | None = "r-1") -> None:
        self.refuse = refuse
        self.reservation = reservation
        self.estimates: list[SpendEstimate] = []
        self.settled: list[tuple[str, SpendSettlement]] = []
        self.released: list[tuple[str, SpendEnd, str]] = []

    async def reserve(self, estimate: SpendEstimate) -> str | None:
        self.estimates.append(estimate)
        if self.refuse is not None:
            raise self.refuse
        return self.reservation

    async def settle(self, reservation: str, settlement: SpendSettlement) -> None:
        self.settled.append((reservation, settlement))

    async def release(self, reservation: str, *, outcome: SpendEnd, detail: str) -> None:
        self.released.append((reservation, outcome, detail))


def _provider(server: FakeServer, admission: ISpendAdmission | None, **kwargs: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(endpoint(), client=server.client(), admission=admission, **kwargs)


USAGE = {"prompt_tokens": 1_000, "completion_tokens": 200}
STREAM = chunk_stream(
    {"choices": [{"delta": {"content": "hi"}}]},
    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    {"choices": [], "usage": USAGE},
)


def test_the_recording_admission_is_an_admission() -> None:
    assert isinstance(RecordingAdmission(), ISpendAdmission)


async def test_a_stream_reserves_with_an_estimate_and_settles_with_its_usage() -> None:
    admission = RecordingAdmission()
    pricing = ModelPricing(input=1.0, output=2.0)
    server = FakeServer(sse_response(STREAM))
    deltas = await collect(
        _provider(server, admission, pricing=lambda model: pricing if model == "model-a" else None)
        .stream_with_tools(user_request(max_tokens=512))
    )

    [estimate] = admission.estimates
    assert estimate.endpoint == "test-endpoint"
    assert estimate.model == "model-a"
    assert estimate.purpose == "stream"
    assert estimate.max_output_tokens == 512
    assert estimate.prompt_tokens > 0
    assert estimate.cost_usd == pytest.approx(pricing.estimate(prompt_tokens=estimate.prompt_tokens,
                                                               max_output_tokens=512))
    assert estimate.observability is not None and estimate.observability.session_id == "sess-1"

    [(reservation, settlement)] = admission.settled
    assert reservation == "r-1"
    assert settlement.complete is True
    assert settlement.usage["input_tokens"] == 1_000
    assert settlement.cost_usd == pytest.approx((1_000 * 1.0 + 200 * 2.0) / 1_000_000)
    assert admission.released == []
    usage = next(d for d in deltas if d.kind is K.usage)
    assert usage.usage is not None and usage.usage["cost_usd"] == pytest.approx(settlement.cost_usd)


async def test_a_cost_the_server_reports_wins_over_the_price_table() -> None:
    admission = RecordingAdmission()
    body = chunk_stream({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]},
                        {"choices": [], "usage": {**USAGE, "cost": 0.5}})
    await collect(_provider(FakeServer(sse_response(body)), admission,
                            pricing=lambda _model: ModelPricing(input=100.0, output=100.0))
                  .stream_with_tools(user_request()))

    assert admission.settled[0][1].cost_usd == 0.5


async def test_a_stream_without_usage_settles_as_incomplete() -> None:
    admission = RecordingAdmission()
    body = chunk_stream({"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]})
    await collect(_provider(FakeServer(sse_response(body)), admission).stream_with_tools(user_request()))

    [(_, settlement)] = admission.settled
    assert settlement.complete is False
    assert settlement.cost_usd is None


async def test_a_refused_admission_sends_nothing_and_raises_a_budget_failure() -> None:
    admission = RecordingAdmission(refuse=SpendRefused("over the monthly cap"))
    server = FakeServer()

    with pytest.raises(LLMError) as raised:
        await collect(_provider(server, admission).stream_with_tools(user_request()))

    assert assert_adapter_failure(raised.value).reason is ProviderFailureReason.budget
    assert server.requests == []
    assert admission.settled == [] and admission.released == []


async def test_a_refused_request_releases_its_reservation_as_refused() -> None:
    admission = RecordingAdmission()
    server = FakeServer(httpx.Response(429, json={"error": {"message": "slow down"}}))

    with pytest.raises(LLMError):
        await collect(_provider(server, admission).stream_with_tools(user_request()))

    assert [(r, outcome) for r, outcome, _ in admission.released] == [("r-1", SpendEnd.refused)]
    assert admission.settled == []


async def test_an_abandoned_stream_releases_its_reservation_as_interrupted() -> None:
    admission = RecordingAdmission()
    body = chunk_stream(*({"choices": [{"delta": {"content": "x"}}]} for _ in range(5)), done=False)
    stream = _provider(FakeServer(sse_response(body)), admission).stream_with_tools(user_request())

    first = await anext(stream)
    assert first.kind is K.text
    await stream.aclose()

    assert [(r, outcome) for r, outcome, _ in admission.released] == [("r-1", SpendEnd.interrupted)]


async def test_a_transport_failure_releases_its_reservation_as_interrupted() -> None:
    admission = RecordingAdmission()

    async def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError):
        await collect(_provider(FakeServer(broken), admission).stream_with_tools(user_request()))

    assert [outcome for _, outcome, _ in admission.released] == [SpendEnd.interrupted]


async def test_no_reservation_means_nothing_to_settle() -> None:
    admission = RecordingAdmission(reservation=None)
    await collect(_provider(FakeServer(sse_response(STREAM)), admission).stream_with_tools(user_request()))

    assert len(admission.estimates) == 1
    assert admission.settled == [] and admission.released == []


async def test_a_plain_completion_is_admitted_and_settled() -> None:
    admission = RecordingAdmission()
    server = FakeServer(httpx.Response(200, json={
        "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}], "usage": USAGE,
    }))
    response = await _provider(server, admission, pricing=lambda _m: ModelPricing(input=1.0, output=1.0)) \
        .complete_text(user_request())

    assert admission.estimates[0].purpose == "text"
    assert admission.settled[0][1].complete is True
    assert response.usage is not None
    assert response.usage.response_cost_usd == pytest.approx(1_200 / 1_000_000)


async def test_a_refused_completion_releases_as_refused() -> None:
    admission = RecordingAdmission()
    server = FakeServer(httpx.Response(500, json={"error": {"message": "boom"}}))

    with pytest.raises(LLMError):
        await _provider(server, admission).complete_text(user_request())

    assert [outcome for _, outcome, _ in admission.released] == [SpendEnd.refused]


async def test_a_failing_settlement_does_not_fail_the_answer() -> None:
    class Broken(RecordingAdmission):
        async def settle(self, reservation: str, settlement: SpendSettlement) -> None:
            raise RuntimeError("ledger down")

    admission = Broken()
    deltas = await collect(_provider(FakeServer(sse_response(STREAM)), admission).stream_with_tools(user_request()))

    assert deltas[-1].kind is K.finish
    # Settlement was attempted, so the reservation is not released as well.
    assert admission.released == []
