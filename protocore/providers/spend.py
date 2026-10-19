"""Prices per model, and the hook a host enforces a spend budget through.

The loop budgets tokens; money is the host's to budget, because only the host
knows what an account costs, what was already spent today, and who pays. What
the core can do is give that decision one place to happen and the numbers to
make it with:

* :class:`ModelPricing` turns a call's usage into dollars, pricing fresh input,
  cached input, cache writes, visible output and reasoning each at its own
  rate, and gives an upper-bound estimate before a call is sent.
* :class:`ISpendAdmission` is asked before every call (``reserve``, with an
  estimate) and told afterwards how it ended: ``settle`` with the usage the
  server reported, or ``release`` when no usage ever came (the server refused,
  the stream broke, the caller stopped reading). Refusing is raising
  :class:`SpendRefused`, which reaches the loop as a ``budget`` failure: every
  rung of the chain meets the same cap, so nothing retries it or moves past it.

Pure Python. The reference adapter takes both (``admission=``, ``pricing=``);
another adapter calls the same three methods around its own transport.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from protocore.contracts.llm import (
    LLMObservabilityContext,
    LLMProviderError,
    ProviderFailure,
    ProviderFailureReason,
)

_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """One model's prices, in US dollars per one million tokens.

    ``cached_input`` and ``cache_write`` default to the ``input`` price and
    ``reasoning`` to the ``output`` price. A rate left out must never make
    tokens free: a table that names no cache price would otherwise book every
    cache hit as costing nothing, and a budget built on it would be spent
    before anyone noticed.
    """

    input: float
    output: float
    cached_input: float | None = None
    cache_write: float | None = None
    reasoning: float | None = None

    def __post_init__(self) -> None:
        prices = {
            "input": self.input,
            "output": self.output,
            "cached_input": self.cached_input,
            "cache_write": self.cache_write,
            "reasoning": self.reasoning,
        }
        for name, value in prices.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"ModelPricing.{name} must be a finite non-negative number, got {value!r}")

    def cost(self, usage: Mapping[str, Any]) -> float:
        """The dollars a call with this usage cost.

        ``usage`` uses the keys the core's adapters report:
        ``input_tokens`` is the whole prompt, cache reads
        (``cache_read_tokens`` or ``cache_read_input_tokens``) and cache writes
        (``cache_creation_input_tokens``) included; ``output_tokens`` is the
        whole reply, ``reasoning_tokens`` included. A missing or unreadable
        count is zero.
        """
        prompt = _count(usage, "input_tokens")
        cache_read = _count(usage, "cache_read_tokens") or _count(usage, "cache_read_input_tokens")
        cache_write = _count(usage, "cache_creation_input_tokens")
        fresh = max(prompt - cache_read - cache_write, 0)
        output = _count(usage, "output_tokens")
        reasoning = min(_count(usage, "reasoning_tokens"), output)
        total = (
            fresh * self.input
            + cache_read * self._cached_input
            + cache_write * self._cache_write
            + (output - reasoning) * self.output
            + reasoning * self._reasoning
        )
        return total / _PER_MILLION

    def estimate(self, *, prompt_tokens: int, max_output_tokens: int) -> float:
        """The most a call of this size can cost: no cache hit, and the whole output cap spent.

        An estimate made to reserve money has to err high: a reservation that
        is too small lets a call through that the budget could not pay for.
        """
        dearest_output = max(self.output, self._reasoning)
        return (max(prompt_tokens, 0) * max(self.input, self._cache_write)
                + max(max_output_tokens, 0) * dearest_output) / _PER_MILLION

    @property
    def _cached_input(self) -> float:
        return self.input if self.cached_input is None else self.cached_input

    @property
    def _cache_write(self) -> float:
        return self.input if self.cache_write is None else self.cache_write

    @property
    def _reasoning(self) -> float:
        return self.output if self.reasoning is None else self.reasoning


def _count(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return 0
    return max(int(value), 0)


PricingLookup = Callable[[str], ModelPricing | None]
"""The prices of a model by its name, or ``None`` when they are not known."""


@dataclass(frozen=True, slots=True)
class SpendEstimate:
    """What a call is about to cost, as far as can be known before it is sent.

    ``prompt_tokens`` is the adapter's estimate of the request, and
    ``max_output_tokens`` the request's output cap. ``cost_usd`` is the
    pricing's upper bound when the model's prices are known, else ``None``,
    and the host prices it itself.
    """

    endpoint: str
    model: str
    purpose: str
    prompt_tokens: int
    max_output_tokens: int
    cost_usd: float | None
    observability: LLMObservabilityContext | None


@dataclass(frozen=True, slots=True)
class SpendSettlement:
    """How much a call actually used.

    ``usage`` is the normalised usage (the keys :meth:`ModelPricing.cost`
    reads) and ``raw`` what the server sent. ``cost_usd`` is what the server
    reported as the cost, else the pricing's price for the usage, else
    ``None``. ``complete`` is ``False`` when the server reported no token
    counts at all: the call was made and may have been billed, but the usage
    says nothing about how much, so a host usually keeps the reservation as
    spent rather than settle it at zero.
    """

    usage: Mapping[str, Any]
    raw: Mapping[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    complete: bool = True


class SpendEnd(StrEnum):
    """How a call ended without a usage report."""

    #: The server answered with a refusal (a 4xx or 5xx status). Such a call
    #: is usually not billed.
    refused = "refused"
    #: The call was sent and never finished: the stream broke, no byte came
    #: back in time, or the caller stopped reading. Output may already have
    #: been generated, and billed.
    interrupted = "interrupted"


class SpendRefused(Exception):
    """Raised by :meth:`ISpendAdmission.reserve` to refuse a call.

    ``retry_after_seconds`` says when the budget frees up (the start of the
    next day, say), for a host that shows it; the loop does not retry a
    budget refusal whatever it says.
    """

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after_seconds = None if retry_after_seconds is None else max(0.0, float(retry_after_seconds))


@runtime_checkable
class ISpendAdmission(Protocol):
    """The host's budget, asked before each call and told how each call ended.

    Exactly one of :meth:`settle` and :meth:`release` follows every
    :meth:`reserve` that returned a reservation. Nothing follows one that
    returned ``None`` (the call needs no accounting) or raised.
    """

    async def reserve(self, estimate: SpendEstimate) -> str | None:
        """Admit the call and return a reservation id, or raise :class:`SpendRefused`."""
        ...

    async def settle(self, reservation: str, settlement: SpendSettlement) -> None:
        """The call finished and the server reported its usage (possibly incomplete)."""
        ...

    async def release(self, reservation: str, *, outcome: SpendEnd, detail: str) -> None:
        """The call ended with no usage report; ``outcome`` says how."""
        ...


def budget_error(endpoint: str, refused: SpendRefused) -> LLMProviderError:
    """The error a refused admission reaches the loop as: a ``budget`` failure that nothing retries."""
    return LLMProviderError(
        f"{endpoint}: refused by the spend budget: {refused.message}",
        classified=ProviderFailure(
            reason=ProviderFailureReason.budget,
            retryable=False,
            retry_after_seconds=refused.retry_after_seconds,
            provider_message=refused.message,
        ),
    )


__all__ = [
    "ISpendAdmission",
    "ModelPricing",
    "PricingLookup",
    "SpendEnd",
    "SpendEstimate",
    "SpendRefused",
    "SpendSettlement",
    "budget_error",
]
