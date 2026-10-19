"""One recovery allowance for a run, and when a host may re-drive it.

A run recovers from a provider failure in several places, and each had a
counter of its own: the same endpoint tried again, the next rung of the provider
chain, a re-ask after a round with nothing usable in it, a request shrunk after
the window refused it. A host then re-drove the whole turn after an outage or an
overflow against a counter of its own. Stacked, the counters multiply: two
in-place retries on each of three rungs, re-driven five times, is thirty
requests for one outage, and nobody had decided thirty.

:class:`RecoveryBudget` is the one allowance every layer draws from. It lives in
the run's state (:attr:`~protocore.contracts.run_state.RunScopedState.recovery_budget`)
and travels in the snapshot. Each layer keeps its own bound as well; with
``recovery_budget_max_attempts`` and ``recovery_budget_max_seconds`` unset (the
default) the budget only counts, so nothing recovers differently until a host
sets one.

A host still owns the re-drive, which is a new run, but it reads the same
budget: the terminal ``message_stop`` carries ``recoverable_after``
(:func:`recoverable_after`), the seconds to wait before a re-drive is worth
trying, or ``None`` when it is not; the host draws one :attr:`RecoveryKind.redrive`
or :attr:`RecoveryKind.overflow_redrive` from :attr:`QueryEngine.recovery_budget`
and passes the budget on to the run it starts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from protocore.contracts.llm import PERMANENT_FAILURE_REASONS, ProviderFailureReason
from protocore.contracts.outcome import RunCause, RunOutcome


class RecoveryKind(StrEnum):
    """Which layer drew from the budget."""

    #: The same provider tried again: a rate limit, a timeout, a stalled
    #: stream, a server error.
    transient_retry = "transient_retry"
    #: The next rung of the provider chain.
    chain_advance = "chain_advance"
    #: A re-ask after a round that came back with nothing usable in it.
    empty_response = "empty_response"
    #: A request shrunk (its output cap lowered, or the history compacted)
    #: after the context window refused it.
    context_overflow = "context_overflow"
    #: A host re-driving a run that failed on an outage, as a new run.
    redrive = "redrive"
    #: A host re-driving a run that failed on an overflow, after compacting.
    overflow_redrive = "overflow_redrive"


#: Failure reasons a later attempt can get past without anything changing but
#: time: the provider was busy, slow, unreachable or briefly broken.
OUTAGE_FAILURE_REASONS: frozenset[str] = frozenset(
    {
        ProviderFailureReason.rate_limit.value,
        ProviderFailureReason.overloaded.value,
        ProviderFailureReason.server_error.value,
        ProviderFailureReason.timeout.value,
        ProviderFailureReason.connection.value,
    }
)

#: Terminal kinds of a run that failed on an outage when the error carried no
#: verdict: the loop's own classes for a rate limit, a timeout and a stall.
_OUTAGE_KINDS: frozenset[str] = frozenset({"llm_rate_limit", "llm_timeout", "llm_stream_idle"})

#: Terminal kind of a run the context window refused.
_OVERFLOW_KIND = "llm_context_window_exceeded"


def _now(now: float | None) -> float:
    return time.time() if now is None else now


@dataclass(slots=True)
class RecoveryBudget:
    """Attempts and wall clock spent recovering since the provider last answered.

    The stretch starts at the first draw and ends with :meth:`settle`, which
    the loop calls when a provider round comes back with words or tool calls:
    an answer is the evidence the outage is over, and a long run that meets an
    unrelated blip an hour later is not charged for this one.

    Wall-clock time is the epoch (``time.time()``), not a monotonic clock,
    because the budget travels in a snapshot to another process. Every method
    that reads the clock takes ``now`` for a caller (and a test) that has one.
    """

    #: How many draws a stretch may make; ``None`` counts without a bound.
    max_attempts: int | None = None
    #: How long a stretch may last from its first draw; ``None`` is no bound.
    max_seconds: float | None = None
    #: Draws made in this stretch.
    attempts: int = 0
    #: The same draws by :class:`RecoveryKind` value.
    by_kind: dict[str, int] = field(default_factory=dict)
    #: The epoch second of the stretch's first draw, or ``None`` before it.
    started_at: float | None = None

    @property
    def remaining_attempts(self) -> int | None:
        """Draws left in this stretch, or ``None`` when attempts are unbounded."""
        if self.max_attempts is None:
            return None
        return max(0, self.max_attempts - self.attempts)

    def elapsed(self, now: float | None = None) -> float:
        """Seconds since the stretch's first draw; ``0.0`` before it."""
        if self.started_at is None:
            return 0.0
        return max(0.0, _now(now) - self.started_at)

    def permits(self, now: float | None = None, *, wait_seconds: float = 0.0) -> bool:
        """Whether one more draw, made ``wait_seconds`` from ``now``, is within the bounds."""
        if self.max_attempts is not None and self.attempts >= self.max_attempts:
            return False
        if self.max_seconds is not None and self.started_at is not None:
            return self.elapsed(now) + wait_seconds <= self.max_seconds
        return True

    def draw(self, kind: RecoveryKind | str, now: float | None = None) -> bool:
        """Spend one attempt on ``kind``; ``False``, with nothing spent, when the bounds refuse."""
        moment = _now(now)
        if not self.permits(moment):
            return False
        if self.started_at is None:
            self.started_at = moment
        self.attempts += 1
        name = str(getattr(kind, "value", kind))
        self.by_kind[name] = self.by_kind.get(name, 0) + 1
        return True

    def spent_on(self, kind: RecoveryKind | str) -> int:
        """Draws made on ``kind`` in this stretch."""
        return self.by_kind.get(str(getattr(kind, "value", kind)), 0)

    def settle(self) -> None:
        """End the stretch: the provider answered, so the allowance starts over."""
        self.attempts = 0
        self.by_kind = {}
        self.started_at = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON form a snapshot carries."""
        return {
            "max_attempts": self.max_attempts,
            "max_seconds": self.max_seconds,
            "attempts": self.attempts,
            "by_kind": dict(self.by_kind),
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> RecoveryBudget | None:
        """Read :meth:`to_dict` back; ``None`` for a payload that is not a mapping.

        A field of the wrong shape falls back to its default rather than
        refusing the snapshot: an allowance read as unspent is recoverable,
        and a run refused over a counter is not.
        """
        if not isinstance(raw, dict):
            return None
        by_kind_raw = raw.get("by_kind")
        by_kind = (
            {str(key): value for key, value in by_kind_raw.items() if _is_count(value)}
            if isinstance(by_kind_raw, dict)
            else {}
        )
        return cls(
            max_attempts=_optional_count(raw.get("max_attempts")),
            max_seconds=_optional_seconds(raw.get("max_seconds")),
            attempts=_optional_count(raw.get("attempts")) or 0,
            by_kind=by_kind,
            started_at=_optional_seconds(raw.get("started_at")),
        )


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _optional_count(value: Any) -> int | None:
    return int(value) if _is_count(value) else None


def _optional_seconds(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def mint_recovery_budget(rc: Any) -> RecoveryBudget:
    """A fresh budget with the bounds ``rc`` states (read defensively)."""
    return RecoveryBudget(
        max_attempts=_optional_count(getattr(rc, "recovery_budget_max_attempts", None)),
        max_seconds=_optional_seconds(getattr(rc, "recovery_budget_max_seconds", None)),
    )


def recoverable_after(
    outcome: RunOutcome,
    cause: RunCause | None,
    budget: RecoveryBudget,
    rc: Any,
    *,
    now: float | None = None,
) -> float | None:
    """Seconds a host should wait before re-driving a failed run, or ``None`` not to.

    Only a ``failed`` run is a candidate. A context-window overflow is
    recoverable at once (``0.0``: the host compacts and re-drives) while the
    stretch has drawn fewer ``overflow_redrive`` than
    ``recovery_overflow_redrive_max_attempts``; a second overflow in a row
    after compacting is the history being too large for the window, not bad
    luck. An outage (a verdict in :data:`OUTAGE_FAILURE_REASONS` that is
    retryable, or the loop's rate-limit, timeout and stall classes when the
    error carried no verdict) is recoverable after an exponential back-off
    from ``recovery_redrive_backoff_base_seconds`` per ``redrive`` already
    drawn, capped at ``recovery_redrive_backoff_max_seconds``, and never sooner
    than the server's own ``retry_after_seconds``. A permanent refusal, a spend
    cap and a crash are not recoverable by waiting. Neither is anything once
    the budget refuses another draw, including a wait that would run past its
    wall clock.
    """
    if outcome is not RunOutcome.failed or cause is None:
        return None
    failure = cause.failure
    reason = "" if failure is None else str(getattr(failure.reason, "value", failure.reason))
    if cause.kind == _OVERFLOW_KIND or reason == ProviderFailureReason.context_overflow.value:
        limit = int(getattr(rc, "recovery_overflow_redrive_max_attempts", 0) or 0)
        if budget.spent_on(RecoveryKind.overflow_redrive) >= limit:
            return None
        return 0.0 if budget.permits(now) else None
    if failure is not None:
        outage = (
            failure.retryable
            and reason in OUTAGE_FAILURE_REASONS
            and reason not in PERMANENT_FAILURE_REASONS
        )
    else:
        outage = cause.kind in _OUTAGE_KINDS
    if not outage:
        return None
    base = float(getattr(rc, "recovery_redrive_backoff_base_seconds", 0.0) or 0.0)
    ceiling = float(getattr(rc, "recovery_redrive_backoff_max_seconds", 0.0) or 0.0)
    delay = min(base * (2 ** budget.spent_on(RecoveryKind.redrive)), ceiling)
    if failure is not None and failure.retry_after_seconds is not None:
        delay = max(delay, float(failure.retry_after_seconds))
    if not budget.permits(now, wait_seconds=delay):
        return None
    return float(delay)


__all__ = [
    "OUTAGE_FAILURE_REASONS",
    "RecoveryBudget",
    "RecoveryKind",
    "mint_recovery_budget",
    "recoverable_after",
]
