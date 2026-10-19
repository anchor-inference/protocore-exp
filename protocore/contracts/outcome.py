"""How a run ended, and why, as the terminal event states it.

``status`` says whether the loop finished cleanly, ``stop_reason`` says what
stopped the last turn, and ``has_final_answer`` says whether the user was given
something to read. None of the three says whether the run did its job: a run the
wind-down closed because the provider refused it ends ``completed`` with a
polite closing paragraph, and from outside it looks like a run that answered.
Every consumer then re-derived the verdict from the wind-down cause, and told
its own spend cap from a vendor's refusal by matching words in a detail string.

The terminal ``message_stop`` therefore carries two more fields: ``outcome``, one
:class:`RunOutcome`, and ``cause``, a :class:`RunCause` in its JSON form (or
``None`` for a run that simply completed). Read them back with
``RunOutcome(payload["outcome"])`` and ``RunCause.model_validate(payload["cause"])``.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

from protocore.contracts.llm import ProviderFailure

#: The terminal ``message_stop`` payload key holding the :class:`RunOutcome`.
RUN_OUTCOME_FIELD: Final[str] = "outcome"

#: The terminal ``message_stop`` payload key holding the :class:`RunCause`.
RUN_CAUSE_FIELD: Final[str] = "cause"

#: The terminal ``message_stop`` payload key listing the ids of the queued
#: steer and follow-up items the run still held and never placed, steers first.
#: Input that arrived while a run was settling, or after it failed, used to sit
#: unread until a host went looking for it; the list says there is some, so the
#: host can open the next run with it. It names what the engine holds: an item
#: a host's store received after the engine's last read of it is the store's
#: to add.
LEFTOVER_QUEUED_FIELD: Final[str] = "leftover_queued"

#: The terminal ``message_stop`` payload key holding the seconds a host should
#: wait before re-driving the run as a new run, or ``None`` when a re-drive is
#: not worth trying: the run did not fail, failed on something waiting will not
#: change, or spent its recovery budget. See
#: :func:`protocore.contracts.recovery.recoverable_after`.
RECOVERABLE_AFTER_FIELD: Final[str] = "recoverable_after"


class RunOutcome(StrEnum):
    """Whether the run did what it was asked, in one word."""

    #: The run finished on its own terms.
    completed = "completed"
    #: The run did not get its job done: the provider failed it, the model made
    #: no progress, or the process crashed. A run wound down for either of the
    #: first two reasons is ``failed`` even when its closing turn wrote prose,
    #: because that prose is a report on why the work stopped, not the work.
    failed = "failed"
    #: The run was cancelled.
    stopped = "stopped"
    #: The run reached a bound it was given (turns, tool calls, output tokens,
    #: wall clock) and closed on the evidence it had.
    wound_down = "wound_down"


class RunCause(BaseModel):
    """Why a run did not simply complete.

    ``kind`` is the core's own name for what ended it: a wind-down cause from
    :mod:`protocore.runtime.soft_stop` (``provider_error``,
    ``model_no_progress``, ``max_turns``, ...) or the terminal kind of a failed
    run (``llm_provider_error``, ``llm_rate_limit``, ``internal_error``, ...).
    ``failure`` is the provider's verdict when a provider failure was the cause;
    a host tells its own spend cap from a vendor's refusal by
    ``failure.reason == "budget"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    #: What the cause said in its own words, typically the provider's error text.
    detail: str = ""
    failure: ProviderFailure | None = None


__all__ = [
    "LEFTOVER_QUEUED_FIELD",
    "RECOVERABLE_AFTER_FIELD",
    "RUN_CAUSE_FIELD",
    "RUN_OUTCOME_FIELD",
    "RunCause",
    "RunOutcome",
]
