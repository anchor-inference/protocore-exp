"""A host's choice of reasoning effort for one turn, made from the message that opens it.

A greeting does not need the effort a refactoring does, and a session that
runs every message at one effort either spends a deep model's thinking on
"thanks" or answers a hard task shallowly. A host that sorts each message into
a level — a small model, a heuristic of its words — supplies an
:class:`IReasoningEffortClassifier`. The core asks it once per turn, after the
``user_prompt_submit`` hook has admitted the turn and before the first request,
and runs that turn at the effort it names.

What it chooses sits below an effort the user set mid-run
(``QueryEngine.apply_live_controls``), which is a standing choice and wins; the
classifier is not asked at all while one is in force. It sits above the
configured ``QueryEngineConfig.reasoning_effort``, which is what it is handed as
the base to move from. The temporary effort the loop lowers to while recovering
from a reasoning cut-off still wins over both for the requests it covers.

The choice lasts for the turn: a re-armed engine asks again for the next one,
and a snapshot carries it, so a turn resumed in another process runs at the
effort it began at instead of asking a second time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from protocore.contracts.types import Message


@dataclass(frozen=True, slots=True)
class EffortRequest:
    """What the core tells a classifier about the turn it is about to run."""

    tenant_id: str
    run_id: str
    session_id: str
    turn_id: str
    #: The model the turn will ask: the live override when one is in force.
    model: str
    #: The newest user message, the one the turn answers; ``None`` for a turn
    #: that continues without one.
    message: Message | None
    #: The effort the turn would run at without a choice.
    base_effort: str


@runtime_checkable
class IReasoningEffortClassifier(Protocol):
    """Chooses the reasoning effort of one turn.

    Returns one of the core's efforts (``minimal``, ``low``, ``medium``,
    ``high``, ``xhigh``) or ``None`` to keep the base. Anything else is logged
    and ignored, and so is an exception. The user's message waits behind the
    call, so a classifier bounds its own latency: a sort that takes longer than
    the turn it shapes costs more than a wrong guess.
    """

    async def choose_effort(self, request: EffortRequest) -> str | None:
        """The effort for this turn, or ``None`` for the base."""
        ...


__all__ = ["EffortRequest", "IReasoningEffortClassifier"]
