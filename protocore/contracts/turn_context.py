"""What a host appends after the cache-stable prefix of each request.

Some of what a model should know changes on every run or every request: the
clock, the state of a board, which background jobs are still running, a mode
the user switched a minute ago. Written into the system prompt, it changes the
first bytes of every request, and the provider re-reads the whole replayed
history behind them: measured at the start of runs, a host that did so was
served almost nothing from the cache, while inside a run the hit was above
eighty percent. Written into a stored message, it stays in the history for
good, repeated once per turn, and a compaction summary quotes it as if
somebody had said it.

The turn context is the third place. A host supplies it through
:class:`ITurnContextProvider`; the core asks for it before every request of
the run's action stream and appends it as one user message after everything
else, so the system prompt, the tool catalogue and the history in front of it
stay byte-identical from request to request. It never enters
``engine.history``, a snapshot or a compaction, it is never marked as a cache
breakpoint, and the next request carries the next one in its place. The
request says how many trailing messages are volatile in
``extra["volatile_suffix_messages"]``, so an adapter that places its own cache
markers, and a reader of request manifests, can leave them out of the prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

#: ``LLMRequest.extra`` key holding the number of trailing request messages
#: that are volatile — present only when there are any.
VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY: Final[str] = "volatile_suffix_messages"

#: ``Message.metadata`` key on the request message that carries a turn
#: context. The message exists only in the request, never in the history.
TURN_CONTEXT_METADATA_KEY: Final[str] = "protocore.turn_context"


@dataclass(frozen=True, slots=True)
class TurnContextRequest:
    """What the core tells a provider about the request it is building."""

    tenant_id: str
    run_id: str
    session_id: str
    #: The turn this request belongs to (``QueryEngine.turn_id()``). A host
    #: that wants one context per turn rather than per request keys on it.
    turn_id: str
    #: The model the request is addressed to: the live override when one is in
    #: force.
    model: str


@runtime_checkable
class ITurnContextProvider(Protocol):
    """Supplies the volatile block appended to a request.

    Called once per request of the action stream, on the hot path between the
    request being assembled and the provider being asked, so it returns
    promptly: a host that gathers its notes from slow sources caches them per
    turn. ``None`` or an empty string appends nothing. A provider that raises
    is logged and the request goes without a turn context rather than failing.
    """

    async def turn_context(self, request: TurnContextRequest) -> str | None:
        """The text to append after the cache-stable prefix, or ``None``."""
        ...


__all__ = [
    "TURN_CONTEXT_METADATA_KEY",
    "VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY",
    "ITurnContextProvider",
    "TurnContextRequest",
]
