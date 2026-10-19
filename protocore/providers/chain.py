"""A concrete :class:`~protocore.contracts.llm.IProviderChain` over an ordered list.

The contract states what the loop needs from a chain — the provider serving
now, its model name, a one-way :meth:`advance` and the record of what was ruled
out — and leaves the policy to the host. Two pieces of policy are general
enough to ship here, because every chain that lacked them failed the same way:

* **A fallback must be able to hold the conversation.** A rung whose context
  window is smaller than the prompt the run is carrying plus the reply it asks
  for is skipped, with that reason recorded. Demoting onto it trades a stalled
  stream for a context-window refusal, and the run ends on the fallback instead
  of retrying where it was.
* **A fallback known to be resting is skipped.** A provider that refused for a
  spent quota a minute ago will refuse again; walking into it costs a call and
  a retry before the run moves on anyway. The primary is never asked, since the
  operator chose it and a stale observation must not veto it.

Rungs may be built lazily: the contract makes :meth:`advance` asynchronous so a
provider (and its connection pool) is created only when a run reaches it.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from protocore.contracts.llm import ILLMProvider, IProviderChain
from protocore.logging_utils import get_logger

_logger = get_logger(__name__)

ProviderFactory = Callable[[], Awaitable[ILLMProvider]]
"""Builds a rung's provider when the run first reaches it."""

RestingCheck = Callable[["ChainRung"], str | None]
"""Why a fallback rung should not be tried right now, or ``None`` when it may be."""


@dataclass(frozen=True, slots=True)
class ChainRung:
    """One provider a run may be served by, and what the chain needs to judge it.

    ``name`` labels the rung in :meth:`ListProviderChain.attempted`; ``model``
    is the model identity the engine's config takes when the rung serves.
    Exactly one of ``provider`` (already built) and ``factory`` (built on
    arrival) is given. ``context_window`` and ``max_output_tokens`` are the
    rung's own limits; either left ``None`` means "unknown", and an unknown
    limit never rules a rung out — unknown is not the same as too small.
    """

    name: str
    model: str
    provider: ILLMProvider | None = None
    factory: ProviderFactory | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if (self.provider is None) == (self.factory is None):
            raise ValueError(
                f"chain rung {self.name!r} needs exactly one of provider and factory"
            )


class ListProviderChain(IProviderChain):
    """A one-way cursor over :class:`ChainRung` entries, in preference order.

    ``prompt_tokens`` reports the size of the prompt the run is carrying (a
    host binds the engine's measurement, e.g. ``engine.prompt_tokens_now``, with
    :meth:`bind_prompt_size` once the engine exists); ``0`` means unknown and
    skips no rung. ``resting`` names fallbacks to pass over for now.

    One caveat on resume: the engine re-seats a restored run by advancing the
    recorded number of times and then checks the rung's model name. A rung that
    was skipped when the run demoted but fits now changes where those advances
    land, and the engine refuses the snapshot rather than serve it from the
    wrong rung.
    """

    def __init__(
        self,
        rungs: Sequence[ChainRung],
        *,
        prompt_tokens: Callable[[], int] | None = None,
        resting: RestingCheck | None = None,
    ) -> None:
        if not rungs:
            raise ValueError("a provider chain needs at least one rung")
        first = rungs[0]
        if first.provider is None:
            raise ValueError(
                "the first rung must carry a built provider: current() is "
                "synchronous and the run asks for it before any advance"
            )
        self._rungs = tuple(rungs)
        self._providers: dict[int, ILLMProvider] = {0: first.provider}
        self._prompt_tokens = prompt_tokens or (lambda: 0)
        self._resting = resting
        self._index = 0
        self._ruled_out: list[tuple[str, str]] = []
        # The check-and-advance is one transition. Building a rung awaits, and
        # two failures arriving together must not both decide from position
        # zero and demote past each other.
        self._advance_lock = asyncio.Lock()

    def bind_prompt_size(self, prompt_tokens: Callable[[], int]) -> None:
        """Where the current prompt size is read from (the engine, once it exists)."""
        self._prompt_tokens = prompt_tokens

    @property
    def position(self) -> int:
        """The index of the rung serving now."""
        return self._index

    @property
    def rungs(self) -> tuple[ChainRung, ...]:
        """Every rung, in preference order."""
        return self._rungs

    def current(self) -> ILLMProvider:
        return self._providers[self._index]

    def current_model_name(self) -> str:
        return self._rungs[self._index].model

    def attempted(self) -> Sequence[tuple[str, str]]:
        return tuple(self._ruled_out)

    async def advance(self, *, reason: str) -> bool:
        async with self._advance_lock:
            skipped: list[tuple[str, str]] = []
            for candidate in range(self._index + 1, len(self._rungs)):
                rung = self._rungs[candidate]
                unusable = self._unusable(rung)
                if unusable is None:
                    unusable = await self._materialise(candidate)
                if unusable is not None:
                    skipped.append((rung.name, unusable))
                    continue
                self._ruled_out.append((self._rungs[self._index].name, reason))
                self._ruled_out.extend(skipped)
                self._index = candidate
                return True
            # Exhausted: the current rung stays and nothing is recorded. The
            # loop's next move (a retry in place, a wind-down) is not a demotion.
            return False

    def _unusable(self, rung: ChainRung) -> str | None:
        """Why ``rung`` cannot serve the run right now, or ``None`` when it can."""
        if self._resting is not None:
            resting = self._resting(rung)
            if resting:
                return resting
        window = rung.context_window
        if not window:
            return None
        prompt = int(self._prompt_tokens() or 0)
        reply = int(rung.max_output_tokens or 0)
        if prompt and prompt + reply > window:
            return (
                f"context window {window} cannot hold the prompt ({prompt} tokens) "
                f"and the reply ({reply})"
            )
        return None

    async def _materialise(self, index: int) -> str | None:
        """Build the rung's provider; why it could not be built, if it could not.

        Called once per rung at most: the cursor only moves forward, and a rung
        it passed over is never a candidate again.
        """
        rung = self._rungs[index]
        if rung.provider is not None:
            self._providers[index] = rung.provider
            return None
        factory = rung.factory
        if factory is None:  # pragma: no cover - ChainRung refuses a rung with neither
            return "has neither a provider nor a factory"
        try:
            self._providers[index] = await factory()
        except Exception as exc:
            # One rung that cannot be built (a missing key, a bad URL) is a fact
            # about that rung; the walk goes on to the next instead of failing
            # the run over a fallback it might not even have needed.
            _logger.warning("provider chain rung %s could not be built: %s", rung.name, exc)
            return f"could not be built: {exc}"
        return None


__all__ = ["ChainRung", "ListProviderChain", "ProviderFactory", "RestingCheck"]
