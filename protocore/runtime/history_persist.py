"""What a session store is told when the working history changes.

A ReAct round adds messages to the end of a sequence it does not otherwise
touch. Telling the store "these two are new" is the whole of what it needs to
hear; handing it the other eight hundred again is a rewrite of everything the
session has ever said, once per round, and the cost of it grows with the
conversation rather than with what the conversation just did. Measured against
a store that writes a row per message, a 800-message history cost 63 ms and
1.2 MB of writes per round to record one new message.

The engine therefore remembers the prefix it last handed over and compares the
current history against it. :class:`~protocore.contracts.types.Message` is
frozen, so a message is replaced rather than edited whenever it changes: an
append leaves every earlier object exactly where it was, and a compaction, a
checkpoint or an eviction builds new ones. Identity comparison of the prefix
reports the difference without anything having to declare it, and the prefix is
held weakly, so the comparison costs a pointer per message and nothing that
left the history is kept alive by having once been persisted.

An engine that has never handed anything over remembers nothing rather than
remembering an empty history, and the two are not the same claim: a run picked
up on another process holds a history the store may already have most of, and
appending onto rows nobody can vouch for is how a transcript acquires a
duplicate. So the first hand-over of a run is a rewrite, and every one after it
is an append until the sequence itself changes.

Two shapes of store are supported, and a store says which it is by what it
attaches to the engine:

* ``persist_session_history(engine)`` — the full write. The only method a
  store must have, and what every store had before the delta existed.
* ``persist_history_delta(engine, delta)`` — the incremental write. A store
  that has it is handed a :class:`HistoryDelta` and writes only
  ``delta.appended``, unless ``delta.rewritten`` says the sequence itself
  changed and ``delta.history`` replaces what the store holds.

Either way the store is handed :class:`~protocore.contracts.types.Message`
objects, and some of them are not the conversation's: compaction writes its
ledger, summaries and floor digest into history as user-role messages, and
their :attr:`~protocore.contracts.types.Message.origin` is
:attr:`~protocore.contracts.types.MessageOrigin.compaction`. A store that keeps
a transcript for people to read, or seeds a later run from it, must not record
them as the user's turn.

A store with only the first is called exactly as it was before. A store that
subclasses :class:`HistoryPersister` and overrides nothing inherits the same
behaviour through the default below.

**Returning is the store's promise that the write landed.** The engine advances
its marker only after the call returns, so a store that raises is offered the
same messages again on the next round — which is the self-healing the old
write-it-all-every-time contract had for free, and the one thing the delta path
would otherwise throw away. A store that defers its write (a task, a queue, a
batch) is promising on that write's behalf, and owes the engine a
:meth:`~protocore.runtime.query_engine.QueryEngine.forget_persisted_history`
call if the deferred write is dropped or fails; without it, the messages it
dropped are never offered again.

:class:`OrderedHistoryPersister` is that deferred store, written once: it
takes an ``async`` write, runs the writes of one engine one at a time in the
order they were handed over, keeps the promise above on every failure, and
offers :meth:`OrderedHistoryPersister.flush` as the point at which the store is
known to hold everything it was given.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from protocore.contracts.types import Message

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.runtime.query_engine import QueryEngine

__all__ = [
    "HistoryDelta",
    "HistoryPersistError",
    "HistoryPersister",
    "HistoryWrite",
    "OrderedHistoryPersister",
    "persist_history",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoryDelta:
    """What changed in the working history since the store last saw it."""

    history: tuple[Message, ...]
    """The whole working history as it stands. Authoritative when
    :attr:`rewritten`; otherwise it is context, and writing it again would be
    the cost this delta exists to avoid."""

    appended: tuple[Message, ...]
    """The messages added since the last hand-over. Empty on a rewrite, because
    a rewrite is not an addition to anything."""

    persisted_through: int
    """How many messages of :attr:`history` the store was last told about. The
    store may use it as a positional check on its own tail; ``0`` on a rewrite,
    where nothing that came before is still true."""

    rewritten: bool
    """The sequence changed rather than grew — a compaction replaced a span
    with its summary, a checkpoint rebuilt it, a host evicted rows the engine
    had been told were durable. The store replaces what it holds with
    :attr:`history`."""

    session_state_changed: bool = False
    """Something durable about the session changed that the message sequence
    does not carry — a checkpoint was taken. A delta may carry this with
    nothing appended and no rewrite, which is a store being told "the run
    changed, write down whatever else you keep about it"."""


class HistoryPersister:
    """The store's side of the contract, for a host that would rather subclass.

    Nothing requires it: the engine looks for two attributes and does not care
    where they came from, so attaching two plain functions is as valid as
    subclassing this. What the class buys is the fallback — a store that can
    only write the history whole overrides :meth:`persist_session_history`
    alone and is driven exactly as it was before the delta existed, without
    having to know the delta exists.
    """

    def persist_session_history(self, engine: QueryEngine) -> None:
        """Write the engine's working history out in full."""
        raise NotImplementedError

    def persist_history_delta(self, engine: QueryEngine, delta: HistoryDelta) -> None:
        """Write what changed. Falls back to the full write."""
        self.persist_session_history(engine)


#: The store's own write: given what changed, make it durable, and raise if it
#: did not land.
HistoryWrite = Callable[[HistoryDelta], Awaitable[None]]


class HistoryPersistError(RuntimeError):
    """The store does not hold everything it was handed.

    Raised by :meth:`OrderedHistoryPersister.flush` when the last write that
    could have repaired the store failed too. The engine has already been told
    (:meth:`~protocore.runtime.query_engine.QueryEngine.forget_persisted_history`),
    so its next hand-over is a whole rewrite; what this error says is that,
    until then, the stored history is short. The store's own exception is the
    ``__cause__``.
    """


class OrderedHistoryPersister(HistoryPersister):
    """A store's write deferred off the loop, in order, with a flush.

    The hand-over returns as soon as the write is queued, so a round does not
    wait for the database. What the class adds over a bare task is the part
    that is easy to get wrong:

    * **Order.** Writes run one at a time, each after the one handed over
      before it, whatever its outcome. Without that, a failed write could have
      its repair overtaken by an append queued behind it, and the append would
      land on rows that are not there.
    * **The promise on failure.** A write that raises (or is cancelled) makes
      the engine forget what it was told at once, from inside the failing
      write rather than from a done-callback a loop iteration later, by which
      time the next round could already have been handed over as an append.
    * **Repair.** While the store is behind, every queued write — including
      appends handed over before the failure was known — is written whole
      instead, from the engine's live history when that is at least as long as
      the snapshot it was handed. If nothing else is queued, one repair write
      is, so a failure in the last round of a run still gets written. A repair
      that fails is not repeated: a store refusing every write is retried by
      the engine's next hand-over, not in a loop here.

    **Durability.** Returning from a hand-over means *queued*, not *landed*.
    :meth:`flush` is the barrier: once it returns, every history handed over
    before it was called has landed. Call it when the run reaches a terminal
    state and before the session moves to another process; it raises
    :class:`HistoryPersistError` if the store is still behind. Writes are
    process-local tasks: a crash before the flush loses what was queued, and
    the next run's first hand-over is a rewrite (an engine that never handed
    anything over trusts nothing), which is what makes that loss recoverable.

    The write is the store's only part. It is ``async``, is given a
    :class:`HistoryDelta`, writes ``delta.history`` whole when
    ``delta.rewritten`` and ``delta.appended`` otherwise, and raises when it
    did not land. Hand-overs need a running event loop. One instance serves
    one engine at a time; :meth:`attach` installs it.
    """

    def __init__(self, write: HistoryWrite, *, name: str = "history-persist") -> None:
        self._write = write
        self._name = name
        self._tail: asyncio.Task[None] | None = None
        self._repair: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._behind = False
        self._error: BaseException | None = None

    def attach(self, engine: QueryEngine) -> None:
        """Make ``engine`` hand its history to this persister."""
        engine.persist_session_history = self.persist_session_history  # type: ignore[attr-defined]
        engine.persist_history_delta = self.persist_history_delta  # type: ignore[attr-defined]

    @property
    def pending(self) -> int:
        """Writes queued or running."""
        return len(self._tasks)

    @property
    def behind(self) -> bool:
        """Whether a write was lost and nothing has written the history whole since."""
        return self._behind

    def persist_session_history(self, engine: QueryEngine) -> None:
        """Queue a whole write of the engine's history."""
        self._enqueue(engine, _whole(tuple(engine.history), engine.session_state_changed))

    def persist_history_delta(self, engine: QueryEngine, delta: HistoryDelta) -> None:
        """Queue ``delta`` behind every write handed over before it."""
        self._enqueue(engine, delta)

    async def flush(self) -> None:
        """Wait until every queued write has finished; raise if the store is behind."""
        while self._tail is not None and not self._tail.done():
            # A failing write queues its repair before it finishes, so a tail
            # that is done is the last write there is.
            await asyncio.wait([self._tail])
        if self._behind:
            raise HistoryPersistError(
                "the session store is missing history it was handed; "
                "the engine's next hand-over rewrites it whole"
            ) from self._error

    def cancel(self) -> None:
        """Cancel every queued write, for a shutdown that cannot wait for them.

        The store is then behind, and the engine is told so by each cancelled
        write; :meth:`flush` afterwards raises :class:`HistoryPersistError`.
        """
        for task in list(self._tasks):
            task.cancel()

    def _enqueue(self, engine: QueryEngine, delta: HistoryDelta, *, repair: bool = False) -> asyncio.Task[None]:
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._run(engine, delta, self._tail, repair), name=self._name)
        self._tail = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(
        self,
        engine: QueryEngine,
        delta: HistoryDelta,
        prior: asyncio.Task[None] | None,
        repair: bool,
    ) -> None:
        try:
            if prior is not None and not prior.done():
                # Its outcome is its own; this waits only for its turn.
                await asyncio.wait([prior])
            if repair and not self._behind:
                return  # a hand-over queued behind the failure already wrote it whole
            if self._behind:
                live = tuple(engine.history)
                history = live if len(live) >= len(delta.history) else delta.history
                delta = _whole(history, delta.session_state_changed)
            await self._write(delta)
        except asyncio.CancelledError as exc:
            self._lost(engine, exc, schedule_repair=False)
            raise
        except Exception as exc:
            logger.warning("session history write failed; the next write rewrites it whole", exc_info=True)
            self._lost(engine, exc, schedule_repair=not repair)
            return
        if delta.rewritten:
            self._behind = False
            self._error = None

    def _lost(self, engine: QueryEngine, exc: BaseException, *, schedule_repair: bool) -> None:
        self._behind = True
        self._error = exc
        engine.forget_persisted_history()
        if not schedule_repair or (self._repair is not None and not self._repair.done()):
            return
        self._repair = self._enqueue(engine, _whole(tuple(engine.history), False), repair=True)


def _whole(history: tuple[Message, ...], state_changed: bool) -> HistoryDelta:
    """A delta that replaces what the store holds with ``history``."""
    return HistoryDelta(
        history=history,
        appended=(),
        persisted_through=0,
        rewritten=True,
        session_state_changed=state_changed,
    )


def persist_history(engine: QueryEngine) -> None:
    """Hand the store what changed in the working history, if anything did.

    The one call site for all three places the loop persists — the turn-start
    reset, the background-wake append and the manual checkpoint — so the
    prefix marker advances wherever the history is handed over and cannot be
    advanced by one path and not another.

    A call that would say nothing is not made at all. The engine reaches this
    function once per round whether or not the round changed anything, and a
    store woken to be told that its copy is already correct pays the full
    write to find that out.

    The marker advances after the store's call returns, never before: see the
    module docstring on what returning promises.
    """
    incremental = getattr(engine, "persist_history_delta", None)
    persister = getattr(engine, "persist_session_history", None)
    if not callable(incremental) and not callable(persister):
        return
    persisted = engine.persisted_history_marker
    history = tuple(engine.history)
    known: tuple[object, ...] = () if persisted is None else persisted
    rewritten = (
        persisted is None
        or len(history) < len(known)
        or any(
            before() is not now
            for before, now in zip(persisted or (), history, strict=False)
        )
    )
    appended = () if rewritten else history[len(known) :]
    state_changed = engine.session_state_changed
    if not rewritten and not appended and not state_changed:
        return
    if persisted is None and not history and not state_changed:
        # Nothing has been said, and nothing was ever stored to correct.
        return
    epoch = engine.persisted_history_epoch
    if callable(incremental):
        incremental(
            engine,
            HistoryDelta(
                history=history,
                appended=appended,
                persisted_through=0 if rewritten else len(known),
                rewritten=rewritten,
                session_state_changed=state_changed,
            ),
        )
    elif callable(persister):
        persister(engine)
    if engine.persisted_history_epoch != epoch:
        # The store disowned the write from inside it. Believe the store.
        return
    engine.note_history_persisted(history)
    engine.note_session_state_persisted()
