"""IToolOutcomeStore Protocol — a tool call's outcome, durable the moment it exists.

The tools of one model message can run together, and their results are placed
in history in the order the model asked for them, not the order they finished.
Placement waits for the slowest call. Without a record of its own, a call that
finished while an earlier sibling was still running exists only in memory
until the whole batch is placed, so a process that dies in that window loses
finished work — and a resumed run that knows nothing of the finished call
either asks the model to repeat it or reports it as failed, and a repeat of a
call with a side effect applies that effect twice.

This port separates the two things placement used to do at once: making an
outcome durable, which happens in completion order as each call returns, and
putting it in the transcript, which still happens in source order. A resumed
run reads the store and places a recorded outcome instead of running the call
again.

What a record promises is narrow. ``started`` says the call was handed to its
tool and nothing is known about what it did; ``finished`` carries the result
block exactly as the run would have placed it. There is no promise that an
arbitrary side effect happened exactly once: a call that started and never
finished is reported to the model as having an unknown outcome, never re-run by
the core.

Reference shape: one row per invocation keyed by ``(tenant_id, run_id,
invocation_id)``, the outcome as a JSON column, written with a conditional
update so a finished row is never overwritten.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

ToolInvocationState = Literal["started", "finished"]


class ToolInvocationRecord(BaseModel):
    """One tool call of one run, as the outcome store keeps it."""

    model_config = ConfigDict(frozen=True)

    invocation_id: str
    """The store's identity for the call.

    Not the provider's tool-call id: a provider is free to reuse one in a
    later step, and an id that is only unique within a message cannot key a
    record that outlives it.
    """
    tenant_id: str
    run_id: str
    tool_call_id: str
    """The provider's id for the call, used to find its ``tool_use`` again."""
    tool_name: str
    arguments_digest: str
    """Digest of the tool name and arguments, so a record only ever answers
    the call it was written for and never a different call that happens to
    carry the same provider id."""
    step: int = 0
    """The model step the call belongs to, for whoever reads the store."""
    repeat_safe: bool = False
    """Whether repeating the call costs nothing but the repeat, as decided
    when the call was made. Carried so a reader on another process does not
    have to re-derive it from settings that may have changed."""
    state: ToolInvocationState = "started"
    outcome: dict[str, Any] | None = Field(default=None)
    """The transcript's result block for the call, as JSON, once it finished.

    The block a run places, not the tool's raw value: a large value has
    already been moved to the blob store by then and the block carries its
    reference, so a record stays small and the replayed transcript is the one
    the run would have written.
    """


class ToolOutcomeStoreError(Exception):
    """Base for outcome-store domain errors."""


@runtime_checkable
class IToolOutcomeStore(Protocol):
    """Where a run makes each tool call's outcome durable as soon as it has one."""

    async def record_started(self, record: ToolInvocationRecord) -> None:
        """Record that a call is about to be handed to its tool.

        Awaited directly before the tool is invoked, and only for calls whose
        repeat is not harmless. Returning means the record is durable. A record
        that is already ``finished`` stays finished: a late or repeated start
        must never make a known outcome look unknown again.
        """
        ...

    async def record_finished(self, record: ToolInvocationRecord) -> bool:
        """Record a call's outcome; ``False`` when one was already recorded.

        Called in completion order, the moment the call returns, before its
        result is placed in history. Creates the record when no start was
        recorded. Conditional: the first outcome recorded for an invocation is
        the one kept, and a second one — a duplicated callback, a late result
        after a cancellation — is refused with ``False`` and changes nothing.
        """
        ...

    async def list_run(self, tenant_id: str, run_id: str) -> Sequence[ToolInvocationRecord]:
        """Every record this run still holds, in no particular order."""
        ...

    async def discard(
        self,
        tenant_id: str,
        run_id: str,
        invocation_ids: Sequence[str],
    ) -> None:
        """Forget these records; an id the store does not hold is ignored.

        Called once history durably answers the calls, so a store holds at most
        the records of the step a run is in. A host expires the records of a
        run that ended without reaching another step on its own schedule.
        """
        ...


__all__ = [
    "IToolOutcomeStore",
    "ToolInvocationRecord",
    "ToolInvocationState",
    "ToolOutcomeStoreError",
]
