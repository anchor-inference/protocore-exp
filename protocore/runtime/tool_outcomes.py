"""The outcome store's in-memory default and the rules a run applies with it.

The store itself is in :mod:`protocore.contracts.tool_outcome`. What lives here
is the part every host shares: which calls are recorded when, how a recorded
outcome is matched back to the call it answers after a restart, and where in
history a replayed result goes.
"""
from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from protocore.contracts.tool_outcome import ToolInvocationRecord
from protocore.contracts.types import (
    Message,
    MessageRole,
    ToolResultBlock,
    ToolUseBlock,
)


class InMemoryToolOutcomeStore:
    """Dict-backed :class:`~protocore.contracts.tool_outcome.IToolOutcomeStore`.

    Durable only as long as the process, so on its own it survives nothing a
    restart would lose. It is the reference behaviour the conformance suite is
    checked against, and it serves a host that drives several engines in one
    process over the same run — a cancelled drive picked up by a fresh engine.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], ToolInvocationRecord] = {}

    @staticmethod
    def _key(record: ToolInvocationRecord) -> tuple[str, str, str]:
        return (record.tenant_id, record.run_id, record.invocation_id)

    async def record_started(self, record: ToolInvocationRecord) -> None:
        key = self._key(record)
        held = self._records.get(key)
        if held is not None and held.state == "finished":
            return
        self._records[key] = record.model_copy(update={"state": "started", "outcome": None})

    async def record_finished(self, record: ToolInvocationRecord) -> bool:
        key = self._key(record)
        held = self._records.get(key)
        if held is not None and held.state == "finished":
            return False
        self._records[key] = record.model_copy(update={"state": "finished"})
        return True

    async def list_run(self, tenant_id: str, run_id: str) -> Sequence[ToolInvocationRecord]:
        return [
            record
            for (tenant, run, _), record in self._records.items()
            if tenant == tenant_id and run == run_id
        ]

    async def discard(
        self,
        tenant_id: str,
        run_id: str,
        invocation_ids: Sequence[str],
    ) -> None:
        for invocation_id in invocation_ids:
            self._records.pop((tenant_id, run_id, invocation_id), None)


def new_invocation_id() -> str:
    """An identity for a call that has no durable intent record to borrow one from."""
    return f"inv_{uuid4().hex[:16]}"


def unanswered_tool_uses(history: Sequence[Message]) -> dict[str, ToolUseBlock]:
    """Every ``tool_use`` in history that no ``tool_result`` answers yet."""
    answered: set[str] = set()
    uses: dict[str, ToolUseBlock] = {}
    for message in history:
        for block in message.content_blocks:
            if isinstance(block, ToolResultBlock):
                answered.add(block.tool_call_id)
            elif isinstance(block, ToolUseBlock):
                uses[block.tool_call_id] = block
    return {call_id: block for call_id, block in uses.items() if call_id not in answered}


def place_result_in_source_order(history: list[Message], block: ToolResultBlock) -> bool:
    """Insert a replayed result after the call it answers, behind its earlier siblings.

    A replayed batch arrives one result at a time and in whatever order the
    store lists it, and a result simply inserted after the assistant message
    would put the batch back in reverse. The model reads results in the order
    it asked for them, and a provider pairs them by position in some dialects,
    so each one goes after every result already present for a call that came
    before it in the same message.
    """
    for index, message in enumerate(history):
        if message.role is not MessageRole.assistant:
            continue
        order = [
            item.tool_call_id
            for item in message.content_blocks
            if isinstance(item, ToolUseBlock)
        ]
        if block.tool_call_id not in order:
            continue
        position = order.index(block.tool_call_id)
        earlier = set(order[:position])
        at = index + 1
        while at < len(history) and history[at].role is MessageRole.tool:
            answers = {
                item.tool_call_id
                for item in history[at].content_blocks
                if isinstance(item, ToolResultBlock)
            }
            if not answers or not answers <= earlier:
                break
            at += 1
        history.insert(at, Message(role=MessageRole.tool, content_blocks=[block]))
        return True
    return False


__all__ = [
    "InMemoryToolOutcomeStore",
    "new_invocation_id",
    "place_result_in_source_order",
    "unanswered_tool_uses",
]
