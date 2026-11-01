"""Tool call ids that are unique within a conversation.

Everything that pairs a result with its call does it by id: the outbound
pairing repair, the dispatcher, approvals, cancellation, the stored transcript
and a host's view of it. Providers do not all promise an id the conversation
has not seen. Models that write their own ids count them per response or from
the history they were shown — Kimi names its calls ``functions.Write:0`` and
starts again from ``:0`` once the context is compacted — so a later call can
arrive carrying an earlier call's id. The pairing repair then kept the first
call with that id and dropped the later one with its result: the call's effect
stayed on disk and the model never saw it.

So every call gets an id of its own the moment its response arrives
(:class:`ToolCallIdAssigner`), and a transcript stored before that was true is
read back with its repeats told apart by position
(:func:`rekey_repeated_tool_call_ids`).
"""
from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Set

from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
from protocore.contracts.types import ContentBlock, Message, ToolResultBlock, ToolUseBlock

_logger = logging.getLogger(__name__)

_TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)$", re.DOTALL)

_CALL_DELTAS = frozenset(
    {ProviderDeltaKind.tool_use_start, ProviderDeltaKind.tool_use_input, ProviderDeltaKind.tool_use_stop}
)


def fresh_tool_call_id(provider_id: str, taken: Set[str]) -> str:
    """An id not in ``taken``, written in the shape of ``provider_id``.

    An id that ends in a number gets the next number above every id of the
    same stem: ``functions.Write:0`` becomes ``functions.Write:3`` when
    ``:2`` is the highest taken. Kimi's ids have that shape and its history is
    rendered back to it verbatim, so a renamed id still looks like one of its
    own; and a number above the highest stays clear of the low numbers a model
    that restarted its count is about to write. Any other id gets ``_2``, ``_3``…,
    which adds no character an id validator (Anthropic's
    ``^[a-zA-Z0-9_-]+$``) could refuse that the provider's own id did not
    already carry.
    """
    match = _TRAILING_NUMBER.match(provider_id)
    if match is None:
        suffix = 2
        while f"{provider_id}_{suffix}" in taken:
            suffix += 1
        return f"{provider_id}_{suffix}"
    stem = match.group(1)
    number = int(match.group(2))
    for other in taken:
        same_stem = _TRAILING_NUMBER.match(other)
        if same_stem is not None and same_stem.group(1) == stem:
            number = max(number, int(same_stem.group(2)))
    number += 1
    while f"{stem}{number}" in taken:
        number += 1
    return f"{stem}{number}"


class ToolCallIdAssigner:
    """Renames the calls of one response so none repeats an id already in use.

    ``taken`` is every id the conversation has used, and is extended in place
    with each id issued here, so a caller that keeps the set sees every id
    this response handed out. A provider id that is not taken is kept as it
    is; only a repeat is renamed, so a provider that echoes its own ids back
    still finds them wherever they were unique.

    Within one response the provider id is all a delta carries, so a repeat
    that is still open is told apart by the order adapters promise: a call's
    arguments arrive after its start and before the next call starts, and the
    stops come in the order the calls started. An input therefore belongs to
    the newest open call with that provider id, and a stop to the oldest.
    """

    def __init__(self, taken: set[str]) -> None:
        self._taken = taken
        self._open: dict[str, deque[str]] = {}

    def assign(self, delta: ProviderDelta) -> ProviderDelta:
        """``delta`` carrying the conversation's id for its call."""
        provider_id = delta.tool_call_id
        if not provider_id or delta.kind not in _CALL_DELTAS:
            return delta
        open_calls = self._open.setdefault(provider_id, deque())
        if delta.kind is ProviderDeltaKind.tool_use_start or not open_calls:
            # A stop or an input with no start before it breaks the adapter
            # contract; it is still a call, and it still needs an id of its own.
            open_calls.append(self._issue(provider_id))
        if delta.kind is ProviderDeltaKind.tool_use_stop:
            call_id = open_calls.popleft()
        else:
            call_id = open_calls[-1]
        if call_id == provider_id:
            return delta
        return delta.model_copy(update={"tool_call_id": call_id})

    def _issue(self, provider_id: str) -> str:
        if provider_id not in self._taken:
            self._taken.add(provider_id)
            return provider_id
        call_id = fresh_tool_call_id(provider_id, self._taken)
        self._taken.add(call_id)
        _logger.warning("the provider repeated tool call id %r; the call is kept as %r", provider_id, call_id)
        return call_id


def rekey_repeated_tool_call_ids(messages: list[Message]) -> list[Message]:
    """``messages`` with every repeated call id replaced by one of its own.

    A transcript stored before ids were made unique at arrival can hold two
    calls with one id, each followed by its own result. The result belongs to
    the oldest call with its id that has none yet, which is how the results
    were appended. A result left over once every call with its id has one is a
    second result for the same call, and keeps its id for the pairing repair
    to drop. Unchanged messages are returned by reference; nothing is mutated.
    The renaming depends only on ``messages``, so the same history gives the
    same request and a prompt cache over it still hits.
    """
    taken = tool_call_ids_in(messages)
    seen_calls: set[str] = set()
    awaiting_result: dict[str, deque[str]] = {}
    rekeyed: list[Message] = []
    for message in messages:
        blocks: list[ContentBlock] = []
        changed = False
        for block in message.content_blocks:
            if isinstance(block, ToolUseBlock):
                stored_id = block.tool_call_id
                if stored_id in seen_calls:
                    call_id = fresh_tool_call_id(stored_id, taken)
                    taken.add(call_id)
                    block = block.model_copy(update={"tool_call_id": call_id})
                    changed = True
                seen_calls.add(stored_id)
                awaiting_result.setdefault(stored_id, deque()).append(block.tool_call_id)
            elif isinstance(block, ToolResultBlock):
                waiting = awaiting_result.get(block.tool_call_id)
                if waiting:
                    call_id = waiting.popleft()
                    if call_id != block.tool_call_id:
                        block = block.model_copy(update={"tool_call_id": call_id})
                        changed = True
            blocks.append(block)
        rekeyed.append(message.model_copy(update={"content_blocks": blocks}) if changed else message)
    return rekeyed


def tool_call_ids_in(messages: list[Message]) -> set[str]:
    """Every call id ``messages`` carry, on a call or on a result."""
    return {
        block.tool_call_id
        for message in messages
        for block in message.content_blocks
        if isinstance(block, (ToolUseBlock, ToolResultBlock))
    }
