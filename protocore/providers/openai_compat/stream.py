"""Turning a chat-completions SSE stream into the core's :class:`ProviderDelta` envelopes.

Split from the transport so the parsing is tested on recorded streams: the
lines a server sent go in, the deltas the loop would see come out.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
from protocore.json_utils import OutputParserException, PartialJSONParser
from protocore.logging_utils import get_logger
from protocore.providers.text_tool_calls import TextToolCallGrammar, TextToolCallGuard

_logger = get_logger(__name__)

FinishReason = Literal["stop", "tool_use", "length", "content_filter"]

_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "tool_use": "tool_use",
    "length": "length",
    "content_filter": "content_filter",
}


def map_finish(reason: str | None) -> FinishReason:
    """The core's finish reason for the server's; an unknown one reads as a plain stop."""
    return _FINISH.get(reason or "stop", "stop")


async def sse_events(lines: AsyncIterator[str]) -> AsyncIterator[str | None]:
    """Yield the ``data`` payload of each SSE event, and ``None`` for each comment line.

    A comment carries no data, but it is the only sign of life some servers
    and proxies give while a model thinks without writing anything, so the
    caller hears about every one. An event that ends without its blank line
    (the server closed the stream right after it) is still delivered.
    """
    buffer: list[str] = []
    async for line in lines:
        if line == "":
            if buffer:
                yield "\n".join(buffer)
                buffer = []
            continue
        if line.startswith(":"):
            yield None
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].removeprefix(" "))
    if buffer:
        yield "\n".join(buffer)


def normalize_usage(raw: dict[str, Any]) -> dict[str, Any]:
    """Map the usage shapes chat-completions servers send onto the keys the core reads.

    ``input_tokens`` is the whole prompt, cached part included, which is what
    the OpenAI shape's ``prompt_tokens`` already means; the cached part is
    reported beside it, never added to it. The cache-hit count comes from
    ``prompt_tokens_details.cached_tokens`` (OpenAI, vLLM),
    ``prompt_cache_hit_tokens`` (DeepSeek) or ``cache_read_input_tokens``
    (Anthropic-shaped gateways). A ``cost`` the server reports (OpenRouter) is
    kept as ``cost_usd``.
    """
    prompt = _int(raw.get("prompt_tokens") or raw.get("input_tokens"))
    completion = _int(raw.get("completion_tokens") or raw.get("output_tokens"))
    details = raw.get("prompt_tokens_details") or {}
    cache_hit = _int(
        raw.get("prompt_cache_hit_tokens")
        or (details.get("cached_tokens") if isinstance(details, dict) else None)
        or raw.get("cache_read_input_tokens")
    )
    completion_details = raw.get("completion_tokens_details") or {}
    reasoning = _int(completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None)
    normalized: dict[str, Any] = {
        "input_tokens": prompt,
        "output_tokens": completion,
        "cache_read_tokens": cache_hit,
        "cache_read_input_tokens": cache_hit,
        "cache_creation_input_tokens": _int(raw.get("cache_creation_input_tokens")),
        "reasoning_tokens": reasoning,
        "total_tokens": _int(raw.get("total_tokens")) or prompt + completion,
    }
    cost = raw.get("cost")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        normalized["cost_usd"] = float(cost)
    return normalized


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class _OpenCall:
    call_id: str
    name: str = ""
    arguments: str = ""
    started: bool = False


@dataclass(slots=True)
class StreamAssembler:
    """Folds chat-completions chunks into deltas, one chunk at a time.

    Owns the state a stream builds up: the tool calls still open, the finish
    reason, the usage chunk, the opaque reasoning entries. :meth:`feed` returns
    the deltas a chunk produces; :meth:`close` returns the ``tool_use_stop``
    for every call still open, flagged when its arguments were cut short.
    """

    repair_partial_arguments: bool = True
    keep_reasoning_details: bool = False
    #: Grammars of tool calls written as reply text; see
    #: :mod:`protocore.providers.text_tool_calls`.
    text_tool_calls: tuple[TextToolCallGrammar, ...] = ()
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)
    _calls: dict[int, _OpenCall] = field(default_factory=dict)
    _slot_by_index: dict[int, int] = field(default_factory=dict)
    _stream_tag: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _guard: TextToolCallGuard | None = None

    def __post_init__(self) -> None:
        if self.text_tool_calls:
            self._guard = TextToolCallGuard(self.text_tool_calls)

    def feed(self, chunk: dict[str, Any]) -> list[ProviderDelta]:
        """The deltas one parsed chunk produces; a ``progress`` delta when it said nothing.

        A chunk with no content (a role-only first chunk, an empty delta while
        the model reasons silently) still proves the server is alive, and the
        core's idle watchdog counts from the last delta it saw: without the
        ``progress`` a long silent think was cut as a stalled stream. The
        trailing usage chunk is bookkeeping, not liveness, and produces nothing.
        """
        out: list[ProviderDelta] = []
        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage:
            self.usage = usage
        spoke = bool(usage)
        for choice in chunk.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            # ``reasoning_content`` (DeepSeek, vLLM's reasoning parsers) and
            # ``reasoning`` (OpenRouter, newer vLLM) carry the same thing.
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                out.append(ProviderDelta(kind=ProviderDeltaKind.thinking, content=reasoning))
                spoke = True
            details = delta.get("reasoning_details")
            if isinstance(details, list) and details:
                spoke = True
                if self.keep_reasoning_details:
                    # Whole entries, one per finished reasoning item: nothing to merge.
                    self.reasoning_details.extend(item for item in details if isinstance(item, dict))
            content = delta.get("content")
            if isinstance(content, str) and content:
                shown = self._guard.feed(content) if self._guard is not None else content
                if shown:
                    out.append(ProviderDelta(kind=ProviderDeltaKind.text, content=shown))
                else:
                    # Held text still proves the server is alive; without a
                    # delta a long held block would read as a stalled stream.
                    out.append(ProviderDelta(kind=ProviderDeltaKind.progress))
                spoke = True
            for call in delta.get("tool_calls") or ():
                if isinstance(call, dict):
                    out.extend(self._tool_call_fragment(call))
                    spoke = True
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])
                spoke = True
        if not spoke:
            out.append(ProviderDelta(kind=ProviderDeltaKind.progress))
        return out

    def _tool_call_fragment(self, fragment: dict[str, Any]) -> list[ProviderDelta]:
        """Assemble one tool-call fragment into the call it belongs to.

        Calls are keyed by ``index``. A fragment whose ``index`` is taken by a
        call with a different server id starts a new call: some servers number
        every call 0 and tell them apart only by id.
        """
        out: list[ProviderDelta] = []
        index = fragment.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            index = len(self._slot_by_index)
        server_id = fragment.get("id") if isinstance(fragment.get("id"), str) else None
        slot = self._slot_by_index.get(index)
        if slot is not None and server_id and self._calls[slot].call_id != server_id and self._calls[slot].started:
            slot = None
        if slot is None:
            slot = len(self._calls)
            self._slot_by_index[index] = slot
            self._calls[slot] = _OpenCall(call_id=server_id or f"call_{self._stream_tag}_{slot}")
        call = self._calls[slot]
        if server_id and not call.started:
            call.call_id = server_id
        function = fragment.get("function") or {}
        name = function.get("name")
        if isinstance(name, str) and name and not call.name:
            call.name = name
        if not call.started and call.name:
            call.started = True
            out.append(
                ProviderDelta(kind=ProviderDeltaKind.tool_use_start, tool_call_id=call.call_id, tool_name=call.name)
            )
            if call.arguments:
                # Arguments that arrived before the name are released with it.
                out.append(
                    ProviderDelta(
                        kind=ProviderDeltaKind.tool_use_input,
                        tool_call_id=call.call_id,
                        tool_input_delta=call.arguments,
                    )
                )
        piece = function.get("arguments")
        if isinstance(piece, str) and piece:
            call.arguments += piece
            if call.started:
                out.append(
                    ProviderDelta(kind=ProviderDeltaKind.tool_use_input, tool_call_id=call.call_id, tool_input_delta=piece)
                )
        return out

    @property
    def has_open_calls(self) -> bool:
        return bool(self._calls)

    def resolved_finish(self) -> str:
        """The finish reason to report: the server's, or the one its calls imply."""
        return self.finish_reason or ("tool_calls" if self._calls else "stop")

    def close(self) -> list[ProviderDelta]:
        """A ``tool_use_stop`` for every open call, in the order the calls began.

        Each stop carries the parsed arguments. Arguments that do not parse as
        they stand were cut short: the stop is flagged ``args_partial_truncated``
        whatever ended the stream, and also ``truncated_by_output_cap`` when it
        was the output cap (``finish_reason == "length"``), which is what sends
        the loop down its resume-the-call branch instead of dispatching half a
        call.
        """
        out = self._release_held_text()
        cut_by_cap = self.resolved_finish() == "length"
        for slot in sorted(self._calls):
            call = self._calls[slot]
            name = call.name or "unknown"
            if not call.started:
                out.append(ProviderDelta(kind=ProviderDeltaKind.tool_use_start, tool_call_id=call.call_id, tool_name=name))
            arguments, partial = self._parse_arguments(call.arguments)
            out.append(
                ProviderDelta(
                    kind=ProviderDeltaKind.tool_use_stop,
                    tool_call_id=call.call_id,
                    tool_name=name,
                    tool_input_final=arguments,
                    is_block_end=True,
                    truncated_by_output_cap=partial and cut_by_cap,
                    args_partial_truncated=partial,
                )
            )
        if self.finish_reason is None and self._calls:
            # Some servers end a stream of calls with ``[DONE]`` and no finish
            # reason. It used to be read after the calls below were cleared, fell
            # back to ``stop``, and the loop took a turn that asked for a tool as
            # finished. A stream with neither never gets here: the provider
            # refuses it as a dropped connection.
            self.finish_reason = "tool_calls"
        self._calls.clear()
        self._slot_by_index.clear()
        return out

    def _release_held_text(self) -> list[ProviderDelta]:
        """The text the guard held, and the calls it encoded opened as ordinary calls.

        A recovered call joins the open calls, so it closes exactly like one
        the server sent: the loop cannot tell them apart, which is the point.
        It also turns a ``stop`` into ``tool_calls``, because the server
        reported the turn as finished when the model had in fact asked for a
        tool.
        """
        if self._guard is None:
            return []
        held = self._guard.finish()
        out: list[ProviderDelta] = []
        if held.prose:
            out.append(ProviderDelta(kind=ProviderDeltaKind.text, content=held.prose))
        for number, call in enumerate(held.calls):
            slot = len(self._calls)
            self._calls[slot] = _OpenCall(
                call_id=f"call_{self._stream_tag}_text{number}", name=call.name, arguments=call.arguments_json
            )
        if held.calls:
            _logger.info("recovered %d tool call(s) written as text", len(held.calls))
            if self.finish_reason in (None, "stop"):
                self.finish_reason = "tool_calls"
        return out

    def _parse_arguments(self, raw: str) -> tuple[dict[str, Any], bool]:
        """The call's arguments as an object, and whether they were cut short.

        Valid JSON is taken as it is. Anything else goes to the core's partial
        parser, which also reads a whole object a model wrapped in a fence or
        prose (not cut short) and closes one the stream ended inside (cut
        short). A cut call keeps the repaired prefix only when
        :attr:`repair_partial_arguments` asks for it.
        """
        if not raw.strip():
            return {}, False
        try:
            value = json.loads(raw)
        except ValueError:
            pass
        else:
            return (value if isinstance(value, dict) else {"value": value}), False
        try:
            parsed, repaired = PartialJSONParser().parse_with_flag(raw)
        except OutputParserException:
            parsed, repaired = None, True
        if parsed is None:
            repaired = True
        if repaired:
            _logger.debug("tool-call arguments were cut short (%d chars)", len(raw))
            if not self.repair_partial_arguments:
                return {}, True
        return (parsed if isinstance(parsed, dict) else {}), repaired


__all__ = ["FinishReason", "StreamAssembler", "map_finish", "normalize_usage", "sse_events"]
