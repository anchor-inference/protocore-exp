"""Tool calls a model writes as text: held back from the reply and turned into real calls."""
from __future__ import annotations

import json
import re
from typing import Any

from protocore.contracts.llm import ProviderDeltaKind
from protocore.providers.openai_compat import ProviderCapabilities, StreamAssembler
from protocore.providers.text_tool_calls import (
    DSML,
    DsmlGrammar,
    ParsedText,
    TextToolCall,
    TextToolCallGrammar,
    TextToolCallGuard,
)

from ._openai_support import FakeServer, chunk_stream, collect, kinds, provider, sse_response, text_of, user_request

K = ProviderDeltaKind

# The full-width bar DeepSeek's tokenizer writes, as an escape so it is not read as "|".
BAR = "\uff5c"
OPEN = f"<{BAR}DSML{BAR}"
CLOSE = f"</{BAR}DSML{BAR}"

BLOCK = (
    f"{OPEN}tool_calls>\n"
    f'{OPEN}invoke name="read_file">\n'
    f'{OPEN}parameter name="path" string="true">notes/todo.md{CLOSE}parameter>\n'
    f'{OPEN}parameter name="limit" string="false">40{CLOSE}parameter>\n'
    f'{OPEN}parameter name="options">{{"follow": true}}{CLOSE}parameter>\n'
    f"{CLOSE}invoke>\n"
    f"{CLOSE}tool_calls>"
)


def _feed_all(guard: TextToolCallGuard, pieces: list[str]) -> str:
    return "".join(guard.feed(piece) for piece in pieces)


def test_a_dsml_block_parses_into_calls_with_typed_arguments() -> None:
    parsed = DSML.parse(BLOCK)

    assert parsed == ParsedText(
        prose="",
        calls=(
            TextToolCall(
                name="read_file",
                arguments={"path": "notes/todo.md", "limit": 40, "options": {"follow": True}},
            ),
        ),
    )


def test_every_spelling_of_the_dsml_bars_is_read() -> None:
    for bar in (BAR, "|", BAR * 2):
        text = BLOCK.replace(f"{BAR}DSML{BAR}", f"{bar}DSML{bar}")
        parsed = DSML.parse(text)
        assert parsed is not None, bar
        assert parsed.calls[0].name == "read_file"


def test_a_block_without_an_invoke_is_not_a_call() -> None:
    assert DSML.parse(f"{OPEN}tool_calls>nothing here{CLOSE}tool_calls>") is None


def test_the_guard_shows_the_prose_and_holds_the_block() -> None:
    guard = TextToolCallGuard([DSML])
    shown = _feed_all(guard, ["Let me look at the file. ", BLOCK[:30], BLOCK[30:]])

    assert shown == "Let me look at the file. "
    assert guard.holding
    finished = guard.finish()
    assert finished.prose == ""
    assert [call.name for call in finished.calls] == ["read_file"]


def test_a_marker_split_across_chunks_is_still_caught() -> None:
    guard = TextToolCallGuard([DSML])
    pieces = ["Checking", " now<", f"{BAR}DS", f"ML{BAR}tool_calls>", BLOCK[len(f"{OPEN}tool_calls>"):]]
    shown = _feed_all(guard, pieces)

    assert shown == "Checking now"
    assert len(guard.finish().calls) == 1


def test_a_partial_marker_that_turns_out_to_be_prose_is_released() -> None:
    guard = TextToolCallGuard([DSML])
    shown = _feed_all(guard, ["a <", "b"])

    assert shown == "a <b"
    assert guard.finish() == ParsedText(prose="", calls=())


def test_text_after_the_block_is_released_with_the_calls() -> None:
    guard = TextToolCallGuard([DSML])
    shown = _feed_all(guard, [BLOCK, "\n\nDone."])

    assert shown == ""
    finished = guard.finish()
    assert finished.prose == "Done."
    assert len(finished.calls) == 1


def test_an_unfinished_block_is_released_unchanged() -> None:
    guard = TextToolCallGuard([DSML])
    cut = BLOCK[:60]
    shown = _feed_all(guard, ["Intro. ", cut])

    assert shown == "Intro. "
    assert guard.finish() == ParsedText(prose=cut, calls=())


def test_a_marker_mentioned_in_prose_is_released_past_the_hold_limit() -> None:
    guard = TextToolCallGuard([DSML], hold_limit=50)
    prose = f"The markup starts with {OPEN} and then " + "x" * 80

    shown = _feed_all(guard, [prose[:30], prose[30:]])

    assert shown == prose
    assert not guard.holding


class _JsonTagGrammar:
    """A test grammar: ``<call>{"name": ..., "arguments": {...}}</call>``."""

    name = "json-tag"
    markers = ("<call>",)

    def is_call_block(self, held: str) -> bool:
        return held.startswith("<call>{")

    def parse(self, held: str) -> ParsedText | None:
        found = re.search(r"<call>(.*?)</call>", held, re.DOTALL)
        if found is None:
            return None
        data: dict[str, Any] = json.loads(found.group(1))
        rest = (held[: found.start()] + held[found.end():]).strip()
        return ParsedText(prose=rest, calls=(TextToolCall(name=data["name"], arguments=data["arguments"]),))


def test_a_grammar_is_pluggable_and_the_earliest_marker_wins() -> None:
    grammar: TextToolCallGrammar = _JsonTagGrammar()
    guard = TextToolCallGuard([DSML, grammar])
    shown = guard.feed('ok <call>{"name": "ping", "arguments": {"n": 1}}</call> then ' + OPEN)

    assert shown == "ok "
    finished = guard.finish()
    assert finished.calls == (TextToolCall(name="ping", arguments={"n": 1}),)
    assert finished.prose == "then " + OPEN


def test_the_dsml_grammar_is_a_grammar() -> None:
    assert isinstance(DsmlGrammar(), TextToolCallGrammar)


def test_the_assembler_turns_held_text_into_tool_calls() -> None:
    assembler = StreamAssembler(text_tool_calls=(DSML,))
    deltas = []
    for piece in ["Reading. ", BLOCK[:40], BLOCK[40:]]:
        deltas += assembler.feed({"choices": [{"delta": {"content": piece}}]})
    deltas += assembler.feed({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    deltas += assembler.close()

    assert text_of(deltas) == "Reading. "
    # Held text still proves the stream is alive.
    assert [d.kind for d in deltas].count(K.progress) >= 2
    stops = [d for d in deltas if d.kind is K.tool_use_stop]
    assert [d.tool_name for d in stops] == ["read_file"]
    assert stops[0].tool_input_final == {"path": "notes/todo.md", "limit": 40, "options": {"follow": True}}
    assert assembler.resolved_finish() == "tool_calls"


async def test_the_adapter_recovers_text_calls_only_when_the_capability_is_on() -> None:
    body = chunk_stream(
        {"choices": [{"delta": {"content": "Reading. "}}]},
        {"choices": [{"delta": {"content": BLOCK}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )

    on = provider(
        FakeServer(sse_response(body)), capabilities=ProviderCapabilities(text_tool_calls=(DSML,))
    )
    deltas = await collect(on.stream_with_tools(user_request()))
    assert kinds(deltas) == ["text", "tool_use_start", "tool_use_stop", "usage", "finish"]
    assert "DSML" not in text_of(deltas)
    finish = next(d for d in deltas if d.kind is K.finish)
    assert finish.finish_reason == "tool_use"

    off = provider(FakeServer(sse_response(body)))
    plain = await collect(off.stream_with_tools(user_request()))
    assert text_of(plain) == "Reading. " + BLOCK
    assert "tool_use_start" not in kinds(plain)
