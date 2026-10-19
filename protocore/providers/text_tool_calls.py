"""Tool calls a model wrote as text, recovered while the reply streams.

Some models now and then write a tool call in their own markup inside the
reply text instead of as a structured call: the server does not recognise it
and delivers it as plain text. Left alone, the markup is shown to the person
reading the reply and the turn ends as if the model had answered, with the
call it meant to make never made.

:class:`TextToolCallGuard` sits on the text stream. From the first marker of a
known grammar on, it holds the text back instead of releasing it; at the end of
the stream it either turns a complete block into :class:`TextToolCall` values,
releasing only the prose around it, or releases everything it held unchanged.
Text before the marker is never delayed beyond the few characters that might
be the start of one.

Each markup is a :class:`TextToolCallGrammar`. The first shipped is
:class:`DsmlGrammar` (:data:`DSML`), the markup DeepSeek models emit. Pure
Python, no dependency: the reference adapter uses it when an endpoint lists
grammars in ``ProviderCapabilities.text_tool_calls``, and any other adapter
can feed it its own text deltas.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

#: How many characters may be held after a marker before the guard asks the
#: grammar whether it is still looking at a call block. A model that merely
#: mentions the markup in prose would otherwise have the rest of its reply
#: held until the stream ends.
DEFAULT_HOLD_LIMIT = 4_000


@dataclass(frozen=True, slots=True)
class TextToolCall:
    """One call recovered from text: the tool's name and its arguments."""

    name: str
    arguments: dict[str, Any]

    @property
    def arguments_json(self) -> str:
        return json.dumps(self.arguments, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ParsedText:
    """What held text turned out to be: prose to release, and the calls it encoded."""

    prose: str
    calls: tuple[TextToolCall, ...] = ()


@runtime_checkable
class TextToolCallGrammar(Protocol):
    """One markup in which a model writes tool calls as text.

    ``markers`` are the strings whose appearance starts holding text back;
    they should be distinctive enough that ordinary prose rarely contains
    them. ``is_call_block`` is asked once the held text passes the hold
    limit: ``False`` releases it as prose. ``parse`` reads the held text,
    which starts at a marker, and returns ``None`` when it holds no complete
    call, in which case the text is released unchanged.
    """

    @property
    def name(self) -> str: ...

    @property
    def markers(self) -> tuple[str, ...]: ...

    def is_call_block(self, held: str) -> bool: ...

    def parse(self, held: str) -> ParsedText | None: ...


class TextToolCallGuard:
    """Releases streamed text, holding back anything from a grammar's marker onwards.

    Feed every text piece through :meth:`feed` and show what it returns; call
    :meth:`finish` once when the stream ends and show its ``prose``, then
    dispatch its ``calls``. One guard per stream.
    """

    def __init__(self, grammars: Sequence[TextToolCallGrammar], *, hold_limit: int = DEFAULT_HOLD_LIMIT) -> None:
        if hold_limit < 1:
            raise ValueError("hold_limit must be positive")
        self._grammars = tuple(grammars)
        self._hold_limit = hold_limit
        self._longest_marker = max((len(m) for g in self._grammars for m in g.markers), default=0)
        self._held = ""
        self._active: TextToolCallGrammar | None = None
        # The end of the last piece when it could be the start of a marker
        # split across two chunks.
        self._tail = ""

    @property
    def holding(self) -> bool:
        """Whether text is being held after a marker."""
        return self._active is not None

    def feed(self, text: str) -> str:
        """The part of ``text`` that can be shown now."""
        if self._active is not None:
            self._held += text
            if len(self._held) > self._hold_limit and not self._active.is_call_block(self._held):
                # A mention of the markup in prose, not a call block.
                held, self._held, self._active = self._held, "", None
                return held
            return ""
        buffer = self._tail + text
        self._tail = ""
        earliest: tuple[int, TextToolCallGrammar] | None = None
        for grammar in self._grammars:
            for marker in grammar.markers:
                index = buffer.find(marker)
                if index != -1 and (earliest is None or index < earliest[0]):
                    earliest = (index, grammar)
        if earliest is not None:
            index, grammar = earliest
            self._active = grammar
            self._held = buffer[index:]
            return buffer[:index]
        for size in range(min(self._longest_marker - 1, len(buffer)), 0, -1):
            suffix = buffer[-size:]
            if any(m.startswith(suffix) for g in self._grammars for m in g.markers):
                self._tail = suffix
                return buffer[:-size]
        return buffer

    def finish(self) -> ParsedText:
        """Everything still held, as prose to release and the calls it encoded."""
        rest, self._tail = self._tail, ""
        grammar, held = self._active, self._held
        self._active, self._held = None, ""
        if grammar is None:
            return ParsedText(prose=rest)
        parsed = grammar.parse(held)
        if parsed is None or not parsed.calls:
            return ParsedText(prose=rest + held)
        return ParsedText(prose=rest + parsed.prose, calls=parsed.calls)


# DSML ----------------------------------------------------------------------

#: The bars around ``DSML`` come as the full-width bar the model's tokenizer
#: uses (U+FF5C, written as an escape so no reader mistakes it for ``|``),
#: doubled, or as the ASCII bar a decoder sometimes substitutes.
_FULLWIDTH_BAR = "\uff5c"
_BARS = (_FULLWIDTH_BAR * 2, _FULLWIDTH_BAR, "|")
_DSML_OPEN = tuple(f"<{bar}DSML{bar}" for bar in _BARS)
_DSML_CLOSE = tuple(f"</{bar}DSML{bar}" for bar in _BARS)
_CANON_OPEN = "<DSML:"
_CANON_CLOSE = "</DSML:"
_BLOCK = re.compile(r"<DSML:tool_calls>(.*?)(?:</DSML:tool_calls>|$)", re.DOTALL)
_INVOKE = re.compile(r"<DSML:invoke name=\"([^\"]+)\">(.*?)</DSML:invoke>", re.DOTALL)
_PARAMETER = re.compile(r"<DSML:parameter name=\"([^\"]+)\"([^>]*)>(.*?)</DSML:parameter>", re.DOTALL)


def _canonical(text: str) -> str:
    """The markup with every spelling of the bars reduced to one, so one set of patterns reads it."""
    for marker in _DSML_CLOSE:
        text = text.replace(marker, _CANON_CLOSE)
    for marker in _DSML_OPEN:
        text = text.replace(marker, _CANON_OPEN)
    return text


class DsmlGrammar:
    """DeepSeek's DSML tool-call markup.

    .. code-block:: text

        <|DSML|tool_calls>
        <|DSML|invoke name="read_file">
        <|DSML|parameter name="path" string="true">notes.md</|DSML|parameter>
        </|DSML|invoke>
        </|DSML|tool_calls>

    The bars are the full-width U+FF5C as the model writes them, single or
    doubled; the ASCII bar shown here is read as well.

    A parameter marked ``string="true"`` is taken as text. Any other is read as
    JSON when it looks like JSON (an object, an array, a number, ``true``,
    ``false``, ``null``) and as text when it does not parse.
    """

    name = "dsml"
    markers = _DSML_OPEN + _DSML_CLOSE

    def is_call_block(self, held: str) -> bool:
        return "tool_calls" in _canonical(held[:200])

    def parse(self, held: str) -> ParsedText | None:
        canon = _canonical(held)
        block = _BLOCK.search(canon)
        if block is None:
            return None
        calls = tuple(
            TextToolCall(name=invoke.group(1), arguments=_dsml_arguments(invoke.group(2)))
            for invoke in _INVOKE.finditer(block.group(1))
        )
        if not calls:
            return None
        before = canon[: block.start()].strip()
        after = canon[block.end():].strip()
        prose = f"{before}\n\n{after}" if before and after else before or after
        return ParsedText(prose=prose, calls=calls)


def _dsml_arguments(body: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for parameter in _PARAMETER.finditer(body):
        name, attributes, raw = parameter.group(1), parameter.group(2), parameter.group(3)
        value = raw.strip()
        if 'string="true"' not in attributes and _looks_like_json(value):
            try:
                arguments[name] = json.loads(value)
                continue
            except ValueError:
                pass
        arguments[name] = value
    return arguments


def _looks_like_json(value: str) -> bool:
    # DSML marks a text parameter ``string="true"``; any other was written as
    # JSON, so a bare number is a number and not the text of one.
    if value[:1] in ("{", "[", '"') or value in ("true", "false", "null"):
        return True
    return bool(re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", value))


#: The DSML grammar, ready to list in ``ProviderCapabilities.text_tool_calls``.
DSML = DsmlGrammar()


__all__ = [
    "DEFAULT_HOLD_LIMIT",
    "DSML",
    "DsmlGrammar",
    "ParsedText",
    "TextToolCall",
    "TextToolCallGrammar",
    "TextToolCallGuard",
]
