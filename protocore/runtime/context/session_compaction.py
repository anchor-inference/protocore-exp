"""Compacting a session between runs: one summary in place of most of its history.

The tiers in :mod:`protocore.runtime.context.compaction` work inside a run.
A session that lives across runs also needs "compact this now" outside one —
after a run settles, before a run whose history no longer fits, on demand —
and every host that kept such a session built the same service around a
summariser call. This module is that service, pure with respect to storage:
:func:`compact_session` takes a history and returns the history that should
replace it, and the caller commits it (:func:`commit_session_compaction` for an
engine, :meth:`SessionCompaction.apply` for a host's own store).

What it does, and the failure behind each part:

* **The previous summary is the base to update.** A summary at the head of the
  history goes to the summariser whole, apart from the transcript, with an
  instruction to carry it forward. When the previous summary was re-summarised
  as part of a clipped transcript instead, the chain broke at every pass and a
  session kept only the last topic it worked on.
* **A history longer than one call is summarised in parts.** Parts run in
  parallel under a bound, their notes are merged in rounds while they are still
  too long for one call, and the last merge is made onto the base. Nothing is
  cut from the front of the transcript.
* **The sections are checked.** A summary missing a required section is asked
  for once more, naming what was wrong; a second miss is accepted with the
  problem reported, because a usable summary beats a session that cannot be
  compacted.
* **A ladder of summarisers.** A cheaper model configured for summaries, then
  the session's own; one that refuses, is out of quota or times out hands over
  to the next. A summary model past its limit used to end a manual compaction
  in a bare error while the session's own model stood ready.
* **Retries that respect a quota.** A transient failure is asked again after a
  pause, twice; a rate limit is not (the next summariser answers sooner); a
  stalled call gets one more try with twice the time, because a model that
  reasons whether asked to or not needs it on a large part.
* **A hold after a pass that failed or freed little** (:class:`CompactionHold`,
  :class:`CompactionHoldPolicy`): a value the host persists, so a session whose
  summariser times out is not held for minutes on every message, and one whose
  kept tail is itself over the trigger does not compact the same nothing again.
* **Safe to cancel, safe to commit late.** The summary is made without holding
  anything; the result remembers, by identity, the messages it summarised, and
  is committed only while the history still starts with exactly those. A turn
  appended meanwhile is kept after the summary; a summarised message that
  changed makes the result stale, and it is discarded with the history intact.
  Cancelling (``cancel``, or cancelling the task) leaves nothing behind.

The texts the summariser is asked with are parameters (:class:`SummaryPrompts`)
with generic defaults; a host that has its own words for its sections passes
them, and so does one that wants another rendering of the transcript
(``render``) or code-written additions to the summary (``decorate``).
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from protocore.contracts.llm import (
    ILLMProvider,
    LLMError,
    LLMObservabilityContext,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    ProviderDelta,
    ProviderFailureReason,
    provider_failure_of,
)
from protocore.contracts.types import (
    COMPACTION_SUMMARY_METADATA_KEY,
    OPERATOR_WORDS_METADATA_KEY,
    Message,
    MessageOrigin,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.logging_utils import get_logger
from protocore.runtime.context.ledger import is_ledger, operator_words

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.contracts.runtime_constants import LoopConstants
    from protocore.runtime.events import TurnEvent
    from protocore.runtime.query_engine import QueryEngine

_logger = get_logger(__name__)

SESSION_COMPACTION_METADATA_KEY: Final[str] = "protocore.session_compaction"
"""``Message.metadata`` key on a session summary: ``{reason, messages, kept,
summariser}`` — what the summary stands for and who wrote it. The summary also
carries :data:`~protocore.contracts.types.COMPACTION_SUMMARY_METADATA_KEY`, so
its origin is :attr:`~protocore.contracts.types.MessageOrigin.compaction`."""

CHARS_PER_TOKEN: Final[int] = 4
"""The approximation the parts are sized by. A part is a request the summariser
must take whole, so being a little off costs a slightly smaller or larger part,
never a refused one: the bound is far below any window it is used with."""

_SCAFFOLDING: Final[frozenset[MessageOrigin]] = frozenset(
    {MessageOrigin.nudge, MessageOrigin.wind_down, MessageOrigin.recovery}
)

DEFAULT_SECTIONS: Final[tuple[str, ...]] = (
    "Instructions",
    "Facts",
    "Progress",
    "Open questions",
    "Identifiers",
)

_DEFAULT_SECTION_RULES = (
    "Markdown with exactly these level-2 sections, each once, in this order: {headings}. "
    "Instructions: every rule, preference and boundary the user stated that still applies, "
    "in the user's words. Facts: what stays true — paths, hosts, commands that worked, "
    "decisions and their reasons. Progress: one entry per task or topic, with what was done, "
    "what is in progress and the exact next step. Open questions: what the transcript does not "
    "show; a missing result means the outcome is unknown, not that it happened. Identifiers: "
    "every path, id, URL, number or name later work may need, exactly as it appeared. "
    "Never write a secret value; name where it is kept instead."
)


class _Missing(dict[str, str]):
    """Leaves an unknown placeholder in place, so a host text with its own braces formats."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class SummaryPrompts:
    """The texts a session summary is asked with.

    Each is formatted with the fields it may use — ``{language}``,
    ``{max_words}``, ``{sections}`` (the section rules), ``{base_rule}`` (the
    instruction about the previous summary, or nothing), and for a part
    ``{index}`` and ``{total}``. A placeholder a text does not use is simply
    absent from it, and one the core does not know is left as written.

    ``sections`` are the level-2 headings every summary must carry once each,
    in order (:func:`validate_sections`); empty turns the check off.
    """

    sections: tuple[str, ...] = DEFAULT_SECTIONS
    section_rules: str = _DEFAULT_SECTION_RULES
    single: str = (
        "You keep the working memory of a long conversation between a user and an agent, so the "
        "agent can continue in a fresh context without the user explaining anything again. Write "
        "it in {language} as {sections}\n{base_rule}At most ~{max_words} words."
    )
    part: str = (
        "The text below is ONE PART (part {index} of {total}) of a longer conversation between a "
        "user and an agent. Write notes on this part only, in {language}, as {sections}\n"
        "At most ~{max_words} words."
    )
    merge_notes: str = (
        "Below are notes on consecutive parts of one conversation, oldest first. Merge them into "
        "one set of notes in {language}, keeping the latest state of each task and every "
        "instruction, fact and identifier, as {sections}\nAt most ~{max_words} words."
    )
    merge: str = (
        "You keep the working memory of a long conversation between a user and an agent. Below "
        "are notes on consecutive parts of it, oldest first. Write the memory from them in "
        "{language} as {sections}\n{base_rule}At most ~{max_words} words."
    )
    base_rule: str = (
        "The text starts with <previous-summary>: the memory of this conversation as it stood "
        "before the transcript that follows. It is the base you update, not material to condense "
        "again: carry every instruction and fact in it forward unless the transcript withdraws "
        "it, keep every task in it until the transcript shows it finished, and add what is new."
    )
    focus: str = "\n\nThe user asks to focus on: {instructions}"
    retry: str = (
        "\n\nYour previous attempt was rejected: {problem}. Write every section, each exactly "
        "once, in the given order."
    )

    def rules(self) -> str:
        headings = " · ".join(f"## {name}" for name in self.sections)
        return self.section_rules.format_map(_Missing(headings=headings))


@dataclass(frozen=True)
class SessionCompactionSettings:
    """The tunables of a session compaction. Defaults are the values a long-lived host settled on."""

    max_words: int = 1200
    """The summary's floor in words; a large window raises it (:meth:`words_for`)."""
    words_per_window_tokens: int = 100
    """One word of summary per this many tokens of window: 4,000 words on a 400k window.
    The summary is the only memory of what it replaces; a fixed 1,200 words for a session
    of five parallel tasks kept only the latest one."""
    words_cap: int = 8000
    tokens_per_word: int = 3
    """Output tokens a summary word may take; some languages run at two and a half."""
    min_output_tokens: int = 6000
    max_output_tokens: int = 32_000
    chunk_tokens: int = 30_000
    """The size of one part. A transcript up to this is summarised in one call."""
    parallel_calls: int = 6
    """Parts in flight at once. Thirty at once is what a provider answers with a rate limit."""
    merge_rounds_max: int = 4
    call_timeout_seconds: float = 180.0
    """One summariser call's deadline; a stalled call gets one more try at twice this."""
    retry_pauses: tuple[float, ...] = (5.0, 20.0)
    """The pauses before a transiently failed call is asked again, one per retry."""
    temperature: float = 0.2
    operator_passages_kept: int = 400
    """The user's passages a summary carries forward in its metadata for the next
    compaction and the in-run ledger to quote again."""
    assistant_chars: int = 8000
    note_chars: int = 12_000
    tool_argument_chars: int = 1000
    tool_result_chars: int = 1500

    def words_for(self, window: int) -> int:
        """The summary's length budget for a model whose window is ``window`` tokens."""
        if window <= 0:
            return self.max_words
        return max(self.max_words, min(self.words_cap, window // max(1, self.words_per_window_tokens)))

    def output_tokens_for(self, words: int) -> int:
        return min(self.max_output_tokens, max(self.min_output_tokens, words * self.tokens_per_word + 1000))


@dataclass(frozen=True)
class Summariser:
    """One rung of the ladder: a provider and the model to ask it for."""

    provider: ILLMProvider
    model: str
    label: str = ""
    """How reports name this rung; the model name when empty."""

    @property
    def name(self) -> str:
        return self.label or self.model


@dataclass(frozen=True)
class SummaryInput:
    """What the summariser is given: the memory as it stood, and what happened since, both whole."""

    previous: str
    transcript: str


@dataclass(frozen=True)
class CompactionProgress:
    """One step of a compaction, for a host that shows it: a long history takes minutes."""

    stage: str
    """``summarising``, ``merging`` or ``writing``."""
    parts_total: int = 0
    parts_done: int = 0
    summariser: str = ""


class NothingToCompact(ValueError):
    """The history holds nothing a summary could replace: empty, or entirely inside the kept tail."""


class SessionCompactionCancelled(Exception):
    """The compaction was cancelled before it finished; nothing was produced."""


class SessionCompactionFailed(RuntimeError):
    """No summariser of the ladder produced a summary. ``failures`` says why, one entry per rung."""

    def __init__(self, failures: Sequence[str]) -> None:
        self.failures: tuple[str, ...] = tuple(failures)
        super().__init__("no summariser could summarise the history: " + "; ".join(self.failures))


class SummaryEmptyError(Exception):
    """A summariser answered with no text. The next rung of the ladder is asked."""


@dataclass(frozen=True)
class SessionCompaction:
    """The outcome of :func:`compact_session`: the summary and what it replaces.

    ``source`` is the summarised prefix, held strongly: it is the revision the
    summary was made against, compared by identity when the result is applied.
    """

    summary: Message
    summary_text: str
    source: tuple[Message, ...]
    carried: tuple[Message, ...]
    """Messages of the prefix carried past the summariser unchanged: the newest
    compaction ledger, whose exact values a model rewriting them would erode."""
    tail: tuple[Message, ...]
    summariser: str
    failures: tuple[str, ...] = ()
    """Rungs of the ladder that failed before ``summariser`` answered."""
    parts: int = 1
    merge_rounds: int = 0
    section_problem: str = ""
    """Why the accepted summary failed the section check, or empty when it passed."""
    reason: str = "session"

    @property
    def replaced(self) -> int:
        return len(self.source)

    @property
    def kept(self) -> int:
        return len(self.tail)

    @property
    def history(self) -> tuple[Message, ...]:
        """The history as of the moment the summary was made, with the summary in place."""
        return (self.summary, *self.carried, *self.tail)

    def applies_to(self, history: Sequence[Message]) -> bool:
        """Whether ``history`` still starts with exactly the messages this summary replaces."""
        if len(history) < len(self.source):
            return False
        return all(before is now for before, now in zip(self.source, history, strict=False))

    def apply(self, history: Sequence[Message]) -> list[Message]:
        """``history`` with the summarised prefix replaced; everything after it is kept as it is now.

        Raises :class:`ValueError` when the prefix changed since the summary
        was made: committing it would put a summary of turns that no longer
        exist in place of turns it never read.
        """
        if not self.applies_to(history):
            raise ValueError("the summarised part of the history changed; this compaction is stale")
        return [self.summary, *self.carried, *history[len(self.source) :]]


# ---------------------------------------------------------------------------
# The hold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionHold:
    """A pause on automatic compaction, as a value a host persists.

    ``until`` is wall-clock seconds (``time.time()``), so it means the same
    after a restart; ``None`` holds until the prompt grows, whatever the time.
    """

    prompt_tokens: int
    failures: int
    until: float | None
    reason: str
    """``failed`` or ``freed_little``."""

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_tokens": self.prompt_tokens, "failures": self.failures, "until": self.until, "reason": self.reason}

    @classmethod
    def from_dict(cls, raw: object) -> CompactionHold | None:
        """The hold a host stored, or ``None`` for nothing or something unreadable — no hold at all."""
        if not isinstance(raw, dict):
            return None
        try:
            until = raw.get("until")
            return cls(
                prompt_tokens=int(raw["prompt_tokens"]),
                failures=int(raw["failures"]),
                until=None if until is None else float(until),
                reason=str(raw["reason"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class CompactionHoldPolicy:
    """When automatic compaction waits, and for how long.

    A compaction that failed, or one that left the prompt about as large as it
    found it, was tried again after every turn: an orchestrating session whose
    summariser timed out was held for three minutes per message and never
    shrank. It waits now until the prompt has really grown or, after a
    failure, until some time has passed. A compaction the user asks for, or
    one a history that no longer fits requires, ignores the hold.
    """

    growth_ratio: float = 0.1
    """The prompt must grow by this share of the window to end a hold."""
    retry_seconds: float = 1800.0
    """A failure holds this long, doubling with each failure in a row."""
    retry_max_seconds: float = 6 * 3600.0
    min_freed_ratio: float = 0.1
    """A pass that took less than this share off the prompt freed nothing."""

    def holds(self, hold: CompactionHold | None, *, prompt_tokens: int, window: int, now: float) -> bool:
        if hold is None:
            return False
        grown = prompt_tokens >= hold.prompt_tokens + self.growth_ratio * window
        if grown:
            return False
        return hold.until is None or now < hold.until

    def after_failure(self, hold: CompactionHold | None, *, prompt_tokens: int, now: float) -> CompactionHold:
        failures = (hold.failures if hold is not None else 0) + 1
        wait = min(self.retry_seconds * 2 ** (failures - 1), self.retry_max_seconds)
        return CompactionHold(prompt_tokens=prompt_tokens, failures=failures, until=now + wait, reason="failed")

    def after_pass(
        self, hold: CompactionHold | None, *, prompt_before: int, prompt_after: int, trigger_tokens: int
    ) -> CompactionHold | None:
        """The hold after a pass that worked: ``None`` unless it left the prompt over the trigger or freed little.

        What is left still over the trigger — the kept tail, or what the host
        puts before every prompt, is itself that large — would compact again
        next turn and free the same nothing; only growth makes another pass
        worth paying for.
        """
        freed = prompt_before - prompt_after
        if prompt_after >= trigger_tokens or freed < self.min_freed_ratio * prompt_before:
            failures = (hold.failures if hold is not None else 0) + 1
            return CompactionHold(prompt_tokens=prompt_after, failures=failures, until=None, reason="freed_little")
        return None


# ---------------------------------------------------------------------------
# Rendering and splitting
# ---------------------------------------------------------------------------


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]} … [{len(text) - limit} chars left out] … {text[-half:]}"


def _summary_text(message: Message) -> str:
    return re.sub(r"</?compacted-turn[^>]*>", "", message.text).strip()


def _is_users_turn(message: Message) -> bool:
    if message.role is not MessageRole.user or message.origin is not MessageOrigin.conversation:
        return False
    if any(isinstance(block, ToolResultBlock) for block in message.content_blocks):
        return False
    marked = message.metadata.get(OPERATOR_WORDS_METADATA_KEY)
    return marked is None or marked is True


def leading_summaries(history: Sequence[Message]) -> int:
    """How many messages at the start of ``history`` are compaction records.

    Only the leading ones are the memory to update. A summary further in is a
    span summarised inside a run, and is read in its place in the transcript.
    """
    index = 0
    while index < len(history) and history[index].origin is MessageOrigin.compaction:
        index += 1
    return index


def render_for_summary(
    history: Sequence[Message], *, settings: SessionCompactionSettings | None = None
) -> SummaryInput:
    """The previous summary and the rest of the history, as text for the summariser.

    Nothing is cut from the front or the back. Per message: what the user wrote
    is never cut, an earlier summary is never cut, an agent's text, a note and
    a tool exchange are cut in the middle past their bounds. The ledger is left
    out (it is carried beside the summary), and so is the loop's scaffolding —
    nudges, the wind-down notice, recovery turns — which says nothing about the
    work.
    """
    limits = settings or SessionCompactionSettings()
    start = leading_summaries(history)
    previous = "\n\n".join(text for m in history[:start] if not is_ledger(m) and (text := _summary_text(m)))
    lines: list[str] = []
    for message in history[start:]:
        if message.role is MessageRole.system or is_ledger(message) or message.origin in _SCAFFOLDING:
            continue
        if message.origin is MessageOrigin.compaction:
            text = _summary_text(message)
            if text:
                lines.append(f"[summary of earlier turns] {text}")
            continue
        users = _is_users_turn(message)
        for block in message.content_blocks:
            if isinstance(block, TextBlock):
                text = block.text.strip()
                if not text:
                    continue
                if users:
                    lines.append(f"[user] {text}")
                elif message.role is MessageRole.assistant:
                    lines.append(f"[assistant] {_clip_middle(text, limits.assistant_chars)}")
                else:
                    # A user-role message that is not the user's own turn — a
                    # wake-up, a relayed report — is a note, and long ones carry
                    # their result at the ends.
                    label = "note" if message.role is MessageRole.user else message.role.value
                    lines.append(f"[{label}] {_clip_middle(text, limits.note_chars)}")
            elif isinstance(block, ToolUseBlock):
                arguments = block.arguments_json or ""
                lines.append(f"[tool call] {block.name} {_clip_middle(arguments, limits.tool_argument_chars)}")
            elif isinstance(block, ToolResultBlock):
                body = (block.content or "").strip()
                flag = " ERROR" if block.is_error else ""
                lines.append(f"[tool result{flag}] {_clip_middle(body, limits.tool_result_chars)}")
            elif isinstance(block, ThinkingBlock):
                continue
    return SummaryInput(previous=previous, transcript="\n".join(lines))


def split_transcript(text: str, limit_chars: int) -> list[str]:
    """Consecutive parts of ``text`` of at most ``limit_chars``, split on line ends.

    A single line longer than a part is split inside rather than sent as a
    part the summariser cannot take.
    """
    limit = max(1, limit_chars)
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                parts.append("\n".join(current))
                current, size = [], 0
            parts.append(line[:limit])
            line = line[limit:]
        if size + len(line) + 1 > limit and current:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        parts.append("\n".join(current))
    return parts


def _pack(notes: Sequence[str], limit: int) -> list[list[str]]:
    """Consecutive notes grouped to stay within ``limit``, at least two to a group.

    Two at least, so each round halves the notes even when every note is long:
    a merge's output is bounded by its word budget, so the rounds converge where
    groups of one never would.
    """
    groups: list[list[str]] = []
    size = 0
    for note in notes:
        if groups and (size + len(note) <= limit or len(groups[-1]) < 2):
            groups[-1].append(note)
            size += len(note)
        else:
            groups.append([note])
            size = len(note)
    return groups


def _joined(notes: Sequence[str]) -> str:
    return "\n\n".join(f"<part {i + 1}>\n{text}\n</part {i + 1}>" for i, text in enumerate(notes) if text)


def _with_base(previous: str, body: str) -> str:
    if not previous:
        return body
    return f"<previous-summary>\n{previous}\n</previous-summary>\n\n{body}"


def validate_sections(summary: str, sections: Sequence[str]) -> str:
    """Why ``summary`` is rejected, or an empty string when every section appears once, in order."""
    positions: list[int] = []
    for name in sections:
        pattern = re.compile(rf"^##\s+{re.escape(name)}\s*$", re.MULTILINE | re.IGNORECASE)
        found = list(pattern.finditer(summary))
        if len(found) != 1:
            return f"section '## {name}' appears {len(found)} times (expected once)"
        positions.append(found[0].start())
    if positions != sorted(positions):
        return "sections are out of order"
    return ""


def session_compaction_cut(
    history: Sequence[Message],
    *,
    keep_recent_messages: int = 0,
    keep_recent_tokens: int = 0,
    rc: LoopConstants | None = None,
) -> int:
    """Where the kept tail starts: at least the requested tail, beginning at a turn boundary.

    A boundary is a user's own turn — not a tool result, not the loop's
    scaffolding, not a compaction record — so a tool call is never separated
    from its result and a summary never ends on an unanswered call. The cut
    moves back to the nearest boundary, keeping a little more than asked rather
    than splitting an exchange. With nothing to keep, the whole history is
    summarised.
    """
    if keep_recent_messages <= 0 and keep_recent_tokens <= 0:
        return len(history)
    cut = len(history)
    if keep_recent_messages > 0:
        cut = min(cut, max(0, len(history) - keep_recent_messages))
    if keep_recent_tokens > 0:
        from protocore.contracts.runtime_constants import LoopConstants
        from protocore.runtime.context.compaction import estimate_message_tokens

        constants = rc or LoopConstants()
        used = 0
        index = len(history)
        while index > 0:
            used += estimate_message_tokens(history[index - 1], constants)
            if used > keep_recent_tokens:
                break
            index -= 1
        cut = min(cut, index)
    while cut > 0 and not _is_users_turn(history[cut]):
        cut -= 1
    return cut


def _operator_passages(history: Sequence[Message], keep: int) -> list[str]:
    found: list[str] = []
    for message in history:
        if is_ledger(message):
            continue
        found += operator_words(message, is_operator=_is_users_turn)
    unique: dict[str, None] = {}
    for passage in found:
        unique.pop(passage, None)  # a passage said again counts where it was said last
        unique[passage] = None
    return list(unique)[-keep:] if keep > 0 else []


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


async def _guard[T](make: Callable[[], Coroutine[Any, Any, T]], cancel: asyncio.Event | None) -> T:
    """Await what ``make`` starts unless ``cancel`` is set first; then it is cancelled and nothing is returned.

    Takes a factory rather than a coroutine so that a cancel already set
    starts no call at all, not even one that is created and dropped.
    """
    if cancel is None:
        return await make()
    if cancel.is_set():
        raise SessionCompactionCancelled("cancelled before the call")
    task: asyncio.Task[T] = asyncio.ensure_future(make())
    waiter = asyncio.ensure_future(cancel.wait())
    try:
        done, _pending = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        task.cancel()
        waiter.cancel()
        raise
    if task in done:
        waiter.cancel()
        return task.result()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    raise SessionCompactionCancelled("cancelled during a summariser call")


@dataclass
class _Run:
    """One compaction's fixed inputs, shared by every call it makes."""

    settings: SessionCompactionSettings
    prompts: SummaryPrompts
    language: str
    focus: str
    observability: LLMObservabilityContext | None
    progress: Callable[[CompactionProgress], Awaitable[None]] | None
    cancel: asyncio.Event | None
    sleep: Callable[[float], Awaitable[None]]
    parts: int = 0
    merge_rounds: int = 0
    section_problem: str = ""
    done: int = field(default=0)

    async def report(self, update: CompactionProgress) -> None:
        if self.progress is not None:
            await self.progress(update)

    def text(self, template: str, **fields: object) -> str:
        values = _Missing({key: str(value) for key, value in fields.items()})
        values.setdefault("language", self.language)
        values.setdefault("sections", self.prompts.rules())
        return template.format_map(values)

    async def request(self, summariser: Summariser, request: LLMRequest) -> LLMResponse:
        """One summariser call, asked again when it failed in a way that passes.

        A compaction once died on a single 5xx and the session went on growing
        to nine tenths of its window. A transient failure is asked again after
        each pause; a rate limit is a quota, not a hiccup, and goes back to the
        ladder at once, as does a refusal. A stall gets one more try at twice
        the deadline, then goes back too.
        """
        timeout = self.settings.call_timeout_seconds
        pauses: tuple[float | None, ...] = (*self.settings.retry_pauses, None)
        for pause in pauses:
            try:
                try:
                    return await _guard(lambda: asyncio.wait_for(summariser.provider.complete_text(request), timeout=timeout), self.cancel)
                except TimeoutError:
                    _logger.warning("session summariser %s gave no answer within %.0fs; one more try with %.0fs", summariser.name, timeout, 2 * timeout)
                    return await _guard(lambda: asyncio.wait_for(summariser.provider.complete_text(request), timeout=2 * timeout), self.cancel)
            except LLMError as exc:
                verdict = provider_failure_of(exc)
                rate_limited = isinstance(exc, LLMRateLimitError) or (
                    verdict is not None and verdict.reason is ProviderFailureReason.rate_limit
                )
                retryable = verdict.retryable if verdict is not None else bool(exc.retryable)
                if pause is None or rate_limited or not retryable:
                    raise
                _logger.warning("session summariser %s failed (%s); asking again in %.0fs", summariser.name, exc, pause)
                await self.sleep(pause)
        raise AssertionError("unreachable")  # pragma: no cover - the last pause is None

    async def call(self, summariser: Summariser, prompt: str, body: str, *, words: int, strict: bool) -> str:
        from protocore.runtime.query import build_llm_request

        candidate = ""
        problem = ""
        for _attempt in range(2 if strict and self.prompts.sections else 1):
            reminder = self.text(self.prompts.retry, problem=problem) if problem else ""
            request = build_llm_request(
                model=summariser.model,
                messages=[Message(role=MessageRole.user, content_blocks=[TextBlock(text=f"{prompt}{reminder}\n\n{body}")])],
                max_tokens=self.settings.output_tokens_for(words),
                temperature=self.settings.temperature,
                # A summary is written, not reasoned towards: a thinking model
                # spends the output cap on reasoning and returns it cut short.
                thinking_enabled=False,
                reasoning_effort="low",
                observability=self.observability,
            )
            response = await self.request(summariser, request)
            candidate = response.message.text.strip()
            if not candidate:
                raise SummaryEmptyError(f"{summariser.name} returned an empty summary")
            problem = validate_sections(candidate, self.prompts.sections) if strict else ""
            if not problem:
                return candidate
            _logger.warning("session summary from %s rejected: %s", summariser.name, problem)
        self.section_problem = problem
        return candidate

    async def summarise(self, summariser: Summariser, rendered: SummaryInput, words: int) -> str:
        settings = self.settings
        base = rendered.previous
        has_base = bool(base)
        base_rule = self.prompts.base_rule + "\n" if has_base else ""
        limit = settings.chunk_tokens * CHARS_PER_TOKEN
        parts = split_transcript(rendered.transcript, limit)
        self.parts = len(parts)
        self.merge_rounds = 0
        self.section_problem = ""
        await self.report(CompactionProgress("summarising", parts_total=len(parts), summariser=summariser.name))
        if len(parts) == 1:
            prompt = self.text(self.prompts.single, max_words=words, base_rule=base_rule) + self.focus
            return await self.call(summariser, prompt, _with_base(base, rendered.transcript), words=words, strict=True)
        part_words = max(400, words // 2)
        gate = asyncio.Semaphore(max(1, settings.parallel_calls))
        self.done = 0

        async def note(prompt: str, body: str) -> str:
            async with gate:
                return await self.call(summariser, prompt + self.focus, body, words=part_words, strict=False)

        async def part_note(index: int, part: str) -> str:
            text = await note(self.text(self.prompts.part, index=index + 1, total=len(parts), max_words=part_words), part)
            self.done += 1
            await self.report(CompactionProgress("summarising", parts_total=len(parts), parts_done=self.done, summariser=summariser.name))
            return text

        notes = [text for text in await asyncio.gather(*(part_note(i, part) for i, part in enumerate(parts))) if text]
        for _ in range(settings.merge_rounds_max):
            # Thirty parts leave notes that alone exceed one call; they are
            # merged in groups first, so the last merge reads all of them
            # rather than the newest.
            if len(_joined(notes)) <= limit or len(notes) < 2:
                break
            groups = _pack(notes, limit)
            self.merge_rounds += 1
            await self.report(CompactionProgress("merging", parts_total=len(parts), parts_done=len(parts), summariser=summariser.name))
            prompt = self.text(self.prompts.merge_notes, max_words=part_words)

            async def merged(group: list[str], merge_prompt: str = prompt) -> str:
                return await note(merge_prompt, _joined(group)) if len(group) > 1 else group[0]

            notes = list(await asyncio.gather(*(merged(group) for group in groups)))
        await self.report(CompactionProgress("merging", parts_total=len(parts), parts_done=len(parts), summariser=summariser.name))
        prompt = self.text(self.prompts.merge, max_words=words, base_rule=base_rule) + self.focus
        return await self.call(summariser, prompt, _with_base(base, _joined(notes)), words=words, strict=True)


async def compact_session(
    history: Sequence[Message],
    *,
    summarisers: Sequence[Summariser],
    keep_recent_messages: int = 0,
    keep_recent_tokens: int = 0,
    instructions: str = "",
    language: str = "the language of the conversation",
    window: int = 0,
    prompts: SummaryPrompts | None = None,
    settings: SessionCompactionSettings | None = None,
    render: Callable[[Sequence[Message]], SummaryInput] | None = None,
    decorate: Callable[[str, Sequence[Message]], str] | None = None,
    observability: LLMObservabilityContext | None = None,
    progress: Callable[[CompactionProgress], Awaitable[None]] | None = None,
    cancel: asyncio.Event | None = None,
    reason: str = "session",
    rc: LoopConstants | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SessionCompaction:
    """Summarise ``history`` (all but its kept tail) and return what should replace it.

    Pure with respect to storage and to ``history``: nothing is written and no
    lock is taken, so a caller runs it without holding its session and commits
    the result afterwards (:func:`commit_session_compaction`,
    :meth:`SessionCompaction.apply`). ``summarisers`` are asked in order; the
    first that produces a summary wins. ``window`` scales the summary's word
    budget; ``instructions`` is what the user asked the summary to focus on.
    ``render`` turns the summarised prefix into text (default
    :func:`render_for_summary`), and ``decorate`` may add to the summary text
    by code before it is wrapped — quotes, an index, a pointer to an archive —
    because what code writes does not decay through the next summary.

    Raises :class:`NothingToCompact` when there is nothing to summarise,
    :class:`SessionCompactionFailed` when no summariser produced a summary, and
    :class:`SessionCompactionCancelled` when ``cancel`` was set first.
    """
    if not summarisers:
        raise ValueError("compact_session needs at least one summariser")
    settings = settings or SessionCompactionSettings()
    prompts = prompts or SummaryPrompts()
    cut = session_compaction_cut(
        history, keep_recent_messages=keep_recent_messages, keep_recent_tokens=keep_recent_tokens, rc=rc
    )
    prefix = tuple(history[:cut])
    tail = tuple(history[cut:])
    if not prefix:
        raise NothingToCompact("nothing to compact: the history is empty or entirely inside the kept tail")
    rendered = (render or (lambda messages: render_for_summary(messages, settings=settings)))(prefix)
    if not rendered.transcript.strip():
        raise NothingToCompact("nothing to compact: the summarised part holds no turns besides earlier summaries")
    focus = prompts.focus.format_map(_Missing(instructions=instructions.strip())) if instructions.strip() else ""
    run = _Run(
        settings=settings,
        prompts=prompts,
        language=language,
        focus=focus,
        observability=observability,
        progress=progress,
        cancel=cancel,
        sleep=sleep,
    )
    words = settings.words_for(window)
    failures: list[str] = []
    text = ""
    used: Summariser | None = None
    for summariser in summarisers:
        if cancel is not None and cancel.is_set():
            raise SessionCompactionCancelled("cancelled before the summary")
        try:
            text = await run.summarise(summariser, rendered, words)
        except (LLMError, TimeoutError, SummaryEmptyError) as exc:
            detail = str(exc) or type(exc).__name__
            failures.append(f"{summariser.name}: {detail}"[:400])
            _logger.warning("session compaction with %s failed (%s)", summariser.name, detail)
            continue
        used = summariser
        break
    if used is None:
        raise SessionCompactionFailed(failures)
    await run.report(CompactionProgress("writing", parts_total=run.parts, parts_done=run.parts, summariser=used.name))
    if decorate is not None:
        text = decorate(text, prefix)
    if cancel is not None and cancel.is_set():
        raise SessionCompactionCancelled("cancelled before the summary was written")
    from protocore.runtime.context.compaction import _wrap_compaction_summary

    passages = _operator_passages(prefix, settings.operator_passages_kept)
    summary = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text=_wrap_compaction_summary(reason, text, attributes=f"messages='{len(prefix)}' kind='session'"))],
        metadata={
            COMPACTION_SUMMARY_METADATA_KEY: True,
            # The user's passages the summary stood for, named so the in-run
            # ledger and the next compaction quote them rather than read the
            # summary as a model's words.
            OPERATOR_WORDS_METADATA_KEY: passages or False,
            SESSION_COMPACTION_METADATA_KEY: {
                "reason": reason,
                "messages": len(prefix),
                "kept": len(tail),
                "summariser": used.name,
            },
        },
    )
    ledgers = [message for message in prefix if is_ledger(message)]
    return SessionCompaction(
        summary=summary,
        summary_text=text,
        source=prefix,
        carried=tuple(ledgers[-1:]),
        tail=tail,
        summariser=used.name,
        failures=tuple(failures),
        parts=run.parts,
        merge_rounds=run.merge_rounds,
        section_problem=run.section_problem,
        reason=reason,
    )


async def commit_session_compaction(engine: QueryEngine, compaction: SessionCompaction) -> TurnEvent | None:
    """Put ``compaction`` into ``engine``'s history if it still applies; ``None`` when it does not.

    It applies while no turn is driving and the history still starts with the
    messages it summarised, by identity. Turns appended since are kept after
    the summary. The check and the replacement happen without an await between
    them, so nothing on the event loop can change the history in between.
    Committed through :meth:`~protocore.runtime.query_engine.QueryEngine.replace_history`,
    which retires the old history's derived state and emits
    ``history_replaced`` with the reason ``session_compaction:<reason>``.
    """
    if engine._current_turn_task is not None or not compaction.applies_to(engine.history):
        return None
    return await engine.replace_history(compaction.apply(engine.history), reason=f"session_compaction:{compaction.reason}")


# ---------------------------------------------------------------------------
# A ladder for the passes inside a run
# ---------------------------------------------------------------------------


class SummariserLadder:
    """A provider that summarises with the first rung that answers, then with the run's own provider.

    Built by :class:`~protocore.runtime.query_engine.QueryEngine` from
    ``compaction_providers``, so the in-run tiers use a model configured for
    summaries and fall back when it refuses, is out of quota or stalls; with a
    single ``compaction_provider`` there was no fallback at all. Each rung is
    asked for its own model; the fallback, the run's own provider, gets the
    request as built, with the run's model. Only :meth:`complete_text` is a
    summariser call; the rest are passed to the run's own provider.
    """

    def __init__(
        self,
        rungs: Sequence[Summariser],
        *,
        fallback: Callable[[], ILLMProvider],
        per_call_timeout_seconds: float | None = None,
    ) -> None:
        self.rungs: tuple[Summariser, ...] = tuple(rungs)
        self._fallback = fallback
        self.per_call_timeout_seconds = per_call_timeout_seconds

    def deadline_for(self, per_call_seconds: float) -> float:
        """The deadline for the whole ladder when each call has ``per_call_seconds``."""
        return per_call_seconds * (len(self.rungs) + 1)

    async def _ask(self, provider: ILLMProvider, request: LLMRequest) -> LLMResponse:
        if self.per_call_timeout_seconds is None:
            return await provider.complete_text(request)
        return await asyncio.wait_for(provider.complete_text(request), timeout=self.per_call_timeout_seconds)

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        for rung in self.rungs:
            try:
                return await self._ask(rung.provider, request.model_copy(update={"model": rung.model}))
            except (LLMError, TimeoutError) as exc:
                _logger.warning("summariser %s failed (%s); trying the next", rung.name, exc or type(exc).__name__)
        return await self._ask(self._fallback(), request)

    def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent | ProviderDelta]:
        return self._fallback().stream_with_tools(request)

    async def complete_structured(self, request: LLMRequest, response_schema: dict[str, Any]) -> LLMResponse:
        return await self._fallback().complete_structured(request, response_schema)

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return self._fallback().count_tokens(text, model)


def _weak_fallback(engine: QueryEngine) -> Callable[[], ILLMProvider]:
    """The engine's current provider, without the ladder keeping the engine alive."""
    ref = weakref.ref(engine)

    def current() -> ILLMProvider:
        live = ref()
        if live is None:  # pragma: no cover - the engine owns the ladder
            raise RuntimeError("the engine that owned this summariser ladder is gone")
        return live.llm

    return current


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_SECTIONS",
    "SESSION_COMPACTION_METADATA_KEY",
    "CompactionHold",
    "CompactionHoldPolicy",
    "CompactionProgress",
    "NothingToCompact",
    "SessionCompaction",
    "SessionCompactionCancelled",
    "SessionCompactionFailed",
    "SessionCompactionSettings",
    "Summariser",
    "SummariserLadder",
    "SummaryEmptyError",
    "SummaryInput",
    "SummaryPrompts",
    "commit_session_compaction",
    "compact_session",
    "leading_summaries",
    "render_for_summary",
    "session_compaction_cut",
    "split_transcript",
    "validate_sections",
]
