"""Compacting a session between runs: one summary in place of the history, made safely.

A long-lived session needs "compact this now" outside a run — after a run, before
one whose history no longer fits, on demand — and every such host built the
same service: the previous summary as the base to update, a history longer
than one call summarised in parts and merged, the sections checked, a ladder of
summarisers, retries that do not hammer a quota, a hold after a pass that
failed or freed nothing, and a commit that cannot overwrite turns that arrived
while the summary was being written.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    ProviderFailure,
    ProviderFailureReason,
)
from protocore.contracts.types import (
    COMPACTION_LEDGER_METADATA_KEY,
    COMPACTION_SUMMARY_METADATA_KEY,
    OPERATOR_WORDS_METADATA_KEY,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    Message,
    MessageOrigin,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.session_compaction import (
    CompactionHold,
    CompactionHoldPolicy,
    CompactionProgress,
    NothingToCompact,
    SessionCompactionCancelled,
    SessionCompactionFailed,
    SessionCompactionSettings,
    Summariser,
    SummariserLadder,
    SummaryPrompts,
    commit_session_compaction,
    compact_session,
    render_for_summary,
    session_compaction_cut,
    validate_sections,
)
from protocore.runtime.events import EventType

SECTIONS = ("Facts", "Open")
GOOD = "## Facts\n- the port is 8080\n## Open\n- nothing"


def _prompts() -> SummaryPrompts:
    return SummaryPrompts(sections=SECTIONS)


Reply = Callable[[LLMRequest], "str | BaseException"]


class _Summariser:
    """A provider whose ``complete_text`` answers from a script, and records what it was asked."""

    def __init__(self, reply: Reply | str = GOOD, *, delay: float = 0.0) -> None:
        self._reply = reply
        self.delay = delay
        self.requests: list[LLMRequest] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            answer = self._reply if isinstance(self._reply, str) else self._reply(request)
            if isinstance(answer, BaseException):
                raise answer
            return LLMResponse(
                message=Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=answer)]),
                stop_reason=StopReason.end_turn,
            )
        finally:
            self.in_flight -= 1

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        raise AssertionError("not used")
        yield  # pragma: no cover

    async def complete_structured(self, request: LLMRequest, response_schema: dict[str, Any]) -> LLMResponse:
        raise AssertionError("not used")

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def body(request: LLMRequest) -> str:
        return request.messages[-1].text


def _user(text: str, **metadata: Any) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)], metadata=metadata)


def _assistant(text: str) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])


def _exchange(n: int, size: int = 40) -> list[Message]:
    call = f"call-{n}"
    return [
        _user(f"task {n} " + "u" * size),
        Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text=f"looking {n}"), ToolUseBlock(tool_call_id=call, name="Read", arguments_json='{"path": "a.py"}')],
        ),
        Message(role=MessageRole.tool, content_blocks=[ToolResultBlock(tool_call_id=call, content=f"result {n} " + "r" * size)]),
        _assistant(f"answer {n} " + "a" * size),
    ]


def _history(turns: int, size: int = 40) -> list[Message]:
    out: list[Message] = []
    for n in range(turns):
        out.extend(_exchange(n, size))
    return out


async def _no_sleep(_seconds: float) -> None:
    return None


# ---------------------------------------------------------------------------
# the summary itself
# ---------------------------------------------------------------------------


async def test_a_short_history_is_summarised_in_one_call() -> None:
    provider = _Summariser()
    history = _history(3)
    result = await compact_session(history, summarisers=[Summariser(provider, "small")], keep_recent_messages=4, prompts=_prompts())

    assert len(provider.requests) == 1
    assert provider.requests[0].model == "small"
    summary = result.history[0]
    assert summary.origin is MessageOrigin.compaction
    assert "the port is 8080" in summary.text
    # The kept tail is the very same objects, from a turn boundary on.
    assert result.history[1:] == tuple(history[-4:])
    assert all(a is b for a, b in zip(result.history[1:], history[-4:], strict=True))
    assert result.replaced == 8 and result.kept == 4
    assert result.summariser == "small"
    assert not result.section_problem
    assert "task 0" in provider.body(provider.requests[0])
    assert "task 2" not in provider.body(provider.requests[0]), "the kept tail is not summarised"


async def test_the_previous_summary_is_the_base_to_update() -> None:
    provider = _Summariser()
    first = await compact_session(_history(3), summarisers=[Summariser(provider, "m")], prompts=_prompts())
    grown = [*first.history, *_history(2)]
    await compact_session(grown, summarisers=[Summariser(provider, "m")], prompts=_prompts())

    body = provider.body(provider.requests[-1])
    assert "<previous-summary>" in body
    assert "the port is 8080" in body.split("</previous-summary>")[0]
    transcript = body.split("</previous-summary>")[1]
    assert "the port is 8080" not in transcript, "the base is not summarised again as transcript"
    assert _prompts().base_rule[:40] in body, "the summariser is told what the base is"


async def test_a_long_history_is_summarised_in_parts_merged_in_rounds() -> None:
    def reply(request: LLMRequest) -> str:
        text = request.messages[-1].text
        if "PART" in text.split("\n", 1)[0]:
            return "## Facts\n- " + "n" * 900 + "\n## Open\n- x"
        return GOOD

    provider = _Summariser(reply, delay=0.01)
    stages: list[str] = []

    async def progress(update: CompactionProgress) -> None:
        stages.append(update.stage)

    prompts = SummaryPrompts(sections=SECTIONS, part="PART {index} of {total}. {sections}")
    settings = SessionCompactionSettings(chunk_tokens=300, parallel_calls=3)
    result = await compact_session(
        _history(30, size=200), summarisers=[Summariser(provider, "m")], prompts=prompts, settings=settings, progress=progress
    )

    assert result.parts > 3
    assert result.merge_rounds >= 1
    assert provider.max_in_flight <= 3
    assert stages[0] == "summarising" and "merging" in stages and stages[-1] == "writing"
    assert "the port is 8080" in result.summary_text


async def test_sections_are_checked_once_more_then_accepted_with_a_warning() -> None:
    provider = _Summariser("## Facts\n- only facts")
    result = await compact_session(_history(2), summarisers=[Summariser(provider, "m")], prompts=_prompts())

    assert len(provider.requests) == 2
    assert "## Open" in provider.requests[1].messages[-1].text and "rejected" in provider.requests[1].messages[-1].text
    assert result.section_problem
    assert "only facts" in result.summary_text


def test_validate_sections() -> None:
    assert validate_sections(GOOD, SECTIONS) == ""
    assert "appears 0 times" in validate_sections("## Facts\n", SECTIONS)
    assert "appears 2 times" in validate_sections("## Facts\n## Facts\n## Open\n", SECTIONS)
    assert "order" in validate_sections("## Open\n## Facts\n", SECTIONS)
    assert validate_sections("anything", ()) == ""


async def test_scaffolding_and_the_ledger_stay_out_of_the_transcript() -> None:
    ledger = _user("LEDGER BODY", **{COMPACTION_LEDGER_METADATA_KEY: {"v": 1}})
    nudge = _user("NUDGE BODY", **{SYNTHETIC_RECOVERY_METADATA_KEY: "terminal_tool_nudge"})
    history = [*_exchange(0), ledger, nudge, *_exchange(1)]
    rendered = render_for_summary(history)
    assert "LEDGER BODY" not in rendered.transcript
    assert "NUDGE BODY" not in rendered.transcript
    assert "[tool call] Read" in rendered.transcript
    assert "[tool result]" in rendered.transcript
    assert rendered.previous == ""


async def test_the_ledger_is_carried_past_the_summariser_verbatim() -> None:
    ledger = _user("LEDGER BODY", **{COMPACTION_LEDGER_METADATA_KEY: {"v": 1}})
    history = [*_exchange(0), ledger, *_exchange(1), *_exchange(2)]
    result = await compact_session(history, summarisers=[Summariser(_Summariser(), "m")], keep_recent_messages=4, prompts=_prompts())
    assert result.history[1] is ledger
    assert result.history[2:] == tuple(history[-4:])


async def test_the_operator_words_and_a_decoration_are_kept() -> None:
    history = _history(2)

    def decorate(text: str, summarised: Any) -> str:
        return text + f"\n\n## Quoted\n{len(summarised)} messages"

    result = await compact_session(history, summarisers=[Summariser(_Summariser(), "m")], prompts=_prompts(), decorate=decorate)
    assert "8 messages" in result.summary_text
    words = result.summary.metadata[OPERATOR_WORDS_METADATA_KEY]
    assert any(w.startswith("task 0") for w in words)


# ---------------------------------------------------------------------------
# the ladder and the retries
# ---------------------------------------------------------------------------


def _failure(reason: ProviderFailureReason, retryable: bool) -> LLMProviderError:
    return LLMProviderError(f"refused: {reason.value}", classified=ProviderFailure(reason=reason, retryable=retryable))


async def test_a_summariser_out_of_balance_hands_over_to_the_next() -> None:
    broke = _Summariser(lambda r: _failure(ProviderFailureReason.billing, False))
    ready = _Summariser()
    result = await compact_session(
        _history(2), summarisers=[Summariser(broke, "cheap", label="summary model"), Summariser(ready, "own")], prompts=_prompts(), sleep=_no_sleep
    )
    assert len(broke.requests) == 1, "a permanent refusal is not asked again"
    assert result.summariser == "own"
    assert result.failures and "summary model" in result.failures[0]


async def test_a_rate_limit_is_not_retried_in_place() -> None:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    limited = _Summariser(lambda r: LLMRateLimitError("429"))
    result = await compact_session(
        _history(2), summarisers=[Summariser(limited, "a"), Summariser(_Summariser(), "b")], prompts=_prompts(), sleep=sleep
    )
    assert len(limited.requests) == 1 and not slept
    assert result.summariser == "b"


async def test_a_transient_failure_is_asked_again_after_a_pause() -> None:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    calls = {"n": 0}

    def flaky(request: LLMRequest) -> str | BaseException:
        calls["n"] += 1
        return _failure(ProviderFailureReason.server_error, True) if calls["n"] == 1 else GOOD

    provider = _Summariser(flaky)
    result = await compact_session(_history(2), summarisers=[Summariser(provider, "a")], prompts=_prompts(), sleep=sleep)
    assert slept == [5.0]
    assert result.summariser == "a" and not result.failures


async def test_a_stalled_call_is_tried_once_more_with_twice_the_time() -> None:
    calls = {"n": 0}

    class _Stalls(_Summariser):
        async def complete_text(self, request: LLMRequest) -> LLMResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                await asyncio.sleep(1.0)
            else:
                await asyncio.sleep(0.06)  # longer than one timeout, shorter than two
            return await super().complete_text(request)

    settings = SessionCompactionSettings(call_timeout_seconds=0.05)
    result = await compact_session(_history(2), summarisers=[Summariser(_Stalls(), "a")], prompts=_prompts(), settings=settings)
    assert calls["n"] == 2
    assert result.summariser == "a"


async def test_when_every_summariser_fails_the_history_is_left_alone() -> None:
    history = _history(2)
    snapshot = list(history)
    with pytest.raises(SessionCompactionFailed) as raised:
        await compact_session(
            history,
            summarisers=[Summariser(_Summariser(lambda r: _failure(ProviderFailureReason.auth, False)), "a"), Summariser(_Summariser(""), "b")],
            prompts=_prompts(),
            sleep=_no_sleep,
        )
    assert len(raised.value.failures) == 2
    assert "empty" in raised.value.failures[1]
    assert history == snapshot


async def test_no_summariser_is_a_refusal() -> None:
    with pytest.raises(ValueError):
        await compact_session(_history(2), summarisers=[], prompts=_prompts())


async def test_nothing_to_compact() -> None:
    with pytest.raises(NothingToCompact):
        await compact_session(_history(1), summarisers=[Summariser(_Summariser(), "m")], keep_recent_messages=10, prompts=_prompts())
    with pytest.raises(NothingToCompact):
        await compact_session([], summarisers=[Summariser(_Summariser(), "m")], prompts=_prompts())


# ---------------------------------------------------------------------------
# the cut, cancellation and the commit
# ---------------------------------------------------------------------------


def test_the_cut_never_splits_a_tool_exchange() -> None:
    history = _history(3)
    for keep in range(1, len(history)):
        cut = session_compaction_cut(history, keep_recent_messages=keep)
        assert cut <= len(history) - keep
        if 0 < cut < len(history):
            first = history[cut]
            assert first.role is MessageRole.user and not any(isinstance(b, ToolResultBlock) for b in first.content_blocks)
    assert session_compaction_cut(history) == len(history)


def test_the_cut_by_tokens_keeps_at_least_the_budget() -> None:
    history = _history(6, size=400)
    cut = session_compaction_cut(history, keep_recent_tokens=300)
    assert 0 < cut < len(history)
    assert history[cut].role is MessageRole.user
    nudged = [*history[:8], _user("NUDGE", **{SYNTHETIC_RECOVERY_METADATA_KEY: "terminal_tool_nudge"}), *history[8:]]
    assert nudged[session_compaction_cut(nudged, keep_recent_messages=len(nudged) - 8)].origin is MessageOrigin.conversation


async def test_cancellation_mid_way_leaves_nothing_behind() -> None:
    cancel = asyncio.Event()

    def reply(request: LLMRequest) -> str:
        cancel.set()
        return GOOD

    provider = _Summariser(reply, delay=0.05)
    history = _history(30, size=200)
    snapshot = list(history)
    with pytest.raises(SessionCompactionCancelled):
        await compact_session(
            history,
            summarisers=[Summariser(provider, "m")],
            prompts=_prompts(),
            settings=SessionCompactionSettings(chunk_tokens=300, parallel_calls=2),
            cancel=cancel,
        )
    assert history == snapshot


async def test_a_cancel_set_before_the_start_makes_no_call() -> None:
    cancel = asyncio.Event()
    cancel.set()
    provider = _Summariser()
    with pytest.raises(SessionCompactionCancelled):
        await compact_session(_history(2), summarisers=[Summariser(provider, "m")], prompts=_prompts(), cancel=cancel)
    assert not provider.requests


async def test_the_commit_replaces_the_engines_history(engine_factory: Any) -> None:
    engine = engine_factory()
    engine.history = _history(3)
    result = await compact_session(engine.history, summarisers=[Summariser(_Summariser(), "m")], keep_recent_messages=4, prompts=_prompts())

    event = await commit_session_compaction(engine, result)
    assert event is not None and event.type is EventType.HISTORY_REPLACED
    assert event.payload["reason"] == "session_compaction:session"
    assert list(engine.history) == list(result.history)


async def test_turns_appended_while_summarising_are_kept(engine_factory: Any) -> None:
    engine = engine_factory()
    engine.history = _history(3)
    result = await compact_session(list(engine.history), summarisers=[Summariser(_Summariser(), "m")], keep_recent_messages=4, prompts=_prompts())
    late = _user("a steer that arrived meanwhile")
    engine.history.append(late)

    assert await commit_session_compaction(engine, result) is not None
    assert engine.history[-1] is late
    assert engine.history[0] is result.summary


async def test_a_summarised_prefix_that_changed_is_not_committed(engine_factory: Any) -> None:
    engine = engine_factory()
    engine.history = _history(3)
    result = await compact_session(list(engine.history), summarisers=[Summariser(_Summariser(), "m")], keep_recent_messages=4, prompts=_prompts())
    engine.history[1] = _assistant("rewritten by someone else")
    before = list(engine.history)

    assert await commit_session_compaction(engine, result) is None
    assert engine.history == before
    assert not result.applies_to(engine.history)
    with pytest.raises(ValueError):
        result.apply(engine.history)


async def test_no_commit_while_a_turn_is_driving(engine_factory: Any) -> None:
    engine = engine_factory()
    engine.history = _history(3)
    result = await compact_session(list(engine.history), summarisers=[Summariser(_Summariser(), "m")], prompts=_prompts())
    async with engine.driving_turn():
        assert await commit_session_compaction(engine, result) is None
    assert len(engine.history) == 12


# ---------------------------------------------------------------------------
# the hold after a pass that failed or freed little
# ---------------------------------------------------------------------------


def test_a_failure_holds_for_a_doubling_while_capped() -> None:
    policy = CompactionHoldPolicy(retry_seconds=60.0, retry_max_seconds=200.0, growth_ratio=0.1)
    first = policy.after_failure(None, prompt_tokens=1_000, now=0.0)
    assert first.failures == 1 and first.until == 60.0 and first.reason == "failed"
    second = policy.after_failure(first, prompt_tokens=1_000, now=100.0)
    assert second.until == 220.0
    third = policy.after_failure(second, prompt_tokens=1_000, now=300.0)
    assert third.until == 500.0, "the wait is capped"

    assert policy.holds(first, prompt_tokens=1_050, window=1_000, now=30.0)
    assert not policy.holds(first, prompt_tokens=1_050, window=1_000, now=61.0), "time passed"
    assert not policy.holds(first, prompt_tokens=1_100, window=1_000, now=30.0), "the history grew by a tenth of the window"
    assert not policy.holds(None, prompt_tokens=1, window=1, now=0.0)


def test_a_pass_that_freed_little_holds_until_the_history_grows() -> None:
    policy = CompactionHoldPolicy()
    held = policy.after_pass(None, prompt_before=10_000, prompt_after=9_500, trigger_tokens=8_000)
    assert held is not None and held.reason == "freed_little" and held.until is None
    assert policy.holds(held, prompt_tokens=9_600, window=10_000, now=1e12)
    assert not policy.holds(held, prompt_tokens=10_600, window=10_000, now=0.0)
    still_over = policy.after_pass(None, prompt_before=10_000, prompt_after=8_500, trigger_tokens=8_000)
    assert still_over is not None, "what is left is still over the trigger"
    assert policy.after_pass(held, prompt_before=10_000, prompt_after=4_000, trigger_tokens=8_000) is None


def test_a_hold_round_trips_through_a_dict() -> None:
    hold = CompactionHold(prompt_tokens=5, failures=2, until=12.5, reason="failed")
    assert CompactionHold.from_dict(hold.to_dict()) == hold
    assert CompactionHold.from_dict(CompactionHold(1, 1, None, "freed_little").to_dict()).until is None
    assert CompactionHold.from_dict({"garbage": True}) is None
    assert CompactionHold.from_dict(None) is None


# ---------------------------------------------------------------------------
# a ladder for the passes inside a run
# ---------------------------------------------------------------------------


def _request(model: str = "session-model") -> LLMRequest:
    return LLMRequest(model=model, messages=[_user("summarise")], max_tokens=100)


async def test_the_ladder_falls_through_to_the_runs_own_model() -> None:
    down = _Summariser(lambda r: _failure(ProviderFailureReason.billing, False))
    also_down = _Summariser(lambda r: LLMRateLimitError("429"))
    own = _Summariser()
    ladder = SummariserLadder([Summariser(down, "cheap"), Summariser(also_down, "cheaper")], fallback=lambda: own)

    response = await ladder.complete_text(_request())
    assert "8080" in response.message.text
    assert down.requests[0].model == "cheap"
    assert also_down.requests[0].model == "cheaper"
    assert own.requests[0].model == "session-model", "the run's own provider gets the run's own model"
    assert ladder.deadline_for(10.0) == 30.0


async def test_the_ladder_answers_from_the_first_rung_that_works() -> None:
    first = _Summariser()
    own = _Summariser()
    ladder = SummariserLadder([Summariser(first, "cheap")], fallback=lambda: own, per_call_timeout_seconds=1.0)
    await ladder.complete_text(_request())
    assert len(first.requests) == 1 and not own.requests


async def test_the_ladder_moves_past_a_stalled_rung() -> None:
    stalled = _Summariser(delay=1.0)
    own = _Summariser()
    ladder = SummariserLadder([Summariser(stalled, "slow")], fallback=lambda: own, per_call_timeout_seconds=0.05)
    await ladder.complete_text(_request())
    assert len(own.requests) == 1


async def test_the_ladder_raises_the_last_failure() -> None:
    own = _Summariser(lambda r: _failure(ProviderFailureReason.server_error, True))
    ladder = SummariserLadder([], fallback=lambda: own)
    with pytest.raises(LLMProviderError):
        await ladder.complete_text(_request())


async def test_the_engine_builds_a_ladder_from_compaction_providers(engine_factory: Any, in_memory_runtime: dict[str, Any]) -> None:
    from protocore.runtime.query_engine import QueryEngine

    base = engine_factory()
    cheap = _Summariser()
    engine = QueryEngine(
        config=base.config,
        llm_provider=in_memory_runtime["llm"],
        tool_registry=in_memory_runtime["tools"],
        event_stream=in_memory_runtime["events"],
        hook_manager=in_memory_runtime["hooks"],
        skill_store=in_memory_runtime["skills"],
        blob_store=in_memory_runtime["blobs"],
        compaction_providers=[Summariser(cheap, "cheap")],
    )
    assert isinstance(engine.compaction_llm, SummariserLadder)
    await engine.compaction_llm.complete_text(_request())
    assert cheap.requests[0].model == "cheap"
    # A re-armed engine keeps the same ladder.
    engine.state = engine.state.__class__.COMPLETED
    engine.rearm()
    assert isinstance(engine.compaction_llm, SummariserLadder)

    with pytest.raises(ValueError):
        QueryEngine(
            config=base.config,
            llm_provider=in_memory_runtime["llm"],
            tool_registry=in_memory_runtime["tools"],
            event_stream=in_memory_runtime["events"],
            hook_manager=in_memory_runtime["hooks"],
            skill_store=in_memory_runtime["skills"],
            blob_store=in_memory_runtime["blobs"],
            compaction_provider=cheap,
            compaction_providers=[Summariser(cheap, "cheap")],
        )


# ---------------------------------------------------------------------------
# the details of the rendering and the settings
# ---------------------------------------------------------------------------


def test_the_rendering_reads_each_kind_of_message_in_its_place() -> None:
    from protocore.contracts.types import ThinkingBlock

    earlier = _user("<compacted-turn id='a'>EARLIER SUMMARY</compacted-turn>", **{COMPACTION_SUMMARY_METADATA_KEY: True})
    middle = _user("<compacted-turn id='b'>MID SUMMARY</compacted-turn>", **{COMPACTION_SUMMARY_METADATA_KEY: True})
    wake = _user("background tasks finished: build ok", **{SYNTHETIC_RECOVERY_METADATA_KEY: "background_tasks_finished"})
    history = [
        earlier,
        Message(role=MessageRole.system, content_blocks=[TextBlock(text="SYSTEM")]),
        _user("do it"),
        Message(role=MessageRole.assistant, content_blocks=[ThinkingBlock(text="HIDDEN"), TextBlock(text="   "), TextBlock(text="z" * 50)]),
        middle,
        wake,
        Message(role=MessageRole.tool, content_blocks=[ToolResultBlock(tool_call_id="c", content="boom", is_error=True)]),
    ]
    rendered = render_for_summary(history, settings=SessionCompactionSettings(assistant_chars=20))
    assert rendered.previous == "EARLIER SUMMARY"
    text = rendered.transcript
    assert "SYSTEM" not in text and "HIDDEN" not in text
    assert "[summary of earlier turns] MID SUMMARY" in text
    assert "[note] background tasks finished: build ok" in text, "a wake-up is a note, not the user's turn"
    assert "chars left out" in text
    assert "[tool result ERROR] boom" in text


async def test_a_history_of_summaries_alone_is_nothing_to_compact() -> None:
    only = [_user("<compacted-turn id='a'>S</compacted-turn>", **{COMPACTION_SUMMARY_METADATA_KEY: True})]
    with pytest.raises(NothingToCompact):
        await compact_session(only, summarisers=[Summariser(_Summariser(), "m")], prompts=_prompts())


def test_the_word_budget_grows_with_the_window_and_stops_at_the_cap() -> None:
    settings = SessionCompactionSettings(max_words=1000, words_cap=5000)
    assert settings.words_for(0) == 1000
    assert settings.words_for(50_000) == 1000
    assert settings.words_for(400_000) == 4000
    assert settings.words_for(10_000_000) == 5000
    assert settings.output_tokens_for(100) == 6000
    assert settings.output_tokens_for(20_000) == 32_000


def test_a_line_longer_than_a_part_is_split_inside() -> None:
    from protocore.runtime.context.session_compaction import split_transcript

    parts = split_transcript("a\n" + "b" * 25 + "\nc", 10)
    assert all(len(part) <= 10 for part in parts)
    assert "".join(parts).replace("\n", "") == "a" + "b" * 25 + "c"
    assert split_transcript("short", 10) == ["short"]


def test_a_host_text_with_its_own_placeholders_still_formats() -> None:
    prompts = SummaryPrompts(sections=("A",), section_rules="sections {headings} and {unknown}")
    assert prompts.rules() == "sections ## A and {unknown}"


def test_the_token_cut_keeps_everything_when_the_budget_covers_it() -> None:
    history = _history(2)
    assert session_compaction_cut(history, keep_recent_tokens=10_000_000) == 0


async def test_a_cancel_after_the_summary_drops_it() -> None:
    cancel = asyncio.Event()

    def decorate(text: str, summarised: Any) -> str:
        cancel.set()
        return text

    with pytest.raises(SessionCompactionCancelled):
        await compact_session(_history(2), summarisers=[Summariser(_Summariser(), "m")], prompts=_prompts(), decorate=decorate, cancel=cancel)


async def test_merging_stops_after_the_last_round() -> None:
    def reply(request: LLMRequest) -> str:
        return "## Facts\n- " + "n" * 2000 + "\n## Open\n- x"

    provider = _Summariser(reply)
    settings = SessionCompactionSettings(chunk_tokens=300, merge_rounds_max=1)
    result = await compact_session(_history(20, size=200), summarisers=[Summariser(provider, "m")], prompts=_prompts(), settings=settings)
    assert result.merge_rounds == 1


async def test_the_ladder_hands_everything_but_summaries_to_the_runs_provider() -> None:
    class _Own(_Summariser):
        def count_tokens(self, text: str, model: str | None = None) -> int:
            return 7

        async def complete_structured(self, request: LLMRequest, response_schema: dict[str, Any]) -> LLMResponse:
            return await self.complete_text(request)

        async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
            if False:  # pragma: no cover
                yield

    own = _Own()
    ladder = SummariserLadder([], fallback=lambda: own)
    assert ladder.count_tokens("abc") == 7
    assert "8080" in (await ladder.complete_structured(_request(), {})).message.text
    assert [event async for event in ladder.stream_with_tools(_request())] == []
