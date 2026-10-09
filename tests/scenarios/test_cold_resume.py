"""A cold resume: a NEW engine, built from a snapshot, finishing the run.

This is how a host survives a pod restart, and it is the reason none of
these scenarios reuse the engine that produced the snapshot. Everything
asserted is what the second process can see: the request its provider gets,
the tools it did or did not invoke again, and the snapshot it writes next.
"""
from __future__ import annotations

import builtins
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from protocore.contracts.llm import LLMStreamEvent, LLMStreamIdleError
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.events import EventType
from protocore.runtime.loop_state import LoopState
from protocore.runtime.permission_widen import CommandGrant, apply_widen, grant_covers

from .conftest import (
    FailingProvider,
    ProviderChainDouble,
    ScenarioFactory,
    ScriptedTool,
    classified,
    default_rc,
)


class DetachedPool:
    """A pool that can name a session's commands and say nothing about them.

    The shape a fresh process is in before the host re-adopts anything.
    """

    def __init__(self) -> None:
        self.refreshed: list[str] = []

    def mark_session_attached(self, session_id: str) -> None:
        return None

    async def ensure_session_attached(self, session_id: str) -> bool:
        return False

    def list(self, session_id: str) -> builtins.list[Any]:
        return []

    def get(self, task_id: str) -> Any:
        return None

    async def refresh(self, task_id: str) -> None:
        self.refreshed.append(task_id)

    def drain_wakes(self, session_id: str) -> builtins.list[str]:
        return []


async def test_the_second_process_is_shown_the_conversation_the_first_had(
    scenario: ScenarioFactory,
) -> None:
    first = scenario(rc=default_rc(model_context_window=32_000))
    first.llm.queue_response(text="the first answer")
    await first.run("the original question")

    second = scenario(rc=default_rc(model_context_window=32_000))
    second.llm.queue_response(text="the second answer")
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    second.engine.rearm()
    await second.run("a follow-up")

    assert any("the original question" in text for text in second.request_texts(0))
    assert any("the first answer" in text for text in second.request_texts(0))


async def test_a_tool_that_already_settled_is_not_run_a_second_time(
    scenario: ScenarioFactory,
) -> None:
    """The side effect happened before the pod died; it does not happen twice."""
    tool = ScriptedTool(tool_name="Charge", content="charged once")
    first = scenario(rc=default_rc(model_context_window=32_000), tools=[tool])
    first.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Charge", tool_input={"v": "1"}
    )
    first.llm.queue_response(text="charged")
    await first.run("charge it")
    assert tool.invocations == [{"v": "1"}]

    second = scenario(rc=default_rc(model_context_window=32_000), tools=[tool])
    second.llm.queue_response(text="nothing more to do")
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    second.engine.rearm()
    await second.run("anything else?")

    assert tool.invocations == [{"v": "1"}]
    assert [block.content for block in second.tool_results()] == ["charged once"]


async def test_an_orphaned_call_is_paired_rather_than_re_issued(
    scenario: ScenarioFactory,
) -> None:
    """The pod died between the call and its result.

    The unanswered ``tool_use`` is paired before the request leaves, because
    a provider rejects an unpaired one outright — and the tool is NOT called
    a second time, which is the part that would cost a duplicate side effect.
    """
    tool = ScriptedTool(tool_name="Charge")
    first = scenario(rc=default_rc(model_context_window=32_000), tools=[tool])
    first.engine.history.extend(
        [
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="charge it")]),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id="call-orphan",
                        name="Charge",
                        arguments_json='{"v": "1"}',
                    )
                ],
            ),
        ]
    )

    second = scenario(rc=default_rc(model_context_window=32_000), tools=[tool])
    second.llm.queue_response(text="I will check before repeating")
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    await second.run("what happened?")

    repaired = [
        block
        for message in second.requests[0].messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "call-orphan"
    ]
    assert len(repaired) == 1
    assert tool.invocations == []


async def test_the_wall_clock_budget_is_carried_across_rather_than_restarted(
    scenario: ScenarioFactory,
) -> None:
    """A resumed run keeps the time it already spent; it does not get it back."""
    first = scenario(rc=default_rc(agent_max_seconds=600.0))
    first.llm.queue_response(text="some work")
    await first.run("work on it")
    spent = first.engine.snapshot()["run_deadline_elapsed_seconds"]
    assert spent > 0.0

    second = scenario(rc=default_rc(agent_max_seconds=600.0))
    await second.engine.resume_from_snapshot(first.engine.snapshot())

    assert second.engine.snapshot()["run_deadline_elapsed_seconds"] >= spent


async def test_a_fresh_run_starts_with_the_whole_budget(
    scenario: ScenarioFactory,
) -> None:
    """The counterpart: nothing is carried into a run that never ran before."""
    run = scenario(rc=default_rc(agent_max_seconds=600.0))

    assert run.engine.snapshot()["run_deadline_elapsed_seconds"] == 0.0


async def test_the_turn_count_survives_the_restart(
    scenario: ScenarioFactory,
) -> None:
    first = scenario()
    first.llm.queue_response(text="one")
    await first.run("go")

    second = scenario()
    await second.engine.resume_from_snapshot(first.engine.snapshot())

    assert second.engine.snapshot()["turn_count"] == first.engine.snapshot()["turn_count"]


async def test_a_run_whose_background_commands_are_unreachable_says_so(
    scenario: ScenarioFactory,
) -> None:
    """A pool that cannot speak for the session is reported, not assumed idle."""
    first = scenario(
        rc=default_rc(background_tasks_enabled=True)
    )
    second = scenario(
        rc=default_rc(background_tasks_enabled=True),
        background_pool=DetachedPool(),
    )
    second.llm.queue_response(text="I cannot see my commands")
    await second.engine.resume_from_snapshot(first.engine.snapshot())

    produced = await second.run("what is running?")

    detached = [
        evt
        for evt in produced
        if evt.type is EventType.STATE_CHANGED
        and evt.payload.get("reason") == "background_tasks_detached"
    ]
    assert detached
    assert detached[0].payload["session_id"] == "sess-scenario"


async def test_the_detach_is_reported_once_within_the_turn_it_is_found(
    scenario: ScenarioFactory,
) -> None:
    """One report per turn, however many times the turn asks the pool."""
    run = scenario(
        rc=default_rc(background_tasks_enabled=True),
        background_pool=DetachedPool(),
        tools=[ScriptedTool(tool_name="Note")],
    )
    fresh = scenario(
        rc=default_rc(background_tasks_enabled=True)
    )
    await run.engine.resume_from_snapshot(fresh.engine.snapshot())
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Note", tool_input={}
    )
    run.llm.queue_response(text="one")

    produced = await run.run("first")

    detached = [
        evt
        for evt in produced
        if evt.type is EventType.STATE_CHANGED
        and evt.payload.get("reason") == "background_tasks_detached"
    ]
    assert len(detached) == 1


async def test_a_pool_that_can_speak_for_the_session_is_not_reported(
    scenario: ScenarioFactory,
) -> None:
    class AttachedPool(DetachedPool):
        async def ensure_session_attached(self, session_id: str) -> bool:
            return True

    run = scenario(
        rc=default_rc(background_tasks_enabled=True),
        background_pool=AttachedPool(),
    )
    fresh = scenario(
        rc=default_rc(background_tasks_enabled=True)
    )
    await run.engine.resume_from_snapshot(fresh.engine.snapshot())
    run.llm.queue_response(text="all quiet")

    produced = await run.run("what is running?")

    assert "background_tasks_detached" not in [
        str(evt.payload.get("reason", ""))
        for evt in produced
        if evt.type is EventType.STATE_CHANGED
    ]


async def test_a_snapshot_for_another_run_is_refused(
    scenario: ScenarioFactory,
) -> None:
    """A resume is bound to the run it belongs to, or it is not a resume."""
    first = scenario(run_id="run-a")
    second = scenario(run_id="run-b")

    with pytest.raises(ValueError):
        await second.engine.resume_from_snapshot(first.engine.snapshot())


async def test_a_snapshot_from_a_future_schema_is_refused_loudly(
    scenario: ScenarioFactory,
) -> None:
    first = scenario()
    snapshot = first.engine.snapshot()
    snapshot["schema_version"] = 10_000

    second = scenario()
    with pytest.raises(Exception):
        await second.engine.resume_from_snapshot(snapshot)


async def test_the_wind_down_the_first_process_started_is_in_what_the_second_shows(
    scenario: ScenarioFactory,
) -> None:
    """The bounds of a run write to history, and history is what survives.

    The first process reached its turn cap and wound the run down: the notice
    it appended and the snapshot it wrote immediately after are the whole of
    what a second process has to go on. A wind-down that lived only in the
    first engine's memory would leave the resumed run believing it had a full
    budget and everything still on its surface. The process died before the
    wind-down's own turn, so the second one resumes the snapshot written while
    the run was winding down, not a settled one.
    """
    from protocore.runtime import soft_stop as _soft_stop

    tool = ScriptedTool(tool_name="Read")
    rc = dict(model_context_window=32_000, soft_stop_enabled=True, max_turns_per_run=2)
    first = scenario(rc=default_rc(**rc), tools=[tool])
    for index in range(6):
        first.llm.queue_tool_call_response(
            tool_call_id=f"call-{index}", tool_name="Read", tool_input={"v": str(index)}
        )
    first.llm.queue_response(text="the wind-down answer")
    written: list[dict[str, Any]] = []
    persist = first.engine._persist_snapshot

    async def _persist_and_keep() -> None:
        if _soft_stop.is_armed(first.engine) and not first.engine.is_terminal and not written:
            written.append(first.engine.snapshot())
        await persist()

    first.engine._persist_snapshot = _persist_and_keep  # type: ignore[method-assign]
    await first.run("go")
    assert "soft_stop_notified" in first.state_reasons()
    assert written

    second = scenario(rc=default_rc(**rc), tools=[tool])
    await second.engine.resume_from_snapshot(written[0])
    second.llm.queue_response(text="the wind-down answer")
    await second.run(None)

    assert any(
        "max_turns" in text and "closing" in text
        for text in second.request_texts(0)
    ), second.request_texts(0)


def _truncated_call_stream(
    calls: Sequence[tuple[str, str, dict[str, Any], bool]],
) -> list[LLMStreamEvent]:
    """One round in which the model did not finish writing every call."""
    import json

    events = [LLMStreamEvent(name="message_start", payload={})]
    for tool_call_id, tool_name, arguments, truncated in calls:
        events.extend(
            [
                LLMStreamEvent(
                    name="tool_use_start",
                    payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
                ),
                LLMStreamEvent(
                    name="tool_use_input_delta",
                    payload={
                        "tool_call_id": tool_call_id,
                        "partial_input_json": json.dumps(arguments),
                    },
                ),
                LLMStreamEvent(
                    name="tool_use_stop",
                    payload={
                        "tool_call_id": tool_call_id,
                        "final_input": arguments,
                        "args_partial_truncated": truncated,
                    },
                ),
            ]
        )
    events.append(
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})
    )
    return events


async def test_the_second_process_teaches_a_truncated_call_to_chunk(
    scenario: ScenarioFactory,
) -> None:
    """The recovery is the resumed process's to make, on its restored count.

    A resume that only carries the transcript would hand the model a call it
    can neither run nor learn from: the arguments were closed by the parser,
    the dispatch fails on validation, and the model reads that as a broken
    tool. The instruction is written by whichever process sees the round.
    """
    write = ScriptedTool(tool_name="Write")
    read = ScriptedTool(tool_name="Read", content="file body")
    first = scenario(
        rc=default_rc(model_context_window=32_000), tools=[write, read]
    )
    first.llm.queue_response(text="starting")
    await first.run("write the file")

    second = scenario(
        rc=default_rc(model_context_window=32_000), tools=[write, read]
    )
    second.llm.queue_scripted_stream(
        _truncated_call_stream(
            [
                ("call-t", "Write", {}, True),
                ("call-clean", "Read", {"v": "a.txt"}, False),
            ]
        )
    )
    second.llm.queue_response(text="chunked instead")
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    second.engine.rearm()
    await second.run("now write it")

    assert write.invocations == []
    taught = [
        block for block in second.tool_results() if block.tool_call_id == "call-t"
    ]
    assert len(taught) == 1
    assert taught[0].is_error is True
    assert str(
        second.engine.config.rc.tool_call_max_input_chunk_bytes
    ) in taught[0].content
    # The sibling the model DID finish is not thrown away with the other.
    assert read.invocations == [{"v": "a.txt"}]


async def test_the_second_process_steps_down_the_chain_it_inherited(
    scenario: ScenarioFactory,
) -> None:
    """A resumed run answers a provider failure with the same ranking.

    The chain, the wind-down and the terminal are decided per round, not per
    process, so the process that never saw the run start still ranks the
    recoveries the same way — and the replacement is shown the partial the
    reader had already been given.
    """
    first = scenario(rc=default_rc(model_context_window=32_000))
    first.llm.queue_response(text="the answer before the restart")
    await first.run("ask")

    # Silent twice: the first silence is retried where it is, the second moves the chain.
    silence = classified(LLMStreamIdleError("went quiet"), "timeout")
    primary = FailingProvider(failures={0: silence, 1: silence}, partial_text="the first half")
    fallback = FailingProvider(failures={}, text="the replacement answer")
    chain = ProviderChainDouble(
        [("primary-model", primary), ("fallback-model", fallback)]
    )
    second = scenario(
        rc=default_rc(model_context_window=32_000, llm_transient_error_retry_backoff_base_seconds=0.0),
        llm_provider=primary,
        provider_chain=chain,
        model_name="primary-model",
    )
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    second.engine.rearm()
    await second.run("ask again")

    assert chain.advance_reasons
    assert fallback.calls
    assert "the replacement answer" in "".join(second.history_texts())


@dataclass
class _StopsItselfTool(ScriptedTool):
    """A tool whose own run is where the operator's stop lands."""

    engine: Any = None

    async def invoke(self, context: Any, arguments: dict[str, Any]) -> Any:
        result = await super().invoke(context, arguments)
        assert self.engine is not None
        self.engine.stop()
        return result


async def test_a_stop_reaching_the_second_process_mid_batch_ends_the_run(
    scenario: ScenarioFactory,
) -> None:
    """A resumed run answers a cancel the same way the first process would.

    The resumed turn is already dispatching when the stop arrives, so what
    ends the run is the checkpoint between two calls: everything already
    dispatched keeps its real result, the call behind it is never made, and
    the transcript the next reader picks up has no unanswered question in it.
    """
    stopper = _StopsItselfTool(tool_name="Stopper", content="stopped")
    later = ScriptedTool(tool_name="Later", content="never")
    first = scenario(
        rc=default_rc(model_context_window=32_000), tools=[stopper, later]
    )
    first.llm.queue_response(text="the answer before the restart")
    await first.run("ask")

    second = scenario(
        rc=default_rc(model_context_window=32_000), tools=[stopper, later]
    )
    stopper.engine = second.engine
    second.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("call-stop", "Stopper", {"v": "one"}),
            ("call-later", "Later", {"v": "two"}),
        ]
    )
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    second.engine.rearm()
    await second.run("carry on")

    assert later.invocations == []
    assert second.engine.state is LoopState.CANCELLED
    # Nothing the turn abandoned is left unanswered in the durable record.
    answered = {block.tool_call_id for block in second.tool_results()}
    assert {"call-stop", "call-later"} <= answered


async def test_the_call_the_process_died_inside_is_answered_honestly(
    scenario: ScenarioFactory,
) -> None:
    """The pod was killed between the ``tool_use`` and its result.

    Three things have to be true at once for that to be an honest answer
    rather than a fabricated one. The call is closed, because a provider
    refuses a transcript carrying an unanswered one. What closes it says the
    call was interrupted — it does not report a success the process never
    saw, and it does not report a failure of the tool, which did not fail.
    And the tool is NOT called again, because the process that died may well
    have completed the side effect before it went, and repeating it is the
    one outcome that turns an interruption into damage.

    The run then finishes normally: an interrupted call is a fact the next
    turn reads, not an error that ends the run.
    """
    tool = ScriptedTool(tool_name="Charge")
    first = scenario(tools=[tool])
    first.engine.history.extend(
        [
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="charge it")]),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id="call-killed",
                        name="Charge",
                        arguments_json='{"v": "1"}',
                    )
                ],
            ),
        ]
    )

    second = scenario(tools=[tool])
    second.llm.queue_response(text="the charge is unconfirmed, so I will not repeat it")
    await second.engine.resume_from_snapshot(first.engine.snapshot())
    produced = await second.run("what happened?")

    answered = [
        block
        for message in second.requests[0].messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "call-killed"
    ]
    assert len(answered) == 1
    # Honest about what it is: the loop's own published text for a result that
    # never arrived, not an invented one and not the tool's own words.
    assert answered[0].content == second.engine.prompt_text("tool_result_pairing_repair")
    assert answered[0].content.strip()
    # Marked as what it is, so the model is never told the call succeeded.
    assert answered[0].is_error is True
    # The side effect is not repeated, which is the whole point.
    assert tool.invocations == []
    # And the run itself finishes as a run: an interrupted call is a fact the
    # next turn reads, not an error that ends the run.
    assert EventType.MESSAGE_STOP in [evt.type for evt in produced]
    assert second.engine.state is LoopState.COMPLETED


async def test_the_permissions_a_person_widened_survive_the_restart_as_grants(
    scenario: ScenarioFactory,
) -> None:
    """A grant travels as a row and has to arrive as a grant.

    What a person widened before the pause is carried in the snapshot as the
    two strings a grant serialises to. The approval gate does not ask a row
    whether it covers a command — it asks the grant — so a resume that handed
    the gate rows would ask for approval again for exactly the commands
    somebody had already widened, and only after the pod died.
    """
    first = scenario(rc=default_rc(permission_widening_enabled=True))
    first.engine.session_grants = list(
        apply_widen([], "git status", kind="multiplexer_verb", rc=first.engine.config.rc)
    )

    second = scenario(rc=default_rc(permission_widening_enabled=True))
    await second.engine.resume_from_snapshot(first.engine.snapshot())

    assert second.engine.session_grants == first.engine.session_grants
    assert all(
        isinstance(grant, CommandGrant) for grant in second.engine.session_grants
    )
    assert grant_covers(second.engine.session_grants[0], "git status --short") is True


async def test_a_run_nobody_asked_anything_is_continued_from_where_it_died(
    scenario: ScenarioFactory,
) -> None:
    """The shape a host reclaims: no decision, no answer, no new user turn.

    An approval resume carries a decision and an ask-user resume carries an
    answer, so both hand the engine something to continue from. A run whose
    process simply vanished mid-tool has none of that — nobody was asked
    anything — and the only honest input is nothing at all.

    ``run(None)`` therefore has to be enough on its own: continue the history as
    it stands, close the call the dead process was inside without repeating it,
    and finish the run. Without this the host would have to invent a user turn
    to resume with, and an invented turn is a message the user never sent
    sitting in their transcript for ever.
    """
    tool = ScriptedTool(tool_name="Charge")
    first = scenario(tools=[tool])
    first.engine.history.extend(
        [
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="charge it")]),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id="call-vanished",
                        name="Charge",
                        arguments_json='{"v": "1"}',
                    )
                ],
            ),
        ]
    )

    second = scenario(tools=[tool])
    second.llm.queue_response(text="the charge is unconfirmed, so I will not repeat it")
    await second.engine.resume_from_snapshot(first.engine.snapshot())

    produced = await second.run(None)

    answered = [
        block
        for message in second.requests[0].messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "call-vanished"
    ]
    assert len(answered) == 1
    assert answered[0].content.strip()
    # The side effect is not repeated: the process that died may well have
    # completed it before it went.
    assert tool.invocations == []
    # No turn was invented to make the resume possible — the history the model
    # is shown ends where the run's own history did, plus the closed call.
    assert [
        block
        for message in second.requests[0].messages
        if message.role is MessageRole.user
        for block in message.content_blocks
        if isinstance(block, TextBlock) and block.text != "charge it"
    ] == []
    assert EventType.MESSAGE_STOP in [evt.type for evt in produced]
    assert second.engine.state is LoopState.COMPLETED


class RecordingEventStream:
    """An event stream a scenario can read the durable writes back out of."""

    def __init__(self) -> None:
        self.names: builtins.list[str] = []

    async def emit(self, event: Any) -> None:
        self.names.append(event.name)

    async def subscribe(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def trim(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None


async def test_a_pickup_point_exists_before_the_first_call_even_a_repeat_safe_one(
    scenario: ScenarioFactory,
) -> None:
    """A run is recoverable from its first tool call, whatever that tool is.

    The dispatcher writes a snapshot immediately before a call whose repeat is
    NOT harmless and nothing before one whose repeat is — the right trade per
    call, since a read costs a second read and nothing else. Read on its own
    that looks as though a run whose FIRST tool is a read reaches that call
    with nothing written down, leaving a host takeover no pickup point to
    rebuild from.

    It does not, and this is the test that says so rather than leaving it to be
    re-derived: ``run()`` persists a snapshot of its own before the first
    provider call, so every run is recoverable from before it does anything.
    The guarantee is a property of the drive, not of which tool happens to come
    first, and the assertion below is what would fail if that ever changed.
    """
    stream = RecordingEventStream()
    seen_before_the_call: builtins.list[int] = []

    class _Reading(ScriptedTool):
        async def invoke(self, context: Any, arguments: dict[str, Any]) -> Any:
            seen_before_the_call.append(
                sum(1 for name in stream.names if name == "state_snapshot")
            )
            return await super().invoke(context, arguments)

    tool = _Reading(tool_name="Read")
    run = scenario(
        tools=[tool],
        event_stream=stream,
        rc=default_rc(intent_repeat_safe_tools="Read"),
    )
    run.llm.queue_tool_call_response(
        tool_call_id="call-read", tool_name="Read", tool_input={"v": "notes.txt"}
    )
    run.llm.queue_response(text="read it")

    await run.run("read the notes")

    assert tool.invocations == [{"v": "notes.txt"}]
    assert seen_before_the_call and seen_before_the_call[0] >= 1, (
        "the run reached its first tool call with nothing written down, so a "
        "process lost inside that call leaves no pickup point"
    )


async def test_the_side_effect_of_the_interrupted_call_lands_at_most_once(
    scenario: ScenarioFactory,
) -> None:
    """Counted, not merely "not re-invoked" — the measurement the gate asks for.

    The first process gets as far as APPLYING the side effect and dies before
    the result is written, which is the only interesting case: the effect is
    already there, and a resume that re-issued the call would make it two. So
    the effect is a counter the tool increments, it stands at one when the
    second process takes the run over, and it still stands at one when that
    run reaches its end.
    """
    applied: builtins.list[str] = []

    class _Mutating(ScriptedTool):
        async def invoke(self, context: Any, arguments: dict[str, Any]) -> Any:
            applied.append(str(arguments.get("v", "")))
            return await super().invoke(context, arguments)

    tool = _Mutating(tool_name="Charge")
    first = scenario(tools=[tool])
    # The dead process's dispatch: the effect landed, its result never did.
    await tool.invoke(None, {"v": "1"})
    assert applied == ["1"]

    first.engine.history.extend(
        [
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="charge it")]),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id="call-mid-effect",
                        name="Charge",
                        arguments_json='{"v": "1"}',
                    )
                ],
            ),
        ]
    )

    second = scenario(tools=[tool])
    second.llm.queue_response(text="the charge is unconfirmed, so I will not repeat it")
    await second.engine.resume_from_snapshot(first.engine.snapshot())

    await second.run(None)

    assert second.engine.state is LoopState.COMPLETED
    assert applied == ["1"], (
        "the interrupted call's side effect landed more than once across the "
        "restart"
    )
