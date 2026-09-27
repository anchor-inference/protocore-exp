"""End-to-end scenarios for answering everything a run is waiting on.

Every scenario drives the engine only through entry points a host is supposed
to reach for — ``QueryEngine.run``, ``resume`` and the engine's own public
park/inspect methods — and observes only what a host can observe: yielded
``TurnEvent``s, the engine's public state and history, and the snapshot it
writes. No private symbol is touched, which is the point: a caller with
nothing else to hold on to has to be able to park a batch, answer it, and be
told plainly when its answers do not fit what the run is actually waiting for.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.hooks import HookActionKind, HookEvent, HookResult
from protocore.contracts.interrupt import (
    InterruptDecision,
    InterruptKind,
    InterruptResolution,
    InterruptResolutionError,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.snapshot import PENDING_INTERRUPTS_SNAPSHOT_KEY
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    OPERATOR_WORDS_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultBlock,
)
from protocore.runtime import (
    EventType,
    LoopState,
    resume,
    resume_approved_tool,
    resume_interrupts,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)


class _CountingTool(Tool):
    """A tool that records every invocation, so "exactly once" is assertable."""

    def __init__(self, name: str = "MyTool") -> None:
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="records its calls",
            parameters=ToolParameterSchema(properties={"v": {"type": "string"}}),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(tool_call_id="", content="ran", is_error=False)


class _Runtime:
    def __init__(self) -> None:
        self.llm = InMemoryLLMProvider()
        self.tools = InMemoryToolRegistry()
        self.events = InMemoryEventStream()
        self.hooks = InMemoryHookManager()
        self.skills = InMemorySkillStore()
        self.blobs = InMemoryBlobStore()

    def engine(self, *, run_id: str = "run-1", rc: LoopConstants | None = None) -> QueryEngine:
        return QueryEngine(
            config=QueryEngineConfig(
                run_id=run_id,
                tenant_id="tenant-1",
                account_id="tenant-1",
                session_id="sess-1",
                model_name="scripted-model",
                rc=rc or _approval_rc(),
            ),
            llm_provider=self.llm,
            tool_registry=self.tools,
            event_stream=self.events,
            hook_manager=self.hooks,
            skill_store=self.skills,
            blob_store=self.blobs,
        )


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _approval_rc() -> LoopConstants:
    return LoopConstants(model_context_window=4_096, approval_gate_web_enabled=True)


def _hold_next_tool_for_approval(runtime: _Runtime) -> None:
    runtime.hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-1"},
            reason="awaiting user",
        ),
    )


def _results(engine: QueryEngine) -> dict[str, str]:
    return {
        block.tool_call_id: block.content
        for message in engine.history_snapshot()
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }


async def _run_to_one_parked_approval(
    runtime: _Runtime, tool: _CountingTool
) -> tuple[QueryEngine, list[Any]]:
    """Drive a fresh turn that stops holding one tool call for a decision."""
    runtime.tools.register(tool)
    _hold_next_tool_for_approval(runtime)
    runtime.llm.queue_tool_call_response(
        tool_call_id="toolu_a",
        tool_name="MyTool",
        tool_input={"v": "first"},
    )
    engine = runtime.engine()
    events = [event async for event in engine.run(_user("use the tool"))]
    assert engine.state is LoopState.AWAITING
    assert tool.calls == []
    return engine, events


# ---------------------------------------------------------------------------
# What the host is told
# ---------------------------------------------------------------------------


async def test_a_parked_call_reaches_the_host_as_a_typed_wait_with_an_identity() -> None:
    """The event a host draws its card from carries the whole open set.

    Before the wait was a value, all a host had was the ``tool_call_pending``
    envelope and a state that said "awaiting". It could not quote an identity
    back, and it could not tell whether there were more waits behind this one.
    """
    runtime = _Runtime()
    engine, events = await _run_to_one_parked_approval(runtime, _CountingTool())

    parked_events = [event for event in events if event.type is EventType.INTERRUPT_PARKED]
    assert len(parked_events) == 1
    payload = parked_events[0].payload
    assert payload["interrupt"]["kind"] == InterruptKind.approval.value
    assert payload["interrupt"]["tool_call_id"] == "toolu_a"
    assert payload["interrupt"]["tool_name"] == "MyTool"
    assert len(payload["pending_interrupts"]) == 1

    # The same identity is on the engine, and it is what a resolution is keyed on.
    assert [item.interrupt_id for item in engine.pending_interrupts] == [
        payload["interrupt"]["interrupt_id"]
    ]


# ---------------------------------------------------------------------------
# A batch, answered in one act
# ---------------------------------------------------------------------------


async def test_a_batch_of_parked_calls_is_answered_in_one_resume() -> None:
    """Three waits, three different answers, one drive.

    This is what the single latch made impossible: it held one call id, so a
    second parked call had to become a second round of stop-ask-resume, with
    its own snapshot and its own redraw. Here the approve, the deny and the
    abandon all land together, and each call gets the result its own decision
    produced.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)

    # Two more calls the host's own gate parked — the same public entry the
    # loop itself goes through when a gate stops a call.
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                _tool_use("toolu_b", "MyTool", '{"v": "second"}'),
                _tool_use("toolu_c", "MyTool", '{"v": "third"}'),
            ],
        )
    )
    engine.mark_pending_approval("toolu_b", tool_name="MyTool")
    engine.mark_pending_approval("toolu_c", tool_name="MyTool")
    parked = engine.pending_interrupts
    assert [item.tool_call_id for item in parked] == ["toolu_a", "toolu_b", "toolu_c"]
    stored = engine.snapshot()

    successor = runtime.engine()
    events = [
        event
        async for event in resume(
            successor,
            stored,
            resolutions={
                parked[0].interrupt_id: InterruptResolution(
                    parked[0].interrupt_id, InterruptDecision.approve
                ),
                parked[1].interrupt_id: InterruptResolution(
                    parked[1].interrupt_id, InterruptDecision.deny, reason="too broad"
                ),
                parked[2].interrupt_id: InterruptResolution(
                    parked[2].interrupt_id, InterruptDecision.abandon
                ),
            },
        )
    ]

    assert [event.type for event in events].count(EventType.MESSAGE_STOP) == 1
    assert successor.pending_interrupts == ()
    # Exactly the approved call ran, and it ran once.
    assert tool.calls == [{"v": "first"}]
    results = _results(successor)
    assert results["toolu_a"] == "ran"
    assert results["toolu_b"] == successor.config.rc.tool_result_approval_denied_placeholder
    assert results["toolu_c"] == successor.config.rc.tool_result_approval_abandoned_placeholder


async def test_denying_one_and_approving_the_other_runs_only_what_was_allowed() -> None:
    """The discriminating pair: two identical calls, two opposite decisions."""
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[_tool_use("toolu_b", "MyTool", '{"v": "second"}')],
        )
    )
    engine.mark_pending_approval("toolu_b", tool_name="MyTool")
    parked = engine.pending_interrupts

    successor = runtime.engine()
    async for _ in resume(
        successor,
        engine.snapshot(),
        resolutions={
            parked[0].interrupt_id: InterruptResolution(
                parked[0].interrupt_id, InterruptDecision.deny
            ),
            parked[1].interrupt_id: InterruptResolution(
                parked[1].interrupt_id, InterruptDecision.approve
            ),
        },
    ):
        pass

    assert tool.calls == [{"v": "second"}]
    results = _results(successor)
    assert results["toolu_a"] == successor.config.rc.tool_result_approval_denied_placeholder
    assert results["toolu_b"] == "ran"


async def test_an_approval_may_correct_the_arguments_it_approves() -> None:
    """A person who wants the call run with the path fixed, not refused.

    Three things have to agree afterwards or the correction is a lie
    somewhere: what the tool was handed, what the transcript says was asked
    for, and the durable record written before the call.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    parked = engine.pending_interrupts[0]

    successor = runtime.engine()
    async for _ in resume(
        successor,
        engine.snapshot(),
        resolutions={
            parked.interrupt_id: InterruptResolution(
                parked.interrupt_id,
                InterruptDecision.approve,
                updated_input={"v": "corrected"},
            )
        },
    ):
        pass

    assert tool.calls == [{"v": "corrected"}]
    asked_for = [
        block.arguments_json
        for message in successor.history_snapshot()
        for block in message.content_blocks
        if getattr(block, "tool_call_id", None) == "toolu_a"
        and hasattr(block, "arguments_json")
    ]
    assert asked_for == ['{"v": "corrected"}']


# ---------------------------------------------------------------------------
# Abandoning
# ---------------------------------------------------------------------------


async def test_abandoning_says_the_call_never_ran_and_leaves_no_gap() -> None:
    """The one true thing about a decision that will never come.

    A record left standing keeps the call open forever, and a ``tool_use``
    left unpaired is filled in on the wire with a synthetic failure — which
    says the opposite of the truth about a call nobody approved.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    parked = engine.pending_interrupts[0]

    successor = runtime.engine()
    async for _ in resume(
        successor,
        engine.snapshot(),
        resolutions={
            parked.interrupt_id: InterruptResolution(
                parked.interrupt_id, InterruptDecision.abandon, reason="operator left"
            )
        },
    ):
        pass

    assert tool.calls == []
    assert successor.pending_interrupts == ()
    assert (
        _results(successor)["toolu_a"]
        == successor.config.rc.tool_result_approval_abandoned_placeholder
    )


# ---------------------------------------------------------------------------
# Answering a question
# ---------------------------------------------------------------------------


async def test_an_answered_question_becomes_the_result_of_the_call_that_asked() -> None:
    """A question is not an approval, and its answer is not a decision.

    Parked as one wait and answered as another, the reply would be written
    where the transcript wants a permission and the permission where it wants
    a reply. Here the answer lands as the result of the call that asked for
    it, which is the only place the model can read it as one.
    """
    runtime = _Runtime()
    engine = runtime.engine()
    engine.history.append(_user("ask me"))
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[_tool_use("toolu_q", "AskUser", "{}")],
        )
    )
    engine.transition_to(LoopState.RUNNING)
    asked = engine.mark_awaiting_answer(
        "toolu_q",
        tool_name="AskUser",
        payload={"questions": [{"question": "which file?"}]},
    )
    engine.transition_to(LoopState.AWAITING)
    assert asked.kind is InterruptKind.question
    # A question is deliberately invisible to the approval accessor: it is not
    # a call waiting to be allowed to run.
    assert engine.pending_approval_tool_call_id() is None

    successor = runtime.engine()
    async for _ in resume(
        successor,
        engine.snapshot(),
        resolutions={
            asked.interrupt_id: InterruptResolution(
                asked.interrupt_id, InterruptDecision.answer, answer="report.md"
            )
        },
    ):
        pass

    assert _results(successor)["toolu_q"] == "report.md"
    assert successor.pending_interrupts == ()
    # The reply is the operator's own words, and says so: compaction quotes a
    # result carrying this mark whole in its ledger.
    answered = next(
        block
        for message in successor.history_snapshot()
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == "toolu_q"
    )
    assert answered.metadata.get(OPERATOR_WORDS_METADATA_KEY) is True


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


async def test_a_map_that_answers_the_wrong_run_is_refused_with_nothing_done() -> None:
    """A stale id names a wait this run is not on; acting on it answers nothing."""
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    parked = engine.pending_interrupts[0]

    successor = runtime.engine()
    with pytest.raises(InterruptResolutionError, match="no interrupt 'int-from-elsewhere'"):
        async for _ in resume(
            successor,
            engine.snapshot(),
            resolutions={
                parked.interrupt_id: InterruptResolution(
                    parked.interrupt_id, InterruptDecision.approve
                ),
                "int-from-elsewhere": InterruptResolution(
                    "int-from-elsewhere", InterruptDecision.approve
                ),
            },
        ):
            pass

    # Refused BEFORE anything ran: the coherent half of an incoherent map is
    # not half-applied.
    assert tool.calls == []
    assert len(successor.pending_interrupts) == 1


async def test_approving_a_question_is_refused() -> None:
    """The confusion the single latch could not even express."""
    runtime = _Runtime()
    engine = runtime.engine()
    engine.history.append(_user("ask me"))
    engine.transition_to(LoopState.RUNNING)
    asked = engine.mark_awaiting_answer("toolu_q", tool_name="AskUser")
    engine.transition_to(LoopState.AWAITING)

    successor = runtime.engine()
    with pytest.raises(InterruptResolutionError, match="cannot be resolved by approve"):
        async for _ in resume(
            successor,
            engine.snapshot(),
            resolutions={
                asked.interrupt_id: InterruptResolution(
                    asked.interrupt_id, InterruptDecision.approve
                )
            },
        ):
            pass


async def test_leaving_one_of_two_undecided_is_refused_unless_it_is_deliberate() -> None:
    """Silence about a wait is far more often a dropped one than a chosen one."""
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[_tool_use("toolu_b", "MyTool", '{"v": "second"}')],
        )
    )
    engine.mark_pending_approval("toolu_b", tool_name="MyTool")
    parked = engine.pending_interrupts
    stored = engine.snapshot()
    one_of_two = {
        parked[0].interrupt_id: InterruptResolution(
            parked[0].interrupt_id, InterruptDecision.deny
        )
    }

    successor = runtime.engine()
    with pytest.raises(InterruptResolutionError, match="would be"):
        async for _ in resume(successor, stored, resolutions=one_of_two):
            pass

    # Said outright, the same map is accepted and the run keeps waiting.
    deliberate = runtime.engine()
    async for _ in resume(
        deliberate, stored, resolutions=one_of_two, allow_partial_resolution=True
    ):
        pass
    assert [item.tool_call_id for item in deliberate.pending_interrupts] == ["toolu_b"]
    assert deliberate.state is LoopState.AWAITING


async def test_a_partial_resume_that_answers_everything_simply_finishes() -> None:
    """The flag relaxes the map, it does not decide where the run ends up.

    Read off the flag instead, a caller that said "I may leave some parked" and
    then answered every one drove out of AWAITING into a completion AWAITING
    has no edge to — after its decisions were already in history and its
    snapshot already written. What survived was a durable run in AWAITING with
    nothing open, answerable by nothing.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    parked = engine.pending_interrupts
    full_map = {
        parked[0].interrupt_id: InterruptResolution(
            parked[0].interrupt_id, InterruptDecision.deny
        )
    }

    successor = runtime.engine()
    async for _ in resume(
        successor,
        engine.snapshot(),
        resolutions=full_map,
        allow_partial_resolution=True,
    ):
        pass

    assert successor.state is LoopState.COMPLETED
    assert successor.pending_interrupts == ()
    assert successor.snapshot()["state"] == LoopState.COMPLETED.value
    assert tool.calls == []


async def test_a_partial_resume_mixing_approve_and_deny_runs_only_the_approved() -> None:
    """Two decisions of different kinds in one map, with a third left parked."""
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    for call_id, value in (("toolu_b", "second"), ("toolu_c", "third")):
        engine.history.append(
            Message(
                role=MessageRole.assistant,
                content_blocks=[_tool_use(call_id, "MyTool", f'{{"v": "{value}"}}')],
            )
        )
        engine.mark_pending_approval(call_id, tool_name="MyTool")
    parked = {item.tool_call_id: item for item in engine.pending_interrupts}
    stored = engine.snapshot()

    successor = runtime.engine()
    async for _ in resume(
        successor,
        stored,
        resolutions={
            parked["toolu_a"].interrupt_id: InterruptResolution(
                parked["toolu_a"].interrupt_id, InterruptDecision.approve
            ),
            parked["toolu_b"].interrupt_id: InterruptResolution(
                parked["toolu_b"].interrupt_id, InterruptDecision.deny
            ),
        },
        allow_partial_resolution=True,
    ):
        pass

    assert [item.tool_call_id for item in successor.pending_interrupts] == ["toolu_c"]
    assert successor.state is LoopState.AWAITING
    assert [call["v"] for call in tool.calls] == ["first"]


async def test_a_plain_re_drive_over_a_parked_wait_is_refused() -> None:
    """The existing guarantee, restated over the typed wait.

    A re-drive is news that something happened elsewhere; it is not a decision
    on the call an operator was asked about. Walking past it would leave the
    call parked in history with no result, and the wire repair would then tell
    the model a call nobody approved was attempted and failed.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)

    successor = runtime.engine()
    with pytest.raises(ValueError, match="this run is waiting on"):
        async for _ in resume(successor, engine.snapshot()):
            pass
    assert tool.calls == []


async def test_a_resolution_map_and_a_single_answer_are_alternatives() -> None:
    """A caller supplying both has not decided which of two things happened."""
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    parked = engine.pending_interrupts[0]

    successor = runtime.engine()
    with pytest.raises(ValueError, match="not both"):
        async for _ in resume(
            successor,
            engine.snapshot(),
            resolutions={
                parked.interrupt_id: InterruptResolution(
                    parked.interrupt_id, InterruptDecision.approve
                )
            },
            abandon_approval=True,
        ):
            pass


def _tool_use(tool_call_id: str, name: str, arguments_json: str) -> Any:
    from protocore.contracts.types import ToolUseBlock

    return ToolUseBlock(
        tool_call_id=tool_call_id, name=name, arguments_json=arguments_json
    )


async def test_a_decision_that_arrives_after_the_deadline_does_not_run_the_call() -> None:
    """A host that sets a deadline gets one, not a number on a record.

    The stamp was written on every parked wait and read by nothing, so an
    operator who configured a deadline still had every stale approval execute
    the moment it was answered — which is worse than no knob, because the
    description said otherwise.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    stored = engine.snapshot()
    for record in stored[PENDING_INTERRUPTS_SNAPSHOT_KEY]:
        record["expires_at_ms"] = 1
    parked_id = stored[PENDING_INTERRUPTS_SNAPSHOT_KEY][0]["interrupt_id"]

    successor = runtime.engine()
    with pytest.raises(InterruptResolutionError, match="expired at 1"):
        async for _ in resume(
            successor,
            stored,
            resolutions={
                parked_id: InterruptResolution(parked_id, InterruptDecision.approve)
            },
        ):
            pass
    assert tool.calls == []

    # Abandoning it stays available, or the run would be stuck on it forever.
    cleared = runtime.engine()
    async for _ in resume(
        cleared,
        stored,
        resolutions={
            parked_id: InterruptResolution(parked_id, InterruptDecision.abandon)
        },
    ):
        pass
    assert cleared.pending_interrupts == ()
    assert tool.calls == []


# ---------------------------------------------------------------------------
# One decision out of a batch
# ---------------------------------------------------------------------------


def _park_second_call(engine: QueryEngine) -> None:
    """Park a second gated call of the same assistant message."""
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[_tool_use("toolu_b", "MyTool", '{"v": "second"}')],
        )
    )
    engine.mark_pending_approval("toolu_b", tool_name="MyTool")


async def test_approving_one_call_of_a_parked_batch_runs_only_that_call() -> None:
    """A single approval is a legal answer when several calls are parked.

    A host that shows one card at a time answers one card at a time, and the
    entry it reaches for takes one call. When the run held only ever one
    parked approval that was the whole story; once an assistant message parks
    three, the same honest approval has to keep working — and the other two
    have to still be parked afterwards, with the run still saying it is
    waiting, or the person who was going to decide them has nothing to come
    back to.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    _park_second_call(engine)
    assert [item.tool_call_id for item in engine.pending_interrupts] == [
        "toolu_a",
        "toolu_b",
    ]

    events = [
        event
        async for event in resume_approved_tool(
            engine, ToolCall(id="toolu_a", name="MyTool", arguments={"v": "first"})
        )
    ]

    assert [event.type for event in events].count(EventType.MESSAGE_STOP) == 1
    # Only the approved call ran; the other is untouched and has no result.
    assert tool.calls == [{"v": "first"}]
    assert _results(engine)["toolu_a"] == "ran"
    assert "toolu_b" not in _results(engine)
    # And the run is still waiting, for exactly what is left.
    assert engine.state is LoopState.AWAITING
    assert [item.tool_call_id for item in engine.pending_interrupts] == ["toolu_b"]

    # Deciding the last one carries the run through to the end.
    remaining = engine.pending_interrupts[0]
    async for _ in resume_interrupts(
        engine,
        {
            remaining.interrupt_id: InterruptResolution(
                remaining.interrupt_id, InterruptDecision.approve
            )
        },
    ):
        pass
    assert tool.calls == [{"v": "first"}, {"v": "second"}]
    assert engine.pending_interrupts == ()
    assert engine.state is LoopState.COMPLETED


async def test_one_call_of_a_batch_is_approved_by_a_process_that_never_saw_it() -> None:
    """The cold-resume variant: the deciding process holds only the snapshot.

    This is the shape the durable pickup actually has — the run's own process
    is gone, a successor rehydrates from the stored snapshot and is handed one
    approved call. It has to run that call, leave the rest of the batch parked
    in the snapshot it writes, and let a second pickup finish the job.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    _park_second_call(engine)
    stored = engine.snapshot()

    successor = runtime.engine()
    async for _ in resume(
        successor,
        stored,
        approved_tool_call=ToolCall(id="toolu_b", name="MyTool", arguments={"v": "second"}),
    ):
        pass

    assert tool.calls == [{"v": "second"}]
    assert successor.state is LoopState.AWAITING
    assert [item.tool_call_id for item in successor.pending_interrupts] == ["toolu_a"]
    assert "toolu_a" not in _results(successor)

    # A third process picks the rest up from what the second one wrote.
    last = runtime.engine()
    async for _ in resume(
        last,
        successor.snapshot(),
        approved_tool_call=ToolCall(id="toolu_a", name="MyTool", arguments={"v": "first"}),
    ):
        pass
    assert tool.calls == [{"v": "second"}, {"v": "first"}]
    assert last.pending_interrupts == ()
    assert last.state is LoopState.COMPLETED
    assert _results(last)["toolu_a"] == "ran"


async def test_a_call_this_run_never_parked_is_still_refused() -> None:
    """Batch-awareness widens what is accepted, not what may run.

    The refusal is the whole value of the check: a call id that is not among
    the parked approvals is a host that has confused two runs, and running it
    would execute a tool on the strength of a decision made about something
    else.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    _park_second_call(engine)

    with pytest.raises(ValueError, match="is not a parked approval"):
        async for _ in resume_approved_tool(
            engine, ToolCall(id="toolu_z", name="MyTool", arguments={"v": "other"})
        ):
            pass

    assert tool.calls == []
    assert [item.tool_call_id for item in engine.pending_interrupts] == [
        "toolu_a",
        "toolu_b",
    ]


async def test_answering_a_question_through_the_approval_door_is_refused() -> None:
    """A wait of another kind is not an approval, even when it is the only one.

    The membership test reads the kind as well as the id: a tool that ran far
    enough to ask something is holding for an answer, and approving it would
    run a second time the call whose reply was what was wanted.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[_tool_use("toolu_q", "MyTool", '{"v": "asked"}')],
        )
    )
    engine.mark_awaiting_answer("toolu_q", tool_name="MyTool")

    with pytest.raises(ValueError, match="is not a parked approval"):
        async for _ in resume_approved_tool(
            engine, ToolCall(id="toolu_q", name="MyTool", arguments={"v": "asked"})
        ):
            pass
    assert tool.calls == []


def _snapshots(runtime: _Runtime, engine: QueryEngine) -> list[dict[str, Any]]:
    """Every durable snapshot the run has written, oldest first."""
    return [
        event.payload["snapshot"]
        for event in runtime.events.stream_for(
            engine.config.tenant_id, engine.config.run_id
        )
        if event.name == "state_snapshot"
    ]


async def test_the_wait_is_durable_before_the_run_announces_it() -> None:
    """A host may stop reading the moment it is told a call is parked.

    That is what a pause IS to a host: record it, release the driver, and let
    another process pick the run up from its snapshot. Everything the run
    would write after that envelope is written into a generator nobody is
    consuming any more — so a snapshot persisted after the announcement is one
    that host never sees. It comes back to a durable record saying the run is
    waiting for nothing, and refuses the approval it was handed for a call it
    can no longer find.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    runtime.tools.register(tool)
    _hold_next_tool_for_approval(runtime)
    runtime.llm.queue_tool_call_response(
        tool_call_id="toolu_a",
        tool_name="MyTool",
        tool_input={"v": "first"},
    )
    engine = runtime.engine()

    # Consume exactly as far as the announcement and no further.
    async for event in engine.run(_user("use the tool")):
        if event.type is EventType.TOOL_CALL_PENDING:
            break

    latest = _snapshots(runtime, engine)[-1]
    assert [item["tool_call_id"] for item in latest[PENDING_INTERRUPTS_SNAPSHOT_KEY]] == [
        "toolu_a"
    ]

    # And a process holding only that snapshot can act on the decision.
    successor = runtime.engine()
    async for _ in resume(
        successor,
        latest,
        approved_tool_call=ToolCall(id="toolu_a", name="MyTool", arguments={"v": "first"}),
    ):
        pass
    assert tool.calls == [{"v": "first"}]
    assert successor.pending_interrupts == ()
