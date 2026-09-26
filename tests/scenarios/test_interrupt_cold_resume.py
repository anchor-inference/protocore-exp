"""A wait outlives the process that recorded it.

The point of making the wait durable is that the process holding it dies. An
operator decision may take a week; the pod does not. So every guarantee below
is asserted across a snapshot boundary — the run is written out by one engine
and picked up by another, built fresh, sharing nothing but the payload — and
the payload alone has to be enough to say what was being waited for, which
call, and of what kind.

The last scenario here is the one that used to be impossible to state at all:
a payload written under the shape that held one call id comes back as a typed
wait, so a run paused across the change is resumable rather than stranded.
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
from protocore.contracts.snapshot import (
    PENDING_INTERRUPTS_SNAPSHOT_KEY,
    SNAPSHOT_SCHEMA_KEY,
    SNAPSHOT_SCHEMA_VERSION,
)
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultBlock,
)
from protocore.runtime import EventType, LoopState, resume
from protocore.runtime.loop_state import UnwitnessedAwaitError
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)

#: The last schema version whose paused run named its wait with one bare call
#: id. Stated as the number rather than as "the version before the current
#: one": later versions change other fields, and the shape these tests build is
#: this one's.
_UNTYPED_WAIT_SCHEMA_VERSION = 5


class _CountingTool(Tool):
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

    def engine(self, *, run_id: str = "run-1") -> QueryEngine:
        return QueryEngine(
            config=QueryEngineConfig(
                run_id=run_id,
                tenant_id="tenant-1",
                account_id="tenant-1",
                session_id="sess-1",
                model_name="scripted-model",
                rc=LoopConstants(model_context_window=4_096, approval_gate_web_enabled=True),
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


def _results(engine: QueryEngine) -> dict[str, str]:
    return {
        block.tool_call_id: block.content
        for message in engine.history_snapshot()
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }


async def _run_to_one_parked_approval(runtime: _Runtime, tool: _CountingTool) -> QueryEngine:
    runtime.tools.register(tool)
    runtime.hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-1"},
            reason="awaiting user",
        ),
    )
    runtime.llm.queue_tool_call_response(
        tool_call_id="toolu_a", tool_name="MyTool", tool_input={"v": "first"}
    )
    engine = runtime.engine()
    async for _ in engine.run(_user("use the tool")):
        pass
    assert engine.state is LoopState.AWAITING
    return engine


# ---------------------------------------------------------------------------


async def test_the_snapshot_states_the_wait_in_full() -> None:
    """Everything a fresh process needs to draw the card and take the answer.

    The kind decides which decisions are legal, the payload is what the person
    is shown, and the identity is what their answer is addressed to. A payload
    that carried only the call id could supply none of the three.
    """
    runtime = _Runtime()
    engine = await _run_to_one_parked_approval(runtime, _CountingTool())
    stored = engine.snapshot()

    assert stored[SNAPSHOT_SCHEMA_KEY] == SNAPSHOT_SCHEMA_VERSION
    assert "pending_approval_tool_call_id" not in stored
    written = stored[PENDING_INTERRUPTS_SNAPSHOT_KEY]
    assert len(written) == 1
    assert written[0]["kind"] == InterruptKind.approval.value
    assert written[0]["tool_call_id"] == "toolu_a"
    assert written[0]["tool_name"] == "MyTool"
    assert written[0]["interrupt_id"]


async def test_a_wait_recorded_by_one_process_is_answered_by_another() -> None:
    """The identity travels, so the answer reaches the wait it was written for."""
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_one_parked_approval(runtime, tool)
    parked_id = engine.pending_interrupts[0].interrupt_id
    stored = engine.snapshot()
    del engine

    successor = runtime.engine()
    events = [
        event
        async for event in resume(
            successor,
            stored,
            resolutions={
                parked_id: InterruptResolution(parked_id, InterruptDecision.approve)
            },
        )
    ]

    assert [event.type for event in events].count(EventType.MESSAGE_STOP) == 1
    assert tool.calls == [{"v": "first"}]
    assert successor.pending_interrupts == ()
    assert successor.state is LoopState.COMPLETED


async def test_an_approval_that_already_landed_is_not_applied_a_second_time() -> None:
    """The crash window between "the result was written" and "the wait was released".

    A process that dies there leaves a payload whose history already holds the
    result and whose open set still holds the wait. The next process is handed
    exactly that, with the same answer, and must not run the tool again: the
    effect is already in place, and applying it twice for one decision is the
    failure the durable record exists to prevent.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_one_parked_approval(runtime, tool)
    parked_id = engine.pending_interrupts[0].interrupt_id
    stored = engine.snapshot()

    first = runtime.engine()
    async for _ in resume(
        first,
        stored,
        resolutions={parked_id: InterruptResolution(parked_id, InterruptDecision.approve)},
    ):
        pass
    assert tool.calls == [{"v": "first"}]

    half_written = dict(first.snapshot())
    half_written[PENDING_INTERRUPTS_SNAPSHOT_KEY] = stored[PENDING_INTERRUPTS_SNAPSHOT_KEY]
    half_written["state"] = LoopState.AWAITING.value

    second = runtime.engine()
    async for _ in resume(
        second,
        half_written,
        resolutions={parked_id: InterruptResolution(parked_id, InterruptDecision.approve)},
    ):
        pass

    assert tool.calls == [{"v": "first"}]
    assert _results(second)["toolu_a"] == "ran"
    assert second.pending_interrupts == ()


async def test_an_answer_addressed_to_the_wrong_wait_is_refused_across_the_boundary() -> None:
    """The identity the host quotes has to be the one the run actually holds.

    Across a process boundary this is the whole check there is: the successor
    has no memory of what was parked beyond the payload, so an id that does
    not appear in it is the only evidence available that the answer belongs to
    something else.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_one_parked_approval(runtime, tool)
    stored = engine.snapshot()

    successor = runtime.engine()
    with pytest.raises(InterruptResolutionError, match="no interrupt 'int-stale'"):
        async for _ in resume(
            successor,
            stored,
            resolutions={"int-stale": InterruptResolution("int-stale", InterruptDecision.approve)},
        ):
            pass
    assert tool.calls == []
    assert successor.state is LoopState.AWAITING


async def test_a_run_paused_under_the_older_shape_comes_back_as_a_typed_wait() -> None:
    """A run paused when the change rolled out is resumable, not stranded.

    Refusing the payload is not recoverable for a paused run: nothing later
    will accept it either. So the older shape — one field holding one call id
    — is read as what it always was in practice, a call parked at a gate, and
    the run picks up and is answered normally.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_one_parked_approval(runtime, tool)
    stored = engine.snapshot()

    # The shape a store written before the wait had a type still holds.
    older = {key: value for key, value in stored.items() if key != PENDING_INTERRUPTS_SNAPSHOT_KEY}
    older["pending_approval_tool_call_id"] = "toolu_a"
    older[SNAPSHOT_SCHEMA_KEY] = _UNTYPED_WAIT_SCHEMA_VERSION

    successor = runtime.engine()
    await successor.resume_from_snapshot(older)
    lifted = successor.pending_interrupts
    assert len(lifted) == 1
    assert lifted[0].kind is InterruptKind.approval
    assert lifted[0].tool_call_id == "toolu_a"
    assert successor.state is LoopState.AWAITING

    driven = runtime.engine()
    async for _ in resume(
        driven,
        older,
        resolutions={
            lifted[0].interrupt_id: InterruptResolution(
                lifted[0].interrupt_id, InterruptDecision.approve
            )
        },
    ):
        pass
    assert tool.calls == [{"v": "first"}]


async def test_an_ask_user_pause_written_under_the_older_shape_can_be_answered() -> None:
    """The older field held both parks; the intent record says which one this was.

    Read as an approval, the question comes back answerable by nothing: a
    message is refused because the wait says approval, and approving it would
    re-run the tool that already asked.
    """
    runtime = _Runtime()
    tool = _CountingTool()
    engine = await _run_to_one_parked_approval(runtime, tool)
    stored = engine.snapshot()

    older = {
        key: value
        for key, value in stored.items()
        if key != PENDING_INTERRUPTS_SNAPSHOT_KEY
    }
    older["pending_approval_tool_call_id"] = "toolu_a"
    older["open_intents"] = [
        {
            "tool_call_id": "toolu_a",
            "tool_name": "MyTool",
            "state": "PAUSED_ASK_USER",
        }
    ]
    older[SNAPSHOT_SCHEMA_KEY] = _UNTYPED_WAIT_SCHEMA_VERSION

    successor = runtime.engine()
    await successor.resume_from_snapshot(older)
    lifted = successor.pending_interrupts
    assert [item.kind for item in lifted] == [InterruptKind.question]

    driven = runtime.engine()
    async for _ in resume(
        driven,
        older,
        resolutions={
            lifted[0].interrupt_id: InterruptResolution(
                lifted[0].interrupt_id,
                InterruptDecision.answer,
                answer="the port is 8080",
            )
        },
    ):
        pass
    assert tool.calls == []
    assert driven.state is LoopState.COMPLETED


async def test_a_stored_run_that_waits_for_nothing_is_refused_at_the_pickup() -> None:
    """The rule the live transition enforces has to hold over the record too.

    A restore assigns the state rather than moving to it, so without the check
    a host could pick up a run that says it is waiting and names nothing that
    could end the wait — and then wait forever on it.
    """
    runtime = _Runtime()
    engine = await _run_to_one_parked_approval(runtime, _CountingTool())
    stored = engine.snapshot()
    stored[PENDING_INTERRUPTS_SNAPSHOT_KEY] = []

    successor = runtime.engine()
    with pytest.raises(UnwitnessedAwaitError):
        await successor.resume_from_snapshot(stored)
