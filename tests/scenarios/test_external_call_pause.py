"""A call handed to an executor outside the run parks the run when that executor goes away.

The executor here is a machine the run does not own: the operator's laptop
running the tools of a remote session. When it disconnects with a call in
flight, failing the call would tell the model a command failed that may still
be running, and answering it with "unknown" would invite the model to repeat a
write. So the tool raises :class:`ExternalCallPauseRequested`, the call is
parked as an ``external_call`` interrupt, and the run waits. When the executor
comes back, :func:`redispatch_external_calls` hands the same call to the tool
again; the executor answers it from its own journal, so a call that already ran
is not run a second time, and the real result lands where the tool's result
belongs.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.interrupt import InterruptKind
from protocore.contracts.tools import ExternalCallPauseRequested, Tool, ToolContext
from protocore.contracts.types import (
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultBlock,
)
from protocore.runtime import EventType, LoopState, redispatch_external_calls
from protocore.runtime.query_engine import QueryEngine
from tests.scenarios.test_interrupt_resolution import _Runtime, _user


class _Machine:
    """The far side: a journal of what it ran, and whether it is reachable."""

    def __init__(self) -> None:
        self.online = False
        self.journal: dict[str, str] = {}
        self.executions: list[str] = []


class _RemoteTool(Tool):
    """Runs a command on :class:`_Machine`, answering a repeat of a call from its journal."""

    is_concurrent_safe = False
    is_destructive = True

    def __init__(self, machine: _Machine, name: str = "RemoteExec") -> None:
        self._machine = machine
        self._name = name
        self.invocations = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="runs a command on the operator's machine",
            parameters=ToolParameterSchema(properties={"command": {"type": "string"}}),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.invocations += 1
        call_id = str(context.metadata.get("tool_call_id") or "")
        if call_id in self._machine.journal:
            return ToolResult(tool_call_id=call_id, content=self._machine.journal[call_id])
        if not self._machine.online:
            raise ExternalCallPauseRequested({"machine": "laptop", "delivery": "not_delivered"})
        self._machine.executions.append(call_id)
        output = f"ran {arguments.get('command')}"
        self._machine.journal[call_id] = output
        return ToolResult(tool_call_id=call_id, content=output)


class _ReadTool(_RemoteTool):
    is_concurrent_safe = True
    is_destructive = False


def _results(engine: QueryEngine) -> dict[str, ToolResultBlock]:
    return {
        block.tool_call_id: block
        for message in engine.history_snapshot()
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }


async def _parked(runtime: _Runtime, machine: _Machine) -> tuple[QueryEngine, _RemoteTool, list[Any]]:
    tool = _RemoteTool(machine)
    runtime.tools.register(tool)
    runtime.llm.queue_tool_call_response(tool_call_id="toolu_x", tool_name="RemoteExec", tool_input={"command": "make"})
    engine = runtime.engine()
    events = [event async for event in engine.run(_user("build it"))]
    return engine, tool, events


async def test_an_executor_that_is_away_parks_the_run_as_an_external_call() -> None:
    runtime = _Runtime()
    engine, tool, events = await _parked(runtime, _Machine())

    assert engine.state is LoopState.AWAITING
    [parked] = engine.pending_interrupts
    assert parked.kind is InterruptKind.external_call
    assert parked.tool_call_id == "toolu_x"
    assert parked.payload["delivery"] == "not_delivered"
    pending = [event for event in events if event.type is EventType.TOOL_CALL_PENDING]
    assert [event.payload["kind"] for event in pending] == ["external_call"]
    assert pending[0].payload["external_call_payload"] == {"machine": "laptop", "delivery": "not_delivered"}
    announced = [event for event in events if event.type is EventType.INTERRUPT_PARKED]
    assert announced[-1].payload["interrupt"]["kind"] == "external_call"
    # Nothing was written as the call's result: not a failure, not an unknown outcome.
    assert "toolu_x" not in _results(engine)
    assert tool.invocations == 1


async def test_the_parked_call_is_handed_over_again_and_the_turn_goes_on() -> None:
    runtime = _Runtime()
    machine = _Machine()
    engine, _, _ = await _parked(runtime, machine)

    machine.online = True
    events = [event async for event in redispatch_external_calls(engine)]

    assert engine.state is LoopState.RUNNING
    assert engine.pending_interrupts == ()
    assert _results(engine)["toolu_x"].content == "ran make"
    assert [event.payload.get("tool_call_id") for event in events if event.type is EventType.TOOL_RESULT] == ["toolu_x"]

    runtime.llm.queue_response(text="built")
    async for _ in engine.run(None):
        pass
    assert engine.state is LoopState.COMPLETED
    sent = {
        block.tool_call_id: block.content
        for message in runtime.llm.calls[-1].messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }
    assert sent["toolu_x"] == "ran make"
    assert machine.executions == ["toolu_x"]


async def test_a_call_the_executor_already_ran_is_answered_not_run_again() -> None:
    """The executor finished the command while the link was down: its journal answers."""
    runtime = _Runtime()
    machine = _Machine()
    engine, _, _ = await _parked(runtime, machine)
    machine.journal["toolu_x"] = "ran make (finished while away)"
    machine.online = True

    async for _ in redispatch_external_calls(engine):
        pass

    assert _results(engine)["toolu_x"].content == "ran make (finished while away)"
    assert machine.executions == []


async def test_an_executor_still_away_parks_the_call_again_under_the_same_identity() -> None:
    runtime = _Runtime()
    engine, _, _ = await _parked(runtime, _Machine())
    [before] = engine.pending_interrupts

    async for _ in redispatch_external_calls(engine):
        pass

    assert engine.state is LoopState.AWAITING
    [after] = engine.pending_interrupts
    assert after.interrupt_id == before.interrupt_id
    assert "toolu_x" not in _results(engine)


async def test_a_parked_call_survives_a_restart_and_is_handed_over_from_the_snapshot() -> None:
    runtime = _Runtime()
    machine = _Machine()
    engine, _, _ = await _parked(runtime, machine)
    snapshot = engine.snapshot()
    intents = {item["tool_call_id"]: item["state"] for item in snapshot.get("open_intents", [])}
    assert intents.get("toolu_x") == "PAUSED_EXTERNAL"

    restored = runtime.engine()
    await restored.resume_from_snapshot(snapshot)
    assert restored.state is LoopState.AWAITING
    machine.online = True
    async for _ in redispatch_external_calls(restored):
        pass

    assert _results(restored)["toolu_x"].content == "ran make"
    assert restored.state is LoopState.RUNNING


async def test_a_parked_call_in_a_parallel_batch_waits_while_its_siblings_keep_their_results() -> None:
    runtime = _Runtime()
    machine = _Machine()
    reader = _ReadTool(machine, name="RemoteRead")
    always_there = _Machine()
    always_there.online = True
    local = _ReadTool(always_there, name="LocalRead")
    runtime.tools.register(reader)
    runtime.tools.register(local)
    runtime.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("toolu_r", "RemoteRead", {"command": "cat a"}),
            ("toolu_l", "LocalRead", {"command": "cat b"}),
        ]
    )
    engine = runtime.engine()
    async for _ in engine.run(_user("read both")):
        pass

    assert engine.state is LoopState.AWAITING
    assert [item.tool_call_id for item in engine.pending_interrupts] == ["toolu_r"]
    assert _results(engine)["toolu_l"].content == "ran cat b"
    assert "toolu_r" not in _results(engine)

    machine.online = True
    async for _ in redispatch_external_calls(engine):
        pass
    assert _results(engine)["toolu_r"].content == "ran cat a"
    assert engine.state is LoopState.RUNNING


async def test_an_abandoned_external_call_is_closed_with_the_hosts_words() -> None:
    """Cancelling a parked turn: the host answers the call with what it knows of it."""
    runtime = _Runtime()
    engine, _, _ = await _parked(runtime, _Machine())

    await engine.answer_pending("toolu_x", "cancelled while the machine was away; it may have run", is_error=True, operator_words=False)

    assert engine.pending_interrupts == ()
    assert _results(engine)["toolu_x"].is_error is True
    assert engine.state is LoopState.RUNNING


async def test_a_stopped_run_hands_nothing_over() -> None:
    runtime = _Runtime()
    machine = _Machine()
    engine, tool, _ = await _parked(runtime, machine)
    machine.online = True
    engine.stop()

    async for _ in redispatch_external_calls(engine):
        pass

    assert tool.invocations == 1
    assert machine.executions == []
