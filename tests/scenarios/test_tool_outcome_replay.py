"""A tool call's outcome outlives the process that produced it.

The calls of one model message can run together, and their results are placed
in the order the model asked for them. When the first call is slow, the ones
behind it finish and wait. These scenarios kill the process in that wait —
by cancelling the drive and resuming a NEW engine from the last snapshot the
first one wrote, which is all a restarted host would have — and assert what
the second process does with the work the first one finished: it places the
recorded results instead of running the calls again, and it says the honest
"outcome unknown" only about the call that never finished.

No sleeps: the slow call waits on an event nobody sets, and the drive is
polled with zero-length yields until the finished calls are on record.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

from protocore.contracts.events import IEventStream
from protocore.contracts.tool_outcome import ToolInvocationRecord
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    Event,
    Message,
    MessageRole,
    TextBlock,
    ToolResult,
    ToolUseBlock,
)
from protocore.runtime.events import EventType
from protocore.runtime.intent import fingerprint_arguments
from protocore.runtime.query import resume
from protocore.runtime.tool_outcomes import InMemoryToolOutcomeStore
from protocore.tests_support.adapters import InMemoryEventStream

from .conftest import (
    DelegationTool,
    ScenarioFactory,
    ScriptedTool,
    default_rc,
    require_approval,
)

TENANT = "tenant-scenario"
RUN = "run-scenario"


class SnapshotRecorder(InMemoryEventStream):
    """The event stream, keeping every snapshot the run asked to make durable.

    The last one written before a crash is exactly what a restarted host would
    load, so it is the only thing the second process is allowed to start from.
    """

    def __init__(self) -> None:
        super().__init__()
        self.snapshots: list[dict[str, Any]] = []

    async def emit(self, event: Event) -> None:
        if event.name == "state_snapshot":
            self.snapshots.append(json.loads(json.dumps(event.payload["snapshot"])))
        await super().emit(event)


assert isinstance(SnapshotRecorder(), IEventStream)


@dataclass
class Child(DelegationTool):
    """A delegation whose ``slow`` child never comes back on its own."""

    hold: asyncio.Event = field(default_factory=asyncio.Event)

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        if arguments.get("v") == "slow":
            self.invocations.append(dict(arguments))
            await self.hold.wait()
        return await super().invoke(context, arguments)


async def _until(predicate: Any, *, budget: int = 20_000) -> None:
    for _ in range(budget):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("the run never reached the point the scenario waits for")


async def _finished(store: InMemoryToolOutcomeStore) -> list[ToolInvocationRecord]:
    return [record for record in await store.list_run(TENANT, RUN) if record.state == "finished"]


async def _crash_mid_batch(
    scenario: ScenarioFactory,
    store: InMemoryToolOutcomeStore | None,
) -> tuple[Child, dict[str, Any]]:
    """Run a three-call batch, kill it while the first call still runs.

    Returns the tool and the last snapshot written before the kill. With a store
    the kill waits until the two fast calls are on record there; without one it
    waits until they have been invoked, which is as far as anything can tell.
    """
    tool = Child()
    recorder = SnapshotRecorder()
    first = scenario(
        rc=default_rc(max_concurrent_subagents=4),
        tools=[tool],
        event_stream=recorder,
        tool_outcome_store=store,
    )
    first.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("d-slow", "Delegate", {"v": "slow"}),
            ("d-b", "Delegate", {"v": "b"}),
            ("d-c", "Delegate", {"v": "c"}),
        ]
    )

    async def drive() -> None:
        async for _ in first.engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="fan out")])
        ):
            pass

    driver = asyncio.create_task(drive())
    if store is not None:
        finished: list[ToolInvocationRecord] = []

        async def probe() -> None:
            finished[:] = await _finished(store)

        for _ in range(20_000):
            await probe()
            if len(finished) == 2:
                break
            await asyncio.sleep(0)
        assert sorted(record.tool_call_id for record in finished) == ["d-b", "d-c"]
    else:
        await _until(lambda: len(tool.invocations) == 3 and not driver.done())
        # Let the two fast children return into the gather.
        for _ in range(200):
            await asyncio.sleep(0)
    crash_point = recorder.snapshots[-1]
    driver.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await driver
    return tool, crash_point


async def test_finished_siblings_are_replayed_not_rerun_after_a_crash(
    scenario: ScenarioFactory,
) -> None:
    store = InMemoryToolOutcomeStore()
    tool, crash_point = await _crash_mid_batch(scenario, store)

    second = scenario(
        rc=default_rc(max_concurrent_subagents=4),
        tools=[tool],
        tool_outcome_store=store,
    )
    second.llm.queue_response(text="all accounted for")
    events = [evt async for evt in resume(second.engine, crash_point)]
    second.events.extend(events)

    # Each child ran once, in the first process. The second ran none of them.
    assert sorted(str(call["v"]) for call in tool.invocations) == ["b", "c", "slow"]
    # The results sit in the order the model asked for them.
    results = {block.tool_call_id: block for block in second.tool_results()}
    assert [block.tool_call_id for block in second.tool_results()] == ["d-slow", "d-b", "d-c"]
    assert results["d-b"].content == "ok:b"
    assert results["d-c"].content == "ok:c"
    assert not results["d-b"].is_error
    # The call that never finished is described honestly, not as a failure.
    rc = second.engine.config.rc
    assert results["d-slow"].content == rc.tool_result_unknown_outcome_placeholder
    assert not results["d-slow"].is_error
    # The model is shown the replayed work.
    assert "ok:b" in second.request_texts(0)
    replayed = [
        evt.payload["tool_call_id"]
        for evt in second.events_of(EventType.TOOL_RESULT)
        if evt.payload.get("replayed")
    ]
    assert sorted(replayed) == ["d-b", "d-c"]
    # Every record has been answered by history, so the store holds none.
    assert await store.list_run(TENANT, RUN) == []


async def test_without_a_store_a_crash_loses_the_finished_siblings(
    scenario: ScenarioFactory,
) -> None:
    """The behaviour a host keeps by passing no store: nothing is replayed."""
    tool, crash_point = await _crash_mid_batch(scenario, None)

    second = scenario(rc=default_rc(max_concurrent_subagents=4), tools=[tool])
    second.llm.queue_response(text="starting over")
    _ = [evt async for evt in resume(second.engine, crash_point)]

    assert "ok:b" not in second.request_texts(0)
    assert sorted(str(call["v"]) for call in tool.invocations) == ["b", "c", "slow"]


async def test_a_result_recorded_before_placement_is_placed_exactly_once(
    scenario: ScenarioFactory,
) -> None:
    """The serial path: the tool returned, the process died before history moved."""
    store = InMemoryToolOutcomeStore()
    recorder = SnapshotRecorder()
    tool = ScriptedTool(tool_name="Send", content="sent")
    first = scenario(tools=[tool], event_stream=recorder, tool_outcome_store=store)
    first.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="Send", tool_input={"v": "x"}
    )
    first.llm.queue_response(text="done")
    await first.run("send it")

    # The last snapshot written before the result reached history is the one a
    # process that died right after the tool returned would have left.
    before_placement = next(
        snapshot
        for snapshot in reversed(recorder.snapshots)
        if not any(
            block.get("kind") == "tool_result"
            for message in snapshot["history"]
            for block in message.get("content_blocks", [])
        )
        and any(
            block.get("kind") == "tool_use"
            for message in snapshot["history"]
            for block in message.get("content_blocks", [])
        )
    )
    # The live run discarded its record once history answered it; put back
    # what the store held at that moment.
    await store.record_finished(
        ToolInvocationRecord(
            invocation_id="inv-s-1",
            tenant_id=TENANT,
            run_id=RUN,
            tool_call_id="s-1",
            tool_name="Send",
            arguments_digest=fingerprint_arguments("Send", {"v": "x"}),
            state="finished",
            outcome={"kind": "tool_result", "tool_call_id": "s-1", "content": "sent"},
        )
    )

    second = scenario(tools=[tool], tool_outcome_store=store)
    second.llm.queue_response(text="done again")
    _ = [evt async for evt in resume(second.engine, before_placement)]

    assert tool.invocations == [{"v": "x"}]
    assert [block.content for block in second.tool_results()] == ["sent"]


async def test_a_live_run_records_each_outcome_and_forgets_it_once_placed(
    scenario: ScenarioFactory,
) -> None:
    class Spy(InMemoryToolOutcomeStore):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[str, str]] = []

        async def record_started(self, record: ToolInvocationRecord) -> None:
            self.calls.append(("started", record.tool_call_id))
            await super().record_started(record)

        async def record_finished(self, record: ToolInvocationRecord) -> bool:
            self.calls.append(("finished", record.tool_call_id))
            return await super().record_finished(record)

    store = Spy()
    write = ScriptedTool(tool_name="Send", content="sent")
    look = ScriptedTool(tool_name="Look", content="seen")
    run = scenario(
        rc=default_rc(intent_repeat_safe_tools="Look"),
        tools=[write, look],
        tool_outcome_store=store,
    )
    run.llm.queue_tool_call_response(tool_call_id="w-1", tool_name="Send", tool_input={"v": "a"})
    run.llm.queue_tool_call_response(tool_call_id="l-1", tool_name="Look", tool_input={"v": "b"})
    run.llm.queue_response(text="done")
    await run.run("go")

    # A call whose repeat would do something twice is on record before it runs;
    # one whose repeat is harmless is recorded only once it has an outcome.
    assert store.calls == [
        ("started", "w-1"),
        ("finished", "w-1"),
        ("finished", "l-1"),
    ]
    assert await store.list_run(TENANT, RUN) == []


async def test_a_call_parked_for_approval_leaves_no_record(
    scenario: ScenarioFactory,
) -> None:
    store = InMemoryToolOutcomeStore()
    tool = ScriptedTool(tool_name="Send")
    run = scenario(
        rc=default_rc(approval_gate_web_enabled=True),
        tools=[tool],
        tool_outcome_store=store,
    )
    require_approval(run.hooks)
    run.llm.queue_tool_call_response(tool_call_id="p-1", tool_name="Send", tool_input={"v": "x"})
    await run.run("send it")

    assert tool.invocations == []
    assert await store.list_run(TENANT, RUN) == []


async def test_a_stale_record_never_answers_a_call_that_reuses_its_id(
    scenario: ScenarioFactory,
) -> None:
    """A provider may reuse a call id; the arguments tell the two calls apart."""
    store = InMemoryToolOutcomeStore()
    await store.record_finished(
        ToolInvocationRecord(
            invocation_id="old",
            tenant_id=TENANT,
            run_id=RUN,
            tool_call_id="c-1",
            tool_name="Send",
            arguments_digest=fingerprint_arguments("Send", {"v": "earlier"}),
            state="finished",
            outcome={"kind": "tool_result", "tool_call_id": "c-1", "content": "old result"},
        )
    )
    first = scenario(tools=[ScriptedTool(tool_name="Send")])
    first.engine.history.extend(
        [
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="send")]),
            Message(
                role=MessageRole.assistant,
                content_blocks=[
                    ToolUseBlock(
                        tool_call_id="c-1", name="Send", arguments_json='{"v": "later"}'
                    )
                ],
            ),
        ]
    )
    snapshot = first.engine.snapshot()

    second = scenario(tools=[ScriptedTool(tool_name="Send")], tool_outcome_store=store)
    second.llm.queue_response(text="ok")
    _ = [evt async for evt in resume(second.engine, snapshot)]

    assert "old result" not in [block.content for block in second.tool_results()]
    assert await store.list_run(TENANT, RUN) == []


async def test_a_stop_on_resume_still_keeps_the_finished_results(
    scenario: ScenarioFactory,
) -> None:
    """A stop cancels what was still to come, not what had already finished."""
    store = InMemoryToolOutcomeStore()
    tool, crash_point = await _crash_mid_batch(scenario, store)

    second = scenario(
        rc=default_rc(max_concurrent_subagents=4),
        tools=[tool],
        tool_outcome_store=store,
    )
    await second.engine.resume_from_snapshot(crash_point)
    second.engine.stop()
    _ = [evt async for evt in second.engine.run(None)]

    results = {block.tool_call_id: block for block in second.tool_results()}
    assert results["d-b"].content == "ok:b"
    assert results["d-c"].content == "ok:c"
    assert results["d-slow"].content != "ok:slow"
    assert sorted(str(call["v"]) for call in tool.invocations) == ["b", "c", "slow"]
