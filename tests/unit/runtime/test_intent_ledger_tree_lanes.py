"""Intent, ledger, fork/clone, lanes, hooks, telemetry — shipped paths."""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.middleware import (
    LifecycleContext,
    LifecycleDecision,
    LifecycleVerdict,
    RegistrationKind,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import HookEvent, Message, MessageRole, TextBlock, ToolUseBlock
from protocore.hooks import HookManager, refuse_lifecycle_when_disabled
from protocore.runtime.events import EventType
from protocore.runtime.intent import (
    DISPATCHED,
    commit_intent,
    refuse_intent_when_disabled,
    replay_policy_for,
)
from protocore.runtime.lanes import (
    acquire_lane,
    create_lane,
    ensure_main,
    mark_diverged,
    refuse_lanes_when_disabled,
    release_lane,
    reviewer_blocks_main,
)
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import _query as query
from protocore.runtime.telemetry import is_prometheus_safe_label, mark_recovery, start_span
from protocore.runtime.usage_ledger import append_usage, from_seq, refuse_ledger_when_disabled, session_total
from protocore.tests_support.adapters import InMemoryLLMProvider

from ._tool_fixtures import MockTool


def _on(**overrides: object) -> LoopConstants:
    values: dict[str, object] = {
        "model_context_window": 4096,
        "intent_settlement_enabled": True,
        "usage_ledger_enabled": True,
        "lanes_enabled": True,
        "typed_hooks_enabled": True,
        "telemetry_spans_enabled": True,
    }
    values.update(overrides)
    return LoopConstants(**values)  # type: ignore[arg-type]


def test_flags_off_refuse() -> None:
    with pytest.raises(ValueError, match="intent_settlement_disabled"):
        refuse_intent_when_disabled(False)
    with pytest.raises(ValueError, match="usage_ledger_disabled"):
        refuse_ledger_when_disabled(False)
    with pytest.raises(ValueError, match="lanes_disabled"):
        refuse_lanes_when_disabled(False)
    with pytest.raises(ValueError, match="lifecycle_hooks_disabled"):
        refuse_lifecycle_when_disabled(False)


def test_replay_policy_write_never_read_safe() -> None:
    rc = _on()
    assert replay_policy_for("Write", rc) == "never"
    assert replay_policy_for("Read", rc) == "safe"


@pytest.mark.asyncio
async def test_query_write_intent_then_crash_does_not_rewrite(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    engine = engine_factory(rc=rc)
    writes: list[dict[str, object]] = []

    async def on_write(args: dict[str, object]) -> None:
        writes.append(args)

    in_memory_runtime["tools"].register(
        MockTool(tool_name="Write", description="write", on_invoke=on_write, response_content="ok")
    )
    llm.queue_tool_call_response(
        tool_call_id="w1",
        tool_name="Write",
        tool_input={"path": "f.txt", "content": "once"},
    )
    llm.queue_response(text="done")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="write")]))
    events = [evt async for evt in query(engine)]
    assert any(evt.type == EventType.INTENT_COMMITTED for evt in events)
    assert len(writes) == 1
    # A settled call leaves no record behind: history holds the result.
    assert engine.open_intents == []
    # The run stops between the call and its result, so the record survives
    # saying "dispatched" and history has no result for it.
    engine.open_intents = [
        commit_intent(
            tool_name="Write",
            tool_call_id="w1",
            rc=rc,
            arguments={"path": "f.txt", "content": "once"},
        )
    ]
    engine.history = [
        message
        for message in engine.history
        if not any(
            getattr(block, "tool_call_id", None) == "w1"
            and block.__class__.__name__ == "ToolResultBlock"
            for block in message.content_blocks
        )
    ]
    assert engine.open_intents[-1].state == DISPATCHED
    llm.queue_tool_call_response(
        tool_call_id="w1",
        tool_name="Write",
        tool_input={"path": "f.txt", "content": "twice"},
    )
    llm.queue_response(text="recovered")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="again")]))
    engine.state = LoopState.PENDING
    more = [evt async for evt in query(engine)]
    # The interrupted call is not run again: the resume answers it as unknown.
    # The model's next call is a new decision, made after reading that answer,
    # and runs under an id of its own even though the model reused ``w1`` —
    # swallowing it because its id repeated was the defect, not the guarantee.
    assert [write["content"] for write in writes] == ["once", "twice"]
    unknown = [
        evt
        for evt in more
        if evt.type == EventType.TOOL_RESULT and evt.payload.get("outcome") == "unknown"
    ]
    assert unknown
    assert unknown[0].payload["is_error"] is False
    assert "never recorded" in unknown[0].payload["content"]
    assert unknown[0].payload["tool_call_id"] == "w1"
    rerun = [evt for evt in more if evt.type == EventType.TOOL_USE_START]
    assert [evt.payload["tool_call_id"] for evt in rerun] != ["w1"]


@pytest.mark.asyncio
async def test_query_read_safe_replays(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    engine = engine_factory(rc=rc)
    reads: list[dict[str, object]] = []

    async def on_read(args: dict[str, object]) -> None:
        reads.append(args)

    in_memory_runtime["tools"].register(
        MockTool(tool_name="Read", description="read", on_invoke=on_read, response_content="hi")
    )
    llm.queue_tool_call_response(tool_call_id="r1", tool_name="Read", tool_input={"path": "a"})
    llm.queue_response(text="ok")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="read")]))
    await _drain(query(engine))
    llm.queue_tool_call_response(tool_call_id="r2", tool_name="Read", tool_input={"path": "a"})
    llm.queue_response(text="ok2")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="again")]))
    engine.state = LoopState.PENDING
    await _drain(query(engine))
    assert len(reads) == 2


async def _drain(agen) -> list:
    return [item async for item in agen]


def test_ledger_fail_retry_compact_sum() -> None:
    rc = _on()
    rows: list[Any] = []
    rows = append_usage(rows, kind="inference", run_id="r", input_tokens=10, output_tokens=1, success=False, rc=rc)
    rows = append_usage(rows, kind="retry", run_id="r", input_tokens=10, output_tokens=2, success=True, rc=rc)
    rows = append_usage(rows, kind="compaction", run_id="r", input_tokens=3, output_tokens=0, success=True, rc=rc)
    rows = append_usage(rows, kind="inference", run_id="r", input_tokens=4, output_tokens=5, success=True, rc=rc)
    assert [item.seq for item in rows] == [1, 2, 3, 4]
    assert session_total(rows) == 10 + 1 + 10 + 2 + 3 + 4 + 5
    assert [item.seq for item in from_seq(rows, 2)] == [3, 4]
    assert append_usage([], kind="inference", run_id="r", input_tokens=1, output_tokens=1, success=True, rc=LoopConstants()) == []


def test_lanes_reviewer_after_diverge_does_not_block_main() -> None:
    rc = _on()
    lanes = ensure_main([])
    lanes = create_lane(lanes, lane_id="reviewer", cursor=0, model="m", toolset=("Read",), rc=rc)
    lanes = acquire_lane(lanes, "reviewer", "pod-b")
    lanes = mark_diverged(lanes, "reviewer")
    assert not reviewer_blocks_main(lanes)
    lanes = acquire_lane(lanes, "main", "pod-a")
    assert lanes[0].locked_by == "pod-a"
    lanes = release_lane(lanes, "reviewer", "pod-b")
    assert next(item.locked_by for item in lanes if item.lane_id == "reviewer") is None
    with pytest.raises(ValueError, match="lanes_disabled"):
        create_lane([], lane_id="x", cursor=0, model="m", toolset=(), rc=LoopConstants())


@pytest.mark.asyncio
async def test_typed_hooks_and_telemetry() -> None:
    rc = _on()
    registry = HookManager()
    registry.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda _ctx: LifecycleDecision(
            verdict=LifecycleVerdict.require_approval, approval_token="tok"
        ),
        owner="test",
    )
    out = await registry.dispatch(
        LifecycleContext(point=HookEvent.pre_tool_use, payload={"tool_name": "Write"})
    )
    assert out.verdict is LifecycleVerdict.require_approval
    span = start_span("tool", rc=rc, tool="Write")
    assert span is not None
    marked = mark_recovery(span, intent_id="op_1")
    assert marked is not None and marked.attributes["recovery"] is True
    assert marked.attributes["intent_id"] == "op_1"
    assert start_span("tool", rc=LoopConstants()) is None
    assert not is_prometheus_safe_label("session_id")
    assert is_prometheus_safe_label("tenant")


@pytest.mark.asyncio
async def test_query_before_tool_approval_pauses(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    engine = engine_factory(rc=rc)
    registry = HookManager()
    registry.register(
        HookEvent.pre_tool_use,
        RegistrationKind.decide,
        lambda _ctx: LifecycleDecision(
            verdict=LifecycleVerdict.require_approval, approval_token="tok-h"
        ),
        owner="test",
    )
    engine.lifecycle_hooks = registry
    in_memory_runtime["tools"].register(MockTool(tool_name="Write", description="w"))
    llm.queue_tool_call_response(tool_call_id="w2", tool_name="Write", tool_input={"path": "x"})
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="w")]))
    events = [evt async for evt in query(engine)]
    assert any(
        evt.type == EventType.TOOL_CALL_PENDING and evt.payload.get("requires_approval")
        for evt in events
    )
    assert engine.pending_approval_tool_call_id() == "w2"
    assert engine.state is LoopState.AWAITING


@pytest.mark.asyncio
async def test_query_inference_fail_then_retry_two_ledger_rows(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    from protocore.contracts.llm import LLMTimeoutError

    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    rc = _on()
    engine = engine_factory(rc=rc)
    calls = {"n": 0}
    orig = llm.stream_with_tools

    async def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMTimeoutError("timeout")
        async for item in orig(request):
            yield item

    llm.stream_with_tools = flaky  # type: ignore[method-assign]
    llm.queue_response(text="ok")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")]))
    events = [evt async for evt in query(engine)]
    kinds = [evt.payload.get("kind") for evt in events if evt.type == EventType.USAGE_COMMITTED]
    assert "inference" in kinds
    assert "retry" in kinds
    assert [item.seq for item in engine.usage_rows] == [1, 2]
    assert session_total(engine.usage_rows) == sum(item.total_tokens for item in engine.usage_rows)


@pytest.mark.asyncio
async def test_query_resume_marks_recovery_span(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    rc = _on()
    engine = engine_factory(rc=rc)
    in_memory_runtime["tools"].register(MockTool(tool_name="Write", description="w"))
    llm.queue_tool_call_response(tool_call_id="w1", tool_name="Write", tool_input={"path": "f"})
    llm.queue_response(text="done")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="w")]))
    await _drain(query(engine))
    engine.open_intents = [
        commit_intent(tool_name="Write", tool_call_id="w9", rc=rc, arguments={"path": "f"})
    ]
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="w9", name="Write", arguments_json='{"path": "f"}')
            ],
        )
    )
    llm.queue_tool_call_response(tool_call_id="w1", tool_name="Write", tool_input={"path": "f"})
    llm.queue_response(text="recovered")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="again")]))
    engine.state = LoopState.PENDING
    more = [evt async for evt in query(engine)]
    assert any(evt.type == EventType.RECOVERY_MARKED for evt in more)
    assert any(getattr(span, "attributes", {}).get("recovery") is True for span in engine.spans)


@pytest.mark.asyncio
async def test_query_fires_published_hooks(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    rc = _on()
    engine = engine_factory(rc=rc)
    seen: list[str] = []
    registry = HookManager()
    for point in (
        HookEvent.run_start,
        HookEvent.turn_start,
        HookEvent.context_transform,
        HookEvent.request_prepare,
        HookEvent.response_received,
        HookEvent.turn_end,
        HookEvent.pre_tool_use,
        HookEvent.post_tool_use,
        HookEvent.run_finalize,
    ):
        registry.register(
            point,
            RegistrationKind.observe,
            lambda _ctx, hooked=point: _record_hook(seen, hooked.value),
            owner="test",
        )
    engine.lifecycle_hooks = registry
    in_memory_runtime["tools"].register(MockTool(tool_name="Read", description="r"))
    llm.queue_tool_call_response(tool_call_id="r1", tool_name="Read", tool_input={"path": "a"})
    llm.queue_response(text="ok")
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="r")]))
    events = [evt async for evt in query(engine)]
    fired = {evt.payload.get("hook") for evt in events if evt.type == EventType.HOOK_FIRED}
    assert "run_start" in fired
    assert "turn_start" in fired
    assert "context_transform" in fired
    assert "request_prepare" in fired
    assert "response_received" in fired
    assert "turn_end" in fired
    assert "pre_tool_use" in fired
    assert "post_tool_use" in fired
    assert "run_finalize" in fired


def _record_hook(seen: list[str], name: str) -> None:
    seen.append(name)
