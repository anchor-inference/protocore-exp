"""``post_tool_use`` sees every result the model will read, failures included.

A redaction hook exists to remove what must not reach the transcript, and an
exception message is where such text turns up most often: a command line
carrying a token, a URL with credentials, a server echoing the key it refused.
The dispatcher used to answer a raised tool, a timeout, an unreadable result
and a refused call without asking the hook, so the failure text went into the
history, the events and the next request exactly as the tool wrote it. These
tests pin the contract that closes that hole: the hook fires once for every
call that produces a result, with ``is_error`` and ``error_kind`` saying what
kind of result it is, and a ``modify`` from it lands everywhere the text goes.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import pytest

from protocore.contracts.hooks import HookActionKind, HookResult
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    HookEvent,
    Message,
    MessageRole,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolResultBlock,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.tool_dispatch import (
    DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY,
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
)
from protocore.runtime.tool_permission import ToolPermissionGate
from protocore.runtime.tool_registry import ToolRegistry
from protocore.tests_support.adapters import InMemoryHookManager
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

from ._tool_fixtures import MockTool, make_default_ctx

_SECRET = "sk-test-4f9a2c7e1b"
_SECRET_PATTERN = re.compile(r"sk-test-[A-Za-z0-9]+")
_MASK = "[masked]"


class _RedactingHookManager(InMemoryHookManager):
    """Masks the fake key in every ``post_tool_use`` output it is shown."""

    async def invoke(
        self,
        event: HookEvent,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> HookResult:
        await super().invoke(event, payload, tenant_id)
        if event is not HookEvent.post_tool_use:
            return HookResult(action=HookActionKind.ALLOW)
        output = str(payload.get("tool_output", ""))
        masked = _SECRET_PATTERN.sub(_MASK, output)
        if masked == output:
            return HookResult(action=HookActionKind.ALLOW)
        return HookResult(
            action=HookActionKind.MODIFY,
            modifications={"tool_output": masked},
        )

    def post_calls(self) -> list[dict[str, Any]]:
        return [
            payload
            for event, payload, _tenant in self.invocations
            if event is HookEvent.post_tool_use
        ]


class _RaisingHookManager(InMemoryHookManager):
    async def invoke(
        self,
        event: HookEvent,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> HookResult:
        await super().invoke(event, payload, tenant_id)
        if event is HookEvent.post_tool_use:
            raise RuntimeError("redaction service unavailable")
        return HookResult(action=HookActionKind.ALLOW)


class _SlowDeferringTool(MockTool):
    """Declares its own short budget, so a timeout costs milliseconds."""

    should_defer = True
    default_timeout_ms = 50


class _ShapelessTool(MockTool):
    async def invoke(  # type: ignore[override]
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> Any:
        del context, arguments
        return {"token": _SECRET}


class _ArgumentCheckingTool(MockTool):
    """Rejects its argument the way a pydantic model would: by echoing it."""

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        del context
        raise ValueError(f"field 'v' is not a valid identifier: {arguments['v']!r}")


class _SecretResolvingTool(_ArgumentCheckingTool):
    """Expands a reference in its argument before rejecting it, as a shell would."""

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        return await super().invoke(context, {"v": str(arguments["v"]).replace("$API_KEY", _SECRET)})


class _ParallelRaisingRead(MockTool):
    is_concurrent_safe = True
    is_destructive = False

    async def invoke(
        self, context: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        del context
        await asyncio.sleep(float(arguments.get("delay", 0.0)))
        if arguments.get("fail"):
            raise RuntimeError(f"GET https://api.example/v1?key={_SECRET} -> 401")
        return ToolResult(tool_call_id="", content="ok", is_error=False)


def _dispatcher(
    tools: list[MockTool], hooks: InMemoryHookManager
) -> ToolDispatcher:
    return ToolDispatcher(
        registry=ToolRegistry(tools),
        permission_gate=ToolPermissionGate(roles=CONVENTIONAL_TOOL_ROLES),
        hook_manager=hooks,
    )


async def _drain(
    dispatcher: ToolDispatcher,
    call: ToolCall,
    *,
    visibility_policy: ToolVisibilityPolicy | None = None,
) -> tuple[list[TurnEvent], DispatchOutcome]:
    events: list[TurnEvent] = []
    outcome: DispatchOutcome | None = None
    async for item in dispatcher.dispatch(
        tool_call=call,
        ctx=make_default_ctx(),
        visibility_policy=visibility_policy or ToolVisibilityPolicy(),
        timeout_seconds=30,
    ):
        if isinstance(item, DispatchOutcome):
            outcome = item
        else:
            events.append(item)
    assert outcome is not None
    return events, outcome


def _dump(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, default=str)


def _assert_secret_absent(events: list[TurnEvent], outcome: DispatchOutcome) -> None:
    for event in events:
        assert _SECRET not in _dump(event.payload), event.type
    assert _SECRET not in outcome.content
    assert _SECRET not in _dump(outcome.metadata or {})


# ----------------------------------------------------------------------
# The dispatcher: every failure path asks the hook, once
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_raised_secret_is_masked_in_the_outcome_and_every_event() -> None:
    hooks = _RedactingHookManager()
    tool = MockTool(
        tool_name="Fetch",
        raise_exception=RuntimeError(f"curl -H 'Authorization: Bearer {_SECRET}' failed"),
    )
    events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="Fetch", arguments={}))

    assert outcome.is_error is True
    assert outcome.error_kind is DispatchErrorKind.execution
    assert _MASK in outcome.content
    _assert_secret_absent(events, outcome)
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload["error"]["message"] == outcome.content
    assert result.payload["content_blocks"][0]["text"] == outcome.content

    (payload,) = hooks.post_calls()
    assert payload["is_error"] is True
    assert payload["success"] is False
    assert payload["error_kind"] == "execution"
    assert _SECRET in payload["tool_output"]
    hook_fired = [e for e in events if e.type is EventType.HOOK_FIRED]
    assert [e.payload["outcome"] for e in hook_fired] == ["modify"]
    # The fired notice comes before the result, as it does on success.
    assert events.index(hook_fired[0]) < events.index(result)


@pytest.mark.asyncio
async def test_a_timeout_reaches_the_hook_with_its_kind() -> None:
    hooks = InMemoryHookManager()
    hooks.queue_action(
        HookEvent.post_tool_use,
        HookResult(action=HookActionKind.MODIFY, modifications={"tool_output": "took too long"}),
    )
    tool = _SlowDeferringTool(tool_name="Slow", sleep_seconds=5.0)
    events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="Slow", arguments={}))

    assert outcome.error_kind is DispatchErrorKind.timeout
    assert outcome.content == "took too long"
    post = [p for e, p, _ in hooks.invocations if e is HookEvent.post_tool_use]
    assert len(post) == 1
    assert post[0]["error_kind"] == "timeout"
    assert "timed out" in post[0]["tool_output"]
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload["error"] == {"kind": "timeout", "message": "took too long"}


@pytest.mark.asyncio
async def test_an_argument_error_echoing_the_secret_is_masked() -> None:
    hooks = _RedactingHookManager()
    tool = _ArgumentCheckingTool(tool_name="Check")
    events, outcome = await _drain(
        _dispatcher([tool], hooks), ToolCall(name="Check", arguments={"v": _SECRET})
    )

    assert outcome.error_kind is DispatchErrorKind.execution
    _assert_secret_absent(events, outcome)
    assert [p["error_kind"] for p in hooks.post_calls()] == ["execution"]


@pytest.mark.asyncio
async def test_an_unreadable_result_shape_reaches_the_hook() -> None:
    hooks = InMemoryHookManager()
    tool = _ShapelessTool(tool_name="Shapeless")
    _events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="Shapeless", arguments={}))

    assert outcome.error_kind is DispatchErrorKind.execution
    post = [p for e, p, _ in hooks.invocations if e is HookEvent.post_tool_use]
    assert len(post) == 1
    assert "non-ToolResult" in post[0]["tool_output"]


@pytest.mark.asyncio
async def test_arguments_too_deep_to_serialise_reach_the_hook_as_validation() -> None:
    hooks = InMemoryHookManager()
    tool = MockTool(tool_name="Deep")
    nested: dict[str, Any] = {}
    cursor = nested
    for _ in range(300):
        cursor["n"] = {}
        cursor = cursor["n"]
    _events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="Deep", arguments=nested))

    assert outcome.error_kind is DispatchErrorKind.validation
    post = [p for e, p, _ in hooks.invocations if e is HookEvent.post_tool_use]
    assert [p["error_kind"] for p in post] == ["validation"]
    assert tool.calls == []


@pytest.mark.asyncio
async def test_an_unknown_tool_and_a_refused_call_reach_the_hook() -> None:
    hooks = InMemoryHookManager()
    tool = MockTool(tool_name="Known")
    dispatcher = _dispatcher([tool], hooks)

    _e1, unknown = await _drain(dispatcher, ToolCall(id="c-unknown", name="Missing", arguments={}))
    _e2, refused = await _drain(
        dispatcher,
        ToolCall(id="c-refused", name="Known", arguments={}),
        visibility_policy=ToolVisibilityPolicy(blocked={"Known"}),
    )

    assert unknown.error_kind is DispatchErrorKind.unknown_tool
    assert refused.error_kind is DispatchErrorKind.permission
    post = {
        p["tool_call_id"]: p["error_kind"]
        for e, p, _ in hooks.invocations
        if e is HookEvent.post_tool_use
    }
    assert post == {"c-unknown": "unknown_tool", "c-refused": "permission"}
    assert tool.calls == []


@pytest.mark.asyncio
async def test_a_success_says_it_is_not_an_error() -> None:
    hooks = InMemoryHookManager()
    tool = MockTool(tool_name="Fine", response_content="all good")
    await _drain(_dispatcher([tool], hooks), ToolCall(name="Fine", arguments={}))

    (payload,) = [p for e, p, _ in hooks.invocations if e is HookEvent.post_tool_use]
    assert payload["is_error"] is False
    assert payload["error_kind"] is None


@pytest.mark.asyncio
async def test_a_hook_that_raises_on_a_failure_leaves_the_failure_standing() -> None:
    hooks = _RaisingHookManager()
    tool = MockTool(tool_name="Boom", raise_exception=RuntimeError("disk full"))
    events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="Boom", arguments={}))

    assert outcome.is_error is True
    assert outcome.error_kind is DispatchErrorKind.execution
    assert "disk full" in outcome.content
    # Isolated as on success: no fired notice, one result, one attempt.
    assert [e for e in events if e.type is EventType.HOOK_FIRED] == []
    assert len([e for e in events if e.type is EventType.TOOL_RESULT]) == 1
    assert len([1 for e, _p, _t in hooks.invocations if e is HookEvent.post_tool_use]) == 1


@pytest.mark.asyncio
async def test_the_raw_failure_text_rides_only_in_memory_once_masked() -> None:
    """The replay keeps the unmasked text for its streak, never the metadata.

    The parallel path recomputes the consecutive-error streak from the text the
    serial path keyed it on, so it needs the tool's own words. Kept in the
    outcome's metadata they would follow the result into the transcript and
    every snapshot; kept on the outcome object they die with the turn.
    """
    hooks = _RedactingHookManager()
    tool = MockTool(tool_name="Fetch", raise_exception=RuntimeError(f"key={_SECRET}"))
    _events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="Fetch", arguments={}))

    assert outcome.replay_message is not None
    assert _SECRET in outcome.replay_message
    metadata = outcome.metadata or {}
    assert metadata[DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY] == outcome.content
    assert _SECRET not in _dump(metadata)


@pytest.mark.asyncio
async def test_a_tool_error_result_masked_by_the_hook_leaves_no_raw_copy() -> None:
    hooks = _RedactingHookManager()
    tool = MockTool(
        tool_name="Login",
        response_content=f"401: key {_SECRET} revoked",
        response_is_error=True,
    )
    events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="Login", arguments={}))

    assert outcome.is_error is True
    (payload,) = hooks.post_calls()
    assert payload["is_error"] is True
    assert payload["error_kind"] == "execution"
    _assert_secret_absent(events, outcome)
    assert outcome.replay_message == f"401: key {_SECRET} revoked"


@pytest.mark.asyncio
async def test_an_unmodified_failure_keeps_its_metadata_as_before() -> None:
    hooks = InMemoryHookManager()
    tool = MockTool(tool_name="Boom", raise_exception=RuntimeError("plain"))
    _events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="Boom", arguments={}))

    assert outcome.replay_message is None
    assert (outcome.metadata or {})[DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY] == (
        "tool 'Boom' execution failed: plain"
    )


# ----------------------------------------------------------------------
# The loop: the masked text is all the transcript and the model ever see
# ----------------------------------------------------------------------


def _history_dump(engine: Any) -> str:
    return "\n".join(_dump(message) for message in engine.history)


@pytest.mark.asyncio
async def test_a_raised_secret_never_reaches_history_events_or_the_next_request(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    hooks = _RedactingHookManager()
    engine.hooks = hooks
    in_memory_runtime["tools"].register(
        MockTool(tool_name="Fetch", raise_exception=RuntimeError(f"token={_SECRET} rejected"))
    )
    llm = in_memory_runtime["llm"]
    llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Fetch", tool_input={})
    llm.queue_response(text="the call failed")

    events: list[TurnEvent] = []
    async for evt in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        events.append(evt)

    results = [b for m in engine.history for b in m.content_blocks if isinstance(b, ToolResultBlock)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert _MASK in results[0].content
    assert _SECRET not in _history_dump(engine)
    for event in events:
        assert _SECRET not in _dump(event.payload), event.type
    assert len(llm.calls) == 2
    assert _SECRET not in _dump(llm.calls[1])
    assert _MASK in _dump(llm.calls[1])
    assert [p["tool_call_id"] for p in hooks.post_calls()] == ["call-1"]


@pytest.mark.asyncio
async def test_a_timeout_modified_by_the_hook_is_what_the_history_keeps(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    hooks = InMemoryHookManager()
    hooks.queue_action(
        HookEvent.post_tool_use,
        HookResult(action=HookActionKind.MODIFY, modifications={"tool_output": "slow tool, try later"}),
    )
    engine.hooks = hooks
    in_memory_runtime["tools"].register(_SlowDeferringTool(tool_name="Slow", sleep_seconds=5.0))
    llm = in_memory_runtime["llm"]
    llm.queue_tool_call_response(tool_call_id="call-slow", tool_name="Slow", tool_input={})
    llm.queue_response(text="ok")

    async for _evt in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    results = [b for m in engine.history for b in m.content_blocks if isinstance(b, ToolResultBlock)]
    assert [r.content for r in results] == ["slow tool, try later"]
    assert "timed out" not in _dump(llm.calls[1])


@pytest.mark.asyncio
async def test_a_bad_argument_secret_never_reaches_the_next_request(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    engine.hooks = _RedactingHookManager()
    in_memory_runtime["tools"].register(_SecretResolvingTool(tool_name="Check"))
    llm = in_memory_runtime["llm"]
    # The model's own call carries no secret; only the tool's echo would.
    llm.queue_tool_call_response(tool_call_id="call-arg", tool_name="Check", tool_input={"v": "$API_KEY"})
    llm.queue_response(text="ok")

    async for _evt in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        pass

    assert _SECRET not in _history_dump(engine)
    assert _SECRET not in _dump(llm.calls[1])


@pytest.mark.asyncio
async def test_a_parallel_failure_is_masked_and_asked_about_once(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    hooks = _RedactingHookManager()
    engine.hooks = hooks
    in_memory_runtime["tools"].register(_ParallelRaisingRead(tool_name="Read"))
    llm = in_memory_runtime["llm"]
    from tests.unit.runtime.test_query_parallel_safe_tools import _queue_multi_tool_stream

    _queue_multi_tool_stream(
        llm,
        tool_calls=[
            # The success finishes last but stands first, so the replay has to
            # put the streak back in the order the model asked for the calls.
            ("call-a", "Read", {"delay": 0.05}),
            ("call-b", "Read", {"fail": True, "delay": 0.0}),
        ],
    )
    llm.queue_response(text="done")

    events: list[TurnEvent] = []
    async for evt in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        events.append(evt)

    results = {
        b.tool_call_id: b
        for m in engine.history
        for b in m.content_blocks
        if isinstance(b, ToolResultBlock)
    }
    assert results["call-b"].is_error is True
    assert _MASK in results["call-b"].content
    assert _SECRET not in _history_dump(engine)
    for event in events:
        assert _SECRET not in _dump(event.payload), event.type
    assert _SECRET not in _dump(llm.calls[1])
    assert sorted(p["tool_call_id"] for p in hooks.post_calls()) == ["call-a", "call-b"]
    # The streak was keyed on the tool's own words, as the serial path keys it.
    streak = engine.run_state.consecutive_error
    assert streak is not None and streak.tool_name == "Read"
    assert streak.signature == ToolDispatcher._error_signature(
        DispatchErrorKind.execution,
        f"tool 'Read' execution failed: GET https://api.example/v1?key={_SECRET} -> 401",
        "Read",
        roles=engine.config.tool_roles,
    )


# ----------------------------------------------------------------------
# A rewritten success: the original leaves no copy outside the process
# ----------------------------------------------------------------------


def _durable_dump(engine: Any, runtime: dict[str, Any]) -> str:
    """Everything a host would persist or send: history, snapshots, blobs."""
    stream = runtime["events"]
    blobs = runtime["blobs"]
    parts = [_history_dump(engine), _dump(engine.snapshot())]
    parts.extend(_dump(e) for events in stream._streams.values() for e in events)
    parts.extend(data.decode("utf-8", "replace") for data, _meta in blobs._store.values())
    return "\n".join(parts)


async def _run_one_call(engine: Any, runtime: dict[str, Any], tool: MockTool) -> list[TurnEvent]:
    runtime["tools"].register(tool)
    llm = runtime["llm"]
    llm.queue_tool_call_response(tool_call_id="call-ok", tool_name=tool.tool_name, tool_input={})
    llm.queue_response(text="ok")
    events: list[TurnEvent] = []
    async for evt in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])):
        events.append(evt)
    return events


def _result_block(engine: Any) -> ToolResultBlock:
    (block,) = [b for m in engine.history for b in m.content_blocks if isinstance(b, ToolResultBlock)]
    return block


@pytest.mark.asyncio
async def test_a_masked_success_leaves_no_original_in_history_snapshots_or_the_request(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    engine.hooks = _RedactingHookManager()
    tool = MockTool(
        tool_name="Env",
        response_content=f"OPENAI_API_KEY={_SECRET}",
        response_ui_payload={"rows": [["OPENAI_API_KEY", _SECRET]]},
    )
    events = await _run_one_call(engine, in_memory_runtime, tool)

    block = _result_block(engine)
    assert block.content == f"OPENAI_API_KEY={_MASK}"
    assert block.canonical_content is None
    assert _SECRET not in _durable_dump(engine, in_memory_runtime)
    for event in events:
        assert _SECRET not in _dump(event.payload), event.type
    assert _SECRET not in _dump(in_memory_runtime["llm"].calls[1])


@pytest.mark.asyncio
async def test_a_masked_over_limit_success_stores_only_the_masked_text(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    engine.hooks = _RedactingHookManager()
    long_output = f"token={_SECRET}\n" + ("a line of build output\n" * 3_000)
    tool = MockTool(tool_name="Build", response_content=long_output)
    await _run_one_call(engine, in_memory_runtime, tool)

    block = _result_block(engine)
    # The result was cut on entry and the whole value offloaded to a blob.
    assert block.canonical_ref is not None
    stored = [data.decode() for data, _meta in in_memory_runtime["blobs"]._store.values()]
    assert any(_MASK in text for text in stored)
    assert _SECRET not in _durable_dump(engine, in_memory_runtime)
    assert _SECRET not in _dump(in_memory_runtime["llm"].calls[1])


@pytest.mark.asyncio
async def test_a_rewritten_result_drops_a_reference_the_hook_could_not_mask() -> None:
    hooks = _RedactingHookManager()
    tool = MockTool(
        tool_name="Dump",
        response_content=f"key={_SECRET}",
        response_canonical_ref="blob://tool-wrote-this-itself",
        response_ui_payload={"raw": _SECRET},
        response_path="config/.env",
    )
    events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="Dump", arguments={}))

    assert outcome.content == f"key={_MASK}"
    assert outcome.canonical_content is None
    assert outcome.canonical_ref is None
    assert outcome.ui_payload is None
    # Where the result came from is not its content, and stays.
    assert outcome.path == "config/.env"
    _assert_secret_absent(events, outcome)
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert "canonical_ref" not in result.payload
    assert "ui_payload" not in result.payload


@pytest.mark.asyncio
async def test_a_hook_that_allows_leaves_the_canonical_value_as_before() -> None:
    hooks = InMemoryHookManager()
    tool = MockTool(
        tool_name="List",
        response_content="line 1\nline 2\nline 3",
        response_projection="line 1 (2 more)",
        response_canonical_ref="blob://listing",
        response_ui_payload={"rows": 3},
    )
    _events, outcome = await _drain(_dispatcher([tool], hooks), ToolCall(name="List", arguments={}))

    assert outcome.content == "line 1 (2 more)"
    assert outcome.canonical_content == "line 1\nline 2\nline 3"
    assert outcome.canonical_ref == "blob://listing"
    assert outcome.ui_payload == {"rows": 3}


@pytest.mark.asyncio
async def test_an_unmodified_success_keeps_its_canonical_value_in_history(
    engine_factory, in_memory_runtime
) -> None:
    engine = engine_factory()
    tool = MockTool(
        tool_name="List",
        response_content="line 1\nline 2\nline 3",
        response_projection="line 1 (2 more)",
    )
    await _run_one_call(engine, in_memory_runtime, tool)

    block = _result_block(engine)
    assert block.content == "line 1 (2 more)"
    assert block.canonical_content == "line 1\nline 2\nline 3"
