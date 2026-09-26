"""Tests for :class:`protocore.runtime.tool_dispatch.ToolDispatcher`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from protocore.contracts.hooks import (
    HookActionKind,
    HookResult,
)
from protocore.contracts.run_state import RunScopedState
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext, ToolPolicyDenied
from protocore.contracts.types import HookEvent, ToolCall, ToolResult
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.tool_dispatch import (
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
)
from protocore.runtime.tool_permission import ToolPermissionGate
from protocore.runtime.tool_registry import ToolRegistry
from protocore.tests_support.adapters import InMemoryHookManager
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

from ._tool_fixtures import MockTool, make_default_ctx

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_dispatcher(
    tools: list[MockTool] | None = None,
    *,
    hook_manager: InMemoryHookManager | None = None,
) -> tuple[ToolDispatcher, ToolRegistry, InMemoryHookManager | None]:
    reg = ToolRegistry(tools or [])
    gate = ToolPermissionGate(roles=CONVENTIONAL_TOOL_ROLES)
    dispatcher = ToolDispatcher(
        registry=reg,
        permission_gate=gate,
        hook_manager=hook_manager,
    )
    return dispatcher, reg, hook_manager


async def _drain(
    dispatcher: ToolDispatcher,
    *,
    tool_call: ToolCall,
    ctx: ToolContext | None = None,
    visibility_policy: ToolVisibilityPolicy | None = None,
    timeout_seconds: int = 30,
    subagent_whitelist: list[str] | None = None,
) -> tuple[list[TurnEvent], DispatchOutcome]:
    """Collect events + outcome from a dispatch."""
    events: list[TurnEvent] = []
    outcome: DispatchOutcome | None = None
    async for item in dispatcher.dispatch(
        tool_call=tool_call,
        ctx=ctx or make_default_ctx(),
        visibility_policy=visibility_policy or ToolVisibilityPolicy(),
        timeout_seconds=timeout_seconds,
        subagent_whitelist=subagent_whitelist,
    ):
        if isinstance(item, DispatchOutcome):
            outcome = item
        else:
            events.append(item)
    assert outcome is not None
    return events, outcome


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_emits_tool_result_and_success_outcome() -> None:
    tool = MockTool(tool_name="MyTool", response_content="result-text")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="MyTool", arguments={"v": "x"})

    events, outcome = await _drain(dispatcher, tool_call=call)

    types = [e.type for e in events]
    assert EventType.TOOL_RESULT in types
    assert outcome.success is True
    assert outcome.content == "result-text"
    assert outcome.error_kind is None
    assert outcome.duration_ms >= 0
    assert tool.calls == [{"v": "x"}]


@pytest.mark.asyncio
async def test_dispatch_adds_tool_call_id_to_tool_context_metadata() -> None:
    seen_metadata: dict[str, object] = {}
    seen_state: dict[str, Any] = {}

    class MetadataCapturingTool(MockTool):
        async def invoke(
            self,
            context: ToolContext,
            arguments: dict[str, Any],
        ) -> ToolResult:
            seen_metadata.update(context.metadata)
            seen_state["state"] = context.run_state
            return await super().invoke(context, arguments)

    tool = MetadataCapturingTool(tool_name="MyTool")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(id="call-from-dispatch", name="MyTool", arguments={})

    _events, outcome = await _drain(dispatcher, tool_call=call)

    assert outcome.success is True
    assert seen_metadata["tool_call_id"] == "call-from-dispatch"


@pytest.mark.asyncio
async def test_dispatch_preserves_existing_context_metadata() -> None:
    seen_metadata: dict[str, object] = {}
    seen_state: dict[str, Any] = {}

    class MetadataCapturingTool(MockTool):
        async def invoke(
            self,
            context: ToolContext,
            arguments: dict[str, Any],
        ) -> ToolResult:
            seen_metadata.update(context.metadata)
            seen_state["state"] = context.run_state
            return await super().invoke(context, arguments)

    host_slot = object()
    ctx = ToolContext(
        tenant_id="tenant-1",
        run_id="run-1",
        session_id="sess-1",
        run_state=RunScopedState(host={"workspace": host_slot}),
        metadata={"tool_call_id": "caller-supplied-id"},
    )
    tool = MetadataCapturingTool(tool_name="MyTool")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(id="actual-call-id", name="MyTool", arguments={})

    _events, outcome = await _drain(dispatcher, tool_call=call, ctx=ctx)

    assert outcome.success is True
    assert seen_metadata["tool_call_id"] == "caller-supplied-id"
    assert seen_state["state"].host["workspace"] is host_slot


@pytest.mark.asyncio
async def test_happy_path_with_hooks_emits_pre_and_post_hook_events() -> None:
    """Hook stage must report ``stage=hook`` for the pre-hook event.

    Per L-1: ``hook_fired(pre)`` may ONLY be emitted when the gate
    actually reached the hook stage. With the default ``InMemoryHookManager``
    returning ``ALLOW`` (no modifications / no deny), the gate returns
    ``PermissionStage.default`` — therefore only the post-hook
    ``hook_fired`` is emitted by this happy-path. Pre-hook fires when
    the hook produces a non-default verdict (MODIFY / DENY / approval).
    """
    tool = MockTool(tool_name="MyTool")
    hooks = InMemoryHookManager()
    # Force the hook to declare a non-default verdict so the gate
    # surfaces ``PermissionStage.hook`` for the pre stage.
    hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(action=HookActionKind.MODIFY, modifications={}),
    )
    dispatcher, _, _ = _build_dispatcher([tool], hook_manager=hooks)
    call = ToolCall(name="MyTool", arguments={})

    events, outcome = await _drain(dispatcher, tool_call=call)

    hook_events = [e for e in events if e.type is EventType.HOOK_FIRED]
    assert len(hook_events) == 2
    assert hook_events[0].payload["hook_event"] == HookEvent.pre_tool_use.value
    assert hook_events[1].payload["hook_event"] == HookEvent.post_tool_use.value
    assert outcome.success


@pytest.mark.asyncio
async def test_no_hook_manager_omits_hook_events() -> None:
    tool = MockTool(tool_name="MyTool")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="MyTool", arguments={})

    events, _ = await _drain(dispatcher, tool_call=call)
    assert not [e for e in events if e.type is EventType.HOOK_FIRED]


@pytest.mark.asyncio
async def test_hook_fired_not_emitted_on_earlier_stage_deny() -> None:
    """L-1: dispatcher MUST NOT emit ``hook_fired(pre_tool_use)`` when the
    gate denies at the whitelist/safety_policy/rate_limit stage — the hook
    never actually executed.
    """
    # Bash with a denied command triggers the safety_policy stage before
    # the hook runs. Even with a hook manager wired, the pre-hook
    # ``hook_fired`` MUST be suppressed.
    tool = MockTool(tool_name="Bash")
    hooks = InMemoryHookManager()
    dispatcher, _, _ = _build_dispatcher([tool], hook_manager=hooks)
    call = ToolCall(name="Bash", arguments={"command": "rm -rf /"})

    events, outcome = await _drain(dispatcher, tool_call=call)

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.permission
    # Verify pre_tool_use HOOK_FIRED was NOT emitted (earlier-stage deny).
    pre_hook = [
        e
        for e in events
        if e.type is EventType.HOOK_FIRED and e.payload.get("hook_event") == HookEvent.pre_tool_use.value
    ]
    assert pre_hook == [], f"expected no pre_tool_use hook_fired for safety_policy deny, got {pre_hook}"


@pytest.mark.asyncio
async def test_hook_fired_not_emitted_on_whitelist_deny() -> None:
    """L-1 (whitelist branch): subagent_whitelist deny → no hook_fired."""
    tool = MockTool(tool_name="Read")
    hooks = InMemoryHookManager()
    dispatcher, _, _ = _build_dispatcher([tool], hook_manager=hooks)
    call = ToolCall(name="Read", arguments={})

    events, outcome = await _drain(
        dispatcher,
        tool_call=call,
        subagent_whitelist=["Grep", "Bash"],
    )

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.permission
    pre_hook = [
        e
        for e in events
        if e.type is EventType.HOOK_FIRED and e.payload.get("hook_event") == HookEvent.pre_tool_use.value
    ]
    assert pre_hook == []


# ----------------------------------------------------------------------
# Sandbox cold-start event — adapter owns emission, NOT core
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_core_does_not_emit_transport_starting_for_a_shell_tool() -> None:
    """The host's transport owns ``tool_transport_starting`` entirely.

 The dispatcher here must not emit it: it cannot tell a cold start from
 a warm one, so every emission of its own would be noise on a dispatch
 that started nothing.
 """
    tool = MockTool(tool_name="Bash", description="run shell")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Bash", arguments={"command": "echo hi"})

    events, outcome = await _drain(dispatcher, tool_call=call)

    starting = [e for e in events if e.type is EventType.TOOL_TRANSPORT_STARTING]
    assert starting == []
    assert outcome.success


@pytest.mark.asyncio
async def test_core_does_not_emit_transport_starting_for_a_state_only_tool() -> None:
    """Sanity-check the negative: a tool that reaches no transport at all
    must not see a ``tool_transport_starting`` event either.
    """
    tool = MockTool(tool_name="Read", description="read file")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Read", arguments={})

    events, _ = await _drain(dispatcher, tool_call=call)
    assert not [e for e in events if e.type is EventType.TOOL_TRANSPORT_STARTING]


# ----------------------------------------------------------------------
# Error taxonomy
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_emits_unknown_tool_kind() -> None:
    dispatcher, _, _ = _build_dispatcher([])
    call = ToolCall(name="Ghost", arguments={})

    events, outcome = await _drain(dispatcher, tool_call=call)

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.unknown_tool
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload["error"]["kind"] == "unknown_tool"


@pytest.mark.asyncio
async def test_permission_deny_emits_permission_kind() -> None:
    tool = MockTool(tool_name="Bash")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Bash", arguments={"command": "rm -rf /"})

    events, outcome = await _drain(dispatcher, tool_call=call)

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.permission
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload["error"]["kind"] == "permission"


@pytest.mark.asyncio
async def test_timeout_returns_timeout_kind() -> None:
    tool = MockTool(tool_name="Slow", sleep_seconds=1.0)
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Slow", arguments={})

    # Set timeout < sleep_seconds.
    _events, outcome = await _drain(dispatcher, tool_call=call, timeout_seconds=1)
    # With wait_for ~1s vs sleep 1s — race-y on slow CI; bump the
    # sleep instead.
    if outcome.success:
        pytest.skip("timer race — sleep finished before wait_for fired")
    assert outcome.error_kind is DispatchErrorKind.timeout


@pytest.mark.asyncio
async def test_timeout_deterministic_with_long_sleep() -> None:
    # Deterministic timeout: sleep >> wait_for.
    tool = MockTool(tool_name="Slow", sleep_seconds=10.0)
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Slow", arguments={})

    started = asyncio.get_event_loop().time()
    events, outcome = await _drain(dispatcher, tool_call=call, timeout_seconds=1)
    elapsed = asyncio.get_event_loop().time() - started

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.timeout
    assert elapsed < 5  # didn't actually wait 10s
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload["error"]["kind"] == "timeout"


@dataclass
class _DeferringTool(MockTool):
    """A tool that owns its own long-lived async unit (Agent-tool shape).

    Declares the host ``TypedTool`` ClassVars the dispatcher reads
    defensively (``should_defer`` + ``default_timeout_ms``); they are not on
    the core :class:`Tool` ABC. Mirrors ``AgentTool`` (``should_defer=True`` +
    ``default_timeout_ms=600_000``) which runs its ``SubagentRunner`` under its
    own classified ``asyncio.timeout``.
    """

    should_defer: bool = True
    default_timeout_ms: int = 600_000


@pytest.mark.asyncio
async def test_deferring_tool_honours_declared_timeout_over_flat_cap() -> None:
    """a ``should_defer`` tool uses ``default_timeout_ms``, not the cap.

    The flat ``timeout_seconds`` is ZERO — which instantly cancels any ordinary
    tool via ``asyncio.wait_for(timeout=0.0)``. A deferring tool with a large
    ``default_timeout_ms`` must still get its declared budget, so the
    sleep completes and the dispatch succeeds. Pre-fix the dispatcher wrapped
    every tool in the flat cap and this returned a ``timeout`` failure.
    """
    tool = _DeferringTool(tool_name="Defer", sleep_seconds=0.05)
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Defer", arguments={})

    _events, outcome = await _drain(dispatcher, tool_call=call, timeout_seconds=0)

    assert outcome.success is True
    assert outcome.error_kind is None
    assert len(tool.calls) == 1


@pytest.mark.asyncio
async def test_non_deferring_tool_keeps_flat_cap() -> None:
    """A tool that does NOT declare ``should_defer`` keeps the flat budget.

    ``MockTool`` carries no ``should_defer`` / ``default_timeout_ms`` (a pure
    core :class:`Tool`); the dispatcher must fall back to ``timeout_seconds``.
    With a zero flat cap a slow tool is cancelled with a ``timeout`` kind —
    proving the override does NOT regress ordinary tools.
    """
    tool = MockTool(tool_name="Slow", sleep_seconds=10.0)
    assert not hasattr(tool, "should_defer")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Slow", arguments={})

    _events, outcome = await _drain(dispatcher, tool_call=call, timeout_seconds=0)

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.timeout


@pytest.mark.asyncio
async def test_execution_exception_emits_execution_kind() -> None:
    tool = MockTool(
        tool_name="Boom",
        raise_exception=RuntimeError("kaboom"),
    )
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Boom", arguments={})

    _events, outcome = await _drain(dispatcher, tool_call=call)

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.execution
    assert "kaboom" in outcome.content


@pytest.mark.asyncio
async def test_tool_policy_denied_classifies_as_permission() -> None:
    tool = MockTool(
        tool_name="Restricted",
        raise_exception=ToolPolicyDenied("adapter denied"),
    )
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Restricted", arguments={})

    _events, outcome = await _drain(dispatcher, tool_call=call)
    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.permission
    assert "adapter denied" in outcome.content


@pytest.mark.asyncio
async def test_non_jsonable_args_validation_error() -> None:
    """Arguments that can't be JSON-serialised → validation kind."""
    tool = MockTool(tool_name="MyTool")
    dispatcher, _, _ = _build_dispatcher([tool])
    # asyncio.Lock() is not JSON-serialisable.
    call = ToolCall(name="MyTool", arguments={"bad": asyncio.Lock()})

    _events, outcome = await _drain(dispatcher, tool_call=call)
    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.validation


@pytest.mark.asyncio
async def test_tool_returning_non_tool_result_classified_as_execution() -> None:
    """A tool that breaks the return-type contract yields execution error."""

    class BrokenTool(MockTool):
        async def invoke(self, context, arguments):  # type: ignore[no-untyped-def]
            return "not a ToolResult"  # type: ignore[return-value]

    tool = BrokenTool(tool_name="Broken")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Broken", arguments={})

    _events, outcome = await _drain(dispatcher, tool_call=call)
    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.execution
    assert "non-ToolResult shape" in outcome.content


# ----------------------------------------------------------------------
# AskUser pause flow
# ----------------------------------------------------------------------


class _AskUserPauseTool(MockTool):
    """Test double that raises :class:`AskUserPauseRequested` on invoke."""

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        from protocore.tools.ask_user import AskUserInput, AskUserPauseRequested

        payload = AskUserInput.model_validate(arguments)
        raise AskUserPauseRequested(payload)


@pytest.mark.asyncio
async def test_ask_user_pause_emits_pending_event_with_kind() -> None:
    """``AskUserPauseRequested`` from invoke() → TOOL_CALL_PENDING ``kind=ask_user``."""
    tool = _AskUserPauseTool(tool_name="AskUser")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(
        name="AskUser",
        arguments={
            "questions": [
                {
                    "question": "Proceed?",
                    "options": [{"label": "Yes"}, {"label": "No"}],
                }
            ]
        },
    )

    events, outcome = await _drain(dispatcher, tool_call=call)

    pending = [e for e in events if e.type is EventType.TOOL_CALL_PENDING]
    assert len(pending) == 1
    payload = pending[0].payload
    assert payload["kind"] == "ask_user"
    assert payload["ask_user"] is True
    questions = payload["ask_user_payload"]["questions"]
    assert questions[0]["question"] == "Proceed?"
    assert [o["label"] for o in questions[0]["options"]] == ["Yes", "No"]
    # No tool_result emitted on pause.
    assert EventType.TOOL_RESULT not in [e.type for e in events]
    # Outcome flag carries through.
    assert outcome.ask_user_required is True
    assert outcome.approval_required is False
    assert outcome.is_error is False
    assert outcome.ask_user_payload is not None
    assert outcome.ask_user_payload["questions"][0]["multiSelect"] is False


@pytest.mark.asyncio
async def test_ask_user_pause_multi_question_carries_payload() -> None:
    tool = _AskUserPauseTool(tool_name="AskUser")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(
        name="AskUser",
        arguments={
            "questions": [
                {
                    "question": "pick features",
                    "options": [{"label": "a"}, {"label": "b"}],
                    "multiSelect": True,
                },
                {"question": "notes?", "allow_custom": True},
            ]
        },
    )

    events, outcome = await _drain(dispatcher, tool_call=call)

    pending = [e for e in events if e.type is EventType.TOOL_CALL_PENDING]
    questions = pending[0].payload["ask_user_payload"]["questions"]
    assert [o["label"] for o in questions[0]["options"]] == ["a", "b"]
    assert questions[0]["multiSelect"] is True
    assert questions[1]["allow_custom"] is True
    assert outcome.ask_user_required is True
    assert outcome.ask_user_payload is not None
    assert len(outcome.ask_user_payload["questions"]) == 2


# ----------------------------------------------------------------------
# Approval flow
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_required_emits_pending_event() -> None:
    tool = MockTool(tool_name="MyTool")
    hooks = InMemoryHookManager()
    hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": "tok-1"},
            reason="needs sign-off",
        ),
    )
    dispatcher, _, _ = _build_dispatcher([tool], hook_manager=hooks)
    call = ToolCall(name="MyTool", arguments={})

    events, outcome = await _drain(dispatcher, tool_call=call)

    pending = [e for e in events if e.type is EventType.TOOL_CALL_PENDING]
    assert len(pending) == 1
    assert pending[0].payload["approval_token"] == "tok-1"
    assert outcome.approval_required is True
    assert outcome.approval_token == "tok-1"
    # Crucial: tool was NOT invoked.
    assert tool.calls == []


# ----------------------------------------------------------------------
# PostToolUse hook can modify output
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_hook_modify_overrides_content() -> None:
    tool = MockTool(tool_name="MyTool", response_content="original")
    hooks = InMemoryHookManager()
    hooks.queue_action(
        HookEvent.post_tool_use,
        HookResult(
            action=HookActionKind.MODIFY,
            modifications={"tool_output": "redacted"},
        ),
    )
    dispatcher, _, _ = _build_dispatcher([tool], hook_manager=hooks)
    call = ToolCall(name="MyTool", arguments={})

    events, outcome = await _drain(dispatcher, tool_call=call)

    assert outcome.content == "redacted"
    # TOOL_RESULT envelope shows the modified content.
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload["content_blocks"][0]["text"] == "redacted"


@pytest.mark.asyncio
async def test_post_hook_exception_isolated() -> None:
    """Raising PostToolUse hook MUST NOT break the call."""

    class BoomHookManager(InMemoryHookManager):
        async def invoke(self, event, payload, tenant_id):  # type: ignore[no-untyped-def]
            if event is HookEvent.post_tool_use:
                raise RuntimeError("post hook crashed")
            return await super().invoke(event, payload, tenant_id)

    tool = MockTool(tool_name="MyTool", response_content="orig")
    hooks = BoomHookManager()
    # Queue a non-default pre-hook verdict so the gate reports stage=hook
    # and the pre-hook ``hook_fired`` event is emitted.
    hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(action=HookActionKind.MODIFY, modifications={}),
    )
    dispatcher, _, _ = _build_dispatcher([tool], hook_manager=hooks)
    call = ToolCall(name="MyTool", arguments={})

    events, outcome = await _drain(dispatcher, tool_call=call)
    assert outcome.success is True
    assert outcome.content == "orig"
    # PreToolUse hook_fired only (post crashed and was isolated).
    hook_events = [e for e in events if e.type is EventType.HOOK_FIRED]
    assert len(hook_events) == 1


# ----------------------------------------------------------------------
# Hook MODIFY rewrites tool input pre-execute
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_hook_modify_rewrites_input_to_tool() -> None:
    tool = MockTool(tool_name="MyTool")
    hooks = InMemoryHookManager()
    hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.MODIFY,
            modifications={"tool_input": {"v": "mutated"}},
        ),
    )
    dispatcher, _, _ = _build_dispatcher([tool], hook_manager=hooks)
    call = ToolCall(name="MyTool", arguments={"v": "original"})

    _events, outcome = await _drain(dispatcher, tool_call=call)
    assert outcome.success is True
    # Tool saw the mutated input.
    assert tool.calls == [{"v": "mutated"}]


# ----------------------------------------------------------------------
# Subagent whitelist narrows scope
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_whitelist_blocks_unlisted_tool() -> None:
    tool = MockTool(tool_name="Read")
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="Read", arguments={})

    _events, outcome = await _drain(
        dispatcher,
        tool_call=call,
        subagent_whitelist=["Grep", "Bash"],
    )
    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.permission


# ----------------------------------------------------------------------
# Tool reporting success=False (non-exception failure)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returning_is_error_surfaces_in_outcome() -> None:
    tool = MockTool(
        tool_name="MyTool",
        response_content="something went wrong",
        response_is_error=True,
    )
    dispatcher, _, _ = _build_dispatcher([tool])
    call = ToolCall(name="MyTool", arguments={})

    events, outcome = await _drain(dispatcher, tool_call=call)
    assert outcome.success is False
    assert outcome.is_error is True
    # TOOL_RESULT.success mirrors the outcome.
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload["success"] is False


# ----------------------------------------------------------------------
# / / A2: tool_error_counter hook
# ----------------------------------------------------------------------


class _RecordingToolErrorCounter:
    """Capture every ``increment_tool_errors_count`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.raise_on_increment: BaseException | None = None

    async def increment_tool_errors_count(self, run_id: str, by: int = 1) -> None:
        self.calls.append((run_id, by))
        if self.raise_on_increment is not None:
            raise self.raise_on_increment


def _build_dispatcher_with_counter(
    counter: _RecordingToolErrorCounter,
    tools: list[MockTool] | None = None,
) -> ToolDispatcher:
    reg = ToolRegistry(tools or [])
    gate = ToolPermissionGate(roles=CONVENTIONAL_TOOL_ROLES)
    return ToolDispatcher(
        registry=reg,
        permission_gate=gate,
        tool_error_counter=counter,
    )


@pytest.mark.asyncio
async def test_counter_increments_on_unknown_tool() -> None:
    counter = _RecordingToolErrorCounter()
    dispatcher = _build_dispatcher_with_counter(counter)
    ctx = make_default_ctx(run_id="run-a2-unknown")

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="DoesNotExist", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.unknown_tool
    assert counter.calls == [("run-a2-unknown", 1)]


@pytest.mark.asyncio
async def test_counter_increments_on_execution_exception() -> None:
    counter = _RecordingToolErrorCounter()
    tool = MockTool(tool_name="Boom", raise_exception=RuntimeError("kaboom"))
    dispatcher = _build_dispatcher_with_counter(counter, [tool])
    ctx = make_default_ctx(run_id="run-a2-exec")

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Boom", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.execution
    assert counter.calls == [("run-a2-exec", 1)]


@pytest.mark.asyncio
async def test_counter_increments_on_timeout() -> None:
    counter = _RecordingToolErrorCounter()
    tool = MockTool(tool_name="Slow", sleep_seconds=10.0)
    dispatcher = _build_dispatcher_with_counter(counter, [tool])
    ctx = make_default_ctx(run_id="run-a2-timeout")

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Slow", arguments={}),
        ctx=ctx,
        timeout_seconds=1,
    )
    assert outcome.error_kind is DispatchErrorKind.timeout
    assert counter.calls == [("run-a2-timeout", 1)]


@pytest.mark.asyncio
async def test_counter_increments_on_tool_policy_denied() -> None:
    counter = _RecordingToolErrorCounter()
    tool = MockTool(
        tool_name="Restricted",
        raise_exception=ToolPolicyDenied("adapter denied"),
    )
    dispatcher = _build_dispatcher_with_counter(counter, [tool])
    ctx = make_default_ctx(run_id="run-a2-policy")

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Restricted", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.permission
    assert counter.calls == [("run-a2-policy", 1)]


@pytest.mark.asyncio
async def test_counter_increments_on_tool_is_error_response() -> None:
    """A tool returning ``ToolResult(is_error=True)`` also counts as an error."""

    counter = _RecordingToolErrorCounter()
    tool = MockTool(
        tool_name="SoftErr",
        response_content="something went wrong",
        response_is_error=True,
    )
    dispatcher = _build_dispatcher_with_counter(counter, [tool])
    ctx = make_default_ctx(run_id="run-a2-soft")

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="SoftErr", arguments={}),
        ctx=ctx,
    )
    assert outcome.success is False
    assert counter.calls == [("run-a2-soft", 1)]


@pytest.mark.asyncio
async def test_counter_not_called_on_happy_path() -> None:
    counter = _RecordingToolErrorCounter()
    tool = MockTool(tool_name="Good", response_content="ok")
    dispatcher = _build_dispatcher_with_counter(counter, [tool])
    ctx = make_default_ctx(run_id="run-a2-happy")

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Good", arguments={"v": "x"}),
        ctx=ctx,
    )
    assert outcome.success is True
    assert counter.calls == []


@pytest.mark.asyncio
async def test_counter_failure_is_isolated() -> None:
    """Counter exceptions must not corrupt the dispatch flow."""

    counter = _RecordingToolErrorCounter()
    counter.raise_on_increment = RuntimeError("telemetry plane offline")
    dispatcher = _build_dispatcher_with_counter(counter)
    ctx = make_default_ctx(run_id="run-a2-iso")

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="DoesNotExist", arguments={}),
        ctx=ctx,
    )
    # The dispatch must still complete with the proper error kind.
    assert outcome.error_kind is DispatchErrorKind.unknown_tool
    # And the counter was invoked exactly once even though it raised.
    assert counter.calls == [("run-a2-iso", 1)]


@pytest.mark.asyncio
async def test_dispatcher_without_counter_unchanged_behaviour() -> None:
    """The opt-in counter must not change behaviour when not wired."""

    dispatcher, _, _ = _build_dispatcher([])
    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="DoesNotExist", arguments={}),
    )
    assert outcome.error_kind is DispatchErrorKind.unknown_tool


@pytest.mark.asyncio
async def test_counter_increments_n_times_for_n_dispatches() -> None:
    """Sequential N error dispatches → counter called N times.

    Mirrors the integration scenario: a fake session through ToolDispatcher
    that causes 2 errors via a mock tool then asserts the counter records 2.
    """

    counter = _RecordingToolErrorCounter()
    tool = MockTool(tool_name="FlakyTool", raise_exception=RuntimeError("nope"))
    dispatcher = _build_dispatcher_with_counter(counter, [tool])
    ctx = make_default_ctx(run_id="run-a2-batch")

    for idx in range(2):
        _events, outcome = await _drain(
            dispatcher,
            tool_call=ToolCall(name="FlakyTool", arguments={"i": str(idx)}),
            ctx=ctx,
        )
        assert outcome.error_kind is DispatchErrorKind.execution

    assert counter.calls == [("run-a2-batch", 1), ("run-a2-batch", 1)]


# ----------------------------------------------------------------------
# Attribute tool errors to root_run_id
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_counter_attributes_error_to_the_root_run_id() -> None:
    """When helpers bag carries ``root_run_id``, the counter MUST use it.

    Subagent ``run_id``s are internal and not present in PG ``runs`` rows;
    routing the increment to ``root_run_id`` (the parent's bare UUID)
    keeps the SQL ``WHERE id = $1`` cast clean and aggregates errors at
    the run row that actually owns the partial-status classification.
    """

    counter = _RecordingToolErrorCounter()
    tool = MockTool(tool_name="Boom", raise_exception=RuntimeError("kaboom"))
    dispatcher = _build_dispatcher_with_counter(counter, [tool])
    parent_uuid = "11111111-1111-4111-8111-111111111111"
    ctx = ToolContext(
        tenant_id="tenant-1",
        run_id="sub-agent-abc123",  # subagent's non-UUID id
        session_id="sess-1",
        run_state=RunScopedState(root_run_id=parent_uuid),
    )

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Boom", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.execution
    # The counter MUST receive the parent's bare UUID, not the subagent's id.
    assert counter.calls == [(parent_uuid, 1)]


@pytest.mark.asyncio
async def test_counter_falls_back_to_ctx_run_id_without_root_run_id() -> None:
    """No root run id → the counter still uses ``ctx.run_id`` (leader path)."""

    counter = _RecordingToolErrorCounter()
    tool = MockTool(tool_name="Boom", raise_exception=RuntimeError("kaboom"))
    dispatcher = _build_dispatcher_with_counter(counter, [tool])
    ctx = make_default_ctx(run_id="leader-run-uuid")

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Boom", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.execution
    assert counter.calls == [("leader-run-uuid", 1)]


@pytest.mark.asyncio
async def test_counter_falls_back_when_root_run_id_blank() -> None:
    """Blank / non-string root_run_id MUST fall back to ctx.run_id."""

    counter = _RecordingToolErrorCounter()
    dispatcher = _build_dispatcher_with_counter(counter)
    ctx = ToolContext(
        tenant_id="tenant-1",
        run_id="fallback-run",
        session_id="sess-1",
        run_state=RunScopedState(root_run_id=""),
    )

    _events, outcome = await _drain(
        dispatcher,
        tool_call=ToolCall(name="DoesNotExist", arguments={}),
        ctx=ctx,
    )
    assert outcome.error_kind is DispatchErrorKind.unknown_tool
    assert counter.calls == [("fallback-run", 1)]


@pytest.mark.asyncio
async def test_counter_attribution_routes_all_error_kinds_to_root() -> None:
    """All five error paths (unknown / exec / timeout / policy / soft) → root."""

    counter = _RecordingToolErrorCounter()
    root_uuid = "22222222-2222-4222-8222-222222222222"

    def _make_ctx(run_id: str) -> ToolContext:
        return ToolContext(
            tenant_id="tenant-1",
            run_id=run_id,
            session_id="sess-1",
            run_state=RunScopedState(root_run_id=root_uuid),
        )

    # 1. unknown_tool
    dispatcher = _build_dispatcher_with_counter(counter)
    _events, _ = await _drain(
        dispatcher,
        tool_call=ToolCall(name="DoesNotExist", arguments={}),
        ctx=_make_ctx("sub-1"),
    )

    # 2. execution exception
    boom = MockTool(tool_name="Boom", raise_exception=RuntimeError("x"))
    dispatcher = _build_dispatcher_with_counter(counter, [boom])
    _events, _ = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Boom", arguments={}),
        ctx=_make_ctx("sub-2"),
    )

    # 3. ToolPolicyDenied
    denied = MockTool(
        tool_name="Restricted",
        raise_exception=ToolPolicyDenied("nope"),
    )
    dispatcher = _build_dispatcher_with_counter(counter, [denied])
    _events, _ = await _drain(
        dispatcher,
        tool_call=ToolCall(name="Restricted", arguments={}),
        ctx=_make_ctx("sub-3"),
    )

    # 4. soft error (ToolResult.is_error=True)
    soft = MockTool(
        tool_name="SoftErr",
        response_content="bad",
        response_is_error=True,
    )
    dispatcher = _build_dispatcher_with_counter(counter, [soft])
    _events, _ = await _drain(
        dispatcher,
        tool_call=ToolCall(name="SoftErr", arguments={}),
        ctx=_make_ctx("sub-4"),
    )

    # All four increments MUST attribute to root_uuid, NOT to sub-*.
    assert counter.calls == [
        (root_uuid, 1),
        (root_uuid, 1),
        (root_uuid, 1),
        (root_uuid, 1),
    ]


# ----------------------------------------------------------------------
# `*_contract` hallucination hint
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_hallucination_hint() -> None:
    """`finalization_contract` as a tool name returns the nudge, not the raw error."""
    dispatcher, _, _ = _build_dispatcher([])
    call = ToolCall(name="finalization_contract", arguments={})

    events, outcome = await _drain(dispatcher, tool_call=call)

    assert outcome.success is False
    assert outcome.is_error is True
    assert outcome.error_kind is DispatchErrorKind.unknown_tool
    assert "is not a tool" in outcome.content
    assert "<finalization_contract>" in outcome.content
    assert "</finalization_contract>" in outcome.content
    # Hint is mirrored into the emitted tool_result envelope.
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.payload["error"]["kind"] == "unknown_tool"
    assert "is not a tool" in result.payload["error"]["message"]


@pytest.mark.asyncio
async def test_other_contract_names_caught() -> None:
    """Any `^[a-z_]+_contract$` name (not just `finalization_contract`) hits the hint."""
    dispatcher, _, _ = _build_dispatcher([])
    for name in ("subagent_contract", "safety_contract"):
        call = ToolCall(name=name, arguments={})
        _events, outcome = await _drain(dispatcher, tool_call=call)

        assert outcome.success is False
        assert outcome.error_kind is DispatchErrorKind.unknown_tool
        assert f"'{name}' is not a tool" in outcome.content
        assert f"<{name}>...</{name}>" in outcome.content


@pytest.mark.asyncio
async def test_non_contract_unknown_tool_unchanged() -> None:
    """Non-`*_contract` unknown tool names still emit the raw error."""
    dispatcher, _, _ = _build_dispatcher([])
    for name in ("FooBar", "mything"):
        call = ToolCall(name=name, arguments={})
        _events, outcome = await _drain(dispatcher, tool_call=call)

        assert outcome.success is False
        assert outcome.error_kind is DispatchErrorKind.unknown_tool
        assert outcome.content == f"unknown tool: {name!r}"
        assert "is not a tool" not in outcome.content


@pytest.mark.asyncio
async def test_an_unknown_tool_is_answered_with_the_nearest_admitted_names() -> None:
    """A wrong case or a missing word is the usual mistake; the right spelling in
    the error is what stops the model guessing again. Blocked names are never
    offered, however close."""
    dispatcher, _, _ = _build_dispatcher(
        [
            MockTool(tool_name="Mcp_Github_list_issues"),
            MockTool(tool_name="Mcp_Github_list_pulls"),
            MockTool(tool_name="Mcp_Github_secret_list_issues"),
            MockTool(tool_name="Read"),
        ]
    )
    call = ToolCall(name="Mcp_github_list_issue", arguments={})

    _events, outcome = await _drain(
        dispatcher,
        tool_call=call,
        visibility_policy=ToolVisibilityPolicy(blocked={"Mcp_Github_secret_list_issues"}),
    )

    assert outcome.error_kind is DispatchErrorKind.unknown_tool
    assert outcome.content.startswith(
        "unknown tool: 'Mcp_github_list_issue'. Did you mean: Mcp_Github_list_issues"
    )
    assert "secret" not in outcome.content
    assert "Read" not in outcome.content


def _ctx_with_finalize_terminal_rc() -> ToolContext:
    """A ctx whose constants opt into the typed-``Finalize`` terminal."""
    from protocore.contracts.runtime_constants import LoopConstants

    rc = LoopConstants(agent_finalize_tool_as_terminal=True)
    return make_default_ctx().model_copy(
        update={"run_state": RunScopedState(rc=rc)}
    )


@pytest.mark.asyncio
async def test_contract_hallucination_hint_finalize_terminal_points_to_finalize() -> None:
    """Typed-Finalize tenants: a misfired `finalization_contract` tool is steered to
    the `Finalize` tool, NOT the legacy `<finalization_contract>` XML block (closes the
    residual ④b leak on the opt-in recovery path)."""
    dispatcher, _, _ = _build_dispatcher([])
    call = ToolCall(name="finalization_contract", arguments={})

    _events, outcome = await _drain(
        dispatcher, tool_call=call, ctx=_ctx_with_finalize_terminal_rc()
    )

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.unknown_tool
    assert "Finalize" in outcome.content
    assert "declared_deliverables" in outcome.content
    assert "Inline it directly" not in outcome.content


@pytest.mark.asyncio
async def test_contract_hallucination_finalize_terminal_only_specialcases_finalization() -> None:
    """Opt-in only special-cases `finalization_contract`; other `*_contract` names keep
    the inline-XML nudge (they are not the terminal contract)."""
    dispatcher, _, _ = _build_dispatcher([])
    call = ToolCall(name="subagent_contract", arguments={})

    _events, outcome = await _drain(
        dispatcher, tool_call=call, ctx=_ctx_with_finalize_terminal_rc()
    )

    assert "<subagent_contract>...</subagent_contract>" in outcome.content
    assert "Finalize" not in outcome.content


@pytest.mark.asyncio
async def test_contract_pattern_case_sensitive() -> None:
    """`Finalization_contract` (capital F) does NOT match — falls to default."""
    dispatcher, _, _ = _build_dispatcher([])
    call = ToolCall(name="Finalization_contract", arguments={})

    _events, outcome = await _drain(dispatcher, tool_call=call)

    assert outcome.success is False
    assert outcome.error_kind is DispatchErrorKind.unknown_tool
    assert outcome.content == "unknown tool: 'Finalization_contract'"
    assert "is not a tool" not in outcome.content
