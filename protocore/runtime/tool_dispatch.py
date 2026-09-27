"""``ToolDispatcher`` — the 7-step dispatch flow.

The dispatcher is the **single core entry point** for executing one
:class:`~protocore.contracts.types.ToolCall` AFTER the LLM has finished
emitting it. Concerns:

* Registry lookup — :class:`DispatchErrorKind.unknown_tool` if missing.
* Permission gate — fans out to :class:`ToolPermissionGate`
 (whitelist → safety policies → ``PreToolUse`` hook).
* Approval requirement — surfaces ``tool_call_pending`` event; emits
 no tool_result (caller transitions engine to AWAITING).
* Timeout-wrapped execute — :func:`asyncio.wait_for` honouring
 ``rc.tool_timeout_seconds`` by default, or the tool's own declared
 ``default_timeout_ms`` when it sets ``should_defer=True`` (a tool that
 runs its own long-lived async unit with an internal classified timeout —
 the ``Agent`` tool / ``SubagentRunner``). See
 :func:`_resolve_tool_timeout_seconds`.
* Event emission — emits ``hook_fired`` (pre/post) and the terminal
 ``tool_result``. **Does NOT emit** ``tool_use_start`` /
 ``tool_use_stop`` / ``tool_use_input_delta`` — those are LLM-stream
 events already produced by the assistant-message stream in
 :func:`protocore.runtime.query._stream_one_assistant_message`.
 **Does NOT emit** ``tool_transport_starting`` — the host's
 transport owns that event entirely, and emits it only on **cold
 start**, with a payload naming the transport and describing how it
 was brought up. The dispatcher here cannot tell a cold start from a
 warm one, so any emission of its own would be double-emit noise.
* ``PostToolUse`` hook — fire-and-await; may rewrite the output.

The dispatcher is **agnostic to the tool implementation** — it depends
only on :class:`~protocore.contracts.tools.Tool.invoke`. Tests inject a
``MockTool`` via :class:`InMemoryToolRegistry`; no the host plumbing
required.

Output: every dispatch yields a stream of :class:`TurnEvent` envelopes
AND returns a final :class:`DispatchOutcome` summarising the call.
The caller (the turn driver) forwards events to the SSE stream and uses the
outcome to mutate engine state (append tool result to history,
transition to ``AWAITING`` on approval).
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import hashlib
import json
import re
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from protocore.constants import MAX_DATA_NESTING_DEPTH
from protocore.contracts.evidence import (
    EvidenceProducerBinding,
    EvidenceRecord,
    ToolEvidenceContext,
)
from protocore.contracts.hooks import HookActionKind, IHookManager
from protocore.contracts.middleware import (
    ILifecycleRegistry,
    LifecycleContext,
    LifecycleVerdict,
)
from protocore.contracts.resilience import (
    TRANSPORT_DOWN_ERROR_CLASSES,
    IResilienceClassifier,
)
from protocore.contracts.run import IRunToolErrorCounter
from protocore.contracts.run_state import (
    ConsecutiveErrorStreak,
    RunScopedState,
    SignatureStreak,
    ToolStreak,
)
from protocore.contracts.tool_registry import (
    ADVERTISED_TOOLS_METADATA_KEY,
    TOOL_ALLOWLIST_METADATA_KEY,
    TOOL_VISIBILITY_POLICY_METADATA_KEY,
    IToolRegistry,
    ToolVisibilityPolicy,
    policy_admits,
)
from protocore.contracts.tool_roles import EMPTY_TOOL_ROLE_MAP, ToolRole, ToolRoleMap
from protocore.contracts.tools import (
    Tool,
    ToolContext,
    ToolPolicyDenied,
    copy_metadata,
    read_metadata,
)
from protocore.contracts.types import (
    TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY,
    TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY,
    HookEvent,
    ToolCall,
    ToolResult,
)
from protocore.logging_utils import get_logger
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.tool_permission import (
    PermissionStage,
    ToolPermissionDecision,
    ToolPermissionGate,
    ToolPermissionOutcome,
)
from protocore.runtime.tool_preconditions import (
    check_preconditions,
    record_satisfaction,
)
from protocore.runtime.tool_retrieval import tool_line
from protocore.tools.ask_user import AskUserPauseRequested

_logger = get_logger(__name__)


def _admission_deferred(ctx: ToolContext) -> bool:
    """Whether a parallel replay, rather than this dispatch, admits evidence.

    An invocation carrying no evidence context defers nothing: it produces no
    evidence, so there is nothing for a replay to order.
    """
    return ctx.evidence is not None and ctx.evidence.admission_deferred


def _metadata_flag(
    metadata: dict[str, Any] | None, key: str, *, default: bool
) -> bool:
    """Read a strict-boolean flag from a ToolResult metadata bag.

 Only an actual ``bool`` value overrides ``default`` — any other type
 (or a missing key) yields ``default``. Used by the soft-error path to
 honour a tool's ``count_as_tool_error`` / ``consecutive_error_cap_eligible``
 classification without trusting arbitrary truthy values that a hook
 might have injected into the bag.
 """
    if not metadata:
        return default
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return value
    return default


def _resolve_max_data_nesting_depth(ctx: ToolContext) -> int:
    """The depth a tool-call payload may nest to before the call is refused.

    Read from the run's constants where it carries them, so an operator can
    raise it for a workload with genuinely deep payloads. A run without
    constants falls back to the structural floor the pure contract validators
    and JSON utilities use, which is this field's own default.
    """
    raw = getattr(_run_constants(ctx), "max_data_nesting_depth", None)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return MAX_DATA_NESTING_DEPTH
    return raw


def _run_constants(ctx: ToolContext) -> Any | None:
    """The constants snapshot this run resolved, when it carries one.

    Duck-typed on purpose: every caller reads one named field off it with a
    defensive default, so a run wired without constants still dispatches under
    the module's own fallbacks rather than refusing to run.
    """
    state = ctx.run_state
    return None if state is None else state.rc


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    if coerced < 0:
        return None
    return coerced


def _resolve_tool_timeout_seconds(tool: Any, *, flat_timeout_seconds: int) -> float:
    """Pick the dispatch wall-clock budget for one tool invocation.

    The flat ``rc.tool_timeout_seconds`` (default 90s) is the budget for an
    ordinary tool call. A tool that runs its OWN long-lived async unit with an
    internal, classified timeout — the ``Agent`` tool, whose ``SubagentRunner``
    wraps the child loop in ``asyncio.timeout(max_duration_ms)`` and owns an
    in-tool stale watchdog — declares this by setting ``should_defer=True`` plus
    a ``default_timeout_ms`` ClassVar (the host ``TypedTool``; not part of the
    core :class:`~protocore.contracts.tools.Tool` ABC, hence read defensively).
    For such a tool the dispatcher MUST give at least the tool's declared budget,
    otherwise the flat cap cancels the unit mid-run and its own classified
    timeout (``DispatchErrorKind.timeout`` / ``abort_kind``) can never fire.

    Override semantics: when ``should_defer`` is a strict ``True`` and
    ``default_timeout_ms`` is a positive int, the dispatch budget IS the tool's
    declared seconds (no min/max with the flat cap — the deferring tool owns its
    own clock). Every other tool keeps the flat budget. One field, one read.
    """
    if getattr(tool, "should_defer", False) is not True:
        return float(flat_timeout_seconds)
    declared_ms = _coerce_non_negative_int(getattr(tool, "default_timeout_ms", None))
    if not declared_ms:
        return float(flat_timeout_seconds)
    return declared_ms / 1000.0


class ToolDispatchCancelled(asyncio.CancelledError):
    """Run-level cancel interrupted an in-flight tool dispatch (#6).

    A subclass of :class:`asyncio.CancelledError` so EVERY existing
    ``except asyncio.CancelledError`` handler up the stack (the core query
    generator's cooperative unwind, the host executor ``_drive_run``
    cancelled-arm, the subagent runner's external-cancel path) treats it
    identically — but the distinct type makes the origin (a SET per-run
    ``cancel_event``, not a generic task cancellation) traceable in logs.
    """


def _run_cancel_event(ctx: ToolContext) -> asyncio.Event | None:
    """The per-run cancel event, when this run carries one.

    A run wired without one (a tool invoked outside a run, a focused fixture)
    returns ``None`` and the dispatcher keeps its plain ``asyncio.wait_for``
    path. One field, one read.
    """
    state = ctx.run_state
    return None if state is None else state.cancel_event


def _tool_cancel_drain_seconds(ctx: ToolContext) -> float:
    """Bounded drain budget for a cancelled in-flight tool task (RC-driven)."""
    rc = _run_constants(ctx)
    drain = getattr(rc, "tool_cancel_drain_seconds", None)
    if isinstance(drain, bool) or not isinstance(drain, (int, float)):
        return _TOOL_CANCEL_DRAIN_FALLBACK_SECONDS
    drain_f = float(drain)
    return drain_f if drain_f > 0 else _TOOL_CANCEL_DRAIN_FALLBACK_SECONDS


async def _invoke_tool_raced_with_cancel(
    *,
    tool: Any,
    ctx: ToolContext,
    final_args: dict[str, Any],
    effective_timeout: float,
    cancel_event: asyncio.Event,
) -> ToolResult:
    """Await ``tool.invoke`` while racing the per-run cancel Event (#6).

 On the happy path this is equivalent to
 ``await asyncio.wait_for(tool.invoke(ctx, final_args), timeout=...)`` — it
 returns the tool result, and the cancel watcher is cancelled cleanly.

 When ``cancel_event`` fires FIRST, the in-flight tool task is cancelled,
 bounded-drained (so the ``Agent`` tool's subagent teardown can unwind), and
 :class:`ToolDispatchCancelled` is raised so the run finalises ``cancelled``
 promptly instead of after the whole tool (subagent) runs.

 Follows the KB anti-pattern fix (→ "asyncio
 .wait does NOT cancel its child tasks"): ``asyncio.wait`` leaves the
 not-yet-done task running, so BOTH branches explicitly cancel + drain the
 loser task; the ``finally`` guards against a cancel of THIS coroutine
 leaking either child task.
 """
    # Already cancelled BEFORE we start → never construct/await the tool
    # coroutine. A coroutine body runs its side-effects synchronously up to its
    # first ``await``, so creating the task at all would let a post-cancel
    # tool/subagent begin. Raise without touching the tool so the "no
    # post-cancel tool calls" contract holds even when the cancel landed between
    # the dispatch pre-checks and this point.
    if cancel_event.is_set():
        _logger.warning(
            "DIAG tool_dispatch.cancelled_before_invoke tool=%s run=%s — "
            "cancel already set; not starting the tool",
            getattr(tool, "name", type(tool).__name__),
            ctx.run_id,
        )
        raise ToolDispatchCancelled(
            f"tool dispatch cancelled for run {ctx.run_id!r}"
        )
    tool_task: asyncio.Task[Any] = asyncio.ensure_future(
        asyncio.wait_for(tool.invoke(ctx, final_args), timeout=effective_timeout)
    )
    cancel_task: asyncio.Task[Any] = asyncio.ensure_future(cancel_event.wait())
    try:
        done, _pending = await asyncio.wait(
            {tool_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        # THIS coroutine was cancelled (e.g. executor drain) — do not leak
        # either child task. Cancel + best-effort settle both, then re-raise.
        for task in (tool_task, cancel_task):
            if not task.done():
                task.cancel()
        for task in (tool_task, cancel_task):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        raise

    # Cancel fired before the tool finished → abort the tool task.
    if cancel_event.is_set() and tool_task not in done:
        tool_task.cancel()
        drain = _tool_cancel_drain_seconds(ctx)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(tool_task, timeout=drain)
        _logger.warning(
            "DIAG tool_dispatch.cancelled_in_flight tool=%s run=%s — "
            "run cancel raced the in-flight tool task; raising cancelled",
            getattr(tool, "name", type(tool).__name__),
            ctx.run_id,
        )
        raise ToolDispatchCancelled(
            f"tool dispatch cancelled for run {ctx.run_id!r}"
        )

    # Tool finished (or raised) first → tear down the cancel watcher and
    # surface the tool's result/exception exactly as the plain path would.
    if not cancel_task.done():
        cancel_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await cancel_task
    result: ToolResult = await tool_task
    return result


async def _record_tool_call_soft_cap_warning(
    *,
    ctx: ToolContext,
    tool_name: str,
) -> dict[str, Any] | None:
    """Increment the optional subagent soft-cap counter and return a warning.

    The counts live on the run's own state, so a child run gets one isolated
    set per subagent run. A cap of 0 means unlimited. The warning is advisory
    only: the caller annotates the already-produced tool result and never alters
    dispatch success/error semantics.
    """
    state = ctx.run_state
    if state is None:
        return None
    limit = _coerce_non_negative_int(state.tool_call_soft_caps.get(tool_name))
    if limit is None or limit == 0:
        return None

    cap_state = state.tool_call_soft_cap_state
    lock = cap_state.lock
    if lock is None:
        lock = state.tool_shared_state_lock
    if lock is None or not hasattr(lock, "__aenter__"):
        lock = asyncio.Lock()
    cap_state.lock = lock

    async with lock:
        counts = cap_state.counts
        count = _coerce_non_negative_int(counts.get(tool_name)) or 0
        count += 1
        counts[tool_name] = count

        if count < limit:
            return None

        status = "reached" if count == limit else "exceeded"
        if status == "reached":
            message = (
                f"Warning: {tool_name} call soft cap {limit} reached "
                f"(call {count}). Continue only if necessary; expected work "
                "quality may decline after this point."
            )
        else:
            message = (
                f"Warning: {tool_name} call soft cap {limit} exceeded "
                f"(call {count}). The tool still ran; expected work quality "
                "may decline."
            )
        warning = {
            "tool_name": tool_name,
            "limit": limit,
            "count": count,
            "status": status,
            "message": message,
        }
        cap_state.warnings.append(dict(warning))
        return warning


def _append_soft_cap_warning_to_content(content: str, warning: Mapping[str, Any]) -> str:
    message = str(warning.get("message") or "").strip()
    if not message:
        return content
    separator = "\n\n" if content else ""
    return f"{content}{separator}[Tool call soft-cap warning]\n{message}"


def _merge_soft_cap_warning_metadata(
    metadata: dict[str, Any] | None,
    warning: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(metadata or {})
    warning_dict = dict(warning)
    merged[TOOL_CALL_SOFT_CAP_METADATA_KEY] = warning_dict
    raw_warnings = merged.get(TOOL_CALL_SOFT_CAP_WARNINGS_METADATA_KEY)
    if isinstance(raw_warnings, list):
        warnings = [*raw_warnings, warning_dict]
    else:
        warnings = [warning_dict]
    merged[TOOL_CALL_SOFT_CAP_WARNINGS_METADATA_KEY] = warnings
    return merged


def _annotate_tool_result_event(
    event: TurnEvent,
    *,
    warning: Mapping[str, Any],
    content: str,
    metadata: dict[str, Any],
) -> TurnEvent:
    if event.type is not EventType.TOOL_RESULT:
        return event
    payload = dict(event.payload)
    blocks = payload.get("content_blocks")
    if isinstance(blocks, list) and blocks:
        next_blocks: list[Any] = []
        replaced = False
        for block in blocks:
            if (
                not replaced
                and isinstance(block, dict)
                and block.get("type") == "text"
            ):
                next_block = dict(block)
                next_block["text"] = content
                next_blocks.append(next_block)
                replaced = True
            else:
                next_blocks.append(block)
        payload["content_blocks"] = next_blocks
    else:
        payload["content_blocks"] = [{"type": "text", "text": content}]
    payload["metadata"] = metadata
    if "error" in payload and isinstance(payload["error"], dict):
        error_payload = dict(payload["error"])
        error_payload["soft_cap_warning"] = dict(warning)
        payload["error"] = error_payload
    return event.model_copy(update={"payload": payload})


# Consecutive same-tool-same-error cap.
# Research found the leader can retry an IDENTICAL failed tool call up to
# 200 times (docgen-en-005, long-en-004 Write storms). The dispatcher tracks
# a single per-run streak on the run's state; when the
# (tool_name, signature) tuple repeats more times than the RC cap, the next
# error is rewritten with the `consecutive_error_cap` kind so the model sees
# a distinct stop signal instead of looping. Streak resets on (a) a different
# (tool, signature) tuple or (b) a successful tool call.
# Fallback default when the run carries no constants (test fixtures /
# legacy dispatch paths). Mirrors the executor's defensive ``getattr`` pattern
# for the analogous ``max_ask_user_calls_per_run``.
_DEFAULT_CONSECUTIVE_ERROR_CAP: int = 4

# Subagent tool-call soft caps. The host declares the limits on a child run's
# state; core treats them as an optional diagnostics layer above normal dispatch
# and never gates tool execution. These two name the fields on the tool result
# that carry a warning back to the caller.
TOOL_CALL_SOFT_CAP_METADATA_KEY: str = "tool_call_soft_cap"
TOOL_CALL_SOFT_CAP_WARNINGS_METADATA_KEY: str = "tool_call_soft_cap_warnings"

# Transport-down canonical-signature streak.
# Tracks a SEPARATE counter from the generic consecutive-error cap so the
# threshold can fire earlier (default 3 vs cap default 4). When the streak
# reaches ``LoopConstants.sandbox_down_system_message_threshold`` the
# dispatcher raises a one-shot signal on the run's state for the loop above
# to consume — that loop appends a synthetic user-role :class:`Message`
# telling the agent to work by another route while the transport is down.
# One-shot per streak: a successful tool call OR a signature that is not a
# transport-down one clears it. What the streak is for: a run can otherwise
# spend its whole token budget retrying a tool whose way out is gone, dozens
# of failed calls deep, because each failure looks new to a counter that
# tells errors apart by their text.
_TRANSPORT_DOWN_CANONICAL_SUFFIX: str = ":TRANSPORT_DOWN"

# Fallback default when no RC is wired (test fixtures, legacy paths).
# Mirrors :data:`_DEFAULT_CONSECUTIVE_ERROR_CAP` defensive pattern.
_DEFAULT_TRANSPORT_DOWN_THRESHOLD: int = 3

# Stabilization Pydantic ``string_type``
# terminal streak. Tracks consecutive validation errors with
# ``type=string_type`` on the SAME tool independently of the generic
# consecutive-error cap so this failure mode can be tuned separately.
# When the streak reaches ``LoopConstants.tool_dispatch_string_type_
# terminal_cap`` the dispatcher rewrites the next dispatch error to
# ``DispatchErrorKind.consecutive_error_cap`` with a stronger terminal
# guidance string (the model is told to stop retrying the same shape).
# The streak resets on a successful tool call OR a fresh non-string_type
# error signature, the same pattern as the transport-down counter.
#
# A host's own write/append/bash tools are expected to coerce the common
# malformed shapes (list/dict instead of str) silently, so this guard is a
# safety net for residual
# cases (e.g. an uncoercible ``content=None`` or a future field that did
# not get a coercion validator).
_STRING_TYPE_CANONICAL_MARKER: str = "string_type"

# Fallback default when no RC is wired (test fixtures, legacy paths).
# Lowered from 5 to 3 so the string_type-specific TERMINAL guidance fires
# BEFORE the generic ``tool_dispatch_consecutive_error_cap`` (default 4)
# wraps the error with vague guidance.
_DEFAULT_STRING_TYPE_TERMINAL_CAP: int = 3

# Normalisation regexes — strip variable identifiers (uuid-shaped call ids,
# absolute paths, line numbers, the dispatcher's own ``after Ns`` timeout
# wording) so a real retry of the same logical error collapses to the same
# signature. Keeps the error_kind in the signature so a Write-validation error
# and a Write-execution error of the same wording are still distinct.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HEX_TOKEN_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
_NUMBER_RE = re.compile(r"\b\d+\b")
_TIMEOUT_DURATION_RE = re.compile(r"timed out after \d+s")

# Broaden signature normalisation so a storm of failures against one
# unreachable transport collapses to a single canonical signature regardless
# of the arguments the model varied on retry. The model changes command text
# between attempts (different quoted strings, paths, regex snippets), so the
# (tool, signature) tuple kept changing and the consecutive-error cap never
# fired on a tool that was simply unreachable.
#
# The host's verdict takes precedence over hashing: when the bound
# :class:`~protocore.contracts.resilience.IResilienceClassifier` says a
# failure belongs to one of
# :data:`~protocore.contracts.resilience.TRANSPORT_DOWN_ERROR_CLASSES`, the
# signature is fixed to ``<tool>:TRANSPORT_DOWN`` and the cap collapses N
# varied retries against a dead transport into one streak. Core recognises
# none of those failures by their wording: the wordings belong to whatever
# the host put behind its tools, and a pattern for them here would be a copy
# of another codebase, kept in step by nothing. Similarly for
# ``command not found``, which is the shell's own wording and is asked of a
# tool the host declared as running a shell.
#
# When no verdict collapses the failure we still hash, but first strip quoted
# strings and absolute file paths so cosmetic variation between retries
# (different file names, regex snippets, search needles) does not bump the
# signature.
_SHELL_CMD_MISSING_PATTERNS = re.compile(
    r"command not found",
    re.IGNORECASE,
)
# Stabilization detect Pydantic
# ``string_type`` validation errors in dispatcher error text.
# The host typed-tool adapter raises ``ToolInvocationError`` whose
# message includes the sanitized ``exc.errors(include_input=False)``
# dump, which still carries ``'type': 'string_type'`` for the offending
# field. The dispatcher uses this to maintain a separate consecutive-
# streak counter so the failure can be terminated faster than the
# generic ``tool_dispatch_consecutive_error_cap``.
#
# The SAME terminal cap must also fire on Pydantic ``type='missing'``
# (``Field required``). A content-less ``Write`` produces
# ``[{'type': 'missing', 'loc': ('content',), 'msg': 'Field required'}]``
# on EVERY retry; matching only ``string_type`` would mean the schema-specific
# TERMINAL guidance never fires and the loop spirals. Both are "the argument
# is the wrong SHAPE / absent, retrying the same shape will not succeed"
# failures, so they share the one terminal streak. (The chunk-recovery logic
# prevents a *truncated* call from ever reaching dispatch; this cap is the
# backstop for a genuinely model-omitted required field.)
_STRING_TYPE_VALIDATION_PATTERN = re.compile(
    r"'type'\s*:\s*'(?:string_type|missing)'",
)
# Alternation order is single / double / back-tick. Each alternative is
# greedy up to the next matching delimiter on the same line. We do not
# handle triple-quoted strings — they do not appear in sanitised error
# messages emitted by ``_safe_error_message``.
_QUOTED_REGEX = re.compile(r"'[^']*'|\"[^\"]*\"|`[^`]*`")
# Match absolute POSIX file paths (``/foo``, ``/foo/bar.py``). Underscores,
# dots, dashes and slashes are common in test/workspace paths and uniquely
# identify the file being acted on, so they should not contribute to the
# signature.
_FILE_PATH_REGEX = re.compile(r"(?:/[a-zA-Z0-9_.-]+)+")

# Strip URLs from error text before hashing. A host that restarts the thing
# behind a tool commonly brings it back at a fresh address, and each failure
# line then embeds a different one — so a single outage reads as a stream of
# distinct errors to a counter that hashes the text. The classifier normally
# collapses those failures first; this is what keeps the hash path honest
# when it does not.
_ERROR_URL_PATTERN = re.compile(
    r"\b[a-z][a-z0-9+.-]*://\S+",
    re.IGNORECASE,
)


# Hallucination hint for `<*_contract>` XML blocks misemitted as tool names.
# Case-sensitive lowercase-only `^[a-z_]+_contract$` to avoid false positives
# on legitimate (e.g. CamelCase) tool names; only the model's
# "I-meant-the-XML-block" failure mode follows this snake_case shape
# (`finalization_contract`, `subagent_contract`, …).
_CONTRACT_HALLUCINATION = re.compile(r"^[a-z_]+_contract$")


def _contract_hallucination_hint(tool_name: str, *, finalize_terminal: bool = False) -> str:
    """Build the nudge message for a hallucinated ``*_contract`` tool name.

    Typed-Finalize tenants (``LoopConstants.agent_finalize_tool_as_terminal``)
    must NOT be steered to the legacy ``<finalization_contract>`` XML block when
    they misfire a ``finalization_contract`` "tool" — that re-introduces the very
    prose-contract leak the ``Finalize`` tool retires. For them the nudge points
    at ``Finalize`` instead. Every other ``*_contract`` shape (and the opt-out /
    default tenants) keep the inline-XML nudge byte-for-byte.
    """
    if finalize_terminal and tool_name == "finalization_contract":
        return (
            f"'{tool_name}' is not a tool. To finish, call the `Finalize` tool "
            f"with declared_deliverables and answer — do NOT write a "
            f"<finalization_contract> text block."
        )
    return (
        f"'{tool_name}' is not a tool — it appears to be an XML contract "
        f"block. Inline it directly in your assistant text as "
        f"<{tool_name}>...</{tool_name}> (especially the "
        f"<finalization_contract> block per the leader persona spec)."
    )


# ---------------------------------------------------------------------------
# Error taxonomy.
# ---------------------------------------------------------------------------


class DispatchErrorKind(StrEnum):
    """Error kinds per

 Every kind wraps into a ``tool_result(success=false)`` content block;
 the model sees the failure and recovers in the next assistant turn.

 ``consecutive_error_cap`` tracks
 dispatcher rewrites a real failure into this kind once the per-run
 consecutive-identical-error streak exceeds
 ``LoopConstants.tool_dispatch_consecutive_error_cap``.
 """

    validation = "validation"
    permission = "permission"
    execution = "execution"
    timeout = "timeout"
    rate_limit = "rate_limit"
    unknown_tool = "unknown_tool"
    consecutive_error_cap = "consecutive_error_cap"


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    """Final outcome of one :meth:`ToolDispatcher.dispatch` call.

 The caller uses this to (a) append a tool-result :class:`Message` to
 history and (b) decide whether to transition to ``AWAITING``.

 Attributes
 ----------
 tool_call:
 The original :class:`ToolCall` (after any hook mutation).
 success:
 ``True`` only if the tool executed and returned a non-error
 :class:`ToolResult`.
 content:
 Text content for the resulting :class:`ToolResultBlock`. On
 error this is the sanitised error message.
 is_error:
 Mirror of ``not success`` — surfaced as
 :attr:`ToolResultBlock.is_error`.
 error_kind:
 Non-None on failure paths only. Drives audit + telemetry.
 approval_required:
 ``True`` if the gate returned ``require_approval`` — the loop
 MUST transition to ``AWAITING`` and not append a tool result.
 approval_token:
 Opaque token to round-trip to the user.
 ask_user_required:
 ``True`` if the tool raised :class:`AskUserPauseRequested`
 — the host's dispatch shim
 transitions the loop to ``AWAITING`` via the ``ask_user`` resume
 path, emits the matching SSE envelope, and writes the pending
 interrupt to Redis. No ``tool_result`` is appended; the answer
 arrives on resume.
 ask_user_payload:
 Validated :class:`~protocore.tools.ask_user.AskUserInput`
 carried out of band (typed dict shape) so the host handler
 does not re-validate. Stored as a plain dict (``model_dump``) to
 keep :class:`DispatchOutcome` import-light and JSON-friendly for
 cross-pod transport.
 duration_ms:
 Wall-time of the execute step (zero on pre-execute denial).
 metadata:
 Optional bag for telemetry (provider-meta, stage, …).
 evidence_records:
 Typed tool-authored observations retained only by the runtime. They are
 intentionally absent from model-visible result blocks and event payloads.
 canonical_content:
 The whole value the tool returned, when ``content`` is only a
 projection of it. ``None`` means the two are the same value.
 ui_payload:
 Structured detail for whoever is watching the run. Rides the result
 event and stops there — it never enters the transcript.
 canonical_ref:
 Where the canonical value can be fetched back from, if the tool
 stored it anywhere.
 path:
 The workspace path this result is a view of, if it is a view of one.
 """

    tool_call: ToolCall
    success: bool
    content: str
    is_error: bool
    error_kind: DispatchErrorKind | None = None
    approval_required: bool = False
    approval_token: str | None = None
    ask_user_required: bool = False
    ask_user_payload: dict[str, Any] | None = None
    duration_ms: int = 0
    metadata: dict[str, Any] | None = None
    evidence_records: tuple[EvidenceRecord, ...] = ()
    evidence_producer: EvidenceProducerBinding | None = None
    canonical_content: str | None = None
    ui_payload: dict[str, Any] | None = None
    canonical_ref: str | None = None
    path: str | None = None


DISPATCH_REPLAY_ERROR_KIND_METADATA_KEY: str = "tool_dispatch.replay_error_kind"
DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY: str = "tool_dispatch.replay_error_message"
DISPATCH_POST_TOOL_OUTPUT_MODIFIED_METADATA_KEY: str = (
    "tool_dispatch.post_tool_output_modified"
)

# The metadata slot carrying a tool's machine-readable give-up signal, and
# the sub-key the loop reads to decide whether to surface a finalize hint to
# the model. A tool attaches ``structured_error`` to its raised exception
# (e.g. a retry-budget exhaustion: ``{"finalization_recommended": True,
# "reason": ...}``); the dispatch except-branch forwards it verbatim under
# :data:`DISPATCH_STRUCTURED_ERROR_METADATA_KEY` on the DispatchOutcome.
DISPATCH_STRUCTURED_ERROR_METADATA_KEY: str = "structured_error"
STRUCTURED_ERROR_FINALIZATION_RECOMMENDED_KEY: str = "finalization_recommended"
STRUCTURED_ERROR_REASON_KEY: str = "reason"

#: Fallback bounded-drain budget (seconds) for a cancelled in-flight tool task
#: when the run carries no constants (older callers / tests). The live path
#: uses ``LoopConstants.tool_cancel_drain_seconds``; this mirrors its default
#: so behaviour is identical when the RC is unreachable.
_TOOL_CANCEL_DRAIN_FALLBACK_SECONDS: Final[float] = 2.0

# How many near names an unknown-tool error offers. Three is enough to contain
# the right one when a case or a word was wrong, and few enough that the model
# is not handed a menu to pick a plausible stranger from.
_UNKNOWN_TOOL_SUGGESTIONS: Final[int] = 3


def _dispatch_replay_metadata(
    kind: DispatchErrorKind,
    message: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(extra or {})
    metadata[DISPATCH_REPLAY_ERROR_KIND_METADATA_KEY] = kind.value
    metadata[DISPATCH_REPLAY_ERROR_MESSAGE_METADATA_KEY] = message
    return metadata


# ---------------------------------------------------------------------------
# The dispatcher
# ---------------------------------------------------------------------------


def _nesting_exceeds(value: Any, max_depth: int) -> bool:
    """True iff ``value`` nests containers deeper than ``max_depth``.

    Iterative on an explicit stack, so measuring the depth of a pathological
    structure cannot itself be what exhausts the stack. Only the container
    types ``json.dumps`` descends into are counted.
    """
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if not isinstance(item, (dict, list, tuple)):
            continue
        if depth >= max_depth:
            return True
        children = item.values() if isinstance(item, dict) else item
        for child in children:
            stack.append((child, depth + 1))
    return False


class ToolDispatcher:
    """Orchestrates the 7-step tool-call lifecycle.

    The dispatcher is **stateless per-call** — all state flows through
    parameters. Construct once per executor pod and re-use across runs.
    """

    def __init__(
        self,
        *,
        registry: IToolRegistry,
        permission_gate: ToolPermissionGate,
        hook_manager: IHookManager | None = None,
        tool_error_counter: IRunToolErrorCounter | None = None,
        roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
        resilience_classifier: IResilienceClassifier | None = None,
    ) -> None:
        self._registry = registry
        self._gate = permission_gate
        self._hooks = hook_manager
        self._tool_error_counter = tool_error_counter
        self._roles = roles
        """What the host said its tools do — the dispatcher asks it instead of
        recognising a tool by a name it spelled itself."""
        self._resilience_classifier = resilience_classifier
        """Who says what kind of failure a message describes. Absent, the
        dispatcher tells failures apart by their own text alone and no run of
        them is ever read as one transport being down."""

    async def _record_tool_error(self, ctx: ToolContext) -> None:
        """Best-effort increment of ``runs.tool_errors_count`` for the run.

 Tool errors are attributed to the **parent (root) run** rather than the
 per-subagent ``ctx.run_id``.

 Why: the dashboard aggregates ``tool_errors_count`` at the root-run
 level for partial-status classification. Subagent ``run_id``s are
 internal and not present in the PG ``runs`` table (no row to update —
 they are not durably persisted as their own ``runs.id`` rows), and
 any non-UUID-shaped string fed into the ``WHERE id = $1`` UUID column
 triggers ``psycopg.errors.InvalidTextRepresentation`` (~20x/run noise).
 The parent's ``root_run_id`` is already plumbed by the host down the run
 tree and lives on the run's state. It is the parent's bare UUID — semantically the right aggregate and SQL-safe.

 Non-raising — the dispatcher must never surface a telemetry failure
 as a tool error (counter leak / cascade). Logged at WARNING so the
 diagnostic is visible in production pod logs.
 """
        counter = self._tool_error_counter
        if counter is None:
            return
        target_run_id = self._resolve_error_attribution_run_id(ctx)
        try:
            await counter.increment_tool_errors_count(target_run_id, 1)
        except Exception:
            _logger.warning(
                "tool_error_counter increment failed run_id=%s",
                target_run_id,
                exc_info=True,
            )

    @staticmethod
    def _resolve_error_attribution_run_id(ctx: ToolContext) -> str:
        """Return the run id to attribute the tool-error increment to.

        Prefers the run state's ``root_run_id`` when present — the parent
        (root) run's bare UUID, plumbed by the host's subagent dispatch path.
        Falls back to ``ctx.run_id`` for leader runs where root_run_id ==
        run_id and for test contexts that wire no state.
        """
        state = ctx.run_state
        if state is not None and state.root_run_id:
            return state.root_run_id
        return ctx.run_id

    # ------------------------------------------------------------------
    # Consecutive same-tool-same-error cap
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_consecutive_error_cap(ctx: ToolContext) -> int:
        """Read ``tool_dispatch_consecutive_error_cap`` from the RC snapshot.

        Falls back to :data:`_DEFAULT_CONSECUTIVE_ERROR_CAP` when no helper
        bag / RC snapshot is wired (legacy test fixtures, dispatch paths
        that pre-date the RC plumbing). Mirrors the defensive ``getattr``
        pattern used for ``max_ask_user_calls_per_run``.
        """
        rc = _run_constants(ctx)
        if rc is None:
            return _DEFAULT_CONSECUTIVE_ERROR_CAP
        raw = getattr(rc, "tool_dispatch_consecutive_error_cap", _DEFAULT_CONSECUTIVE_ERROR_CAP)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_CONSECUTIVE_ERROR_CAP
        # ``ge=2`` is enforced by the LoopConstants validator; defence
        # in depth here so a corrupted snapshot cannot push the cap to 1
        # (which would reject the very first error).
        return value if value >= 2 else _DEFAULT_CONSECUTIVE_ERROR_CAP

    @staticmethod
    def _resolve_transport_down_threshold(ctx: ToolContext) -> int:
        """Read ``sandbox_down_system_message_threshold`` from the RC snapshot.

        Falls back to :data:`_DEFAULT_TRANSPORT_DOWN_THRESHOLD` when no RC
        snapshot is wired. Mirrors :meth:`_resolve_consecutive_error_cap`.
        """
        rc = _run_constants(ctx)
        if rc is None:
            return _DEFAULT_TRANSPORT_DOWN_THRESHOLD
        raw = getattr(
            rc, "sandbox_down_system_message_threshold", _DEFAULT_TRANSPORT_DOWN_THRESHOLD
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_TRANSPORT_DOWN_THRESHOLD
        # ``gt=0`` is enforced by the LoopConstants validator; defence in
        # depth so a corrupted snapshot cannot push the threshold to 0
        # (which would fire on every dispatch).
        return value if value >= 1 else _DEFAULT_TRANSPORT_DOWN_THRESHOLD

    @staticmethod
    def _resolve_string_type_terminal_cap(ctx: ToolContext) -> int:
        """Read ``tool_dispatch_string_type_terminal_cap`` from the RC snapshot.

 Falls back to
 :data:`_DEFAULT_STRING_TYPE_TERMINAL_CAP` when no run state / RC
 snapshot is wired (legacy test fixtures). Mirrors
 :meth:`_resolve_consecutive_error_cap` defensive pattern.
 """
        rc = _run_constants(ctx)
        if rc is None:
            return _DEFAULT_STRING_TYPE_TERMINAL_CAP
        raw = getattr(
            rc,
            "tool_dispatch_string_type_terminal_cap",
            _DEFAULT_STRING_TYPE_TERMINAL_CAP,
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_STRING_TYPE_TERMINAL_CAP
        # ``ge=2`` is enforced by the LoopConstants validator; defence
        # in depth here so a corrupted snapshot cannot push the cap to 1.
        return value if value >= 2 else _DEFAULT_STRING_TYPE_TERMINAL_CAP

    @staticmethod
    def _is_string_type_error(kind: DispatchErrorKind, message: str) -> bool:
        """Return ``True`` when *message* carries a Pydantic
 ``string_type`` validation error.

 The host typed-tool adapter raises
 ``ToolInvocationError`` whose message format is
 ``tool 'X': invalid arguments: [{'type': 'string_type', ...}]``.
 Matching on the textual presence of ``'type': 'string_type'``
 avoids tight coupling to either the validation or execution
 error kind: both routes can carry the same payload depending on
 which dispatcher branch caught the error.
 """
        if not message:
            return False
        del kind  # currently informational; both validation+execution carry the text
        return bool(_STRING_TYPE_VALIDATION_PATTERN.search(message))

    @staticmethod
    def _normalize_error_text(message: str) -> str:
        """Collapse variable identifiers so retries of the same logical
 failure produce identical signatures.

 Order matters:

 1. Strip URLs, which collapse to ``<url>`` before any other
 normalisation. Without this, a failing transport that comes back at
 a fresh address each time produces a fresh hashed signature every
 iteration whenever the host's classifier returned no verdict.
 2. Strip quoted content (`'foo'`, `"bar"`, ```baz```) and absolute
 file paths.
 3. Strip uuid-shaped tokens, long hex blobs, the dispatcher's own
 ``timed out after Ns`` suffix, and bare decimal numbers.
 4. Collapse whitespace so the hash stays stable across cosmetic
 spacing differences between retries.
 """
        normalized = _ERROR_URL_PATTERN.sub("<url>", message)
        normalized = _QUOTED_REGEX.sub("<quoted>", normalized)
        normalized = _FILE_PATH_REGEX.sub("<path>", normalized)
        normalized = _UUID_RE.sub("<uuid>", normalized)
        normalized = _HEX_TOKEN_RE.sub("<hex>", normalized)
        normalized = _TIMEOUT_DURATION_RE.sub("timed out after <n>s", normalized)
        normalized = _NUMBER_RE.sub("<n>", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _is_transport_down(
        message: str,
        tool_name: str | None,
        classifier: IResilienceClassifier | None,
    ) -> bool:
        """Whether the host calls this failure one of the transport being down.

        Best-effort in both directions: no classifier means no, and a
        classifier that raises means no. A failing verdict must not turn a
        tool error into a dispatch crash, and the only thing lost by
        answering no is one collapsed signature.
        """
        if classifier is None:
            return False
        try:
            verdict = classifier.classify_error_text(message, tool_name=tool_name)
        except Exception:
            _logger.warning(
                "resilience classifier raised on a dispatch error for tool=%s; "
                "treating the failure as unclassified",
                tool_name or "<unknown>",
                exc_info=True,
            )
            return False
        return verdict in TRANSPORT_DOWN_ERROR_CLASSES

    @classmethod
    def _error_signature(
        cls,
        kind: DispatchErrorKind,
        message: str,
        tool_name: str | None = None,
        roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
        classifier: IResilienceClassifier | None = None,
    ) -> str:
        """Return the per-error-streak signature.

 Canonical signatures take precedence: a failure the host's
 ``classifier`` calls a transport-down one collapses to
 ``<tool>:TRANSPORT_DOWN``, and a shell tool's ``command not found``
 collapses to ``<tool>:SHELL_CMD_MISSING``, both regardless of
 surrounding text — without this the consecutive-error cap never fires
 when the model varies its argument shape on retry. Core asks the host
 for the first of those verdicts and never reads the wording itself; a
 run with no classifier bound simply has no transport-down signature.

 Otherwise we hash ``(error_kind, normalised_message)`` where the
 normalisation already strips quoted content + file paths via
 :meth:`_normalize_error_text`. Stable across retries of the same
 logical failure but distinct across error kinds + distinct logical
 failures.

 ``tool_name`` is optional for backwards compatibility with the few
 legacy callers / tests that build a signature without a tool. When
 omitted the canonical match falls back to the ``unknown`` namespace.
 """
        # Canonical-match short-circuit traces — logged so that when the
        # consecutive-error cap fires we can verify which branch triggered.
        if cls._is_transport_down(message, tool_name, classifier):
            _logger.debug(
                "DIAG tool_dispatch.canonical_error_match "
                "tool=%s kind=%s match=TRANSPORT_DOWN excerpt=%r",
                tool_name or "<unknown>",
                kind.value,
                message[:100],
            )
            return f"{tool_name or 'unknown'}{_TRANSPORT_DOWN_CANONICAL_SUFFIX}"
        if (
            roles.has_role(tool_name, ToolRole.runs_shell)
            and _SHELL_CMD_MISSING_PATTERNS.search(message)
        ):
            _logger.debug(
                "DIAG tool_dispatch.canonical_error_match "
                "tool=%s kind=%s match=SHELL_CMD_MISSING excerpt=%r",
                tool_name,
                kind.value,
                message[:100],
            )
            return f"{tool_name}:SHELL_CMD_MISSING"
        normalized = cls._normalize_error_text(message)
        payload = f"{kind.value}|{normalized}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _apply_consecutive_error_cap(
        self,
        ctx: ToolContext,
        tool_name: str,
        kind: DispatchErrorKind,
        message: str,
        *,
        emit_diagnostics: bool = True,
    ) -> tuple[DispatchErrorKind, str]:
        """Track the consecutive streak and rewrite the surfaced error
 once the per-run cap is exceeded.

 Returns the (kind, message) tuple to surface to the caller — either
 the original pair (streak under cap) or a synthesised
 :class:`DispatchErrorKind.consecutive_error_cap` with guidance to
 try a different approach + the original error text appended.

 State shape: one record on the run's state holding the last
 (tool_name, signature) and the current consecutive count. Switching
 tool OR signature resets the count to 1.

 In addition to the generic cap, the method maintains a SEPARATE
 counter for transport-down canonical signatures
 (``<tool>:TRANSPORT_DOWN``) and raises a one-shot injection signal on
 the run's state when that counter reaches
 ``LoopConstants.sandbox_down_system_message_threshold``. The signal is
 consumed by the loop above, which appends a synthetic user-role
 message telling the agent to reach its goal by another route.
 """
        state = ctx.run_state
        if state is None:
            # A tool invoked outside a run counts no streak: there is nothing
            # for a per-run cap to be a cap on.
            return kind, message

        cls = type(self)
        signature = cls._error_signature(
            kind,
            message,
            tool_name,
            roles=self._roles,
            classifier=self._resilience_classifier,
        )
        prior = state.consecutive_error
        if (
            prior is not None
            and prior.tool_name == tool_name
            and prior.signature == signature
        ):
            count = prior.count + 1
        else:
            count = 1

        state.consecutive_error = ConsecutiveErrorStreak(
            tool_name=tool_name, signature=signature, count=count
        )

        # Separately track the transport-down streak
        # so the injection threshold can fire earlier (default 3) than the
        # generic cap (default 4). The two counters are kept independent so
        # operators can tune them independently per category.
        cls._track_transport_down_streak(
            ctx, state, signature, emit_diagnostics=emit_diagnostics
        )

        # separately track Pydantic
        # ``string_type`` validation errors so the terminal guard for the
        # malformed-args loop can fire independently of (and typically
        # later than) the generic cap. The streak resets on either a
        # non-string_type error or a successful tool call (handled in
        # :meth:`_reset_consecutive_error_streak`).
        string_type_terminal = cls._track_string_type_streak(
            ctx,
            state,
            tool_name,
            kind,
            message,
            emit_diagnostics=emit_diagnostics,
        )
        if string_type_terminal is not None:
            return string_type_terminal

        cap = cls._resolve_consecutive_error_cap(ctx)
        if count >= cap:
            if emit_diagnostics:
                _logger.warning(
                    "DIAG tool_dispatch.consecutive_error_cap "
                    "run=%s tool=%s count=%d cap=%d",
                    ctx.run_id,
                    tool_name,
                    count,
                    cap,
                )
            wrapped = (
                f"This tool+error combination repeated {count} consecutive "
                "times. Try a different tool or argument shape. "
                f"{message}"
            )
            return DispatchErrorKind.consecutive_error_cap, wrapped

        return kind, message

    @classmethod
    def _track_transport_down_streak(
        cls,
        ctx: ToolContext,
        state: RunScopedState,
        signature: str,
        *,
        emit_diagnostics: bool = True,
    ) -> None:
        """Update the transport-down canonical-signature streak counter.

 Runs every time
 :meth:`_apply_consecutive_error_cap` records a dispatch error. When
 ``signature`` ends in :data:`_TRANSPORT_DOWN_CANONICAL_SUFFIX` the
 counter increments; otherwise it resets to zero (any other error
 breaks the streak so the injection signal does not fire across a mix
 of unrelated tool failures).

 Once the counter reaches the RC threshold, the run's state raises a
 one-shot injection-pending flag. The loop above consumes it by
 appending a synthetic user-role message and then clearing it via
 :meth:`_consume_transport_down_injection_signal`. Until the flag is
 consumed the counter does NOT re-arm the signal on every subsequent
 transport-down failure — re-arming only happens after a successful
 tool call or another error breaks the streak.

 Best-effort: no exceptions propagate out of this helper. The cap
 path keeps working even if the signal cannot be posted.
 """
        if not signature.endswith(_TRANSPORT_DOWN_CANONICAL_SUFFIX):
            state.transport_down = None
            return

        prior = state.transport_down
        new_count = (prior.count if prior is not None else 0) + 1
        state.transport_down = SignatureStreak(signature=signature, count=new_count)

        threshold = cls._resolve_transport_down_threshold(ctx)
        # Only arm the signal at the exact threshold crossing so
        # the loop above sees the nudge once per streak. Subsequent
        # transport-down errors keep incrementing the counter for telemetry
        # but do not re-arm — preventing a flood of synthetic messages.
        if new_count == threshold:
            if emit_diagnostics:
                _logger.warning(
                    "DIAG tool_dispatch.transport_down_threshold_reached "
                    "run=%s signature=%s count=%d threshold=%d",
                    ctx.run_id,
                    signature,
                    new_count,
                    threshold,
                )
            state.transport_down_injection_pending = True

    @classmethod
    def _consume_transport_down_injection_signal(cls, ctx: ToolContext) -> bool:
        """Pop the transport-down injection-pending flag.

 The loop driving the run calls this
 after every TurnEvent dispatch step. Returns ``True`` exactly once
 per streak (when the flag is armed by
 :meth:`_track_transport_down_streak`). Subsequent calls return
 ``False`` until the streak resets and re-arms.

 Best-effort: a run with no state returns ``False`` silently. The
 method never raises — failure to consume the flag must not break
 the loop that calls it.
 """
        state = ctx.run_state
        if state is None or not state.transport_down_injection_pending:
            return False
        state.transport_down_injection_pending = False
        return True

    @classmethod
    def _track_string_type_streak(
        cls,
        ctx: ToolContext,
        state: RunScopedState,
        tool_name: str,
        kind: DispatchErrorKind,
        message: str,
        *,
        emit_diagnostics: bool = True,
    ) -> tuple[DispatchErrorKind, str] | None:
        """Update the Pydantic ``string_type`` streak counter and return
 a synthesised terminal-cap tuple when the cap is crossed.

 A model can loop many minutes retrying ``Write`` with a wrong
 argument shape (e.g. ``{content: [array]}``). The mainline coercion
 validators on ``WriteInput.content`` / ``AppendFileInput.content`` /
 ``BashInput.command`` handle common shapes silently; this terminal
 guard is the safety net for residual cases (uncoercible values,
 future fields that did not get a coercion validator).

 Returns ``None`` when the call is not a ``string_type`` error
 OR the streak is still under the cap. Returns the terminal
 ``(DispatchErrorKind.consecutive_error_cap, message)`` tuple
 when the cap is crossed so the caller can short-circuit the
 rest of :meth:`_apply_consecutive_error_cap`.

 Streak state is keyed on ``(tool_name, "string_type")`` so a
 switch to a different tool (e.g. Write -> Edit) restarts the
 counter at 1 while still surfacing the same failure mode.
 Non-``string_type`` errors break the streak (the model has
 moved to a different shape).
 """
        is_string_type = cls._is_string_type_error(kind, message)
        prior = state.string_type

        if not is_string_type:
            # Any non-string_type error breaks the streak. We don't
            # clear it here on success — that path lives in
            # :meth:`_reset_consecutive_error_streak`.
            state.string_type = None
            return None

        if prior is not None and prior.tool_name == tool_name:
            new_count = prior.count + 1
        else:
            new_count = 1

        state.string_type = ToolStreak(tool_name=tool_name, count=new_count)

        cap = cls._resolve_string_type_terminal_cap(ctx)
        if new_count >= cap:
            if emit_diagnostics:
                _logger.warning(
                    "DIAG tool_dispatch.string_type_terminal_cap "
                    "run=%s tool=%s count=%d cap=%d",
                    ctx.run_id,
                    tool_name,
                    new_count,
                    cap,
                )
            wrapped = (
                f"TERMINAL: {tool_name!r} rejected {new_count} consecutive "
                "calls with the same schema error (wrong-shape or "
                "missing-required-field). The argument is the wrong shape or a "
                "required field is absent — retrying the same call will not "
                "succeed. STOP and change the call: include EVERY required "
                "field, pass `content`/`command` as a single JSON string (use "
                "embedded `\\n` for newlines), and if a file's `content` is too "
                "large to fit in one call, write it in chunks "
                "(`Write` header -> `AppendFile` chunks -> `FinalizeFile`) "
                "instead of repeating the same oversized call. "
                f"Original error: {message}"
            )
            return DispatchErrorKind.consecutive_error_cap, wrapped
        return None

    @classmethod
    def _reset_consecutive_error_streak(cls, ctx: ToolContext) -> None:
        """Clear the streak on a successful tool call.

 Called after a tool returns a non-error :class:`ToolResult` — the
 next error (even for the same tool + same signature) starts a
 fresh streak. Best-effort: a run with no state is a silent no-op.

 Also clears the transport-down counter and
 the pending injection signal. A successful tool call (even of a
 different tool) is empirical evidence the transport recovered, so the
 next storm should restart at count=1 and the nudge can be re-armed at
 the threshold.

 also clears the ``string_type``
 terminal-cap counter so a subsequent malformed-args storm starts
 fresh.
 """
        state = ctx.run_state
        if state is None:
            return
        state.consecutive_error = None
        state.transport_down = None
        state.transport_down_injection_pending = False
        state.string_type = None

    # ------------------------------------------------------------------
    # DAG tool-precondition mechanism
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_preconditions_enabled(cls, ctx: ToolContext) -> bool:
        """Read ``tool_preconditions_enabled`` from the RC snapshot.

 Falls back to ``True`` when no RC is wired (legacy test fixtures /
 dispatch paths that pre-date the RC plumbing). Mirrors
 the defensive ``getattr`` pattern used for
 ``tool_dispatch_consecutive_error_cap`` etc.
 """
        rc = _run_constants(ctx)
        if rc is None:
            return True
        return bool(getattr(rc, "tool_preconditions_enabled", True))

    @classmethod
    def _check_tool_preconditions(
        cls,
        *,
        tool: Any,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> str | None:
        """Run the precondition check for *tool* against the per-run satisfied set.

        Returns ``None`` if all preconditions are met (or none declared),
        otherwise a human-readable denial reason. Mirrors the v1
        ``dispatch.py:1179-1216`` flow ported from commit ``7dfa1ff``.

        Sources the tool's ``preconditions`` list from
        :attr:`ToolDefinition.preconditions` via ``tool.definition``. If
        the tool has no ``preconditions`` (None or empty list) the check
        returns ``None`` without consulting the run's state.

        Disabled (returns ``None``) when
        ``LoopConstants.tool_preconditions_enabled`` is False.
        """
        if not cls._resolve_preconditions_enabled(ctx):
            return None
        # Read preconditions from the tool's published definition.
        # Some tool implementations may not expose ``preconditions`` on
        # their definition (legacy tools, mock tools in tests) — getattr
        # with default ``None`` covers both shapes.
        try:
            definition = tool.definition
        except Exception:
            return None
        preconditions = getattr(definition, "preconditions", None)
        if not preconditions:
            return None
        state = ctx.run_state
        satisfied = set(state.satisfied_preconditions) if state is not None else set()
        return check_preconditions(
            preconditions=list(preconditions),
            arguments=arguments,
            satisfied=satisfied,
        )

    @classmethod
    def _record_precondition_satisfaction(
        cls,
        *,
        tool: Any,
        tool_name: str,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> None:
        """Persist that *tool_name* was successfully called.

        Updates the run's satisfied-precondition set so subsequent dispatches
        in the same run can see the satisfaction.

        Honours :attr:`ToolDefinition.path_fields` when present so tools
        with non-standard path argument names (``copy_path`` /
        ``move_path`` in v1) record the correct ``tool_name:path`` entry.

        No-op when ``tool_preconditions_enabled`` is False or the call carries
        no run state — legacy test fixtures still dispatch successfully without
        the satisfaction set.
        """
        if not cls._resolve_preconditions_enabled(ctx):
            return
        state = ctx.run_state
        if state is None:
            return
        path_fields: list[str] | None = None
        try:
            definition = tool.definition
        except Exception:
            definition = None
        if definition is not None:
            raw_fields = getattr(definition, "path_fields", None)
            if isinstance(raw_fields, list):
                path_fields = [field for field in raw_fields if isinstance(field, str)]
        satisfied = set(state.satisfied_preconditions)
        record_satisfaction(
            tool_name=tool_name,
            arguments=arguments,
            satisfied=satisfied,
            path_fields=path_fields,
        )
        state.satisfied_preconditions = satisfied

    # ------------------------------------------------------------------
    # Public dispatch entry
    # ------------------------------------------------------------------

    async def _invoke_within_lifecycle(
        self,
        *,
        lifecycle: ILifecycleRegistry | None,
        tool: Any,
        tool_call: ToolCall,
        ctx: ToolContext,
        final_args: dict[str, Any],
        effective_timeout: float,
        cancel_event: asyncio.Event | None,
    ) -> ToolResult:
        """Invoke the tool, wrapped in the ``around`` chain at ``tool_execute``.

        With no registry, or with nothing registered at the coordinate, this is
        the plain invocation and nothing about it changes — the timeout, the
        cancel race and every exception type the arms below catch are the ones
        they always were.

        With a chain, three things are true and each of them is deliberate.
        The tool's own failure is re-raised **as itself** after the chain has
        unwound, so a handler sees the real exception and cannot convert a
        timeout into a success by swallowing it. A handler that refuses — by
        deciding, by raising, by timing out, or by answering unreadably —
        stops the call: :class:`ToolPolicyDenied` carries its reason into the
        permission arm, because a seam that could not say yes has not said it.
        And a handler that skips its ``next`` skips the tool itself, which is
        the same refusal: there is no result to report, so the call is refused
        rather than answered with a silence.
        """

        async def _run_tool() -> ToolResult:
            if cancel_event is None:
                return await asyncio.wait_for(
                    tool.invoke(ctx, final_args),
                    timeout=effective_timeout,
                )
            return await _invoke_tool_raced_with_cancel(
                tool=tool,
                ctx=ctx,
                final_args=final_args,
                effective_timeout=effective_timeout,
                cancel_event=cancel_event,
            )

        if lifecycle is None or not lifecycle.registrations(HookEvent.tool_execute):
            return await _run_tool()

        produced: list[ToolResult] = []
        raised: list[BaseException] = []

        async def _next(_context: LifecycleContext) -> None:
            try:
                produced.append(await _run_tool())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raised.append(exc)
                raise

        outcome = await lifecycle.around(
            LifecycleContext(
                point=HookEvent.tool_execute,
                run_id=ctx.run_id,
                session_id=ctx.session_id,
                tenant_id=ctx.tenant_id,
                payload={
                    "tool_name": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "arguments": dict(final_args),
                },
            ),
            _next,
        )
        if raised:
            raise raised[0]
        if outcome.verdict is not LifecycleVerdict.allow:
            raise ToolPolicyDenied(
                outcome.reason
                or f"tool execution refused at {HookEvent.tool_execute.value}"
            )
        if not produced:
            raise ToolPolicyDenied(
                f"{outcome.decided_by or 'a lifecycle handler'} skipped the "
                f"invocation of {tool_call.name!r}"
            )
        return produced[0]

    def _nearest_admitted_names(
        self,
        name: str,
        policy: ToolVisibilityPolicy | None,
        allowlist: frozenset[str] | None = None,
    ) -> list[str]:
        """Up to three registered names close to ``name`` that the policy admits.

        A model that misremembers a name is usually one case change or one
        word away — ``Mcp_github_x`` for ``Mcp_Github_x`` — and without the
        right spelling in front of it, it tends to keep guessing. Only admitted
        names are offered, so a blocked tool is never revealed by being near.
        """
        admitted = {
            tool.name.casefold(): tool.name
            for tool in self._registry.list_all()
            if policy_admits(policy, tool.name)
            and (allowlist is None or tool.name in allowlist)
        }
        close = difflib.get_close_matches(
            name.casefold(), list(admitted), n=_UNKNOWN_TOOL_SUGGESTIONS, cutoff=0.6
        )
        return [admitted[candidate] for candidate in close]

    async def dispatch(
        self,
        *,
        tool_call: ToolCall,
        ctx: ToolContext,
        visibility_policy: ToolVisibilityPolicy,
        timeout_seconds: int,
        subagent_whitelist: Iterable[str] | None = None,
        child_run: bool = False,
        preapproved_tool_call_id: str | None = None,
        admit_evidence: Callable[[tuple[EvidenceRecord, ...], EvidenceProducerBinding], None]
        | None = None,
        on_dispatch_start: Callable[[ToolCall], Awaitable[None]] | None = None,
        lifecycle: ILifecycleRegistry | None = None,
    ) -> AsyncIterator[TurnEvent | DispatchOutcome]:
        """Drive one dispatch lifecycle.

        Yields :class:`TurnEvent` envelopes in Anthropic-style ordering
        (``hook_fired`` for PreToolUse … optional ``tool_transport_starting``
        … ``hook_fired`` for PostToolUse … ``tool_result``). On approval
        path the terminal event is ``tool_call_pending`` (no
        tool_result). The final item yielded is always a
        :class:`DispatchOutcome` — callers iterate yields and switch on
        type.

        Caller responsibility: the LLM-stream ``tool_use_start`` /
        ``tool_use_input_delta`` / ``tool_use_stop`` are emitted by
        :func:`protocore.runtime.query._stream_one_assistant_message`
        BEFORE this dispatcher is invoked. We pick up after the LLM
        finishes the tool_call block.

        The dispatcher never RAISES on tool errors — all failure modes
        surface as :class:`DispatchOutcome` with ``success=False`` and a
        populated :attr:`DispatchOutcome.error_kind`.

        ``on_dispatch_start`` is awaited at exactly one point: after every gate,
        hook, precondition and cancel check has passed, and immediately before
        the tool is invoked. It is the only moment at which a caller can record
        durably that this call is about to happen — earlier is a lie for every
        call a gate then parks or denies, and later is too late for a run that
        dies inside the tool. A caller that persists there gets a record whose
        four outcomes (parked, in flight, asking, settled) can be told apart
        after a crash.

        ``lifecycle`` is the run's registry (:mod:`protocore.contracts.middleware`).
        When one is passed, the invocation is wrapped in the ``around`` chain at
        :attr:`HookEvent.tool_execute` — the one coordinate that can see the call
        go in and the result come out. A handler that skips its ``next`` skips
        the tool; a handler that raises, times out or answers unreadably denies
        the call, which surfaces as a permission failure like any other refusal.
        """
        metadata = copy_metadata(ctx)
        metadata.setdefault("tool_call_id", tool_call.id)
        # The live policy and the child's declared tool set, for policy-aware
        # tools (ToolSearch), so discovery honours the SAME contract the
        # permission gate enforces below. Assigned, never set-if-absent: a
        # value already in the bag is a host's or a stale one, and a search
        # that trusted it could report as loaded a tool the gate then refuses.
        metadata[TOOL_VISIBILITY_POLICY_METADATA_KEY] = visibility_policy
        allowlist = _declared_allowlist(subagent_whitelist)
        if allowlist is None:
            metadata.pop(TOOL_ALLOWLIST_METADATA_KEY, None)
        else:
            metadata[TOOL_ALLOWLIST_METADATA_KEY] = allowlist
        ctx = ctx.model_copy(update={"metadata": metadata})

        # ── Step 1: registry lookup ────────────────────────────────
        tool = self._registry.get(tool_call.name)
        if tool is None:
            # Detect `<*_contract>` XML blocks the
            # model misemits as tool names and return a nudge instead of the
            # raw "unknown tool" error, prompting the model to inline the XML
            # block in its assistant text.
            if _CONTRACT_HALLUCINATION.fullmatch(tool_call.name):
                _rc = _run_constants(ctx)
                msg = _contract_hallucination_hint(
                    tool_call.name,
                    finalize_terminal=bool(
                        getattr(_rc, "agent_finalize_tool_as_terminal", False)
                    ),
                )
            else:
                msg = f"unknown tool: {tool_call.name!r}"
                nearest = self._nearest_admitted_names(
                    tool_call.name, visibility_policy, allowlist
                )
                if nearest:
                    msg += f". Did you mean: {', '.join(nearest)}?"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.unknown_tool, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.unknown_tool, msg),
            )
            return

        # The binding is runtime execution metadata: it is taken from the
        # registered tool after lookup, frozen in the context, and never read
        # from a model-visible result payload.
        ctx = ctx.model_copy(
            update={
                "evidence": (ctx.evidence or ToolEvidenceContext())
                .with_producer_binding(tool.evidence_producer)
            }
        )

        # A call of a tool the request did not advertise was written without
        # its schema; when such a call fails on its arguments, the answer
        # carries the tool's line so the retry is right the first time.
        unseen_signature = _unseen_tool_signature(ctx, tool, visibility_policy, allowlist)

        # ── Step 2: schema validation (input dict shape) ───────────
        # Core ABC accepts dict; tool-specific Pydantic input_model
        # validation lives in the host adapter. We
        # only enforce the byte-cap invariant here.
        #
        # The depth check comes FIRST because the serialisation below cannot be
        # relied on to report the failure: ``json.dumps`` recurses per level and
        # signals exhaustion with ``RecursionError``, which is neither
        # ``TypeError`` nor ``ValueError``, so it escapes the handler and takes
        # the run down from inside the dispatcher. Refusing the call as a
        # validation error keeps the failure where the model can see it and
        # re-issue the call with a sane payload.
        depth_ceiling = _resolve_max_data_nesting_depth(ctx)
        if _nesting_exceeds(tool_call.arguments, depth_ceiling):
            msg = (
                "tool arguments nest deeper than "
                f"{depth_ceiling} levels — re-issue the call with a "
                "flatter payload"
            ) + unseen_signature
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.validation, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.validation, msg),
            )
            return
        try:
            arguments_json = json.dumps(tool_call.arguments, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            msg = f"tool arguments not JSON-serialisable: {exc}" + unseen_signature
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.validation, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.validation, msg),
            )
            return
        del arguments_json  # validation only

        # ── Step 3-4: permission gate ──────────────────────────────
        # Approved resume still runs the full gate. The only narrowed
        # exception is a repeated PreToolUse ``require_approval`` decision for
        # the exact call id that was already approved outside core.
        decision = await self._gate.check(
            tool=tool,
            arguments=tool_call.arguments,
            ctx=ctx,
            visibility_policy=visibility_policy,
            # Frozen once above: an iterator handed in here would otherwise
            # reach the gate already spent, which reads as no declaration.
            subagent_whitelist=allowlist,
            child_run=child_run,
            hook_manager=self._hooks,
            skip_pre_tool_approval=preapproved_tool_call_id == tool_call.id,
        )

        # Emit a HOOK_FIRED(pre_tool_use) event ONLY when the gate
        # actually invoked the PreToolUse hook (decision.stage == hook).
        # Earlier-stage decisions (whitelist / safety_policy / rate_limit)
        # MUST NOT advertise a hook fire — the hook stage never ran.
        # Telemetry only — never changes loop behaviour.
        if self._hooks is not None and decision.stage is PermissionStage.hook:
            yield TurnEvent(
                type=EventType.HOOK_FIRED,
                run_id=ctx.run_id,
                payload={
                    "hook_event": HookEvent.pre_tool_use.value,
                    "outcome": decision.outcome.value,
                    "stage": decision.stage.value,
                    "tool_call_id": tool_call.id,
                },
            )

        if decision.requires_approval:
            yield TurnEvent(
                type=EventType.TOOL_CALL_PENDING,
                run_id=ctx.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "tool_input": tool_call.arguments,
                    "requires_approval": True,
                    "approval_token": decision.approval_token,
                    "reason": decision.reason,
                },
            )
            # NO tool_use_stop yet — the call is paused; pending state
            # owns the lifecycle until user resumes.
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content="",
                is_error=False,
                approval_required=True,
                approval_token=decision.approval_token,
            )
            return

        if decision.denied:
            denial_reason = decision.reason or "blocked by policy"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.permission, denial_reason
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(
                    DispatchErrorKind.permission,
                    denial_reason,
                    {"stage": decision.stage.value},
                ),
            )
            return

        # Apply hook-driven mutation.
        final_args = decision.modified_input if decision.modified_input is not None else tool_call.arguments

        # ── DAG tool-precondition check ──────
        # When the tool's ``ToolDefinition.preconditions`` are unsatisfied
        # by the per-run satisfied-precondition set, return a
        # ``[PRECONDITION NOT MET: ...]`` error envelope BEFORE invoking
        # the tool. The check is gated by
        # ``LoopConstants.tool_preconditions_enabled`` so operators
        # can disable enforcement without redeploying the tool registry.
        precondition_reason = self._check_tool_preconditions(
            tool=tool,
            arguments=final_args,
            ctx=ctx,
        )
        if precondition_reason is not None:
            msg = f"[PRECONDITION NOT MET: {precondition_reason}]"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.permission, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                metadata=_dispatch_replay_metadata(
                    DispatchErrorKind.permission,
                    msg,
                    {"precondition_denial": precondition_reason},
                ),
            )
            return

        # NOTE: ``tool_transport_starting`` is intentionally NOT emitted
        # here. The host's transport owns that event entirely and emits it on
        # a cold start only, with a payload describing how it was brought up.
        # The dispatcher cannot tell a cold start from a warm one and would
        # emit a spurious event on every warm dispatch.

        # ── Step 5: timeout-wrapped execute ────────────────────────
        # A tool that owns its own long-lived async unit (the ``Agent`` tool,
        # whose ``SubagentRunner`` runs an inner ``asyncio.timeout`` + stale
        # watchdog) declares a per-tool budget via ``should_defer`` +
        # ``default_timeout_ms``; honour it so the flat ``rc.tool_timeout_seconds``
        # cap does not cancel the unit before its own classified timeout fires.
        effective_timeout = _resolve_tool_timeout_seconds(
            tool, flat_timeout_seconds=timeout_seconds
        )
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        # #6 cancel propagation — when the executor put a per-run cancel
        # ``asyncio.Event`` on the run state, RACE the tool task against it so a
        # leader parked inside the synchronous ``Agent`` tool (whole subagent)
        # unblocks within the cancel-poll latency instead of ~2 min. Absent ⇒
        # the plain ``await asyncio.wait_for`` path (byte-identical, no-op for
        # older callers / tests). ``ToolDispatchCancelled`` (a CancelledError
        # subclass) propagates PAST the typed except-arms below (none catch
        # BaseException) up to the engine's cooperative cancel unwind + the
        # executor ``_drive_run`` cancelled-arm.
        cancel_event = _run_cancel_event(ctx)
        # Pre-dispatch guard: the gate / hook / precondition checks above ran
        # ``await``s, so a cancel may have landed since the last check. Test it
        # HERE — as close to the invoke point as possible, before the tool
        # coroutine is ever constructed — so a cancelled run never starts a new
        # tool/subagent (the raced helper repeats this check, but this covers
        # the path symmetrically and keeps the contract independent of the
        # helper's internals).
        if cancel_event is not None and cancel_event.is_set():
            _logger.warning(
                "DIAG tool_dispatch.cancelled_pre_dispatch tool=%s run=%s — "
                "cancel set before invoke; not starting the tool",
                tool_call.name,
                ctx.run_id,
            )
            raise ToolDispatchCancelled(
                f"tool dispatch cancelled for run {ctx.run_id!r}"
            )
        if on_dispatch_start is not None:
            await on_dispatch_start(tool_call)
        try:
            tool_result = await self._invoke_within_lifecycle(
                lifecycle=lifecycle,
                tool=tool,
                tool_call=tool_call,
                ctx=ctx,
                final_args=final_args,
                effective_timeout=effective_timeout,
                cancel_event=cancel_event,
            )
        except TimeoutError:
            duration_ms = int((loop.time() - started_at) * 1000)
            timeout_label = (
                int(effective_timeout)
                if effective_timeout.is_integer()
                else effective_timeout
            )
            msg = f"tool {tool_call.name!r} timed out after {timeout_label}s"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.timeout, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                duration_ms=duration_ms,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.timeout, msg),
            )
            return
        except ToolPolicyDenied as exc:
            duration_ms = int((loop.time() - started_at) * 1000)
            reason = str(exc) or "tool policy denied"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.permission, reason
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                duration_ms=duration_ms,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.permission, reason),
            )
            return
        except AskUserPauseRequested as pause:
            # AskUser invoke raises this
            # typed signal to pause the loop. We yield a
            # ``TOOL_CALL_PENDING`` envelope (the same shape used for
            # approval gating) tagged with ``kind="ask_user"`` so
            # the host dispatch shim can fan out to the ask_user
            # resume path instead of the approval store. NO tool_result
            # is emitted — the answer arrives on resume and
            # the host side builds the final ToolResult then.
            duration_ms = int((loop.time() - started_at) * 1000)
            payload_dict = pause.payload.model_dump()
            yield TurnEvent(
                type=EventType.TOOL_CALL_PENDING,
                run_id=ctx.run_id,
                payload={
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "tool_input": tool_call.arguments,
                    "requires_approval": False,
                    "ask_user": True,
                    "kind": "ask_user",
                    "ask_user_payload": payload_dict,
                },
            )
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content="",
                is_error=False,
                ask_user_required=True,
                ask_user_payload=payload_dict,
                duration_ms=duration_ms,
            )
            return
        except Exception as exc:
            duration_ms = int((loop.time() - started_at) * 1000)
            msg = f"tool {tool_call.name!r} execution failed: {exc}"
            if isinstance(exc, _ARGUMENT_ERRORS):
                msg += unseen_signature
            _logger.warning(
                "tool dispatch raised for tool=%s call_id=%s",
                tool_call.name,
                tool_call.id,
                exc_info=True,
            )
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.execution, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            # Surface a machine-readable give-up signal. A tool may attach a
            # ``structured_error`` mapping to the exception it raises (e.g. a
            # retry-budget exhaustion carries
            # ``{"finalization_recommended": True, ...}``). Forward it verbatim
            # on the DispatchOutcome metadata so the loop / model can act on the
            # finalize hint instead of treating it as an opaque execution error.
            # Generic: any exception with a dict ``structured_error`` qualifies;
            # absent (the common case) leaves the metadata bit-identical.
            structured_error = getattr(exc, "structured_error", None)
            structured_error_extra: dict[str, Any] | None = (
                {DISPATCH_STRUCTURED_ERROR_METADATA_KEY: structured_error}
                if isinstance(structured_error, dict)
                else None
            )
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                duration_ms=duration_ms,
                metadata=_dispatch_replay_metadata(
                    DispatchErrorKind.execution, msg, structured_error_extra
                ),
            )
            return

        duration_ms = int((loop.time() - started_at) * 1000)
        if not isinstance(tool_result, ToolResult):
            msg = f"tool {tool_call.name!r} returned non-ToolResult shape: {type(tool_result).__name__}"
            final_kind, final_msg = self._apply_consecutive_error_cap(
                ctx, tool_call.name, DispatchErrorKind.execution, msg
            )
            async for evt in self._emit_failure(tool_call, ctx, final_kind, final_msg):
                yield evt
            yield DispatchOutcome(
                tool_call=tool_call,
                success=False,
                content=final_msg,
                is_error=True,
                error_kind=final_kind,
                duration_ms=duration_ms,
                metadata=_dispatch_replay_metadata(DispatchErrorKind.execution, msg),
            )
            return

        success = not tool_result.is_error
        # ``content`` from here on is the MODEL PROJECTION — the text that
        # will sit in the transcript. The canonical value is kept beside it
        # and is never rewritten by anything below: the error paths, the
        # consecutive-error cap and the PostToolUse hook all speak to the
        # model, and a message written for the model is not the value the
        # tool produced.
        canonical_content = tool_result.content
        content = tool_result.model_content
        # What the parallel path's replay rebuilds a failed result's text from.
        replay_message = tool_result.content
        ui_payload = tool_result.ui_payload
        canonical_ref = tool_result.canonical_ref
        result_path = tool_result.path
        result_metadata = dict(tool_result.metadata)
        evidence_records = tool_result.evidence_records
        evidence_producer: EvidenceProducerBinding | None = None
        if success and evidence_records:
            evidence_producer = (
                ctx.evidence.producer_binding if ctx.evidence is not None else None
            )
            if evidence_producer is None:
                success = False
                content = "tool evidence rejected: registered tool has no evidence producer binding"
                canonical_content = content
                ui_payload = None
                canonical_ref = None
                evidence_records = ()
            else:
                origin = ctx.evidence.origin if ctx.evidence is not None else None
                if origin is None:
                    success = False
                    content = "tool evidence rejected: dispatch context has no evidence origin"
                    canonical_content = content
                    ui_payload = None
                    canonical_ref = None
                    evidence_records = ()
                    evidence_producer = None
                else:
                    evidence_records = tuple(
                        record.model_copy(
                            update={
                                "origin": origin,
                                "producer_id": evidence_producer.producer_id,
                                "producer_revision": evidence_producer.producer_revision,
                            }
                        )
                        for record in evidence_records
                    )
                    if not _admission_deferred(ctx):
                        if admit_evidence is None:
                            success = False
                            content = "tool evidence rejected: dispatcher has no private ledger admission channel"
                            canonical_content = content
                            ui_payload = None
                            canonical_ref = None
                            evidence_records = ()
                            evidence_producer = None
                        else:
                            try:
                                admit_evidence(evidence_records, evidence_producer)
                            except ValueError as exc:
                                _logger.warning(
                                    "tool evidence rejected run=%s call_id=%s error=%s",
                                    ctx.run_id,
                                    tool_call.id,
                                    type(exc).__name__,
                                )
                                success = False
                                content = f"tool evidence rejected: {exc}"
                                canonical_content = content
                                ui_payload = None
                                canonical_ref = None
                                evidence_records = ()
                                evidence_producer = None
        # A tool that returns ``ToolResult(is_error=True)`` still counts as
        # a tool error for the per-run counter (parity with the
        # dispatcher-detected error paths in :meth:`_emit_failure`). The
        # PostToolUse hook below may rewrite
        # content but cannot flip ``success`` back to ``True``, so this is
        # the canonical point to record the increment.
        if not success:
            # A tool may stamp its soft ``is_error=True`` result as NOT a
            # genuine failure for counting/capping purposes.
            # ``Bash`` does this for an ordinary nonzero process exit (a
            # ``grep -q`` no-match, a ``test``/``[`` false, a ``diff``
            # difference — exit status used as DATA, not a tool failure). Such
            # results must NOT increment ``tool_errors_count`` (which would
            # downgrade an otherwise-successful run to ``partial`` via
            # the host terminal classifier) and must NOT feed the generic
            # consecutive-error cap. The flags default to True (absent ⇒
            # count + cap) so every historical soft/hard error path is
            # unchanged. A genuine hard failure (e.g. a Bash timeout) leaves
            # both flags True and remains fully count/cap eligible.
            count_as_error = _metadata_flag(
                result_metadata,
                TOOL_RESULT_COUNT_AS_ERROR_METADATA_KEY,
                default=True,
            )
            cap_eligible = _metadata_flag(
                result_metadata,
                TOOL_RESULT_CONSECUTIVE_CAP_ELIGIBLE_METADATA_KEY,
                default=True,
            )
            if count_as_error and unseen_signature:
                # A tool that checks its own arguments reports a bad one as an
                # error result, not an exception, so every counted failure of an
                # unseen tool gets the line; a benign exit status does not.
                content += unseen_signature
                replay_message += unseen_signature
            if count_as_error:
                await self._record_tool_error(ctx)
            if cap_eligible:
                # Soft is_error path participates in
                # the consecutive-error cap. The streak is keyed on
                # ``(tool_name, signature(execution, content))`` which matches
                # how dispatcher-detected execution errors are keyed below.
                soft_kind, soft_msg = self._apply_consecutive_error_cap(
                    ctx, tool_call.name, DispatchErrorKind.execution, content
                )
                content = soft_msg
                soft_error_kind = soft_kind
            else:
                # Not cap-eligible: leave the streak untouched and surface the
                # original error kind/content. An ordinary nonzero exit is not
                # a "repeat the same failing call" signal, so it must neither
                # advance nor reset the streak (a real failure interleaved with
                # benign predicate exits should still accumulate).
                soft_error_kind = DispatchErrorKind.execution
        else:
            soft_error_kind = None
            # A gathered call is replayed in transcript order.  Its helper
            # mutations must wait for that replay, where deferred evidence is
            # admitted first; otherwise a completion-order success could
            # satisfy a dependency before a later rejection turns it into an
            # error.  Serial dispatch commits the normal state immediately.
            if not _admission_deferred(ctx):
                # Successful tool result breaks the consecutive-error streak so a
                # subsequent failure restarts at count=1. Without this reset,
                # ``ToolA(err)…ToolA(ok)…ToolA(err)`` would carry the count
                # across the success, contrary to the "consecutive" semantics.
                self._reset_consecutive_error_streak(ctx)

                # ── DAG record_satisfaction ──────
                # On successful dispatch, persist that this (tool_name, path)
                # tuple has been satisfied so future tools whose
                # ``preconditions`` reference it can proceed.
                self._record_precondition_satisfaction(
                    tool=tool,
                    tool_name=tool_call.name,
                    arguments=final_args,
                    ctx=ctx,
                )

        # ── Step 7: PostToolUse hook (may mutate output) ───────────
        post_tool_output_modified = False
        if self._hooks is not None:
            try:
                post_hook = await self._hooks.invoke(
                    HookEvent.post_tool_use,
                    {
                        "run_id": ctx.run_id,
                        "tenant_id": ctx.tenant_id,
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "tool_output": content,
                        "success": success,
                        "duration_ms": duration_ms,
                    },
                    ctx.tenant_id,
                )
            except Exception:
                _logger.warning(
                    "PostToolUse hook raised for tool=%s; isolating",
                    tool_call.name,
                    exc_info=True,
                )
                post_hook = None
            if post_hook is not None:
                yield TurnEvent(
                    type=EventType.HOOK_FIRED,
                    run_id=ctx.run_id,
                    payload={
                        "hook_event": HookEvent.post_tool_use.value,
                        "outcome": ("modify" if post_hook.action == HookActionKind.MODIFY else "success"),
                        "tool_call_id": tool_call.id,
                    },
                )
                if post_hook.action == HookActionKind.MODIFY:
                    modified = post_hook.modifications.get("tool_output")
                    if isinstance(modified, str):
                        content = modified
                        post_tool_output_modified = True

        outcome_metadata: dict[str, Any] | None = result_metadata or None
        if not success:
            replay_extra = (
                {DISPATCH_POST_TOOL_OUTPUT_MODIFIED_METADATA_KEY: True}
                if post_tool_output_modified
                else None
            )
            outcome_metadata = {
                **result_metadata,
                **_dispatch_replay_metadata(
                    DispatchErrorKind.execution,
                    replay_message,
                    replay_extra,
                ),
            }

        # ── Step 6: ToolResult envelope event ──────────────────────
        # ``success`` and ``is_error`` are both stated, and they mirror each
        # other by construction. Two readers grew up on this payload asking
        # opposite questions, and a client that reads only the one its author
        # happened to pick must not see a failed call as a success.
        #
        # ``ui_payload`` leaves the runtime here and nowhere else: this event
        # is the UI's channel, and the transcript built further down carries
        # the projection only.
        yield TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=ctx.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "success": success,
                "is_error": not success,
                "duration_ms": duration_ms,
                "content_blocks": [{"type": "text", "text": content}],
                **({"ui_payload": ui_payload} if ui_payload else {}),
                **({"canonical_ref": canonical_ref} if canonical_ref else {}),
                **({"path": result_path} if result_path else {}),
                **({"metadata": outcome_metadata} if outcome_metadata else {}),
            },
        )

        yield DispatchOutcome(
            tool_call=tool_call,
            success=success,
            content=content,
            is_error=not success,
            error_kind=None if success else soft_error_kind,
            duration_ms=duration_ms,
            metadata=outcome_metadata,
            evidence_records=evidence_records,
            evidence_producer=evidence_producer,
            canonical_content=(
                None if canonical_content == content else canonical_content
            ),
            ui_payload=ui_payload,
            canonical_ref=canonical_ref,
            path=result_path,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _emit_failure(
        self,
        tool_call: ToolCall,
        ctx: ToolContext,
        kind: DispatchErrorKind,
        message: str,
    ) -> AsyncIterator[TurnEvent]:
        """Emit the terminal ``tool_result(error=...)`` envelope.

 The LLM-stream ``tool_use_start`` / ``tool_use_stop`` for the
 originating call were emitted upstream — they are not
 synthesised here. Every failure path of :meth:`dispatch` calls
 this helper exactly once for symmetry.

 Increments the run's ``tool_errors_count`` (atomic, idempotent) BEFORE
 yielding the error envelope so the terminal classifier can downgrade runs
 with tool errors to ``partial`` status.
 """
        await self._record_tool_error(ctx)
        yield TurnEvent(
            type=EventType.TOOL_RESULT,
            run_id=ctx.run_id,
            payload={
                "tool_call_id": tool_call.id,
                "success": False,
                "is_error": True,
                "error": {
                    "kind": kind.value,
                    "message": message,
                },
                "content_blocks": [{"type": "text", "text": message}],
            },
        )


# The exceptions a tool raises when the arguments it was called with do not
# fit it: a pydantic model's ValidationError (a ValueError), a missing or
# unexpected keyword (TypeError), a required key read straight off the dict.
_ARGUMENT_ERRORS: Final[tuple[type[Exception], ...]] = (TypeError, ValueError, KeyError)


def _declared_allowlist(subagent_whitelist: Iterable[str] | None) -> frozenset[str] | None:
    """A child's declared tool set as the gate reads it: ``None`` or empty admits all."""
    if subagent_whitelist is None:
        return None
    allow = frozenset(subagent_whitelist)
    return allow or None


def _unseen_tool_signature(
    ctx: ToolContext,
    tool: Tool,
    policy: ToolVisibilityPolicy,
    allowlist: frozenset[str] | None,
) -> str:
    """The tool's line, to append to a failure, when the model never saw its schema.

    Empty when the loop did not say what it advertised (a dispatch outside a
    run) or when the tool was on the list, where the schema is already in
    front of the model and repeating it is noise. Empty, too, for a tool the
    policy or the child's declared set refuses: the argument checks run before
    the gate, so a blocked tool called with bad arguments was answered with
    its parameters and told it was "loaded now", which it never is.
    """
    advertised = read_metadata(ctx, ADVERTISED_TOOLS_METADATA_KEY)
    if not isinstance(advertised, frozenset | set | tuple | list) or tool.name in advertised:
        return ""
    if not policy_admits(policy, tool.name) or (allowlist is not None and tool.name not in allowlist):
        return ""
    return (
        "\nThis tool was not in your tool list, so it was called without its "
        f"parameters; it is loaded now. It takes: {tool_line(tool.definition)}"
    )


def consume_transport_down_injection_signal(
    state: RunScopedState | None,
) -> bool:
    """Public consumer for the transport-down injection-pending flag.

 The loop driving a run calls this once per dispatched ``TurnEvent``.
 Returns ``True`` exactly once per streak (when
 :meth:`ToolDispatcher._track_transport_down_streak` raised the flag).
 Subsequent calls in the same streak return ``False`` until a successful
 tool call or another error resets the counter.

 Parameters
 ----------
 state:
 The run's state. ``None`` is treated as "no signal" — the function
 never raises.

 Returns
 -------
 bool
 ``True`` iff the dispatcher raised the flag and this is the first
 consumer to take it. ``False`` otherwise.
 """
    if state is None or not state.transport_down_injection_pending:
        return False
    state.transport_down_injection_pending = False
    return True


__all__ = [
    "DispatchErrorKind",
    "DispatchOutcome",
    "ToolDispatcher",
    "ToolPermissionDecision",
    "ToolPermissionOutcome",
    "consume_transport_down_injection_signal",
]
