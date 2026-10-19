"""``ToolPermissionGate`` — the 4-stage permission decision pipeline.

md`. Orders the gating concerns:

1. Tenant tool whitelist (or subagent restricted scope).
2. Per-side-effect-class safety policy
 (``DefaultShellSafetyPolicy`` for Bash, DNS allowlist for WebFetch,
 denied-paths for workspace tools).
3. Rate limit (host-side — left as a Protocol seam in core; the gate
 only knows how to ASK).
4. ``PreToolUse`` hook fires last and may override (allow / deny /
 require_approval / mutate input). The hook is the **highest-leverage
 point** for tenants who want an LLM-as-a-policy gate.

Core ships the *default* safety policies (shell deny patterns are in
:mod:`protocore.safety.shell`; DNS allowlist is a static-set check).
The host can stack additional policies by composing via
:meth:`ToolPermissionGate.register_policy` — keeps the core API frozen.

Result envelope: :class:`ToolPermissionDecision` carries an outcome
(allow / deny / require_approval), the optional rewritten input args
(``modify`` action from hooks), the reason string, and the originating
stage for telemetry/audit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable
from urllib.parse import urlparse

from protocore.contracts.hooks import HookActionKind, HookResult, IHookManager
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tool_roles import (
    EMPTY_TOOL_ROLE_MAP,
    WORKSPACE_MUTATION_ROLES,
    ToolArgumentSlot,
    ToolRole,
    ToolRoleMap,
)
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import HookEvent
from protocore.logging_utils import get_logger
from protocore.runtime.tool_arguments import (
    argument_names,
    present_string_argument,
    string_argument,
)
from protocore.safety.shell import (
    DefaultShellSafetyPolicy,
    ShellPolicyVerdict,
)

_logger = get_logger(__name__)

_warned_undeclared_slots: set[str] = set()


def _warn_once(message: str, subject: str) -> None:
    """Say a structural misconfiguration out loud, once per subject.

    A missing declaration is a property of the installation, not of the call,
    so repeating it on every call would bury the log without telling anyone
    anything new.
    """
    key = f"{message}:{subject}"
    if key in _warned_undeclared_slots:
        return
    _warned_undeclared_slots.add(key)
    _logger.warning(message, subject)


def _session_grant_covers(ctx: ToolContext, arguments: dict[str, Any]) -> bool:
    """A session grant skips approval only; denials still stand."""
    command = str((arguments or {}).get("command") or "")
    if not command:
        return False
    state = ctx.run_state
    grants = list(state.session_grants) if state is not None else []
    if not grants:
        return False
    from protocore.runtime.permission_widen import grant_covers

    return any(grant_covers(grant, command) for grant in grants)


# Side-effect class strings used to fan out to per-class safety policies.
# Defined as constants here rather than RC because they are protocol
# values — not user-tunable. Matches the tool catalog in ``docs/tools.md``.
SIDE_EFFECT_SANDBOX: Final[str] = "sandbox"
SIDE_EFFECT_HTTP: Final[str] = "http"
SIDE_EFFECT_WORKSPACE: Final[str] = "workspace"
SIDE_EFFECT_STATE_ONLY: Final[str] = "state_only"

def side_effect_class_for_roles(held: frozenset[ToolRole]) -> str | None:
    """The side-effect class a set of roles implies, or None when they imply none.

    The classification is a protocol invariant — which classes exist and what
    each one costs is core's business — but WHICH tool is in which class is the
    host's, and the roles it declared say so: a tool that runs a command line
    is sandbox-class whatever it is called, a tool that reaches a named network
    host is http-class, and a tool that changes a file is workspace-class.
    Roles that change nothing outside the run map to no class at all, which the
    caller reads as the most permissive one.
    """
    if ToolRole.runs_shell in held:
        return SIDE_EFFECT_SANDBOX
    if ToolRole.fetches_url in held:
        return SIDE_EFFECT_HTTP
    if held & WORKSPACE_MUTATION_ROLES:
        return SIDE_EFFECT_WORKSPACE
    return None


class ToolPermissionOutcome(StrEnum):
    """Permission gate decision outcome."""

    allow = "allow"
    deny = "deny"
    require_approval = "require_approval"


class PermissionStage(StrEnum):
    """Stage at which a decision was reached (for telemetry)."""

    whitelist = "whitelist"
    safety_policy = "safety_policy"
    rate_limit = "rate_limit"
    hook = "hook"
    default = "default"


@dataclass(frozen=True, slots=True)
class ToolPermissionDecision:
    """Outcome envelope from :meth:`ToolPermissionGate.check`.

    Attributes
    ----------
    outcome:
        ``allow`` — proceed with execute.
        ``deny`` — abort, surface as ``tool_result(success=false)``.
        ``require_approval`` — pause turn (``AWAITING``), emit
        ``tool_call_pending``, resume via user approval.
    reason:
        Human-readable explanation (logged + surfaced in audit + payload
        for ``deny`` results).
    stage:
        Which gate produced the verdict — drives telemetry/buckets.
    modified_input:
        ``None`` if input passes through unchanged; otherwise the
        mutated dict produced by a ``MODIFY``-action ``PreToolUse`` hook.
    approval_token:
        Opaque token issued by the hook manager when an approval is
        required; carried back via the user-resume payload.
    """

    outcome: ToolPermissionOutcome
    reason: str = ""
    stage: PermissionStage = PermissionStage.default
    modified_input: dict[str, Any] | None = None
    approval_token: str | None = None

    @property
    def allowed(self) -> bool:
        return self.outcome is ToolPermissionOutcome.allow

    @property
    def denied(self) -> bool:
        return self.outcome is ToolPermissionOutcome.deny

    @property
    def requires_approval(self) -> bool:
        return self.outcome is ToolPermissionOutcome.require_approval


# ---------------------------------------------------------------------------
# Per-side-effect-class safety policies
# ---------------------------------------------------------------------------


@runtime_checkable
class IToolSafetyPolicy(Protocol):
    """Per-side-effect-class safety policy Protocol.

    Implementations check tool input args against per-class rules
    (denied paths for workspace tools, deny patterns for sandbox tools,
    DNS allowlist for HTTP tools). Stateless — config is bound at
    construction time.
    """

    def applies_to(self, side_effect_class: str) -> bool:
        """Return ``True`` if this policy handles ``side_effect_class``."""
        ...

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolPermissionDecision:
        """Evaluate the call; return :class:`ToolPermissionDecision`."""
        ...


@dataclass(frozen=True, slots=True)
class ShellSafetyPolicyAdapter(IToolSafetyPolicy):
    """Wraps :class:`DefaultShellSafetyPolicy` for the sandbox class.

    Reads the shell command from every spelling the host declared for it and
    evaluates via the deny-pattern policy. The command must be found under any
    of them: the gate sees raw arguments, before the tool's own input model has
    resolved an alias, so a call that spells the command differently would
    otherwise skip the deny patterns and still execute.

    A host that declares no spelling at all leaves this policy unable to read
    the command it exists to inspect. That is a policy failure, not a pass: the
    call goes to a person instead of running unexamined.
    """

    policy: DefaultShellSafetyPolicy = field(default_factory=DefaultShellSafetyPolicy)
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP

    def applies_to(self, side_effect_class: str) -> bool:
        return side_effect_class == SIDE_EFFECT_SANDBOX

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolPermissionDecision:
        del ctx
        if not argument_names(ToolArgumentSlot.shell_command, roles=self.roles):
            _warn_once(
                "shell deny patterns cannot run for %r: no argument spelling is "
                "declared for the command, so the call is sent for approval "
                "instead of being examined",
                tool.name,
            )
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.require_approval,
                reason="shell command argument spelling is not declared",
                stage=PermissionStage.safety_policy,
            )
        command = present_string_argument(
            arguments, ToolArgumentSlot.shell_command, roles=self.roles
        )
        if not isinstance(command, str):
            return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)
        verdict = self.policy.evaluate(command)
        if verdict.verdict is ShellPolicyVerdict.deny:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=verdict.reason,
                stage=PermissionStage.safety_policy,
            )
        return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)


@dataclass(frozen=True, slots=True)
class HttpDnsAllowlistPolicy(IToolSafetyPolicy):
    """DNS allowlist policy for the ``http`` side-effect class.

 Reads the target from every spelling the host declared for the ``url``
 slot. Empty allowlist means "allow any host" (default) — the host
 registers a populated allowlist via
 :meth:`ToolPermissionGate.register_policy`. Blocked hosts always reject
 regardless of allowlist. A host that declares no spelling leaves the
 policy unable to read the address it exists to check, so the call is sent
 for approval rather than allowed unexamined.

 NOTE: this is the *core* policy and is intentionally lightweight —
 it does NOT perform DNS resolution (no network in pure core).
 Full SSRF guard (private-IP / cloud-metadata / link-local resolution)
 lives in the host adapter and stacks on top of this gate.

 Host matching is **case-insensitive** (RFC 4343): the configured
 ``allowed_hosts`` / ``blocked_hosts`` are ASCII-lowercased at
 construction so a mixed-case operator entry (e.g. ``"Evil.com"``) still
 matches the always-lowercase host that :func:`urllib.parse.urlparse`
 produces . We use ``str.lower`` (not ``str.casefold``) to mirror
 ``urlparse``'s own ASCII-lowercasing exactly — ``casefold`` would
 over-fold some non-ASCII characters (e.g. ``faß`` → ``fass``) and could
 silently widen the allowlist to distinct hostnames.
 """

    allowed_hosts: frozenset[str] = field(default_factory=frozenset)
    blocked_hosts: frozenset[str] = field(default_factory=frozenset)
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP

    def __post_init__(self) -> None:
        # Normalise host sets once at construction so comparisons against
        # ``urlparse().hostname`` (always ASCII-lowercase) are
        # case-insensitive. The dataclass is frozen+slots, so mutate via
        # object.__setattr__.
        object.__setattr__(
            self,
            "allowed_hosts",
            frozenset(h.lower() for h in self.allowed_hosts),
        )
        object.__setattr__(
            self,
            "blocked_hosts",
            frozenset(h.lower() for h in self.blocked_hosts),
        )

    def applies_to(self, side_effect_class: str) -> bool:
        return side_effect_class == SIDE_EFFECT_HTTP

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolPermissionDecision:
        del ctx
        if not argument_names(ToolArgumentSlot.url, roles=self.roles):
            _warn_once(
                "the host allowlist cannot run for %r: no argument spelling is "
                "declared for the target address, so the call is sent for "
                "approval instead of being checked",
                tool.name,
            )
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.require_approval,
                reason="target address argument spelling is not declared",
                stage=PermissionStage.safety_policy,
            )
        url = string_argument(arguments, ToolArgumentSlot.url, roles=self.roles)
        if not isinstance(url, str):
            return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)
        parsed = urlparse(url)
        # ``urlparse`` ASCII-lowercases the hostname; lowercase again (no-op
        # for ASCII) so both sides of the membership test agree without the
        # over-folding ``casefold`` would introduce for non-ASCII hosts.
        host = (parsed.hostname or "").lower()
        if not host:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason="url missing host component",
                stage=PermissionStage.safety_policy,
            )
        if host in self.blocked_hosts:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=f"host {host!r} is on the blocklist",
                stage=PermissionStage.safety_policy,
            )
        if self.allowed_hosts and host not in self.allowed_hosts:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=f"host {host!r} not in DNS allowlist",
                stage=PermissionStage.safety_policy,
            )
        return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)


@dataclass(frozen=True, slots=True)
class WorkspacePathPolicy(IToolSafetyPolicy):
    """Path-prefix denial policy for the ``workspace`` side-effect class.

 Reads the path from every spelling the host declared for it and denies
 any resolved path with a prefix in ``denied_path_prefixes``. Empty set
 means "allow all".
 """

    denied_path_prefixes: frozenset[str] = field(default_factory=frozenset)
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP

    def applies_to(self, side_effect_class: str) -> bool:
        return side_effect_class == SIDE_EFFECT_WORKSPACE

    def evaluate(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolPermissionDecision:
        del tool, ctx
        path = present_string_argument(
            arguments, ToolArgumentSlot.path, roles=self.roles
        )
        if not isinstance(path, str):
            return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)
        for prefix in self.denied_path_prefixes:
            if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
                return ToolPermissionDecision(
                    outcome=ToolPermissionOutcome.deny,
                    reason=f"path {path!r} is denied by tenant policy",
                    stage=PermissionStage.safety_policy,
                )
        return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class ToolPermissionGate:
    """4-stage permission decision pipeline.

 Stage ordering **Whitelist** — :class:`ToolVisibilityPolicy` (and optional
 subagent narrowing whitelist) must permit the tool name.
 2. **Safety policies** — per-side-effect-class checks via
 :class:`IToolSafetyPolicy` chain.
 3. **Rate limit** — host-only; baseline returns
 :class:`ToolPermissionOutcome.allow`. Subclass or compose to
 inject a Redis-backed bucket.
 4. **PreToolUse hook** — final, highest-leverage stage; may flip
 allow → deny / require_approval / modify.

 The gate is async-callable (``check``) — the hook stage is async by
 nature of :class:`IHookManager.invoke`.
 """

    def __init__(
        self,
        *,
        policies: Iterable[IToolSafetyPolicy] | None = None,
        side_effect_map: dict[str, str] | None = None,
        roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
    ) -> None:
        self._roles = roles
        # ``None`` is the defaults and an empty list is none. An ``or`` here
        # once read the two alike, so a host that passed ``[]`` to turn the
        # shell deny list off got it back without being told.
        self._policies: list[IToolSafetyPolicy] = list(
            self._default_policies(roles) if policies is None else policies
        )
        self._side_effect_map: dict[str, str] = dict(side_effect_map or {})

    @staticmethod
    def _default_policies(roles: ToolRoleMap) -> list[IToolSafetyPolicy]:
        """Default policy stack — shell deny patterns only (no DNS / path
        config in core baseline; the host stacks tenant-specific
        instances on top via :meth:`register_policy`)."""
        return [ShellSafetyPolicyAdapter(roles=roles)]

    def register_policy(self, policy: IToolSafetyPolicy) -> None:
        """Stack an additional policy. Evaluated after the defaults."""
        self._policies.append(policy)

    def set_side_effect_class(self, tool_name: str, side_effect_class: str) -> None:
        """Override the side-effect class for a tool name.

        Used by tests + the host registry to state a classification the tool's
        roles do not already imply.
        """
        self._side_effect_map[tool_name] = side_effect_class

    def classify(self, tool: Tool) -> str:
        """Return the side-effect class for ``tool``.

 Falls back to ``state_only`` for unregistered tool names — the
 most permissive class. The host adapters typically declare
 their classification via a ``ClassVar`` on the concrete impl
 set via ``set_side_effect_class`` on registration.
 """
        # Honour an explicit attribute on the tool first (a host's adapters
        # declare ``side_effect_class`` as a ``ClassVar[str]``).
        attr = getattr(tool, "side_effect_class", None)
        if isinstance(attr, str):
            return attr
        override = self._side_effect_map.get(tool.name)
        if override is not None:
            return override
        implied = side_effect_class_for_roles(self._roles.roles_of(tool.name))
        return implied if implied is not None else SIDE_EFFECT_STATE_ONLY

    # ------------------------------------------------------------------
    # check — the entry point
    # ------------------------------------------------------------------

    async def check(
        self,
        *,
        tool: Tool,
        arguments: dict[str, Any],
        ctx: ToolContext,
        visibility_policy: ToolVisibilityPolicy,
        subagent_whitelist: Iterable[str] | None = None,
        child_run: bool = False,
        hook_manager: IHookManager | None = None,
        skip_pre_tool_approval: bool = False,
    ) -> ToolPermissionDecision:
        """Run the 4-stage pipeline; return the first non-allow decision.

        Parameters
        ----------
        tool:
            The registered :class:`Tool` instance.
        arguments:
            Already-parsed input dict.
        ctx:
            :class:`ToolContext` for this dispatch.
        visibility_policy:
            Tenant-level :class:`ToolVisibilityPolicy` (visible/blocked).
        subagent_whitelist:
            Optional narrow scope when invoked inside a subagent — if
            non-None and non-empty, only these names are permitted in
            addition to the tenant policy.
        child_run:
            Whether this call is being made by a delegated run. A tool the
            host declared ``never_delegated`` is refused for one, whatever
            else permits it — including the tool-surface floor, and including
            a subagent that declared nothing and therefore has no allow-list
            stage to narrow. The catalogue applies the same rule when it
            resolves a child's surface; both halves are needed, because a
            child that named a tool it was never shown used to reach it.
        hook_manager:
            Optional :class:`IHookManager` to fire ``PreToolUse``. If
            ``None``, the hook stage is skipped (used by tests that
            isolate the policy stack).
        skip_pre_tool_approval:
            Treat a hook-level ``require_approval`` result as already
            satisfied for a previously approved resume. Earlier gate stages
            and hook denials/modifications still apply.
        """
        # Apply the whitelist.
        if visibility_policy.blocked and tool.name in visibility_policy.blocked:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=f"tool {tool.name!r} is on the tenant blocked list",
                stage=PermissionStage.whitelist,
            )
        # The allowed set under a non-empty ``visible`` whitelist is
        # ``visible | pinned | forced_pinned``. ``forced_pinned`` is the core
        # tool-surface floor and ``pinned`` is the progressive-
        # discovery set — both are advertised to the model unconditionally
        # (``ToolRegistry.compute_effective_surface`` / ``_floored_visible_tools``),
        # so dispatch MUST permit them too, or the model receives a callable
        # tool schema that deterministically fails at execution — recreating the
        # cause-#3 collapse one step later. ``blocked`` already overrode above.
        if visibility_policy.visible:
            allowed = (
                visibility_policy.visible
                | visibility_policy.pinned
                | visibility_policy.forced_pinned
            )
            if tool.name not in allowed:
                return ToolPermissionDecision(
                    outcome=ToolPermissionOutcome.deny,
                    reason=f"tool {tool.name!r} is not in the tenant visible set",
                    stage=PermissionStage.whitelist,
                )
        if child_run and ToolRole.never_delegated in self._roles.roles_of(tool.name):
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=f"tool {tool.name!r} is never offered to a delegated run",
                stage=PermissionStage.whitelist,
            )
        if subagent_whitelist is not None:
            allow = frozenset(subagent_whitelist)
            if allow and tool.name not in allow:
                return ToolPermissionDecision(
                    outcome=ToolPermissionOutcome.deny,
                    reason=(f"tool {tool.name!r} not in subagent whitelist"),
                    stage=PermissionStage.whitelist,
                )

        # Apply per-side-effect safety policies.
        side_effect = self.classify(tool)
        for policy in self._policies:
            if not policy.applies_to(side_effect):
                continue
            decision = policy.evaluate(tool, arguments, ctx)
            if decision.outcome is not ToolPermissionOutcome.allow:
                return decision

        # Apply the rate limit (currently a no-op).
        # The host composes a RedisLuaRateLimitPolicy in via
        # :meth:`register_policy` — it is just another
        # :class:`IToolSafetyPolicy` that returns deny with
        # PermissionStage.rate_limit when the bucket is exhausted.

        # Run the PreToolUse hook.
        if hook_manager is None:
            return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)

        try:
            hook_result = await hook_manager.invoke(
                HookEvent.pre_tool_use,
                {
                    "run_id": ctx.run_id,
                    "tenant_id": ctx.tenant_id,
                    "tool_name": tool.name,
                    "tool_input": arguments,
                    "side_effect_class": side_effect,
                },
                ctx.tenant_id,
            )
        except Exception:
            # Fail CLOSED. This stage exists to answer whether a call may run,
            # and a stage that cannot answer has not said yes. Treating its
            # failure as consent turned every outage of the hook executor into
            # a silent, run-wide removal of the permission layer — the caller
            # saw an allowed call and no sign that anything had been skipped.
            _logger.warning(
                "PreToolUse hook raised for tool=%s; denying",
                tool.name,
                exc_info=True,
            )
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason="hook dispatch failed; denying",
                stage=PermissionStage.hook,
            )

        decision = self._project_hook_result(hook_result, arguments)
        if skip_pre_tool_approval and decision.requires_approval:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.allow,
                reason="pre-tool approval already satisfied",
                stage=PermissionStage.hook,
            )
        if decision.requires_approval and _session_grant_covers(ctx, arguments):
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.allow,
                reason="session_grant_covers",
                stage=PermissionStage.hook,
            )
        return decision

    @staticmethod
    def _project_hook_result(
        hook_result: HookResult,
        original_args: dict[str, Any],
    ) -> ToolPermissionDecision:
        """Translate a :class:`HookResult` to a :class:`ToolPermissionDecision`."""
        if hook_result.modifications.get("requires_approval"):
            token = hook_result.modifications.get("approval_token")
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.require_approval,
                reason=hook_result.reason or "",
                stage=PermissionStage.hook,
                approval_token=token if isinstance(token, str) else None,
            )

        if hook_result.action == HookActionKind.DENY:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason=hook_result.reason or "blocked by policy",
                stage=PermissionStage.hook,
            )

        if hook_result.action == HookActionKind.MODIFY:
            modified = hook_result.modifications.get("tool_input")
            if isinstance(modified, dict):
                return ToolPermissionDecision(
                    outcome=ToolPermissionOutcome.allow,
                    reason=hook_result.reason or "modified by hook",
                    stage=PermissionStage.hook,
                    modified_input=dict(modified),
                )
            # MODIFY without tool_input: fall through as allow.
            del original_args  # unchanged
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.allow,
                reason=hook_result.reason or "",
                stage=PermissionStage.hook,
            )

        return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)


__all__ = [
    "SIDE_EFFECT_HTTP",
    "SIDE_EFFECT_SANDBOX",
    "SIDE_EFFECT_STATE_ONLY",
    "SIDE_EFFECT_WORKSPACE",
    "HttpDnsAllowlistPolicy",
    "IToolSafetyPolicy",
    "PermissionStage",
    "ShellSafetyPolicyAdapter",
    "ToolPermissionDecision",
    "ToolPermissionGate",
    "ToolPermissionOutcome",
    "WorkspacePathPolicy",
]
