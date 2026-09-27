"""Hold back the tool groups a surface cannot afford, and name them instead.

Advertising every tool is the best surface there is for as long as it fits:
the model sees each schema, picks the dedicated tool, and fills its arguments
under the provider's own constraints. It stops fitting in two ways, and both
are walls rather than slopes. The definitions can take so much of the context
window that the conversation no longer has room, and a provider can refuse a
request outright above a fixed number of tools. A host that connects a large
MCP server reaches either wall on the first request.

So a host may sort its catalogue into GROUPS
(:class:`~protocore.contracts.tool_registry.ToolGroup`), and when the surface
is over, whole groups are held back: left off ``tools``, named one line each
in a catalogue in the system prompt, and loaded on request through a
discovery tool (the core's is :class:`~protocore.tools.tool_search.ToolSearchTool`).
A loaded tool is APPENDED to the surface in the order it was discovered, so
what came before it — the name-sorted base surface and every earlier load —
keeps its bytes, and a prefix-caching provider re-reads only the tail.

The decision is made once per run and remade only when the catalogue itself
changes. Deciding per request would move the catalogue in the middle of the
cached prefix every time the answer flipped, which is worse than either answer.
Held-back tools stay admitted by the visibility policy: a model that calls one
by its exact name is served, and the tool is loaded for the rest of the run.

Nothing here runs when nothing is over. No group declared, no discovery tool
registered, or a surface inside both limits: the surface and the system prompt
are exactly what they would be without this module.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import (
    TOOLS_LOADED_METADATA_KEY,
    ToolGroup,
    ToolVisibilityPolicy,
    policy_admits,
    tool_group_of,
)
from protocore.contracts.tool_retrieval import RetrievalSettings
from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
from protocore.contracts.tools import Tool
from protocore.contracts.types import ToolDefinition
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.token_counting import estimate_tokens

if TYPE_CHECKING:
    from protocore.runtime.query_engine import QueryEngine
    from protocore.runtime.tool_dispatch import DispatchOutcome

__all__ = [
    "NO_DEFERRAL",
    "ToolDeferral",
    "build_tool_surface",
    "discovery_tool_names",
    "ensure_tool_deferral",
    "observe_dispatched_tool",
    "plan_tool_deferral",
    "render_tool_catalogue",
    "tool_catalogue_block",
]

#: The sentence under the catalogue. A model that cannot see a tool reaches for
#: the nearest one it can — a service started with a shell command, two edits
#: instead of one multi-edit — and that, rather than a failed search, is how a
#: held-back tool goes unused. Saying so once is what moved it.
_CATALOGUE_ADVICE: Final[str] = (
    "A dedicated tool for the job is better than a workaround with a general "
    "one such as a shell command, so load the tool rather than improvising."
)


@dataclass(frozen=True)
class ToolDeferral:
    """What one run holds back, and the catalogue that names it."""

    deferred_groups: tuple[str, ...] = ()
    """The held-back groups, in the order they were held back."""

    deferred_names: frozenset[str] = frozenset()
    """Every tool left off the surface because its group was held back."""

    catalogue: str = ""
    """The system-prompt block naming the held-back groups; empty for none."""

    reasons: tuple[str, ...] = ()
    """Why anything was held back: ``dynamic``, ``tokens``, ``count``, ``restored``."""


NO_DEFERRAL: Final[ToolDeferral] = ToolDeferral()


def discovery_tool_names(tools: Iterable[Tool], roles: ToolRoleMap) -> frozenset[str]:
    """The registered tools that load other tools: role ``discovers_tools``.

    The role comes from the host's :class:`ToolRoleMap` or from a tool that
    states its own roles in a ``tool_roles`` class attribute, the way the
    core's own tools do, so the core's ToolSearch needs no declaration.
    """
    declared = roles.names_with(ToolRole.discovers_tools)
    names: set[str] = set()
    for tool in tools:
        own = getattr(tool, "tool_roles", ())
        if tool.name in declared or (
            isinstance(own, tuple | frozenset | list) and ToolRole.discovers_tools in own
        ):
            names.add(tool.name)
    return frozenset(names)


def _definition_tokens(definition: ToolDefinition, rc: LoopConstants) -> int:
    return estimate_tokens(definition.model_dump_json(), rc)


def plan_tool_deferral(
    *,
    tools: Sequence[Tool],
    groups: Sequence[ToolGroup],
    protected: frozenset[str],
    discovery_names: frozenset[str],
    rc: LoopConstants,
    restored: Sequence[str] | None = None,
) -> ToolDeferral:
    """Decide which groups ``tools`` — the would-be surface — holds back.

    ``protected`` tools are never held back whatever their group: the forced
    floor, explicitly pinned tools and always-load tools. A tool in no group
    is never held back either; a host that wants a tool deferrable says so by
    grouping it.

    Every dynamic group is held back as soon as deferral is on. The others go
    only while the surface is over — its definitions above
    ``tool_definitions_ratio`` of the window, or its count above
    ``max_advertised_tools`` with room left for the tools the run may load —
    largest first, so the fewest groups leave. ``restored`` replays a
    decision a snapshot carried instead of measuring again, because the
    catalogue it produced is at the head of the resumed run's cached prompt.
    """
    if rc.tool_deferral_mode == "off" or not discovery_names:
        return NO_DEFERRAL
    if not any(tool.name in discovery_names for tool in tools):
        return NO_DEFERRAL
    declared = {group.name: group for group in groups}
    members: dict[str, list[Tool]] = {}
    for tool in tools:
        if tool.name in protected or tool.name in discovery_names:
            continue
        group_name = tool_group_of(tool, groups)
        if group_name:
            members.setdefault(group_name, []).append(tool)
    if not members:
        return NO_DEFERRAL

    if restored is not None:
        chosen = [name for name in restored if name in members]
        return _deferral(chosen, members, declared, rc, ("restored",), discovery_names)

    tokens_of = {tool.name: _definition_tokens(tool.definition, rc) for tool in tools}
    group_tokens = {
        name: sum(tokens_of[tool.name] for tool in grouped)
        for name, grouped in members.items()
    }

    def largest_first(name: str) -> tuple[int, int, str]:
        return (-group_tokens[name], -len(members[name]), name)

    discovery_tokens = sum(tokens_of[name] for name in discovery_names if name in tokens_of)
    discovery_count = sum(1 for tool in tools if tool.name in discovery_names)
    # The surface as it goes out when nothing is held back carries no
    # discovery tool, so that is the one measured against the limits.
    plain_tokens = sum(tokens_of.values()) - discovery_tokens
    plain_count = len(tools) - discovery_count
    # The same budget the context layers are sized with; until this module it
    # was computed and never held to anything.
    budget = derive_budgets(rc).tool_definitions_budget_tokens
    limit = rc.max_advertised_tools
    over_tokens = plain_tokens > budget
    over_count = limit > 0 and plain_count > limit

    chosen = sorted((name for name in members if _is_dynamic(declared, name)), key=largest_first)
    reasons: list[str] = ["dynamic"] if chosen else []
    if over_tokens or over_count:
        tokens = plain_tokens + discovery_tokens - sum(group_tokens[name] for name in chosen)
        count = plain_count + discovery_count - sum(len(members[name]) for name in chosen)
        # A held-back tool the run loads comes back onto the surface, so the
        # count leaves room for as many as the run may keep loaded.
        headroom = rc.pinned_tool_max_count
        for name in sorted((n for n in members if n not in chosen), key=largest_first):
            fits_tokens = tokens <= budget
            fits_count = limit <= 0 or count + headroom <= limit
            if fits_tokens and fits_count:
                break
            chosen.append(name)
            tokens -= group_tokens[name]
            count -= len(members[name])
        if over_tokens:
            reasons.append("tokens")
        if over_count:
            reasons.append("count")
    if not chosen:
        return NO_DEFERRAL
    return _deferral(chosen, members, declared, rc, tuple(reasons), discovery_names)


def _is_dynamic(declared: dict[str, ToolGroup], name: str) -> bool:
    group = declared.get(name)
    return group is not None and group.dynamic


def _deferral(
    chosen: Sequence[str],
    members: dict[str, list[Tool]],
    declared: dict[str, ToolGroup],
    rc: LoopConstants,
    reasons: tuple[str, ...],
    discovery_names: frozenset[str],
) -> ToolDeferral:
    if not chosen:
        return NO_DEFERRAL
    deferred = {
        name: sorted(tool.name for tool in members[name]) for name in chosen
    }
    return ToolDeferral(
        deferred_groups=tuple(chosen),
        deferred_names=frozenset(n for names in deferred.values() for n in names),
        catalogue=render_tool_catalogue(
            deferred,
            declared,
            discovery_tool=min(discovery_names),
            max_listed_names=rc.tool_catalogue_max_listed_names,
        ),
        reasons=reasons,
    )


def render_tool_catalogue(
    deferred: dict[str, list[str]],
    declared: dict[str, ToolGroup],
    *,
    discovery_tool: str,
    max_listed_names: int,
) -> str:
    """The system-prompt block naming each held-back group, one line apiece.

    Groups in name order and names in name order, so the same decision renders
    the same bytes on every request and after every resume. Each line gives
    the EXACT names — a model that guesses a name gets the case wrong — or, for
    a group declared by prefix with more tools than ``max_listed_names``, the
    exact prefix and the count.
    """
    if not deferred:
        return ""
    lines: list[str] = []
    for name in sorted(deferred):
        names = deferred[name]
        group = declared.get(name)
        description = group.description.strip() if group is not None else ""
        prefix = group.prefix if group is not None else ""
        if prefix and len(names) > max_listed_names and all(n.startswith(prefix) for n in names):
            listed = f"{prefix}* ({len(names)} tools)"
        else:
            listed = ", ".join(names)
        summary = description.rstrip(". ")
        if summary:
            lines.append(f"- {name}: {summary}. Tools: {listed}")
        else:
            lines.append(f"- {name}: tools {listed}")
    header = (
        "These tools are available but not loaded. Before calling one, load it "
        f'with {discovery_tool}: describe what you need, or pass "select:" and '
        f'exact names, e.g. "select:Name1,Name2". {_CATALOGUE_ADVICE}'
    )
    body = "\n".join(lines)
    return f"<system-reminder>\n{header}\n\n{body}\n</system-reminder>"


# ---------------------------------------------------------------------------
# The engine side
# ---------------------------------------------------------------------------


def _catalogue_key(engine: QueryEngine) -> tuple[Any, ...]:
    """What the decision depends on that can change under a running process.

    The host's policy is part of it. A host that switches a server's tools on
    mid-run replaces the policy and registers nothing, when the tools were
    already registered for another session; keyed on the catalogue alone, the
    decision went stale and those tools reached the surface whole, however
    many there were. The pins the run adds for the tools it loaded are left
    out: loading a tool must not reopen the decision.
    """
    policy = engine.config.tool_visibility_policy
    return (
        tuple(tool.name for tool in engine.tools.list_all()),
        tuple(engine.tools.tool_groups()),
        frozenset(policy.visible),
        frozenset(policy.blocked),
        frozenset(policy.pinned),
        policy.forced_pinned,
    )


def _unpinned_policy(engine: QueryEngine, policy: ToolVisibilityPolicy) -> ToolVisibilityPolicy:
    """``policy`` without the pins that stand for discovered tools.

    The engine folds the discovered tools into ``pinned`` so dispatch admits
    them; the surface places them itself, after the base, so they must not
    also be sorted into it.
    """
    loaded = set(engine.context_manager.discovered_tool_names())
    loaded -= set(engine.config.tool_visibility_policy.pinned)
    if not loaded & set(policy.pinned):
        return policy
    return policy.model_copy(update={"pinned": set(policy.pinned) - loaded})


def ensure_tool_deferral(
    engine: QueryEngine, policy: ToolVisibilityPolicy | None = None
) -> ToolDeferral:
    """The run's deferral decision, made on first use and when the catalogue changes."""
    key = _catalogue_key(engine)
    current = engine._tool_deferral
    if current is not None and engine._tool_deferral_key == key:
        return current
    if policy is None:
        policy = engine.effective_tool_policy
    base = _unpinned_policy(engine, policy)
    registry = engine.tools
    candidates: list[Tool] = []
    for definition in registry.compute_effective_surface(
        tenant_id=engine.config.tenant_id, policy=base, top_k=None
    ):
        tool = registry.get(definition.name)
        if tool is not None:
            candidates.append(tool)
    protected = (
        frozenset(policy.forced_pinned)
        | frozenset(engine.config.tool_visibility_policy.pinned)
        | frozenset(t.name for t in candidates if bool(getattr(t, "always_load", False)))
    )
    restored = engine._restored_deferred_groups
    engine._restored_deferred_groups = None
    decision = plan_tool_deferral(
        tools=candidates,
        groups=registry.tool_groups(),
        protected=protected,
        discovery_names=discovery_tool_names(candidates, engine.config.tool_roles),
        rc=engine.config.rc,
        restored=restored,
    )
    engine._tool_deferral = decision
    engine._tool_deferral_key = key
    return decision


def tool_catalogue_block(engine: QueryEngine) -> str:
    """The run's catalogue of held-back tools, or ``""``."""
    decision = engine._tool_deferral
    return decision.catalogue if decision is not None else ""


def build_tool_surface(engine: QueryEngine) -> list[ToolDefinition]:
    """The tool definitions the next request advertises.

    The base surface is the registry's, in name order, minus held-back groups
    and — while nothing is held back — minus the discovery tool, which has
    nothing to find then. Discovered tools follow in discovery order. The
    order is the cache contract: the base does not move when a tool is loaded,
    and a loaded tool does not move when another one is.
    """
    policy = engine.effective_tool_policy
    decision = ensure_tool_deferral(engine, policy)
    registry = engine.tools
    rc = engine.config.rc
    discovery = discovery_tool_names(registry.list_all(), engine.config.tool_roles)
    hidden = set(decision.deferred_names)
    if not decision.deferred_groups:
        explicit = set(policy.forced_pinned) | set(engine.config.tool_visibility_policy.pinned)
        hidden |= discovery - explicit
    surface_policy = _unpinned_policy(engine, policy)
    if hidden - set(surface_policy.blocked):
        surface_policy = surface_policy.model_copy(
            update={"blocked": set(surface_policy.blocked) | hidden}
        )
    base = list(
        registry.compute_effective_surface(
            tenant_id=engine.config.tenant_id,
            policy=surface_policy,
            query=engine.latest_user_message.text if engine.latest_user_message else "",
            top_k=rc.tool_retrieval_top_k or None,
            retrieval=RetrievalSettings.from_constants(rc),
        )
    )
    present = {definition.name for definition in base}
    appended: list[ToolDefinition] = []
    for name in engine.context_manager.discovered_tool_names():
        # ``pinned`` is where the engine admits a discovered tool, and where a
        # wind-down or an execution profile withdraws it again.
        if name in present or name not in policy.pinned or name in policy.blocked:
            continue
        tool = registry.get(name)
        if tool is None:
            continue
        appended.append(tool.definition)
        present.add(name)
    limit = rc.max_advertised_tools
    if limit > 0 and len(base) + len(appended) > limit:
        appended = _fit_loaded(engine, appended, max(0, limit - len(base)))
    return base + appended


def _fit_loaded(
    engine: QueryEngine, appended: list[ToolDefinition], room: int
) -> list[ToolDefinition]:
    """Keep the ``room`` most recently used loaded tools, in discovery order.

    Only reached when a provider's tool limit would otherwise refuse the
    request. Dropping from the middle of the loaded tail costs the cache
    everything after it, which is still better than a request that fails.
    """
    recency = engine.context_manager.discovered_tool_last_used()
    keep = {
        definition.name
        for definition in sorted(appended, key=lambda d: recency.get(d.name, 0), reverse=True)[:room]
    }
    return [definition for definition in appended if definition.name in keep]


def observe_dispatched_tool(
    engine: QueryEngine,
    tool_name: str,
    tool_call_id: str,
    outcome: DispatchOutcome,
) -> list[TurnEvent]:
    """Fold one dispatched call into the run's loaded tools; return what to announce.

    Two things load a tool. A discovery tool's result names what it loaded,
    under :data:`TOOLS_LOADED_METADATA_KEY`; only a tool in the discovery role
    is believed, and only for names the live policy admits. And a call of a
    registered tool the request did not advertise — a held-back tool called by
    its exact name, or one the per-message clip left out — has already been
    served, since dispatch checks the policy and not the surface; loading it
    puts its schema in front of the model for the next call rather than
    leaving the model to keep calling a tool it cannot see.
    """
    manager = engine.context_manager
    registry = engine.tools
    tool = registry.get(tool_name)
    if tool is None:
        return []
    manager.note_tool_used(tool_name)
    policy = engine.effective_tool_policy
    allowlist = engine.effective_subagent_tool_allowlist

    def admitted(name: str) -> bool:
        # The same two stages dispatch applies: the policy, and a child's
        # declared tool set. Loading a tool the next call would be refused
        # hands the model a schema it cannot use.
        return policy_admits(policy, name) and (allowlist is None or name in allowlist)

    events: list[TurnEvent] = []
    discovery = discovery_tool_names([tool], engine.config.tool_roles)
    if tool_name in discovery:
        raw = (outcome.metadata or {}).get(TOOLS_LOADED_METADATA_KEY)
        if outcome.is_error or not isinstance(raw, list):
            return events
        advertised_now = engine._advertised_tool_names or frozenset()
        loaded: list[str] = []
        newly: list[str] = []
        for name in raw:
            if not isinstance(name, str) or name in loaded:
                continue
            if registry.get(name) is None or not admitted(name):
                continue
            loaded.append(name)
            # A match already on the surface is callable as it is; loading it
            # again would only spend a place under the cap.
            if name in advertised_now and name not in manager.discovered_tool_last_used():
                continue
            if manager.discover_tool(name):
                newly.append(name)
        if loaded:
            events.append(
                TurnEvent(
                    type=EventType.TOOL_DISCOVERED,
                    run_id=engine.config.run_id,
                    payload={
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "loaded_tool_names": loaded,
                        "newly_loaded_tool_names": newly,
                        "discovered_tool_names": list(manager.discovered_tool_names()),
                    },
                )
            )
        return events
    advertised = engine._advertised_tool_names
    if advertised is None or tool_name in advertised:
        return events
    if outcome.error_kind == "permission":
        return events
    if not admitted(tool_name):
        return events
    newly_loaded = manager.discover_tool(tool_name)
    events.append(
        TurnEvent(
            type=EventType.TOOL_UNADVERTISED_CALL,
            run_id=engine.config.run_id,
            payload={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "success": not outcome.is_error,
                "deferred": tool_name in (
                    engine._tool_deferral.deferred_names
                    if engine._tool_deferral is not None
                    else frozenset()
                ),
                "loaded": newly_loaded,
                "discovered_tool_names": list(manager.discovered_tool_names()),
            },
        )
    )
    return events
