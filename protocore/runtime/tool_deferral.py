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

The decision is made once per run. When the catalogue or the policy changes
it is made again on top of itself: what is held back stays held back, and only
a newly admitted dynamic group or a limit now exceeded adds to it. Deciding
afresh would move the catalogue in the middle of the cached prefix every time
the answer flipped, which is worse than either answer.
Held-back tools stay admitted by the visibility policy: a model that calls one
by its exact name is served, and the tool is loaded for the rest of the run.

A group may also be held back by choice. Its load mode — the declaration's,
or the run's own override — is ``auto`` for the rule above, ``lazy`` for a
family the run rarely needs, held back whenever there is a discovery tool to
load it, and ``eager`` for one that stays on the surface whatever the size,
giving way only to a provider's hard limit on the number of tools. And a group
may carry rules for its tools, given once per run where the tools first come
in front of the model: in the catalogue for tools there from the start, in the
result that loads them, or instead of running a call made to one blind.

Nothing here runs when nothing is over. No group declared, no discovery tool
registered, or a surface inside both limits, with no group lazy and none
carrying rules: the surface and the system prompt are exactly what they would
be without this module.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import (
    TOOL_GROUP_LOADS,
    TOOL_GROUP_RULES_METADATA_KEY,
    TOOL_GROUPS_LOADED_METADATA_KEY,
    TOOLS_LOADED_METADATA_KEY,
    ToolGroup,
    ToolVisibilityPolicy,
    group_rules_text,
    policy_admits,
    tool_group_of,
)
from protocore.contracts.tool_retrieval import RetrievalSettings
from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
from protocore.contracts.tools import Tool
from protocore.contracts.types import ToolCall, ToolDefinition
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.token_counting import estimate_tokens

if TYPE_CHECKING:
    from protocore.runtime.query_engine import QueryEngine
    from protocore.runtime.tool_dispatch import DispatchOutcome

#: How the tools of a group came to be loaded, as ``tool_group_loaded`` says.
#: ``search`` and ``select`` are a discovery tool's two forms, ``group`` its
#: whole-group form, ``direct_call`` a call of a tool that was not advertised,
#: and ``seed`` the tools a host started the run with.
GROUP_LOAD_VIAS: Final[tuple[str, ...]] = ("search", "select", "group", "direct_call", "seed")

__all__ = [
    "GROUP_LOAD_VIAS",
    "NO_DEFERRAL",
    "ToolDeferral",
    "build_tool_surface",
    "calls_held_for_rules",
    "discovery_tool_names",
    "ensure_tool_deferral",
    "hold_call_for_rules",
    "note_prompt_prefix_restarted",
    "observe_dispatched_tool",
    "plan_tool_deferral",
    "render_tool_catalogue",
    "seed_group_events",
    "tool_catalogue_block",
    "tool_group_states",
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
    """Why anything was held back: ``lazy``, ``dynamic``, ``tokens``, ``count``, ``restored``."""

    group_loads: tuple[tuple[str, str], ...] = ()
    """Every group with a tool on the would-be surface and its load mode, by name."""

    ruled_groups: tuple[str, ...] = ()
    """The groups whose rules the catalogue carries, by name."""


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


def _load_of(
    name: str, declared: Mapping[str, ToolGroup], loads: Mapping[str, str]
) -> str:
    """The load mode ``name`` runs with: the run's override, else the declaration.

    A group only a tool's ``tool_group`` attribute names was never declared,
    and is ``auto`` — what it was before load modes existed.
    """
    override = loads.get(name)
    if override in TOOL_GROUP_LOADS:
        return override
    group = declared.get(name)
    return group.load if group is not None else "auto"


def plan_tool_deferral(
    *,
    tools: Sequence[Tool],
    groups: Sequence[ToolGroup],
    protected: frozenset[str],
    discovery_names: frozenset[str],
    rc: LoopConstants,
    restored: Sequence[str] | None = None,
    restored_reasons: Sequence[str] = ("restored",),
    loads: Mapping[str, str] | None = None,
    loaded: Iterable[str] = (),
    ruled: Iterable[str] = (),
) -> ToolDeferral:
    """Decide which groups ``tools`` — the would-be surface — holds back.

    ``protected`` tools are never held back whatever their group: the forced
    floor, explicitly pinned tools and always-load tools. A tool in no group
    is never held back either; a host that wants a tool deferrable says so by
    grouping it.

    Each group's load mode — ``loads`` for this run, else its declaration —
    comes first. A ``lazy`` group is held back whenever a discovery tool is
    on the surface, whatever the size; an ``eager`` one is never held back
    for size. The rest is the ``auto`` rule: every dynamic group is held back
    as soon as deferral is on, and the others only while the surface is over
    — its definitions above ``tool_definitions_ratio`` of the window, or its
    count above ``max_advertised_tools`` with room left for the tools the run
    may load — largest first, so the fewest groups leave. The provider's count
    limit is the one thing an ``eager`` group yields to, and only while the
    count is over the limit itself: a request over it is refused, which no
    load mode is worth.

    ``restored`` is an earlier decision to keep: one a snapshot carried, or the
    one this run already made. Its groups stay held back while they have tools,
    even where a fresh measurement would let them back, because the catalogue
    they produced is at the head of the cached prompt. It is a floor and not
    the answer: a dynamic group it does not name is still held back, and the
    limits are still enforced on top of it. Replayed as the whole answer, a
    snapshot taken before a large server connected put that server on the
    surface whole, over a provider's limit on the number of tools. A group
    now ``eager`` leaves the floor: the operator asked for it by name.

    Without a discovery tool nothing can load a held-back tool but a call by
    its exact name, so groups — ``lazy`` ones too — are held back only when
    the provider would otherwise refuse the request (the count limit); the
    catalogue then says to call by exact name.

    The rules of a group (:attr:`ToolGroup.instructions`) are written into the
    catalogue when its tools are in front of the model from the start: a
    group left on the surface, a group of one of the ``loaded`` tools, and
    every ``ruled`` group, which is the floor the catalogue already carries.
    """
    declared = {group.name: group for group in groups}
    overrides: Mapping[str, str] = loads or {}
    chosen, reasons = _choose_deferred(
        tools=tools,
        groups=groups,
        declared=declared,
        overrides=overrides,
        protected=protected,
        discovery_names=discovery_names,
        rc=rc,
        restored=restored,
        restored_reasons=restored_reasons,
    )
    group_of = {tool.name: tool_group_of(tool, groups) for tool in tools}
    deferred: dict[str, list[str]] = {name: [] for name in chosen}
    for tool in tools:
        group_name = group_of[tool.name]
        if (
            group_name in deferred
            and tool.name not in protected
            and tool.name not in discovery_names
        ):
            deferred[group_name].append(tool.name)
    deferred_names = frozenset(name for names in deferred.values() for name in names)
    present = sorted({name for name in group_of.values() if name})
    on_surface = {
        group_of[tool.name]
        for tool in tools
        if tool.name not in deferred_names and group_of[tool.name]
    }
    with_loaded = {group_of.get(name, "") for name in loaded} - {""}
    rules = {
        name: declared[name].instructions
        for name in sorted(set(ruled) | on_surface | with_loaded)
        if name in declared and declared[name].instructions
    }
    if not chosen and not rules:
        return replace(
            NO_DEFERRAL,
            group_loads=tuple((name, _load_of(name, declared, overrides)) for name in present),
        )
    listed_discovery = discovery_names & {tool.name for tool in tools}
    return ToolDeferral(
        deferred_groups=tuple(chosen),
        deferred_names=deferred_names,
        catalogue=render_tool_catalogue(
            {name: sorted(names) for name, names in deferred.items()},
            declared,
            discovery_tool=min(listed_discovery) if listed_discovery else "",
            max_listed_names=rc.tool_catalogue_max_listed_names,
            rules=rules,
        ),
        reasons=tuple(dict.fromkeys(reasons)) if chosen else (),
        group_loads=tuple((name, _load_of(name, declared, overrides)) for name in present),
        ruled_groups=tuple(rules),
    )


def _choose_deferred(
    *,
    tools: Sequence[Tool],
    groups: Sequence[ToolGroup],
    declared: Mapping[str, ToolGroup],
    overrides: Mapping[str, str],
    protected: frozenset[str],
    discovery_names: frozenset[str],
    rc: LoopConstants,
    restored: Sequence[str] | None,
    restored_reasons: Sequence[str],
) -> tuple[list[str], list[str]]:
    """The groups to hold back, in the order they go, and why."""
    if rc.tool_deferral_mode == "off":
        return [], []
    discovering = any(tool.name in discovery_names for tool in tools)
    limit = rc.max_advertised_tools
    if not discovering and limit <= 0:
        return [], []
    members: dict[str, list[Tool]] = {}
    for tool in tools:
        if tool.name in protected or tool.name in discovery_names:
            continue
        group_name = tool_group_of(tool, groups)
        if group_name:
            members.setdefault(group_name, []).append(tool)
    if not members:
        return [], []
    load = {name: _load_of(name, declared, overrides) for name in members}

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
    # Without a discovery tool the token budget is not enforced: a group held
    # back for it could only come back by a blind call, which costs more than
    # the definitions it saves. The count limit is a refusal, so it is.
    over_tokens = discovering and plain_tokens > budget
    over_count = limit > 0 and plain_count > limit
    if not discovering and not over_count:
        return [], []

    def eligible(name: str) -> bool:
        return name not in chosen and load[name] != "eager"

    chosen: list[str] = []
    reasons: list[str] = []
    if restored is not None:
        chosen = list(
            dict.fromkeys(
                name for name in restored if name in members and load[name] != "eager"
            )
        )
        if chosen:
            reasons.extend(restored_reasons)
    if discovering:
        lazy = sorted(
            (name for name in members if eligible(name) and load[name] == "lazy"),
            key=largest_first,
        )
        if lazy:
            chosen.extend(lazy)
            reasons.append("lazy")
    dynamic = sorted(
        (name for name in members if eligible(name) and _is_dynamic(declared, name)),
        key=largest_first,
    )
    if dynamic:
        chosen.extend(dynamic)
        reasons.append("dynamic")
    if over_tokens or over_count:
        tokens = plain_tokens + discovery_tokens - sum(group_tokens[name] for name in chosen)
        count = plain_count + discovery_count - sum(len(members[name]) for name in chosen)
        # A held-back tool the run loads comes back onto the surface, so the
        # count leaves room for as many as the run may keep loaded.
        headroom = rc.pinned_tool_max_count
        for name in sorted((n for n in members if eligible(n)), key=largest_first):
            fits_tokens = not discovering or tokens <= budget
            fits_count = limit <= 0 or count + headroom <= limit
            if fits_tokens and fits_count:
                break
            chosen.append(name)
            tokens -= group_tokens[name]
            count -= len(members[name])
        # An eager group gives way to the provider's limit and to nothing
        # else: not to the token budget, and not to the room kept for loaded
        # tools, which the request drops before it would exceed the limit.
        for name in sorted(
            (n for n in members if n not in chosen and load[n] == "eager"), key=largest_first
        ):
            if limit <= 0 or count <= limit:
                break
            chosen.append(name)
            count -= len(members[name])
        if over_tokens:
            reasons.append("tokens")
        if over_count:
            reasons.append("count")
    return chosen, reasons


def _is_dynamic(declared: Mapping[str, ToolGroup], name: str) -> bool:
    group = declared.get(name)
    return group is not None and group.dynamic


def render_tool_catalogue(
    deferred: dict[str, list[str]],
    declared: Mapping[str, ToolGroup],
    *,
    discovery_tool: str,
    max_listed_names: int,
    rules: Mapping[str, str] | None = None,
) -> str:
    """The system-prompt block naming each held-back group, one line apiece.

    Groups in name order and names in name order, so the same decision renders
    the same bytes on every request and after every resume. Each line gives
    the EXACT names — a model that guesses a name gets the case wrong — or, for
    a group declared by prefix with more tools than ``max_listed_names``, the
    exact prefix and the count. An empty ``discovery_tool`` means the run has
    none, and the header says to call by exact name instead.

    ``rules`` are the rules of the groups whose tools the model has in front
    of it from the start. A held-back group's rules follow its line, indented;
    the rules of a group on the surface follow the list. Only rules and no
    held-back group make a block without the header, which would describe
    nothing.
    """
    rules = rules or {}
    if not deferred and not rules:
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
        if name in rules:
            lines.extend(
                f"  {line}" if line else ""
                for line in group_rules_text(name, rules[name]).splitlines()
            )
    trailing = [
        group_rules_text(name, text) for name, text in sorted(rules.items()) if name not in deferred
    ]
    if not deferred:
        body = "\n\n".join(trailing)
        return f"<system-reminder>\n{body}\n</system-reminder>"
    if discovery_tool:
        header = (
            "These tools are available but not loaded, and can be loaded at any "
            f"time. Before calling one, load it with {discovery_tool}: pass "
            'group="<name>" to load a whole group, or "select:" and exact names '
            'to load particular tools (e.g. "select:Name1,Name2"), or describe '
            f"what you need. {_CATALOGUE_ADVICE}"
        )
    else:
        # No discovery tool: the one way in is a call by exact name, which is
        # served and loads the tool, and a call with the wrong arguments is
        # answered with the tool's parameters.
        header = (
            "These tools are available but not in your tool list, which would be "
            "over the provider's limit. Call one by its exact name and it is "
            "loaded; if the arguments are wrong, the answer gives its parameters. "
            f"{_CATALOGUE_ADVICE}"
        )
    body = "\n".join(lines)
    if trailing:
        body = body + "\n\n" + "\n\n".join(trailing)
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
        # Whole declarations, so a group redeclared with another load mode or
        # other rules is decided again.
        tuple(engine.tools.tool_groups()),
        tuple(sorted(engine.config.tool_group_loads.items())),
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
    """The run's deferral decision, made on first use and kept for the run.

    A change of the catalogue or of the policy makes it again, with the
    decision in force as the floor (see ``restored`` in
    :func:`plan_tool_deferral`): a group held back stays held back while it
    has tools, a dynamic group this run is newly admitted to joins it, and the
    limits are enforced on top. Made from scratch instead, it flipped whenever
    anything in the process moved — another session switching a server on
    grows this session's refusals, a tool switched off shrinks the surface —
    and every flip rewrote the catalogue at the head of the cached prompt, or
    let a group that had been held back onto the surface mid-run.
    """
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
    restored: Sequence[str] | None = engine._restored_deferred_groups
    restored_reasons: tuple[str, ...] = ("restored",)
    engine._restored_deferred_groups = None
    if restored is None and current is not None:
        restored = current.deferred_groups
        restored_reasons = current.reasons
    # The loaded tools' rules go into the catalogue where the run's prompt
    # begins — its first decision, or after a compaction — and not when a
    # change of catalogue makes it again mid-run: the tools loaded since came
    # with their rules in a result, and writing them into the catalogue as
    # well would move the head of the cached prompt for nothing.
    loaded: Sequence[str] = ()
    if current is None or engine._catalogue_takes_loaded_rules:
        loaded = engine.context_manager.discovered_tool_names()
    engine._catalogue_takes_loaded_rules = False
    decision = plan_tool_deferral(
        tools=candidates,
        groups=registry.tool_groups(),
        protected=protected,
        discovery_names=discovery_tool_names(candidates, engine.config.tool_roles),
        rc=engine.config.rc,
        restored=restored,
        restored_reasons=restored_reasons,
        loads=engine.config.tool_group_loads,
        loaded=loaded,
        ruled=current.ruled_groups if current is not None else (),
    )
    engine._tool_deferral = decision
    engine._tool_deferral_key = key
    # Rules in the catalogue are rules given: a call of one of those tools
    # is not held back to give them again.
    engine._tool_group_rules_given.update(decision.ruled_groups)
    return decision


def note_prompt_prefix_restarted(engine: QueryEngine) -> None:
    """Make the catalogue again at the next request, with the loaded tools' rules.

    Called where the cached prefix is gone anyway — after a compaction. The
    rules a load put in a result may have been summarised away with it, and
    the catalogue is the one place that outlives a summary.
    """
    engine._tool_deferral_key = None
    engine._catalogue_takes_loaded_rules = True


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


def _group_of(engine: QueryEngine, name: str) -> str:
    tool = engine.tools.get(name)
    return tool_group_of(tool, engine.tools.tool_groups()) if tool is not None else ""


def _admits(engine: QueryEngine) -> Callable[[str], bool]:
    """The two stages dispatch applies: the policy, and a child's declared tool set.

    Loading a tool the next call would be refused hands the model a schema it
    cannot use, and holding one back to give its rules gives rules about a
    tool it may not call.
    """
    policy = engine.effective_tool_policy
    allowlist = engine.effective_subagent_tool_allowlist

    def admitted(name: str) -> bool:
        return policy_admits(policy, name) and (allowlist is None or name in allowlist)

    return admitted


def _group_loaded_events(
    engine: QueryEngine, loads: Mapping[str, tuple[str, list[str]]]
) -> list[TurnEvent]:
    """One ``tool_group_loaded`` per group, in name order: ``{group: (via, tools)}``."""
    return [
        TurnEvent(
            type=EventType.TOOL_GROUP_LOADED,
            run_id=engine.config.run_id,
            payload={"group": group, "via": via, "tools": list(tools)},
        )
        for group, (via, tools) in sorted(loads.items())
        if tools
    ]


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

    A discovery tool also names the groups it loaded whole
    (:data:`TOOL_GROUPS_LOADED_METADATA_KEY`), whose tools are then one entry
    under the loaded-tool cap, and the groups whose rules its result carries
    (:data:`TOOL_GROUP_RULES_METADATA_KEY`), which count as given.
    """
    manager = engine.context_manager
    registry = engine.tools
    tool = registry.get(tool_name)
    if tool is None:
        return []
    manager.note_tool_used(tool_name)
    admitted = _admits(engine)

    events: list[TurnEvent] = []
    discovery = discovery_tool_names([tool], engine.config.tool_roles)
    if tool_name in discovery:
        metadata = outcome.metadata or {}
        raw = metadata.get(TOOLS_LOADED_METADATA_KEY)
        if outcome.is_error or not isinstance(raw, list):
            return events
        raw_groups = metadata.get(TOOL_GROUPS_LOADED_METADATA_KEY)
        whole = (
            {name for name in raw_groups if isinstance(name, str)}
            if isinstance(raw_groups, list)
            else set()
        )
        via_query = metadata.get("load_via") == "search"
        advertised_now = engine._advertised_tool_names or frozenset()
        loaded: list[str] = []
        newly: list[str] = []
        by_group: dict[str, tuple[str, list[str]]] = {}
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
            group = _group_of(engine, name)
            as_group = group if group and group in whole else ""
            if manager.discover_tool(name, group=as_group):
                newly.append(name)
                if group:
                    via = "group" if as_group else ("search" if via_query else "select")
                    by_group.setdefault(group, (via, []))[1].append(name)
        declared = {group.name: group for group in registry.tool_groups()}
        raw_rules = metadata.get(TOOL_GROUP_RULES_METADATA_KEY)
        if isinstance(raw_rules, list):
            engine._tool_group_rules_given.update(
                name
                for name in raw_rules
                if isinstance(name, str) and name in declared and declared[name].instructions
            )
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
        events.extend(_group_loaded_events(engine, by_group))
        return events
    advertised = engine._advertised_tool_names
    if advertised is None or tool_name in advertised:
        return events
    if outcome.error_kind == "permission":
        return events
    if not admitted(tool_name):
        return events
    newly_loaded = manager.discover_tool(tool_name)
    # Loaded by being called, so it has been called: the use noted above came
    # before it was a discovered tool and did not count.
    manager.note_tool_used(tool_name)
    events.append(
        TurnEvent(
            type=EventType.TOOL_UNADVERTISED_CALL,
            run_id=engine.config.run_id,
            payload={
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "executed": True,
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
    group = _group_of(engine, tool_name)
    if newly_loaded and group:
        events.extend(_group_loaded_events(engine, {group: ("direct_call", [tool_name])}))
    return events


def calls_held_for_rules(engine: QueryEngine, calls: Sequence[ToolCall]) -> dict[str, str]:
    """The calls of one model message to answer with their group's rules instead of running.

    A call of a tool the request did not advertise was written without its
    schema, and — when its group has rules the run has not been given — without
    its rules. Running it anyway acts on a guess about exactly what the rules
    are there to settle (which browser to drive, whether to ask first), so the
    call is answered with the rules and the tool is loaded; the call again,
    one step later, runs. A group without rules keeps the older bargain: the
    call runs and loads the tool. Maps each held call's id to its group.
    """
    advertised = engine._advertised_tool_names
    if advertised is None:
        return {}
    registry = engine.tools
    groups = registry.tool_groups()
    declared = {group.name: group for group in groups}
    given = engine._tool_group_rules_given
    discovery = discovery_tool_names(registry.list_all(), engine.config.tool_roles)
    admitted = _admits(engine)
    held: dict[str, str] = {}
    for call in calls:
        if call.name in advertised or call.name in discovery:
            continue
        tool = registry.get(call.name)
        if tool is None or not admitted(call.name):
            continue
        group = tool_group_of(tool, groups)
        declaration = declared.get(group)
        if declaration is None or not declaration.instructions or group in given:
            continue
        held[call.id] = group
    return held


def hold_call_for_rules(
    engine: QueryEngine, call: ToolCall, group: str
) -> tuple[str, list[TurnEvent]]:
    """Load ``call``'s tool and answer it with its group's rules; return the answer.

    The answer is not an error. Nothing failed: the model is one step from the
    call it wanted, and an error would count against the tool and teach the
    model that the tool is broken.
    """
    manager = engine.context_manager
    declaration = next((g for g in engine.tools.tool_groups() if g.name == group), None)
    first = group not in engine._tool_group_rules_given
    newly_loaded = manager.discover_tool(call.name)
    manager.note_tool_used(call.name)
    if first and declaration is not None and declaration.instructions:
        engine._tool_group_rules_given.add(group)
        content = (
            f"Not run yet: {call.name} was not in your tool list, and the {group} "
            "tools come with rules to read before the first call.\n\n"
            f"{group_rules_text(group, declaration.instructions)}\n\n"
            "The tool is loaded now; call it again."
        )
    elif not first:
        # A second call of the same group in the same message: its rules are
        # in the answer to the first, a few lines up, and once is enough.
        content = (
            f"Not run yet: {call.name} was not in your tool list. The rules for "
            f"the {group} tools are in another result of this step. The tool is "
            "loaded now; call it again."
        )
    else:
        # The group lost its rules between the start of the message and this
        # call (redeclared, or forgotten): there is nothing left to give, and
        # the call was still made blind.
        content = f"Not run yet: {call.name} was not in your tool list. It is loaded now; call it again."
    events = [
        TurnEvent(
            type=EventType.TOOL_UNADVERTISED_CALL,
            run_id=engine.config.run_id,
            payload={
                "tool_call_id": call.id,
                "tool_name": call.name,
                "executed": False,
                "success": False,
                "deferred": engine._tool_deferral is not None
                and call.name in engine._tool_deferral.deferred_names,
                "loaded": newly_loaded,
                "discovered_tool_names": list(manager.discovered_tool_names()),
            },
        )
    ]
    if newly_loaded:
        events.extend(_group_loaded_events(engine, {group: ("direct_call", [call.name])}))
    return content, events


def seed_group_events(engine: QueryEngine) -> list[TurnEvent]:
    """``tool_group_loaded`` for the groups the host's seed loaded; once per run.

    Emitted beside the first advertisement rather than at construction, where
    no stream is being read yet.
    """
    seeded = engine._seeded_tool_names
    engine._seeded_tool_names = ()
    if not seeded:
        return []
    present = set(engine.context_manager.discovered_tool_names())
    by_group: dict[str, tuple[str, list[str]]] = {}
    for name in seeded:
        group = _group_of(engine, name) if name in present else ""
        if group:
            by_group.setdefault(group, ("seed", []))[1].append(name)
    return _group_loaded_events(engine, by_group)


def tool_group_states(
    decision: ToolDeferral | None, advertised: Iterable[str], engine: QueryEngine
) -> list[dict[str, str]]:
    """Each group's load mode and state on one request, by name.

    ``advertised``: on the surface as a group. ``deferred``: held back, none
    of it loaded. ``loaded``: held back, and some of its tools loaded onto
    this request.
    """
    if decision is None:
        return []
    names = set(advertised)
    held = set(decision.deferred_groups)
    loaded_groups = {
        _group_of(engine, name)
        for name in engine.context_manager.discovered_tool_names()
        if name in names
    }
    states: list[dict[str, str]] = []
    for group, load in decision.group_loads:
        if group not in held:
            state = "advertised"
        elif group in loaded_groups:
            state = "loaded"
        else:
            state = "deferred"
        states.append({"name": group, "load": load, "state": state})
    return states
