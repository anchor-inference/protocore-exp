"""IToolRegistry Protocol — tenant-aware tool surface.

Core implements the concrete 3-layer filter (policy + clipping + progressive
discovery) in :mod:`protocore.runtime.tool_registry`; this Protocol exists
for tests and future plugins that may want to substitute the registry wholesale.

The registry also holds the catalogue's GROUPS: named sets of tools the loop may
leave off the advertised surface as a unit when the whole catalogue no longer
fits, and name in the system prompt instead (see
:mod:`protocore.runtime.tool_deferral`).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.tool_retrieval import RetrievalSettings
from protocore.contracts.tools import Tool
from protocore.contracts.types import ToolDefinition

#: ``ToolContext.metadata`` key under which the dispatcher stamps the live
#: per-run :class:`ToolVisibilityPolicy` so policy-aware tools (ToolSearch)
#: can honour the same visible/blocked contract the dispatch gate enforces, and
#: so a blocked tool's schema never leaks through a search. The value is the
#: policy MODEL instance — a live object, never a serialised copy. Namespaced
#: and assigned on every dispatch: under a bare name set only when absent, a
#: host key of the same spelling, or a stale policy carried in the bag, won
#: over the one the gate was about to enforce.
TOOL_VISIBILITY_POLICY_METADATA_KEY: Final[str] = "protocore.tool_visibility_policy"

#: ``ToolContext.metadata`` key under which the dispatcher stamps a child run's
#: declared tool set, as a frozenset, beside the policy. The gate refuses a
#: name outside it, so a search must neither load nor report one. Absent when
#: the run declared none.
TOOL_ALLOWLIST_METADATA_KEY: Final[str] = "protocore.tool_allowlist"

#: ``ToolResult.metadata`` key under which a discovery tool (role
#: ``discovers_tools``) lists the tool names it loaded. The loop reads it from
#: discovery tools only and keeps only names the live policy admits, so a tool
#: cannot widen the surface by writing this key into its result.
TOOLS_LOADED_METADATA_KEY: Final[str] = "protocore.tools_loaded"

#: ``ToolContext.metadata`` key under which the loop stamps the names of the
#: tools the current request advertised, as a frozenset. A tool the model called
#: from outside that list was called blind — its schema never shown — and
#: ``ToolSearch`` must not report a tool the model can already see as newly
#: loaded: a model told "Loaded: WebSearch" of a tool it had all along reads it
#: as having loaded something else it asked for, such as a skill. Absent when
#: the tool runs outside a loop, which then reads as "nothing is known to be
#: advertised".
ADVERTISED_TOOLS_METADATA_KEY: Final[str] = "protocore.advertised_tools"

#: ``ToolContext.metadata`` key under which the loop stamps the names of the
#: tool groups whose rules (:attr:`ToolGroup.instructions`) the run has already
#: been given, as a frozenset. A discovery tool gives a group's rules with the
#: first tools of it that it loads, and gives them once: repeated on every
#: load they are only tokens. Absent outside a loop, which reads as "none
#: given yet".
TOOL_GROUP_RULES_GIVEN_METADATA_KEY: Final[str] = "protocore.tool_group_rules_given"

#: ``ToolResult.metadata`` key under which a discovery tool names the groups it
#: loaded whole (``ToolSearch(group=...)``). The loop keeps the tools of such a
#: group as one entry under the loaded-tool cap, so it is never left holding
#: half a group the model asked for as a unit.
TOOL_GROUPS_LOADED_METADATA_KEY: Final[str] = "protocore.tool_groups_loaded"

#: ``ToolResult.metadata`` key under which a discovery tool names the groups
#: whose rules its result carries. The loop counts those rules as given and
#: does not hold back a later call of the group's tools to give them again.
TOOL_GROUP_RULES_METADATA_KEY: Final[str] = "protocore.tool_group_rules"

#: How a group is advertised. ``eager`` is never held back for size (only a
#: provider's hard limit on the number of tools can still push it off);
#: ``auto`` is held back when the surface is over its limits; ``lazy`` is held
#: back whenever the run has a discovery tool, and named in the catalogue.
ToolGroupLoad = Literal["eager", "auto", "lazy"]

TOOL_GROUP_LOADS: Final[tuple[str, ...]] = ("eager", "auto", "lazy")


class ToolVisibilityPolicy(BaseModel):
    """Per-tenant tool-visibility policy.

    Built by the host from PG ``tenant_tool_policy`` table; passed into
    :meth:`IToolRegistry.compute_effective_surface`.
    """

    model_config = ConfigDict(frozen=True)

    visible: set[str] = Field(default_factory=set)
    """Whitelist — empty = all visible."""

    blocked: set[str] = Field(default_factory=set)
    """Explicit deny list (overrides visible)."""

    pinned: set[str] = Field(default_factory=set)
    """Always-include tools (always in surface, even with clipping)."""

    forced_pinned: frozenset[str] = Field(default_factory=frozenset)
    """Core tool-surface floor.

    Tools that bypass the BM25 clip unconditionally — the per-turn surface
    ALWAYS carries them, regardless of query language or score. Built by
    the host from :attr:`LoopConstants.tool_surface_forced_pins`
    (default: ``Agent`` plus the six core file tools
    ``Read/Write/Edit/Bash/Glob/Grep``).

    Distinct from :attr:`pinned`: ``pinned`` is the ToolSearch/progressive-
    discovery pin set (per-session, mutable); ``forced_pinned`` is the
    tenant-policy floor that prevents catastrophic RU-prompt zero-score
    collapse. ``compute_effective_surface`` unions the two before clipping.
    """


def policy_admits(policy: ToolVisibilityPolicy | None, name: str) -> bool:
    """Dispatchability predicate — the single visible/blocked contract.

    Mirrors the dispatch permission gate's stage-1 whitelist semantics
    (``protocore.runtime.tool_permission``): ``blocked`` always denies; under a
    non-empty ``visible`` whitelist the allowed set is
    ``visible | pinned | forced_pinned`` (pinned/forced tools are advertised
    unconditionally, so they must be admitted here too). ``None`` policy =
    no restriction. Used by :meth:`IToolRegistry.search` and the ToolSearch
    ``select:`` path so progressive discovery can never return a schema the
    dispatch gate would refuse (tools-initiative A2 info-leak fix).
    """
    if policy is None:
        return True
    if policy.blocked and name in policy.blocked:
        return False
    if policy.visible:
        return (
            name in policy.visible
            or name in policy.pinned
            or name in policy.forced_pinned
        )
    return True


class ToolGroup(BaseModel):
    """A named set of tools that is advertised, or held back, as one unit.

    A tool joins a group in one of two ways: its class names it with the
    optional ``tool_group`` attribute (read with ``getattr``, like
    ``search_hint``), or its name starts with the group's :attr:`prefix`. The
    first is for tools a host writes; the second is for tools it does not —
    the proxies an MCP client registers for a server arrive by the dozen under
    one name prefix, and nobody sets a class attribute on them.

    Membership never reaches the wire: it is not part of a tool's definition,
    so it cannot change the advertised surface or its digest.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    """One line the system prompt shows for the group when it is held back."""

    dynamic: bool = False
    """The group's membership is not the host's own code — an MCP server's tools.

    Such a group is the part of a catalogue that grows without anyone deciding
    it should, and its tools are the likeliest to crowd a search result with
    near-matches, so it is the first thing held back and is held back whenever
    deferral is on at all.
    """

    prefix: str = ""
    """Tools whose name starts with this join the group; empty joins none this way."""

    load: ToolGroupLoad = "auto"
    """When the group is advertised; see :data:`ToolGroupLoad`.

    ``lazy`` is for families a run rarely needs: a dozen browser tools used in
    one session of a hundred cost their definitions on every request of the
    other ninety-nine. Held back, they cost one line of the catalogue, and a
    model that needs them loads the group with one call. A run with no
    discovery tool treats ``lazy`` as ``auto``: a blind call by exact name
    would then be the only way in, which costs more than the definitions.
    """

    instructions: str = ""
    """Rules for using the group's tools, given once, when they are first loaded.

    Written into the prompt beside the tools only when they are in front of
    the model: a rule about tools the run never loads is tokens spent on
    nothing, and a rule next to a tool list the model cannot see yet is
    easily taken for a rule about something else.
    """


def make_tool_group(
    name: str,
    description: str,
    *,
    dynamic: bool = False,
    prefix: str | None = None,
    load: str = "auto",
    instructions: str = "",
) -> ToolGroup:
    """The :class:`ToolGroup` a ``declare_group`` call describes, checked.

    One place for every registry to build it, so an unknown load mode is
    refused with the same message whichever registry a host wired.
    """
    if not name:
        raise ValueError("a tool group needs a name")
    if load not in TOOL_GROUP_LOADS:
        raise ValueError(
            f"tool group {name!r}: load must be one of {TOOL_GROUP_LOADS!r}, got {load!r}"
        )
    return ToolGroup(
        name=name,
        description=description,
        dynamic=dynamic,
        prefix=prefix or "",
        load=cast(ToolGroupLoad, load),
        instructions=instructions.strip(),
    )


def group_rules_text(group: str, instructions: str) -> str:
    """The block a group's rules are given in, wherever they are given.

    One spelling for the catalogue, a search result and a held call, so a
    model that has read it once recognises it everywhere.
    """
    return f"Rules for the {group} tools:\n{instructions.strip()}"


def tool_group_of(tool: Tool, groups: Sequence[ToolGroup]) -> str:
    """The name of the group ``tool`` belongs to, or ``""`` for none.

    An explicit ``tool_group`` attribute wins over a prefix: a host that says
    where a tool belongs has settled the question. Among prefixes the longest
    match wins, so ``Mcp_Github_`` and ``Mcp_`` can both be declared and a
    GitHub proxy lands in the narrower one.
    """
    explicit = getattr(tool, "tool_group", "")
    if isinstance(explicit, str) and explicit:
        return explicit
    best = ""
    best_length = 0
    for group in groups:
        if group.prefix and tool.name.startswith(group.prefix) and len(group.prefix) > best_length:
            best = group.name
            best_length = len(group.prefix)
    return best


@runtime_checkable
class IToolRegistry(Protocol):
    """Tenant-aware tool registry."""

    def register(self, tool: Tool) -> None:
        """Register a tool. Idempotent on ``tool.name``."""
        ...

    def unregister(self, name: str) -> None:
        """Remove a tool by name. Idempotent — no error if absent."""
        ...

    def get(self, name: str) -> Tool | None:
        """Fetch tool by name; ``None`` if not registered."""
        ...

    def list_all(self) -> Sequence[Tool]:
        """All registered tools, sorted by :attr:`Tool.name` ASC.

        The sort invariant matters for KV-prefix cache stability across turns.
        """
        ...

    def list_for_tenant(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
    ) -> Sequence[Tool]:
        """List tools visible to a tenant (after policy filter)."""
        ...

    def filter_by_whitelist(self, names: Sequence[str]) -> Sequence[Tool]:
        """Resolve a flat list of names to :class:`Tool` instances.

        Used by subagent dispatch before building the per-turn surface:
        the subagent's whitelist is a flat name list — this method
        resolves it against the catalogue, dropping unknown names
        silently. Result sorted by name ASC (sort invariant).
        """
        ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        tenant_id: str = "",
        whitelist: Sequence[str] | None = None,
        policy: ToolVisibilityPolicy | None = None,
        retrieval: RetrievalSettings | None = None,
    ) -> Sequence[Tool]:
        """Ranked search across the policy-filtered subset, best match first.

        ``whitelist`` (if provided) narrows the candidate pool — used by
        the :class:`ToolSearch` tool to restrict matches to the
        subagent's allowed surface. ``policy`` (if provided) applies the
        per-run visibility contract via :func:`policy_admits` so blocked
        tools never leak through discovery. ``retrieval`` carries the run's
        ranking settings; ``None`` means the defaults. Empty query returns
        the first ``top_k`` candidates by name order (deterministic).
        """
        ...

    def compute_effective_surface(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
        *,
        query: str = "",
        top_k: int | None = None,
        retrieval: RetrievalSettings | None = None,
    ) -> Sequence[ToolDefinition]:
        """3-layer filter: policy → clipping → progressive discovery.

        ``query`` is the recent user message (for retrieval); ``top_k`` is
        how many tools the clip retrieves BESIDES the pinned ones (pinned,
        forced and always-load tools never count against it), ``None``
        meaning no clip. ``retrieval`` carries the run's ranking settings;
        ``None`` means the defaults.
        """
        ...

    def declare_group(
        self,
        name: str,
        description: str,
        *,
        dynamic: bool = False,
        prefix: str | None = None,
        load: str = "auto",
        instructions: str = "",
    ) -> None:
        """Declare (or redeclare) a tool group. See :class:`ToolGroup`.

        ``load`` is one of :data:`TOOL_GROUP_LOADS`, and anything else is
        refused. Redeclaring replaces the description, the load mode and the
        instructions, and a running loop decides again what to hold back.
        """
        ...

    def undeclare_group(self, name: str) -> None:
        """Forget a group. Idempotent — no error if it was never declared.

        For a group whose tools are gone for good, such as the proxies of an
        MCP server the operator removed: a declaration outlives its tools, and
        one left behind would still claim their prefix if a server of the same
        name came back with another description.
        """
        ...

    def tool_groups(self) -> Sequence[ToolGroup]:
        """Every declared group, sorted by name."""
        ...


__all__ = [
    "ADVERTISED_TOOLS_METADATA_KEY",
    "TOOLS_LOADED_METADATA_KEY",
    "TOOL_ALLOWLIST_METADATA_KEY",
    "TOOL_GROUPS_LOADED_METADATA_KEY",
    "TOOL_GROUP_LOADS",
    "TOOL_GROUP_RULES_GIVEN_METADATA_KEY",
    "TOOL_GROUP_RULES_METADATA_KEY",
    "TOOL_VISIBILITY_POLICY_METADATA_KEY",
    "IToolRegistry",
    "ToolGroup",
    "ToolGroupLoad",
    "ToolVisibilityPolicy",
    "group_rules_text",
    "make_tool_group",
    "policy_admits",
    "tool_group_of",
]
