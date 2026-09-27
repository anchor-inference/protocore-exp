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
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.tool_retrieval import RetrievalSettings
from protocore.contracts.tools import Tool
from protocore.contracts.types import ToolDefinition

#: ``ToolContext.metadata`` key under which the dispatcher injects the live
#: per-run :class:`ToolVisibilityPolicy` so policy-aware tools (ToolSearch)
#: can honour the same visible/blocked contract the dispatch gate enforces
#: (tools-initiative A2 — closes the blocked-schema info leak). The value is
#: the policy MODEL instance — a live object, never a serialised copy.
TOOL_VISIBILITY_POLICY_METADATA_KEY: Final[str] = "tool_visibility_policy"

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
        prefix: str = "",
    ) -> None:
        """Declare (or redeclare) a tool group. See :class:`ToolGroup`."""
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
    "TOOL_VISIBILITY_POLICY_METADATA_KEY",
    "IToolRegistry",
    "ToolGroup",
    "ToolVisibilityPolicy",
    "policy_admits",
    "tool_group_of",
]
