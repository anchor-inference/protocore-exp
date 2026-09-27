"""Tool ABC + ToolError hierarchy.

Core defines the contract and ships only the tools whose whole surface is the
protocol itself: ``AskUser``, the memory tools and ``ToolSearch`` (see
:mod:`protocore.tools`). The baseline tools a host is expected to provide
(Read, Write, Edit, Grep, Glob, Bash, WebFetch, Skill, Agent, TodoWrite) are
host-side adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.evidence import EvidenceProducerBinding, ToolEvidenceContext
from protocore.contracts.run_state import RunScopedState
from protocore.contracts.types import ToolDefinition, ToolResult

#: ``ToolContext.metadata`` key carrying a delegation child's position in its
#: concurrently-dispatched batch (0-based, LLM-requested order). Set only on the
#: concurrent delegation fan-out; absent on serial/single dispatch. The host
#: forwards it onto the child so same-path parent-ledger declarations resolve in
#: batch order (deterministic) rather than gather-completion order.
SUBAGENT_DISPATCH_ORDER_METADATA_KEY: Final[str] = "protocore.subagent_dispatch_order"

#: ``ToolContext.metadata`` key carrying a stable identity for the whole
#: concurrent fan-out GROUP (shared by every child in one ``asyncio.gather``;
#: distinct across groups and across turns). Paired with
#: :data:`SUBAGENT_DISPATCH_ORDER_METADATA_KEY` so the parent ledger can scope
#: batch-order resolution PER GROUP — a later group's same-path declaration wins
#: over an earlier group's regardless of batch width, matching cross-turn
#: last-writer-wins.
SUBAGENT_DISPATCH_GROUP_METADATA_KEY: Final[str] = "protocore.subagent_dispatch_group"

#: ``ToolContext.metadata`` key carrying the executing child's tree-budget permit
#: handle (a :class:`~protocore.runtime.subagent_budget.SubagentTreePermit`, an
#: in-memory object — not serialized, like the cancel ``asyncio.Event`` on the
#: run state). The parent acquires a tree slot at the dispatch site and stamps
#: the handle here so the host runner can put it on the CHILD's run state as
#: :attr:`~protocore.contracts.run_state.RunScopedState.subagent_tree_permit`;
#: the child engine then release-while-awaits around its OWN nested delegation
#: gather. Absent on serial/single dispatch and on the root leader (which owns no
#: permit) ⇒ the child simply never releases/reacquires a tree slot.
SUBAGENT_TREE_PERMIT_METADATA_KEY: Final[str] = "protocore.subagent_tree_permit"


#: Every ``ToolContext.metadata`` key the core READS. The bag belongs to the
#: host — it carries the run's own envelope and whatever else a host spells —
#: and core reaches into it only for what is named here. :func:`read_metadata`
#: is the only way core reads it, and it refuses a key absent from this set, so
#: the channel between a host and the loop is this list and nothing else.
#:
#: Stated as literals rather than assembled from the constants that name them,
#: because those constants live in the modules that read them and a contract
#: module may not import upward from any of those. The test beside this module
#: pins every such constant against this set, so the two cannot drift.
CORE_TOOL_CONTEXT_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "tool_call_id",
        "memory_default_scope",
        "memory_default_scope_key",
        "memory_allowed_scopes",
        "memory_scope_keys",
        "memory_enabled",
        "memory_write_similarity_threshold",
        "memory_max_records_per_scope",
        "protocore.tool_visibility_policy",
        "protocore.tool_allowlist",
        "protocore.advertised_tools",
    }
)

#: The keys core STAMPS on the bag for a tool or a host to read back. The other
#: direction of the same channel, and listed for the same reason: a value core
#: writes here is one a host may branch on, so it is part of the contract even
#: though core never reads it again.
CORE_STAMPED_TOOL_CONTEXT_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "tool_call_id",
        "protocore.tool_visibility_policy",
        "protocore.tool_allowlist",
        "protocore.advertised_tools",
        "protocore.subagent_dispatch_order",
        "protocore.subagent_dispatch_group",
        "protocore.subagent_tree_permit",
        "protocore.synthetic_recovery",
    }
)


class ToolError(Exception):
    """Base class for tool-domain errors."""


class ToolInvocationError(ToolError):
    """Tool raised during invocation; caught by dispatcher."""


class ToolPolicyDenied(ToolError):
    """Tool invocation blocked by safety policy."""


class ToolContext(BaseModel):
    """Per-invocation context — injected by the loop.

    Tools must NOT mutate the context; treat as read-only.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tenant_id: str
    """The run's scope, as an opaque key.

    Core neither parses it nor knows what a host addresses with it: it keys
    hooks, secrets, constants, workspace and sessions, and every one of those
    is reached through a contract the host implements. Whatever else a host
    knows about the scope — who owns it, what it is billed to — stays with the
    host, in :attr:`RunScopedState.host` or under a host key in
    :attr:`metadata`.
    """
    run_id: str
    session_id: str
    work_scope: str = ""
    """The pool scope work this call starts belongs to, or ``""`` for the session's.

    A tool that launches background work files it under this scope, so the run
    that started it is the run that can stop it. A delegated run shares the
    session — same workspace, same wake delivery — while owning the work it
    starts on its own; every other run leaves this empty and its work is the
    session's, exactly as it was before a run could be delegated.
    """
    evidence: ToolEvidenceContext | None = None
    """Provenance for evidence this invocation produces, or ``None`` for none.

    One value rather than three loose fields, because they are only meaningful
    together: see
    :class:`~protocore.contracts.evidence.ToolEvidenceContext`. A tool that
    produces no evidence never reads it.
    """
    run_state: RunScopedState | None = None
    """The run's own state, shared by reference with every call in the run.

    Typed and named — a tool reads an allowance off an attribute, and a reader
    that asks for one this run does not carry gets a type error rather than a
    silently missing value. ``None`` only where a caller invokes a tool outside
    a run at all; every path the loop takes supplies one.
    """
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC):
    """Adapter-bound tool. All 11 default tools live in the host.

    Concrete subclass implements:
        - :attr:`name`: stable identifier surfaced to LLM
        - :attr:`definition`: full :class:`ToolDefinition` for surface
        - :meth:`invoke`: actual side-effect
    """

    @property
    def evidence_producer(self) -> EvidenceProducerBinding | None:
        """Trusted binding configured for evidence this registered tool may emit.

        ``None`` is the safe default: returning evidence without a registry
        binding fails closed.  Tool implementations cannot select the binding
        for an individual invocation; the dispatcher stamps it onto records.
        """
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable tool name. Matches ``definition.name``."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Full tool definition (name + description + JSON Schema)."""

    @abstractmethod
    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Invoke the tool with validated arguments.

        Contract:
            - Caller has already JSON-Schema-validated ``arguments``.
            - Caller has already passed safety policy gate.
            - On error: raise :class:`ToolInvocationError`; never swallow.
            - On policy denial: raise :class:`ToolPolicyDenied`.
        """


def read_metadata(context: ToolContext, key: str, default: Any = None) -> Any:
    """The value a host put under ``key``, or ``default``.

    The one way core reads the metadata bag, and it refuses a key core has not
    declared in :data:`CORE_TOOL_CONTEXT_METADATA_KEYS`. Without that, reading
    the bag is an open channel: any module could start depending on a string a
    host happened to spell, and nothing anywhere would say so — least of all
    the host, which would learn it had been supplying an input the day it
    stopped.
    """
    if key not in CORE_TOOL_CONTEXT_METADATA_KEYS:
        raise KeyError(
            f"{key!r} is not a metadata key the core declares; add it to "
            "CORE_TOOL_CONTEXT_METADATA_KEYS if the core is to read it"
        )
    return (context.metadata or {}).get(key, default)


def has_metadata(context: ToolContext, key: str) -> bool:
    """Whether the host stated anything at all under ``key``.

    Distinct from a ``None`` value on purpose: "the host has no opinion" and
    "the host said no" are different answers, and several guards turn on which
    one they got. Refuses an undeclared key exactly as :func:`read_metadata`
    does.
    """
    if key not in CORE_TOOL_CONTEXT_METADATA_KEYS:
        raise KeyError(
            f"{key!r} is not a metadata key the core declares; add it to "
            "CORE_TOOL_CONTEXT_METADATA_KEYS if the core is to read it"
        )
    return key in (context.metadata or {})


def copy_metadata(context: ToolContext) -> dict[str, Any]:
    """A mutable copy of the whole bag, for a caller building the next one.

    The dispatcher stamps its own keys onto a copy before it hands the context
    to a tool. It carries the host's keys through untouched and reads none of
    them, which is why this is not a read and needs no declaration.
    """
    return dict(context.metadata)


__all__ = [
    "CORE_STAMPED_TOOL_CONTEXT_METADATA_KEYS",
    "CORE_TOOL_CONTEXT_METADATA_KEYS",
    "SUBAGENT_DISPATCH_GROUP_METADATA_KEY",
    "SUBAGENT_DISPATCH_ORDER_METADATA_KEY",
    "SUBAGENT_TREE_PERMIT_METADATA_KEY",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolInvocationError",
    "ToolPolicyDenied",
    "copy_metadata",
    "has_metadata",
    "read_metadata",
]
