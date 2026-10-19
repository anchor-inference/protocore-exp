"""Pure-core runtime utilities + agent loop.

 shipped: token counting, compaction-threshold derivation,
wire-format placeholders, chain parser, BM25 tool retrieval.

 ships:
 - :class:`~protocore.runtime.query_engine.QueryEngine` per-conversation engine
 - :func:`~protocore.runtime.query.resume` pick a stored run back up and drive it
 - :func:`~protocore.runtime.query.resume_approved_tool` run a call held for approval
 - :func:`~protocore.runtime.query.resume_interrupts` answer everything a run waits on
 - :func:`~protocore.runtime.query.answer_pending` answer one parked question in place
 - :class:`~protocore.runtime.context.manager.ContextManager` + two-tier compaction
 - :class:`~protocore.runtime.events.types.EventType` + :class:`~protocore.runtime.events.envelope.TurnEvent`
 - :class:`~protocore.runtime.loop_state.LoopState` state machine
 - :class:`~protocore.runtime.usage.TokenUsage` accumulator
"""
from __future__ import annotations

from protocore.runtime.candidate_delivery import CandidateDeliveryGate
from protocore.runtime.context import (
    CompactionAttempt,
    CompactionExhaustedError,
    CompactionState,
    ContextBundle,
    ContextManager,
    Tier1Result,
    Tier2Result,
    TokenBudgets,
    derive_budgets,
)
from protocore.runtime.events import EventType, InMemoryTurnEventBuffer, TurnEvent
from protocore.runtime.loop_state import (
    InvalidStateTransitionError,
    LoopState,
    assert_transition,
    is_terminal,
)
from protocore.runtime.query import (
    answer_pending,
    resume,
    resume_approved_tool,
    resume_interrupts,
)
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.runtime.tool_dispatch import (
    DispatchErrorKind,
    DispatchOutcome,
    ToolDispatcher,
    consume_transport_down_injection_signal,
)
from protocore.runtime.tool_permission import (
    HttpDnsAllowlistPolicy,
    PermissionStage,
    ShellSafetyPolicyAdapter,
    ToolPermissionDecision,
    ToolPermissionGate,
    ToolPermissionOutcome,
    WorkspacePathPolicy,
)
from protocore.runtime.tool_registry import ToolRegistry
from protocore.runtime.usage import TokenUsage

__all__ = [
    "CandidateDeliveryGate",
    "CompactionAttempt",
    "CompactionExhaustedError",
    "CompactionState",
    "ContextBundle",
    "ContextManager",
    "DispatchErrorKind",
    "DispatchOutcome",
    "EventType",
    "HttpDnsAllowlistPolicy",
    "InMemoryTurnEventBuffer",
    "InvalidStateTransitionError",
    "LoopState",
    "PermissionStage",
    "QueryEngine",
    "QueryEngineConfig",
    "ShellSafetyPolicyAdapter",
    "Tier1Result",
    "Tier2Result",
    "TokenBudgets",
    "TokenUsage",
    "ToolDispatcher",
    "ToolPermissionDecision",
    "ToolPermissionGate",
    "ToolPermissionOutcome",
    "ToolRegistry",
    "TurnEvent",
    "WorkspacePathPolicy",
    "answer_pending",
    "assert_transition",
    "consume_transport_down_injection_signal",
    "derive_budgets",
    "is_terminal",
    "resume",
    "resume_approved_tool",
    "resume_interrupts",
]
