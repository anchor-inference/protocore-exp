"""``ContextManager`` + token-budget derivation + three-tier compaction.

"""
from __future__ import annotations

from protocore.runtime.context.budgets import TokenBudgets, derive_budgets
from protocore.runtime.context.compaction import (
    CompactionAttempt,
    CompactionExhaustedError,
    CompactionState,
    Tier1Result,
    Tier2Result,
    Tier3Result,
    TokenEstimator,
    estimate_history_tokens,
    estimate_message_tokens,
)
from protocore.runtime.context.manager import ContextBundle, ContextManager
from protocore.runtime.context.session_compaction import (
    CompactionHold,
    CompactionHoldPolicy,
    SessionCompaction,
    SessionCompactionSettings,
    Summariser,
    SummariserLadder,
    SummaryPrompts,
    commit_session_compaction,
    compact_session,
)
from protocore.runtime.context.session_memory import (
    SUMMARY_SYSTEM,
    ArtifactLedger,
    FoldResult,
    SessionMemory,
    build_seed,
    build_summary_user_message,
    extract_artifacts,
    fold_run,
    render_ledger,
    running_summary_needed,
    summary_fold_threshold_tokens,
)

__all__ = [
    "SUMMARY_SYSTEM",
    "ArtifactLedger",
    "CompactionAttempt",
    "CompactionExhaustedError",
    "CompactionHold",
    "CompactionHoldPolicy",
    "CompactionState",
    "ContextBundle",
    "ContextManager",
    "FoldResult",
    "SessionCompaction",
    "SessionCompactionSettings",
    "SessionMemory",
    "Summariser",
    "SummariserLadder",
    "SummaryPrompts",
    "Tier1Result",
    "Tier2Result",
    "Tier3Result",
    "TokenBudgets",
    "TokenEstimator",
    "build_seed",
    "build_summary_user_message",
    "commit_session_compaction",
    "compact_session",
    "derive_budgets",
    "estimate_history_tokens",
    "estimate_message_tokens",
    "extract_artifacts",
    "fold_run",
    "render_ledger",
    "running_summary_needed",
    "summary_fold_threshold_tokens",
]
