"""The per-run state a tool call can reach: one typed object, not a bag of keys.

Every cross-call allowance a run carries — the streaks the dispatcher counts,
the cumulative work ledger the whole run tree draws on, the cancel event, the
locks, the satisfied preconditions — used to live in one untyped dictionary
threaded through ``ToolContext.metadata`` under a single agreed string. Three
independent declarations of that string existed across core and the host, and
nothing checked that a reader spelled a key the way its writer did: a typo
produced a missing allowance, which reads at runtime exactly like a run that
legitimately had none.

This is that state with names. A field that moves is a type error at the reader,
a field that is absent has one spelling, and a host wiring its own slots keeps
them in :attr:`RunScopedState.host` — one opaque compartment, so core never has
to know what a host puts there and a host never has to guess which key core
reads.

Only two things in here outlive the process: the run tree's work ledger and the
capacity of its concurrency budget. :meth:`RunScopedState.to_snapshot` states
exactly those, and :meth:`RunScopedState.apply_snapshot` puts them back — live
objects (an ``asyncio.Event``, a semaphore, a lock) are per-process by nature
and are rebuilt by whoever wires the resumed run, never restored from a payload.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from protocore.contracts.recovery import RecoveryBudget
    from protocore.runtime.run_work_budget import RunWorkLedger
    from protocore.runtime.subagent_budget import SubagentTreeBudget, SubagentTreePermit


@dataclass
class ConsecutiveErrorStreak:
    """How many times in a row the same tool failed the same way."""

    tool_name: str | None = None
    signature: str | None = None
    count: int = 0


@dataclass
class SignatureStreak:
    """A streak counted against one canonical error signature."""

    signature: str | None = None
    count: int = 0


@dataclass
class ToolStreak:
    """A streak counted against one tool name."""

    tool_name: str | None = None
    count: int = 0


@dataclass
class ToolCallSoftCapState:
    """Per-tool call counts and the advisory warnings they have produced.

    ``lock`` guards the counts when a run dispatches tools concurrently. It is
    duck-typed (anything with ``__aenter__``) because a host may hand its own
    shared lock down a run tree; ``None`` means the counter mints a private one
    and the count is only correct within this run.
    """

    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    lock: Any | None = None


@dataclass
class RunScopedState:
    """Everything one run carries across its own tool calls.

    Constructed by whoever starts the run and threaded by reference: a subagent
    that must share an allowance with its parent gets the same object in that
    field, and one that must not gets its own. Mutable on purpose — a streak
    that could not be written back would not be a streak.
    """

    #: The constants snapshot this run resolved. Duck-typed: core reads named
    #: fields off it defensively so a run wired without one still dispatches.
    rc: Any | None = None

    #: The root run's id, for a subagent whose own id names no durable row.
    root_run_id: str | None = None

    #: Set by whoever cancels the run; the dispatcher races tool calls against
    #: it so a leader blocked inside a long tool unblocks promptly.
    cancel_event: asyncio.Event | None = None

    #: Shared across a run tree so concurrently-dispatched calls serialise
    #: their writes to the counts below. Duck-typed for the same reason as
    #: :attr:`ToolCallSoftCapState.lock`.
    tool_shared_state_lock: Any | None = None

    #: The tree's CUMULATIVE work budget — child runs started, tokens charged.
    run_work_ledger: RunWorkLedger | None = None

    #: The one allowance every recovery layer draws from: in-place retries,
    #: chain advances, empty-round re-asks, overflow shrinking, and a host's
    #: re-drives. Per run, not per tree: a subagent's outage is its own.
    recovery_budget: RecoveryBudget | None = None

    #: The tree's INSTANTANEOUS concurrency budget, minted at the first
    #: parallel fan-out and shared by reference with every descendant.
    subagent_tree_budget: SubagentTreeBudget | None = None

    #: This run's own slot in the budget above, held while it executes. Absent
    #: on a root leader, which was never dispatched under a budget.
    subagent_tree_permit: SubagentTreePermit | None = None

    #: Advisory per-tool call limits, and the counts they are measured against.
    tool_call_soft_caps: dict[str, int] = field(default_factory=dict)
    tool_call_soft_cap_state: ToolCallSoftCapState = field(
        default_factory=ToolCallSoftCapState
    )

    #: The three streaks whose caps are stated per-run, and the one-shot signal
    #: the transport streak raises for the loop to consume.
    consecutive_error: ConsecutiveErrorStreak | None = None
    transport_down: SignatureStreak | None = None
    transport_down_injection_pending: bool = False
    string_type: ToolStreak | None = None

    #: Preconditions this run has already satisfied, as normalised entries.
    satisfied_preconditions: set[str] = field(default_factory=set)

    #: Command grants a person gave this session; a grant skips approval only.
    session_grants: list[Any] = field(default_factory=list)

    #: Telemetry sinks a host may wire. Duck-typed; absent means no telemetry.
    tool_error_counter: Any | None = None
    adaptive_safety_band: Any | None = None

    #: The operator-supplied per-run envelope. Merged onto tool metadata with
    #: runtime-internal names skipped, so it can never shadow trusted state.
    run_metadata: dict[str, Any] = field(default_factory=dict)

    #: The host's own slots. Core neither reads nor writes this compartment; it
    #: exists so a host has one place to put what only it understands, reached
    #: through the same object rather than through a second channel.
    host: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Carry this object through a model by reference, never by value.

        A model field holding run state must hold THE run's state: the whole
        point is that the loop, the dispatcher and every tool mutate one object.
        Pydantic's default treatment of a dataclass field is to validate it,
        which builds a copy — and a copy is a second run's worth of allowances
        that nothing ever reconciles with the first. An instance check is the
        whole contract here.
        """
        return core_schema.is_instance_schema(cls)

    # ------------------------------------------------------------------
    # Allowances that are minted on demand
    # ------------------------------------------------------------------

    def ensure_run_work_ledger(self, rc: Any) -> RunWorkLedger:
        """Return the tree's ledger, minting it here if the run has none.

        The live path finds one already set by whoever composed the run. Minting
        is the fallback for a run composed without one: it still bounds this
        run's own subtree, which is the most a run sharing nothing can be held
        to, and it cannot be mistaken for a tree-wide bound because nothing else
        can reach it.
        """
        if self.run_work_ledger is None:
            from protocore.runtime.run_work_budget import mint_run_work_ledger

            self.run_work_ledger = mint_run_work_ledger(rc)
        return self.run_work_ledger

    def ensure_recovery_budget(self, rc: Any) -> RecoveryBudget:
        """Return the run's recovery budget, minting one with ``rc``'s bounds if it has none."""
        if self.recovery_budget is None:
            from protocore.contracts.recovery import mint_recovery_budget

            self.recovery_budget = mint_recovery_budget(rc)
        return self.recovery_budget

    def ensure_subagent_tree_budget(self, cap: int) -> SubagentTreeBudget:
        """Return the tree's concurrency budget, minting it at first fan-out.

        Deliberately lazy, unlike the ledger: a bound on how many children run
        AT ONCE only means anything once a run fans out, and minting it earlier
        would state a bound the run never operated under.
        """
        if self.subagent_tree_budget is None:
            from protocore.runtime.subagent_budget import SubagentTreeBudget as _Budget

            self.subagent_tree_budget = _Budget(cap)
        return self.subagent_tree_budget

    # ------------------------------------------------------------------
    # Per-run cells that a re-armed turn must not inherit
    # ------------------------------------------------------------------

    def clear_run_scoped_streaks(self) -> None:
        """Drop the streaks and the one-shot signal that go with them.

        A re-armed engine that kept them would open its next turn with the
        previous turn's streak already counted and its one-shot already spent,
        so an agent repeating one failing call each turn would cross a per-run
        cap that no single turn ever reached.

        Everything else survives on purpose. The ledger above all: it is the
        run *tree's*, shared with subagents that may still be drawing on it, so
        emptying it from the leader would be a decision about their budget too.
        """
        self.consecutive_error = None
        self.transport_down = None
        self.transport_down_injection_pending = False
        self.string_type = None
        self.tool_call_soft_cap_state = ToolCallSoftCapState(
            lock=self.tool_call_soft_cap_state.lock
        )

    # ------------------------------------------------------------------
    # Transcript-order state, for a batch dispatched in parallel
    # ------------------------------------------------------------------

    def transcript_state(self) -> TranscriptOrderedState:
        """Copy the cells whose meaning is "in the order the model asked".

        A parallel batch mutates these in completion order, which is not the
        order the run is a record of. The caller takes this copy before the
        batch, puts it back after, and then replays the batch's outcomes in the
        model's order so the next turn's caps fire on the right count.
        """
        return TranscriptOrderedState(
            consecutive_error=_copy_streak(self.consecutive_error),
            transport_down=_copy_streak(self.transport_down),
            transport_down_injection_pending=self.transport_down_injection_pending,
            string_type=_copy_streak(self.string_type),
            satisfied_preconditions=set(self.satisfied_preconditions),
            soft_cap_counts=dict(self.tool_call_soft_cap_state.counts),
        )

    def restore_transcript_state(self, saved: TranscriptOrderedState) -> None:
        """Put the cells back the way :meth:`transcript_state` found them."""
        self.consecutive_error = _copy_streak(saved.consecutive_error)
        self.transport_down = _copy_streak(saved.transport_down)
        self.transport_down_injection_pending = saved.transport_down_injection_pending
        self.string_type = _copy_streak(saved.string_type)
        self.satisfied_preconditions = set(saved.satisfied_preconditions)
        self.tool_call_soft_cap_state.counts = dict(saved.soft_cap_counts)

    # ------------------------------------------------------------------
    # What survives the process
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict[str, Any]:
        """State the two allowances a resumed run must not be handed afresh.

        The ledger is the only bound on the TOTAL work a run tree may do, and
        the budget's ``(capacity, in use)`` pair is all that a live semaphore
        leaves behind. A run resumed without them would get a full budget of
        child runs and tokens for a second time, and nothing downstream could
        tell that apart from a run that had never spent any.
        """
        budget = self.subagent_tree_budget
        return {
            "run_work_ledger": (
                self.run_work_ledger.to_snapshot()
                if self.run_work_ledger is not None
                else None
            ),
            "subagent_tree_budget": (
                {"capacity": budget.capacity, "in_use": budget.in_use}
                if budget is not None
                else None
            ),
            # What the run has spent recovering in the current stretch. A run
            # resumed without it would retry an outage from a full allowance,
            # the multiplication the budget exists to stop.
            "recovery_budget": (
                self.recovery_budget.to_dict() if self.recovery_budget is not None else None
            ),
        }

    def apply_snapshot(self, payload: Any) -> None:
        """Restore what :meth:`to_snapshot` stated, leaving the rest alone.

        A payload that is not a mapping, or that carries neither allowance,
        leaves this run exactly as it was — the caller has already decided
        whether that is a resume it will accept.

        The concurrency budget comes back with its CAPACITY and with none of
        its occupancy. A semaphore is a per-process object: the slots the
        payload records were held by coroutines in a process that is gone,
        nothing here can ever release them, and a budget rebuilt with them taken
        loses that much capacity for the rest of the run's life — or, at full
        occupancy, waits forever on its first delegation. A run that already
        holds a live budget keeps it untouched, because its own descendants may
        be holding permits on that very object.
        """
        if not isinstance(payload, dict):
            return
        from protocore.runtime.run_work_budget import RunWorkLedger as _Ledger
        from protocore.runtime.subagent_budget import SubagentTreeBudget as _Budget

        ledger_payload = payload.get("run_work_ledger")
        if isinstance(ledger_payload, dict):
            self.run_work_ledger = _Ledger.from_snapshot(ledger_payload)

        from protocore.contracts.recovery import RecoveryBudget as _Recovery

        restored_recovery = _Recovery.from_dict(payload.get("recovery_budget"))
        if restored_recovery is not None:
            self.recovery_budget = restored_recovery

        budget_payload = payload.get("subagent_tree_budget")
        if isinstance(budget_payload, dict):
            capacity: Any = budget_payload.get("capacity")
            if _is_count(capacity) and self.subagent_tree_budget is None:
                self.subagent_tree_budget = _Budget(int(capacity))
        elif budget_payload is None:
            # A tree that never fanned out persists no budget, and inventing one
            # on resume would state a bound the run never operated under.
            pass


@dataclass
class TranscriptOrderedState:
    """A copy of the cells a parallel batch must not decide the order of."""

    consecutive_error: ConsecutiveErrorStreak | None = None
    transport_down: SignatureStreak | None = None
    transport_down_injection_pending: bool = False
    string_type: ToolStreak | None = None
    satisfied_preconditions: set[str] = field(default_factory=set)
    soft_cap_counts: dict[str, int] = field(default_factory=dict)


def _copy_streak[
    Streak: (ConsecutiveErrorStreak, SignatureStreak, ToolStreak)
](streak: Streak | None) -> Streak | None:
    """Copy one streak record, or pass ``None`` through.

    A copy, not a reference: the caller takes it to hold a value across a batch
    that is about to mutate the original.
    """
    return None if streak is None else replace(streak)


def _is_count(value: Any) -> bool:
    """Whether ``value`` is a real non-negative integer count.

    ``bool`` is rejected explicitly: it is an ``int`` subclass, so a malformed
    ``true`` would otherwise restore as a capacity of one.
    """
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


__all__ = [
    "ConsecutiveErrorStreak",
    "RunScopedState",
    "SignatureStreak",
    "ToolCallSoftCapState",
    "ToolStreak",
    "TranscriptOrderedState",
]
