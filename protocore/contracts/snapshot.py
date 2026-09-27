"""The run snapshot's schema: how it states its own version, and who may read it.

A snapshot is written by one process and read by another, and the two are not
guaranteed to be the same build. That gap is the whole problem this module
exists for. A reader that quietly accepts a payload it does not understand does
not fail — it resumes a run with fields missing, latches unset and budgets
refilled, and nothing downstream can tell that apart from a run that legitimately
had none of those things. The failure surfaces much later, as an agent that
repeats work it already did or spends an allowance it already spent.

So the snapshot states its schema version in the payload, and a reader that
cannot recognise that version refuses the snapshot outright. Refusing is the
recoverable outcome: the run stays where it was and an operator sees why.

An older payload is not refused where it can be brought forward instead. An
upcaster chain does that: one registered step per version, each reading the
shape one version below it and filling in what that version introduced. A
version with no step is a refusal — skipping one leaves its fields unset, which
is the same silent half-restore.

Versions are integers, they only go up, and the newest is
:data:`SNAPSHOT_SCHEMA_VERSION`. A payload with no version field at all is
version 1: version 1 is exactly the shape the field was added on top of, and it
is the shape every store already holds, because the field has never shipped
without a value above it. Reading it as anything else would strand every run
that was paused when the change rolled out — and a paused run is the one case
where refusal is not recoverable, since nothing later will accept the payload
either.
"""
from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import Any

#: Where the version lives inside the snapshot payload.
SNAPSHOT_SCHEMA_KEY = "schema_version"

#: The schema version this build writes and can read after upcasting.
SNAPSHOT_SCHEMA_VERSION = 7

#: Where the run's own state sits inside the snapshot payload: the tree's
#: cumulative work ledger and the capacity of its concurrency budget, the two
#: allowances a resumed run must not be handed a second time.
RUN_SCOPED_STATE_SNAPSHOT_KEY = "run_scoped_state"

#: Where everything the run is waiting for sits inside the snapshot payload.
PENDING_INTERRUPTS_SNAPSHOT_KEY = "pending_interrupts"

#: One step of the chain: read a payload at version N, return it at N+1.
SnapshotUpcaster = Callable[[dict[str, Any]], dict[str, Any]]


class SnapshotSchemaError(ValueError):
    """A snapshot cannot be read as the schema this build understands.

    Subclasses :class:`ValueError` because every other refusal
    ``resume_from_snapshot`` raises is one, and a caller that already treats a
    refused snapshot as "leave the run alone" keeps doing so unchanged.
    """


def read_schema_version(snapshot: dict[str, Any]) -> int:
    """The version ``snapshot`` declares, or raise.

    A payload that carries no version at all is version 1 — that is the shape
    the field was added on top of, and the only shape a store can hold without
    it. Everything else is refused: a version below 1, a version above what
    this build reads, or a value that is not an integer at all.

    ``bool`` is rejected explicitly: it is an ``int`` subclass, so a payload
    carrying ``schema_version: true`` would otherwise read as version 1 and be
    accepted by the build that happens to be at version 1.
    """
    if not isinstance(snapshot, dict):
        raise SnapshotSchemaError(
            f"snapshot must be a mapping, got {type(snapshot).__name__}"
        )
    if SNAPSHOT_SCHEMA_KEY not in snapshot:
        return 1
    version: object = snapshot[SNAPSHOT_SCHEMA_KEY]
    if isinstance(version, bool) or not isinstance(version, int):
        raise SnapshotSchemaError(
            f"snapshot {SNAPSHOT_SCHEMA_KEY!r} must be an integer, got "
            f"{type(version).__name__}"
        )
    if version < 1:
        raise SnapshotSchemaError(
            f"snapshot {SNAPSHOT_SCHEMA_KEY!r} must be at least 1, got {version}"
        )
    return version


class UpcasterRegistry:
    """The chain that brings an older payload forward, one version at a time.

    Each step is registered against the version it reads and produces the next
    one up, so bringing a version-1 payload to version 3 runs the 1→2 step and
    then the 2→3 step. Steps are written once, in the change that introduced the
    version they produce, and never revised afterwards — a step is a statement
    about a shape that already exists in some store somewhere, and editing it
    rewrites history that has already been written.

    A gap in the chain is a refusal, not a skip: jumping a version means the
    fields that version introduced are never filled in, which is the silent
    half-restore the schema exists to prevent.
    """

    def __init__(self, current_version: int) -> None:
        self._current_version = current_version
        self._steps: dict[int, SnapshotUpcaster] = {}

    @property
    def current_version(self) -> int:
        """The version this registry upgrades to."""
        return self._current_version

    def register(self, from_version: int, upcaster: SnapshotUpcaster) -> None:
        """Add the step that reads ``from_version`` and produces the next one."""
        if from_version < 1:
            raise ValueError(f"upcaster source version must be at least 1, got {from_version}")
        if from_version >= self._current_version:
            raise ValueError(
                f"upcaster from version {from_version} produces "
                f"{from_version + 1}, which is not below the current version "
                f"{self._current_version}"
            )
        if from_version in self._steps:
            raise ValueError(f"an upcaster from version {from_version} is already registered")
        self._steps[from_version] = upcaster

    def upgrade(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Return ``snapshot`` at :attr:`current_version`, or raise.

        The input is left alone — a caller that refuses the result still holds
        the payload it read, unchanged, to log or to hand back to its store.
        """
        version = read_schema_version(snapshot)
        if version > self._current_version:
            raise SnapshotSchemaError(
                f"snapshot schema version {version} was written by a newer "
                f"build; this one reads at most {self._current_version}"
            )
        upgraded = dict(snapshot)
        while version < self._current_version:
            step = self._steps.get(version)
            if step is None:
                raise SnapshotSchemaError(
                    f"no upcaster brings a snapshot from schema version "
                    f"{version} to {version + 1}; this build cannot read it"
                )
            upgraded = step(upgraded)
            # The step is trusted to fill the fields, not to remember the
            # bookkeeping: the version is stamped here so every step is written
            # as a plain transformation of the payload.
            upgraded[SNAPSHOT_SCHEMA_KEY] = version + 1
            version += 1
        return upgraded


#: The chain this build reads with. Steps are registered by the modules that
#: introduce a version, at import time.
UPCASTERS = UpcasterRegistry(SNAPSHOT_SCHEMA_VERSION)


def _v1_to_v2(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Version 2 states the run continuity that a re-armed turn keeps.

    Version 1 is the shape written before the payload stated a version at all.
    It carried the run's counters, latches and budgets but left the
    continuity implicit — the compaction checkpoint, the discovered and
    activated rule files, the pinned-tool order, the session grants, the
    execution-profile audit and the spans. A reader had to invent a default for
    each, and an invented default is indistinguishable from a run that
    genuinely had none.

    Version 1 runs really did have none of it recorded, so the honest lift is
    the empty one: no checkpoint, nothing discovered, nothing activated, nothing
    pinned, no records. That is a real loss for a run mid-rollout — the turns
    folded away behind a checkpoint written under version 1 do not come back —
    and it is stated here rather than spread across the restore as a chain of
    permissive defaults.
    """
    return {
        "compact_checkpoint": None,
        "active_rule_paths": [],
        "discovered_rules": [],
        "session_grants": [],
        "profile_audit": [],
        "spans": [],
        "context_manager_pinned_tools": [],
        "skill_catalog_block_sha256": None,
        **snapshot,
    }


UPCASTERS.register(1, _v1_to_v2)


def _v2_to_v3(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Version 3 names the manifest of the last request the run made.

    Version 2 carried the run's history but nothing about the request that
    history was turned into, so no reader of a version-2 payload could say
    which provider call the run was on when it was written. The lift is
    ``None``: a version-2 run really has no manifest recorded — the machinery
    that writes one did not exist — and ``None`` is the same value a run whose
    host keeps no manifests writes today, which is exactly the right reading.
    Nothing downstream treats it as a failure; a manifest that is not there is
    evidence the run does not have.
    """
    return {"last_request_manifest": None, **snapshot}


UPCASTERS.register(2, _v2_to_v3)


def _v3_to_v4(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Version 4 gathers the run's allowances under one named state.

    Version 3 wrote the tree's cumulative work ledger and its concurrency
    budget as two top-level fields named after the keys they used to occupy in
    an untyped per-run bag. The bag is gone and the state has a type, so the
    two fields move under one name that says what they are. The lift carries
    both values across verbatim — a version-3 run really did spend that ledger,
    and dropping it here would hand a resumed tree its whole budget again, which
    is the exact failure the ledger exists to close.
    """
    lifted = {
        key: value
        for key, value in snapshot.items()
        if key not in ("run_work_ledger", "subagent_tree_budget")
    }
    lifted[RUN_SCOPED_STATE_SNAPSHOT_KEY] = {
        "run_work_ledger": snapshot.get("run_work_ledger"),
        "subagent_tree_budget": snapshot.get("subagent_tree_budget"),
    }
    return lifted


UPCASTERS.register(3, _v3_to_v4)


def _v4_to_v5(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Version 5 separates a tool result's value from the text shown for it.

    Up to version 4 a recorded tool result was one string doing three jobs at
    once, so a reader could not tell a whole value from a summary of one that
    had been shed. Version 5 states both facts beside the text: where the whole
    value is kept, and which file the result describes.

    A version-4 result has neither, and no honest lift invents them. Its text
    is the whole of what was kept, so there is no other copy to point at; and
    nothing recorded which file it was about, so no later write can be said to
    have made it stale. Both are written out as absent rather than left off,
    so a reader of a lifted payload sees a run that HAD no reference and no
    path, not a payload whose fields it forgot to read.
    """
    history = snapshot.get("history")
    if not isinstance(history, list):
        return dict(snapshot)
    lifted_history: list[Any] = []
    for message in history:
        if not isinstance(message, dict):
            lifted_history.append(message)
            continue
        blocks = message.get("content_blocks")
        if not isinstance(blocks, list):
            lifted_history.append(message)
            continue
        lifted_blocks: list[Any] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("kind") == "tool_result":
                lifted_blocks.append({"canonical_ref": None, "path": None, **block})
            else:
                lifted_blocks.append(block)
        lifted_history.append({**message, "content_blocks": lifted_blocks})
    return {**snapshot, "history": lifted_history}


UPCASTERS.register(4, _v4_to_v5)


#: The intent state a version-5 build stamped on a call a gate had parked. Any
#: other state on the parked call means the tool had already been reached.
_V5_PENDING_APPROVAL_STATE = "PENDING_APPROVAL"


def _open_intent_for(snapshot: dict[str, Any], tool_call_id: str) -> dict[str, Any] | None:
    """The stored intent record for one call, when the payload carries one."""
    records = snapshot.get("open_intents")
    if not isinstance(records, list):
        return None
    for record in records:
        if isinstance(record, dict) and record.get("tool_call_id") == tool_call_id:
            return record
    return None


def _v5_to_v6(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Version 6 says what the run is waiting FOR, not merely that it waits.

    Up to version 5 a paused run carried one field holding one tool-call id.
    It could say that a call was parked and nothing else: not whether a person
    owed a decision on a call that had not run, or an answer to a question a
    tool had already asked, and not that more than one call was parked at once,
    because there was only ever room for one id.

    So the lift has to work out which of the two a stored run was doing, and
    the payload already says: the intent record for the parked call is the one
    place the old build distinguished them. A call held at a gate was marked
    pending approval; a call that ran and asked was not. Reading the id as an
    approval unconditionally handed every stopped question back as an approval,
    and both routes out then refuse it — a question cannot be answered because
    the wait says approval, and approving it would re-run the tool that already
    asked. A payload with no intent record at all keeps the historical reading,
    approval, because that is the only wait the field was ever set for on a
    build old enough to have no records.

    Nothing else is invented: the version 5 payload recorded no question text
    and no expiry, so the lifted interrupt has an empty payload and never
    expires, which is the truth about a wait nobody put a deadline on.

    The identity is derived from the call id rather than minted fresh, so
    lifting the same version-5 payload twice — a retried resume, two readers of
    the same stored run — names the same interrupt both times. A fresh id per
    read would hand two processes two names for one wait.

    A version-5 payload with no id parked lifts to an empty list, which is the
    same shape a run that never waited writes today.
    """
    lifted = {key: value for key, value in snapshot.items() if key != "pending_approval_tool_call_id"}
    parked = snapshot.get("pending_approval_tool_call_id")
    if isinstance(parked, str) and parked:
        intent = _open_intent_for(snapshot, parked)
        kind = "approval"
        tool_name = ""
        if intent is not None:
            tool_name = str(intent.get("tool_name") or "")
            if intent.get("state") != _V5_PENDING_APPROVAL_STATE:
                kind = "question"
        lifted[PENDING_INTERRUPTS_SNAPSHOT_KEY] = [
            {
                "interrupt_id": f"int_{sha256(parked.encode()).hexdigest()[:16]}",
                "kind": kind,
                "tool_call_id": parked,
                "tool_name": tool_name,
                "payload": {},
                "created_at_ms": 0,
                "expires_at_ms": None,
            }
        ]
    else:
        if snapshot.get("state") == "awaiting":
            # A version-5 run that says it is waiting and names no parked call
            # is a run nothing could ever answer: the one field that could have
            # said what it waits for is empty. Refused here, where the payload
            # is still in hand and can be looked at, rather than lifted into a
            # shape the loop forbids and discovered later as a run that stopped
            # for no stated reason.
            raise SnapshotSchemaError(
                "snapshot is AWAITING with no parked call: nothing recorded "
                "in it could resume the run"
            )
        lifted[PENDING_INTERRUPTS_SNAPSHOT_KEY] = []
    return lifted


UPCASTERS.register(5, _v5_to_v6)


def _v6_to_v7(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Version 7 keeps discovered tools in discovery order and states what was held back.

    Up to version 6 the run carried one list of tool names, in least-recently
    pinned order, because the tools the agent loaded were kept by recency and
    placed by name. Version 7 places them by the order they were discovered
    and evicts by last use, so each needs both facts; and it records which
    tool groups the run held back, because the catalogue naming them sits at
    the head of the cached prompt and must come back the same.

    The version 6 order is the only order there is, so it is taken for both:
    discovery order and recency, the first name the least recently used. Nothing
    was held back under version 6 — the mechanism did not exist — so the lift
    says exactly that, an empty list, and the resumed run keeps the whole
    surface it had rather than deciding afresh partway through. No group's rules
    had been given either, since groups had none, so that list is empty too,
    and there is no rules mark (the resumed run uses its own).
    """
    lifted = {key: value for key, value in snapshot.items() if key != "context_manager_pinned_tools"}
    names = snapshot.get("context_manager_pinned_tools")
    rows = [
        {"name": name, "last_used": tick}
        for tick, name in enumerate(names if isinstance(names, list) else [], start=1)
        if isinstance(name, str)
    ]
    lifted.setdefault("discovered_tools", rows)
    lifted.setdefault("deferred_tool_groups", [])
    lifted.setdefault("tool_group_rules_given", [])
    lifted.setdefault("tool_group_rules_mark", "")
    return lifted


UPCASTERS.register(6, _v6_to_v7)


def migrate_snapshot(
    snapshot: dict[str, Any], *, registry: UpcasterRegistry | None = None
) -> dict[str, Any]:
    """Refuse a snapshot this build cannot read; return the rest at the current version.

    ``registry`` is an injection seam for tests of the chain itself. Production
    callers pass nothing and get :data:`UPCASTERS`.
    """
    return (registry or UPCASTERS).upgrade(snapshot)
