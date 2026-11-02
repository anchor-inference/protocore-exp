"""Durable tool intent: the record a run writes BEFORE it calls a tool.

A run that dies between "the tool was called" and "the result was written to
history" leaves a ``tool_use`` block with no ``tool_result``. Without a durable
record of the call, a resumed run cannot tell three very different situations
apart:

* the call never started, because a permission gate parked it for approval;
* the call started and its outcome is genuinely unknown;
* the call started, asked the user a question, and is waiting for the answer.

Guessing costs correctness in the worst direction. Reporting an interrupted
call as a failed one invites the model to repeat it, and a repeated call with a
side effect applies that effect twice. Reporting an approval-parked call as
"outcome unknown" invites the opposite mistake: a call the operator never
approved is described to the model as one that may already have run.

So the record is written before the call and carries an explicit lifecycle:

``RESERVED``
    The call is recorded and nothing has been tried yet: no gate has decided,
    no tool has been touched. A resumed run must neither execute it nor say
    anything about its outcome, because there is no outcome to report.
``PENDING_APPROVAL``
    A gate required approval. Dispatch never began. A resumed run must NOT
    execute the call; it waits for the approval to arrive.
``DISPATCHED``
    The tool was invoked and the result is not yet known. A resumed run must
    NOT repeat the call; it reports the honest unknown outcome instead.
``PAUSED_ASK_USER``
    The tool began executing and asked the user a question. A resumed run must
    neither execute nor declare the outcome unknown; the answer settles it.
``PAUSED_EXTERNAL``
    The call was handed to an executor outside the run (another machine), which
    went away before it answered. A resumed run must not declare the outcome
    unknown: the call is handed to that executor again, and the executor, which
    keeps its own journal of the calls it ran, answers it without running it a
    second time.
``SETTLED``
    A result exists. Nothing to recover; repeats are no-ops.

The state is what a resumed run reads. ``replay`` stays alongside it as the
policy that says whether repeating THIS tool is acceptable at all.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Final, Literal
from uuid import uuid4

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_roles import (
    EMPTY_TOOL_ROLE_MAP,
    REPEAT_SAFE_ROLES,
    ToolRoleMap,
)

ReplayPolicy = Literal["never", "safe"]
IntentState = Literal[
    "RESERVED", "PENDING_APPROVAL", "DISPATCHED", "PAUSED_ASK_USER", "PAUSED_EXTERNAL", "SETTLED"
]
IntentOutcome = Literal["pending", "known", "unknown"]

RESERVED: Final[IntentState] = "RESERVED"
PENDING_APPROVAL: Final[IntentState] = "PENDING_APPROVAL"
DISPATCHED: Final[IntentState] = "DISPATCHED"
PAUSED_ASK_USER: Final[IntentState] = "PAUSED_ASK_USER"
PAUSED_EXTERNAL: Final[IntentState] = "PAUSED_EXTERNAL"
SETTLED: Final[IntentState] = "SETTLED"

class IntentPauseMismatch(ValueError):
    """A pause envelope disagrees with the durable intent it claims to pause.

    Neither side is treated as the truth. The two disagree about which call is
    parked, and a resume that picks one of them either runs a call nobody
    approved or answers a question nobody asked.
    """


def _fingerprint(tool_name: str, arguments: Any) -> str:
    """Stable short digest of the call a record stands for."""
    try:
        payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover — default=str covers it
        payload = repr(arguments)
    digest = hashlib.sha256(f"{tool_name}\x00{payload}".encode()).hexdigest()
    return digest[:32]


def fingerprint_arguments(tool_name: str, arguments: Any) -> str:
    """The digest a record carries for the call it stands for.

    Public because a call whose arguments a person corrected while approving it
    is a different call from the one the model proposed, and the record has to
    be able to say so — otherwise the pause check refuses the very call the
    correction was made for.
    """
    return _fingerprint(tool_name, arguments)


@dataclass(slots=True)
class IntentRecord:
    """One durable record of "this run is about to call this tool"."""

    operation_id: str
    tool_name: str
    tool_call_id: str
    reserved_result_ids: list[str]
    replay: ReplayPolicy
    repeat_safe: bool = False
    state: IntentState = DISPATCHED
    outcome: IntentOutcome = "pending"
    result: str | None = None
    arguments_fingerprint: str = ""
    pause_fingerprint: str | None = None
    reported: bool = False
    _key: str = field(default="", repr=False)

    @property
    def idempotency_key(self) -> str:
        """Key under which a repeat of this exact call is the same call.

        Derived from the call id and the arguments, so the same tool invoked
        twice with different arguments is two operations while a re-drive of
        one call is one. Consumed by whatever decides whether repeating a
        side-effecting tool is allowed.
        """
        if self._key:
            return self._key
        return f"{self.tool_call_id}:{self.arguments_fingerprint}"

    @property
    def repeat_is_safe(self) -> bool:
        """Whether re-issuing this call would change anything a second time.

        Settled when the record is written, from the operator's list, and
        carried on the record from then on: the answer has to be the same for
        a run picked up on another pod as it was for the run that wrote it,
        and re-deriving it there would read whatever the list says at that
        later moment instead.
        """
        return self.repeat_safe

    @property
    def outcome_is_unknown(self) -> bool:
        return self.state == SETTLED and self.outcome == "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "reserved_result_ids": list(self.reserved_result_ids),
            "replay": self.replay,
            "repeat_safe": self.repeat_safe,
            "state": self.state,
            "outcome": self.outcome,
            "result": self.result,
            "arguments_fingerprint": self.arguments_fingerprint,
            "pause_fingerprint": self.pause_fingerprint,
            "reported": self.reported,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntentRecord:
        state = data.get("state")
        outcome = data.get("outcome")
        record = cls(
            operation_id=str(data.get("operation_id", "")),
            tool_name=str(data.get("tool_name", "")),
            tool_call_id=str(data.get("tool_call_id", "")),
            reserved_result_ids=[str(item) for item in data.get("reserved_result_ids") or []],
            replay="never" if data.get("replay") == "never" else "safe",
            repeat_safe=bool(data.get("repeat_safe", False)),
            state=state if state in _STATES else DISPATCHED,
            outcome=outcome if outcome in _OUTCOMES else "pending",
            result=data.get("result") if isinstance(data.get("result"), str) else None,
            arguments_fingerprint=str(data.get("arguments_fingerprint", "")),
            reported=bool(data.get("reported", False)),
        )
        pause = data.get("pause_fingerprint")
        record.pause_fingerprint = pause if isinstance(pause, str) else None
        key = data.get("idempotency_key")
        if isinstance(key, str) and key:
            record._key = key
        return record


_STATES: Final[frozenset[str]] = frozenset(
    {RESERVED, PENDING_APPROVAL, DISPATCHED, PAUSED_ASK_USER, PAUSED_EXTERNAL, SETTLED}
)
_OUTCOMES: Final[frozenset[str]] = frozenset({"pending", "known", "unknown"})


def _named_tools(names: str) -> set[str]:
    return {item.strip() for item in names.split(",") if item.strip()}


def replay_policy_for(tool_name: str, rc: LoopConstants) -> ReplayPolicy:
    return "never" if tool_name in _named_tools(rc.intent_never_replay_tools) else "safe"


def repeat_is_safe_for(
    tool_name: str, rc: LoopConstants, roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP
) -> bool:
    """Whether an interrupted call of this tool may simply be re-issued.

    The tools answer it themselves now: a call that only looks at state costs a
    second look and nothing else if it runs twice. An operator's list is still
    read, and can still add a tool whose repeat is harmless for a reason the
    roles cannot express — but a host that declared its roles no longer has to
    keep that list in step with its registry to keep the guarantee.
    """
    if tool_name in _named_tools(rc.intent_repeat_safe_tools):
        return True
    return bool(roles.roles_of(tool_name) & REPEAT_SAFE_ROLES)


def commit_intent(
    *,
    tool_name: str,
    tool_call_id: str,
    rc: LoopConstants,
    arguments: Any = None,
    state: IntentState = DISPATCHED,
    roles: ToolRoleMap = EMPTY_TOOL_ROLE_MAP,
) -> IntentRecord:
    """Build the record written before the call is made."""
    reserved = f"res_{uuid4().hex[:12]}"
    return IntentRecord(
        operation_id=f"op_{uuid4().hex[:12]}",
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        reserved_result_ids=[reserved],
        replay=replay_policy_for(tool_name, rc),
        repeat_safe=repeat_is_safe_for(tool_name, rc, roles),
        state=state,
        arguments_fingerprint=_fingerprint(tool_name, arguments),
    )


def mark_pending_approval(intent: IntentRecord) -> IntentRecord:
    """The call is parked at a gate; dispatch has not begun."""
    if intent.state != SETTLED:
        intent.state = PENDING_APPROVAL
    return intent


def mark_dispatched(intent: IntentRecord) -> IntentRecord:
    """The tool is being invoked right now; the outcome is not yet known."""
    if intent.state != SETTLED:
        intent.state = DISPATCHED
    return intent


def mark_paused_ask_user(
    intent: IntentRecord,
    *,
    pause_payload: Any = None,
) -> IntentRecord:
    """The tool ran far enough to ask the user something."""
    if intent.state != SETTLED:
        intent.state = PAUSED_ASK_USER
        intent.pause_fingerprint = _fingerprint(intent.tool_name, pause_payload)
    return intent


def mark_paused_external(intent: IntentRecord) -> IntentRecord:
    """The executor the call was handed to went away before it answered."""
    if intent.state != SETTLED:
        intent.state = PAUSED_EXTERNAL
    return intent


def settle_intent(intent: IntentRecord, *, result: str) -> IntentRecord:
    intent.state = SETTLED
    intent.outcome = "known"
    intent.result = result
    return intent


def settle_unknown(intent: IntentRecord) -> IntentRecord:
    """Close a record whose call may or may not have happened."""
    intent.state = SETTLED
    intent.outcome = "unknown"
    intent.result = None
    return intent


def unknown_outcome_text(intent: IntentRecord, rc: LoopConstants) -> str:
    """What the model is told about a call whose outcome nobody recorded."""
    if intent.repeat_is_safe:
        return rc.tool_result_unknown_outcome_repeatable_placeholder
    return rc.tool_result_unknown_outcome_placeholder


def assert_pause_matches(
    intent: IntentRecord,
    *,
    tool_call_id: str,
    tool_name: str,
    arguments: Any = None,
) -> None:
    """Refuse a pause envelope that does not describe the recorded intent.

    Raises :class:`IntentPauseMismatch`. The refusal is the point: with the
    durable record and the pause envelope disagreeing, no third party can say
    which of them describes the call the operator is about to approve.
    """
    if intent.tool_call_id != tool_call_id:
        raise IntentPauseMismatch(
            "paused tool call is not the recorded intent: intent holds "
            f"{intent.tool_call_id!r}, pause holds {tool_call_id!r}"
        )
    if intent.tool_name != tool_name:
        raise IntentPauseMismatch(
            f"paused tool call {tool_call_id!r} names tool {tool_name!r}, "
            f"the recorded intent names {intent.tool_name!r}"
        )
    if arguments is not None and intent.arguments_fingerprint:
        seen = _fingerprint(tool_name, arguments)
        if seen != intent.arguments_fingerprint:
            raise IntentPauseMismatch(
                f"paused tool call {tool_call_id!r} carries arguments that differ "
                "from the recorded intent"
            )


def find_intent(intents: list[IntentRecord], tool_call_id: str) -> IntentRecord | None:
    for item in intents:
        if item.tool_call_id == tool_call_id:
            return item
    return None


def orphaned_intents(
    intents: list[IntentRecord],
    *,
    resolved_tool_call_ids: set[str],
) -> list[IntentRecord]:
    """Records left ``DISPATCHED`` with no result anywhere in history.

    Every other state is deliberately excluded. ``PENDING_APPROVAL`` and
    ``PAUSED_ASK_USER`` are waiting for something that has not arrived, and
    ``RESERVED`` was recorded before anything was tried — none of the three is
    an unknown outcome, and describing one as such would tell the model a call
    that never ran may already have taken effect.
    """
    return [
        item
        for item in intents
        if item.state == DISPATCHED and item.tool_call_id not in resolved_tool_call_ids
    ]


def should_dispatch(intent: IntentRecord | None) -> bool:
    """Whether a call carrying this record may be handed to a tool."""
    if intent is None:
        return True
    return intent.state != SETTLED


def refuse_intent_when_disabled(enabled: bool) -> None:
    if not enabled:
        raise ValueError("intent_settlement_disabled")


__all__ = [
    "DISPATCHED",
    "PAUSED_ASK_USER",
    "PAUSED_EXTERNAL",
    "PENDING_APPROVAL",
    "RESERVED",
    "SETTLED",
    "IntentOutcome",
    "IntentPauseMismatch",
    "IntentRecord",
    "IntentState",
    "ReplayPolicy",
    "assert_pause_matches",
    "commit_intent",
    "find_intent",
    "fingerprint_arguments",
    "mark_dispatched",
    "mark_paused_ask_user",
    "mark_paused_external",
    "mark_pending_approval",
    "orphaned_intents",
    "refuse_intent_when_disabled",
    "repeat_is_safe_for",
    "replay_policy_for",
    "settle_intent",
    "settle_unknown",
    "should_dispatch",
    "unknown_outcome_text",
]
