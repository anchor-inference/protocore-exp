"""``EventType`` StrEnum — 18-type taxonomy for per-turn streaming events.

Names mirror v1 :class:`protocore.events.EventName` where possible but are
strict to the turn-streaming subset — :class:`protocore.events.EventName`
remains for in-process EventBus signalling.
"""
from __future__ import annotations

from enum import StrEnum

# The live stream states a block's visibility on every ``content_block_start`` /
# ``content_block_stop``, and a durable :class:`~protocore.contracts.types.TextBlock`
# can carry the same judgement as its own property — so the vocabulary belongs to
# neither surface and is defined once, in contracts. Re-exported here because the
# stream event types are where a reader of the wire format looks for it.
from protocore.contracts.types import BlockVisibility

# The lifecycle of a host's tool transport is named in both event vocabularies,
# and the key that says WHICH transport is the same key in both. Imported
# rather than restated so the two cannot drift.
from protocore.events import TOOL_TRANSPORT_EVENT_FIELD


class EventType(StrEnum):
    """Per-turn streaming event taxonomy.

    Wire compatible with Anthropic Messages API streaming events plus
    Protocore-specific extensions. Each value is the string surfaced to SSE
    clients as ``event:`` line.
    """

    # ----- Anthropic-aligned (mandatory) -----
    MESSAGE_START = "message_start"
    MESSAGE_STOP = "message_stop"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_INPUT_DELTA = "tool_use_input_delta"
    TOOL_USE_STOP = "tool_use_stop"
    TOOL_RESULT = "tool_result"
    TOOL_SURFACE_ADVERTISED = "tool_surface_advertised"
    # A discovery tool (ToolSearch) loaded tools onto the surface; they are
    # advertised from the next request on, appended in discovery order.
    TOOL_DISCOVERED = "tool_discovered"
    # The model called a registered tool the request did not advertise. The
    # policy admitted it, so it ran; it is loaded for the rest of the run.
    TOOL_UNADVERTISED_CALL = "tool_unadvertised_call"
    # Tools of a group were loaded: payload ``{group, via, tools}``, ``via``
    # one of search, select, group, direct_call or seed.
    TOOL_GROUP_LOADED = "tool_group_loaded"
    ERROR = "error"

    # ----- Protocore extensions -----
    # Whatever the host puts between a tool call and the place it runs. Its
    # payload names the transport under
    # :data:`~protocore.events.TOOL_TRANSPORT_EVENT_FIELD`.
    TOOL_TRANSPORT_STARTING = "tool_transport_starting"
    TOOL_TRANSPORT_READY = "tool_transport_ready"
    SUBAGENT_SPAWN = "subagent_spawn"
    SUBAGENT_PROGRESS = "subagent_progress"
    SUBAGENT_COMPLETE = "subagent_complete"
    HOOK_FIRED = "hook_fired"
    TOOL_CALL_PENDING = "tool_call_pending"
    # A wait was recorded, typed and with an identity. Carries the interrupt
    # that was just parked and the whole open set beside it, so a host that
    # renders one card per wait can draw a parked batch in one pass and quote
    # the ids back when the answers arrive.
    INTERRUPT_PARKED = "interrupt_parked"
    STATE_CHANGED = "state_changed"
    # Deep-mode SGR plan step. Carries the
    # structured ordered plan + the single next tool the model recorded
    # BEFORE acting. Distinct from native CoT, which continues to flow as
    # ``CONTENT_BLOCK_DELTA`` thinking deltas (``ProviderDeltaKind.thinking``).
    REASONING_STEP = "reasoning_step"

    # ----- Loop lifecycle -----
    RUN_STARTED = "run_started"
    HEARTBEAT = "heartbeat"
    COMPACTION_STARTED = "compaction_started"
    COMPACTION_COMPLETED = "compaction_completed"
    RUN_SETTLED = "run_settled"
    LOOP_GUARD_FIRED = "loop_guard_fired"
    TOOL_RESULT_EVICTED = "tool_result_evicted"
    STEER_QUEUED = "steer_queued"
    FOLLOW_UP_QUEUED = "follow_up_queued"
    QUEUE_UPDATE = "queue_update"
    MODEL_CHANGED = "model_changed"
    THINKING_CHANGED = "thinking_changed"
    BACKGROUND_TASK_UPDATED = "background_task_updated"
    BACKGROUND_WAKE = "background_wake"
    PROFILE_CHANGED = "profile_changed"
    COMPACT_CHECKPOINT = "compact_checkpoint"
    RULES_ACTIVATED = "rules_activated"
    PATH_DENIED = "path_denied"
    INTENT_COMMITTED = "intent_committed"
    USAGE_COMMITTED = "usage_committed"
    SESSION_FORKED = "session_forked"
    LANE_LOCKED = "lane_locked"
    RECOVERY_MARKED = "recovery_marked"

    # ----- Candidate verification lifecycle -----
    # These events describe stage progress only. Projection of content to a
    # public delivery channel remains the responsibility of the caller.
    CANDIDATE_READY = "candidate_ready"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_REPORTED = "verification_reported"
    REPAIR_REQUESTED = "repair_requested"
    RELEASE_DECIDED = "release_decided"
    # One atomic projection of an immutable candidate after a release/warn
    # decision.  It is distinct from the incremental content-block frames that
    # gated delivery intentionally withholds.
    CANDIDATE_RELEASED = "candidate_released"


__all__ = ["TOOL_TRANSPORT_EVENT_FIELD", "BlockVisibility", "EventType"]
