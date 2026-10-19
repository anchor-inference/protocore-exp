"""IEventStream Protocol — cross-pod event fanout.

Reference shape: a Redis stream driven with ``XADD``/``XREAD`` and a
``MAXLEN ~`` trim.

Distinct from in-process :class:`protocore.events.EventBus`:
    - ``EventBus``: in-process typed pub/sub for sibling-handler signalling
    - ``IEventStream``: durable cross-pod stream (SSE reconnect/replay)

Not every event is worth keeping, and not every payload is worth keeping more
than once. :func:`event_durability` says which events a stream may fan out and
drop, and :func:`payload_by_reference` which payload members a store may keep
once under a digest. Both are hints a store is free to ignore: a stream that
keeps everything verbatim is still correct, only larger.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from protocore.contracts.types import Event


class EventStreamError(Exception):
    """Base for event-stream domain errors."""


class EventDurability(StrEnum):
    """Whether a stream has to keep an event for replay."""

    durable = "durable"
    """Kept: a subscriber that reconnects is replayed it."""

    live_only = "live_only"
    """Fanned out to whoever is subscribed, then free to drop. What it carried
    reaches durable state another way, so a replay without it is complete."""


#: Streaming fragments. The text and the tool input they carry are in the
#: message by the time ``content_block_stop`` / ``tool_use_stop`` arrives, and
#: in the session's history after that; kept one row per token, they were the
#: larger part of a store's event log, for nobody who was not already watching.
LIVE_ONLY_EVENT_NAMES: frozenset[str] = frozenset({"content_block_delta", "tool_use_input_delta"})


def event_durability(
    name: str, *, live_only: frozenset[str] = LIVE_ONLY_EVENT_NAMES
) -> EventDurability:
    """Whether the event called ``name`` must be kept for replay.

    Everything is durable unless named: an unknown event, a host's own, is
    kept, because dropping one a reader needed is a silent loss and keeping
    one nobody needed is only space. A host with live-only events of its own
    passes ``LIVE_ONLY_EVENT_NAMES | {...}`` as ``live_only``.
    """
    return EventDurability.live_only if name in live_only else EventDurability.durable


#: Where the runtime names the tool surface an advertisement describes, and
#: whether this advertisement carries the descriptions. Spelled here so a store
#: and the runtime read and write the same keys.
TOOL_SURFACE_DIGEST_FIELD: str = "tool_surface_digest"
TOOL_SURFACE_DESCRIBED_FIELD: str = "tool_surface_described"


@dataclass(frozen=True)
class ByReferenceField:
    """One payload member a store may keep once, under a digest."""

    field: str
    """The member whose value repeats across events."""

    digest_field: str | None = None
    """A member in which the emitter names the value's digest itself. The
    emitter's name wins over one computed here, so a complete and an abridged
    copy of one value land on one key."""

    complete_field: str | None = None
    """A boolean member that is ``False`` when this event carries an abridged
    copy of the value. Absent means complete."""


#: Events whose payload repeats one large value run after run. The tool surface
#: is tens of kilobytes and the same on every run of a deployment; a store that
#: keeps it once and names it from each row keeps one copy instead of hundreds.
BY_REFERENCE_EVENTS: Mapping[str, ByReferenceField] = MappingProxyType(
    {
        "tool_surface_advertised": ByReferenceField(
            field="tools",
            digest_field=TOOL_SURFACE_DIGEST_FIELD,
            complete_field=TOOL_SURFACE_DESCRIBED_FIELD,
        ),
    }
)


@dataclass(frozen=True)
class PayloadReference:
    """An event's payload split into what a row keeps and what is kept once."""

    payload: dict[str, Any]
    """The payload to store with the event: the referenced member replaced by
    ``{"digest": ..., "count": ...}``, everything else as it was."""

    field: str
    """The member that was replaced."""

    digest: str
    """The key the value is kept under."""

    value: Any
    """The value to keep under :attr:`digest`."""

    complete: bool
    """Whether :attr:`value` is the full copy. A store writes a complete value
    over whatever it holds for the digest, and an abridged one only where it
    holds nothing: an abridged copy must never replace the full one."""


def payload_by_reference(
    name: str,
    payload: Mapping[str, Any],
    *,
    fields: Mapping[str, ByReferenceField] = BY_REFERENCE_EVENTS,
) -> PayloadReference | None:
    """Split the payload of an event called ``name`` for a store that keeps
    repeated values once.

    Takes the name and payload rather than an envelope, so it serves an
    :class:`~protocore.contracts.types.Event` and a runtime turn event alike.
    ``None`` when there is nothing to reference (not a by-reference event, or
    the member is absent), in which case the payload is stored as it is. The
    payload is not modified: live subscribers still receive the value. Without
    an emitter-named digest, the digest is the SHA-256 of the value's
    canonical JSON.
    """
    spec = fields.get(name)
    if spec is None:
        return None
    value = payload.get(spec.field)
    if value is None:
        return None
    named = payload.get(spec.digest_field) if spec.digest_field else None
    if named:
        digest = str(named)
    else:
        body = json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    complete = (
        bool(payload.get(spec.complete_field, True)) if spec.complete_field else True
    )
    count = len(value) if isinstance(value, (list, dict)) else 1
    return PayloadReference(
        payload={**payload, spec.field: {"digest": digest, "count": count}},
        field=spec.field,
        digest=digest,
        value=value,
        complete=complete,
    )


@runtime_checkable
class IEventStream(Protocol):
    """Adapter Protocol over a per-run durable event stream."""

    async def emit(self, event: Event) -> None:
        """Append event to the per-run stream.

        Durability: an event :func:`event_durability` calls ``live_only`` must
        reach current subscribers and need not be kept for replay; every other
        event must be kept. A stream may store a by-reference event's payload
        as :func:`payload_by_reference` splits it, provided a reader can get the
        value back by its digest. Live subscribers are handed ``event`` as
        emitted either way.
        """
        ...

    def subscribe(
        self,
        run_id: str,
        tenant_id: str,
        *,
        from_event_id: str | None = None,
    ) -> AsyncIterator[Event]:
        """Replay (if cursor) then tail. Yields until client disconnects.

        Implementations are async generators; declared here without
        ``async def`` so the Protocol typechecks against async iterators.
        """
        ...

    async def trim(self, run_id: str, tenant_id: str, *, max_len: int) -> None:
        """Apply ``MAXLEN ~`` trim. Called by GC on terminal runs."""
        ...


__all__ = [
    "BY_REFERENCE_EVENTS",
    "LIVE_ONLY_EVENT_NAMES",
    "TOOL_SURFACE_DESCRIBED_FIELD",
    "TOOL_SURFACE_DIGEST_FIELD",
    "ByReferenceField",
    "EventDurability",
    "EventStreamError",
    "IEventStream",
    "PayloadReference",
    "event_durability",
    "payload_by_reference",
]
