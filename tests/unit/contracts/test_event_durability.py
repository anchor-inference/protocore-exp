"""What an event stream has to keep, and what it may keep once.

Streaming fragments are wanted by whoever is watching and by nobody after:
the text they carry is in the transcript by the time the message stops, and a
store that kept them wrote one row per token. The tool-surface advertisement
repeats the same large list run after run. The contract says both in one
place so every store does not have to rediscover them.
"""
from __future__ import annotations

import pytest

from protocore.contracts.events import (
    BY_REFERENCE_EVENTS,
    LIVE_ONLY_EVENT_NAMES,
    ByReferenceField,
    EventDurability,
    PayloadReference,
    event_durability,
    payload_by_reference,
)
from protocore.contracts.types import Event
from protocore.runtime.events.types import EventType


def test_streaming_fragments_are_live_only() -> None:
    assert event_durability(EventType.CONTENT_BLOCK_DELTA.value) is EventDurability.live_only
    assert event_durability(EventType.TOOL_USE_INPUT_DELTA.value) is EventDurability.live_only


@pytest.mark.parametrize(
    "name",
    [
        EventType.MESSAGE_START.value,
        EventType.CONTENT_BLOCK_STOP.value,
        EventType.TOOL_RESULT.value,
        EventType.COMPACTION_STARTED.value,
        "a_host_event",
    ],
)
def test_everything_else_is_durable_unless_said_otherwise(name: str) -> None:
    assert event_durability(name) is EventDurability.durable


def test_a_host_can_name_its_own_live_only_events() -> None:
    names = LIVE_ONLY_EVENT_NAMES | {"progress_tick"}
    assert event_durability("progress_tick", live_only=names) is EventDurability.live_only
    assert event_durability("progress_tick") is EventDurability.durable


def test_every_live_only_name_is_one_the_runtime_emits() -> None:
    emitted = {member.value for member in EventType}
    assert LIVE_ONLY_EVENT_NAMES <= emitted
    assert set(BY_REFERENCE_EVENTS) <= emitted


def _advertisement(*, described: bool | None = True, digest: str | None = "d1") -> Event:
    payload: dict[str, object] = {
        "turn_id": "t1",
        "tools": [{"name": "Read", "description": "Read a file"}, {"name": "Write"}],
    }
    if digest is not None:
        payload["tool_surface_digest"] = digest
    if described is not None:
        payload["tool_surface_described"] = described
    return Event(run_id="run-1", name=EventType.TOOL_SURFACE_ADVERTISED.value, payload=payload)


def _split(event: Event) -> PayloadReference | None:
    return payload_by_reference(event.name, event.payload)


def test_the_surface_is_kept_once_under_the_digest_the_runtime_named() -> None:
    event = _advertisement()
    reference = payload_by_reference(event.name, event.payload)
    assert isinstance(reference, PayloadReference)
    assert reference.field == "tools"
    assert reference.digest == "d1"
    assert reference.value == event.payload["tools"]
    assert reference.complete
    assert reference.payload["tools"] == {"digest": "d1", "count": 2}
    assert reference.payload["turn_id"] == "t1"
    # The event itself is untouched: live subscribers still get the full value.
    assert isinstance(event.payload["tools"], list)


def test_an_advertisement_without_descriptions_must_not_replace_a_full_copy() -> None:
    reference = _split(_advertisement(described=False))
    assert reference is not None
    assert not reference.complete
    assert reference.digest == "d1"


def test_without_a_named_digest_one_is_taken_from_the_value() -> None:
    first = _split(_advertisement(digest=None, described=None))
    second = _split(_advertisement(digest=None, described=None))
    assert first is not None and second is not None
    assert first.digest == second.digest
    assert len(first.digest) == 64
    # A runtime that says nothing about descriptions sends them every time.
    assert first.complete


def test_an_event_with_nothing_to_reference_is_stored_as_it_is() -> None:
    assert payload_by_reference("message_start", {"x": 1}) is None
    assert payload_by_reference(EventType.TOOL_SURFACE_ADVERTISED.value, {"turn_id": "t"}) is None


def test_a_host_can_name_its_own_by_reference_events() -> None:
    fields = {"config_published": ByReferenceField(field="config")}
    reference = payload_by_reference("config_published", {"config": {"a": 1}, "rev": 3}, fields=fields)
    assert reference is not None
    assert reference.complete
    assert reference.payload == {"config": {"digest": reference.digest, "count": 1}, "rev": 3}
    scalar = payload_by_reference("config_published", {"config": "x" * 10}, fields=fields)
    assert scalar is not None and scalar.payload["config"]["count"] == 1
