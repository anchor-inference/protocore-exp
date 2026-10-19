"""Queued steer / follow-up items keep their attachments when they are placed.

A screenshot sent while a run was busy used to reach the model as its caption
alone: the placed message joined only the items' text and left their files
behind, so a host had to put the images back itself before the next request.
"""
from __future__ import annotations

import json

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    IMAGE_REFS_METADATA_KEY,
    QUEUED_ITEMS_METADATA_KEY,
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    message_image_refs,
)
from protocore.runtime.context.compaction import estimate_message_tokens
from protocore.runtime.events import EventType
from protocore.runtime.live_control import new_queued_prompt, placed_as_user_message
from protocore.runtime.query import _query as query
from protocore.tests_support.adapters import InMemoryLLMProvider


def _enabled() -> LoopConstants:
    return LoopConstants(model_context_window=4_096, steer_follow_up_enabled=True)


def _image(ref: str, mime: str = "image/png") -> dict[str, object]:
    return {"blob_ref": ref, "mime_type": mime, "name": f"{ref}.png"}


def test_placed_message_carries_the_image_of_a_queued_item() -> None:
    item = new_queued_prompt("steer", "look at this", attachments=[_image("blob-1", "image/jpeg")])

    message = placed_as_user_message([item])

    assert message is not None
    assert message.text == "look at this"
    assert message_image_refs(message) == (ImageRefBlock(blob_ref="blob-1", mime_type="image/jpeg"),)
    # The user turn holds one block, so the image rides beside it in metadata.
    assert len(message.content_blocks) == 1


def test_merged_items_keep_every_attachment_in_order() -> None:
    log = {"name": "build.log", "blob_ref": "blob-log", "mime_type": "text/plain"}
    first = new_queued_prompt("steer", "first", attachments=[_image("a"), log], item_id="q1")
    second = new_queued_prompt("steer", "second", attachments=[_image("b"), _image("c")], item_id="q2")

    message = placed_as_user_message([first, second])

    assert message is not None
    assert message.text == "first\n\nsecond"
    assert [block.blob_ref for block in message_image_refs(message)] == ["a", "b", "c"]
    # A file that is not an image is not sent to the model as one, but the host
    # still learns it was part of the placed item.
    assert message.metadata[QUEUED_ITEMS_METADATA_KEY] == [
        {"id": "q1", "kind": "steer", "attachments": [_image("a"), log]},
        {"id": "q2", "kind": "steer", "attachments": [_image("b"), _image("c")]},
    ]


def test_metadata_names_each_placed_item_and_its_origin() -> None:
    steer = new_queued_prompt("steer", "now", item_id="s1")
    follow = new_queued_prompt("follow_up", "later", item_id="f1")

    message = placed_as_user_message([steer, follow])

    assert message is not None
    assert message.metadata[QUEUED_ITEMS_METADATA_KEY] == [
        {"id": "s1", "kind": "steer", "attachments": []},
        {"id": "f1", "kind": "follow_up", "attachments": []},
    ]
    # Nothing to show: no image key at all, so an adapter has nothing to read.
    assert IMAGE_REFS_METADATA_KEY not in message.metadata


def test_an_item_with_only_an_image_is_not_dropped_as_empty() -> None:
    item = new_queued_prompt("steer", "  ", attachments=[_image("only")])

    message = placed_as_user_message([item])

    assert message is not None
    assert message.content_blocks == ()
    assert [block.blob_ref for block in message_image_refs(message)] == ["only"]


def test_an_item_with_neither_text_nor_an_image_still_places_nothing() -> None:
    assert placed_as_user_message([new_queued_prompt("steer", " \n ")]) is None
    log = {"name": "build.log", "blob_ref": "blob-log", "mime_type": "text/plain"}
    # A file the model cannot be shown would make an empty turn.
    assert placed_as_user_message([new_queued_prompt("steer", "", attachments=[log])]) is None


def test_message_image_refs_reads_blocks_before_metadata_and_skips_malformed_entries() -> None:
    message = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="x"), ImageRefBlock(blob_ref="in-block")],
        metadata={IMAGE_REFS_METADATA_KEY: [{"blob_ref": "in-meta"}, "junk", {"mime_type": "image/png"}]},
    )

    assert message_image_refs(message) == (
        ImageRefBlock(blob_ref="in-block"),
        ImageRefBlock(blob_ref="in-meta"),
    )


def test_a_queued_image_is_counted_in_the_prompt_estimate() -> None:
    rc = _enabled()
    caption = placed_as_user_message([new_queued_prompt("steer", "caption")])
    with_image = placed_as_user_message(
        [new_queued_prompt("steer", "caption", attachments=[_image("blob-1")])]
    )
    assert caption is not None
    assert with_image is not None

    # An image the estimate never saw would let a history overrun the window it
    # was sized against.
    assert estimate_message_tokens(with_image, rc) == (
        estimate_message_tokens(caption, rc) + rc.token_count_image_tokens
    )


@pytest.mark.asyncio
async def test_steer_with_an_image_sent_while_a_tool_runs_reaches_the_next_request(
    engine_factory, in_memory_runtime: dict[str, object]
) -> None:
    llm = in_memory_runtime["llm"]
    assert isinstance(llm, InMemoryLLMProvider)
    llm.queue_tool_call_response(tool_call_id="once", tool_name="Read", tool_input={"path": "x"})
    llm.queue_response(text="seen it")
    engine = engine_factory(rc=_enabled())
    engine.history.append(Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")]))
    reloads = {"n": 0}

    async def reload(eng) -> None:
        # The first reload is before the first request; the second follows the
        # tool, which is when an operator's mid-run screenshot shows up.
        reloads["n"] += 1
        if reloads["n"] == 2:
            eng._steer_queue = [
                new_queued_prompt(
                    "steer", "this screen", attachments=[_image("shot-1")], item_id="q-shot"
                ).to_dict()
            ]

    engine.reload_live_control = reload
    events = [event async for event in query(engine)]

    placed = next(
        event for event in events
        if event.type is EventType.QUEUE_UPDATE and event.payload.get("kind") == "steer"
    )
    assert placed.payload["placed"] == ["q-shot"]
    assert len(llm.calls) == 2
    sent = [m for m in llm.calls[1].messages if m.role is MessageRole.user and "this screen" in m.text]
    assert len(sent) == 1
    assert message_image_refs(sent[0]) == (ImageRefBlock(blob_ref="shot-1"),)
    assert sent[0].metadata[QUEUED_ITEMS_METADATA_KEY][0]["id"] == "q-shot"


@pytest.mark.asyncio
async def test_snapshot_round_trip_keeps_queued_and_placed_attachments(engine_factory) -> None:
    engine = engine_factory(rc=_enabled())
    pending = new_queued_prompt("follow_up", "later", attachments=[_image("wait-1")], item_id="q-wait")
    engine._follow_up_queue = [pending.to_dict()]
    placed = placed_as_user_message(
        [new_queued_prompt("steer", "now", attachments=[_image("seen-1")], item_id="q-now")]
    )
    assert placed is not None
    engine.history.append(placed)

    # Through JSON, as a host stores it, not just through a dict copy.
    snapshot = json.loads(json.dumps(engine.snapshot(), default=str))
    other = engine_factory(rc=_enabled())
    await other.resume_from_snapshot(snapshot)

    assert other._follow_up_queue == [pending.to_dict()]
    restored = next(m for m in other.history if m.text == "now")
    assert message_image_refs(restored) == (ImageRefBlock(blob_ref="seen-1"),)
    assert restored.metadata[QUEUED_ITEMS_METADATA_KEY][0]["id"] == "q-now"
