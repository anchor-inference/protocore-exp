"""A tool that returns images, seen from outside: the request, the events, the history.

Computer use takes a screenshot after every action and the acting model reads
it, so a tool result has to be able to carry pixels to the model that made the
call. These scenarios drive ``QueryEngine.run()`` with such a tool and assert
on what the provider was handed, what a reader of the events was told, and
what a resumed process gets back.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    TOOL_IMAGES_ELIDED_METADATA_KEY,
    ToolResult,
    ToolResultBlock,
    ToolResultImage,
)
from protocore.runtime.events import EventType
from protocore.tests_support.adapters import InMemoryBlobStore

from .conftest import ScenarioFactory, ScriptedTool, default_rc

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 8


@dataclass
class ScreenshotTool(ScriptedTool):
    """Returns the screen as an inline image, a new one on every call."""

    tool_name: str = "Screenshot"
    shots: list[bytes] = field(default_factory=list)

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        await super().invoke(context, arguments)
        shot = PNG + str(len(self.invocations)).encode()
        self.shots.append(shot)
        return ToolResult(
            tool_call_id="",
            content=f"screenshot {len(self.invocations)}",
            images=(ToolResultImage(data=shot, width=1280, height=800, metadata={"display": 0}),),
        )


class RefusingBlobStore(InMemoryBlobStore):
    """A store that is down: every write fails."""

    async def put(self, tenant_id: str, content: bytes, **kwargs: Any) -> Any:
        raise OSError("store unavailable")


def _request_images(run: Any, index: int) -> list[ToolResultImage]:
    return [
        image
        for message in run.requests[index].messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
        for image in block.images
    ]


async def test_a_screenshot_reaches_the_model_by_reference_and_the_events_carry_no_bytes(
    scenario: ScenarioFactory,
) -> None:
    tool = ScreenshotTool()
    run = scenario(tools=[tool])
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Screenshot", tool_input={})
    run.llm.queue_response(text="I see the screen")

    await run.run()

    [result] = run.tool_results()
    [image] = result.images
    assert image.data is None
    assert image.blob_ref is not None
    assert (image.width, image.height, image.metadata) == (1280, 800, {"display": 0})
    assert await run.blobs.get("tenant-scenario", image.blob_ref) == tool.shots[0]
    # The model's next request is handed the image with its result.
    assert _request_images(run, 1) == [image]
    # The reader is told about it by reference; no event carries the bytes.
    [event] = run.events_of(EventType.TOOL_RESULT)
    assert event.payload["images"] == [image.event_entry()]
    assert "data" not in event.payload["images"][0]
    serialised = json.dumps([evt.payload for evt in run.events], default=str)
    assert "pixels" not in serialised


async def test_only_the_newest_images_ride_along_and_they_leave_in_batches(
    scenario: ScenarioFactory,
) -> None:
    """Six screenshots, three kept, two evicted at a time. After the fifth the
    view is two over the limit, one whole batch, and the two oldest leave;
    after the sixth it is three over, still one batch, so the same two stay
    out and nothing before the newest result changes."""
    tool = ScreenshotTool()
    run = scenario(tools=[tool], rc=default_rc(model_context_window=200_000))
    for step in range(6):
        run.llm.queue_tool_call_response(tool_call_id=f"call-{step}", tool_name="Screenshot", tool_input={})
    run.llm.queue_response(text="done")

    await run.run()

    counts = [len(_request_images(run, index)) for index in range(len(run.requests))]
    # Request i follows i screenshots: 0,1,2,3 kept whole, then 4 (one over,
    # no batch yet), then 5 → 3 (a batch of two left), then 6 → 4.
    assert counts == [0, 1, 2, 3, 4, 3, 4]
    last = [
        block
        for message in run.requests[-1].messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert [block.metadata.get(TOOL_IMAGES_ELIDED_METADATA_KEY) for block in last] == [1, 1, None, None, None, None]
    assert "no longer shown" in last[0].content
    assert last[0].content.startswith("screenshot 1")
    # The cut between the fifth and sixth request is the same cut: the
    # results before the newest one are byte for byte what they were.
    fifth = [
        block
        for message in run.requests[5].messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert last[: len(fifth)] == fifth
    # The history keeps every image.
    assert [len(block.images) for block in run.tool_results()] == [1] * 6


async def test_an_image_the_store_refuses_stays_inline_and_survives_a_resume(
    scenario: ScenarioFactory,
) -> None:
    tool = ScreenshotTool()
    first = scenario(tools=[tool], blob_store=RefusingBlobStore())
    first.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Screenshot", tool_input={})
    first.llm.queue_response(text="seen")
    await first.run()

    [event] = first.events_of(EventType.TOOL_RESULT)
    assert event.payload["images"][0]["inline"] is True
    assert event.payload["images"][0]["byte_size"] == len(tool.shots[0])

    stored = json.loads(json.dumps(first.engine.snapshot()))
    second = scenario(tools=[ScreenshotTool()])
    await second.engine.resume_from_snapshot(stored)

    [result] = second.tool_results()
    assert result.images[0].data == tool.shots[0]
    assert result.images[0].width == 1280


async def test_a_text_tool_is_unchanged(scenario: ScenarioFactory) -> None:
    run = scenario(tools=[ScriptedTool(tool_name="Note", content="noted")])
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Note", tool_input={})
    run.llm.queue_response(text="done")

    await run.run()

    [event] = run.events_of(EventType.TOOL_RESULT)
    assert "images" not in event.payload
    [result] = run.tool_results()
    assert "images" not in result.model_dump(mode="json")
