"""The image a tool hands back: what it may hold, how it is stored, how it is named."""
from __future__ import annotations

import base64
import json
import math

import pytest

from protocore.constants import MAX_INLINE_TOOL_IMAGE_BYTES, MAX_TOOL_RESULT_IMAGES
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResult,
    ToolResultBlock,
    ToolResultImage,
    message_tool_images,
)

PNG = b"\x89PNG\r\n\x1a\n\x00\xff"


def test_inline_bytes_survive_a_json_round_trip_as_standard_base64() -> None:
    """A snapshot is JSON, read back with ``model_validate`` on the decoded
    mapping; pydantic's default would write the bytes as UTF-8, which a PNG is
    not, and read text back as the bytes themselves."""
    image = ToolResultImage(data=PNG, width=10, height=20, metadata={"scale": 0.5})

    dumped = image.model_dump(mode="json")
    assert dumped["data"] == base64.b64encode(PNG).decode("ascii")
    assert ToolResultImage.model_validate(dumped) == image
    assert ToolResultImage.model_validate_json(json.dumps(dumped)) == image
    assert image.model_dump()["data"] == PNG


def test_unset_fields_are_left_out_of_the_dump() -> None:
    assert ToolResultImage(blob_ref="ref-1").model_dump(mode="json") == {
        "blob_ref": "ref-1",
        "mime_type": "image/png",
    }


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"blob_ref": "ref", "data": PNG},
        {"blob_ref": ""},
        {"data": b""},
        {"data": "not base64!"},
        {"blob_ref": "ref", "mime_type": "text/plain"},
        {"blob_ref": "ref", "width": 0},
        {"blob_ref": "ref", "metadata": {"scale": math.inf}},
    ],
)
def test_an_image_that_cannot_be_read_back_is_refused(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ToolResultImage.model_validate(fields)


def test_inline_bytes_are_bounded() -> None:
    with pytest.raises(ValueError, match="store it"):
        ToolResultImage(data=b"0" * (MAX_INLINE_TOOL_IMAGE_BYTES + 1))


def test_a_result_carries_a_bounded_number_of_images() -> None:
    images = [ToolResultImage(blob_ref=f"ref-{n}") for n in range(MAX_TOOL_RESULT_IMAGES + 1)]
    with pytest.raises(ValueError, match="images"):
        ToolResult(tool_call_id="c", content="x", images=images)
    with pytest.raises(ValueError, match="images"):
        ToolResultBlock(tool_call_id="c", content="x", images=images)


def test_an_event_names_an_image_and_never_carries_its_bytes() -> None:
    stored = ToolResultImage(blob_ref="ref-1", width=4, height=3, metadata={"display": 1})
    inline = ToolResultImage(data=PNG, mime_type="image/jpeg")

    assert stored.event_entry() == {
        "mime_type": "image/png",
        "blob_ref": "ref-1",
        "width": 4,
        "height": 3,
        "metadata": {"display": 1},
    }
    assert inline.event_entry() == {"mime_type": "image/jpeg", "inline": True, "byte_size": len(PNG)}


def test_a_result_without_images_is_stored_exactly_as_before() -> None:
    """The key is left out when empty, so every result written before the
    field existed — and every request digest over one — keeps its bytes."""
    block = ToolResultBlock(tool_call_id="c", content="x")

    assert "images" not in block.model_dump(mode="json")
    with_images = ToolResultBlock(tool_call_id="c", content="x", images=[ToolResultImage(data=PNG)])
    dumped = with_images.model_dump(mode="json")
    assert dumped["images"] == [{"data": base64.b64encode(PNG).decode("ascii"), "mime_type": "image/png"}]
    assert ToolResultBlock.model_validate(dumped) == with_images


def test_message_tool_images_reads_every_result_in_order() -> None:
    first = ToolResultImage(blob_ref="a")
    second = ToolResultImage(blob_ref="b")
    message = Message(
        role=MessageRole.tool,
        content_blocks=[
            ToolResultBlock(tool_call_id="c1", content="x", images=[first]),
            ToolResultBlock(tool_call_id="c2", content="y"),
            ToolResultBlock(tool_call_id="c3", content="z", images=[second]),
        ],
    )

    assert message_tool_images(message) == (first, second)
    assert message_tool_images(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ) == ()
