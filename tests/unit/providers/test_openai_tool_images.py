"""Images a tool returned, rendered into chat-completions JSON and checked on the JSON."""
from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

import pytest

from protocore.conformance import VisionCapabilityConformance
from protocore.contracts.llm import CacheBreakpoint, IVisionCapability, VisionSupport, vision_support_of
from protocore.contracts.types import (
    IMAGE_REFS_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolResultImage,
    ToolUseBlock,
)
from protocore.providers.openai_compat import ImageLimits, ProviderCapabilities, messages_to_wire
from protocore.providers.openai_compat.wire import TOOL_IMAGES_CARRIER_HEADER, apply_cache_control

from ._openai_support import FakeServer, provider

TAKES_IMAGES = ProviderCapabilities(images=ImageLimits())


class _Loader:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    async def __call__(self, ref: str) -> tuple[bytes, str]:
        self.loaded.append(ref)
        return b"\x89PNG" + ref.encode(), "image/png"


def _url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _calls(*ids: str) -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[ToolUseBlock(tool_call_id=call_id, name="Screenshot", arguments_json="{}") for call_id in ids],
    )


def _result(call_id: str, *images: ToolResultImage, content: str = "done") -> Message:
    return Message(
        role=MessageRole.tool,
        content_blocks=[ToolResultBlock(tool_call_id=call_id, content=content, images=images)],
    )


def _user(text: str, **kwargs: Any) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)], **kwargs)


async def _wire(messages: list[Message], caps: ProviderCapabilities, **kwargs: Any) -> Any:
    return await messages_to_wire(
        messages, capabilities=caps, thinking=None, endpoint_name="ep", model="model-a", **kwargs
    )


async def test_tool_images_ride_in_one_user_turn_after_the_whole_batch() -> None:
    """A user turn between two results of one batch would close the batch, and
    the server would refuse the call left unanswered; so the carrier comes after
    the last result, and the tool messages themselves stay strings."""
    loader = _Loader()
    wire = await _wire(
        [
            _user("click the button"),
            _calls("c1", "c2"),
            _result("c1", ToolResultImage(blob_ref="shot-1", width=1280, height=800)),
            _result("c2", ToolResultImage(blob_ref="shot-2")),
            _user("thanks"),
        ],
        TAKES_IMAGES,
        image_loader=loader,
    )

    roles = [entry["role"] for entry in wire.entries]
    assert roles == ["user", "assistant", "tool", "tool", "user", "user"]
    assert wire.entries[2] == {"role": "tool", "tool_call_id": "c1", "content": "done"}
    assert wire.entries[3] == {"role": "tool", "tool_call_id": "c2", "content": "done"}
    carrier = wire.entries[4]["content"]
    assert carrier[0] == {"type": "text", "text": TOOL_IMAGES_CARRIER_HEADER}
    assert carrier[1] == {"type": "text", "text": "[images returned by tool call c1]"}
    assert carrier[2] == {"type": "text", "text": "[image 1 of tool call c1: 1280x800 pixels]"}
    assert carrier[3] == {"type": "image_url", "image_url": {"url": _url(b"\x89PNGshot-1")}}
    assert carrier[4] == {"type": "text", "text": "[images returned by tool call c2]"}
    assert carrier[5]["type"] == "image_url"
    assert loader.loaded == ["shot-1", "shot-2"]
    # The operator's next turn is its own message, never merged into the carrier.
    assert wire.entries[5] == {"role": "user", "content": "thanks"}
    # The carrier stands for the end of the batch: the last tool message maps to it.
    assert wire.index_map == [0, 1, 2, 4, 5]


async def test_an_endpoint_that_takes_parts_in_a_tool_message_gets_them_there() -> None:
    caps = ProviderCapabilities(images=ImageLimits(), tool_result_images=True)
    wire = await _wire(
        [_calls("c1"), _result("c1", ToolResultImage(data=b"\x89PNGraw", width=4, height=2))],
        caps,
    )

    assert [entry["role"] for entry in wire.entries] == ["assistant", "tool"]
    assert wire.entries[1]["content"] == [
        {"type": "text", "text": "done"},
        {"type": "text", "text": "[image 1 of tool call c1: 4x2 pixels]"},
        {"type": "image_url", "image_url": {"url": _url(b"\x89PNGraw")}},
    ]


async def test_inline_bytes_are_sent_without_a_loader() -> None:
    wire = await _wire([_calls("c1"), _result("c1", ToolResultImage(data=b"\x89PNGraw"))], TAKES_IMAGES)

    assert wire.entries[2]["content"][2] == {"type": "image_url", "image_url": {"url": _url(b"\x89PNGraw")}}


async def test_a_text_only_model_is_told_in_the_result_and_gets_no_extra_turn() -> None:
    wire = await _wire(
        [_calls("c1"), _result("c1", ToolResultImage(blob_ref="shot-1"), ToolResultImage(data=b"x"))],
        ProviderCapabilities(),
        image_loader=_Loader(),
    )

    assert [entry["role"] for entry in wire.entries] == ["assistant", "tool"]
    assert wire.entries[1]["content"] == (
        "done\n[image not sent, this model does not take images: shot-1]"
        "\n[image not sent, this model does not take images: inline image]"
    )


async def test_tool_and_user_images_share_the_endpoints_image_limit() -> None:
    """The limit is per request, so the newest images win whoever sent them."""
    caps = ProviderCapabilities(images=ImageLimits(max_images=2))
    wire = await _wire(
        [
            _user("look", metadata={IMAGE_REFS_METADATA_KEY: [{"blob_ref": "photo", "mime_type": "image/png"}]}),
            _calls("c1"),
            _result("c1", ToolResultImage(blob_ref="shot-1"), ToolResultImage(blob_ref="shot-2")),
        ],
        caps,
        image_loader=_Loader(),
    )

    assert wire.entries[0]["content"] == (
        "look\n[image not sent, only the most recent images fit in one request: photo]"
    )
    carrier = wire.entries[3]["content"]
    assert [part["type"] for part in carrier] == ["text", "text", "image_url", "image_url"]


async def test_an_image_the_endpoint_cannot_take_is_named_in_the_result() -> None:
    caps = ProviderCapabilities(images=ImageLimits(mime_types=frozenset({"image/png"})))
    wire = await _wire(
        [
            _calls("c1"),
            _result(
                "c1",
                ToolResultImage(data=b"GIF89a", mime_type="image/gif"),
                ToolResultImage(data=b"\x89PNG"),
            ),
        ],
        caps,
    )

    assert wire.entries[1]["content"] == (
        "done\n[image not sent, image/gif is not a format this model takes: inline image]"
    )
    assert [part["type"] for part in wire.entries[2]["content"]] == ["text", "text", "image_url"]


async def test_the_images_of_a_result_the_repair_drops_are_dropped_with_it() -> None:
    wire = await _wire(
        [_user("hi"), _result("orphan", ToolResultImage(blob_ref="shot-1"))],
        TAKES_IMAGES,
        image_loader=_Loader(),
    )

    assert wire.entries == [{"role": "user", "content": "hi"}]


async def test_a_call_the_repair_answers_carries_no_images() -> None:
    """An inserted "interrupted" answer belongs to no core message; the carrier
    for the batch is still placed after it, at the end of the run."""
    wire = await _wire(
        [_calls("c1", "c2"), _result("c1", ToolResultImage(blob_ref="shot-1"))],
        TAKES_IMAGES,
        image_loader=_Loader(),
    )

    assert [entry["role"] for entry in wire.entries] == ["assistant", "tool", "tool", "user"]
    assert wire.entries[2]["tool_call_id"] == "c2"
    assert wire.index_map == [0, 3]


async def test_a_cache_breakpoint_at_the_last_result_lands_on_the_carrier() -> None:
    """A tool message's content must stay a string for most servers, so a
    marker there is skipped; the carrier is a user turn and can hold one."""
    wire = await _wire(
        [_calls("c1"), _result("c1", ToolResultImage(blob_ref="shot-1"))],
        TAKES_IMAGES,
        image_loader=_Loader(),
    )
    apply_cache_control(
        wire.entries, [CacheBreakpoint(message_index=1, cache_control_type="ephemeral", rationale="x")], wire.index_map
    )

    assert wire.entries[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert wire.entries[1]["content"] == "done"


# The capability check ------------------------------------------------------


def test_a_provider_that_says_nothing_takes_no_images() -> None:
    assert vision_support_of(object()) == VisionSupport()


def test_the_adapter_says_what_its_endpoint_does_with_images() -> None:
    assert isinstance(provider(FakeServer()), IVisionCapability)
    assert provider(FakeServer()).vision_support() == VisionSupport()
    # Capabilities alone are not enough: without a loader no stored image is sent.
    assert provider(FakeServer(), capabilities=TAKES_IMAGES).vision_support() == VisionSupport()
    seeing = provider(
        FakeServer(),
        capabilities=ProviderCapabilities(images=ImageLimits(max_images=5)),
        image_loader=_Loader(),
    )
    assert vision_support_of(seeing, "model-a") == VisionSupport(
        images=True, tool_result_images=True, native_tool_result_images=False, max_images=5
    )
    native = provider(
        FakeServer(),
        capabilities=ProviderCapabilities(images=ImageLimits(), tool_result_images=True),
        image_loader=_Loader(),
    )
    assert native.vision_support().native_tool_result_images is True


class TestOpenAIVisionCapabilityConformance(VisionCapabilityConformance):
    @pytest.fixture
    def subject_factory(self) -> Callable[[], Any]:
        return lambda: provider(
            FakeServer(),
            capabilities=ProviderCapabilities(images=ImageLimits(), tool_result_images=True),
            image_loader=_Loader(),
        )


class TestBlindProviderVisionCapabilityConformance(VisionCapabilityConformance):
    @pytest.fixture
    def subject_factory(self) -> Callable[[], Any]:
        return lambda: provider(FakeServer())


class _Inconsistent:
    def vision_support(self, model: str | None = None) -> VisionSupport:
        return VisionSupport(images=False, tool_result_images=True, native_tool_result_images=True)


def test_the_suite_refuses_an_answer_that_contradicts_itself() -> None:
    suite = VisionCapabilityConformance()
    with pytest.raises(AssertionError, match="no images"):
        suite.test_the_answer_is_consistent(_Inconsistent())
