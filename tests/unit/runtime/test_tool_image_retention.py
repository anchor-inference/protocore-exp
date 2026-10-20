"""Only the newest tool images ride in a request; the rest leave in batches, by count alone."""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    TOOL_IMAGES_ELIDED_METADATA_KEY,
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolResultImage,
    ToolUseBlock,
)
from protocore.prompts import bundled_prompt_provider
from protocore.runtime.context.compaction import (
    TokenEstimator,
    estimate_message_tokens,
    render_span_for_summary,
)
from protocore.runtime.tool_image_retention import (
    evicted_image_count,
    evicted_image_tokens,
    retain_recent_tool_images,
)

PROMPTS = bundled_prompt_provider()


def _rc(**overrides: object) -> LoopConstants:
    values: dict[str, object] = {"tool_result_image_keep": 3, "tool_result_image_evict_batch": 2}
    values.update(overrides)
    return LoopConstants(**values)  # type: ignore[arg-type]


def _step(call_id: str, images: int = 1) -> list[Message]:
    return [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id=call_id, name="Screenshot", arguments_json="{}")],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id=call_id,
                    content=f"shot {call_id}",
                    images=[ToolResultImage(blob_ref=f"{call_id}-{n}") for n in range(images)],
                )
            ],
        ),
    ]


def _history(steps: int, images: int = 1) -> list[Message]:
    history: list[Message] = [Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")])]
    for index in range(steps):
        history.extend(_step(f"c{index}", images))
    return history


def _results(view: list[Message]) -> list[ToolResultBlock]:
    return [block for message in view for block in message.content_blocks if isinstance(block, ToolResultBlock)]


@pytest.mark.parametrize(
    ("total", "evicted"),
    [(0, 0), (3, 0), (4, 0), (5, 2), (6, 2), (7, 4), (8, 4), (9, 6)],
)
def test_images_leave_a_whole_batch_at_a_time(total: int, evicted: int) -> None:
    assert evicted_image_count(total, _rc()) == evicted


def test_a_batch_of_one_evicts_as_soon_as_the_limit_is_passed() -> None:
    assert [evicted_image_count(total, _rc(tool_result_image_evict_batch=1)) for total in range(6)] == [
        0, 0, 0, 0, 1, 2,
    ]


def test_no_limit_keeps_everything_and_a_keep_of_zero_sends_nothing() -> None:
    assert evicted_image_count(50, _rc(tool_result_image_keep=None)) == 0
    assert evicted_image_count(5, _rc(tool_result_image_keep=0)) == 5


def test_the_oldest_images_leave_the_view_and_say_so_while_history_keeps_them() -> None:
    history = _history(5)

    view = retain_recent_tool_images(history, _rc(), PROMPTS)

    results = _results(view)
    assert [len(block.images) for block in results] == [0, 0, 1, 1, 1]
    assert [block.metadata.get(TOOL_IMAGES_ELIDED_METADATA_KEY) for block in results] == [1, 1, None, None, None]
    assert results[0].content.startswith("shot c0\n[1 image this tool returned is no longer shown")
    assert "newest 3 tool images" in results[0].content
    assert results[2] is _results(history)[2]
    assert [len(block.images) for block in _results(history)] == [1] * 5


def test_a_result_can_lose_its_older_images_and_keep_its_newer_ones() -> None:
    history = _history(2, images=3)

    view = retain_recent_tool_images(history, _rc(), PROMPTS)

    first, second = _results(view)
    assert [image.blob_ref for image in first.images] == ["c0-2"]
    assert first.metadata[TOOL_IMAGES_ELIDED_METADATA_KEY] == 2
    assert "2 images this tool returned are no longer shown" in first.content
    assert len(second.images) == 3


def test_a_result_with_no_text_gets_the_line_alone() -> None:
    history = [
        Message(
            role=MessageRole.tool,
            content_blocks=[
                ToolResultBlock(tool_call_id="c", content="", images=[ToolResultImage(blob_ref="r")])
            ],
        )
    ]

    [result] = _results(retain_recent_tool_images(history, _rc(tool_result_image_keep=0), PROMPTS))

    assert result.content.startswith("[1 image")
    assert result.images == ()


def test_the_cut_is_the_same_until_the_next_batch_so_the_prefix_holds() -> None:
    """The view after five images and the view after six agree on everything
    before the sixth: the provider's cache of that prefix is still good."""
    history = _history(5)
    five = retain_recent_tool_images(history, _rc(), PROMPTS)
    six = retain_recent_tool_images([*history, *_step("c5")], _rc(), PROMPTS)

    assert six[: len(five)] == five


def test_building_the_view_again_changes_nothing() -> None:
    view = retain_recent_tool_images(_history(7), _rc(), PROMPTS)

    assert retain_recent_tool_images(view, _rc(), PROMPTS) == view


def test_a_view_without_tool_images_is_handed_back_as_it_is() -> None:
    history = _history(0)

    assert retain_recent_tool_images(history, _rc(), PROMPTS) == history


# Pricing -------------------------------------------------------------------


def test_a_tool_image_is_priced_like_a_user_image() -> None:
    rc = _rc()
    plain = estimate_message_tokens(_step("c0", images=0)[1], rc)
    with_two = estimate_message_tokens(_step("c0", images=2)[1], rc)

    assert with_two - plain == 2 * rc.token_count_image_tokens


def test_the_history_sum_measures_the_request_that_will_be_sent() -> None:
    """Pricing every screenshot of a long run would compact it long before the
    request is anywhere near the window."""
    rc = _rc()
    history = _history(9)
    view = retain_recent_tool_images(history, rc, PROMPTS)
    estimator = TokenEstimator()

    assert evicted_image_tokens(view, rc) == 0
    assert evicted_image_tokens(history, rc) > 5 * (rc.token_count_image_tokens - 100)
    gap = abs(estimator.estimate_history(history, rc) - estimator.estimate_history(view, rc))
    assert gap < rc.token_count_image_tokens // 10
    assert abs(
        estimator.estimate_history_uncalibrated(history, rc)
        - estimator.estimate_history_uncalibrated(view, rc)
    ) < rc.token_count_image_tokens // 10


def test_the_summariser_is_told_a_result_had_images() -> None:
    text = render_span_for_summary(
        _step("c0", images=2), _rc(), tool_names={"c0": "Screenshot"}, originals={}, max_tokens=1_000
    )

    assert "[2 image(s) returned]" in text
