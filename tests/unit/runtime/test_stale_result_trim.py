"""Stale tool results are cut down; fresh ones, pins and persist are not."""
from __future__ import annotations

from protocore.constants import PROTOCOL_COMPACTED_TOOL_RESULT_V1
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.prompts import bundled_prompt_provider
from protocore.runtime.stale_result_trim import trim_stale_results
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

PROMPTS = bundled_prompt_provider()


def _rc(**overrides: object) -> LoopConstants:
    values: dict[str, object] = {
        "tool_result_stale_trim_enabled": True,
        "tool_result_fresh_count": 2,
        "tool_result_stale_max_chars": 100,
        "tool_result_stale_trim_batch_chars": 500,
    }
    values.update(overrides)
    return LoopConstants(**values)  # type: ignore[arg-type]


def _pair(call_id: str, size: int, **result_fields: object) -> list[Message]:
    """One call and its answer, the answer ``size`` characters long."""
    return [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id=call_id, name="Read", arguments_json="{}")],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[
                ToolResultBlock(tool_call_id=call_id, content="x" * size, **result_fields)
            ],
        ),
    ]


def _results(view: list[Message]) -> dict[str, ToolResultBlock]:
    return {
        block.tool_call_id: block
        for message in view
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }


def _transcript(count: int, size: int = 900, prefix: str = "call") -> list[Message]:
    history: list[Message] = []
    for index in range(count):
        history.extend(_pair(f"{prefix}_{index}", size))
    return history


def test_the_oldest_results_are_cut_and_the_newest_are_not() -> None:
    history = _transcript(5)
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert sorted(trimmed) == ["call_0", "call_1", "call_2"]
    results = _results(view)
    for call_id in ("call_0", "call_1", "call_2"):
        assert results[call_id].content.startswith("x" * 100)
        assert "call the tool again" in results[call_id].content
        assert results[call_id].metadata["stale_trimmed"] is True
    for call_id in ("call_3", "call_4"):
        assert results[call_id].content == "x" * 900
        assert "stale_trimmed" not in results[call_id].metadata


def test_the_pointer_says_what_was_kept_and_what_went() -> None:
    view, _ = trim_stale_results(_transcript(4), _rc(), PROMPTS)
    content = _results(view)["call_0"].content
    assert "[trimmed: 100 chars kept, 800 removed" in content
    assert "NOT complete evidence" in content


def test_persist_is_never_touched() -> None:
    history = _transcript(5)
    before = [message.model_dump() for message in history]
    trim_stale_results(history, _rc(), PROMPTS)
    assert [message.model_dump() for message in history] == before


def test_nothing_happens_below_the_batch_threshold() -> None:
    # Three trimmable results at 800 chars of excess each is 2400 — under a
    # batch of 10_000, so the prefix is left exactly where it was.
    history = _transcript(5)
    view, trimmed = trim_stale_results(
        history, _rc(tool_result_stale_trim_batch_chars=10_000), PROMPTS
    )
    assert trimmed == frozenset()
    assert view == list(history)


def test_short_results_are_left_alone_however_old() -> None:
    history = _transcript(2, size=50, prefix="short") + _transcript(3)
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert "short_0" not in trimmed
    assert _results(view)["short_0"].content == "x" * 50


def test_a_trimmed_result_stays_trimmed_after_the_batch_that_cut_it() -> None:
    history = _transcript(5)
    _, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    # The next build has nothing new worth a batch of its own — the decision
    # still has to hold, or the whole result comes back and the prefix moves.
    view, still = trim_stale_results(
        history,
        _rc(tool_result_stale_trim_batch_chars=10_000),
        PROMPTS,
        already_trimmed=trimmed,
    )
    assert still == trimmed
    assert len(_results(view)["call_0"].content) < 900


def test_a_pinned_result_is_never_cut() -> None:
    history = _transcript(5)
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS, pinned_ids={"call_0"})
    assert "call_0" not in trimmed
    assert _results(view)["call_0"].content == "x" * 900


def test_a_marked_result_is_never_cut() -> None:
    history = _pair("kept", 900, metadata={"retention": "pinned"}) + _transcript(4)
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert "kept" not in trimmed
    assert _results(view)["kept"].content == "x" * 900


def test_a_compacted_placeholder_is_never_rewritten() -> None:
    placeholder = PROTOCOL_COMPACTED_TOOL_RESULT_V1 + " " + "y" * 900
    history = [
        *_pair("folded", 0),
        *_transcript(4),
    ]
    history[1] = Message(
        role=MessageRole.user,
        content_blocks=[ToolResultBlock(tool_call_id="folded", content=placeholder)],
    )
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert "folded" not in trimmed
    assert _results(view)["folded"].content == placeholder


def test_the_latest_round_is_never_cut() -> None:
    # Three calls issued together; the fresh window holds only two of them, and
    # the third is still an answer the model is reading right now.
    history = _transcript(4)
    history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="a", name="Read", arguments_json="{}"),
                ToolUseBlock(tool_call_id="b", name="Read", arguments_json="{}"),
                ToolUseBlock(tool_call_id="c", name="Read", arguments_json="{}"),
            ],
        )
    )
    for call_id in ("a", "b", "c"):
        history.append(
            Message(
                role=MessageRole.user,
                content_blocks=[ToolResultBlock(tool_call_id=call_id, content="x" * 900)],
            )
        )
    _, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert trimmed == {"call_0", "call_1", "call_2", "call_3"}


def _long_run() -> list[Message]:
    """One question answered by ten sequential rounds of a 20 KB read."""
    history: list[Message] = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="one question")])
    ]
    for index in range(10):
        history.extend(_pair(f"round_{index}", 20_000))
    return history


def test_a_run_of_many_rounds_keeps_its_latest_round_and_its_fresh_window() -> None:
    # Below the batch threshold nothing moves at all, and what the guarantee
    # names — the latest round, and the newest six results — is carried whole.
    view, trimmed = trim_stale_results(
        _long_run(),
        LoopConstants(
            tool_result_stale_trim_enabled=True,
            tool_result_stale_trim_batch_chars=100_000,
        ),
        PROMPTS,
    )
    assert trimmed == frozenset()
    results = _results(view)
    for index in range(4, 10):  # the fresh six, the last of them the latest round
        assert results[f"round_{index}"].content == "x" * 20_000


def test_the_older_rounds_of_the_same_run_go_once_the_batch_is_worth_it() -> None:
    # The run this is for: one question, twenty files read, the window filling
    # up. Past the fresh seven the run has moved on, and 54_000 characters of
    # excess is worth the changed prefix.
    view, trimmed = trim_stale_results(
        _long_run(),
        LoopConstants(
            tool_result_stale_trim_enabled=True,
            tool_result_fresh_count=7,
            tool_result_stale_max_chars=2000,
            tool_result_stale_trim_batch_chars=40_000,
        ),
        PROMPTS,
    )
    assert sorted(trimmed) == ["round_0", "round_1", "round_2"]
    results = _results(view)
    for index in range(3):
        assert results[f"round_{index}"].content.startswith("x" * 2000)
        assert "call the tool again" in results[f"round_{index}"].content
    for index in range(3, 10):
        assert results[f"round_{index}"].content == "x" * 20_000


def test_a_sticky_id_the_view_no_longer_carries_is_forgotten() -> None:
    history = _transcript(5)
    _, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    # The checkpoint dropped the oldest turn; its id must not ride along in
    # the set the engine keeps and every snapshot writes.
    _, still = trim_stale_results(history[2:], _rc(), PROMPTS, already_trimmed=trimmed)
    assert "call_0" not in still
    assert "call_1" in still


def test_a_trimmed_result_fits_the_split_that_runs_after_it() -> None:
    # The two limits are independent: a deployment that keeps more head than
    # the split allows would otherwise have the split cut the trim pointer in
    # half, and the line saying how to get the result back with it.
    history = _transcript(5, size=4000)
    view, trimmed = trim_stale_results(
        history,
        _rc(
            tool_result_stale_max_chars=3000,
            tool_result_split_enabled=True,
            tool_result_content_max_chars=500,
            # What is measured here is the head sizing, not whether the batch
            # pays for itself: four results of which a quarter goes would not.
            tool_result_stale_trim_min_share=0.0,
        ),
        PROMPTS,
    )
    assert "call_0" in trimmed
    content = _results(view)["call_0"].content
    assert len(content) <= 500
    assert content.endswith("call the tool again if you need the rest]")


def test_the_switch_off_leaves_the_view_alone() -> None:
    history = _transcript(5)
    view, trimmed = trim_stale_results(
        history, _rc(tool_result_stale_trim_enabled=False), PROMPTS
    )
    assert view == list(history)
    assert trimmed == frozenset()


def test_prose_blocks_are_carried_through_untouched() -> None:
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")]),
        *_transcript(5),
    ]
    view, _ = trim_stale_results(history, _rc(), PROMPTS)
    assert view[0] is history[0]


def test_a_pin_the_run_has_since_falsified_is_cut_like_any_other() -> None:
    # The pinned read describes a file the run rewrote two calls later. Holding
    # it whole keeps the version before the write in front of the model, which
    # is worse than handing back its head and saying to read the file again.
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="read", name="Read", arguments_json='{"path": "a.py"}'
                )
            ],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[
                ToolResultBlock(tool_call_id="read", content="x" * 900, path="a.py")
            ],
        ),
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(
                    tool_call_id="write", name="Write", arguments_json='{"path": "a.py"}'
                )
            ],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[ToolResultBlock(tool_call_id="write", content="written")],
        ),
        *_transcript(4),
    ]
    kept, _ = trim_stale_results(history, _rc(), PROMPTS, pinned_ids={"read"})
    assert _results(kept)["read"].content == "x" * 900

    cut, trimmed = trim_stale_results(
        history, _rc(), PROMPTS, pinned_ids={"read"}, roles=CONVENTIONAL_TOOL_ROLES
    )
    assert "read" in trimmed
    assert "call the tool again" in _results(cut)["read"].content


def test_a_pin_over_a_file_nothing_rewrote_survives_the_roles_being_declared() -> None:
    history = _pair("kept", 900, path="untouched.py") + _transcript(4)
    view, trimmed = trim_stale_results(
        history, _rc(), PROMPTS, pinned_ids={"kept"}, roles=CONVENTIONAL_TOOL_ROLES
    )
    assert "kept" not in trimmed
    assert _results(view)["kept"].content == "x" * 900


def _page(call_id: str, *, body: int, citation: str) -> list[Message]:
    """A page-shaped result: a citation line, then a body far past the head."""
    return [
        *_pair(call_id, 0)[:1],
        Message(
            role=MessageRole.user,
            content_blocks=[
                ToolResultBlock(
                    tool_call_id=call_id,
                    content="\n".join([citation, "y" * body, citation]),
                )
            ],
        )
    ]


def test_the_line_that_identifies_the_result_survives_the_body() -> None:
    # The citation sits past the head, so a head-only cut loses it — and a
    # citation the model cannot see is one it makes up.
    citation = "Cite exactly: [[catalog:9912/p414]]"
    history = [*_page("page", body=900, citation=citation), *_transcript(4)]
    view, trimmed = trim_stale_results(history, _rc(), PROMPTS)
    assert "page" in trimmed
    content = _results(view)["page"].content
    assert citation in content
    assert content.index(citation) < content.index("[trimmed:")
    assert "y" * 900 not in content


def test_every_configured_prefix_is_carried_and_others_are_not() -> None:
    body = "\n".join(
        [
            "Cite: Ross, Atlas of Mosses, 1961",
            "Source: https://example.invalid/records/8",
            "note: the middle of the page nobody needs twice",
            "z" * 900,
        ]
    )
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id="page", name="Read", arguments_json="{}")],
        ),
        Message(
            role=MessageRole.user,
            content_blocks=[ToolResultBlock(tool_call_id="page", content=body)],
        ),
        *_transcript(4),
    ]
    view, _ = trim_stale_results(history, _rc(tool_result_stale_max_chars=10), PROMPTS)
    content = _results(view)["page"].content
    assert "Cite: Ross, Atlas of Mosses, 1961" in content
    assert "Source: https://example.invalid/records/8" in content
    assert "note: the middle of the page" not in content


def test_the_carry_over_can_be_turned_off() -> None:
    citation = "Cite exactly: [[catalog:9912/p414]]"
    history = [*_page("page", body=900, citation=citation), *_transcript(4)]
    view, _ = trim_stale_results(
        history,
        _rc(tool_result_stale_max_chars=10, tool_result_stale_trim_protected_prefixes=""),
        PROMPTS,
    )
    assert citation not in _results(view)["page"].content


def test_a_carried_citation_still_fits_the_split_that_runs_after_it() -> None:
    citation = "Cite exactly: [[catalog:9912/p414]]"
    history = [*_page("page", body=4000, citation=citation), *_transcript(4, size=4000)]
    view, trimmed = trim_stale_results(
        history,
        _rc(
            tool_result_stale_max_chars=3000,
            tool_result_split_enabled=True,
            tool_result_content_max_chars=500,
            # What is measured here is the head sizing, not whether the batch
            # pays for itself: four results of which a quarter goes would not.
            tool_result_stale_trim_min_share=0.0,
        ),
        PROMPTS,
    )
    assert "page" in trimmed
    content = _results(view)["page"].content
    assert len(content) <= 500
    assert citation in content
    assert content.endswith("call the tool again if you need the rest]")


def _prose(size: int) -> list[Message]:
    """A plain exchange of ``size`` characters that no trim can shorten."""
    return [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="q" * size)]),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="a" * size)]),
    ]


def test_a_trim_that_saves_little_of_the_view_never_happens() -> None:
    # The case that cost a full cache write of a long context on every trim:
    # a few pages cut to save three per cent of the view. The batch threshold
    # is crossed, but re-writing the conversation into the cache would take
    # hundreds of calls to pay back.
    history = [*_transcript(3, size=6000), *_prose(250_000), *_transcript(2, size=50, prefix="new")]
    view, trimmed = trim_stale_results(
        history, _rc(tool_result_stale_trim_batch_chars=10_000), PROMPTS
    )
    assert trimmed == frozenset()
    assert view == list(history)


def test_a_pile_of_stale_output_is_shed_in_one_batch() -> None:
    history = [*_prose(20_000), *_transcript(12, size=30_000), *_transcript(2, size=50, prefix="new")]
    _, trimmed = trim_stale_results(
        history, _rc(tool_result_stale_trim_batch_chars=10_000), PROMPTS
    )
    assert trimmed == frozenset(f"call_{index}" for index in range(12))


def test_a_share_of_zero_leaves_only_the_batch_threshold() -> None:
    history = [*_transcript(3, size=6000), *_prose(250_000), *_transcript(2, size=50, prefix="new")]
    _, trimmed = trim_stale_results(
        history,
        _rc(tool_result_stale_trim_batch_chars=10_000, tool_result_stale_trim_min_share=0.0),
        PROMPTS,
    )
    assert trimmed == frozenset({"call_0", "call_1", "call_2"})


def test_results_already_cut_count_at_their_head_in_the_stretch() -> None:
    # The request carries a sticky result at its head, so that is what a later
    # trim invalidates of it — counting it whole would hold back a batch that
    # in fact pays for itself.
    history = [*_transcript(4, size=40_000, prefix="cut"), *_transcript(3, size=6000), *_transcript(2, size=50, prefix="new")]
    _, trimmed = trim_stale_results(
        history,
        _rc(tool_result_stale_trim_batch_chars=1000, tool_result_stale_trim_min_share=0.5),
        PROMPTS,
        already_trimmed={"cut_0", "cut_1", "cut_2", "cut_3"},
    )
    assert {"call_0", "call_1", "call_2"} <= trimmed


def test_a_result_the_pointer_would_lengthen_is_left_whole() -> None:
    # Thirteen characters over the head: the pointer alone is longer than what
    # the cut would remove, so cutting it moves the prefix to grow the request.
    history = [*_transcript(6, size=2000), *_pair("barely", 113), *_transcript(2, size=50, prefix="new")]
    _, trimmed = trim_stale_results(
        history, _rc(tool_result_stale_trim_batch_chars=1000), PROMPTS
    )
    assert "barely" not in trimmed
    assert {f"call_{index}" for index in range(6)} <= trimmed


def test_on_a_large_window_the_window_share_sets_the_batch() -> None:
    # A batch sized for a small window is a drip on a million-token one: every
    # trim rewrites the cache behind it, so there the window's share decides.
    # 0.05 of 1_000_000 tokens at four characters a token is 200_000 chars, and
    # five stale pages of 30_000 save about 150_000.
    history = [*_transcript(5, size=30_000), *_transcript(2, size=50, prefix="new")]
    rc = _rc(
        model_context_window=1_000_000,
        tool_result_stale_trim_batch_chars=10_000,
        tool_result_stale_trim_min_share=0.0,
        tool_result_stale_trim_window_share=0.05,
    )
    _, trimmed = trim_stale_results(history, rc, PROMPTS)
    assert trimmed == frozenset()

    history = [*_transcript(12, size=30_000), *_transcript(2, size=50, prefix="new")]
    _, trimmed = trim_stale_results(history, rc, PROMPTS)
    assert trimmed == frozenset(f"call_{index}" for index in range(12))


def test_the_window_share_never_lowers_the_configured_batch() -> None:
    # On a small window the share comes to less than the batch, and the batch
    # stands: the larger of the two is the threshold.
    history = _transcript(5)
    _, trimmed = trim_stale_results(
        history,
        _rc(
            model_context_window=8_192,
            tool_result_stale_trim_batch_chars=10_000,
            tool_result_stale_trim_window_share=0.01,
        ),
        PROMPTS,
    )
    assert trimmed == frozenset()


def test_a_window_share_of_zero_is_the_fixed_batch() -> None:
    history = [*_transcript(5, size=30_000), *_transcript(2, size=50, prefix="new")]
    _, trimmed = trim_stale_results(
        history,
        _rc(
            model_context_window=1_000_000,
            tool_result_stale_trim_batch_chars=10_000,
            tool_result_stale_trim_min_share=0.0,
        ),
        PROMPTS,
    )
    assert trimmed == frozenset(f"call_{index}" for index in range(5))
