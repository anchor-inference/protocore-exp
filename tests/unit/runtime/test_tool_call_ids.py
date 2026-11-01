"""The rules that give every tool call an id of its own within a conversation."""
from __future__ import annotations

from protocore.contracts.llm import ProviderDelta, ProviderDeltaKind
from protocore.contracts.types import Message, MessageRole, TextBlock, ToolResultBlock, ToolUseBlock
from protocore.runtime.tool_call_ids import (
    ToolCallIdAssigner,
    fresh_tool_call_id,
    rekey_repeated_tool_call_ids,
)

START = ProviderDeltaKind.tool_use_start
INPUT = ProviderDeltaKind.tool_use_input
STOP = ProviderDeltaKind.tool_use_stop


def _delta(kind: ProviderDeltaKind, call_id: str, piece: str = "") -> ProviderDelta:
    return ProviderDelta(kind=kind, tool_call_id=call_id, tool_name="Write", tool_input_delta=piece or None)


def test_a_numbered_id_takes_the_next_number_above_its_stem() -> None:
    taken = {"functions.Write:0", "functions.Write:2", "functions.Read:7"}
    assert fresh_tool_call_id("functions.Write:0", taken) == "functions.Write:3"


def test_an_unnumbered_id_takes_a_suffix() -> None:
    assert fresh_tool_call_id("toolu_abc", {"toolu_abc"}) == "toolu_abc_2"
    assert fresh_tool_call_id("toolu_abc", {"toolu_abc", "toolu_abc_2"}) == "toolu_abc_3"


def test_an_unused_provider_id_is_kept() -> None:
    taken: set[str] = set()
    assigner = ToolCallIdAssigner(taken)
    delta = _delta(START, "call_1")
    assert assigner.assign(delta) is delta
    assert taken == {"call_1"}


def test_a_repeat_still_open_is_routed_by_the_adapter_order() -> None:
    """Both calls start, each sends its arguments, then the stops come in start order."""
    taken = {"functions.Write:0"}
    assigner = ToolCallIdAssigner(taken)
    seen = [
        assigner.assign(_delta(START, "functions.Write:0")),
        assigner.assign(_delta(INPUT, "functions.Write:0", "a")),
        assigner.assign(_delta(START, "functions.Write:0")),
        assigner.assign(_delta(INPUT, "functions.Write:0", "b")),
        assigner.assign(_delta(STOP, "functions.Write:0")),
        assigner.assign(_delta(STOP, "functions.Write:0")),
    ]
    first, second = "functions.Write:1", "functions.Write:2"
    assert [(d.kind, d.tool_call_id) for d in seen] == [
        (START, first), (INPUT, first), (START, second), (INPUT, second), (STOP, first), (STOP, second),
    ]
    assert {first, second} <= taken


def test_a_stop_without_a_start_still_gets_an_id_of_its_own() -> None:
    assigner = ToolCallIdAssigner({"x"})
    assert assigner.assign(_delta(STOP, "x")).tool_call_id == "x_2"


def test_deltas_that_name_no_call_pass_through() -> None:
    assigner = ToolCallIdAssigner({"x"})
    text = ProviderDelta(kind=ProviderDeltaKind.text, content="hi")
    assert assigner.assign(text) is text


def _call(call_id: str, value: str) -> Message:
    return Message(
        role=MessageRole.assistant,
        content_blocks=[ToolUseBlock(tool_call_id=call_id, name="Write", arguments_json=f'{{"v": "{value}"}}')],
    )


def _result(call_id: str, text: str) -> Message:
    return Message(role=MessageRole.tool, content_blocks=[ToolResultBlock(tool_call_id=call_id, content=text)])


def test_a_stored_repeat_is_paired_by_position() -> None:
    history = [_call("c", "a"), _result("c", "A"), _call("c", "b"), _result("c", "B")]
    rekeyed = rekey_repeated_tool_call_ids(history)
    ids = [block.tool_call_id for message in rekeyed for block in message.content_blocks]
    assert ids == ["c", "c", "c_2", "c_2"]
    # Untouched messages are the same objects, so nothing downstream sees a rewrite.
    assert rekeyed[0] is history[0] and rekeyed[1] is history[1]


def test_a_second_result_for_one_call_keeps_its_id_for_the_repair_to_drop() -> None:
    history = [_call("c", "a"), _result("c", "first"), _result("c", "again")]
    assert rekey_repeated_tool_call_ids(history) == history


def test_rekeying_is_a_function_of_the_messages_alone() -> None:
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="go")]),
        _call("functions.Write:0", "a"),
        _result("functions.Write:0", "A"),
        _call("functions.Write:0", "b"),
        _result("functions.Write:0", "B"),
    ]
    assert rekey_repeated_tool_call_ids(history) == rekey_repeated_tool_call_ids(list(history))
