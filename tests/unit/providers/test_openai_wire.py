"""Rendering core requests into chat-completions JSON, checked on the JSON itself."""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.llm import CacheBreakpoint
from protocore.contracts.types import (
    IMAGE_REFS_METADATA_KEY,
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.providers.openai_compat import (
    INTERRUPTED_TOOL_RESULT,
    REASONING_ENVELOPE,
    ImageLimits,
    ProviderCapabilities,
    ReasoningReplay,
    ThinkingDialect,
    messages_to_wire,
    tools_to_wire,
)
from protocore.providers.openai_compat.wire import (
    apply_cache_control,
    apply_thinking,
    reasoning_envelope,
    tool_choice_for,
)

CAPS = ProviderCapabilities()


def _user(text: str, **kwargs: Any) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)], **kwargs)


def _assistant(*blocks: Any, **kwargs: Any) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=list(blocks), **kwargs)


def _results(*pairs: tuple[str, str]) -> Message:
    return Message(
        role=MessageRole.tool,
        content_blocks=[ToolResultBlock(tool_call_id=call_id, content=content) for call_id, content in pairs],
    )


def _call(call_id: str, name: str = "Read", arguments: str = '{"path": "a"}') -> ToolUseBlock:
    return ToolUseBlock(tool_call_id=call_id, name=name, arguments_json=arguments)


async def _wire(messages: list[Message], caps: ProviderCapabilities = CAPS, **kwargs: Any) -> Any:
    kwargs.setdefault("thinking", None)
    return await messages_to_wire(messages, capabilities=caps, endpoint_name="ep", model="model-a", **kwargs)


# Tool schemas --------------------------------------------------------------


def _strict_tool() -> ToolDefinition:
    return ToolDefinition(
        name="plan",
        description="Record the plan.",
        parameters=ToolParameterSchema(
            properties={
                "steps": {"type": "array", "items": {"type": "object", "additionalProperties": False}},
            },
            required=["steps"],
            additional_properties=False,
        ),
    )


def test_a_strict_schema_keeps_additional_properties_on_the_wire() -> None:
    # The core names the flag ``additional_properties``; a serialiser that
    # copied only ``properties`` and ``required`` dropped it, and the forced
    # plan tool lost its strictness.
    [entry] = tools_to_wire([_strict_tool()], CAPS)

    schema = entry["function"]["parameters"]
    assert schema == {
        "type": "object",
        "properties": {"steps": {"type": "array", "items": {"type": "object", "additionalProperties": False}}},
        "required": ["steps"],
        "additionalProperties": False,
    }
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "plan"
    assert entry["function"]["description"] == "Record the plan."


def test_a_schema_that_states_no_flag_sends_none() -> None:
    tool = ToolDefinition(name="t", description="d", parameters=ToolParameterSchema())
    [entry] = tools_to_wire([tool], CAPS)
    assert "additionalProperties" not in entry["function"]["parameters"]


def test_an_endpoint_that_refuses_the_keyword_can_turn_it_off() -> None:
    caps = ProviderCapabilities(schema_additional_properties=False)
    [entry] = tools_to_wire([_strict_tool()], caps)
    assert "additionalProperties" not in entry["function"]["parameters"]
    # Nested schemas are the tool's own text and stay as written.
    assert entry["function"]["parameters"]["properties"]["steps"]["items"]["additionalProperties"] is False


# Messages ------------------------------------------------------------------


async def test_roles_render_as_the_api_expects() -> None:
    wire = await _wire(
        [
            Message(role=MessageRole.system, content_blocks=[TextBlock(text="be brief")]),
            _user("read a"),
            _assistant(TextBlock(text="Reading."), _call("c1")),
            _results(("c1", "contents of a")),
            _assistant(TextBlock(text="It says hello.")),
        ]
    )

    assert wire.entries == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "read a"},
        {
            "role": "assistant",
            "content": "Reading.",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": '{"path": "a"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "contents of a"},
        {"role": "assistant", "content": "It says hello."},
    ]
    assert wire.index_map == [0, 1, 2, 3, 4]


async def test_an_assistant_turn_with_only_calls_has_null_content() -> None:
    wire = await _wire([_user("go"), _assistant(_call("c1", arguments="")), _results(("c1", "ok"))])
    assistant = wire.entries[1]
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["function"]["arguments"] == "{}"


async def test_an_assistant_turn_that_said_nothing_sends_an_empty_string() -> None:
    wire = await _wire([_user("go"), _assistant()])
    assert wire.entries[1] == {"role": "assistant", "content": ""}


async def test_one_tool_message_fans_out_and_the_index_map_follows() -> None:
    wire = await _wire(
        [_user("go"), _assistant(_call("c1"), _call("c2")), _results(("c1", "one"), ("c2", "two")), _user("next")]
    )
    assert [entry["role"] for entry in wire.entries] == ["user", "assistant", "tool", "tool", "user"]
    assert wire.index_map == [0, 1, 3, 4]


# Reasoning replay ------------------------------------------------------------


async def test_reasoning_never_goes_back_by_default() -> None:
    wire = await _wire([_user("q"), _assistant(TextBlock(text="a"), reasoning_content="thought")])
    assert "reasoning_content" not in wire.entries[1]


async def test_reasoning_goes_back_when_present_if_asked() -> None:
    caps = ProviderCapabilities(reasoning_replay=ReasoningReplay.when_present)
    wire = await _wire(
        [
            _user("q"),
            _assistant(ThinkingBlock(text="from blocks"), TextBlock(text="a")),
            _assistant(TextBlock(text="b")),
        ],
        caps,
    )
    assert wire.entries[1]["reasoning_content"] == "from blocks"
    assert "reasoning_content" not in wire.entries[2]


async def test_required_replay_fills_every_turn_and_drops_reasoning_only_ones_while_thinking() -> None:
    caps = ProviderCapabilities(reasoning_replay=ReasoningReplay.required, thinking=ThinkingDialect.deepseek)
    messages = [
        _user("q"),
        _assistant(TextBlock(text="no reasoning here")),
        _assistant(reasoning_content="cut off mid-thought"),
        _assistant(_call("c1"), reasoning_content="why"),
        _results(("c1", "r")),
    ]

    thinking = await _wire(messages, caps, thinking=True)
    assistants = [entry for entry in thinking.entries if entry["role"] == "assistant"]
    assert [entry.get("reasoning_content") for entry in assistants] == ["", "why"]
    assert thinking.index_map[2] == -1

    plain = await _wire(messages, caps, thinking=False)
    assistants = [entry for entry in plain.entries if entry["role"] == "assistant"]
    assert len(assistants) == 3
    assert "reasoning_content" not in assistants[0]
    assert assistants[1]["reasoning_content"] == "cut off mid-thought"


async def test_opaque_reasoning_goes_back_only_to_the_endpoint_and_model_that_wrote_it() -> None:
    mine = reasoning_envelope("ep", "model-a", [{"type": "reasoning.encrypted", "data": "x"}], 12)
    other_model = reasoning_envelope("ep", "model-b", [{"data": "y"}], 3)
    other_endpoint = reasoning_envelope("other", "model-a", [{"data": "z"}], 3)
    message = _assistant(TextBlock(text="a"), reasoning_items=(mine, other_model, other_endpoint, {"format": "else"}))
    assert mine["format"] == REASONING_ENVELOPE
    assert mine["tokens"] == 12

    on = await _wire([_user("q"), message], ProviderCapabilities(replay_reasoning_details=True))
    assert on.entries[1]["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "x"}]

    off = await _wire([_user("q"), message])
    assert "reasoning_details" not in off.entries[1]


# Tool pairing --------------------------------------------------------------


async def test_an_unanswered_call_gets_an_interrupted_result_and_an_orphan_result_is_dropped() -> None:
    wire = await _wire(
        [
            _user("go"),
            _results(("ghost", "a result whose call compaction removed")),
            _assistant(_call("c1"), _call("c2")),
            _results(("c2", "two"), ("c2", "a duplicate")),
            _user("are you there?"),
        ]
    )

    assert wire.entries == [
        {"role": "user", "content": "go"},
        wire.entries[1],
        {"role": "tool", "tool_call_id": "c2", "content": "two"},
        {"role": "tool", "tool_call_id": "c1", "content": INTERRUPTED_TOOL_RESULT},
        {"role": "user", "content": "are you there?"},
    ]
    assert wire.entries[1]["role"] == "assistant"
    # The inserted result belongs to no core message; the dropped one leaves
    # its message with no entry.
    assert wire.index_map == [0, -1, 1, 2, 4]


async def test_calls_left_open_at_the_end_are_answered_too() -> None:
    wire = await _wire([_user("go"), _assistant(_call("c1"))])
    assert wire.entries[-1] == {"role": "tool", "tool_call_id": "c1", "content": INTERRUPTED_TOOL_RESULT}


async def test_pair_repair_can_be_turned_off() -> None:
    caps = ProviderCapabilities(repair_tool_pairs=False)
    wire = await _wire([_user("go"), _results(("ghost", "x")), _assistant(_call("c1"))], caps)
    assert [entry["role"] for entry in wire.entries] == ["user", "tool", "assistant"]


# Images --------------------------------------------------------------------


def _image_message(*refs: str, text: str = "look") -> Message:
    return _user(text, metadata={IMAGE_REFS_METADATA_KEY: [{"blob_ref": ref, "mime_type": "image/png"} for ref in refs]})


class _Loader:
    def __init__(self, sizes: dict[str, int] | None = None, mime: str = "image/png") -> None:
        self.sizes = sizes or {}
        self.mime = mime
        self.loaded: list[str] = []

    async def __call__(self, ref: str) -> tuple[bytes, str]:
        self.loaded.append(ref)
        if ref == "broken":
            raise OSError("gone")
        return b"\x89PNG" + b"0" * self.sizes.get(ref, 4), self.mime


async def test_images_are_inlined_when_the_endpoint_takes_them() -> None:
    caps = ProviderCapabilities(images=ImageLimits())
    wire = await _wire([_image_message("blob-1")], caps, image_loader=_Loader())

    content = wire.entries[0]["content"]
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_an_endpoint_without_images_gets_a_note_instead() -> None:
    wire = await _wire([_image_message("blob-1")], image_loader=_Loader())
    assert wire.entries[0] == {
        "role": "user",
        "content": "look\n[image not sent, this model does not take images: blob-1]",
    }


async def test_image_limits_keep_the_most_recent_and_explain_the_rest() -> None:
    loader = _Loader(sizes={"big": 10_000})
    caps = ProviderCapabilities(images=ImageLimits(max_images=3, max_bytes=1_000, mime_types=frozenset({"image/png"})))
    wire = await _wire(
        [
            _image_message("old"),
            Message(
                role=MessageRole.user,
                content_blocks=[ImageRefBlock(blob_ref="big", mime_type="image/png")],
            ),
            _image_message("broken", "new"),
        ],
        caps,
        image_loader=loader,
    )

    old, big, last = (entry["content"] for entry in wire.entries)
    assert "only the most recent images" in old
    assert "over this model's limit of 1000" in big
    assert "could not be loaded" in last[1]["text"]
    assert last[2]["type"] == "image_url"
    assert "old" not in loader.loaded


async def test_an_image_in_a_format_the_endpoint_refuses_is_explained() -> None:
    caps = ProviderCapabilities(images=ImageLimits(mime_types=frozenset({"image/jpeg"})))
    wire = await _wire([_image_message("blob-1")], caps, image_loader=_Loader(mime="image/webp"))
    assert "image/webp is not a format this model takes" in wire.entries[0]["content"]


# Cache control -------------------------------------------------------------


async def test_cache_breakpoints_land_on_the_right_wire_entries() -> None:
    wire = await _wire(
        [
            Message(role=MessageRole.system, content_blocks=[TextBlock(text="system")]),
            _user("q"),
            _assistant(_call("c1")),
            _results(("c1", "r")),
            _image_message("blob-1"),
        ],
        ProviderCapabilities(images=ImageLimits()),
        image_loader=_Loader(),
    )
    apply_cache_control(
        wire.entries,
        [
            CacheBreakpoint(message_index=0, cache_control_type="1h", rationale="system_prefix"),
            CacheBreakpoint(message_index=3, cache_control_type="ephemeral", rationale="tool"),
            CacheBreakpoint(message_index=4, cache_control_type="5m", rationale="last_message"),
            CacheBreakpoint(message_index=2, cache_control_type="ephemeral", rationale="no text"),
            CacheBreakpoint(message_index=99, cache_control_type="ephemeral", rationale="out of range"),
        ],
        wire.index_map,
    )

    assert wire.entries[0]["content"] == [
        {"type": "text", "text": "system", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    assert wire.entries[3]["content"] == "r"
    assert wire.entries[4]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert wire.entries[2]["content"] is None


async def test_cache_breakpoint_on_a_message_with_no_entry_is_skipped() -> None:
    caps = ProviderCapabilities(reasoning_replay=ReasoningReplay.required)
    wire = await _wire([_user("q"), _assistant(reasoning_content="only")], caps, thinking=True)
    apply_cache_control(
        wire.entries, [CacheBreakpoint(message_index=1, cache_control_type="ephemeral", rationale="x")], wire.index_map
    )
    assert wire.entries == [{"role": "user", "content": "q"}]


# Thinking dialects -----------------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "thinking", "effort", "expected"),
    [
        (ThinkingDialect.none, True, "high", {}),
        (ThinkingDialect.openai, True, "high", {"reasoning_effort": "high"}),
        (ThinkingDialect.openai, False, "high", {}),
        (ThinkingDialect.chat_template, True, "high", {"chat_template_kwargs": {"enable_thinking": True}}),
        (ThinkingDialect.chat_template, False, "low", {"chat_template_kwargs": {"enable_thinking": False}}),
        (
            ThinkingDialect.llama_cpp,
            True,
            "low",
            {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}, "reasoning_effort": "low"},
        ),
        (ThinkingDialect.llama_cpp, False, "low", {"chat_template_kwargs": {"enable_thinking": False}}),
        (ThinkingDialect.deepseek, True, "medium", {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
        (ThinkingDialect.deepseek, True, "xhigh", {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}),
        (ThinkingDialect.deepseek, False, "medium", {"thinking": {"type": "disabled"}}),
        (ThinkingDialect.openrouter, True, "low", {"reasoning": {"effort": "low"}}),
        (ThinkingDialect.openrouter, False, "low", {"reasoning": {"enabled": False}}),
    ],
)
def test_thinking_reaches_the_wire_in_each_dialect(
    dialect: ThinkingDialect, thinking: bool, effort: str, expected: dict[str, Any]
) -> None:
    body: dict[str, Any] = {}
    apply_thinking(body, ProviderCapabilities(thinking=dialect), thinking=thinking, effort=effort)
    assert body == expected


def test_no_thinking_switch_on_the_request_leaves_the_server_default() -> None:
    body: dict[str, Any] = {}
    apply_thinking(body, ProviderCapabilities(thinking=ThinkingDialect.deepseek), thinking=None, effort="high")
    assert body == {}


def test_off_under_the_openai_dialect_uses_the_configured_word() -> None:
    body: dict[str, Any] = {}
    caps = ProviderCapabilities(reasoning_off_effort="none")
    apply_thinking(body, caps, thinking=False, effort="high")
    assert body == {"reasoning_effort": "none"}


def test_effort_names_can_be_overridden_and_existing_kwargs_are_kept() -> None:
    body: dict[str, Any] = {"chat_template_kwargs": {"custom": 1}}
    caps = ProviderCapabilities(thinking=ThinkingDialect.llama_cpp, effort_names={"high": "xhigh"})
    apply_thinking(body, caps, thinking=True, effort="high")
    assert body["chat_template_kwargs"] == {"custom": 1, "enable_thinking": True, "reasoning_effort": "xhigh"}
    assert caps.effort_name("low") == "low"


def test_thinking_on_without_an_effort_still_says_on() -> None:
    body: dict[str, Any] = {}
    apply_thinking(body, ProviderCapabilities(thinking=ThinkingDialect.openrouter), thinking=True, effort=None)
    assert body == {"reasoning": {"enabled": True}}


# Tool choice ---------------------------------------------------------------


def test_tool_choice_renders_the_core_constraints() -> None:
    assert tool_choice_for({"forced_tool_choice": "plan"}, CAPS) == {"type": "function", "function": {"name": "plan"}}
    assert tool_choice_for({"tool_choice_required": True}, CAPS) == "required"
    assert tool_choice_for({}, CAPS) is None


def test_a_constraint_the_endpoint_cannot_express_is_left_out() -> None:
    caps = ProviderCapabilities(forced_tool_choice=False, tool_choice_required=False)
    assert tool_choice_for({"forced_tool_choice": "plan"}, caps) is None
    assert tool_choice_for({"tool_choice_required": True}, caps) is None
