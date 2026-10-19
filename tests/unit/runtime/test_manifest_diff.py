"""Which part of a request moved between two calls, read off their manifests.

A cache miss has one cause that matters: the first byte of the request the
provider had not seen before. Two manifests say whether the model, the tool
catalogue, the system prompt or the history changed, and where the history
stopped being a prefix of the next request. A host used to compare the slot
digests by hand and could not say which message broke the prefix.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.llm import LLMRequest
from protocore.contracts.observability import RequestManifest, build_request_manifest
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
)
from protocore.runtime.manifest_diff import ManifestDiff, diff_request_manifests

_SYSTEM = Message(role=MessageRole.system, content_blocks=[TextBlock(text="You are helpful.")])
_TOOLS = [
    ToolDefinition(
        name="Read",
        description="read a file",
        parameters=ToolParameterSchema(properties={"path": {"type": "string"}}),
    )
]


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def _assistant(text: str) -> Message:
    return Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])


def _manifest(
    messages: list[Message],
    *,
    model: str = "model-a",
    tools: list[ToolDefinition] | None = None,
    extra: dict[str, Any] | None = None,
    inline: int = 1_000_000,
) -> tuple[RequestManifest, dict[str, bytes]]:
    request = LLMRequest(
        model=model,
        messages=messages,
        tools=_TOOLS if tools is None else tools,
        extra=extra or {},
    )
    return build_request_manifest(
        request=request,
        attempt_scope="run/turn/run",
        constants_sha256="c" * 64,
        inline_value_max_bytes=inline,
    )


_HISTORY = [_SYSTEM, _user("read the file"), _assistant("reading it")]


def test_a_history_that_only_grew_keeps_the_prefix() -> None:
    previous, _ = _manifest(_HISTORY)
    current, _ = _manifest([*_HISTORY, _user("and the next one")])

    diff = diff_request_manifests(previous, current)

    assert diff.prefix_break is None
    assert diff.first_changed_message is None
    assert diff.history_known is True
    assert "messages" in diff.changed
    assert diff.describe().startswith("prefix kept")


def test_a_rewritten_message_is_named_by_its_index() -> None:
    previous, _ = _manifest(_HISTORY)
    rewritten = [_SYSTEM, _user("read the file"), _assistant("[trimmed]"), _user("next")]
    current, _ = _manifest(rewritten)

    diff = diff_request_manifests(previous, current)

    assert diff.prefix_break == "messages"
    assert diff.first_changed_message == 2
    assert "message 2" in diff.describe()


def test_a_changed_system_prompt_is_the_break() -> None:
    previous, _ = _manifest(_HISTORY)
    other_system = Message(role=MessageRole.system, content_blocks=[TextBlock(text="You are terse.")])
    current, _ = _manifest([other_system, *_HISTORY[1:]])

    diff = diff_request_manifests(previous, current)

    assert diff.prefix_break == "system_prompt"
    assert "system_prompt" in diff.changed


def test_a_changed_catalogue_comes_before_the_system_prompt() -> None:
    previous, _ = _manifest(_HISTORY)
    other_system = Message(role=MessageRole.system, content_blocks=[TextBlock(text="You are terse.")])
    current, _ = _manifest([other_system, *_HISTORY[1:]], tools=[])

    diff = diff_request_manifests(previous, current)

    assert diff.prefix_break == "tools"
    assert diff.changed[:2] == ("tools", "system_prompt")


def test_another_model_shares_no_cache() -> None:
    previous, _ = _manifest(_HISTORY)
    current, _ = _manifest(_HISTORY, model="model-b")

    diff = diff_request_manifests(previous, current)

    assert diff.prefix_break == "model"


def test_identical_requests_change_nothing() -> None:
    previous, _ = _manifest(_HISTORY)
    current, _ = _manifest(_HISTORY)

    diff = diff_request_manifests(previous, current)

    assert diff == ManifestDiff(changed=(), prefix_break=None, first_changed_message=None, history_known=True)


def test_the_volatile_suffix_is_not_part_of_the_prefix() -> None:
    """A turn context that differs on every request must not read as a break."""
    previous, _ = _manifest(
        [*_HISTORY, _user("<turn-context>12:00</turn-context>")],
        extra={"volatile_suffix_messages": 1},
    )
    current, _ = _manifest(
        [*_HISTORY, _user("go on"), _user("<turn-context>12:01</turn-context>")],
        extra={"volatile_suffix_messages": 1},
    )

    diff = diff_request_manifests(previous, current)

    assert diff.prefix_break is None
    assert diff.first_changed_message is None


def test_bodies_stored_outside_the_manifest_can_be_supplied() -> None:
    previous, previous_bodies = _manifest(_HISTORY, inline=0)
    rewritten = [_SYSTEM, _user("read the file"), _assistant("[trimmed]")]
    current, current_bodies = _manifest(rewritten, inline=0)

    unknown = diff_request_manifests(previous, current)
    assert unknown.history_known is False
    assert unknown.prefix_break is None, "without the bodies no break is claimed"
    assert "unknown" in unknown.describe()

    known = diff_request_manifests(
        previous,
        current,
        previous_messages=previous_bodies["messages"],
        current_messages=current_bodies["messages"],
    )
    assert known.history_known is True
    assert known.prefix_break == "messages"
    assert known.first_changed_message == 2


def test_a_shorter_history_breaks_where_it_ends() -> None:
    previous, _ = _manifest([*_HISTORY, _user("more")])
    current, _ = _manifest(_HISTORY[:2])

    diff = diff_request_manifests(previous, current)

    assert diff.prefix_break == "messages"
    assert diff.first_changed_message == 2
