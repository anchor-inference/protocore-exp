"""A provider that repeats a tool call id must not cost the model a call.

Some models write their own call ids and count them per response, or from
whatever history they were shown: Kimi names its calls ``functions.Write:0``,
``functions.Read:1`` and starts again from ``:0`` after the context is
compacted. The loop pairs every result with its call by id, so a repeated id
used to make the pairing repair keep the first call with that id and drop the
later one together with its result: the effect stayed on disk and the model
never saw it. Every call now gets an id of its own within the conversation the
moment its response arrives, and each call and its own result reach the next
request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from protocore.contracts.tools import ToolContext
from protocore.contracts.types import (
    Message,
    MessageRole,
    TextBlock,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.providers.openai_compat import EndpointSpec, OpenAICompatibleProvider
from protocore.runtime.events import EventType

from ..unit.providers._openai_support import FakeServer, chunk_stream, sse_response
from .conftest import ScenarioFactory, ScriptedTool

KIMI_ID = "functions.Note:0"


@dataclass
class EchoTool(ScriptedTool):
    """A tool whose result names the call it answers, so a pairing is checkable."""

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(arguments))
        return ToolResult(tool_call_id="", content=f"noted {arguments.get('v')}")


def _pairs(messages: list[Message]) -> list[tuple[str, str, str]]:
    """``(call id, argument v, result text)`` for every call in ``messages``."""
    results = {
        block.tool_call_id: block.content
        for message in messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }
    return [
        (block.tool_call_id, json.loads(block.arguments_json)["v"], results.get(block.tool_call_id, ""))
        for message in messages
        for block in message.content_blocks
        if isinstance(block, ToolUseBlock)
    ]


def _assert_every_call_kept_with_its_own_result(messages: list[Message], values: list[str]) -> None:
    pairs = _pairs(messages)
    assert [v for _, v, _ in pairs] == values
    ids = [call_id for call_id, _, _ in pairs]
    assert len(set(ids)) == len(ids), ids
    for _, v, result in pairs:
        assert result == f"noted {v}"


async def test_a_repeated_id_across_two_responses_keeps_both_calls(scenario: ScenarioFactory) -> None:
    note = EchoTool(tool_name="Note")
    run = scenario(tools=[note])
    run.llm.queue_tool_call_response(tool_call_id=KIMI_ID, tool_name="Note", tool_input={"v": "a"})
    run.llm.queue_tool_call_response(tool_call_id=KIMI_ID, tool_name="Note", tool_input={"v": "b"})
    run.llm.queue_response(text="done")

    await run.run("take two notes")

    assert note.invocations == [{"v": "a"}, {"v": "b"}]
    _assert_every_call_kept_with_its_own_result(list(run.requests[2].messages), ["a", "b"])
    # The provider's own id is kept where it was unique; only the repeat is renamed.
    assert run.tool_uses()[0].tool_call_id == KIMI_ID


async def test_a_repeated_id_inside_one_response_keeps_both_calls(scenario: ScenarioFactory) -> None:
    note = EchoTool(tool_name="Note")
    run = scenario(tools=[note])
    run.llm.queue_multi_tool_call_response(tool_calls=[(KIMI_ID, "Note", {"v": "a"}), (KIMI_ID, "Note", {"v": "b"})])
    run.llm.queue_response(text="done")

    await run.run("take two notes")

    assert sorted(call["v"] for call in note.invocations) == ["a", "b"]
    _assert_every_call_kept_with_its_own_result(list(run.requests[1].messages), ["a", "b"])


async def test_the_reader_sees_the_same_id_the_history_keeps(scenario: ScenarioFactory) -> None:
    """The host draws its live rows from the events and its transcript from the history."""
    note = EchoTool(tool_name="Note")
    run = scenario(tools=[note])
    run.llm.queue_tool_call_response(tool_call_id=KIMI_ID, tool_name="Note", tool_input={"v": "a"})
    run.llm.queue_tool_call_response(tool_call_id=KIMI_ID, tool_name="Note", tool_input={"v": "b"})
    run.llm.queue_response(text="done")

    await run.run("take two notes")

    kept = [block.tool_call_id for block in run.tool_uses()]
    assert [str(evt.payload["tool_call_id"]) for evt in run.events_of(EventType.TOOL_USE_START)] == kept
    assert [str(evt.payload["tool_call_id"]) for evt in run.events_of(EventType.TOOL_USE_STOP)] == kept
    assert [str(evt.payload["tool_call_id"]) for evt in run.events_of(EventType.TOOL_RESULT)] == kept
    assert [block.tool_call_id for block in run.tool_results()] == kept


async def test_an_id_the_host_reserved_is_not_issued_again(scenario: ScenarioFactory) -> None:
    """An id from before a compaction is gone from the history but not from the transcript."""
    note = EchoTool(tool_name="Note")
    run = scenario(tools=[note])
    run.engine.reserve_tool_call_ids([KIMI_ID])
    run.llm.queue_tool_call_response(tool_call_id=KIMI_ID, tool_name="Note", tool_input={"v": "a"})
    run.llm.queue_response(text="done")

    await run.run("take a note")

    assert [block.tool_call_id for block in run.tool_uses()] != [KIMI_ID]


async def test_a_stored_history_with_a_repeated_id_loses_nothing(scenario: ScenarioFactory) -> None:
    """A conversation stored before ids were made unique is read back whole."""
    run = scenario(tools=[EchoTool(tool_name="Note")])
    for value in ("a", "b"):
        run.engine.history.append(
            Message(
                role=MessageRole.assistant,
                content_blocks=[ToolUseBlock(tool_call_id=KIMI_ID, name="Note", arguments_json=json.dumps({"v": value}))],
            )
        )
        run.engine.history.append(
            Message(role=MessageRole.tool, content_blocks=[ToolResultBlock(tool_call_id=KIMI_ID, content=f"noted {value}")])
        )
    run.engine.history.insert(0, Message(role=MessageRole.user, content_blocks=[TextBlock(text="take two notes")]))
    run.llm.queue_response(text="done")

    await run.run("and now?")

    _assert_every_call_kept_with_its_own_result(list(run.requests[0].messages), ["a", "b"])


def _adapter(server: FakeServer) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        EndpointSpec(name="kimi", base_url="http://127.0.0.1:9/v1"),
        client=server.client(),
    )


def _two_calls_one_id() -> bytes:
    """Two calls at different indexes that share one id, the way a per-response counter collides."""
    return chunk_stream(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": KIMI_ID, "function": {"name": "Note", "arguments": '{"v": "a"}'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": KIMI_ID, "function": {"name": "Note", "arguments": '{"v": "b"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 10}},
    )


def _answer() -> bytes:
    return chunk_stream(
        {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 320, "completion_tokens": 4}},
    )


async def test_the_wire_carries_each_call_and_its_own_result(scenario: ScenarioFactory) -> None:
    server = FakeServer(sse_response(_two_calls_one_id()), sse_response(_answer()))
    note = EchoTool(tool_name="Note")
    run = scenario(llm_provider=_adapter(server), tools=[note])

    await run.run("take two notes")

    _, second = server.bodies()
    calls = [call for entry in second["messages"] for call in entry.get("tool_calls") or []]
    ids = [call["id"] for call in calls]
    assert ids[0] == KIMI_ID
    assert len(set(ids)) == 2
    # The renamed id keeps the provider's shape: Kimi reads ``functions.<name>:<n>``.
    assert ids[1].startswith("functions.Note:") and ids[1].rsplit(":", 1)[1].isdigit()
    results = {entry["tool_call_id"]: entry["content"] for entry in second["messages"] if entry["role"] == "tool"}
    for call in calls:
        assert f"noted {json.loads(call['function']['arguments'])['v']}" in results[call["id"]]
