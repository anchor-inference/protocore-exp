"""Recorded chat-completions streams, through the adapter, as the loop sees them."""
from __future__ import annotations

from collections.abc import AsyncIterator

from protocore.conformance import assert_adapter_failure
from protocore.contracts.llm import LLMError, ProviderDeltaKind, ProviderFailureReason
from protocore.providers.openai_compat import (
    REASONING_ENVELOPE,
    ProviderCapabilities,
    StreamAssembler,
    normalize_usage,
)
from protocore.providers.openai_compat.stream import map_finish, sse_events

from ._openai_support import (
    FakeServer,
    chunk_stream,
    collect,
    kinds,
    provider,
    recorded,
    sse_response,
    text_of,
    user_request,
)

K = ProviderDeltaKind


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def test_sse_events_yield_data_comments_and_an_unterminated_last_event() -> None:
    events = [
        event
        async for event in sse_events(
            _lines(": keepalive", "", "data: {\"a\":", "data: 1}", "", "event: ignored", "data:no-space")
        )
    ]
    assert events == [None, '{"a":\n1}', "no-space"]


async def test_openai_tool_calls_stream() -> None:
    server = FakeServer(sse_response(recorded("openai_tool_calls.sse")))
    deltas = await collect(provider(server).stream_with_tools(user_request()))

    assert kinds(deltas) == [
        "text",
        "text",
        "tool_use_start",
        "tool_use_input",
        "tool_use_input",
        "tool_use_start",
        "tool_use_input",
        "tool_use_stop",
        "tool_use_stop",
        "usage",
        "finish",
    ]
    assert text_of(deltas) == "Looking it up."
    stops = [delta for delta in deltas if delta.kind is K.tool_use_stop]
    assert [(stop.tool_call_id, stop.tool_name, stop.tool_input_final) for stop in stops] == [
        ("call_read", "Read", {"path": "README.md"}),
        ("call_list", "List", {}),
    ]
    assert not any(stop.args_partial_truncated or stop.truncated_by_output_cap for stop in stops)
    usage = next(delta.usage for delta in deltas if delta.kind is K.usage)
    assert usage is not None
    assert usage["input_tokens"] == 1200
    assert usage["cache_read_input_tokens"] == 1024
    assert deltas[-1].finish_reason == "tool_use"
    # The role-only first chunk said nothing and kept the idle clock alive.
    assert deltas[0].kind is K.progress


async def test_vllm_reasoning_parser_stream() -> None:
    server = FakeServer(sse_response(recorded("vllm_reasoning.sse")))
    deltas = await collect(provider(server).stream_with_tools(user_request()))

    assert text_of(deltas, K.thinking) == "The user wants a greeting."
    assert text_of(deltas) == "Hello!"
    assert deltas[-1].finish_reason == "stop"
    usage = next(delta.usage for delta in deltas if delta.kind is K.usage)
    assert usage == {
        "input_tokens": 50,
        "output_tokens": 12,
        "cache_read_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 62,
    }


async def test_deepseek_stream_reports_its_cache_hits() -> None:
    server = FakeServer(sse_response(recorded("deepseek_reasoning.sse")))
    deltas = await collect(provider(server).stream_with_tools(user_request()))

    assert text_of(deltas, K.thinking) == "Check the file first."
    assert text_of(deltas) == "Done."
    usage = next(delta.usage for delta in deltas if delta.kind is K.usage)
    assert usage is not None
    assert usage["cache_read_input_tokens"] == 768
    assert usage["reasoning_tokens"] == 20
    assert usage["input_tokens"] == 900


async def test_openrouter_stream_keeps_opaque_reasoning_and_cost() -> None:
    server = FakeServer(sse_response(recorded("openrouter_reasoning_details.sse")))
    deltas = await collect(
        provider(server, capabilities=ProviderCapabilities(replay_reasoning_details=True)).stream_with_tools(
            user_request(model="model-o")
        )
    )

    # The processing comments arrived before any data and count as liveness.
    assert [delta.kind for delta in deltas[:2]] == [K.progress, K.progress]
    assert text_of(deltas, K.thinking) == "Planning the answer."
    finish = deltas[-1]
    assert finish.kind is K.finish
    assert finish.reasoning_items is not None
    [envelope] = finish.reasoning_items
    assert envelope["format"] == REASONING_ENVELOPE
    assert envelope["endpoint"] == "test-endpoint"
    assert envelope["model"] == "model-o"
    assert envelope["tokens"] == 18
    assert envelope["details"][0]["data"] == "opaque-blob-1"
    usage = next(delta.usage for delta in deltas if delta.kind is K.usage)
    assert usage is not None
    assert usage["cost_usd"] == 0.00042


async def test_opaque_reasoning_is_not_kept_unless_asked() -> None:
    server = FakeServer(sse_response(recorded("openrouter_reasoning_details.sse")))
    deltas = await collect(provider(server).stream_with_tools(user_request()))
    assert deltas[-1].reasoning_items is None


async def test_llama_cpp_split_tool_call_with_late_name() -> None:
    server = FakeServer(sse_response(recorded("llamacpp_split_tool.sse")))
    deltas = await collect(provider(server).stream_with_tools(user_request()))

    start = next(delta for delta in deltas if delta.kind is K.tool_use_start)
    assert start.tool_name == "Read"
    assert start.tool_call_id
    inputs = "".join(delta.tool_input_delta or "" for delta in deltas if delta.kind is K.tool_use_input)
    # The fragment that arrived before the name is released with it, in order.
    assert inputs == '{"path": "a.txt"}'
    stop = next(delta for delta in deltas if delta.kind is K.tool_use_stop)
    assert stop.tool_input_final == {"path": "a.txt"}
    assert stop.tool_call_id == start.tool_call_id


async def test_a_length_cut_mid_arguments_sets_both_flags() -> None:
    server = FakeServer(sse_response(recorded("length_cut_tool.sse")))
    deltas = await collect(provider(server).stream_with_tools(user_request()))

    stop = next(delta for delta in deltas if delta.kind is K.tool_use_stop)
    assert stop.truncated_by_output_cap is True
    assert stop.args_partial_truncated is True
    assert stop.tool_input_final is not None
    assert stop.tool_input_final["path"] == "notes.md"
    assert deltas[-1].finish_reason == "length"


async def test_without_repair_a_cut_call_hands_the_loop_no_arguments() -> None:
    server = FakeServer(sse_response(recorded("length_cut_tool.sse")))
    caps = ProviderCapabilities(repair_partial_arguments=False)
    deltas = await collect(provider(server, capabilities=caps).stream_with_tools(user_request()))
    stop = next(delta for delta in deltas if delta.kind is K.tool_use_stop)
    assert stop.tool_input_final == {}
    assert stop.args_partial_truncated is True


async def test_an_error_chunk_after_the_200_is_a_classified_refusal() -> None:
    server = FakeServer(sse_response(recorded("error_after_200.sse")))
    seen: list[str] = []
    try:
        async for delta in provider(server).stream_with_tools(user_request()):
            if delta.content:
                seen.append(delta.content)
    except LLMError as exc:
        failure = assert_adapter_failure(exc)
    else:
        raise AssertionError("the stream error was swallowed")
    assert seen == ["Partial "]
    assert failure.reason is ProviderFailureReason.overloaded
    assert failure.retryable is True


# The assembler on its own --------------------------------------------------


async def test_calls_streamed_without_a_finish_reason_end_the_turn_as_tool_use() -> None:
    """Some servers end a stream of tool calls with no ``finish_reason`` at all.
    The finish was read after the open calls were closed and cleared, so it fell
    back to ``stop`` and the loop took a turn that asked for a tool as finished."""
    body = chunk_stream(
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "Read", "arguments": '{"path": "a"}'}}]}}]},
    )
    deltas = await collect(provider(FakeServer(sse_response(body))).stream_with_tools(user_request()))

    finish = [delta for delta in deltas if delta.kind is K.finish]
    assert [delta.finish_reason for delta in finish] == [map_finish("tool_calls")]


async def test_a_stream_cut_before_its_end_is_a_connection_failure_not_a_tool_call() -> None:
    """The same calls with neither a finish reason nor ``[DONE]``: the connection
    dropped. Reading that as ``tool_calls`` ran a call the model may not have
    finished; reading it as ``stop`` ended the turn as if it were answered."""
    body = chunk_stream(
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "Write", "arguments": '{"path": "a", "content": "par'}}]}}]},
        done=False,
    )
    seen: list[K] = []
    try:
        async for delta in provider(FakeServer(sse_response(body))).stream_with_tools(user_request()):
            seen.append(delta.kind)
    except LLMError as exc:
        failure = assert_adapter_failure(exc)
    else:
        raise AssertionError("a cut stream was read as finished")
    assert failure.reason is ProviderFailureReason.connection
    assert failure.retryable is True
    assert K.tool_use_stop not in seen and K.finish not in seen


async def test_text_cut_before_the_end_of_the_stream_is_a_connection_failure() -> None:
    body = chunk_stream({"choices": [{"index": 0, "delta": {"content": "Half an ans"}}]}, done=False)
    try:
        await collect(provider(FakeServer(sse_response(body))).stream_with_tools(user_request()))
    except LLMError as exc:
        assert assert_adapter_failure(exc).reason is ProviderFailureReason.connection
    else:
        raise AssertionError("a cut stream was read as finished")


async def test_a_finish_reason_without_done_still_ends_the_turn() -> None:
    body = chunk_stream({"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}, done=False)
    deltas = await collect(provider(FakeServer(sse_response(body))).stream_with_tools(user_request()))
    assert [delta.finish_reason for delta in deltas if delta.kind is K.finish] == [map_finish("stop")]


def test_calls_numbered_alike_are_told_apart_by_id() -> None:
    assembler = StreamAssembler()
    assembler.feed({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "One", "arguments": "{}"}}]}}]})
    assembler.feed({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "b", "function": {"name": "Two", "arguments": "{}"}}]}}]})
    stops = assembler.close()
    assert [(stop.tool_call_id, stop.tool_name) for stop in stops] == [("a", "One"), ("b", "Two")]


def test_a_call_that_never_got_a_name_still_opens_and_closes() -> None:
    assembler = StreamAssembler()
    deltas = assembler.feed({"choices": [{"delta": {"tool_calls": [{"function": {"arguments": '{"x": 1}'}}]}}]})
    assert deltas == []
    assert assembler.has_open_calls
    start, stop = assembler.close()
    assert start.kind is K.tool_use_start and start.tool_name == "unknown"
    assert stop.tool_input_final == {"x": 1}
    assert not assembler.has_open_calls


def test_arguments_wrapped_in_prose_are_whole_and_a_bare_value_is_wrapped() -> None:
    assembler = StreamAssembler()
    assembler.feed({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "a", "function": {"name": "A", "arguments": '```json\n{"k": "v"}\n```'}},
        {"index": 1, "id": "b", "function": {"name": "B", "arguments": "[1, 2]"}},
        {"index": 2, "id": "c", "function": {"name": "C", "arguments": "not json"}},
    ]}}]})
    first, second, third = (stop for stop in assembler.close() if stop.kind is K.tool_use_stop)
    assert first.tool_input_final == {"k": "v"} and first.args_partial_truncated is False
    assert second.tool_input_final == {"value": [1, 2]}
    assert third.tool_input_final == {} and third.args_partial_truncated is True


def test_odd_chunks_are_tolerated() -> None:
    assembler = StreamAssembler()
    assert [delta.kind for delta in assembler.feed({"choices": ["junk", {"delta": {}}]})] == [K.progress]
    assert assembler.feed({"choices": [], "usage": {"prompt_tokens": 3}}) == []
    assert assembler.usage == {"prompt_tokens": 3}
    assert assembler.resolved_finish() == "stop"


def test_usage_normalisation_handles_every_shape() -> None:
    assert normalize_usage({}) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    anthropic_shaped = normalize_usage(
        {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 7, "cache_creation_input_tokens": 3}
    )
    assert anthropic_shaped["cache_read_input_tokens"] == 7
    assert anthropic_shaped["cache_creation_input_tokens"] == 3
    assert anthropic_shaped["total_tokens"] == 15
    odd = normalize_usage({"prompt_tokens": "x", "completion_tokens": True, "cost": True, "prompt_tokens_details": []})
    assert odd["input_tokens"] == 0 and odd["output_tokens"] == 0 and "cost_usd" not in odd


def test_unknown_finish_reasons_read_as_a_stop() -> None:
    assert map_finish("function_call") == "tool_use"
    assert map_finish("eos_token") == "stop"
    assert map_finish(None) == "stop"
