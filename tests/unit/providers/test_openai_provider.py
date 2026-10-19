"""The adapter's request side, transport failures, watchdog and non-streamed calls."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from protocore.conformance import LLMProviderConformance, assert_adapter_failure
from protocore.contracts.llm import (
    CacheBreakpoint,
    ILLMProvider,
    LLMContextWindowExceeded,
    LLMObservabilityContext,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    ProviderDeltaKind,
    ProviderFailureReason,
)
from protocore.contracts.types import StopReason, ToolDefinition, ToolParameterSchema
from protocore.providers.openai_compat import (
    EndpointSpec,
    FailureObservation,
    OpenAICompatibleProvider,
    ProviderCapabilities,
    ThinkingDialect,
    UsageObservation,
)

from ._openai_support import (
    FakeServer,
    chunk_stream,
    collect,
    provider,
    recorded,
    sse_response,
    user_request,
)

R = ProviderFailureReason

_DONE = chunk_stream({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})


class TestOpenAICompatibleProviderConformance(LLMProviderConformance):
    @pytest.fixture
    def subject_factory(self) -> Callable[[], Any]:
        return lambda: provider(FakeServer())


def test_is_the_core_contract() -> None:
    assert isinstance(provider(FakeServer()), ILLMProvider)


# Endpoint validation -------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_url": "ftp://host"}, "http"),
        ({"timeout_seconds": 0}, "positive"),
        ({"first_byte_timeout_seconds": -1}, "negative"),
        ({"first_byte_retries": 4}, "between 0 and 3"),
    ],
)
def test_an_endpoint_refuses_settings_that_cannot_work(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EndpointSpec(name="e", base_url=overrides.pop("base_url", "http://127.0.0.1:1/v1"), **overrides)


def test_the_key_stays_out_of_repr() -> None:
    assert "secret-value" not in repr(EndpointSpec(name="e", base_url="http://127.0.0.1:1/v1", api_key="secret-value"))


# The request -----------------------------------------------------------------


async def test_the_body_carries_what_the_request_asks_for() -> None:
    server = FakeServer(sse_response(_DONE))
    caps = ProviderCapabilities(thinking=ThinkingDialect.chat_template, cache_control=True, cache_prompt=True)
    adapter = provider(
        server,
        capabilities=caps,
        api_key="k-123",
        headers={"x-extra": "1"},
        cache_key_field="prompt_cache_key",
        cache_key_header="x-session-affinity",
        extra_body={"top_k": 20, "model": "never-wins"},
    )
    tool = ToolDefinition(
        name="plan",
        description="d",
        parameters=ToolParameterSchema(properties={"a": {"type": "string"}}, additional_properties=False),
    )
    request = user_request(
        tools=[tool],
        max_tokens=512,
        temperature=0.3,
        extra={
            "forced_tool_choice": "plan",
            "enable_thinking": True,
            "reasoning_effort": "low",
            "cache_breakpoints": [CacheBreakpoint(message_index=0, cache_control_type="ephemeral", rationale="last")],
        },
        observability=LLMObservabilityContext(session_id="sess-fork", cache_key="sess-source"),
    )

    await collect(adapter.stream_with_tools(request))

    [body] = server.bodies()
    assert body["model"] == "model-a"
    assert body["max_tokens"] == 512
    assert body["temperature"] == 0.3
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert body["tool_choice"] == {"type": "function", "function": {"name": "plan"}}
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["prompt_cache_key"] == "sess-source"
    assert body["cache_prompt"] is True
    assert body["top_k"] == 20
    [sent] = server.requests
    assert sent.url == "http://127.0.0.1:8000/v1/chat/completions"
    assert sent.headers["authorization"] == "Bearer k-123"
    assert sent.headers["x-extra"] == "1"
    assert sent.headers["x-session-affinity"] == "sess-source"


async def test_a_plain_request_sends_only_the_plain_fields() -> None:
    server = FakeServer(sse_response(_DONE))
    await collect(provider(server).stream_with_tools(user_request(observability=None)))

    [body] = server.bodies()
    assert set(body) == {"model", "messages", "max_tokens", "stream", "stream_options"}
    assert "authorization" not in server.requests[0].headers


async def test_routing_falls_back_to_the_session_and_a_pinned_temperature_wins() -> None:
    server = FakeServer(sse_response(_DONE))
    caps = ProviderCapabilities(max_tokens_field="max_completion_tokens", stream_usage=False)
    adapter = provider(server, capabilities=caps, cache_key_field="prompt_cache_key", temperature=0.0)
    await collect(adapter.stream_with_tools(user_request(temperature=0.9, extra={"response_format": {"type": "json_object"}})))

    [body] = server.bodies()
    assert body["prompt_cache_key"] == "sess-1"
    assert body["temperature"] == 0.0
    assert body["max_completion_tokens"] == 4096
    assert "max_tokens" not in body and "stream_options" not in body
    assert body["response_format"] == {"type": "json_object"}


# Refusals and transport failures ------------------------------------------------


async def test_a_status_refusal_is_raised_with_its_verdict_and_observed() -> None:
    seen: list[FailureObservation] = []

    async def on_failure(observation: FailureObservation) -> None:
        seen.append(observation)

    server = FakeServer(
        httpx.Response(
            429,
            json={"error": {"message": "slow down", "type": "rate_limit_error"}},
            headers={"retry-after": "7", "set-cookie": "secret"},
        )
    )
    with pytest.raises(LLMRateLimitError) as caught:
        await collect(provider(server, on_failure=on_failure).stream_with_tools(user_request()))

    failure = assert_adapter_failure(caught.value)
    assert failure.retry_after_seconds == 7.0
    [observation] = seen
    assert observation.status == 429
    assert observation.failure is not None and observation.failure.reason is R.rate_limit
    assert observation.headers == {"retry-after": "7"}
    assert observation.observability is not None


async def test_a_refused_constrained_request_offers_the_prompted_fallback() -> None:
    server = FakeServer(httpx.Response(400, json={"error": {"message": "tool_choice is not supported"}}))
    request = user_request(extra={"tool_choice_required": True}, tools=[
        ToolDefinition(name="t", description="d", parameters=ToolParameterSchema())
    ])
    with pytest.raises(LLMProviderError) as caught:
        await collect(provider(server).stream_with_tools(request))
    assert assert_adapter_failure(caught.value).should_fallback is True


async def test_an_overflow_is_raised_for_the_loop_to_compact() -> None:
    server = FakeServer(
        httpx.Response(400, json={"error": {"message": "maximum context length is 8192 tokens. However, your messages resulted in 9000 tokens."}})
    )
    seen: list[FailureObservation] = []

    async def on_failure(observation: FailureObservation) -> None:
        seen.append(observation)
        raise RuntimeError("an observer that breaks must not hide the refusal")

    with pytest.raises(LLMContextWindowExceeded) as caught:
        await collect(provider(server, on_failure=on_failure).stream_with_tools(user_request()))
    assert caught.value.context_window == 8192
    assert seen[0].failure is not None and seen[0].failure.reason is R.context_overflow


@pytest.mark.parametrize(
    ("exc", "error_type", "reason"),
    [
        (httpx.ConnectError("connection refused"), LLMProviderError, R.connection),
        (httpx.RemoteProtocolError("peer closed connection"), LLMProviderError, R.connection),
        (httpx.ReadTimeout("read timed out"), LLMTimeoutError, R.timeout),
        (httpx.DecodingError("bad gzip"), LLMProviderError, R.unknown),
    ],
)
async def test_a_request_that_got_no_answer_carries_a_verdict(
    exc: httpx.HTTPError, error_type: type[Exception], reason: ProviderFailureReason
) -> None:
    async def fail(_request: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(error_type) as caught:
        await collect(provider(FakeServer(fail)).stream_with_tools(user_request()))
    failure = assert_adapter_failure(caught.value)
    assert failure.reason is reason
    assert failure.retryable is True


async def test_a_refused_socket_under_the_client_error_is_read_from_the_cause() -> None:
    async def fail(_request: httpx.Request) -> httpx.Response:
        try:
            raise ConnectionRefusedError(111, "Connection refused")
        except ConnectionRefusedError as cause:
            raise httpx.ConnectError("all connection attempts failed") from cause

    with pytest.raises(LLMProviderError) as caught:
        await collect(provider(FakeServer(fail)).stream_with_tools(user_request()))
    assert assert_adapter_failure(caught.value).reason is R.connection


async def test_a_stream_that_breaks_midway_is_a_connection_failure() -> None:
    async def body() -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"half"}}]}\n\n'
        raise httpx.ReadError("connection reset")

    server = FakeServer(httpx.Response(200, content=body()))
    received: list[str] = []
    with pytest.raises(LLMProviderError) as caught:
        async for delta in provider(server).stream_with_tools(user_request()):
            if delta.content:
                received.append(delta.content)
    assert received == ["half"]
    assert assert_adapter_failure(caught.value).reason is R.connection


async def test_unreadable_events_are_skipped() -> None:
    body = b'data: not json\n\ndata: [1]\n\ndata: {"choices":[{"delta":{"content":"fine"},"finish_reason":"stop"}]}\n\n'
    deltas = await collect(provider(FakeServer(sse_response(body))).stream_with_tools(user_request()))
    assert [delta.content for delta in deltas if delta.kind is ProviderDeltaKind.text] == ["fine"]


# First-byte watchdog ---------------------------------------------------------


async def test_a_silent_request_is_sent_again_byte_for_byte() -> None:
    async def silent(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        raise AssertionError("the watchdog should have given up on this attempt")

    server = FakeServer(silent, sse_response(recorded("vllm_reasoning.sse")))
    adapter = provider(server, first_byte_timeout_seconds=0.2, first_byte_retries=1)

    deltas = await collect(adapter.stream_with_tools(user_request()))

    assert len(server.requests) == 2
    assert server.requests[0].content == server.requests[1].content
    # The core's idle clock heard about the resend before the second window.
    assert deltas[0].kind is ProviderDeltaKind.progress
    assert deltas[-1].kind is ProviderDeltaKind.finish


async def test_headers_without_a_first_line_count_as_silence() -> None:
    async def stalled_body() -> AsyncIterator[bytes]:
        await asyncio.sleep(30)
        yield b""

    server = FakeServer(httpx.Response(200, content=stalled_body()), sse_response(_DONE))
    adapter = provider(server, first_byte_timeout_seconds=0.2)

    deltas = await collect(adapter.stream_with_tools(user_request()))

    assert len(server.requests) == 2
    assert deltas[-1].finish_reason == "stop"


async def test_silence_on_every_attempt_is_a_timeout_that_moves_the_chain() -> None:
    async def silent(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        raise AssertionError("unreachable")

    server = FakeServer(silent, silent)
    adapter = provider(server, first_byte_timeout_seconds=0.1, first_byte_retries=1)

    with pytest.raises(LLMTimeoutError) as caught:
        await collect(adapter.stream_with_tools(user_request()))
    assert assert_adapter_failure(caught.value).reason is R.timeout
    assert len(server.requests) == 2


async def test_a_refusal_is_an_answer_and_is_never_resent() -> None:
    server = FakeServer(httpx.Response(401, json={"error": {"message": "bad key"}}))
    adapter = provider(server, first_byte_timeout_seconds=0.2, first_byte_retries=3)
    with pytest.raises(LLMProviderError):
        await collect(adapter.stream_with_tools(user_request()))
    assert len(server.requests) == 1


async def test_an_empty_body_ends_the_stream_cleanly() -> None:
    deltas = await collect(provider(FakeServer(sse_response(b""))).stream_with_tools(user_request()))
    assert [delta.kind for delta in deltas] == [ProviderDeltaKind.usage, ProviderDeltaKind.finish]


async def test_a_timeout_of_the_caller_is_not_mistaken_for_the_watchdog() -> None:
    async def slow(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        raise AssertionError("unreachable")

    adapter = provider(FakeServer(slow), first_byte_timeout_seconds=10)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.1):
            await collect(adapter.stream_with_tools(user_request()))


# Non-streamed calls and observers --------------------------------------------


def _completion(content: Any, finish: str = "stop", **extra: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 16}},
            **extra,
        },
    )


async def test_complete_text_returns_the_reply_and_its_usage() -> None:
    usages: list[UsageObservation] = []

    async def on_usage(observation: UsageObservation) -> None:
        usages.append(observation)

    server = FakeServer(_completion("A summary."))
    response = await provider(server, on_usage=on_usage).complete_text(user_request())

    assert response.message.text == "A summary."
    assert response.stop_reason is StopReason.end_turn
    assert response.usage.input_tokens == 20
    assert response.usage.cache_read_input_tokens == 16
    assert server.bodies()[0]["stream"] is False
    [usage] = usages
    assert usage.purpose == "text"
    assert usage.normalized["output_tokens"] == 5


async def test_complete_text_reads_content_parts_and_a_length_cut() -> None:
    server = FakeServer(_completion([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, "x"], finish="length"))
    response = await provider(server).complete_text(user_request())
    assert response.message.text == "ab"
    assert response.stop_reason is StopReason.max_tokens


async def test_complete_structured_returns_the_object_as_json() -> None:
    server = FakeServer(_completion('Here it is:\n```json\n{"answer": 42}\n```'))
    response = await provider(server).complete_structured(user_request(), {"type": "object"})
    assert json.loads(response.message.text) == {"answer": 42}
    assert server.bodies()[0]["response_format"] == {"type": "json_object"}


async def test_complete_structured_asks_for_the_schema_where_the_endpoint_takes_it() -> None:
    server = FakeServer(_completion('{"a": 1}'))
    adapter = provider(server, capabilities=ProviderCapabilities(json_schema_response=True))
    await adapter.complete_structured(user_request(), {"type": "object", "properties": {"a": {"type": "integer"}}})
    assert server.bodies()[0]["response_format"]["type"] == "json_schema"
    assert server.bodies()[0]["response_format"]["json_schema"]["schema"]["properties"] == {"a": {"type": "integer"}}


async def test_a_structured_reply_without_json_offers_the_fallback() -> None:
    server = FakeServer(_completion("I cannot answer that.", finish="length"))
    with pytest.raises(LLMProviderError) as caught:
        await provider(server).complete_structured(user_request(), {"type": "object"})
    failure = assert_adapter_failure(caught.value)
    assert failure.should_fallback is True
    assert "cut by the output cap" in str(caught.value)


async def test_non_streamed_refusals_and_odd_replies() -> None:
    refused = FakeServer(httpx.Response(503, text="busy"))
    with pytest.raises(LLMProviderError) as caught:
        await provider(refused).complete_text(user_request())
    assert assert_adapter_failure(caught.value).reason is R.overloaded

    in_body = FakeServer(httpx.Response(200, json={"error": {"message": "quota", "code": "insufficient_quota"}}))
    with pytest.raises(LLMProviderError) as caught:
        await provider(in_body).complete_text(user_request())
    assert assert_adapter_failure(caught.value).reason is R.billing

    not_json = FakeServer(httpx.Response(200, text="<html>"))
    with pytest.raises(LLMProviderError) as caught:
        await provider(not_json).complete_text(user_request())
    assert assert_adapter_failure(caught.value).reason is R.server_error

    async def refused_socket(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMProviderError):
        await provider(FakeServer(refused_socket)).complete_text(user_request())

    odd = FakeServer(httpx.Response(200, json=[1, 2]), httpx.Response(200, json={"choices": ["x"]}))
    adapter = provider(odd)
    assert (await adapter.complete_text(user_request())).message.text == ""
    assert (await adapter.complete_text(user_request())).message.text == ""


async def test_a_failing_usage_observer_does_not_fail_the_answer() -> None:
    async def on_usage(_observation: UsageObservation) -> None:
        raise RuntimeError("ledger down")

    deltas = await collect(provider(FakeServer(sse_response(_DONE)), on_usage=on_usage).stream_with_tools(user_request()))
    assert deltas[-1].kind is ProviderDeltaKind.finish


async def test_count_tokens_uses_the_core_estimate() -> None:
    adapter = provider(FakeServer())
    assert adapter.count_tokens("") == 0
    assert adapter.count_tokens("a few words of English text") > 0


async def test_an_owned_client_is_closed_and_a_borrowed_one_is_not() -> None:
    owned = OpenAICompatibleProvider(EndpointSpec(name="e", base_url="http://127.0.0.1:1/v1"))
    async with owned:
        pass
    assert owned._client.is_closed

    server = FakeServer()
    client = server.client()
    borrowed = OpenAICompatibleProvider(EndpointSpec(name="e", base_url="http://127.0.0.1:1/v1"), client=client)
    await borrowed.aclose()
    assert not client.is_closed
    await client.aclose()
