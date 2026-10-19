"""Whole runs on the reference chat-completions adapter, against a fake server.

The unit tests pin each rule of the adapter on its own; these drive the loop
through it, so the deltas it produces are the ones the loop actually consumes
and the second request is built from the history the first one left.
"""
from __future__ import annotations

import json

import httpx

from protocore.providers.chain import ChainRung, ListProviderChain
from protocore.providers.openai_compat import EndpointSpec, OpenAICompatibleProvider
from protocore.runtime.events import EventType

from ..unit.providers._openai_support import FakeServer, chunk_stream, sse_response
from .conftest import ScenarioFactory, ScriptedTool, default_rc


def _adapter(server: FakeServer, name: str = "primary", **spec: object) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        EndpointSpec(name=name, base_url="http://127.0.0.1:9/v1", **spec),  # type: ignore[arg-type]
        client=server.client(),
    )


def _tool_call_turn() -> bytes:
    return chunk_stream(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "Note", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"v": "x"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 10}},
    )


def _answer_turn(text: str = "All noted.") -> bytes:
    return chunk_stream(
        {"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 320, "completion_tokens": 4}},
    )


async def test_a_tool_round_trip_through_the_adapter(scenario: ScenarioFactory) -> None:
    server = FakeServer(sse_response(_tool_call_turn()), sse_response(_answer_turn()))
    note = ScriptedTool(tool_name="Note", content="noted")
    run = scenario(
        llm_provider=_adapter(server, cache_key_field="prompt_cache_key"),
        tools=[note],
        cache_key="sess-source",
    )

    produced = await run.run("take a note")

    assert note.invocations == [{"v": "x"}]
    assert [evt for evt in produced if evt.type is EventType.MESSAGE_STOP]
    first, second = server.bodies()
    assert first["prompt_cache_key"] == second["prompt_cache_key"] == "sess-source"
    assert any(tool["function"]["name"] == "Note" for tool in first["tools"])
    assistant = next(entry for entry in second["messages"] if entry.get("tool_calls"))
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"v": "x"}
    result = next(entry for entry in second["messages"] if entry["role"] == "tool")
    assert result["tool_call_id"] == "call_1"
    assert "noted" in result["content"]
    assert "All noted." in "".join(run.history_texts())


async def test_a_dead_endpoint_moves_the_run_to_the_next_rung(scenario: ScenarioFactory) -> None:
    async def refused(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    dead = FakeServer(refused, refused, refused, refused)
    healthy = FakeServer(sse_response(_answer_turn("Served by the fallback.")))
    primary = _adapter(dead)
    chain = ListProviderChain(
        [
            ChainRung(name="primary", model="model-a", provider=primary),
            ChainRung(name="fallback", model="model-b", provider=_adapter(healthy, name="fallback")),
        ]
    )
    run = scenario(
        rc=default_rc(llm_transient_error_retry_backoff_base_seconds=0.0),
        llm_provider=primary,
        provider_chain=chain,
        model_name="model-a",
    )

    await run.run("hello")

    assert "Served by the fallback." in "".join(run.history_texts())
    assert chain.current_model_name() == "model-b"
    assert chain.attempted()[0][0] == "primary"
    assert healthy.bodies()[0]["model"] == "model-b"


async def test_a_fallback_too_small_for_the_prompt_is_passed_over(scenario: ScenarioFactory) -> None:
    async def refused(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    dead = FakeServer(refused, refused, refused, refused)
    small = FakeServer()
    large = FakeServer(sse_response(_answer_turn("Served by the large rung.")))
    primary = _adapter(dead)
    chain = ListProviderChain(
        [
            ChainRung(name="primary", model="model-a", provider=primary),
            ChainRung(name="small", model="model-s", provider=_adapter(small, name="small"), context_window=200, max_output_tokens=50),
            ChainRung(name="large", model="model-l", provider=_adapter(large, name="large"), context_window=100_000),
        ]
    )
    run = scenario(
        rc=default_rc(llm_transient_error_retry_backoff_base_seconds=0.0),
        llm_provider=primary,
        provider_chain=chain,
        model_name="model-a",
    )
    chain.bind_prompt_size(run.engine.prompt_tokens_now)

    await run.run("a question with a long paste in it " * 60)

    assert small.requests == []
    assert chain.current_model_name() == "model-l"
    assert "small" in dict(chain.attempted())
