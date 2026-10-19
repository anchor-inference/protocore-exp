"""An exact prompt size from the server's counting route (``IRequestTokenCounter``)."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from protocore.conformance import RequestTokenCounterConformance
from protocore.contracts.llm import IRequestTokenCounter, LLMProviderError
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import ToolDefinition, ToolParameterSchema
from protocore.providers.openai_compat import ProviderCapabilities, ThinkingDialect, TokenizeDialect
from protocore.providers.openai_compat.tokenize import TokenizeCounter
from protocore.runtime.request_budget import count_request_tokens_exactly

from ._openai_support import FakeServer, endpoint, provider, user_request


class TestCountingProviderConformance(RequestTokenCounterConformance):
    @pytest.fixture
    def subject_factory(self) -> Callable[[], Any]:
        return lambda: provider(FakeServer(), tokenize=TokenizeDialect.vllm)


def _tool() -> ToolDefinition:
    return ToolDefinition(name="Read", description="read", parameters=ToolParameterSchema())


async def test_vllm_counts_the_chat_request_as_it_will_be_sent() -> None:
    server = FakeServer(httpx.Response(200, json={"count": 321, "max_model_len": 32768, "tokens": [1, 2]}))
    adapter = provider(
        server,
        tokenize=TokenizeDialect.vllm,
        api_key="k",
        capabilities=ProviderCapabilities(thinking=ThinkingDialect.chat_template),
    )
    request = user_request(tools=[_tool()], extra={"enable_thinking": False, "reasoning_effort": "low"})

    assert isinstance(adapter, IRequestTokenCounter)
    assert await adapter.count_request_tokens(request) == 321

    [sent] = server.requests
    # Mounted at the server root, beside /v1, not under it.
    assert sent.url == "http://127.0.0.1:8000/tokenize"
    assert sent.headers["authorization"] == "Bearer k"
    payload = json.loads(sent.content)
    assert payload["model"] == "model-a"
    assert payload["add_generation_prompt"] is True
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["tools"][0]["function"]["name"] == "Read"
    # What the template renders depends on the thinking switch, so it is counted too.
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


async def test_vllm_without_a_count_field_counts_the_token_list() -> None:
    server = FakeServer(httpx.Response(200, json={"tokens": [5, 6, 7]}), httpx.Response(200, json={}))
    adapter = provider(server, tokenize=TokenizeDialect.vllm)
    assert await adapter.count_request_tokens(user_request()) == 3
    assert await adapter.count_request_tokens(user_request()) is None


async def test_llama_cpp_renders_the_template_then_counts_the_prompt() -> None:
    server = FakeServer(
        httpx.Response(200, json={"prompt": "<|user|>hello<|assistant|>"}),
        httpx.Response(200, json={"tokens": [1, 2, 3, 4, 5]}),
    )
    adapter = provider(server, tokenize=TokenizeDialect.llama_cpp, tokenize_url="http://127.0.0.1:8080/")

    assert await adapter.count_request_tokens(user_request()) == 5

    render, count = server.requests
    assert render.url == "http://127.0.0.1:8080/apply-template"
    assert json.loads(render.content) == {"messages": [{"role": "user", "content": "hello"}]}
    assert count.url == "http://127.0.0.1:8080/tokenize"
    assert json.loads(count.content) == {"content": "<|user|>hello<|assistant|>", "add_special": True}


async def test_a_server_without_the_route_stops_being_asked() -> None:
    server = FakeServer(httpx.Response(404, text="Not Found"))
    adapter = provider(server, tokenize=TokenizeDialect.vllm)

    assert await adapter.count_request_tokens(user_request()) is None
    assert await adapter.count_request_tokens(user_request()) is None
    assert len(server.requests) == 1


async def test_no_counting_route_configured_means_no_count() -> None:
    server = FakeServer()
    assert await provider(server).count_request_tokens(user_request()) is None
    assert server.requests == []


@pytest.mark.parametrize(
    "responses",
    [
        [httpx.Response(500, text="boom")],
        [httpx.Response(200, text="not json")],
        [httpx.Response(200, json=[1])],
        [httpx.Response(200, json={"no_prompt": True})],
    ],
)
async def test_a_count_that_failed_is_raised_for_the_core_to_log(responses: list[httpx.Response]) -> None:
    adapter = provider(FakeServer(*responses), tokenize=TokenizeDialect.llama_cpp)
    with pytest.raises(LLMProviderError):
        await adapter.count_request_tokens(user_request())


async def test_the_core_uses_the_count_and_falls_back_to_its_estimate() -> None:
    rc = LoopConstants(exact_token_count_enabled=True)
    counted = provider(FakeServer(httpx.Response(200, json={"count": 77})), tokenize=TokenizeDialect.vllm)
    assert await count_request_tokens_exactly(user_request(), counted, rc) == 77

    failing = provider(FakeServer(httpx.Response(500, text="boom")), tokenize=TokenizeDialect.vllm)
    assert await count_request_tokens_exactly(user_request(), failing, rc) is None


def test_the_counting_root_drops_the_api_version() -> None:
    assert endpoint().counting_root() == "http://127.0.0.1:8000"
    assert endpoint(base_url="http://127.0.0.1:8000").counting_root() == "http://127.0.0.1:8000"
    assert endpoint(tokenize_url="http://127.0.0.1:9/x/").counting_root() == "http://127.0.0.1:9/x"


async def test_the_counter_reports_whether_the_route_exists() -> None:
    server = FakeServer(httpx.Response(405))
    counter = TokenizeCounter(server.client(), root="http://127.0.0.1:1", dialect=TokenizeDialect.vllm)
    assert counter.available
    assert await counter.count({"messages": []}) is None
    assert not counter.available
