"""Shared pieces for the OpenAI-compatible adapter tests: recorded streams and a fake server.

Every test talks to ``httpx.MockTransport``; nothing here opens a socket.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from protocore.contracts.llm import LLMObservabilityContext, LLMRequest, ProviderDelta, ProviderDeltaKind
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.providers.openai_compat import EndpointSpec, OpenAICompatibleProvider, ProviderCapabilities

FIXTURES = Path(__file__).parent / "fixtures"

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def recorded(name: str) -> bytes:
    """A recorded SSE body from ``fixtures/``."""
    return (FIXTURES / name).read_bytes()


def sse_response(body: bytes, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, content=body, headers={"content-type": "text/event-stream", **(headers or {})})


def chunk_stream(*chunks: dict[str, Any], done: bool = True) -> bytes:
    """An SSE body made of ``chunks``."""
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


class FakeServer:
    """Answers each request with the next queued response and keeps what it was sent."""

    def __init__(self, *responses: httpx.Response | Handler) -> None:
        self.queue: list[httpx.Response | Handler] = list(responses)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.queue:
            raise AssertionError(f"unexpected request to {request.url}")
        answer = self.queue.pop(0)
        if isinstance(answer, httpx.Response):
            return answer
        return await answer(request)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def endpoint(**overrides: Any) -> EndpointSpec:
    values: dict[str, Any] = {"name": "test-endpoint", "base_url": "http://127.0.0.1:8000/v1"}
    values.update(overrides)
    return EndpointSpec(**values)


def provider(server: FakeServer, *, capabilities: ProviderCapabilities | None = None, **overrides: Any) -> OpenAICompatibleProvider:
    if capabilities is not None:
        overrides["capabilities"] = capabilities
    observers = {key: overrides.pop(key) for key in ("on_usage", "on_failure", "image_loader") if key in overrides}
    return OpenAICompatibleProvider(endpoint(**overrides), client=server.client(), **observers)


def user_request(text: str = "hello", **kwargs: Any) -> LLMRequest:
    kwargs.setdefault("model", "model-a")
    kwargs.setdefault(
        "observability", LLMObservabilityContext(session_id="sess-1", run_id="run-1")
    )
    return LLMRequest(
        messages=[Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])],
        **kwargs,
    )


async def collect(stream: AsyncIterator[ProviderDelta]) -> list[ProviderDelta]:
    return [delta async for delta in stream]


def kinds(deltas: list[ProviderDelta]) -> list[str]:
    return [delta.kind.value for delta in deltas if delta.kind is not ProviderDeltaKind.progress]


def text_of(deltas: list[ProviderDelta], kind: ProviderDeltaKind = ProviderDeltaKind.text) -> str:
    return "".join(delta.content or "" for delta in deltas if delta.kind is kind)
