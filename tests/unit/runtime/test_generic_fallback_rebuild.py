"""A provider swap hands the replacement the context the user already saw.

Every recovery branch that steps down the provider chain first appends the
failed attempt's partial assistant turn to history, because the subscribers
have already been shown it. The request that goes to the next provider must
therefore be built AFTER that append. The generic ``LLMProviderError`` branch —
the one an adapter's whole catch-all lands in, so the most travelled of the
three — used to swap providers and continue on the context object built before
the append, which sent the replacement a conversation missing the output the
user was looking at. It answered from scratch, and a model that had already
emitted a ``tool_use`` block could re-issue the same side effect.

The pairing is one act now — the policy that answers a failed stream steps the
chain down and asks for the rebuild in the same breath — and these tests hold
all three failures to it: the generic provider error, the rate limit, and the
stream that went silent.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from protocore.contracts.llm import (
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    LLMStreamIdleError,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock

PARTIAL = "PARTIAL-LIVE"


class _Verdict:
    """The verdict shape a provider adapter pins onto the error it raises."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


def _classified(exc: BaseException, reason: str) -> BaseException:
    object.__setattr__(exc, "classified", _Verdict(reason))
    return exc


class _FakeProviderChain:
    """A two-rung chain: the run starts on the first name and steps to the next."""

    def __init__(self, provider: Any, names: list[str]) -> None:
        self._provider = provider
        self._names = names
        self._index = 0

    def current(self) -> Any:
        return self._provider

    def current_model_name(self) -> str:
        return self._names[self._index]

    async def advance(self, *, reason: str) -> bool:
        if self._index + 1 >= len(self._names):
            return False
        self._index += 1
        return True


def _attach_chain(engine: Any, llm: Any, *names: str) -> _FakeProviderChain:
    chain = _FakeProviderChain(llm, list(names))
    engine.llm = llm
    engine.provider_chain = chain
    return chain


class _PartialThenFailLLM:
    """Streams a partial, fails the first attempt, then answers normally."""

    #: Raised on the first attempt, once the partial is on the wire.
    failure: BaseException = _classified(LLMProviderError("primary 5xx"), "server_error")

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []
        self._n = 0

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.calls.append(request)
        n = self._n
        self._n += 1
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"index": 0, "type": "text"})
        if n == 0:
            yield LLMStreamEvent(
                name="content_block_delta", payload={"index": 0, "text": PARTIAL}
            )
            raise self.failure
        yield LLMStreamEvent(name="content_block_delta", payload={"index": 0, "text": "recovered"})
        yield LLMStreamEvent(name="content_block_stop", payload={"index": 0})
        yield LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})

    async def complete_structured(self, request: LLMRequest, schema: Any) -> LLMResponse:
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return max(1, len(text) // 4)


class _RateLimitedLLM(_PartialThenFailLLM):
    """The control: the branch that has always rebuilt."""

    failure: BaseException = _classified(LLMRateLimitError("429"), "rate_limit")


class _IdleStreamLLM(_PartialThenFailLLM):
    """The third branch: a stream that stopped speaking mid-answer.

    Classified the way the idle watchdog classifies what it raises, since it is
    the only thing that raises this and nothing above core can label it.
    """

    failure: BaseException = _classified(LLMStreamIdleError("stalled"), "timeout")


def _request_text(request: LLMRequest) -> str:
    return " ".join(
        getattr(block, "text", "") or ""
        for message in request.messages
        for block in message.content_blocks
    )


def _history_text(engine: Any) -> str:
    return " ".join(
        getattr(block, "text", "") or ""
        for message in engine.history
        for block in message.content_blocks
    )


async def _run_two_attempts(
    engine_factory: Any, llm: _PartialThenFailLLM, **rc_overrides: Any
) -> Any:
    engine = engine_factory(rc=LoopConstants(model_context_window=4096, **rc_overrides))
    _attach_chain(engine, llm, "primary-model", "fallback-model-x")
    async for _ in engine.run(
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")])
    ):
        pass
    return engine


async def test_generic_provider_error_fallback_carries_the_streamed_partial(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    llm = _PartialThenFailLLM()
    engine = await _run_two_attempts(engine_factory, llm)

    assert len(llm.calls) == 2, [call.model for call in llm.calls]
    second = llm.calls[1]
    assert second.model == "fallback-model-x"
    assert PARTIAL in _history_text(engine), "the partial belongs in durable history"
    assert PARTIAL in _request_text(second), (
        "the replacement provider must be sent the output the subscribers already saw"
    )


async def test_rate_limit_fallback_carries_the_streamed_partial(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    """Control: the branch the generic one is now held to."""
    llm = _RateLimitedLLM()
    engine = await _run_two_attempts(engine_factory, llm)

    assert len(llm.calls) == 2, [call.model for call in llm.calls]
    assert llm.calls[1].model == "fallback-model-x"
    assert PARTIAL in _history_text(engine)
    assert PARTIAL in _request_text(llm.calls[1])


async def test_idle_stream_fallback_carries_the_streamed_partial(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    """A silent endpoint is answered by the next provider, not by giving up.

    With no in-place retry budget the first silence goes straight to the chain,
    which is the branch this file is about; the retry that normally comes first
    is pinned in ``tests/scenarios/test_provider_failure.py``.
    """
    llm = _IdleStreamLLM()
    engine = await _run_two_attempts(
        engine_factory, llm, llm_transient_error_retry_max_attempts=0
    )

    assert len(llm.calls) == 2, [call.model for call in llm.calls]
    assert llm.calls[1].model == "fallback-model-x"
    assert PARTIAL in _history_text(engine)
    assert PARTIAL in _request_text(llm.calls[1])


async def test_unclassified_provider_error_does_not_move_the_chain(
    engine_factory: Any, in_memory_runtime: dict[str, object]
) -> None:
    """No verdict, no swap — and so no rebuild either, which the helper reports
    by returning no context rather than by building one nothing will send."""

    class _Unclassified(_PartialThenFailLLM):
        failure: BaseException = LLMProviderError("policy refusal")

    llm = _Unclassified()
    started_on = engine_factory().config.model_name
    engine = await _run_two_attempts(engine_factory, llm)

    assert engine._provider_chain_advances == 0
    assert engine.config.model_name == started_on
    assert {call.model for call in llm.calls} == {started_on}
