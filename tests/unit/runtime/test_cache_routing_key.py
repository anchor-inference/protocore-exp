"""The key a provider routes a conversation's prompt cache by travels on the request.

Providers that keep prompt caches per machine route a request by a key the
client names (a body field such as ``prompt_cache_key``, or a header). A run
knows which conversation it belongs to; an adapter does not, and a host that
had to tell it through a side channel (a context variable set around the run)
got it wrong whenever a call left that context. A fork that starts with its
source's bytes is cached with its source only when it is routed by the
source's key, so the key is configurable rather than always the session.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.llm import LLMObservabilityContext
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)


def _engine(llm: InMemoryLLMProvider, **config: Any) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-cache",
            tenant_id="tenant-cache",
            session_id="sess-fork",
            model_name="model-a",
            rc=LoopConstants(model_context_window=8_192),
            **config,
        ),
        llm_provider=llm,
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


async def _first_request_observability(engine: QueryEngine, llm: InMemoryLLMProvider) -> LLMObservabilityContext:
    llm.queue_response(text="done", stop_reason=StopReason.end_turn)
    message = Message(role=MessageRole.user, content_blocks=[TextBlock(text="hello")])
    [event async for event in engine.run(message)]
    observability = llm.calls[0].observability
    assert observability is not None
    return observability


async def test_configured_cache_key_reaches_the_request() -> None:
    llm = InMemoryLLMProvider()
    engine = _engine(llm, cache_key="sess-source")

    observability = await _first_request_observability(engine, llm)

    assert observability.cache_key == "sess-source"
    assert observability.session_id == "sess-fork"
    assert observability.routing_key() == "sess-source"


async def test_without_a_configured_key_the_request_routes_by_its_session() -> None:
    llm = InMemoryLLMProvider()
    engine = _engine(llm)

    observability = await _first_request_observability(engine, llm)

    # Left unset on the request, so a manifest recorded before the field
    # existed and one recorded now carry the same identity.
    assert observability.cache_key is None
    assert observability.routing_key() == "sess-fork"


def test_routing_key_falls_back_to_the_run_and_then_to_nothing() -> None:
    assert LLMObservabilityContext(run_id="run-1").routing_key() == "run-1"
    assert LLMObservabilityContext(session_id="", run_id="run-1").routing_key() == "run-1"
    # No fixed fallback: one key shared by every caller would pin them all to
    # the same machine.
    assert LLMObservabilityContext().routing_key() is None


def test_configured_cache_key_must_not_be_blank() -> None:
    with pytest.raises(ValueError, match="cache_key"):
        QueryEngineConfig(
            run_id="run-cache",
            tenant_id="tenant-cache",
            session_id="sess-fork",
            model_name="model-a",
            cache_key="  ",
        )
