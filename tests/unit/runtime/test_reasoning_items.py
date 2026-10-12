"""Opaque reasoning records ride on the assistant message they came with.

A provider that keeps a model's reasoning on its own side hands the client an
encrypted copy, and the model reasons from that copy on the next request. These
tests pin the three things that make that work: the stream's records end up on
the message, the message carries them into the next request, and compaction
sheds them together with the reasoning text rather than leaving half a turn.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from protocore.contracts.llm import LLMRequest, ProviderDelta, ProviderDeltaKind
from protocore.contracts.observability import model_visible
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.context.compaction import (
    estimate_message_tokens,
    run_tier1_truncation,
)
from protocore.runtime.query import (
    _drive_one_stream,
    _rebuild_context_for_recovery,
    _StreamAttemptResult,
)
from protocore.tests_support.adapters import InMemoryBlobStore

ITEM = {"endpoint": "codex", "model": "gpt-test", "details": [{"type": "reasoning.encrypted", "data": "opaque=="}], "tokens": 120}


def _user(text: str) -> Message:
    return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])


def test_reasoning_items_round_trip_and_are_absent_from_the_dump_when_unset() -> None:
    plain = Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="hi")])
    assert "reasoning_items" not in plain.model_dump(mode="json")
    assert "reasoning_items" not in model_visible(plain)

    carried = Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="hi")], reasoning_items=(ITEM,))
    rebuilt = Message.model_validate_json(carried.model_dump_json())
    assert rebuilt.reasoning_items == (ITEM,)
    assert model_visible(carried)["reasoning_items"] == [ITEM]


def test_an_empty_tuple_is_stored_as_none() -> None:
    message = Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="hi")], reasoning_items=())
    assert message.reasoning_items is None


def test_reasoning_items_are_refused_on_a_user_turn() -> None:
    with pytest.raises(ValueError, match="reasoning_items"):
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="hi")], reasoning_items=(ITEM,))


def _stream(deltas: list[ProviderDelta], seen: list[LLMRequest] | None = None):
    def _stream_with_tools(request: LLMRequest) -> AsyncIterator[ProviderDelta]:
        if seen is not None:
            seen.append(request)

        async def _gen() -> AsyncIterator[ProviderDelta]:
            for delta in deltas:
                yield delta

        return _gen()

    return _stream_with_tools


@pytest.mark.asyncio
async def test_the_stream_collects_reasoning_items_from_any_delta(engine_factory: Any) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=4_096))
    engine.history.append(_user("hi"))
    second = {**ITEM, "details": [{"type": "reasoning.encrypted", "data": "more=="}]}
    engine.llm.stream_with_tools = _stream([  # type: ignore[method-assign]
        ProviderDelta(kind=ProviderDeltaKind.text, content="A", reasoning_items=(ITEM,)),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop", reasoning_items=(second,)),
    ])
    result = _StreamAttemptResult()
    async for _evt in _drive_one_stream(engine, await _rebuild_context_for_recovery(engine), result):
        pass
    assert result.reasoning_items == [ITEM, second]


@pytest.mark.asyncio
async def test_the_next_request_carries_the_previous_turns_reasoning_items(engine_factory: Any) -> None:
    engine = engine_factory()
    seen: list[LLMRequest] = []
    engine.llm.stream_with_tools = _stream([  # type: ignore[method-assign]
        ProviderDelta(kind=ProviderDeltaKind.text, content="done"),
        ProviderDelta(kind=ProviderDeltaKind.finish, finish_reason="stop", reasoning_items=(ITEM,)),
    ], seen)
    async for _ in engine.run(_user("first")):
        pass
    answer = next(m for m in engine.history if m.role is MessageRole.assistant)
    assert answer.reasoning_items == (ITEM,)

    # A later run resumes from the stored history, as a session's next turn does.
    resumed = engine_factory(run_id="run-next")
    resumed.history.extend(engine.history)
    resumed.llm.stream_with_tools = engine.llm.stream_with_tools  # type: ignore[method-assign]
    async for _ in resumed.run(_user("second")):
        pass
    replayed = [m for m in seen[-1].messages if m.role is MessageRole.assistant]
    assert replayed and replayed[0].reasoning_items == (ITEM,)


def test_the_estimate_takes_the_providers_token_count_for_an_item() -> None:
    rc = LoopConstants(model_context_window=4_096)
    bare = Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="hi")])
    carried = bare.model_copy(update={"reasoning_items": (ITEM,)})
    assert estimate_message_tokens(carried, rc) - estimate_message_tokens(bare, rc) == 120


@pytest.mark.asyncio
async def test_tier1_sheds_reasoning_items_with_the_reasoning_text() -> None:
    rc = LoopConstants(model_context_window=4_096)
    history = [
        _user("go"),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ok")], reasoning_content="thought", reasoning_items=(ITEM,)),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="ok")], reasoning_items=(ITEM,)),
        _user("next"),
    ]
    from protocore.runtime.context.budgets import derive_budgets

    result = await run_tier1_truncation(
        history=history,
        blob_store=InMemoryBlobStore(),
        tenant_id="t1",
        rc=rc,
        truncation_threshold_tokens=derive_budgets(rc).tool_result_truncation_threshold,
        keep_recent_turns=0,
    )
    assert result.messages_modified == 2
    assert all(m.reasoning_items is None and m.reasoning_content is None for m in history)
