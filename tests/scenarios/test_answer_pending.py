"""A question a run parked on is answered with one call, and the run goes on.

Resuming a run paused on the core's own AskUser meant building the tool result
message by hand, appending it, releasing the wait, moving the state off
AWAITING and only then driving the run again. Every consumer repeated those
steps, and each one that got a step wrong left a call with no result (which the
wire repair then reports to the model as a failure) or a run that would not
start. ``QueryEngine.answer_pending`` takes those steps.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.interrupt import InterruptResolutionError
from protocore.contracts.types import (
    OPERATOR_WORDS_METADATA_KEY,
    Message,
    MessageRole,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime import LoopState
from protocore.runtime.query_engine import QueryEngine
from tests.scenarios.test_interrupt_resolution import (
    _CountingTool,
    _run_to_one_parked_approval,
    _Runtime,
    _user,
)


def _asked(runtime: _Runtime, *, tool_call_id: str = "toolu_q") -> QueryEngine:
    engine = runtime.engine()
    engine.history.append(_user("ask me"))
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id=tool_call_id, name="AskUser", arguments_json="{}")
            ],
        )
    )
    engine.transition_to(LoopState.RUNNING)
    engine.mark_awaiting_answer(
        tool_call_id,
        tool_name="AskUser",
        payload={"questions": [{"question": "which file?"}]},
    )
    engine.transition_to(LoopState.AWAITING)
    return engine


def _result_blocks(engine: QueryEngine, tool_call_id: str) -> list[ToolResultBlock]:
    return [
        block
        for message in engine.history_snapshot()
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock) and block.tool_call_id == tool_call_id
    ]


def _sent_results(request: Any) -> dict[str, str]:
    return {
        block.tool_call_id: block.content
        for message in request.messages
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    }


async def test_the_answer_reaches_the_next_request_as_the_calls_result() -> None:
    runtime = _Runtime()
    engine = _asked(runtime)
    runtime.llm.queue_response(text="opening report.md")

    await engine.answer_pending("toolu_q", "report.md")

    assert engine.state is LoopState.RUNNING
    assert engine.pending_interrupts == ()
    [block] = _result_blocks(engine, "toolu_q")
    assert block.content == "report.md"
    assert block.is_error is False
    # The operator's own words: compaction quotes this result whole.
    assert block.metadata.get(OPERATOR_WORDS_METADATA_KEY) is True

    async for _ in engine.run(None):
        pass

    assert _sent_results(runtime.llm.calls[-1])["toolu_q"] == "report.md"
    assert engine.state is LoopState.COMPLETED


async def test_an_answer_another_agent_gave_is_not_marked_as_the_operators() -> None:
    runtime = _Runtime()
    engine = _asked(runtime)

    await engine.answer_pending("toolu_q", '{"answers": []}', operator_words=False)

    [block] = _result_blocks(engine, "toolu_q")
    assert OPERATOR_WORDS_METADATA_KEY not in block.metadata


async def test_an_error_answer_is_recorded_as_one() -> None:
    runtime = _Runtime()
    engine = _asked(runtime)

    await engine.answer_pending("toolu_q", "nobody answered in time", is_error=True)

    [block] = _result_blocks(engine, "toolu_q")
    assert block.is_error is True


async def test_a_call_the_run_is_not_waiting_on_is_refused_with_nothing_changed() -> None:
    runtime = _Runtime()
    engine = _asked(runtime)

    with pytest.raises(InterruptResolutionError, match="toolu_other"):
        await engine.answer_pending("toolu_other", "report.md")

    assert engine.state is LoopState.AWAITING
    assert len(engine.pending_interrupts) == 1
    assert _result_blocks(engine, "toolu_q") == []


async def test_an_approval_is_decided_not_answered() -> None:
    runtime = _Runtime()
    tool = _CountingTool()
    engine, _ = await _run_to_one_parked_approval(runtime, tool)
    parked = engine.pending_interrupts[0]

    with pytest.raises(InterruptResolutionError, match="approval"):
        await engine.answer_pending(parked.tool_call_id, "yes")

    assert tool.calls == []
    assert engine.pending_interrupts == (parked,)


async def test_a_run_paused_in_another_process_takes_the_answer() -> None:
    runtime = _Runtime()
    stored = _asked(runtime).snapshot()
    runtime.llm.queue_response(text="opening report.md")

    successor = runtime.engine()
    await successor.resume_from_snapshot(stored)
    await successor.answer_pending("toolu_q", "report.md")
    async for _ in successor.run(None):
        pass

    assert _sent_results(runtime.llm.calls[-1])["toolu_q"] == "report.md"
    assert successor.state is LoopState.COMPLETED


async def test_an_answer_that_already_landed_is_not_written_twice() -> None:
    """A process that wrote the answer and died before releasing the wait."""
    runtime = _Runtime()
    engine = _asked(runtime)
    engine.history.append(
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="toolu_q", content="report.md")],
        )
    )

    await engine.answer_pending("toolu_q", "report.md")

    assert len(_result_blocks(engine, "toolu_q")) == 1
    assert engine.pending_interrupts == ()
    assert engine.state is LoopState.RUNNING


async def test_one_answer_of_two_leaves_the_run_waiting_for_the_other() -> None:
    runtime = _Runtime()
    engine = _asked(runtime)
    engine.history.append(
        Message(
            role=MessageRole.assistant,
            content_blocks=[
                ToolUseBlock(tool_call_id="toolu_r", name="AskUser", arguments_json="{}")
            ],
        )
    )
    engine.mark_awaiting_answer("toolu_r", tool_name="AskUser")

    await engine.answer_pending("toolu_q", "report.md")

    assert engine.state is LoopState.AWAITING
    assert [item.tool_call_id for item in engine.pending_interrupts] == ["toolu_r"]
