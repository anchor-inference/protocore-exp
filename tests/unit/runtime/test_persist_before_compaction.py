"""The rounds a run produced reach the store before a compaction inside the run rewrites them.

The loop hands the history over at the start of a run, at a background wake
and at a checkpoint. A compaction in the middle of a run replaced rounds the
store had never been given, so they existed nowhere but in the summary: a host
transcribed the whole history itself on ``compaction_started`` to keep them.
The completed event names, by position, which of the messages just handed
over the pass replaced and which messages it wrote, so a host can label the
summary without diffing two histories.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageOrigin, MessageRole, TextBlock
from protocore.runtime.events import EventType
from protocore.runtime.history_persist import HistoryDelta, history_rewrite
from protocore.runtime.query_engine import QueryEngine

from ._tool_fixtures import MockTool


class _LoggingStore:
    def __init__(self, log: list[tuple[Any, ...]]) -> None:
        self.log = log
        self.deltas: list[HistoryDelta] = []

    def persist_session_history(self, engine: QueryEngine) -> None:
        raise AssertionError("the delta path is attached")

    def persist_history_delta(self, engine: QueryEngine, delta: HistoryDelta) -> None:
        self.deltas.append(delta)
        self.log.append(("delta", len(self.deltas) - 1))


async def _drive(engine_factory: Any, in_memory_runtime: dict[str, Any], *, rc: LoopConstants, iterations: int = 12) -> tuple[QueryEngine, _LoggingStore, list[tuple[Any, ...]], list[Any]]:
    engine = engine_factory(rc=rc)
    log: list[tuple[Any, ...]] = []
    store = _LoggingStore(log)
    engine.persist_session_history = store.persist_session_history  # type: ignore[attr-defined]
    engine.persist_history_delta = store.persist_history_delta  # type: ignore[attr-defined]
    in_memory_runtime["tools"].register(MockTool(tool_name="ReadBig", description="read a big file", response_content="DATA " * 1200))
    llm = in_memory_runtime["llm"]
    for i in range(iterations):
        llm.queue_tool_call_response(tool_call_id=f"t{i}", tool_name="ReadBig", tool_input={"path": f"f{i}"})
    llm.queue_response(text="final answer")
    events = []
    async for event in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="ORIGINAL TASK: read all the files")])):
        events.append(event)
        log.append(("event", event.type, event))
    return engine, store, log, events


def _rc() -> LoopConstants:
    return LoopConstants(
        model_context_window=4_096,
        request_context_safety_tokens=0,
        compaction_per_iteration_enabled=True,
        compaction_keep_recent_turns=2,
        empty_completion_guard_enabled=False,
    )


async def test_the_rounds_are_handed_over_before_the_compaction_starts(engine_factory: Any, in_memory_runtime: dict[str, Any]) -> None:
    _engine, store, log, _events = await _drive(engine_factory, in_memory_runtime, rc=_rc())

    started = [i for i, entry in enumerate(log) if entry[0] == "event" and entry[1] is EventType.COMPACTION_STARTED]
    assert started, "no compaction inside the run"
    first = started[0]
    # The last hand-over before the event is the history the pass is about to
    # rewrite, and it holds the tool rounds of this run, not only its opening.
    handed = [entry for entry in log[:first] if entry[0] == "delta"]
    assert handed, "nothing reached the store before the compaction"
    delta = store.deltas[handed[-1][1]]
    started_event = log[first][2]
    assert len(delta.history) == started_event.payload["history_messages"]
    assert any(m.role is MessageRole.tool for m in delta.history)

    # The next hand-over after the pass is a rewrite: the store replaces what it holds.
    later = [entry for entry in log[first:] if entry[0] == "delta"]
    assert later and store.deltas[later[0][1]].rewritten


async def test_the_completed_event_names_what_was_replaced_and_written(engine_factory: Any, in_memory_runtime: dict[str, Any]) -> None:
    _engine, store, log, _events = await _drive(engine_factory, in_memory_runtime, rc=_rc())

    first_started = next(i for i, e in enumerate(log) if e[0] == "event" and e[1] is EventType.COMPACTION_STARTED)
    before = store.deltas[[e for e in log[:first_started] if e[0] == "delta"][-1][1]].history
    completed = next(e[2] for e in log[first_started:] if e[0] == "event" and e[1] is EventType.COMPACTION_COMPLETED)
    replaced = completed.payload["replaced_indices"]
    inserted = completed.payload["inserted_indices"]
    assert completed.payload["history_messages_before"] == len(before)
    assert replaced, "the pass replaced nothing"
    assert all(0 <= index < len(before) for index in replaced)
    after = completed.payload["history_messages_after"]
    assert all(0 <= index < after for index in inserted)
    # The opening task is never among what a pass replaces.
    assert 0 not in replaced


def test_history_rewrite_compares_by_identity() -> None:
    def say(text: str) -> Message:
        return Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])

    a, b, c, d = say("a"), say("b"), say("c"), say("d")
    summary = Message(role=MessageRole.user, content_blocks=[TextBlock(text="s")], metadata={"protocore.compaction_summary": True})
    same_text = say("c")
    rewrite = history_rewrite((a, b, c, d), (a, summary, same_text, d))
    assert rewrite == {"replaced_indices": [1, 2], "inserted_indices": [1, 2], "history_messages_before": 4, "history_messages_after": 4}
    assert summary.origin is MessageOrigin.compaction
    assert history_rewrite((a, b), (a, b)) == {"replaced_indices": [], "inserted_indices": [], "history_messages_before": 2, "history_messages_after": 2}
