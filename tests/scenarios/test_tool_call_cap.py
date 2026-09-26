"""One model message may ask for only so many tool calls.

A model can emit a runaway batch — the same search a thousand times in one
message — and every call of it would run. The calls past the cap are each
answered with an error instead, so the transcript stays paired and the run
carries on.
"""
from __future__ import annotations

from .conftest import ScenarioFactory, ScriptedTool, default_rc


async def test_calls_past_the_per_message_cap_are_answered_but_not_run(
    scenario: ScenarioFactory,
) -> None:
    note = ScriptedTool(tool_name="Note", description="record a note")
    run = scenario(tools=[note], rc=default_rc(max_tool_calls_per_turn=2))
    run.llm.queue_multi_tool_call_response(
        tool_calls=[(f"n-{i}", "Note", {"v": str(i)}) for i in range(5)]
    )
    run.llm.queue_response(text="done")
    await run.run("note five things")

    assert note.invocations == [{"v": "0"}, {"v": "1"}]
    results = run.tool_results()
    assert [result.tool_call_id for result in results] == [f"n-{i}" for i in range(5)]
    assert [result.is_error for result in results] == [False, False, True, True, True]
    assert "at most 2 tool calls" in results[-1].content
    # Every call is paired, so the next request went out and the run finished.
    assert len(run.requests) == 2
    assert run.history_texts()[-1] == "done"
