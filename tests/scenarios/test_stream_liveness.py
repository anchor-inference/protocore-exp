"""Time spent running tools is not time the model was silent.

The operator's reading of a run that kept losing its model was that long tools
were being charged to the stream. They are not, and this pins it: the idle
watchdog wraps each provider call alone, so a tool that runs far longer than
the idle budget between two calls leaves both calls with their whole budget.
The run is driven on a clock that jumps instead of waiting, so the "200 s"
tool costs nothing.
"""
from __future__ import annotations

from protocore.runtime.events import EventType
from tests._fixtures.virtual_clock import run_on_virtual_clock

from .conftest import ScenarioFactory, ScriptedTool, default_rc


def test_a_tool_longer_than_the_idle_budget_does_not_count_against_the_next_call(
    scenario: ScenarioFactory,
) -> None:
    slow = ScriptedTool(tool_name="Exec", content="built", delay_seconds=200.0)
    run = scenario(
        tools=[slow],
        rc=default_rc(llm_stream_idle_timeout_seconds=90.0, tool_timeout_seconds=600),
    )
    run.llm.queue_tool_call_response(tool_call_id="call-1", tool_name="Exec", tool_input={"v": "x"})
    run.llm.queue_response(text="all built")

    produced, elapsed = run_on_virtual_clock(lambda: run.run("build it"))

    assert elapsed >= 200.0
    assert len(slow.invocations) == 1
    assert len(run.requests) == 2
    assert "all built" in "".join(run.history_texts())
    assert not [
        evt for evt in produced
        if evt.type is EventType.STATE_CHANGED
        and evt.payload.get("error_class") == "llm_stream_idle"
    ]
