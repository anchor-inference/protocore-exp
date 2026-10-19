"""Which settings each model step actually ran with.

A setting changed while a run is going — another model, another effort — takes
effect at the next request, not at the moment it is set. A host showing "this
answer came from model X" needs the record of what the request was built with,
not a reading of the current settings afterwards, which may already describe
the step after. These scenarios assert the record each step leaves: on the
event stream when asked for, on the engine and in the snapshot always, and
with the fields that changed since the step before named on it.
"""
from __future__ import annotations

from typing import Any

from protocore.contracts.step_settings import StepSettings
from protocore.runtime.events import EventType

from .conftest import ScenarioFactory, ScriptedTool, default_rc


def _records(events: list[Any]) -> list[dict[str, Any]]:
    return [dict(evt.payload) for evt in events if evt.type is EventType.STEP_SETTINGS]


async def test_each_step_records_the_settings_its_request_was_built_with(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Note")
    run = scenario(rc=default_rc(step_settings_event_enabled=True), tools=[tool])
    run.llm.queue_tool_call_response(tool_call_id="n-1", tool_name="Note", tool_input={"v": "a"})
    run.llm.queue_response(text="done")

    events = await run.run("note it")

    records = _records(events)
    assert len(records) == 2
    first, second = records
    request = run.requests[0]
    assert first["model"] == request.model == "scenario-model"
    assert first["thinking_enabled"] == request.extra["enable_thinking"]
    assert first["reasoning_effort"] == request.extra["reasoning_effort"]
    assert first["max_output_tokens"] == request.max_tokens
    advertised = [
        evt.payload["tool_surface_digest"]
        for evt in events
        if evt.type is EventType.TOOL_SURFACE_ADVERTISED
    ]
    assert first["tool_surface_digest"] == advertised[0]
    assert first["tool_count"] == len(request.tools)
    assert first["context_window"] == run.engine.config.rc.model_context_window
    # The first step of a run has nothing to differ from.
    assert first["changed"] == []
    # Nothing was changed between the two steps, so nothing is reported.
    assert second["changed"] == []
    assert second["digest"] == first["digest"]
    assert second["turn_id"] != first["turn_id"]
    # The record is the one the snapshot keeps.
    assert run.engine.last_step_settings == StepSettings.model_validate(second)
    assert run.engine.snapshot()["last_step_settings"] == second


async def test_a_change_made_mid_step_takes_effect_at_the_next_and_is_reported(
    scenario: ScenarioFactory,
) -> None:
    holder: dict[str, Any] = {}

    def switch(_arguments: dict[str, Any]) -> None:
        holder["engine"].apply_live_controls(model_name="other-model", reasoning_effort="high")

    class Switching(ScriptedTool):
        async def invoke(self, context: Any, arguments: dict[str, Any]) -> Any:
            switch(arguments)
            return await super().invoke(context, arguments)

    run = scenario(
        rc=default_rc(step_settings_event_enabled=True),
        tools=[Switching(tool_name="Note")],
    )
    holder["engine"] = run.engine
    run.llm.queue_tool_call_response(tool_call_id="n-1", tool_name="Note", tool_input={"v": "a"})
    run.llm.queue_response(text="done")

    events = await run.run("switch")

    first, second = _records(events)
    # The step that was running when the change was made kept its settings.
    assert first["model"] == "scenario-model"
    assert second["model"] == "other-model" == run.requests[1].model
    assert second["reasoning_effort"] == "high"
    assert set(second["changed"]) >= {"model", "reasoning_effort"}
    assert "tool_surface_digest" not in second["changed"]
    assert second["digest"] != first["digest"]


async def test_the_record_is_kept_without_the_event(
    scenario: ScenarioFactory,
) -> None:
    """The event is opt-in; the record on the engine and the snapshot is not."""
    run = scenario()
    run.llm.queue_response(text="hello")

    events = await run.run("hi")

    assert _records(events) == []
    settings = run.engine.last_step_settings
    assert settings is not None
    assert settings.model == "scenario-model"
    assert run.engine.snapshot()["last_step_settings"] == settings.model_dump(mode="json")


async def test_a_resumed_run_reports_what_changed_since_the_step_before_the_restart(
    scenario: ScenarioFactory,
) -> None:
    first = scenario()
    first.llm.queue_response(text="first")
    await first.run("one")
    stored = first.engine.snapshot()

    second = scenario(rc=default_rc(step_settings_event_enabled=True))
    second.llm.queue_response(text="second")
    await second.engine.resume_from_snapshot(stored)
    assert second.engine.last_step_settings == first.engine.last_step_settings
    second.engine.rearm()
    # Changed between the processes, before the resumed run's first step.
    second.engine.apply_live_controls(model_name="another-model")
    events = await second.run("two")

    [record] = _records(events)
    assert record["model"] == "another-model"
    assert "model" in record["changed"]


def test_a_retry_within_one_step_is_a_second_attempt_of_that_step() -> None:
    from protocore.contracts.step_settings import settle_step_settings

    base: dict[str, Any] = {
        "turn_id": "turn-1",
        "model": "m",
        "provider_model": None,
        "provider_rung": 0,
        "thinking_enabled": False,
        "reasoning_effort": "low",
        "forced_tool": None,
        "tool_surface_digest": "d",
        "tool_count": 1,
        "tool_policy_digest": "p",
        "rules_mark": "",
        "execution_profile": "default",
        "run_mode": "direct",
        "max_output_tokens": 100,
        "context_window": 1000,
        "max_turns_per_run": 10,
        "tool_timeout_seconds": 30.0,
        "request_manifest_id": None,
    }
    one = settle_step_settings(base, previous=None)
    again = settle_step_settings({**base, "model": "fallback"}, previous=one)
    assert one.attempt == 1
    assert again.attempt == 2
    assert again.changed == ("model",)
    later = settle_step_settings({**base, "turn_id": "turn-2"}, previous=again)
    assert later.attempt == 1
    assert later.changed == ("model",)


async def test_a_snapshot_with_a_malformed_record_is_refused_untouched(
    scenario: ScenarioFactory,
) -> None:
    import pytest

    first = scenario()
    first.llm.queue_response(text="first")
    await first.run("one")
    stored = first.engine.snapshot()

    for broken in ("a string", {"model": 3}):
        second = scenario()
        with pytest.raises(ValueError, match="last_step_settings"):
            await second.engine.resume_from_snapshot({**stored, "last_step_settings": broken})
        assert second.engine.last_step_settings is None
        assert second.engine.history == []
