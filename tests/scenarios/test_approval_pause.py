"""An approval pause, and the resume that must execute the call exactly once.

The pause is the one place where the run stops with a tool call decided and
not performed, so it is where a duplicate side effect is cheapest to cause:
the operator approves, the pod dies, another pod picks the run up and
approves again. These scenarios drive the pause through ``run()`` and the
lift through ``resume_approved_tool`` — including across a cold start, on an
engine instance that never saw the pause.
"""
from __future__ import annotations

import json

import pytest

from protocore.contracts.llm import LLMStreamEvent, LLMStreamIdleError
from protocore.contracts.types import ToolCall
from protocore.runtime.events import EventType
from protocore.runtime.loop_state import LoopState
from protocore.runtime.query import resume_approved_tool

from .conftest import (
    FailingProvider,
    ProviderChainDouble,
    ScenarioFactory,
    ScriptedTool,
    classified,
    default_rc,
    require_approval,
)


def _approving_rc(**overrides: object) -> object:
    values: dict[str, object] = {"approval_gate_web_enabled": True}
    values.update(overrides)
    return default_rc(**values)


async def test_a_call_that_needs_approval_pauses_instead_of_running(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Deploy")
    run = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(run.hooks, token="tok-a")
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Deploy", tool_input={"v": "prod"}
    )

    produced = await run.run("deploy it")

    pending = [evt for evt in produced if evt.type is EventType.TOOL_CALL_PENDING]
    assert len(pending) == 1
    assert pending[0].payload["approval_token"] == "tok-a"
    assert tool.invocations == []
    assert run.engine.state is LoopState.AWAITING


async def test_the_approved_call_runs_once_and_its_result_lands_in_history(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(run.hooks)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Deploy", tool_input={"v": "prod"}
    )
    await run.run("deploy it")

    lifted = [
        evt
        async for evt in resume_approved_tool(
            run.engine, ToolCall(id="call-1", name="Deploy", arguments={"v": "prod"})
        )
    ]

    assert tool.invocations == [{"v": "prod"}]
    assert [evt.type for evt in lifted].count(EventType.TOOL_RESULT) == 1
    assert [block.content for block in run.tool_results()] == ["deployed"]


async def test_a_second_approval_for_the_same_call_is_a_no_op(
    scenario: ScenarioFactory,
) -> None:
    """The operator clicked twice, or two pods delivered the same approval."""
    tool = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(run.hooks)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Deploy", tool_input={"v": "prod"}
    )
    await run.run("deploy it")
    call = ToolCall(id="call-1", name="Deploy", arguments={"v": "prod"})

    async for _ in resume_approved_tool(run.engine, call):
        pass
    async for _ in resume_approved_tool(run.engine, call):
        pass

    assert tool.invocations == [{"v": "prod"}]
    assert len(run.tool_results()) == 1


async def test_the_pause_survives_a_cold_start_and_the_call_still_runs_once(
    scenario: ScenarioFactory,
) -> None:
    """The engine that lifts the pause is not the engine that took it."""
    tool = ScriptedTool(tool_name="Deploy", content="deployed")
    first = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(first.hooks)
    first.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Deploy", tool_input={"v": "prod"}
    )
    await first.run("deploy it")
    snapshot = first.engine.snapshot()

    second = scenario(rc=_approving_rc(), tools=[tool])
    await second.engine.resume_from_snapshot(snapshot)
    async for _ in resume_approved_tool(
        second.engine, ToolCall(id="call-1", name="Deploy", arguments={"v": "prod"})
    ):
        pass

    assert tool.invocations == [{"v": "prod"}]
    assert [block.content for block in second.tool_results()] == ["deployed"]


async def test_a_resumed_run_does_not_execute_the_pending_call_on_its_own(
    scenario: ScenarioFactory,
) -> None:
    """Restoring a paused run is not approving it."""
    tool = ScriptedTool(tool_name="Deploy")
    first = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(first.hooks)
    first.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Deploy", tool_input={"v": "prod"}
    )
    await first.run("deploy it")

    second = scenario(rc=_approving_rc(), tools=[tool])
    await second.engine.resume_from_snapshot(first.engine.snapshot())

    assert tool.invocations == []
    assert second.engine.state is LoopState.AWAITING


async def test_an_approval_landing_on_a_cancelled_run_does_not_run_the_tool(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Deploy")
    run = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(run.hooks)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Deploy", tool_input={"v": "prod"}
    )
    await run.run("deploy it")
    run.engine.stop()

    async for _ in resume_approved_tool(
        run.engine, ToolCall(id="call-1", name="Deploy", arguments={"v": "prod"})
    ):
        pass

    assert tool.invocations == []
    results = run.tool_results()
    assert [block.tool_call_id for block in results] == ["call-1"]
    assert results[0].is_error is True


async def test_an_approval_for_a_call_the_run_never_made_is_refused(
    scenario: ScenarioFactory,
) -> None:
    tool = ScriptedTool(tool_name="Deploy")
    run = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(run.hooks)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Deploy", tool_input={"v": "prod"}
    )
    await run.run("deploy it")

    with pytest.raises(Exception):
        async for _ in resume_approved_tool(
            run.engine,
            ToolCall(id="call-other", name="Deploy", arguments={"v": "prod"}),
        ):
            pass

    assert tool.invocations == []


async def test_an_approval_re_serialised_by_the_host_still_runs(
    scenario: ScenarioFactory,
) -> None:
    """The host may hand the call back spelled differently, not changed.

    An approval travels out of this run and back through the host's own
    transport, which is free to serialise the arguments its own way — sorted
    keys are the usual choice. That is the same call, and a run that refuses
    it strands an approval a person really gave.
    """
    tool = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(run.hooks)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1",
        tool_name="Deploy",
        tool_input={"path": "/srv/app", "content": "prod"},
    )
    await run.run("deploy it")

    # The key order the model emitted is not the order that comes back.
    round_tripped = json.loads(
        json.dumps({"path": "/srv/app", "content": "prod"}, sort_keys=True)
    )
    assert list(round_tripped) != ["path", "content"]

    lifted = [
        evt
        async for evt in resume_approved_tool(
            run.engine,
            ToolCall(id="call-1", name="Deploy", arguments=round_tripped),
        )
    ]

    assert tool.invocations == [{"content": "prod", "path": "/srv/app"}]
    assert [evt.type for evt in lifted].count(EventType.TOOL_RESULT) == 1


async def test_an_approval_whose_arguments_really_changed_is_still_refused(
    scenario: ScenarioFactory,
) -> None:
    """Different values, not a different spelling: the call never runs."""
    tool = ScriptedTool(tool_name="Deploy")
    run = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(run.hooks)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1",
        tool_name="Deploy",
        tool_input={"path": "/srv/app", "content": "prod"},
    )
    await run.run("deploy it")

    with pytest.raises(ValueError):
        async for _ in resume_approved_tool(
            run.engine,
            ToolCall(
                id="call-1",
                name="Deploy",
                arguments={"content": "staging", "path": "/srv/app"},
            ),
        ):
            pass

    assert tool.invocations == []


async def test_the_gate_is_downgraded_when_the_installation_says_so(
    scenario: ScenarioFactory,
) -> None:
    """Same hook verdict, switch off: the call proceeds without a pause."""
    tool = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=_approving_rc(approval_gate_web_enabled=False), tools=[tool])
    require_approval(run.hooks)
    run.llm.queue_tool_call_response(
        tool_call_id="call-1", tool_name="Deploy", tool_input={"v": "prod"}
    )
    run.llm.queue_response(text="deployed and reported")

    produced = await run.run("deploy it")

    assert EventType.TOOL_CALL_PENDING not in [evt.type for evt in produced]
    assert tool.invocations == [{"v": "prod"}]


def _truncated_then(
    calls: list[tuple[str, str, dict[str, object], bool]],
) -> list[LLMStreamEvent]:
    """One round in which the model did not finish writing every call."""
    events = [LLMStreamEvent(name="message_start", payload={})]
    for tool_call_id, tool_name, arguments, truncated in calls:
        events.extend(
            [
                LLMStreamEvent(
                    name="tool_use_start",
                    payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
                ),
                LLMStreamEvent(
                    name="tool_use_input_delta",
                    payload={
                        "tool_call_id": tool_call_id,
                        "partial_input_json": json.dumps(arguments),
                    },
                ),
                LLMStreamEvent(
                    name="tool_use_stop",
                    payload={
                        "tool_call_id": tool_call_id,
                        "final_input": arguments,
                        "args_partial_truncated": truncated,
                    },
                ),
            ]
        )
    events.append(
        LLMStreamEvent(name="message_stop", payload={"stop_reason": "end_turn"})
    )
    return events


async def test_the_pause_is_the_same_pause_when_a_recovery_took_the_batch(
    scenario: ScenarioFactory,
) -> None:
    """A gate held a call the recovery dispatched, and the run still pauses.

    The recovery is a second entry into dispatch, so what a host sees when a
    gate fires there has to be what it sees anywhere else: the run in
    AWAITING, one wait per held call, one envelope carrying the open set, and
    the tool not run.
    """
    tool = ScriptedTool(tool_name="Deploy", content="deployed")
    run = scenario(rc=_approving_rc(), tools=[tool])
    require_approval(run.hooks, token="tok-recovery")
    run.llm.queue_scripted_stream(
        _truncated_then(
            [
                ("call-trunc", "Deploy", {}, True),
                ("call-gated", "Deploy", {"v": "prod"}, False),
            ]
        )
    )

    produced = await run.run("deploy it")

    assert tool.invocations == []
    assert run.engine.state is LoopState.AWAITING
    assert [item.tool_call_id for item in run.engine.pending_interrupts] == [
        "call-gated"
    ]
    announcements = [
        evt for evt in produced if evt.type is EventType.INTERRUPT_PARKED
    ]
    assert len(announcements) == 1


async def test_the_pause_survives_the_provider_the_run_had_to_step_down_from(
    scenario: ScenarioFactory,
) -> None:
    """The replacement provider's round is gated, and the pause is unchanged.

    A run that stepped down the chain rebuilt its context to do it. What the
    gate then holds is a call from the new rung, and the pause it takes is
    the same pause — otherwise a host would learn that a fallback quietly
    changes what an approval means.
    """
    tool = ScriptedTool(tool_name="Deploy", content="deployed")
    # Silent twice: the first silence is retried where it is, the second moves the chain.
    silence = classified(LLMStreamIdleError("went quiet"), "timeout")
    primary = FailingProvider(failures={0: silence, 1: silence}, partial_text="about to deploy")
    fallback = FailingProvider(
        failures={},
        tool_calls=[("call-1", "Deploy", {"v": "prod"})],
    )
    chain = ProviderChainDouble(
        [("primary-model", primary), ("fallback-model", fallback)]
    )
    run = scenario(
        rc=_approving_rc(llm_transient_error_retry_backoff_base_seconds=0.0),
        tools=[tool],
        llm_provider=primary,
        provider_chain=chain,
        model_name="primary-model",
    )
    require_approval(run.hooks, token="tok-fallback")

    await run.run("deploy it")

    assert chain.advance_reasons
    assert tool.invocations == []
    assert run.engine.state is LoopState.AWAITING
    assert [item.tool_call_id for item in run.engine.pending_interrupts] == ["call-1"]
