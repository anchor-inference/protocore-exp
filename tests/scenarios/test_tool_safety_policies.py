"""A host's own tool safety policies, passed to the engine.

They used to be reachable only by writing an engine's private attribute after
construction. A host that replaced the tool dispatcher to install its policies
built it without the run's error counter, the tool roles and the resilience
classifier, so every one of them went missing from its runs without a test
failing. These scenarios state what the constructor gives instead.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from protocore.contracts.resilience import ResilienceErrorClass
from protocore.contracts.tools import Tool, ToolContext
from protocore.runtime.tool_permission import (
    PermissionStage,
    ToolPermissionDecision,
    ToolPermissionOutcome,
)
from protocore.runtime.tool_surface import forget_tool_surfaces
from tests._fixtures.tool_roles import CONVENTIONAL_TOOL_ROLES

from .conftest import ScenarioFactory, ScriptedTool


@pytest.fixture(autouse=True)
def _forget_surfaces() -> Iterator[None]:
    forget_tool_surfaces()
    yield
    forget_tool_surfaces()


@dataclass
class _RefuseByName:
    """A host policy that refuses one tool by name, whatever its class."""

    refused: str
    seen: list[str] = field(default_factory=list)

    def applies_to(self, side_effect_class: str) -> bool:
        return True

    def evaluate(
        self, tool: Tool, arguments: dict[str, Any], ctx: ToolContext
    ) -> ToolPermissionDecision:
        self.seen.append(tool.name)
        if tool.name == self.refused:
            return ToolPermissionDecision(
                outcome=ToolPermissionOutcome.deny,
                reason="refused by the host's policy",
                stage=PermissionStage.safety_policy,
            )
        return ToolPermissionDecision(outcome=ToolPermissionOutcome.allow)


@dataclass
class _Counter:
    calls: list[tuple[str, int]] = field(default_factory=list)

    async def increment_tool_errors_count(self, run_id: str, by: int = 1) -> None:
        self.calls.append((run_id, by))


@dataclass
class _Classifier:
    shown: list[tuple[str, str | None]] = field(default_factory=list)

    def classify_error_text(
        self, message: str, *, tool_name: str | None = None
    ) -> ResilienceErrorClass | None:
        self.shown.append((message, tool_name))
        return None


async def test_a_policy_passed_to_the_engine_refuses_the_call_and_the_error_is_counted(
    scenario: ScenarioFactory,
) -> None:
    note = ScriptedTool(tool_name="Note", description="record a note")
    policy = _RefuseByName(refused="Note")
    run = scenario(tools=[note], tool_safety_policies=[policy])
    counter = _Counter()
    run.engine.run_state.tool_error_counter = counter
    run.llm.queue_tool_call_response(tool_call_id="n-1", tool_name="Note", tool_input={})
    run.llm.queue_response(text="done")
    await run.run("note it")

    assert policy.seen == ["Note"]
    assert note.invocations == []
    assert len(counter.calls) == 1


async def test_the_classifier_and_the_roles_stay_beside_the_hosts_policies(
    scenario: ScenarioFactory,
) -> None:
    """The dispatcher built for the host's policies is the core's own: a
    failing call is shown to the host's classifier, and a shell tool's
    missing command is told apart by its role."""
    bash = ScriptedTool(
        tool_name="Bash",
        description="run a command",
        content="bash: frobnicate: command not found",
        is_error=True,
    )
    classifier = _Classifier()
    run = scenario(
        tools=[bash],
        tool_safety_policies=[_RefuseByName(refused="Nothing")],
        tool_roles=CONVENTIONAL_TOOL_ROLES,
        resilience_classifier=classifier,
    )
    run.llm.queue_tool_call_response(
        tool_call_id="b-1", tool_name="Bash", tool_input={"command": "frobnicate"}
    )
    run.llm.queue_response(text="done")
    await run.run("run it")

    assert len(bash.invocations) == 1
    assert any(tool_name == "Bash" for _message, tool_name in classifier.shown)
    streak = run.engine.run_state.consecutive_error
    assert streak is not None
    assert streak.signature == "Bash:SHELL_CMD_MISSING"


async def _run_destructive_command(scenario: ScenarioFactory, **overrides: Any) -> ScriptedTool:
    bash = ScriptedTool(tool_name="Bash", description="run a command")
    run = scenario(tools=[bash], tool_roles=CONVENTIONAL_TOOL_ROLES, **overrides)
    run.llm.queue_tool_call_response(
        tool_call_id="b-1", tool_name="Bash", tool_input={"command": "rm -rf /"}
    )
    run.llm.queue_response(text="done")
    await run.run("clean up")
    return bash


async def test_without_policies_named_the_core_s_shell_deny_list_applies(
    scenario: ScenarioFactory,
) -> None:
    bash = await _run_destructive_command(scenario)
    assert bash.invocations == []


async def test_an_empty_list_installs_no_policy_at_all(scenario: ScenarioFactory) -> None:
    """A host whose sandbox is the boundary says so with ``[]``; the core's
    deny list is a default, not a floor under the host's choice."""
    bash = await _run_destructive_command(scenario, tool_safety_policies=[])
    assert bash.invocations == [{"command": "rm -rf /"}]


async def test_a_rearmed_engine_keeps_the_hosts_policies(scenario: ScenarioFactory) -> None:
    note = ScriptedTool(tool_name="Note", description="record a note")
    policy = _RefuseByName(refused="Note")
    run = scenario(tools=[note], tool_safety_policies=[policy])
    run.llm.queue_response(text="first")
    await run.run("hello")
    run.engine.rearm()
    run.llm.queue_tool_call_response(tool_call_id="n-1", tool_name="Note", tool_input={})
    run.llm.queue_response(text="done")
    await run.run("note it")

    assert policy.seen == ["Note"]
    assert note.invocations == []
