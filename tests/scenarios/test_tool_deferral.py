"""Held-back tool groups, driven through the loop as a host drives it.

What a host and a model can see is asserted from outside the engine: the tool
list and system prompt of every request the provider received, the events the
reader was handed, and what survives a snapshot. The surface and the catalogue
are compared byte for byte between requests, because the point of the design
is that loading a tool changes the END of the tool list and nothing else.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import pytest

from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import MessageRole, TextBlock, ToolResult
from protocore.runtime.events import EventType
from protocore.runtime.tool_surface import forget_tool_surfaces
from protocore.tools import ToolSearchTool

from .conftest import Scenario, ScenarioFactory, ScriptedTool, default_rc

_GITHUB_DESCRIPTION = "GitHub issues and pull requests"


@pytest.fixture(autouse=True)
def _forget_surfaces() -> Iterator[None]:
    forget_tool_surfaces()
    yield
    forget_tool_surfaces()


def _tools() -> list[ScriptedTool]:
    return [
        ScriptedTool(tool_name="Note", description="record a note"),
        ScriptedTool(tool_name="Mcp_Github_list_issues", description="list the issues"),
        ScriptedTool(tool_name="Mcp_Github_create_issue", description="open an issue"),
        ScriptedTool(tool_name="Zeta", description="the last tool by name"),
    ]


def _with_search(run: Scenario, *, dynamic: bool = True) -> Scenario:
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group(
        "github", _GITHUB_DESCRIPTION, dynamic=dynamic, prefix="Mcp_Github_"
    )
    return run


def _system_text(run: Scenario, index: int) -> str:
    return "\n".join(
        block.text
        for message in run.requests[index].messages
        if message.role is MessageRole.system
        for block in message.content_blocks
        if isinstance(block, TextBlock)
    )


def _definitions(run: Scenario, index: int) -> list[str]:
    return [definition.model_dump_json() for definition in run.requests[index].tools]


def _adverts(run: Scenario) -> list[dict[str, Any]]:
    return [dict(evt.payload) for evt in run.events_of(EventType.TOOL_SURFACE_ADVERTISED)]


# ── below every threshold nothing changes ───────────────────────────────────


async def test_a_surface_that_fits_is_sent_exactly_as_without_groups(
    scenario: ScenarioFactory,
) -> None:
    """Declaring groups and registering ToolSearch costs a run that fits nothing:
    the same tools, the same system prompt, and no search tool to spend turns on."""
    plain = scenario(tools=_tools())
    plain.llm.queue_response(text="done")
    await plain.run("hello")

    grouped = _with_search(scenario(tools=_tools()), dynamic=False)
    grouped.llm.queue_response(text="done")
    await grouped.run("hello")

    assert _definitions(grouped, 0) == _definitions(plain, 0)
    assert _system_text(grouped, 0) == _system_text(plain, 0)
    advert = _adverts(grouped)[0]
    assert advert["deferred_tool_groups"] == []
    assert advert["deferred_tool_count"] == 0
    assert "ToolSearch" not in grouped.advertised_tool_names(0)


# ── a dynamic group is held back and loaded on request ──────────────────────


async def test_a_loaded_tool_is_appended_and_nothing_before_it_moves(
    scenario: ScenarioFactory,
) -> None:
    run = _with_search(scenario(tools=_tools()))
    run.llm.queue_tool_call_response(
        tool_call_id="s-1",
        tool_name="ToolSearch",
        tool_input={"query": "select:Mcp_Github_create_issue"},
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_Github_create_issue", tool_input={"v": "bug"}
    )
    run.llm.queue_response(text="filed")
    await run.run("file a bug")

    first, second, third = (_definitions(run, i) for i in range(3))
    assert run.advertised_tool_names(0) == ["Note", "Zeta", "ToolSearch"]
    # The base is untouched and the loaded tool is its tail, from then on.
    assert second[: len(first)] == first
    assert run.advertised_tool_names(1) == ["Note", "Zeta", "ToolSearch", "Mcp_Github_create_issue"]
    assert third == second

    # The catalogue names the group once, with exact names, and does not move
    # when a tool of it is loaded.
    catalogue = _system_text(run, 0)
    assert (
        f"- github: {_GITHUB_DESCRIPTION}. Tools: "
        "Mcp_Github_create_issue, Mcp_Github_list_issues"
    ) in catalogue
    assert _system_text(run, 1) == catalogue == _system_text(run, 2)

    discovered = run.events_of(EventType.TOOL_DISCOVERED)
    assert [evt.payload["loaded_tool_names"] for evt in discovered] == [
        ["Mcp_Github_create_issue"]
    ]
    assert discovered[0].payload["tool_call_id"] == "s-1"
    adverts = _adverts(run)
    assert adverts[0]["deferred_tool_groups"] == ["github"]
    assert adverts[0]["tool_deferral_reasons"] == ["dynamic"]
    assert adverts[1]["discovered_tool_names"] == ["Mcp_Github_create_issue"]
    sources = {entry["name"]: entry["sources"] for entry in adverts[1]["tools"]}
    assert sources["Mcp_Github_create_issue"] == ["discovered"]
    assert [result.content for result in run.tool_results()][-1] == "ok"


async def test_a_search_loads_its_best_matches(scenario: ScenarioFactory) -> None:
    run = _with_search(scenario(tools=_tools(), rc=default_rc(tool_search_autoload_count=1)))
    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"query": "list the issues"}
    )
    run.llm.queue_response(text="found it")
    await run.run("what is open?")

    assert run.advertised_tool_names(1)[-1] == "Mcp_Github_list_issues"
    assert "Loaded, and callable from your next step: Mcp_Github_list_issues." in (
        run.tool_results()[0].content
    )


# ── a held-back tool called by name is served and loaded ────────────────────


async def test_calling_a_held_back_tool_by_name_runs_it_and_loads_it(
    scenario: ScenarioFactory,
) -> None:
    tools = _tools()
    run = _with_search(scenario(tools=tools))
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_Github_list_issues", tool_input={"v": "open"}
    )
    run.llm.queue_response(text="three open")
    await run.run("what is open?")

    assert tools[1].invocations == [{"v": "open"}]
    assert "Mcp_Github_list_issues" not in run.advertised_tool_names(0)
    assert run.advertised_tool_names(1)[-1] == "Mcp_Github_list_issues"
    (event,) = run.events_of(EventType.TOOL_UNADVERTISED_CALL)
    assert event.payload["tool_name"] == "Mcp_Github_list_issues"
    assert event.payload["deferred"] is True
    assert event.payload["loaded"] is True
    assert event.payload["success"] is True


async def test_a_misspelt_name_is_answered_with_the_nearest_admitted_ones(
    scenario: ScenarioFactory,
) -> None:
    run = _with_search(scenario(tools=_tools()))
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_github_list_issues", tool_input={}
    )
    run.llm.queue_response(text="sorry")
    await run.run("what is open?")

    (result,) = run.tool_results()
    assert result.is_error
    assert "Did you mean: Mcp_Github_list_issues" in result.content
    assert run.events_of(EventType.TOOL_UNADVERTISED_CALL) == []


async def test_a_blocked_tool_is_never_offered_as_a_near_name(
    scenario: ScenarioFactory,
) -> None:
    run = _with_search(
        scenario(
            tools=_tools(),
            tool_visibility_policy=ToolVisibilityPolicy(blocked={"Mcp_Github_list_issues"}),
        )
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_github_list_issues", tool_input={}
    )
    run.llm.queue_response(text="sorry")
    await run.run("what is open?")

    (result,) = run.tool_results()
    assert "Mcp_Github_list_issues" not in result.content


# ── what survives a process ─────────────────────────────────────────────────


async def test_the_loaded_tools_and_the_decision_survive_a_resume(
    scenario: ScenarioFactory,
) -> None:
    first = _with_search(scenario(tools=_tools()))
    first.llm.queue_tool_call_response(
        tool_call_id="s-1",
        tool_name="ToolSearch",
        tool_input={"query": "select:Mcp_Github_list_issues"},
    )
    first.llm.queue_response(text="loaded")
    await first.run("load it")
    snapshot = first.engine.snapshot()
    assert snapshot["deferred_tool_groups"] == ["github"]
    assert [row["name"] for row in snapshot["discovered_tools"]] == ["Mcp_Github_list_issues"]

    # The new process has more tools in the group and declares it NOT dynamic,
    # so a fresh decision would hold nothing back; the snapshot's is replayed.
    second = _with_search(scenario(tools=_tools()), dynamic=False)
    await second.engine.resume_from_snapshot(snapshot)
    second.engine.rearm()
    second.llm.queue_response(text="still here")
    await second.run("and now?")

    assert _definitions(second, 0) == _definitions(first, 1)
    catalogue = _system_text(first, 1)
    assert "- github:" in catalogue
    assert "- github:" in _system_text(second, 0)


async def test_a_host_seeds_a_new_run_with_what_the_last_one_loaded(
    scenario: ScenarioFactory,
) -> None:
    run = _with_search(
        scenario(
            tools=_tools(),
            discovered_tools=("Mcp_Github_create_issue", "Mcp_Github_list_issues"),
            rc=default_rc(pinned_tool_max_count=1),
        )
    )
    run.llm.queue_response(text="ready")
    await run.run("hello")

    # Over the cap, the oldest of the seed is left out; the rest keep their order.
    assert run.advertised_tool_names(0)[-1] == "Mcp_Github_list_issues"
    assert "Mcp_Github_create_issue" not in run.advertised_tool_names(0)
    assert _adverts(run)[0]["discovered_tool_names"] == ["Mcp_Github_list_issues"]


async def test_loaded_tools_over_the_cap_stay_until_the_next_turn(
    scenario: ScenarioFactory,
) -> None:
    """Unloading mid-run would pull a schema from under the model; a turn
    boundary is where the prefix starts over, so that is where the cap bites."""
    run = _with_search(scenario(tools=_tools(), rc=default_rc(pinned_tool_max_count=1)))
    run.llm.queue_tool_call_response(
        tool_call_id="s-1",
        tool_name="ToolSearch",
        tool_input={"query": "select:Mcp_Github_create_issue,Mcp_Github_list_issues"},
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_Github_create_issue", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="done")
    await run.run("both")

    assert run.advertised_tool_names(2)[-2:] == [
        "Mcp_Github_create_issue",
        "Mcp_Github_list_issues",
    ]
    run.engine.rearm()
    run.llm.queue_response(text="next")
    await run.run("next")
    # The one used most recently survives.
    assert run.advertised_tool_names(3)[-1] == "Mcp_Github_create_issue"
    assert "Mcp_Github_list_issues" not in run.advertised_tool_names(3)


# ── the two walls ───────────────────────────────────────────────────────────


async def test_a_surface_over_its_token_budget_holds_back_a_declared_group(
    scenario: ScenarioFactory,
) -> None:
    long = "does a great many things, each of them described at length. " * 20
    tools = [
        ScriptedTool(tool_name="Note", description="record a note"),
        *(ScriptedTool(tool_name=f"Browser{i}", description=long) for i in range(4)),
    ]
    run = scenario(tools=tools)
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group("browser", "Drive a web browser", prefix="Browser")
    run.llm.queue_response(text="done")
    await run.run("hello")

    assert run.advertised_tool_names(0) == ["Note", "ToolSearch"]
    advert = _adverts(run)[0]
    assert advert["deferred_tool_groups"] == ["browser"]
    assert advert["tool_deferral_reasons"] == ["tokens"]
    assert "- browser: Drive a web browser. Tools: Browser0, Browser1, Browser2, Browser3" in (
        _system_text(run, 0)
    )


async def test_loaded_tools_past_a_provider_limit_are_not_advertised(
    scenario: ScenarioFactory,
) -> None:
    """A provider that refuses more than N tools fails the whole request; the
    least recently used loaded tools are left off instead. They stay loaded."""
    tools = [*_tools(), ScriptedTool(tool_name="Mcp_Github_merge", description="merge")]
    run = _with_search(
        scenario(
            tools=tools,
            discovered_tools=(
                "Mcp_Github_merge",
                "Mcp_Github_create_issue",
                "Mcp_Github_list_issues",
            ),
            rc=default_rc(max_advertised_tools=5),
        )
    )
    run.llm.queue_response(text="ready")
    await run.run("hello")

    assert run.advertised_tool_names(0) == [
        "Note",
        "Zeta",
        "ToolSearch",
        "Mcp_Github_create_issue",
        "Mcp_Github_list_issues",
    ]
    assert _adverts(run)[0]["discovered_tool_names"] == [
        "Mcp_Github_merge",
        "Mcp_Github_create_issue",
        "Mcp_Github_list_issues",
    ]


async def test_parallel_searches_load_in_the_order_the_model_asked(
    scenario: ScenarioFactory,
) -> None:
    run = _with_search(scenario(tools=_tools()))
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("s-1", "ToolSearch", {"query": "select:Mcp_Github_list_issues"}),
            ("s-2", "ToolSearch", {"query": "select:Mcp_Github_create_issue"}),
        ]
    )
    run.llm.queue_response(text="both loaded")
    await run.run("load both")

    assert run.advertised_tool_names(1)[-2:] == [
        "Mcp_Github_list_issues",
        "Mcp_Github_create_issue",
    ]
    assert [evt.payload["tool_call_id"] for evt in run.events_of(EventType.TOOL_DISCOVERED)] == [
        "s-1",
        "s-2",
    ]


# ── what a result tells a model about tools it could not see ────────────────


async def test_a_search_says_which_tools_were_already_in_the_list(
    scenario: ScenarioFactory,
) -> None:
    """A model told "Loaded" of a tool it had all along takes it for something else
    it asked for having been loaded; the loop tells the tool what it advertised."""
    run = _with_search(scenario(tools=_tools()))
    run.llm.queue_tool_call_response(
        tool_call_id="s-1",
        tool_name="ToolSearch",
        tool_input={"select": ["Note", "Mcp_Github_list_issues"]},
    )
    run.llm.queue_response(text="loaded")
    await run.run("load it")

    lines = run.tool_results()[0].content.splitlines()
    assert lines[0] == "Loaded, and callable from your next step: Mcp_Github_list_issues."
    assert lines[1] == "Already in your tool list, nothing to load: Note."
    assert run.advertised_tool_names(1) == ["Note", "Zeta", "ToolSearch", "Mcp_Github_list_issues"]


@pytest.mark.parametrize(
    "failure",
    [{"raises": TypeError("got an unexpected keyword argument 'v'")}, {"is_error": True}],
    ids=["raised", "error-result"],
)
async def test_a_blind_call_that_fails_is_answered_with_the_tools_line(
    scenario: ScenarioFactory, failure: dict[str, Any]
) -> None:
    tools = _tools()
    for field_name, value in failure.items():
        setattr(tools[1], field_name, value)
    run = _with_search(scenario(tools=tools))
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_Github_list_issues", tool_input={"state": "open"}
    )
    run.llm.queue_response(text="retrying")
    await run.run("what is open?")

    (result,) = run.tool_results()
    assert result.is_error
    assert result.content.endswith("It takes: Mcp_Github_list_issues(v) — list the issues")
    assert run.advertised_tool_names(1)[-1] == "Mcp_Github_list_issues"


async def test_a_failure_of_a_listed_tool_is_not_given_the_line(
    scenario: ScenarioFactory,
) -> None:
    tools = _tools()
    tools[0].raises = TypeError("got an unexpected keyword argument 'text'")
    run = _with_search(scenario(tools=tools))
    run.llm.queue_tool_call_response(tool_call_id="c-1", tool_name="Note", tool_input={"text": "x"})
    run.llm.queue_response(text="retrying")
    await run.run("note it")

    (result,) = run.tool_results()
    assert result.is_error
    assert "It takes:" not in result.content


async def test_blind_calls_in_one_message_each_get_the_line(
    scenario: ScenarioFactory,
) -> None:
    """Calls gathered in parallel have their failures rebuilt in transcript
    order; the line must survive that rebuild."""
    tools = _tools()
    for tool in tools[1:3]:
        tool.is_error = True
        tool.is_concurrent_safe = True
    run = _with_search(scenario(tools=tools))
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("c-1", "Mcp_Github_list_issues", {"state": "open"}),
            ("c-2", "Mcp_Github_create_issue", {"title": "bug"}),
        ]
    )
    run.llm.queue_response(text="retrying")
    await run.run("list and file")

    first, second = run.tool_results()
    assert first.content.endswith("It takes: Mcp_Github_list_issues(v) — list the issues")
    assert second.content.endswith("It takes: Mcp_Github_create_issue(v) — open an issue")


# ── a policy that changes under a running run ───────────────────────────────


@dataclass
class _AdmitEverything(ScriptedTool):
    """Stands in for a host tool that switches a server on mid-run, as a host's
    MCP switch does: it replaces the engine's policy while the run goes on."""

    tool_name: str = "Enable"
    description: str = "switch a server on"
    engine: Any = None

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.engine.config = replace(self.engine.config, tool_visibility_policy=ToolVisibilityPolicy())
        return await super().invoke(context, arguments)


async def test_tools_a_changed_policy_admits_are_held_back_like_the_rest(
    scenario: ScenarioFactory,
) -> None:
    """The decision was keyed on the catalogue alone, so a server switched on
    mid-run — its tools already registered for another session — arrived on
    the surface whole, however large, instead of in the catalogue."""
    switch = _AdmitEverything()
    run = _with_search(
        scenario(
            tools=[*_tools(), switch],
            tool_visibility_policy=ToolVisibilityPolicy(
                blocked={"Mcp_Github_list_issues", "Mcp_Github_create_issue"}
            ),
        )
    )
    switch.engine = run.engine
    run.llm.queue_tool_call_response(tool_call_id="e-1", tool_name="Enable", tool_input={})
    run.llm.queue_response(text="on")
    await run.run("switch github on")

    assert run.advertised_tool_names(0) == ["Note", "Zeta", "Enable"]
    assert run.advertised_tool_names(1) == ["Note", "Zeta", "Enable", "ToolSearch"]
    assert "Mcp_Github_create_issue, Mcp_Github_list_issues" in _system_text(run, 1)


@dataclass
class _SwitchOff(ScriptedTool):
    """Stands in for the operator switching an unrelated tool off mid-run."""

    tool_name: str = "SwitchOff"
    description: str = "switch a tool off"
    engine: Any = None
    off: frozenset[str] = frozenset()

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.engine.config = replace(
            self.engine.config, tool_visibility_policy=ToolVisibilityPolicy(blocked=set(self.off))
        )
        return await super().invoke(context, arguments)


async def test_a_group_held_back_stays_held_back_when_the_surface_shrinks_mid_run(
    scenario: ScenarioFactory,
) -> None:
    """Remade from scratch on every policy change, the decision flipped when an
    unrelated tool was switched off: the surface fitted again, the held-back
    group came back, and the catalogue at the head of the cached prompt was
    rewritten mid-run."""
    long = "does a great many things, each of them described at length. "
    switch = _SwitchOff(off=frozenset({"Huge"}))
    tools = [
        ScriptedTool(tool_name="Note", description="record a note"),
        ScriptedTool(tool_name="Huge", description=long * 60),
        *(ScriptedTool(tool_name=f"Browser{i}", description=long * 6) for i in range(2)),
        switch,
    ]
    run = scenario(tools=tools)
    switch.engine = run.engine
    run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group("browser", "Drive a web browser", prefix="Browser")
    run.llm.queue_tool_call_response(tool_call_id="o-1", tool_name="SwitchOff", tool_input={})
    run.llm.queue_response(text="done")
    await run.run("hello")

    assert run.advertised_tool_names(0) == ["Note", "Huge", "SwitchOff", "ToolSearch"]
    assert run.advertised_tool_names(1) == ["Note", "SwitchOff", "ToolSearch"]
    assert _system_text(run, 1) == _system_text(run, 0)
    assert "- browser: Drive a web browser" in _system_text(run, 1)
