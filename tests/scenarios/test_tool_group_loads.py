"""Load modes and group rules, driven through the loop as a host drives it.

A ``lazy`` group is the operator saying "this family is rare": its tools stay
off the surface even when everything fits, one line of the catalogue names
it, and a model that needs it loads it whole in one call. Its rules reach the
model once, with the first of its tools — and a model that skips the loading
and calls a tool by the name it read in the catalogue gets the rules first
and the call on its next step. Everything here is asserted from what the
provider received and what the reader was handed.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.types import MessageRole, TextBlock
from protocore.runtime.events import EventType
from protocore.runtime.tool_deferral import ensure_tool_deferral, note_prompt_prefix_restarted
from protocore.runtime.tool_surface import forget_tool_surfaces
from protocore.tools import ToolSearchTool

from .conftest import Scenario, ScenarioFactory, ScriptedTool, default_rc

_RULES = "Ask the user before submitting a form.\nClose the page when you are done."


@pytest.fixture(autouse=True)
def _forget_surfaces() -> Iterator[None]:
    forget_tool_surfaces()
    yield
    forget_tool_surfaces()


def _tools(description: str = "drive the browser") -> list[ScriptedTool]:
    return [
        ScriptedTool(tool_name="Note", description="record a note"),
        ScriptedTool(tool_name="BrowserOpen", description=f"open a page; {description}"),
        ScriptedTool(tool_name="BrowserClick", description=f"click on a page; {description}"),
        ScriptedTool(tool_name="Zeta", description="the last tool by name"),
    ]


def _lazy(
    run: Scenario, *, search: bool = True, load: str = "lazy", rules: str = _RULES
) -> Scenario:
    if search:
        run.tools.register(ToolSearchTool(run.tools))
    run.tools.declare_group(
        "browser", "Drive a web browser", prefix="Browser", load=load, instructions=rules
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


def _adverts(run: Scenario) -> list[dict[str, Any]]:
    return [dict(evt.payload) for evt in run.events_of(EventType.TOOL_SURFACE_ADVERTISED)]


def _group_loads(run: Scenario) -> list[tuple[str, str, list[str]]]:
    return [
        (evt.payload["group"], evt.payload["via"], list(evt.payload["tools"]))
        for evt in run.events_of(EventType.TOOL_GROUP_LOADED)
    ]


def _invocations(run: Scenario, name: str) -> list[dict[str, Any]]:
    tool = run.tools.get(name)
    assert isinstance(tool, ScriptedTool)
    return tool.invocations


# ── when a lazy group is held back ──────────────────────────────────────────


async def test_a_lazy_group_is_held_back_while_everything_fits(
    scenario: ScenarioFactory,
) -> None:
    run = _lazy(scenario(tools=_tools()))
    run.llm.queue_response(text="done")
    await run.run("hello")

    assert run.advertised_tool_names(0) == ["Note", "Zeta", "ToolSearch"]
    catalogue = _system_text(run, 0)
    assert "- browser: Drive a web browser. Tools: BrowserClick, BrowserOpen\n" in catalogue
    assert 'pass group="<name>" to load a whole group' in catalogue
    assert "A dedicated tool for the job is better than a workaround" in catalogue
    # Nothing of it is loaded, so its rules are not spent yet.
    assert "Rules for the browser tools" not in catalogue
    advert = _adverts(run)[0]
    assert advert["deferred_tool_groups"] == ["browser"]
    assert advert["tool_deferral_reasons"] == ["lazy"]
    assert advert["tool_groups"] == [{"name": "browser", "load": "lazy", "state": "deferred"}]


async def test_a_lazy_group_is_advertised_when_nothing_could_load_it(
    scenario: ScenarioFactory,
) -> None:
    run = _lazy(scenario(tools=_tools()), search=False)
    run.llm.queue_response(text="done")
    await run.run("hello")

    assert sorted(run.advertised_tool_names(0)) == ["BrowserClick", "BrowserOpen", "Note", "Zeta"]
    # Its tools are in front of the model, so its rules are too.
    assert f"Rules for the browser tools:\n{_RULES}" in _system_text(run, 0)
    assert _adverts(run)[0]["tool_groups"] == [
        {"name": "browser", "load": "lazy", "state": "advertised"}
    ]


async def test_a_per_run_override_beats_the_declaration(scenario: ScenarioFactory) -> None:
    run = _lazy(scenario(tools=_tools(), tool_group_loads={"browser": "auto"}))
    run.llm.queue_response(text="done")
    await run.run("hello")

    assert sorted(run.advertised_tool_names(0)) == ["BrowserClick", "BrowserOpen", "Note", "Zeta"]
    assert _adverts(run)[0]["tool_groups"] == [
        {"name": "browser", "load": "auto", "state": "advertised"}
    ]


async def test_an_eager_group_stays_over_the_budget_but_not_over_the_provider_limit(
    scenario: ScenarioFactory,
) -> None:
    long = "does a great many things, each of them described at length. " * 20
    over_budget = _lazy(scenario(tools=_tools(long)), load="eager", rules="")
    over_budget.llm.queue_response(text="done")
    await over_budget.run("hello")
    assert sorted(over_budget.advertised_tool_names(0)) == [
        "BrowserClick",
        "BrowserOpen",
        "Note",
        "Zeta",
    ]

    over_limit = _lazy(
        scenario(tools=_tools(), rc=default_rc(max_advertised_tools=3)), load="eager", rules=""
    )
    over_limit.llm.queue_response(text="done")
    await over_limit.run("hello")
    assert over_limit.advertised_tool_names(0) == ["Note", "Zeta", "ToolSearch"]
    assert _adverts(over_limit)[0]["tool_deferral_reasons"] == ["count"]


# ── loading a whole group ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "arguments",
    [{"group": "browser"}, {"query": "select:group:browser"}],
    ids=["group", "select"],
)
async def test_a_group_is_loaded_whole_with_its_rules_once(
    scenario: ScenarioFactory, arguments: dict[str, Any]
) -> None:
    run = _lazy(scenario(tools=_tools()))
    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input=arguments
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="BrowserOpen", tool_input={"v": "https://example.org"}
    )
    run.llm.queue_response(text="opened")
    await run.run("open the page")

    base = run.advertised_tool_names(0)
    assert run.advertised_tool_names(1) == [*base, "BrowserClick", "BrowserOpen"]
    search_result, open_result = run.tool_results()
    assert search_result.content.startswith(
        "Loaded, and callable from your next step: BrowserClick, BrowserOpen."
    )
    assert search_result.content.endswith(f"Rules for the browser tools:\n{_RULES}")
    # Loaded and advertised, the tool now simply runs.
    assert open_result.content == "ok"
    assert _invocations(run, "BrowserOpen") == [{"v": "https://example.org"}]
    assert _group_loads(run) == [("browser", "group", ["BrowserClick", "BrowserOpen"])]
    # The catalogue at the head of the prompt did not move for the load.
    assert _system_text(run, 0) == _system_text(run, 1) == _system_text(run, 2)
    assert run.engine.context_manager.discovered_tool_groups() == {
        "BrowserClick": "browser",
        "BrowserOpen": "browser",
    }
    assert _adverts(run)[1]["tool_groups"] == [
        {"name": "browser", "load": "lazy", "state": "loaded"}
    ]


async def test_rules_are_given_with_the_first_tool_of_the_group_only(
    scenario: ScenarioFactory,
) -> None:
    run = _lazy(scenario(tools=_tools()))
    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"query": "select:BrowserOpen"}
    )
    run.llm.queue_tool_call_response(
        tool_call_id="s-2", tool_name="ToolSearch", tool_input={"select": "BrowserClick"}
    )
    run.llm.queue_response(text="both loaded")
    await run.run("load them")

    first, second = run.tool_results()
    assert "Rules for the browser tools" in first.content
    assert "Rules for the browser tools" not in second.content
    assert _group_loads(run) == [
        ("browser", "select", ["BrowserOpen"]),
        ("browser", "select", ["BrowserClick"]),
    ]


# ── a call by the name the catalogue gave ───────────────────────────────────


async def test_a_blind_call_gets_the_rules_first_and_runs_on_the_retry(
    scenario: ScenarioFactory,
) -> None:
    run = _lazy(scenario(tools=_tools()))
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="BrowserOpen", tool_input={"v": "https://example.org"}
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-2", tool_name="BrowserOpen", tool_input={"v": "https://example.org"}
    )
    run.llm.queue_response(text="opened")
    await run.run("open the page")

    held, ran = run.tool_results()
    assert not held.is_error
    assert held.content == (
        "Not run yet: BrowserOpen was not in your tool list, and the browser tools "
        "come with rules to read before the first call.\n\n"
        f"Rules for the browser tools:\n{_RULES}\n\n"
        "The tool is loaded now; call it again."
    )
    assert ran.content == "ok"
    assert _invocations(run, "BrowserOpen") == [{"v": "https://example.org"}]
    assert run.advertised_tool_names(1)[-1] == "BrowserOpen"
    unadvertised = run.events_of(EventType.TOOL_UNADVERTISED_CALL)
    assert [evt.payload["executed"] for evt in unadvertised] == [False]
    assert _group_loads(run) == [("browser", "direct_call", ["BrowserOpen"])]
    # Nothing failed, so nothing is charged to the tool.
    assert run.engine.context_manager.called_discovered_tool_names() == ("BrowserOpen",)


async def test_two_blind_calls_of_one_group_in_one_message_both_wait(
    scenario: ScenarioFactory,
) -> None:
    run = _lazy(scenario(tools=_tools()))
    run.llm.queue_multi_tool_call_response(
        tool_calls=[
            ("c-1", "BrowserOpen", {"v": "a"}),
            ("c-2", "BrowserClick", {"v": "b"}),
        ]
    )
    run.llm.queue_response(text="read the rules")
    await run.run("open and click")

    first, second = run.tool_results()
    assert "Rules for the browser tools" in first.content
    assert second.content == (
        "Not run yet: BrowserClick was not in your tool list. The rules for the "
        "browser tools are in another result of this step. The tool is loaded "
        "now; call it again."
    )
    assert _invocations(run, "BrowserOpen") == []
    assert _invocations(run, "BrowserClick") == []


async def test_a_group_without_rules_keeps_running_blind_calls(
    scenario: ScenarioFactory,
) -> None:
    run = _lazy(scenario(tools=_tools()), rules="")
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="BrowserOpen", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="opened")
    await run.run("open")

    assert _invocations(run, "BrowserOpen") == [{"v": "x"}]
    assert _group_loads(run) == [("browser", "direct_call", ["BrowserOpen"])]


# ── tools loaded before the run began ───────────────────────────────────────


async def test_a_seeded_group_brings_its_rules_in_the_catalogue(
    scenario: ScenarioFactory,
) -> None:
    run = _lazy(scenario(tools=_tools(), discovered_tools=("BrowserOpen",)))
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="BrowserClick", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="clicked")
    await run.run("click")

    catalogue = _system_text(run, 0)
    assert (
        "- browser: Drive a web browser. Tools: BrowserClick, BrowserOpen\n"
        "  Rules for the browser tools:\n"
        "  Ask the user before submitting a form.\n"
        "  Close the page when you are done.\n"
    ) in catalogue
    # The rules were given in the catalogue, so a blind call of the other tool
    # runs at once.
    assert _invocations(run, "BrowserClick") == [{"v": "x"}]
    assert _system_text(run, 1) == catalogue
    assert _group_loads(run)[0] == ("browser", "seed", ["BrowserOpen"])
    assert ("browser", "direct_call", ["BrowserClick"]) in _group_loads(run)


async def test_after_a_compaction_the_catalogue_carries_the_loaded_rules(
    scenario: ScenarioFactory,
) -> None:
    run = _lazy(scenario(tools=_tools()))
    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"group": "browser"}
    )
    run.llm.queue_response(text="loaded")
    await run.run("load the browser")
    assert "Rules for the browser tools" not in _system_text(run, 1)

    # Where the prefix starts over, the rules a result gave are written into
    # the catalogue, which outlives the summary.
    note_prompt_prefix_restarted(run.engine)
    run.engine.rearm()
    run.llm.queue_response(text="still here")
    await run.run("and now?")
    assert "  Rules for the browser tools:" in _system_text(run, 2)


# ── across a resume ─────────────────────────────────────────────────────────


async def test_a_resumed_run_does_not_hold_back_a_call_it_already_cleared(
    scenario: ScenarioFactory,
) -> None:
    first = _lazy(scenario(tools=_tools()))
    first.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="BrowserOpen", tool_input={"v": "x"}
    )
    first.llm.queue_response(text="read them")
    await first.run("open")
    snapshot = first.engine.snapshot()
    assert snapshot["tool_group_rules_given"] == ["browser"]

    # BrowserOpen is blocked in the new process, so no loaded tool of the
    # group brings the rules back through the catalogue: only the snapshot
    # says they were given.
    second = _lazy(
        scenario(
            tools=_tools(),
            tool_visibility_policy=ToolVisibilityPolicy(blocked={"BrowserOpen"}),
        )
    )
    await second.engine.resume_from_snapshot(snapshot)
    second.engine.rearm()
    second.llm.queue_tool_call_response(
        tool_call_id="c-2", tool_name="BrowserClick", tool_input={"v": "y"}
    )
    second.llm.queue_response(text="clicked")
    await second.run("click")

    assert "Rules for the browser tools" not in _system_text(second, 0)
    assert _invocations(second, "BrowserClick") == [{"v": "y"}]


# ── groups seeded whole ─────────────────────────────────────────────────────


def _many() -> list[ScriptedTool]:
    return [
        *_tools(),
        ScriptedTool(tool_name="ScheduleAdd", description="schedule a job"),
        ScriptedTool(tool_name="ScheduleList", description="list the jobs"),
        ScriptedTool(tool_name="ScheduleDrop", description="drop a job"),
    ]


def _with_schedule(run: Scenario) -> Scenario:
    _lazy(run)
    run.tools.declare_group("schedule", "Jobs that run later", prefix="Schedule", load="lazy")
    return run


async def test_seeded_groups_are_one_entry_each_under_the_cap(
    scenario: ScenarioFactory,
) -> None:
    # Five tools under a cap of two: as bare names three would be left out.
    run = _with_schedule(
        scenario(
            tools=_many(),
            rc=default_rc(pinned_tool_max_count=2),
            loaded_tool_groups=["BROWSER", "schedule", "nothing"],
        )
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="ScheduleList", tool_input={"v": "all"}
    )
    run.llm.queue_response(text="listed")
    await run.run("what is scheduled?")

    base = ["Note", "Zeta", "ToolSearch"]
    loaded = ["BrowserClick", "BrowserOpen", "ScheduleAdd", "ScheduleDrop", "ScheduleList"]
    assert run.advertised_tool_names(0) == [*base, *loaded]
    # The browser's rules are in the catalogue, as for any group loaded
    # before the run began, so its tools run at once.
    assert (
        "- browser: Drive a web browser. Tools: BrowserClick, BrowserOpen\n"
        "  Rules for the browser tools:\n"
    ) in _system_text(run, 0)
    assert _group_loads(run) == [
        ("browser", "seed", ["BrowserClick", "BrowserOpen"]),
        ("schedule", "seed", ["ScheduleAdd", "ScheduleDrop", "ScheduleList"]),
    ]
    assert _adverts(run)[0]["tool_groups"] == [
        {"name": "browser", "load": "lazy", "state": "loaded"},
        {"name": "schedule", "load": "lazy", "state": "loaded"},
    ]
    manager = run.engine.context_manager
    assert manager.loaded_tool_group_names() == ("browser", "schedule")
    # Seeded is not called: only the group the run used is worth carrying.
    assert manager.called_discovered_tool_names() == ("ScheduleList",)
    rows = {row["name"]: row for row in run.engine.snapshot()["discovered_tools"]}
    assert rows["ScheduleAdd"]["group"] == "schedule"
    assert rows["ScheduleAdd"]["called"] is False


async def test_a_seeded_group_over_the_cap_goes_whole_and_the_older_seed_first(
    scenario: ScenarioFactory,
) -> None:
    run = _with_schedule(
        scenario(
            tools=_many(),
            rc=default_rc(pinned_tool_max_count=1),
            discovered_tools=("Zeta",),
            loaded_tool_groups=("schedule",),
        )
    )
    run.llm.queue_response(text="done")
    await run.run("hello")

    # The named tool is the older entry, so it is the one left out; the group
    # comes in whole.
    assert run.engine.context_manager.discovered_tool_names() == (
        "ScheduleAdd",
        "ScheduleDrop",
        "ScheduleList",
    )
    assert _group_loads(run) == [
        ("schedule", "seed", ["ScheduleAdd", "ScheduleDrop", "ScheduleList"])
    ]


async def test_a_seeded_group_loads_only_what_the_run_may_call(
    scenario: ScenarioFactory,
) -> None:
    run = _with_schedule(
        scenario(
            tools=_many(),
            loaded_tool_groups=("schedule", "browser"),
            tool_visibility_policy=ToolVisibilityPolicy(
                blocked={"ScheduleDrop", "BrowserOpen", "BrowserClick"}
            ),
        )
    )
    run.llm.queue_response(text="done")
    await run.run("hello")

    manager = run.engine.context_manager
    assert manager.discovered_tool_names() == ("ScheduleAdd", "ScheduleList")
    assert manager.loaded_tool_group_names() == ("schedule",)
    assert _group_loads(run) == [("schedule", "seed", ["ScheduleAdd", "ScheduleList"])]


async def test_a_resumed_run_takes_its_groups_from_the_snapshot_not_the_seed(
    scenario: ScenarioFactory,
) -> None:
    first = _with_schedule(scenario(tools=_many()))
    first.llm.queue_response(text="nothing loaded")
    await first.run("hello")
    snapshot = first.engine.snapshot()

    second = _with_schedule(scenario(tools=_many(), loaded_tool_groups=("schedule",)))
    await second.engine.resume_from_snapshot(snapshot)
    second.engine.rearm()
    second.llm.queue_response(text="still nothing")
    await second.run("and now?")

    assert second.engine.context_manager.discovered_tool_names() == ()
    assert _group_loads(second) == []


async def test_a_decision_asked_for_before_the_first_request_sees_the_seeded_groups(
    scenario: ScenarioFactory,
) -> None:
    # A host writes its prompt for the surface the first request will have,
    # and asks for the decision to do it.
    run = _with_schedule(scenario(tools=_many(), loaded_tool_groups=("browser",)))
    decision = ensure_tool_deferral(run.engine)
    assert f"  Rules for the browser tools:\n  {_RULES.splitlines()[0]}" in decision.catalogue
    assert run.engine.context_manager.loaded_tool_group_names() == ("browser",)
    run.llm.queue_response(text="done")
    await run.run("hello")
    assert run.advertised_tool_names(0)[-2:] == ["BrowserClick", "BrowserOpen"]
    assert _group_loads(run) == [("browser", "seed", ["BrowserClick", "BrowserOpen"])]
