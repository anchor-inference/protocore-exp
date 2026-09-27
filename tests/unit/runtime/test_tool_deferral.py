"""Which tool groups a run holds back, and how the catalogue names them.

The planner is a pure function of the would-be surface, the declared groups
and the constants, so these tests state its rules one at a time: nothing moves
while the surface fits, dynamic groups go first and always, the others go
largest first only while the surface is over, and the protected floor never
goes at all. The catalogue is asserted byte for byte because its bytes are the
head of a cached prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import ToolGroup, tool_group_of
from protocore.contracts.tool_roles import ToolRole, ToolRoleMap
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.events import EventType
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.runtime.tool_deferral import (
    NO_DEFERRAL,
    discovery_tool_names,
    plan_tool_deferral,
    render_tool_catalogue,
)
from protocore.runtime.tool_registry import ToolRegistry
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)
from protocore.tools.tool_search import ToolSearchTool
from tests.unit.runtime._tool_fixtures import MockTool


@dataclass
class GroupedTool(MockTool):
    """A tool that names its own group, as a host's own tool would."""

    tool_group: str = ""


def _padded(name: str, *, words: int = 40, group: str = "") -> GroupedTool:
    # A description long enough that a handful of these weigh something
    # against a small window.
    return GroupedTool(tool_name=name, description="word " * words, tool_group=group)


def _search() -> ToolSearchTool:
    return ToolSearchTool(ToolRegistry())


def _plan(
    tools: list[MockTool | ToolSearchTool],
    groups: list[ToolGroup],
    *,
    protected: frozenset[str] = frozenset(),
    rc: LoopConstants | None = None,
    restored: list[str] | None = None,
    loads: dict[str, str] | None = None,
) -> tuple[tuple[str, ...], frozenset[str], str, tuple[str, ...]]:
    decision = plan_tool_deferral(
        tools=tools,
        groups=groups,
        protected=protected,
        discovery_names=discovery_tool_names(tools, ToolRoleMap()),
        rc=rc or LoopConstants(),
        restored=restored,
        loads=loads,
    )
    return (
        decision.deferred_groups,
        decision.deferred_names,
        decision.catalogue,
        decision.reasons,
    )


# ── membership ──────────────────────────────────────────────────────────────


def test_an_explicit_group_wins_and_the_longest_prefix_wins_among_prefixes() -> None:
    groups = [
        ToolGroup(name="mcp", prefix="Mcp_"),
        ToolGroup(name="github", prefix="Mcp_Github_"),
    ]
    assert tool_group_of(MockTool(tool_name="Mcp_Github_list"), groups) == "github"
    assert tool_group_of(MockTool(tool_name="Mcp_Jira_get"), groups) == "mcp"
    assert tool_group_of(GroupedTool(tool_name="Mcp_Github_x", tool_group="own"), groups) == "own"
    assert tool_group_of(MockTool(tool_name="Read"), groups) == ""


def test_a_discovery_tool_is_known_by_its_own_role_or_the_hosts() -> None:
    host_search = MockTool(tool_name="FindTools")
    roles = ToolRoleMap.declare({"FindTools": [ToolRole.discovers_tools]})
    names = discovery_tool_names([_search(), host_search, MockTool(tool_name="Read")], roles)
    assert names == frozenset({"ToolSearch", "FindTools"})


# ── when nothing moves ──────────────────────────────────────────────────────


def test_nothing_is_held_back_while_the_surface_fits() -> None:
    tools = [_search(), _padded("Browse", group="browser"), MockTool(tool_name="Read")]
    groups = [ToolGroup(name="browser", description="Drive a browser")]
    assert _plan(tools, groups) == ((), frozenset(), "", ())


def test_nothing_is_held_back_with_deferral_off_or_without_a_discovery_tool() -> None:
    tools = [_padded(f"Big{i}", words=400, group="big") for i in range(10)]
    groups = [ToolGroup(name="big", dynamic=True)]
    assert _plan(tools, groups)[0] == ()
    off = LoopConstants(tool_deferral_mode="off")
    assert _plan([_search(), *tools], groups, rc=off)[0] == ()


def test_a_tool_in_no_group_is_never_held_back() -> None:
    tools = [_search(), *(_padded(f"Loose{i}", words=400) for i in range(10))]
    rc = LoopConstants(model_context_window=4_096)
    assert _plan(tools, [], rc=rc)[0] == ()


# ── what moves, and in which order ──────────────────────────────────────────


def test_a_dynamic_group_is_held_back_even_when_everything_fits() -> None:
    tools = [
        _search(),
        MockTool(tool_name="Read"),
        MockTool(tool_name="Mcp_Github_list"),
        MockTool(tool_name="Mcp_Github_create"),
    ]
    groups = [ToolGroup(name="github", description="GitHub", dynamic=True, prefix="Mcp_Github_")]
    deferred, names, catalogue, reasons = _plan(tools, groups)
    assert deferred == ("github",)
    assert names == {"Mcp_Github_list", "Mcp_Github_create"}
    assert reasons == ("dynamic",)
    assert "Mcp_Github_create, Mcp_Github_list" in catalogue


def test_over_the_token_budget_the_largest_groups_go_first_until_it_fits() -> None:
    # A 4k window gives the definitions a 1k budget. "large" alone is well
    # over it; without it the rest fits, so "small" and "medium" stay.
    tools = [
        _search(),
        *(_padded(f"Small{i}", words=10, group="small") for i in range(2)),
        *(_padded(f"Medium{i}", words=40, group="medium") for i in range(2)),
        *(_padded(f"Large{i}", words=200, group="large") for i in range(4)),
    ]
    groups = [ToolGroup(name=name) for name in ("small", "medium", "large")]
    rc = LoopConstants(model_context_window=4_096)
    deferred, names, _, reasons = _plan(tools, groups, rc=rc)
    assert deferred == ("large",)
    assert names == {f"Large{i}" for i in range(4)}
    assert reasons == ("tokens",)


def test_dynamic_groups_go_before_any_other_group() -> None:
    tools = [
        _search(),
        *(_padded(f"Own{i}", words=200, group="own") for i in range(4)),
        *(_padded(f"Mcp_Small_{i}", words=5) for i in range(2)),
    ]
    groups = [ToolGroup(name="own"), ToolGroup(name="small", dynamic=True, prefix="Mcp_Small_")]
    rc = LoopConstants(model_context_window=4_096)
    deferred, _, _, reasons = _plan(tools, groups, rc=rc)
    assert deferred == ("small", "own")
    assert reasons == ("dynamic", "tokens")


def test_over_the_provider_tool_limit_room_is_left_for_loaded_tools() -> None:
    tools = [
        _search(),
        *(MockTool(tool_name=f"Core{i}") for i in range(6)),
        *(GroupedTool(tool_name=f"A{i}", tool_group="a") for i in range(4)),
        GroupedTool(tool_name="B0", tool_group="b"),
    ]
    groups = [ToolGroup(name="a"), ToolGroup(name="b")]
    # Eleven tools against a limit of ten. Holding back "a" leaves eight with
    # the search tool, and eight plus two loaded tools is exactly ten; with
    # room for three loaded tools, "b" has to go as well.
    rc = LoopConstants(max_advertised_tools=10, pinned_tool_max_count=2)
    deferred, _, _, reasons = _plan(tools, groups, rc=rc)
    assert deferred == ("a",)
    assert reasons == ("count",)
    tighter = LoopConstants(max_advertised_tools=10, pinned_tool_max_count=3)
    assert _plan(tools, groups, rc=tighter)[0] == ("a", "b")


def test_the_protected_floor_stays_whatever_its_group() -> None:
    tools = [
        _search(),
        _padded("Mcp_Github_list"),
        _padded("Mcp_Github_create"),
        GroupedTool(tool_name="Loud", always_load=True, tool_group="github"),
    ]
    groups = [ToolGroup(name="github", dynamic=True, prefix="Mcp_Github_")]
    _, names, _, _ = _plan(tools, groups, protected=frozenset({"Mcp_Github_list", "Loud"}))
    assert names == {"Mcp_Github_create"}


def test_a_restored_decision_is_kept_rather_than_remeasured() -> None:
    tools = [_search(), _padded("Browse", group="browser"), _padded("Mcp_X_a")]
    groups = [
        ToolGroup(name="browser", description="Drive a browser"),
        ToolGroup(name="x", dynamic=True, prefix="Mcp_X_"),
    ]
    # Everything fits and "browser" is not dynamic, so a fresh decision would
    # let it back; the snapshot said "browser", and "gone" no longer has any
    # tools. "x" is dynamic, so it is held back whatever the snapshot said.
    deferred, names, _, reasons = _plan(tools, groups, restored=["browser", "gone"])
    assert deferred == ("browser", "x")
    assert names == {"Browse", "Mcp_X_a"}
    assert reasons == ("restored", "dynamic")
    assert _plan(tools, groups, restored=[])[0] == ("x",)


def test_a_restored_decision_still_holds_back_a_server_that_connected_since() -> None:
    """A snapshot taken before a large server connected named nothing; replayed
    as the whole answer, it put that server's every tool on the surface, over a
    provider's limit on the number of tools."""
    tools = [
        _search(),
        *(MockTool(tool_name=f"Own{i}") for i in range(20)),
        *(MockTool(tool_name=f"Mcp_Big_t{i}") for i in range(400)),
    ]
    groups = [ToolGroup(name="big", description="Big", dynamic=True, prefix="Mcp_Big_")]
    rc = LoopConstants(model_context_window=128_000, max_advertised_tools=350)
    deferred, names, _, reasons = _plan(tools, groups, rc=rc, restored=[])
    assert deferred == ("big",)
    assert len(names) == 400
    assert "dynamic" in reasons


def test_a_restored_decision_is_held_to_the_limits_on_top() -> None:
    tools = [
        _search(),
        *(MockTool(tool_name=f"Core{i}") for i in range(6)),
        *(GroupedTool(tool_name=f"A{i}", tool_group="a") for i in range(4)),
        *(GroupedTool(tool_name=f"B{i}", tool_group="b") for i in range(2)),
    ]
    groups = [ToolGroup(name="a"), ToolGroup(name="b")]
    rc = LoopConstants(max_advertised_tools=10, pinned_tool_max_count=2)
    # The snapshot held back only the small group; with "a" still on the
    # surface the request carries 11 tools against a limit of 10.
    deferred, _, _, reasons = _plan(tools, groups, rc=rc, restored=["b"])
    assert deferred == ("b", "a")
    assert reasons == ("restored", "count")


# ── without a discovery tool ────────────────────────────────────────────────


def test_without_a_discovery_tool_dynamic_groups_go_only_over_the_provider_limit() -> None:
    """A registry without an admitted ToolSearch used to hold nothing back, so a
    large server reached a provider that refuses more than its limit whole."""
    tools = [
        *(MockTool(tool_name=f"Own{i}") for i in range(20)),
        *(MockTool(tool_name=f"Mcp_Big_t{i}") for i in range(40)),
        *(GroupedTool(tool_name=f"Board{i}", tool_group="board") for i in range(3)),
    ]
    groups = [
        ToolGroup(name="big", description="Big", dynamic=True, prefix="Mcp_Big_"),
        ToolGroup(name="board"),
    ]
    under = LoopConstants(max_advertised_tools=100)
    assert _plan(tools, groups, rc=under)[0] == ()
    assert _plan(tools, groups)[0] == ()  # no limit at all
    over = LoopConstants(max_advertised_tools=50, pinned_tool_max_count=2)
    deferred, names, catalogue, reasons = _plan(tools, groups, rc=over)
    assert deferred == ("big",)
    assert len(names) == 40
    assert reasons == ("dynamic", "count")
    assert "Call one by its exact name" in catalogue
    assert "ToolSearch" not in catalogue
    # A token budget alone is not a reason without a way to load them back.
    small = LoopConstants(model_context_window=4_096)
    assert _plan(tools, groups, rc=small)[0] == ()


# ── the catalogue ───────────────────────────────────────────────────────────


def test_the_catalogue_names_exact_tools_and_prefixes_for_large_dynamic_groups() -> None:
    declared = {
        "github": ToolGroup(
            name="github",
            description="GitHub issues and pull requests.",
            dynamic=True,
            prefix="Mcp_Github_",
        ),
        "schedule": ToolGroup(name="schedule", description="Timed and recurring jobs"),
    }
    catalogue = render_tool_catalogue(
        {
            "schedule": ["ScheduleCreate", "IntentCreate"],
            "github": [f"Mcp_Github_t{i}" for i in range(3)],
            "bare": ["Lonely"],
        },
        declared,
        discovery_tool="ToolSearch",
        max_listed_names=2,
    )
    assert catalogue == (
        "<system-reminder>\n"
        "These tools are available but not loaded, and can be loaded at any time. "
        'Before calling one, load it with ToolSearch: pass group="<name>" to load a '
        'whole group, or "select:" and exact names to load particular tools (e.g. '
        '"select:Name1,Name2"), or describe what you need. A dedicated tool for the '
        "job is better than a workaround with a general one such as a shell command, "
        "so load the tool rather than improvising.\n"
        "\n"
        "- bare: tools Lonely\n"
        "- schedule: Timed and recurring jobs. Tools: ScheduleCreate, IntentCreate\n"
        "\n"
        "Tools of connected servers:\n"
        "- github: GitHub issues and pull requests. Tools: Mcp_Github_* (3 tools)\n"
        "</system-reminder>"
    )
    assert render_tool_catalogue({}, declared, discovery_tool="ToolSearch", max_listed_names=2) == ""


def test_the_hosts_own_groups_come_before_the_connected_servers() -> None:
    # Sorted by name alone, a server spelt in capitals would come before
    # every one of the host's groups.
    declared = {
        "MCP server aaa": ToolGroup(name="MCP server aaa", description="A", dynamic=True),
        "MCP server zzz": ToolGroup(name="MCP server zzz", description="Z", dynamic=True),
        "browser": ToolGroup(name="browser", description="Drive a browser"),
        "loop": ToolGroup(name="loop", description="This session's loop"),
    }
    deferred = {
        "MCP server zzz": ["Mcp_Zzz_b"],
        "loop": ["LoopStop"],
        "MCP server aaa": ["Mcp_Aaa_a"],
        "browser": ["BrowserOpen"],
    }
    catalogue = render_tool_catalogue(
        deferred, declared, discovery_tool="ToolSearch", max_listed_names=12
    )
    body = catalogue.split("\n\n", 1)[1]
    assert body.splitlines()[:6] == [
        "- browser: Drive a browser. Tools: BrowserOpen",
        "- loop: This session's loop. Tools: LoopStop",
        "",
        "Tools of connected servers:",
        "- MCP server aaa: A. Tools: Mcp_Aaa_a",
        "- MCP server zzz: Z. Tools: Mcp_Zzz_b",
    ]
    # The same decision in another order is the same bytes.
    assert catalogue == render_tool_catalogue(
        dict(reversed(list(deferred.items()))),
        declared,
        discovery_tool="ToolSearch",
        max_listed_names=12,
    )

    # Servers alone need no sections; and the order holds without a
    # discovery tool too.
    servers_only = render_tool_catalogue(
        {"MCP server aaa": ["Mcp_Aaa_a"]}, declared, discovery_tool="ToolSearch", max_listed_names=12
    )
    assert "Tools of connected servers" not in servers_only
    assert "- MCP server aaa: A. Tools: Mcp_Aaa_a" in servers_only
    no_discovery = render_tool_catalogue(deferred, declared, discovery_tool="", max_listed_names=12)
    assert "\n- browser: Drive a browser. Tools: BrowserOpen\n- loop:" in no_discovery
    assert "\n\nTools of connected servers:\n- MCP server aaa" in no_discovery


def test_the_same_decision_renders_the_same_bytes() -> None:
    tools = [_search(), *(MockTool(tool_name=f"Mcp_G_{c}") for c in "cab")]
    groups = [ToolGroup(name="g", description="G", dynamic=True, prefix="Mcp_G_")]
    first = _plan(tools, groups)[2]
    again = _plan(list(reversed(tools)), groups)[2]
    assert first == again
    assert "Mcp_G_a, Mcp_G_b, Mcp_G_c" in first


def test_no_deferral_is_the_empty_decision() -> None:
    assert NO_DEFERRAL.catalogue == ""
    assert NO_DEFERRAL.deferred_names == frozenset()


@pytest.mark.parametrize("name", ["", "   "])
def test_a_group_needs_a_name(name: str) -> None:
    registry = ToolRegistry()
    if name:
        registry.declare_group(name, "whitespace is a name, if an odd one")
        assert [g.name for g in registry.tool_groups()] == [name]
    else:
        with pytest.raises(ValueError):
            registry.declare_group(name, "nothing")


@pytest.mark.parametrize("registry_type", [ToolRegistry, InMemoryToolRegistry])
def test_an_undeclared_group_no_longer_claims_its_prefix(
    registry_type: type[ToolRegistry] | type[InMemoryToolRegistry],
) -> None:
    """A removed MCP server's declaration outlives its tools unless the host
    drops it; a server of the same name added later would inherit it."""
    registry = registry_type()
    registry.declare_group("github", "old description", dynamic=True, prefix="Mcp_Github_")
    registry.declare_group("jira", "Jira", dynamic=True, prefix="Mcp_Jira_")
    registry.undeclare_group("github")
    registry.undeclare_group("github")  # idempotent
    registry.undeclare_group("never-declared")
    assert [group.name for group in registry.tool_groups()] == ["jira"]
    tool = MockTool(tool_name="Mcp_Github_list_issues")
    assert tool_group_of(tool, registry.tool_groups()) == ""


# ── load modes ──────────────────────────────────────────────────────────────


def _browser_and_notes() -> list[MockTool | ToolSearchTool]:
    return [
        _search(),
        MockTool(tool_name="Read"),
        GroupedTool(tool_name="BrowserOpen", tool_group="browser"),
        GroupedTool(tool_name="BrowserClick", tool_group="browser"),
        GroupedTool(tool_name="NoteAdd", tool_group="notes"),
    ]


def test_a_lazy_group_is_held_back_below_every_limit_when_a_search_is_there() -> None:
    groups = [ToolGroup(name="browser", description="Drive a browser", load="lazy")]
    deferred, names, catalogue, reasons = _plan(_browser_and_notes(), groups)
    assert deferred == ("browser",)
    assert names == {"BrowserOpen", "BrowserClick"}
    assert reasons == ("lazy",)
    assert "- browser: Drive a browser. Tools: BrowserClick, BrowserOpen" in catalogue
    assert 'ToolSearch: pass group="<name>" to load a whole group' in catalogue


def test_a_lazy_group_is_advertised_when_nothing_could_load_it() -> None:
    """Without a discovery tool a blind call would be the only way in, which
    costs more than the definitions: lazy is auto then."""
    tools = [tool for tool in _browser_and_notes() if not isinstance(tool, ToolSearchTool)]
    groups = [ToolGroup(name="browser", load="lazy")]
    assert _plan(tools, groups)[0] == ()


def test_a_per_run_load_overrides_the_declaration() -> None:
    groups = [ToolGroup(name="browser", load="lazy"), ToolGroup(name="notes")]
    assert _plan(_browser_and_notes(), groups, loads={"browser": "auto"})[0] == ()
    assert _plan(_browser_and_notes(), groups, loads={"notes": "lazy"})[0] == (
        "browser",
        "notes",
    )
    # An undeclared group is auto, and an override reaches it too.
    assert _plan(_browser_and_notes(), [], loads={"notes": "lazy"})[0] == ("notes",)


def test_an_eager_group_stays_over_the_token_budget_and_over_the_dynamic_rule() -> None:
    tools = [
        _search(),
        *(_padded(f"Own{i}", words=200, group="own") for i in range(4)),
        *(_padded(f"Mcp_X_{i}", words=5) for i in range(2)),
    ]
    groups = [
        ToolGroup(name="own", load="eager"),
        ToolGroup(name="x", dynamic=True, prefix="Mcp_X_", load="eager"),
    ]
    rc = LoopConstants(model_context_window=4_096)
    assert _plan(tools, groups, rc=rc)[0] == ()


def test_an_eager_group_gives_way_to_the_provider_limit_and_only_to_it() -> None:
    """A request over the provider's count is refused, which no load mode is
    worth; the room kept for loaded tools is not worth pushing an eager group
    off for, since the request drops loaded tools before it would exceed."""
    tools = [
        _search(),
        *(MockTool(tool_name=f"Core{i}") for i in range(6)),
        *(GroupedTool(tool_name=f"E{i}", tool_group="e") for i in range(4)),
    ]
    groups = [ToolGroup(name="e", load="eager")]
    at_limit = LoopConstants(max_advertised_tools=10, pinned_tool_max_count=5)
    assert _plan(tools, groups, rc=at_limit)[0] == ()
    over = LoopConstants(max_advertised_tools=9, pinned_tool_max_count=5)
    deferred, _, _, reasons = _plan(tools, groups, rc=over)
    assert deferred == ("e",)
    assert reasons == ("count",)


def test_a_group_made_eager_leaves_the_restored_floor() -> None:
    groups = [ToolGroup(name="browser", load="eager")]
    assert _plan(_browser_and_notes(), groups, restored=["browser"])[0] == ()


def test_the_decision_reports_every_group_and_its_mode() -> None:
    groups = [ToolGroup(name="browser", load="lazy")]
    decision = plan_tool_deferral(
        tools=_browser_and_notes(),
        groups=groups,
        protected=frozenset(),
        discovery_names=frozenset({"ToolSearch"}),
        rc=LoopConstants(),
        loads={"notes": "eager"},
    )
    assert decision.group_loads == (("browser", "lazy"), ("notes", "eager"))


# ── group rules in the catalogue ────────────────────────────────────────────


def _ruled_groups() -> list[ToolGroup]:
    return [
        ToolGroup(
            name="browser",
            description="Drive a browser",
            load="lazy",
            instructions="Ask before submitting a form.\nClose the page when done.",
        ),
        ToolGroup(name="notes", instructions="Keep notes short."),
    ]


def test_rules_are_written_only_for_tools_in_front_of_the_model() -> None:
    decision = plan_tool_deferral(
        tools=_browser_and_notes(),
        groups=_ruled_groups(),
        protected=frozenset(),
        discovery_names=frozenset({"ToolSearch"}),
        rc=LoopConstants(),
    )
    # notes is on the surface, so its rules are; browser is held back and
    # brings its rules when it is loaded.
    assert decision.ruled_groups == ("notes",)
    assert "Rules for the notes tools:\nKeep notes short." in decision.catalogue
    assert "Ask before submitting" not in decision.catalogue


def test_a_loaded_groups_rules_sit_under_its_line() -> None:
    decision = plan_tool_deferral(
        tools=_browser_and_notes(),
        groups=_ruled_groups(),
        protected=frozenset(),
        discovery_names=frozenset({"ToolSearch"}),
        rc=LoopConstants(),
        loaded=["BrowserOpen"],
    )
    assert decision.ruled_groups == ("browser", "notes")
    assert decision.catalogue.endswith(
        "- browser: Drive a browser. Tools: BrowserClick, BrowserOpen\n"
        "  Rules for the browser tools:\n"
        "  Ask before submitting a form.\n"
        "  Close the page when done.\n"
        "\n"
        "Rules for the notes tools:\n"
        "Keep notes short.\n"
        "</system-reminder>"
    )


def test_rules_alone_make_a_block_without_the_header() -> None:
    groups = [ToolGroup(name="notes", instructions="Keep notes short.")]
    tools = [MockTool(tool_name="Read"), GroupedTool(tool_name="NoteAdd", tool_group="notes")]
    decision = plan_tool_deferral(
        tools=tools,
        groups=groups,
        protected=frozenset(),
        discovery_names=frozenset(),
        rc=LoopConstants(),
    )
    assert decision.deferred_groups == ()
    assert decision.catalogue == (
        "<system-reminder>\nRules for the notes tools:\nKeep notes short.\n</system-reminder>"
    )


def test_a_mark_goes_into_every_rules_heading_and_is_named_once() -> None:
    decision = plan_tool_deferral(
        tools=_browser_and_notes(),
        groups=_ruled_groups(),
        protected=frozenset(),
        discovery_names=frozenset({"ToolSearch"}),
        rc=LoopConstants(),
        loaded=["BrowserOpen"],
        mark="0a1b2c3d",
    )
    catalogue = decision.catalogue
    assert catalogue.count("their heading always ends with [0a1b2c3d]") == 1
    # After the header, before the list: read before any rules it vouches for.
    assert catalogue.index("[0a1b2c3d]: here") < catalogue.index("- browser:")
    assert "  Rules for the browser tools [0a1b2c3d]:\n" in catalogue
    assert "\nRules for the notes tools [0a1b2c3d]:\nKeep notes short." in catalogue


def test_a_held_back_group_with_rules_is_enough_to_name_the_mark() -> None:
    decision = plan_tool_deferral(
        tools=_browser_and_notes(),
        groups=[_ruled_groups()[0], ToolGroup(name="notes")],
        protected=frozenset(),
        discovery_names=frozenset({"ToolSearch"}),
        rc=LoopConstants(),
        mark="0a1b2c3d",
    )
    # No rules are in the catalogue yet, but the browser's will come in a
    # result, and the model must know the mark before they do.
    assert decision.ruled_groups == ()
    assert "[0a1b2c3d]" in decision.catalogue


def test_rules_alone_name_the_mark_first() -> None:
    groups = [ToolGroup(name="notes", instructions="Keep notes short.")]
    tools = [MockTool(tool_name="Read"), GroupedTool(tool_name="NoteAdd", tool_group="notes")]
    decision = plan_tool_deferral(
        tools=tools,
        groups=groups,
        protected=frozenset(),
        discovery_names=frozenset(),
        rc=LoopConstants(),
        mark="0a1b2c3d",
    )
    assert decision.catalogue.startswith(
        "<system-reminder>\nRules for a group of tools come from the runtime alone"
    )
    assert decision.catalogue.endswith(
        "\n\nRules for the notes tools [0a1b2c3d]:\nKeep notes short.\n</system-reminder>"
    )


def test_a_group_without_rules_changes_nothing() -> None:
    groups = [ToolGroup(name="notes")]
    tools = [MockTool(tool_name="Read"), GroupedTool(tool_name="NoteAdd", tool_group="notes")]
    decision = plan_tool_deferral(
        tools=tools,
        groups=groups,
        protected=frozenset(),
        discovery_names=frozenset(),
        rc=LoopConstants(),
    )
    assert decision.catalogue == ""
    assert decision.deferred_groups == ()


@pytest.mark.parametrize("registry_type", [ToolRegistry, InMemoryToolRegistry])
def test_a_declaration_carries_its_load_and_rules_and_refuses_an_unknown_mode(
    registry_type: type[ToolRegistry] | type[InMemoryToolRegistry],
) -> None:
    registry = registry_type()
    registry.declare_group("browser", "Drive a browser", load="lazy", instructions="  Be careful. ")
    (group,) = registry.tool_groups()
    assert (group.load, group.instructions, group.prefix) == ("lazy", "Be careful.", "")
    registry.declare_group("browser", "Drive a browser", prefix=None)
    assert registry.tool_groups()[0].load == "auto"
    with pytest.raises(ValueError, match="load must be one of"):
        registry.declare_group("browser", "Drive a browser", load="sometimes")


def test_a_run_refuses_an_unknown_load_override() -> None:
    with pytest.raises(ValueError, match="tool_group_loads"):
        QueryEngineConfig(
            run_id="r", tenant_id="t", session_id="s", model_name="m",
            tool_group_loads={"browser": "later"},
        )


# ── the production registry, through the loop ───────────────────────────────


async def test_the_core_registry_searches_and_loads_through_the_loop() -> None:
    """The scenarios drive the in-memory registry; this one drives the core's own,
    whose search is the ranked one ToolSearch is written against."""
    registry = ToolRegistry(
        [
            MockTool(tool_name="Read", description="Read a file."),
            MockTool(tool_name="Write", description="Write a file."),
            MockTool(tool_name="Mcp_Jira_transition_issue", description="Move a Jira issue to another status."),
            MockTool(tool_name="Mcp_Jira_get_issue", description="Fetch one Jira issue."),
            MockTool(tool_name="Mcp_Jira_search", description="Search Jira with JQL."),
        ]
    )
    registry.register(ToolSearchTool(registry))
    registry.declare_group("jira", "Jira tickets", dynamic=True, prefix="Mcp_Jira_")
    llm = InMemoryLLMProvider()
    llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"query": "move jira issue status"}
    )
    llm.queue_response(text="moved")
    engine = QueryEngine(
        config=QueryEngineConfig(
            run_id="run-test",
            tenant_id="tenant-test",
            session_id="sess-test",
            model_name="model",
            rc=LoopConstants(model_context_window=32_000, tool_search_autoload_count=1),
        ),
        llm_provider=llm,
        tool_registry=registry,
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )
    events = [
        evt
        async for evt in engine.run(
            Message(role=MessageRole.user, content_blocks=[TextBlock(text="move ABC-1 to done")])
        )
    ]

    first, second = ([d.name for d in request.tools] for request in llm.calls)
    assert first == ["Read", "ToolSearch", "Write"]
    assert second == ["Read", "ToolSearch", "Write", "Mcp_Jira_transition_issue"]
    loaded = [evt.payload["loaded_tool_names"] for evt in events if evt.type is EventType.TOOL_DISCOVERED]
    assert loaded == [["Mcp_Jira_transition_issue"]]
