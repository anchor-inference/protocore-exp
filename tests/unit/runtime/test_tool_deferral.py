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
) -> tuple[tuple[str, ...], frozenset[str], str, tuple[str, ...]]:
    decision = plan_tool_deferral(
        tools=tools,
        groups=groups,
        protected=protected,
        discovery_names=discovery_tool_names(tools, ToolRoleMap()),
        rc=rc or LoopConstants(),
        restored=restored,
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


def test_a_restored_decision_is_replayed_rather_than_remeasured() -> None:
    tools = [_search(), _padded("Browse", group="browser"), _padded("Mcp_X_a")]
    groups = [
        ToolGroup(name="browser", description="Drive a browser"),
        ToolGroup(name="x", dynamic=True, prefix="Mcp_X_"),
    ]
    # Everything fits and "browser" is not dynamic, so a fresh decision would
    # hold back only "x"; the snapshot said "browser", and "gone" no longer
    # has any tools.
    deferred, names, _, reasons = _plan(tools, groups, restored=["browser", "gone"])
    assert deferred == ("browser",)
    assert names == {"Browse"}
    assert reasons == ("restored",)
    assert _plan(tools, groups, restored=[]) == ((), frozenset(), "", ())


# ── the catalogue ───────────────────────────────────────────────────────────


def test_the_catalogue_names_exact_tools_and_prefixes_for_large_dynamic_groups() -> None:
    declared = {
        "github": ToolGroup(
            name="github", description="GitHub issues and pull requests.", prefix="Mcp_Github_"
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
        "These tools are available but not loaded. Before calling one, load it with "
        'ToolSearch: describe what you need, or pass "select:" and exact names, e.g. '
        '"select:Name1,Name2". A dedicated tool for the job is better than a '
        "workaround with a general one such as a shell command, so load the tool "
        "rather than improvising.\n"
        "\n"
        "- bare: tools Lonely\n"
        "- github: GitHub issues and pull requests. Tools: Mcp_Github_* (3 tools)\n"
        "- schedule: Timed and recurring jobs. Tools: ScheduleCreate, IntentCreate\n"
        "</system-reminder>"
    )
    assert render_tool_catalogue({}, declared, discovery_tool="ToolSearch", max_listed_names=2) == ""


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
