"""``ToolSearch``: what it lists, what it loads, and what it never reveals."""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from protocore.contracts.run_state import RunScopedState
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import (
    ADVERTISED_TOOLS_METADATA_KEY,
    TOOL_ALLOWLIST_METADATA_KEY,
    TOOL_VISIBILITY_POLICY_METADATA_KEY,
    TOOLS_LOADED_METADATA_KEY,
    ToolVisibilityPolicy,
)
from protocore.contracts.tools import ToolContext
from protocore.contracts.types import ToolDefinition, ToolParameterSchema
from protocore.runtime.tool_registry import ToolRegistry
from protocore.tools import ToolSearchTool
from tests.unit.runtime._tool_fixtures import MockTool


def _tool(name: str, description: str) -> MockTool:
    return MockTool(
        tool_name=name,
        description=description,
        parameters_schema={"target": {"type": "string"}, "when": {"type": "string"}},
    )


class _RequiredTool(MockTool):
    def __init__(self, *, tool_name: str, description: str, required: tuple[str, ...]) -> None:
        super().__init__(tool_name=tool_name, description=description)
        self._required = required

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.tool_name,
            description=self.description,
            parameters=ToolParameterSchema(
                properties={"run_at": {"type": "string"}, "command": {"type": "string"}},
                required=list(self._required),
            ),
        )


def _catalogue() -> ToolRegistry:
    registry = ToolRegistry(
        [
            _RequiredTool(
                tool_name="ScheduleCreate",
                description="Schedule a command to run at a time. Runs once or on a cron.",
                required=("run_at",),
            ),
            _tool("ServiceStart", "Start a long-running background service. It keeps running."),
            _tool("BrowserOpen", "Open a web page in the browser."),
            _tool("SecretVault", "Read a stored secret by name."),
            _tool("Mcp_Github_list_issues", "List the issues of a GitHub repository."),
            _tool("Mcp_Github_create_issue", "Create an issue in a GitHub repository."),
        ]
    )
    registry.register(ToolSearchTool(registry))
    return registry


_NO_POLICY: Any = object()


def _context(
    policy: Any = None,
    rc: Any = None,
    advertised: frozenset[str] | None = None,
    allowlist: frozenset[str] | None = None,
) -> ToolContext:
    metadata: dict[str, Any] = {"tool_call_id": "call-1"}
    if policy is not _NO_POLICY:
        metadata[TOOL_VISIBILITY_POLICY_METADATA_KEY] = (
            ToolVisibilityPolicy() if policy is None else policy
        )
    if allowlist is not None:
        metadata[TOOL_ALLOWLIST_METADATA_KEY] = allowlist
    if advertised is not None:
        metadata[ADVERTISED_TOOLS_METADATA_KEY] = advertised
    state = RunScopedState(rc=rc) if rc is not None else None
    return ToolContext(
        tenant_id="t", run_id="r", session_id="s", metadata=metadata, run_state=state
    )


async def _call(
    registry: ToolRegistry,
    query: str | None,
    *,
    arguments: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[str, list[str], list[str]]:
    search = registry.get("ToolSearch")
    assert search is not None
    payload = dict(arguments or {})
    if query is not None:
        payload["query"] = query
    result = await search.invoke(_context(**kwargs), payload)
    assert result.tool_call_id == "call-1"
    return (
        result.content,
        list(result.metadata[TOOLS_LOADED_METADATA_KEY]),
        list(result.metadata["matches"]),
    )


async def test_select_loads_exactly_the_named_tools_in_any_case() -> None:
    content, loaded, _ = await _call(
        _catalogue(), "select:ServiceStart, mcp_github_create_issue,ServiceStart"
    )
    assert loaded == ["ServiceStart", "Mcp_Github_create_issue"]
    assert content.splitlines()[0] == (
        "Loaded, and callable from your next step: ServiceStart, Mcp_Github_create_issue."
    )
    assert "ServiceStart(target, when) — Start a long-running background service." in content


async def test_an_unknown_select_name_is_answered_with_the_nearest_admitted_names() -> None:
    policy = ToolVisibilityPolicy(blocked={"SecretVault"})
    content, loaded, _ = await _call(
        _catalogue(), "select:SecretVaults,ServiceStrt,Nonsense", policy=policy
    )
    assert loaded == []
    assert "Nothing was loaded." in content
    assert "No tool named 'ServiceStrt'. Closest: ServiceStart" in content
    assert "No tool named 'Nonsense'; describe what you need instead." in content
    # The blocked tool is the closest name there is, and it is still never said.
    assert "SecretVault" not in content.replace("'SecretVaults'", "")


async def test_a_query_lists_matches_best_first_and_loads_the_first_few() -> None:
    rc = LoopConstants(tool_search_max_results=3, tool_search_autoload_count=2)
    content, loaded, matches = await _call(_catalogue(), "create a github issue", rc=rc)
    assert matches[0] == "Mcp_Github_create_issue"
    assert len(matches) <= 3
    assert loaded == matches[:2]
    lines = content.splitlines()
    assert lines[0].startswith("Loaded, and callable from your next step: Mcp_Github_create_issue")
    assert "Matches, best first:" in lines
    assert "ToolSearch" not in matches


async def test_required_parameters_are_starred_and_the_summary_is_one_sentence() -> None:
    content, _, _ = await _call(_catalogue(), "select:ScheduleCreate")
    assert "ScheduleCreate(run_at*, command) — Schedule a command to run at a time." in content
    assert "cron" not in content


async def test_the_live_policy_decides_what_can_be_found() -> None:
    policy = ToolVisibilityPolicy(visible={"ServiceStart", "ToolSearch"})
    content, loaded, matches = await _call(_catalogue(), "github issue", policy=policy)
    assert not any(name.startswith("Mcp_") for name in matches + loaded)
    assert "Mcp_" not in content
    _, selected, _ = await _call(_catalogue(), "select:Mcp_Github_list_issues", policy=policy)
    assert selected == []


async def test_a_query_that_matches_nothing_says_so() -> None:
    content, loaded, matches = await _call(_catalogue(), "zzqx")
    assert (loaded, matches) == ([], [])
    assert content.startswith("No tool matches 'zzqx'.")


async def test_without_run_constants_the_defaults_apply() -> None:
    _, loaded, matches = await _call(_catalogue(), "github")
    assert len(loaded) <= 3
    assert len(matches) <= 8


def test_the_tool_describes_itself_as_a_discovery_tool() -> None:
    tool = ToolSearchTool(ToolRegistry())
    assert tool.name == "ToolSearch"
    # Either argument will do, so the schema requires neither.
    assert tool.definition.parameters.required == []
    assert set(tool.definition.parameters.properties) == {"query", "select"}
    # A model that took the tool for a skill loader stopped short of the skill.
    assert "skills are not tools" in tool.definition.description
    assert tool.always_load is True
    assert tool.is_concurrent_safe is True


async def test_a_tool_already_in_the_list_is_not_reported_as_loaded() -> None:
    """Told "Loaded: WebSearch" of a tool it had all along, a model believed a
    skill of that name had been loaded and never opened the skill."""
    advertised = frozenset({"ServiceStart", "ToolSearch"})
    content, loaded, _ = await _call(
        _catalogue(), "select:ServiceStart,BrowserOpen", advertised=advertised
    )
    lines = content.splitlines()
    assert lines[0] == "Loaded, and callable from your next step: BrowserOpen."
    assert lines[1] == "Already in your tool list, nothing to load: ServiceStart."
    # Still named to the loop, which keeps its recency and loads nothing twice.
    assert loaded == ["ServiceStart", "BrowserOpen"]


async def test_a_search_whose_matches_are_all_listed_loads_nothing_new() -> None:
    rc = LoopConstants(tool_search_max_results=2, tool_search_autoload_count=1)
    content, loaded, matches = await _call(
        _catalogue(),
        "start a background service",
        rc=rc,
        advertised=frozenset({"ServiceStart"}),
    )
    assert matches[0] == "ServiceStart"
    assert loaded == ["ServiceStart"]
    assert content.splitlines()[0] == "Already in your tool list, nothing to load: ServiceStart."
    assert "Loaded, and callable" not in content


async def test_outside_a_loop_everything_found_reads_as_loaded() -> None:
    content, _, _ = await _call(_catalogue(), "select:ServiceStart")
    assert content.splitlines()[0] == "Loaded, and callable from your next step: ServiceStart."


@pytest.mark.parametrize(
    "arguments",
    [
        {"select": ["ServiceStart", "browseropen"]},
        {"select": "ServiceStart, browseropen"},
        {"select": ["ServiceStart"], "query": "select:BrowserOpen"},
        # Names win over a description sent beside them.
        {"select": "ServiceStart,BrowserOpen", "query": "create a github issue"},
    ],
)
async def test_select_is_also_its_own_argument(arguments: dict[str, Any]) -> None:
    _, loaded, _ = await _call(_catalogue(), None, arguments=arguments)
    assert loaded == ["ServiceStart", "BrowserOpen"]


@pytest.mark.parametrize("arguments", [{}, {"query": "  "}, {"select": []}, {"select": " , "}])
async def test_a_call_that_asks_for_nothing_is_refused(arguments: dict[str, Any]) -> None:
    search = _catalogue().get("ToolSearch")
    assert search is not None
    with pytest.raises(ValidationError, match="query"):
        await search.invoke(_context(), arguments)


@pytest.mark.parametrize("policy", [_NO_POLICY, {"blocked": ["SecretVault"]}, "everything"])
async def test_without_a_policy_nothing_is_listed_or_loaded(policy: Any) -> None:
    """A missing or unreadable policy used to read as "no restriction", so the
    search listed and loaded tools the gate was about to refuse."""
    content, loaded, matches = await _call(_catalogue(), "select:SecretVault", policy=policy)
    assert loaded == []
    assert matches == []
    assert "SecretVault" not in content
    _, loaded, matches = await _call(_catalogue(), "read a stored secret", policy=policy)
    assert loaded == [] and matches == []


async def test_a_child_is_never_told_it_loaded_a_tool_outside_its_declared_set() -> None:
    allowlist = frozenset({"ServiceStart", "ToolSearch"})
    content, loaded, _ = await _call(
        _catalogue(), "select:ServiceStart,BrowserOpen", allowlist=allowlist
    )
    assert loaded == ["ServiceStart"]
    assert "No tool named 'BrowserOpen'" in content
    _, loaded, matches = await _call(_catalogue(), "open a web page in the browser", allowlist=allowlist)
    assert "BrowserOpen" not in matches
    assert "BrowserOpen" not in loaded
