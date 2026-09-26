"""``ToolSearch``: what it lists, what it loads, and what it never reveals."""
from __future__ import annotations

from typing import Any

from protocore.contracts.run_state import RunScopedState
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import (
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


def _context(policy: ToolVisibilityPolicy | None = None, rc: Any = None) -> ToolContext:
    metadata: dict[str, Any] = {"tool_call_id": "call-1"}
    if policy is not None:
        metadata[TOOL_VISIBILITY_POLICY_METADATA_KEY] = policy
    state = RunScopedState(rc=rc) if rc is not None else None
    return ToolContext(
        tenant_id="t", run_id="r", session_id="s", metadata=metadata, run_state=state
    )


async def _call(registry: ToolRegistry, query: str, **kwargs: Any) -> tuple[str, list[str], list[str]]:
    search = registry.get("ToolSearch")
    assert search is not None
    result = await search.invoke(_context(**kwargs), {"query": query})
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
    assert tool.definition.parameters.required == ["query"]
    assert tool.always_load is True
    assert tool.is_concurrent_safe is True
