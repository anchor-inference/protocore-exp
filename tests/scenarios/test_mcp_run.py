"""A whole run whose tools come from an MCP server, held back and loaded on request."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from protocore.contracts.types import MessageRole, TextBlock
from protocore.mcp import McpServer
from protocore.runtime.tool_surface import forget_tool_surfaces
from protocore.tools import ToolSearchTool

from ..unit.mcp._fake_server import FakeMcpServer, MemoryTransport, text_result
from .conftest import ScenarioFactory, ScriptedTool


@pytest.fixture(autouse=True)
def _forget_surfaces() -> Iterator[None]:
    forget_tool_surfaces()
    yield
    forget_tool_surfaces()


def _fake() -> tuple[FakeMcpServer, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def create(args: dict[str, Any]) -> dict[str, Any]:
        calls.append(args)
        return text_result(f"opened issue 7: {args['title']}")

    fake = FakeMcpServer(
        tools=[
            {"name": "create_issue", "description": "Open an issue.",
             "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}},
                             "required": ["title"], "additionalProperties": False}},
            {"name": "list_issues", "description": "List the open issues.", "inputSchema": {"type": "object"}},
        ],
        handlers={"create_issue": create},
        instructions="Always give an issue a short title.",
    )
    return fake, calls


async def test_an_mcp_servers_tools_are_held_back_loaded_and_called(scenario: ScenarioFactory) -> None:
    fake, calls = _fake()
    run = scenario(tools=[ScriptedTool(tool_name="Note", description="record a note")])
    run.tools.register(ToolSearchTool(run.tools))
    server = McpServer("tracker", lambda: MemoryTransport(fake), registry=run.tools,
                       description="The team's issue tracker")
    await server.start()

    run.llm.queue_tool_call_response(
        tool_call_id="s-1", tool_name="ToolSearch", tool_input={"query": "select:Mcp_Tracker_create_issue"})
    run.llm.queue_tool_call_response(
        tool_call_id="c-1", tool_name="Mcp_Tracker_create_issue", tool_input={"title": "Crash on save"})
    run.llm.queue_response(text="Filed as issue 7.")
    await run.run("file the crash")
    await server.close()

    # A dynamic group is held back from the first request and named in the catalogue.
    assert run.advertised_tool_names(0) == ["Note", "ToolSearch"]
    system = "\n".join(block.text for message in run.requests[0].messages if message.role is MessageRole.system
                       for block in message.content_blocks if isinstance(block, TextBlock))
    assert "MCP server tracker: The team's issue tracker." in system
    assert "Mcp_Tracker_create_issue, Mcp_Tracker_list_issues" in system
    # Loaded, the tool keeps the server's strict schema and the server's rules come with it.
    loaded = next(d for d in run.requests[1].tools if d.name == "Mcp_Tracker_create_issue")
    assert loaded.parameters.additional_properties is False
    assert any("Always give an issue a short title." in text for text in run.request_texts(1))
    assert not any("Always give an issue a short title." in text for text in run.request_texts(0))
    assert calls == [{"title": "Crash on save"}]
    assert any("opened issue 7: Crash on save" in result.content for result in run.tool_results())
    assert "Filed as issue 7." in "".join(run.history_texts())
    assert run.tools.get("Mcp_Tracker_create_issue") is None


async def test_a_failing_mcp_call_reaches_the_model_as_a_tool_error(scenario: ScenarioFactory) -> None:
    fake, _ = _fake()
    run = scenario()
    server = McpServer("tracker", lambda: MemoryTransport(fake), registry=run.tools, load="eager")
    await server.start()

    run.llm.queue_tool_call_response(tool_call_id="c-1", tool_name="Mcp_Tracker_list_issues", tool_input={})
    run.llm.queue_response(text="The tracker refused.")
    await run.run("what is open?")
    await server.close()

    [result] = run.tool_results()
    assert result.is_error
    assert "Unknown tool" in result.content
