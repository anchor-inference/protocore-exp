"""An MCP server's tools in the core's registry: schemas, results, errors, groups and lifecycle."""
from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest

from protocore.constants import MAX_TOOL_RESULT_IMAGES
from protocore.contracts.tool_registry import tool_group_of
from protocore.contracts.tools import Tool, ToolContext, ToolInvocationError
from protocore.mcp import (
    McpServer,
    McpServerState,
    McpTool,
    McpTransportClosed,
    mcp_tool_name,
    mcp_tool_prefix,
    parameters_from_schema,
    tool_result_from_mcp,
)
from protocore.runtime.tool_registry import ToolRegistry

from ._fake_server import FakeMcpServer, MemoryTransport, text_result

SEARCH = {
    "name": "search",
    "title": "Search issues",
    "description": "Find issues by text.",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "filter": {"$ref": "#/$defs/Filter"}},
        "required": ["query"],
        "additionalProperties": False,
        "$defs": {"Filter": {"type": "object", "properties": {"state": {"enum": ["open", "closed"]}}}},
    },
}
CREATE = {"name": "create.issue", "description": "Open an issue.", "inputSchema": {"type": "object"}}


def _context(call_id: str = "call-1") -> ToolContext:
    return ToolContext(tenant_id="t", run_id="r", session_id="s", metadata={"tool_call_id": call_id})


class Transports:
    """A factory that records every transport it built."""

    def __init__(self, server: FakeMcpServer) -> None:
        self.server = server
        self.built: list[MemoryTransport] = []

    def __call__(self) -> MemoryTransport:
        transport = MemoryTransport(self.server)
        self.built.append(transport)
        return transport


def _server(fake: FakeMcpServer, registry: ToolRegistry | None = None, **kwargs: Any) -> tuple[McpServer, Transports]:
    transports = Transports(fake)
    return McpServer("issue tracker", transports, registry=registry, **kwargs), transports


# Names and schemas ----------------------------------------------------------------


def test_names_are_prefixed_sanitised_and_short_enough() -> None:
    assert mcp_tool_prefix("issue tracker") == "Mcp_IssueTracker_"
    assert mcp_tool_prefix("!!!") == "Mcp_Server_"
    assert mcp_tool_name("Mcp_X_", "create.issue/v2") == "Mcp_X_create_issue_v2"
    assert len(mcp_tool_name("Mcp_X_", "a" * 100)) == 64


def test_a_strict_schema_stays_strict_and_references_are_inlined() -> None:
    parameters = parameters_from_schema(SEARCH["inputSchema"])

    assert parameters.additional_properties is False
    assert parameters.required == ["query"]
    assert parameters.properties["filter"] == {
        "type": "object", "properties": {"state": {"enum": ["open", "closed"]}}}


def test_a_recursive_schema_is_inlined_to_a_bounded_depth() -> None:
    schema = {
        "type": "object",
        "properties": {"root": {"$ref": "#/definitions/Node"}},
        "definitions": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/definitions/Node"}}}},
    }
    node = parameters_from_schema(schema).properties["root"]
    depth = 0
    while "properties" in node:
        node = node["properties"]["child"]
        depth += 1
    assert depth == 8
    assert "$ref" not in node


def test_odd_schemas_are_read_permissively() -> None:
    assert parameters_from_schema(None).properties == {}
    loose = parameters_from_schema({"properties": {"a": {}}, "required": ["a", 3],
                                    "additionalProperties": {"type": "string"}})
    assert loose.required == ["a"]
    assert loose.additional_properties is None


# Results --------------------------------------------------------------------------


def test_every_kind_of_content_becomes_text() -> None:
    result = tool_result_from_mcp("c", {
        "content": [
            {"type": "text", "text": "found 2"},
            {"type": "image", "mimeType": "image/png", "data": "..."},
            {"type": "audio", "mimeType": "audio/wav", "data": "..."},
            {"type": "resource_link", "uri": "file:///a.txt", "name": "a.txt"},
            {"type": "resource", "resource": {"uri": "file:///b.txt", "text": "inside b"}},
            {"type": "resource", "resource": {"uri": "file:///c.bin", "blob": "AAAA", "mimeType": "x/bin"}},
            {"type": "something-new", "value": 1},
        ],
        "structuredContent": {"count": 2},
    }, server="s", tool="t")

    assert result.content.splitlines() == [
        "found 2",
        "[image image/png returned by the tool]",
        "[audio audio/wav returned by the tool]",
        "[resource a.txt: file:///a.txt]",
        "inside b",
        "[binary resource file:///c.bin (x/bin)]",
        '{"type": "something-new", "value": 1}',
    ]
    assert result.ui_payload == {"structured_content": {"count": 2}}
    assert result.metadata == {"mcp_server": "s", "mcp_tool": "t"}
    assert result.is_error is False


def test_an_image_is_attached_for_the_model_as_well_as_named() -> None:
    """The line stays, so a model that takes no images still knows one came
    back; the image itself rides the result for a model that can look."""
    png = base64.b64encode(b"\x89PNG-shot").decode("ascii")
    result = tool_result_from_mcp("c", {
        "content": [
            {"type": "image", "mimeType": "image/png", "data": png},
            {"type": "image", "mimeType": "image/png", "data": "not base64!"},
            {"type": "image", "mimeType": "text/plain", "data": png},
            {"type": "image", "data": png},
            {"type": "audio", "mimeType": "audio/wav", "data": png},
        ],
    }, server="s", tool="t")

    assert result.content.splitlines()[0] == "[image image/png returned by the tool]"
    assert [image.data for image in result.images] == [b"\x89PNG-shot"]
    assert result.images[0].mime_type == "image/png"


def test_no_more_images_are_attached_than_a_result_may_carry() -> None:
    png = base64.b64encode(b"\x89PNG").decode("ascii")
    items = [{"type": "image", "mimeType": "image/png", "data": png}] * (MAX_TOOL_RESULT_IMAGES + 2)
    result = tool_result_from_mcp("c", {"content": items}, server="s", tool="t")

    assert len(result.images) == MAX_TOOL_RESULT_IMAGES


def test_structured_content_alone_is_the_result() -> None:
    result = tool_result_from_mcp("c", {"content": [], "structuredContent": {"ok": True}}, server="s", tool="t")
    assert result.content == '{"ok": true}'
    assert result.ui_payload is None


def test_an_empty_or_failed_result_says_so() -> None:
    assert tool_result_from_mcp("c", {}, server="s", tool="t").content == "(empty result)"
    assert tool_result_from_mcp("c", text_result("bad input", is_error=True), server="s", tool="t").is_error


# Publishing -----------------------------------------------------------------------


async def test_the_tools_are_published_as_one_dynamic_group() -> None:
    registry = ToolRegistry()
    fake = FakeMcpServer(tools=[SEARCH, CREATE], instructions="Search before you create.")
    server, _ = _server(fake, registry, search_hint="bug ticket", load="lazy")
    await server.start()

    assert server.state is McpServerState.ready
    names = sorted(tool.name for tool in registry.list_all())
    assert names == ["Mcp_IssueTracker_create_issue", "Mcp_IssueTracker_search"]
    [group] = registry.tool_groups()
    assert group.name == "MCP server issue tracker"
    assert group.dynamic is True
    assert group.prefix == "Mcp_IssueTracker_"
    assert group.load == "lazy"
    assert group.description == "Fake server"
    assert group.instructions == "Search before you create."
    search = registry.get("Mcp_IssueTracker_search")
    assert isinstance(search, McpTool) and isinstance(search, Tool)
    assert tool_group_of(search, registry.tool_groups()) == group.name
    assert search.search_hint == "bug ticket Search issues"
    assert search.output_trust == "untrusted"
    assert search.definition.description == "[MCP issue tracker] Find issues by text."
    assert search.definition.parameters.additional_properties is False
    await server.close()


async def test_a_description_given_by_the_host_wins() -> None:
    registry = ToolRegistry()
    server, _ = _server(FakeMcpServer(tools=[SEARCH]), registry, description="The team's\n issues")
    await server.start()
    assert registry.tool_groups()[0].description == "The team's issues"
    await server.close()


async def test_close_withdraws_the_tools_and_the_group() -> None:
    registry = ToolRegistry()
    server, transports = _server(FakeMcpServer(tools=[SEARCH]), registry)
    await server.start()
    await server.close()

    assert registry.list_all() == []
    assert registry.tool_groups() == []
    assert transports.built[0].closed
    assert server.tools == ()


async def test_a_changed_tool_list_updates_the_registry() -> None:
    registry = ToolRegistry()
    fake = FakeMcpServer(tools=[SEARCH], handlers={"search": lambda args: text_result("ok")})
    server, _ = _server(fake, registry)
    await server.start()

    fake.tools = [CREATE]
    fake.push({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    await server.call_tool("search", {"query": "x"})
    for _ in range(20):
        if registry.get("Mcp_IssueTracker_create_issue") is not None:
            break
        await asyncio.sleep(0.01)

    assert [tool.name for tool in registry.list_all()] == ["Mcp_IssueTracker_create_issue"]
    fake.tools = []
    await server.refresh_tools()
    assert registry.list_all() == [] and registry.tool_groups() == []
    await server.close()


async def test_colliding_names_are_numbered() -> None:
    fake = FakeMcpServer(tools=[{"name": "a.b"}, {"name": "a_b"}, {"name": "a/b"}])
    server, _ = _server(fake)
    await server.start()
    assert [tool.name for tool in server.tools] == [
        "Mcp_IssueTracker_a_b", "Mcp_IssueTracker_a_b_2", "Mcp_IssueTracker_a_b_3"]
    await server.close()


# Calls ----------------------------------------------------------------------------


async def test_a_call_reaches_the_remote_tool_and_comes_back_as_a_result() -> None:
    seen: list[dict[str, Any]] = []

    def search(args: dict[str, Any]) -> dict[str, Any]:
        seen.append(args)
        return text_result(f"2 issues match {args['query']}")

    registry = ToolRegistry()
    server, _ = _server(FakeMcpServer(tools=[SEARCH], handlers={"search": search}), registry)
    await server.start()
    tool = registry.get("Mcp_IssueTracker_search")
    assert tool is not None

    result = await tool.invoke(_context("call-9"), {"query": "crash"})

    assert seen == [{"query": "crash"}]
    assert result.tool_call_id == "call-9"
    assert result.content == "2 issues match crash"
    assert result.is_error is False
    await server.close()


async def test_a_protocol_error_is_a_tool_invocation_error() -> None:
    server, _ = _server(FakeMcpServer(tools=[SEARCH]))
    await server.start()
    with pytest.raises(ToolInvocationError, match=r"search failed: MCP error -32602: Unknown tool"):
        await server.tools[0].invoke(_context(), {"query": "x"})
    await server.close()


async def test_a_call_that_times_out_says_its_outcome_is_unknown() -> None:
    async def slow(_args: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(5)
        return text_result("late")

    server, _ = _server(FakeMcpServer(tools=[SEARCH], handlers={"search": slow}), call_timeout_seconds=0.05)
    await server.start()
    with pytest.raises(ToolInvocationError, match="unknown"):
        await server.tools[0].invoke(_context(), {"query": "x"})
    await server.close()


async def test_the_first_call_connects_and_a_stopped_server_reconnects() -> None:
    fake = FakeMcpServer(tools=[SEARCH], handlers={"search": lambda args: text_result("ok")})
    server, transports = _server(fake)

    assert (await server.call_tool("search", {}))["content"][0]["text"] == "ok"
    assert len(transports.built) == 1
    await server.stop()
    assert server.state is McpServerState.stopped

    await server.call_tool("search", {})
    assert len(transports.built) == 2
    assert fake.initializations == 2
    await server.close()


async def test_a_connection_that_broke_between_calls_is_reopened_before_the_next() -> None:
    fake = FakeMcpServer(tools=[SEARCH], handlers={"search": lambda args: text_result("ok")})
    server, transports = _server(fake)
    await server.start()
    await transports.built[0].drop()
    await asyncio.sleep(0.01)
    assert not server.connected

    await server.call_tool("search", {})
    assert len(transports.built) == 2
    await server.close()


async def test_a_connection_that_breaks_during_a_call_is_not_retried() -> None:
    fake = FakeMcpServer(tools=[SEARCH], handlers={"search": lambda args: text_result("ok")})
    server, transports = _server(fake)
    await server.start()
    transports.built[0].fail_sends = True

    with pytest.raises(ToolInvocationError, match="whether the call took effect is unknown"):
        await server.call_tool("search", {})
    assert server.state is McpServerState.failed
    assert fake.methods().count("tools/call") == 0

    # The broken connection is not reused: the next call opens a new one.
    assert (await server.call_tool("search", {}))["content"][0]["text"] == "ok"
    assert len(transports.built) == 2
    await server.close()


async def test_a_server_that_cannot_connect_fails_the_call_without_sending_it() -> None:
    class Unreachable(MemoryTransport):
        async def open(self) -> None:
            raise McpTransportClosed("connection refused")

    server = McpServer("down", lambda: Unreachable(FakeMcpServer()))
    with pytest.raises(McpTransportClosed):
        await server.start()
    assert server.state is McpServerState.failed
    assert server.error == "connection refused"

    with pytest.raises(ToolInvocationError, match="not reachable, so search was not called"):
        await server.call_tool("search", {})


async def test_a_handshake_that_never_finishes_times_out() -> None:
    class Silent(MemoryTransport):
        async def send(self, message: dict[str, Any]) -> None:
            return None

    server = McpServer("silent", lambda: Silent(FakeMcpServer()), connect_timeout_seconds=0.05)
    with pytest.raises(McpTransportClosed, match="no handshake"):
        await server.start()
    assert server.state is McpServerState.failed


async def test_concurrent_starts_open_one_connection() -> None:
    fake = FakeMcpServer(tools=[SEARCH])
    server, transports = _server(fake)
    await asyncio.gather(*(server.start() for _ in range(4)))
    assert len(transports.built) == 1
    await server.close()


def test_a_server_needs_a_name_and_positive_timeouts() -> None:
    with pytest.raises(ValueError, match="name"):
        McpServer(" ", lambda: MemoryTransport(FakeMcpServer()))
    with pytest.raises(ValueError, match="timeouts"):
        McpServer("x", lambda: MemoryTransport(FakeMcpServer()), call_timeout_seconds=0)
