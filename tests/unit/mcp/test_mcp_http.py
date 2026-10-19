"""MCP over streamable HTTP, against an in-process server behind ``httpx.MockTransport``."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from protocore.mcp import McpServer, McpTransportClosed
from protocore.mcp.http import McpHttpError, StreamableHttpTransport

from ._fake_server import FakeMcpServer, http_handler, text_result

URL = "http://127.0.0.1:9000/mcp"
ECHO = {"name": "echo", "inputSchema": {"type": "object"}}


def _transport(handler: Any, **kwargs: Any) -> StreamableHttpTransport:
    return StreamableHttpTransport(URL, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)


@pytest.mark.parametrize("sse", [False, True])
async def test_a_call_goes_out_as_posts_and_comes_back_as_json_or_sse(sse: bool) -> None:
    seen: list[httpx.Request] = []
    fake = FakeMcpServer(tools=[ECHO], handlers={"echo": lambda args: text_result("pong")})
    server = McpServer("remote", lambda: _transport(http_handler(fake, sse=sse, seen=seen), headers={"x-team": "a"}))

    result = await server.call_tool("echo", {})
    await server.close()

    assert result["content"][0]["text"] == "pong"
    posts = [r for r in seen if r.method == "POST"]
    assert [json.loads(r.content).get("method") for r in posts] == [
        "initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert "text/event-stream" in posts[0].headers["accept"]
    assert posts[0].headers["x-team"] == "a"
    assert "mcp-session-id" not in posts[0].headers
    # Every request after the handshake carries the session and the revision.
    assert all(r.headers["mcp-session-id"] == "session-1" for r in posts[1:])
    assert all(r.headers["mcp-protocol-version"] == "2025-11-25" for r in posts[1:])
    [delete] = [r for r in seen if r.method == "DELETE"]
    assert delete.headers["mcp-session-id"] == "session-1"


async def test_the_server_can_speak_first_inside_an_sse_answer() -> None:
    heard: list[str] = []
    fake = FakeMcpServer(tools=[ECHO], handlers={"echo": lambda args: text_result("pong")})
    fake.push({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}})
    from protocore.mcp import McpClientSession

    session = McpClientSession(_transport(http_handler(fake, sse=True)), on_tools_changed=lambda: heard.append("x"))
    await session.start()
    fake.push({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    assert (await session.call_tool("echo", {}))["content"][0]["text"] == "pong"
    await session.close()
    assert heard == ["x"]


async def test_auth_headers_are_fetched_for_every_connection() -> None:
    tokens = iter(["first", "second"])
    seen: list[httpx.Request] = []
    fake = FakeMcpServer(tools=[ECHO], handlers={"echo": lambda args: text_result("ok")})

    async def auth() -> dict[str, str]:
        return {"authorization": f"Bearer {next(tokens)}"}

    server = McpServer("remote", lambda: _transport(http_handler(fake, seen=seen), auth=auth))
    await server.call_tool("echo", {})
    await server.stop()
    await server.call_tool("echo", {})
    await server.close()

    used = {r.headers.get("authorization") for r in seen if r.method == "POST"}
    assert used == {"Bearer first", "Bearer second"}


async def test_a_refusal_closes_the_session_and_the_next_call_reconnects() -> None:
    fake = FakeMcpServer(tools=[ECHO], handlers={"echo": lambda args: text_result("ok")})
    serve = http_handler(fake)
    refuse = {"next": False}

    async def handler(request: httpx.Request) -> httpx.Response:
        if refuse["next"] and request.method == "POST":
            refuse["next"] = False
            return httpx.Response(401, text="token expired")
        return await serve(request)

    server = McpServer("remote", lambda: _transport(handler))
    await server.start()
    refuse["next"] = True
    with pytest.raises(Exception, match="HTTP 401: token expired"):
        await server.call_tool("echo", {})
    assert (await server.call_tool("echo", {}))["content"][0]["text"] == "ok"
    assert fake.initializations == 2
    await server.close()


async def test_an_unreachable_server_is_a_closed_transport() -> None:
    async def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    transport = _transport(down)
    await transport.open()
    with pytest.raises(McpTransportClosed, match="could not be reached"):
        await transport.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    await transport.close()
    with pytest.raises(McpTransportClosed):
        await transport.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    with pytest.raises(McpTransportClosed):
        await transport.receive()


async def test_a_body_that_is_not_json_is_skipped_and_a_batch_is_unpacked() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        message = json.loads(request.content)
        if message["id"] == 1:
            return httpx.Response(200, content=b"<html>", headers={"content-type": "application/json"})
        return httpx.Response(200, json=[{"jsonrpc": "2.0", "id": 2, "result": {}}, "junk"])

    transport = _transport(handler)
    await transport.open()
    await transport.send({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    await transport.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert await transport.receive() == {"jsonrpc": "2.0", "id": 2, "result": {}}
    await transport.close()


def test_the_http_error_keeps_its_status() -> None:
    error = McpHttpError(404, "session not found")
    assert error.status == 404
    assert isinstance(error, McpTransportClosed)


async def test_a_transport_it_made_owns_its_client() -> None:
    transport = StreamableHttpTransport(URL)
    await transport.open()
    await transport.close()
    await transport.close()
