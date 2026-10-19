"""The JSON-RPC session under the MCP tool adapter."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from protocore.mcp import (
    LATEST_PROTOCOL_VERSION,
    McpClientSession,
    McpError,
    McpTransportClosed,
)

from ._fake_server import FakeMcpServer, MemoryTransport, text_result


async def _session(server: FakeMcpServer, **kwargs: Any) -> tuple[McpClientSession, MemoryTransport]:
    transport = MemoryTransport(server)
    session = McpClientSession(transport, **kwargs)
    await session.start()
    return session, transport


async def test_the_handshake_offers_the_latest_version_and_says_initialized() -> None:
    server = FakeMcpServer(instructions="Use the search tool first.")
    session, transport = await _session(server, client_name="host", client_version="9.9")

    initialize = server.received[0]
    assert initialize["method"] == "initialize"
    assert initialize["params"]["protocolVersion"] == LATEST_PROTOCOL_VERSION
    assert initialize["params"]["clientInfo"] == {"name": "host", "version": "9.9"}
    assert server.methods()[:2] == ["initialize", "notifications/initialized"]
    assert session.protocol_version == LATEST_PROTOCOL_VERSION
    assert transport.negotiated_version == LATEST_PROTOCOL_VERSION
    assert session.instructions == "Use the search tool first."
    assert session.server_info["name"] == "fake"
    await session.close()


async def test_an_older_supported_version_is_accepted() -> None:
    session, _ = await _session(FakeMcpServer(protocol_version="2025-03-26"))
    assert session.protocol_version == "2025-03-26"
    await session.close()


async def test_a_version_the_client_does_not_speak_is_refused() -> None:
    transport = MemoryTransport(FakeMcpServer(protocol_version="1999-01-01"))
    with pytest.raises(McpError, match="1999-01-01"):
        await McpClientSession(transport).start()
    assert transport.closed


async def test_tools_are_listed_across_pages() -> None:
    tools = [{"name": f"tool{i}", "inputSchema": {"type": "object"}} for i in range(5)]
    session, _ = await _session(FakeMcpServer(tools=tools, page_size=2))

    assert [tool["name"] for tool in await session.list_tools()] == [f"tool{i}" for i in range(5)]
    await session.close()


async def test_a_call_returns_the_result_and_an_error_response_raises() -> None:
    server = FakeMcpServer(handlers={"echo": lambda args: text_result(args["text"])})
    session, _ = await _session(server)

    assert await session.call_tool("echo", {"text": "hi"}) == text_result("hi")
    with pytest.raises(McpError) as raised:
        await session.call_tool("missing", {})
    assert raised.value.code == -32602
    assert raised.value.data == {"name": "missing"}
    assert "Unknown tool" in str(raised.value)
    await session.close()


async def test_a_request_that_times_out_is_cancelled_on_the_server() -> None:
    gate = asyncio.Event()

    async def slow(_args: dict[str, Any]) -> dict[str, Any]:
        await gate.wait()
        return text_result("late")

    server = FakeMcpServer(handlers={"slow": slow})
    transport = MemoryTransport(server)
    session = McpClientSession(transport)
    await session.start()

    # The memory transport answers inside send(), so the slow handler blocks
    # the send itself; the timeout must still fire.
    with pytest.raises(TimeoutError):
        await session.call_tool("slow", {}, timeout=0.05)
    gate.set()
    await asyncio.sleep(0)
    cancelled = [m for m in server.received if m.get("method") == "notifications/cancelled"]
    assert cancelled and cancelled[0]["params"]["requestId"] == server.received[2]["id"]
    await session.close()


async def test_the_server_can_ping_and_is_refused_other_requests() -> None:
    server = FakeMcpServer(handlers={"echo": lambda args: text_result("ok")})
    session, _ = await _session(server)
    server.push({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
    server.push({"jsonrpc": "2.0", "id": "srv-2", "method": "sampling/createMessage", "params": {}})

    await session.call_tool("echo", {})
    await asyncio.sleep(0.01)

    replies = {m.get("id"): m for m in server.received if "method" not in m}
    assert replies["srv-1"]["result"] == {}
    assert replies["srv-2"]["error"]["code"] == -32601
    await session.close()


async def test_a_changed_tool_list_is_reported() -> None:
    heard: list[str] = []
    server = FakeMcpServer(handlers={"echo": lambda args: text_result("ok")})
    session, _ = await _session(server, on_tools_changed=lambda: heard.append("changed"))
    server.push({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    server.push({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "x"}})

    await session.call_tool("echo", {})
    await asyncio.sleep(0.01)

    assert heard == ["changed"]
    await session.close()


async def test_a_broken_connection_fails_every_pending_request() -> None:
    gate = asyncio.Event()

    async def never(_args: dict[str, Any]) -> dict[str, Any]:
        await gate.wait()
        return text_result("x")

    server = FakeMcpServer(handlers={"never": never})
    session, transport = await _session(server)
    # Answers arrive through the queue, not from send(): send returns at once.
    original_send = transport.send

    async def send_without_waiting(message: dict[str, Any]) -> None:
        asyncio.get_running_loop().create_task(original_send(message))

    transport.send = send_without_waiting  # type: ignore[method-assign]
    call = asyncio.create_task(session.call_tool("never", {}))
    await asyncio.sleep(0.01)
    await transport.drop()

    with pytest.raises(McpTransportClosed):
        await call
    assert session.closed
    with pytest.raises(McpTransportClosed):
        await session.call_tool("never", {})
    gate.set()
    await session.close()


async def test_a_session_must_be_started_once() -> None:
    session, _ = await _session(FakeMcpServer())
    with pytest.raises(RuntimeError, match="started"):
        await session.start()
    await session.close()
    await session.close()
