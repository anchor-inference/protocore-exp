"""An MCP server that lives in the test process, and transports that reach it.

Nothing here opens a socket or starts a process: the server answers JSON-RPC
messages handed to it as dictionaries, and each transport is the shortest
path from the client under test to it.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from protocore.mcp import McpTransportClosed

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


class FakeMcpServer:
    """Answers the parts of the protocol the client uses, and records what it was sent."""

    def __init__(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        handlers: dict[str, ToolHandler] | None = None,
        instructions: str = "",
        protocol_version: str | None = None,
        page_size: int = 0,
    ) -> None:
        self.tools = list(tools or [])
        self.handlers = dict(handlers or {})
        self.instructions = instructions
        self.protocol_version = protocol_version
        self.page_size = page_size
        self.received: list[dict[str, Any]] = []
        self.initializations = 0
        #: Messages the server sends on its own (notifications, requests) with
        #: its next answer.
        self.outbox: list[dict[str, Any]] = []

    def methods(self) -> list[str]:
        return [str(message.get("method")) for message in self.received if "method" in message]

    def push(self, message: dict[str, Any]) -> None:
        self.outbox.append(message)

    async def handle(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """The messages the server sends back for ``message``, in order."""
        self.received.append(message)
        method = message.get("method")
        if method is None or "id" not in message:
            return self._drain()
        request_id = message["id"]
        params = message.get("params") or {}
        try:
            result = await self._dispatch(str(method), params)
        except _RpcError as error:
            return [*self._drain(), {"jsonrpc": "2.0", "id": request_id,
                                     "error": {"code": error.code, "message": error.message, "data": error.data}}]
        return [*self._drain(), {"jsonrpc": "2.0", "id": request_id, "result": result}]

    def _drain(self) -> list[dict[str, Any]]:
        out, self.outbox = self.outbox, []
        return out

    async def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            self.initializations += 1
            return {
                "protocolVersion": self.protocol_version or params.get("protocolVersion"),
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "fake", "title": "Fake server", "version": "1.0"},
                "instructions": self.instructions,
            }
        if method == "tools/list":
            if not self.page_size:
                return {"tools": self.tools}
            start = int(params.get("cursor") or 0)
            page = self.tools[start:start + self.page_size]
            result: dict[str, Any] = {"tools": page}
            if start + self.page_size < len(self.tools):
                result["nextCursor"] = str(start + self.page_size)
            return result
        if method == "tools/call":
            name = params.get("name")
            handler = self.handlers.get(str(name))
            if handler is None:
                raise _RpcError(-32602, f"Unknown tool: {name}", {"name": name})
            answer = handler(params.get("arguments") or {})
            if isinstance(answer, Awaitable):
                answer = await answer
            return answer
        if method == "ping":
            return {}
        raise _RpcError(-32601, f"Method not found: {method}")


class _RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class MemoryTransport:
    """Hands each message straight to a :class:`FakeMcpServer`."""

    def __init__(self, server: FakeMcpServer) -> None:
        self.server = server
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.opened = 0
        self.closed = False
        self.negotiated_version: str | None = None
        self.fail_sends = False

    async def open(self) -> None:
        self.opened += 1

    def negotiated(self, protocol_version: str) -> None:
        self.negotiated_version = protocol_version

    async def send(self, message: dict[str, Any]) -> None:
        if self.closed or self.fail_sends:
            raise McpTransportClosed("the fake transport is closed")
        for answer in await self.server.handle(json.loads(json.dumps(message))):
            await self.queue.put(answer)

    async def receive(self) -> dict[str, Any]:
        message = await self.queue.get()
        if message is None:
            raise McpTransportClosed("the fake server went away")
        return message

    async def close(self) -> None:
        self.closed = True
        await self.queue.put(None)

    async def drop(self) -> None:
        """The connection breaks from the server's side."""
        await self.queue.put(None)


def http_handler(server: FakeMcpServer, *, sse: bool = False, session_id: str = "session-1",
                 seen: list[httpx.Request] | None = None) -> Callable[[httpx.Request], Awaitable[httpx.Response]]:
    """An ``httpx.MockTransport`` handler that serves ``server`` as streamable HTTP."""

    async def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.method == "DELETE":
            return httpx.Response(200)
        message = json.loads(request.content)
        answers = await server.handle(message)
        if "id" not in message or "method" not in message:
            return httpx.Response(202)
        headers = {"mcp-session-id": session_id} if message.get("method") == "initialize" else {}
        if sse:
            body = "".join(f"event: message\ndata: {json.dumps(answer)}\n\n" for answer in answers)
            return httpx.Response(200, content=(": keepalive\n\n" + body).encode(),
                                  headers={"content-type": "text/event-stream", **headers})
        [answer] = [a for a in answers if a.get("id") == message["id"]]
        return httpx.Response(200, json=answer, headers=headers)

    return handle
