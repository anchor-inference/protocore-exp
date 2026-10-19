"""The client half of an MCP connection: JSON-RPC over a transport.

The Model Context Protocol is JSON-RPC 2.0 with a handshake. A tool adapter
needs a small part of it: ``initialize``, ``tools/list``, ``tools/call``,
answering the server's ``ping``, and hearing that the tool list changed. This
module is that part and nothing more, over any :class:`McpTransport`; the
transports in :mod:`protocore.mcp.stdio` and :mod:`protocore.mcp.http` move
the messages.

Written here rather than taken from the official SDK because the SDK brings a
web server, a JWT library and telemetry into every host that only wants to
call a few tools, and its API changed incompatibly between major versions. A
host that already uses the SDK can wrap its session in the few methods
:class:`~protocore.mcp.tools.McpServer` calls instead.
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from protocore.logging_utils import get_logger

_logger = get_logger(__name__)

#: The revision of the protocol this client offers first.
LATEST_PROTOCOL_VERSION = "2025-11-25"

#: Every revision whose tool calls this client can make. A server that answers
#: the handshake with another one is refused: guessing at a protocol the client
#: does not know produces failures that look like the server's fault.
SUPPORTED_PROTOCOL_VERSIONS: frozenset[str] = frozenset(
    {"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"}
)

_METHOD_NOT_FOUND = -32601


class McpError(Exception):
    """A JSON-RPC error the server answered with, or a handshake the client could not accept."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"MCP error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class McpTransportClosed(Exception):
    """The connection is gone: closed, broken, or never opened.

    Raised to a request that was waiting when it broke, and to every request
    after. Whether a call that was in flight took effect is unknown.
    """


@runtime_checkable
class McpTransport(Protocol):
    """Moves JSON-RPC messages, as dictionaries, between the client and one server.

    :meth:`send` may return before or after the answer has been received;
    answers are read with :meth:`receive`, which raises
    :class:`McpTransportClosed` once the connection is gone.
    :meth:`negotiated` tells the transport the protocol revision the handshake
    settled on (HTTP sends it as a header on every later request).
    """

    async def open(self) -> None: ...

    async def send(self, message: dict[str, Any]) -> None: ...

    async def receive(self) -> dict[str, Any]: ...

    def negotiated(self, protocol_version: str) -> None: ...

    async def close(self) -> None: ...


class McpClientSession:
    """One initialised MCP session over ``transport``.

    ``request_timeout`` bounds every request that names no timeout of its
    own. ``on_tools_changed`` is called, synchronously, when the server says
    its tool list changed; it should only schedule work.
    """

    def __init__(
        self,
        transport: McpTransport,
        *,
        client_name: str = "protocore",
        client_version: str = "",
        request_timeout: float = 60.0,
        on_tools_changed: Callable[[], None] | None = None,
    ) -> None:
        self._transport = transport
        self._client_info = {"name": client_name, "version": client_version or _core_version()}
        self._request_timeout = request_timeout
        self._on_tools_changed = on_tools_changed
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False
        self._failure: BaseException | None = None
        self.protocol_version = ""
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.instructions = ""

    @property
    def closed(self) -> bool:
        """Whether the session can no longer make requests."""
        return self._closed

    async def start(self) -> None:
        """Open the transport and complete the handshake."""
        if self._started:
            raise RuntimeError("an MCP session is started once")
        self._started = True
        await self._transport.open()
        self._reader = asyncio.get_running_loop().create_task(self._read(), name="mcp-session-reader")
        try:
            result = await self.request(
                "initialize",
                {
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": self._client_info,
                },
            )
            version = str(result.get("protocolVersion") or "")
            if version not in SUPPORTED_PROTOCOL_VERSIONS:
                raise McpError(
                    -32602,
                    f"the server speaks protocol revision {version!r}, which this client does not",
                    {"supported": sorted(SUPPORTED_PROTOCOL_VERSIONS)},
                )
            self.protocol_version = version
            self._transport.negotiated(version)
            info = result.get("serverInfo")
            self.server_info = info if isinstance(info, dict) else {}
            capabilities = result.get("capabilities")
            self.capabilities = capabilities if isinstance(capabilities, dict) else {}
            instructions = result.get("instructions")
            self.instructions = instructions if isinstance(instructions, str) else ""
            await self.notify("notifications/initialized")
        except BaseException:
            await self.close()
            raise

    async def list_tools(self) -> list[dict[str, Any]]:
        """Every tool the server lists, following its pages."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        # A server that hands back the cursor it was given would page forever.
        seen: set[str] = set()
        while True:
            result = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            page = result.get("tools")
            tools.extend(tool for tool in page or [] if isinstance(tool, dict) and tool.get("name"))
            next_cursor = result.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen:
                return tools
            seen.add(next_cursor)
            cursor = next_cursor

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        """The server's result for one tool call (a tool's own failure is a result with ``isError``)."""
        return await self.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)

    async def request(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Send one request and wait for its result; an error answer raises :class:`McpError`.

        On a timeout the server is told the request is cancelled, so a slow
        tool does not go on working for a caller that has stopped waiting.
        """
        if self._closed:
            raise McpTransportClosed("the MCP session is closed") from self._failure
        request_id = next(self._ids)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            async with asyncio.timeout(timeout if timeout is not None else self._request_timeout):
                await self._transport.send(message)
                answer = await future
        except TimeoutError:
            await self._cancel_on_server(request_id, method)
            raise
        except McpTransportClosed as exc:
            # A message that could not be sent means the connection is gone;
            # marking the session closed lets the next call open a new one
            # instead of failing on this one forever. This request's own
            # future leaves the table first: it is answered by this raise, and
            # failing it as well would leave an exception nobody retrieves.
            self._pending.pop(request_id, None)
            self._mark_closed(exc)
            raise
        finally:
            self._pending.pop(request_id, None)
        error = answer.get("error")
        if error is not None:
            if not isinstance(error, dict):
                raise McpError(-32603, str(error))
            raise McpError(int(error.get("code") or -32603), str(error.get("message") or ""), error.get("data"))
        result = answer.get("result")
        return result if isinstance(result, dict) else {}

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        await self._transport.send(message)

    async def close(self) -> None:
        """Close the transport and fail whatever is still waiting. Safe to call twice."""
        if self._closed and self._reader is None:
            return
        self._mark_closed(McpTransportClosed("the MCP session was closed"))
        reader, self._reader = self._reader, None
        try:
            await self._transport.close()
        except Exception as exc:
            _logger.debug("closing an MCP transport failed: %s", exc)
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader

    async def _cancel_on_server(self, request_id: int, method: str) -> None:
        if self._closed:
            return
        try:
            async with asyncio.timeout(5.0):
                await self.notify(
                    "notifications/cancelled",
                    {"requestId": request_id, "reason": f"the client stopped waiting for {method}"},
                )
        except Exception as exc:
            _logger.debug("could not tell the MCP server to cancel request %s: %s", request_id, exc)

    async def _read(self) -> None:
        try:
            while True:
                message = await self._transport.receive()
                await self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except McpTransportClosed as exc:
            self._mark_closed(exc)
        except Exception as exc:
            _logger.warning("an MCP connection failed: %s", exc)
            self._mark_closed(McpTransportClosed(f"the MCP connection failed: {exc}"))

    async def _dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if method is None:
            request_id = message.get("id")
            future = self._pending.get(request_id) if isinstance(request_id, int) else None
            if future is not None and not future.done():
                future.set_result(message)
            return
        if "id" in message:
            await self._answer_server_request(message)
            return
        if method == "notifications/tools/list_changed" and self._on_tools_changed is not None:
            try:
                self._on_tools_changed()
            except Exception:
                _logger.exception("an MCP tool-list listener failed")

    async def _answer_server_request(self, message: dict[str, Any]) -> None:
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        if message.get("method") == "ping":
            reply["result"] = {}
        else:
            # The client declared no capabilities, so a server that asks for
            # sampling, roots or elicitation is asking for something it was
            # told is not there.
            reply["error"] = {"code": _METHOD_NOT_FOUND, "message": f"not supported: {message.get('method')}"}
        try:
            await self._transport.send(reply)
        except McpTransportClosed:
            pass

    def _mark_closed(self, failure: BaseException) -> None:
        self._closed = True
        if self._failure is None:
            self._failure = failure
        for future in self._pending.values():
            if not future.done():
                future.set_exception(McpTransportClosed(str(failure)))


def _core_version() -> str:
    from protocore import __version__

    return __version__


__all__ = [
    "LATEST_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "McpClientSession",
    "McpError",
    "McpTransport",
    "McpTransportClosed",
]
