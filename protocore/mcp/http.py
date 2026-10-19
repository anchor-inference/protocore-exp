"""MCP over streamable HTTP.

Each message the client sends is one ``POST`` to the server's endpoint. The
server answers a request with either one JSON message or an SSE stream that
carries the answer, possibly after notifications and requests of its own; a
notification or a reply is acknowledged with ``202``. The session id the
server hands out on the handshake (``Mcp-Session-Id``) goes on every later
request, with the negotiated ``MCP-Protocol-Version``, and the session is
ended with a ``DELETE`` on close.

The optional ``GET`` stream for messages the server sends unprompted is not
opened. A server's announcement that its tool list changed then reaches the
client only inside the answer to some request; a host that needs changes at
once calls :meth:`~protocore.mcp.tools.McpServer.refresh_tools` on its own
schedule.

Needs ``httpx``, installed by the ``mcp`` extra.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "protocore.mcp.http needs httpx; install the extra with 'pip install \"protocore[mcp]\"'"
    ) from exc

from protocore.logging_utils import get_logger
from protocore.mcp.session import McpTransportClosed

_logger = get_logger(__name__)

HeaderSource = Callable[[], Awaitable[Mapping[str, str]]]
"""Headers fetched once per connection, such as a bearer token that may have been refreshed."""

_SESSION_HEADER = "mcp-session-id"
_VERSION_HEADER = "mcp-protocol-version"


class McpHttpError(McpTransportClosed):
    """The server refused a message with an HTTP status.

    A subclass of :class:`McpTransportClosed` because the session cannot go
    on: ``401``/``403`` means the credentials need renewing (the next
    connection asks ``auth`` again), and ``404`` on a request that carried a
    session id means the server forgot the session.
    """

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


class StreamableHttpTransport:
    """Speaks streamable HTTP to the MCP endpoint at ``url``.

    ``headers`` go on every request; ``auth`` is awaited on each
    :meth:`open` and its headers added, so a reconnect picks up a renewed
    token. ``client`` is the HTTP client to use (one is made, and closed with
    the transport, when none is given); redirects are not followed, since a
    request may carry a credential.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        auth: HeaderSource | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self._url = url
        self._headers = dict(headers or {})
        self._auth = auth
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=30.0))
        self._connection_headers: dict[str, str] = {}
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        self._incoming: asyncio.Queue[dict[str, Any] | BaseException] = asyncio.Queue()
        self._closed = False

    async def open(self) -> None:
        self._connection_headers = dict(await self._auth()) if self._auth is not None else {}

    def negotiated(self, protocol_version: str) -> None:
        self._protocol_version = protocol_version

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise McpTransportClosed("the HTTP transport is closed")
        headers = {
            **self._headers,
            **self._connection_headers,
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
        }
        if self._session_id:
            headers[_SESSION_HEADER] = self._session_id
        if self._protocol_version:
            headers[_VERSION_HEADER] = self._protocol_version
        request = self._client.build_request("POST", self._url, json=message, headers=headers)
        try:
            response = await self._client.send(request, stream=True, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise McpTransportClosed(f"the MCP server could not be reached: {type(exc).__name__}: {exc}") from exc
        try:
            await self._read_response(response, message)
        except httpx.HTTPError as exc:
            raise McpTransportClosed(f"the MCP response broke off: {type(exc).__name__}: {exc}") from exc
        finally:
            await response.aclose()

    async def receive(self) -> dict[str, Any]:
        item = await self._incoming.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session_id:
            headers = {**self._headers, **self._connection_headers, _SESSION_HEADER: self._session_id}
            with contextlib.suppress(httpx.HTTPError):
                await self._client.delete(self._url, headers=headers, follow_redirects=False)
        if self._owns_client:
            await self._client.aclose()
        await self._incoming.put(McpTransportClosed("the HTTP transport was closed"))

    async def _read_response(self, response: httpx.Response, sent: dict[str, Any]) -> None:
        if response.status_code >= 400:
            body = (await response.aread()).decode("utf-8", "replace")
            raise McpHttpError(response.status_code, body)
        session_id = response.headers.get(_SESSION_HEADER)
        if session_id and sent.get("method") == "initialize":
            self._session_id = session_id
        if response.status_code == 202 or "id" not in sent or "method" not in sent:
            return
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("text/event-stream"):
            async for data in _sse_data(response.aiter_lines()):
                self._deliver(data)
            return
        self._deliver((await response.aread()).decode("utf-8", "replace"))

    def _deliver(self, payload: str) -> None:
        try:
            parsed = json.loads(payload)
        except ValueError:
            _logger.debug("the MCP server sent a body that is not JSON: %s", payload[:200])
            return
        for message in parsed if isinstance(parsed, list) else [parsed]:
            if isinstance(message, dict):
                self._incoming.put_nowait(message)


async def _sse_data(lines: Any) -> Any:
    """The ``data`` of each SSE event; comments and other fields are skipped."""
    buffer: list[str] = []
    async for line in lines:
        if line == "":
            if buffer:
                yield "\n".join(buffer)
                buffer = []
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].removeprefix(" "))
    if buffer:
        yield "\n".join(buffer)


__all__ = ["HeaderSource", "McpHttpError", "StreamableHttpTransport"]
