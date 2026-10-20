"""An MCP server's tools as core tools.

:class:`McpServer` owns one connection and publishes the server's tools in an
:class:`~protocore.contracts.tool_registry.IToolRegistry` as :class:`McpTool`
proxies, all under one name prefix and declared as one **dynamic** tool group.
That is what lets the loop hold a server of three hundred tools back behind
``ToolSearch`` and name it in one catalogue line, instead of putting every
definition on every request (see "Holding tool groups back" in
``docs/tools.md``).

What crosses the boundary, and how:

* **Schemas.** ``properties``, ``required`` and a boolean
  ``additionalProperties`` reach the definition. Local ``$ref`` pointers into
  ``$defs`` / ``definitions`` are inlined, because the core's schema keeps no
  definitions section and a reference to nothing is a schema no model can
  satisfy.
* **Results.** Text content is the result; images, audio and binary resources
  are named in a line each, and an image is also attached to the result
  (:attr:`~protocore.contracts.types.ToolResult.images`) for a model that can
  look at it; structured
  content becomes the result when the server sent no text, and rides as
  ``ui_payload`` when it did. A result the server marks ``isError`` is an
  error result.
* **Errors.** A protocol error, a timeout or a lost connection raises
  :class:`~protocore.contracts.tools.ToolInvocationError`, the core's tool
  error contract; a call lost with its connection says its outcome is unknown,
  because the server may have acted on it.
* **Lifecycle.** The connection opens on :meth:`McpServer.start` or on the
  first call, and a call that finds it closed reconnects once before sending:
  nothing was sent, so nothing can be repeated. A server that says its tool
  list changed is listed again and the registry updated.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from protocore.constants import MAX_TOOL_RESULT_IMAGES
from protocore.contracts.tool_registry import TOOL_GROUP_INSTRUCTIONS_MAX_CHARS, IToolRegistry
from protocore.contracts.tools import Tool, ToolContext, ToolInvocationError, read_metadata
from protocore.contracts.types import ToolDefinition, ToolParameterSchema, ToolResult, ToolResultImage
from protocore.logging_utils import get_logger
from protocore.mcp.session import McpClientSession, McpError, McpTransport, McpTransportClosed

if TYPE_CHECKING:
    from protocore.mcp.http import HeaderSource

_logger = get_logger(__name__)

#: The longest tool name providers accept.
_NAME_LIMIT = 64
#: How deep inlined ``$ref`` pointers may nest before the rest is left open.
#: A recursive schema (a tree node holding nodes) has no finite inline form.
_REF_DEPTH = 8
#: One line of the catalogue per group, so a server's description is kept short.
_GROUP_DESCRIPTION_CHARS = 200
_UNREADABLE_ITEM_CHARS = 2_000


def mcp_tool_prefix(server: str) -> str:
    """The name prefix every tool of ``server`` carries, and so the prefix its group is declared by."""
    return f"Mcp_{re.sub(r'[^A-Za-z0-9]+', '', server.title()) or 'Server'}_"


def mcp_tool_name(prefix: str, tool: str) -> str:
    """A provider-safe tool name: the prefix, the remote name with other characters as ``_``, at most 64."""
    return f"{prefix}{re.sub(r'[^A-Za-z0-9_-]+', '_', tool)}"[:_NAME_LIMIT]


def parameters_from_schema(schema: Mapping[str, Any] | None) -> ToolParameterSchema:
    """The core's parameter schema for an MCP tool's ``inputSchema``.

    A boolean ``additionalProperties`` is kept, so a server's strict schema
    stays strict on the wire. A schema-valued one (the extra keys must match a
    schema) has no place in the core's field and is left out, which is the
    permissive reading: the server still validates what it receives.
    """
    schema = schema if isinstance(schema, Mapping) else {}
    definitions: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        section = schema.get(key)
        if isinstance(section, Mapping):
            definitions.update({f"#/{key}/{name}": value for name, value in section.items()})
    properties = schema.get("properties")
    inlined = _inline_refs(dict(properties) if isinstance(properties, Mapping) else {}, definitions, 0)
    required = [name for name in schema.get("required") or [] if isinstance(name, str)]
    additional = schema.get("additionalProperties")
    return ToolParameterSchema(
        properties=inlined if isinstance(inlined, dict) else {},
        required=required,
        additional_properties=additional if isinstance(additional, bool) else None,
    )


def _inline_refs(node: Any, definitions: Mapping[str, Any], depth: int) -> Any:
    if isinstance(node, list):
        return [_inline_refs(item, definitions, depth) for item in node]
    if not isinstance(node, Mapping):
        return node
    reference = node.get("$ref")
    if isinstance(reference, str) and reference in definitions:
        if depth >= _REF_DEPTH:
            # Left open rather than cut: an empty schema accepts the value the
            # server will check anyway.
            return {key: value for key, value in node.items() if key != "$ref"}
        target = definitions[reference]
        merged = {**(target if isinstance(target, Mapping) else {}),
                  **{key: value for key, value in node.items() if key != "$ref"}}
        return _inline_refs(merged, definitions, depth + 1)
    return {key: _inline_refs(value, definitions, depth) for key, value in node.items()}


def tool_result_from_mcp(call_id: str, result: Mapping[str, Any], *, server: str, tool: str) -> ToolResult:
    """The core's :class:`ToolResult` for an MCP ``tools/call`` result."""
    parts: list[str] = []
    images: list[ToolResultImage] = []
    for item in result.get("content") or []:
        if isinstance(item, Mapping):
            parts.append(_render_item(item))
            image = _image_of(item)
            if image is not None and len(images) < MAX_TOOL_RESULT_IMAGES:
                images.append(image)
    structured = result.get("structuredContent")
    ui_payload: dict[str, Any] | None = None
    if structured is not None:
        if parts:
            ui_payload = {"structured_content": structured}
        else:
            parts.append(json.dumps(structured, ensure_ascii=False, default=str))
    return ToolResult(
        tool_call_id=call_id,
        content="\n".join(parts) or "(empty result)",
        is_error=bool(result.get("isError")),
        metadata={"mcp_server": server, "mcp_tool": tool},
        ui_payload=ui_payload,
        images=tuple(images),
    )


def _image_of(item: Mapping[str, Any]) -> ToolResultImage | None:
    """The image an ``image`` content item carries, for the model to look at.

    The line :func:`_render_item` writes for it stays in the text, so a model
    that takes no images still knows one was returned. An item whose data is
    not base64 of an image the core accepts is left as that line alone.
    """
    if item.get("type") != "image":
        return None
    data, mime = item.get("data"), item.get("mimeType")
    if not isinstance(data, str) or not isinstance(mime, str):
        return None
    try:
        # Validated from the mapping so the contract's own base64 reader
        # decodes the text, rather than a second decoder here.
        return ToolResultImage.model_validate({"data": data, "mime_type": mime})
    except ValueError:
        return None


def _render_item(item: Mapping[str, Any]) -> str:
    kind = item.get("type")
    if kind == "text":
        return str(item.get("text") or "")
    if kind in ("image", "audio"):
        return f"[{kind} {item.get('mimeType') or ''} returned by the tool]".replace("  ", " ")
    if kind == "resource_link":
        return f"[resource {item.get('name') or item.get('uri')}: {item.get('uri')}]"
    if kind == "resource":
        resource = item.get("resource")
        if isinstance(resource, Mapping):
            if isinstance(resource.get("text"), str):
                return str(resource["text"])
            return f"[binary resource {resource.get('uri')} ({resource.get('mimeType') or 'unknown type'})]"
    return json.dumps(dict(item), ensure_ascii=False, default=str)[:_UNREADABLE_ITEM_CHARS]


class McpServerState(StrEnum):
    """Where an :class:`McpServer`'s connection stands."""

    stopped = "stopped"
    connecting = "connecting"
    ready = "ready"
    failed = "failed"


class McpTool(Tool):
    """A core tool that forwards to one remote MCP tool.

    Joins its server's group explicitly (``tool_group``), carries the server's
    search hint, and declares its output untrusted: it is text an outside
    server wrote, and a runtime that fences such output fences this.
    """

    # Spelled as the literal type a tool's trust takes, so the attribute stays
    # compatible with a ``Tool`` base that declares it.
    output_trust: Literal["trusted", "untrusted"] = "untrusted"

    def __init__(self, server: McpServer, remote_name: str, definition: ToolDefinition, *, search_hint: str) -> None:
        self._server = server
        self._definition = definition
        self.remote_name = remote_name
        self.tool_group = server.group_name
        self.search_hint = search_hint

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    @property
    def server(self) -> McpServer:
        return self._server

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        call_id = str(read_metadata(context, "tool_call_id", "") or "")
        result = await self._server.call_tool(self.remote_name, arguments)
        return tool_result_from_mcp(call_id, result, server=self._server.name, tool=self.remote_name)


TransportFactory = Callable[[], McpTransport]
"""Builds a fresh transport; called for every connection, so a reconnect starts clean."""


class McpServer:
    """One MCP server: its connection, and its tools published in a registry.

    ``transport`` builds a transport per connection (see :meth:`stdio` and
    :meth:`http` for the two the core ships). With ``registry`` given, the
    server's tools are registered there under ``name_prefix`` (by default
    ``Mcp_<Name>_``) and declared as the dynamic group ``group_name``, with
    ``description`` (else the server's title) as its catalogue line, the
    server's own ``instructions`` as the rules given when the group is first
    loaded, and ``load`` as its load mode. ``search_hint`` joins every tool's
    retrieval hint. ``call_timeout_seconds`` bounds a call and
    ``connect_timeout_seconds`` a connection and its handshake.
    """

    def __init__(
        self,
        name: str,
        transport: TransportFactory,
        *,
        registry: IToolRegistry | None = None,
        description: str = "",
        search_hint: str = "",
        load: str = "auto",
        group_name: str | None = None,
        name_prefix: str | None = None,
        call_timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 30.0,
        client_name: str = "protocore",
    ) -> None:
        if not name.strip():
            raise ValueError("an MCP server needs a name")
        if call_timeout_seconds <= 0 or connect_timeout_seconds <= 0:
            raise ValueError("MCP timeouts must be positive")
        self.name = name
        self.group_name = group_name or f"MCP server {name}"
        self.name_prefix = name_prefix or mcp_tool_prefix(name)
        self._transport_factory = transport
        self._registry = registry
        self._description = description
        self._search_hint = search_hint
        self._load = load
        self._call_timeout = call_timeout_seconds
        self._connect_timeout = connect_timeout_seconds
        self._client_name = client_name
        self._session: McpClientSession | None = None
        self._start_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[None] | None = None
        self._published: dict[str, McpTool] = {}
        self.tools: tuple[McpTool, ...] = ()
        self.state = McpServerState.stopped
        self.error: str | None = None
        self.instructions = ""

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        **options: Any,
    ) -> McpServer:
        """A server started as a local process that speaks MCP on its standard streams."""
        from protocore.mcp.stdio import StdioTransport

        return cls(name, lambda: StdioTransport(command, args, env=env, cwd=cwd), **options)

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        auth: HeaderSource | None = None,
        **options: Any,
    ) -> McpServer:
        """A server reached over streamable HTTP. Needs the ``mcp`` extra (``httpx``)."""
        from protocore.mcp.http import StreamableHttpTransport

        return cls(name, lambda: StreamableHttpTransport(url, headers=headers, auth=auth), **options)

    @property
    def connected(self) -> bool:
        return self._session is not None and not self._session.closed

    async def start(self) -> None:
        """Connect, list the tools and publish them. Does nothing when already connected.

        A failure leaves the state ``failed`` with the reason in :attr:`error`
        and is raised as :class:`McpTransportClosed` or :class:`McpError`; the
        tools already published stay, so a later call can reconnect.
        """
        async with self._start_lock:
            if self.connected:
                return
            self.state = McpServerState.connecting
            self.error = None
            session = McpClientSession(
                self._transport_factory(),
                client_name=self._client_name,
                request_timeout=self._call_timeout,
                on_tools_changed=self._tools_changed,
            )
            try:
                async with asyncio.timeout(self._connect_timeout):
                    await session.start()
                    listing = await session.list_tools()
            except BaseException as exc:
                await session.close()
                self.state = McpServerState.failed
                self.error = _describe(exc)
                _logger.warning("MCP server %s did not connect: %s", self.name, self.error)
                if isinstance(exc, TimeoutError):
                    raise McpTransportClosed(f"MCP server {self.name}: no handshake within "
                                             f"{self._connect_timeout:g}s") from exc
                raise
            self._session = session
            self.instructions = session.instructions
            self.state = McpServerState.ready
            self._publish(listing, session)

    async def stop(self) -> None:
        """Close the connection. The tools stay published; a later call reconnects."""
        task, self._refresh_task = self._refresh_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        session, self._session = self._session, None
        if session is not None:
            await session.close()
        self.state = McpServerState.stopped

    async def close(self) -> None:
        """Close the connection and withdraw the tools and the group from the registry."""
        await self.stop()
        self._withdraw()

    async def refresh_tools(self) -> tuple[McpTool, ...]:
        """List the tools again and update the registry; for servers whose changes are not announced."""
        async with self._refresh_lock:
            session = await self._ensure()
            self._publish(await session.list_tools(), session)
            return self.tools

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The server's result for one call, as JSON; every failure raises :class:`ToolInvocationError`."""
        try:
            session = await self._ensure()
        except (McpError, McpTransportClosed, OSError) as exc:
            raise ToolInvocationError(
                f"MCP server {self.name} is not reachable, so {tool} was not called: {_describe(exc)}"
            ) from exc
        try:
            return await session.call_tool(tool, arguments, timeout=self._call_timeout)
        except McpError as exc:
            detail = f" ({json.dumps(exc.data, default=str)[:600]})" if exc.data not in (None, "", {}) else ""
            raise ToolInvocationError(f"MCP server {self.name}: {tool} failed: {exc}{detail}") from exc
        except TimeoutError as exc:
            raise ToolInvocationError(
                f"MCP server {self.name}: {tool} gave no answer within {self._call_timeout:g}s; "
                "it was asked to stop, and whether it took effect is unknown"
            ) from exc
        except McpTransportClosed as exc:
            self.state = McpServerState.failed
            self.error = _describe(exc)
            raise ToolInvocationError(
                f"MCP server {self.name}: the connection broke during {tool}; whether the call took "
                f"effect is unknown, so check before calling it again ({_describe(exc)})"
            ) from exc

    async def _ensure(self) -> McpClientSession:
        session = self._session
        if session is not None and not session.closed:
            return session
        # Nothing was sent on the broken connection for this call, so opening a
        # new one cannot repeat anything.
        self._session = None
        await self.start()
        if self._session is None:  # pragma: no cover - start() raises instead
            raise McpTransportClosed(f"MCP server {self.name} did not connect")
        return self._session

    def _tools_changed(self) -> None:
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        self._refresh_task = asyncio.get_running_loop().create_task(
            self._refresh_quietly(), name=f"mcp-refresh:{self.name}"
        )

    async def _refresh_quietly(self) -> None:
        try:
            await self.refresh_tools()
        except Exception as exc:
            _logger.warning("MCP server %s: listing the changed tools failed: %s", self.name, _describe(exc))

    def _publish(self, listing: list[dict[str, Any]], session: McpClientSession) -> None:
        tools: dict[str, McpTool] = {}
        for remote in listing:
            proxy = self._proxy(remote, taken=tools)
            tools[proxy.name] = proxy
        self.tools = tuple(tools[name] for name in sorted(tools))
        if self._registry is None:
            self._published = tools
            return
        if tools:
            # Declared again with every listing, so an edited description or
            # new instructions reach the next run.
            self._registry.declare_group(
                self.group_name,
                self._group_description(session),
                dynamic=True,
                prefix=self.name_prefix,
                load=self._load,
                instructions=self.instructions[:TOOL_GROUP_INSTRUCTIONS_MAX_CHARS],
            )
        else:
            self._registry.undeclare_group(self.group_name)
        for name in self._published.keys() - tools.keys():
            self._registry.unregister(name)
        for tool in tools.values():
            self._registry.register(tool)
        self._published = tools

    def _withdraw(self) -> None:
        if self._registry is not None:
            for name in self._published:
                self._registry.unregister(name)
            # A declaration outlives its tools, and one left behind would claim
            # the prefix of a server added later under the same name.
            self._registry.undeclare_group(self.group_name)
        self._published = {}
        self.tools = ()

    def _proxy(self, remote: Mapping[str, Any], *, taken: Mapping[str, McpTool]) -> McpTool:
        remote_name = str(remote["name"])
        name = mcp_tool_name(self.name_prefix, remote_name)
        # Two remote names can sanitise or truncate to one; the later one gets
        # a numbered name rather than replacing the first.
        number = 2
        while name in taken:
            suffix = f"_{number}"
            name = mcp_tool_name(self.name_prefix, remote_name)[: _NAME_LIMIT - len(suffix)] + suffix
            number += 1
        annotations = remote.get("annotations")
        title = str(remote.get("title") or (annotations.get("title") if isinstance(annotations, Mapping) else "")
                    or "")
        description = str(remote.get("description") or title or remote_name)
        definition = ToolDefinition(
            name=name,
            description=f"[MCP {self.name}] {description}",
            parameters=parameters_from_schema(remote.get("inputSchema")),
        )
        hint = " ".join(part for part in (self._search_hint, title) if part)
        return McpTool(self, remote_name, definition, search_hint=hint)

    def _group_description(self, session: McpClientSession) -> str:
        info = session.server_info
        text = self._description or str(info.get("title") or "") or f"Tools of the MCP server {self.name}"
        return " ".join(text.split())[:_GROUP_DESCRIPTION_CHARS]


def _describe(exc: BaseException) -> str:
    """The first real cause, out of the exception groups task runners wrap failures in."""
    inner = exc
    while isinstance(inner, BaseExceptionGroup) and inner.exceptions:
        inner = inner.exceptions[0]
    text = str(inner)
    return text if isinstance(inner, McpError | McpTransportClosed) else f"{type(inner).__name__}: {text}"


__all__ = [
    "McpServer",
    "McpServerState",
    "McpTool",
    "TransportFactory",
    "mcp_tool_name",
    "mcp_tool_prefix",
    "parameters_from_schema",
    "tool_result_from_mcp",
]
