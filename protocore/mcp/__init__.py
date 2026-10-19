"""MCP servers' tools as core tools.

An :class:`McpServer` connects to one server over its standard streams
(:meth:`McpServer.stdio`) or streamable HTTP (:meth:`McpServer.http`, which
needs ``httpx`` from the ``mcp`` extra: ``pip install "protocore[mcp]"``),
publishes its tools in a tool registry as one deferrable group, and forwards
calls. Nothing else in the core imports this package.

.. code-block:: python

    from protocore.mcp import McpServer

    files = McpServer.stdio(
        "files", "npx", ["-y", "some-mcp-files-server", "/srv/data"],
        registry=registry, search_hint="file folder directory read list",
    )
    await files.start()          # optional: the first call connects too
    ...
    await files.close()

See ``docs/mcp.md``.
"""
from __future__ import annotations

from protocore.mcp.session import (
    LATEST_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    McpClientSession,
    McpError,
    McpTransport,
    McpTransportClosed,
)
from protocore.mcp.stdio import INHERITED_ENVIRONMENT, StdioTransport
from protocore.mcp.tools import (
    McpServer,
    McpServerState,
    McpTool,
    TransportFactory,
    mcp_tool_name,
    mcp_tool_prefix,
    parameters_from_schema,
    tool_result_from_mcp,
)

__all__ = [
    "INHERITED_ENVIRONMENT",
    "LATEST_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "McpClientSession",
    "McpError",
    "McpServer",
    "McpServerState",
    "McpTool",
    "McpTransport",
    "McpTransportClosed",
    "StdioTransport",
    "TransportFactory",
    "mcp_tool_name",
    "mcp_tool_prefix",
    "parameters_from_schema",
    "tool_result_from_mcp",
]
