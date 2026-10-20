# MCP — an MCP server's tools as core tools

> Audience: a host that wants a Model Context Protocol server's tools in its
> runs.
> Code: `protocore/mcp/` (`session.py` the JSON-RPC client, `stdio.py` and
> `http.py` the transports, `tools.py` the server and its tool proxies).

`protocore.mcp` connects to an MCP server, publishes its tools in the tool
registry as one deferrable group, and forwards the model's calls. Nothing else
in the core imports it.

```bash
pip install "protocore[mcp]==2.0.0a25"   # adds httpx, for the HTTP transport
```

The stdio transport needs no extra; the streamable-HTTP transport needs
`httpx`, which the `mcp` extra installs. The client is written in the core
rather than taken from the official MCP SDK: a tool adapter needs a small part
of the protocol (the handshake, `tools/list`, `tools/call`, `ping`, the
tool-list notification), and the SDK brings a web server, a JWT library and
telemetry into every host that only wants to call a few tools. A host that
already runs the SDK can still pass its own transport (below).

## A server in five lines

```python
from protocore.mcp import McpServer

tracker = McpServer.stdio(
    "tracker", "tracker-mcp", ["--project", "core"],
    env={"TRACKER_TOKEN": token},
    registry=registry,
    search_hint="issue bug ticket report",
)
await tracker.start()      # optional: the first call connects too
...                        # runs see Mcp_Tracker_* tools, held back behind ToolSearch
await tracker.close()      # stops the server and withdraws its tools
```

`McpServer.http(name, url, *, headers=None, auth=None, **options)` is the
same over streamable HTTP. Both are shorthands for
`McpServer(name, transport_factory, **options)`, where the factory builds a
fresh `McpTransport` for every connection.

| Option | Default | What it does |
|---|---|---|
| `registry` | `None` | where the tools are published; without one, read them from `server.tools` |
| `description` | the server's title | the group's line in the catalogue (one line, 200 characters) |
| `search_hint` | `""` | joins every tool's retrieval hint, with the tool's title |
| `load` | `"auto"` | the group's load mode: `eager`, `auto`, `lazy` |
| `group_name` | `"MCP server <name>"` | the group's name |
| `name_prefix` | `Mcp_<Name>_` | the prefix of every tool name, and the group's prefix |
| `call_timeout_seconds` | 120 | how long one call may take |
| `connect_timeout_seconds` | 30 | how long a connection and its handshake may take |
| `client_name` | `"protocore"` | the `clientInfo.name` sent on the handshake |

## Tools, groups and deferral

Each remote tool becomes an `McpTool`, named `<prefix><remote name>` with every
character outside `[A-Za-z0-9_-]` as `_`, cut to 64 characters; two remote
names that come out the same get `_2`, `_3`. The server's tools are declared
as one **dynamic** group (`declare_group(..., dynamic=True, prefix=...)`):
its membership is an outside server's, not the host's code, so the loop holds
it back first and names it in one catalogue line, and a model loads the tools
it needs through `ToolSearch` (see [Holding tool groups back](tools.md#holding-tool-groups-back)).
The server's own `instructions` from the handshake become the group's
`instructions`, written beside the tools only once they are loaded. The group
is declared again with every listing, so an edited description reaches the
next run, and withdrawn with the last tool, because a declaration left behind
would claim the prefix of a server added later under the same name.

Each tool carries `tool_group` (its group, explicitly), `search_hint` and
`output_trust = "untrusted"`: its output is text an outside server wrote, and a
runtime that fences untrusted output fences it.

## Schemas

`properties`, `required` and a boolean `additionalProperties` reach the tool
definition, so a server's strict schema stays strict on the wire. Local
`$ref` pointers into `$defs` or `definitions` are inlined, since the core's
schema keeps no definitions section and a reference to nothing is a schema no
model can satisfy; a recursive schema is inlined eight levels deep and left
open below. A schema-valued `additionalProperties` has no place in the core's
field and is left out — the permissive reading, and the server still
validates what it receives.

## Results and errors

| The server returns | The tool result |
|---|---|
| `text` content | the text, one block per line |
| `image`, `audio` | `[image image/png returned by the tool]`; an image is also attached as `ToolResult.images`, so a model that takes images sees it |
| `resource_link` | `[resource <name>: <uri>]` |
| an embedded `resource` | its text, or `[binary resource <uri> (<type>)]` |
| `structuredContent` | the result when there is no text, else `ui_payload["structured_content"]` |
| `isError: true` | an error result (`is_error=True`), the text as the server wrote it |
| nothing | `(empty result)` |

The result's `metadata` names the server and the remote tool. Everything that
is not a tool's own answer raises `ToolInvocationError`, the core's tool error
contract, which the dispatcher turns into an error result the model reads:

- a protocol error (unknown tool, invalid arguments) — with the server's code,
  message and data;
- no answer within `call_timeout_seconds` — the server is sent
  `notifications/cancelled`, and the message says whether the call took effect
  is unknown;
- the connection breaking during the call — whether it took effect is unknown,
  and the message tells the model to check before calling again;
- a server that cannot be reached — the call was not sent.

## Lifecycle

`start()` connects, completes the handshake, lists the tools (following
`nextCursor`) and publishes them; concurrent starts open one connection.
`stop()` closes the connection and keeps the tools published, so the next
call reconnects. `close()` also withdraws the tools and the group.

A call that finds the connection closed reconnects once before sending:
nothing was sent on the broken connection, so nothing can be repeated. A call
whose connection breaks while it is in flight is not retried, because the
server may have acted on it. `state` (`stopped`, `connecting`, `ready`,
`failed`) and `error` say where the server stands.

A server that sends `notifications/tools/list_changed` is listed again and the
registry updated in the background. `refresh_tools()` does it on demand — the
HTTP transport opens no stream for unprompted messages, so over HTTP a change
is heard only inside the answer to some request.

## Transports

**stdio** (`StdioTransport(command, args, *, env=None, cwd=None)`) starts the
server as a child process and speaks newline-delimited JSON on its standard
streams. The process inherits only a few variables of the host's environment
(`INHERITED_ENVIRONMENT`: `HOME`, `PATH`, `USER`, the locale and a few more)
plus what `env` names, so a host's keys do not leak to a server that was not
given them. Lines up to 32 MiB are read whole; what the server writes to
stderr goes to the debug log; a line on stdout that is not JSON is skipped.
On close, stdin is closed, then the process is terminated and finally killed
if it does not exit.

**Streamable HTTP** (`StreamableHttpTransport(url, *, headers=None, auth=None,
client=None)`) posts each message and reads the answer as JSON or as an SSE
stream, which may carry the server's own notifications and requests before the
answer. The `Mcp-Session-Id` the handshake hands out and the negotiated
`MCP-Protocol-Version` go on every later request, and the session is ended
with a `DELETE` on close. `auth` is an async callable returning headers,
awaited on every connection, so a renewed bearer token is picked up on
reconnect. A refusal (`McpHttpError`, with its `status`) ends the session: a
`401` means the token needs renewing, a `404` that the server forgot the
session; the next call reconnects. Redirects are not followed, since requests
may carry a credential.

**Your own.** Anything with `open()`, `send(message)`, `receive()`,
`negotiated(protocol_version)` and `close()` is an `McpTransport` — for
instance a bridge to an SDK session a host already runs, or an in-process
server in tests. Pass `lambda: MyTransport(...)` as the factory.

## The session

`McpClientSession(transport, *, client_name, client_version, request_timeout,
on_tools_changed)` is the JSON-RPC client under `McpServer`, usable on its
own: `start()`, `list_tools()`, `call_tool(name, arguments, timeout=)`,
`request(method, params)`, `notify(method, params)`, `close()`. It offers
protocol revision `2025-11-25` and accepts the revisions in
`SUPPORTED_PROTOCOL_VERSIONS`; a server that answers with another is refused
with `McpError` rather than half understood. It answers the server's `ping`
and refuses other server requests (sampling, roots, elicitation) with
"method not found", since it declared none of those capabilities. An error
answer raises `McpError(code, message, data)`; a connection that is gone
raises `McpTransportClosed` to every waiting and later request.

## Not covered

OAuth flows (a host obtains the token and passes it through `auth`), the
legacy HTTP+SSE transport, resources and prompts, sampling and elicitation,
and the HTTP `GET` stream for unprompted server messages.
