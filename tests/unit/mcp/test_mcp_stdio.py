"""MCP over a child process's standard streams, with the process faked in the test."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from protocore.mcp import McpServer, McpTransportClosed, StdioTransport
from protocore.mcp import stdio as stdio_module

from ._fake_server import FakeMcpServer, text_result


class _Stdin:
    def __init__(self, process: FakeProcess) -> None:
        self._process = process
        self._buffer = b""
        self._closing = False

    def write(self, data: bytes) -> None:
        if self._closing:
            raise ConnectionResetError("stdin closed")
        self._buffer += data
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._process.lines_in.append(line)
            asyncio.get_running_loop().create_task(self._process.answer(json.loads(line)))

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True
        if self._process.exits_on_eof:
            self._process.exit(0)

    async def wait_closed(self) -> None:
        return None


class FakeProcess:
    """A child process whose other end is a :class:`FakeMcpServer`."""

    def __init__(self, server: FakeMcpServer, *, exits_on_eof: bool = True, banner: bytes = b"") -> None:
        self.server = server
        self.exits_on_eof = exits_on_eof
        self.lines_in: list[bytes] = []
        self.stdin = _Stdin(self)
        self.stdout = asyncio.StreamReader(limit=stdio_module._LINE_LIMIT)
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(b"server starting\n")
        self.stderr.feed_eof()
        if banner:
            self.stdout.feed_data(banner)
        self.returncode: int | None = None
        self.signals: list[str] = []
        self._exited = asyncio.Event()

    async def answer(self, message: dict[str, Any]) -> None:
        for reply in await self.server.handle(message):
            self.stdout.feed_data(json.dumps(reply).encode() + b"\n")

    def exit(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
            self.stdout.feed_eof()
            self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.signals.append("terminate")

    def kill(self) -> None:
        self.signals.append("kill")
        self.exit(-9)


class Spawner:
    """Stands in for process creation: records each start and hands out the next fake process."""

    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.processes: list[FakeProcess] = []

    async def __call__(self, program: str, *args: str, **kwargs: Any) -> FakeProcess:
        self.starts.append({"program": program, "args": args, **kwargs})
        return self.processes.pop(0)


@pytest.fixture
def spawner(monkeypatch: pytest.MonkeyPatch) -> Spawner:
    spawner = Spawner()
    monkeypatch.setattr(stdio_module.asyncio, "create_subprocess_exec", spawner)
    return spawner


async def test_a_stdio_server_is_started_and_spoken_to_line_by_line(
    spawner: Spawner, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("SECRET_TOKEN", "do-not-leak")
    fake = FakeMcpServer(tools=[{"name": "echo", "inputSchema": {"type": "object"}}],
                         handlers={"echo": lambda args: text_result(args["text"] * 3)})
    process = FakeProcess(fake, banner=b"starting up...\n")
    spawner.processes.append(process)
    caplog.set_level(logging.DEBUG, logger="protocore.mcp.stdio")

    server = McpServer.stdio("local", "server-bin", ["--flag"], env={"API_ROOT": "/data"}, cwd="/tmp")
    result = await server.call_tool("echo", {"text": "ab"})

    assert result["content"][0]["text"] == "ababab"
    [start] = spawner.starts
    assert start["program"] == "server-bin"
    assert start["args"] == ("--flag",)
    assert start["cwd"] == "/tmp"
    assert start["env"]["API_ROOT"] == "/data"
    assert start["env"]["PATH"] == "/usr/bin"
    # Only the named variables and a few basics reach a server.
    assert "SECRET_TOKEN" not in start["env"]
    assert start["limit"] >= 1024 * 1024
    assert all(line.endswith(b"}") and b"\n" not in line for line in process.lines_in)
    await server.close()
    assert process.returncode == 0
    assert process.signals == []
    assert any("server starting" in record.getMessage() for record in caplog.records)


async def test_a_large_result_line_is_read_whole(spawner: Spawner) -> None:
    big = "x" * 300_000
    fake = FakeMcpServer(tools=[{"name": "dump"}], handlers={"dump": lambda args: text_result(big)})
    spawner.processes.append(FakeProcess(fake))
    server = McpServer.stdio("local", "server-bin")

    result = await server.call_tool("dump", {})

    assert result["content"][0]["text"] == big
    await server.close()


async def test_a_server_that_ignores_the_end_of_its_input_is_stopped(
    spawner: Spawner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stdio_module, "_EXIT_GRACE_SECONDS", 0.01)
    process = FakeProcess(FakeMcpServer(), exits_on_eof=False)
    spawner.processes.append(process)
    transport = StdioTransport("stubborn")
    await transport.open()
    await transport.close()

    assert process.signals == ["terminate", "kill"]
    await transport.close()


async def test_a_process_that_exits_closes_the_transport(spawner: Spawner) -> None:
    process = FakeProcess(FakeMcpServer())
    spawner.processes.append(process)
    transport = StdioTransport("short-lived")
    await transport.open()
    process.exit(3)

    with pytest.raises(McpTransportClosed, match="status 3"):
        await transport.receive()
    await transport.close()


async def test_a_command_that_cannot_start_is_a_closed_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(stdio_module.asyncio, "create_subprocess_exec", missing)
    with pytest.raises(McpTransportClosed, match="could not start"):
        await StdioTransport("missing-bin").open()


async def test_sending_before_open_or_after_close_is_refused() -> None:
    transport = StdioTransport("never-started")
    with pytest.raises(McpTransportClosed):
        await transport.send({"jsonrpc": "2.0", "method": "x"})
    with pytest.raises(McpTransportClosed):
        await transport.receive()
    transport.negotiated("2025-11-25")
    await transport.close()
