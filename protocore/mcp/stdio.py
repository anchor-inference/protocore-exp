"""MCP over a local process's standard streams.

The server is started as a child process and speaks newline-delimited JSON on
its stdin and stdout; what it writes to stderr is its log, read and passed to
the debug log so a chatty server can never fill the pipe and stall.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from protocore.logging_utils import get_logger
from protocore.mcp.session import McpTransportClosed

_logger = get_logger(__name__)

#: The variables a server inherits from the host's environment, as the
#: official SDKs do. The rest of the host's environment (its keys, its tokens)
#: stays with the host: a server gets what its ``env`` names and nothing more.
INHERITED_ENVIRONMENT: tuple[str, ...] = (
    ("APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "PATHEXT", "PROCESSOR_ARCHITECTURE",
     "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "USERNAME", "USERPROFILE", "PROGRAMFILES")
    if sys.platform == "win32"
    else ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER", "LANG", "LC_ALL", "TMPDIR")
)

#: The longest line a server may send. A tool result arrives as one line, and
#: the stream reader's default of 64 KiB refuses an ordinary large result.
_LINE_LIMIT = 32 * 1024 * 1024
#: How long a server gets to exit on its own once its stdin is closed.
_EXIT_GRACE_SECONDS = 2.0
_STDERR_LINE_CHARS = 500


class StdioTransport:
    """Starts ``command`` with ``args`` and exchanges MCP messages with it.

    ``env`` is added to the few variables inherited from the host
    (:data:`INHERITED_ENVIRONMENT`); ``cwd`` is the process's working
    directory. :meth:`close` closes stdin, waits briefly, then terminates and
    finally kills the process, so a server that ignores the end of its input
    is not left running.
    """

    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self._command = command
        self._args = tuple(args)
        self._env = dict(env or {})
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None

    def environment(self) -> dict[str, str]:
        """The environment the process is started with."""
        inherited = {key: value for key in INHERITED_ENVIRONMENT if (value := os.environ.get(key)) is not None}
        return {**inherited, **self._env}

    async def open(self) -> None:
        try:
            self._process = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.environment(),
                cwd=self._cwd,
                limit=_LINE_LIMIT,
            )
        except OSError as exc:
            raise McpTransportClosed(f"could not start {self._command!r}: {exc}") from exc
        if self._process.stderr is not None:
            self._stderr_task = asyncio.get_running_loop().create_task(self._drain_stderr(self._process.stderr))

    def negotiated(self, protocol_version: str) -> None:
        """Nothing to do: a process speaks one revision for its whole life."""

    async def send(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.stdin.is_closing():
            raise McpTransportClosed(f"{self._command!r} is not running")
        try:
            process.stdin.write(json.dumps(message, ensure_ascii=False).encode() + b"\n")
            await process.stdin.drain()
        except (ConnectionError, RuntimeError) as exc:
            raise McpTransportClosed(f"{self._command!r} stopped reading: {exc}") from exc

    async def receive(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise McpTransportClosed(f"{self._command!r} is not running")
        while True:
            try:
                line = await process.stdout.readline()
            except ValueError as exc:
                raise McpTransportClosed(f"{self._command!r} sent a line over the size limit") from exc
            if not line:
                raise McpTransportClosed(f"{self._command!r} exited (status {process.returncode})")
            text = line.strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except ValueError:
                # A server that prints a banner to stdout breaks the framing
                # of nothing as long as the line is skipped.
                _logger.debug("%s wrote a line that is not JSON: %s", self._command, text[:200])
                continue
            if isinstance(message, dict):
                return message

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None and not process.stdin.is_closing():
            process.stdin.close()
            with contextlib.suppress(Exception):
                await process.stdin.wait_closed()
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), _EXIT_GRACE_SECONDS)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), _EXIT_GRACE_SECONDS)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
        task, self._stderr_task = self._stderr_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        while True:
            try:
                line = await stream.readline()
            except ValueError:
                continue
            if not line:
                return
            _logger.debug("%s: %s", self._command, line.decode("utf-8", "replace").rstrip()[:_STDERR_LINE_CHARS])


__all__ = ["INHERITED_ENVIRONMENT", "StdioTransport"]
