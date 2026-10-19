"""Tests for :func:`protocore.tools.tool` decorator."""
from __future__ import annotations

import pytest

from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import ToolResult
from protocore.tools import tool


def test_decorator_builds_tool_class() -> None:
    @tool(name="echo", description="Echo back text.")
    async def _echo(context: ToolContext, text: str) -> ToolResult:
        return ToolResult(tool_call_id="t1", content=text)

    instance = _echo()
    assert isinstance(instance, Tool)
    assert instance.name == "echo"
    assert instance.definition.description == "Echo back text."
    assert "text" in instance.definition.parameters.properties
    assert "text" in instance.definition.parameters.required


def test_decorator_rejects_sync_function() -> None:
    with pytest.raises(TypeError):

        @tool(name="bad", description="bad")
        def _bad(context: ToolContext, x: str) -> ToolResult:  # type: ignore[empty-body]
            ...


async def test_decorator_invoke_passes_arguments() -> None:
    @tool(name="upper", description="Uppercase the text.")
    async def _upper(context: ToolContext, text: str) -> ToolResult:
        return ToolResult(tool_call_id="t1", content=text.upper())

    instance = _upper()
    context = ToolContext(
        tenant_id="test",
        run_id="r1",
        session_id="s1",
    )
    result = await instance.invoke(context, {"text": "hi"})
    assert result.content == "HI"


def test_optional_params_not_required() -> None:
    @tool(name="opt", description="optional param tool")
    async def _opt(context: ToolContext, text: str, count: int = 1) -> ToolResult:
        return ToolResult(tool_call_id="t1", content=text * count)

    instance = _opt()
    assert "text" in instance.definition.parameters.required
    assert "count" not in instance.definition.parameters.required


def test_decorator_rejects_unannotated_param() -> None:
    """A non-``context`` parameter without an annotation must fail at
    decoration time.

    Otherwise the param is silently dropped from the schema + ``required``
    list while remaining a required positional arg of the callable, so a
    caller validating ``arguments`` against the schema produces a dict
    missing the key and ``invoke`` raises ``TypeError`` only on first call
    (schema-matches-callable invariant violated).
    """
    with pytest.raises(TypeError, match="must be annotated"):

        @tool(name="noanno", description="x")
        async def _f(  # type: ignore[no-untyped-def]
            context: ToolContext, text: str, extra
        ) -> ToolResult:
            return ToolResult(tool_call_id="t", content="")


def test_decorator_states_the_tool_s_output_trust() -> None:
    @tool(name="fetch", description="Fetch a page.", output_trust="untrusted")
    async def _fetch(context: ToolContext, url: str) -> ToolResult:
        return ToolResult(tool_call_id="t1", content=url)

    @tool(name="note", description="Write a note.")
    async def _note(context: ToolContext, text: str) -> ToolResult:
        return ToolResult(tool_call_id="t1", content=text)

    assert _fetch().output_trust == "untrusted"
    assert _note().output_trust == "trusted"
