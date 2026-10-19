"""``@tool`` decorator — Pydantic TypeAdapter → JSON Schema.

Lightweight in-core tool registration helper. Pydantic ``TypeAdapter`` generates
JSON Schema from type hints; a host's tools use this decorator (but
live in the host package).
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar, cast

from pydantic import TypeAdapter

from protocore.contracts.tools import Tool, ToolContext, ToolOutputTrust
from protocore.contracts.types import ToolDefinition, ToolParameterSchema, ToolResult

P = ParamSpec("P")
R = TypeVar("R")

ToolFunc = Callable[..., Awaitable[ToolResult]]


def _build_schema(func: Callable[..., Any], name: str) -> ToolParameterSchema:
    """Build a JSON Schema fragment from a function's type hints.

    Skips the special ``context: ToolContext`` parameter.

    Raises :class:`TypeError` at decoration time if a non-``context``
    parameter lacks an annotation: such a parameter would be silently
    dropped from the schema and ``required`` list while remaining a
    required positional argument of the callable, breaking the
    ``schema matches the actual callable`` invariant and only failing on
    first invocation. Mirrors the ``iscoroutinefunction`` guard in
    :func:`tool`.
    """
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name == "context":
            continue
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"@tool {name!r} parameter {param_name!r} must be annotated"
            )
        if param.annotation is ToolContext:
            continue
        adapter = TypeAdapter(param.annotation)
        properties[param_name] = adapter.json_schema()
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    return ToolParameterSchema(properties=properties, required=required)


def tool(
    *,
    name: str,
    description: str,
    output_trust: ToolOutputTrust = "trusted",
) -> Callable[[ToolFunc], type[Tool]]:
    """Decorate an async function to build a :class:`Tool` subclass.

    Usage::

        @tool(name="echo", description="Echo back the input.")
        async def echo(context: ToolContext, text: str) -> ToolResult:
            return ToolResult(tool_call_id="...", content=text)

    The decorated callable is replaced with a :class:`Tool` subclass.
    ``output_trust="untrusted"`` marks a tool whose results carry text from
    outside the run, which the core then fences as data.
    """

    def _wrap(func: ToolFunc) -> type[Tool]:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"@tool {name!r} must wrap an async function")

        schema = _build_schema(func, name)
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=schema,
        )

        trust = output_trust

        class _DecoratedTool(Tool):
            output_trust = trust

            @property
            def name(self) -> str:
                return definition.name

            @property
            def definition(self) -> ToolDefinition:
                return definition

            async def invoke(
                self,
                context: ToolContext,
                arguments: dict[str, Any],
            ) -> ToolResult:
                return await func(context=context, **arguments)

        _DecoratedTool.__name__ = f"Tool_{name}"
        return cast(type[Tool], _DecoratedTool)

    return _wrap


__all__ = ["tool"]
