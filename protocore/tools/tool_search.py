"""``ToolSearch`` — find tools that are available but not loaded, and load them.

When a run holds tool groups back (:mod:`protocore.runtime.tool_deferral`),
this is how the model gets them: it describes what it needs, or names the
tools exactly with ``select:``, and the best matches are loaded onto the
surface from the next request on. The tool itself only ranks and reports; it
names what it loaded under
:data:`~protocore.contracts.tool_registry.TOOLS_LOADED_METADATA_KEY`, and the
loop — which owns the surface — does the loading, for names the live policy
admits.

The loop advertises this tool only while something is held back. With the whole
catalogue on the surface it has nothing to find, and a model offered a search
tool it does not need spends turns on it.
"""
from __future__ import annotations

import difflib
from collections.abc import Sequence
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.tool_registry import (
    TOOL_VISIBILITY_POLICY_METADATA_KEY,
    TOOLS_LOADED_METADATA_KEY,
    IToolRegistry,
    ToolVisibilityPolicy,
    policy_admits,
)
from protocore.contracts.tool_retrieval import RetrievalSettings
from protocore.contracts.tool_roles import ToolRole
from protocore.contracts.tools import Tool, ToolContext, read_metadata
from protocore.contracts.types import ToolDefinition, ToolParameterSchema, ToolResult
from protocore.runtime.tool_retrieval import split_summary

TOOL_SEARCH_TOOL_NAME: Final[str] = "ToolSearch"

#: The query form that names tools exactly instead of describing them.
SELECT_PREFIX: Final[str] = "select:"

# Used when the run carries no constants, as a tool invoked outside a run does.
_DEFAULT_MAX_RESULTS: Final[int] = 8
_DEFAULT_AUTOLOAD: Final[int] = 3
# How many near names an unknown ``select`` name is answered with, and how
# close they must be. Loose on purpose: a wrong case or a missing word is the
# usual mistake, and a suggestion that is merely plausible still beats none.
_SUGGESTIONS: Final[int] = 3
_SUGGESTION_CUTOFF: Final[float] = 0.5


class ToolSearchInput(BaseModel):
    """``ToolSearch`` LLM-facing input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "What you need the tool to do, in a few words (e.g. 'start a "
            "background service'), or 'select:' followed by exact tool names "
            "separated by commas (e.g. 'select:Name1,Name2') to load those."
        ),
    )


def _signature(definition: ToolDefinition) -> str:
    """``Name(param1, param2*)`` — required parameters marked with a star."""
    required = set(definition.parameters.required)
    params = [
        f"{name}*" if name in required else name
        for name in definition.parameters.properties
    ]
    return f"{definition.name}({', '.join(params)})"


def _hit_line(tool: Tool) -> str:
    definition = tool.definition
    summary, _ = split_summary(definition.description)
    return f"{_signature(definition)} — {summary}" if summary else _signature(definition)


class ToolSearchTool(Tool):
    """Search the admitted catalogue and load the best matches."""

    name_: ClassVar[str] = TOOL_SEARCH_TOOL_NAME
    # The loop recognises a discovery tool by this role, and advertises one only
    # while something is held back.
    tool_roles: ClassVar[tuple[ToolRole, ...]] = (ToolRole.discovers_tools,)
    # Survives the per-message clip like AskUser does; the loop still hides it
    # when there is nothing to find.
    always_load: ClassVar[bool] = True
    # Reads the catalogue and nothing else, so parallel calls are safe; the
    # loop applies what they loaded in the order the model asked.
    is_concurrent_safe: ClassVar[bool] = True
    is_destructive: ClassVar[bool] = False
    description_: ClassVar[str] = (
        "Find and load tools that are available but not loaded yet. Describe "
        "what you need, or pass 'select:' and exact names to load those tools. "
        "The best matches are loaded at once and can be called from your next "
        "step; the result says which."
    )
    search_hint: ClassVar[str] = (
        "find load discover tool capability "
        "найти загрузить инструмент возможность"
    )

    def __init__(self, registry: IToolRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return self.name_

    @property
    def definition(self) -> ToolDefinition:
        schema = ToolSearchInput.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        return ToolDefinition(
            name=self.name_,
            description=self.description_,
            parameters=ToolParameterSchema(
                properties=properties if isinstance(properties, dict) else {},
                required=required if isinstance(required, list) else [],
            ),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        payload = ToolSearchInput.model_validate(arguments)
        call_id = str(read_metadata(context, "tool_call_id", "") or "")
        raw_policy = read_metadata(context, TOOL_VISIBILITY_POLICY_METADATA_KEY)
        policy = raw_policy if isinstance(raw_policy, ToolVisibilityPolicy) else None
        rc = context.run_state.rc if context.run_state is not None else None
        query = payload.query.strip()
        if query[: len(SELECT_PREFIX)].lower() == SELECT_PREFIX:
            names = [n.strip() for n in query[len(SELECT_PREFIX) :].split(",") if n.strip()]
            return self._select(call_id, names, policy)
        return self._search(call_id, query, policy, rc)

    # ------------------------------------------------------------------

    def _admitted(self, policy: ToolVisibilityPolicy | None) -> dict[str, Tool]:
        """Every tool the policy admits, except the discovery tools themselves."""
        return {
            tool.name: tool
            for tool in self._registry.list_all()
            if policy_admits(policy, tool.name) and not _discovers_tools(tool)
        }

    def _select(
        self,
        call_id: str,
        names: Sequence[str],
        policy: ToolVisibilityPolicy | None,
    ) -> ToolResult:
        admitted = self._admitted(policy)
        by_folded = {name.casefold(): name for name in admitted}
        loaded: list[str] = []
        missing: list[str] = []
        for requested in names:
            # A name in the wrong case is still unambiguous; loading the one it
            # means is kinder than refusing it.
            actual = requested if requested in admitted else by_folded.get(requested.casefold())
            if actual is None:
                missing.append(requested)
            elif actual not in loaded:
                loaded.append(actual)
        lines = [_loaded_header(loaded)]
        for requested in missing:
            # Suggestions come from the admitted names only, so a name the
            # policy blocks is never revealed by being "close".
            close = difflib.get_close_matches(
                requested.casefold(), list(by_folded), n=_SUGGESTIONS, cutoff=_SUGGESTION_CUTOFF
            )
            if close:
                nearest = ", ".join(by_folded[name] for name in close)
                lines.append(f"No tool named {requested!r}. Closest: {nearest}.")
            else:
                lines.append(f"No tool named {requested!r}; describe what you need instead.")
        if loaded:
            lines.append("")
            lines.extend(_hit_line(admitted[name]) for name in loaded)
        return ToolResult(
            tool_call_id=call_id,
            content="\n".join(lines),
            metadata={TOOLS_LOADED_METADATA_KEY: loaded, "matches": loaded},
        )

    def _search(
        self,
        call_id: str,
        query: str,
        policy: ToolVisibilityPolicy | None,
        rc: Any,
    ) -> ToolResult:
        max_results = _positive(getattr(rc, "tool_search_max_results", None), _DEFAULT_MAX_RESULTS)
        autoload = _non_negative(
            getattr(rc, "tool_search_autoload_count", None), _DEFAULT_AUTOLOAD
        )
        retrieval = _retrieval_settings(rc)
        # Discovery tools are excluded after the search, so ask for enough
        # extra that excluding them cannot shorten the list.
        discovery = [t.name for t in self._registry.list_all() if _discovers_tools(t)]
        hits = [
            tool
            for tool in self._registry.search(
                query,
                top_k=max_results + len(discovery),
                policy=policy,
                retrieval=retrieval,
            )
            if not _discovers_tools(tool)
        ][:max_results]
        loaded = [tool.name for tool in hits[:autoload]]
        if not hits:
            content = (
                f"No tool matches {query!r}. Try other words, or 'select:' with "
                "a name from the list of available tools."
            )
        else:
            lines = [_loaded_header(loaded), "", "Matches, best first:"]
            lines.extend(_hit_line(tool) for tool in hits)
            if len(hits) > len(loaded):
                lines.append("")
                lines.append("Load any other match with 'select:' and its name.")
            content = "\n".join(lines)
        return ToolResult(
            tool_call_id=call_id,
            content=content,
            metadata={
                TOOLS_LOADED_METADATA_KEY: loaded,
                "matches": [tool.name for tool in hits],
            },
        )


def _loaded_header(loaded: Sequence[str]) -> str:
    if not loaded:
        return "Nothing was loaded."
    return f"Loaded, and callable from your next step: {', '.join(loaded)}."


def _discovers_tools(tool: Tool) -> bool:
    roles = getattr(tool, "tool_roles", ())
    return isinstance(roles, tuple | frozenset | list) and ToolRole.discovers_tools in roles


def _positive(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


def _non_negative(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _retrieval_settings(rc: Any) -> RetrievalSettings | None:
    """The run's ranking settings, or ``None`` (the registry's defaults)."""
    if rc is None:
        return None
    try:
        return RetrievalSettings.from_constants(rc)
    except AttributeError:
        return None


__all__ = [
    "SELECT_PREFIX",
    "TOOL_SEARCH_TOOL_NAME",
    "ToolSearchInput",
    "ToolSearchTool",
]
