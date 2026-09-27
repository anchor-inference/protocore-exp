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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from protocore.contracts.tool_registry import (
    ADVERTISED_TOOLS_METADATA_KEY,
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
from protocore.runtime.tool_retrieval import tool_line

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
    """``ToolSearch`` LLM-facing input: a description, exact names, or both."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(
        default="",
        max_length=1000,
        description=(
            "What you need the tool to do, in a few words (e.g. 'start a "
            "background service'), or 'select:' followed by exact tool names "
            "separated by commas (e.g. 'select:Name1,Name2') to load those."
        ),
    )
    # Models reach for a separate key as often as for the prefix inside the
    # query; refusing it only costs them a turn to learn the other spelling.
    select: list[str] = Field(
        default_factory=list,
        description="Exact tool names to load, instead of or besides a query.",
    )

    @field_validator("select", mode="before")
    @classmethod
    def _split_names(cls, value: Any) -> Any:
        """A comma-separated string is the same request as a list of names."""
        if isinstance(value, str):
            return [name for name in (part.strip() for part in value.split(",")) if name]
        return value

    @model_validator(mode="after")
    def _something_asked(self) -> ToolSearchInput:
        if not self.query.strip() and not self.select:
            raise ValueError("pass a 'query' describing the tool, or 'select' with exact names")
        return self


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
        "what you need, or pass exact names in 'select' (or 'select:' and the "
        "names as the query) to load those tools. The best matches are loaded "
        "at once and can be called from your next step; the result says which, "
        "and which were already in your tool list. This loads tools only: "
        "skills are not tools, and no search here loads one."
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
        advertised = _advertised(read_metadata(context, ADVERTISED_TOOLS_METADATA_KEY))
        query = payload.query.strip()
        names = list(payload.select)
        if query[: len(SELECT_PREFIX)].lower() == SELECT_PREFIX:
            names.extend(n.strip() for n in query[len(SELECT_PREFIX) :].split(",") if n.strip())
        if names:
            # Names win over a description sent beside them: the model already
            # knows what it wants, and a search would load other tools too.
            return self._select(call_id, names, policy, advertised)
        return self._search(call_id, query, policy, rc, advertised)

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
        advertised: frozenset[str] | None,
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
        lines = _loaded_header(loaded, advertised)
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
            lines.extend(tool_line(admitted[name].definition) for name in loaded)
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
        advertised: frozenset[str] | None,
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
            lines = [*_loaded_header(loaded, advertised), "", "Matches, best first:"]
            lines.extend(tool_line(tool.definition) for tool in hits)
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


def _loaded_header(loaded: Sequence[str], advertised: frozenset[str] | None) -> list[str]:
    """What the call loaded, told apart from what the model could call already.

    A tool already in the list is reported as such and never as "Loaded": a
    model asked to use a skill, shown "Loaded: WebSearch" of a tool it had all
    along, concluded the skill was loaded and never opened it.
    """
    already = [name for name in loaded if advertised is not None and name in advertised]
    fresh = [name for name in loaded if name not in already]
    lines: list[str] = []
    if fresh:
        lines.append(f"Loaded, and callable from your next step: {', '.join(fresh)}.")
    if already:
        lines.append(f"Already in your tool list, nothing to load: {', '.join(already)}.")
    return lines or ["Nothing was loaded."]


def _advertised(raw: Any) -> frozenset[str] | None:
    """The names the calling request advertised, or ``None`` when not told."""
    if isinstance(raw, frozenset | set | tuple | list):
        return frozenset(name for name in raw if isinstance(name, str))
    return None


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
