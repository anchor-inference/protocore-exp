"""``ToolSearch`` — find tools that are available but not loaded, and load them.

When a run holds tool groups back (:mod:`protocore.runtime.tool_deferral`),
this is how the model gets them: it describes what it needs, names the tools
exactly with ``select:``, or names a whole group with ``group``, and the tools
are loaded onto the surface from the next request on. The first tools of a
group with rules to follow bring the rules with them, once per run. The tool itself only ranks and reports; it
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
    TOOL_ALLOWLIST_METADATA_KEY,
    TOOL_GROUP_RULES_GIVEN_METADATA_KEY,
    TOOL_GROUP_RULES_MARK_METADATA_KEY,
    TOOL_GROUP_RULES_METADATA_KEY,
    TOOL_GROUPS_LOADED_METADATA_KEY,
    TOOL_VISIBILITY_POLICY_METADATA_KEY,
    TOOLS_LOADED_METADATA_KEY,
    IToolRegistry,
    ToolVisibilityPolicy,
    group_rules_text,
    policy_admits,
    tool_group_of,
)
from protocore.contracts.tool_retrieval import RetrievalSettings
from protocore.contracts.tool_roles import ToolRole
from protocore.contracts.tools import Tool, ToolContext, read_metadata
from protocore.contracts.types import ToolDefinition, ToolParameterSchema, ToolResult
from protocore.runtime.tool_retrieval import tool_line

TOOL_SEARCH_TOOL_NAME: Final[str] = "ToolSearch"

#: The query form that names tools exactly instead of describing them.
SELECT_PREFIX: Final[str] = "select:"

#: A ``select`` entry that names a whole group rather than one tool.
GROUP_PREFIX: Final[str] = "group:"

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
    group: str = Field(
        default="",
        max_length=200,
        description=(
            "The exact name of a tool group from the list of available tools, "
            "to load every tool in it."
        ),
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
        if not self.query.strip() and not self.select and not self.group.strip():
            raise ValueError(
                "pass a 'query' describing the tool, 'select' with exact names, "
                "or 'group' with a group's name"
            )
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
        "what you need, pass exact names in 'select' (or 'select:' and the "
        "names as the query) to load those tools, or pass a group's name in "
        "'group' to load all of its tools. The tools are loaded at once and "
        "can be called from your next step; the result says which, and which "
        "were already in your tool list. This loads tools only: skills are "
        "not tools, and no search here loads one."
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
        if not isinstance(raw_policy, ToolVisibilityPolicy):
            # No policy is no answer about what is admitted, and reading it as
            # "everything" listed and loaded tools the gate then refused.
            return ToolResult(
                tool_call_id=call_id,
                content="Tool search is not available here: nothing says which tools this run may use.",
                is_error=True,
                metadata={TOOLS_LOADED_METADATA_KEY: [], "matches": []},
            )
        policy = self._within_allowlist(
            raw_policy, _allowlist(read_metadata(context, TOOL_ALLOWLIST_METADATA_KEY))
        )
        rc = context.run_state.rc if context.run_state is not None else None
        advertised = _advertised(read_metadata(context, ADVERTISED_TOOLS_METADATA_KEY))
        # Outside a loop nothing says what was given (``None``).
        given = _advertised(read_metadata(context, TOOL_GROUP_RULES_GIVEN_METADATA_KEY))
        raw_mark = read_metadata(context, TOOL_GROUP_RULES_MARK_METADATA_KEY, "")
        mark = raw_mark if isinstance(raw_mark, str) else ""
        query = payload.query.strip()
        entries = list(payload.select)
        if query[: len(SELECT_PREFIX)].lower() == SELECT_PREFIX:
            entries.extend(n.strip() for n in query[len(SELECT_PREFIX) :].split(",") if n.strip())
        names: list[str] = []
        groups: list[str] = [payload.group.strip()] if payload.group.strip() else []
        for entry in entries:
            if entry[: len(GROUP_PREFIX)].lower() == GROUP_PREFIX:
                if entry[len(GROUP_PREFIX) :].strip():
                    groups.append(entry[len(GROUP_PREFIX) :].strip())
            else:
                names.append(entry)
        if names or groups:
            # Names win over a description sent beside them: the model already
            # knows what it wants, and a search would load other tools too.
            return self._select(call_id, names, groups, policy, advertised, given, mark)
        return self._search(call_id, query, policy, rc, advertised, given, mark)

    # ------------------------------------------------------------------

    def _within_allowlist(
        self, policy: ToolVisibilityPolicy, allowlist: frozenset[str] | None
    ) -> ToolVisibilityPolicy:
        """``policy`` narrowed to a child's declared tool set, as the gate narrows it.

        Without it a child was told "Loaded" of a tool outside its set, and
        the call that followed was refused. Folded into ``blocked`` and not
        ``visible``: an empty ``visible`` means everything, so an intersection
        that came out empty would have widened the search to the catalogue.
        """
        if allowlist is None:
            return policy
        outside = {tool.name for tool in self._registry.list_all() if tool.name not in allowlist}
        if not outside - set(policy.blocked):
            return policy
        return policy.model_copy(update={"blocked": set(policy.blocked) | outside})

    def _admitted(self, policy: ToolVisibilityPolicy) -> dict[str, Tool]:
        """Every tool the policy admits, except the discovery tools themselves."""
        return {
            tool.name: tool
            for tool in self._registry.list_all()
            if policy_admits(policy, tool.name) and not _discovers_tools(tool)
        }

    def _group_members(self, admitted: dict[str, Tool]) -> dict[str, list[str]]:
        """Each group with an admitted tool, and those tools by name."""
        declared = self._registry.tool_groups()
        members: dict[str, list[str]] = {}
        for name in sorted(admitted):
            group = tool_group_of(admitted[name], declared)
            if group:
                members.setdefault(group, []).append(name)
        return members

    def _rules(
        self,
        admitted: dict[str, Tool],
        loaded: Sequence[str],
        advertised: frozenset[str] | None,
        given: frozenset[str] | None,
    ) -> list[tuple[str, str]]:
        """The rules owed with this load: of each loaded tool's group not yet given them.

        Inside a loop ``given`` is the whole answer. The loop keeps it to the
        groups whose rules the model can still read, and after a compaction
        that may leave out a group whose tools are still listed: a tool on
        the list is no proof its rules are in view, and skipping it on that
        ground reloaded a group with no rules at all. Outside a loop nothing
        says what was given, and a tool the model already had is taken to
        have come with its rules.
        """
        declared = {group.name: group for group in self._registry.tool_groups()}
        owed: list[tuple[str, str]] = []
        for name in loaded:
            if given is None and advertised is not None and name in advertised:
                continue
            group = tool_group_of(admitted[name], list(declared.values()))
            declaration = declared.get(group)
            if (
                declaration is None
                or not declaration.instructions
                or (given is not None and group in given)
                or any(group == seen for seen, _ in owed)
            ):
                continue
            owed.append((group, declaration.instructions))
        return owed

    def _select(
        self,
        call_id: str,
        names: Sequence[str],
        groups: Sequence[str],
        policy: ToolVisibilityPolicy,
        advertised: frozenset[str] | None,
        given: frozenset[str] | None,
        mark: str,
    ) -> ToolResult:
        admitted = self._admitted(policy)
        by_folded = {name.casefold(): name for name in admitted}
        loaded: list[str] = []
        missing: list[str] = []
        whole: list[str] = []
        missing_groups: list[str] = []
        if groups:
            members = self._group_members(admitted)
            groups_by_folded = {name.casefold(): name for name in members}
            for requested in groups:
                group = requested if requested in members else groups_by_folded.get(
                    requested.casefold()
                )
                if group is None:
                    missing_groups.append(requested)
                    continue
                if group not in whole:
                    whole.append(group)
                loaded.extend(name for name in members[group] if name not in loaded)
        for requested in names:
            # A name in the wrong case is still unambiguous; loading the one it
            # means is kinder than refusing it.
            actual = requested if requested in admitted else by_folded.get(requested.casefold())
            if actual is None:
                missing.append(requested)
            elif actual not in loaded:
                loaded.append(actual)
        lines = _loaded_header(loaded, advertised)
        for requested in missing_groups:
            # Only groups with a tool the run may use are named, so a group
            # the policy blocks whole is never revealed by the list.
            available = ", ".join(sorted(self._group_members(admitted)))
            if available:
                lines.append(f"No tool group named {requested!r}. Groups: {available}.")
            else:
                lines.append(f"No tool group named {requested!r}.")
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
        rules = self._rules(admitted, loaded, advertised, given)
        for group, text in rules:
            lines.append("")
            lines.append(group_rules_text(group, text, mark))
        return ToolResult(
            tool_call_id=call_id,
            content="\n".join(lines),
            metadata={
                TOOLS_LOADED_METADATA_KEY: loaded,
                TOOL_GROUPS_LOADED_METADATA_KEY: whole,
                TOOL_GROUP_RULES_METADATA_KEY: [group for group, _ in rules],
                "matches": loaded,
                "load_via": "select",
            },
        )

    def _search(
        self,
        call_id: str,
        query: str,
        policy: ToolVisibilityPolicy,
        rc: Any,
        advertised: frozenset[str] | None,
        given: frozenset[str] | None,
        mark: str,
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
        rules = self._rules({tool.name: tool for tool in hits}, loaded, advertised, given)
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
            for group, text in rules:
                lines.append("")
                lines.append(group_rules_text(group, text, mark))
            content = "\n".join(lines)
        return ToolResult(
            tool_call_id=call_id,
            content=content,
            metadata={
                TOOLS_LOADED_METADATA_KEY: loaded,
                TOOL_GROUP_RULES_METADATA_KEY: [group for group, _ in rules],
                "matches": [tool.name for tool in hits],
                "load_via": "search",
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


def _allowlist(raw: Any) -> frozenset[str] | None:
    """A child's declared tool set, or ``None`` when the run declared none."""
    if isinstance(raw, frozenset | set | tuple | list):
        names = frozenset(name for name in raw if isinstance(name, str))
        return names or None
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
    "GROUP_PREFIX",
    "SELECT_PREFIX",
    "TOOL_SEARCH_TOOL_NAME",
    "ToolSearchInput",
    "ToolSearchTool",
]
