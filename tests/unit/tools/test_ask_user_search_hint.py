# ruff: noqa: RUF001, RUF002 — Russian queries are the point of these tests.
"""The core's AskUser can be found by what a person says when they want to be asked.

AskUser is always loaded into the surface, but ToolSearch still ranks it, and
its description shares no word with "clarify with me" or a Russian "уточни у
меня". Every other core tool carries a ``search_hint`` for that; a host had to
subclass this one only to give it one.
"""
from __future__ import annotations

import pytest

from protocore.runtime.tool_registry import ToolRegistry, tool_document
from protocore.tools.ask_user import ASK_USER_TOOL_NAME, AskUserTool

from ..runtime._tool_fixtures import MockTool


def test_ask_user_carries_a_search_hint() -> None:
    hint = tool_document(AskUserTool()).search_hint
    assert hint.strip()


def test_the_hint_stays_off_the_wire_description() -> None:
    """The hint is for finding the tool, never for the model to read."""
    tool = AskUserTool()
    assert tool.search_hint not in tool.definition.description


@pytest.mark.parametrize("query", ["clarify with me", "уточни у меня", "спроси меня какой вариант выбрать"])
def test_tool_search_finds_ask_user_by_the_words_people_use(query: str) -> None:
    registry = ToolRegistry()
    registry.register(AskUserTool())
    for name in ("GitDiff", "RunShell", "ListFiles", "GrepText", "EditCode"):
        registry.register(MockTool(tool_name=name, description=f"{name} helper"))

    found = [tool.name for tool in registry.search(query, top_k=1)]

    assert found == [ASK_USER_TOOL_NAME]
