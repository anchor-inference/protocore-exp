"""Every message the loop writes for itself names an origin that is not the conversation's.

A host that shows or stores a conversation used to hide the loop's own control
notes by matching their wording ("[internal control", "The run has reached its
budget"); a reworded template put them back in front of the reader. The origin
is derived from the marker each injection site already sets, so a host filters
on a value instead of a phrase.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import protocore.contracts.types as types_module
from protocore import MessageOrigin
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_SUMMARY_METADATA_KEY,
    ORIGIN_METADATA_KEY,
    SYNTHETIC_RECOVERY_BACKGROUND_WAKE,
    SYNTHETIC_RECOVERY_METADATA_KEY,
    SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE,
    SYNTHETIC_RECOVERY_TRUNCATION_CONTINUE,
    Message,
    MessageRole,
    TextBlock,
    synthetic_recovery_origin,
)
from protocore.runtime.soft_stop import SYNTHETIC_RECOVERY_SOFT_STOP
from protocore.runtime.soft_stop import _append_notice as append_soft_stop_notice

RUNTIME = Path(__file__).resolve().parents[3] / "protocore" / "runtime"


def _marked(value: object) -> Message:
    return Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="note")],
        metadata={SYNTHETIC_RECOVERY_METADATA_KEY: value},
    )


def _recovery_kinds() -> list[str]:
    """Every value the core defines for the synthetic-recovery marker."""
    kinds = [
        value
        for name, value in vars(types_module).items()
        if name.startswith("SYNTHETIC_RECOVERY_") and name != "SYNTHETIC_RECOVERY_METADATA_KEY"
    ]
    return [*kinds, SYNTHETIC_RECOVERY_SOFT_STOP]


@pytest.mark.parametrize("kind", _recovery_kinds())
def test_every_recovery_kind_the_core_defines_has_a_non_conversation_origin(kind: str) -> None:
    origin = _marked(kind).origin
    assert origin is not MessageOrigin.conversation
    assert origin is not MessageOrigin.compaction
    assert synthetic_recovery_origin(kind) is origin


def test_the_kinds_fall_into_the_documented_groups() -> None:
    assert _marked(SYNTHETIC_RECOVERY_SOFT_STOP).origin is MessageOrigin.wind_down
    assert _marked(SYNTHETIC_RECOVERY_BACKGROUND_WAKE).origin is MessageOrigin.background_wake
    assert _marked(SYNTHETIC_RECOVERY_POST_TOOL_EMPTY_NUDGE).origin is MessageOrigin.nudge
    assert _marked(SYNTHETIC_RECOVERY_TRUNCATION_CONTINUE).origin is MessageOrigin.recovery


def test_a_marker_the_core_does_not_know_still_reads_as_the_loops() -> None:
    # A truthy marker from an older build or a host's own kind is still
    # scaffolding; only its group is unknown.
    assert _marked(True).origin is MessageOrigin.recovery
    assert _marked("some_future_kind").origin is MessageOrigin.recovery
    # An empty marker marks nothing.
    assert _marked("").origin is MessageOrigin.conversation
    assert _marked(False).origin is MessageOrigin.conversation


def test_an_explicit_origin_is_honoured_and_a_bad_one_is_ignored() -> None:
    def tagged(value: object) -> Message:
        return Message(role=MessageRole.user, metadata={ORIGIN_METADATA_KEY: value})

    assert tagged("background_wake").origin is MessageOrigin.background_wake
    assert tagged(MessageOrigin.nudge).origin is MessageOrigin.nudge
    assert tagged("conversation").origin is MessageOrigin.conversation
    assert tagged("not-an-origin").origin is MessageOrigin.conversation
    assert tagged(3).origin is MessageOrigin.conversation


def test_compaction_outranks_every_other_tag() -> None:
    record = Message(
        role=MessageRole.user,
        metadata={
            COMPACTION_SUMMARY_METADATA_KEY: True,
            ORIGIN_METADATA_KEY: "nudge",
            SYNTHETIC_RECOVERY_METADATA_KEY: SYNTHETIC_RECOVERY_SOFT_STOP,
        },
    )
    assert record.origin is MessageOrigin.compaction


def test_the_origin_survives_a_dump_and_is_rederived() -> None:
    dumped = _marked(SYNTHETIC_RECOVERY_SOFT_STOP).model_dump(mode="json")
    assert dumped["origin"] == "wind_down"
    assert Message.model_validate(dumped).origin is MessageOrigin.wind_down


def test_the_wind_down_notice_the_loop_appends_is_a_wind_down(engine_factory) -> None:  # type: ignore[no-untyped-def]
    engine = engine_factory()
    append_soft_stop_notice(engine, cause_name="provider_error")
    assert engine.history, "the notice was not appended"
    assert engine.history[-1].origin is MessageOrigin.wind_down


async def test_the_post_tool_empty_nudge_pair_is_a_nudge(engine_factory, in_memory_runtime) -> None:  # type: ignore[no-untyped-def]
    from tests.unit.runtime._tool_fixtures import MockTool

    rc = LoopConstants(model_context_window=8_192, empty_completion_guard_enabled=True)
    engine = engine_factory(rc=rc)
    in_memory_runtime["tools"].register(MockTool(tool_name="Look", description="look", response_content="seen"))
    llm = in_memory_runtime["llm"]
    llm.queue_tool_call_response(tool_call_id="t1", tool_name="Look", tool_input={})
    llm.queue_response(text="")
    llm.queue_response(text="done")

    async for _ in engine.run(Message(role=MessageRole.user, content_blocks=[TextBlock(text="look")])):
        pass

    nudges = [m for m in engine.history if m.metadata.get(SYNTHETIC_RECOVERY_METADATA_KEY)]
    assert nudges, "the run never nudged"
    assert all(m.origin is MessageOrigin.nudge for m in nudges)
    assert engine.history[0].origin is MessageOrigin.conversation


#: Message constructions in the runtime that carry no marker on purpose, with
#: the reason. A new injection site that is not marked fails the scan below
#: until it is either marked or listed here with its reason.
_UNMARKED_ON_PURPOSE: dict[tuple[str, str], str] = {
    ("compact_checkpoint.py", "apply_checkpoint"): "a placeholder replaced two lines later by the tagged summary",
    ("context/compaction.py", "_run_summariser"): "the summariser's own request, never put into history",
    ("context/session_compaction.py", "call"): "a session summariser's own request, never put into history",
    ("context/session_memory.py", "bound_catchup_source"): "a gap marker inside a summariser's source text",
    ("context/session_memory.py", "_summary_message"): "a cross-run seed, tagged by _tag_seeded_reference",
    ("context/session_memory.py", "_ledger_message"): "a cross-run seed, tagged by _tag_seeded_reference",
    ("loop_strategies.py", "_fetch_plan_fallback"): "a one-off structured request, never put into history",
    ("loop_strategies.py", "_append_plan_turn"): "the model's own plan, recorded as its turn",
}


def _enclosing_function(tree: ast.AST, target: ast.AST) -> str:
    best = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.lineno <= target.lineno <= (node.end_lineno or node.lineno):
                best = node.name  # innermost wins: ast.walk visits outer first
    return best


def test_every_user_or_assistant_message_the_runtime_builds_is_marked_or_listed() -> None:
    unlisted: list[str] = []
    for path in sorted(RUNTIME.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(RUNTIME).as_posix()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Message"):
                continue
            keywords = {k.arg: k.value for k in node.keywords}
            role = ast.unparse(keywords["role"]) if "role" in keywords else ""
            if role not in {"MessageRole.user", "MessageRole.assistant"}:
                continue
            if "metadata" in keywords:
                continue
            where = (relative, _enclosing_function(tree, node))
            if where not in _UNMARKED_ON_PURPOSE:
                unlisted.append(f"{relative}:{node.lineno} in {where[1]}")
    assert not unlisted, "unmarked injection sites: " + ", ".join(unlisted)


def test_the_queued_message_is_the_callers() -> None:
    from protocore.runtime.live_control import QueuedPrompt, placed_as_user_message

    placed = placed_as_user_message([QueuedPrompt(id="q1", kind="steer", text="and also this")])
    assert placed is not None
    assert placed.origin is MessageOrigin.conversation
