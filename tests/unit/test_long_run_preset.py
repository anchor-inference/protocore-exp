"""``LoopConstants.long_run()``: the loop guard tuned for runs of hundreds of turns.

The defaults (three identical calls, one nudge, three failures of one tool)
were tuned for short runs; a long one legitimately re-runs the same test
command and re-reads the same file many times. The preset names the values a
long-running host settled on instead of changing the defaults under everyone.
"""
from __future__ import annotations

from protocore.contracts.runtime_constants import LONG_RUN_LOOP_GUARD, LoopConstants


def test_the_preset_turns_the_loop_guard_on_with_long_run_limits() -> None:
    rc = LoopConstants.long_run()
    assert rc.loop_guard_enabled is True
    assert rc.loop_guard_identical_tool_limit == 30
    assert rc.loop_guard_nudge_max == 3
    assert rc.max_consecutive_tool_errors == 8


def test_the_preset_leaves_every_other_field_at_its_default() -> None:
    preset = LoopConstants.long_run().model_dump()
    default = LoopConstants().model_dump()
    changed = {name for name in default if preset[name] != default[name]}
    assert changed == set(LONG_RUN_LOOP_GUARD)


def test_overrides_win_over_the_preset() -> None:
    rc = LoopConstants.long_run(model_context_window=200_000, loop_guard_nudge_max=5)
    assert rc.model_context_window == 200_000
    assert rc.loop_guard_nudge_max == 5
    assert rc.loop_guard_identical_tool_limit == 30


def test_the_defaults_are_unchanged() -> None:
    rc = LoopConstants()
    assert rc.loop_guard_enabled is False
    assert rc.loop_guard_identical_tool_limit == 3
    assert rc.loop_guard_nudge_max == 1
    assert rc.max_consecutive_tool_errors == 3
