"""A fresh tool result larger than the per-result limit is cut where it enters.

The per-result limit (``tool_result_truncation_ratio`` of the window) used to
be applied only by compaction, and only to results outside the protected tail.
The result a tool had just returned sits in that tail, so a single oversized
output went to the model whole — and one larger than the window ended the run
with ``llm_context_window_exceeded`` even after the reactive pass, which may not
touch the newest round.
"""
from __future__ import annotations

import pytest

from protocore.contracts.types import ToolResultBlock
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.token_counting import estimate_tokens

from .conftest import ScenarioFactory, ScriptedTool, default_rc

pytestmark = pytest.mark.asyncio

_BIG = "строка журнала сборки номер " * 11_000


def _results_in(request: object) -> list[ToolResultBlock]:
    return [
        block
        for message in request.messages  # type: ignore[attr-defined]
        for block in message.content_blocks
        if isinstance(block, ToolResultBlock)
    ]


async def test_a_result_larger_than_the_window_does_not_end_the_run(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(
        rc=default_rc(model_context_window=65_536),
        tools=[ScriptedTool(tool_name="BigRead", content=_BIG)],
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c1", tool_name="BigRead", tool_input={"v": "log"}
    )
    run.llm.queue_response(text="read the head, the rest is stored")

    await run.run("read the build log")

    assert run.engine.state.value == "completed"
    [result] = _results_in(run.requests[1])
    rc = run.engine.config.rc
    assert estimate_tokens(result.content, rc) <= derive_budgets(rc).tool_result_truncation_threshold
    assert result.canonical_ref is not None
    assert result.canonical_ref in result.content
    assert result.content.startswith(_BIG[:200])


async def test_the_whole_result_is_kept_behind_the_reference(
    scenario: ScenarioFactory,
) -> None:
    run = scenario(
        rc=default_rc(model_context_window=65_536),
        tools=[ScriptedTool(tool_name="BigRead", content=_BIG)],
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c1", tool_name="BigRead", tool_input={"v": "log"}
    )
    run.llm.queue_response(text="done")

    await run.run("read the build log")

    [result] = _results_in(run.requests[1])
    stored = await run.blobs.get(tenant_id=run.engine.config.tenant_id, ref=result.canonical_ref)
    assert stored.decode("utf-8") == _BIG


async def test_a_result_within_the_limit_reaches_the_model_untouched(
    scenario: ScenarioFactory,
) -> None:
    small = "короткий ответ инструмента"
    run = scenario(
        rc=default_rc(model_context_window=65_536),
        tools=[ScriptedTool(tool_name="SmallRead", content=small)],
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c1", tool_name="SmallRead", tool_input={"v": "x"}
    )
    run.llm.queue_response(text="done")

    await run.run("read it")

    [result] = _results_in(run.requests[1])
    assert result.content == small
    assert result.canonical_ref is None


_FAILING_TAIL = "\nerror: the build failed in the last step, exit status 2"


async def test_by_default_only_the_head_is_kept(scenario: ScenarioFactory) -> None:
    run = scenario(
        rc=default_rc(model_context_window=65_536),
        tools=[ScriptedTool(tool_name="BigRead", content=_BIG + _FAILING_TAIL)],
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c1", tool_name="BigRead", tool_input={"v": "log"}
    )
    run.llm.queue_response(text="done")

    await run.run("read the build log")

    [result] = _results_in(run.requests[1])
    assert _FAILING_TAIL.strip() not in result.content


async def test_a_tail_share_keeps_the_end_of_the_output_too(
    scenario: ScenarioFactory,
) -> None:
    """The line that says why a build failed is usually its last one."""
    run = scenario(
        rc=default_rc(
            model_context_window=65_536, tool_result_entry_cap_tail_share=0.25
        ),
        tools=[ScriptedTool(tool_name="BigRead", content=_BIG + _FAILING_TAIL)],
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c1", tool_name="BigRead", tool_input={"v": "log"}
    )
    run.llm.queue_response(text="done")

    await run.run("read the build log")

    [result] = _results_in(run.requests[1])
    rc = run.engine.config.rc
    assert estimate_tokens(result.content, rc) <= derive_budgets(rc).tool_result_truncation_threshold
    assert result.content.startswith(_BIG[:200])
    assert result.content.endswith(_FAILING_TAIL)
    assert result.canonical_ref is not None
    head, _, rest = result.content.partition("[Output cut where it entered the run")
    assert rest, "the omitted middle is marked between the head and the tail"
    assert result.canonical_ref in rest
    tail = rest.split("]", 1)[1]
    # Roughly the configured split: the tail is a quarter of what was kept.
    kept = len(head) + len(tail)
    assert 0.15 < len(tail) / kept < 0.35


async def test_the_cut_metadata_says_how_much_of_each_end_was_kept(
    scenario: ScenarioFactory,
) -> None:
    from protocore.runtime.tool_result_entry_cap import (
        TOOL_RESULT_CUT_ON_ENTRY_METADATA_KEY,
    )

    run = scenario(
        rc=default_rc(
            model_context_window=65_536, tool_result_entry_cap_tail_share=0.25
        ),
        tools=[ScriptedTool(tool_name="BigRead", content=_BIG + _FAILING_TAIL)],
    )
    run.llm.queue_tool_call_response(
        tool_call_id="c1", tool_name="BigRead", tool_input={"v": "log"}
    )
    run.llm.queue_response(text="done")

    await run.run("read the build log")

    [result] = run.tool_results()
    cut = result.metadata[TOOL_RESULT_CUT_ON_ENTRY_METADATA_KEY]
    assert cut["kept_tail_chars"] > 0
    assert cut["kept_chars"] > cut["kept_tail_chars"]
