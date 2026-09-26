"""Run-mode + thinking + reasoning_effort contracts.

Covers the contracts the sibling units (the host transport / run
persistence, chat) code against:

* :class:`QueryEngineConfig` gains ``run_mode`` (``direct``/``deep``),
  ``thinking_enabled``, ``reasoning_effort`` with safe defaults + pattern
  validation (the config is a frozen dataclass, so validation lives in
  ``__post_init__``).
* :class:`EventType` gains ``REASONING_STEP = "reasoning_step"`` (the SGR plan
  event, distinct from native CoT which keeps flowing as ``CONTENT_BLOCK_DELTA``
  thinking deltas).
* The runtime thinking/effort knobs land on ``LLMRequest.extra`` so
  a host's vLLM adapter can translate them to
  ``chat_template_kwargs.enable_thinking`` + top-level ``reasoning_effort``.

Measured against a vLLM endpoint:
``deep`` pairs native CoT with ``reasoning_effort=low`` so CoT stays bounded
("Qwen thinking eats all tokens" otherwise).
"""
from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.types import Message, MessageRole, TextBlock
from protocore.runtime.events.types import EventType
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)


def _build_engine(
    *,
    rc: LoopConstants | None = None,
    run_mode: str = "direct",
    thinking_enabled: bool = False,
    reasoning_effort: str = "low",
    llm: InMemoryLLMProvider | None = None,
) -> QueryEngine:
    return QueryEngine(
        config=QueryEngineConfig(
            run_id="run-test",
            tenant_id="tenant-test",
            session_id="sess-test",
            model_name="qwen3.6-35b-a3b",
            rc=rc or LoopConstants(model_context_window=4_096),
            run_mode=run_mode,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
        ),
        llm_provider=llm or InMemoryLLMProvider(),
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )


# ---------------------------------------------------------------------------
# Step 1 — config + event contracts
# ---------------------------------------------------------------------------


def test_config_run_mode_defaults_to_direct_non_thinking() -> None:
    """§A.2 — a default-constructed config is Direct, non-thinking, effort=low."""
    config = QueryEngineConfig(
        run_id="r",
        tenant_id="t",
        session_id="s",
        model_name="m",
    )
    assert config.run_mode == "direct"
    assert config.thinking_enabled is False
    assert config.reasoning_effort == "low"


def test_config_accepts_deep_thinking() -> None:
    config = QueryEngineConfig(
        run_id="r",
        tenant_id="t",
        session_id="s",
        model_name="m",
        run_mode="deep",
        thinking_enabled=True,
        reasoning_effort="medium",
    )
    assert config.run_mode == "deep"
    assert config.thinking_enabled is True
    assert config.reasoning_effort == "medium"


def test_engine_effective_tool_policy_applies_rc_floor_without_manual_policy() -> None:
    """The engine applies the RC core floor itself.

    A default ``QueryEngineConfig`` carries ``forced_pinned=frozenset()``. The
    core engine must merge ``rc.tool_surface_forced_pins`` into the per-turn
    policy on the live surface path so the floor is universal — NOT dependent on
    every external caller copying the tuple into ``ToolVisibilityPolicy``. This
    asserts the seam (``effective_tool_policy``) without manually constructing
    a forced_pinned policy in the test.
    """
    engine = _build_engine()  # default RC, default (empty) policy
    assert engine.config.tool_visibility_policy.forced_pinned == frozenset()
    floor = engine.effective_tool_policy.forced_pinned
    assert {"Agent", "Read", "Write", "Edit", "Bash", "Glob", "Grep"} <= floor


def test_engine_effective_tool_policy_unions_explicit_forced_pins() -> None:
    """An explicit policy ``forced_pinned`` is unioned with the RC floor, not lost."""
    policy = ToolVisibilityPolicy(forced_pinned=frozenset({"CustomTool"}))
    engine = QueryEngine(
        config=QueryEngineConfig(
            run_id="r",
            tenant_id="t",
            session_id="s",
            model_name="m",
            tool_visibility_policy=policy,
        ),
        llm_provider=InMemoryLLMProvider(),
        tool_registry=InMemoryToolRegistry(),
        event_stream=InMemoryEventStream(),
        hook_manager=InMemoryHookManager(),
        skill_store=InMemorySkillStore(),
        blob_store=InMemoryBlobStore(),
    )
    floor = engine.effective_tool_policy.forced_pinned
    assert "CustomTool" in floor
    assert {"Agent", "Read", "Write", "Edit", "Bash", "Glob", "Grep"} <= floor


def test_engine_effective_tool_policy_unions_toolsearch_dynamic_pins() -> None:
    """Discovered tools enter the policy's pins, so dispatch admits them."""
    engine = _build_engine()
    engine.context_manager.discover_tool("LibrarySearch")

    assert "LibrarySearch" in engine.effective_tool_policy.pinned


def test_tool_surface_advertised_event_type_contract() -> None:
    assert EventType.TOOL_SURFACE_ADVERTISED.value == "tool_surface_advertised"


def test_config_rejects_unknown_run_mode() -> None:
    """The frozen dataclass validates ``run_mode`` against ^(direct|deep)$."""
    with pytest.raises(ValueError, match="run_mode"):
        QueryEngineConfig(
            run_id="r",
            tenant_id="t",
            session_id="s",
            model_name="m",
            run_mode="turbo",
        )


def test_config_accepts_xhigh_reasoning_effort() -> None:
    config = QueryEngineConfig(
        run_id="r",
        tenant_id="t",
        session_id="s",
        model_name="m",
        thinking_enabled=True,
        reasoning_effort="xhigh",
    )
    assert config.reasoning_effort == "xhigh"


def test_config_rejects_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        QueryEngineConfig(
            run_id="r",
            tenant_id="t",
            session_id="s",
            model_name="m",
            reasoning_effort="ultra",
        )


def test_event_type_has_reasoning_step() -> None:
    """§A.4 — the typed SGR-plan event exists with the wire string."""
    assert EventType.REASONING_STEP == "reasoning_step"
    assert EventType.REASONING_STEP.value == "reasoning_step"


def test_runtime_constants_carry_run_mode_defaults() -> None:
    """§A.1 — per-tenant run-mode defaults are RC fields (dashboard-tunable)."""
    rc = LoopConstants(model_context_window=4_096)
    assert rc.agent_deep_plan_include_summary is False


def test_runtime_constants_reject_bad_default_mode() -> None:
    with pytest.raises(ValueError):
        LoopConstants(model_context_window=4_096, agent_loop_default_mode="turbo")


# ---------------------------------------------------------------------------
# Step 2 — thinking / effort thread into LLMRequest.extra
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_and_effort_land_in_llm_request_extra() -> None:
    """Direct-Thinking mode sets ``enable_thinking``/``reasoning_effort`` on the
    request the provider receives — the host adapter reads these keys."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done", stop_reason=__import__(
        "protocore.contracts.types", fromlist=["StopReason"]
    ).StopReason.end_turn)
    engine = _build_engine(
        run_mode="direct",
        thinking_enabled=True,
        reasoning_effort="low",
        llm=llm,
    )
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="напиши лендинг для кофейни")],
    )
    async for _ in engine.run(initial):
        pass

    assert llm.calls, "the loop must have called the LLM at least once"
    extra = dict(llm.calls[0].extra)
    assert extra.get("enable_thinking") is True
    assert extra.get("reasoning_effort") == "low"


@pytest.mark.asyncio
async def test_thinking_disabled_by_default_in_request_extra() -> None:
    """Direct non-thinking — ``enable_thinking`` is False on the request."""
    llm = InMemoryLLMProvider()
    llm.queue_response(text="done")
    engine = _build_engine(run_mode="direct", thinking_enabled=False, llm=llm)
    initial = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="write a landing")],
    )
    async for _ in engine.run(initial):
        pass

    assert llm.calls
    extra = dict(llm.calls[0].extra)
    assert extra.get("enable_thinking") is False
    assert extra.get("reasoning_effort") == "low"
