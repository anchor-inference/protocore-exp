"""Tests for :class:`protocore.runtime.tool_registry.ToolRegistry`."""
from __future__ import annotations

import asyncio

import pytest

from protocore.contracts.tool_registry import IToolRegistry, ToolVisibilityPolicy
from protocore.runtime.tool_registry import ToolRegistry
from protocore.tests_support.adapters import InMemoryToolRegistry

from ._tool_fixtures import MockTool

# ----------------------------------------------------------------------
# register / get / unregister
# ----------------------------------------------------------------------


def test_register_and_get() -> None:
    reg = ToolRegistry()
    tool = MockTool(tool_name="Alpha", description="alpha desc")
    reg.register(tool)
    fetched = reg.get("Alpha")
    assert fetched is tool


def test_get_missing_returns_none() -> None:
    reg = ToolRegistry()
    assert reg.get("nope") is None


def test_register_is_idempotent_on_name() -> None:
    reg = ToolRegistry()
    first = MockTool(tool_name="Alpha", description="v1")
    second = MockTool(tool_name="Alpha", description="v2")
    reg.register(first)
    reg.register(second)
    assert reg.get("Alpha") is second


def test_unregister_idempotent() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="Alpha"))
    reg.unregister("Alpha")
    reg.unregister("Alpha")  # second call must not raise
    assert reg.get("Alpha") is None


def test_constructor_accepts_iterable() -> None:
    tools = [MockTool(tool_name=f"T{i}") for i in range(3)]
    reg = ToolRegistry(tools)
    assert len(reg) == 3
    for tool in tools:
        assert reg.get(tool.name) is tool


def test_membership_operator() -> None:
    reg = ToolRegistry([MockTool(tool_name="Alpha")])
    assert "Alpha" in reg
    assert "Beta" not in reg
    assert 42 not in reg  # type: ignore[operator]  # non-str non-membership


# ----------------------------------------------------------------------
# Sort invariant
# ----------------------------------------------------------------------


def test_list_all_sorted_by_name() -> None:
    reg = ToolRegistry()
    for name in ["Gamma", "Alpha", "Bravo"]:
        reg.register(MockTool(tool_name=name))
    names = [t.name for t in reg.list_all()]
    assert names == ["Alpha", "Bravo", "Gamma"]


def test_list_for_tenant_visible_set_filters() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo", "Gamma"]:
        reg.register(MockTool(tool_name=name))
    policy = ToolVisibilityPolicy(visible={"Alpha", "Gamma"})
    visible = [t.name for t in reg.list_for_tenant("tnt", policy)]
    assert visible == ["Alpha", "Gamma"]


def test_list_for_tenant_blocked_overrides_visible() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo", "Gamma"]:
        reg.register(MockTool(tool_name=name))
    policy = ToolVisibilityPolicy(
        visible={"Alpha", "Bravo", "Gamma"},
        blocked={"Bravo"},
    )
    visible = [t.name for t in reg.list_for_tenant("tnt", policy)]
    assert visible == ["Alpha", "Gamma"]


def test_list_for_tenant_empty_visible_means_all() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo"]:
        reg.register(MockTool(tool_name=name))
    policy = ToolVisibilityPolicy()  # empty visible == all
    assert {t.name for t in reg.list_for_tenant("tnt", policy)} == {"Alpha", "Bravo"}


# ----------------------------------------------------------------------
# filter_by_whitelist
# ----------------------------------------------------------------------


def test_filter_by_whitelist_resolves_known_names() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo", "Gamma"]:
        reg.register(MockTool(tool_name=name))
    resolved = reg.filter_by_whitelist(["Bravo", "Alpha"])
    assert [t.name for t in resolved] == ["Alpha", "Bravo"]  # sort invariant


def test_filter_by_whitelist_drops_unknown_silently() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="Alpha"))
    resolved = reg.filter_by_whitelist(["Alpha", "Missing", "AlsoMissing"])
    assert [t.name for t in resolved] == ["Alpha"]


def test_filter_by_whitelist_empty_returns_empty() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="Alpha"))
    assert reg.filter_by_whitelist([]) == []


# ----------------------------------------------------------------------
# search (BM25)
# ----------------------------------------------------------------------


def test_search_ranks_by_relevance() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="ReadFile", description="Read a file from workspace"))
    reg.register(MockTool(tool_name="WriteFile", description="Write text to a file"))
    reg.register(MockTool(tool_name="GitDiff", description="Show git diff of changes"))

    results = reg.search("read a file", top_k=2)
    names = [t.name for t in results]
    assert "ReadFile" in names


def test_search_empty_query_returns_first_k_by_name() -> None:
    reg = ToolRegistry()
    for name in ["Charlie", "Alpha", "Bravo"]:
        reg.register(MockTool(tool_name=name))
    results = reg.search("", top_k=2)
    assert [t.name for t in results] == ["Alpha", "Bravo"]


def test_search_with_whitelist_narrows_pool() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="ReadFile", description="Read a file"))
    reg.register(MockTool(tool_name="WriteFile", description="Write a file"))
    reg.register(MockTool(tool_name="GitDiff", description="Show git diff"))

    results = reg.search("file", top_k=5, whitelist=["ReadFile", "GitDiff"])
    assert {t.name for t in results} <= {"ReadFile", "GitDiff"}


def test_search_returns_empty_for_empty_corpus() -> None:
    reg = ToolRegistry()
    assert reg.search("anything", top_k=5) == []


def test_search_handles_russian_query() -> None:
    """BM25 corpus is multilingual — Cyrillic queries must score."""
    reg = ToolRegistry()
    reg.register(
        MockTool(tool_name="ReadRu", description="Прочитать файл из рабочей области")
    )
    reg.register(MockTool(tool_name="Bash", description="Run shell commands"))
    results = reg.search("прочитать файл", top_k=1)
    assert results
    assert results[0].name == "ReadRu"


def test_search_breaks_bm25_ties_by_name_asc_not_registration_order() -> None:
    """BM25 tie-breaking must be name-ASC (docstring invariant).

 The :meth:`ToolRegistry.search` docstring promises that ties within
 ranked groups are broken by ``name`` ASC. A stable sort by score alone
 would keep equal-score rows in registration order, so the tie-break
 has to be the name itself for the contract to hold across pods that
 registered the same tools in different orders.

 Regression test for with three tools sharing an identical
 description (hence identical BM25 score for any matching query),
 registration order is Z, A, B. With ``top_k=2`` the result must be
 the two name-ASC-first (``Alpha``, ``Bravo``), not the two
 registration-first (``Zebra``, ``Alpha``).
 """
    desc = "Read a file from workspace"
    reg = ToolRegistry()
    for name in ["Zebra", "Alpha", "Bravo"]:
        reg.register(MockTool(tool_name=name, description=desc))

    results = reg.search("read a file", top_k=2)
    assert [t.name for t in results] == ["Alpha", "Bravo"]


def test_search_deterministic_across_registration_orders() -> None:
    """Two pods with divergent registration orders must return the same result.

    Cross-pod determinism is the contract the search docstring's
    "ties broken by name ASC" rule exists to uphold: a discovery result
    must not depend on which order a pod happened to call ``register``
    in. With three tied tools and ``top_k=2``, the two pods below are
    guaranteed to differ under the bug (one returns ``[Alpha, Bravo]``,
    the other ``[Zebra, Bravo]``).
    """
    desc = "Read a file from workspace"
    reg_forward = ToolRegistry()
    for name in ["Alpha", "Bravo", "Zebra"]:
        reg_forward.register(MockTool(tool_name=name, description=desc))

    reg_reverse = ToolRegistry()
    for name in ["Zebra", "Bravo", "Alpha"]:
        reg_reverse.register(MockTool(tool_name=name, description=desc))

    names_forward = [t.name for t in reg_forward.search("read a file", top_k=2)]
    names_reverse = [t.name for t in reg_reverse.search("read a file", top_k=2)]
    assert names_forward == names_reverse
    # And the result is the name-ASC first two — the docstring's promise.
    assert names_forward == ["Alpha", "Bravo"]


# ----------------------------------------------------------------------
# compute_effective_surface (3-layer)
# ----------------------------------------------------------------------


def test_effective_surface_returns_all_when_under_topk() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo"]:
        reg.register(MockTool(tool_name=name))
    policy = ToolVisibilityPolicy()
    defs = reg.compute_effective_surface("tnt", policy, top_k=10)
    assert [d.name for d in defs] == ["Alpha", "Bravo"]


def test_effective_surface_no_top_k_returns_all_filtered() -> None:
    reg = ToolRegistry()
    for name in ["Alpha", "Bravo", "Gamma"]:
        reg.register(MockTool(tool_name=name))
    policy = ToolVisibilityPolicy(blocked={"Bravo"})
    defs = reg.compute_effective_surface("tnt", policy)
    assert [d.name for d in defs] == ["Alpha", "Gamma"]


def test_effective_surface_clips_at_top_k_with_retrieval() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="ReadFile", description="Read file from workspace"))
    reg.register(MockTool(tool_name="WriteFile", description="Write text to a file"))
    reg.register(MockTool(tool_name="GitDiff", description="Show git diff"))
    reg.register(MockTool(tool_name="Cat", description="Echo cat picture"))
    policy = ToolVisibilityPolicy()
    defs = reg.compute_effective_surface("tnt", policy, query="read a file", top_k=2)
    assert len(defs) <= 2
    names = [d.name for d in defs]
    # ReadFile must be in the top-2.
    assert "ReadFile" in names
    # Sort invariant: names are sorted ASC.
    assert names == sorted(names)


def test_effective_surface_pinned_always_included() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="Bash", description="Run shell commands"))
    reg.register(MockTool(tool_name="Read", description="Read a file"))
    reg.register(MockTool(tool_name="Write", description="Write a file"))
    reg.register(MockTool(tool_name="Grep", description="Search files"))
    policy = ToolVisibilityPolicy(pinned={"Bash"})
    defs = reg.compute_effective_surface(
        "tnt",
        policy,
        query="read a text file",  # NOT bash-y
        top_k=2,
    )
    names = [d.name for d in defs]
    # Bash MUST be included even though the query is about reading text.
    assert "Bash" in names


def test_effective_surface_blocked_takes_precedence() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="ReadFile", description="Read a file"))
    reg.register(MockTool(tool_name="WriteFile", description="Write a file"))
    policy = ToolVisibilityPolicy(blocked={"ReadFile"})
    defs = reg.compute_effective_surface("tnt", policy, top_k=5)
    assert [d.name for d in defs] == ["WriteFile"]


def test_effective_surface_search_hint_feeds_per_turn_clip_ru() -> None:
    """A tool's RU ``search_hint`` must surface it in the CLIPPED per-turn
    payload — not only via ToolSearch.

    The target tool has a purely-English name + description that score 0.0
    against a Cyrillic query. Its multilingual
    ``search_hint`` carries the RU keyword. With enough English filler tools
    to force the ``top_k`` clip, the hinted tool must still be advertised —
    proving ``compute_effective_surface`` passes ``search_hint`` into the
    BM25 candidate corpus, mirroring :meth:`ToolRegistry.search`.
    """
    reg = ToolRegistry()
    # Target: EN name/description (zero Cyrillic overlap), RU+EN hint.
    reg.register(
        MockTool(
            tool_name="WeatherForecast",
            description="Return the meteorological outlook",
            search_hint="погода прогноз weather forecast",
        )
    )
    # English filler that scores 0 against the RU query — competes for the
    # top_k slots so an unhinted target would be clipped out.
    for name in ["GitDiff", "RunShell", "ListFiles", "GrepText", "EditCode"]:
        reg.register(MockTool(tool_name=name, description=f"{name} helper"))

    policy = ToolVisibilityPolicy()
    defs = reg.compute_effective_surface(
        "tnt", policy, query="прогноз погоды", top_k=2
    )
    names = [d.name for d in defs]
    assert "WeatherForecast" in names, (
        "RU search_hint must surface the tool in the clipped surface; "
        f"got {names}"
    )


def test_effective_surface_without_hint_ru_query_clips_tool() -> None:
    """Control for the search_hint fix: an IDENTICAL tool WITHOUT a hint is
    clipped out by the same RU query + ``top_k`` — confirming the hint (not
    some incidental match) is what surfaces it in the test above.

    The two RU-described filler tools score positively against the query and
    occupy both ``top_k`` slots; the hint-less English target scores 0.0 and
    is genuinely clipped (filler wins), so the surface is non-empty and the
    target is absent — a real clip, not a degenerate empty result.
    """
    reg = ToolRegistry()
    reg.register(
        MockTool(
            tool_name="WeatherForecast",
            description="Return the meteorological outlook",
            # No search_hint → nothing in the corpus matches the RU query.
        )
    )
    # RU-described filler that DOES match the query → fills the top_k slots.
    reg.register(MockTool(tool_name="RuOne", description="прогноз на сегодня"))
    reg.register(MockTool(tool_name="RuTwo", description="погоды сводка дня"))
    for name in ["GitDiff", "RunShell", "EditCode"]:
        reg.register(MockTool(tool_name=name, description=f"{name} helper"))

    policy = ToolVisibilityPolicy()
    defs = reg.compute_effective_surface(
        "tnt", policy, query="прогноз погоды", top_k=2
    )
    names = [d.name for d in defs]
    assert "WeatherForecast" not in names
    # The surface IS populated by the RU-matching filler (real clip).
    assert set(names) <= {"RuOne", "RuTwo"}
    assert names, "control must produce a non-empty clipped surface"


# ----------------------------------------------------------------------
# Concurrency (thread/async-safety smoke test)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# L-2: Protocol conformance — IToolRegistry surface must match both impls
# ----------------------------------------------------------------------


_REQUIRED_PROTOCOL_METHODS: tuple[str, ...] = (
    "register",
    "unregister",
    "get",
    "list_all",
    "list_for_tenant",
    "filter_by_whitelist",
    "search",
    "compute_effective_surface",
    "declare_group",
    "undeclare_group",
    "tool_groups",
)


def test_protocol_advertises_all_required_methods() -> None:
    """IToolRegistry Protocol must declare every method concrete impls expose."""
    for method in _REQUIRED_PROTOCOL_METHODS:
        assert hasattr(IToolRegistry, method), (
            f"IToolRegistry Protocol missing required method {method!r}"
        )


def test_concrete_tool_registry_satisfies_protocol() -> None:
    """The in-core concrete :class:`ToolRegistry` matches the Protocol."""
    reg = ToolRegistry()
    assert isinstance(reg, IToolRegistry)
    for method in _REQUIRED_PROTOCOL_METHODS:
        assert hasattr(reg, method), f"ToolRegistry missing {method!r}"


def test_in_memory_tool_registry_satisfies_protocol() -> None:
    """The test fixture adapter matches the extended Protocol."""
    reg = InMemoryToolRegistry()
    assert isinstance(reg, IToolRegistry)
    for method in _REQUIRED_PROTOCOL_METHODS:
        assert hasattr(reg, method), f"InMemoryToolRegistry missing {method!r}"


@pytest.mark.asyncio
async def test_concurrent_register_and_lookup_safe() -> None:
    """Hammer register/get from many tasks; no corruption."""
    reg = ToolRegistry()

    async def register_many(prefix: str, count: int) -> None:
        for i in range(count):
            reg.register(MockTool(tool_name=f"{prefix}_{i}"))
            await asyncio.sleep(0)

    async def lookup_many(prefix: str, count: int) -> None:
        for i in range(count):
            reg.get(f"{prefix}_{i}")
            await asyncio.sleep(0)

    await asyncio.gather(
        register_many("A", 50),
        register_many("B", 50),
        lookup_many("A", 50),
        lookup_many("B", 50),
    )
    # Every A_* and B_* must be present.
    names = {t.name for t in reg.list_all()}
    assert all(f"A_{i}" in names for i in range(50))
    assert all(f"B_{i}" in names for i in range(50))
