"""Bringing an older snapshot forward, one version at a time.

A snapshot outlives the build that wrote it. It sits in the host's hot store
while pods are replaced under it, so the payload a resuming process picks up is
routinely one shape behind the code reading it. Refusing every one of those
would mean a rollout strands every run that was mid-turn when it started.

The chain is what makes an older payload readable without making the restore
code carry a memory of every shape the snapshot has ever had: each step reads
one version and produces the next, the restore reads only the newest, and a
version nothing knows how to lift is still a refusal.
"""
from __future__ import annotations

from typing import Any

import pytest

from protocore.contracts.snapshot import (
    SNAPSHOT_SCHEMA_KEY,
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotSchemaError,
    UpcasterRegistry,
    migrate_snapshot,
)


def _at(version: int, **fields: Any) -> dict[str, Any]:
    return {SNAPSHOT_SCHEMA_KEY: version, **fields}


def test_a_payload_with_no_version_is_lifted_as_version_one() -> None:
    """The shape the version field was added on top of, and the only shape a
    store can hold without one. Refusing it would strand every run that was
    paused when the field shipped, and a paused run has no later build that
    would accept the payload instead."""
    registry = UpcasterRegistry(current_version=2)
    registry.register(1, lambda snapshot: {**snapshot, "lanes": []})

    assert registry.upgrade({"run_id": "run-1"}) == _at(2, run_id="run-1", lanes=[])


def test_one_step_is_applied() -> None:
    registry = UpcasterRegistry(current_version=2)
    registry.register(1, lambda snapshot: {**snapshot, "lanes": []})

    upgraded = registry.upgrade(_at(1, run_id="run-1"))

    assert upgraded == _at(2, run_id="run-1", lanes=[])


def test_steps_run_in_order_up_to_the_current_version() -> None:
    """Each step sees what the one below it produced, not the original."""
    registry = UpcasterRegistry(current_version=4)
    registry.register(1, lambda snapshot: {**snapshot, "trace": [*snapshot["trace"], "1->2"]})
    registry.register(2, lambda snapshot: {**snapshot, "trace": [*snapshot["trace"], "2->3"]})
    registry.register(3, lambda snapshot: {**snapshot, "trace": [*snapshot["trace"], "3->4"]})

    upgraded = registry.upgrade(_at(1, trace=[]))

    assert upgraded["trace"] == ["1->2", "2->3", "3->4"]
    assert upgraded[SNAPSHOT_SCHEMA_KEY] == 4


def test_a_snapshot_already_at_the_current_version_runs_no_step() -> None:
    registry = UpcasterRegistry(current_version=2)
    registry.register(1, lambda snapshot: {**snapshot, "touched": True})

    upgraded = registry.upgrade(_at(2, run_id="run-1"))

    assert "touched" not in upgraded


def test_a_gap_in_the_chain_is_refused_rather_than_skipped() -> None:
    """Skipping a version leaves the fields it introduced unset, and an unset
    field is indistinguishable from one the run legitimately never had."""
    registry = UpcasterRegistry(current_version=3)
    registry.register(2, lambda snapshot: {**snapshot, "lanes": []})

    with pytest.raises(SnapshotSchemaError, match="from schema version 1 to 2"):
        registry.upgrade(_at(1))


def test_a_newer_payload_is_refused_by_the_chain_too() -> None:
    registry = UpcasterRegistry(current_version=2)

    with pytest.raises(SnapshotSchemaError, match="newer build"):
        registry.upgrade(_at(3))


def test_the_input_payload_is_left_alone() -> None:
    """A caller that refuses the upgraded copy still holds what it read."""
    registry = UpcasterRegistry(current_version=2)
    registry.register(1, lambda snapshot: {**snapshot, "lanes": []})
    original = _at(1, run_id="run-1")

    registry.upgrade(original)

    assert original == _at(1, run_id="run-1")


def test_a_step_cannot_be_registered_twice() -> None:
    """Two steps out of the same version is an ambiguity with no right answer."""
    registry = UpcasterRegistry(current_version=3)
    registry.register(1, lambda snapshot: snapshot)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(1, lambda snapshot: snapshot)


@pytest.mark.parametrize("from_version", [0, -1])
def test_a_step_below_version_one_is_rejected(from_version: int) -> None:
    registry = UpcasterRegistry(current_version=3)

    with pytest.raises(ValueError, match="at least 1"):
        registry.register(from_version, lambda snapshot: snapshot)


@pytest.mark.parametrize("from_version", [2, 3])
def test_a_step_that_would_overshoot_the_current_version_is_rejected(from_version: int) -> None:
    """A registry that upgrades to 2 has no use for a step producing 3 or 4 —
    accepting one would leave a chain that silently never runs it."""
    registry = UpcasterRegistry(current_version=2)

    with pytest.raises(ValueError, match="not below the current version"):
        registry.register(from_version, lambda snapshot: snapshot)


def test_migrate_snapshot_reads_the_builds_own_chain() -> None:
    """The production entry point. No registry passed, no version invented."""
    from protocore.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION

    assert migrate_snapshot(_at(SNAPSHOT_SCHEMA_VERSION))[SNAPSHOT_SCHEMA_KEY] == (
        SNAPSHOT_SCHEMA_VERSION
    )


def test_a_versionless_payload_reaches_the_current_version_with_nothing_invented() -> None:
    """The continuity keys version 2 states were never recorded before it, so
    the honest lift is the empty one rather than a permissive default."""
    upgraded = migrate_snapshot({"run_id": "run-1", "turn_count": 7})

    assert upgraded[SNAPSHOT_SCHEMA_KEY] == SNAPSHOT_SCHEMA_VERSION
    assert upgraded["turn_count"] == 7
    assert upgraded["compact_checkpoint"] is None
    assert upgraded["skill_catalog_block_sha256"] is None
    for key in (
        "active_rule_paths",
        "discovered_rules",
        "session_grants",
        "profile_audit",
        "spans",
        "discovered_tools",
        "deferred_tool_groups",
    ):
        assert upgraded[key] == []
    assert "context_manager_pinned_tools" not in upgraded


def test_a_version_six_pin_list_becomes_discovered_tools_in_the_same_order() -> None:
    """Version 6 knew one order, least recently pinned first. It is the only
    order there is, so it is both the discovery order and the recency; and
    version 6 held nothing back, so the lift says so rather than leaving the
    resumed run to decide afresh partway through."""
    upgraded = migrate_snapshot(
        {
            SNAPSHOT_SCHEMA_KEY: 6,
            "run_id": "run-1",
            "context_manager_pinned_tools": ["Grep", "Read", 3, "Write"],
        }
    )

    assert upgraded[SNAPSHOT_SCHEMA_KEY] == SNAPSHOT_SCHEMA_VERSION
    assert upgraded["discovered_tools"] == [
        {"name": "Grep", "last_used": 1},
        {"name": "Read", "last_used": 2},
        {"name": "Write", "last_used": 4},
    ]
    assert upgraded["deferred_tool_groups"] == []
    assert "context_manager_pinned_tools" not in upgraded
