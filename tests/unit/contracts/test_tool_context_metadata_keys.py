"""The metadata bag is a host's, and this is the whole of what core takes from it.

``ToolContext.metadata`` is an opaque compartment: a host fills it, and core
reaches in for the handful of values it has said it reads. That arrangement
only holds while the handful is written down. Without a list, any module could
start reading a string a host happened to spell, and the host would find out it
had been supplying an input on the day it stopped supplying it.

So the list is
:data:`~protocore.contracts.tools.CORE_TOOL_CONTEXT_METADATA_KEYS`, the reads go
through :func:`~protocore.contracts.tools.read_metadata`, which refuses an
undeclared key, and this file keeps the arrangement honest from both ends: the
declared set is pinned here, every constant naming one of those keys is checked
against the set, and no module may reach into the bag behind the accessors'
back.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from protocore.contracts import tools
from protocore.contracts.tool_registry import (
    ADVERTISED_TOOLS_METADATA_KEY,
    TOOL_ALLOWLIST_METADATA_KEY,
    TOOL_GROUP_RULES_GIVEN_METADATA_KEY,
    TOOL_VISIBILITY_POLICY_METADATA_KEY,
)
from protocore.contracts.tools import (
    CORE_STAMPED_TOOL_CONTEXT_METADATA_KEYS,
    CORE_TOOL_CONTEXT_METADATA_KEYS,
    SUBAGENT_DISPATCH_GROUP_METADATA_KEY,
    SUBAGENT_DISPATCH_ORDER_METADATA_KEY,
    SUBAGENT_TREE_PERMIT_METADATA_KEY,
    ToolContext,
    copy_metadata,
    has_metadata,
    read_metadata,
)
from protocore.contracts.types import SYNTHETIC_RECOVERY_METADATA_KEY
from protocore.tools.memory import (
    MEMORY_ALLOWED_SCOPES_CONTEXT_KEY,
    MEMORY_ENABLED_CONTEXT_KEY,
    MEMORY_MAX_RECORDS_PER_SCOPE_CONTEXT_KEY,
    MEMORY_SCOPE_CONTEXT_KEY,
    MEMORY_SCOPE_KEY_CONTEXT_KEY,
    MEMORY_SCOPE_KEYS_CONTEXT_KEY,
    MEMORY_WRITE_SIMILARITY_THRESHOLD_CONTEXT_KEY,
    TOOL_CALL_ID_CONTEXT_KEY,
)

CORE_ROOT = Path(tools.__file__).resolve().parents[1]

#: What core reads off the bag, spelled out here rather than imported, so that
#: adding a key is two deliberate edits in two places and not one.
EXPECTED_READ = frozenset(
    {
        "tool_call_id",
        "memory_default_scope",
        "memory_default_scope_key",
        "memory_allowed_scopes",
        "memory_scope_keys",
        "memory_enabled",
        "memory_write_similarity_threshold",
        "memory_max_records_per_scope",
        "protocore.tool_visibility_policy",
        "protocore.tool_allowlist",
        "protocore.advertised_tools",
        "protocore.tool_group_rules_given",
    }
)

#: What core stamps on the bag for a tool or a host to read back.
EXPECTED_STAMPED = frozenset(
    {
        "tool_call_id",
        "protocore.tool_visibility_policy",
        "protocore.tool_allowlist",
        "protocore.advertised_tools",
        "protocore.tool_group_rules_given",
        "protocore.subagent_dispatch_order",
        "protocore.subagent_dispatch_group",
        "protocore.subagent_tree_permit",
        "protocore.synthetic_recovery",
    }
)

#: The constants that name a bag key, each paired with the direction it is
#: declared in. A constant whose value drifts from the set, or a new one that
#: never reaches either set, fails here.
DECLARED_CONSTANTS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("TOOL_CALL_ID_CONTEXT_KEY", TOOL_CALL_ID_CONTEXT_KEY, EXPECTED_READ),
    ("MEMORY_SCOPE_CONTEXT_KEY", MEMORY_SCOPE_CONTEXT_KEY, EXPECTED_READ),
    ("MEMORY_SCOPE_KEY_CONTEXT_KEY", MEMORY_SCOPE_KEY_CONTEXT_KEY, EXPECTED_READ),
    ("MEMORY_ALLOWED_SCOPES_CONTEXT_KEY", MEMORY_ALLOWED_SCOPES_CONTEXT_KEY, EXPECTED_READ),
    ("MEMORY_SCOPE_KEYS_CONTEXT_KEY", MEMORY_SCOPE_KEYS_CONTEXT_KEY, EXPECTED_READ),
    ("MEMORY_ENABLED_CONTEXT_KEY", MEMORY_ENABLED_CONTEXT_KEY, EXPECTED_READ),
    (
        "MEMORY_WRITE_SIMILARITY_THRESHOLD_CONTEXT_KEY",
        MEMORY_WRITE_SIMILARITY_THRESHOLD_CONTEXT_KEY,
        EXPECTED_READ,
    ),
    (
        "MEMORY_MAX_RECORDS_PER_SCOPE_CONTEXT_KEY",
        MEMORY_MAX_RECORDS_PER_SCOPE_CONTEXT_KEY,
        EXPECTED_READ,
    ),
    (
        "TOOL_VISIBILITY_POLICY_METADATA_KEY",
        TOOL_VISIBILITY_POLICY_METADATA_KEY,
        EXPECTED_STAMPED,
    ),
    ("TOOL_ALLOWLIST_METADATA_KEY", TOOL_ALLOWLIST_METADATA_KEY, EXPECTED_STAMPED),
    ("ADVERTISED_TOOLS_METADATA_KEY", ADVERTISED_TOOLS_METADATA_KEY, EXPECTED_STAMPED),
    (
        "TOOL_GROUP_RULES_GIVEN_METADATA_KEY",
        TOOL_GROUP_RULES_GIVEN_METADATA_KEY,
        EXPECTED_STAMPED,
    ),
    (
        "SUBAGENT_DISPATCH_ORDER_METADATA_KEY",
        SUBAGENT_DISPATCH_ORDER_METADATA_KEY,
        EXPECTED_STAMPED,
    ),
    (
        "SUBAGENT_DISPATCH_GROUP_METADATA_KEY",
        SUBAGENT_DISPATCH_GROUP_METADATA_KEY,
        EXPECTED_STAMPED,
    ),
    (
        "SUBAGENT_TREE_PERMIT_METADATA_KEY",
        SUBAGENT_TREE_PERMIT_METADATA_KEY,
        EXPECTED_STAMPED,
    ),
    (
        "SYNTHETIC_RECOVERY_METADATA_KEY",
        SYNTHETIC_RECOVERY_METADATA_KEY,
        EXPECTED_STAMPED,
    ),
)

#: The module allowed to touch ``ToolContext.metadata`` directly: the one that
#: publishes the accessors.
ACCESSOR_MODULE = "contracts/tools.py"


def _context(metadata: dict[str, object] | None = None) -> ToolContext:
    return ToolContext(
        tenant_id="a-scope",
        run_id="a-run",
        session_id="a-session",
        metadata=dict(metadata or {}),
    )


def test_the_declared_sets_are_exactly_these() -> None:
    assert CORE_TOOL_CONTEXT_METADATA_KEYS == EXPECTED_READ
    assert CORE_STAMPED_TOOL_CONTEXT_METADATA_KEYS == EXPECTED_STAMPED


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    DECLARED_CONSTANTS,
    ids=[name for name, _, _ in DECLARED_CONSTANTS],
)
def test_every_constant_that_names_a_bag_key_is_declared(
    name: str, value: str, expected: frozenset[str]
) -> None:
    assert value in expected, f"{name} names a bag key nothing declares"


def test_reading_an_undeclared_key_is_refused() -> None:
    """The point of the list: a key not on it cannot be read at all.

    Not "returns nothing" — refused. A silent ``None`` for an undeclared key is
    how an undocumented channel starts: the caller writes the read, the host
    later spells the key, and the two are bound to each other with nothing
    stating it.
    """
    context = _context({"a_host_key": "a value"})

    with pytest.raises(KeyError, match="a_host_key"):
        read_metadata(context, "a_host_key")
    with pytest.raises(KeyError, match="a_host_key"):
        has_metadata(context, "a_host_key")


def test_a_declared_key_reads_back_what_the_host_put_there() -> None:
    context = _context({"memory_enabled": False, "tool_call_id": "call-1"})

    assert read_metadata(context, "memory_enabled") is False
    assert read_metadata(context, "tool_call_id") == "call-1"
    assert read_metadata(context, "memory_scope_keys") is None
    assert read_metadata(context, "memory_scope_keys", {}) == {}


def test_an_absent_key_is_told_apart_from_one_the_host_set_to_none() -> None:
    """"No opinion" and "no" are different answers, and guards turn on which."""
    context = _context({"memory_enabled": None})

    assert has_metadata(context, "memory_enabled") is True
    assert has_metadata(context, "memory_default_scope") is False


def test_the_whole_bag_copies_without_being_read() -> None:
    """Carrying a host's keys through is not a read and needs no declaration."""
    context = _context({"a_host_key": "a value", "tool_call_id": "call-1"})

    carried = copy_metadata(context)
    carried["tool_call_id"] = "call-2"

    assert carried == {"a_host_key": "a value", "tool_call_id": "call-2"}
    assert context.metadata["tool_call_id"] == "call-1"


def _metadata_accesses(path: Path) -> list[int]:
    """Lines in ``path`` reaching a tool context's metadata bag directly."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "metadata"
        and isinstance(node.value, ast.Name)
        and node.value.id in {"ctx", "context"}
    ]


def test_nothing_reaches_into_the_bag_behind_the_accessors() -> None:
    """The refusal above is only a guard while every read goes through it.

    One module reads the bag directly — the one that publishes the accessors.
    Anywhere else, a direct ``ctx.metadata`` is a read of a key nobody declared,
    and it would be invisible: the bag is typed ``dict[str, Any]``, so no type
    check anywhere would have anything to say about it.
    """
    offenders = {
        str(path.relative_to(CORE_ROOT)): lines
        for path in sorted(CORE_ROOT.rglob("*.py"))
        if (lines := _metadata_accesses(path))
        and str(path.relative_to(CORE_ROOT)) != ACCESSOR_MODULE
    }

    assert offenders == {}, (
        "these read the metadata bag directly instead of through "
        f"read_metadata / has_metadata / copy_metadata: {offenders}"
    )
