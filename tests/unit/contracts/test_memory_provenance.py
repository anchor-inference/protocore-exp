"""Who wrote a memory, recorded beside it and handed back with it.

A recalled fact that cannot be traced to the run, the session and the call
that wrote it cannot be weighed, corrected at its source, or told apart from
one an import brought in. The provenance travels as a typed value on the
record, written only by a store that says it can keep it, so a store written
before the field existed keeps working unchanged.
"""
from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any

import pytest

from protocore.contracts.evidence import RunTreeOrigin, ToolEvidenceContext
from protocore.contracts.memory import (
    IMemory,
    IProvenanceMemory,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryWriteDecision,
)
from protocore.contracts.run_state import RunScopedState
from protocore.contracts.tools import ToolContext
from protocore.tests_support.adapters import InMemoryMemory
from protocore.tools.memory import RememberTool, build_memory_tools

_TENANT = "tenant-1"
_SESSION = "session-1"


def _ctx(metadata: dict[str, Any] | None = None, **fields: Any) -> ToolContext:
    return ToolContext(
        tenant_id=_TENANT,
        run_id=fields.pop("run_id", "run-1"),
        session_id=_SESSION,
        metadata=metadata or {},
        **fields,
    )


def test_a_record_written_before_provenance_existed_still_loads() -> None:
    now = datetime.now(UTC)
    stored = {
        "id": "mem-1",
        "tenant_id": _TENANT,
        "scope": "session",
        "scope_key": _SESSION,
        "text": "the deploy key lives in the vault",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    record = MemoryRecord.model_validate(stored)
    assert record.provenance is None


def test_an_unknown_source_is_said_to_be_unknown_rather_than_guessed() -> None:
    provenance = MemoryProvenance()
    assert provenance.source_kind == "unknown"
    # An import without a timestamp must not acquire a freshness it never had.
    assert provenance.observed_at is None
    assert provenance.run_id is None


def test_provenance_round_trips_through_a_dump() -> None:
    provenance = MemoryProvenance(
        source_kind="agent",
        run_id="run-9",
        root_run_id="run-1",
        session_id=_SESSION,
        subagent_id="child-2",
        message_id="message-4",
        tool_call_id="call-7",
        observed_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert MemoryProvenance.model_validate(provenance.model_dump(mode="json")) == provenance


def test_the_provenance_store_is_a_memory_store_with_one_more_keyword() -> None:
    assert isinstance(InMemoryMemory(), IProvenanceMemory)
    assert "provenance" in inspect.signature(IProvenanceMemory.write).parameters
    # The base contract is untouched, so a store written against it still conforms.
    assert "provenance" not in inspect.signature(IMemory.write).parameters


async def test_the_reference_store_keeps_provenance_on_create() -> None:
    store = InMemoryMemory()
    provenance = MemoryProvenance(source_kind="agent", run_id="run-1", tool_call_id="call-1")
    result = await store.write(
        _TENANT, MemoryScope.session, _SESSION, "the api lives on port 9000", provenance=provenance
    )
    assert result.decision is MemoryWriteDecision.created
    assert result.record.provenance == provenance
    fetched = await store.get(_TENANT, result.record.id)
    assert fetched.provenance == provenance


async def test_provenance_alone_does_not_turn_a_skip_into_a_merge() -> None:
    store = InMemoryMemory()
    first = MemoryProvenance(source_kind="agent", run_id="run-1", tool_call_id="call-1")
    second = MemoryProvenance(source_kind="agent", run_id="run-2", tool_call_id="call-9")
    await store.write(_TENANT, MemoryScope.session, _SESSION, "the api lives on port 9000", provenance=first)
    again = await store.write(
        _TENANT, MemoryScope.session, _SESSION, "the api lives on port 9000", provenance=second
    )
    assert again.decision is MemoryWriteDecision.skipped
    assert again.record.version == 1
    assert again.record.provenance == first


async def test_a_merge_records_the_write_that_changed_the_record() -> None:
    store = InMemoryMemory()
    first = MemoryProvenance(source_kind="agent", run_id="run-1")
    second = MemoryProvenance(source_kind="operator", run_id="run-2")
    await store.write(_TENANT, MemoryScope.session, _SESSION, "the api lives on port 9000", provenance=first)
    merged = await store.write(
        _TENANT,
        MemoryScope.session,
        _SESSION,
        "the api lives on port 9000",
        source_refs=["file:README"],
        provenance=second,
    )
    assert merged.decision is MemoryWriteDecision.merged
    assert merged.record.provenance == second


async def test_a_merge_without_provenance_keeps_what_the_record_had() -> None:
    store = InMemoryMemory()
    first = MemoryProvenance(source_kind="agent", run_id="run-1")
    await store.write(_TENANT, MemoryScope.session, _SESSION, "the api lives on port 9000", provenance=first)
    merged = await store.write(
        _TENANT, MemoryScope.session, _SESSION, "the api lives on port 9000", source_refs=["file:README"]
    )
    assert merged.decision is MemoryWriteDecision.merged
    assert merged.record.provenance == first


async def test_remember_records_nothing_unless_asked() -> None:
    store = InMemoryMemory()
    remember = RememberTool(store)
    await remember.invoke(_ctx({"tool_call_id": "call-1"}), {"text": "the api lives on port 9000"})
    (record,) = await store.list(_TENANT)
    assert record.provenance is None


async def test_remember_records_who_wrote_the_memory_when_asked() -> None:
    store = InMemoryMemory()
    remember = RememberTool(store, record_provenance=True)
    before = datetime.now(UTC)
    result = await remember.invoke(
        _ctx({"tool_call_id": "call-1"}, run_id="run-2", run_state=RunScopedState(root_run_id="run-1")),
        {"text": "the api lives on port 9000"},
    )
    assert not result.is_error
    (record,) = await store.list(_TENANT)
    provenance = record.provenance
    assert provenance is not None
    assert provenance.source_kind == "agent"
    assert provenance.run_id == "run-2"
    assert provenance.root_run_id == "run-1"
    assert provenance.session_id == _SESSION
    assert provenance.tool_call_id == "call-1"
    assert provenance.observed_at is not None and provenance.observed_at >= before
    assert result.metadata["provenance"]["tool_call_id"] == "call-1"


async def test_remember_takes_the_run_tree_from_the_evidence_origin() -> None:
    store = InMemoryMemory()
    remember = RememberTool(store, record_provenance=True)
    origin = RunTreeOrigin(
        run_id="run-3", root_run_id="run-1", depth=1, parent_run_id="run-1", subagent_id="child-1"
    )
    await remember.invoke(
        _ctx(run_id="run-3", evidence=ToolEvidenceContext(origin=origin)),
        {"text": "the api lives on port 9000"},
    )
    (record,) = await store.list(_TENANT)
    assert record.provenance is not None
    assert record.provenance.root_run_id == "run-1"
    assert record.provenance.subagent_id == "child-1"


async def test_remember_names_the_source_kind_it_was_built_with() -> None:
    store = InMemoryMemory()
    remember = RememberTool(store, record_provenance=True, provenance_source="import")
    await remember.invoke(_ctx(), {"text": "the api lives on port 9000"})
    (record,) = await store.list(_TENANT)
    assert record.provenance is not None and record.provenance.source_kind == "import"


class _StoreWithoutProvenance(InMemoryMemory):
    """A store written against the base contract: it has never heard of provenance."""

    async def write(  # type: ignore[override]
        self,
        tenant_id: str,
        scope: MemoryScope,
        scope_key: str,
        text: str,
        *,
        kind: str = "fact",
        metadata: dict[str, Any] | None = None,
        source_refs: Any = None,
        salience: float = 0.0,
        similarity_threshold: float | None = None,
        max_records_per_scope: int | None = None,
        expected_version: int | None = None,
    ) -> Any:
        return await super().write(
            tenant_id,
            scope,
            scope_key,
            text,
            kind=kind,
            metadata=metadata,
            source_refs=source_refs,
            salience=salience,
            similarity_threshold=similarity_threshold,
            max_records_per_scope=max_records_per_scope,
            expected_version=expected_version,
        )


async def test_a_store_without_provenance_keeps_working_with_the_default_tools() -> None:
    store = _StoreWithoutProvenance()
    tools = {tool.name: tool for tool in build_memory_tools(store)}
    result = await tools["Remember"].invoke(_ctx(), {"text": "the api lives on port 9000"})
    assert not result.is_error


def test_asking_for_provenance_from_a_store_that_cannot_keep_it_fails_at_wiring() -> None:
    with pytest.raises(TypeError, match="provenance"):
        RememberTool(_StoreWithoutProvenance(), record_provenance=True)


def test_build_memory_tools_passes_the_provenance_choice_through() -> None:
    tools = {tool.name: tool for tool in build_memory_tools(InMemoryMemory(), record_provenance=True)}
    remember = tools["Remember"]
    assert isinstance(remember, RememberTool)
    assert remember.records_provenance
