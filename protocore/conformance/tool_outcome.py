"""The conformance suite for the tool outcome store.

Its own module, with behavioural cases, because the shape of this contract
says almost nothing about whether an adapter keeps its promise. A store with
every method present and asynchronous can still let a late result overwrite
the one a run already placed, let a duplicated start make a known outcome
look unknown again, or answer one run's resume with another run's records —
and each of those is found only after a crash, which is the one moment the
store exists for. So the rules the core relies on are asked here, against the
host's own adapter, in the host's own test run.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from protocore.conformance.suite import ContractSuite
from protocore.contracts.tool_outcome import IToolOutcomeStore, ToolInvocationRecord


def _record(run: str, **overrides: Any) -> ToolInvocationRecord:
    values: dict[str, Any] = {
        "invocation_id": "inv-1",
        "tenant_id": "tenant-a",
        "run_id": run,
        "tool_call_id": "call-1",
        "tool_name": "Send",
        "arguments_digest": "digest-1",
        "step": 3,
    }
    values.update(overrides)
    return ToolInvocationRecord(**values)


def _outcome(text: str) -> dict[str, Any]:
    return {"kind": "tool_result", "tool_call_id": "call-1", "content": text}


class ToolOutcomeStoreConformance(ContractSuite):
    """Where a run makes each tool call's outcome durable the moment it exists."""

    protocol = IToolOutcomeStore

    @staticmethod
    def _run() -> str:
        """A run id no other case writes under.

        A host may hand every case the same store, so each case keeps to runs
        of its own rather than relying on an empty store.
        """
        return f"run-{uuid4().hex}"

    async def test_a_started_call_is_listed_with_no_outcome(self, subject: Any) -> None:
        run = self._run()
        await subject.record_started(_record(run))
        [held] = list(await subject.list_run("tenant-a", run))
        assert held.state == "started"
        assert held.outcome is None
        assert held.tool_call_id == "call-1"
        assert held.arguments_digest == "digest-1"

    async def test_a_finished_call_keeps_its_outcome_exactly(self, subject: Any) -> None:
        """The outcome is placed in history as stored, so it must come back whole."""
        run = self._run()
        await subject.record_started(_record(run))
        accepted = await subject.record_finished(
            _record(run, state="finished", outcome=_outcome("sent"))
        )
        assert accepted is True
        [held] = list(await subject.list_run("tenant-a", run))
        assert held.state == "finished"
        assert held.outcome == _outcome("sent")
        assert held.step == 3

    async def test_a_call_can_finish_without_having_started(self, subject: Any) -> None:
        """A call whose repeat is harmless is only recorded when it returns."""
        run = self._run()
        assert await subject.record_finished(
            _record(run, state="finished", outcome=_outcome("seen"), repeat_safe=True)
        )
        [held] = list(await subject.list_run("tenant-a", run))
        assert held.outcome == _outcome("seen")
        assert held.repeat_safe is True

    async def test_the_first_outcome_wins(self, subject: Any) -> None:
        """A duplicated callback or a late result after a cancellation is refused."""
        run = self._run()
        await subject.record_finished(_record(run, state="finished", outcome=_outcome("first")))
        accepted = await subject.record_finished(
            _record(run, state="finished", outcome=_outcome("late"))
        )
        assert accepted is False
        [held] = list(await subject.list_run("tenant-a", run))
        assert held.outcome == _outcome("first")

    async def test_a_late_start_never_unsettles_a_finished_call(self, subject: Any) -> None:
        run = self._run()
        await subject.record_finished(_record(run, state="finished", outcome=_outcome("done")))
        await subject.record_started(_record(run))
        [held] = list(await subject.list_run("tenant-a", run))
        assert held.state == "finished"
        assert held.outcome == _outcome("done")

    async def test_records_are_kept_apart_by_run_and_tenant(self, subject: Any) -> None:
        run = self._run()
        other = self._run()
        await subject.record_started(_record(run))
        await subject.record_started(_record(other))
        await subject.record_started(_record(run, tenant_id="tenant-b"))
        assert len(list(await subject.list_run("tenant-a", run))) == 1
        assert len(list(await subject.list_run("tenant-a", other))) == 1
        assert len(list(await subject.list_run("tenant-b", run))) == 1
        assert list(await subject.list_run("tenant-a", self._run())) == []

    async def test_the_invocation_not_the_call_id_is_the_identity(self, subject: Any) -> None:
        """A provider may reuse a call id; two invocations stay two records."""
        run = self._run()
        await subject.record_finished(
            _record(run, state="finished", outcome=_outcome("one"))
        )
        await subject.record_finished(
            _record(run, invocation_id="inv-2", state="finished", outcome=_outcome("two"))
        )
        held = list(await subject.list_run("tenant-a", run))
        assert sorted(item.invocation_id for item in held) == ["inv-1", "inv-2"]

    async def test_discard_forgets_only_what_it_names(self, subject: Any) -> None:
        run = self._run()
        other = self._run()
        await subject.record_started(_record(run))
        await subject.record_started(_record(run, invocation_id="inv-2"))
        await subject.discard("tenant-a", run, ["inv-1", "never-written"])
        held = list(await subject.list_run("tenant-a", run))
        assert [item.invocation_id for item in held] == ["inv-2"]
        # Another run's record by the same id is not touched.
        await subject.record_started(_record(other))
        await subject.discard("tenant-a", run, ["inv-1"])
        assert len(list(await subject.list_run("tenant-a", other))) == 1


__all__ = ["ToolOutcomeStoreConformance"]
