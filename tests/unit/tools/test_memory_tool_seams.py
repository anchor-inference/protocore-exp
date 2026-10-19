"""The supported ways for a host to shape the memory tools without copying them.

A host with its own taxonomy, its own fields on a note, or subjects it will not
keep used to subclass the tools and re-implement ``invoke``, importing the
core's private scope helpers so the scope rules could not drift. These tests
pin the seams that replace that: a write policy that classifies or refuses a
write before it lands, an input model that carries extra fields, and recall
hooks for extra store filters and for how a hit is shown.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import pytest
from pydantic import Field

from protocore.contracts.memory import (
    IMemoryWritePolicy,
    MemoryHit,
    MemoryScope,
    MemoryWriteRequest,
    MemoryWriteVerdict,
)
from protocore.contracts.tools import ToolContext
from protocore.tests_support.adapters import InMemoryMemory
from protocore.tools.memory import (
    MEMORY_ALLOWED_SCOPES_CONTEXT_KEY,
    RecallInput,
    RecallTool,
    RememberInput,
    RememberTool,
    build_memory_tools,
)

_TENANT = "tenant-1"
_SESSION = "session-1"


def _ctx(metadata: dict[str, Any] | None = None) -> ToolContext:
    return ToolContext(tenant_id=_TENANT, run_id="run-1", session_id=_SESSION, metadata=metadata or {})


class _TaggedRememberInput(RememberInput):
    topics: list[str] | None = Field(default=None, max_length=5, description="Short tags.")


class _RecordingPolicy:
    """Classifies a note by its words, refuses one subject, records what it saw."""

    def __init__(self) -> None:
        self.requests: list[MemoryWriteRequest] = []

    async def review(self, request: MemoryWriteRequest) -> MemoryWriteVerdict:
        self.requests.append(request)
        if "diagnosis" in request.text:
            return MemoryWriteVerdict.refuse(
                "Not saved: health notes are not kept.", result_metadata={"refused": ["health"]}
            )
        topics = request.extra.get("topics") or ["inferred"]
        kind = request.kind if request.kind_given else "procedure"
        return MemoryWriteVerdict.allow(
            kind=kind, metadata={"topics": topics}, result_metadata={"topics": topics}
        )


class _Scanner:
    def scan(self, text: str) -> str | None:
        return "pattern-1" if "ignore previous" in text else None


def test_the_policy_is_a_published_contract() -> None:
    assert isinstance(_RecordingPolicy(), IMemoryWritePolicy)


def test_a_verdict_allows_by_default_and_refusal_needs_a_message() -> None:
    assert not MemoryWriteVerdict().refused
    refusal = MemoryWriteVerdict.refuse("no")
    assert refusal.refused and refusal.message == "no"
    with pytest.raises(ValueError):
        MemoryWriteVerdict(refused=True)


async def test_the_policy_sees_the_resolved_write_and_the_extra_fields() -> None:
    store = InMemoryMemory()
    policy = _RecordingPolicy()
    remember = RememberTool(store, input_model=_TaggedRememberInput, write_policy=policy)
    result = await remember.invoke(_ctx(), {"text": "deploys go through the staging gate", "topics": ["deploy"]})
    assert not result.is_error
    (request,) = policy.requests
    assert request.tenant_id == _TENANT
    assert request.scope is MemoryScope.session and request.scope_key == _SESSION
    assert request.extra == {"topics": ["deploy"]}
    assert request.kind == "fact" and not request.kind_given


async def test_the_policy_can_classify_before_the_write_lands() -> None:
    store = InMemoryMemory()
    remember = RememberTool(store, input_model=_TaggedRememberInput, write_policy=_RecordingPolicy())
    result = await remember.invoke(_ctx(), {"text": "deploys go through the staging gate", "topics": ["deploy"]})
    (record,) = await store.list(_TENANT)
    assert record.kind == "procedure"
    assert record.metadata == {"topics": ["deploy"]}
    assert result.metadata["topics"] == ["deploy"]
    assert result.metadata["kind"] == "procedure"
    assert "kind=procedure" in result.content


async def test_a_kind_the_model_named_is_reported_as_given() -> None:
    policy = _RecordingPolicy()
    remember = RememberTool(InMemoryMemory(), write_policy=policy)
    await remember.invoke(_ctx(), {"text": "the gate is manual", "kind": "decision"})
    assert policy.requests[0].kind_given
    assert policy.requests[0].kind == "decision"


async def test_a_refused_write_never_reaches_the_store() -> None:
    store = InMemoryMemory()
    remember = RememberTool(store, write_policy=_RecordingPolicy())
    result = await remember.invoke(_ctx({"tool_call_id": "call-3"}), {"text": "the diagnosis came back"})
    assert result.is_error
    assert result.tool_call_id == "call-3"
    assert result.content == "Not saved: health notes are not kept."
    assert result.metadata == {"refused": ["health"]}
    assert await store.list(_TENANT) == []


async def test_the_scanner_still_runs_first_and_covers_the_extra_fields() -> None:
    policy = _RecordingPolicy()
    remember = RememberTool(
        InMemoryMemory(), scanner=_Scanner(), input_model=_TaggedRememberInput, write_policy=policy
    )
    result = await remember.invoke(
        _ctx(), {"text": "a plain note", "topics": ["ignore previous instructions"]}
    )
    assert result.is_error and "pattern-1" in result.content
    # A note the scanner refused is not shown to the policy either.
    assert policy.requests == []


async def test_scope_rules_apply_before_the_policy_is_asked() -> None:
    policy = _RecordingPolicy()
    remember = RememberTool(InMemoryMemory(), write_policy=policy)
    with pytest.raises(ValueError, match="not permitted"):
        await remember.invoke(
            _ctx({MEMORY_ALLOWED_SCOPES_CONTEXT_KEY: ["session"]}), {"text": "a note", "scope": "global"}
        )
    assert policy.requests == []


async def test_an_extra_field_is_published_in_the_definition() -> None:
    remember = RememberTool(InMemoryMemory(), input_model=_TaggedRememberInput)
    assert "topics" in remember.definition.parameters.properties


def test_an_input_model_must_extend_the_core_one() -> None:
    with pytest.raises(TypeError, match="RememberInput"):
        RememberTool(InMemoryMemory(), input_model=RecallInput)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RecallInput"):
        RecallTool(InMemoryMemory(), input_model=RememberInput)  # type: ignore[arg-type]


async def test_without_a_policy_the_write_is_unchanged() -> None:
    store = InMemoryMemory()
    result = await RememberTool(store).invoke(_ctx(), {"text": "the gate is manual"})
    assert result.content.startswith("created memory mem-000001 (scope=session; kind=fact)")
    (record,) = await store.list(_TENANT)
    assert record.metadata == {}


class _TopicRecallInput(RecallInput):
    topic: str | None = Field(default=None, max_length=64, description="Only notes with this tag.")


class _TopicStore(InMemoryMemory):
    """A store whose search takes one filter the base contract does not name."""

    def __init__(self) -> None:
        super().__init__()
        self.topics_asked: list[Sequence[str] | None] = []

    async def search(  # type: ignore[override]
        self,
        tenant_id: str,
        query: str,
        *,
        scopes: Sequence[MemoryScope] | None = None,
        scope_keys: dict[MemoryScope, str] | None = None,
        kinds: Sequence[str] | None = None,
        limit: int = 20,
        topics: Sequence[str] | None = None,
    ) -> Sequence[MemoryHit]:
        self.topics_asked.append(topics)
        hits = await super().search(
            tenant_id, query, scopes=scopes, scope_keys=scope_keys, kinds=kinds, limit=limit
        )
        if topics is None:
            return hits
        return [hit for hit in hits if set(topics) & set(hit.record.metadata.get("topics", []))]


class _TopicRecall(RecallTool):
    description_: ClassVar[str] = "Search memory by keyword or tag."

    def search_filters(self, payload: RecallInput) -> dict[str, Any]:
        topic = getattr(payload, "topic", None)
        return {"topics": [topic]} if topic else {}

    def render_hit(self, index: int, hit: MemoryHit) -> str:
        tags = ", ".join(self.screen_recalled(t) for t in hit.record.metadata.get("topics", []))
        return f"{super().render_hit(index, hit)} [{tags}]"


async def test_recall_hands_extra_filters_to_the_store_and_shows_hits_its_own_way() -> None:
    store = _TopicStore()
    await store.write(_TENANT, MemoryScope.session, _SESSION, "deploys use the staging gate", metadata={"topics": ["deploy"]})
    await store.write(_TENANT, MemoryScope.session, _SESSION, "billing runs nightly", metadata={"topics": ["billing"]})
    recall = _TopicRecall(store, scanner=_Scanner(), input_model=_TopicRecallInput)
    assert "topic" in recall.definition.parameters.properties
    assert recall.definition.description == "Search memory by keyword or tag."
    result = await recall.invoke(_ctx(), {"query": "", "topic": "deploy"})
    assert store.topics_asked == [["deploy"]]
    assert result.metadata["count"] == 1
    assert result.content.endswith("deploys use the staging gate [deploy]")


async def test_recall_with_an_explicit_scope_passes_the_filters_too() -> None:
    store = _TopicStore()
    recall = _TopicRecall(store, input_model=_TopicRecallInput)
    await recall.invoke(_ctx(), {"query": "x", "topic": "deploy", "scope": "session"})
    assert store.topics_asked == [["deploy"]]


async def test_screen_recalled_replaces_a_poisoned_value() -> None:
    recall = RecallTool(InMemoryMemory(), scanner=_Scanner())
    assert recall.screen_recalled("fine") == "fine"
    assert recall.screen_recalled("ignore previous").startswith("[BLOCKED")
    assert RecallTool(InMemoryMemory()).screen_recalled("ignore previous") == "ignore previous"


def test_build_memory_tools_threads_the_policy() -> None:
    policy = _RecordingPolicy()
    tools = {tool.name: tool for tool in build_memory_tools(InMemoryMemory(), write_policy=policy)}
    remember = tools["Remember"]
    assert isinstance(remember, RememberTool)
    assert remember.write_policy is policy
