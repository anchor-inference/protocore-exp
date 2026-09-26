"""In-memory adapter implementations of all 12 interfaces.

Each mock implements the same Protocol / ABC the real adapter targets,
keeping tests honest about contract compliance. Per
.

All mocks are deliberately simple — they exist to wire up integration
tests, not benchmark performance. Programmable surfaces (e.g.
``InMemoryLLMProvider.queue_response``) let tests script behavior.
"""
from __future__ import annotations

import asyncio
import builtins
import hashlib
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from contextlib import suppress
from datetime import UTC as _UTC
from datetime import datetime
from typing import Any, cast

from protocore.contracts.agent_dispatch import (
    IAgentDispatch,
    SubagentHandle,
    SubagentNotFoundError,
)
from protocore.contracts.background import (
    BACKGROUND_TERMINAL_STATUSES,
    AgentRef,
    StartWork,
    TaskRecord,
    TerminalObserver,
    WorkHandle,
    WorkSpec,
)
from protocore.contracts.blob import BlobNotFoundError, IBlobStore
from protocore.contracts.events import IEventStream
from protocore.contracts.hooks import (
    HookActionKind,
    HookResult,
    HookSpec,
    IHookManager,
)
from protocore.contracts.llm import (
    ILLMProvider,
    LLMRequest,
    LLMResponse,
    LLMResponseUsage,
    LLMStreamEvent,
)
from protocore.contracts.memory import (
    DEFAULT_RECALL_SCOPES,
    IMemory,
    MemoryConflictError,
    MemoryHit,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryScope,
    MemoryWriteDecision,
    MemoryWriteResult,
)
from protocore.contracts.observability import (
    IRequestManifestSink,
    ManifestValue,
    RequestManifest,
    request_digest,
)
from protocore.contracts.run import IRunStore, RunNotFoundError
from protocore.contracts.search import Hit, IndexDoc, ISearchIndex
from protocore.contracts.session import ISessionStore, SessionNotFoundError
from protocore.contracts.skills import (
    SKILL_ENTRY_MIME_TYPE,
    SKILL_ENTRY_PATH,
    ISkillStore,
    SkillBundle,
    SkillFileRef,
    SkillIndexEntry,
    SkillUpsertInput,
)
from protocore.contracts.todo import ITodoStorage
from protocore.contracts.tool_registry import (
    IToolRegistry,
    ToolVisibilityPolicy,
    policy_admits,
)
from protocore.contracts.tool_retrieval import RetrievalSettings
from protocore.contracts.tools import Tool
from protocore.contracts.types import (
    BlobMetadata,
    Event,
    HookEvent,
    Message,
    MessageRole,
    Run,
    RunStatus,
    Session,
    SkillManifest,
    StopReason,
    SubagentDef,
    SubagentResult,
    SubagentTask,
    TextBlock,
    Todo,
    ToolDefinition,
)
from protocore.contracts.workspace import (
    DEFAULT_WORKSPACE_LIFECYCLE,
    IWorkspace,
    WorkspaceConflictError,
    WorkspaceHit,
    WorkspaceLifecycle,
    WorkspaceNotFoundError,
    WorkspacePathError,
    WorkspaceQuotaExceededError,
    WorkspaceScope,
    WorkspaceUnit,
    WorkspaceWriteDecision,
    WorkspaceWriteOutcome,
)

# ---------------------------------------------------------------------------
# Scripted LLM provider
# ---------------------------------------------------------------------------


class InMemoryLLMProvider(ILLMProvider):
    """Scripted LLM responses for tests.

    Queue responses via :meth:`queue_response`; each call to
    :meth:`stream_with_tools` pops the next queued response and emits a
    minimal event sequence ending in ``message_stop``.

    Use :meth:`queue_tool_call_response` to script a tool-use turn (one
    or more ``tool_use_*`` deltas + ``message_stop`` with
    ``stop_reason=tool_use``).
    """

    def __init__(self) -> None:
        self._queue: deque[LLMResponse] = deque()
        # Each entry is a list of LLMStreamEvent dicts to emit verbatim.
        self._scripted_streams: deque[list[LLMStreamEvent]] = deque()
        self._default_tool_call_stream: list[LLMStreamEvent] | None = None
        self._calls: list[LLMRequest] = []

    @property
    def calls(self) -> Sequence[LLMRequest]:
        """Read-only view of all calls received (for assertions)."""
        return tuple(self._calls)

    def queue_response(
        self,
        *,
        text: str = "",
        stop_reason: StopReason = StopReason.end_turn,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Enqueue a scripted response."""
        message = Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text=text)] if text else [],
        )
        self._queue.append(
            LLMResponse(
                message=message,
                stop_reason=stop_reason,
                usage=LLMResponseUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )
        )

    def queue_tool_call_response(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        text_prefix: str = "",
        usage_input_tokens: int = 0,
    ) -> None:
        """Enqueue a scripted tool-use stream.

        Emits ``message_start`` → optional text block → ``tool_use_start``
        → ``tool_use_input_delta`` → ``tool_use_stop`` → optional ``usage``
        → ``message_stop`` with ``stop_reason=tool_use``.

        ``usage_input_tokens`` (when > 0) injects a ``usage`` stream event
        carrying the provider-reported prompt size, so tests can drive the
        real observed-prompt-tokens path (the compaction gate floor).
        """
        import json as _json
        args = tool_input or {}
        args_json = _json.dumps(args)
        stream: list[LLMStreamEvent] = [
            LLMStreamEvent(name="message_start", payload={}),
        ]
        if text_prefix:
            stream.extend(
                [
                    LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
                    LLMStreamEvent(
                        name="content_block_delta",
                        payload={"text": text_prefix, "kind": "text"},
                    ),
                    LLMStreamEvent(name="content_block_stop", payload={}),
                ]
            )
        stream.extend(
            [
                LLMStreamEvent(
                    name="tool_use_start",
                    payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
                ),
                LLMStreamEvent(
                    name="tool_use_input_delta",
                    payload={
                        "tool_call_id": tool_call_id,
                        "partial_input_json": args_json,
                    },
                ),
                LLMStreamEvent(
                    name="tool_use_stop",
                    payload={"tool_call_id": tool_call_id, "final_input": args},
                ),
            ]
        )
        if usage_input_tokens > 0:
            stream.append(
                LLMStreamEvent(
                    name="usage",
                    payload={"input_tokens": usage_input_tokens},
                )
            )
        stream.append(
            LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.tool_use.value},
            )
        )
        self._scripted_streams.append(stream)

    def queue_scripted_stream(self, events: Sequence[LLMStreamEvent]) -> None:
        """Enqueue one raw provider stream, emitted verbatim on the next call.

        The escape hatch behind the shaped helpers above: a scenario that
        needs a turn no helper spells — several ``tool_use`` blocks in one
        assistant message, reasoning deltas with no text, a stream that
        stops mid-block — scripts the events itself instead of reaching
        into this double's queue.
        """
        self._scripted_streams.append(list(events))

    def queue_multi_tool_call_response(
        self,
        *,
        tool_calls: Sequence[tuple[str, str, dict[str, Any]]],
        text_prefix: str = "",
    ) -> None:
        """Enqueue ONE assistant turn emitting several ``tool_use`` blocks.

        ``tool_calls`` is ``(tool_call_id, tool_name, arguments)`` in the
        order the model asks for them — the order the run must preserve in
        history whatever order the tools finish in.
        """
        import json as _json

        stream: list[LLMStreamEvent] = [
            LLMStreamEvent(name="message_start", payload={}),
        ]
        if text_prefix:
            stream.extend(
                [
                    LLMStreamEvent(name="content_block_start", payload={"kind": "text"}),
                    LLMStreamEvent(
                        name="content_block_delta",
                        payload={"text": text_prefix, "kind": "text"},
                    ),
                    LLMStreamEvent(name="content_block_stop", payload={}),
                ]
            )
        for tool_call_id, tool_name, args in tool_calls:
            stream.extend(
                [
                    LLMStreamEvent(
                        name="tool_use_start",
                        payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
                    ),
                    LLMStreamEvent(
                        name="tool_use_input_delta",
                        payload={
                            "tool_call_id": tool_call_id,
                            "partial_input_json": _json.dumps(args),
                        },
                    ),
                    LLMStreamEvent(
                        name="tool_use_stop",
                        payload={"tool_call_id": tool_call_id, "final_input": args},
                    ),
                ]
            )
        stream.append(
            LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.tool_use.value},
            )
        )
        self._scripted_streams.append(stream)

    def set_default_tool_call(
        self,
        *,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
        tool_call_id_prefix: str = "toolu_loop",
    ) -> None:
        """Configure a default tool_use stream returned ad infinitum.

        Used to test endless-tool-call termination guards. Each call to
        :meth:`stream_with_tools` allocates a fresh ``tool_call_id`` based
        on the call counter to avoid id collisions.
        """
        self._default_tool_name = tool_name
        self._default_tool_input = tool_input or {}
        self._default_tool_call_id_prefix = tool_call_id_prefix
        # Marker: a non-empty list signals the toggle is on.
        self._default_tool_call_stream = [
            LLMStreamEvent(name="message_start", payload={})
        ]

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self._calls.append(request)
        # 1. Scripted tool-call streams take priority.
        if self._scripted_streams:
            scripted = self._scripted_streams.popleft()
            for evt in scripted:
                yield evt
            return

        # 2. Default tool_use stream (endless mode).
        if self._default_tool_call_stream:
            import json as _json
            call_idx = len(self._calls)
            tool_call_id = f"{self._default_tool_call_id_prefix}_{call_idx}"
            tool_name = self._default_tool_name
            args = self._default_tool_input
            args_json = _json.dumps(args)
            yield LLMStreamEvent(name="message_start", payload={})
            yield LLMStreamEvent(
                name="tool_use_start",
                payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
            )
            yield LLMStreamEvent(
                name="tool_use_input_delta",
                payload={
                    "tool_call_id": tool_call_id,
                    "partial_input_json": args_json,
                },
            )
            yield LLMStreamEvent(
                name="tool_use_stop",
                payload={"tool_call_id": tool_call_id, "final_input": args},
            )
            yield LLMStreamEvent(
                name="message_stop",
                payload={"stop_reason": StopReason.tool_use.value},
            )
            return

        # 3. Fallback text response.
        if not self._queue:
            response = LLMResponse(
                message=Message(role=MessageRole.assistant, content_blocks=[]),
                stop_reason=StopReason.end_turn,
            )
        else:
            response = self._queue.popleft()

        yield LLMStreamEvent(name="message_start", payload={})
        for block in response.message.content_blocks:
            if isinstance(block, TextBlock):
                yield LLMStreamEvent(name="content_block_start", payload={"kind": "text"})
                yield LLMStreamEvent(
                    name="content_block_delta",
                    payload={"text": block.text},
                )
                yield LLMStreamEvent(name="content_block_stop", payload={})
        yield LLMStreamEvent(
            name="message_stop",
            payload={"stop_reason": response.stop_reason.value},
        )

    async def complete_structured(
        self,
        request: LLMRequest,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        self._calls.append(request)
        if self._queue:
            return self._queue.popleft()
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        self._calls.append(request)
        if self._queue:
            return self._queue.popleft()
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        # 1 token per ~4 chars heuristic — fixture-only.
        return max(1, len(text) // 4) if text else 0


# ---------------------------------------------------------------------------
# In-memory BlobStore
# ---------------------------------------------------------------------------


class InMemoryBlobStore(IBlobStore):
    """``dict[(tenant_id, ref), bytes]`` backed."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], tuple[bytes, BlobMetadata]] = {}
        self._lock = asyncio.Lock()

    async def put(
        self,
        tenant_id: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> BlobMetadata:
        sha = hashlib.sha256(content).hexdigest()
        ref = f"{tenant_id}/{sha}"
        async with self._lock:
            md = BlobMetadata(
                ref=ref,
                tenant_id=tenant_id,
                content_type=content_type,
                size_bytes=len(content),
                sha256=sha,
                metadata=metadata or {},
            )
            self._store[(tenant_id, ref)] = (content, md)
        return md

    async def get(self, tenant_id: str, ref: str) -> bytes:
        try:
            content, _ = self._store[(tenant_id, ref)]
        except KeyError as e:
            raise BlobNotFoundError(ref) from e
        return content

    async def get_stream(self, tenant_id: str, ref: str) -> AsyncIterator[bytes]:
        content = await self.get(tenant_id, ref)
        # Chunked yield for stream-compat fidelity.
        chunk_size = 64 * 1024
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    async def head(self, tenant_id: str, ref: str) -> BlobMetadata:
        try:
            _, md = self._store[(tenant_id, ref)]
        except KeyError as e:
            raise BlobNotFoundError(ref) from e
        return md

    async def exists(self, tenant_id: str, ref: str) -> bool:
        return (tenant_id, ref) in self._store

    async def delete(self, tenant_id: str, ref: str) -> bool:
        return self._store.pop((tenant_id, ref), None) is not None

    async def list_prefix(
        self,
        tenant_id: str,
        prefix: str,
        *,
        limit: int = 1000,
    ) -> list[BlobMetadata]:
        results = [
            md
            for (t, ref), (_, md) in self._store.items()
            if t == tenant_id and ref.startswith(prefix)
        ]
        return results[:limit]


# ---------------------------------------------------------------------------
# In-memory SearchIndex
# ---------------------------------------------------------------------------


class InMemorySearchIndex(ISearchIndex):
    """Substring + dot-product matcher for tests."""

    def __init__(self) -> None:
        self._docs: dict[tuple[str, str], IndexDoc] = {}

    async def index(self, doc: IndexDoc) -> None:
        self._docs[(doc.tenant_id, doc.doc_id)] = doc

    async def search(
        self,
        query: str,
        tenant_id: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
    ) -> Sequence[Hit]:
        q = query.lower()
        hits: list[Hit] = []
        for (t, doc_id), doc in self._docs.items():
            if t != tenant_id:
                continue
            # Apply filter predicates if provided.
            if filters and any(doc.fields.get(k) != v for k, v in filters.items()):
                continue
            text_blob = " ".join(str(v).lower() for v in doc.fields.values())
            if q and q not in text_blob:
                continue
            hits.append(Hit(doc_id=doc_id, score=1.0, fields=doc.fields))
        return hits[:limit]

    async def delete(self, doc_id: str, tenant_id: str) -> bool:
        return self._docs.pop((tenant_id, doc_id), None) is not None


# ---------------------------------------------------------------------------
# In-memory Memory (IMemory)
# ---------------------------------------------------------------------------


def _normalise_memory_text(text: str) -> str:
    """Lower + collapse whitespace — the canonical form used for the
    idempotency/similarity gate in the fake (and conceptually mirrored by the
    real FTS adapter's ``plainto_tsquery`` normalisation)."""
    return " ".join(text.lower().split())


def _token_jaccard(a: str, b: str) -> float:
    """Cheap lexical similarity in [0, 1] for the fake's CREATE/MERGE/SKIP gate.

    The real adapter uses Postgres ``ts_rank`` / ``pg_trgm``; the fake only
    needs a deterministic, monotone proxy so the contract tests can assert the
    branch decisions without a database.
    """
    ta = set(_normalise_memory_text(a).split())
    tb = set(_normalise_memory_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class InMemoryMemory(IMemory):
    """Dict-backed :class:`IMemory` implementing the full behavioural contract.

    Faithful to the spec the docstrings in :mod:`protocore.contracts.memory`
    pin: two-stage idempotent write (lexical similarity gate →
    CREATE/MERGE/SKIP), optimistic drift-guard, scope fan-out recall/search,
    reinforcement signals, tenant isolation. Ranking is a lexical proxy
    (token Jaccard + substring), NOT BM25 — that is the real adapter's job;
    the fake exists so core tests run with no database.
    """

    #: Default dedup threshold the fake applies when the caller passes
    #: ``similarity_threshold=None`` (the real adapter resolves this from
    #: ``LoopConstants.memory_write_similarity_threshold``).
    default_similarity_threshold: float = 0.85

    #: Default per-scope soft cap the fake applies when the caller passes
    #: ``max_records_per_scope=None`` (0 = unbounded). Mirrors the real
    #: adapter's ``max_records_per_scope`` dataclass field.
    default_max_records_per_scope: int = 0

    def __init__(self) -> None:
        # keyed by (tenant_id, id)
        self._store: dict[tuple[str, str], MemoryRecord] = {}
        self._counter = 0

    # -- internal helpers ---------------------------------------------------

    def _new_id(self) -> str:
        self._counter += 1
        return f"mem-{self._counter:06d}"

    def _bucket(
        self,
        tenant_id: str,
        scope: MemoryScope,
        scope_key: str,
        kind: str | None,
    ) -> list[MemoryRecord]:
        out: list[MemoryRecord] = []
        for (t, _), rec in self._store.items():
            if t != tenant_id:
                continue
            if rec.scope is not scope or rec.scope_key != scope_key:
                continue
            if kind is not None and rec.kind != kind:
                continue
            out.append(rec)
        return out

    def _trim_bucket(
        self,
        tenant_id: str,
        scope: MemoryScope,
        scope_key: str,
        cap: int,
    ) -> None:
        """Drop least-recently-accessed rows beyond ``cap`` in the bucket.

        Mirrors :meth:`PgMemoryStore._trim_bucket` (LRU by
        ``last_accessed_at`` then ``created_at``). The bucket here is by
        ``(scope, scope_key)`` across all kinds — same as the real adapter.
        """
        rows = [
            rec
            for (t, _), rec in self._store.items()
            if t == tenant_id and rec.scope is scope and rec.scope_key == scope_key
        ]
        if len(rows) <= cap:
            return

        def _recency(rec: MemoryRecord) -> float:
            stamp = rec.last_accessed_at or rec.created_at
            return stamp.timestamp() if stamp else 0.0

        # newest first; everything past the cap is evicted.
        rows.sort(key=lambda r: (_recency(r), r.id), reverse=True)
        for rec in rows[cap:]:
            self._store.pop((tenant_id, rec.id), None)

    # -- write --------------------------------------------------------------

    async def write(
        self,
        tenant_id: str,
        scope: MemoryScope,
        scope_key: str,
        text: str,
        *,
        kind: str = "fact",
        metadata: dict[str, Any] | None = None,
        source_refs: Sequence[str] | None = None,
        salience: float = 0.0,
        similarity_threshold: float | None = None,
        max_records_per_scope: int | None = None,
        expected_version: int | None = None,
    ) -> MemoryWriteResult:
        if not text or not text.strip():
            raise ValueError("memory text must be non-empty")
        if scope is not MemoryScope.global_ and not scope_key:
            raise ValueError(f"scope_key required for scope={scope.value!r}")
        if scope is MemoryScope.global_:
            scope_key = ""

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.default_similarity_threshold
        )
        cap = (
            max_records_per_scope
            if max_records_per_scope is not None
            else self.default_max_records_per_scope
        )
        new_refs = list(dict.fromkeys(source_refs or []))
        new_meta = dict(metadata or {})

        # Apply the similarity gate over the same (scope, scope_key, kind) bucket.
        best: MemoryRecord | None = None
        best_sim = 0.0
        for rec in self._bucket(tenant_id, scope, scope_key, kind):
            sim = _token_jaccard(rec.text, text)
            if sim > best_sim:
                best_sim, best = sim, rec

        now = datetime.now(_UTC)

        # Resolve the result as CREATE, MERGE, or SKIP.
        if best is None or best_sim < threshold:
            rec = MemoryRecord(
                id=self._new_id(),
                tenant_id=tenant_id,
                scope=scope,
                scope_key=scope_key,
                kind=kind,
                text=text,
                metadata=new_meta,
                source_refs=new_refs,
                salience=salience,
                version=1,
                created_at=now,
                updated_at=now,
            )
            self._store[(tenant_id, rec.id)] = rec
            if cap > 0:
                self._trim_bucket(tenant_id, scope, scope_key, cap)
            return MemoryWriteResult(
                record=rec,
                decision=MemoryWriteDecision.created,
                similarity=best_sim,
            )

        # drift-guard on the MERGE/SKIP target.
        if expected_version is not None and best.version != expected_version:
            raise MemoryConflictError(
                f"memory {best.id} version drift: expected {expected_version}, "
                f"live {best.version}"
            )

        merged_refs = list(dict.fromkeys([*best.source_refs, *new_refs]))
        merged_meta = {**best.metadata, **new_meta}
        normalised_equal = _normalise_memory_text(best.text) == _normalise_memory_text(text)
        adds_information = (
            not normalised_equal
            or merged_refs != best.source_refs
            or merged_meta != best.metadata
            or (salience > best.salience)
        )
        if not adds_information:
            return MemoryWriteResult(
                record=best,
                decision=MemoryWriteDecision.skipped,
                similarity=best_sim,
            )

        merged = best.model_copy(
            update={
                # Prefer the longer/richer text on merge (information-additive).
                "text": text if len(text) > len(best.text) else best.text,
                "metadata": merged_meta,
                "source_refs": merged_refs,
                "salience": max(best.salience, salience),
                "version": best.version + 1,
                "updated_at": now,
            }
        )
        self._store[(tenant_id, merged.id)] = merged
        return MemoryWriteResult(
            record=merged,
            decision=MemoryWriteDecision.merged,
            similarity=best_sim,
        )

    # -- retrieval ----------------------------------------------------------

    def _rank(
        self,
        tenant_id: str,
        query: str,
        scopes: Sequence[MemoryScope] | None,
        scope_keys: dict[MemoryScope, str] | None,
        kinds: Sequence[str] | None,
        limit: int,
    ) -> list[MemoryHit]:
        eff_scopes = tuple(scopes) if scopes else DEFAULT_RECALL_SCOPES
        keys = scope_keys or {}
        kind_set = set(kinds) if kinds else None
        q = _normalise_memory_text(query)

        candidates: list[tuple[float, MemoryRecord]] = []
        for (t, _), rec in self._store.items():
            if t != tenant_id:
                continue
            if rec.scope not in eff_scopes:
                continue
            # resolve the scope key: global is fixed "", others must match the
            # caller-supplied key for that scope.
            if rec.scope is MemoryScope.global_:
                pass
            else:
                want = keys.get(rec.scope)
                if want is None or want != rec.scope_key:
                    continue
            if kind_set is not None and rec.kind not in kind_set:
                continue
            if not q:
                # empty query → recency-ordered (score by updated_at epoch).
                candidates.append((rec.updated_at.timestamp(), rec))
                continue
            sim = _token_jaccard(rec.text, query)
            substring_bonus = 0.5 if q in _normalise_memory_text(rec.text) else 0.0
            score = sim + substring_bonus
            if score <= 0.0:
                continue
            candidates.append((score, rec))

        candidates.sort(key=lambda pair: (pair[0], pair[1].updated_at.timestamp()), reverse=True)
        top = candidates[:limit]

        # reinforcement: bump access_count + last_accessed_at on returned rows.
        now = datetime.now(_UTC)
        hits: list[MemoryHit] = []
        for score, rec in top:
            reinforced = rec.model_copy(
                update={
                    "access_count": rec.access_count + 1,
                    "last_accessed_at": now,
                }
            )
            self._store[(tenant_id, rec.id)] = reinforced
            hits.append(MemoryHit(record=reinforced, score=float(score)))
        return hits

    async def recall(
        self,
        tenant_id: str,
        query: str,
        *,
        scopes: Sequence[MemoryScope] | None = None,
        scope_keys: dict[MemoryScope, str] | None = None,
        kinds: Sequence[str] | None = None,
        limit: int = 10,
    ) -> Sequence[MemoryHit]:
        return self._rank(tenant_id, query, scopes, scope_keys, kinds, limit)

    async def search(
        self,
        tenant_id: str,
        query: str,
        *,
        scopes: Sequence[MemoryScope] | None = None,
        scope_keys: dict[MemoryScope, str] | None = None,
        kinds: Sequence[str] | None = None,
        limit: int = 20,
    ) -> Sequence[MemoryHit]:
        return self._rank(tenant_id, query, scopes, scope_keys, kinds, limit)

    # -- crud ---------------------------------------------------------------

    async def get(self, tenant_id: str, memory_id: str) -> MemoryRecord:
        rec = self._store.get((tenant_id, memory_id))
        if rec is None:
            raise MemoryNotFoundError(f"memory {memory_id} not found")
        return rec

    async def delete(
        self,
        tenant_id: str,
        memory_id: str,
        *,
        expected_version: int | None = None,
    ) -> bool:
        rec = self._store.get((tenant_id, memory_id))
        if rec is None:
            return False
        if expected_version is not None and rec.version != expected_version:
            raise MemoryConflictError(
                f"memory {memory_id} version drift: expected {expected_version}, "
                f"live {rec.version}"
            )
        del self._store[(tenant_id, memory_id)]
        return True

    async def list(
        self,
        tenant_id: str,
        *,
        scope: MemoryScope | None = None,
        scope_key: str | None = None,
        kinds: Sequence[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[MemoryRecord]:
        kind_set = set(kinds) if kinds else None
        rows = [
            rec
            for (t, _), rec in self._store.items()
            if t == tenant_id
            and (scope is None or rec.scope is scope)
            and (scope_key is None or rec.scope_key == scope_key)
            and (kind_set is None or rec.kind in kind_set)
        ]
        rows.sort(key=lambda r: (r.updated_at.timestamp(), r.id), reverse=True)
        return rows[offset : offset + limit]


# ---------------------------------------------------------------------------
# In-memory Workspace
# ---------------------------------------------------------------------------


def _path_safe(path: str) -> str:
    """Validate + canonicalise a workspace-relative path (fake mirror of the
    real adapter's ``_validate_relative_path``)."""
    if not path or not path.strip():
        raise WorkspacePathError("empty path denied")
    if any(0 <= ord(c) < 0x20 for c in path):
        raise WorkspacePathError(f"control character denied: {path!r}")
    if "\\" in path:
        raise WorkspacePathError(f"backslash denied: {path!r}")
    if path.startswith("/"):
        raise WorkspacePathError(f"absolute path denied: {path!r}")
    parts = path.split("/")
    if any(p == ".." for p in parts):
        raise WorkspacePathError(f"parent-traversal denied: {path!r}")
    if any(p == "" for p in parts):
        raise WorkspacePathError(f"empty segment denied: {path!r}")
    return path


class InMemoryWorkspace(IWorkspace):
    """Dict-backed :class:`IWorkspace` implementing the full behavioural contract.

    Faithful to the spec the docstrings in
    :mod:`protocore.contracts.workspace` pin: atomic idempotent write
    (CREATE/REPLACE/UNCHANGED keyed by ``(scope, scope_key, path)``), hard quota
    refusal, scratch soft-cap LRU GC, optimistic drift-guard, scope-fan-out
    lexical search, reinforcement signals, tenant isolation, lifecycle teardown.
    Ranking is a lexical proxy (token Jaccard + substring), NOT BM25 — that is
    the real adapter's job; the fake exists so core tests run with no database.
    """

    #: Defaults applied when the caller passes the corresponding cap as ``None``
    #: (0 = unbounded). The real adapter resolves these from LoopConstants.
    default_max_bytes: int = 1_048_576
    default_max_units_per_scope: int = 256
    default_max_scope_bytes: int = 33_554_432
    #: Bounded prefix indexed as searchable_text for a text body.
    searchable_text_max_bytes: int = 65_536

    def __init__(self) -> None:
        # keyed by (tenant_id, scope_value, scope_key, path) → the full unit
        # (with body bytes in ``content``).
        self._store: dict[tuple[str, str, str, str], WorkspaceUnit] = {}
        self._counter = 0

    # -- internal helpers ---------------------------------------------------

    def _new_id(self) -> str:
        self._counter += 1
        return f"ws-{self._counter:06d}"

    def _key(
        self, tenant_id: str, scope: WorkspaceScope, scope_key: str, path: str
    ) -> tuple[str, str, str, str]:
        return (tenant_id, scope.value, scope_key, path)

    def _scope_units(
        self, tenant_id: str, scope: WorkspaceScope, scope_key: str
    ) -> list[WorkspaceUnit]:
        return [
            u
            for (t, sv, sk, _), u in self._store.items()
            if t == tenant_id and sv == scope.value and sk == scope_key
        ]

    def _scope_bytes(
        self, tenant_id: str, scope: WorkspaceScope, scope_key: str
    ) -> int:
        return sum(
            u.size_bytes for u in self._scope_units(tenant_id, scope, scope_key)
        )

    def _gc(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        max_units: int,
        max_scope_bytes: int,
        *,
        protect_id: str | None = None,
        refuse_if_over: bool = True,
    ) -> list[str]:
        """Evict least-recently-accessed SCRATCH units until within both caps.

        Mirrors :meth:`PgWorkspaceStore._gc_scope` exactly: durable units are
        never evicted, the just-written unit (``protect_id``) is never a
        victim, and if the scope is still over cap after exhausting every other
        eligible scratch victim the write is REFUSED with
        :class:`WorkspaceQuotaExceededError` (the caller has already mutated
        ``self._store``, so the caller must roll back) rather than returning a
        unit that GC then evicted. The PRE-write byte reclaim passes
        ``refuse_if_over=False`` (it only opens byte headroom before the new row
        exists; the post-write pass owns refusal).
        """
        evicted: list[str] = []

        def _over() -> bool:
            units = self._scope_units(tenant_id, scope, scope_key)
            if max_units > 0 and len(units) > max_units:
                return True
            if max_scope_bytes > 0:
                total = sum(u.size_bytes for u in units)
                if total > max_scope_bytes:
                    return True
            return False

        while _over():
            scratch = [
                u
                for u in self._scope_units(tenant_id, scope, scope_key)
                if u.lifecycle is WorkspaceLifecycle.scratch
                and u.id != protect_id  # never evict the just-written unit
            ]
            if not scratch:
                break  # only durable / the protected new unit left — give up

            def _recency(u: WorkspaceUnit) -> float:
                stamp = u.last_accessed_at or u.created_at
                return stamp.timestamp() if stamp else 0.0

            # oldest first → evict the least-recently-accessed scratch unit.
            scratch.sort(key=lambda u: (_recency(u), u.id))
            victim = scratch[0]
            self._store.pop(
                self._key(tenant_id, scope, victim.scope_key, victim.path), None
            )
            evicted.append(victim.path)

        if refuse_if_over and _over():
            raise WorkspaceQuotaExceededError(
                f"scope {scope.value}/{scope_key} cannot admit a new unit within "
                f"caps (units<={max_units}, bytes<={max_scope_bytes}) after "
                "evicting all reclaimable scratch units"
            )
        return evicted

    # -- write --------------------------------------------------------------

    async def write(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        path: str,
        content: bytes,
        *,
        lifecycle: WorkspaceLifecycle = DEFAULT_WORKSPACE_LIFECYCLE,
        content_type: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_refs: Sequence[str] | None = None,
        max_bytes: int | None = None,
        max_units_per_scope: int | None = None,
        max_scope_bytes: int | None = None,
        searchable_text_max_bytes: int | None = None,
        expected_version: int | None = None,
    ) -> WorkspaceWriteOutcome:
        if not scope_key:
            raise WorkspacePathError(f"scope_key required for scope={scope.value!r}")
        path = _path_safe(path)
        hard_bytes = max_bytes if max_bytes is not None else self.default_max_bytes
        cap_units = (
            max_units_per_scope
            if max_units_per_scope is not None
            else self.default_max_units_per_scope
        )
        cap_scope_bytes = (
            max_scope_bytes
            if max_scope_bytes is not None
            else self.default_max_scope_bytes
        )
        cap_searchable = (
            searchable_text_max_bytes
            if searchable_text_max_bytes is not None
            else self.searchable_text_max_bytes
        )
        size = len(content)
        if hard_bytes > 0 and size > hard_bytes:
            raise WorkspaceQuotaExceededError(
                f"unit {path!r} is {size} bytes > per-unit cap {hard_bytes}"
            )

        sha = hashlib.sha256(content).hexdigest()
        new_meta = dict(metadata or {})
        new_refs = list(dict.fromkeys(source_refs or []))
        # text decode for the searchable index (binary → empty searchable_text).
        # A NUL byte marks the body BINARY (canonical indicator; mirrors the real
        # PgWorkspaceStore where a PG ``text`` column cannot hold a NUL).
        if b"\x00" in content:
            searchable = ""
            ctype = content_type or "application/octet-stream"
        else:
            try:
                decoded = content.decode("utf-8")
                # 0 = index the whole body; >0 = bounded prefix (char proxy for
                # the real adapter's byte cap — the fake exists so core tests run
                # with no DB).
                searchable = decoded if cap_searchable <= 0 else decoded[:cap_searchable]
                ctype = content_type or "text/plain; charset=utf-8"
            except UnicodeDecodeError:
                searchable = ""
                ctype = content_type or "application/octet-stream"

        now = datetime.now(_UTC)
        key = self._key(tenant_id, scope, scope_key, path)
        existing = self._store.get(key)

        if existing is None:
            # hard scope-byte ceiling check on CREATE.
            if cap_scope_bytes > 0:
                projected = (
                    self._scope_bytes(tenant_id, scope, scope_key) + size
                )
                if projected > cap_scope_bytes:
                    # PRE-write byte reclaim: BYTE-only (max_units=0) + non-
                    # refusing — no new-unit row exists to protect yet; the unit
                    # cap + refusal are owned by the POST-write GC below.
                    self._gc(
                        tenant_id, scope, scope_key, 0, cap_scope_bytes - size,
                        protect_id=None, refuse_if_over=False,
                    )
                    projected = (
                        self._scope_bytes(tenant_id, scope, scope_key) + size
                    )
                    if projected > cap_scope_bytes:
                        raise WorkspaceQuotaExceededError(
                            f"scope {scope.value}/{scope_key} would exceed byte "
                            f"budget {cap_scope_bytes} (projected {projected})"
                        )
            unit = WorkspaceUnit(
                id=self._new_id(),
                tenant_id=tenant_id,
                scope=scope,
                scope_key=scope_key,
                path=path,
                lifecycle=lifecycle,
                size_bytes=size,
                sha256=sha,
                content_type=ctype,
                searchable_text=searchable,
                summary=summary,
                metadata=new_meta,
                source_refs=new_refs,
                content=content,
                version=1,
                created_at=now,
                updated_at=now,
            )
            self._store[key] = unit
            try:
                evicted = self._gc(
                    tenant_id, scope, scope_key, cap_units, cap_scope_bytes,
                    protect_id=unit.id,
                )
            except WorkspaceQuotaExceededError:
                # Roll back the just-inserted row (the real adapter's
                # transaction does this automatically) so we never leak a unit
                # that GC could not fit.
                self._store.pop(key, None)
                raise
            return WorkspaceWriteOutcome(
                unit=unit.model_copy(update={"content": None}),
                decision=WorkspaceWriteDecision.created,
                evicted_paths=evicted,
            )

        # exists — drift-guard on REPLACE.
        if expected_version is not None and existing.version != expected_version:
            raise WorkspaceConflictError(
                f"workspace unit {path!r} version drift: expected "
                f"{expected_version}, live {existing.version}"
            )

        unchanged = (
            existing.sha256 == sha
            and existing.lifecycle is lifecycle
            and existing.summary == summary
            and existing.metadata == new_meta
            and list(existing.source_refs) == new_refs
        )
        if unchanged:
            return WorkspaceWriteOutcome(
                unit=existing.model_copy(update={"content": None}),
                decision=WorkspaceWriteDecision.unchanged,
                evicted_paths=[],
            )

        replaced = existing.model_copy(
            update={
                "lifecycle": lifecycle,
                "size_bytes": size,
                "sha256": sha,
                "content_type": ctype,
                "searchable_text": searchable,
                "summary": summary,
                "metadata": new_meta,
                "source_refs": new_refs,
                "content": content,
                "version": existing.version + 1,
                "updated_at": now,
            }
        )
        self._store[key] = replaced
        try:
            evicted = self._gc(
                tenant_id, scope, scope_key, cap_units, cap_scope_bytes,
                protect_id=replaced.id,
            )
        except WorkspaceQuotaExceededError:
            # Restore the prior row (the real adapter's transaction rolls the
            # UPDATE back) so a refused replace leaves the unit untouched.
            self._store[key] = existing
            raise
        return WorkspaceWriteOutcome(
            unit=replaced.model_copy(update={"content": None}),
            decision=WorkspaceWriteDecision.replaced,
            evicted_paths=evicted,
        )

    # -- read ---------------------------------------------------------------

    async def read(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        path: str,
    ) -> WorkspaceUnit:
        path = _path_safe(path)
        key = self._key(tenant_id, scope, scope_key, path)
        unit = self._store.get(key)
        if unit is None:
            raise WorkspaceNotFoundError(
                f"workspace unit {path!r} not found in {scope.value}/{scope_key}"
            )
        reinforced = unit.model_copy(
            update={
                "access_count": unit.access_count + 1,
                "last_accessed_at": datetime.now(_UTC),
            }
        )
        self._store[key] = reinforced
        # read returns the body populated.
        return reinforced

    # -- search -------------------------------------------------------------

    async def search(
        self,
        tenant_id: str,
        query: str,
        *,
        scopes: Sequence[WorkspaceScope] | None = None,
        scope_keys: dict[WorkspaceScope, str] | None = None,
        lifecycles: Sequence[WorkspaceLifecycle] | None = None,
        limit: int = 20,
    ) -> Sequence[WorkspaceHit]:
        eff_scopes = tuple(scopes) if scopes else tuple(WorkspaceScope)
        keys = scope_keys or {}
        life_set = set(lifecycles) if lifecycles else None
        q = _normalise_memory_text(query)

        candidates: list[tuple[float, tuple[str, str, str, str], WorkspaceUnit]] = []
        for k, unit in self._store.items():
            t, _sv, sk, _ = k
            if t != tenant_id:
                continue
            if unit.scope not in eff_scopes:
                continue
            want = keys.get(unit.scope)
            if want is None or want != sk:
                continue
            if life_set is not None and unit.lifecycle not in life_set:
                continue
            haystack = " ".join(
                [unit.path, unit.searchable_text, unit.summary or ""]
            )
            if not q:
                candidates.append((unit.updated_at.timestamp(), k, unit))
                continue
            sim = _token_jaccard(haystack, query)
            substring_bonus = 0.5 if q in _normalise_memory_text(haystack) else 0.0
            score = sim + substring_bonus
            if score <= 0.0:
                continue
            candidates.append((score, k, unit))

        candidates.sort(
            key=lambda c: (c[0], c[2].updated_at.timestamp()), reverse=True
        )
        top = candidates[:limit]

        now = datetime.now(_UTC)
        hits: list[WorkspaceHit] = []
        for score, k, unit in top:
            reinforced = unit.model_copy(
                update={
                    "access_count": unit.access_count + 1,
                    "last_accessed_at": now,
                }
            )
            self._store[k] = reinforced
            # search returns manifest-only (no body).
            hits.append(
                WorkspaceHit(
                    unit=reinforced.model_copy(update={"content": None}),
                    score=float(score),
                )
            )
        return hits

    # -- list ---------------------------------------------------------------

    async def list(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        *,
        lifecycles: Sequence[WorkspaceLifecycle] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[WorkspaceUnit]:
        if not scope_key:
            raise WorkspacePathError(f"scope_key required for scope={scope.value!r}")
        life_set = set(lifecycles) if lifecycles else None
        rows = [
            u
            for u in self._scope_units(tenant_id, scope, scope_key)
            if life_set is None or u.lifecycle in life_set
        ]
        rows.sort(key=lambda u: (u.updated_at.timestamp(), u.id), reverse=True)
        return [
            u.model_copy(update={"content": None})
            for u in rows[offset : offset + limit]
        ]

    # -- delete / clear -----------------------------------------------------

    async def delete(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        path: str,
        *,
        expected_version: int | None = None,
    ) -> bool:
        path = _path_safe(path)
        key = self._key(tenant_id, scope, scope_key, path)
        unit = self._store.get(key)
        if unit is None:
            return False
        if expected_version is not None and unit.version != expected_version:
            raise WorkspaceConflictError(
                f"workspace unit {path!r} version drift: expected "
                f"{expected_version}, live {unit.version}"
            )
        del self._store[key]
        return True

    async def clear_scope(
        self,
        tenant_id: str,
        scope: WorkspaceScope,
        scope_key: str,
        *,
        lifecycles: Sequence[WorkspaceLifecycle] | None = None,
    ) -> int:
        if not scope_key:
            raise WorkspacePathError(f"scope_key required for scope={scope.value!r}")
        life_set = set(lifecycles) if lifecycles else None
        victims = [
            self._key(tenant_id, scope, scope_key, u.path)
            for u in self._scope_units(tenant_id, scope, scope_key)
            if life_set is None or u.lifecycle in life_set
        ]
        for k in victims:
            self._store.pop(k, None)
        return len(victims)


# ---------------------------------------------------------------------------
# In-memory SkillStore
# ---------------------------------------------------------------------------


class InMemorySkillStore(ISkillStore):
    """``dict[(account_id, skill_id), (manifest, body)]`` backed.

    Flat account-scoped catalog: each skill is ``id``/``name``/
    ``description``/``enabled``. Mirrors the real ``PgSkillStore`` which
    keys on ``(account_id, name)``.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], tuple[SkillManifest, str]] = {}
        # Multi-file overlay keyed by (account_id, skill_id). When a key is
        # missing we synthesise a single SKILL.md entry from the legacy body
        # (lazy backward path).
        self._files: dict[tuple[str, str], dict[str, bytes]] = {}
        # ``SkillManifest`` (the frozen core contract) carries no ``enabled``
        # flag, but the real store filters disabled skills out of ``list`` and
        # ``list_enabled_subset``. Track the toggle as fixture-only side state
        # keyed by ``(account_id, skill_id)``. Missing key ⇒ enabled.
        self._enabled: dict[tuple[str, str], bool] = {}

    async def list(self, account_id: str) -> Sequence[SkillIndexEntry]:
        entries: list[SkillIndexEntry] = []
        for (t, skill_id), (manifest, _) in self._store.items():
            if t != account_id:
                continue
            # The real store only returns ``enabled = TRUE`` skills.
            enabled = self._enabled.get((t, skill_id), True)
            if not enabled:
                continue
            entries.append(
                SkillIndexEntry(
                    id=manifest.id,
                    name=manifest.name,
                    description=manifest.description,
                    enabled=enabled,
                )
            )
        return entries

    async def load(self, account_id: str, skill_id: str) -> SkillBundle:
        manifest, body = self._store[(account_id, skill_id)]
        return SkillBundle(manifest=manifest, body=body)

    async def upsert(self, account_id: str, manifest: SkillManifest, body: str) -> None:
        self._store[(account_id, manifest.id)] = (manifest, body)

    async def create(
        self,
        account_id: str,
        payload: SkillUpsertInput,
    ) -> SkillIndexEntry:
        import uuid

        new_id = str(uuid.uuid4())
        manifest = SkillManifest(
            id=new_id,
            name=payload.name,
            description=payload.description,
            tenant_id=account_id,
        )
        self._store[(account_id, new_id)] = (manifest, payload.body_md)
        self._enabled[(account_id, new_id)] = payload.enabled
        return SkillIndexEntry(
            id=new_id,
            name=payload.name,
            description=payload.description,
            enabled=payload.enabled,
        )

    async def update(
        self,
        account_id: str,
        skill_id: str,
        payload: SkillUpsertInput,
    ) -> SkillIndexEntry:
        if (account_id, skill_id) not in self._store:
            from protocore.contracts.skills import SkillNotFoundError

            raise SkillNotFoundError(f"skill {skill_id} not found")
        manifest = SkillManifest(
            id=skill_id,
            name=payload.name,
            description=payload.description,
            tenant_id=account_id,
        )
        self._store[(account_id, skill_id)] = (manifest, payload.body_md)
        self._enabled[(account_id, skill_id)] = payload.enabled
        return SkillIndexEntry(
            id=skill_id,
            name=payload.name,
            description=payload.description,
            enabled=payload.enabled,
        )

    async def delete(self, account_id: str, skill_id: str) -> None:
        self._store.pop((account_id, skill_id), None)
        self._enabled.pop((account_id, skill_id), None)

    async def set_enabled(self, account_id: str, skill_id: str, *, enabled: bool) -> None:
        # Track the toggle in fixture side state so ``list`` /
        # ``list_enabled_subset`` drop disabled skills with store parity.
        self._enabled[(account_id, skill_id)] = enabled

    def _subset_entries(
        self,
        account_id: str,
        names: Sequence[str],
        *,
        enabled_only: bool,
    ) -> Sequence[SkillIndexEntry]:
        # NB: return type is ``Sequence`` (not ``list``) because the bare name
        # ``list`` resolves to this class's ``list`` METHOD in the class-scope
        # annotation namespace, which mypy rejects as a type. The local ``out``
        # below is fine — it is annotated in the function scope where ``list``
        # is still the builtin.
        wanted = set(names)
        out: list[SkillIndexEntry] = []
        for (t, skill_id), (manifest, _) in self._store.items():
            if t != account_id:
                continue
            if manifest.name not in wanted:
                continue
            enabled = self._enabled.get((t, skill_id), True)
            if enabled_only and not enabled:
                continue
            out.append(
                SkillIndexEntry(
                    id=manifest.id,
                    name=manifest.name,
                    description=manifest.description,
                    enabled=enabled,
                )
            )
        return out

    async def list_subset(
        self,
        account_id: str,
        names: Sequence[str],
    ) -> Sequence[SkillIndexEntry]:
        # Whitelist resolution by bare name — like the real store, ignores
        # the ``enabled`` toggle.
        return self._subset_entries(account_id, names, enabled_only=False)

    async def list_enabled_subset(
        self,
        account_id: str,
        names: Sequence[str],
    ) -> Sequence[SkillIndexEntry]:
        # Prompt-surfacing path: same bare-name matching as ``list_subset``
        # plus the ``enabled = TRUE`` filter so a disabled skill is not
        # resurfaced by a stale project pin.
        return self._subset_entries(account_id, names, enabled_only=True)

    # ------------------------------------------------------------------
    # Multi-file bundle surface.
    # ------------------------------------------------------------------

    def put_file(
        self,
        account_id: str,
        skill_id: str,
        path: str,
        body: bytes,
    ) -> None:
        """Test helper — add a file to the bundle overlay.

        Not part of the :class:`ISkillStore` Protocol; used by
        integration tests to exercise the multi-file surface without
        needing the real PG-backed adapter + IBlobStore wiring.
        """
        bundle = self._files.setdefault((account_id, skill_id), {})
        bundle[path] = body

    async def list_files(
        self,
        account_id: str,
        skill_id: str,
    ) -> Sequence[SkillFileRef]:
        bundle = self._files.get((account_id, skill_id))
        if bundle is None:
            # Lazy backward path — synthesise SKILL.md from legacy body.
            stored = self._store.get((account_id, skill_id))
            if stored is None:
                return []
            _, body = stored
            body_bytes = body.encode("utf-8")
            return [
                SkillFileRef(
                    path=SKILL_ENTRY_PATH,
                    size_bytes=len(body_bytes),
                    mime_type=SKILL_ENTRY_MIME_TYPE,
                    content_hash=hashlib.sha256(body_bytes).hexdigest(),
                )
            ]
        out: list[SkillFileRef] = []
        for path, body_bytes in bundle.items():
            mime = (
                SKILL_ENTRY_MIME_TYPE
                if path == SKILL_ENTRY_PATH
                else "application/octet-stream"
            )
            out.append(
                SkillFileRef(
                    path=path,
                    size_bytes=len(body_bytes),
                    mime_type=mime,
                    content_hash=hashlib.sha256(body_bytes).hexdigest(),
                )
            )
        return out

    async def load_file(
        self,
        account_id: str,
        skill_id: str,
        path: str,
    ) -> bytes | None:
        bundle = self._files.get((account_id, skill_id))
        if bundle is None:
            if path != SKILL_ENTRY_PATH:
                return None
            stored = self._store.get((account_id, skill_id))
            if stored is None:
                return None
            _, body = stored
            return body.encode("utf-8")
        return bundle.get(path)


# ---------------------------------------------------------------------------
# In-memory AgentDispatch
# ---------------------------------------------------------------------------


class InMemoryAgentDispatch(IAgentDispatch):
    """In-memory subagent registry + scripted dispatch result."""

    def __init__(self, pool: InMemoryWorkPool | None = None) -> None:
        self._defs: dict[tuple[str, str], SubagentDef] = {}
        self._next_result: SubagentResult | None = None
        #: Where launched children live. Given one by the host in production;
        #: minted here so a test that only wants a scripted result gets a
        #: working pool without having to assemble one.
        self.pool = pool if pool is not None else InMemoryWorkPool()

    def register(self, definition: SubagentDef) -> None:
        self._defs[(definition.tenant_id, definition.id)] = definition

    def queue_result(self, result: SubagentResult) -> None:
        """Script the next :meth:`dispatch` return value."""
        self._next_result = result

    async def list_subagents(self, tenant_id: str) -> Sequence[SubagentDef]:
        return [d for (t, _), d in self._defs.items() if t == tenant_id]

    async def get(self, tenant_id: str, subagent_id: str) -> SubagentDef:
        try:
            return self._defs[(tenant_id, subagent_id)]
        except KeyError as e:
            raise SubagentNotFoundError(subagent_id) from e

    async def dispatch(self, task: SubagentTask) -> SubagentHandle:
        """Launch the scripted child and hand back its handle.

        The double launches through the same pool a host would, so a test that
        exercises the handle exercises the shape the host has to supply rather
        than a shortcut only the double has.
        """
        result = self._next_result
        self._next_result = None
        if result is None:
            result = SubagentResult(
                subagent_id=task.subagent_id,
                parent_run_id=task.parent_run_id,
                output="",
                success=True,
            )
        settled = result

        async def _start(task_id: str) -> WorkHandle[SubagentResult]:
            async def _run() -> SubagentResult:
                return settled

            return InMemoryWorkHandle(task_id=task_id, pool=self.pool, run=_run())

        record = await self.pool.launch(
            WorkSpec(
                session_id=task.parent_run_id,
                kind="agent",
                label=task.subagent_id,
                notify_on_finish=task.notify_on_finish,
                expected_seconds=task.expected_seconds,
                timeout_seconds=task.timeout_seconds,
                agent=AgentRef(name=task.subagent_id),
            ),
            _start,
        )
        handle = self.pool.handle(record.id)
        if handle is None:  # pragma: no cover - launch always binds one
            raise RuntimeError(f"pool lost the handle for {record.id}")
        return cast("SubagentHandle", handle)


# ---------------------------------------------------------------------------
# In-memory work pool — commands and child runs under one address space
# ---------------------------------------------------------------------------


class InMemoryWorkHandle:
    """One launched unit of work, backed by a coroutine.

    The handle owns the task, not the other way round: cancelling it is how the
    work stops, and awaiting it is how the work is collected. Both are done
    through the same object because a caller that could wait but not stop would
    have to reach past the handle to cancel, and whatever it reached for would
    be the real handle.
    """

    def __init__(
        self, *, task_id: str, pool: InMemoryWorkPool, run: Coroutine[Any, Any, Any]
    ) -> None:
        self._task_id = task_id
        self._pool = pool
        self._task: asyncio.Task[Any] = asyncio.ensure_future(run)
        self._task.add_done_callback(self._settle)

    def identity(self) -> TaskRecord:
        record = self._pool.get(self._task_id)
        if record is None:  # pragma: no cover - the record outlives the handle
            raise RuntimeError(f"pool lost the record for {self._task_id}")
        return record

    async def wait(self) -> Any:
        return await asyncio.shield(self._task)

    async def stop(self, grace_seconds: float = 0.0) -> TaskRecord:
        if not self._task.done():
            if grace_seconds > 0:
                with suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(
                        asyncio.shield(self._task), timeout=grace_seconds
                    )
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        return self.identity()

    def _settle(self, task: asyncio.Task[Any]) -> None:
        record = self._pool.get(self._task_id)
        if record is None or record.terminal:
            return
        record.finished_at = self._pool.clock()
        if task.cancelled():
            record.status = "stopped"
        elif task.exception() is not None:
            record.status = "failed"
            record.error = str(task.exception())
            record.exit_code = 1
        else:
            record.status = "succeeded"
            record.exit_code = 0
        self._pool.announce(record)


class InMemoryWorkPool:
    """A pool that holds both kinds of work, for tests and for host reference."""

    def __init__(self) -> None:
        self.records: list[TaskRecord] = []
        self.handles: dict[str, InMemoryWorkHandle] = {}
        self.attached_sessions: set[str] = set()
        self.observers: list[TerminalObserver] = []
        self.stopped_sessions: list[tuple[str, float]] = []
        self._minted = 0
        self._woken: set[str] = set()
        self._now = 0.0

    # -- what a test drives --------------------------------------------------

    def clock(self) -> float:
        """Monotonic-enough time. Advanced explicitly so durations are stated."""
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def announce(self, record: TaskRecord) -> None:
        for observer in list(self.observers):
            observer(record)

    # -- IWorkPool -----------------------------------------------------------

    async def launch(self, spec: WorkSpec, start: StartWork) -> TaskRecord:
        self._minted += 1
        record = TaskRecord.minted(f"task-{self._minted}", spec)
        self.records.append(record)
        self.attached_sessions.add(spec.session_id)
        record.status = "running"
        record.started_at = self._now
        handle = await start(record.id)
        self.handles[record.id] = cast("InMemoryWorkHandle", handle)
        return record

    def handle(self, task_id: str) -> WorkHandle[Any] | None:
        return self.handles.get(task_id)

    async def stop(
        self, task_id: str, grace_seconds: float = 0.0
    ) -> TaskRecord | None:
        handle = self.handles.get(task_id)
        if handle is None:
            return self.get(task_id)
        return await handle.stop(grace_seconds)

    async def stop_session(
        self, scope: str, grace_seconds: float = 0.0
    ) -> Sequence[TaskRecord]:
        self.stopped_sessions.append((scope, grace_seconds))
        stopped: list[TaskRecord] = []
        for record in list(self.owned_by(scope)):
            if record.terminal:
                continue
            result = await self.stop(record.id, grace_seconds)
            if result is not None:
                stopped.append(result)
        return stopped

    def owned_by(self, scope: str) -> builtins.list[TaskRecord]:
        """Records ``scope`` is responsible for ending — its own, or its session's."""
        return [
            record
            for record in self.records
            if scope in (record.owner_scope, record.session_id)
        ]

    def subscribe(self, on_terminal: TerminalObserver) -> Callable[[], None]:
        self.observers.append(on_terminal)

        def _cancel() -> None:
            if on_terminal in self.observers:
                self.observers.remove(on_terminal)

        return _cancel

    def mark_session_attached(self, session_id: str) -> None:
        self.attached_sessions.add(session_id)

    async def ensure_session_attached(self, session_id: str) -> bool:
        if session_id not in self.attached_sessions:
            return False
        return all(
            record.terminal or record.id in self.handles
            for record in self.list(session_id)
        )

    def list(self, session_id: str) -> builtins.list[TaskRecord]:
        return [item for item in self.records if item.session_id == session_id]

    def get(self, task_id: str) -> TaskRecord | None:
        for item in self.records:
            if item.id == task_id:
                return item
        return None

    async def refresh(self, task_id: str) -> object:
        return self.get(task_id)

    def drain_wakes(self, session_id: str) -> builtins.list[str]:
        ids = [
            item.id
            for item in self.list(session_id)
            if item.notify_on_finish
            and item.status in BACKGROUND_TERMINAL_STATUSES
            and item.id not in self._woken
        ]
        self._woken.update(ids)
        return ids


# ---------------------------------------------------------------------------
# In-memory SessionStore
# ---------------------------------------------------------------------------


class InMemorySessionStore(ISessionStore):
    """``dict[(tenant_id, session_id), Session]`` + per-session message log."""

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], Session] = {}
        self._messages: dict[tuple[str, str], list[Message]] = defaultdict(list)

    async def create(self, session: Session) -> None:
        self._sessions[(session.tenant_id, session.id)] = session

    async def get(self, session_id: str, tenant_id: str) -> Session:
        try:
            return self._sessions[(tenant_id, session_id)]
        except KeyError as e:
            raise SessionNotFoundError(session_id) from e

    async def append_message(self, session_id: str, tenant_id: str, message: Message) -> None:
        if (tenant_id, session_id) not in self._sessions:
            raise SessionNotFoundError(session_id)
        self._messages[(tenant_id, session_id)].append(message)

    async def list_messages(
        self,
        session_id: str,
        tenant_id: str,
        *,
        since: datetime | None = None,
        limit: int = 200,
    ) -> Sequence[Message]:
        if (tenant_id, session_id) not in self._sessions:
            raise SessionNotFoundError(session_id)
        messages = self._messages[(tenant_id, session_id)]
        if since is not None:
            messages = [m for m in messages if m.created_at > since]
        return list(messages[:limit])


# ---------------------------------------------------------------------------
# In-memory RunStore
# ---------------------------------------------------------------------------


class InMemoryRunStore(IRunStore):
    """``dict[(tenant_id, run_id), Run]`` backed."""

    def __init__(self) -> None:
        self._runs: dict[tuple[str, str], Run] = {}

    async def create(self, run: Run) -> None:
        self._runs[(run.tenant_id, run.id)] = run

    async def get(self, run_id: str, tenant_id: str) -> Run:
        try:
            return self._runs[(tenant_id, run_id)]
        except KeyError as e:
            raise RunNotFoundError(run_id) from e

    async def update_status(self, run_id: str, tenant_id: str, status: RunStatus) -> None:
        run = await self.get(run_id, tenant_id)
        self._runs[(tenant_id, run_id)] = run.model_copy(update={"status": status})

    async def list(
        self,
        tenant_id: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> Sequence[Run]:
        items = [r for (t, _), r in self._runs.items() if t == tenant_id]
        if filters:
            items = [r for r in items if all(getattr(r, k, None) == v for k, v in filters.items())]
        return items[:limit]

    async def flush_terminal(
        self,
        run_id: str,
        tenant_id: str,
        detail_blob_ref: str,
    ) -> None:
        run = await self.get(run_id, tenant_id)
        self._runs[(tenant_id, run_id)] = run.model_copy(
            update={"detail_blob_ref": detail_blob_ref}
        )


# ---------------------------------------------------------------------------
# In-memory EventStream
# ---------------------------------------------------------------------------


class InMemoryEventStream(IEventStream):
    """``dict[(tenant_id, run_id), list[Event]]`` + live-tail via Event."""

    def __init__(self) -> None:
        self._streams: dict[tuple[str, str], list[Event]] = defaultdict(list)
        self._new_event: dict[tuple[str, str], asyncio.Event] = {}

    async def emit(self, event: Event) -> None:
        # ``event.run_id`` is required; tenant lives in payload metadata.
        # We key by (tenant from payload OR 'default', run_id).
        tenant = str(event.payload.get("tenant_id", "default"))
        key = (tenant, event.run_id)
        self._streams[key].append(event)
        if key in self._new_event:
            self._new_event[key].set()

    async def subscribe(
        self,
        run_id: str,
        tenant_id: str,
        *,
        from_event_id: str | None = None,
    ) -> AsyncIterator[Event]:
        key = (tenant_id, run_id)
        ev = self._new_event.setdefault(key, asyncio.Event())
        # Replay backlog (optionally starting after from_event_id).
        backlog = list(self._streams.get(key, []))
        if from_event_id is not None:
            idx = next(
                (i for i, e in enumerate(backlog) if e.id == from_event_id),
                -1,
            )
            backlog = backlog[idx + 1 :]
        cursor = len(self._streams.get(key, []))
        for event in backlog:
            yield event
        # Tail.
        while True:
            await ev.wait()
            ev.clear()
            tail = self._streams.get(key, [])
            for event in tail[cursor:]:
                yield event
            cursor = len(tail)

    async def trim(self, run_id: str, tenant_id: str, *, max_len: int) -> None:
        key = (tenant_id, run_id)
        stream = self._streams.get(key)
        if stream is None or len(stream) <= max_len:
            return
        self._streams[key] = stream[-max_len:]

    def stream_for(self, tenant_id: str, run_id: str) -> Sequence[Event]:
        """Test helper — read backlog directly (no subscription)."""
        return list(self._streams.get((tenant_id, run_id), []))


# ---------------------------------------------------------------------------
# In-memory HookManager (captures invocations)
# ---------------------------------------------------------------------------


class InMemoryHookManager(IHookManager):
    """Captures every invocation; configurable per-event response."""

    def __init__(self) -> None:
        self._specs: dict[str, HookSpec] = {}
        self._scripted: dict[HookEvent, list[HookResult]] = {}
        self.invocations: list[tuple[HookEvent, dict[str, Any], str]] = []

    def queue_action(self, event: HookEvent, result: HookResult) -> None:
        """Script one more :meth:`invoke` return value for an event.

        A queue rather than a slot, because a single assistant message can
        ask for several calls and a test about what happens to the SECOND one
        has to be able to say what the hook did to the first.
        """
        self._scripted.setdefault(event, []).append(result)

    async def invoke(
        self,
        event: HookEvent,
        payload: dict[str, Any],
        tenant_id: str,
    ) -> HookResult:
        self.invocations.append((event, payload, tenant_id))
        scripted = self._scripted.get(event)
        if scripted:
            return scripted.pop(0)
        return HookResult(action=HookActionKind.ALLOW)

    async def register(self, spec: HookSpec) -> None:
        self._specs[spec.id] = spec

    async def unregister(self, hook_id: str, tenant_id: str) -> None:
        spec = self._specs.get(hook_id)
        if spec is not None and spec.tenant_id == tenant_id:
            self._specs.pop(hook_id)

    async def list(
        self,
        tenant_id: str,
        *,
        event: HookEvent | None = None,
    ) -> Sequence[HookSpec]:
        items = [s for s in self._specs.values() if s.tenant_id == tenant_id]
        if event is not None:
            items = [s for s in items if s.event is event]
        return items


# ---------------------------------------------------------------------------
# In-memory TodoStorage
# ---------------------------------------------------------------------------


class InMemoryTodoStorage(ITodoStorage):
    """``dict[(tenant_id, session_id), list[Todo]]`` backed."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[Todo]] = {}

    async def read(self, session_id: str, tenant_id: str) -> Sequence[Todo]:
        return list(self._store.get((tenant_id, session_id), []))

    async def write(
        self,
        session_id: str,
        tenant_id: str,
        todos: Sequence[Todo],
    ) -> None:
        self._store[(tenant_id, session_id)] = list(todos)


# ---------------------------------------------------------------------------
# In-memory ToolRegistry
# ---------------------------------------------------------------------------


class InMemoryToolRegistry(IToolRegistry):
    """Simple in-memory registry with name-keyed lookup.

 The 3-layer surface compute is intentionally minimal here — it returns
 the policy-filtered set ordered by registration. will ship the
 full retrieval-aware variant.
 """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_all(self) -> Sequence[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def list_for_tenant(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
    ) -> Sequence[Tool]:
        del tenant_id
        visible = (
            self._tools.values()
            if not policy.visible
            else (t for t in self._tools.values() if t.name in policy.visible)
        )
        return [t for t in visible if t.name not in policy.blocked]

    def filter_by_whitelist(self, names: Sequence[str]) -> Sequence[Tool]:
        tools = [self._tools[n] for n in names if n in self._tools]
        return sorted(tools, key=lambda t: t.name)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        tenant_id: str = "",
        whitelist: Sequence[str] | None = None,
        policy: ToolVisibilityPolicy | None = None,
        retrieval: RetrievalSettings | None = None,
    ) -> Sequence[Tool]:
        del tenant_id, retrieval
        pool: list[Tool] = list(self._tools.values())
        if whitelist is not None:
            allow = frozenset(whitelist)
            pool = [t for t in pool if t.name in allow]
        if policy is not None:
            pool = [t for t in pool if policy_admits(policy, t.name)]
        # Trivial substring rank over description; ties broken by name.
        q = query.lower().strip()
        if not q:
            return sorted(pool, key=lambda t: t.name)[:top_k]
        scored = [
            (1 if q in t.definition.description.lower() else 0, t.name, t)
            for t in pool
        ]
        scored.sort(key=lambda triple: (-triple[0], triple[1]))
        return [t for _, _, t in scored[:top_k]]

    def compute_effective_surface(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
        *,
        query: str = "",
        top_k: int | None = None,
        retrieval: RetrievalSettings | None = None,
    ) -> Sequence[ToolDefinition]:
        del query, retrieval
        tools = self.list_for_tenant(tenant_id, policy)
        defs = [t.definition for t in tools]
        if top_k is not None:
            defs = defs[:top_k]
        return defs


# ---------------------------------------------------------------------------
# In-memory request-manifest sink + the replay provider it feeds
# ---------------------------------------------------------------------------


class ReplayMismatchError(AssertionError):
    """A replayed run asked for something the recording does not contain.

    An ``AssertionError`` on purpose: a replay that quietly served the nearest
    recorded answer would be a fixture that agrees with whatever the code now
    does, which is the one thing a recording exists to rule out.
    """


class InMemoryRequestManifestSink(IRequestManifestSink):
    """Keeps every manifest, and puts the oversized bodies in a blob store.

    This is the shape a host is expected to have — the core hands over a
    manifest and the parts too large to inline, and the host decides where they
    live. Here they go into an :class:`InMemoryBlobStore`, whose ``put`` is
    content-addressed and idempotent exactly as the contract requires, and the
    ref comes back onto a copy of the manifest through
    :meth:`RequestManifest.with_blob_refs`.

    Recording is what makes it useful twice: the manifests it holds are the
    input to :class:`ReplayLLMProvider`, so a test can drive a run, keep what
    it asked for, and then prove that the same run rebuilt elsewhere asks for
    the same thing.
    """

    def __init__(self, blobs: InMemoryBlobStore | None = None) -> None:
        self.blobs = blobs if blobs is not None else InMemoryBlobStore()
        self._manifests: list[RequestManifest] = []
        self._by_id: dict[str, RequestManifest] = {}

    @property
    def manifests(self) -> Sequence[RequestManifest]:
        """Every manifest recorded, in the order the run made the calls."""
        return tuple(self._manifests)

    @property
    def request_digests(self) -> Sequence[str]:
        """The recorded requests' content digests, in call order."""
        return tuple(item.request_sha256 for item in self._manifests)

    def get(self, manifest_id: str) -> RequestManifest:
        return self._by_id[manifest_id]

    async def body_of(self, manifest: RequestManifest, slot: str) -> bytes:
        """The canonical bytes of one manifest slot, inline or out of the store."""
        value: ManifestValue = getattr(manifest, slot)
        if value.inline is not None:
            return value.inline.encode("utf-8")
        if value.blob_ref is None:
            raise KeyError(f"{slot} is neither inline nor stored")
        tenant_id = manifest.identity.get("tenant_id") or _REPLAY_TENANT
        return await self.blobs.get(tenant_id, value.blob_ref)

    async def record_request_manifest(
        self,
        *,
        manifest: RequestManifest,
        manifest_id: str,
        bodies: Mapping[str, bytes],
    ) -> None:
        tenant_id = manifest.identity.get("tenant_id") or _REPLAY_TENANT
        refs: dict[str, str] = {}
        for slot, body in bodies.items():
            stored = await self.blobs.put(
                tenant_id, body, content_type="application/json"
            )
            refs[slot] = stored.ref
        kept = manifest.with_blob_refs(refs) if refs else manifest
        # The id is a property of what was SENT, so storing the bodies must not
        # have moved it. Checked rather than assumed: this double is what the
        # replay provider keys on.
        if kept.manifest_id != manifest_id:
            raise ValueError(
                f"storing the bodies moved the manifest id from {manifest_id} "
                f"to {kept.manifest_id}"
            )
        self._manifests.append(kept)
        self._by_id[manifest_id] = kept


#: Tenant used for blob puts when a manifest carries no correlation identity.
#: Obviously synthetic, and never a real scope.
_REPLAY_TENANT = "replay"


class ReplayLLMProvider(ILLMProvider):
    """Serves a recorded run's provider events back, in order, or refuses.

    A recording is a list of (request digest, stream events) pairs — what a
    real provider was asked and what it said. Replaying it re-drives the loop
    without an endpoint and without paying for the tokens twice, which is what
    makes a recorded incident into a regression test.

    The refusal is the point. The provider matches the request it is handed
    against the digest recorded for that position, and a run whose assembly has
    drifted — a prompt section that moved, a tool that is no longer advertised,
    a constant that changed a budget — raises :class:`ReplayMismatchError`
    instead of being served an answer to a question it did not ask. Correlation
    metadata is excluded from the digest (see
    :func:`~protocore.contracts.observability.request_digest`), so replaying a
    recording under a new run id is not itself a mismatch.

    Build one with :meth:`from_recording`, or from a sink and the streams that
    were seen alongside it with :meth:`from_sink`.
    """

    def __init__(self, recording: Sequence[tuple[str, Sequence[LLMStreamEvent]]]) -> None:
        self._recording = list(recording)
        self._position = 0
        self._calls: list[LLMRequest] = []

    @classmethod
    def from_recording(
        cls, recording: Sequence[tuple[str, Sequence[LLMStreamEvent]]]
    ) -> ReplayLLMProvider:
        return cls(recording)

    @classmethod
    def from_sink(
        cls,
        sink: InMemoryRequestManifestSink,
        streams: Sequence[Sequence[LLMStreamEvent]],
    ) -> ReplayLLMProvider:
        """Pair each recorded manifest with the stream that answered it."""
        if len(streams) != len(sink.manifests):
            raise ValueError(
                f"{len(sink.manifests)} manifest(s) recorded but "
                f"{len(streams)} stream(s) supplied; a replay needs one answer "
                "per request"
            )
        return cls(list(zip(sink.request_digests, streams, strict=True)))

    @property
    def calls(self) -> Sequence[LLMRequest]:
        """Every request replayed, for assertions."""
        return tuple(self._calls)

    @property
    def exhausted(self) -> bool:
        return self._position >= len(self._recording)

    def _next(self, request: LLMRequest) -> Sequence[LLMStreamEvent]:
        digest = request_digest(request)
        if self.exhausted:
            raise ReplayMismatchError(
                f"the recording holds {len(self._recording)} call(s) and the "
                f"run is making another one (digest {digest[:16]})"
            )
        expected, events = self._recording[self._position]
        if digest != expected:
            raise ReplayMismatchError(
                f"call {self._position} of the recording was made with request "
                f"{expected[:16]} and this run is making {digest[:16]}: the "
                "request assembly has changed, so the recorded answer is an "
                "answer to a different question"
            )
        self._position += 1
        self._calls.append(request)
        return events

    async def stream_with_tools(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]:
        for event in self._next(request):
            yield event

    async def complete_structured(
        self,
        request: LLMRequest,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        del response_schema
        return _response_from_events(self._next(request))

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        return _response_from_events(self._next(request))

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return max(1, len(text) // 4) if text else 0


def _response_from_events(events: Sequence[LLMStreamEvent]) -> LLMResponse:
    """Fold a recorded stream into the one response a non-streaming call wants."""
    text_parts: list[str] = []
    stop_reason = StopReason.end_turn
    for event in events:
        if event.name == "content_block_delta":
            piece = event.payload.get("text")
            if isinstance(piece, str):
                text_parts.append(piece)
        elif event.name == "message_stop":
            raw = event.payload.get("stop_reason")
            if isinstance(raw, str):
                stop_reason = StopReason(raw)
    text = "".join(text_parts)
    return LLMResponse(
        message=Message(
            role=MessageRole.assistant,
            content_blocks=[TextBlock(text=text)] if text else [],
        ),
        stop_reason=stop_reason,
    )


__all__ = [
    "InMemoryAgentDispatch",
    "InMemoryBlobStore",
    "InMemoryEventStream",
    "InMemoryHookManager",
    "InMemoryLLMProvider",
    "InMemoryMemory",
    "InMemoryRequestManifestSink",
    "InMemoryRunStore",
    "InMemorySearchIndex",
    "InMemorySessionStore",
    "InMemorySkillStore",
    "InMemoryTodoStorage",
    "InMemoryToolRegistry",
    "InMemoryWorkHandle",
    "InMemoryWorkPool",
    "InMemoryWorkspace",
    "ReplayLLMProvider",
    "ReplayMismatchError",
]
