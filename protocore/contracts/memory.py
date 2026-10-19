"""``IMemory`` Protocol — universal, scoped, searchable agent memory.

Greenfield subsystem. Protocore had **no** memory subsystem before this
contract: the existing stores cover transcripts
(:class:`~protocore.contracts.session.ISessionStore`), content-addressed blobs
(:class:`~protocore.contracts.blob.IBlobStore`), a generic search index
(:class:`~protocore.contracts.search.ISearchIndex`), and per-session todos
(:class:`~protocore.contracts.todo.ITodoStorage`). None of them is a *typed,
scope-aware, idempotent, retrieval-ranked* memory of facts the agent learns and
re-uses — the capability this contract adds.

Design north-star (all behaviours below are the *contract*, not implementation
hints):

* **Scope grammar** — every record carries a ``(scope, scope_key)`` tuple:
  ``global | user | project | session | agent | custom``. A single-scope
  (user/global only) design leaks cross-task context, so the full grammar is
  first-class. The most-isolated default is **session scope only** (no
  cross-session leak); the product supports the full grammar — controlled
  per-tenant via LoopConstants, never hard-coded.
* **Lexical/BM25 retrieval is mandatory in v1** — pure-vector recall misses the
  exact SKUs / order-IDs / table names / filesystem paths that dominate ops and
  data tasks. v1 ranks with Postgres full-text search (``tsvector`` +
  ``ts_rank``) / ``pg_trgm``. The contract is intentionally **backend-agnostic**
  so a hybrid vector + decay/reinforce + cross-encoder rerank upgrade is a
  non-breaking v2 drop-in (``MemoryRecord.embedding`` is reserved;
  :meth:`IMemory.search` ranking is opaque to the caller).
* **Idempotent writes** — :meth:`IMemory.write` is two-stage: a cheap
  similarity pre-filter then a CREATE / MERGE / SKIP decision, so a retried /
  re-captured fact does not bloat the store or self-contradict. The returned
  :class:`MemoryWriteResult` tells the caller (and ultimately the *model*) which
  branch fired — the runtime never silently double-writes.
* **Non-fatal, off the hot path, bounded** — memory must never take down the
  agent. Recall is time-boxable; a slow or failed memory op degrades to "no
  memory this turn", never an error that aborts the run. (Enforced by
  the host adapter + the RC budget; the contract documents the expectation.)
* **Untrusted content** — a record's ``text`` is **untrusted input**: it may be
  authored by the model, by a sister session sharing a non-session scope, or by
  a compromised tool, and it re-enters model context on recall (and, via the
  optional auto-recall hook, the prompt). It SHOULD therefore be scanned for
  prompt-injection / exfiltration content both *before* it is persisted and
  *before* it is injected. The seam for that scan is
  :class:`IMemoryContentScanner` (the host adapter populates it; the
  fence around recalled memory is defense-in-depth, not the only control).
* **Horizontal-scale-safe** — durable state lives in Postgres; there is no
  module-level / per-pod authority. Every method is tenant-scoped.

Reference shape: a relational store with a vector index for recall
(Postgres FTS/BM25, atomic two-stage idempotent write, drift-guard on
concurrent write). Core defines the contract only and never imports it
(boundary guard: ``protocore/tests/test_core_import_boundary.py``).
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MemoryError(Exception):
    """Base class for memory-subsystem domain errors."""


class MemoryNotFoundError(MemoryError):
    """Requested memory id is unknown for the given tenant+scope."""


class MemoryStoreUnavailableError(MemoryError):
    """Backing store is temporarily unreachable.

    Callers (the runtime) MUST treat this as **non-fatal**: a recall that
    raises this degrades to "no memory this turn"; a write that raises it is
    dropped with a ``logger.warning`` — never an error that aborts the run
    (memory must never take down the agent).
    """


class MemoryConflictError(MemoryError):
    """Optimistic-concurrency (drift-guard) failure on a write.

    Raised by :meth:`IMemory.write` / :meth:`IMemory.delete` when an
    ``expected_version`` was supplied and the live record's version no longer
    matches — i.e. a concurrent writer mutated the same record between the
    caller's read and write. The caller re-reads and retries (or surfaces the
    conflict to the model). The store NEVER blind-overwrites a drifted record.
    """


# ---------------------------------------------------------------------------
# Scope grammar
# ---------------------------------------------------------------------------


class MemoryScope(StrEnum):
    """The scope a memory record is filed under.

    A record's *address* is the tuple ``(tenant_id, scope, scope_key)``:

    * :attr:`global_` — tenant-wide, shared across every user/project/session.
      ``scope_key`` is conventionally the empty string ``""`` (there is exactly
      one global bucket per tenant).
    * :attr:`user` — per end-user. ``scope_key`` = the user id.
    * :attr:`project` — per project / repository / workspace.
      ``scope_key`` = the project id.
    * :attr:`session` — per conversation/session. ``scope_key`` = the session
      id. **This is the safest default** (isolated, no cross-session leak).
    * :attr:`agent` — per agent / subagent persona. ``scope_key`` = the agent
      id. Lets a specialised subagent keep its own working notes.
    * :attr:`custom` — an escape hatch for a caller-defined namespace.
      ``scope_key`` = the custom namespace string.

    The enum *value* is the lowercase string stored in Postgres and surfaced on
    the wire. ``global`` is a Python keyword, so the member is :attr:`global_`
    with value ``"global"``.
    """

    global_ = "global"
    user = "user"
    project = "project"
    session = "session"
    agent = "agent"
    custom = "custom"


# Default fan-out used by :meth:`IMemory.recall` when the caller does not pin an
# explicit scope set: the union of the durable + working scopes. The host
# resolver narrows / widens this per-tenant via LoopConstants (the most
# isolated configuration collapses it to ``(session,)`` only).
DEFAULT_RECALL_SCOPES: tuple[MemoryScope, ...] = (
    MemoryScope.session,
    MemoryScope.project,
    MemoryScope.user,
    MemoryScope.global_,
)


# ---------------------------------------------------------------------------
# Record + value types
# ---------------------------------------------------------------------------


class ScopeRef(BaseModel):
    """A resolved ``(scope, scope_key)`` address.

    ``scope_key`` is the empty string for :attr:`MemoryScope.global_` (one
    bucket per tenant) and the relevant id otherwise. Frozen — a record's
    address is immutable once written.
    """

    model_config = ConfigDict(frozen=True)

    scope: MemoryScope
    scope_key: str = ""

    def as_tuple(self) -> tuple[str, str]:
        """``(scope_value, scope_key)`` — the canonical storage key fragment."""
        return (self.scope.value, self.scope_key)


#: Conventional (non-exhaustive, non-enforced) values of
#: :attr:`MemoryProvenance.source_kind`. A host may use its own; these are the
#: ones the core itself writes or that a reader can expect to understand.
MEMORY_SOURCE_KINDS: tuple[str, ...] = (
    "agent",
    "operator",
    "import",
    "system",
    "unknown",
)


class MemoryProvenance(BaseModel):
    """Who or what wrote a memory, and from where.

    Every field is optional and an absent one means *unknown*, never a guess:
    a record carried over from an import without a timestamp keeps
    ``observed_at=None`` rather than acquiring the freshness of the moment it
    was copied, and a record written outside a run names no run. The store
    keeps the value as given; it never fills one in.

    It names the *write that last changed the record*: set on CREATE, replaced
    on MERGE when the merging write carries one, left alone on SKIP. The
    record's ``created_at`` still says when the fact first arrived, and
    ``source_refs`` stays the additive list of evidence it was derived from.
    """

    model_config = ConfigDict(frozen=True)

    source_kind: str = Field(
        default="unknown",
        max_length=64,
        description=(
            "What kind of writer this was: conventionally one of "
            ":data:`MEMORY_SOURCE_KINDS` — the agent through its tool, the "
            "operator by hand, an import, or the system itself."
        ),
    )
    run_id: str | None = Field(default=None, description="The run that wrote it.")
    root_run_id: str | None = Field(
        default=None,
        description="The root of that run's tree, when the writer was a subagent.",
    )
    session_id: str | None = Field(default=None, description="The session the write was made in.")
    subagent_id: str | None = Field(
        default=None, description="The subagent that wrote it, when one did."
    )
    message_id: str | None = Field(
        default=None,
        description="The message the fact was taken from, when the writer knows it.",
    )
    tool_call_id: str | None = Field(
        default=None, description="The tool call that wrote it, when one did."
    )
    observed_at: datetime | None = Field(
        default=None,
        description="When the writer observed the fact. ``None`` = not known.",
    )


class MemoryRecord(BaseModel):
    """A single typed memory.

    A memory is a *typed record with text*, never a raw transcript append
    (the prior-art anti-pattern: "chat history merely appends text linearly,
    generating clutter"). The ``kind`` taxonomy lets retrieval and future
    consolidation reason about the record without re-parsing ``text``.

    Idempotency identity
    --------------------
    Logical identity for the two-stage idempotent write is
    ``(tenant_id, scope, scope_key, kind, normalised(text))`` PLUS the
    similarity gate — i.e. two writes of the *same fact* into the *same scope*
    collapse to one record (CREATE then MERGE/SKIP). ``id`` is the surrogate
    primary key the store assigns; callers do not supply it on write.

    Versioning (drift-guard)
    ------------------------
    ``version`` is a monotonically-increasing integer the store bumps on every
    mutation of the record. A caller that read a record at ``version=N`` may
    pass ``expected_version=N`` to :meth:`IMemory.write` / :meth:`IMemory.delete`;
    the store raises :class:`MemoryConflictError` if the live row has advanced.
    This is the concurrent-write drift-guard — no blind overwrite.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Store-assigned surrogate id (UUID).")
    tenant_id: str
    scope: MemoryScope
    scope_key: str = Field(
        default="",
        description=(
            "Empty for MemoryScope.global_; the user/project/session/agent/"
            "custom id otherwise."
        ),
    )
    kind: str = Field(
        default="fact",
        description=(
            "Typed category. v1 free-form string; the conventional taxonomy "
            "(non-exhaustive, see :data:`MEMORY_KINDS`) is profile / preference "
            "/ fact / decision / entity / reflection / skill / observation / "
            "other. The store does NOT reject unknown kinds (forward-compatible); "
            "callers SHOULD use a value from :data:`MEMORY_KINDS`."
        ),
    )
    text: str = Field(
        description=(
            "The memory body in natural language. This is what FTS/BM25 indexes "
            "and what is injected on recall. Keep it a single self-contained "
            "fact/note; large dumps belong in IWorkspace/IBlobStore, not here."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary structured side-channel (e.g. S-R-O triplet, entity ids, "
            "confidence). NOT a fallback content store; retrieval ranks ``text``, "
            "not ``metadata``."
        ),
    )
    source_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Provenance: canonical source references this memory was derived "
            "from (e.g. ``CompactionSourceRef`` ids, file paths, run ids). Lets "
            "a recalled memory be traced back to evidence and dovetails with the "
            "grounding/citation discipline. Never invented by the store."
        ),
    )
    provenance: MemoryProvenance | None = Field(
        default=None,
        description=(
            "Who wrote the record and from where (see :class:`MemoryProvenance`). "
            "``None`` for a record written by a store or a caller that does not "
            "record provenance — including every record written before the "
            "field existed, which therefore still loads."
        ),
    )
    salience: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Intrinsic importance hint in [0, 1] (0 = unweighted). v1 stores it "
            "and MAY use it as a tie-break in ranking; the v2 decay/reinforce "
            "upgrade folds it into the composite score "
            "(relevance + recency + frequency + salience). Caller-supplied or "
            "0.0; the store never fabricates a non-zero value."
        ),
    )
    embedding: list[float] | None = Field(
        default=None,
        description=(
            "RESERVED for the v2 hybrid-vector upgrade. v1 leaves it ``None`` "
            "and ranks lexically (FTS/BM25 + trgm). Present in the contract so "
            "adding dense retrieval is non-breaking."
        ),
    )
    version: int = Field(
        default=1,
        ge=1,
        description=(
            "Optimistic-concurrency version. Bumped by the store on every "
            "mutation; used by the drift-guard (see class docstring)."
        ),
    )
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = Field(
        default=None,
        description=(
            "Set when the record is returned by :meth:`IMemory.recall` / "
            ":meth:`IMemory.search` (reinforcement signal for v2 decay). v1 MAY "
            "update it best-effort; ``None`` = never recalled."
        ),
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of times this record has been surfaced by recall/search. "
            "Reinforcement signal for the v2 decay/promotion lane. v1 MAY "
            "increment it best-effort."
        ),
    )


# Conventional (non-exhaustive, non-enforced) ``kind`` taxonomy. The store
# accepts any string so the taxonomy can grow without a migration; callers
# SHOULD pick from here for cross-tenant consistency.
MEMORY_KINDS: tuple[str, ...] = (
    "profile",
    "preference",
    "fact",
    "decision",
    "entity",
    "reflection",
    "skill",
    "observation",
    "other",
)


class MemoryWriteDecision(StrEnum):
    """Which branch the two-stage idempotent write took."""

    created = "created"
    """No similar prior record — a new row was inserted."""

    merged = "merged"
    """A similar prior record existed and was UPDATED in place (text/metadata/
    source_refs reconciled, ``version`` bumped). The returned record is the
    surviving merged row."""

    skipped = "skipped"
    """A similar (effectively identical) prior record existed and the write was
    a no-op. The returned record is the pre-existing row, unchanged."""


class MemoryWriteResult(BaseModel):
    """Outcome of :meth:`IMemory.write`.

    The ``decision`` makes the idempotency observable: the runtime can tell the
    *model* "I already knew that (skipped)" / "I updated what I knew (merged)"
    rather than silently double-writing. ``record`` is the surviving row in all
    three branches.
    """

    model_config = ConfigDict(frozen=True)

    record: MemoryRecord
    decision: MemoryWriteDecision
    similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity score of the best prior match considered by the gate "
            "(0.0 when ``created`` with no candidate). Lets callers / tests "
            "inspect why a branch was chosen."
        ),
    )


class MemoryHit(BaseModel):
    """A ranked retrieval result from :meth:`IMemory.recall` / :meth:`search`."""

    model_config = ConfigDict(frozen=True)

    record: MemoryRecord
    score: float = Field(
        description=(
            "Opaque relevance score, higher = better. v1 = lexical rank "
            "(``ts_rank`` + trgm similarity). The scale is NOT stable across "
            "backends/versions — use only for ORDER BY / thresholding within a "
            "single result set, never as an absolute."
        ),
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IMemory(Protocol):
    """Universal, per-tenant, scoped, searchable memory subsystem.

    Every method is **tenant-scoped** (first positional ``tenant_id``) for
    multi-tenant isolation, and **non-fatal by contract**: a backing-store
    outage surfaces as :class:`MemoryStoreUnavailableError`, which the runtime
    treats as "no memory this turn" rather than aborting the run.

    The reference adapter is Postgres-backed (durable, N-pod safe). The
    Protocol is the seam tests fake against (see the in-memory fake in
    ``protocore/tests/contracts/test_memory_contract.py``) and the seam a future
    vector/hybrid backend substitutes behind.
    """

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
        """Idempotently persist a memory; return the write outcome.

        This is the canonical *remember* entrypoint. It is **two-stage** and
        **idempotent**:

        1. **Similarity gate.** Find the most similar existing record in the
           same ``(tenant_id, scope, scope_key, kind)`` bucket. v1 measures
           similarity lexically (FTS/trgm); a v2 backend may use cosine on
           embeddings — the contract only requires *a* similarity in ``[0, 1]``.
        2. **CREATE / MERGE / SKIP.**
           * similarity below ``similarity_threshold`` (or no candidate) →
             **CREATE** a new row (``decision=created``).
           * similarity at/above the threshold but the new text adds
             information (e.g. new ``source_refs`` / richer ``text``) →
             **MERGE** into the existing row in place, bump ``version``
             (``decision=merged``).
           * similarity at/above the threshold and the new text is effectively
             identical → **SKIP**, return the existing row unchanged
             (``decision=skipped``).

        ``similarity_threshold`` defaults (when ``None``) to the value
        the host adapter resolves from
        :attr:`~protocore.contracts.runtime_constants.LoopConstants.memory_write_similarity_threshold`
        — core never hard-codes the number. The per-tenant runtime
        (``executor``/dispatcher) passes the RESOLVED value in per call so
        per-tenant overrides take effect on a pod-wide store instance; the
        adapter's own field is only the final static fallback.

        ``max_records_per_scope`` is the per-call resolved value of
        :attr:`~protocore.contracts.runtime_constants.LoopConstants.memory_max_records_per_scope`
        (``0`` = unbounded). Passed in per call for the same per-tenant
        reason as ``similarity_threshold``: a single pod-wide store serves
        every tenant, so the soft cap must travel with the call, not live in
        the adapter's dataclass default. When ``None`` the adapter falls back
        to its own field.

        **Drift-guard.** When ``expected_version`` is supplied AND the gate
        selects an existing record to MERGE, the store raises
        :class:`MemoryConflictError` if that record's live ``version`` no longer
        equals ``expected_version`` (a concurrent writer won the race). The
        store NEVER blind-overwrites a drifted record. ``expected_version`` is
        ignored on the CREATE branch (there is nothing to conflict with).

        **Idempotency guarantee.** Calling ``write`` twice with the same
        ``(scope, scope_key, kind, text)`` yields exactly one stored record:
        the second call returns ``decision=skipped`` (or ``merged`` if it
        carried new ``source_refs``/``metadata``), never a duplicate row.

        Args:
            tenant_id: Tenant isolation key (required).
            scope: The :class:`MemoryScope` to file under.
            scope_key: The scope's key (empty string for ``global``).
            text: The memory body (FTS-indexed). Must be non-empty after strip.
            kind: Typed category (see :data:`MEMORY_KINDS`).
            metadata: Optional structured side-channel.
            source_refs: Optional provenance references (merged, de-duplicated,
                on the MERGE branch — provenance is additive).
            salience: Intrinsic importance hint in ``[0, 1]``.
            similarity_threshold: Per-tenant resolved dedup threshold (``None``
                = use the adapter's static fallback).
            max_records_per_scope: Per-tenant resolved soft cap on the bucket
                (``0`` = unbounded, ``None`` = use the adapter's fallback).
            expected_version: Optimistic-concurrency guard (see above).

        Returns:
            :class:`MemoryWriteResult` — the surviving record + which branch
            fired + the best-match similarity.

        Raises:
            ValueError: ``text`` empty/whitespace, or ``scope_key`` empty for a
                non-global scope.
            MemoryConflictError: drift-guard tripped on the MERGE branch.
            MemoryStoreUnavailableError: backing store unreachable (non-fatal
                to the caller's run — drop + warn).
        """
        ...

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
        """Context-aware recall — ranked memories relevant to ``query``.

        The *auto-recall by context* entrypoint: given the current turn's text
        (``query``) and the ambient scope keys, return the top-``limit`` records
        ranked by relevance across the requested ``scopes``. This is the method
        the optional auto-recall hook calls before the first LLM turn to inject
        ambient memory (gated by
        :attr:`~protocore.contracts.runtime_constants.LoopConstants.memory_auto_recall_enabled`).

        Scope fan-out:
            * ``scopes=None`` → :data:`DEFAULT_RECALL_SCOPES` (the union of
              session/project/user/global). The host resolver narrows this
              per-tenant (the most isolated configuration = ``(session,)`` only).
            * ``scope_keys`` maps each requested scope to its concrete key
              (e.g. ``{MemoryScope.session: "<sid>", MemoryScope.user: "<uid>"}``).
              A scope present in ``scopes`` but absent from ``scope_keys`` is
              skipped unless it is :attr:`MemoryScope.global_` (whose key is the
              fixed empty string).

        Ranking is the backend's concern (v1 lexical, v2 hybrid+decay) and is
        opaque to the caller — only the ``score`` ordering within the returned
        set is meaningful. Implementations SHOULD record a reinforcement signal
        (``last_accessed_at`` / ``access_count``) on the returned records,
        best-effort and never fatally.

        An empty ``query`` returns the most-recent records in the requested
        scopes (deterministic recency order), so "what do I know here?" works
        without a search term.

        This call is expected to be **time-boxable** by the caller (the runtime
        wraps it in the recall budget); on timeout the runtime proceeds WITHOUT
        memory rather than hanging. The contract itself does not impose a
        timeout — it just must not block unboundedly on its own.

        Returns:
            Up to ``limit`` :class:`MemoryHit` ordered by descending score.
            Empty sequence when nothing matches (never ``None``).

        Raises:
            MemoryStoreUnavailableError: backing store unreachable (non-fatal).
        """
        ...

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
        """Explicit lexical/BM25 search — ranked hits for an agent-issued query.

        Semantically the *agent-facing* counterpart to :meth:`recall`: same
        ranked-hit return shape, but framed as an explicit search the model runs
        (the ``recall``/``search`` tool) rather than the ambient auto-injection.
        Both are backed by the same retrieval; ``search`` defaults to a larger
        ``limit`` and is the method the memory *search tool* binds to.

        **Lexical/BM25 is mandatory here** — this is the path that must find the
        exact ``SKU-12345`` / ``/proc/catalog/.../x.json`` / ``orders`` table
        name a vector-only recall would miss. v1 = ``to_tsvector`` +
        ``plainto_tsquery`` ranked by ``ts_rank`` with a ``pg_trgm`` substring
        fallback for partial tokens.

        Same scope-fan-out + reinforcement + non-fatal semantics as
        :meth:`recall`.

        Returns:
            Up to ``limit`` :class:`MemoryHit` ordered by descending score.

        Raises:
            MemoryStoreUnavailableError: backing store unreachable (non-fatal).
        """
        ...

    async def get(
        self,
        tenant_id: str,
        memory_id: str,
    ) -> MemoryRecord:
        """Fetch one record by id (tenant-scoped).

        Raises:
            MemoryNotFoundError: no such record for this tenant.
            MemoryStoreUnavailableError: backing store unreachable.
        """
        ...

    async def delete(
        self,
        tenant_id: str,
        memory_id: str,
        *,
        expected_version: int | None = None,
    ) -> bool:
        """Delete one record by id (tenant-scoped). The *forget* entrypoint.

        ``expected_version`` activates the drift-guard: if supplied and the live
        record's ``version`` differs, raise :class:`MemoryConflictError` instead
        of deleting (a concurrent writer changed the record the caller intended
        to forget).

        Returns:
            ``True`` if a row was removed, ``False`` if the id was already
            absent (idempotent delete — absence is not an error).

        Raises:
            MemoryConflictError: drift-guard tripped.
            MemoryStoreUnavailableError: backing store unreachable.
        """
        ...

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
        """List records for inspection/administration (NOT ranked retrieval).

        Returns records in deterministic ``updated_at DESC, id`` order, filtered
        by the optional ``scope`` / ``scope_key`` / ``kinds``. This backs the
        admin "inspect memory per tenant+scope" surface and bulk export; it does
        NOT apply relevance ranking (use :meth:`search` for that). Pagination via
        ``limit`` / ``offset``.

        Raises:
            MemoryStoreUnavailableError: backing store unreachable.
        """
        ...


@runtime_checkable
class IProvenanceMemory(IMemory, Protocol):
    """An :class:`IMemory` that also keeps who wrote each record.

    A separate contract rather than a new keyword on :meth:`IMemory.write`,
    because the core calls a store by keyword: a keyword added to the base
    contract would make every store written before it fail its conformance
    suite and raise ``TypeError`` on the first write. A store opts in by
    accepting ``provenance``; a caller asks for it only when it knows the
    store has (the memory tools check at construction, see
    :class:`~protocore.tools.memory.RememberTool`).
    """

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
        provenance: MemoryProvenance | None = None,
    ) -> MemoryWriteResult:
        """:meth:`IMemory.write`, keeping ``provenance`` on the surviving record.

        * CREATE stores ``provenance`` as given (``None`` stays ``None``).
        * MERGE replaces the record's provenance with ``provenance`` when one
          is given, and keeps the old one when not: the record names the last
          write that changed it.
        * SKIP leaves the record untouched. A different ``provenance`` alone
          is not new information — the same fact restated by another run must
          stay a SKIP, or every re-save would bump ``version``.
        """
        ...


# ---------------------------------------------------------------------------
# Write policy seam (classify or refuse a write before it lands)
# ---------------------------------------------------------------------------


class MemoryWriteRequest(BaseModel):
    """A memory write as the ``Remember`` tool is about to make it.

    Handed to an :class:`IMemoryWritePolicy` after the content scan and the
    scope rules have passed, so the policy decides about the write that would
    actually land — the resolved scope and key, not what the model asked for —
    and never about one the core would have refused anyway.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    scope: MemoryScope
    scope_key: str
    text: str
    kind: str
    kind_given: bool = Field(
        description=(
            "Whether the caller named ``kind``. ``False`` means ``kind`` is the "
            "default, so a policy that infers a category knows it may replace it."
        ),
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The fields the tool's input model adds to the core's, by name, as "
            "the model sent them (``None`` for one it left out). Model-authored "
            "and already content-scanned, never trusted for anything else."
        ),
    )
    provenance: MemoryProvenance | None = Field(
        default=None,
        description="Who is writing, when the tool records provenance.",
    )


class MemoryWriteVerdict(BaseModel):
    """What an :class:`IMemoryWritePolicy` decided about one write.

    Either a refusal, whose ``message`` is the whole of what the model reads
    (so it should say what was refused, why, and what to do instead), or an
    allowance that may amend the write. Amendments come from the host and are
    trusted as such: they are not content-scanned again.
    """

    model_config = ConfigDict(frozen=True)

    refused: bool = False
    message: str = Field(
        default="",
        description="The refusal the model reads. Required when ``refused``.",
    )
    kind: str | None = Field(
        default=None,
        description="A replacement ``kind`` for the write; ``None`` keeps the request's.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata to store with the write.",
    )
    source_refs: list[str] = Field(
        default_factory=list,
        description="Source references to store with the write.",
    )
    salience: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="A salience for the write; ``None`` keeps the default.",
    )
    result_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Merged into the tool result's metadata, refused or not, for the "
            "host's own readers of the result."
        ),
    )

    @model_validator(mode="after")
    def _refusal_says_why(self) -> Self:
        # A bare refusal reads to the model as "blocked", and a model told
        # only that tends to try again in other words.
        if self.refused and not self.message.strip():
            raise ValueError("a refused memory write needs a message for the model")
        return self

    @classmethod
    def allow(
        cls,
        *,
        kind: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_refs: Sequence[str] | None = None,
        salience: float | None = None,
        result_metadata: dict[str, Any] | None = None,
    ) -> MemoryWriteVerdict:
        """Let the write land, amended as given."""
        return cls(
            kind=kind,
            metadata=dict(metadata or {}),
            source_refs=list(source_refs or ()),
            salience=salience,
            result_metadata=dict(result_metadata or {}),
        )

    @classmethod
    def refuse(
        cls, message: str, *, result_metadata: dict[str, Any] | None = None
    ) -> MemoryWriteVerdict:
        """Refuse the write; ``message`` is what the model reads."""
        return cls(refused=True, message=message, result_metadata=dict(result_metadata or {}))


@runtime_checkable
class IMemoryWritePolicy(Protocol):
    """The host's say over a memory write before it reaches the store.

    The place for what is the host's own and not the core's: a taxonomy to
    classify a note into (possibly by asking a model), subjects the operator
    has not allowed in long-term memory, a field the host's input model adds
    that has to become metadata. Optional: the ``Remember`` tool without one
    writes exactly as it always did.

    Asked once per write, after the content scan and scope resolution and
    before the store. A raise is not a refusal: it propagates like any tool
    failure, so a policy that wants the model to read a refusal returns
    :meth:`MemoryWriteVerdict.refuse`.
    """

    async def review(self, request: MemoryWriteRequest) -> MemoryWriteVerdict:
        """Decide about ``request``: allow it (possibly amended) or refuse it."""
        ...


# ---------------------------------------------------------------------------
# Untrusted-content scan seam (memory text is untrusted input)
# ---------------------------------------------------------------------------


#: Marker that prefixes a recalled/snapshotted memory entry whose text tripped
#: the content scanner. The flagged entry's body is replaced by a placeholder
#: built from this marker so the raw (potentially injecting) text never
#: re-enters model context, while the model still SEES that something was
#: filtered (so it does not silently rely on a blocked fact).
MEMORY_BLOCKED_PLACEHOLDER: str = "[BLOCKED: memory entry filtered"


def blocked_memory_placeholder(text: str, *, reason: str | None = None) -> str:
    """Return the [BLOCKED] replacement for a flagged memory entry.

    The original ``text`` is **never** echoed back (echoing it would defeat the
    point — the injection payload would still reach the prompt). Only the marker
    and an optional short ``reason`` (a threat-pattern id / message, NOT the raw
    text) are returned, so a poisoned entry can be swapped for this placeholder
    before it is rendered into a recall result or a system-prompt snapshot.

    Args:
        text: The flagged entry text (used only to confirm there is content to
            replace; not included in the output).
        reason: Optional short threat descriptor to surface to the model.

    Returns:
        A safe placeholder string beginning with :data:`MEMORY_BLOCKED_PLACEHOLDER`.
    """
    _ = text  # intentionally not echoed (would re-inject the payload)
    if reason:
        return f"{MEMORY_BLOCKED_PLACEHOLDER}: {reason}]"
    return f"{MEMORY_BLOCKED_PLACEHOLDER}]"


@runtime_checkable
class IMemoryContentScanner(Protocol):
    """Seam for scanning **untrusted** memory text for injection/exfil content.

    Memory ``text`` is untrusted input (model-authored, sister-session-authored,
    or written by a compromised tool) that re-enters model context on recall and
    — via the optional auto-recall hook — the system prompt. The runtime SHOULD
    scan it at BOTH boundaries:

    * at **write**: a flagged write is refused with a corrective error so poison
      never reaches the store; and
    * at **recall / injection**: a flagged entry that is already on disk (a
      poisoned-on-disk record the write-side scan never saw) is replaced with a
      :func:`blocked_memory_placeholder` rather than rendered verbatim.

    Core defines only the seam; the concrete pattern set lives in the host
    adapter (so the pattern library is configurable + updatable without a core
    release). The seam is OPTIONAL: a caller that supplies no scanner gets no
    scan (pure-core unit tests and minimal callers are unaffected). The fence
    around recalled memory is defense-in-depth, NOT a substitute for the scan.
    """

    def scan(self, text: str) -> str | None:
        """Scan ``text``; return a short threat message if it is unsafe, else None.

        The returned message is a threat descriptor (e.g. a matched pattern id),
        suitable for surfacing to the model or embedding in a placeholder — it
        MUST NOT echo back the offending text. ``None`` means "clean".
        """
        ...


__all__ = [
    "DEFAULT_RECALL_SCOPES",
    "MEMORY_BLOCKED_PLACEHOLDER",
    "MEMORY_KINDS",
    "MEMORY_SOURCE_KINDS",
    "IMemory",
    "IMemoryContentScanner",
    "IMemoryWritePolicy",
    "IProvenanceMemory",
    "MemoryConflictError",
    "MemoryError",
    "MemoryHit",
    "MemoryNotFoundError",
    "MemoryProvenance",
    "MemoryRecord",
    "MemoryScope",
    "MemoryStoreUnavailableError",
    "MemoryWriteDecision",
    "MemoryWriteRequest",
    "MemoryWriteResult",
    "MemoryWriteVerdict",
    "ScopeRef",
    "blocked_memory_placeholder",
]
