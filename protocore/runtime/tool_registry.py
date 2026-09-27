"""Concrete :class:`ToolRegistry` — in-core implementation of :class:`IToolRegistry`.

Implements the 3-layer tool surface (policy → clipping →
progressive discovery) using the BM25F retrieval of
:mod:`protocore.runtime.tool_retrieval`.

This is the **default** registry shipped by core. A host may substitute a
database-backed variant that adds tenant overrides and cross-process cache
invalidation; that adapter satisfies the same :class:`IToolRegistry` Protocol.

Thread / async safety
---------------------
* In-memory state is guarded by a single :class:`threading.RLock` — the
 registry is mutated from background tasks (tool installation hooks)
 AND queried from the per-turn driver async generator; both must
 see a consistent catalogue.
* Snapshot reads (``list_all`` / ``compute_effective_surface``) take a
 copy under the lock and operate lock-free thereafter — keeps the hot
 per-turn path cheap.
* The retrieval index is built outside both locks and cached per catalogue
 generation (see :meth:`ToolRegistry._index_for`), so a registration never
 waits for an index build and a query never rebuilds one it can reuse.

Tenant scoping
--------------
* Single global namespace in core. The host adapter introduces
 per-tenant override semantics; this baseline simply honours the
 :class:`ToolVisibilityPolicy` whitelist passed by the loop.
* The ``filter_by_whitelist`` helper extends the Protocol by exposing
 registry-level resolution from a flat name list — used by subagent
 dispatch before the per-turn surface is computed.
"""
from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from typing import Final, Literal

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import (
    IToolRegistry,
    ToolGroup,
    ToolVisibilityPolicy,
    policy_admits,
)
from protocore.contracts.tool_retrieval import (
    IToolRetriever,
    RetrievalSettings,
    ToolDocument,
    reciprocal_rank_fusion,
)
from protocore.contracts.tools import Tool
from protocore.contracts.types import ToolDefinition
from protocore.runtime.tool_retrieval import (
    AnalyzedCatalogue,
    Lexicon,
    ToolIndex,
    parameter_text,
)

# Indexes kept per catalogue generation, one per distinct settings. A pod
# normally serves one settings value; a few tenants with their own weights fit,
# and the bound keeps a stream of distinct weights from growing the cache
# without limit. Not a tunable: it bounds memory, not behaviour.
_MAX_CACHED_INDEXES: Final[int] = 4


def _search_hint(tool: Tool) -> str:
    """The tool's optional multilingual retrieval hint (``search_hint``).

    Read via ``getattr`` so plain core tools without the ClassVar stay
    supported. The hint joins the DISCOVERY corpus only (never the wire
    description) — see :class:`~protocore.contracts.tool_retrieval.ToolDocument`.
    """
    raw = getattr(tool, "search_hint", "")
    return raw if isinstance(raw, str) else ""


def tool_document(tool: Tool) -> ToolDocument:
    """The searchable form of ``tool``: its wire text plus hint and parameters."""
    definition = tool.definition
    return ToolDocument(
        name=tool.name,
        description=definition.description,
        search_hint=_search_hint(tool),
        parameters=parameter_text(definition.parameters.properties),
    )


class ToolRegistry(IToolRegistry):
    """In-core, thread-safe, BM25-ranked tool catalogue.

 Implements :class:`IToolRegistry`. Lifecycle:

 * Constructed once per executor pod (or per test).
 * Tools registered at startup via :meth:`register` (idempotent on
 ``tool.name`` — re-registration overwrites the previous binding).
 * Looked up per-turn via :meth:`get` (O(1) dict access).
 * Surfaced per-turn via :meth:`compute_effective_surface` (BM25F over
 the policy-filtered subset).

 Sort invariant: every public list result, and the advertised surface,
 is sorted by ``Tool.name`` ascending for KV-prefix-cache stability
 across turns. :meth:`search` is the exception: it returns rank order,
 because which hit is best is the information its caller asked for.

 ``lexicon`` is the Russian-to-English query expansion: ``"bundled"``
 (the default) loads the one shipped with the package on first use, a
 :class:`Lexicon` supplies a host's own, ``None`` turns expansion off.
 ``retriever`` is an optional host ranker fused with the lexical ranking
 by reciprocal rank fusion.
 """

    def __init__(
        self,
        tools: Iterable[Tool] | None = None,
        *,
        lexicon: Lexicon | Literal["bundled"] | None = "bundled",
        retriever: IToolRetriever | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._tools: dict[str, Tool] = {}
        self._groups: dict[str, ToolGroup] = {}
        # Bumped by every register/unregister; an index built for an older
        # generation is never used again.
        self._generation = 0
        self._retriever = retriever
        self._lexicon_choice = lexicon
        self._lexicon: Lexicon | None = lexicon if isinstance(lexicon, Lexicon) else None
        self._index_lock = threading.Lock()
        self._catalogue: tuple[int, AnalyzedCatalogue] | None = None
        self._indexes: dict[RetrievalSettings, ToolIndex] = {}
        # The settings used when a caller passes none — the defaults of the
        # constants model, built once rather than per query.
        self._default_settings = RetrievalSettings.from_constants(LoopConstants())
        if tools is not None:
            for tool in tools:
                self.register(tool)

    # ------------------------------------------------------------------
    # IToolRegistry — register / get
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a tool; idempotent on :attr:`Tool.name`.

        Re-registering the same name overwrites the existing entry —
        used by tenant-override adapters that swap in a tenant-specific
        implementation at admission time.
        """
        with self._lock:
            self._tools[tool.name] = tool
            self._generation += 1

    def unregister(self, name: str) -> None:
        """Remove a tool by name. Idempotent — no error if absent."""
        with self._lock:
            if self._tools.pop(name, None) is not None:
                self._generation += 1

    def get(self, name: str) -> Tool | None:
        """Fetch tool by name; ``None`` if not registered."""
        with self._lock:
            return self._tools.get(name)

    # ------------------------------------------------------------------
    # IToolRegistry — groups
    # ------------------------------------------------------------------

    def declare_group(
        self,
        name: str,
        description: str,
        *,
        dynamic: bool = False,
        prefix: str = "",
    ) -> None:
        """Declare a tool group, replacing any earlier declaration of ``name``.

        A group changes no search result and no advertised definition, so
        declaring one does not bump the catalogue generation.
        """
        if not name:
            raise ValueError("a tool group needs a name")
        with self._lock:
            self._groups[name] = ToolGroup(
                name=name, description=description, dynamic=dynamic, prefix=prefix
            )

    def undeclare_group(self, name: str) -> None:
        """Forget a group; idempotent.

        Like declaring one, this leaves the catalogue generation alone: groups
        change no search result, and the loop's deferral decision is keyed on
        the declared groups themselves, so it is made again on the next request.
        """
        with self._lock:
            self._groups.pop(name, None)

    def tool_groups(self) -> Sequence[ToolGroup]:
        """Every declared group, sorted by name."""
        with self._lock:
            groups = list(self._groups.values())
        return sorted(groups, key=lambda group: group.name)

    # ------------------------------------------------------------------
    # IToolRegistry — listing / filtering
    # ------------------------------------------------------------------

    def list_all(self) -> Sequence[Tool]:
        """All registered tools, sorted by :attr:`Tool.name` ASC."""
        with self._lock:
            tools = list(self._tools.values())
        return sorted(tools, key=lambda t: t.name)

    def list_for_tenant(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
    ) -> Sequence[Tool]:
        """List tools visible to a tenant after the visibility policy filter.

 Order: sorted by ``Tool.name`` ASC (sort invariant —
 cache-prefix stability).

 Note: ``tenant_id`` is accepted for Protocol compliance but
 unused in this baseline (single-namespace registry);
 the host PG-backed variant uses it for tenant scoping.
 """
        del tenant_id  # baseline: single namespace
        with self._lock:
            all_tools = list(self._tools.values())

        if policy.visible:
            filtered = [t for t in all_tools if t.name in policy.visible]
        else:
            filtered = list(all_tools)
        filtered = [t for t in filtered if t.name not in policy.blocked]
        return sorted(filtered, key=lambda t: t.name)

    def filter_by_whitelist(self, names: Sequence[str]) -> Sequence[Tool]:
        """Resolve a list of names to :class:`Tool` instances.

 Used by subagent dispatch before building the per-turn
 surface: the subagent's ``tool_whitelist_json`` is a flat name
 list — this method resolves it against the catalogue, dropping
 unknown names silently (registration is the source of truth).

 Result is sorted by name ASC (sort invariant).
 """
        with self._lock:
            tools: list[Tool] = [self._tools[n] for n in names if n in self._tools]
        return sorted(tools, key=lambda t: t.name)

    # ------------------------------------------------------------------
    # IToolRegistry — 3-layer surface (policy → clipping → retrieval)
    # ------------------------------------------------------------------

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
        """The ``top_k`` tools that best match ``query``, best first.

        ``whitelist`` (if provided) narrows the candidate pool — used by
        the :class:`ToolSearch` tool to restrict matches to the
        subagent's allowed surface. ``policy`` (if provided) applies the
        per-run visibility contract (:func:`policy_admits` — ``blocked``
        always denied; a non-empty ``visible`` admits only
        ``visible | pinned | forced_pinned``) so discovery can never return
        a schema the dispatch gate would refuse.

        Results are in RANK order, ties broken by name. This used to be
        re-sorted by name, which threw away the one thing a caller cannot
        work out for itself — which hit fits best — and a model reading a
        name-sorted list picked the alphabetically first plausible tool.
        Byte-stability matters for the advertised surface, not for a
        search result the model reads once.

        ``retrieval`` is the run's :class:`RetrievalSettings`; without one
        the constants model's defaults apply. An empty query returns the
        first ``top_k`` candidates by name (deterministic).
        """
        del tenant_id  # baseline: single namespace
        if top_k <= 0:
            return []
        generation, tools = self._snapshot()
        pool = list(tools.values())
        if whitelist is not None:
            allow = frozenset(whitelist)
            pool = [t for t in pool if t.name in allow]
        if policy is not None:
            pool = [t for t in pool if policy_admits(policy, t.name)]
        pool.sort(key=lambda t: t.name)

        if not query.strip():
            return pool[:top_k]
        allowed = frozenset(t.name for t in pool)
        ranked = self._rank(query, generation, tools, allowed, top_k, retrieval)
        by_name = {t.name: t for t in pool}
        return [by_name[name] for name in ranked]

    def compute_effective_surface(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
        *,
        query: str = "",
        top_k: int | None = None,
        retrieval: RetrievalSettings | None = None,
    ) -> Sequence[ToolDefinition]:
        """3-layer filter: policy → clipping → progressive discovery.

 Returns the ranked :class:`ToolDefinition` list ready to be
 embedded in the LLM context (system-prompt tools section).

 Implementation:

 1. Apply :class:`ToolVisibilityPolicy` (visible + blocked), then
 re-admit ``forced_pinned`` tools that ``blocked`` did NOT deny —
 the core tool-surface floor must survive a tenant ``visible``
 whitelist, not just the retrieval clip (see
 :meth:`_floored_visible_tools`).
 2. If ``top_k`` is ``None``, or the tools that are not pinned number
 no more than ``top_k``: return the floored set sorted by name (no
 retrieval).
 3. Otherwise: pinned (policy.pinned) + forced_pinned + ``always_load``
 tools always included, and ``top_k`` more go to the best-ranked of
 the rest for ``query`` (the same ranking as :meth:`search`, fallback
 stage included). A query that matches nothing leaves pinned tools
 only; an empty one, the pinned tools plus the first others by name.

 Clipping per message is opt-in and not recommended: it re-ranks the
 surface on every user message, which changes the tool list and with
 it the provider's prompt cache, and a message in one language
 against descriptions in another finds too little. Deferring whole
 groups behind ToolSearch (:mod:`protocore.runtime.tool_deferral`)
 is the supported way to keep a large catalogue off the surface.

 ``always_load``: a tool whose class sets ``always_load = True``
 (e.g. ``ToolSearch``) is ALWAYS part of the advertised surface,
 independent of its score and ``top_k`` — pin semantics, but
 class-driven instead of policy-driven. Precedence: the visibility
 policy still wins — ``blocked`` denies an always-load tool outright,
 and a non-empty ``visible`` whitelist that omits it keeps it out
 (only ``forced_pinned`` re-admits past the whitelist). It survives
 ONLY the layer-2/3 clip.

 The result is sorted by name whichever layer ran: the ranking
 drives *selection*, but the LLM context must stay byte-deterministic
 for the prefix cache.
 """
        visible_tools = self._floored_visible_tools(tenant_id, policy)

        # Layer 2 + 3: clipping + retrieval
        if top_k is None:
            return [t.definition for t in visible_tools]

        # Layer-3 pins + the core floor. ``forced_pinned`` is already in the
        # pool (``_floored_visible_tools`` re-admitted it past the ``visible``
        # whitelist), but it must ALSO bypass the clip: a prompt that matches
        # none of the floor's words would otherwise drop it at the top-K cut.
        # ``pinned`` is the per-session ToolSearch/progressive-discovery set.
        # ``always_load`` names are the class-driven floor (policy-admitted
        # only — derived from the post-policy pool, so ``blocked``/whitelist
        # still win). Union (not replace) so all three survive the clip.
        always_load_names = frozenset(
            t.name for t in visible_tools if bool(getattr(t, "always_load", False))
        )
        pinned_names = (
            frozenset(policy.pinned) | policy.forced_pinned | always_load_names
        )
        chosen = [t for t in visible_tools if t.name in pinned_names]
        others = [t for t in visible_tools if t.name not in pinned_names]
        # ``top_k`` counts retrieved tools only. It used to count the pinned
        # ones too, so a floor of fourteen pinned tools under the default of
        # twelve left no room at all and the clip retrieved nothing, whatever
        # the message said.
        if len(others) <= top_k:
            return [t.definition for t in visible_tools]
        remaining = top_k
        if remaining and others:
            if query.strip():
                # ``search_hint`` joins this corpus too, not only ToolSearch's:
                # a tool's Russian hint must surface it in the advertised
                # payload, not only through progressive discovery.
                generation, tools = self._snapshot()
                allowed = frozenset(t.name for t in others)
                ranked = set(self._rank(query, generation, tools, allowed, remaining, retrieval))
                chosen.extend(t for t in others if t.name in ranked)
            else:
                # No retrieval signal (autonomous batch, synthetic resume): the
                # first others by name rather than none — a zero-tool surface
                # leaves the model unable to act.
                chosen.extend(others[:remaining])

        chosen.sort(key=lambda t: t.name)
        return [t.definition for t in chosen]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _snapshot(self) -> tuple[int, dict[str, Tool]]:
        """The catalogue and its generation, read together under the lock."""
        with self._lock:
            return self._generation, dict(self._tools)

    def _rank(
        self,
        query: str,
        generation: int,
        tools: dict[str, Tool],
        allowed: frozenset[str],
        limit: int,
        retrieval: RetrievalSettings | None,
    ) -> list[str]:
        """Names from ``allowed``, best first, at most ``limit``.

        Lexical BM25F first; if it scores nothing, the normalized fallback
        (partial names, unusual inflections). With a host retriever, its
        ranking and the lexical one are fused by reciprocal rank fusion —
        over the full rankings, so a tool either ranker places just below
        the cut can still be lifted into it by the other.
        """
        settings = retrieval if retrieval is not None else self._default_settings
        index = self._index_for(generation, tools, settings)
        depth = limit if self._retriever is None else len(allowed)
        ranked = index.rank(query, depth, allowed)
        if not ranked:
            ranked = index.fallback(query, depth, allowed)
        if self._retriever is None:
            return ranked
        documents = [document for document in index.documents if document.name in allowed]
        host_ranked = [
            name for name in self._retriever.rank(query, documents, len(documents)) if name in allowed
        ]
        return reciprocal_rank_fusion(
            [ranked, host_ranked],
            limit=limit,
            rank_constant=settings.fusion_rank_constant,
        )

    def _index_for(
        self,
        generation: int,
        tools: dict[str, Tool],
        settings: RetrievalSettings,
    ) -> ToolIndex:
        """The index of this catalogue generation under ``settings``.

        Analysis (tokenising and stemming every tool) happens once per
        generation; scoring constants once per generation and settings.
        Builds run outside the locks: two threads that miss together both
        build, and one result wins — cheaper than making every query wait
        behind a lock for the rare rebuild.
        """
        with self._index_lock:
            cached = self._catalogue
            if cached is not None and cached[0] == generation:
                index = self._indexes.get(settings)
                if index is not None:
                    return index
                catalogue = cached[1]
            else:
                catalogue = None
            lexicon = self._lexicon
        if lexicon is None and self._lexicon_choice == "bundled" and settings.lexicon_weight > 0:
            lexicon = Lexicon.bundled()
        if catalogue is None:
            catalogue = AnalyzedCatalogue(tool_document(tool) for tool in tools.values())
        index = ToolIndex(catalogue, settings, lexicon)
        with self._index_lock:
            if self._lexicon is None and lexicon is not None:
                self._lexicon = lexicon
            current = self._catalogue
            if current is None or current[0] < generation:
                self._catalogue = (generation, catalogue)
                self._indexes = {}
            elif current[0] > generation:
                # A newer catalogue was indexed while this one was built;
                # serve this query from its own snapshot but keep the newer.
                return index
            if len(self._indexes) >= _MAX_CACHED_INDEXES:
                self._indexes.pop(next(iter(self._indexes)))
            self._indexes[settings] = index
        return index

    def _floored_visible_tools(
        self,
        tenant_id: str,
        policy: ToolVisibilityPolicy,
    ) -> list[Tool]:
        """Policy-visible tools with the ``forced_pinned`` floor re-admitted.

 :meth:`list_for_tenant` applies the ``visible`` whitelist + ``blocked``
 deny. But ``forced_pinned`` (the core tool-surface floor) must be
 present *regardless of the whitelist* — a tenant ``visible`` set that
 omits the six core file tools would otherwise recreate the cause-#3
 collapse for that tenant, exactly what the floor exists to prevent.
 So we re-admit any
 registered ``forced_pinned`` tool that ``blocked`` did NOT explicitly
 deny — ``blocked`` still wins (an operator can hard-deny a tool even
 against the floor). Result is name-ASC (cache-prefix stable).

 ``policy.pinned`` (the ToolSearch / progressive-discovery pin set) is
 re-admitted the SAME way:
 the dispatch permission gate and :func:`policy_admits` treat the
 allowed set under a non-empty ``visible`` whitelist as
 ``visible | pinned | forced_pinned``, so the advertised surface must
 agree — otherwise a tool the model just pinned via ToolSearch would
 be dispatch-callable yet vanish from the next turn's schema.
 """
        visible = list(self.list_for_tenant(tenant_id, policy))
        readmittable = (frozenset(policy.pinned) | policy.forced_pinned)
        if not readmittable:
            return visible
        present = {t.name for t in visible}
        missing = readmittable - present - policy.blocked
        if not missing:
            return visible
        with self._lock:
            readmit = [
                self._tools[name] for name in missing if name in self._tools
            ]
        if not readmit:
            return visible
        merged = visible + readmit
        merged.sort(key=lambda t: t.name)
        return merged

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        with self._lock:
            return name in self._tools


__all__ = ["ToolRegistry", "tool_document"]
