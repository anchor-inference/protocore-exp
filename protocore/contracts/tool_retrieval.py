"""Tool retrieval — the searchable form of a tool, and the seam for a host ranker.

The core ranks tools lexically (:mod:`protocore.runtime.tool_retrieval`). A host
that wants a different signal — an embedding model, a learned reranker — does
not replace that engine; it supplies an :class:`IToolRetriever`, and the
registry fuses the two rankings with :func:`reciprocal_rank_fusion`. Fusing by
rank rather than by score is what makes the two combinable at all: a BM25 score
and a cosine similarity are on unrelated scales, while a rank means the same
thing in both.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self, runtime_checkable

from protocore.contracts.runtime_constants import LoopConstants


@dataclass(frozen=True, slots=True)
class ToolDocument:
    """Everything retrieval may read about one tool.

    ``search_hint`` and ``parameters`` are for finding the tool only; neither
    is ever shown to the model as part of the tool's schema.
    """

    name: str
    description: str
    search_hint: str = ""
    parameters: str = ""
    """Parameter names and parameter descriptions, flattened to one text."""


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """The tunable part of lexical tool retrieval, read from :class:`LoopConstants`.

    Frozen and hashable on purpose: the registry keys its prebuilt index by
    these values, so two runs with different weights never share an index
    built for the other's.
    """

    name_weight: float
    search_hint_weight: float
    summary_weight: float
    description_weight: float
    parameters_weight: float
    bm25_k1: float
    bm25_b: float
    lexicon_weight: float
    fusion_rank_constant: int

    @classmethod
    def from_constants(cls, rc: LoopConstants) -> Self:
        """The settings a run's constants snapshot asks for."""
        return cls(
            name_weight=rc.tool_retrieval_name_weight,
            search_hint_weight=rc.tool_retrieval_search_hint_weight,
            summary_weight=rc.tool_retrieval_summary_weight,
            description_weight=rc.tool_retrieval_description_weight,
            parameters_weight=rc.tool_retrieval_parameters_weight,
            bm25_k1=rc.tool_retrieval_bm25_k1,
            bm25_b=rc.tool_retrieval_bm25_b,
            lexicon_weight=rc.tool_retrieval_lexicon_weight,
            fusion_rank_constant=rc.tool_retrieval_fusion_rank_constant,
        )


@runtime_checkable
class IToolRetriever(Protocol):
    """A host-supplied tool ranker, fused with the core's lexical ranking.

    ``rank`` is synchronous because it runs inside the per-turn surface
    computation, which is synchronous. A ranker that needs a model should
    embed the catalogue ahead of time, keyed by document, and do no more per
    call than embed the query and compare.
    """

    def rank(self, query: str, documents: Sequence[ToolDocument], limit: int) -> Sequence[str]:
        """Names of the best ``limit`` documents for ``query``, best first.

        Only names of the given ``documents`` count; any other name is ignored.
        Returning fewer than ``limit`` names, or none, is allowed: a tool this
        ranker does not return simply gets no support from it in the fusion.
        """
        ...


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    limit: int,
    rank_constant: int,
) -> list[str]:
    """Fuse several best-first rankings into one.

    A name at 1-based rank ``r`` in a ranking scores ``1 / (rank_constant + r)``
    from it, and its scores are summed across rankings. Ties go to the name
    that sorts first, so the fused order does not depend on which ranking
    happened to be listed first. A name repeated within one ranking counts at
    its best rank only.
    """
    if limit <= 0:
        return []
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for position, name in enumerate(ranking, start=1):
            if name in seen:
                continue
            seen.add(name)
            scores[name] = scores.get(name, 0.0) + 1.0 / (rank_constant + position)
    return sorted(scores, key=lambda name: (-scores[name], name))[:limit]


__all__ = [
    "IToolRetriever",
    "RetrievalSettings",
    "ToolDocument",
    "reciprocal_rank_fusion",
]
