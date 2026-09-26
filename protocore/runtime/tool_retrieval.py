"""Lexical tool retrieval: BM25F over tool fields, with Russian-to-English expansion.

How a tool is found
-------------------
Each tool is indexed as five fields — its name, its ``search_hint``, the first
sentence of its description, the rest of the description, and its parameter
names and descriptions — each run through :func:`protocore.runtime.text_analysis.analyze`
(identifier splitting, folding, stopwords, stemming). A query is scored with
BM25F: a term's frequency in each field is length-normalised against that
field's average, weighted, summed across fields, and saturated once with ``k1``;
``idf`` is computed over whole documents. Field weights, ``k1`` and ``b`` are
:class:`~protocore.contracts.tool_retrieval.RetrievalSettings`, read from the
run's constants.

A query in Russian against a catalogue described in English shares no words
with it. The :class:`Lexicon` bridges that: each Russian stem of the query also
looks up the English stems it translates to, at a reduced weight. The bundled
lexicon is generic developer vocabulary, translated from the English of tool
descriptions — never from queries — so it is not fitted to any one catalogue.

When the query scores nothing at all, :func:`normalized_fallback_match` re-ranks
by loose substring and prefix overlap, which still finds a partial tool name or
an unusual inflection.

Why the index is built once
---------------------------
Tokenising and stemming a catalogue of hundreds of tools costs far more than
scoring one query against it, so :class:`ToolIndex` is built once per catalogue
version and settings, and a query only walks the postings of its own terms. The
index is owned by the registry instance that built it — nothing here keeps
state at module level.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Final

from protocore.contracts.tool_retrieval import RetrievalSettings, ToolDocument
from protocore.runtime.stemmers import stem
from protocore.runtime.text_analysis import STOPWORDS, content_words, fold

#: Package data: English word -> the Russian words and phrases for it.
BUNDLED_LEXICON_RESOURCE: Final[str] = "tool_retrieval_lexicon.json"

# A pasted log or a long message can carry hundreds of distinct words; scoring
# all of them costs time and lets common words drown the few that name a tool.
# Past this many distinct terms, only the rarest (highest idf) are kept — the
# approach of Elasticsearch's more_like_this. Not a tunable: it bounds cost
# rather than shaping relevance, and ordinary queries never reach it.
_MAX_QUERY_TERMS: Final[int] = 25

# Russian words this short are mostly prepositions and particles the stopword
# list missed; mapping them to English would expand to noise.
_MIN_LEXICON_WORD_LENGTH: Final[int] = 3

# The fallback ignores tokens shorter than this on both sides, and matches a
# shared prefix of at least this many letters (Russian inflections share long
# stems: "память"/"памяти").
_FALLBACK_MIN_TOKEN_LENGTH: Final[int] = 3
_FALLBACK_MIN_PREFIX_LENGTH: Final[int] = 4

# name, search_hint, first sentence of the description, the rest of it, parameters.
_FIELD_COUNT: Final[int] = 5

_SENTENCE_END: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")
_WORD: Final[re.Pattern[str]] = re.compile(r"\w+")


def split_summary(description: str) -> tuple[str, str]:
    """The first sentence of ``description``, and everything after it."""
    parts = _SENTENCE_END.split(description.strip(), maxsplit=1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def parameter_text(properties: Mapping[str, object]) -> str:
    """Parameter names and their descriptions from a JSON-Schema ``properties`` map."""
    words: list[str] = []
    for name, spec in properties.items():
        words.append(name)
        if isinstance(spec, Mapping):
            description = spec.get("description")
            if isinstance(description, str) and description:
                words.append(description)
    return " ".join(words)


class Lexicon:
    """Russian stem -> the English stems a Russian query word expands to.

    Built from translations keyed the other way round (English word -> Russian
    words), because that is how such a list is written: take the vocabulary of
    tool descriptions and say how a Russian speaker refers to each word. Both
    sides are stemmed with the retrieval analyser, so the lexicon matches
    whatever inflection the query uses.
    """

    __slots__ = ("_expansions",)

    def __init__(self, expansions: Mapping[str, Iterable[str]]) -> None:
        self._expansions: dict[str, tuple[str, ...]] = {
            source: tuple(sorted(set(targets))) for source, targets in expansions.items() if targets
        }

    @classmethod
    def from_translations(cls, translations: Mapping[str, Sequence[str]]) -> Lexicon:
        """Build from ``{english word: [russian word or phrase, ...]}``."""
        expansions: dict[str, set[str]] = {}
        for english, russian_phrases in translations.items():
            english_stems = {stem(word) for word in content_words(english)}
            for phrase in russian_phrases:
                for word in content_words(phrase):
                    if len(word) < _MIN_LEXICON_WORD_LENGTH:
                        continue
                    source = stem(word)
                    # A loanword spelled the same after stemming would only
                    # expand to itself.
                    expansions.setdefault(source, set()).update(english_stems - {source})
        return cls(expansions)

    @classmethod
    def bundled(cls) -> Lexicon:
        """The Russian-to-English lexicon shipped with the package.

        Read and built on every call; the caller keeps the result. The tool
        registry builds it once, on its first query.
        """
        text = resources.files("protocore.runtime").joinpath(BUNDLED_LEXICON_RESOURCE).read_text(encoding="utf-8")
        translations = json.loads(text)
        if not isinstance(translations, dict):
            raise ValueError(f"{BUNDLED_LEXICON_RESOURCE} must hold a JSON object")
        return cls.from_translations(translations)

    def expand(self, term: str) -> tuple[str, ...]:
        """English stems ``term`` expands to; empty when it has none."""
        return self._expansions.get(term, ())

    def __len__(self) -> int:
        return len(self._expansions)


@dataclass(frozen=True, slots=True)
class _Posting:
    """One term's field frequencies in one document."""

    document: int
    frequencies: tuple[int, int, int, int, int]
    """Occurrences in name, hint, summary, rest of description, parameters."""


class AnalyzedCatalogue:
    """The analysed text of a catalogue: what is expensive and settings-independent.

    Documents are held in name order, so a document's position doubles as the
    name-ascending tie-break and the result never depends on the order tools
    were registered in.
    """

    __slots__ = (
        "_fallback",
        "_stems",
        "average_lengths",
        "documents",
        "field_lengths",
        "names",
        "postings",
    )

    def __init__(self, documents: Iterable[ToolDocument]) -> None:
        self.documents: tuple[ToolDocument, ...] = tuple(sorted(documents, key=lambda document: document.name))
        self.names: tuple[str, ...] = tuple(document.name for document in self.documents)
        # Every word seen, with its stem: a query mostly uses the catalogue's
        # own words, and this spares stemming them again on each query.
        self._stems: dict[str, str] = {}
        field_counts: list[tuple[Counter[str], ...]] = []
        for document in self.documents:
            summary, rest = split_summary(document.description)
            fields = (document.name, document.search_hint, summary, rest, document.parameters)
            field_counts.append(tuple(Counter(self._analyze(text)) for text in fields))
        self.field_lengths: list[tuple[int, ...]] = [
            tuple(sum(counter.values()) for counter in counters) for counters in field_counts
        ]
        # The floor keeps a field no tool uses (no hints anywhere) from
        # dividing by zero; its frequencies are all zero, so the value is moot.
        document_count = max(1, len(self.documents))
        self.average_lengths: tuple[float, ...] = tuple(
            max(1e-9, sum(lengths[index] for lengths in self.field_lengths) / document_count)
            for index in range(_FIELD_COUNT)
        )
        postings: dict[str, list[_Posting]] = {}
        for position, counters in enumerate(field_counts):
            for term in set().union(*counters):
                frequencies = (
                    counters[0][term], counters[1][term], counters[2][term], counters[3][term], counters[4][term]
                )
                postings.setdefault(term, []).append(_Posting(position, frequencies))
        self.postings: dict[str, list[_Posting]] = postings
        self._fallback = _FallbackMatcher(self.documents)

    def _analyze(self, text: str) -> list[str]:
        terms: list[str] = []
        for word in content_words(text):
            stemmed = self._stems.get(word)
            if stemmed is None:
                stemmed = self._stems[word] = stem(word)
            terms.append(stemmed)
        return terms

    def query_terms(self, query: str) -> list[str]:
        """Stems of ``query``'s content words, repeats kept, in query order."""
        stems = self._stems
        return [stems.get(word) or stem(word) for word in content_words(query)]

    def fallback(self, query: str, limit: int, allowed: frozenset[str] | None) -> list[str]:
        return self._fallback.match(query, limit, allowed)


class ToolIndex:
    """A catalogue scored under one :class:`RetrievalSettings`, ready to query.

    Everything that depends only on the catalogue and the settings — each
    term's idf and its saturated, weighted frequency in each document — is
    folded into one number per posting here, so a query does one multiply-add
    per matching document and term.
    """

    __slots__ = ("_catalogue", "_contributions", "_idf", "_lexicon", "_settings")

    def __init__(self, catalogue: AnalyzedCatalogue, settings: RetrievalSettings, lexicon: Lexicon | None) -> None:
        self._catalogue = catalogue
        self._settings = settings
        self._lexicon = lexicon if settings.lexicon_weight > 0 else None
        weights = (
            settings.name_weight,
            settings.search_hint_weight,
            settings.summary_weight,
            settings.description_weight,
            settings.parameters_weight,
        )
        k1, b = settings.bm25_k1, settings.bm25_b
        count = len(catalogue.documents)
        averages = catalogue.average_lengths
        self._idf: dict[str, float] = {}
        self._contributions: dict[str, tuple[tuple[int, float], ...]] = {}
        for term, postings in catalogue.postings.items():
            idf = math.log(1.0 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            self._idf[term] = idf
            scored: list[tuple[int, float]] = []
            for posting in postings:
                lengths = catalogue.field_lengths[posting.document]
                frequency = 0.0
                for index, occurrences in enumerate(posting.frequencies):
                    if occurrences and weights[index]:
                        normaliser = 1.0 - b + b * lengths[index] / averages[index]
                        frequency += weights[index] * occurrences / normaliser
                if frequency:
                    scored.append((posting.document, idf * frequency / (k1 + frequency)))
            if scored:
                self._contributions[term] = tuple(scored)

    @property
    def names(self) -> tuple[str, ...]:
        return self._catalogue.names

    @property
    def documents(self) -> tuple[ToolDocument, ...]:
        return self._catalogue.documents

    def weighted_query(self, query: str) -> dict[str, float]:
        """The query as term -> weight, expansion included."""
        terms = self._catalogue.query_terms(query)
        distinct = set(terms)
        if len(distinct) > _MAX_QUERY_TERMS:
            keep = set(sorted(distinct, key=lambda term: (-self._idf.get(term, 0.0), term))[:_MAX_QUERY_TERMS])
            terms = [term for term in terms if term in keep]
        weights: dict[str, float] = {}
        for term in terms:
            weights[term] = weights.get(term, 0.0) + 1.0
        if self._lexicon is not None:
            expansion_weight = self._settings.lexicon_weight
            for term in terms:
                for target in self._lexicon.expand(term):
                    weights[target] = weights.get(target, 0.0) + expansion_weight
        return weights

    def scores(self, query: str) -> dict[int, float]:
        """Document position -> BM25F score, for documents that score at all."""
        totals: dict[int, float] = {}
        for term, weight in self.weighted_query(query).items():
            for document, contribution in self._contributions.get(term, ()):
                totals[document] = totals.get(document, 0.0) + weight * contribution
        return totals

    def rank(self, query: str, limit: int, allowed: frozenset[str] | None = None) -> list[str]:
        """Names of the best ``limit`` scoring tools, best first; ties by name.

        ``allowed`` restricts the result to those names. ``idf`` still comes
        from the whole catalogue: how rare a word is among all tools is a
        property of the catalogue, not of one run's visibility policy.
        """
        if limit <= 0:
            return []
        totals = self.scores(query)
        names = self._catalogue.names
        order = sorted(
            (document for document in totals if allowed is None or names[document] in allowed),
            key=lambda document: (-totals[document], document),
        )
        return [names[document] for document in order[:limit]]

    def fallback(self, query: str, limit: int, allowed: frozenset[str] | None = None) -> list[str]:
        """:func:`normalized_fallback_match` over this index's documents."""
        return self._catalogue.fallback(query, limit, allowed)


class _FallbackMatcher:
    """Loose token overlap, for queries the main ranking scores nothing for.

    Scored per query token: a document earns a point for each query token that
    one of its words contains, is contained in, or shares a long prefix with.
    The vocabulary is gathered once, so a query compares its few tokens with
    each distinct word of the catalogue rather than with every document.
    """

    __slots__ = ("_names", "_word_documents")

    def __init__(self, documents: Sequence[ToolDocument]) -> None:
        self._names = tuple(document.name for document in documents)
        word_documents: dict[str, set[int]] = {}
        for position, document in enumerate(documents):
            text = fold(f"{document.name} {document.description} {document.search_hint}")
            for word in _WORD.findall(text):
                # Short words and stopwords are left out of the vocabulary. The
                # previous version kept them, and because a document word
                # contained in a query token counts as a match, the article
                # "a" in nearly every English description matched any query
                # token with an "a" in it.
                if len(word) >= _FALLBACK_MIN_TOKEN_LENGTH and word not in STOPWORDS:
                    word_documents.setdefault(word, set()).add(position)
        self._word_documents = word_documents

    def match(self, query: str, limit: int, allowed: frozenset[str] | None) -> list[str]:
        if limit <= 0:
            return []
        tokens = [token for token in _WORD.findall(fold(query)) if len(token) >= _FALLBACK_MIN_TOKEN_LENGTH]
        tokens = [token for token in tokens if token not in STOPWORDS]
        if not tokens:
            return []
        hits: dict[int, int] = {}
        for token, repeats in Counter(tokens).items():
            matched: set[int] = set()
            prefix = token[:_FALLBACK_MIN_PREFIX_LENGTH]
            for word, positions in self._word_documents.items():
                # The cheap containment and prefix tests come first: this loop
                # runs over the whole vocabulary for every query token.
                if (
                    token in word
                    or word in token
                    or (word.startswith(prefix) and _shares_long_prefix(token, word))
                ):
                    matched |= positions
            for position in matched:
                hits[position] = hits.get(position, 0) + repeats
        names = self._names
        order = sorted(
            (position for position in hits if allowed is None or names[position] in allowed),
            key=lambda position: (-hits[position], names[position]),
        )
        return [names[position] for position in order[:limit]]


def _shares_long_prefix(query_token: str, word: str) -> bool:
    """Whether the two share a prefix long enough to absorb an inflectional ending.

    The prefix must cover all but the last two letters of the shorter word and
    never fewer than :data:`_FALLBACK_MIN_PREFIX_LENGTH` letters.
    """
    required = max(_FALLBACK_MIN_PREFIX_LENGTH, min(len(query_token), len(word)) - 2)
    if len(query_token) < required or len(word) < required:
        return False
    return query_token[:required] == word[:required]


def normalized_fallback_match(query: str, documents: Sequence[ToolDocument], *, limit: int) -> list[str]:
    """Names of ``documents`` by loose token overlap with ``query``; best first.

    The second stage of retrieval, for a query the main ranking scores nothing
    for. Deterministic: most matched tokens first, then name ascending. Tools
    with no match are left out. A registry calls this through its prebuilt
    index; this form builds the vocabulary on each call.
    """
    ordered = sorted(documents, key=lambda document: document.name)
    return _FallbackMatcher(ordered).match(query, limit, None)


__all__ = [
    "BUNDLED_LEXICON_RESOURCE",
    "AnalyzedCatalogue",
    "Lexicon",
    "ToolIndex",
    "normalized_fallback_match",
    "parameter_text",
    "split_summary",
]
