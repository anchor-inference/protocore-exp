"""Tool retrieval: tokeniser, stemmers, BM25F, lexicon, index caching, fusion."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import replace
from importlib import resources

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_registry import ToolVisibilityPolicy
from protocore.contracts.tool_retrieval import (
    IToolRetriever,
    RetrievalSettings,
    ToolDocument,
    reciprocal_rank_fusion,
)
from protocore.runtime import tool_registry as tool_registry_module
from protocore.runtime.stemmers import stem, stem_english, stem_russian
from protocore.runtime.text_analysis import analyze, content_words
from protocore.runtime.tool_registry import ToolRegistry, tool_document
from protocore.runtime.tool_retrieval import (
    BUNDLED_LEXICON_RESOURCE,
    AnalyzedCatalogue,
    Lexicon,
    ToolIndex,
    normalized_fallback_match,
    parameter_text,
    split_summary,
)

from ._tool_fixtures import MockTool

_DEFAULTS = RetrievalSettings.from_constants(LoopConstants())


def _index(documents: Sequence[ToolDocument], *, lexicon: Lexicon | None = None, **overrides: float) -> ToolIndex:
    settings = replace(_DEFAULTS, **overrides) if overrides else _DEFAULTS
    return ToolIndex(AnalyzedCatalogue(documents), settings, lexicon)


# ----------------------------------------------------------------------
# tokeniser
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BrowserOpen", ["browser", "open", "browseropen"]),
        ("HTTPServer", ["http", "server", "httpserver"]),
        ("mcp_github-create.issue", ["mcp", "github", "create", "issue", "mcpgithubcreateissue"]),
        ("list_pull_requests", ["list", "pull", "requests", "listpullrequests"]),
        ("e5-small", ["e", "5", "small", "e5small"]),
        ("utf8", ["utf", "8", "utf8"]),
        ("read the file", ["read", "file"]),
        ("please get me the logs", ["logs"]),
        ("Ёлка и ЁЖ", ["елка", "еж"]),
        ("Слушай, короче, найди плз файлы", ["найди", "файлы"]),
        ("пожалуйста, открой браузер", ["открой", "браузер"]),
        ("", []),
    ],
)
def test_content_words(text: str, expected: list[str]) -> None:
    assert content_words(text) == expected


def test_analysis_is_the_same_for_index_and_query() -> None:
    """A word must reach the index in the form a query looks it up in."""
    assert analyze("Reminders") == analyze("reminder")
    assert analyze("напоминания") == analyze("напоминание")
    assert analyze("ёлки") == analyze("елки")


# ----------------------------------------------------------------------
# stemmers — expected values are the reference Snowball implementations'
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("файлы", "файл"),
        ("файлов", "файл"),
        ("открыть", "откр"),
        ("открой", "откр"),
        ("открываю", "открыва"),
        ("напоминание", "напоминан"),
        ("напоминания", "напоминан"),
        ("напомни", "напомн"),
        ("запусти", "запуст"),
        ("сервером", "сервер"),
        ("удалить", "удал"),
        ("удалённый", "удален"),
        ("задачу", "задач"),
        ("истории", "истор"),
        ("память", "памя"),
        ("памяти", "памят"),
        ("закоммитить", "закоммит"),
        ("отправь", "отправ"),
        ("сообщения", "сообщен"),
        ("пользователя", "пользовател"),
        ("прочитай", "прочита"),
        ("отредактируй", "отредактир"),
        ("быстрейший", "быстр"),
        ("вероятность", "вероятн"),
        ("красивейшая", "красив"),
    ],
)
def test_russian_stemmer_matches_snowball(word: str, expected: str) -> None:
    assert stem_russian(word) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("files", "file"),
        ("running", "run"),
        ("connections", "connect"),
        ("relational", "relat"),
        ("generalization", "gener"),
        ("happiness", "happi"),
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("agreed", "agre"),
        ("plastered", "plaster"),
        ("motoring", "motor"),
        ("sing", "sing"),
        ("conflated", "conflat"),
        ("sized", "size"),
        ("hopping", "hop"),
        ("falling", "fall"),
        ("filing", "file"),
        ("happy", "happi"),
        ("controlling", "control"),
        ("electrical", "electr"),
        ("adjustable", "adjust"),
        ("adoption", "adopt"),
        # The 1980 table, not Porter's later code: no "logi" rule, "abli" not "bli".
        ("topology", "topologi"),
        ("possibly", "possibli"),
        ("reliably", "reliabl"),
    ],
)
def test_english_stemmer_matches_porter(word: str, expected: str) -> None:
    assert stem_english(word) == expected


def test_short_tokens_are_not_stemmed() -> None:
    assert stem("is") == "is"
    assert stem("ui") == "ui"
    assert stem("да") == "да"


def test_tokens_that_are_not_words_pass_through() -> None:
    assert stem("utf8") == "utf8"
    assert stem("42") == "42"


# ----------------------------------------------------------------------
# BM25F ranking
# ----------------------------------------------------------------------


def test_summary_outweighs_the_tail_of_a_description() -> None:
    documents = [
        ToolDocument("Alpha", "Delete a record. Mentions archive once in passing, as a caveat."),
        ToolDocument("Bravo", "Archive a record for later."),
    ]
    assert _index(documents).rank("archive", 2) == ["Bravo", "Alpha"]


def test_field_weights_come_from_the_settings() -> None:
    documents = [
        ToolDocument("Alpha", "Shows the weather.", parameters="archive"),
        ToolDocument("Bravo", "Shows the weather.", search_hint="archive"),
    ]
    assert _index(documents).rank("archive", 2) == ["Bravo", "Alpha"]
    reversed_weights = _index(documents, search_hint_weight=0.1, parameters_weight=3.0)
    assert reversed_weights.rank("archive", 2) == ["Alpha", "Bravo"]


def test_identifier_split_name_matches_spaced_query() -> None:
    documents = [
        ToolDocument("BrowserOpen", "Load a page."),
        ToolDocument("Grep", "Search text."),
    ]
    assert _index(documents).rank("browser open", 2)[0] == "BrowserOpen"
    assert _index(documents).rank("browseropen", 2) == ["BrowserOpen"]


def test_russian_query_reaches_english_tools_through_the_lexicon() -> None:
    documents = [
        ToolDocument("ReadFile", "Read a text file from the workspace and return its lines."),
        ToolDocument("BrowserOpen", "Open a web page in the browser."),
        ToolDocument("GitCommit", "Record staged changes in the repository as a commit."),
    ]
    lexicon = Lexicon.bundled()
    assert _index(documents, lexicon=lexicon).rank("открой страницу в браузере", 1) == ["BrowserOpen"]
    assert _index(documents, lexicon=lexicon).rank("прочитай файл", 1) == ["ReadFile"]
    assert _index(documents, lexicon=lexicon).rank("закоммить изменения", 1) == ["GitCommit"]
    # The control: without the lexicon the query shares no word with the catalogue.
    assert _index(documents).rank("открой страницу в браузере", 3) == []


def test_lexicon_weight_zero_turns_expansion_off() -> None:
    documents = [ToolDocument("BrowserOpen", "Open a web page in the browser.")]
    index = _index(documents, lexicon=Lexicon.bundled(), lexicon_weight=0.0)
    assert index.rank("открой браузер", 1) == []


def test_a_host_lexicon_replaces_the_bundled_one() -> None:
    lexicon = Lexicon.from_translations({"forecast": ["погода", "прогноз"]})
    documents = [ToolDocument("Weather", "Return the forecast for a city."), ToolDocument("Grep", "Search text.")]
    assert _index(documents, lexicon=lexicon).rank("какая погода", 2) == ["Weather"]


def test_the_bundled_lexicon_is_shipped() -> None:
    text = resources.files("protocore.runtime").joinpath(BUNDLED_LEXICON_RESOURCE).read_text(encoding="utf-8")
    translations = json.loads(text)
    assert len(translations) > 1500
    assert all(isinstance(words, list) and words for words in translations.values())
    assert len(Lexicon.bundled()) > 1500


def test_ties_are_broken_by_name_whatever_the_input_order() -> None:
    documents = [ToolDocument(name, "Read a file.") for name in ("Zebra", "Alpha", "Mike")]
    assert _index(documents).rank("read file", 3) == ["Alpha", "Mike", "Zebra"]
    assert _index(list(reversed(documents))).rank("read file", 3) == ["Alpha", "Mike", "Zebra"]


def test_allowed_restricts_the_result() -> None:
    documents = [ToolDocument(name, "Read a file.") for name in ("Alpha", "Bravo", "Charlie")]
    assert _index(documents).rank("read", 5, frozenset({"Bravo"})) == ["Bravo"]


def test_a_long_query_keeps_its_rarest_terms() -> None:
    """Past the query-term cap only the rarest terms are scored.

    A pasted log can carry hundreds of distinct words; scoring every one costs
    time and lets the common ones outvote the few that name a tool.
    """
    words = [f"{chr(97 + i // 26)}{chr(97 + i % 26)}qx" for i in range(40)]
    documents = [ToolDocument(f"Filler{i}", " ".join(words)) for i in range(5)]
    documents.append(ToolDocument("Target", "zanzibar"))
    index = _index(documents)
    weighted = index.weighted_query(" ".join([*words, "zanzibar"]))
    assert len(weighted) == 25
    assert "zanzibar" in weighted


def test_empty_catalogue_and_zero_limit() -> None:
    assert _index([]).rank("anything", 5) == []
    assert _index([ToolDocument("A", "alpha")]).rank("alpha", 0) == []


def test_split_summary_and_parameter_text() -> None:
    assert split_summary("First one. Second one! Third?") == ("First one.", "Second one! Third?")
    assert split_summary("No full stop") == ("No full stop", "")
    properties = {"path": {"type": "string", "description": "Where to read"}, "limit": {"type": "integer"}, "odd": "x"}
    assert parameter_text(properties) == "path Where to read limit odd"


@pytest.mark.parametrize(
    ("description", "summary"),
    [
        # Cut after "(e.g." the line ToolSearch shows ended mid-bracket.
        (
            "Transition a Jira issue to a new status (e.g. Done, In Progress). Needs the key.",
            "Transition a Jira issue to a new status (e.g. Done, In Progress).",
        ),
        ("Find files by pattern, e.g. a glob. Then read them.", "Find files by pattern, e.g. a glob."),
        ("Compare two runs, i.e. their outputs. More.", "Compare two runs, i.e. their outputs."),
        ("Local vs. remote copies. More.", "Local vs. remote copies."),
        ("Ищет файлы, т.е. по маске. Потом читает.", "Ищет файлы, т.е. по маске."),  # noqa: RUF001
        ("Ищет файлы, напр. по маске. Потом читает.", "Ищет файлы, напр. по маске."),
        # "etc." ends the sentence only when a capital starts the next one.
        ("Reads CSV, JSON, etc. Use it for data.", "Reads CSV, JSON, etc."),
        ("Reads CSV, JSON, etc. and more. Use it.", "Reads CSV, JSON, etc. and more."),
        ("Read a file (up to 2000 lines. Then stop). Next.", "Read a file (up to 2000 lines. Then stop)."),
        # A bracket that never closes must not swallow the whole description.
        ("Stray (bracket. Second. Third", "Stray (bracket."),
    ],
)
def test_split_summary_does_not_stop_at_an_abbreviation_or_inside_brackets(
    description: str, summary: str
) -> None:
    head, rest = split_summary(description)
    assert head == summary
    assert f"{head} {rest}".strip() == description


def test_fallback_limit_zero_and_stopword_only_query() -> None:
    documents = [ToolDocument("Read", "Read a file.")]
    assert normalized_fallback_match("read", documents, limit=0) == []
    assert normalized_fallback_match("the and", documents, limit=5) == []


# ----------------------------------------------------------------------
# the registry: rank order, index caching, the clip's fallback
# ----------------------------------------------------------------------


def test_search_returns_rank_order_not_name_order() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="Alpha", description="Mentions a screenshot in passing among many other words here."))
    reg.register(MockTool(tool_name="Zulu", description="Take a screenshot of the page."))
    assert [t.name for t in reg.search("take a screenshot", top_k=2)] == ["Zulu", "Alpha"]


def test_search_zero_top_k_is_empty() -> None:
    reg = ToolRegistry([MockTool(tool_name="Alpha", description="alpha")])
    assert reg.search("alpha", top_k=0) == []


def test_search_falls_back_when_nothing_scores() -> None:
    reg = ToolRegistry(lexicon=None)
    reg.register(MockTool(tool_name="Remember", description="Save a fact.", search_hint="память запомнить"))
    reg.register(MockTool(tool_name="Bash", description="Run a shell command."))
    assert [t.name for t in reg.search("памяти", top_k=5)] == ["Remember"]


def test_parameters_are_searchable() -> None:
    reg = ToolRegistry()
    reg.register(
        MockTool(
            tool_name="Convert",
            description="Convert a value.",
            parameters_schema={"timezone": {"type": "string", "description": "IANA zone"}},
        )
    )
    reg.register(MockTool(tool_name="Other", description="Something else."))
    assert [t.name for t in reg.search("timezone", top_k=1)] == ["Convert"]
    convert = reg.get("Convert")
    assert convert is not None
    assert tool_document(convert).parameters == "timezone IANA zone"


class _CountingCatalogue(AnalyzedCatalogue):
    builds = 0

    def __init__(self, documents: Iterable[ToolDocument]) -> None:
        type(self).builds += 1
        super().__init__(documents)


class _CountingIndex(ToolIndex):
    builds = 0

    def __init__(self, catalogue: AnalyzedCatalogue, settings: RetrievalSettings, lexicon: Lexicon | None) -> None:
        type(self).builds += 1
        super().__init__(catalogue, settings, lexicon)


@pytest.fixture
def counting(monkeypatch: pytest.MonkeyPatch) -> tuple[type[_CountingCatalogue], type[_CountingIndex]]:
    _CountingCatalogue.builds = 0
    _CountingIndex.builds = 0
    monkeypatch.setattr(tool_registry_module, "AnalyzedCatalogue", _CountingCatalogue)
    monkeypatch.setattr(tool_registry_module, "ToolIndex", _CountingIndex)
    return _CountingCatalogue, _CountingIndex


def test_index_is_rebuilt_only_when_the_catalogue_changes(
    counting: tuple[type[_CountingCatalogue], type[_CountingIndex]],
) -> None:
    catalogue, index = counting
    reg = ToolRegistry([MockTool(tool_name="Alpha", description="alpha tool"), MockTool(tool_name="Bravo", description="bravo")])
    reg.search("alpha", top_k=1)
    reg.search("bravo", top_k=1)
    reg.compute_effective_surface("t", ToolVisibilityPolicy(), query="alpha", top_k=1)
    assert (catalogue.builds, index.builds) == (1, 1)

    reg.register(MockTool(tool_name="Charlie", description="charlie"))
    assert [t.name for t in reg.search("charlie", top_k=1)] == ["Charlie"]
    assert (catalogue.builds, index.builds) == (2, 2)

    reg.unregister("Charlie")
    reg.unregister("Charlie")  # absent: no new generation
    assert reg.search("charlie", top_k=1) == []
    reg.search("alpha", top_k=1)
    assert (catalogue.builds, index.builds) == (3, 3)


def test_other_settings_reuse_the_analysis(
    counting: tuple[type[_CountingCatalogue], type[_CountingIndex]],
) -> None:
    catalogue, index = counting
    reg = ToolRegistry([MockTool(tool_name="Alpha", description="alpha tool")])
    reg.search("alpha", top_k=1)
    reg.search("alpha", top_k=1, retrieval=replace(_DEFAULTS, bm25_b=0.75))
    reg.search("alpha", top_k=1, retrieval=replace(_DEFAULTS, bm25_b=0.75))
    assert (catalogue.builds, index.builds) == (1, 2)


def test_the_index_cache_is_bounded(
    counting: tuple[type[_CountingCatalogue], type[_CountingIndex]],
) -> None:
    _, index = counting
    reg = ToolRegistry([MockTool(tool_name="Alpha", description="alpha tool")])
    variants = [replace(_DEFAULTS, bm25_k1=1.0 + step / 10) for step in range(10)]
    for settings in variants:
        reg.search("alpha", top_k=1, retrieval=settings)
    assert len(reg._indexes) <= tool_registry_module._MAX_CACHED_INDEXES
    # The oldest were evicted, so asking for the first again builds it again.
    reg.search("alpha", top_k=1, retrieval=variants[0])
    assert index.builds == len(variants) + 1


def test_an_index_for_an_older_generation_is_not_cached() -> None:
    reg = ToolRegistry([MockTool(tool_name="Alpha", description="alpha tool")])
    reg.search("alpha", top_k=1)
    generation, tools = reg._snapshot()
    reg.register(MockTool(tool_name="Bravo", description="bravo"))
    reg.search("bravo", top_k=1)
    newer = reg._catalogue
    stale = reg._index_for(generation, tools, _DEFAULTS)
    assert stale.names == ("Alpha",)
    assert reg._catalogue is newer


def test_clip_uses_the_fallback_when_nothing_scores() -> None:
    reg = ToolRegistry(lexicon=None)
    reg.register(MockTool(tool_name="Remember", description="Save a fact.", search_hint="память запомнить"))
    for name in ("Alpha", "Bravo", "Charlie"):
        reg.register(MockTool(tool_name=name, description=f"{name} helper"))
    defs = reg.compute_effective_surface("t", ToolVisibilityPolicy(), query="памяти", top_k=1)
    assert [d.name for d in defs] == ["Remember"]


def test_clip_keeps_name_order() -> None:
    reg = ToolRegistry()
    reg.register(MockTool(tool_name="Zulu", description="Take a screenshot of the page."))
    reg.register(MockTool(tool_name="Alpha", description="A screenshot is mentioned here among other words."))
    reg.register(MockTool(tool_name="Mike", description="Unrelated."))
    defs = reg.compute_effective_surface("t", ToolVisibilityPolicy(), query="take a screenshot", top_k=2)
    assert [d.name for d in defs] == ["Alpha", "Zulu"]


def test_empty_query_clip_fills_by_name() -> None:
    reg = ToolRegistry([MockTool(tool_name=name) for name in ("Delta", "Charlie", "Alpha", "Bravo")])
    defs = reg.compute_effective_surface("t", ToolVisibilityPolicy(pinned={"Charlie"}), query=" ", top_k=2)
    assert [d.name for d in defs] == ["Alpha", "Bravo", "Charlie"]


def test_pinned_tools_do_not_count_against_the_clip() -> None:
    """``top_k`` is how many tools retrieval adds. Counted with the pins, a
    floor as large as the default left no room at all and the clip retrieved
    nothing, whatever the message asked for."""
    pinned = {f"Core{i:02d}" for i in range(14)}
    tools = [MockTool(tool_name=name) for name in sorted(pinned)]
    tools.append(MockTool(tool_name="ScheduleCreate", description="Schedule a job to run later."))
    tools.extend(MockTool(tool_name=f"Other{i:02d}", description="Unrelated.") for i in range(20))
    reg = ToolRegistry(tools)

    defs = reg.compute_effective_surface(
        "t", ToolVisibilityPolicy(forced_pinned=frozenset(pinned)), query="schedule a job", top_k=12
    )

    names = [d.name for d in defs]
    assert "ScheduleCreate" in names
    assert pinned <= set(names)
    assert len(names) <= len(pinned) + 12


def test_no_clip_when_the_unpinned_tools_already_fit() -> None:
    reg = ToolRegistry([MockTool(tool_name=name) for name in ("A", "B", "C", "D")])
    defs = reg.compute_effective_surface(
        "t", ToolVisibilityPolicy(pinned={"A", "B"}), query="anything", top_k=2
    )
    assert [d.name for d in defs] == ["A", "B", "C", "D"]


def test_settings_default_to_the_constants_model() -> None:
    assert ToolRegistry()._default_settings == RetrievalSettings.from_constants(LoopConstants())
    rc = LoopConstants(tool_retrieval_bm25_b=0.9, tool_retrieval_lexicon_weight=0.0)
    settings = RetrievalSettings.from_constants(rc)
    assert (settings.bm25_b, settings.lexicon_weight) == (0.9, 0.0)


# ----------------------------------------------------------------------
# host retriever and reciprocal rank fusion
# ----------------------------------------------------------------------


def test_reciprocal_rank_fusion() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], limit=3, rank_constant=60)
    # a and c tie at 1/61 + 1/63, just above b's 2/62; the tie goes by name.
    assert fused == ["a", "c", "b"]
    assert reciprocal_rank_fusion([["a", "a", "b"]], limit=5, rank_constant=60) == ["a", "b"]
    assert reciprocal_rank_fusion([["a"]], limit=0, rank_constant=60) == []


class _PreferLast:
    """A toy host ranker: the reverse of name order, plus a name it does not own."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def rank(self, query: str, documents: Sequence[ToolDocument], limit: int) -> Sequence[str]:
        self.calls.append((query, len(documents), limit))
        return ["NotInTheCatalogue", *sorted((d.name for d in documents), reverse=True)][:limit]


def test_a_host_retriever_is_fused_with_the_lexical_ranking() -> None:
    retriever = _PreferLast()
    assert isinstance(retriever, IToolRetriever)
    reg = ToolRegistry(retriever=retriever)
    reg.register(MockTool(tool_name="Alpha", description="Take a screenshot of the page."))
    reg.register(MockTool(tool_name="Bravo", description="Unrelated."))
    reg.register(MockTool(tool_name="Zulu", description="Unrelated too."))
    names = [t.name for t in reg.search("screenshot", top_k=2, whitelist=["Alpha", "Bravo", "Zulu"])]
    # Alpha is first lexically; Zulu is first for the host ranker and has no
    # lexical support; both beat Bravo, which neither ranks first.
    assert names == ["Alpha", "Zulu"]
    assert retriever.calls == [("screenshot", 3, 3)]


def test_a_host_retriever_can_surface_what_the_lexicon_misses() -> None:
    reg = ToolRegistry(lexicon=None, retriever=_PreferLast())
    reg.register(MockTool(tool_name="Alpha", description="alpha"))
    reg.register(MockTool(tool_name="Zulu", description="zulu"))
    assert [t.name for t in reg.search("совсем другое", top_k=1)] == ["Zulu"]


# ----------------------------------------------------------------------
# cost
# ----------------------------------------------------------------------


def test_a_catalogue_of_700_tools_builds_once_and_queries_fast() -> None:
    """A smoke bound, not a benchmark: generous enough for a loaded CI box.

    The engine measures about 0.3 ms per query and 150 ms per build on 700
    realistic tools; the old engine re-tokenised the catalogue on every query
    and took 15-20 ms. Bounds two orders of magnitude above the measurement
    catch a return to per-query work without flaking under load.
    """
    verbs = ["read", "write", "list", "delete", "create", "search", "update", "send", "open", "close"]
    nouns = ["file", "issue", "page", "message", "event", "container", "pod", "commit", "task", "record"]
    reg = ToolRegistry()
    for i in range(700):
        verb, noun = verbs[i % 10], nouns[(i // 10) % 10]
        reg.register(
            MockTool(
                tool_name=f"Server{i // 100}_{verb}_{noun}_{i}",
                description=f"{verb.title()} a {noun} on server {i // 100}. Returns the {noun} as JSON with id {i}.",
                parameters_schema={"id": {"type": "string", "description": f"The {noun} id"}},
            )
        )
    started = time.perf_counter()
    reg.search("warm", top_k=5)
    build_seconds = time.perf_counter() - started
    queries = ["read a file", "open the page", "send a message", "удалить задачу", "list pods on server 3"] * 20
    started = time.perf_counter()
    for query in queries:
        assert reg.search(query, top_k=5)
    per_query = (time.perf_counter() - started) / len(queries)
    assert build_seconds < 10.0
    assert per_query < 0.05
