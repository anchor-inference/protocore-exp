"""Offline quality floor for tool retrieval: a fixed catalogue and fixed queries.

The unit tests pin mechanisms; this pins the outcome. A change to the
tokeniser, the stopwords, the stemmers, the field weights or the lexicon can
keep every mechanism test green and still quietly make the right tool harder
to find. Recall@5 over a small labelled set is the cheapest signal that
catches that.

The catalogue is written the way a real one is: English names and
descriptions, a search_hint on only a few tools, and near-neighbours that
compete for the same words (three ways to read something, two schedulers, two
messaging tools). Half of the queries are Russian and reach English-only tools
through the bundled lexicon alone. Each query names the tools that would be an
acceptable first call.

The floors sit a little under what the engine scores today (English 20 of
20, Russian 17 of 20: "запиши" is not in the lexicon's entry for write,
"помнишь" stems apart from the hint's "вспомни", and "погода" has no entry at
all), so ordinary tuning passes and a real regression does not. Raising a floor after an
improvement is the right move; lowering one needs a reason in the commit.
"""
# ruff: noqa: RUF001 — Russian queries are the point of this file

from __future__ import annotations

from collections.abc import Sequence

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_retrieval import RetrievalSettings, ToolDocument
from protocore.runtime.tool_retrieval import AnalyzedCatalogue, Lexicon, ToolIndex

_CATALOGUE: tuple[ToolDocument, ...] = (
    ToolDocument("Read", "Read a text file from the workspace and return its numbered lines.", parameters="path Path of the file"),
    ToolDocument("Write", "Create or overwrite a file in the workspace with the given content.", parameters="path content"),
    ToolDocument("Edit", "Replace an exact string in an existing file with new text.", parameters="path old_string new_string"),
    ToolDocument("Glob", "Find files whose paths match a glob pattern such as **/*.py.", parameters="pattern"),
    ToolDocument("Grep", "Search file contents for a regular expression and list matching lines.", parameters="pattern path"),
    ToolDocument("Bash", "Run a shell command in the sandbox and return its output and exit code.", parameters="command timeout"),
    ToolDocument("WebSearch", "Search the internet and return result titles, links and snippets.", parameters="query limit"),
    ToolDocument("WebFetch", "Download a web page by URL and return its readable text.", parameters="url"),
    ToolDocument("BrowserOpen", "Open a page in the controlled browser and wait until it loads.", parameters="url"),
    ToolDocument("BrowserClick", "Click a button, link or other element on the current browser page.", parameters="selector"),
    ToolDocument("BrowserScreenshot", "Take a screenshot of the current browser page.", parameters="full_page"),
    ToolDocument("GitCommit", "Record the staged changes in the repository as a new commit.", parameters="message"),
    ToolDocument("GitDiff", "Show the uncommitted changes in the repository as a diff."),
    ToolDocument("GitLog", "List recent commits of the repository with authors and dates.", parameters="limit"),
    ToolDocument("CreateIssue", "Open a new issue in the project tracker with a title and body.", parameters="title body"),
    ToolDocument("ListPullRequests", "List open pull requests of a repository.", parameters="repository state"),
    ToolDocument("SendMessage", "Send a chat message to a user or a channel.", parameters="recipient text"),
    ToolDocument("SendEmail", "Send an email with a subject and body to one or more addresses.", parameters="to subject body"),
    ToolDocument("ScheduleReminder", "Schedule a reminder that notifies the user at a given time.", parameters="when text"),
    ToolDocument("ListReminders", "List the reminders that are scheduled and not yet delivered."),
    ToolDocument("CreateCalendarEvent", "Create a calendar event with a start time, duration and attendees.", parameters="title start attendees"),
    ToolDocument(
        "Remember",
        "Save a durable fact to long-term memory so later sessions can recall it.",
        search_hint="remember memorize save note запомнить запомни память сохранить",
    ),
    ToolDocument(
        "Recall",
        "Look up facts saved in long-term memory that match a query.",
        search_hint="recall memory lookup вспомнить вспомни память",
    ),
    ToolDocument("Forget", "Delete a fact from long-term memory.", parameters="fact_id"),
    ToolDocument("TodoWrite", "Replace the task checklist of the current run with an updated list."),
    ToolDocument("AskUser", "Pause and ask the user a question, then return the answer."),
    ToolDocument("SqlQuery", "Run a read-only SQL query against the database and return the rows.", parameters="sql"),
    ToolDocument("DockerLogs", "Show the logs of a running container.", parameters="container tail"),
    ToolDocument("DockerRestart", "Restart a container by name.", parameters="container"),
    ToolDocument("KubeListPods", "List pods in a Kubernetes namespace with their status.", parameters="namespace"),
    ToolDocument("Translate", "Translate text into another language.", parameters="text target_language"),
    ToolDocument("Summarize", "Summarize a long text into a few sentences."),
    ToolDocument("GenerateImage", "Generate an image from a text prompt.", parameters="prompt size"),
    ToolDocument("ConvertTime", "Convert a time between time zones.", parameters="time from_zone to_zone"),
    ToolDocument("WeatherForecast", "Return the weather forecast for a city.", parameters="city days"),
    ToolDocument("UploadFile", "Upload a local file and return a link to share it.", parameters="path"),
    ToolDocument("ArchiveFiles", "Pack files into a zip archive.", parameters="paths destination"),
    ToolDocument("SpawnAgent", "Start a sub-agent that works on a delegated task in the background.", parameters="task"),
    ToolDocument("Notify", "Show a push notification to the operator.", parameters="text"),
    ToolDocument("ReadPdf", "Extract the text of a PDF document.", parameters="path pages"),
)

# (query, acceptable first calls)
_ENGLISH: tuple[tuple[str, frozenset[str]], ...] = (
    ("read the config file", frozenset({"Read"})),
    ("create a new file with this content", frozenset({"Write"})),
    ("replace a string in a file", frozenset({"Edit"})),
    ("find all python files", frozenset({"Glob", "Grep"})),
    ("search the code for a regex", frozenset({"Grep"})),
    ("run a shell command", frozenset({"Bash"})),
    ("search the web for the latest release notes", frozenset({"WebSearch"})),
    ("fetch this url and give me the text", frozenset({"WebFetch", "BrowserOpen"})),
    ("open the page in the browser", frozenset({"BrowserOpen"})),
    ("click the submit button", frozenset({"BrowserClick"})),
    ("take a screenshot", frozenset({"BrowserScreenshot"})),
    ("commit my changes", frozenset({"GitCommit"})),
    ("what changed since the last commit", frozenset({"GitDiff"})),
    ("file a bug in the tracker", frozenset({"CreateIssue"})),
    ("which pull requests are open", frozenset({"ListPullRequests"})),
    ("email the report to the team", frozenset({"SendEmail"})),
    ("remind me tomorrow at 9", frozenset({"ScheduleReminder"})),
    ("remember that the deploy key rotates monthly", frozenset({"Remember"})),
    ("how many users signed up, query the database", frozenset({"SqlQuery"})),
    ("show the container logs", frozenset({"DockerLogs"})),
)

_RUSSIAN: tuple[tuple[str, frozenset[str]], ...] = (
    ("прочитай файл с настройками", frozenset({"Read"})),
    ("запиши это в новый файл", frozenset({"Write"})),
    ("найди все файлы по шаблону", frozenset({"Glob", "Grep"})),
    ("выполни команду в терминале", frozenset({"Bash"})),
    ("поищи в интернете новости", frozenset({"WebSearch"})),
    ("открой страницу в браузере", frozenset({"BrowserOpen"})),
    ("нажми на кнопку", frozenset({"BrowserClick"})),
    ("сделай скриншот страницы", frozenset({"BrowserScreenshot"})),
    ("закоммить изменения", frozenset({"GitCommit"})),
    ("покажи последние коммиты", frozenset({"GitLog"})),
    ("отправь сообщение в канал", frozenset({"SendMessage"})),
    ("напомни мне завтра позвонить", frozenset({"ScheduleReminder"})),
    ("запомни, что сервер переехал", frozenset({"Remember"})),
    ("что ты помнишь про проект", frozenset({"Recall"})),
    ("перезапусти контейнер", frozenset({"DockerRestart"})),
    ("переведи текст на английский", frozenset({"Translate"})),
    ("какая погода в городе", frozenset({"WeatherForecast"})),
    ("сгенерируй картинку", frozenset({"GenerateImage"})),
    ("упакуй файлы в архив", frozenset({"ArchiveFiles"})),
    ("спроси пользователя", frozenset({"AskUser"})),
)


def _recall_at_5(index: ToolIndex, cases: Sequence[tuple[str, frozenset[str]]]) -> float:
    hits = 0
    for query, gold in cases:
        ranked = index.rank(query, 5) or index.fallback(query, 5)
        hits += bool(gold & set(ranked))
    return hits / len(cases)


@pytest.fixture(scope="module")
def index() -> ToolIndex:
    settings = RetrievalSettings.from_constants(LoopConstants())
    return ToolIndex(AnalyzedCatalogue(_CATALOGUE), settings, Lexicon.bundled())


def test_the_corpus_is_the_size_it_claims() -> None:
    assert len(_CATALOGUE) == 40
    assert len({document.name for document in _CATALOGUE}) == 40
    names = {document.name for document in _CATALOGUE}
    for _, gold in _ENGLISH + _RUSSIAN:
        assert gold <= names


def test_english_recall_at_5(index: ToolIndex) -> None:
    assert _recall_at_5(index, _ENGLISH) >= 0.9


def test_russian_recall_at_5(index: ToolIndex) -> None:
    """Russian queries against English descriptions, through the lexicon."""
    assert _recall_at_5(index, _RUSSIAN) >= 0.8


def test_russian_recall_depends_on_the_lexicon() -> None:
    """The control: without the lexicon the same Russian queries mostly miss.

    If this stopped holding, the Russian floor above would no longer be
    measuring the lexicon at all.
    """
    settings = RetrievalSettings.from_constants(LoopConstants())
    bare = ToolIndex(AnalyzedCatalogue(_CATALOGUE), settings, None)
    assert _recall_at_5(bare, _RUSSIAN) <= 0.3
