"""Russian requests against the English vocabulary of MCP servers.

An MCP server is described by whoever wrote it, in English, in the words of its
own domain: pull requests, tickets, events, deployments, alerts. The operator
asks in Russian, and in the Russian a developer actually speaks — "пулреквест",
"ишью", "смержи", "выкати", "созвон" — much of which is loanword slang no
dictionary lists. Nothing here carries a ``search_hint``, as nothing on a real
server does, so every Russian query reaches its tool through the bundled
lexicon alone.

The catalogue mixes servers that compete for the same words (a GitHub issue and
a support ticket, a chat message and an email, a calendar event and a
reminder), and a host tool that also talks about pull requests. Each query
names the tools that would be an acceptable first call, and each must be among
the first three: ToolSearch loads the first three matches, so a tool ranked
fourth is a tool the model has to ask for again.
"""
# ruff: noqa: RUF001 — Russian queries are the point of this file

from __future__ import annotations

import pytest

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tool_retrieval import RetrievalSettings, ToolDocument
from protocore.runtime.tool_retrieval import AnalyzedCatalogue, Lexicon, ToolIndex

_CATALOGUE: tuple[ToolDocument, ...] = (
    # A few host tools, the usual distractors.
    ToolDocument("Write", "Create or overwrite a file in the workspace with the given content.", parameters="path content"),
    ToolDocument("Read", "Read a text file from the workspace and return its numbered lines.", parameters="path"),
    ToolDocument("Bash", "Run a shell command in the sandbox and return its output and exit code.", parameters="command"),
    ToolDocument("WebSearch", "Search the internet and return result titles, links and snippets.", parameters="query"),
    ToolDocument(
        "ProposeChange",
        "Propose a change to the agent's own code: commit it on a branch and open a pull request for review.",
        parameters="title summary",
    ),
    # GitHub.
    ToolDocument("list_pull_requests", "List pull requests in a GitHub repository, filtered by state (open, closed, all).", parameters="owner repo state"),
    ToolDocument("create_pull_request", "Create a new pull request in a GitHub repository from a head branch into a base branch.", parameters="owner repo title head base"),
    ToolDocument("merge_pull_request", "Merge a pull request in a GitHub repository.", parameters="owner repo pull_number merge_method"),
    ToolDocument("create_pull_request_review", "Create a review on a pull request, approving it or requesting changes.", parameters="owner repo pull_number event body"),
    ToolDocument("list_commits", "Get a list of commits of a branch in a GitHub repository.", parameters="owner repo sha"),
    ToolDocument("create_branch", "Create a new branch in a GitHub repository.", parameters="owner repo branch from_branch"),
    ToolDocument("create_issue", "Create a new issue in a GitHub repository.", parameters="owner repo title body labels"),
    ToolDocument("list_issues", "List issues in a GitHub repository with filtering options.", parameters="owner repo state labels"),
    # Jira.
    ToolDocument("jira_transition_issue", "Transition a Jira issue to a new status, e.g. In Progress or Done.", parameters="issue_key transition_id"),
    ToolDocument("jira_search", "Search Jira issues using JQL.", parameters="jql limit"),
    ToolDocument("jira_create_issue", "Create a new Jira issue: a task, bug or story in a project.", parameters="project_key summary issue_type"),
    # Helpdesk.
    ToolDocument("archive_ticket", "Archive a support ticket so it no longer shows in the queue.", parameters="ticket_id"),
    ToolDocument("list_tickets", "List support tickets in the helpdesk, filtered by status or requester.", parameters="status requester"),
    ToolDocument("reply_to_ticket", "Post a public reply to the customer on a support ticket.", parameters="ticket_id body"),
    # Chat and mail.
    ToolDocument("slack_post_message", "Post a new message to a Slack channel.", parameters="channel_id text"),
    ToolDocument("slack_list_channels", "List public channels in the Slack workspace.", parameters="limit"),
    ToolDocument("send_email", "Send an email message to one or more recipients.", parameters="to subject body"),
    # Calendar.
    ToolDocument("create_event", "Create a calendar event with a title, start and end time and attendees.", parameters="summary start end attendees"),
    ToolDocument("list_events", "List upcoming events in a calendar.", parameters="calendar_id time_min"),
    ToolDocument("create_reminder", "Create a reminder that notifies the user at a given time.", parameters="when text"),
    # Deployments and observability.
    ToolDocument("create_deployment", "Deploy a new version of a service to an environment.", parameters="service version environment"),
    ToolDocument("get_deployment_logs", "Fetch the build and runtime logs of a deployment.", parameters="deployment_id"),
    ToolDocument("list_alerts", "List the alerts that are currently firing.", parameters="severity"),
    ToolDocument("get_dashboard", "Get a Grafana dashboard with its panels and metrics.", parameters="uid"),
    ToolDocument("create_incident", "Open an incident, page the on-call responder and start a timeline.", parameters="title severity"),
    # Data.
    ToolDocument("execute_sql", "Execute a SQL query against the database and return the rows.", parameters="query"),
    ToolDocument("create_record", "Create a record in a table of the base.", parameters="table fields"),
)

#: (query, the tools that would be an acceptable first call)
_QUERIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("покажи открытые пулреквесты", frozenset({"list_pull_requests"})),
    ("открой пул-реквест из моей ветки", frozenset({"create_pull_request"})),
    ("смержи пулреквест", frozenset({"merge_pull_request"})),
    ("оставь ревью на пр и заапрувь", frozenset({"create_pull_request_review"})),
    ("какие коммиты были в ветке", frozenset({"list_commits"})),
    ("создай новую ветку", frozenset({"create_branch"})),
    ("заведи ишью про падение тестов", frozenset({"create_issue", "jira_create_issue"})),
    ("переведи задачу в статус готово", frozenset({"jira_transition_issue"})),
    ("создай таску в джире", frozenset({"jira_create_issue"})),
    ("заархивируй тикет поддержки", frozenset({"archive_ticket"})),
    ("ответь клиенту в тикете", frozenset({"reply_to_ticket"})),
    ("напиши сообщение в канал слака", frozenset({"slack_post_message"})),
    ("отправь письмо на почту", frozenset({"send_email"})),
    ("создай событие в календаре", frozenset({"create_event"})),
    ("назначь созвон на завтра", frozenset({"create_event"})),
    ("какие встречи у меня на неделе", frozenset({"list_events"})),
    ("выкати новую версию сервиса", frozenset({"create_deployment"})),
    ("покажи логи деплоя", frozenset({"get_deployment_logs"})),
    ("какие алерты сейчас горят", frozenset({"list_alerts"})),
    ("открой дашборд с метриками", frozenset({"get_dashboard"})),
    ("заведи инцидент", frozenset({"create_incident"})),
    ("выполни запрос к базе данных", frozenset({"execute_sql"})),
    ("добавь запись в таблицу", frozenset({"create_record"})),
)


@pytest.fixture(scope="module")
def index() -> ToolIndex:
    settings = RetrievalSettings.from_constants(LoopConstants())
    return ToolIndex(AnalyzedCatalogue(_CATALOGUE), settings, Lexicon.bundled())


@pytest.mark.parametrize(("query", "acceptable"), _QUERIES, ids=[query for query, _ in _QUERIES])
def test_a_russian_request_finds_the_servers_tool_in_the_first_three(
    index: ToolIndex, query: str, acceptable: frozenset[str]
) -> None:
    top = index.rank(query, 3)
    assert acceptable & set(top), f"{query!r} ranked {index.rank(query, 8)}"
