"""AskUser tool — pause the agent loop for a human-in-the-loop answer.

Rich multi-question contract modeled on Anthropic's
``AskUserQuestion`` tool. This REPLACES the legacy single-question /
3-render-mode contract (no backward compatibility — dev-version project).

Contract surface (core)
=======================

The tool is a **state-only** side-effect kind: it does NOT mutate the
workspace, hit the network, or touch the sandbox. Instead it raises a
typed :class:`AskUserPauseRequested` signal that the core dispatcher
turns into a ``tool_call_pending``-shaped ``ask_user`` interrupt. The
loop transitions to ``AWAITING``; the host's resume route feeds the
user's per-question answers back as the tool result.

Rich schema
-----------

* :class:`AskUserOption` — one selectable option: ``label`` plus an
 optional ``description`` and an optional ``preview`` (ASCII/diagram
 text the UI renders in a split pane).
* :class:`AskUserQuestion` — one question: ``question`` text, an optional
 short ``header`` chip, a list of ``options``, ``multiSelect`` and
 ``allow_custom`` flags. A question MUST carry ``options`` OR set
 ``allow_custom=True`` (otherwise it is unanswerable); ``multiSelect``
 is only meaningful with options.
* :class:`AskUserInput` — ``questions[]``: one or more questions, each
 independently single/multi-select, each optionally allowing a
 free-text custom answer.
* :class:`AskUserAnswer` — the user's answer to one question:
 ``selected`` labels (verbatim) plus an optional ``custom`` free-text.
* :class:`AskUserOutput` — ``answers[]`` echoed back into the model
 context once the user submits.

The three legacy render modes collapse into this one schema:

* plain text → no ``options`` + ``allow_custom=True``;
* yes/no → two options (``Yes``/``No``), ``multiSelect=False``;
* choices → ``options`` + ``multiSelect`` as needed.

The actual interrupt emit + answer resume lives in the host, because
delivering the question and waiting for the reply needs a durable queue
and an event bus, neither of which is visible to core.
"""
from __future__ import annotations

import copy
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from protocore.contracts.tool_roles import ToolRole
from protocore.contracts.tools import Tool, ToolContext, ToolError
from protocore.contracts.types import (
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
)

ASK_USER_TOOL_NAME: str = "AskUser"
"""Canonical, stable name surfaced to the LLM. Alphabetic sort puts it
before ``Bash`` — matches the registry's alphabetic-order invariant.
"""

# Bound the surface so a runaway model call cannot wedge the Redis
# pending-interrupt blob. The caps are deliberately generous for an
# interactive panel while staying well inside the chat modal budget.
ASK_USER_MAX_QUESTIONS: int = 10
"""Max questions in one AskUser panel (paginated per-question on chat)."""

ASK_USER_QUESTION_MAX_LENGTH: int = 2000
"""Per-question text ceiling (markdown permitted, not required)."""

ASK_USER_HEADER_MAX_LENGTH: int = 24
"""Short chip label shown above the question (≈12 visible chars; the cap
leaves headroom for multibyte glyphs)."""

ASK_USER_MAX_OPTIONS: int = 20
"""Max selectable options per question."""

ASK_USER_OPTION_LABEL_MAX_LENGTH: int = 200
"""Per-option label ceiling — mirrors the chat modal's truncation."""

ASK_USER_OPTION_DESCRIPTION_MAX_LENGTH: int = 1000
"""Per-option helper-text ceiling."""

ASK_USER_PREVIEW_MAX_LENGTH: int = 8000
"""Per-option ASCII/diagram preview ceiling (split-pane content)."""


def _inline_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a self-contained JSON schema with all ``$defs`` inlined.

    Pydantic emits nested model definitions under a top-level ``$defs``
    map referenced by ``$ref``. LLM tool-schema consumers receive only
    the per-tool ``properties`` block (:class:`ToolParameterSchema`), so
    a bare ``$ref`` would dangle. This walker dereferences every
    ``#/$defs/<name>`` pointer in place (recursively, preserving any
    sibling keys such as ``description``) so the published schema is
    fully resolved and provider-portable.
    """

    defs: dict[str, Any] = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.rsplit("/", 1)[-1])
                if isinstance(target, dict):
                    merged = _resolve(copy.deepcopy(target))
                    # Carry any sibling keys declared alongside the $ref
                    # (e.g. a field-level ``description``) onto the
                    # resolved object.
                    for key, value in node.items():
                        if key != "$ref":
                            merged[key] = _resolve(value)
                    return merged
            return {
                key: _resolve(value)
                for key, value in node.items()
                if key != "$defs"
            }
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    resolved = _resolve({k: v for k, v in schema.items() if k != "$defs"})
    return resolved if isinstance(resolved, dict) else {}


class AskUserPauseRequested(ToolError):
    """Sentinel raised by :meth:`AskUserTool.invoke`.

    The dispatcher catches this typed signal and translates it into a
    pending ``ask_user`` interrupt. Carries the validated
    :class:`AskUserInput` so the host handler can hand the chat UI
    a typed envelope without re-parsing the LLM tool_input dict.

    Why a sentinel exception (not a return value)?
    --------------------------------------------
    The core :class:`Tool` ABC contract is ``invoke -> ToolResult``. A
    successful AskUser does NOT have a synchronous result — it pauses the
    loop and the answer arrives via the resume route on a different pod.
    Raising a typed signal keeps the synchronous ``ToolResult`` contract
    honest and lets the host handler decide how to surface the
    pause.
    """

    def __init__(self, payload: AskUserInput) -> None:
        super().__init__(f"ask_user paused: {len(payload.questions)} question(s)")
        self.payload = payload


class AskUserOption(BaseModel):
    """One selectable option for an :class:`AskUserQuestion`.

    ``preview`` carries ASCII/diagram text the chat UI renders in a
    split pane next to the option list (the harness preview layout).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(
        ...,
        min_length=1,
        max_length=ASK_USER_OPTION_LABEL_MAX_LENGTH,
        description="Option label shown to the user and returned verbatim if picked.",
    )
    description: str | None = Field(
        default=None,
        max_length=ASK_USER_OPTION_DESCRIPTION_MAX_LENGTH,
        description="Optional helper text shown under the option label.",
    )
    preview: str | None = Field(
        default=None,
        max_length=ASK_USER_PREVIEW_MAX_LENGTH,
        description=(
            "Optional ASCII/diagram preview (block-scheme / flowchart) the UI "
            "renders in a split pane when this option is selected."
        ),
    )


class AskUserQuestion(BaseModel):
    """One question in an AskUser panel.

    Invariants:

    * A question MUST carry ``options`` OR set ``allow_custom=True`` —
      a question with neither is unanswerable.
    * ``multiSelect=True`` is only meaningful with ``options``.
    * Option labels are unique within the question.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(
        ...,
        min_length=1,
        max_length=ASK_USER_QUESTION_MAX_LENGTH,
        description="The question text. Plain text; markdown permitted, not required.",
    )
    header: str | None = Field(
        default=None,
        min_length=1,
        max_length=ASK_USER_HEADER_MAX_LENGTH,
        description="Optional short chip label (≈12 chars) shown above the question.",
    )
    options: list[AskUserOption] = Field(
        default_factory=list,
        max_length=ASK_USER_MAX_OPTIONS,
        description=(
            "Selectable options. Omit (and set allow_custom=True) for a "
            "free-text question."
        ),
    )
    multiSelect: bool = Field(
        default=False,
        description="When true, the user may select multiple options (only with options).",
    )
    allow_custom: bool = Field(
        default=False,
        description=(
            "When true, the user may submit a free-text custom answer in addition "
            "to (or instead of) the options. Required for a pure free-text question."
        ),
    )

    @model_validator(mode="after")
    def _validate_question(self) -> AskUserQuestion:
        if not self.options and not self.allow_custom:
            raise ValueError(
                "a question must carry options or set allow_custom=True"
            )
        if self.multiSelect and not self.options:
            raise ValueError("multiSelect is only meaningful with options")
        if self.options:
            labels = [opt.label for opt in self.options]
            if len(labels) != len(set(labels)):
                raise ValueError("option labels must be unique within a question")
        return self


class AskUserInput(BaseModel):
    """``AskUser`` LLM-facing input — one or more questions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: list[AskUserQuestion] = Field(
        ...,
        min_length=1,
        max_length=ASK_USER_MAX_QUESTIONS,
        description=(
            "One or more questions to ask the user, each independently "
            "single/multi-select with an optional free-text custom answer."
        ),
    )

    @model_validator(mode="after")
    def _validate_input(self) -> AskUserInput:
        # The output contract (:class:`AskUserAnswer`) pairs each answer to
        # its question by ``question`` text — there is no stable per-question
        # id. Two questions with identical text would make a submitted answer
        # ambiguous (and the host resume validator keys ``parsed_questions``
        # on the text, silently dropping the duplicate). Reject duplicate
        # question text within one input so every posed question is uniquely
        # addressable on resume.
        texts = [q.question for q in self.questions]
        if len(texts) != len(set(texts)):
            raise ValueError("question texts must be unique within one AskUser call")
        return self


class AskUserAnswer(BaseModel):
    """The user's answer to one :class:`AskUserQuestion`.

    ``selected`` carries the picked option labels verbatim (empty for a
    pure free-text answer); ``custom`` carries the free-text answer when
    the question allowed one.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        description="The originating question text, echoed for unambiguous pairing.",
    )
    selected: list[str] = Field(
        default_factory=list,
        description="Picked option labels (verbatim). Empty for a free-text answer.",
    )
    custom: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Free-text custom answer when the question allowed one. Omit "
            "(null) when no custom answer is supplied — an empty string is "
            "not a valid custom answer."
        ),
    )


class AskUserOutput(BaseModel):
    """``AskUser`` LLM-facing output — per-question answers."""

    model_config = ConfigDict(extra="forbid")

    answers: list[AskUserAnswer] = Field(
        ...,
        min_length=1,
        description="One answer per posed question (at least one).",
    )


class AskUserTool(Tool):
    """Core ``AskUser`` tool. Pauses the loop pending the user's answers.

    The host handler intercepts the typed
    :class:`AskUserPauseRequested` raised by :meth:`invoke` and translates
    it into the canonical ``ask_user`` interrupt envelope (event +
    pending interrupt store entry). The LLM-visible :class:`ToolResult`
    is never constructed synchronously — the loop pauses and the answers
    arrive via the resume route, which the host handler re-wraps
    into an :class:`AskUserOutput` for tool-result emission.

    The tool is intentionally **stateless** — the same instance is shared
    across runs and :class:`ToolContext` is the only per-call mutable
    surface.
    """

    name_: ClassVar[str] = ASK_USER_TOOL_NAME
    # What this tool does, so the loop and the surface it advertises can reason
    # about the human-in-the-loop pause by role rather than by the name a
    # deployment happened to give it.
    tool_roles: ClassVar[tuple[ToolRole, ...]] = (ToolRole.asks_user,)
    description_: ClassVar[str] = (
        "Pause the agent loop and ask the user one or more questions, then "
        "return the user's answers as the tool result. Each question may offer "
        "options (single- or multi-select, each with an optional description "
        "and an optional ASCII preview) and/or allow a free-text custom answer. "
        "Use sparingly: each call blocks the run until the user responds."
    )
    # Human-in-the-loop is a first-class interaction primitive, not something
    # the model can "discover" via ToolSearch — and its description shares no
    # lexical overlap with most user prompts, so BM25 clipping silently drops
    # it from the advertised surface on any turn that does not literally name
    # it (measured: a Russian-language "ask me something interesting" turn
    # advertised no AskUser, so
    # the model answered in plain text; the user had to type "Используй
    # AskUser" to make the token match and surface it). Mark it ``always_load``
    # so it survives the layer-2/3 clip exactly like ``ToolSearch``. Precedence
    # is unchanged: ``blocked`` still wins (the headless_eval / autonomous_batch
    # run-mode mask removes it outright), and a non-empty ``visible`` whitelist
    # that omits it still keeps it out — see
    # ``ToolRegistry.compute_effective_surface`` always-load semantics.
    always_load: ClassVar[bool] = True
    # Always loading the tool does not take it out of ToolSearch's ranking,
    # and a request to be asked ("clarify with me", or the same in Russian)
    # shares no word with the description above. Without a hint a host subclassed the
    # tool only to give it one. Joined to the discovery corpus, never sent to
    # the model.
    search_hint: ClassVar[str] = (
        "ask user question options choose clarify confirm pause for answer "
        "ask me which one "
        "спросить спроси пользователя уточнить уточни выбрать выбери варианты "
        "вопрос спроси меня подтвердить"
    )

    @property
    def name(self) -> str:
        return self.name_

    @property
    def definition(self) -> ToolDefinition:
        """Auto-publish :class:`ToolDefinition` from the input JSON schema.

        The nested ``questions[]`` / ``options[]`` models produce a
        ``$defs``/``$ref`` schema; :func:`_inline_json_schema` flattens
        it into a self-contained ``properties`` block so the LLM provider
        sees a fully-resolved tool schema.
        """
        raw_schema = _inline_json_schema(AskUserInput.model_json_schema())
        properties = raw_schema.get("properties", {})
        required = raw_schema.get("required", [])
        if not isinstance(properties, dict):
            properties = {}
        if not isinstance(required, list):
            required = []
        return ToolDefinition(
            name=self.name_,
            description=self.description_,
            parameters=ToolParameterSchema(
                properties=properties,
                required=required,
            ),
        )

    async def invoke(
        self,
        context: ToolContext,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Validate input → raise :class:`AskUserPauseRequested`.

        Contract:
            * Validation failures surface as standard Pydantic
              ``ValidationError`` (the dispatcher wraps it as an
              execution error).
            * Successful validation raises
              :class:`AskUserPauseRequested`, which the dispatcher catches
              and translates into the ``ask_user`` interrupt envelope.
            * The :class:`ToolResult` return value is **never reached**
              on the success path — declared only to satisfy the ABC.
        """
        payload = AskUserInput.model_validate(arguments)
        raise AskUserPauseRequested(payload)


__all__ = [
    "ASK_USER_HEADER_MAX_LENGTH",
    "ASK_USER_MAX_OPTIONS",
    "ASK_USER_MAX_QUESTIONS",
    "ASK_USER_OPTION_DESCRIPTION_MAX_LENGTH",
    "ASK_USER_OPTION_LABEL_MAX_LENGTH",
    "ASK_USER_PREVIEW_MAX_LENGTH",
    "ASK_USER_QUESTION_MAX_LENGTH",
    "ASK_USER_TOOL_NAME",
    "AskUserAnswer",
    "AskUserInput",
    "AskUserOption",
    "AskUserOutput",
    "AskUserPauseRequested",
    "AskUserQuestion",
    "AskUserTool",
]
