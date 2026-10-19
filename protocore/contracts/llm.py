"""ILLMProvider — Protocol for LLM access via provider adapters.

Core sees only structured ``ToolCall(name, args_json)``; the provider
adapter decides provider routing and wire-format details.
"""
from __future__ import annotations

import errno
import socket
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from protocore.contracts.types import Message, StopReason, ToolDefinition


class ProviderFailureReason(StrEnum):
    """What kind of failure a provider reported, as far as the loop acts on it.

    The vendor mapping (which status, which body, which header means which
    reason) belongs to the adapter; this list is what the loop branches on.
    Every member is a decision somewhere: whether a sibling provider is worth a
    try (:data:`PROVIDER_ADVANCE_REASONS`), whether the same request is worth
    another attempt (:data:`PERMANENT_FAILURE_REASONS`), and whether narrowing
    the tool surface could get past it (:data:`SURFACE_INDEPENDENT_FAILURE_REASONS`).
    A reason the adapter cannot place is ``unknown``, which is treated as
    transient and never moves the chain.
    """

    #: The credentials were refused; another key or vendor may accept.
    auth = "auth"
    #: The credentials are revoked or invalid for good.
    auth_permanent = "auth_permanent"
    #: The account is out of balance at this vendor.
    billing = "billing"
    #: A spend cap the HOST enforces refused the call before or instead of the
    #: vendor. Every rung of the chain and every narrowed request meets the same
    #: cap, so nothing in the loop retries or moves past it.
    budget = "budget"
    #: The endpoint could not be reached: refused, reset, unreachable, a name
    #: that does not resolve. Says nothing about the request, so a sibling
    #: provider gets the next try.
    connection = "connection"
    #: The prompt does not fit the model's window; fixed by compaction.
    context_overflow = "context_overflow"
    #: The structured-output form of the request was rejected.
    format_error = "format_error"
    #: An attached image is over the endpoint's limit.
    image_too_large = "image_too_large"
    #: A grammar the endpoint compiles from the tool schemas was refused.
    llama_cpp_grammar_pattern = "llama_cpp_grammar_pattern"
    #: The request needs a long-context tier this account does not have.
    long_context_tier = "long_context_tier"
    #: The endpoint does not serve the model name.
    model_not_found = "model_not_found"
    #: A long-context beta refused for the account's credential type.
    oauth_long_context_beta_forbidden = "oauth_long_context_beta_forbidden"
    #: The endpoint is over capacity right now.
    overloaded = "overloaded"
    #: The request body is over the endpoint's size limit.
    payload_too_large = "payload_too_large"
    #: The vendor refused the content on policy grounds.
    provider_policy_blocked = "provider_policy_blocked"
    #: A quota or rate window is closed.
    rate_limit = "rate_limit"
    #: A 5xx, or a reply the endpoint could not finish.
    server_error = "server_error"
    #: A reasoning payload replayed from another model was refused.
    thinking_signature = "thinking_signature"
    #: The request or the stream ran out of time.
    timeout = "timeout"
    #: The adapter could not place the failure.
    unknown = "unknown"


_R = ProviderFailureReason

#: Failure classes a DIFFERENT provider could plausibly serve. Each names a
#: property of the endpoint that failed (its quota, its capacity, its key, its
#: balance, its catalogue, its reachability) and never a property of the
#: request, so the next provider gets a real chance rather than reproducing the
#: same failure at a second vendor's expense.
#:
#: The omissions are the load-bearing part. A prompt that overflows the window
#: is fixed by compaction, which advancing would skip; an oversized body is
#: fixed by compressing it; a malformed request by the structured-output
#: ladder; an invalid reasoning payload is CAUSED by a model swap, so swapping
#: again compounds it; a policy refusal re-routed to a vendor that might accept
#: the content is a compliance decision, not a default; and a spend cap is the
#: host's, the same on every rung. An error with no verdict at all is excluded
#: on principle: advancing on a failure nobody has characterised turns one
#: broken run into N provider calls and N bills.
#:
#: Members compare equal to their string values, so a verdict whose reason is a
#: plain string from an adapter's own taxonomy is tested against this set too.
PROVIDER_ADVANCE_REASONS: Final[frozenset[ProviderFailureReason]] = frozenset(
    {
        _R.auth,
        _R.auth_permanent,
        _R.billing,
        _R.connection,
        _R.model_not_found,
        _R.overloaded,
        _R.rate_limit,
        _R.server_error,
        _R.timeout,
    }
)

#: Reasons that name a permanent answer. An adapter that attaches one of these
#: without a ``retryable`` flag has already said the request will not succeed by
#: being sent again, and the class default of the exception it raised must not
#: overrule it.
PERMANENT_FAILURE_REASONS: Final[frozenset[ProviderFailureReason]] = frozenset(
    {
        _R.auth,
        _R.auth_permanent,
        _R.billing,
        _R.budget,
        _R.context_overflow,
        _R.format_error,
        _R.image_too_large,
        _R.llama_cpp_grammar_pattern,
        _R.long_context_tier,
        _R.model_not_found,
        _R.oauth_long_context_beta_forbidden,
        _R.payload_too_large,
        _R.provider_policy_blocked,
        _R.thinking_signature,
    }
)

#: Permanent refusals that the wind-down's request would meet again. The
#: wind-down keeps the history and the endpoint and changes only the tool
#: surface (down to the finalizing tool), so a refusal about the account, the
#: model, a content policy, the host's spend cap or something already in the
#: history is not rescued by it. Any other permanent reason (a rejected tool
#: schema, a request or grammar the full surface made too large) may be.
SURFACE_INDEPENDENT_FAILURE_REASONS: Final[frozenset[ProviderFailureReason]] = frozenset(
    {
        _R.auth,
        _R.auth_permanent,
        _R.billing,
        _R.budget,
        _R.image_too_large,
        _R.model_not_found,
        _R.oauth_long_context_beta_forbidden,
        _R.provider_policy_blocked,
        _R.thinking_signature,
    }
)


class ProviderFailure(BaseModel):
    """An adapter's verdict on one provider failure, attached to the raised error.

    Raise it as ``LLMProviderError(message, classified=ProviderFailure(...))``
    (or on any other :class:`LLMError`). The loop reads it to choose between
    moving to the next provider, trying the same one again after a pause,
    winding the run down, and failing it; the terminal event carries it as the
    run's cause. Only the adapter saw the response, so only the adapter can
    fill it in; the loop never guesses a verdict for an error that has none.

    Frozen and JSON-serialisable, because it travels: onto the terminal event,
    into the run snapshot, and across a process boundary on resume.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: ProviderFailureReason
    #: Whether sending the SAME request to the same endpoint again could work.
    #: Outranks the reason when the two disagree: the adapter set both from one
    #: response, and the flag is its answer to exactly this question.
    retryable: bool
    #: The request's structured-output form (a forced tool, a JSON schema) was
    #: refused, and the same question asked as prompted JSON may be answered.
    #: Read only by the strategies that have such a fallback.
    should_fallback: bool = False
    #: A server-stated wait (``Retry-After``, a rate-limit reset header), in
    #: seconds from now. Lengthens the backoff before a retry on the same
    #: endpoint, never past the configured ceiling.
    retry_after_seconds: float | None = Field(default=None, ge=0.0)
    #: The provider's own words, for a person reading why the run ended.
    provider_message: str = ""


class LLMError(Exception):
    """Base for all LLM provider errors.

    Carries the one thing the runtime has to know about a failure it did not
    produce: whether trying the SAME request again could plausibly work.

    The class default is the honest reading of the type — a rate limit, a
    timeout and a silent stream are transient by definition, a context
    overflow never is — and the adapter overrides it per raise when it knows
    better. That order matters: only the adapter has the response in front of
    it, and :class:`LLMProviderError` is the adapters' catch-all, so a 503 and
    a provider's policy refusal arrive as the same type. An adapter that
    classified the response says so by passing ``retryable``; one that did not
    gets the class default and a bounded number of attempts, which costs one
    more call against a cached prompt and is the cheaper of the two mistakes.

    The runtime never retries a context overflow through this flag whatever an
    adapter sets, because :class:`LLMContextWindowExceeded` is answered by
    shrinking the request before the flag is ever read.
    """

    #: Whether re-issuing the same request is worth an attempt. The class
    #: default; a raise may override it.
    retryable: bool = False

    #: The adapter's verdict, when it classified the failure. Hosts that attach
    #: a verdict object of their own after construction still work, since the
    #: loop reads the fields by name; :func:`provider_failure_of` lifts such an
    #: object onto :class:`ProviderFailure`.
    classified: ProviderFailure | None = None

    def __init__(
        self,
        *args: object,
        retryable: bool | None = None,
        classified: ProviderFailure | None = None,
    ) -> None:
        super().__init__(*args)
        if classified is not None:
            self.classified = classified
            # The policy reads the error's own flag before the verdict's, so a
            # catch-all class default of ``True`` would let a permanent verdict
            # be retried. The verdict is the adapter's reading of the response;
            # the error's flag follows it unless the raise says otherwise.
            self.retryable = classified.retryable
        if retryable is not None:
            self.retryable = retryable


class LLMRateLimitError(LLMError):
    """Provider returned 429 / rate-limit signal.

    Transient. The stream loop in :mod:`protocore.runtime.query` handles it
    (and :class:`LLMTimeoutError`) BEFORE :class:`LLMProviderError`: it first
    steps down the run's :class:`IProviderChain` (when one is configured and the
    advance budget is not spent), otherwise re-opens the stream with a bounded
    backoff (``llm_transient_error_retry_max_attempts``), and only goes terminal
    FAILED once both are unavailable / exhausted — preserving any
    already-delivered answer. Distinct type from :class:`LLMProviderError` so
    this differentiated recovery is reachable, and because the loop treats these
    two types as their own verdict: an adapter raises them only for a quota it
    hit or a stream that stopped, both properties of the endpoint rather than of
    the request.
    """

    retryable: bool = True
    """A quota that refilled is the ordinary case; the same request is retried."""


class LLMTimeoutError(LLMError):
    """Provider request timed out (connect/read/write/pool) or the stream stalled.

    Transient — recovered on the same fallback-then-bounded-retry path as
    :class:`LLMRateLimitError` (see its docstring). Distinct from
    :class:`LLMStreamIdleError`, which is the core idle-watchdog's own
    no-delta timeout and drives terminal FAILED via the provider-error path.
    """

    retryable: bool = True
    """A request that ran out of time is retried; nothing was delivered."""


class LLMContextWindowExceeded(LLMError):
    """Request exceeded provider context window.

    Triggers bounded reactive-413 recovery in :mod:`protocore.runtime.query`.
    When exact provider sizes prove that a smaller output cap fits, the request
    may receive one corrective retry before history is rewritten. Otherwise the
    runtime force-compacts once, then retries with strictly decreasing output
    caps until one succeeds or the configured attempt bound is exhausted.
    """

    retryable: bool = False
    """Never. The request is the problem, so re-sending it unchanged fails
    identically; the runtime shrinks it instead, on a path taken before the
    flag is read. An adapter cannot opt a context overflow into the transient
    ladder — :meth:`__init__` does not accept the keyword."""

    def __init__(
        self,
        message: str,
        *,
        context_window: int | None = None,
        input_tokens: int | None = None,
        requested_output_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.context_window = context_window
        self.input_tokens = input_tokens
        self.requested_output_tokens = requested_output_tokens


class LLMProviderError(LLMError):
    """Provider-level error (5xx, transient 4xx, network).

    Distinct from :class:`LLMRateLimitError` (429-specific) and
    :class:`LLMContextWindowExceeded` (context length).

    The adapters' catch-all: a 503 and a provider's policy refusal both arrive
    as this type, so :mod:`protocore.runtime.query` does not advance the run's
    :class:`IProviderChain` on the type alone. It reads the classification the
    adapter pinned on the raised error and steps down only for the failures a
    different endpoint could plausibly serve; an unclassified one, or one whose
    fix is compaction / compression / the structured-output ladder, keeps its
    existing recovery and otherwise goes terminal FAILED.
    """

    retryable: bool = True
    """A 5xx, a reset connection and a refused body are what this type mostly
    carries, and the first two pass on a retry often enough to be worth the
    bounded ladder. An adapter that knows the response is permanent — a bad
    API key, a model name that does not exist — raises it with
    ``retryable=False`` and the run fails on the first answer it got."""


class LLMStreamIdleError(LLMError):
    """LLM stream produced no deltas within the idle-timeout window.

    Raised by the watchdog wrapping :meth:`ILLMProvider.stream_with_tools`
    when no provider event arrived within
    :attr:`LoopConstants.llm_stream_idle_timeout_seconds`.

    The first silence of a streak is tried again on the same provider, since a
    model thinking past the budget looks exactly like a hang from here. A
    silence that repeats is a statement about the endpoint, not about the
    request, so it is answered by the provider chain: the run moves to the next
    configured provider carrying the partial text it had already streamed. Only
    when there is no further provider to move to does it wind the run down, and
    drive terminal FAILED, via the same path as :class:`LLMProviderError`.
    """

    retryable: bool = True
    """A stream that went quiet is as often a queue as a hang, and nothing was
    delivered, so the same request is tried again before the run gives up."""


class MaxOutputTokensExhausted(LLMError):
    """Max-output-token recovery loop exhausted ``max_output_recovery_rounds``.

    Raised after the loop synthesised a "Resume directly" continuation
    nudge :attr:`LoopConstants.max_output_recovery_rounds` times and
    the model still terminated with ``finish_reason="length"``. Drives
    terminal FAILED with ``reason="output_length_exhausted"``.
    """


class LLMObservabilityContext(BaseModel):
    """Non-secret provider-call correlation metadata."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str | None = None
    run_id: str | None = None
    parent_run_id: str | None = None
    session_id: str | None = None
    principal_id: str | None = None
    agent_id: str | None = None
    call_purpose: str | None = None
    call_category: str | None = None


class LLMRequest(BaseModel):
    """Single inference request (tool-using turn or structured completion)."""

    model_config = ConfigDict(frozen=True)

    model: str
    messages: Sequence[Message]
    tools: Sequence[ToolDefinition] = Field(default_factory=list)
    max_tokens: int = 4096
    #: ``None`` means the caller has no opinion: an adapter omits the field and
    #: the host (a per-model setting, or the server's own generation config)
    #: decides. A number is the caller's explicit choice and wins over both.
    temperature: float | None = None
    #: Provider-neutral control keys; an adapter reads the ones it knows and
    #: never sends them to the wire verbatim. Two of them constrain the tool
    #: choice, and a request carries at most one:
    #:
    #: * ``forced_tool_choice`` — a tool NAME the reply must call; render it as
    #:   the provider's native single-tool choice (OpenAI-compatible:
    #:   ``{"type": "function", "function": {"name": ...}}``).
    #: * ``tool_choice_required`` — ``True`` when the reply must call SOME
    #:   advertised tool and may not answer in prose; render it as the native
    #:   "any tool" choice (OpenAI-compatible: ``tool_choice="required"``).
    #:
    #: An adapter without support for either ignores it — the runtime drops
    #: what a constrained turn returns beyond what it admits, so the request is
    #: merely less efficient. A provider that rejects the constraint outright
    #: should surface that as a non-retryable error, not a transient one.
    #: ``enable_thinking`` travels with ``reasoning_effort``; an explicit
    #: ``False`` must reach the wire as thinking off, not as "provider default".
    extra: dict[str, Any] = Field(default_factory=dict)
    observability: LLMObservabilityContext | None = None


class LLMResponseUsage(BaseModel):
    """Token usage summary.

    Adapters normalise to the Anthropic-style shape (``cache_read_input_tokens``
    / ``cache_creation_input_tokens``); OpenAI-style ``cached_tokens`` is
    surfaced separately for adapters (vLLM) that emit it under
    ``usage.prompt_tokens_details.cached_tokens``.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_read_input_tokens: int | None = None
    """Anthropic-shape — tokens served from cache this turn."""

    cache_creation_input_tokens: int | None = None
    """Anthropic-shape — tokens written to cache this turn."""

    cached_tokens: int | None = None
    """OpenAI / vLLM shape — cached prompt tokens (mirrors
    ``prompt_tokens_details.cached_tokens``)."""

    response_cost_usd: float | None = None
    """Best-effort response cost in USD when the adapter can
    source it from provider metadata or configured local-model pricing."""


class LLMStreamEvent(BaseModel):
    """Anthropic-aligned streaming event.

    Names: ``message_start`` / ``content_block_start`` / ``content_block_delta``
    / ``content_block_stop`` / ``tool_use_start`` / ``tool_use_input_delta``
    / ``tool_use_stop`` / ``message_stop``.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ProviderDeltaKind(StrEnum):
    """Normalised provider-stream delta kinds.

    Adapters translate upstream provider events (Anthropic SSE, OpenAI
    chat.completions stream, vLLM custom stream, ...) into a sequence of
    :class:`ProviderDelta` envelopes keyed on these kinds.
    """

    text = "text"
    thinking = "thinking"
    progress = "progress"
    tool_use_start = "tool_use_start"
    tool_use_input = "tool_use_input"
    tool_use_stop = "tool_use_stop"
    finish = "finish"
    usage = "usage"


class ProviderDelta(BaseModel):
    """Normalised provider streaming envelope.

    Every adapter translates upstream events to this primitive. Core sees
    only this — the loop never inspects vendor-specific shapes.

    Invariants enforced by the adapter:

    1. Every ``tool_use_start`` is followed by exactly one
       ``tool_use_stop`` with matching ``tool_call_id``, even on early
       termination.
    2. Order is causal — thinking before text before tool_use (per
       provider's emission); core does not reorder.
    3. Final ``kind=usage`` envelope MUST surface cumulative
       ``{input_tokens, output_tokens}``.
    """

    model_config = ConfigDict(frozen=True)

    kind: ProviderDeltaKind
    content: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_input_delta: str | None = None
    tool_input_final: dict[str, Any] | None = None
    finish_reason: Literal["stop", "tool_use", "length", "content_filter"] | None = None
    usage: dict[str, Any] | None = None
    is_block_end: bool = False
    truncated_by_output_cap: bool = False
    """Set ``True`` ONLY on a synthetic ``tool_use_stop`` delta emitted because
    ``finish_reason="length"`` arrived while the parser still held an open
    tool_call buffer (incomplete args JSON). Default ``False`` everywhere else,
    including:
    * the natural close path (``finish_reason="tool_calls"``);
    * the connection-drop recovery path
      (:func:`synthesize_tool_use_stops_for_open_tools`) — the open call
      was killed by a transport-layer abort, not by the output-token cap.

    The flag propagates from this envelope to :class:`ToolCall.truncated_by_output_cap`
    and drives the mid-tool-call recovery branch in
    :func:`protocore.runtime.query._stream_one_assistant_message`."""

    args_partial_truncated: bool = False
    """Set ``True`` on a ``tool_use_stop`` delta when the SSE parser saw an
    incomplete JSON args stream (stage-4 brace balancing had to synthesise
    closers), REGARDLESS of the ``finish_reason`` that closed the stream.
    Default ``False`` for clean parses.

    Distinct from :attr:`truncated_by_output_cap`: the latter is the narrow
    "output-token cap" signal (``finish_reason="length"``), while this flag
    covers every wire-level truncation signature — including the case where
    models emit ``finish_reason="stop"`` after only ``{`` of the args JSON,
    leaving the call orphaned. The loop's surface-truncation-to-agent branch in
    :func:`protocore.runtime.query._stream_one_assistant_message` keys off this
    flag in combination with ``finish_reason="stop"`` to emit a synthetic error
    :class:`~protocore.contracts.types.ToolResult` so the agent learns to chunk
    large outputs instead of silently failing."""

    reasoning_items: tuple[dict[str, Any], ...] | None = None
    """Opaque reasoning records to keep on the assistant message this stream
    builds (:attr:`~protocore.contracts.types.Message.reasoning_items`).

    Read from any delta that carries them, in order; adapters put them on the
    ``finish`` delta, since a provider hands its encrypted reasoning over once
    the item is complete. A stream that never finishes keeps none, which is
    right: a partial turn is not one the provider can resume from."""


class LLMResponse(BaseModel):
    """Final response after stream completion."""

    model_config = ConfigDict(frozen=True)

    message: Message
    stop_reason: StopReason
    usage: LLMResponseUsage = Field(default_factory=LLMResponseUsage)


@dataclass(frozen=True, slots=True)
class CacheBreakpoint:
    """A single cache-control placement hint for prompt caching.

    Pure-data, immutable. Computed by
    :func:`protocore.runtime.prompt_caching.apply_system_and_3` from a
    list of :class:`Message` envelopes; consumed by LLM adapters
    (Anthropic / OpenRouter / Bedrock) that translate breakpoints into
    wire-format ``cache_control`` blocks.

    Fields:
        message_index: zero-based index into the messages list.
        cache_control_type: TTL bucket — ``"ephemeral"`` (Anthropic
            default), ``"5m"`` (Anthropic explicit), or ``"1h"``
            (Anthropic 1-hour beta — not available on Bedrock).
        rationale: short human-readable label for telemetry / audit
            (``"system_prefix"``, ``"last_message"``, etc.).
    """

    message_index: int
    cache_control_type: Literal["ephemeral", "5m", "1h"]
    rationale: str


@runtime_checkable
class ILLMProvider(Protocol):
    """Adapter Protocol over an LLM endpoint.

    Concrete provider adapters live outside the core.
    """

    def stream_with_tools(
        self, request: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent | ProviderDelta]:
        """Stream provider events; yields until ``message_stop``.

        Implementations are async generators; declared here without
        ``async def`` per `mypy.readthedocs.io/en/stable/more_types.html#asynchronous-iterators`.
        """
        ...

    async def complete_structured(
        self,
        request: LLMRequest,
        response_schema: dict[str, Any],
    ) -> LLMResponse:
        """Post-loop structured completion.

        Invariant: do NOT combine ``response_format`` with ``tools`` in
        the same call. Use this only AFTER the tool loop terminates (per
        v1 invariant — broken on vLLM #16313, llama.cpp #11847,
        SGLang #5178).
        """
        ...

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        """Post-loop plain completion — free-form text, no decoding constraint.

        Deliberately a separate method, NOT :meth:`complete_structured`
        called without a schema. The two ask the model for different
        things and are not interchangeable:

        * :meth:`complete_structured` constrains decoding to a JSON
          schema. The caller fixes the shape in advance and gets back a
          machine-parsable object; the model spends its budget filling
          the declared fields.
        * :meth:`complete_text` states the required structure in the
          prompt and lets the model write the document. Headings,
          ordering and level of detail are part of the answer, not of a
          wire contract, and the caller consumes the text as text.

        Callers whose output is a *document* — a summary, a memory fold,
        a report — must use this method. Under a schema a model fills
        every declared field even when the source material says nothing
        about it, so a fact that is simply absent comes back as a
        plausible-looking default instead of being left out: the reply
        stays well-formed while its content drifts away from the input.
        A schema constrains syntax, never fidelity.

        Non-streaming: the caller wants one finished document rather than
        deltas. Like :meth:`complete_structured`, it is a post-loop call —
        no ``tools`` are expected on the request.
        """
        ...

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Provider-exact token count. Used for budget pre-flight."""
        ...


@runtime_checkable
class IRequestTokenCounter(Protocol):
    """Optional provider capability: the size of a request as the server renders it.

    Not part of :class:`ILLMProvider`, so a provider that cannot count is left
    exactly as it is. The runtime looks the method up on the provider's class and
    asks it only when its own estimate is close to a limit
    (:attr:`LoopConstants.exact_token_count_margin_ratio`), so an endpoint pays
    for the round-trip only on the requests whose size decides something.

    The count covers what the server would tokenize for this request: the
    messages rendered through its chat template together with the tool
    definitions and the generation prompt. It is the prompt's size, not the
    prompt plus ``max_tokens``.

    Return ``None`` when this endpoint cannot count the request (the capability
    is switched off for it, or the server has no counting route); the runtime
    then uses its estimate and says nothing. Raise when counting failed; the
    runtime logs a warning and uses its estimate. Either way the request that is
    eventually sent is the one that would have been sent without the count,
    apart from an output cap fitted to the measured size.
    """

    async def count_request_tokens(self, request: LLMRequest) -> int | None:
        """The prompt tokens ``request`` renders to, or ``None`` when unknown."""
        ...


@runtime_checkable
class IProviderChain(Protocol):
    """The ordered providers one consumer may run on, plus a cursor over them.

    An installation states a preference order per consumer — the leader of a
    scope, or one subagent definition — and the adapter layer filters it down to
    what is actually usable before the run starts (a deleted row, a disabled
    row, a model the caller's tariff does not include). What reaches core is the
    surviving order and a cursor, because only core is standing in the right
    place when a provider fails mid-stream: it owns the partial assistant turn
    the user has already seen, the context rebuild, and the state event that
    explains the swap. A wrapper below core cannot un-publish deltas that have
    already streamed.

    :meth:`advance` moves the whole provider — endpoint, key, capability flags,
    declared window — and not a model name. A name alone reaches the previous
    vendor's transport asking for a model it does not serve.

    The cursor is one-way. Demoting is a statement that the current provider is
    unhealthy right now; promoting back inside the same run would re-pay the
    cold prefix cache in both directions and multiply the model swaps, and a
    swap is itself the documented cause of an invalid reasoning payload on an
    earlier assistant turn.
    """

    def current(self) -> ILLMProvider:
        """The provider serving right now."""
        ...

    def current_model_name(self) -> str:
        """The model identity of :meth:`current`, for the engine's config."""
        ...

    async def advance(self, *, reason: str) -> bool:
        """Step to the next usable provider. ``False`` when exhausted.

        Async because materialising a provider is: a rung is built when the run
        actually reaches it, never before. A chain that constructed every rung
        up front would open a connection pool per configured model on every run,
        including the overwhelming majority that never leave position 0.
        """
        ...

    def attempted(self) -> Sequence[tuple[str, str]]:
        """``(provider_name, reason)`` for every rung already ruled out.

        Ordered as they were ruled out, and the material for an error message
        that says what was tried rather than only that nothing worked.
        """
        ...


#: Socket errors that mean the endpoint could not be reached, whatever the
#: exception type a platform or client wraps them in.
_UNREACHABLE_ERRNOS: Final[frozenset[int]] = frozenset(
    {
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.EPIPE,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
        errno.ENETDOWN,
        errno.EHOSTDOWN,
    }
)

#: How far down a ``__cause__`` / ``__context__`` chain the classifier looks.
#: HTTP clients wrap the socket error two or three levels deep; a longer chain
#: is an application error that happened to be raised while handling one.
_CAUSE_CHAIN_DEPTH: Final[int] = 8


def classify_transport_exception(exc: BaseException) -> ProviderFailure | None:
    """The verdict for a failure to reach the endpoint at all, or ``None``.

    For adapter authors: call it in the ``except`` around the HTTP call and
    attach what it returns, so a dead endpoint moves the run to the next
    provider instead of ending it::

        except Exception as exc:
            failure = classify_transport_exception(exc)
            if failure is None:
                raise
            raise LLMProviderError(str(exc), classified=failure) from exc

    Recognises the standard library's view of a transport failure, which every
    HTTP client ends up wrapping: a refused, reset or aborted connection, a
    broken pipe, an unreachable host or network, a name that does not resolve
    (``connection``), and a timeout (``timeout``). The exception's own type is
    checked first, then its ``__cause__`` and ``__context__`` chain, since a
    client raises its own type ``from`` the socket error. Anything else returns
    ``None``: a certificate failure or a permission error is a configuration
    problem a retry or a sibling provider does not fix, and guessing a verdict
    for it would spend the chain on it.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_CAUSE_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        failure = _transport_failure(current)
        if failure is not None:
            return failure
        current = current.__cause__ or current.__context__
    return None


def _transport_failure(exc: BaseException) -> ProviderFailure | None:
    message = str(exc) or type(exc).__name__
    if isinstance(exc, TimeoutError):
        return ProviderFailure(
            reason=ProviderFailureReason.timeout, retryable=True, provider_message=message
        )
    unreachable = (
        isinstance(exc, ConnectionError | socket.gaierror)
        or (isinstance(exc, OSError) and exc.errno in _UNREACHABLE_ERRNOS)
    )
    if unreachable:
        return ProviderFailure(
            reason=ProviderFailureReason.connection, retryable=True, provider_message=message
        )
    return None


def provider_failure_of(exc: BaseException) -> ProviderFailure | None:
    """The typed verdict a raised error carries, or ``None`` when it has none.

    A :class:`ProviderFailure` attached by the adapter is returned as is. A
    verdict object of the adapter's own type is lifted field by field, with a
    reason outside :class:`ProviderFailureReason` read as ``unknown``. The two
    error types that are their own verdict (a rate limit and a timeout, both
    properties of the endpoint) read as one, as do the idle watchdog's stall and
    a context overflow. A bare :class:`LLMProviderError`, or anything outside
    the :class:`LLMError` family, says nothing and nothing is invented for it.
    """
    classified = getattr(exc, "classified", None)
    if isinstance(classified, ProviderFailure):
        return classified
    lifted = _lift_verdict(exc, classified) if classified is not None else None
    if lifted is not None:
        return lifted
    for error_type, reason in _FAILURE_BY_TYPE.items():
        if isinstance(exc, error_type):
            return ProviderFailure(
                reason=reason,
                retryable=bool(getattr(exc, "retryable", False)),
                provider_message=str(exc),
            )
    return None


_FAILURE_BY_TYPE: Final[dict[type[BaseException], ProviderFailureReason]] = {
    LLMRateLimitError: ProviderFailureReason.rate_limit,
    LLMTimeoutError: ProviderFailureReason.timeout,
    LLMStreamIdleError: ProviderFailureReason.timeout,
    LLMContextWindowExceeded: ProviderFailureReason.context_overflow,
}


def _positive_seconds(value: object) -> float | None:
    """A usable wait in seconds, or ``None`` for a missing or nonsensical one."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds > 0.0 else None


def _lift_verdict(exc: BaseException, verdict: object) -> ProviderFailure | None:
    raw_reason = getattr(verdict, "reason", None)
    if raw_reason is None:
        return None
    value = str(getattr(raw_reason, "value", raw_reason))
    try:
        reason = ProviderFailureReason(value)
    except ValueError:
        reason = ProviderFailureReason.unknown
    flag = getattr(verdict, "retryable", None)
    if not isinstance(flag, bool):
        flag = bool(getattr(exc, "retryable", False)) and value not in PERMANENT_FAILURE_REASONS
    retry_after = _positive_seconds(getattr(verdict, "retry_after_seconds", None))
    message = getattr(verdict, "provider_message", "")
    return ProviderFailure(
        reason=reason,
        retryable=flag,
        should_fallback=bool(getattr(verdict, "should_fallback", False)),
        retry_after_seconds=retry_after,
        provider_message=message if isinstance(message, str) else "",
    )


__all__ = [
    "PERMANENT_FAILURE_REASONS",
    "PROVIDER_ADVANCE_REASONS",
    "SURFACE_INDEPENDENT_FAILURE_REASONS",
    "CacheBreakpoint",
    "ILLMProvider",
    "IProviderChain",
    "IRequestTokenCounter",
    "LLMContextWindowExceeded",
    "LLMError",
    "LLMObservabilityContext",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseUsage",
    "LLMStreamEvent",
    "LLMStreamIdleError",
    "LLMTimeoutError",
    "MaxOutputTokensExhausted",
    "ProviderDelta",
    "ProviderDeltaKind",
    "ProviderFailure",
    "ProviderFailureReason",
    "classify_transport_exception",
    "provider_failure_of",
]
