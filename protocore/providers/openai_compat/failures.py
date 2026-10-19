"""Reading a refusal: from status, body and headers to the core's verdict.

The loop decides retry, fallback and wind-down from the
:class:`~protocore.contracts.llm.ProviderFailure` attached to the error an
adapter raises. Only the adapter has the response in front of it, so this
module is where a chat-completions refusal becomes that verdict: which reason,
whether the same request is worth another try, how long the server asked to
wait, and, for a prompt that did not fit, the sizes the server stated.

Everything here is a pure function of the response, so the mapping is tested
against recorded refusals rather than live servers.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Final

from protocore.contracts.llm import (
    LLMContextWindowExceeded,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    ProviderFailure,
    ProviderFailureReason,
)

_R = ProviderFailureReason

#: The longest wait a server's header is believed for. A rate window that
#: reopens tomorrow is a quota, not a pause; the loop clamps the wait to its own
#: ceiling anyway, and a header further out is more likely a clock error.
MAX_STATED_WAIT_SECONDS: Final[float] = 86_400.0

#: Words a context-window refusal is written in, across servers. Read only on
#: the statuses that carry a refused request (400, 413, 422) and on errors
#: inside a stream, never on a 429, where "tokens" means a rate window.
_CONTEXT_MARKERS: Final[tuple[str, ...]] = (
    "context length",
    "context_length",
    "maximum context",
    "context window",
    "exceeds the available context size",
    "exceed_context_size",
    "prompt is too long",
    "too many tokens",
    "reduce the length",
    "exceeds the model",
)

#: Body codes or types that mean the account is out of money or allowance,
#: whatever status carried them (the OpenAI API sends ``insufficient_quota`` on
#: a 429 that would otherwise read as a short pause).
_BILLING_CODES: Final[frozenset[str]] = frozenset(
    {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "billing_error"}
)
_RATE_CODES: Final[frozenset[str]] = frozenset(
    {"rate_limit_error", "rate_limit_exceeded", "requests_limit_exceeded", "tokens_limit_exceeded"}
)
_OVERLOAD_CODES: Final[frozenset[str]] = frozenset({"overloaded_error", "capacity_exceeded", "server_overloaded"})
_AUTH_CODES: Final[frozenset[str]] = frozenset({"authentication_error", "permission_error", "invalid_api_key"})
_POLICY_CODES: Final[frozenset[str]] = frozenset({"content_policy_violation", "content_filter"})

#: A refusal of the model rather than of the request. Read as an unknown model,
#: because another rung of the chain may be served where this one is not.
_MODEL_MARKERS: Final[tuple[str, ...]] = (
    "model_not_found",
    "does not support this model",
    "model does not exist",
    "no such model",
    "unknown model",
)


@dataclass(frozen=True, slots=True)
class ErrorBody:
    """What an error body says: its ``type``, ``code`` and ``message``, and the object itself.

    Empty strings for what it does not say. Accepts the wrapped shape
    (``{"error": {...}}``) a status response carries and the bare error object
    a stream chunk carries.
    """

    type: str = ""
    code: str = ""
    message: str = ""
    fields: Mapping[str, Any] | None = None


def read_error_body(text: str) -> ErrorBody:
    """Parse an error body; an unreadable one says nothing beyond its text."""
    try:
        data = json.loads(text)
    except ValueError:
        return ErrorBody(message=text.strip()[:500])
    if not isinstance(data, dict):
        return ErrorBody()
    error = data.get("error", data)
    if isinstance(error, str):
        return ErrorBody(message=error)
    if not isinstance(error, dict):
        return ErrorBody()
    code = error.get("code")
    return ErrorBody(
        type=str(error.get("type") or ""),
        code="" if code is None else str(code),
        message=str(error.get("message") or ""),
        fields=error,
    )


def is_context_overflow(status: int, text: str) -> bool:
    """Whether a refusal says the prompt did not fit the model's window."""
    if status not in (400, 413, 422, 500):
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _CONTEXT_MARKERS)


_WINDOW = re.compile(r"maximum context length is (\d+) tokens", re.IGNORECASE)
_SPLIT = re.compile(r"\((\d+) (?:tokens )?in the messages, (\d+) (?:tokens )?in the completion\)", re.IGNORECASE)
_MESSAGES_RESULTED = re.compile(r"your messages resulted in (\d+) tokens", re.IGNORECASE)
_OUTPUT_AND_PROMPT = re.compile(
    r"you requested (\d+) output tokens and your prompt contains at least (\d+) input tokens", re.IGNORECASE
)
_ANTHROPIC = re.compile(r"prompt is too long: (\d+) tokens > (\d+) maximum", re.IGNORECASE)
_LLAMA_CPP = re.compile(
    r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)", re.IGNORECASE
)


def context_overflow_sizes(text: str) -> dict[str, int]:
    """The window, prompt and requested output a context refusal states.

    Keys are the keyword arguments of
    :class:`~protocore.contracts.llm.LLMContextWindowExceeded`; a size the
    refusal does not state is absent. The core sizes its one corrective retry
    from these numbers when it has them (the output cap it can still afford)
    and otherwise from its own estimate, which is what ran short in the first
    place. Recognised wordings:

    * vLLM and OpenAI: "maximum context length is 32768 tokens. However, you
      requested 33000 tokens (30000 in the messages, 3000 in the completion)",
      "... your messages resulted in 130000 tokens", and "you requested 1000
      output tokens and your prompt contains at least 32001 input tokens";
    * Anthropic-shaped gateways: "prompt is too long: 210000 tokens > 200000
      maximum";
    * llama.cpp server: ``n_prompt_tokens`` and ``n_ctx`` on the error object,
      or "request (40000 tokens) exceeds the available context size (32768
      tokens)".
    """
    sizes: dict[str, int] = {}
    body = read_error_body(text)
    if body.fields is not None:
        prompt = body.fields.get("n_prompt_tokens")
        window = body.fields.get("n_ctx")
        if isinstance(window, int) and not isinstance(window, bool) and window > 0:
            sizes["context_window"] = window
        if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt > 0:
            sizes["input_tokens"] = prompt
    if match := _WINDOW.search(text):
        sizes["context_window"] = int(match.group(1))
    if match := _SPLIT.search(text):
        sizes["input_tokens"] = int(match.group(1))
        sizes["requested_output_tokens"] = int(match.group(2))
    elif match := _OUTPUT_AND_PROMPT.search(text):
        sizes["requested_output_tokens"] = int(match.group(1))
        sizes["input_tokens"] = int(match.group(2))
    elif match := _MESSAGES_RESULTED.search(text):
        sizes["input_tokens"] = int(match.group(1))
    if match := _ANTHROPIC.search(text):
        sizes["input_tokens"] = int(match.group(1))
        sizes["context_window"] = int(match.group(2))
    if match := _LLAMA_CPP.search(text):
        sizes["input_tokens"] = int(match.group(1))
        sizes["context_window"] = int(match.group(2))
    return sizes


_DURATION = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


def _duration_seconds(value: str) -> float | None:
    """A Go-style duration (``6m0s``, ``1.5s``, ``20ms``) in seconds."""
    match = _DURATION.fullmatch(value.strip())
    if match is None or not any(match.groups()):
        return None
    hours, minutes, seconds, millis = (float(group) if group else 0.0 for group in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def _bounded(seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0 or seconds > MAX_STATED_WAIT_SECONDS:
        return None
    return seconds


def retry_after_seconds(
    headers: Mapping[str, str], *, status: int, now: datetime | None = None
) -> float | None:
    """The wait the server asked for, in seconds from now, or ``None``.

    ``retry-after-ms`` first (the OpenAI API's precise form), then
    ``Retry-After`` in seconds or as an HTTP date. On a 429 with neither, the
    later of ``x-ratelimit-reset-requests`` and ``x-ratelimit-reset-tokens``:
    the response does not say which window closed, and asking again before
    both reopen is the cheapest way to be refused twice. A value that is
    unparseable, in the past, or more than a day out is ignored.
    """
    lowered = {str(key).lower(): str(value).strip() for key, value in headers.items()}
    if "retry-after-ms" in lowered:
        try:
            waited = _bounded(float(lowered["retry-after-ms"]) / 1000)
        except ValueError:
            waited = None
        if waited is not None:
            return waited
    stated = lowered.get("retry-after")
    if stated:
        try:
            return _bounded(float(stated))
        except ValueError:
            pass
        try:
            instant = parsedate_to_datetime(stated)
        except (TypeError, ValueError):
            instant = None
        if instant is not None:
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=UTC)
            return _bounded((instant - (now or datetime.now(UTC))).total_seconds())
    if status != 429:
        return None
    resets = [
        _duration_seconds(lowered[key])
        for key in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")
        if key in lowered
    ]
    known = [value for value in resets if value is not None]
    return _bounded(max(known)) if known else None


def classify_failure(
    status: int,
    text: str,
    headers: Mapping[str, str] | None = None,
    *,
    constrained: bool = False,
) -> ProviderFailure:
    """The verdict on a request the endpoint answered with ``status`` and ``text``.

    ``constrained`` says the request carried a structured-output constraint (a
    forced or required tool choice, a ``response_format``); a plain refusal of
    such a request is marked ``should_fallback``, because the same question
    asked as prompted JSON may be answered where the constraint was not.

    Rate limits, timeouts, overload and server errors stay retryable and
    carry the wait the server asked for. Every other 4xx is the request or the
    account, and so is a body that names one of those errors whatever status
    carried it: a stream reports its error in a chunk, after the 200.
    """
    body = read_error_body(text)
    kind, code = body.type.lower(), body.code.lower()
    names = {kind, code}
    lowered = f"{code} {kind} {body.message}".lower()
    message = body.message or text.strip()[:500]
    wait = retry_after_seconds(headers or {}, status=status)

    def verdict(reason: ProviderFailureReason, retryable: bool, **extra: Any) -> ProviderFailure:
        return ProviderFailure(reason=reason, retryable=retryable, provider_message=message, **extra)

    if status == 402 or names & _BILLING_CODES:
        return verdict(_R.billing, False)
    if status in (401, 403) or names & _AUTH_CODES:
        return verdict(_R.auth, False)
    if names & _POLICY_CODES:
        return verdict(_R.provider_policy_blocked, False)
    if status == 429 or names & _RATE_CODES:
        return verdict(_R.rate_limit, True, retry_after_seconds=wait)
    if status in (408, 504):
        return verdict(_R.timeout, True, retry_after_seconds=wait)
    if status in (503, 529) or names & _OVERLOAD_CODES:
        return verdict(_R.overloaded, True, retry_after_seconds=wait)
    if status == 404 or any(marker in lowered for marker in _MODEL_MARKERS):
        return verdict(_R.model_not_found, False)
    if is_context_overflow(status, text):
        return verdict(_R.context_overflow, False)
    if status == 413:
        return verdict(_R.payload_too_large, False)
    if 400 <= status < 500:
        return verdict(_R.format_error, False, should_fallback=constrained)
    return verdict(_R.server_error, True, retry_after_seconds=wait)


def stream_error_status(chunk_error: Any) -> int:
    """The status an error reported inside a stream stands for.

    Servers put the HTTP-style status in ``code`` when they have one (vLLM,
    llama.cpp); otherwise the stream failed after it started, which is the
    server's fault until shown otherwise.
    """
    if isinstance(chunk_error, dict):
        code = chunk_error.get("code")
        if isinstance(code, int) and not isinstance(code, bool) and 400 <= code <= 599:
            return code
    return 500


def error_for_response(
    endpoint_name: str,
    status: int,
    text: str,
    headers: Mapping[str, str] | None = None,
    *,
    constrained: bool = False,
) -> LLMError:
    """The error to raise for a refused request, verdict attached.

    A prompt that did not fit is raised as
    :class:`~protocore.contracts.llm.LLMContextWindowExceeded` with the sizes
    the server stated, so the loop compacts instead of retrying. A rate limit
    and a timeout get their own types, which the loop answers on the
    fallback-then-retry path; everything else is an
    :class:`~protocore.contracts.llm.LLMProviderError`.
    """
    if is_context_overflow(status, text):
        return LLMContextWindowExceeded(
            f"{endpoint_name}: HTTP {status}: {read_error_body(text).message or text[:300]}",
            **context_overflow_sizes(text),
        )
    failure = classify_failure(status, text, headers, constrained=constrained)
    said = f"{endpoint_name}: HTTP {status}: {failure.provider_message[:500]}"
    body = read_error_body(text)
    # A numeric code repeats the status; a named one is what to search for.
    if body.code and not body.code.isdigit():
        said = f"{said} ({body.code})"
    if failure.reason is _R.rate_limit:
        return LLMRateLimitError(said, classified=failure)
    if failure.reason is _R.timeout:
        return LLMTimeoutError(said, classified=failure)
    return LLMProviderError(said, classified=failure)


__all__ = [
    "MAX_STATED_WAIT_SECONDS",
    "ErrorBody",
    "classify_failure",
    "context_overflow_sizes",
    "error_for_response",
    "is_context_overflow",
    "read_error_body",
    "retry_after_seconds",
    "stream_error_status",
]
