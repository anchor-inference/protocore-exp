"""From a refusal to the core's verdict, checked on recorded refusal bodies."""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from protocore.conformance import assert_adapter_failure
from protocore.contracts.llm import (
    PERMANENT_FAILURE_REASONS,
    PROVIDER_ADVANCE_REASONS,
    LLMContextWindowExceeded,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    ProviderFailureReason,
)
from protocore.providers.openai_compat import (
    classify_failure,
    context_overflow_sizes,
    error_for_response,
    retry_after_seconds,
)
from protocore.providers.openai_compat.failures import read_error_body, stream_error_status

R = ProviderFailureReason


def _body(message: str, type_: str = "", code: object = None) -> str:
    error: dict[str, object] = {"message": message}
    if type_:
        error["type"] = type_
    if code is not None:
        error["code"] = code
    return json.dumps({"error": error})


@pytest.mark.parametrize(
    ("status", "body", "reason", "retryable"),
    [
        (401, _body("Incorrect API key provided", "invalid_request_error", "invalid_api_key"), R.auth, False),
        (403, _body("forbidden"), R.auth, False),
        (402, _body("Insufficient credits"), R.billing, False),
        # The OpenAI API sends a spent quota on a 429; it is not a pause.
        (429, _body("You exceeded your current quota", "insufficient_quota", "insufficient_quota"), R.billing, False),
        (429, _body("Rate limit reached for requests", "requests", "rate_limit_exceeded"), R.rate_limit, True),
        (400, _body("slow down", "rate_limit_error"), R.rate_limit, True),
        (408, "request timeout", R.timeout, True),
        (504, "<html>gateway timeout</html>", R.timeout, True),
        (503, _body("Loading model", "unavailable_error", 503), R.overloaded, True),
        (529, _body("Overloaded", "overloaded_error"), R.overloaded, True),
        (404, _body("The model `x` does not exist", "invalid_request_error", "model_not_found"), R.model_not_found, False),
        (400, _body("this account does not support this model"), R.model_not_found, False),
        (400, _body("flagged by the safety system", "invalid_request_error", "content_policy_violation"), R.provider_policy_blocked, False),
        (413, _body("Request Entity Too Large"), R.payload_too_large, False),
        (400, _body("'messages' must contain the word 'json'", "invalid_request_error"), R.format_error, False),
        (500, _body("Internal error", "server_error"), R.server_error, True),
        (502, "bad gateway", R.server_error, True),
    ],
)
def test_each_refusal_reads_as_the_verdict_the_loop_acts_on(
    status: int, body: str, reason: ProviderFailureReason, retryable: bool
) -> None:
    failure = classify_failure(status, body)
    assert failure.reason is reason
    assert failure.retryable is retryable
    assert failure.provider_message


def test_verdicts_line_up_with_what_the_loop_does_with_them() -> None:
    # A dead or saturated endpoint moves the chain; a refused request does not.
    assert classify_failure(503, "").reason in PROVIDER_ADVANCE_REASONS
    assert classify_failure(400, _body("bad field")).reason in PERMANENT_FAILURE_REASONS
    assert classify_failure(400, _body("bad field")).reason not in PROVIDER_ADVANCE_REASONS


def test_a_constrained_request_refused_outright_may_fall_back_to_prompted_json() -> None:
    body = _body("tool_choice 'required' is not supported", "invalid_request_error")
    assert classify_failure(400, body, constrained=True).should_fallback is True
    assert classify_failure(400, body).should_fallback is False
    # A refusal that is not about the request's form never asks for the fallback.
    assert classify_failure(503, body, constrained=True).should_fallback is False


def test_the_server_stated_wait_reaches_the_verdict() -> None:
    failure = classify_failure(429, _body("slow down"), {"Retry-After": "30"})
    assert failure.retry_after_seconds == 30.0
    overloaded = classify_failure(503, "", {"retry-after-ms": "1500"})
    assert overloaded.retry_after_seconds == 1.5
    assert classify_failure(400, _body("x"), {"Retry-After": "30"}).retry_after_seconds is None


@pytest.mark.parametrize(
    ("headers", "status", "expected"),
    [
        ({"retry-after": "12"}, 429, 12.0),
        ({"Retry-After": "2.5"}, 503, 2.5),
        ({"retry-after-ms": "250"}, 429, 0.25),
        ({"retry-after-ms": "junk", "retry-after": "4"}, 429, 4.0),
        ({"retry-after": "Tue, 20 Oct 2026 12:00:30 GMT"}, 429, 30.0),
        ({"retry-after": "Tue, 20 Oct 2026 11:59:00 GMT"}, 429, None),
        ({"retry-after": "soon"}, 429, None),
        ({"retry-after": "0"}, 429, None),
        ({"retry-after": "999999"}, 429, None),
        ({"x-ratelimit-reset-requests": "1s", "x-ratelimit-reset-tokens": "6m0s"}, 429, 360.0),
        ({"x-ratelimit-reset-tokens": "20ms"}, 429, 0.02),
        ({"x-ratelimit-reset-tokens": "1h2m3.5s"}, 429, 3723.5),
        ({"x-ratelimit-reset-tokens": "later"}, 429, None),
        ({"x-ratelimit-reset-tokens": "6m0s"}, 503, None),
        ({}, 429, None),
    ],
)
def test_retry_after_reads_every_way_a_server_says_it(
    headers: dict[str, str], status: int, expected: float | None
) -> None:
    now = datetime(2026, 10, 20, 12, 0, 0, tzinfo=UTC)
    assert retry_after_seconds(headers, status=status, now=now) == expected


# Context overflow ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            _body(
                "This model's maximum context length is 32768 tokens. However, you requested 33000 tokens "
                "(30000 in the messages, 3000 in the completion). Please reduce the length of the messages."
            ),
            {"context_window": 32768, "input_tokens": 30000, "requested_output_tokens": 3000},
        ),
        (
            _body(
                "This model's maximum context length is 128000 tokens. However, your messages resulted in "
                "130000 tokens. Please reduce the length of the messages.",
                "invalid_request_error",
                "context_length_exceeded",
            ),
            {"context_window": 128000, "input_tokens": 130000},
        ),
        (
            _body(
                "This model's maximum context length is 32768 tokens. However, you requested 1000 output tokens "
                "and your prompt contains at least 32001 input tokens, for a total of at least 33001 tokens."
            ),
            {"context_window": 32768, "input_tokens": 32001, "requested_output_tokens": 1000},
        ),
        (
            _body("prompt is too long: 210000 tokens > 200000 maximum", "invalid_request_error"),
            {"context_window": 200000, "input_tokens": 210000},
        ),
        (
            json.dumps(
                {
                    "error": {
                        "code": 400,
                        "message": "the request exceeds the available context size, try increasing it",
                        "type": "exceed_context_size_error",
                        "n_prompt_tokens": 40000,
                        "n_ctx": 32768,
                    }
                }
            ),
            {"context_window": 32768, "input_tokens": 40000},
        ),
        (
            _body("request (40000 tokens) exceeds the available context size (32768 tokens), try increasing it"),
            {"context_window": 32768, "input_tokens": 40000},
        ),
        (_body("context length exceeded"), {}),
    ],
)
def test_overflow_sentences_of_each_server_parse(text: str, expected: dict[str, int]) -> None:
    assert context_overflow_sizes(text) == expected
    error = error_for_response("ep", 400, text)
    assert isinstance(error, LLMContextWindowExceeded)
    assert error.context_window == expected.get("context_window")
    assert error.input_tokens == expected.get("input_tokens")
    assert error.requested_output_tokens == expected.get("requested_output_tokens")


def test_a_rate_limit_that_mentions_tokens_is_not_an_overflow() -> None:
    text = _body("Rate limit reached: too many tokens per minute", "tokens", "rate_limit_exceeded")
    error = error_for_response("ep", 429, text, {"retry-after": "5"})
    assert isinstance(error, LLMRateLimitError)
    assert assert_adapter_failure(error).retry_after_seconds == 5.0


def test_an_overflow_classified_without_raising_reads_as_overflow() -> None:
    failure = classify_failure(400, _body("maximum context length is 10 tokens"))
    assert failure.reason is R.context_overflow
    assert failure.retryable is False


# Raised errors -------------------------------------------------------------


def test_raised_errors_carry_a_verdict_the_conformance_check_accepts() -> None:
    timeout = error_for_response("ep", 504, "gateway timeout")
    assert isinstance(timeout, LLMTimeoutError)
    assert assert_adapter_failure(timeout).reason is R.timeout

    refused = error_for_response("ep", 401, _body("Incorrect API key", "invalid_request_error", "invalid_api_key"))
    assert isinstance(refused, LLMProviderError)
    assert assert_adapter_failure(refused).retryable is False
    # The provider's own sentence, and the named code to search for.
    assert str(refused) == "ep: HTTP 401: Incorrect API key (invalid_api_key)"

    numeric = error_for_response("ep", 500, _body("boom", code=500))
    assert str(numeric) == "ep: HTTP 500: boom"


def test_error_bodies_are_read_in_every_shape() -> None:
    assert read_error_body("not json at all").message == "not json at all"
    assert read_error_body("[1, 2]").message == ""
    assert read_error_body(json.dumps({"error": "plain words"})).message == "plain words"
    assert read_error_body(json.dumps({"error": 7})).message == ""
    bare = read_error_body(json.dumps({"message": "bare", "type": "t", "code": "c"}))
    assert (bare.type, bare.code, bare.message) == ("t", "c", "bare")


def test_an_error_inside_a_stream_stands_for_its_code_or_a_server_fault() -> None:
    assert stream_error_status({"code": 429}) == 429
    assert stream_error_status({"code": "rate_limit"}) == 500
    assert stream_error_status({"code": True}) == 500
    assert stream_error_status("text") == 500
