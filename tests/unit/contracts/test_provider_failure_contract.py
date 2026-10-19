"""The verdict an adapter attaches to a provider failure, as a type.

The loop has always decided retry, wind-down and chain advance from a
``classified`` attribute an adapter pins on the error it raises, read by
attribute name. Without a published shape an adapter author had to read the
loop to learn the fields, and the one real consumer missed three of them: the
server's ``Retry-After`` never reached the backoff, the prompted-JSON fallback
flag was never set, and a refused connection carried no verdict at all, so the
provider chain never moved past a dead endpoint.
"""
from __future__ import annotations

import errno
import socket

import pytest
from pydantic import ValidationError

from protocore.conformance import assert_adapter_failure
from protocore.contracts.llm import (
    PERMANENT_FAILURE_REASONS,
    PROVIDER_ADVANCE_REASONS,
    SURFACE_INDEPENDENT_FAILURE_REASONS,
    LLMContextWindowExceeded,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMStreamIdleError,
    LLMTimeoutError,
    ProviderFailure,
    ProviderFailureReason,
    classify_transport_exception,
    provider_failure_of,
)

Reason = ProviderFailureReason


def test_the_reason_sets_are_drawn_from_the_enum() -> None:
    for reasons in (
        PROVIDER_ADVANCE_REASONS,
        PERMANENT_FAILURE_REASONS,
        SURFACE_INDEPENDENT_FAILURE_REASONS,
    ):
        assert reasons <= frozenset(Reason)
    # A refusal the wind-down's narrowed request meets again is, by
    # definition, a permanent one.
    assert SURFACE_INDEPENDENT_FAILURE_REASONS <= PERMANENT_FAILURE_REASONS


def test_a_dead_endpoint_moves_the_chain_and_is_worth_another_attempt() -> None:
    assert Reason.connection in PROVIDER_ADVANCE_REASONS
    assert Reason.connection not in PERMANENT_FAILURE_REASONS


def test_a_spend_cap_neither_moves_the_chain_nor_winds_down() -> None:
    """The cap is the host's, not the endpoint's: every rung and every narrowed
    request is refused by it the same way."""
    assert Reason.budget not in PROVIDER_ADVANCE_REASONS
    assert Reason.budget in PERMANENT_FAILURE_REASONS
    assert Reason.budget in SURFACE_INDEPENDENT_FAILURE_REASONS


def test_the_sets_still_answer_for_a_plain_string_reason() -> None:
    """Adapters that attach their own taxonomy compare by value."""
    assert "rate_limit" in PROVIDER_ADVANCE_REASONS
    assert "format_error" in PERMANENT_FAILURE_REASONS
    assert "spelled_differently" not in PROVIDER_ADVANCE_REASONS


def test_the_verdict_is_frozen_and_takes_the_reason_by_value() -> None:
    failure = ProviderFailure(reason="connection", retryable=True)  # type: ignore[arg-type]

    assert failure.reason is Reason.connection
    assert failure.should_fallback is False
    assert failure.retry_after_seconds is None
    assert failure.provider_message == ""
    with pytest.raises(ValidationError):
        failure.retryable = False  # type: ignore[misc]


@pytest.mark.parametrize("bad", [{"reason": "no_such_reason"}, {"retry_after_seconds": -1.0}])
def test_a_verdict_that_cannot_be_acted_on_is_refused(bad: dict[str, object]) -> None:
    fields: dict[str, object] = {"reason": "timeout", "retryable": True, **bad}
    with pytest.raises(ValidationError):
        ProviderFailure(**fields)  # type: ignore[arg-type]


def test_the_verdict_survives_its_json_form() -> None:
    failure = ProviderFailure(
        reason=Reason.rate_limit,
        retryable=True,
        retry_after_seconds=30.0,
        provider_message="slow down",
    )

    assert ProviderFailure.model_validate(failure.model_dump(mode="json")) == failure


def test_an_error_carries_the_verdict_it_was_raised_with() -> None:
    failure = ProviderFailure(reason=Reason.auth, retryable=False)

    error = LLMProviderError("HTTP 401", classified=failure)

    assert error.classified is failure
    # The flag on the error follows the verdict, because the policy reads the
    # error's flag first and a catch-all class default would overrule it.
    assert error.retryable is False


def test_an_explicit_flag_on_the_raise_still_wins() -> None:
    failure = ProviderFailure(reason=Reason.server_error, retryable=True)

    error = LLMProviderError("HTTP 503", retryable=False, classified=failure)

    assert error.retryable is False


def test_an_error_raised_the_old_way_is_unchanged() -> None:
    error = LLMProviderError("HTTP 503")

    assert error.classified is None
    assert error.retryable is True
    assert LLMError("x").retryable is False


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (ConnectionRefusedError(111, "Connection refused"), Reason.connection),
        (ConnectionResetError(104, "Connection reset by peer"), Reason.connection),
        (BrokenPipeError(32, "Broken pipe"), Reason.connection),
        (socket.gaierror(-2, "Name or service not known"), Reason.connection),
        (TimeoutError("timed out"), Reason.timeout),
        (OSError(errno.EHOSTUNREACH, "No route to host"), Reason.connection),
    ],
    ids=["refused", "reset", "broken-pipe", "dns", "timeout", "unreachable"],
)
def test_common_transport_failures_are_classified(exc: BaseException, reason: Reason) -> None:
    failure = classify_transport_exception(exc)

    assert failure is not None
    assert failure.reason is reason
    assert failure.retryable is True
    assert failure.provider_message


def test_a_transport_failure_is_found_under_the_clients_own_wrapper() -> None:
    """HTTP clients wrap the socket error in their own type; the cause chain
    still names it."""

    class ClientConnectError(Exception):
        pass

    try:
        try:
            raise ConnectionRefusedError(111, "Connection refused")
        except ConnectionRefusedError as inner:
            raise ClientConnectError("All connection attempts failed") from inner
    except ClientConnectError as outer:
        failure = classify_transport_exception(outer)

    assert failure is not None
    assert failure.reason is Reason.connection


def test_a_failure_that_is_not_the_transport_is_left_alone() -> None:
    assert classify_transport_exception(ValueError("bad json")) is None
    assert classify_transport_exception(PermissionError(13, "denied")) is None


def test_a_cyclic_cause_chain_does_not_hang_the_classifier() -> None:
    first = ValueError("a")
    second = ValueError("b")
    first.__context__ = second
    second.__context__ = first

    assert classify_transport_exception(first) is None


def test_the_typed_verdict_is_read_back_from_any_error() -> None:
    failure = ProviderFailure(reason=Reason.overloaded, retryable=True)

    assert provider_failure_of(LLMProviderError("529", classified=failure)) is failure
    # An unclassified catch-all says nothing, and nothing is invented for it.
    assert provider_failure_of(LLMProviderError("boom")) is None
    assert provider_failure_of(RuntimeError("crash")) is None


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (LLMRateLimitError("429"), Reason.rate_limit),
        (LLMTimeoutError("read timeout"), Reason.timeout),
        (LLMStreamIdleError("idle"), Reason.timeout),
        (LLMContextWindowExceeded("too long"), Reason.context_overflow),
    ],
)
def test_a_type_that_is_its_own_verdict_reads_back_as_one(
    error: LLMError, reason: Reason
) -> None:
    failure = provider_failure_of(error)

    assert failure is not None
    assert failure.reason is reason
    assert failure.retryable is error.retryable


def test_a_duck_typed_verdict_is_lifted_onto_the_type() -> None:
    class _Theirs:
        reason = "billing"
        retryable = False
        retry_after_seconds = "12"
        should_fallback = None

    error = LLMProviderError("HTTP 402")
    object.__setattr__(error, "classified", _Theirs())

    failure = provider_failure_of(error)

    assert failure == ProviderFailure(
        reason=Reason.billing, retryable=False, retry_after_seconds=12.0
    )


def test_a_duck_typed_reason_outside_the_enum_reads_as_unknown() -> None:
    class _Theirs:
        reason = "vendor_specific"

    error = LLMProviderError("HTTP 400")
    object.__setattr__(error, "classified", _Theirs())

    failure = provider_failure_of(error)

    assert failure is not None
    assert failure.reason is Reason.unknown
    assert failure.retryable is True


def test_the_conformance_helper_accepts_a_classified_error() -> None:
    error = LLMProviderError(
        "connect failed",
        classified=ProviderFailure(reason=Reason.connection, retryable=True),
    )

    assert assert_adapter_failure(error).reason is Reason.connection


@pytest.mark.parametrize(
    "error",
    [
        LLMProviderError("no verdict"),
        RuntimeError("not an LLM error"),
        LLMProviderError(
            "disagrees",
            retryable=False,
            classified=ProviderFailure(reason=Reason.server_error, retryable=True),
        ),
    ],
    ids=["unclassified", "wrong-type", "flag-disagrees"],
)
def test_the_conformance_helper_names_what_an_adapter_got_wrong(error: BaseException) -> None:
    with pytest.raises(AssertionError):
        assert_adapter_failure(error)
