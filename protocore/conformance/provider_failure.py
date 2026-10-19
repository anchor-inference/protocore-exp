"""A check an adapter's own tests run on the errors it raises.

The loop acts on a provider failure only through the verdict attached to it: an
error without one is never retried on a sibling provider, and a verdict whose
flag disagrees with the error's own flag is read by the error's flag first.
Both mistakes are silent at run time; the run simply ends, or retries what it
should not. So an adapter's test raises each failure it maps (a refused
connection, a 429 with ``Retry-After``, a 401, a 5xx) through the adapter and
hands the error to :func:`assert_adapter_failure`.
"""
from __future__ import annotations

from protocore.contracts.llm import LLMError, ProviderFailure


def assert_adapter_failure(exc: BaseException) -> ProviderFailure:
    """Return the verdict ``exc`` carries, or raise :class:`AssertionError` naming the fault.

    Passes when ``exc`` is an :class:`~protocore.contracts.llm.LLMError` with a
    :class:`~protocore.contracts.llm.ProviderFailure` attached as
    ``classified`` and the error's ``retryable`` flag agrees with the verdict's.
    The returned verdict is for the caller's own assertions on the reason and
    the wait.
    """
    if not isinstance(exc, LLMError):
        raise AssertionError(
            f"a provider failure must be raised as an LLMError subclass, got {type(exc).__name__}"
        )
    failure = exc.classified
    if not isinstance(failure, ProviderFailure):
        raise AssertionError(
            f"{type(exc).__name__} carries no ProviderFailure verdict "
            f"(classified={failure!r}); the loop will not move the provider chain for it"
        )
    if exc.retryable is not failure.retryable:
        raise AssertionError(
            f"{type(exc).__name__}.retryable is {exc.retryable} but its verdict says "
            f"{failure.retryable}; the loop reads the error's flag first"
        )
    return failure


__all__ = ["assert_adapter_failure"]
