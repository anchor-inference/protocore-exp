"""A reference adapter for servers that speak the OpenAI chat-completions API.

Install it with the ``openai`` extra (``pip install "protocore[openai]"``),
which adds ``httpx``. Nothing else in the core imports this package.

.. code-block:: python

    from protocore.providers.openai_compat import (
        EndpointSpec,
        OpenAICompatibleProvider,
        ProviderCapabilities,
        ThinkingDialect,
    )

    provider = OpenAICompatibleProvider(
        EndpointSpec(
            name="local",
            base_url="http://127.0.0.1:8000/v1",
            capabilities=ProviderCapabilities(thinking=ThinkingDialect.chat_template),
        )
    )

See ``docs/providers.md`` for the capabilities, the failure mapping and the
first-byte watchdog.
"""
from __future__ import annotations

try:
    import httpx  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "protocore.providers.openai_compat needs httpx; install the extra with "
        "'pip install \"protocore[openai]\"'"
    ) from exc

from protocore.providers.openai_compat.failures import (
    classify_failure,
    context_overflow_sizes,
    error_for_response,
    retry_after_seconds,
)
from protocore.providers.openai_compat.provider import (
    FailureObservation,
    FailureObserver,
    OpenAICompatibleProvider,
    UsageObservation,
    UsageObserver,
)
from protocore.providers.openai_compat.spec import (
    DEFAULT_EFFORT_NAMES,
    EndpointSpec,
    ImageLimits,
    ProviderCapabilities,
    ReasoningReplay,
    ThinkingDialect,
)
from protocore.providers.openai_compat.stream import StreamAssembler, normalize_usage
from protocore.providers.openai_compat.wire import (
    INTERRUPTED_TOOL_RESULT,
    REASONING_ENVELOPE,
    ImageLoader,
    messages_to_wire,
    tools_to_wire,
)

__all__ = [
    "DEFAULT_EFFORT_NAMES",
    "INTERRUPTED_TOOL_RESULT",
    "REASONING_ENVELOPE",
    "EndpointSpec",
    "FailureObservation",
    "FailureObserver",
    "ImageLimits",
    "ImageLoader",
    "OpenAICompatibleProvider",
    "ProviderCapabilities",
    "ReasoningReplay",
    "StreamAssembler",
    "ThinkingDialect",
    "UsageObservation",
    "UsageObserver",
    "classify_failure",
    "context_overflow_sizes",
    "error_for_response",
    "messages_to_wire",
    "normalize_usage",
    "retry_after_seconds",
    "tools_to_wire",
]
