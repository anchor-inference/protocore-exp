"""What the adapter needs to know about one endpoint, as typed values.

Servers that speak the chat-completions API agree on the core of it and
disagree at the edges: how thinking is switched on and off, whether a
reasoning trace is sent back on later turns, whether ``tool_choice`` may say
``"required"``, whether images are taken and how large, whether the prompt
cache is steered by markers. :class:`ProviderCapabilities` names those edges,
one field per decision the adapter makes, so an endpoint is described rather
than recognised by its URL.

The defaults are the plain OpenAI API and nothing more. An unknown capability
is treated as absent, never as present: sending a field a server does not know
is a refused request, while leaving out one it would have accepted costs at
most some efficiency.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from protocore.providers.text_tool_calls import TextToolCallGrammar


class ThinkingDialect(StrEnum):
    """How a request says "think" and "do not think".

    Each member is a public API convention, named after where it is
    documented. The adapter sends a thinking control only when the request
    carries one (``extra["enable_thinking"]`` with ``extra["reasoning_effort"]``),
    and an explicit "off" reaches the wire as off wherever the dialect can say
    it, rather than as the server's default.
    """

    #: Send no thinking control at all.
    none = "none"
    #: ``reasoning_effort`` (OpenAI). Off is
    #: :attr:`ProviderCapabilities.reasoning_off_effort` when set, else omitted,
    #: because the API has no universal "off" value.
    openai = "openai"
    #: ``chat_template_kwargs.enable_thinking`` (vLLM and SGLang serving a model
    #: whose chat template reads the flag).
    chat_template = "chat_template"
    #: ``chat_template_kwargs`` with ``enable_thinking`` and, when on,
    #: ``reasoning_effort``, plus the top-level ``reasoning_effort`` (llama.cpp
    #: server, which passes the kwargs to its Jinja template).
    llama_cpp = "llama_cpp"
    #: ``thinking: {"type": "enabled" | "disabled"}`` with ``reasoning_effort``
    #: when on (DeepSeek, and gateways that pass its fields through).
    deepseek = "deepseek"
    #: ``reasoning: {"effort": ...}`` or ``reasoning: {"enabled": false}``
    #: (OpenRouter).
    openrouter = "openrouter"


#: The effort names each dialect understands, keyed by the core's names. A
#: dialect missing here takes the core's names as they are.
#:
#: DeepSeek accepts ``low``, ``high`` and ``max`` and reads anything else as
#: ``high``, so ``medium`` (the model's default) and ``minimal`` (the step below
#: it) are sent as what they mean rather than as a word it would misread.
DEFAULT_EFFORT_NAMES: Mapping[ThinkingDialect, Mapping[str, str]] = {
    ThinkingDialect.deepseek: {
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    },
}


class ReasoningReplay(StrEnum):
    """Whether an assistant turn's reasoning text goes back to the server.

    Some servers read the reasoning of earlier turns (``reasoning_content`` on
    an assistant message) and some refuse a message that carries a field they
    do not know, so the adapter sends it only where it was asked to.
    """

    #: Never send reasoning back.
    never = "never"
    #: Send ``reasoning_content`` on the assistant turns that have some.
    when_present = "when_present"
    #: While thinking is on, every assistant turn carries ``reasoning_content``
    #: (empty when it has none) and a turn with nothing but reasoning is left
    #: out. DeepSeek refuses a thinking request whose earlier assistant turns
    #: lack the field, and a reasoning-only turn (the partial a stream left when
    #: it died mid-thought) is not a turn it can use. With thinking off this
    #: behaves as :attr:`when_present`.
    required = "required"


@dataclass(frozen=True, slots=True)
class ImageLimits:
    """What an endpoint accepts as image input. ``None`` in a field means no limit is known.

    An image the endpoint would refuse is not sent; the model is told in its
    place that an image was left out and why, so a request never fails on a
    picture and the model still knows one was attached.
    """

    #: How many images one request may carry. The most recent ones are kept,
    #: since the latest picture is the one a conversation is usually about.
    max_images: int | None = None
    #: The largest image, in bytes, as loaded (before base64).
    max_bytes: int | None = None
    #: The media types accepted, e.g. ``{"image/png", "image/jpeg"}``.
    mime_types: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """The edges of the chat-completions API an endpoint supports.

    Every field defaults to the plain OpenAI behaviour or to "absent".
    """

    #: How thinking is switched on and off.
    thinking: ThinkingDialect = ThinkingDialect.openai
    #: The endpoint's names for the core's effort levels. Empty takes
    #: :data:`DEFAULT_EFFORT_NAMES` for the dialect, then the core's names.
    effort_names: Mapping[str, str] = field(default_factory=dict)
    #: The ``reasoning_effort`` that means "off" under
    #: :attr:`ThinkingDialect.openai` (e.g. ``"none"`` or ``"minimal"`` on the
    #: models that accept one). ``None`` leaves thinking at the model default.
    reasoning_off_effort: str | None = None
    #: Whether reasoning text goes back on later assistant turns.
    reasoning_replay: ReasoningReplay = ReasoningReplay.never
    #: Keep the opaque ``reasoning_details`` a stream carries on the assistant
    #: message (:attr:`~protocore.contracts.types.Message.reasoning_items`) and
    #: send them back to the same endpoint and model. Off by default: a server
    #: that does not define the field may refuse a message carrying it.
    replay_reasoning_details: bool = False
    #: Whether ``tool_choice`` may name one function.
    forced_tool_choice: bool = True
    #: Whether ``tool_choice`` may be ``"required"``.
    tool_choice_required: bool = True
    #: Whether a tool schema's ``additionalProperties`` reaches the wire. A
    #: strict schema (the core's forced ``plan`` tool) depends on it; turn it
    #: off only for a server that refuses the keyword.
    schema_additional_properties: bool = True
    #: Repair the pairing of tool calls and tool results before sending: drop a
    #: result whose call is not in the history, and answer a call that has no
    #: result with an explicit "interrupted" error result. The API refuses a
    #: request with either, and a history cut by a cancel or a compaction can
    #: hold both.
    repair_tool_pairs: bool = True
    #: Image input; ``None`` means the endpoint takes none.
    images: ImageLimits | None = None
    #: Translate the core's cache breakpoints into ``cache_control`` markers on
    #: message content (Anthropic models behind OpenRouter and similar
    #: gateways). Servers that cache prefixes on their own need nothing.
    cache_control: bool = False
    #: Send ``cache_prompt: true`` (llama.cpp server), asking it to reuse the
    #: common prefix held in its slot.
    cache_prompt: bool = False
    #: Ask for the usage chunk at the end of a stream
    #: (``stream_options.include_usage``).
    stream_usage: bool = True
    #: The body field the output cap is sent in. Newer OpenAI reasoning models
    #: take ``max_completion_tokens`` and refuse ``max_tokens``.
    max_tokens_field: str = "max_tokens"
    #: Ask :meth:`complete_structured` for ``json_schema`` output. Off sends
    #: ``json_object``, which every JSON-mode server accepts.
    json_schema_response: bool = False
    #: Repair a tool call's arguments that were cut short (an object left open
    #: by the output cap) into the object they began, flagged as partial. Off
    #: hands the loop an empty object for them instead.
    repair_partial_arguments: bool = True
    #: The markups in which this endpoint's models sometimes write a tool call
    #: as reply text (:data:`~protocore.providers.text_tool_calls.DSML` for
    #: DeepSeek models). Text from a marker on is held back from the reply and
    #: a complete block becomes a structured call; empty, the default, passes
    #: text through untouched, since holding text costs a little latency on
    #: every reply that happens to contain a marker's first character.
    text_tool_calls: tuple[TextToolCallGrammar, ...] = ()

    def effort_name(self, effort: str) -> str:
        """The endpoint's word for the core's ``effort``."""
        if effort in self.effort_names:
            return self.effort_names[effort]
        return DEFAULT_EFFORT_NAMES.get(self.thinking, {}).get(effort, effort)


class TokenizeDialect(StrEnum):
    """Which counting route an endpoint exposes beside its chat API."""

    #: ``POST /tokenize`` with the chat ``messages`` and ``tools``; the answer
    #: carries ``count`` (vLLM).
    vllm = "vllm"
    #: ``POST /apply-template`` renders the messages through the chat template,
    #: then ``POST /tokenize`` counts the rendered prompt (llama.cpp server).
    llama_cpp = "llama_cpp"


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """One chat-completions endpoint: where it is, how to reach it, what it supports.

    ``base_url`` is the API root the paths hang from, usually ending in
    ``/v1``. ``api_key`` is sent as a bearer token and kept out of ``repr``.
    """

    #: The label errors, logs and observations name the endpoint by.
    name: str
    base_url: str
    api_key: str = field(default="", repr=False)
    #: Headers sent on every request, after the adapter's own.
    headers: Mapping[str, str] = field(default_factory=dict)
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    #: The whole-request timeout of the HTTP client.
    timeout_seconds: float = 600.0
    connect_timeout_seconds: float = 30.0
    #: How long a streamed request may get no line back at all (headers and
    #: the first line of the body together) before it is abandoned and sent
    #: again, byte for byte. ``0`` turns the watchdog off, which is right for a
    #: local server that may spend minutes reading a long prompt before its
    #: first byte.
    first_byte_timeout_seconds: float = 0.0
    #: How many times such a request is sent again before the silence is raised
    #: as a timeout.
    first_byte_retries: int = 1
    #: The body field that carries the request's prompt-cache routing key
    #: (``prompt_cache_key`` on the OpenAI API). ``None`` sends none.
    cache_key_field: str | None = None
    #: The header that carries the routing key, for servers that route by a
    #: header (xAI documents ``x-grok-conv-id``). ``None`` sends none.
    cache_key_header: str | None = None
    #: A sampling temperature every request to this endpoint uses, whatever the
    #: request says. ``None`` takes the request's own, or none.
    temperature: float | None = None
    #: Fields added to every request body. The adapter's own fields win, so
    #: this extends a request and cannot rewrite one.
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    #: The counting route, for an exact prompt size; ``None`` counts nothing.
    tokenize: TokenizeDialect | None = None
    #: Where the counting route lives. ``None`` takes ``base_url`` without a
    #: trailing ``/v1``, since both servers mount it at their root.
    tokenize_url: str | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"endpoint {self.name!r}: base_url must be an http(s) URL")
        if self.timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            raise ValueError(f"endpoint {self.name!r}: timeouts must be positive")
        if self.first_byte_timeout_seconds < 0:
            raise ValueError(f"endpoint {self.name!r}: first_byte_timeout_seconds cannot be negative")
        # Each resend waits a whole window again; more than a few turns one
        # silent endpoint into minutes of a run waiting for nothing.
        if not 0 <= self.first_byte_retries <= 3:
            raise ValueError(f"endpoint {self.name!r}: first_byte_retries must be between 0 and 3")

    def url(self, path: str) -> str:
        """``path`` under the API root."""
        return self.base_url.rstrip("/") + path

    def counting_root(self) -> str:
        """Where the counting route is mounted."""
        if self.tokenize_url:
            return self.tokenize_url.rstrip("/")
        base = self.base_url.rstrip("/")
        return base[: -len("/v1")] if base.endswith("/v1") else base


__all__ = [
    "DEFAULT_EFFORT_NAMES",
    "EndpointSpec",
    "ImageLimits",
    "ProviderCapabilities",
    "ReasoningReplay",
    "ThinkingDialect",
    "TokenizeDialect",
]
