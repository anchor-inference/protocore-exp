"""The reference :class:`~protocore.contracts.llm.ILLMProvider` for chat-completions servers.

One class serves every endpoint that speaks the OpenAI chat-completions API: a
hosted vendor, a gateway, a self-hosted vLLM, SGLang or llama.cpp server. What
differs between them is data, an :class:`~protocore.providers.openai_compat.spec.EndpointSpec`
with its :class:`~protocore.providers.openai_compat.spec.ProviderCapabilities`,
never a subclass per vendor.

A host extends it at three points, each a method meant to be overridden:
:meth:`OpenAICompatibleProvider.build_body` (fields a deployment adds),
:meth:`OpenAICompatibleProvider.request_headers` (headers it adds), and the two
observers it passes in, which hear every usage report and every refusal (for a
ledger, for learning a model's real context window from what the server said).
A spend budget hangs on ``admission`` (see :mod:`protocore.providers.spend`).
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Self

import httpx

from protocore.contracts.llm import (
    ILLMProvider,
    LLMError,
    LLMObservabilityContext,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMResponseUsage,
    LLMTimeoutError,
    ProviderDelta,
    ProviderDeltaKind,
    ProviderFailure,
    ProviderFailureReason,
    VisionSupport,
    classify_transport_exception,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, StopReason, TextBlock
from protocore.json_utils import structured_json_candidates
from protocore.logging_utils import get_logger
from protocore.providers.openai_compat.failures import (
    classify_failure,
    error_for_response,
    read_error_body,
    stream_error_status,
)
from protocore.providers.openai_compat.spec import EndpointSpec
from protocore.providers.openai_compat.stream import (
    StreamAssembler,
    map_finish,
    normalize_usage,
    sse_events,
)
from protocore.providers.openai_compat.tokenize import TokenizeCounter
from protocore.providers.openai_compat.wire import (
    ImageLoader,
    apply_cache_control,
    apply_thinking,
    messages_to_wire,
    reasoning_envelope,
    tool_choice_for,
    tools_to_wire,
)
from protocore.providers.spend import (
    ISpendAdmission,
    PricingLookup,
    SpendEnd,
    SpendEstimate,
    SpendRefused,
    SpendSettlement,
    budget_error,
)
from protocore.runtime.token_counting import estimate_tokens

_logger = get_logger(__name__)

CallPurpose = Literal["stream", "text", "structured"]


@dataclass(frozen=True, slots=True)
class UsageObservation:
    """One call's usage, as the server reported it and as the core reads it."""

    endpoint: str
    model: str
    purpose: CallPurpose
    raw: Mapping[str, Any]
    normalized: Mapping[str, Any]
    duration_seconds: float
    observability: LLMObservabilityContext | None


@dataclass(frozen=True, slots=True)
class FailureObservation:
    """One refused request: what the server said, and the verdict read from it.

    ``body`` is the response body as text, cut to a few kilobytes; ``headers``
    keeps only the ones that say when to retry, so an observer that stores it
    cannot store a credential a proxy echoed back.
    """

    endpoint: str
    model: str
    status: int
    failure: ProviderFailure | None
    error: LLMError
    body: str
    headers: Mapping[str, str]
    observability: LLMObservabilityContext | None


UsageObserver = Callable[[UsageObservation], Awaitable[None]]
FailureObserver = Callable[[FailureObservation], Awaitable[None]]

_KEPT_HEADERS = frozenset(
    {"retry-after", "retry-after-ms", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"}
)
_OBSERVED_BODY_CHARS = 4_000

_DEFAULT_CONSTANTS: LoopConstants | None = None


def _estimate(text: str) -> int:
    """The core's own token estimate, under its default constants."""
    global _DEFAULT_CONSTANTS
    if _DEFAULT_CONSTANTS is None:
        _DEFAULT_CONSTANTS = LoopConstants()
    return estimate_tokens(text, _DEFAULT_CONSTANTS)


class OpenAICompatibleProvider(ILLMProvider):
    """Streams ``POST {base_url}/chat/completions`` into core :class:`ProviderDelta` envelopes.

    ``client`` is the HTTP client to send through; one is created (and closed
    by :meth:`aclose`) when none is given. Tests pass a client over
    ``httpx.MockTransport``. ``image_loader`` resolves an image's blob
    reference to its bytes; without it no image is sent, whatever the
    capabilities say. ``admission`` is asked before every call and told how
    it ended; ``pricing`` prices a model's usage where the server reports no
    cost of its own (both in :mod:`protocore.providers.spend`).
    """

    def __init__(
        self,
        endpoint: EndpointSpec,
        *,
        client: httpx.AsyncClient | None = None,
        image_loader: ImageLoader | None = None,
        on_usage: UsageObserver | None = None,
        on_failure: FailureObserver | None = None,
        admission: ISpendAdmission | None = None,
        pricing: PricingLookup | None = None,
    ) -> None:
        self.endpoint = endpoint
        self._admission = admission
        self._pricing = pricing
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(endpoint.timeout_seconds, connect=endpoint.connect_timeout_seconds)
        )
        self._image_loader = image_loader
        self._on_usage = on_usage
        self._on_failure = on_failure
        self._counter = (
            TokenizeCounter(
                self._client,
                root=endpoint.counting_root(),
                dialect=endpoint.tokenize,
                headers=self.request_headers(None),
            )
            if endpoint.tokenize is not None
            else None
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client, when this provider created it."""
        if self._owns_client:
            await self._client.aclose()

    # ILLMProvider ---------------------------------------------------------

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[ProviderDelta]:
        """Yield deltas in the order the core reads them: content, tool stops, usage, finish.

        The core stops reading at ``finish``, and the usage arrives in a
        trailing chunk, so the stream is read to its end before ``usage`` and
        ``finish`` are yielded.
        """
        body = await self.build_body(request, stream=True)
        reservation = await self._reserve(request, body, "stream")
        ended = _SpendState()
        try:
            # aclosing: a caller that stops reading closes this generator, and
            # the inner one must close with it, now, so its response is closed
            # rather than left to the garbage collector.
            async with aclosing(self._stream(request, body, ended, reservation)) as stream:
                async for delta in stream:
                    yield delta
        finally:
            await self._release_unsettled(reservation, ended)

    async def _stream(
        self, request: LLMRequest, body: dict[str, Any], ended: _SpendState, reservation: str | None
    ) -> AsyncGenerator[ProviderDelta, None]:
        started = time.monotonic()
        caps = self.endpoint.capabilities
        assembler = StreamAssembler(
            repair_partial_arguments=caps.repair_partial_arguments,
            keep_reasoning_details=caps.replay_reasoning_details,
            text_tool_calls=caps.text_tool_calls,
        )
        try:
            silent = 0
            while True:
                opened = await self._open_stream(request, body, silent)
                if opened is not None:
                    break
                silent += 1
                if silent > self.endpoint.first_byte_retries:
                    raise LLMTimeoutError(
                        f"{self.endpoint.name}: no response byte within "
                        f"{self.endpoint.first_byte_timeout_seconds:g}s on {silent} identical attempt(s)",
                        classified=ProviderFailure(
                            reason=ProviderFailureReason.timeout,
                            retryable=True,
                            provider_message="no response byte in time",
                        ),
                    )
                # Said before the next attempt waits its own window: the core's
                # idle watchdog counts from the last delta, and two silent
                # windows back to back would pass its baseline and be cut as a
                # stalled stream half-way through the resend.
                yield ProviderDelta(kind=ProviderDeltaKind.progress)
            response, lines = opened
            ended_cleanly = False
            try:
                if response.status_code >= 400:
                    text = (await response.aread()).decode("utf-8", "replace")
                    ended.outcome = SpendEnd.refused
                    raise await self._refused(request, body, response.status_code, text, response.headers)
                async for data in sse_events(lines):
                    if data is None:
                        yield ProviderDelta(kind=ProviderDeltaKind.progress)
                        continue
                    if data.strip() == "[DONE]":
                        ended_cleanly = True
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        _logger.debug("%s: skipped an unreadable stream event", self.endpoint.name)
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if chunk.get("error") and not chunk.get("choices"):
                        # An error after the 200: the status line said
                        # nothing, so the chunk is the refusal.
                        error = chunk["error"]
                        raise await self._refused(
                            request, body, stream_error_status(error), json.dumps({"error": error}), {}
                        )
                    for delta in assembler.feed(chunk):
                        yield delta
            finally:
                await response.aclose()
            if not ended_cleanly and assembler.finish_reason is None:
                # The body ended with neither ``[DONE]`` nor a finish reason: the
                # connection dropped mid-answer. Read as a stop it ended the turn
                # on half an answer; read as calls it ran a call the model may
                # not have finished. Neither is said: the round is retried.
                raise LLMProviderError(
                    f"{self.endpoint.name}: the stream ended before the response did",
                    classified=ProviderFailure(
                        reason=ProviderFailureReason.connection,
                        retryable=True,
                        provider_message="stream closed before the response was complete",
                    ),
                )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        for delta in assembler.close():
            yield delta
        normalized = self._priced(request, normalize_usage(assembler.usage or {}))
        await self._observe_usage(request, "stream", assembler.usage or {}, normalized, started)
        # Settled before the usage is yielded: the loop may stop reading at
        # ``finish``, and a settlement placed after the last yield would then
        # never run, leaving the reservation to be released as interrupted.
        await self._settle(reservation, ended, assembler.usage or {}, normalized)
        yield ProviderDelta(kind=ProviderDeltaKind.usage, usage=normalized)
        kept = None
        if assembler.reasoning_details:
            kept = (
                reasoning_envelope(
                    self.endpoint.name, request.model, assembler.reasoning_details, normalized["reasoning_tokens"]
                ),
            )
        yield ProviderDelta(
            kind=ProviderDeltaKind.finish,
            finish_reason=map_finish(assembler.resolved_finish()),
            reasoning_items=kept,
        )

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        body = await self.build_body(request, stream=False)
        return await self._complete(request, body, "text")

    async def complete_structured(self, request: LLMRequest, response_schema: dict[str, Any]) -> LLMResponse:
        """A JSON completion: the object is validated here and returned as its JSON text.

        Asks for ``json_schema`` output where the endpoint takes it and
        ``json_object`` otherwise. A reply that holds no JSON object is refused
        as a ``format_error`` marked ``should_fallback``: the same question as
        prompted JSON, or on a plain completion, may still be answered.
        """
        body = await self.build_body(request, stream=False)
        if self.endpoint.capabilities.json_schema_response:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema},
            }
        else:
            body["response_format"] = {"type": "json_object"}
        response = await self._complete(request, body, "structured")
        candidates = structured_json_candidates(response.message.text)
        if not candidates:
            truncated = response.stop_reason is StopReason.max_tokens
            raise LLMProviderError(
                f"{self.endpoint.name}: the structured reply holds no JSON object"
                + (" (cut by the output cap)" if truncated else ""),
                classified=ProviderFailure(
                    reason=ProviderFailureReason.format_error,
                    retryable=False,
                    should_fallback=True,
                    provider_message="the reply holds no JSON object",
                ),
            )
        text = json.dumps(candidates[0], ensure_ascii=False)
        return response.model_copy(
            update={"message": Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=text)])}
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """The core's estimate; the loop re-anchors on the size the server reports."""
        return max(1, _estimate(text)) if text else 0

    def vision_support(self, model: str | None = None) -> VisionSupport:
        """What this endpoint does with images, read from its capabilities.

        An endpoint takes images only when its capabilities say so and an
        image loader was given, since without one no image held by reference
        is ever sent. Tool images always reach a model that takes images: in
        the tool message where :attr:`ProviderCapabilities.tool_result_images`
        allows it, in the carrier turn after it otherwise. The answer is the
        endpoint's whatever ``model`` names: one endpoint is one model's
        capabilities here.
        """
        caps = self.endpoint.capabilities
        if caps.images is None or self._image_loader is None:
            return VisionSupport()
        return VisionSupport(
            images=True,
            tool_result_images=True,
            native_tool_result_images=caps.tool_result_images,
            max_images=caps.images.max_images,
        )

    async def count_request_tokens(self, request: LLMRequest) -> int | None:
        """The exact prompt size from the endpoint's counting route, or ``None``.

        ``None`` when the endpoint has no counting route configured, or the
        route turned out not to exist; the loop then uses its own estimate.
        The body is built exactly as the call would build it, so the count
        covers the chat template, the tool definitions and the thinking switch.
        """
        if self._counter is None:
            return None
        body = await self.build_body(request, stream=False)
        return await self._counter.count(body)

    # Extension points -----------------------------------------------------

    async def build_body(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        """The JSON body for ``request``. Override to add fields; call this first.

        The body is built once per call and reused as it is for a first-byte
        resend, so a server's prompt cache sees the identical bytes.
        """
        caps = self.endpoint.capabilities
        extra = dict(request.extra or {})
        thinking_flag = extra.get("enable_thinking")
        thinking = bool(thinking_flag) if thinking_flag is not None else None
        effort = extra.get("reasoning_effort")
        wire = await messages_to_wire(
            request.messages,
            capabilities=caps,
            thinking=thinking,
            endpoint_name=self.endpoint.name,
            model=request.model,
            image_loader=self._image_loader,
        )
        body: dict[str, Any] = dict(self.endpoint.extra_body)
        body.update(
            {
                "model": request.model,
                "messages": wire.entries,
                caps.max_tokens_field: request.max_tokens,
                "stream": stream,
            }
        )
        # The core leaves the temperature unset unless a caller states one, so
        # the server's own generation config applies; a null on the wire is
        # not "unset" to every server, so the key is left out instead.
        temperature = self.endpoint.temperature if self.endpoint.temperature is not None else request.temperature
        if temperature is not None:
            body["temperature"] = temperature
        if stream and caps.stream_usage:
            body["stream_options"] = {"include_usage": True}
        if request.tools:
            body["tools"] = tools_to_wire(request.tools, caps)
            choice = tool_choice_for(extra, caps)
            if choice is not None:
                body["tool_choice"] = choice
        response_format = extra.get("response_format")
        if isinstance(response_format, dict):
            body["response_format"] = dict(response_format)
        apply_thinking(body, caps, thinking=thinking, effort=str(effort) if effort else None)
        breakpoints = extra.get("cache_breakpoints")
        if caps.cache_control and breakpoints:
            apply_cache_control(body["messages"], breakpoints, wire.index_map)
        routing_key = request.observability.routing_key() if request.observability is not None else None
        if self.endpoint.cache_key_field and routing_key:
            body[self.endpoint.cache_key_field] = routing_key
        if caps.cache_prompt:
            body["cache_prompt"] = True
        return body

    def request_headers(self, request: LLMRequest | None) -> dict[str, str]:
        """The headers for ``request`` (``None`` for a call outside one). Override to add some."""
        headers = {"content-type": "application/json"}
        if self.endpoint.api_key:
            headers["authorization"] = f"Bearer {self.endpoint.api_key}"
        headers.update(self.endpoint.headers)
        observability = request.observability if request is not None else None
        routing_key = observability.routing_key() if observability is not None else None
        # No fallback to a fixed value: one key shared by every caller would
        # pin them all to the same machine.
        if self.endpoint.cache_key_header and routing_key:
            headers[self.endpoint.cache_key_header] = routing_key
        return headers

    # Internals ------------------------------------------------------------

    async def _open_stream(
        self, request: LLMRequest, body: dict[str, Any], attempt: int
    ) -> tuple[httpx.Response, AsyncIterator[str]] | None:
        """Send the request and wait for its first line back; ``None`` when nothing came in time.

        One deadline covers the response headers and the first line of the body
        together, because a server that answers its headers at once and then
        says nothing is as stuck as one that never answers. Any line counts, an
        SSE comment or an empty one included, so a model that thinks silently
        behind a proxy that sends keepalives is never taken for a dead one. A
        line, not a byte, is waited for because servers flush whole lines.

        Abandoning such a request and sending the same one again is safe
        because nothing came back: no partial output reached the run, so none
        can be repeated. A refusal (status 400 and up) is an answer, so its
        body is read with no deadline and it is never resent.
        """
        timeout = self.endpoint.first_byte_timeout_seconds
        sent = self._client.build_request(
            "POST", self.endpoint.url("/chat/completions"), json=body, headers=self.request_headers(request)
        )
        began = time.monotonic()
        response: httpx.Response | None = None
        answered_at: float | None = None
        try:
            async with asyncio.timeout(timeout if timeout > 0 else None) as deadline:
                response = await self._client.send(sent, stream=True)
                answered_at = time.monotonic()
                lines = response.aiter_lines()
                if response.status_code >= 400:
                    return response, lines
                first = await anext(lines, None)
        except TimeoutError:
            if response is not None:
                await response.aclose()
            if not deadline.expired():
                raise
            heard = "no headers" if answered_at is None else f"headers after {answered_at - began:.1f}s"
            _logger.warning(
                "%s: no byte of the %s response within %gs (%s); sending the identical request again (%d of %d)",
                self.endpoint.name,
                request.model,
                timeout,
                heard,
                attempt + 1,
                self.endpoint.first_byte_retries,
            )
            return None
        except BaseException:
            if response is not None:
                await response.aclose()
            raise
        if attempt:
            _logger.warning(
                "%s: the resent %s request answered after %.1fs",
                self.endpoint.name,
                request.model,
                time.monotonic() - began,
            )
        return response, _prepend(first, lines)

    async def _complete(self, request: LLMRequest, body: dict[str, Any], purpose: CallPurpose) -> LLMResponse:
        reservation = await self._reserve(request, body, purpose)
        ended = _SpendState()
        try:
            return await self._complete_admitted(request, body, purpose, ended, reservation)
        finally:
            await self._release_unsettled(reservation, ended)

    async def _complete_admitted(
        self,
        request: LLMRequest,
        body: dict[str, Any],
        purpose: CallPurpose,
        ended: _SpendState,
        reservation: str | None,
    ) -> LLMResponse:
        started = time.monotonic()
        try:
            response = await self._client.post(
                self.endpoint.url("/chat/completions"), json=body, headers=self.request_headers(request)
            )
        except httpx.HTTPError as exc:
            raise self._transport_error(exc) from exc
        if response.status_code >= 400:
            ended.outcome = SpendEnd.refused
            raise await self._refused(request, body, response.status_code, response.text, response.headers)
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMProviderError(
                f"{self.endpoint.name}: the reply is not JSON",
                classified=ProviderFailure(
                    reason=ProviderFailureReason.server_error, retryable=True, provider_message=response.text[:300]
                ),
            ) from exc
        if not isinstance(data, dict):
            data = {}
        if data.get("error") and not data.get("choices"):
            error = data["error"]
            raise await self._refused(request, body, stream_error_status(error), json.dumps({"error": error}), {})
        choice = (data.get("choices") or [{}])[0]
        message = (choice.get("message") or {}) if isinstance(choice, dict) else {}
        usage = data.get("usage")
        raw_usage: dict[str, Any] = usage if isinstance(usage, dict) else {}
        normalized = self._priced(request, normalize_usage(raw_usage))
        await self._observe_usage(request, purpose, raw_usage, normalized, started)
        await self._settle(reservation, ended, raw_usage, normalized)
        finish = choice.get("finish_reason") if isinstance(choice, dict) else None
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[TextBlock(text=_message_text(message))]),
            stop_reason=StopReason.max_tokens if finish == "length" else StopReason.end_turn,
            usage=LLMResponseUsage(
                input_tokens=normalized["input_tokens"],
                output_tokens=normalized["output_tokens"],
                cache_read_tokens=normalized["cache_read_tokens"],
                cache_read_input_tokens=normalized["cache_read_input_tokens"],
                cache_creation_input_tokens=normalized["cache_creation_input_tokens"],
                cached_tokens=normalized["cache_read_tokens"],
                response_cost_usd=normalized.get("cost_usd"),
            ),
        )

    async def _refused(
        self,
        request: LLMRequest,
        body: dict[str, Any],
        status: int,
        text: str,
        headers: Mapping[str, str],
    ) -> LLMError:
        """The error for a refusal, after the failure observer has heard it."""
        constrained = bool(
            body.get("tool_choice") not in (None, "auto", "none") or body.get("response_format")
        )
        error = error_for_response(self.endpoint.name, status, text, headers, constrained=constrained)
        if status == 400:
            # The message shape, not its content, so a refused request can be
            # read from the log without the conversation in it.
            _logger.warning(
                "%s refused a request (%s); message shape: %s",
                self.endpoint.name,
                read_error_body(text).message[:200],
                _message_shape(body.get("messages") or []),
            )
        if self._on_failure is not None:
            observation = FailureObservation(
                endpoint=self.endpoint.name,
                model=request.model,
                status=status,
                failure=error.classified
                if error.classified is not None
                else classify_failure(status, text, headers, constrained=constrained),
                error=error,
                body=text[:_OBSERVED_BODY_CHARS],
                headers={
                    str(key).lower(): str(value)
                    for key, value in headers.items()
                    if str(key).lower() in _KEPT_HEADERS
                },
                observability=request.observability,
            )
            try:
                await self._on_failure(observation)
            except Exception:
                # Observing is a side effect; the refusal stays the caller's error.
                _logger.exception("%s: the failure observer raised", self.endpoint.name)
        return error

    def _transport_error(self, exc: httpx.HTTPError) -> LLMError:
        """The error for a request that never got an answer, verdict attached.

        A timeout is a timeout; a connection that could not be made, was
        reset, or broke mid-stream is ``connection``, so a dead endpoint moves
        the run to the next provider instead of ending it.
        """
        message = f"{self.endpoint.name}: {type(exc).__name__}: {exc}"
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError(
                message,
                classified=ProviderFailure(
                    reason=ProviderFailureReason.timeout, retryable=True, provider_message=str(exc)
                ),
            )
        failure = classify_transport_exception(exc)
        if failure is None:
            reason = (
                ProviderFailureReason.connection
                if isinstance(exc, httpx.TransportError)
                else ProviderFailureReason.unknown
            )
            failure = ProviderFailure(reason=reason, retryable=True, provider_message=str(exc) or type(exc).__name__)
        return LLMProviderError(message, classified=failure)

    async def estimate_spend(self, request: LLMRequest, body: dict[str, Any], purpose: CallPurpose) -> SpendEstimate:
        """What the call is about to cost, for the admission. Override to count exactly.

        The prompt size is the core's estimate of the messages and tools as
        they go on the wire; asking a counting route here would add a round
        trip to every call for a number that only has to err high.
        """
        prompt_tokens = _estimate(
            json.dumps([body.get("messages") or [], body.get("tools") or []], ensure_ascii=False)
        )
        pricing = self._pricing(request.model) if self._pricing is not None else None
        return SpendEstimate(
            endpoint=self.endpoint.name,
            model=request.model,
            purpose=purpose,
            prompt_tokens=prompt_tokens,
            max_output_tokens=request.max_tokens,
            cost_usd=pricing.estimate(prompt_tokens=prompt_tokens, max_output_tokens=request.max_tokens)
            if pricing is not None
            else None,
            observability=request.observability,
        )

    async def _reserve(self, request: LLMRequest, body: dict[str, Any], purpose: CallPurpose) -> str | None:
        if self._admission is None:
            return None
        estimate = await self.estimate_spend(request, body, purpose)
        try:
            return await self._admission.reserve(estimate)
        except SpendRefused as refused:
            raise budget_error(self.endpoint.name, refused) from refused

    def _priced(self, request: LLMRequest, normalized: dict[str, Any]) -> dict[str, Any]:
        """The usage with ``cost_usd`` from the price table, when the server reported none."""
        if "cost_usd" in normalized or self._pricing is None or not _reported_counts(normalized):
            return normalized
        pricing = self._pricing(request.model)
        if pricing is not None:
            normalized["cost_usd"] = pricing.cost(normalized)
        return normalized

    async def _settle(
        self,
        reservation: str | None,
        ended: _SpendState,
        raw: Mapping[str, Any],
        normalized: Mapping[str, Any],
    ) -> None:
        if reservation is None or self._admission is None:
            return
        ended.settled = True
        settlement = SpendSettlement(
            usage=dict(normalized),
            raw=dict(raw),
            cost_usd=normalized.get("cost_usd"),
            complete=_reported_counts(normalized),
        )
        try:
            await self._admission.settle(reservation, settlement)
        except Exception:
            # The answer arrived and was paid for; a ledger that failed to book
            # it must not take the answer away from the run.
            _logger.exception("%s: the spend admission failed to settle %s", self.endpoint.name, reservation)

    async def _release_unsettled(self, reservation: str | None, ended: _SpendState) -> None:
        if reservation is None or self._admission is None or ended.settled:
            return
        detail = (
            "the server refused the request"
            if ended.outcome is SpendEnd.refused
            else "the request ended without a usage report"
        )
        try:
            await self._admission.release(reservation, outcome=ended.outcome, detail=detail)
        except Exception:
            # Runs while the call's own error or cancellation is in flight;
            # raising here would replace it.
            _logger.exception("%s: the spend admission failed to release %s", self.endpoint.name, reservation)

    async def _observe_usage(
        self,
        request: LLMRequest,
        purpose: CallPurpose,
        raw: Mapping[str, Any],
        normalized: Mapping[str, Any],
        started: float,
    ) -> None:
        if self._on_usage is None:
            return
        observation = UsageObservation(
            endpoint=self.endpoint.name,
            model=request.model,
            purpose=purpose,
            raw=dict(raw),
            normalized=dict(normalized),
            duration_seconds=time.monotonic() - started,
            observability=request.observability,
        )
        try:
            await self._on_usage(observation)
        except Exception:
            # A ledger that failed to record must not fail the answer it recorded.
            _logger.exception("%s: the usage observer raised", self.endpoint.name)


@dataclass(slots=True)
class _SpendState:
    """How one admitted call is ending: settled, or which way it failed."""

    settled: bool = False
    outcome: SpendEnd = SpendEnd.interrupted


def _reported_counts(normalized: Mapping[str, Any]) -> bool:
    """Whether the server reported any token count; all zeros means it reported none."""
    return bool(normalized.get("input_tokens") or normalized.get("output_tokens"))


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def _message_shape(messages: list[dict[str, Any]]) -> str:
    """One token per wire message: its role's initial, and for an assistant turn
    ``t`` (text), ``c`` (tool calls), ``r`` (reasoning) and ``e`` (opaque reasoning)."""
    parts = []
    for message in messages:
        role = str(message.get("role") or "?")
        token = role[0]
        if role == "assistant":
            token += "t" if message.get("content") else ""
            token += "c" if message.get("tool_calls") else ""
            token += "r" if message.get("reasoning_content") else ""
            token += "e" if message.get("reasoning_details") else ""
        parts.append(token)
    return " ".join(parts)


async def _prepend(first: str | None, rest: AsyncIterator[str]) -> AsyncIterator[str]:
    """``first`` (the line the watchdog already read; ``None`` for a body that ended empty), then ``rest``."""
    if first is None:
        return
    yield first
    async for line in rest:
        yield line


__all__ = [
    "CallPurpose",
    "FailureObservation",
    "FailureObserver",
    "OpenAICompatibleProvider",
    "UsageObservation",
    "UsageObserver",
]
