"""Rendering core requests into chat-completions JSON.

Pure functions over the core's types and :class:`ProviderCapabilities`: no
I/O except the image loader the caller passes, so every rule here is tested
against the JSON it produces.
"""
from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from protocore.contracts.types import (
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolResultImage,
    ToolUseBlock,
    message_image_refs,
    message_tool_images,
)
from protocore.logging_utils import get_logger
from protocore.providers.openai_compat.spec import (
    ProviderCapabilities,
    ReasoningReplay,
    ThinkingDialect,
)

_logger = get_logger(__name__)

ImageLoader = Callable[[str], Awaitable[tuple[bytes, str]]]
"""Loads an image by its blob reference: ``(bytes, media type)``."""

#: The ``format`` of the record kept on an assistant message for the
#: ``reasoning_details`` its stream carried.
REASONING_ENVELOPE = "openai_compat.reasoning_details"

#: What a tool call with no result is answered with when the pairing is
#: repaired. An error, never a made-up success: the call did not complete, and
#: the model has to know that to decide whether to make it again.
INTERRUPTED_TOOL_RESULT = "[no result: the call was interrupted before it returned]"

#: The line that opens the ``user`` message carrying tool images. The images
#: ride in a user turn only because most servers take nothing but a string in a
#: tool message; without this line the model would read a screenshot it asked
#: for as something the user sent it.
TOOL_IMAGES_CARRIER_HEADER = (
    "[Tool output, not a message from the user: the images below were returned "
    "by the tool calls above.]"
)


def tools_to_wire(
    tools: Sequence[ToolDefinition], capabilities: ProviderCapabilities
) -> list[dict[str, Any]]:
    """The ``tools`` array for ``tools``.

    The schema is the tool's own, including ``additionalProperties`` when the
    tool states it. The core spells that flag ``additional_properties`` on
    :class:`~protocore.contracts.types.ToolParameterSchema`, so a serialiser
    that copies only ``properties`` and ``required`` silently drops it, and the
    strict schema of a forced tool becomes a permissive one: the model may then
    add fields the loop never asked for. Nested schemas inside ``properties``
    are passed through as written.
    """
    wire: list[dict[str, Any]] = []
    for tool in tools:
        params = tool.parameters
        schema: dict[str, Any] = {
            "type": "object",
            "properties": dict(params.properties),
            "required": list(params.required),
        }
        if params.additional_properties is not None and capabilities.schema_additional_properties:
            schema["additionalProperties"] = params.additional_properties
        wire.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                },
            }
        )
    return wire


@dataclass(frozen=True, slots=True)
class WireMessages:
    """The ``messages`` array, and where each core message landed in it.

    ``index_map[i]`` is the position of the LAST wire entry core message ``i``
    produced, or ``-1`` when it produced none. A tool message fans out into one
    entry per result and a message can be left out, so a core index is not a
    wire index; prompt-cache markers placed by core index need the map.
    """

    entries: list[dict[str, Any]]
    index_map: list[int]


async def messages_to_wire(
    messages: Sequence[Message],
    *,
    capabilities: ProviderCapabilities,
    thinking: bool | None,
    endpoint_name: str,
    model: str,
    image_loader: ImageLoader | None = None,
) -> WireMessages:
    """Render ``messages`` as the chat-completions ``messages`` array.

    ``thinking`` is the request's thinking switch (``None`` when it carries
    none); it decides what :attr:`ReasoningReplay.required` sends.
    ``endpoint_name`` and ``model`` guard the replay of opaque reasoning: a
    record goes back only to the endpoint and model that wrote it.
    """
    allowed_images = _allowed_image_positions(messages, capabilities)
    items: list[tuple[int, dict[str, Any]]] = []
    image_position = 0
    #: Each tool result's images, by call id, with the request-wide position
    #: of the first: placed only after the pairing repair, which decides which
    #: results are sent at all.
    tool_images: dict[str, tuple[tuple[ToolResultImage, ...], int]] = {}
    for index, message in enumerate(messages):
        if message.role is MessageRole.system:
            items.append((index, {"role": "system", "content": message.text}))
        elif message.role is MessageRole.tool:
            for block in message.content_blocks:
                if isinstance(block, ToolResultBlock):
                    items.append(
                        (index, {"role": "tool", "tool_call_id": block.tool_call_id, "content": block.content})
                    )
                    if block.images:
                        tool_images[block.tool_call_id] = (block.images, image_position)
                        image_position += len(block.images)
        elif message.role is MessageRole.assistant:
            entry = _assistant_entry(
                message,
                capabilities=capabilities,
                thinking=thinking,
                endpoint_name=endpoint_name,
                model=model,
            )
            if entry is not None:
                items.append((index, entry))
        else:
            refs = message_image_refs(message)
            entry, image_position = await _user_entry(
                message,
                refs,
                first_position=image_position,
                allowed=allowed_images,
                capabilities=capabilities,
                image_loader=image_loader,
            )
            items.append((index, entry))
    if capabilities.repair_tool_pairs:
        items = _repair_tool_pairs(items)
    if tool_images:
        items = await _place_tool_images(
            items,
            tool_images,
            allowed=allowed_images,
            capabilities=capabilities,
            image_loader=image_loader,
        )
    index_map = [-1] * len(messages)
    for position, (index, _entry) in enumerate(items):
        if index >= 0:
            index_map[index] = position
    return WireMessages(entries=[entry for _index, entry in items], index_map=index_map)


def _assistant_entry(
    message: Message,
    *,
    capabilities: ProviderCapabilities,
    thinking: bool | None,
    endpoint_name: str,
    model: str,
) -> dict[str, Any] | None:
    text: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in message.content_blocks:
        if isinstance(block, ToolUseBlock):
            tool_calls.append(
                {
                    "id": block.tool_call_id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": block.arguments_json or "{}"},
                }
            )
        elif isinstance(block, ThinkingBlock):
            thoughts.append(block.text)
        elif isinstance(block, TextBlock):
            text.append(block.text)
    content = "".join(text)
    reasoning = message.reasoning_content or "".join(thoughts)
    entry: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        entry["tool_calls"] = tool_calls
    replay = capabilities.reasoning_replay
    if replay is ReasoningReplay.required and thinking:
        if not content and not tool_calls:
            return None
        entry["reasoning_content"] = reasoning
    elif replay is not ReasoningReplay.never and reasoning:
        entry["reasoning_content"] = reasoning
    if entry["content"] is None and not tool_calls:
        # An assistant message needs content or tool calls; an empty string is
        # the one value every server accepts for a turn that said nothing.
        entry["content"] = ""
    if capabilities.replay_reasoning_details:
        details = replayed_details(message, endpoint_name, model)
        if details:
            entry["reasoning_details"] = details
    return entry


def _allowed_image_positions(
    messages: Sequence[Message], capabilities: ProviderCapabilities
) -> range:
    """The positions, counted over the whole request, of the images that may be sent."""
    limits = capabilities.images
    total = sum(
        len(message_image_refs(message)) for message in messages if message.role is MessageRole.user
    ) + sum(
        len(message_tool_images(message)) for message in messages if message.role is MessageRole.tool
    )
    if limits is None or limits.max_images is None:
        return range(total)
    return range(max(0, total - limits.max_images), total)


async def _user_entry(
    message: Message,
    refs: Iterable[ImageRefBlock],
    *,
    first_position: int,
    allowed: range,
    capabilities: ProviderCapabilities,
    image_loader: ImageLoader | None,
) -> tuple[dict[str, Any], int]:
    texts = [block.text for block in message.content_blocks if isinstance(block, TextBlock)]
    parts: list[dict[str, Any]] = [{"type": "text", "text": text} for text in texts]
    position = first_position
    sent_image = False
    for ref in refs:
        part = await _image_part(
            ref,
            in_window=position in allowed,
            capabilities=capabilities,
            image_loader=image_loader,
        )
        position += 1
        sent_image = sent_image or part["type"] == "image_url"
        parts.append(part)
    if sent_image:
        return {"role": "user", "content": parts}, position
    return {"role": "user", "content": "\n".join(part["text"] for part in parts)}, position


async def _image_part(
    ref: ImageRefBlock | ToolResultImage,
    *,
    in_window: bool,
    capabilities: ProviderCapabilities,
    image_loader: ImageLoader | None,
) -> dict[str, Any]:
    """The content part for one image: the image itself, or a note saying why not."""
    limits = capabilities.images
    inline = ref.data if isinstance(ref, ToolResultImage) else None
    if limits is None or (image_loader is None and inline is None):
        return _image_note(ref, "this model does not take images")
    if not in_window:
        return _image_note(ref, "only the most recent images fit in one request")
    # An image without inline bytes always has a reference (the contract
    # holds exactly one), and the check above left a loader for it.
    if inline is None and image_loader is not None and ref.blob_ref is not None:
        try:
            data, mime = await image_loader(ref.blob_ref)
        except Exception as exc:
            # One unreadable attachment must not fail the whole request.
            _logger.warning("image %s could not be loaded: %s", ref.blob_ref, exc)
            return _image_note(ref, "it could not be loaded")
    else:
        data, mime = inline or b"", ref.mime_type
    mime = mime or ref.mime_type
    if limits.mime_types is not None and mime not in limits.mime_types:
        return _image_note(ref, f"{mime} is not a format this model takes")
    if limits.max_bytes is not None and len(data) > limits.max_bytes:
        return _image_note(ref, f"it is {len(data)} bytes, over this model's limit of {limits.max_bytes}")
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _image_note(ref: ImageRefBlock | ToolResultImage, reason: str) -> dict[str, Any]:
    return {"type": "text", "text": f"[image not sent, {reason}: {ref.blob_ref or 'inline image'}]"}


def _image_label(call_id: str, number: int, image: ToolResultImage) -> dict[str, Any] | None:
    """A line naming an image's size, for a model that reasons in its pixels.

    A computer-use model points at coordinates in the image it was shown; the
    size it was shown at is what turns them back into screen positions, and a
    model that is not told it guesses. Nothing is said when the tool gave none.
    """
    if image.width is None or image.height is None:
        return None
    return {
        "type": "text",
        "text": f"[image {number} of tool call {call_id}: {image.width}x{image.height} pixels]",
    }


async def _place_tool_images(
    items: list[tuple[int, dict[str, Any]]],
    tool_images: Mapping[str, tuple[tuple[ToolResultImage, ...], int]],
    *,
    allowed: range,
    capabilities: ProviderCapabilities,
    image_loader: ImageLoader | None,
) -> list[tuple[int, dict[str, Any]]]:
    """Put each sent tool result's images where this endpoint takes them.

    With :attr:`ProviderCapabilities.tool_result_images` the images are parts
    of the tool message itself, after its text — the Anthropic ``tool_result``
    shape. Otherwise they ride in one ``user`` message placed after the whole
    run of consecutive tool messages: a user turn between two results of one
    batch would close the batch, and the server would refuse the calls left
    without an answer. The carrier opens with
    :data:`TOOL_IMAGES_CARRIER_HEADER`. An image that cannot be sent becomes a
    line in its result's own text, so a text-only model gets no extra turn.
    """
    out: list[tuple[int, dict[str, Any]]] = []
    carrier: list[dict[str, Any]] = []
    carrier_index = -1

    def close_run() -> None:
        nonlocal carrier, carrier_index
        if carrier:
            out.append(
                (carrier_index, {"role": "user", "content": [{"type": "text", "text": TOOL_IMAGES_CARRIER_HEADER}, *carrier]})
            )
        carrier = []
        carrier_index = -1

    for index, entry in items:
        if entry["role"] != "tool":
            close_run()
            out.append((index, entry))
            continue
        out.append((index, entry))
        if index >= 0:
            carrier_index = index
        call_id = str(entry.get("tool_call_id"))
        found = tool_images.get(call_id) if index >= 0 else None
        if found is None:
            continue
        images, first_position = found
        sent: list[dict[str, Any]] = []
        notes: list[str] = []
        for offset, image in enumerate(images):
            part = await _image_part(
                image,
                in_window=first_position + offset in allowed,
                capabilities=capabilities,
                image_loader=image_loader,
            )
            if part["type"] != "image_url":
                notes.append(part["text"])
                continue
            label = _image_label(call_id, offset + 1, image)
            if label is not None:
                sent.append(label)
            sent.append(part)
        text = "\n".join([entry["content"], *notes]) if notes else entry["content"]
        if sent and capabilities.tool_result_images:
            entry["content"] = [{"type": "text", "text": text}, *sent]
            continue
        entry["content"] = text
        if sent:
            carrier.append({"type": "text", "text": f"[images returned by tool call {call_id}]"})
            carrier.extend(sent)
    close_run()
    return out


def _repair_tool_pairs(items: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
    """Make every tool result answer a call, and every call have a result.

    A result is kept only directly after the assistant turn that made its call,
    and once; anything else is dropped. A call still unanswered when the next
    non-tool entry arrives (or the history ends) gets
    :data:`INTERRUPTED_TOOL_RESULT`. Inserted entries belong to no core message.
    """
    out: list[tuple[int, dict[str, Any]]] = []
    pending: dict[str, None] = {}

    def answer_pending() -> None:
        for call_id in pending:
            out.append(
                (-1, {"role": "tool", "tool_call_id": call_id, "content": INTERRUPTED_TOOL_RESULT})
            )
        pending.clear()

    for index, entry in items:
        if entry["role"] == "tool":
            call_id = entry.get("tool_call_id")
            if call_id in pending:
                del pending[call_id]
                out.append((index, entry))
            else:
                _logger.debug("dropped a tool result with no open call: %s", call_id)
            continue
        answer_pending()
        for call in entry.get("tool_calls") or ():
            pending[call["id"]] = None
        out.append((index, entry))
    answer_pending()
    return out


def reasoning_envelope(endpoint_name: str, model: str, details: list[dict[str, Any]], tokens: int) -> dict[str, Any]:
    """The record kept on an assistant message for the ``reasoning_details`` its stream carried.

    ``tokens`` is the provider's count of the reasoning, which is what sending
    the entries back costs; the core's estimate reads it instead of the size of
    an encrypted blob.
    """
    return {
        "format": REASONING_ENVELOPE,
        "endpoint": endpoint_name,
        "model": model,
        "details": details,
        "tokens": max(0, int(tokens)),
    }


def replayed_details(message: Message, endpoint_name: str, model: str) -> list[dict[str, Any]]:
    """The ``reasoning_details`` to send back with ``message``, for this endpoint and model only.

    Encrypted reasoning is readable only by the model that wrote it. A session
    that switched model or endpoint keeps the records on its earlier turns, and
    they go back only if the session returns to the model that wrote them.
    """
    out: list[dict[str, Any]] = []
    for item in message.reasoning_items or ():
        if (
            item.get("format") == REASONING_ENVELOPE
            and item.get("endpoint") == endpoint_name
            and item.get("model") == model
        ):
            out.extend(detail for detail in item.get("details") or () if isinstance(detail, dict))
    return out


def apply_cache_control(entries: list[dict[str, Any]], breakpoints: Iterable[Any], index_map: Sequence[int]) -> None:
    """Put a ``cache_control`` marker where each of the core's breakpoints falls.

    A breakpoint names a core message index; the marker goes on the last
    content part of that message's last wire entry, turning string content into
    a one-part list. A tool result is skipped: its content must stay a string
    on the wire for most servers.
    """
    for breakpoint_ in breakpoints:
        index = getattr(breakpoint_, "message_index", None)
        if not isinstance(index, int) or not 0 <= index < len(index_map):
            continue
        position = index_map[index]
        if position < 0:
            continue
        entry = entries[position]
        if entry.get("role") == "tool":
            continue
        marker: dict[str, Any] = {"type": "ephemeral"}
        ttl = getattr(breakpoint_, "cache_control_type", "ephemeral")
        if ttl in ("5m", "1h"):
            marker["ttl"] = ttl
        content = entry.get("content")
        if isinstance(content, str) and content:
            entry["content"] = [{"type": "text", "text": content, "cache_control": marker}]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            content[-1]["cache_control"] = marker


def apply_thinking(
    body: dict[str, Any],
    capabilities: ProviderCapabilities,
    *,
    thinking: bool | None,
    effort: str | None,
) -> None:
    """Write the request's thinking switch into ``body`` in the endpoint's dialect.

    ``thinking=None`` writes nothing: the request carried no switch, so the
    server's default stands. An explicit ``False`` is written as "off" wherever
    the dialect can say it.
    """
    dialect = capabilities.thinking
    if thinking is None or dialect is ThinkingDialect.none:
        return
    mapped = capabilities.effort_name(effort) if effort else None
    if dialect is ThinkingDialect.openai:
        value = mapped if thinking else capabilities.reasoning_off_effort
        if value:
            body["reasoning_effort"] = value
    elif dialect is ThinkingDialect.chat_template:
        body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "enable_thinking": thinking}
    elif dialect is ThinkingDialect.llama_cpp:
        kwargs: dict[str, Any] = {**body.get("chat_template_kwargs", {}), "enable_thinking": thinking}
        if thinking and mapped:
            kwargs["reasoning_effort"] = mapped
            body["reasoning_effort"] = mapped
        body["chat_template_kwargs"] = kwargs
    elif dialect is ThinkingDialect.deepseek:
        body["thinking"] = {"type": "enabled" if thinking else "disabled"}
        if thinking and mapped:
            body["reasoning_effort"] = mapped
    else:
        body["reasoning"] = {"effort": mapped} if thinking and mapped else {"enabled": thinking}


def tool_choice_for(extra: Mapping[str, Any], capabilities: ProviderCapabilities) -> Any | None:
    """The ``tool_choice`` for the request's constraint, or ``None`` for none.

    A constraint the endpoint cannot express is left out, as the contract asks:
    the loop drops what a constrained turn returns beyond what it admits, so
    the request is only less efficient, never wrong.
    """
    forced = extra.get("forced_tool_choice")
    if forced:
        if capabilities.forced_tool_choice:
            return {"type": "function", "function": {"name": str(forced)}}
        return None
    if extra.get("tool_choice_required") and capabilities.tool_choice_required:
        return "required"
    return None


__all__ = [
    "INTERRUPTED_TOOL_RESULT",
    "REASONING_ENVELOPE",
    "TOOL_IMAGES_CARRIER_HEADER",
    "ImageLoader",
    "WireMessages",
    "apply_cache_control",
    "apply_thinking",
    "messages_to_wire",
    "reasoning_envelope",
    "replayed_details",
    "tool_choice_for",
    "tools_to_wire",
]
