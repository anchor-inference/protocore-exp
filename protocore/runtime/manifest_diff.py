"""Which part of a request moved between two provider calls.

A provider's prompt cache serves a request up to the first byte it has not seen
before, so a cache miss has one cause that matters: the earliest part of the
request that changed. Two :class:`~protocore.contracts.observability.RequestManifest`
records hold enough to name it. The model, the tool catalogue and the system
prompt are compared by digest; the history is compared message by message,
which needs the message bodies — inline in the manifest when they were small
enough, or supplied by the host from the store its manifest sink wrote them to.

The parts are read in the order a cache reads a request: the model first (a
different model shares no cache at all), then the tool catalogue, then the
system prompt, then the history. Several providers render the catalogue ahead
of the system prompt; a host whose provider does not reads :attr:`ManifestDiff.changed`,
which names every part that differs whatever the order.

The volatile suffix a request declares (``extra["volatile_suffix_messages"]``,
the host's turn context) is left out of the history on both sides: it differs
on every request by design and is never part of the prefix.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from protocore.contracts.observability import ManifestValue, RequestManifest
from protocore.contracts.turn_context import VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY

#: The parts a manifest names, in the order a prompt cache reads a request,
#: then the parts that are not part of any prefix.
_PREFIX_ORDER: Final[tuple[str, ...]] = ("model", "tools", "system_prompt", "messages")
_OTHER_PARTS: Final[tuple[str, ...]] = (
    "extra",
    "max_tokens",
    "temperature",
    "constants",
    "provider_chain",
)

_PART_WORDS: Final[Mapping[str, str]] = {
    "model": "the model",
    "tools": "the tool catalogue",
    "system_prompt": "the system prompt",
    "messages": "the history",
}

MessageBodies = bytes | str | Sequence[Mapping[str, Any]]
"""Message bodies as a sink stored them (the canonical JSON bytes) or decoded."""


@dataclass(frozen=True, slots=True)
class ManifestDiff:
    """What changed from one request to the next.

    ``changed`` names every part whose recorded value differs, prefix parts
    first in cache order: ``model``, ``tools``, ``system_prompt``,
    ``messages``, then ``extra``, ``max_tokens``, ``temperature``,
    ``constants`` and ``provider_chain``. ``messages`` is there whenever the
    message list differs at all, a history that only grew included.

    ``prefix_break`` is the earliest part that cost the cache: ``"model"``,
    ``"tools"``, ``"system_prompt"``, ``"messages"`` (a message the previous
    request carried is not where it was), or ``None`` when the previous
    request is still a prefix of this one. ``first_changed_message`` is the
    index of that message, counted in the previous request's messages.
    ``history_known`` is ``False`` when the message bodies were neither inline
    nor supplied: a break in the history is then never claimed, since a
    history that grew and one that was rewritten have different digests alike.
    """

    changed: tuple[str, ...]
    prefix_break: str | None
    first_changed_message: int | None
    history_known: bool

    def describe(self) -> str:
        """One line for a log or a status view."""
        if self.prefix_break is None:
            head = "prefix kept"
            if not self.history_known and "messages" in self.changed:
                head = "prefix kept up to the history; whether the history only grew is unknown"
        elif self.prefix_break == "messages":
            head = f"prefix broken in the history at message {self.first_changed_message}"
        else:
            head = f"prefix broken by {_PART_WORDS[self.prefix_break]}"
        others = [part for part in self.changed if part != self.prefix_break]
        if others:
            return f"{head}; also changed: {', '.join(others)}"
        return head


def _decode(bodies: MessageBodies | None) -> list[Any] | None:
    if bodies is None:
        return None
    if isinstance(bodies, (bytes, str)):
        decoded = json.loads(bodies)
        return list(decoded) if isinstance(decoded, list) else None
    return list(bodies)


def _inline(value: ManifestValue) -> list[Any] | None:
    return _decode(value.inline) if value.inline is not None else None


def _volatile_suffix(manifest: RequestManifest) -> int:
    """How many trailing messages the request declared volatile; 0 when unknown."""
    if manifest.extra.inline is None:
        return 0
    extra = json.loads(manifest.extra.inline)
    count = extra.get(VOLATILE_SUFFIX_MESSAGES_EXTRA_KEY) if isinstance(extra, dict) else None
    return count if isinstance(count, int) and count > 0 else 0


def _first_difference(previous: list[Any], current: list[Any]) -> int | None:
    """The first index where ``previous`` stops being a prefix of ``current``."""
    for index, message in enumerate(previous):
        if index >= len(current) or current[index] != message:
            return index
    return None


def diff_request_manifests(
    previous: RequestManifest,
    current: RequestManifest,
    *,
    previous_messages: MessageBodies | None = None,
    current_messages: MessageBodies | None = None,
) -> ManifestDiff:
    """Compare two manifests of consecutive requests.

    ``previous_messages`` and ``current_messages`` are the message bodies when
    the manifests carry them by digest only: the ``bodies["messages"]`` a sink
    was handed, or the same decoded. Inline bodies are used when present.
    """
    changed: list[str] = []
    if previous.model != current.model:
        changed.append("model")
    for slot, before_value, after_value in (
        ("tools", previous.tools, current.tools),
        ("system_prompt", previous.system_prompt, current.system_prompt),
        ("messages", previous.messages, current.messages),
    ):
        if before_value.sha256 != after_value.sha256:
            changed.append(slot)
    if previous.extra.sha256 != current.extra.sha256:
        changed.append("extra")
    if previous.max_tokens != current.max_tokens:
        changed.append("max_tokens")
    if previous.temperature != current.temperature:
        changed.append("temperature")
    if previous.constants_sha256 != current.constants_sha256:
        changed.append("constants")
    if (previous.provider_chain_position, previous.provider_chain_model) != (
        current.provider_chain_position,
        current.provider_chain_model,
    ):
        changed.append("provider_chain")
    ordered = tuple(part for part in (*_PREFIX_ORDER, *_OTHER_PARTS) if part in changed)

    before = _decode(previous_messages) if previous_messages is not None else _inline(previous.messages)
    after = _decode(current_messages) if current_messages is not None else _inline(current.messages)
    history_known = before is not None and after is not None

    first_changed: int | None = None
    if "messages" in ordered and before is not None and after is not None:
        stable_before = before[: len(before) - _volatile_suffix(previous)]
        stable_after = after[: len(after) - _volatile_suffix(current)]
        first_changed = _first_difference(stable_before, stable_after)

    prefix_break: str | None = next(
        (part for part in ("model", "tools", "system_prompt") if part in ordered), None
    )
    if prefix_break is None and first_changed is not None:
        prefix_break = "messages"
    return ManifestDiff(
        changed=ordered,
        prefix_break=prefix_break,
        first_changed_message=first_changed,
        history_known=history_known or "messages" not in ordered,
    )


__all__ = ["ManifestDiff", "MessageBodies", "diff_request_manifests"]
