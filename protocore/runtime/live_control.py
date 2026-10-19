"""Steer / follow-up queues, live model/thinking, and settled helper."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    IMAGE_REFS_METADATA_KEY,
    QUEUED_ITEMS_METADATA_KEY,
    ImageRefBlock,
    Message,
    MessageRole,
    TextBlock,
)

QueueKind = Literal["steer", "follow_up"]
QueueMode = Literal["one-at-a-time", "all"]


@dataclass(slots=True)
class QueuedPrompt:
    """One pending steer or follow-up item."""

    id: str
    kind: QueueKind
    text: str
    attachments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "attachments": list(self.attachments),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> QueuedPrompt:
        return cls(
            id=str(raw["id"]),
            kind=raw["kind"],
            text=str(raw.get("text") or ""),
            attachments=list(raw.get("attachments") or []),
        )


def new_queued_prompt(
    kind: QueueKind,
    text: str,
    *,
    attachments: list[dict[str, Any]] | None = None,
    item_id: str | None = None,
) -> QueuedPrompt:
    return QueuedPrompt(
        id=item_id or f"q_{uuid4().hex[:12]}",
        kind=kind,
        text=text,
        attachments=list(attachments or []),
    )


def enqueue(
    queue: list[QueuedPrompt],
    item: QueuedPrompt,
    rc: LoopConstants,
) -> list[QueuedPrompt]:
    """Append if under caps. Raises ValueError when the item or queue is over cap."""
    if not rc.steer_follow_up_enabled:
        raise ValueError("steer_follow_up_disabled")
    if len(item.text) > rc.max_queued_chars:
        raise ValueError("queued_item_too_long")
    if len(queue) >= rc.max_queued_items:
        raise ValueError("queue_full")
    return [*queue, item]


def place_items(
    queue: list[QueuedPrompt],
    mode: QueueMode,
) -> tuple[list[QueuedPrompt], list[QueuedPrompt]]:
    """Split ``queue`` into (placed, remaining) according to ``mode``."""
    if not queue:
        return [], []
    if mode == "all":
        return list(queue), []
    return [queue[0]], list(queue[1:])


def _image_ref(attachment: dict[str, Any]) -> ImageRefBlock | None:
    """The image an attachment names, or ``None`` when it is not one.

    An attachment is an image when it names a blob (``blob_ref``) and its
    ``mime_type`` is an image type, the two fields of :class:`ImageRefBlock`.
    A blob of any other type, a PDF or a log, is not sent to the model as an
    image; it stays in the placed message's queued-items record for the host.
    """
    blob_ref = attachment.get("blob_ref")
    mime_type = attachment.get("mime_type")
    if not isinstance(blob_ref, str) or not blob_ref:
        return None
    if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
        return None
    return ImageRefBlock(blob_ref=blob_ref, mime_type=mime_type)


def placed_as_user_message(items: SequenceLike) -> Message | None:
    """Join placed items into one user message, their files included.

    The texts are joined in queue order. Every item's image attachments ride on
    the message under :data:`IMAGE_REFS_METADATA_KEY`, in order, so a
    screenshot sent while the run was busy reaches the model with its caption;
    joining only the texts used to leave the files behind. Each item's id, kind
    and attachments as queued are recorded under
    :data:`QUEUED_ITEMS_METADATA_KEY`. An image with no words is still placed.
    Items with neither words nor an image give ``None``: the core has no way to
    show the model any other kind of file, so a host that wants one read names
    it in the item's text.
    """
    texts = [item.text.strip() for item in items if item.text.strip()]
    images = [
        image
        for item in items
        for attachment in item.attachments
        if isinstance(attachment, dict) and (image := _image_ref(attachment)) is not None
    ]
    if not texts and not images:
        return None
    metadata: dict[str, Any] = {
        QUEUED_ITEMS_METADATA_KEY: [
            {"id": item.id, "kind": item.kind, "attachments": list(item.attachments)}
            for item in items
        ],
    }
    if images:
        metadata[IMAGE_REFS_METADATA_KEY] = [image.model_dump(mode="json") for image in images]
    # No empty text block for a message of files alone: several providers
    # refuse an empty text part, and the adapter builds the turn from the
    # images in that case.
    blocks = [TextBlock(text="\n\n".join(texts))] if texts else []
    return Message(role=MessageRole.user, content_blocks=blocks, metadata=metadata)


def validate_thinking_for_mode(run_mode: str, thinking_enabled: bool) -> None:
    """Raise ValueError when deep would turn thinking off."""
    if run_mode not in ("direct", "deep"):
        raise ValueError("invalid_run_mode")
    if run_mode == "deep" and not thinking_enabled:
        raise ValueError("deep_requires_thinking")


def lease_key(scope_id: str, session_id: str, lane: str) -> str:
    return f"pc:lease:{scope_id}:{session_id}:{lane}"


def _restorable(raw: Any, kind: QueueKind) -> QueuedPrompt | None:
    """One queue entry as a :class:`QueuedPrompt`, or ``None`` when it carries nothing.

    A queue is a list of dicts that a host or a live store wrote, and a stop
    must hand back whatever was in it: an entry without an ``id`` or ``kind``
    is still restored, under the queue it sat in, rather than raising on the
    cancel path.
    """
    if isinstance(raw, QueuedPrompt):
        item = raw
    elif isinstance(raw, dict):
        item = QueuedPrompt(
            id=str(raw.get("id") or ""),
            kind=raw.get("kind") or kind,
            text=str(raw.get("text") or ""),
            attachments=list(raw.get("attachments") or []),
        )
    else:
        item = QueuedPrompt(id="", kind=kind, text=str(getattr(raw, "text", "") or ""))
    if not item.text.strip() and not item.attachments:
        return None
    return item


def restore_queued_items(engine: Any) -> list[QueuedPrompt]:
    """Drain steer + follow-up queues and return their items whole.

    Each item keeps its ``id``, ``kind`` and ``attachments`` beside its text,
    steer items first. A stopped run hands these back so a host can put them
    in the composer again; returning texts alone lost a queued screenshot,
    and the whole item when it had no caption.
    """
    items: list[QueuedPrompt] = []
    queues: tuple[tuple[QueueKind, Any], ...] = (
        ("steer", getattr(engine, "_steer_queue", [])),
        ("follow_up", getattr(engine, "_follow_up_queue", [])),
    )
    for kind, queue in queues:
        for raw in list(queue or []):
            item = _restorable(raw, kind)
            if item is not None:
                items.append(item)
    engine._steer_queue = []
    engine._follow_up_queue = []
    return items


def restore_queued_prompts(engine: Any) -> list[str]:
    """Drain steer + follow-up queues and return their texts.

    The text-only view of :func:`restore_queued_items`, kept for callers that
    put words back in a composer and nothing else; it drops attachments and
    items with no text. A caller that can take files back uses
    :func:`restore_queued_items`.
    """
    return [item.text.strip() for item in restore_queued_items(engine) if item.text.strip()]


# SequenceLike avoids importing Sequence in the public helper signature noise.
SequenceLike = list[QueuedPrompt]


__all__ = [
    "QueueKind",
    "QueueMode",
    "QueuedPrompt",
    "enqueue",
    "lease_key",
    "new_queued_prompt",
    "place_items",
    "placed_as_user_message",
    "restore_queued_items",
    "restore_queued_prompts",
    "validate_thinking_for_mode",
]
