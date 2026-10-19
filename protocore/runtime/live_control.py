"""Steer / follow-up queues, live model/thinking, and settled helper."""
from __future__ import annotations

from collections.abc import Sequence
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


def _queue_item_id(raw: Any) -> str:
    """The id of one queue entry, whether a dict a store wrote or a :class:`QueuedPrompt`."""
    if isinstance(raw, QueuedPrompt):
        return raw.id
    if isinstance(raw, dict):
        return str(raw.get("id") or "")
    return ""


@dataclass(slots=True)
class QueueSync:
    """What the engine's queues owe a host's durable copy of them.

    A host that keeps the queues in a store reads them into the engine before a
    model call (:meth:`QueryEngine.load_queued`) and writes back what the engine
    did not place (:meth:`QueryEngine.merge_queued`). Between the two, the store
    can gain an item the engine never saw, and the engine can lose one the host
    took back. These four records are what lets the write-back tell those apart
    from an item the model read.
    """

    #: The ids the last read handed the engine, per queue.
    seen_steer: list[str] = field(default_factory=list)
    seen_follow_up: list[str] = field(default_factory=list)
    #: Ids taken back since that read; a read that predates the withdrawal
    #: must not put them back, and the write-back must not call them consumed.
    withdrawn: list[str] = field(default_factory=list)
    #: Follow-ups turned into steers since that read, in the order turned.
    promoted: list[str] = field(default_factory=list)

    def seen(self) -> list[str]:
        return [*self.seen_steer, *self.seen_follow_up]


@dataclass(frozen=True, slots=True)
class QueueMerge:
    """What a host writes back to its store, and what it may say the model read.

    ``steer`` and ``follow_up`` are the engine's remainder followed by whatever
    the store gained after the engine's last read. ``consumed`` are the ids the
    engine was handed and no longer holds: placed into the history, and only
    those. An item taken back is not consumed, and neither is one that arrived
    after the read.
    """

    steer: list[dict[str, Any]]
    follow_up: list[dict[str, Any]]
    consumed: list[str]


def merge_queue(
    kept: list[dict[str, Any]],
    stored: list[dict[str, Any]],
    seen: Sequence[str],
) -> list[dict[str, Any]]:
    """What the engine hands back, followed by what the store gained meanwhile.

    The seen rule: an item in ``stored`` that the engine still holds, or was
    handed by its last read (``seen``), is the engine's to decide about, and it
    decided: kept, or placed. Any other stored item arrived after that read and
    is kept, behind the engine's remainder, so the order stays the order things
    were queued in. Writing ``kept`` over the store instead destroys a message
    that landed while the run held the queue, and the sender is told it was
    delivered.
    """
    known = {_queue_item_id(item) for item in kept} | set(seen)
    return [*kept, *(item for item in stored if _queue_item_id(item) not in known)]


def _queue_sync(engine: Any) -> QueueSync:
    sync = getattr(engine, "_queue_sync", None)
    if not isinstance(sync, QueueSync):
        sync = QueueSync()
        engine._queue_sync = sync
    return sync


def queued_items(engine: Any) -> list[QueuedPrompt]:
    """Every item the engine holds and has not placed, steers first, in queue order."""
    items: list[QueuedPrompt] = []
    for kind, queue in (("steer", engine._steer_queue), ("follow_up", engine._follow_up_queue)):
        for raw in list(queue or []):
            item = _restorable(raw, kind)  # type: ignore[arg-type]
            if item is not None:
                items.append(item)
    return items


def queue_input(
    engine: Any,
    kind: QueueKind,
    text: str,
    *,
    attachments: list[dict[str, Any]] | None = None,
    item_id: str | None = None,
) -> QueuedPrompt:
    """Add one item to the engine's own queue, under the configured caps."""
    item = new_queued_prompt(kind, text, attachments=attachments, item_id=item_id)
    queue = list((engine._steer_queue if kind == "steer" else engine._follow_up_queue) or [])
    # Only the length of the queue is checked, so a stand-in per entry is
    # enough; converting what a store wrote could raise on an entry with no id.
    enqueue([item] * len(queue), item, engine.config.rc)
    if kind == "steer":
        engine._steer_queue = [*queue, item.to_dict()]
    else:
        engine._follow_up_queue = [*queue, item.to_dict()]
    return item


def withdraw_queued(engine: Any, item_id: str) -> bool:
    """Take one waiting item out of the engine's queues; ``False`` when it holds none.

    ``False`` covers two cases a host answers differently. An item the engine
    was handed and no longer holds has been placed: the model is reading it,
    and nothing can take it back. An item the engine was never handed arrived
    in the store after the engine's last read, so the host removes it there;
    the engine still remembers the id until its next write-back, because a read
    already in flight returns the row as it stood and would bring it back.
    """
    sync = _queue_sync(engine)
    steer = list(engine._steer_queue or [])
    follow_up = list(engine._follow_up_queue or [])
    kept_steer = [raw for raw in steer if _queue_item_id(raw) != item_id]
    kept_follow_up = [raw for raw in follow_up if _queue_item_id(raw) != item_id]
    removed = len(kept_steer) != len(steer) or len(kept_follow_up) != len(follow_up)
    engine._steer_queue = kept_steer
    engine._follow_up_queue = kept_follow_up
    if removed or item_id not in sync.seen():
        if item_id not in sync.withdrawn:
            sync.withdrawn.append(item_id)
    return removed


def promote_queued(engine: Any, item_id: str) -> bool:
    """Turn a waiting follow-up into the last steer; ``False`` when the engine holds no such follow-up.

    A follow-up waits for the run to finish; a steer is read at the next tool
    round. The item moves in the engine first, before the host moves it in its
    store, because once it is out of the follow-up list the engine cannot start
    placing it as a follow-up and the store move cannot race a placement into a
    message the model reads twice.
    """
    queue = list(engine._follow_up_queue or [])
    index = next((i for i, raw in enumerate(queue) if _queue_item_id(raw) == item_id), None)
    if index is None:
        return False
    raw = queue.pop(index)
    moved = dict(raw) if isinstance(raw, dict) else _restorable(raw, "follow_up")
    if isinstance(moved, QueuedPrompt):
        moved = moved.to_dict()
    if not isinstance(moved, dict):
        return False
    moved["kind"] = "steer"
    engine._follow_up_queue = queue
    engine._steer_queue = [*list(engine._steer_queue or []), moved]
    sync = _queue_sync(engine)
    if item_id not in sync.promoted:
        sync.promoted.append(item_id)
    return True


def load_queued(
    engine: Any,
    steer: list[dict[str, Any]],
    follow_up: list[dict[str, Any]],
) -> None:
    """Replace the engine's queues with a store's, and remember what it was handed.

    Withdrawn items stay out, and a promoted follow-up stays a steer, even when
    the read returns a row written before the withdrawal or the promotion.
    """
    sync = _queue_sync(engine)
    withdrawn = set(sync.withdrawn)
    promoted = set(sync.promoted)
    next_steer = [
        dict(item) for item in steer
        if isinstance(item, dict) and _queue_item_id(item) not in withdrawn
    ]
    next_follow_up = [
        dict(item) for item in follow_up
        if isinstance(item, dict) and _queue_item_id(item) not in withdrawn | promoted
    ]
    # A promotion the store has not written yet keeps its place: the engine's
    # copy is the only one that says it is a steer now.
    steer_ids = {_queue_item_id(item) for item in next_steer}
    for raw in list(engine._steer_queue or []):
        raw_id = _queue_item_id(raw)
        if raw_id in promoted and raw_id not in steer_ids and isinstance(raw, dict):
            next_steer.append(dict(raw))
    sync.seen_steer = [_queue_item_id(item) for item in next_steer]
    sync.seen_follow_up = [_queue_item_id(item) for item in next_follow_up]
    engine._steer_queue = next_steer
    engine._follow_up_queue = next_follow_up


def merge_queued(
    engine: Any,
    stored_steer: list[dict[str, Any]],
    stored_follow_up: list[dict[str, Any]],
) -> QueueMerge:
    """Merge the engine's remainder into a store's current row by the seen rule.

    Call it with the row as the store holds it now, inside the store's own
    transaction, and write the result back. The engine then counts what it holds
    as handed, and forgets its withdrawals and promotions: the write-back is the
    record of both.
    """
    sync = _queue_sync(engine)
    kept_steer = [dict(raw) for raw in engine._steer_queue or [] if isinstance(raw, dict)]
    kept_follow_up = [dict(raw) for raw in engine._follow_up_queue or [] if isinstance(raw, dict)]
    known = [*sync.seen(), *sync.withdrawn, *sync.promoted]
    merged_steer = merge_queue(kept_steer, stored_steer, [*known, *map(_queue_item_id, kept_follow_up)])
    merged_follow_up = merge_queue(kept_follow_up, stored_follow_up, [*known, *map(_queue_item_id, kept_steer)])
    held = {_queue_item_id(item) for item in [*kept_steer, *kept_follow_up]}
    withdrawn = set(sync.withdrawn)
    consumed: list[str] = []
    for item_id in [*sync.seen(), *sync.promoted]:
        if item_id and item_id not in held and item_id not in withdrawn and item_id not in consumed:
            consumed.append(item_id)
    sync.seen_steer = [_queue_item_id(item) for item in kept_steer]
    sync.seen_follow_up = [_queue_item_id(item) for item in kept_follow_up]
    sync.withdrawn = []
    sync.promoted = []
    return QueueMerge(steer=merged_steer, follow_up=merged_follow_up, consumed=consumed)


def leftover_queued_ids(engine: Any) -> list[str]:
    """The ids of the items the engine still holds, steers first."""
    return [item.id for item in queued_items(engine) if item.id]


# SequenceLike avoids importing Sequence in the public helper signature noise.
SequenceLike = list[QueuedPrompt]


__all__ = [
    "QueueKind",
    "QueueMerge",
    "QueueMode",
    "QueueSync",
    "QueuedPrompt",
    "enqueue",
    "lease_key",
    "leftover_queued_ids",
    "load_queued",
    "merge_queue",
    "merge_queued",
    "new_queued_prompt",
    "place_items",
    "placed_as_user_message",
    "promote_queued",
    "queue_input",
    "queued_items",
    "restore_queued_items",
    "restore_queued_prompts",
    "validate_thinking_for_mode",
    "withdraw_queued",
]
