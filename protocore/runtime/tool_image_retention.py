"""Keep only the newest tool images in the request.

A tool that returns an image — a computer-use tool's screenshot after every
click, a browser's rendered page — returns one per step, and each is priced at
a thousand tokens or more. Only the last few say what the screen looks like
now; the twentieth screenshot back is a picture of a state that no longer
exists, sent again on every call for the rest of the run.

So the copy of the history handed to the provider carries the newest
:attr:`~LoopConstants.tool_result_image_keep` tool images and no others. An
older image leaves the request; its result keeps its text and gains one line,
rendered from the ``tool_image_elided`` template, saying how many images were
taken out and why, so the model knows it was shown something and can ask for a
fresh one. ``engine.history`` is never touched: like stale-trimming and
eviction, this rewrites the view, and a resumed or compacted run still has
every reference.

*Batches, not drips.* Every image taken out changes the prompt at the point
where it stood, and a provider's prompt cache serves a request only up to the
first byte that differs. Taking the oldest image out on every step would move
that point back on every request. Instead images leave
:attr:`~LoopConstants.tool_result_image_evict_batch` at a time, once that many
are over the limit, so the view holds between ``keep`` and ``keep + batch - 1``
of them and the prefix moves once per batch — the idea stale-trimming batches
its cuts on.

*Stateless.* Which images leave follows from how many the view holds and
nothing else: the count of evicted images is always a whole number of batches,
the oldest ones. Building the view again from the same history evicts the same
images, so no decision has to be remembered on the engine or in the snapshot,
and a run resumed in another process makes the same cut.
"""
from __future__ import annotations

from collections.abc import Sequence

from protocore.contracts.prompts import IPromptTemplateProvider
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    TOOL_IMAGES_ELIDED_METADATA_KEY,
    ContentBlock,
    Message,
    ToolResultBlock,
)
from protocore.runtime.token_counting import estimate_tokens

#: The bundled template's words with the largest count a result can name,
#: for pricing the line without a template provider. The estimator that needs
#: it sizes histories far from any provider, and a host's own template is
#: priced as the bundled one: a line of a few dozen tokens beside images of
#: thousands.
_PLACEHOLDER_PROBE = (
    "[16 images this tool returned are no longer shown: only the newest {keep} "
    "tool images stay in view. Call the tool again for a current one]"
)


def evicted_image_count(total: int, rc: LoopConstants) -> int:
    """How many of a view's ``total`` tool images, the oldest, leave the request.

    A whole number of batches, and never one of the newest
    :attr:`~LoopConstants.tool_result_image_keep`. A keep of zero sends none.
    """
    keep = rc.tool_result_image_keep
    if keep is None or total <= keep:
        return 0
    if keep == 0:
        return total
    batch = max(rc.tool_result_image_evict_batch, 1)
    return (total - keep) // batch * batch


def _evictions(history: Sequence[Message], rc: LoopConstants) -> dict[tuple[int, int], int]:
    """For each result that loses images, how many of its oldest it loses.

    Keyed by (message index, block index). Images are counted in transcript
    order, and within one result in the order the tool returned them, so a
    result can lose its first images and keep its last.
    """
    positions = [
        (message_index, block_index, len(block.images))
        for message_index, message in enumerate(history)
        for block_index, block in enumerate(message.content_blocks)
        if isinstance(block, ToolResultBlock) and block.images
    ]
    remaining = evicted_image_count(sum(count for _m, _b, count in positions), rc)
    evictions: dict[tuple[int, int], int] = {}
    for message_index, block_index, count in positions:
        if remaining <= 0:
            break
        taken = min(count, remaining)
        evictions[(message_index, block_index)] = taken
        remaining -= taken
    return evictions


def evicted_image_tokens(history: Sequence[Message], rc: LoopConstants) -> int:
    """What the request saves over ``history`` by leaving its older tool images out.

    The per-message estimate prices every image a result carries, because it
    cannot see where in the history the message sits. A budget that sizes the
    whole history takes this off, so it measures the request that will really
    be sent rather than one carrying every screenshot of the run. Over a view
    this policy already built it is zero: what was left out is no longer there
    to count.
    """
    evictions = _evictions(history, rc)
    if not evictions:
        return 0
    images = sum(evictions.values())
    line = estimate_tokens(_PLACEHOLDER_PROBE.format(keep=rc.tool_result_image_keep), rc)
    return max(0, images * rc.token_count_image_tokens - len(evictions) * line)


def retain_recent_tool_images(
    history: Sequence[Message],
    rc: LoopConstants,
    prompts: IPromptTemplateProvider,
) -> list[Message]:
    """Return the view with all but the newest tool images taken out.

    A result that loses images keeps its text, gains the elision line, and
    says in :data:`TOOL_IMAGES_ELIDED_METADATA_KEY` how many it lost.
    """
    evictions = _evictions(history, rc)
    if not evictions:
        return list(history)
    rewritten: list[Message] = []
    for message_index, message in enumerate(history):
        new_blocks: list[ContentBlock] = []
        changed = False
        for block_index, block in enumerate(message.content_blocks):
            taken = evictions.get((message_index, block_index), 0)
            if not taken or not isinstance(block, ToolResultBlock):
                new_blocks.append(block)
                continue
            line = prompts.render(
                "tool_image_elided",
                {"count": taken, "keep": rc.tool_result_image_keep},
            )
            new_blocks.append(
                block.model_copy(
                    update={
                        "content": f"{block.content}\n{line}" if block.content else line,
                        "images": block.images[taken:],
                        "metadata": {**block.metadata, TOOL_IMAGES_ELIDED_METADATA_KEY: taken},
                    }
                )
            )
            changed = True
        rewritten.append(
            message.model_copy(update={"content_blocks": new_blocks}) if changed else message
        )
    return rewritten


__all__ = ["evicted_image_count", "evicted_image_tokens", "retain_recent_tool_images"]
