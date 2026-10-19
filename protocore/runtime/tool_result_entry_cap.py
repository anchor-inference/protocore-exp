"""Cutting a fresh tool result down to the per-result limit where it enters.

``tool_result_truncation_ratio`` of the window is the most one tool result may
take. Compaction has always enforced it, but only on results outside the
protected tail, and the result a tool has just returned is always inside it: the
newest round is exactly what a pass keeps. So a single oversized output reached
the model whole, and one larger than the window ended the run — the reactive
pass freed nothing, since it may not touch the newest round either.

Here the limit is applied at the moment the result is recorded. What the model
is shown is the head of the output, as much as fits the limit, and a line saying
how much was left out and where the whole value is stored. With
:attr:`~LoopConstants.tool_result_entry_cap_tail_share` above zero the kept
part is split between the head and the end of the output, and the line marks
the omitted middle: the reason a build or a test run failed is usually its last
line, which a head-only cut drops. The whole value is
written to the blob store first, so the reference the line names is real before
anything is cut, and it becomes the result's canonical reference: compaction,
eviction and persistence then treat it as the address of the value rather than
storing it a second time.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import replace

from protocore.contracts.blob import IBlobStore
from protocore.contracts.runtime_constants import LoopConstants
from protocore.runtime.context.budgets import derive_budgets
from protocore.runtime.token_counting import estimate_tokens
from protocore.runtime.tool_dispatch import DispatchOutcome

logger = logging.getLogger(__name__)

#: Metadata key set on a result that was cut on entry, so a host can tell a
#: shortened result from one the tool returned short.
TOOL_RESULT_CUT_ON_ENTRY_METADATA_KEY = "protocore.tool_result_cut_on_entry"


def _pointer(*, original_tokens: int, limit: int, kept_chars: int, ref: str) -> str:
    return (
        f"\n[Output truncated where it entered the run: {original_tokens} tokens is over "
        f"the per-result limit of {limit}, so only the first {kept_chars} characters are "
        f"shown. The whole result is stored as {ref}.]"
    )


def _middle_marker(
    *, original_tokens: int, limit: int, omitted_chars: int, ref: str
) -> str:
    """The line standing where the middle of a head-and-tail cut was."""
    return (
        f"\n\n[Output cut where it entered the run: {original_tokens} tokens is over "
        f"the per-result limit of {limit}, so {omitted_chars} characters are omitted "
        f"here, between the head and the end. The whole result is stored as {ref}.]\n\n"
    )


def _head_within(content: str, budget_tokens: int, rc: LoopConstants) -> str:
    """The longest prefix of ``content`` whose estimate fits ``budget_tokens``."""
    if budget_tokens <= 0:
        return ""
    low, high = 0, len(content)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(content[:mid], rc) <= budget_tokens:
            low = mid
        else:
            high = mid - 1
    return content[:low]


def _tail_within(content: str, budget_tokens: int, rc: LoopConstants) -> str:
    """The longest suffix of ``content`` whose estimate fits ``budget_tokens``."""
    if budget_tokens <= 0:
        return ""
    low, high = 0, len(content)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(content[len(content) - mid :], rc) <= budget_tokens:
            low = mid
        else:
            high = mid - 1
    return content[len(content) - low :] if low else ""


async def cap_fresh_result(
    outcome: DispatchOutcome,
    *,
    rc: LoopConstants,
    blob_store: IBlobStore,
    tenant_id: str,
) -> DispatchOutcome:
    """``outcome`` with its text cut to the per-result limit, or unchanged.

    A result within the limit, a call parked for approval or a question to the
    user are returned as they are. Otherwise the whole value is stored (unless
    the tool already names where it lives) and the text becomes its head plus a
    pointer, together no larger than the limit.
    """
    if outcome.approval_required or outcome.ask_user_required or not outcome.content:
        return outcome
    limit = derive_budgets(rc).tool_result_truncation_threshold
    original_tokens = estimate_tokens(outcome.content, rc)
    if original_tokens <= limit:
        return outcome
    if outcome.canonical_ref is not None:
        ref = outcome.canonical_ref
    else:
        value = outcome.canonical_content or outcome.content
        stored = await blob_store.put(
            tenant_id=tenant_id,
            content=value.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
            metadata={
                "tool_call_id": outcome.tool_call.id,
                "label": "tool_result",
                "stage": "entry",
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            },
        )
        ref = stored.ref
    tail_share = rc.tool_result_entry_cap_tail_share
    tail = ""
    if tail_share > 0:
        # Measured with the longest number it can carry, like the pointer
        # below, so head, marker and tail together never land above the limit.
        marker_budget = estimate_tokens(
            _middle_marker(
                original_tokens=original_tokens,
                limit=limit,
                omitted_chars=len(outcome.content),
                ref=ref,
            ),
            rc,
        )
        kept_budget = limit - marker_budget
        while True:
            tail_budget = int(kept_budget * tail_share)
            head = _head_within(outcome.content, kept_budget - tail_budget, rc)
            tail = _tail_within(outcome.content[len(head) :], tail_budget, rc)
            text = (
                head
                + _middle_marker(
                    original_tokens=original_tokens,
                    limit=limit,
                    omitted_chars=len(outcome.content) - len(head) - len(tail),
                    ref=ref,
                )
                + tail
            )
            # The estimate of three joined pieces can round a token above the
            # sum of their estimates; give the overshoot back and cut again.
            overshoot = estimate_tokens(text, rc) - limit
            if overshoot <= 0 or kept_budget <= 0:
                break
            kept_budget -= overshoot
    else:
        # The pointer is measured with the longest numbers it can carry, so the
        # head plus the pointer never lands above the limit.
        pointer_budget = estimate_tokens(
            _pointer(original_tokens=original_tokens, limit=limit, kept_chars=len(outcome.content), ref=ref),
            rc,
        )
        head = _head_within(outcome.content, limit - pointer_budget, rc)
        text = head + _pointer(
            original_tokens=original_tokens, limit=limit, kept_chars=len(head), ref=ref
        )
    logger.warning(
        "DIAG tool_result.cut_on_entry tool=%s call=%s tokens=%d limit=%d kept_chars=%d "
        "kept_tail_chars=%d ref=%s",
        outcome.tool_call.name,
        outcome.tool_call.id,
        original_tokens,
        limit,
        len(head) + len(tail),
        len(tail),
        ref,
    )
    return replace(
        outcome,
        content=text,
        canonical_content=None,
        canonical_ref=ref,
        metadata={
            **(outcome.metadata or {}),
            TOOL_RESULT_CUT_ON_ENTRY_METADATA_KEY: {
                "original_tokens": original_tokens,
                "limit": limit,
                "kept_chars": len(head) + len(tail),
                "kept_tail_chars": len(tail),
            },
        },
    )


__all__ = ["TOOL_RESULT_CUT_ON_ENTRY_METADATA_KEY", "cap_fresh_result"]
