"""Outside text a tool returns, handed to the model fenced as data.

A fetched page, a search result or an MCP server's reply can carry words that
read like instructions. A tool that says so — ``output_trust = "untrusted"`` on
the tool, or :data:`TOOL_OUTPUT_UNTRUSTED_METADATA_KEY` on one result — has the
text the model reads wrapped in a fence: an opening line naming the tool, a
sentence saying the text is data and not instructions, the text, and a closing
line. Nothing changes for a tool that says nothing.

The fence's id is the head of a SHA-256 digest of the text inside it. A text
cannot contain the closing line of its own fence without containing its own
digest, so it cannot end the fence early and speak outside it; a forged closing
line carries some other id, which the model is told not to accept. A random id
would do the same job and be worse: the same text would be fenced differently
on each run, and a result replayed from a snapshot would no longer match the
request the provider has cached.

The fence is applied where a result enters the run, after the per-result cut,
so the closing line is never what the cut removes. ``canonical_content`` keeps
the text as the tool returned it: the fence is how the model is shown the
value, not the value. The result event carries the transcript text, as it does
for a cut result, together with the mark, so a viewer that wants the tool's own
words reads them from the stored value.

The fence does not detect anything. It frames every result of a tool marked
untrusted, whatever the text says; a host that scans for injected instructions
does so in its own tool or in a ``post_tool_use`` hook, which runs before this.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Final

from protocore.contracts.tools import (
    TOOL_OUTPUT_UNTRUSTED_METADATA_KEY,
    ToolOutputTrust,
    is_untrusted_tool,
)
from protocore.runtime.tool_dispatch import DispatchOutcome

#: The prompt template the fence is rendered from, so a host can reword it or
#: say it in another language without touching the mechanism.
UNTRUSTED_TOOL_OUTPUT_TEMPLATE: Final[str] = "untrusted_tool_output"

#: Hex characters of the digest the fence carries. Forty-eight bits: a text
#: that guessed its own fence's id would have to find a fixed point of the
#: digest, and a collision between two results of one run is not a concern —
#: each fence only has to be distinct from what is inside it.
_FENCE_ID_HEX: Final[int] = 12


def fence_id(text: str) -> str:
    """The id of the fence around ``text``: the head of its SHA-256 digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_FENCE_ID_HEX]


def fence_untrusted_output(
    text: str,
    *,
    tool_name: str,
    render: Callable[..., str] | None = None,
) -> str:
    """``text`` inside the fence that marks it as data from ``tool_name``.

    ``render`` is a prompt renderer taking the template name and its context
    (``QueryEngine.prompt_text``); without one the bundled template is used.
    """
    if render is None:
        from protocore.prompts import bundled_prompt_provider

        provider = bundled_prompt_provider()

        def render(name: str, /, **context: Any) -> str:
            return provider.render(name, dict(context))

    return render(
        UNTRUSTED_TOOL_OUTPUT_TEMPLATE,
        tool_name=tool_name,
        fence_id=fence_id(text),
        content=text,
    )


def fence_overhead_text(
    *, tool_name: str, render: Callable[..., str] | None = None
) -> str:
    """The fence around an empty text, for sizing the room it needs."""
    return fence_untrusted_output("", tool_name=tool_name, render=render)


def fence_untrusted_result(
    outcome: DispatchOutcome,
    *,
    render: Callable[..., str] | None = None,
) -> DispatchOutcome:
    """``outcome`` with its text fenced, when it is marked untrusted.

    A result not marked, a call parked for approval, a question to the user and
    an empty result are returned as they are. The text the tool returned stays
    the canonical value unless a stored reference already stands for it.
    """
    if (
        outcome.approval_required
        or outcome.ask_user_required
        or not outcome.content
        or (outcome.metadata or {}).get(TOOL_OUTPUT_UNTRUSTED_METADATA_KEY) is not True
    ):
        return outcome
    fenced = fence_untrusted_output(
        outcome.content, tool_name=outcome.tool_call.name, render=render
    )
    canonical = outcome.canonical_content
    if canonical is None and outcome.canonical_ref is None:
        canonical = outcome.content
    return replace(outcome, content=fenced, canonical_content=canonical)


__all__ = [
    "TOOL_OUTPUT_UNTRUSTED_METADATA_KEY",
    "UNTRUSTED_TOOL_OUTPUT_TEMPLATE",
    "ToolOutputTrust",
    "fence_id",
    "fence_overhead_text",
    "fence_untrusted_output",
    "fence_untrusted_result",
    "is_untrusted_tool",
]
