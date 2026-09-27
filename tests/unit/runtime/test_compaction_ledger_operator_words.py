"""The operator's words in the compaction ledger (``docs/compaction.md``, clauses 4 and 10).

The ledger once clipped every quote of the operator at 1,200 characters without
a word, and never quoted a reply the operator gave by any route other than a
typed turn — the answer to a question, a decision a host relays inside one of
its own notes. After compaction the model then worked from half an instruction,
or from a summary's paraphrase of an answer. These tests pin the repair: a
quote is whole; the other sections give up their room first; a quote that
still cannot fit is cut with a marker that says how much is missing and where
the whole text is; and a host can say which words are the operator's.
"""
from __future__ import annotations

import json
import re

import pytest

from protocore.contracts.llm import LLMRequest, LLMResponse
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import (
    COMPACTION_SUMMARY_METADATA_KEY,
    OPERATOR_WORDS_METADATA_KEY,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.context.compaction import (
    CompactionState,
    _is_compaction_artefact,
    _is_plain_operator_turn,
    place_ledger,
    run_floor,
    run_tier3_fold,
)
from protocore.runtime.context.ledger import Ledger, OperatorQuote, is_ledger, ledger_from_history
from protocore.runtime.token_counting import estimate_tokens
from protocore.tests_support.adapters import InMemoryBlobStore, InMemoryLLMProvider


class _DroppingSummariser(InMemoryLLMProvider):
    """A summariser that keeps nothing of what it was shown."""

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=Message(
                role=MessageRole.assistant,
                content_blocks=[TextBlock(text="## Progress\nchecked more sections; nothing notable")],
            ),
            stop_reason=StopReason.end_turn,
        )


def _cyrillic_instruction(chars: int = 3_000) -> str:
    """A multi-line Russian instruction of about ``chars`` characters, with values in it."""
    lines = [
        "Никогда не перезапускай основной сервер без моего подтверждения.",
        "Порт администратора — 47031, конфигурация лежит в /srv/app/config/main.toml.",
        "Если сборка упадёт, сначала пришли мне журнал, потом предлагай исправление.",
    ]
    text: list[str] = []
    index = 0
    while sum(len(line) + 1 for line in text) < chars:
        text.append(f"{index + 1}. {lines[index % len(lines)]} Пункт {index + 1} обязателен.")
        index += 1
    return "\n".join(text)


def _summary(i: int) -> Message:
    return Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text=f"<compacted-turn id='k{i}'>\n## Progress\nstep {i} " + "done " * 80 + "\n</compacted-turn>")],
        metadata={COMPACTION_SUMMARY_METADATA_KEY: True},
    )


def _absorb(ledger: Ledger, messages: list[Message], *, source: str = "") -> None:
    ledger.absorb(messages, is_operator=_is_plain_operator_turn, skip=_is_compaction_artefact, source=source)


# ---------------------------------------------------------------------------
# A quote is whole
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_long_multi_line_cyrillic_instruction_survives_compaction_verbatim() -> None:
    instruction = _cyrillic_instruction()
    assert len(instruction) >= 3_000 and instruction.count("\n") > 20
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the task")]),
        _summary(0), _summary(1),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text=instruction)]),
        _summary(2), _summary(3),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
        Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="working")]),
    ]
    rc = LoopConstants(
        model_context_window=256_000,
        compaction_keep_recent_turns=2,
        compaction_fold_min_tokens=0,
        compaction_fold_keep_operator_turns=1,
    )
    ledger = Ledger()
    result = await run_tier3_fold(history, _DroppingSummariser(), CompactionState(), rc, model_name="m", ledger=ledger)
    assert result.spans_folded == 1
    assert all(message.text != instruction for message in history)

    place_ledger(history, ledger, rc)
    carried = next(message for message in history if is_ledger(message))
    assert instruction in carried.text
    assert "more characters of the operator's words" not in carried.text
    # And it is carried by the state, so the next pass rebuilds it whole.
    assert [quote.text for quote in ledger_from_history(history).operator] == [instruction]


# ---------------------------------------------------------------------------
# A host says which words are the operator's
# ---------------------------------------------------------------------------


def test_a_runtime_note_relaying_an_operator_answer_is_quoted_and_the_rest_is_not() -> None:
    answer = 'the operator answered your request [a1b2] "Which port?": 8443 — note: keep the old one running until Friday'
    note = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text=f"3 events:\n- build 412 finished\n- {answer}\n- disk at 71%")],
        metadata={OPERATOR_WORDS_METADATA_KEY: [answer]},
    )
    # The note is the runtime's: a summary may stand for it once the ledger
    # has quoted what the operator said in it.
    assert _is_plain_operator_turn(note) is False

    ledger = Ledger()
    _absorb(ledger, [note], source="blob fold-1")

    assert ledger.operator == [OperatorQuote(answer, "blob fold-1")]
    assert "8443" in ledger.identifiers
    rendered = ledger.render(LoopConstants())
    assert answer in rendered
    assert "build 412 finished" not in rendered


def test_a_runtime_note_marked_false_is_not_taken_for_the_operator() -> None:
    wake = Message(
        role=MessageRole.user,
        content_blocks=[TextBlock(text="the loop woke up: nothing to do")],
        metadata={OPERATOR_WORDS_METADATA_KEY: False},
    )
    assert _is_plain_operator_turn(wake) is False
    ledger = Ledger()
    _absorb(ledger, [wake])
    assert ledger.operator == []


def test_a_message_marked_true_is_quoted_whatever_its_role() -> None:
    relayed = Message(
        role=MessageRole.assistant,
        content_blocks=[TextBlock(text="Operator, over the phone: ship it on Monday, not before.")],
        metadata={OPERATOR_WORDS_METADATA_KEY: True},
    )
    ledger = Ledger()
    _absorb(ledger, [relayed])
    assert [quote.text for quote in ledger.operator] == ["Operator, over the phone: ship it on Monday, not before."]


def test_a_reply_to_a_question_is_quoted_whole_with_what_it_answered() -> None:
    reply = "Use the staging database.\nNo, because production is frozen until the audit is over."
    history = [
        Message(
            role=MessageRole.assistant,
            content_blocks=[ToolUseBlock(tool_call_id="q1", name="AskUser", arguments_json=json.dumps({"question": "Which database?"}))],
        ),
        Message(
            role=MessageRole.tool,
            content_blocks=[ToolResultBlock(tool_call_id="q1", content=reply, metadata={OPERATOR_WORDS_METADATA_KEY: True})],
        ),
    ]
    ledger = Ledger()
    _absorb(ledger, history, source="blob span-7")

    assert ledger.operator == [OperatorQuote(reply, "blob span-7", "reply to AskUser")]
    assert f"- (reply to AskUser) {reply}" in ledger.render(LoopConstants())


def test_an_unmarked_tool_result_is_not_taken_for_the_operator() -> None:
    ledger = Ledger()
    ledger.absorb_result(ToolResultBlock(tool_call_id="c1", content="Use the staging database."), tool_name="Exec")
    assert ledger.operator == []


@pytest.mark.asyncio
async def test_the_floor_points_a_quote_at_the_copy_it_kept() -> None:
    rule = "Deploy only from the release branch."
    history = [
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="the task")]),
        *(
            message
            for i in range(6)
            for message in (
                Message(role=MessageRole.user, content_blocks=[TextBlock(text=f"{rule} ({i})")]),
                Message(role=MessageRole.assistant, content_blocks=[TextBlock(text="noted " * 200)]),
            )
        ),
        Message(role=MessageRole.user, content_blocks=[TextBlock(text="recent")]),
    ]
    ledger = Ledger()
    result = await run_floor(
        history,
        LoopConstants(model_context_window=32_768, compaction_keep_recent_turns=1),
        free_target_tokens=10_000,
        ledger=ledger,
        blob_store=InMemoryBlobStore(),
        tenant_id="t",
    )
    assert result.messages_dropped > 0
    assert ledger.operator
    assert all(quote.source.startswith("blob ") for quote in ledger.operator)


# ---------------------------------------------------------------------------
# The budget: other sections give way first
# ---------------------------------------------------------------------------


def _crowded_ledger(instruction: str) -> Ledger:
    ledger = Ledger()
    _absorb(ledger, [Message(role=MessageRole.user, content_blocks=[TextBlock(text=instruction)])], source="blob op-1")
    for i in range(300):
        ledger._touch(f"/srv/app/module_{i:03d}/handler.py", "read")
        ledger._identifier("id", f"job-{i:05d}-x", 1, f"started job-{i:05d}-x on worker {i}")
        ledger._error(f"Exec: step {i} failed with exit code {i % 7 + 1}")
    return ledger


def test_other_sections_give_up_their_room_before_an_operator_quote() -> None:
    rc = LoopConstants()
    instruction = _cyrillic_instruction(2_000)
    ledger = _crowded_ledger(instruction)
    quote_tokens = estimate_tokens(instruction, rc)
    # More than the operator's 30 % share, and well within the whole budget.
    budget = int(quote_tokens * 1.8)

    rendered = ledger.render(rc, budget_tokens=budget)

    assert instruction in rendered
    assert "more characters of the operator's words" not in rendered
    assert "omitted)" in rendered  # the identifiers, files and failures were cut instead
    assert estimate_tokens(rendered, rc) <= budget + 8


def test_a_quote_that_cannot_fit_is_cut_with_a_marker_naming_the_copy() -> None:
    rc = LoopConstants()
    older = "Always write the migration notes in English."
    instruction = _cyrillic_instruction(6_000)
    ledger = Ledger()
    _absorb(ledger, [Message(role=MessageRole.user, content_blocks=[TextBlock(text=older)])], source="blob op-0")
    _absorb(ledger, [Message(role=MessageRole.user, content_blocks=[TextBlock(text=instruction)])], source="blob op-1")
    budget = estimate_tokens(instruction, rc) // 3

    rendered = ledger.render(rc, budget_tokens=budget)

    cut = re.search(r"^- (1\. .*?)\n(\[… \d+ more characters of the operator's words[^\n]*\])$", rendered, re.S | re.M)
    assert cut is not None
    head, marker = cut.group(1), cut.group(2)
    assert instruction.startswith(head) and len(head) > 200
    assert marker == f"[… {len(instruction) - len(head)} more characters of the operator's words — full text in blob op-1]"
    # The older instruction is not dropped in silence either.
    assert "[1 earlier message(s) of the operator's words omitted for room — full text in blob op-0]" in rendered
    assert older not in rendered
    assert estimate_tokens(rendered, rc) <= budget + 8


def test_quotes_the_state_no_longer_holds_are_still_accounted_for() -> None:
    ledger = Ledger()
    for i in range(45):
        _absorb(ledger, [Message(role=MessageRole.user, content_blocks=[TextBlock(text=f"instruction number {i}")])])
    assert len(ledger.operator) == 40 and ledger.operator_dropped == 5

    restored = Ledger.from_dict(json.loads(json.dumps(ledger.to_dict())))
    assert restored.operator == ledger.operator and restored.operator_dropped == 5
    assert "[5 earlier message(s) of the operator's words omitted for room — full text in the session transcript]" in restored.render(
        LoopConstants()
    )


def test_a_ledger_stored_with_bare_quotes_is_still_read() -> None:
    restored = Ledger.from_dict({"operator": ["keep the old port"], "compacted_messages": 3})
    assert restored.operator == [OperatorQuote("keep the old port")]
