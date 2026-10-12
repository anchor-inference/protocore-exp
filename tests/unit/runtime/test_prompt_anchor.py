"""The compaction gate sizes the prompt from the provider's count of the last request.

The estimate alone ran 45% above the provider's count on a real session (277k against 191k) and
compacted a third of a window that was still free. Anchored on the count, the gate is exact for
everything but the newest round, and falls back to the estimate once the history is rewritten.
"""

from __future__ import annotations

from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.types import Message, MessageRole, TextBlock


def _say(role: MessageRole, text: str) -> Message:
    return Message(role=role, content_blocks=[TextBlock(text=text)])


def test_the_prompt_is_the_providers_count_plus_what_came_after_it(engine_factory) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=1_000_000, token_estimate_calibration=1.5))
    engine.history = [_say(MessageRole.user, "x" * 40_000), _say(MessageRole.assistant, "y" * 40_000)]
    estimated = engine.prompt_tokens_now()
    engine.note_dispatched_history()
    engine.note_observed_prompt(engine.effective_model_name, 12_000)
    assert engine.prompt_tokens_now() == 12_000, "nothing was added since the count"
    engine.history.append(_say(MessageRole.user, "z" * 400))
    tail = engine.context_manager.current_prompt_tokens(engine.history[2:])
    assert engine.prompt_tokens_now() == 12_000 + tail
    assert estimated > 12_000 + tail, "the calibrated estimate of the whole was the larger number"


def test_a_rewritten_history_or_another_model_falls_back_to_the_estimate(engine_factory) -> None:
    engine = engine_factory(rc=LoopConstants(model_context_window=1_000_000))
    engine.history = [_say(MessageRole.user, "x" * 4_000), _say(MessageRole.assistant, "y" * 4_000)]
    engine.note_dispatched_history()
    engine.note_observed_prompt(engine.effective_model_name, 99_000)
    engine.history[0] = engine.history[0].model_copy(update={"content_blocks": [TextBlock(text="[masked]")]})
    assert engine.prompt_tokens_now() < 99_000, "a compaction replaced a counted message: the count no longer applies"
    engine.history = [_say(MessageRole.user, "x" * 4_000)]
    engine.note_dispatched_history()
    engine.note_observed_prompt("another-model", 99_000)
    assert engine.prompt_tokens_now() < 99_000


def test_the_gate_decides_on_the_anchored_size(engine_factory) -> None:
    rc = LoopConstants(model_context_window=100_000, compaction_trigger_ratio=0.8, token_estimate_calibration=2.0)
    engine = engine_factory(rc=rc)
    engine.history = [_say(MessageRole.user, "x" * 200_000)]  # about 50k raw, 100k at the calibration
    assert engine.needs_compaction(), "the over-calibrated estimate alone would compact"
    engine.note_dispatched_history()
    engine.note_observed_prompt(engine.effective_model_name, 55_000)
    assert not engine.needs_compaction(), "the provider counted 55k of a 100k window"
