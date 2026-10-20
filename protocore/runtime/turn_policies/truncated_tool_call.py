"""A tool call the model started writing and never finished.

The model opens a call, streams part of its arguments, and then the round
ends with anything but a length cap: an ordinary stop, or a ``tool_use`` some
servers report for a call they cut. What arrives is a call
whose arguments were salvaged by closing the braces the model never sent —
parseable, and wrong. Dispatching it fails on validation, and the model reads
that as "the tool is broken" rather than as "the arguments were too big",
which is the one thing it needed to learn.

So the call is not dispatched. Each truncated call gets a synthetic error
result that says, in both languages, how much it managed to emit and what
size to chunk to, the transcript keeps that instruction, and the turn is run
again from a rebuilt context. The count is bounded per message, because a
model stuck in a "``{``, stop" loop would otherwise spend the whole run on it,
and a clean round gives the whole count back.

The calls the model DID finish in the same message are still dispatched: the
truncation is one call's problem, and dropping its siblings would throw away
work the model completed. That dispatch is a second entry into the dispatch
path, so it carries the same two guarantees the main one does — no new call
after a stop is observed, and every call a gate holds is parked, not just the
first.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from protocore.contracts.llm import LLMProviderError
from protocore.contracts.turn_policy import (
    TurnContext,
    TurnCoordinate,
    TurnDirective,
)
from protocore.contracts.types import Message, MessageRole, ToolCall, ToolResultBlock
from protocore.runtime.events import TurnEvent
from protocore.runtime.loop_state import LoopState
from protocore.runtime.turn_policies import RunCounter
from protocore.runtime.turn_policies.run_ceilings import TerminalEmitter
from protocore.runtime.turn_policies.sibling_walk import (
    CallParker,
    ToolDispatcher,
    dispatch_parking_holds,
    prose_gate_just_injected,
)

#: The one envelope that tells the host everything the run is now waiting on.
ParkedAnnouncer = Callable[[Any, Any], TurnEvent]

#: Whether the result this call left in history ends the run.
ResultIsTerminal = Callable[[Any, str], bool]

#: The end-of-turn frame between two assistant messages of the same turn.
ToolUseStop = Callable[[Any], TurnEvent]

#: The envelope a synthetic, never-dispatched call's error result travels in.
RecoveryResultEvent = Callable[[Any, str, str], TurnEvent]

#: A rough lines-per-chunk proxy: source and prose both run near this.
_CHARS_PER_LINE = 50

#: The payload size the chunk-count estimate is quoted against.
_ESTIMATE_TARGET_BYTES = 10_240


class TruncatedToolCallRecoveryPolicy:
    """Teach the model to chunk, instead of dispatching half a call."""

    name = "truncated_tool_call_recovery"
    coordinates = frozenset({TurnCoordinate.tool_calls_ready})

    __slots__ = (
        "_dispatch",
        "_llm_terminal",
        "_pair_orphans",
        "_park",
        "_parked_event",
        "_recoveries",
        "_recovery_result_event",
        "_result_is_terminal",
        "_tool_use_stop",
    )

    def __init__(
        self,
        *,
        recoveries: RunCounter,
        llm_terminal: TerminalEmitter,
        dispatch: ToolDispatcher,
        park: CallParker,
        parked_event: ParkedAnnouncer,
        result_is_terminal: ResultIsTerminal,
        pair_orphans: Callable[[Any], None],
        tool_use_stop: ToolUseStop,
        recovery_result_event: RecoveryResultEvent,
    ) -> None:
        self._recoveries = recoveries
        self._llm_terminal = llm_terminal
        self._dispatch = dispatch
        self._park = park
        self._parked_event = parked_event
        self._result_is_terminal = result_is_terminal
        self._pair_orphans = pair_orphans
        self._tool_use_stop = tool_use_stop
        self._recovery_result_event = recovery_result_event

    async def apply(self, turn: TurnContext) -> AsyncIterator[TurnEvent]:
        engine = turn.engine
        truncated = [
            call
            for call in turn.pending_tool_calls
            if call.args_partial_truncated and not call.truncated_by_output_cap
        ]
        # Any finish but the length cap, which the output-cap policy resumes. Only
        # ``stop`` used to count, so a cut call a server closed with ``tool_use``
        # was dispatched on its brace-repaired arguments and wrote half a file.
        if not truncated:
            # A clean round. Give the whole per-message count back, so one
            # early truncation does not permanently consume a slot.
            self._recoveries.reset(engine)
            return

        bound = engine.rc.tool_call_max_truncation_recoveries_per_message
        if self._recoveries.read(engine) >= bound:
            async for event in self._llm_terminal(
                engine,
                LLMProviderError(
                    "tool_call_max_truncation_recoveries_per_message exhausted "
                    "(model kept emitting partial tool args + stop)"
                ),
                kind="tool_call_truncated_exhausted",
            ):
                yield event
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = "tool_call_truncated_exhausted"
            return

        self._recoveries.charge(engine)
        for call in truncated:
            yield self._teach_one(engine, call)
        await engine.persist_snapshot()

        async for event in self._dispatch_the_rest(turn, truncated):
            yield event
        if turn.outcome.directive is not TurnDirective.proceed:
            return

        yield self._tool_use_stop(engine)
        turn.outcome.directive = TurnDirective.restart_turn
        turn.outcome.rebuild_context = True
        turn.outcome.reason = "tool_call_truncated"

    # -- the instruction ----------------------------------------------------

    def _teach_one(self, engine: Any, call: ToolCall) -> TurnEvent:
        """Pin a synthetic error result onto one truncated call.

        The message is rendered in both languages and joined, because the run
        does not know which one the reader speaks, and it carries three
        numbers rather than a scolding: how much the model emitted, what a
        chunk should weigh, and how many chunks a large payload will take.
        The chunk count is floored at two — a message that says "send it in
        one chunk" contradicts the instruction not to retry as-is.
        """
        rc = engine.rc
        chunk_bytes = rc.tool_call_max_input_chunk_bytes
        partial_length = len(json.dumps(call.arguments, ensure_ascii=False))
        placeholders: dict[str, Any] = {
            "tool_name": call.name,
            "partial_length": partial_length,
            "chunk_bytes": chunk_bytes,
            "chunk_bytes_lines": max(1, chunk_bytes // _CHARS_PER_LINE),
            "chunk_count_estimate": max(
                2, (_ESTIMATE_TARGET_BYTES // chunk_bytes) + 1
            ),
        }
        message = "{}\n\n{}".format(
            engine.prompt_text("tool_call_truncation_recovery_en", **placeholders),
            engine.prompt_text("tool_call_truncation_recovery_ru", **placeholders),
        )
        engine.history.append(
            Message(
                role=MessageRole.tool,
                content_blocks=[
                    ToolResultBlock(
                        tool_call_id=call.id, content=message, is_error=True
                    )
                ],
            )
        )
        # The dispatcher has no further use for the id → name mapping once the
        # synthetic result is written; the same cleanup a real dispatch does.
        engine.forget_tool_name(call.id)
        return self._recovery_result_event(engine, call.id, message)

    # -- the siblings the model did finish ----------------------------------

    async def _dispatch_the_rest(
        self, turn: TurnContext, truncated: Sequence[ToolCall]
    ) -> AsyncIterator[TurnEvent]:
        """Dispatch every non-truncated call of the message, in order.

        The stop check is repeated before each one rather than taken once: the
        yields above, the snapshot, and each dispatch's own awaits are all gaps
        a cancel can land in, and a cancel that landed in one of them must not
        be followed by a NEW side effect. It is asked through the turn's own
        checkpoint rather than read off the engine, so a host that replaced
        what a stop MEANS is obeyed here too — this seam was the one place
        that still answered a stop in its own words.
        """
        engine = turn.engine
        flags = turn.flags
        flags.approval_pending = False
        flags.terminal_tool_completed = False
        # The calls to skip are named by id, which is what the sibling
        # recovery walk does. Identity is what the question is actually
        # about, and asking it the same way in both places is what keeps the
        # two from answering it differently later.
        cut_ids = {call.id for call in truncated}
        parked: list[Any] = []
        for call in turn.pending_tool_calls:
            if call.id in cut_ids:
                continue
            if parked and engine.stop_requested:
                # A cancelled run waits for nothing: the decisions this same
                # message parked a moment ago can no longer be acted on, and
                # the teardown below is about to answer their calls with
                # synthetic errors — a wait witnessing an answered call is a
                # question the host can never close.
                for held_wait in parked:
                    engine.release_interrupt(held_wait.interrupt_id)
                parked.clear()
                flags.approval_pending = False
            async for event in turn.cancel_checkpoint():
                yield event
            if turn.flags.terminal_yielded:
                turn.outcome.directive = TurnDirective.end_turn
                turn.outcome.reason = "stop_requested"
                return
            held_count = len(parked)
            async for event in dispatch_parking_holds(
                engine, call, dispatch=self._dispatch, park=self._park, parked=parked
            ):
                yield event
            if len(parked) > held_count:
                continue
            if parked:
                # Something in this message is already held for a decision, so
                # nothing behind it may end the run: the seal below and the
                # announcement both assume the turn is free to move on, and a
                # run already on its way to AWAITING cannot also be completed.
                continue
            if prose_gate_just_injected(engine):
                # The gate refused this call and asked for the answer in prose.
                # Dispatching the siblings behind it would file their results
                # after that question, so the batch stops here and the
                # corrective round carries what is left.
                break
            if self._result_is_terminal(engine, call.id):
                flags.terminal_tool_completed = True
                break

        if parked:
            flags.approval_pending = True
            engine.transition_to(LoopState.AWAITING)
            await engine.persist_snapshot()
            yield self._parked_event(engine, parked[0])
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = "approval_pending"
            return

        if flags.terminal_tool_completed:
            self._pair_orphans(engine)
            turn.outcome.directive = TurnDirective.end_turn
            turn.outcome.reason = "terminal_tool_completed"


__all__ = ["TruncatedToolCallRecoveryPolicy"]
