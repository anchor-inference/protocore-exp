# The compaction contract

> Audience: anyone changing `runtime/context/`, or hosting the core on a model
> whose context window a long run outgrows.
> Code: `runtime/context/manager.py` (the pass), `runtime/context/compaction.py`
> (the tiers and the floor), `runtime/context/carrier.py` (what a summary is),
> `runtime/context/ledger.py` (what is carried by code),
> `runtime/turn_policies/compaction.py` and `runtime/query.py` (the gates).

A long run outgrows its window. Compaction is what keeps it running: it takes
the oldest part of the transcript out of the prompt and leaves something
smaller in its place. This page states what compaction guarantees, in terms a
test can check, and why each clause is there. Every clause has a failure
behind it that happened in a real long-running session.

## Vocabulary

| Term | Meaning |
|---|---|
| window `W` | `model_context_window`, the provider's context window in tokens |
| prompt `P` | the whole request: the history **plus** the system prompt and the tool definitions (the *overhead*). The provider's count of the last request plus the calibrated estimate of the messages added since, while the counted history is still the start of the history (`QueryEngine.prompt_tokens_now`); otherwise the calibrated estimate of the whole |
| trigger `T` | `compaction_trigger_tokens` (see [Budget arithmetic](#budget-arithmetic)); a pass opens when `P > T` |
| fixed part `F` | what no tier can remove: the overhead plus the protected set |
| target `R` | where an opened pass aims; `R < T` unless `F ≥ T` |
| pass | one run of the cascade below, opened by a gate |
| span | what one summary stands for: one or more adjacent tool-pairing units |
| carrier | the text that replaces a span: a summary, a fold or a floor digest |
| ledger | the one message that carries exact values out of compacted spans, written by code |
| protected set | what no tier may remove (see [Invariant 10](#10-what-is-never-removed)) |

## The invariants

### 1. Progress

Every opened pass ends with `P ≤ T`, or at the **floor**: nothing outside the
protected set is left to remove. A pass that could not change anything is not
opened at all. No run fails because compaction "made no progress", and no gate
fires on every iteration.

*Why.* A run over a history of hundreds of short rounds, under a large tool
surface, opened a pass on every iteration, freed 0–3 tokens each time and
failed on the retry budget after 75 passes. The gate measured the whole
prompt; the passes could shrink only the history, and no tier had anything it
was willing to touch.

The retry budget (`compaction_failed_max_retries`) survives only for the case
the floor cannot help with: a history already at its floor on which a tier
raised. A proactive pass that exhausts it suspends the summariser tiers for a
bounded stretch (masking and the floor keep running); a reactive pass hands the
turn to the output-cap ladder, and fails the run only when no smaller cap is
left — the one situation in which the history cannot be made to fit.

### 2. Tier order

A pass runs, cheapest first, only while the prompt is above the target:

1. **Masking** (`run_tier1_truncation`). Tool outputs outside the kept tail are
   replaced by a placeholder: any output of at least
   `tool_result_truncation_ratio × W`, and — oldest first, until the target is
   met — any output older than the `compaction_mask_keep_recent_results` most
   recent and at least `compaction_mask_min_tokens` long, except the output of
   a call that read an instruction file or a memory note
   (`compaction_mask_keep_reads_pattern`). Needs no model and cannot fail for
   want of one.
2. **Summarising** (`run_tier2_summarisation`, then `run_tier3_fold`). The
   oldest spans are replaced by a model summary; then runs of old summaries and
   old operator turns are merged into one fold, which a later fold can absorb.
3. **The floor** (`run_floor`). If the prompt is still above the trigger (above
   the target for a forced pass — a provider refusal, or an estimate past the
   emergency line), the oldest spans are removed, raw spans before summaries,
   until the target is met or nothing removable is left. One digest written by
   code stands where they were.

A summary is an improvement on the floor, never a precondition for progress.

### 3. A failed or absent summary is a normal outcome

A summariser call ends in one of: a summary, `transport` (the provider
raised), `timeout` (no answer within `compaction_summary_timeout_seconds`),
`empty` (no usable text), `too_large` (the request does not fit the
summariser's window) or `not_smaller` (the summary is no smaller than the
span). None of them fails the pass: the span stays, and the floor takes it if
the room is needed. Only `empty` and `too_large` are counted against the span
(`compaction_summary_failed_unit_max_attempts`), because only they repeat.

The summariser's **input is bounded by construction** to
`min(compaction_summariser_input_max_tokens, W / 4)`: a span larger than that is
shown with its tool outputs cut to head and tail lines, never mid-line. By
default the summariser is the model that just refused a request as too long;
its input has to fit anyway.

### 4. Facts from state, not from the model

Exact values are carried by code, in the **ledger**, not by the summariser.
Whenever a tier takes something out of the window it first records, from the
outgoing messages:

- the operator's words, verbatim and whole — the turns the operator typed,
  and the replies a host marks as the operator's (see
  [Who wrote a message](#who-wrote-a-message));
- the files the agent read, wrote or edited, by the tool's declared role;
- identifiers and exact values of recognisable shape — URLs, UUIDs, paths,
  hashes, timestamps, versions, `host:port` pairs, handles, id-shaped tokens —
  each with the line it stood on;
- the tool calls that failed, with the first line of what they said;
- the latest plan the agent recorded and the questions it asked.

The ledger is one user-role message tagged `protocore.compaction_ledger`; its
metadata holds its own structured state, and every pass rebuilds it from that
state plus what the pass removed. It is **never** shown to the summariser —
a record a model rewrites on every fold compounds its errors — and no tier
compacts it. Values of noisy shape (timestamps, bare numbers, hashes) are read
only from the operator's turns, the agent's own text and arguments, and the
*distinct lines* of a tool output: the lines whose shape, numbers aside, the
output does not repeat. A log of a thousand lines says its setting, its error
or its result once; that is the line kept.

What the ledger does not do: recognise a value with no shape. A count with its
unit, or a name, stays the summary's job.

**The operator's words are never cut in silence.** A quote is recorded whole —
there is no per-quote length limit — and within the ledger's budget the
operator's words are placed first: the identifiers, files, open items and
failures give up their room, oldest and lowest-ranked first, before a quote is
touched. Newest quotes are kept first, because the latest word on a matter is
the one in force. Only when the quotes alone outgrow the whole budget is one
cut, at a line or word boundary, and it ends with an explicit marker:

```text
[… 4395 more characters of the operator's words — full text in blob <ref>]
```

The reference is the copy the pass kept of what it removed — the same blob
store the placeholders of masked outputs and the summaries point into (clause
9) — or `the session transcript` when no copy could be made. Older quotes that
then find no room, and quotes past the state's bound of 40, are named on one
line, `[N earlier message(s) of the operator's words omitted for room — full
text in …]`, never dropped without a word.

*Why.* The ledger once clipped every quote at 1,200 characters with nothing to
show it had, and a long instruction reached the model after compaction as its
first third, read as the whole of it.

#### Who wrote a message

The core takes a user-role message to be the operator's when it carries no tool
result and none of the compaction, reference, seed or recovery flags. A host
that delivers its own events on the same channel, or the operator's replies on
another, says who wrote what with `OPERATOR_WORDS_METADATA_KEY`
(`"protocore.operator_words"`, in `protocore.contracts.types`):

| Where | Value | Meaning |
|---|---|---|
| `Message.metadata` | `True` | every text block of the message is the operator's |
| `Message.metadata` | `False` | a runtime note: nothing in it is the operator's, and it is compacted like any other message |
| `Message.metadata` | `list[str]` | a runtime note relaying the operator: each string is a passage of the operator's, quoted verbatim; the rest is the runtime's |
| `ToolResultBlock.metadata` | `True` | the result is the operator's reply to the question the call asked; quoted with `(reply to <tool>)` |

The core marks the result it writes when a question interrupt is resolved
with an answer; a host sets the key on anything else it delivers. A
`list[str]` note is not an operator turn for clause 10: it may be summarised,
because its operator passages are in the ledger before it goes. The key is
trusted as it is set, so a host must not pass a tool's or a remote server's
own metadata through under it.

### 5. The carrier

A summary is **plain text under five fixed headings**: `## Progress`,
`## Facts and values`, `## Decisions and constraints`, `## Failures`, `## Open`.
It is requested with `ILLMProvider.complete_text` — never under a response
schema — and read tolerantly by `read_carrier`:

- reasoning blocks and code fences are removed;
- a JSON envelope the model wrote anyway is unwrapped, whole or unterminated;
- a reply without headings is kept under `## Notes`;
- a reply the output cap cut loses its last, unfinished line, not the rest;
- each section is clamped to its own share of the budget, at line boundaries —
  a single overlong line at a word boundary — so a long account of progress
  cannot push the exact values out.

*Why.* One JSON `summary` string under a 1,024-character ceiling lost the
verbatim identifiers on every model it was measured on, while headed plain
text with an adequate budget kept them. And a schema request is the one
request that can come back as a parse failure: an object the model never
closed, prose where an object was due. Headed text has no way to be malformed.

### 6. The summariser's instructions are not the user's

The instruction is the **system** message; the material is fenced as data in
the user message (`<transcript>…</transcript>`); any sentence of the
instruction echoed into the reply is removed; and every carrier opens with a
line saying it was written by the runtime, not by the user, and is reference,
not instruction. A summary that carried the compactor's own imperative forward
has been seen attributed to the user, after which the run stopped acting.

### 7. Re-injection

The system prompt and the tool definitions are rebuilt for every request, so
compaction never touches them. The ledger is put back after every pass, at the
boundary between the compacted past and the kept present, and it lists the
files touched most recently first, so the agent knows what to re-read rather
than trusting a summary of its contents.

### 8. The gate

The gate measures the whole prompt — history plus the system prompt and tools
the last request carried — against the trigger, which already sits below the
window less the output reserve (see below). An opened pass aims at the target
`R = F + compaction_target_ratio × (min(T, P) − F)`: it keeps that share of the
room it can actually work in, above the fixed part `F` that no tier can remove,
so the next pass is many turns away, not one. A pass that frees less than
`compaction_min_gain_ratio` backs the per-iteration gate off for
`compaction_no_gain_backoff_iterations`.

*Why.* The target used to be `compaction_target_ratio × min(T, P)` on the whole
prompt. Under a large tool surface, or behind turns seeded from an earlier run,
the fixed part alone sat above it: no amount of compaction could reach it, so
every tier and the floor ran to exhaustion and a routine pass drained the run's
own work to the task and the tail, when removing a round or two would have
brought the prompt back under the trigger.

### 9. Reversibility

Nothing is lost that cannot be found again. A masked output leaves a
placeholder that names the tool, its size and the blob reference of the
original. A summary, fold or floor digest names the blob holding the messages
it replaced. A host that keeps its own durable transcript (and most do) is the
second copy.

### 10. What is never removed

The first user turn of the run (the task), the ledger, frozen reference blocks
(they may be blobbed by masking), the `compaction_keep_recent_turns` trailing
messages (one after a provider refusal) as long as they fit
`compaction_keep_recent_max_ratio` of the trigger, and the batch of tool
results the model has not read yet with the message that asked for it. When
this fixed part is still at or over the trigger, a routine pass runs only the
model-free tiers: the summariser cannot bring the prompt under the trigger
then, and is not called for nothing on every iteration. Outside reactive recovery, turns seeded from an
earlier run are untouched; reactive recovery may compact them, never in one
span with this run's turns, and every replacement keeps the seed tag. An
operator's turn — a typed one, or one a host marked `True` — is never
summarised; a fold or the floor may take it only after the ledger has quoted
it, whole. The operator's replies that arrive as tool results or inside runtime
notes are quoted by the ledger before any tier takes them.

### 11. Tool pairing stays whole

A span is a closed component of tool calls and the results answering them;
no tier ever leaves a call without its result or a result without its call.

### 12. Replays stay replayable

A carrier depends only on the messages it replaces: no timestamps, no
identifiers minted at random. The same history produces the same requests,
which is what lets a recorded run be replayed past its first compaction.

### 13. Its records are marked as its own

Everything compaction writes into the history — the ledger, the summaries and
folds, the floor digest — is user-role, because a provider accepts no system
message past the first. None of it is the user's message, and a host must not
store or show it as one. Every such message answers `Message.origin` with
`MessageOrigin.compaction`; the loop's own notes answer one of their own
origins (see [contracts](contracts.md#messages--content)), and the
conversation answers `MessageOrigin.conversation`. The origin is derived from the tags compaction
already sets (`protocore.compaction_summary`, `protocore.compaction_ledger`), is
serialised with the message — each entry of a snapshot's `history` carries
`"origin"` — and is never sent to a provider, so what the model reads is
unchanged. It is not a field one sets: `Message(origin=...)` is ignored, and a
message is marked as compaction's by its tag. Both tags are exported from
`protocore.contracts` as `COMPACTION_SUMMARY_METADATA_KEY` and
`COMPACTION_LEDGER_METADATA_KEY`.

A host that persists history (`persist_history_delta`, `persist_session_history`,
or its own reading of `engine.history`) filters on it: a compaction record
stands for turns the host already holds in their original form, so it is
either dropped or stored under its own origin, never as the user's turn. The
runtime reads it the same way: a compaction record does not start a new round
of the run, so work done before a pass still counts as the run's output.

*Why.* The ledger was stored by a host as the user's own message, drawn as a
user bubble, and seeded into the next run as an operator turn.

## Between runs

The tiers above run inside a run. A host that keeps a session alive across
runs also rewrites the history between them — a compaction after a run, a
revert of the last turn, a clear, an import — and those rewrites go through the
engine, not around it.

### Replacing the history

`await engine.replace_history(messages, reason=...)` puts `messages` in place
of the history and retires what was derived from the old one: the compaction
checkpoint, the compaction state and its back-offs, the store's persist marker
(the next hand-over is a rewrite, so a revert to a prefix does not leave the
dropped rows in the store) and the prompt measurement. It emits
`history_replaced` (`reason`, `messages_before`, `messages_after`,
`prompt_tokens_before`, `prompt_tokens_after`) on the engine's event stream and
returns it. It raises while a turn is driving.

The prompt is re-sized without a call: the provider's last count, less the
fixed part, is scaled by how much of the history survived, and never exceeds
the count. The estimator's calibration cancels out of the ratio, so the answer
errs only low — the next compaction comes a turn late at most — and messages
added afterwards are estimated on top of it, as after any count. With no count
to scale, the estimate stays the authority. The scaled anchor rides the
snapshot (`prompt_anchor`), so a run resumed on another engine sizes its prompt
the same way.

*Why.* A host assigned the history and forgot the measurement; the trigger read
the size of the history it had just shrunk, and nearly half the compactions of
a long session re-summarised a tail the previous one had already made small.

### Compacting a session

`protocore.runtime.context.session_compaction.compact_session(history,
summarisers=[...], ...)` replaces all but a kept tail of a history with one
summary, and returns the new history without writing anything
(`SessionCompaction`); `commit_session_compaction(engine, result)` puts it into
an engine through `replace_history`, and `result.apply(history)` builds it for a
host's own store. Its clauses:

- **The previous summary is the base.** Compaction records at the head of the
  history (`leading_summaries`) go to the summariser whole, inside
  `<previous-summary>`, with `SummaryPrompts.base_rule`; the transcript after
  them is rendered whole (`render_for_summary`): the user's words never cut,
  agents' text, notes and tool exchanges cut in the middle past their bounds,
  the ledger and the loop's scaffolding (`nudge`, `wind_down`, `recovery`) left
  out. *Why.* A summary re-condensed as part of a clipped transcript broke the
  chain at every pass, and a session kept only its last topic.
- **Parts, merged in rounds.** A transcript longer than
  `SessionCompactionSettings.chunk_tokens` is cut on line ends into parts
  summarised in parallel (at most `parallel_calls` at once); while the notes are
  still longer than one part they are merged in groups of at least two, up to
  `merge_rounds_max` rounds; the last merge is made onto the base.
- **Sections, checked once more.** A summary that lacks one of
  `SummaryPrompts.sections`, repeats one or reorders them is asked for again
  with `SummaryPrompts.retry` naming the problem; a second miss is accepted and
  reported as `section_problem`. The word budget grows with the window
  (`words_for`).
- **A ladder of summarisers.** `summarisers` are tried in order; a provider
  error, a timeout or an empty answer hands over to the next, and
  `SessionCompactionFailed` lists every rung's failure when none answered.
  *Why.* A summary model past its limit ended a compaction in a bare error
  while the session's own model stood ready.
- **Retries.** A retryable failure is asked again after each of
  `retry_pauses`; a rate limit and a permanent refusal go to the next rung at
  once; a call that exceeds `call_timeout_seconds` gets one more try at twice
  that.
- **The kept tail starts at a turn boundary** (`session_compaction_cut`): a
  user's own turn, never a tool result or scaffolding, so no tool exchange is
  split. The newest ledger is carried past the summariser unchanged, after the
  summary. `decorate` adds code-written text (quotes, an index, a pointer to an
  archive) to the summary before it is wrapped, and the summary carries the
  user's passages under `OPERATOR_WORDS_METADATA_KEY` for the next pass.
- **Atomic and cancellable.** Nothing is held while the summary is made. The
  result keeps the summarised prefix by identity and applies only while the
  history still starts with exactly those messages: turns appended meanwhile
  are kept after the summary; a changed summarised message makes it stale, and
  `commit_session_compaction` returns `None` with the history untouched — as it
  does while a turn is driving. Setting `cancel`, or cancelling the task,
  raises `SessionCompactionCancelled` and leaves nothing behind.

`CompactionHold` and `CompactionHoldPolicy` are the pause on automatic
compaction a host persists (`to_dict` / `from_dict`): after a failure for
`retry_seconds`, doubling per failure up to `retry_max_seconds`; after a pass
that left the prompt over the trigger or freed less than `min_freed_ratio`,
until the prompt grows; either ends when the prompt grows by `growth_ratio` of
the window. *Why.* A session whose summariser timed out was held three minutes
per message and never shrank.

Inside a run, `QueryEngine(compaction_providers=[Summariser(provider, model),
...])` gives the tiers the same ladder (`SummariserLadder`): each rung is asked
with its own model and the run's own provider is the last rung. Each call gets
`compaction_summary_timeout_seconds`, and the whole exchange one such deadline
per rung. `compaction_provider` (one provider, no fallback) still works and
cannot be combined with it.

## Budget arithmetic

With `O = min(W × llm_output_max_tokens_ratio, llm_output_min_tokens)` when the
provider reserves the output inside the window
(`provider_reserves_output_in_context_window`), else 0. Every request is fitted
to the window — its output cap is cut to what the prompt leaves — so the trigger
keeps back room for a usable answer, not for the whole configured output cap.
When the ceiling is lower than `W × compaction_trigger_ratio`, the trigger in
force is the ceiling and every compaction event says so:
`configured_trigger_tokens` is the configured trigger, `trigger_threshold` the
one in force and `trigger_limited_by` is `accept_ceiling`.

```text
T      = min(W × compaction_trigger_ratio,
             W − O − request_context_safety_tokens − W × compaction_trigger_turn_headroom_ratio)
F      = overhead + tokens of the protected set        # the pass's own profile
R      = F + compaction_target_ratio × max(0, min(T, P_before) − F)
need   = P + ledger growth − R                 # what the pass still has to free
span   ≤ compaction_summary_group_max_tokens    # adjacent units joined, oldest first
sent   when span > max(empty-summary size, compaction_summary_min_unit_tokens)
input  ≤ min(compaction_summariser_input_max_tokens, W / 4)
budget = clamp(span × compaction_summary_ratio,
               compaction_summary_min_output_tokens,
               compaction_summary_max_output_tokens)   # a fold: compaction_fold_max_output_tokens
wire   = 2 × budget                             # room to overshoot; the clamp trims
ledger = clamp(W × compaction_ledger_ratio,
               compaction_ledger_min_tokens, compaction_ledger_max_tokens)
```

Section shares of a carrier's budget: Facts and values 35 %, Progress 25 %,
Decisions and constraints 15 %, Open 15 %, Failures 10 %; whatever a section
does not use is shared out in that order. The ledger places the operator's
words first, whole (clause 4); what they leave is shared in the proportions
identifiers 30, files 15, open items 15, failures 10, and whatever a section
does not use goes to the next that needs it.

Example, a 256k window with a 64k output cap and a trigger ratio of 0.59:
`T = min(151,040, 256,000 − 65,536 − 1,024 − 38,400) = 151,040`,
`R = 90,624` with nothing fixed; with a 56,000-token tool surface and 75,000
tokens of seeded turns, `F = 131,000` and `R = 143,024`, where the whole-prompt
ratio would have asked for 90,624, below what the pass can reach; a 6,000-token span is summarised within 1,200 tokens, sent with
a 2,400-token cap; the ledger may spend 5,120 tokens.

## Failure states

| Condition | What happens | Visible as |
|---|---|---|
| Summariser raises (5xx, rate limit, reset) | span stays; floor if needed | `tier2_failures.transport` |
| Summariser gives no answer in time | call abandoned; span stays | `tier2_failures.timeout` |
| Reply has no usable text | counted against the span; span stays | `tier2_failures.empty` |
| Request larger than the summariser's window | counted against the span | `tier2_failures.too_large` |
| Summary no smaller than its span | discarded | `tier2_failures.not_smaller` |
| Reply is JSON, unterminated JSON, unheaded, or cut | recovered and kept | `tier2_recovered.*` |
| Tiers stop short of the trigger | the floor removes oldest spans | `floor_dropped`, `outcome` |
| Nothing left to remove, still over the trigger | the gate stops opening passes | `outcome = at_floor`, then no pass |
| A tier raises at the floor (proactive) | summariser tiers suspended for a stretch | `compaction_exhausted_proactive_suspended` |
| A tier raises at the floor (after a refusal) | output-cap ladder; the run fails only when no smaller cap is left | `reactive_413_compaction_exhausted` |
| A `pre_compact` hook refuses | nothing written; the run returns to `RUNNING` | `compaction_refused_by_hook` |

## Observability

`compaction_completed` carries, besides the historical `tokens_before`,
`tokens_after`, `tier1_freed`, `tier2_summarised` and `tier3_folded`:

| Key | Meaning |
|---|---|
| `outcome` | `below_target`, `below_trigger`, `above_trigger`, `at_floor` or `unchanged` |
| `prompt_before`, `prompt_after` | the whole prompt, overhead included |
| `trigger_threshold`, `target_tokens` | the pass's `T` and `R` |
| `fixed_tokens` | the pass's `F`: overhead plus the protected history |
| `tier1_masked_by_age` | outputs masked for their age rather than their size |
| `tier2_attempted`, `tier2_failures`, `tier2_recovered` | calls made, failures by kind, replies recovered by kind |
| `tier3_failures` | fold failures by kind |
| `floor_dropped`, `floor_reached` | messages the floor removed; whether it ran out of things to remove |
| `ledger_tokens` | the size of the rebuilt ledger |
| `replaced_indices` | positions, in the history handed to the store just before the pass, of the messages the pass removed or rewrote |
| `inserted_indices` | positions, in the history after the pass, of the messages it wrote (summaries, folds, the ledger, masked outputs) |
| `history_messages_before`, `history_messages_after` | the lengths of those two histories |

Before `compaction_started` the loop hands the history to the session store
(`persist_history`), so the rounds the run produced since the last hand-over
are durable before the pass rewrites them, and `replaced_indices` index exactly
the rows the store was just given. Messages carry no id of their own; the
positions are compared by object identity, like the persist marker. *Why.* A
host transcribed the whole history itself on `compaction_started`, because
otherwise those rounds existed only as the summary, and labelled the summary on
`compaction_completed` by diffing two histories.

Every pass also logs one line, `DIAG compaction.pass`, with the same figures,
and the floor logs `DIAG compaction.floor`.

## Validating on an endpoint

The carrier depends on the model that writes it: a format one endpoint handles
well another may not. Two harnesses in the test suite are the shape to rerun
against a real endpoint whenever the model or the endpoint changes:

- `tests/_fixtures/compaction/planted.py` — a synthetic agent session of
  about 150k tokens in six chunks, with fifty planted values of which fourteen
  are graded by exact match after every chunk has been compacted and the
  summaries folded; two of the fourteen have no shape the ledger can recognise.
- `tests/_fixtures/compaction/long_loop_shape.json` — the shape, without its
  content, of a long-running loop's history at the moment its compaction
  stopped making progress: 357 messages, 153 tool definitions.

`tests/unit/runtime/test_compaction_contract.py` drives both with summariser
doubles that succeed, fail, hang, return JSON, return it unterminated, return
nothing or keep nothing; point the same code at a real provider to measure it.

## Configuration

| Constant | Default | Role |
|---|---|---|
| `compaction_trigger_ratio` | 0.8 | upper bound on `T` as a share of the window |
| `compaction_trigger_turn_headroom_ratio` | 0.10 | one turn's room below the acceptance ceiling |
| `llm_output_min_tokens` | 4096 | answer room the trigger keeps back when smaller than `llm_output_max_tokens_ratio` of the window |
| `compaction_target_ratio` | 0.6 | where a pass aims, below the trigger |
| `compaction_emergency_ratio` | 0.95 | the proactive emergency line |
| `compaction_keep_recent_turns` | 4 | trailing messages kept verbatim, while they fit the bound below |
| `compaction_keep_recent_max_ratio` | 0.25 | share of the trigger the kept tail may hold; at least `compaction_force_keep_recent_turns` messages and the unread batch are kept whatever they weigh |
| `compaction_force_keep_recent_turns` | 1 | the same after a provider refusal |
| `tool_result_truncation_ratio` | 0.10 | the most one tool result may take: a larger fresh result is cut to its head where it enters (the whole value stored, its reference named), and an older output this size is masked whatever its age |
| `compaction_mask_keep_recent_results` | 8 | outputs never masked for age |
| `compaction_mask_min_tokens` | 300 | smaller outputs are not masked for age |
| `compaction_mask_keep_reads_pattern` | instruction and memory files | outputs of matching reads are not masked for age |
| `compaction_mask_distinct_lines` | 6 | distinct lines a placeholder keeps |
| `compaction_placeholder_preview_chars` | 240 | head and tail preview in a placeholder |
| `compaction_summary_group_max_tokens` | 6,000 | the largest span one summary stands for |
| `compaction_summary_min_unit_tokens` | 0 | the smallest span worth a call |
| `compaction_summary_ratio` | 0.2 | summary budget as a share of its span |
| `compaction_summary_min_output_tokens` | 256 | floor on a summary's budget |
| `compaction_summary_max_output_tokens` | 2,048 | ceiling on a span summary's budget |
| `compaction_fold_max_output_tokens` | 3,000 | ceiling on a fold's budget |
| `compaction_summariser_input_max_tokens` | 12,000 | ceiling on what one call is shown |
| `compaction_summary_timeout_seconds` | 120 | deadline for one call |
| `compaction_summariser_parallelism` | 4 | calls in flight at once |
| `compaction_summary_failed_unit_max_attempts` | 2 | repeatable failures before a span is left to the floor |
| `compaction_fold_min_messages` | 4 | shortest run the fold merges |
| `compaction_fold_min_tokens` | 1,500 | lightest run the fold merges |
| `compaction_fold_keep_operator_turns` | 4 | recent operator turns never folded |
| `compaction_fold_max_spans_per_pass` | 2 | folds per pass |
| `compaction_ledger_ratio` | 0.02 | ledger budget as a share of the window |
| `compaction_ledger_min_tokens` | 1,000 | floor on the ledger budget |
| `compaction_ledger_max_tokens` | 6,000 | ceiling on the ledger budget |
| `compaction_failed_max_retries` | 2 | failed passes at the floor before suspension |
