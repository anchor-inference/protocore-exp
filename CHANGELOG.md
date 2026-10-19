# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`OPERATOR_WORDS_METADATA_KEY` (`"protocore.operator_words"`) says who
  wrote a message.** `True` on `Message.metadata` makes the message the
  operator's, `False` makes a user-role message a runtime note, and a list of
  strings names the operator's passages inside a runtime note; `True` on
  `ToolResultBlock.metadata` makes the result the operator's reply. The
  compaction ledger quotes whatever it marks, whole. The core sets it itself on
  the result it writes when a question interrupt is answered. See
  `docs/compaction.md`.
- **`QueryEngineConfig.trimmed_tool_result_ids`** seeds the results
  stale-trimming already cut, as `QueryEngine.trimmed_tool_result_ids` hands
  them out. A host that starts each run of a session on a new engine started
  the set empty, so the next run cut, at its first request, whatever the last
  one had left whole, and counted the results already cut as new savings.
- **`Message.reasoning_items` and `ProviderDelta.reasoning_items` keep a
  provider's opaque reasoning on the turn it belongs to.** A provider that
  holds a model's reasoning on its own side (the Responses API with `store`
  off) hands back an encrypted copy, and the model reasons from it on the next
  request; without it a reasoning model starts every tool step from nothing.
  An adapter puts the records on a delta, the core keeps them on the assistant
  message it builds, and the adapter decides on the next request whether to
  send them again. The field is omitted from a dump when unset, so stored
  messages and request digests without it are unchanged. Tier 1 sheds it
  together with `reasoning_content`, and the token estimate uses an item's own
  `tokens` count where it gives one.
- **`LoopConstants.compaction_mask_keep_reads_pattern`** keeps the outputs of
  calls that read an instruction file or a memory note out of the age pass of
  Tier 1. The default matches `AGENTS.md`, `MEMORY.md` and notes in a
  `memory/` or `memories/` folder: masked minutes after it was read, the
  operator's standing knowledge was gone from a long run. Empty keeps nothing.
- **`QueryEngine.prompt_tokens_now()`** sizes the prompt from the provider's
  count of the last request plus an estimate of the messages added since, while
  the counted history is still the start of the history; otherwise from the
  calibrated estimate. The compaction gates (`needs_compaction`,
  `needs_emergency_compaction`) decide on it. The estimate alone once ran 45%
  above the count and compacted a third of a window that was still free.
  `ContextManager.needs_compaction` and `needs_emergency_compaction` take the
  measured size as `current_tokens`.
- **`ProviderFailure`, the typed verdict on a provider failure
  (`protocore.contracts.llm`).** A frozen model with `reason`
  (`ProviderFailureReason`), `retryable`, `should_fallback`,
  `retry_after_seconds` and `provider_message`, attached as
  `LLMProviderError(message, classified=ProviderFailure(...))`; every
  `LLMError` now takes `classified=`. The loop has always read these fields by
  name off a `classified` attribute, and nothing published them, so an adapter
  author had to read the loop to learn them. What the loop does with each
  reason is three public sets drawn from the enum:
  `PROVIDER_ADVANCE_REASONS`, `PERMANENT_FAILURE_REASONS` and
  `SURFACE_INDEPENDENT_FAILURE_REASONS`. Two reasons are new: `connection` (the
  endpoint could not be reached) and `budget` (a spend cap the host enforces).
  `ProviderFailure` and `ProviderFailureReason` are re-exported from
  `protocore.contracts`. See "Reporting a provider failure" in
  `docs/contracts.md`.
- **`classify_transport_exception(exc)` and `provider_failure_of(exc)`.** The
  first gives an adapter the verdict for a refused, reset, aborted or
  unreachable connection, an unresolvable name (`connection`) or a timeout
  (`timeout`), found on the exception or down its cause chain, so an HTTP
  client's own wrapper type is classified too; it returns `None` for anything
  else. The second reads the typed verdict back from any raised error, lifting
  an adapter's own verdict object onto `ProviderFailure`.
- **`protocore.conformance.assert_adapter_failure(exc)`** checks, in an
  adapter's own tests, that a raised error is an `LLMError` carrying a
  `ProviderFailure` whose `retryable` agrees with the error's flag.
- **Every terminal `message_stop` carries `outcome` and `cause`.**
  `outcome` is a `RunOutcome` (`completed`, `failed`, `stopped`,
  `wound_down`); `cause` is the JSON form of a `RunCause` (`kind`, `detail`,
  and `failure`, the `ProviderFailure` when a provider failure ended the run),
  or `None` for a run that simply completed. Both live in the new
  `protocore.contracts.outcome` and are re-exported from `protocore.contracts`.
  A host tells its own spend cap from a vendor's refusal by
  `cause.failure.reason == "budget"` instead of matching words in a detail.
  `soft_stop.enter` takes the verdict as `failure=`, and
  `soft_stop.FAILING_CAUSES` / `soft_stop.outcome_of` name which wind-down
  causes are failures.
- **`QueryEngine(tool_safety_policies=...)`** sets the safety policies the
  tool permission gate runs: `None` for the core's default stack, `[]` for
  none, a sequence for exactly those. The engine builds the dispatcher around
  them with the run's tool-error counter, the tool roles and the resilience
  classifier. A host that replaced the engine's dispatcher to install its own
  policies built it without those three, and lost the error count, role-aware
  transport-down detection and classification from every run. A re-arm keeps
  the policies. See `docs/extending.md`.
- **`QueryEngineConfig.tool_rules_session_id`** scopes the rules mark to a
  session instead of the tenant, derived under `tool_rules_mark_key` like the
  tenant's, so a host that runs every session under one tenant gives each its
  own mark, the same in every process that holds the key, and a fork names its
  source's session to keep its source's mark. `tool_rules_scope(tenant_id,
  session_id=None)` in `protocore.runtime.tool_deferral` is the scope the mark
  is derived from. Left unset, the mark is the tenant's, as before.
- **Seams for a host to shape the memory tools without copying them**
  (`protocore.contracts.memory`, `protocore.tools.memory`). `RememberTool`
  and `RecallTool` take `input_model=`, a subclass of `RememberInput` /
  `RecallInput` whose extra fields the model is shown; on `Remember` they are
  content-scanned like the text. `RememberTool(write_policy=...)` asks an
  `IMemoryWritePolicy` about every write after the scan and the scope rules,
  with a `MemoryWriteRequest` (resolved scope and key, `kind` and whether the
  model named it, the extra fields); its `MemoryWriteVerdict` refuses with a
  message the model reads, or amends kind, metadata, source refs and salience.
  `RecallTool.search_filters()` passes extra keywords to the store's `search`,
  `RecallTool.render_hit()` and `RememberTool.describe_write()` shape the lines
  the model reads, and `RecallTool.screen_recalled()` is the public scan of a
  recalled value. A host that subclassed the tools and imported their private
  scope helpers to keep the scope rules from drifting no longer has to.
  `build_memory_tools(write_policy=...)` threads a policy through. Every seam is
  off by default and the tools behave as before without them.
- **Memory provenance: `MemoryProvenance` on `MemoryRecord.provenance`.** Who
  wrote a record and from where — `source_kind`, `run_id`, `root_run_id`,
  `session_id`, `subagent_id`, `message_id`, `tool_call_id`, `observed_at` —
  with every field optional and an absent one meaning unknown. A store opts in
  by implementing `IProvenanceMemory`, an `IMemory` whose `write` takes
  `provenance=`; it is a separate contract because a keyword added to
  `IMemory.write` would fail every existing store's conformance suite. The
  reference `InMemoryMemory` implements it: set on CREATE, replaced on MERGE,
  untouched on SKIP, and a different provenance alone never turns a SKIP into
  a MERGE. `RememberTool(record_provenance=True)` (or
  `build_memory_tools(record_provenance=True)`) stamps each write with the
  run, root run, session, subagent and call id the loop vouches for — never
  anything from the model's arguments — and refuses at construction a store
  that cannot take the keyword. Off by default: a store written against
  `IMemory` alone is called exactly as before, and the field defaults to
  `None`, so stored records without it still load. The `Remember` result's
  metadata also names the stored `kind`, and `provenance` when the record has
  one.

### Changed

- **`ToolPermissionGate(policies=[])` installs no policy.** An empty list read
  as "use the defaults", so a host that passed one to turn the shell deny list
  off still had it. Only `None` means the defaults now.
- **A group's rules join the catalogue in the request after the load.** The
  catalogue took a group loaded mid-run only at the next run's first decision,
  so the system prompt changed one run after the tool list did and a session
  paid two misses on its whole context for one load. The request after a load
  already misses from the head of the prompt, so the catalogue is made again
  there, and the next run seeded with the same tools makes the same one.
- **`QueryEngineConfig.discovered_tools` takes the rows
  `ContextManager.discovered_tool_state()` writes**, as well as bare names.
  Rows bring the loaded tools back in the order they were loaded, with a tool
  loaded as part of a whole group still one entry with it; names come back
  before `loaded_tool_groups`, which swapped a group loaded before a single
  tool and moved the tool list.
- **A wind-down for `provider_error` or `model_no_progress` reports
  `outcome="failed"`,** with the cause's detail and the provider's verdict,
  even when the closing turn wrote prose; a wind-down for a bound the run was
  given reports `wound_down`. Nothing else about such a run changes: its
  `status`, `stop_reason`, `has_final_answer` and `soft_stop_cause` are what
  they were, so a consumer that wants the old reading keeps reading those and
  ignores the two new fields.
- **`LLMError(..., classified=verdict)` sets the error's `retryable` from the
  verdict** unless the raise passes `retryable=` as well, which still wins.
  The error's flag is what every reader sees first, and the provider-failure
  policy treats a `False` there as final, so a class default (`False` on a bare
  `LLMError`, `True` on `LLMProviderError`) must not contradict the verdict the
  error was raised with. Raising without `classified=` is unchanged.
- **`LLMError.classified` is a declared attribute typed
  `ProviderFailure | None`.** Attaching a verdict object of another type after
  construction still works at run time (the loop reads the fields by name), but
  a type checker now flags the assignment; attach a `ProviderFailure` instead,
  or keep the old object and silence the check at that line.
- **The snapshot schema is version 8.** It adds `run_cause`, the wind-down's
  cause with the provider's verdict, so a run resumed while winding down still
  reports what failed it. A version 7 payload is lifted with `run_cause: null`.
  A build that reads at most version 7 refuses a version 8 snapshot, as with
  every version bump: roll forward, not back.
- **The private reason sets are gone.** `query._PROVIDER_ADVANCE_REASONS`,
  `provider_failure._PERMANENT_FAILURE_REASONS` and
  `provider_failure._SURFACE_INDEPENDENT_REASONS` are replaced by the public
  sets in `protocore.contracts.llm`, and the idle watchdog attaches a
  `ProviderFailure(reason="timeout")` instead of a private verdict type. The
  sets compare by value, so a verdict whose reason is a plain string is still
  matched.
- **The resilience map knows the two new reasons:** `connection` is
  `timeout_rebuild` and `budget` is `deterministic_abort`.

- **`post_tool_use` now also fires for failed calls**, with two new payload
  fields on every call: `is_error` (`False` on success) and `error_kind` (the
  `DispatchErrorKind` value of the surfaced failure, `None` on success). A
  hook that should only see successes returns `ALLOW` when `is_error` is true.
  `success` is still sent. A `MODIFY` on a failure replaces the error text
  everywhere it goes, as on a success.
- **A `post_tool_use` hook that rewrites a result now replaces the tool's own
  value, not only the transcript text.** The canonical value used to be kept
  beside the projection whatever the hook did. After a `MODIFY` with a string
  `tool_output` the result carries no `canonical_content`, no `canonical_ref`
  the tool stored itself (the core cannot mask what it points at) and no
  `ui_payload` (the hook is not shown it); `path` stays. A host that relies on
  one of them surviving a rewrite has to mask it before the tool returns. A
  hook that allows the result changes nothing.

### Fixed

- **A queued steer or follow-up keeps its attachments.**
  `placed_as_user_message` joined only the items' texts, so a screenshot sent
  while a run was busy reached the model as its caption alone. Each item's
  image attachments (a `blob_ref` with an image `mime_type`) now ride on the
  placed turn under the new `IMAGE_REFS_METADATA_KEY`, every image in queue
  order, and `QUEUED_ITEMS_METADATA_KEY` records each placed item's `id`,
  `kind` and attachments, so a host can tell which items reached the model
  without reloading its queue. An item with an image and no words is placed
  rather than dropped. `message_image_refs` reads a message's images from its
  blocks and that key, and the prompt estimate now prices images carried in
  the key. See `docs/contracts.md`.
- **A stopped run hands back its queued items with their files.**
  `restore_queued_prompts` returned texts only, so a screenshot queued while a
  run was busy was lost when the run stopped, and an item with no caption was
  lost whole. The new `restore_queued_items` drains both queues and returns
  each item as a `QueuedPrompt` (`id`, `kind`, text, attachments), and the
  stop's `message_stop` event carries them as `restored_queue_items`.
  `restore_queued_prompts` keeps its `list[str]` return as the text-only view.
- **A failed tool call no longer bypasses the `post_tool_use` hook.** A tool
  that raised, timed out, returned something other than a `ToolResult`, was
  called with arguments that could not be serialised, was unknown, or was
  refused by the gate got its error text — the exception message included —
  written into the history, the `tool_result` event and the next request
  without the hook being asked, so a redaction hook missed a token in a failed
  command line or a key a server echoed back. Every such path now asks the
  hook once, after the consecutive-error cap, and a `MODIFY` lands in the
  outcome, the event and the transcript. The hook's rewrite also replaces the
  unmasked text in the result's replay metadata; the parallel path keys its
  error streak on the original words, held in memory only. A hook that raises
  on a failure is logged and ignored, as on a success.
- **The original of a result a `post_tool_use` hook rewrote no longer leaves
  the process (security).** The rewritten text went to the model, but the
  original stayed on the result block as `canonical_content`: in the history,
  every snapshot and the request handed to the adapter, and a result over the
  entry limit wrote the original to the blob store. A redaction hook masked
  the transcript and left the secret beside it. The hook's text is now the
  value everywhere, and the entry cap offloads that text.
- **The compaction ledger no longer clips the operator's words.** Each quote
  was cut at 1,200 characters without a marker. A quote is now kept whole, the
  other sections of the ledger give up their room before it, and a quote that
  alone outgrows the budget is cut with
  `[… N more characters of the operator's words — full text in blob <ref>]`,
  pointing at the copy the pass kept. Quotes left out for room are named on
  one line with the same pointer.
- **A request that asked for thinking may stay silent for the reasoning budget
  before its first token.** The idle watchdog gave a stream the 90 s baseline
  until its first delta, and the reasoning window only opened on a `thinking`
  delta. A provider that reasons without streaming the reasoning (Anthropic
  with the thinking display omitted, a Responses model without a summary)
  never sends one, so every think longer than the baseline was killed as an
  outage. Until the first delta that is not a `progress` heartbeat the budget
  is now `llm_stream_reasoning_idle_timeout_seconds` when the request carries
  `enable_thinking=True`; the idle error names the window as `first_token`.
- **The first stream silence of a streak is retried on the same provider
  before the provider chain moves.** One slow think demoted the run onto the
  next provider for the rest of the run. An `LLMStreamIdleError` now takes one
  in-place retry (from the `llm_transient_error_retry_max_attempts` budget)
  first; a second silence in a row moves the chain as before, and with that
  budget at 0 the chain moves at once.
- **Stale-result trimming no longer re-writes a long context to save a few
  pages.** A cache hit is found only at a breakpoint or a few blocks short of
  one, so with breakpoints on the system prompt and the last messages a trim
  further back loses the whole conversation; one measured trim saved 42,000
  characters of a 1.6-million-character view and was served nothing from the
  cache past the system prompt. The new `tool_result_stale_trim_min_share`
  (default `0.25`) requires a batch to save at least that share of the whole
  view. `0` restores the batch threshold alone.
- **Stale-result trimming no longer cuts a result into a longer one.** A
  result a few characters over the head was cut and given a pointer longer
  than what was removed. A result counts only for what the cut really saves,
  the pointer and the carried lines paid for.
- **A refused or reset connection moves the run to the next provider.** It
  used to reach the loop with no verdict, or with one the advance set did not
  name, so a dead endpoint ended the run even when a healthy fallback was
  configured. A verdict with `reason="connection"` now advances the chain; an
  error with no verdict still does not, on principle. An adapter that wants a
  connection failure to stay on the current provider attaches a reason outside
  `PROVIDER_ADVANCE_REASONS` (for instance `unknown`) or no verdict.
- **A spend cap the host enforces ends the run at once.** Reported as
  `reason="budget"`, it is permanent, never moves the chain and never winds
  down, because every rung and every narrowed request meets the same cap.
  Before, a reason the core did not know, with `retryable=False`, was wound
  down once the run had produced anything, and the closing turn was refused by
  the same cap.

## [2.0.0a25] - 2026-09-29

### Added

- `LoopConstants.compaction_keep_recent_max_ratio` (default 0.25): the share of
  the compaction trigger the kept tail may hold.
- `LoopConstants.llm_output_min_tokens` (default 4096): the answer room the
  compaction trigger keeps back.
- `TokenBudgets.configured_trigger_tokens` and `TokenBudgets.trigger_limited_by`,
  also carried by `compaction_started` and `compaction_completed`
  (`configured_trigger_tokens`, `trigger_limited_by`).

### Changed

- `compaction_trigger_turn_headroom_ratio` defaults to 0.10 (was 0.15): a fresh
  tool result is now cut to `tool_result_truncation_ratio` of the window where
  it enters, so one turn fits a tenth of it.

### Fixed

- **A tool result larger than the per-result limit is cut where it enters the
  run.** `tool_result_truncation_ratio` of the window was enforced only by
  compaction, on results outside the protected tail; the result a tool had just
  returned is always inside it, so a single oversized output reached the model
  whole, and one larger than the window ended the run with
  `llm_context_window_exceeded` even after the reactive pass. The whole value is
  now stored in the blob store when the result is recorded, the model is shown
  its head within the limit and a line naming the stored reference, and the
  reference becomes the result's `canonical_ref`. The result's metadata carries
  `protocore.tool_result_cut_on_entry` (`TOOL_RESULT_CUT_ON_ENTRY_METADATA_KEY`).
- **The kept tail is bounded in tokens, and a pass that cannot reach the
  trigger calls no summariser.** The routine keep window protected the last
  `compaction_keep_recent_turns` messages whatever they weighed, so recent
  messages carrying large tool results held the part of the prompt a pass
  cannot change above the trigger: every pass ended at the floor and the
  summariser was called on every iteration for nothing. The tail now keeps
  those messages only while they fit `compaction_keep_recent_max_ratio` (new,
  0.25) of the trigger — never fewer than `compaction_force_keep_recent_turns`,
  and never the unread batch of results or the message that asked for it. When
  the fixed part is still at or over the trigger, a routine pass runs only the
  model-free tiers; a forced or reactive pass keeps its model tiers.
- **The configured `compaction_trigger_ratio` is the trigger in force, and a
  clamp is visible.** The trigger was the lower of the ratio and a ceiling that
  kept back the whole configured output cap (`llm_output_max_tokens_ratio` of
  the window) although every request is fitted to the window, so on a 65k window
  a ratio of 0.8 became 0.57 without a word anywhere. The ceiling now keeps back
  `min(that cap, llm_output_min_tokens)`; on a 65k window the default 0.8 is the
  trigger. A ratio the ceiling still clamps is reported by
  `configured_trigger_tokens` and `trigger_limited_by` on the budgets and on
  every compaction event.

## [2.0.0a24] - 2026-09-28

### Added

- **`Message.origin` tells compaction's records from the conversation.** The
  ledger, the turn and fold summaries and the floor digest are user-role
  messages, because a provider accepts no system message past the first, and a
  host that persisted history by role stored them as the user's own turns and
  seeded them into the next run as such. Every message now answers
  `Message.origin` with `MessageOrigin.conversation` or
  `MessageOrigin.compaction`; the value is derived from the tags compaction
  already sets (`protocore.compaction_summary` and `protocore.compaction_ledger`),
  is serialised with the message in snapshots, and is never sent to a provider.
  It is derived and cannot be set: `Message(origin=...)` is accepted and
  ignored, and a host marks a message as compaction's by setting the tag. The
  request-only checkpoint summary is tagged too. See `docs/compaction.md`,
  invariant 13.
- **`COMPACTION_SUMMARY_METADATA_KEY` and `COMPACTION_LEDGER_METADATA_KEY` are
  exported** from `protocore.contracts` (and `protocore.contracts.types`), for
  a host that tells compaction's records apart or tags a message itself.
- **`group_load_overflow(rc, advertised, adding)`** in
  `protocore.runtime.tool_deferral` says whether loading a group whole beside
  a surface would go over `max_advertised_tools` (`"count"`) or the
  tool-definition budget (`"tokens"`).
- **`removable_indices`** in `protocore.runtime.context.compaction` names the
  history entries a compaction pass may remove, and `compaction_completed`
  carries `fixed_tokens`, the part of the prompt no tier can remove.
- **`TOOL_GROUP_INSTRUCTIONS_MAX_CHARS`** (8 000) in
  `protocore.contracts.tool_registry`, the longest a group's rules may be.
- **`QueryEngineConfig.tool_rules_mark_key`**, the host's secret the rules mark
  is derived with (see Fixed).
- **`TOOLS_LOADED_THIS_STEP_METADATA_KEY`** (`protocore.tools_loaded_this_step`),
  stamped beside `ADVERTISED_TOOLS_METADATA_KEY`: the tools loaded by earlier
  calls of the same model message, which the next request carries.

### Changed

- **`max_tool_calls_per_turn` defaults to 24** (was 64). It bounds the calls
  dispatched from one model message; 24 still fits a genuine fan-out, such as
  reading a couple of dozen files at once, while 64 let a single runaway
  message come close to the per-run tool-call soft caps
  (`subagent_tool_call_soft_cap` is 40) before any call was refused.
- **Breaking: `declare_group` raises `ValueError` for rules over
  `TOOL_GROUP_INSTRUCTIONS_MAX_CHARS`** (8 000 characters) at declaration.
  Rules of a group that cannot be held back go into every request, so their
  length is bounded where they are declared. A host that takes a group's
  instructions from operators should validate or shorten them before
  declaring the group.
- **`tool_rules_mark(session_id)` is now `tool_rules_mark(scope, key="")`**,
  and the engine passes the tenant id as the scope, so the mark is the same
  for every session of a tenant (see Fixed).
- **`ToolSearchTool.is_concurrent_safe` is `False`.** Several `ToolSearch`
  calls in one model message are dispatched one at a time, so each is stamped
  with what the ones before it loaded and which rules they gave. A message
  such as `[Read, Read, ToolSearch, Read]` is dispatched in three batches.
- **`Message.model_dump()` carries an `"origin"` key on every message** (see
  Added). A host that stores dumps under a strict schema, or compares them
  byte for byte, sees the new key; it is ignored when a message is read back.

### Fixed

- **A compaction pass no longer drains the history when the fixed part of the
  prompt sits above its target.** The target was `compaction_target_ratio ×
  min(T, P)` on the whole prompt; when the system prompt, the tool definitions
  and the protected history (the task, the ledger, the kept tail, seeded turns)
  alone were above it, it was unreachable, and every tier and the floor ran to
  exhaustion, leaving the task and the last messages. The target is now
  `F + compaction_target_ratio × (min(T, P) − F)`, where `F` is what no tier
  can remove; with nothing fixed it is unchanged. `compaction_completed`
  carries `fixed_tokens`, and `removable_indices` names what a pass may remove.
- **A compaction record no longer starts a new round of the run.** The ledger,
  summaries and floor digest are user-role, and the round boundary took them
  for a caller's message. After a reactive pass put the ledger before the
  runtime's continue prompt, the run read as having produced nothing: a
  provider failure then failed it instead of winding it down, and an answer
  written before the ledger was no longer preserved. They are now skipped as
  boundaries, and a summary inside the round counts as the round's output.
- **A permanent provider refusal the tool surface could have caused is wound
  down again.** Since 2.0.0a22 every refusal the adapter classified as final
  skipped the wind-down, on the premise that its request only adds one message.
  It also narrows the tool surface to the finalizing tool, so a rejected tool
  schema or a request made too large by the surface is served there and the run
  reports what it found. Only refusals the surface cannot cause (`auth`,
  `auth_permanent`, `billing`, `model_not_found`,
  `oauth_long_context_beta_forbidden`, `provider_policy_blocked`,
  `thinking_signature`, `image_too_large`) still skip it.
- **A settled wound-down run resumed from its snapshot no longer gets the
  wind-down notice back.** `resume_from_snapshot` restored the notice for every
  snapshot whose wind-down state was armed, a finished run's included, and
  `rearm` kept it, so the next turn opened with "your tools are gone" although
  every tool was back. The notice is now restored only for a run that is not
  terminal, and `rearm` removes any notice left in history.
- **Russian imperatives reach the tools their infinitives reach.** The bundled
  lexicon lists verbs as infinitives, and the stemmer keeps "отредактируй",
  "найди" and "скопируй" apart from "редактировать", "найти" and
  "копировать", so the most common form of a request expanded to nothing.
  `Lexicon.from_translations` now also registers each infinitive under the
  stems of its imperatives, and `Lexicon.expand` looks a stem it does not know
  up once more without a perfective prefix. The bundled lexicon gains
  "искать"/"поискать", "compare" and "translate".
- **A long Russian message keeps the words that name the tool.** Past 25
  distinct terms a query keeps the rarest, and rarity was the term's own idf in
  the catalogue — 0 for every Russian word against English tools, so the 25
  alphabetically first stems survived and a request at the end of a long
  message could be cut. A term is now worth the larger of its own idf and its
  rarest lexicon expansion's idf at `tool_retrieval_lexicon_weight`, and terms
  that can add nothing to a score are dropped before the cap.
- **`compute_effective_surface(top_k=0)` no longer clips to the pinned tools.**
  `tool_retrieval_top_k` documents `0` as "no clip" and defaults to it, but the
  registry read `0` as "retrieve no tool", so a host passing the constant
  straight through advertised the pinned tools alone. `ToolRegistry` and
  `InMemoryToolRegistry` now read `0` like `None`, and a host that passes the
  constant straight through gets the same unclipped surface the loop
  advertises.
- **The bare name of a namespaced tool ranks that tool first.** `get_issue`
  ranked `mcp__github__add_issue_comment` above `mcp__github__get_issue`: the
  index kept an identifier's parts and its whole joined form, "get" is a
  stopword, and what was left of the query was "issue". The analyser now also
  keeps the joined form of every tail of an identifier's parts (`getissue`,
  `githubgetissue`), on both the catalogue and the query side.
- **Building a retrieval index no longer repeats work the process has done.**
  `Lexicon.bundled()` read and built the 2000-entry lexicon on every call, so
  every new registry paid for it, and every registry re-tokenised and
  re-stemmed each tool; a host that builds a registry per request paid about
  a quarter of a second per request at 500 tools. The bundled lexicon is now
  built once per process and is read-only; each tool's analysed fields are
  cached by `ToolDocument` and each word's stem by word (both bounded); plain
  words skip identifier splitting; and the fallback matcher is built only when
  a query needs it. At 500 tools a rebuild over known tools takes about 15 ms
  and a first build about 90 ms, against about 250 ms before.
- **A loaded tool is admitted like any other tool.** Loaded names were folded
  into `pinned` unconditionally, and `pinned` is admitted past a `visible`
  whitelist, so a tool seeded through `QueryEngineConfig.discovered_tools`,
  restored from a snapshot, or loaded before the host narrowed its policy was
  advertised and callable although the whitelist excluded it. Only names the
  host's policy admits on its own are pinned now, checked on every request, and
  the surface appends a loaded tool only when dispatch would admit it — a
  child's declared tool set included.
- **`ToolSearch` stays advertised while the per-message clip leaves tools
  off.** It was hidden whenever no group was held back, so with
  `tool_retrieval_top_k` on the tools the clip left off could not be found.
  The discovery tool is hidden only when every admitted tool is on the
  surface; a tool outside a child's declared tool set does not count, since
  the search could never load it.
- **The catalogue names no group a child's declared tool set cannot reach.**
  The deferral was planned from the policy's surface alone, so a child run with
  a `subagent_tool_allowlist` was shown groups every load of which came back
  empty and every call of which the gate refused. The would-be surface is now
  narrowed to the declared set first, as dispatch narrows it.
- **The deferral decision is made again when the execution profile changes.**
  It was keyed on the host's policy but not on the profile, which reshapes the
  same surface: planned under a plan profile that admitted none of a server's
  tools, it held nothing back, and when the profile ended mid-run the whole
  server landed on the surface instead of in the catalogue. The profile and
  the child's declared tool set are part of the key now.
- **The rules mark can be stable across processes and sessions.** It was an
  HMAC of the session id under a key drawn anew in every process, so it sat in
  the system prompt as a value that changed with every worker, every restart
  and every session, and the prefix cache missed from the catalogue on (and on
  every tool definition, with a template that renders tools after the system
  text). It is now an HMAC of the tenant id. A host that wants the prompt
  stable across workers and restarts sets `QueryEngineConfig.tool_rules_mark_key`,
  a secret it keeps the same in every worker; the mark then lasts for the
  tenant until the key is rotated. Without a key it is derived under a random
  key drawn once per process, so it cannot be computed from the tenant id.
  The key is left out of the config's `repr`, and the mark is handed to the
  discovery tool alone rather than to every tool's metadata.
- **A group's rules count against the tool budget, and their length is
  capped.** The rules of a group on the surface go into the system prompt on
  every request but were never measured, so a group whose definitions fit
  stayed on the surface with rules of any length, and a run could fail on the
  context window before its first request. The deferral now counts
  `group_rules_text` with the group's definitions against
  `tool_definitions_ratio`, and their length is capped at declaration (see
  Changed).
- **A group is loaded whole only while the whole of it fits.** A blind call
  held for its group's rules, and `ToolSearch(group=...)`, loaded every tool
  of the group with no check against `max_advertised_tools` or the
  tool-definition budget, so one blind call of a large server's tool put the
  next request's definitions far over the budget the deferral had enforced.
  Both are judged by `group_load_overflow` now: a held call loads the called
  tool alone and says how many more there are and how to load them, and
  `ToolSearch` lists a group that would not fit instead of loading it. Each
  group is judged beside what the next request will carry: the groups taken
  earlier in the same call, and the tools loaded by earlier calls of the same
  model message, which the loop stamps as `TOOLS_LOADED_THIS_STEP_METADATA_KEY`.
- **A loaded tail over `max_advertised_tools` is trimmed by whole entries.**
  It was trimmed one tool at a time by recency, so a group loaded whole was
  left half on the list while the model had been told all of it was loaded.
  A group now leaves the request whole, as it leaves the cap, and a smaller
  older entry may take its place.
- **A group's rules are given once even when one message loads it twice.**
  `ToolSearch` was concurrent-safe, so two calls in one message were both
  stamped with the groups given before the message ran and both carried the
  full rules. It is dispatched one call at a time now (see Changed); each
  result is folded in before the next call is stamped, and a blind call beside a search of its
  group is pointed at the search's rules instead of repeating them.

## [2.0.0a23] - 2026-09-27

### Changed

- **Tool retrieval is BM25F over five fields, with stemming and Russian query
  expansion.** A tool is indexed as its name, `search_hint`, the first sentence
  of its description, the rest of it, and its parameter names and
  descriptions, each weighted (`tool_retrieval_name_weight`,
  `tool_retrieval_search_hint_weight`, `tool_retrieval_summary_weight`,
  `tool_retrieval_description_weight`, `tool_retrieval_parameters_weight`,
  `tool_retrieval_bm25_k1`, `tool_retrieval_bm25_b`). Identifiers are split
  (`BrowserOpen` matches "browser open"), `ё` is folded, conversational
  fillers are stopwords, and both sides are stemmed with the Snowball Russian
  and Porter English stemmers. A Russian query is expanded through a bundled
  Russian-to-English lexicon (`tool_retrieval_lexicon_weight`), which is what
  lets it find tools described only in English. On a labelled set of 656
  English and Russian queries against about 700 tools, Recall@5 rose from 0.41
  to 0.67, and to 0.78 with a Russian `search_hint` on every host tool. See
  `docs/tools.md`.
- **`ToolRegistry.search` returns tools in rank order**, best first, ties by
  name. It used to re-sort its result by name, which hid which hit fit best.
  The advertised surface from `compute_effective_surface` stays in name order.
- **The index is built once per catalogue version** and cached on the registry
  instance, instead of re-tokenising every tool on every query: about 0.3 ms per
  query at 700 tools, down from 15–20 ms.
- **The per-turn clip uses the normalized fallback too** when nothing scores,
  as `search` already did. The fallback no longer counts one- and two-letter
  words and stopwords of a description, which made the article "a" match any
  query token containing the letter.
- `IToolRegistry.search` and `compute_effective_surface` take an optional
  `retrieval: RetrievalSettings`; the loop passes
  `RetrievalSettings.from_constants(rc)`.
- **The per-message clip is off by default, and pinned tools no longer count
  against it.** `tool_retrieval_top_k` defaults to `0` (off) and now counts only
  the tools retrieval adds; counted with the pins, a floor of fourteen under the
  old default of twelve left the clip no room to retrieve anything. Clipping per
  user message is kept as an opt-in and documented as not recommended.
- **`tool_definitions_ratio` is enforced** and defaults to `0.25`. It was
  derived into a budget that nothing held the surface to.
- **Discovered tools are appended to the surface in discovery order** instead of
  being sorted into it, so loading a tool changes only the end of the tool list.
  The context manager's pin LRU (`pin_tool`, `pinned_tool_names`) is replaced by
  `discover_tool`, `note_tool_used`, `discovered_tool_names` and
  `evict_discovered_tools`; `pinned_tool_max_count` is applied, least recently
  used first, only after compaction, at a turn boundary and when a run starts.
- **The snapshot schema is version 7.** `context_manager_pinned_tools` becomes
  `discovered_tools` (name and last use, discovery order) beside the new
  `deferred_tool_groups`; the upcaster lifts a version 6 payload.
- **`tool_surface_advertised`** carries `discovered_tool_names` (discovery
  order) in place of `toolsearch_pinned_tool_names`, the held-back groups
  (`deferred_tool_groups`, `deferred_tool_count`, `tool_deferral_reasons`), and
  the source `discovered` in place of `toolsearch_pin`.
- **An unknown tool name is answered with up to three near names** the policy
  admits: `unknown tool: 'X'. Did you mean: A, B, C?`
- An engine built without a host run state now carries its own constants on
  `run_state.rc`, so core tools read the run's tunables.
- **A description's first sentence no longer ends at an abbreviation** such as
  "e.g.", "i.e.", "vs.", "т.е." or "напр.", nor at a full stop inside
  brackets; "etc." and "т.д." end it only before a capital. `ToolSearch` shows
  that sentence as a tool's line, and it used to read "Transition an issue to a
  new status (e.g.".

### Added

- **Tool groups and deferral.** `ToolGroup`, `IToolRegistry.declare_group`,
  `undeclare_group` (for a group whose tools are gone, such as a removed MCP
  server's) and `tool_groups`; a tool joins a group by its `tool_group` class attribute or a
  declared name prefix. With `tool_deferral_mode = "auto"` (the default, inert
  until a group is declared and a discovery tool registered) dynamic groups are
  held back, and other groups largest first while the surface is over
  `tool_definitions_ratio` of the window or over `max_advertised_tools`.
  Held-back groups are named in a byte-stable catalogue block in the system
  prompt, with exact tool names or an exact prefix and count
  (`tool_catalogue_max_listed_names`). The decision is made again when the
  catalogue, the groups or the host's visibility policy change, so tools a
  host switches on mid-run are held back like the rest, with the decision in
  force as its floor, so the catalogue does not flip mid-run. A decision a
  snapshot carries is a floor too: a dynamic group it does not name is still
  held back and the limits still apply. Without a discovery tool, groups are
  held back only over `max_advertised_tools`, and the catalogue says to call
  by exact name. See `docs/tools.md`.
- **`ToolSearch`** (`protocore.tools.ToolSearchTool`): free-text search with up
  to `tool_search_max_results` hits in rank order, one line each, loading the
  first `tool_search_autoload_count`; `select:Name1,Name2` (or a `select`
  argument, a list or a comma-separated string) loads exact names and answers
  an unknown one with the nearest admitted names. A tool already in the tool
  list is reported as such, never as loaded, and the description says the tool
  loads tools only — skills are not tools. It respects the live visibility
  policy and a child run's declared tool set, which the dispatcher stamps on
  every call (`TOOL_VISIBILITY_POLICY_METADATA_KEY`, now
  `protocore.tool_visibility_policy`, and `TOOL_ALLOWLIST_METADATA_KEY`); with
  no policy it admits nothing. It is advertised only while something is held
  back.
- `ADVERTISED_TOOLS_METADATA_KEY`: the loop stamps the names the calling
  request advertised on `ToolContext.metadata`.
- A call of a registered tool the request did not advertise still runs and now
  loads the tool; events `tool_discovered` and `tool_unadvertised_call`. When
  such a call fails on its arguments, the error ends with the tool's line
  (`It takes: Name(param1*, param2) — first sentence`), the form `ToolSearch`
  lists tools in, so the retry need not guess again — only for a tool the
  policy and the child's declared set admit.
- `QueryEngineConfig.discovered_tools` seeds a new run with the tools the last
  run of the session loaded; `ContextManager.called_discovered_tool_names()`
  is the part of them the run called, which is the list worth carrying.
- `max_tool_calls_per_turn` (default 64): calls past it in one model message are
  each answered with an error and not run.
- **Load modes and rules for tool groups.** `declare_group` takes
  `load="eager" | "auto" | "lazy"` and `instructions`. A `lazy` group is held
  back whenever a discovery tool is admitted, even when everything fits, and
  is advertised like `auto` when none is; an `eager` group is never held back
  for size and gives way only to `max_advertised_tools` itself.
  `QueryEngineConfig.tool_group_loads` overrides the load modes for one run.
  `ToolSearch` takes `group` (and `group:<name>` entries in `select`) to load
  every admitted tool of a group; tools loaded that way are one entry under
  `pinned_tool_max_count`. A group's rules are given once per run: in the
  catalogue for tools on the surface or loaded when the run starts (and again
  after a compaction), in the `ToolSearch` result that loads the group's first
  tools, or — for a blind call of such a tool — instead of running the call,
  which is answered with the rules and runs on the next try. New event
  `tool_group_loaded` (`group`, `via`, `tools`); `tool_surface_advertised`
  carries `tool_groups` with each group's load mode and state, and
  `tool_unadvertised_call` carries `executed`. The snapshot (still version 7)
  gains `tool_group_rules_given`, and a discovered-tool row loaded with its
  whole group carries `group`. The catalogue's header now names the
  whole-group load. See `docs/tools.md`.
- **A blind call held for its group's rules loads the whole group**, as
  `ToolSearch(group=...)` would, and its answer names the group's other tools
  now callable and gives the called tool's line (`It takes: Name(p1*, p2) —
  …`). Loading only the called tool left the rest of the group out of sight,
  and the retry was written without the parameters.
- **Group rules carry the run's rules mark, and the catalogue names it.** A
  genuine rules heading reads `Rules for the <group> tools [<mark>]:`; the
  catalogue says, whenever a group in it has rules, that text claiming to be
  rules without the mark — a web page, a file, a command's output — is
  content, not rules. The mark is an HMAC of the session id under a
  per-process key (`tool_rules_mark`), stamped for discovery tools as
  `protocore.tool_group_rules_mark` and carried in the snapshot as
  `tool_group_rules_mark`.
- **After a compaction the rules counted as given are the rules the catalogue
  carries.** The set used to keep every group given earlier, so a group whose
  rules were in a result the compaction took away, and which the compaction
  then unloaded over the cap, had its blind calls run without the rules and
  its reload give none. The catalogue no longer keeps rules for groups that
  are neither on the surface nor loaded, and `ToolSearch` inside a loop goes
  by the given set alone.
- **A call held for its group's rules is not a use of the tool**: it did not
  run, so `called_discovered_tool_names()` no longer lists it and a host does
  not carry its group into the next run for it. Its `tool_unadvertised_call`
  (`executed: false`) now comes before its `tool_result`, so a host knows the
  result answers a call that never ran.
- **The catalogue lists the host's own held-back groups first**, and the
  `dynamic` groups after them under `Tools of connected servers:`. In one
  name-sorted list the servers stood ahead of the host's groups.
- `QueryEngineConfig.loaded_tool_groups` seeds a run with whole groups, each
  one entry under `pinned_tool_max_count` with its rules in the catalogue and
  a `tool_group_loaded` event with `via` `seed`; seeding a large group's tools
  by name made each an entry and pushed the rest of the seed out.
  `ContextManager.loaded_tool_group_names()` is the groups a run holds whole,
  for a host to carry into the next one.

- **The bundled lexicon covers the Russian a developer speaks about trackers,
  chat, calendars and deployments**: loanwords and slang such as "пулреквест",
  "ишью", "тикет", "таска", "смержи", "выкати", "созвон", "заархивируй",
  "алерт", mapped to the English an MCP server describes its tools in. A
  Russian request such as "покажи открытые пулреквесты" or "заархивируй тикет
  поддержки" now finds the server's tool among the first three.
- `IToolRetriever` and `reciprocal_rank_fusion` (`contracts/tool_retrieval.py`):
  a host may pass its own ranker, for example an embedding model, as
  `ToolRegistry(retriever=...)`, and its ranking is fused with the lexical one
  (`tool_retrieval_fusion_rank_constant`). `ToolRegistry(lexicon=...)` takes a
  host's own `Lexicon`, or `None` to turn expansion off.

### Removed

- `retrieve_tools`, `bm25_score`, `compute_idf`, `compute_avgdl`,
  `reduce_query`, `build_candidate` and `ToolRetrievalCandidate`: replaced by
  `ToolIndex`, `AnalyzedCatalogue` and `ToolDocument`.
  `normalized_fallback_match` now takes `ToolDocument`s and returns names.

## [2.0.0a22] - 2026-09-26

### Changed

- **Compaction ends below its trigger or at a floor, whatever the summariser
  does.** A pass now masks old tool outputs, summarises the oldest spans,
  folds old summaries, and — when those stop short — removes the oldest spans
  without a model, until the whole prompt is at `compaction_target_ratio` of
  the trigger or nothing removable is left. A summariser that fails, hangs,
  returns nothing or returns something unparseable is a normal outcome, and a
  pass that could change nothing is not opened. See `docs/compaction.md`.
- **A summary is plain text under five fixed headings**, requested with
  `complete_text` under a system-role instruction, read tolerantly (a JSON
  envelope, an unterminated one, a reply without headings and a reply the
  output cap cut are all kept) and clamped section by section at line
  boundaries. The single JSON `summary` string and its 1,024-character ceiling
  are gone.
- **Exact values are carried by code.** Every tier records what it takes out
  of the window — operator instructions, files touched, identifiers of
  recognisable shape with their line, failed calls, the latest plan — in one
  ledger message, rebuilt by code on every pass and never shown to the
  summariser.
- **Old tool outputs are masked by age** (`compaction_mask_keep_recent_results`,
  `compaction_mask_min_tokens`), keeping the lines the output said only once
  (`compaction_mask_distinct_lines`) and a readable pointer to the stored
  original. The summariser is shown a masked output in full.
- **Adjacent units are summarised as one span** up to
  `compaction_summary_group_max_tokens`, whatever their size, and a span's
  output budget is `compaction_summary_ratio` of it, clamped by
  `compaction_summary_min_output_tokens` and `compaction_summary_max_output_tokens`
  (now 2,048). The summariser's input is bounded
  (`compaction_summariser_input_max_tokens`, and a quarter of the window) and
  each call has a deadline (`compaction_summary_timeout_seconds`).
- **`compaction_completed` reports every tier and the pass's `outcome`**, and
  the whole prompt before and after; `compaction_fold_min_messages` is now 4.

### Removed

- `compaction_routine_min_clear_ratio` (replaced by `compaction_target_ratio`),
  `compaction_summary_string_max_chars`, `compaction_summary_envelope_tokens`,
  `compaction_summary_chars_per_word`, `compaction_summary_output_tokens_per_word`,
  `compaction_summary_tokens_per_word`, `compaction_summary_min_words` and
  `compaction_fold_summary_target_words`: they sized a JSON string in words.

### Fixed

- **A compaction the `pre_compact` hook refuses leaves the run running.** The
  refusal returned from inside `COMPACTING`, and the run stayed in that state
  for the rest of the turn. It now returns to `RUNNING` with the reason
  `compaction_refused_by_hook`.
- **The compaction gate sizes the whole prompt.** The trigger and the
  emergency cliff are whole-prompt sizes, but the gate held the history's
  estimate against them. With a large system prompt and tool surface — 56k of
  a 256k window in the case that showed it — the history reached the trigger
  only after the request had passed the provider's ceiling, so the fit clipped
  the output cap turn after turn and compaction first ran on a refusal. The
  gate now adds what the last request carried besides the history (its system
  messages and tool definitions, calibrated); the figure rides the snapshot.
- **A history made only of short rounds can be compacted.** Tier 2 skipped
  every unit below `compaction_summary_min_unit_tokens`, and the fold takes only
  summaries and operator turns, so a run that works in many rounds of one short
  tool call each built a history no tier could reduce: every pass freed
  nothing and the run failed on the retry budget. Adjacent small units of the
  same provenance, with nothing between them, are now joined up to
  `compaction_summary_group_max_tokens` (6,000; 0 disables it) and summarised
  as one unit when the group clears the floor.
- **A failed round never completes on an earlier run's answer.** Completing a
  failed run "on its preserved answer" read every message not tagged as
  seeded, so on a host that hands the engine a session's earlier turns
  untagged, the previous run's reply counted: a run whose every request was
  refused completed with nothing written and no error reported. The answer
  must now have been written in the round that failed, the span after the
  last message a caller put in.
- **A refusal the adapter classified as final is neither retried nor wound
  down.** The retry decision read only the `reason` of an attached
  classification and ignored its `retryable` flag, although `ClassifiedLike`
  carries both; the flag now decides when it is set. And a permanent
  `LLMProviderError` no longer enters the wind-down: its one turn is a request
  to the endpoint that has just refused the run, and it was refused the same
  way. The fallback chain is still tried first.
- **The wind-down notice names a deadline and a stalled model for what they
  are.** A run stopped by its wall clock was told it had reached its budget,
  and a run whose model kept returning reasoning with no answer and no tool
  call was told the model endpoint had failed — which the model then passed on
  to the operator as the reason. `soft_stop_notice_text_deadline` names the
  time limit, and the new `model_no_progress` cause, entered by the
  empty-round policy, carries `soft_stop_notice_text_model_no_progress`. Blank
  texts fall back to the general notice.
- **The wind-down notice does not outlive its run.** The "tools are withdrawn,
  write your answer" message the loop writes into history when a run winds
  down stayed there after the run ended. When the wind-down's own final turn
  failed — a provider outage that caused the wind-down also killed the answer —
  the notice was the last thing in the session, and the next turn obeyed it:
  the model reported what it had not finished and called nothing, although
  every tool was back. The notice is removed once the drive reaches a terminal
  state and written back only for a resumed run that is still wound down.

## [2.0.0a21] - 2026-09-25

### Added

- **`TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY`** (`protocore.contracts.types`).
  A terminal tool, or the host check behind it, sets this on an error result
  when the call was refused because work is missing, for example a declared
  file that does not exist. See the forcing below for what it changes.
- **`LLMRequest.extra["tool_choice_required"]`**: `True` asks for some tool
  call and no prose. The `LLMRequest.extra` contract now documents it next to
  `forced_tool_choice`.

### Changed

- **A delivered answer's terminal call is forced, not requested.** This
  applies to a run with `expected_terminal_tool` and
  `terminal_tool_nudge_enabled` that ends a turn with a substantive visible
  answer and no terminal-tool result.
  - Every request after that names the terminal tool in
    `extra["forced_tool_choice"]` and appends nothing to the transcript. The
    free-form `terminal_tool_nudge` text is no longer sent on this path.
  - A terminal-tool error is answered by forcing the tool again by name. The
    one exception is the run's first refusal marked with
    `TERMINAL_REFUSAL_NEEDS_WORK_METADATA_KEY`: the request after it carries
    `extra["tool_choice_required"] = True` instead, so the model can do the
    missing work.
  - A forced turn's text is kept off the live stream and out of history. Calls
    that the forced mode does not admit are withheld from the live stream and
    dropped before they are recorded or run, so a provider that ignores the
    forced choice cannot start new work under the answer.
  - A reasoning-only or finish-less forced round is retried instead of
    receiving `continue_prompt_text`, the reasoning length-cut note or the
    output-cap resume prompt. Before this, a thinking model answered the nudge
    with reasoning only, was told "Please continue.", and started a second
    answer, often to an earlier question in the session.

  **When the forcing steps aside.** The model's text is shown again when it has
  to act rather than seal:
  - a gate refuses the call with a corrective (the prose gate, a host
    pre-dispatch check);
  - a background result, or a user message placed at a tool boundary, arrives;
  - the marked refusal's required call is answered with other work.

  The answer the model ends on is then forced again. A gate's corrective closes
  the answer window, so the refused prose no longer counts as the answer.

  **Budget.** Forced requests and step-asides share
  `terminal_tool_forced_max_attempts` (default 3). The run completes on the
  answer it delivered when the budget is spent (`state_changed` reason
  `terminal_tool_forced_exhausted`), or at once when the terminal tool is not
  registered or the consecutive-error circuit breaker has stopped it
  (`terminal_tool_unavailable`). That completion is a hard stop: it does not
  pass the answer floor, the voluntary artifact seal or follow-up placement.

  **Thinking.** Forced requests go out with thinking off unless
  `terminal_tool_forced_thinking_enabled` is set. Measured on a self-hosted
  thinking model behind vLLM:
  - a named `tool_choice` with thinking on spent a 2,048-token budget reasoning
    in most attempts and returned no call;
  - with thinking off it returned exactly the call every time;
  - `tool_choice="required"` after a refused call returned a tool call in 10 of
    10 samples, the requested write in 8 of them.

  **Other effects.**
  - The terminal tool is pinned to the surface while its call is owed.
  - A continuation run (`run(None)`, including `resume`) keeps a forcing that
    was in progress, with its terminal-only latch.
  - A turn that ends with no answer yet still gets the one-time
    `terminal_tool_nudge` text; the answer it then writes is forced.
  - Prose that only announces work ("now let me write the report") counts as an
    answer and is sealed. A host check on the declared deliverables can refuse
    it with the needs-work marker. A host whose models tend to announce and
    stop can instead set the new `terminal_tool_nudge_write_first_before_forcing`,
    which sends the one-time write-first nudge before sealing while the run has
    written no file.

  **Policy constructors.** A host that assembles its own turn-policy set
  must pass the new arguments. `TerminalNudgePolicy` now requires
  `forced=ForcedTerminalCall(...)`, and `EmptyModelTurnPolicy` requires
  `forcing_terminal_call=`. A host that uses the core's policy set is
  unaffected.

  **Host adapters** must render `forced_tool_choice` as the native single-tool
  choice and `tool_choice_required` as `tool_choice="required"`. When a request
  carries `enable_thinking=False`, they must send thinking off explicitly.

### Fixed

- **A failed terminal call no longer hides the answer before it.** A
  terminal-tool result is recognised through the call it answers, not only by a
  name or terminal flag on the result, so an error result from the terminal
  tool is no longer counted as work. Before, one failed call made the answer
  that preceded it read as progress narration, and the prose gate then refused
  the retried call and asked the model to write its answer again.

## [2.0.0a20]

### Changed

- **`LLMRequest.temperature` is unset unless the caller states one.** The
  contract defaulted it to `0.7`, and `build_llm_request` filled that default
  into every request, so the action stream and the deep loop's plan calls
  always carried a temperature nobody had chosen, and an adapter could not tell
  "no opinion" from "0.7". The field is now `float | None = None`; the
  builder passes `None` through, and `RequestManifest.temperature` records it
  as such. Paths that need a value — the compaction summariser, with
  `compaction_summary_temperature` — still state it. A host adapter should omit
  the wire field when it is `None` (or apply its own per-model setting); an
  adapter that serialises it as-is now sends `null`. The temperature is part of
  `request_digest`, so a request recorded before this change does not
  replay-match the same request rebuilt after it.

- **The wind-down notice asks for the answer, not the work log.** The default
  `soft_stop_notice_text` told the model to report what it did and what it
  found, which invited a narration of the run's steps into the user-facing
  reply — something many hosts' personas forbid. It now asks for the best answer
  from what the run already has and where any results are, says plainly what
  could not be established or finished, and tells the model not to describe its
  steps. Both languages changed; the text is no longer than before. A host that
  seeded the old default into its own constants store keeps the old wording
  until that row is updated.

### Fixed

- **The compaction retry budget charges failed passes, not passes.** Only a
  pass that tried and failed is charged, and only once however many tiers
  failed; a tier that raised is no longer counted again by the no-progress
  rule. The reactive pass after a provider rejection keeps its own count,
  `CompactionState.reactive_retry_count` (same `compaction_failed_max_retries`
  bound, rides the run snapshot), so proactive failures can no longer use up
  the one profile that may compact seeded history. Before, an idle proactive
  pass plus two reactive passes that lost their summariser calls exhausted the
  default budget. Progress by either profile clears both counts, and so does
  `rearm()`. A routine pass that rewrote a Tier 1 block now counts as progress
  even when the estimate did not move.
- **A proactive pass with nothing to do is not opened.** The routine,
  turn-start and per-iteration gates ask `ContextManager.has_proactive_work`
  first. When no tier would change anything under the proactive profile — a
  history of turns seeded from earlier runs is the usual case — there is no
  `COMPACTING` state, no compaction events, hooks, usage row or snapshot. The
  engine then skips the gate without asking again until the history or the
  constants change.
  Before, such a pass ran on every iteration once the estimate crossed the
  gate. `Tier2Result.units_attempted` and `Tier3Result.spans_attempted` report
  the calls a pass made. `tier1_has_work`, `tier2_has_work` and
  `tier3_has_work` answer the same question for each tier. A fold that raises
  now reports a zero `Tier3Result` instead of `None`.
- **Proactive compaction running out of budget no longer fails the run.**
  Nothing has been rejected at that point. The proactive summariser tiers are
  suspended (a `compaction_exhausted_proactive_suspended` state change each
  time, no `ERROR` event) and the request goes out; Tier 1, which needs no LLM, keeps
  running. The suspension ends after `compaction_proactive_suspension_iterations`
  gate visits (default 6) or once the prompt has grown by
  `compaction_proactive_suspension_growth_ratio` (default 0.1) of its size when
  it began. A context refusal — the provider's, or the local fit's — lifts it
  at once and runs the reactive pass, as do `rearm()` and a snapshot resume.
  `ContextManager.run_compaction` and `force_compaction` take `llm_tiers` to run
  Tier 1 alone. `PerIterationCompactionPolicy` no longer takes `pair_orphans`
  or `message_stop`, since it no longer ends the turn.
- **`compaction_no_gain_backoff_iterations` takes effect.** The per-message
  recovery reset zeroed the backoff before the per-iteration gate could read
  it, so a low-gain routine pass ran on every iteration. The count now
  survives the reset. It ends early once the prompt has grown by
  `compaction_no_gain_backoff_growth_ratio` (default 0.1) of its size when it
  was set, so a new large tool result is not left alone, and on any context
  refusal. `PerIterationCompactionPolicy` takes a `prompt_tokens` callable for
  that measurement, and `ITurnState` gains `compaction_backoff_prompt_tokens`.
- **`RequestTokenCounterConformance` and `LifecycleRegistryConformance` are
  importable from `protocore.conformance`.** Both were in `SUITES` but only
  reachable through `protocore.conformance.suites`. The package's own tests now
  fail if a suite in `SUITES` is not re-exported by the package, or if a suite
  class is defined in `suites` without being listed in `SUITES`.

## [2.0.0a19]

### Added

- **A provider can size the rendered request, and is asked near the edge.**
  The optional `IRequestTokenCounter` capability
  (`async count_request_tokens(request) -> int | None`) reports the prompt
  tokens the server renders a request to. When the estimate is within
  `exact_token_count_margin_ratio` (default 0.75) of the point where the output
  cap starts being clipped, or of the compaction trigger, the loop asks for it
  and fits the request — or runs the gate — on that number. The default covers
  every undercount calibration can express (`1 - 1/4`); measured against a vLLM
  tokenizer the heuristic ran 1.76x short on JSON and 3.53x on hexadecimal
  text. Counts are kept per request content
  (`exact_token_count_cache_max_entries`, default 32), bounded by
  `exact_token_count_timeout_seconds` (default 5), and can be switched off with
  `exact_token_count_enabled`. Once the fit has counted a turn's request, the
  compaction gate reuses the factor that count set instead of counting the
  history a second time. After a count, the next one is made only when the
  content that count did not see — messages whose digest is not among the
  counted ones, so compaction or eviction cannot hide dense content arriving
  with it — sized at the worst undercount the margin assumes, could carry the
  prompt over the limit; a count that fails or times out stops
  counting for `exact_token_count_failure_backoff_seconds` (default 300). A provider without the capability sends exactly the
  requests it sent before; a count that fails or times out is logged and the
  estimate is used.

### Fixed

- **The token estimate learns from a count and from a rejection.** An exact
  count sets `token_estimate_calibration` to the measured ratio at once, and
  before the fit is attempted, so a count that proves the request cannot fit
  raises the factor and the refusal goes to compaction sized in the counted
  tokens. A rejection for length, which never carries usage, now raises the
  factor to the floor it proves — the prompt was at least the window less the
  output cap that was sent. On hexadecimal text the estimate had run at 0.28 of
  the server's count. The factor still starts each new run at the configured
  `token_estimate_calibration`; a host whose content is dense sets that per
  scope.

## [2.0.0a18]

### Fixed

- **A proactive compaction pass no longer forfeits the reactive one.** With the
  lower effective trigger, a history made of turns seeded from earlier runs
  crosses the proactive window at turn start; the routine profile leaves seeds
  alone and frees nothing, but the pass was carried into the message as "the
  compaction attempt", so the provider's rejection was answered by the
  output-cap ladder alone and the run died on a prompt that fit the window
  once seeds were summarised. The rejection handler now tracks the reactive
  pass separately and runs it once, after which the ladder applies.
- **An exhausted compaction budget hands the turn back to the output-cap
  ladder** when a strictly smaller cap is still available, instead of ending
  the run on the budget.

## [2.0.0a17]

### Added

- **Stale tool results can be cut down in the request view.** With
  `tool_result_stale_trim_enabled` (off by default), a tool result the run has
  moved past is sent as its first `tool_result_stale_max_chars` characters plus
  a line naming how much was kept and how much was removed, while
  `engine.history` keeps the whole value. The newest `tool_result_fresh_count`
  results and every result of the latest round of tool calls are never cut, and
  nothing is cut until the trimmable excess crosses
  `tool_result_stale_trim_batch_chars`, so the prompt prefix moves for a batch
  rather than for one result. A trimmed result stays trimmed for the run and the
  decision travels in the snapshot as `trimmed_tool_result_ids`. Lines starting
  with one of `tool_result_stale_trim_protected_prefixes` are carried over
  verbatim, so a result whose text names how to cite it keeps that line when it
  loses its body. Pinned results are spared unless a later write falsified the
  pin, and compacted placeholders are never rewritten. It does not delay
  compaction: the compaction gate measures the whole durable transcript, not
  the trimmed request view, so what it buys is a smaller and cheaper prompt at
  the provider, not fewer compactions.

- **A provider failure with nothing behind it no longer ends as an invented
  answer.** A run whose requests the endpoint refused before it wrote a word,
  called a tool or read a result is no longer wound down and asked to deliver
  a final answer — it goes terminal FAILED carrying the provider's own error.
  Once a tool result exists the partial outcome is real, and the wind-down
  still runs; its notice now names a provider failure instead of a budget
  (`soft_stop_notice_text_provider_error`), and its `state_changed` events
  carry the upstream's message as `soft_stop_detail`.

### Changed

- **A summariser failure costs its own unit, not the whole pass.** Tier 2 used
  to discard everything a pass had produced as soon as one call failed, so one
  unit the summariser cannot handle kept a run from shedding a single token
  and the next pass bought the same failure again. Failures are now counted
  per unit in `CompactionState.failed_anchor_keys`; past
  `compaction_summary_failed_unit_max_attempts` (2) the unit is left to the
  fold tier, and the other units in the batch commit. The census is carried in
  the run snapshot. A reply carrying no readable summary now counts as a
  failed call — that is the shape a cut-off reply takes — while a summary that
  is merely no smaller than the original does not — nor does a transport
  failure (a rate limit, a 5xx, a reset socket): only a unit that does not fit
  the summariser's window or whose reply the output cap cut is counted, since
  only those repeat. The forced passes ignore the census entirely and try
  every unit, and entries whose unit has left the history are pruned.
- **The per-turn summariser prompt states its budget in characters too.** It
  now also says that a longer reply is cut off and discarded, asks for a
  count and the records that matter instead of a copy of a long tool result,
  and names the single key it wants. A model that listed every record of a
  long result wrote a reply the output cap cut, and a cut reply is never
  parsed. `compaction_summary_chars_per_word` (6) converts the word budget,
  `compaction_summary_envelope_tokens` (32) reserves room for the JSON around
  the words before the budget is derived, and the budget is also clamped to
  what `compaction_summary_string_max_chars` will accept at decode time.
  The template gains a `max_chars` variable; a per-tenant override that does
  not use it is unaffected.

- **Transient provider failures are retried on the verdict the adapter gives,
  not on the exception type alone.** `LLMError` now carries `retryable`, which
  an adapter sets per raise — `LLMProviderError("no such model",
  retryable=False)` is terminal at once, while a 5xx, a reset connection and a
  silent stream take the bounded ladder that 429s and timeouts already took
  (`llm_transient_error_retry_max_attempts`, backoff doubling from
  `llm_transient_error_retry_backoff_base_seconds`). A context-window overflow
  is never retried this way and takes no such keyword. A cancelled run, or one
  whose wall-clock budget leaves room only to finalise, starts no further
  attempt, and a cancel during a backoff ends the pause immediately. Every
  attempt logs a WARNING naming the run and the attempt number.

### Fixed

- **The compaction trigger now sits below the prompt size the provider will
  still accept.** `derive_budgets` takes the lower of
  `model_context_window * compaction_trigger_ratio` and the window less the
  output reserve, less `request_context_safety_tokens`, less the new
  `compaction_trigger_turn_headroom_ratio` (0.15) for the turn about to be
  added. A server that reserves the output budget inside the context window
  rejects any prompt above `window - max output`; with the stock 0.8 trigger
  and 0.25 output reserve the trigger on a 65 536-token window stood above
  that cliff, so proactive compaction could not fire before the rejection.
  The emergency cliff is held strictly above the effective trigger. The
  output-reserve part of the deduction is gated on
  `provider_reserves_output_in_context_window` (default true): a provider that
  sizes its input window independently of the requested output gives that
  share of the window back by setting it false.
- **A summary's word budget is sized for the script it is written in.** The
  ceiling was half the summariser's output cap, two tokens a word being the
  English figure; JSON escaping and a non-Latin script cost three or four, so
  the summary of every large unit outgrew the cap, came back cut off, never
  parsed and never committed — and the next pass paid for the same units
  again. The rate is now `compaction_summary_output_tokens_per_word` (4), and
  it caps the fold target the same way.

## [2.0.0a16]

### Fixed

- **Reactive compaction can now shrink history seeded from earlier runs.**
  When a provider rejects a request as exceeding its context window, recovery
  keeps only `compaction_force_keep_recent_turns` trailing messages (one by
  default) and may summarise or fold prior-run seed turns, copying the seed tag
  onto every replacement so the host still excludes them from persistence.
  Previously a history made of protected seeds left reactive compaction with
  nothing to compact, and the run failed after the output-cap ladder ran out.
  Proactive emergency compaction keeps the routine window and seed protection.
- Fold summaries label prior-session turns separately from the operator's own
  instructions.

## [2.0.0a15]

### Fixed

- **Bounded context-window recovery when providers report only a prompt-size
  lower bound.** Repeated retries strictly reduce output headroom under
  `context_overflow_retry_max_attempts`, and proactive compaction is not
  repeated reactively for the same assistant message.

## [2.0.0a14]

### Fixed

- A measured context overflow is retried with a smaller output cap before
  history is rewritten; a repeated overflow gets exactly one compaction attempt.

## [2.0.0a13]

### Fixed

- **Context retries now use provider-reported prompt sizes when available.**
  OpenAI-compatible adapters can attach the provider's measured context window
  and input-token count to an overflow. The retry then reserves its configured
  safety margin against that measured size instead of relying only on the local
  estimator, preventing a second request from missing the hard limit by a small
  provider-tokenization difference.

## [2.0.0a12]

### Fixed

- **A context-overflow retry now reduces its output allowance.** Reactive
  compaction still rebuilds the prompt once, and the rebuilt request uses at
  most half the fitted output allowance the provider rejected by default. A
  smaller rebuilt prompt therefore cannot raise the retry back toward the
  normal cap and hit the same context limit twice.

## [2.0.0a11]

### Fixed

- **The compaction summary string limit remains at 1,024 characters.** This
  corrects an unrelated default change that slipped into the preceding
  provider-framing release while retaining its larger request safety margin.
- **Published installation examples now name the current pre-release**, both
  for the runtime package and the optional conformance test extra.

## [2.0.0a10]

### Fixed

- **Near-limit requests reserve more space for provider-rendered framing.**
  The default request margin is 1,024 tokens. Regression coverage now models
  the boundary where an unchanged output cap previously let a provider-side
  prompt count exceed the context window even though the local estimate fit.

## [2.0.0a9]

### Fixed

- **Near-limit requests now keep enough headroom for additive provider
  framing.** The default request margin is 512 tokens, covering chat-template
  and tokenizer overhead that a content-based local estimate cannot see while
  using less than one percent of a 65,536-token context window. Installations
  can still tune the margin, and boundary-focused synthetic tests can disable
  it explicitly.

## [2.0.0a8]

### Fixed

- **The default provider-framing margin now covers real chat-template
  variance.** Live OpenAI-compatible providers may add several tokens that are
  absent from a local message estimate. The default request margin is now 64
  tokens; installations can still tune it, and tiny synthetic test windows can
  explicitly disable it when they are testing an unrelated boundary.

## [2.0.0a7]

### Fixed

- **Provider framing can no longer push an exactly fitted request one token
  beyond the context window.** Complete requests now leave a configurable
  `request_context_safety_tokens` margin after prompt estimation and before
  choosing `max_tokens`. The default one-token margin covers providers that
  count a framing token absent from the local estimate, while the runtime
  constants contract lets an installation reserve more for another tokenizer.
  Invalid margins that consume the whole context window are rejected by both
  snapshot validation and the constants registry.

## [2.0.0a6]

This release makes long-running conversations safer at the two points where a
model can otherwise lose the task: history compaction and the final request
budget sent to a provider.

### Added

- **Bounded recovery for a response that spends its output on reasoning.** A
  length-limited response containing reasoning but no answer or tool call is
  discarded and retried through a small recovery ladder: lower the reasoning
  effort, then disable thinking when the run mode permits it. The ladder and
  its original controls survive snapshot pickup, do not overwrite live
  operator controls, and end with one honest best-effort wind-down request.
- **A hard context-window fit for every provider request.** The complete
  normalized messages and tool schemas are estimated together after the
  ordinary output cap, adaptive safety band and terminal reserve have been
  chosen. `max_tokens` is clipped before the request is manifested or sent;
  when no positive output fits, the action path gets one bounded compaction
  retry and secondary planning or summarisation calls fail locally without
  sending a request already known to be too large.

### Fixed

- **Runtime recovery messages no longer become operator intent during
  compaction.** User-role control messages are identified by provenance rather
  than role alone. Aged recovery nudges are removed on the same atomic working
  copy as the rest of Tier 2, so a failed summariser or recorder cannot leave
  history and compaction accounting disagreeing.
- **Token-estimate calibration follows the model that produced it.** Learned
  calibration survives snapshot pickup for the same model, respects a newer
  configured baseline, and resets on live or provider-chain model changes.
  Late usage from an earlier model is still accounted for but cannot replace
  the active model's calibration. A provider count from one wire request is no
  longer reused as a size floor for a different request or for a history-only
  compaction decision.

## [2.0.0a5]

The repository moved to `https://github.com/anchor-inference/protocore` and is
now the single home of the core; the package on PyPI is still `protocore` and
the import path is unchanged. Everything else in this release is the loop
spending less per round: a session store hears what a round appended instead of
being handed the conversation again, the tool surface travels by digest, and
compaction gains a third pass over what the first two cannot shrink.

### Added

- **A session store can be told what a round added.** The loop used to hand the
  store the whole working history once per round to record the one or two
  messages the round appended, so the cost of writing a turn down grew with the
  length of the conversation rather than with what the conversation just did.
  The engine now remembers the prefix the store already holds and compares the
  current history against it by object identity — `Message` is frozen, so an
  append leaves every earlier object where it was and a compaction, checkpoint
  or eviction builds new ones. `persist_session_history(engine)` is unchanged
  and is still the only method a store must have; a store that can write
  incrementally also attaches `persist_history_delta(engine, delta)` and is
  handed a `HistoryDelta` (`protocore.runtime.history_persist`) naming what was
  appended, or the whole history when the sequence was rewritten.
  `HistoryPersister` carries a default for the second that calls the first.
  Returning is the store's promise that the write landed: the marker advances
  only after the call returns, so a store that raises is offered the same
  messages again, and a store that defers a write it then drops calls
  `QueryEngine.forget_persisted_history()` — which is honoured even from inside
  the write. A hand-over with nothing to say is not made at all, and
  `QueryEngine.note_session_state_changed()` raises the notice for a session
  that changed in a way its messages do not show, such as a checkpoint.
- **The advertised tool surface is named by a digest.**
  `tool_surface_advertised` carries `tool_surface_digest` and
  `tool_surface_described`, and `protocore.runtime.tool_surface` answers
  `surface_descriptions(digest)` for a reader that has nothing kept against a
  digest it met.
- **A third compaction pass, for what the first two cannot touch.** Tier 2
  leaves one summary per tool batch and never re-summarises one, and it now
  refuses operator turns outright, so a long session ends up with a window made
  almost entirely of small summaries and instructions that no pass can shrink.
  The fold replaces each contiguous run of them with a single consolidated
  summary in which the operator's instructions survive as exact quotes. The
  task turn and the most recent instructions stay verbatim, a turn seeded from
  an earlier run of the session is never folded, and a fold is a summary like
  any other — a later fold absorbs it once its neighbourhood has grown again.
  `compaction_fold_enabled`, `compaction_fold_min_messages`,
  `compaction_fold_min_tokens`, `compaction_fold_keep_operator_turns`,
  `compaction_fold_max_spans_per_pass`, `compaction_fold_max_output_tokens` and
  `compaction_fold_summary_target_words` govern it; the completion event
  carries `tier3_folded` beside the counts the other tiers report.
- **The summariser instructions are templates.** `compaction_turn_summary` and
  `compaction_fold_summary` join the bundled registry, so the wording is
  reviewable as prose and an operator serving another language has somewhere to
  put the translation.

### Changed

- **`tool_surface_advertised` no longer repeats every tool's description on
  every run.** This is a wire change. The descriptions are decided by the
  registry and are the same on every run of a deployment, and they were nearly
  the whole event; they now travel with the first advertisement of a digest to
  reach each reader — the session, which is the unit a host fans events out
  over — and `tool_surface_described` says which kind of advertisement this is.
  A reader keeps the descriptions against `tool_surface_digest` and treats a
  missing `description` as "look it up", not "there is none";
  `protocore.runtime.tool_surface.surface_descriptions(digest)` answers a
  reader that has none. What is run-specific — `name`, `sources`, `roles`, and
  which tools are on the surface at all — is on every advertisement. The claim
  that a reader has been described to is recorded only once the event has been
  handed to the stream, so a run cancelled at that point does not spend its
  reader's one description on an event nobody received. The request manifest
  still records the tool definitions in full, so what was sent to the provider
  remains recoverable from the durable record.
- **Tool definitions are costed for tokens once per surface, not once per
  call.** The estimate is cached by digest and by the chars-per-token ratios it
  was computed under. The digest itself is recomputed every call on purpose:
  `ToolDefinition` is frozen but its parameter schema holds a plain `dict`, so
  a surface remembered against object identity would be handed back a digest
  that had stopped describing a schema edited in place.
- **The token estimate cache is no longer split by the calibration factor.**
  `token_estimate_calibration` is a single multiplier over the whole
  per-message partition; it was folded in before the number was cached and then
  keyed on, so the loop's calibrated reading and the calibrator's uncalibrated
  one evicted each other's entries and an alternating pair both walked every
  character of every message. The partition is now cached as the heuristic
  computes it and the factor is applied where the number is handed out, which
  is the same arithmetic for every caller.
- **Compaction summaries keep exact identifiers.** The summariser is told to
  carry every path, id, port, URL, number and error code through verbatim
  rather than substituting a plausible value, and to state an outcome with no
  tool result or confirmation behind it as UNKNOWN rather than as done or not
  done. A summary that quietly rounds an identifier is worse than no summary,
  because the run reads it back as fact.
- **An operator turn is never summarised.** An instruction is short enough that
  paraphrasing it frees almost nothing and specific enough that the paraphrase
  is a rewrite. Tier 3 is where those turns are condensed, with their wording
  quoted rather than restated.
- **A compaction pass no longer crawls.** Summariser calls go out
  `compaction_summariser_parallelism` at a time instead of one after another
  while the run sits in `COMPACTING`; a unit below
  `compaction_summary_min_unit_tokens` is not sent at all, since a summariser
  writes a sentence or three whatever it is handed and below some size the call
  is spent only to discover the summary is no smaller; and the word budget in
  the prompt scales with the unit being replaced rather than being a fixed
  sentence count. A reply that carries no usable summary is logged with its
  head, so the next one can be diagnosed rather than guessed at.

### Fixed

- **`protocore.__version__` reports the installed version.** It had been left
  at the string the first pre-release was cut with, so a host reading it back
  was told `2.0.0-alpha.1` whatever it had installed.

## [2.0.0a4]

This release is the result of a long pass over the core with one question in
front of it: what belongs in a universal agent runtime, and what only ever
belonged to the layer above it. The answer moved a great deal of code out,
tightened what remains into declared contracts, and made several things that
were conventions into checks. The public surface is narrower than it was and
says what it means; that is the point of the release, and it is a breaking one.

### Added

- **Conformance suites, shipped in the package.** `protocore.conformance` is a
  pytest suite a host runs against its **own** implementations of the contracts:
  `pip install "protocore[testing]"`, then `pytest --pyargs protocore.conformance`.
  It replaces "read the Protocol and hope" with a suite that fails when an
  adapter is subtly wrong — a store that loses ordering, a client that reports a
  stream idle without ending it, a sink that drops a field.
- **A constants registry.** Every tunable is declared once, with its bounds, its
  type, its default and the relationships it must hold with its neighbours, in a
  form a program can read. Coercion, validation, a whole-snapshot check and a
  repair that resets an out-of-range field are part of the model rather than
  something each caller reimplements.
- **A request manifest, and a provider that replays it.** Every model request
  now records what it was assembled from, with a digest taken over exactly the
  fields the model can see, so a request is reproducible and a replay that
  diverges is a real difference rather than a timestamp. Observability metadata
  is deliberately outside the digest: the same request stays the same request
  when only its labels differ.
- **A durable record of a tool call before it runs.** The intent is written
  before the effect, so a process that dies between deciding to call a tool and
  calling it resumes with the decision intact, and the charge against a run's
  budget is idempotent per call rather than per attempt.
- **An optional native token estimator**, released separately as
  `protocore-native` and built from source until there are wheels for it. The
  core stays pure Python and selects the extension only when it can import it,
  so having it changes speed and nothing else — the same numbers either way, and
  both arrangements are tested on every supported Python.
  `PROTOCORE_DISABLE_NATIVE=1` keeps the Python implementation in force when the
  extension is installed; it is read once, at import.
- **A `testing` extra** carrying just a test runner, so a host using the
  conformance suites does not inherit the core's linting and typing toolchain.
- **Turn policy as a contract.** The driver of an assistant turn kept the
  mechanics — open a stream, translate deltas, dispatch calls, close the round —
  and every product opinion that had grown into a branch inside it is now an
  object: it declares the named seams of a turn at which it wants to be
  consulted, is consulted in an order the core owns, and answers with events to
  forward and one directive saying what the loop does next. A host's policies
  are **merged with the core's by name rather than replacing them**, so the
  core's own guarantees cannot be switched off by omission, and a directive a
  seam cannot honour — asking to restart a turn at a completion seam — is
  refused with a named error instead of being ignored.
- **One session work pool for both kinds of work.** A background command and a
  delegated child run are the same thing from the loop's side: a unit of work
  with an address, a status, a way to wait for it and a way to stop it. A child
  run therefore has an address, can be asked how far along it is, and can be
  stopped — and a parent waiting on one no longer holds its turn and its slot in
  the tree budget for the whole descendant run. A pool is a collaborator the
  host injects, and a cold start that fails to re-attach a session's still
  running work says so on the run instead of looking like a session with nothing
  running.
- **Interrupts are parked, declared and resumed as a set.** A turn parks every
  held call, announces the whole set in one event, and is resumed with one call
  carrying the resolutions — instead of a turn that could only ever be stopped
  by the first thing that interrupted it.

### Changed

- **The constants snapshot carries the loop's settings and not the host's.**
  What used to be one enormous per-tenant model was a mixture: values the run
  loop reads on every turn, and values only a service layer above the core ever
  looked at. The second group has left the core entirely, and what remains is
  named for what it is. A host that kept its own settings in this model moves
  them into its own; the registry above is how it declares them.
- **One public entry point for resuming a run.** `resume(engine, snapshot, ...)`
  restores from the snapshot first — identity, delivery mode and schema are all
  checked before the first mutation, so a refusal drives nothing — and only then
  chooses how to continue: an approved tool call, a new message, or neither.
  Asking for two at once is an error rather than a guess. The weaker per-turn
  entry point is no longer part of the public surface.
- **One canonical tool result, with projections taken from it.** The typed
  result carries both its success flag and its content blocks, and the shapes a
  model, a user interface and a store each need are derived from it rather than
  maintained beside it.
- **Tool identity comes from a declared role map**, not from tool names spelled
  as literals across the runtime. Which tools delegate, which have side effects,
  which may never be delegated — each is now a property something declares once
  and the loop reads, instead of a name repeated in fifty-five places where a
  rename could silently miss one.

### Fixed

- **The streaming JSON parser no longer costs more than the text it reads.**
  Repairing a partial document used regular expressions that were retried at
  every quote and, on an unterminated string, walked to the end of the buffer
  each time; growth was quadratic. It is now a single string-aware pass, and the
  incremental parser keeps a mirror of the value being built so that the cost of
  a chunk is the depth of the structure rather than the length of the buffer.
  Measured: 28 KiB delivered in 64-byte chunks, 4.84 s to 0.017 s; a single
  repair of a 64 KiB truncated string full of escapes, 18.65 s to 0.016 s.
  A repair that reached a Python-only literal one level down could also return a
  `set` from a JSON parser; every level is now normalised or refused.
- **Token estimates are memoised across a turn**, so a long history is not
  re-measured from scratch on every pass over it.
- **The transient-retry counter resets when the stream settles**, not only on a
  clean round, so a round that ended by tripping the backstop now refreshes the
  retry budget in the same place as every other round.
- `QueryEngine.rearm()` now also restarts the state that is attached to an
  engine *after* it is constructed. The re-arm rebuilds from a fresh engine, and
  `vars()` of a fresh engine cannot see what the host or the run loop attaches
  later, so three things survived every re-arm in silence: the cached tool
  dispatcher, which holds a tool-error counter read out of a helper bag the host
  may since have replaced; the fire-once warning latch for a normalised outbound
  system prompt, which made its warning fire once per engine rather than once
  per run; and the per-run streaks the dispatcher keeps inside the helper bag —
  the consecutive same-tool-same-error cell, the sandbox-down streak and its
  one-shot injection flag, the string-type streak, and the subagent soft-cap
  counts. An agent that repeats one failing call at the start of each turn could
  cross a cap documented as per-run that no single turn ever reached. The bag
  itself belongs to the host and is left alone, as is the run tree's shared work
  ledger. A test now reads the package for every `engine.x = ...` and
  `setattr(engine, "x", ...)` outside the constructor and fails when one is
  classified as neither dropped nor kept.
- **A run resumed from a snapshot is bound to the run it came from.** A snapshot
  whose identity does not match is refused instead of quietly driving another
  run's state.
- **A cold resume restores the whole of a run's accounting**, not the part that
  happened to be constructed with the engine: the position of the provider
  chain, the cumulative budgets of a run tree (a dead process holds no permits,
  so occupied slots come back released), the durable fact that a run was
  cancelled, and the session's background tasks.
- **Context is rebuilt when a request falls back to a generic shape**, and the
  idle-stream branch of provider fallback is now reachable and covered — it
  previously could not be entered at all.

## [2.0.0a3]

Supersedes 2.0.0a2, whose files were removed from the index. That build carried
comments and docstrings that cited internal working documents as the authority
for public behaviour, and that kept the labels a review round leaves behind in
the code it reviewed. A reader outside the project could see that closed
documents govern this library and could read nothing of them. No functional
difference; the prose now states each reason in its own terms.

The publication scanner gained the rules that would have caught it — a document
cited as authority, a review-round trace, a work-package label, a planning or
triage label — so the class cannot come back silently.

### Fixed

- `QueryEngine.rearm()` now restarts every per-run allowance rather than a
  named twelve. An engine that takes an unbounded number of turns on one
  history carried the rest across, and each one ended the agent quietly: the
  identical-tool loop guard counted a fingerprint for the life of the engine,
  so an agent that opened every turn with the same observing call was refused
  it from the fourth turn on; the repeated-error circuit breaker's block list
  is unioned into the visible tool surface, so a tool that failed for a reason
  that had since passed was withdrawn for good; the cooperative stop flag had
  no lowering seam, so an agent interrupted once never spoke again. The reset
  is now expressed the other way round — the engine names what SURVIVES a
  re-arm (history, compaction state, live-control queues, lanes, the injected
  collaborators) and rebuilds everything else — so a field added to the
  constructor resets by default instead of quietly accumulating. A test walks
  the constructor and fails when an attribute is classified as neither.

## [2.0.0a2]

Supersedes 2.0.0a1, whose files were removed from the index. That build
carried comments naming the tooling used to write them and a pointer to a
working document that ships with nothing — no functional difference, but not
what belongs in a published artifact. Nothing else changed.

## [2.0.0a1]

First public release. Withdrawn.

Protocore has existed for some time as the closed core of an agent product.
This is that core, extracted and published under the MPL — the same code, with
the parts that only made sense inside one company's repository rewritten to
describe the boundary rather than the company.

### Added

- The protocol boundary: 20 interface `Protocol`s and an `IBlobStore` ABC in
  `protocore/contracts/`, covering the model client, run and session stores,
  the tool registry, memory, workspace, search, blobs, skills, todos, hooks,
  event transport, and subagent dispatch.
- The ReAct runtime: `QueryEngine` plus `query()`, driving one agent turn at a
  time and emitting typed `TurnEvent`s. Snapshot and resume are first-class.
- A three-layer tool surface — tenant policy, a lean clipped surface, and
  progressive discovery over BM25 retrieval — with a permission gate ahead of
  dispatch and a shell-safety policy behind a real command-chain parser.
- Two-tier context compaction, session memory folding, and a token-budget model
  that keeps a long run inside its window.
- `RuntimeConstants`: 524 per-tenant tunables as a frozen Pydantic snapshot,
  every one of them documented, with new behaviour defaulting off.
- In-memory adapters (`protocore.tests_support.adapters`) that implement the
  same protocols the real ones do, so a turn runs end to end with no external
  services.
- Documentation in English and Russian under `docs/`.
- 2964 tests, a 90% coverage floor, strict typing, lint, and a security scan,
  all gated on Python 3.12, 3.13, and 3.14.

[Unreleased]: https://github.com/anchor-inference/protocore/compare/v2.0.0a6...HEAD
[2.0.0a6]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a6
[2.0.0a5]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a5
[2.0.0a4]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a4
[2.0.0a3]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a3
[2.0.0a2]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a2
[2.0.0a1]: https://github.com/anchor-inference/protocore/releases/tag/v2.0.0a1
