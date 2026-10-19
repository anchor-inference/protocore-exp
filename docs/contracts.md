# Contracts — the core boundary

> Audience: an engineer wiring a host application onto the
> pure core, or anyone who needs to know exactly where the core ends and the
> outside world begins. Scope: the core library `protocore/`.

`protocore` is a set of **contracts** (Python `Protocol`s + typed Pydantic
models) and a protocol-first ReAct runtime. Everything outside-facing is a
`Protocol` that a host implements; the core ships **no** database driver, HTTP
endpoint, or LLM client. This document is the catalogue of that boundary: the
interface protocols the host provides, the core type system that flows across
them, and the conventions that keep the surface stable.

For the deeper "how it fits together" view — the loop, the subsystems, the data
flow of one turn — read [`architecture.md`](architecture.md), the structural
source this page indexes.

---

## The contract package: no monolithic `protocols.py`

The interface surface lives in `protocore/contracts/`, **one module per
domain**. There is deliberately **no single `protocols.py`** — the old
monolithic file was split so each concern (LLM, runs, sessions, memory,
workspace, …) owns a self-contained module with its protocol, its typed models,
and its errors together.

```
protocore/contracts/
  types.py        # the core type system (Message, ContentBlock union, Run, …)
  llm.py          # ILLMProvider + IProviderChain + LLMRequest/LLMResponse + provider deltas
  run.py          # IRunStore
  session.py      # ISessionStore
  blob.py         # IBlobStore
  search.py       # ISearchIndex
  todo.py         # ITodoStorage
  tool_registry.py# IToolRegistry + ToolVisibilityPolicy
  tools.py        # Tool (ABC) + ToolContext + tool errors
  run_state.py    # RunScopedState — the run's live state, incl. the host compartment
  tool_roles.py   # ToolRole / ToolRoleMap / ToolArgumentSlot — what a host's tools DO
  skills.py       # ISkillStore (+ SkillBundle / SkillIndexEntry / SkillFileRef)
  agent_dispatch.py# IAgentDispatch
  events.py       # IEventStream
  hooks.py        # IHookManager + HookResult / HookSpec / HookActionKind
  memory.py       # IMemory (+ IMemoryContentScanner)
  workspace.py    # IWorkspace
  resilience.py   # IToolTransport + IResilienceClassifier (+ the taxonomy)
  prompts.py      # IPromptTemplateProvider
  observability.py# CacheObserverProtocol + IRequestManifestSink + RequestManifest
  background.py   # IWorkPool / IBackgroundTaskPool / WorkHandle / TaskRecord
  middleware.py   # the lifecycle contract — ILifecycleRegistry + the five kinds
  interrupt.py    # PendingInterrupt / InterruptKind / InterruptResolution
  turn_policy.py  # ITurnPolicy / ITurnState / TurnFlags / TurnCoordinate
  config.py       # ConstantSpec / ConstantGroup / IConstantsRegistry
  snapshot.py     # the snapshot's schema version + the upcaster chain
  runtime_constants.py        # LoopConstants + RuntimeConstantsProvider
  attempt_ledger.py
  evidence.py                 # observed evidence + candidate lifecycle (imported by QueryEngine)
  tool_chunking.py            # chunkable-content truncation recovery (imported by the loop)
```

### Contract-first re-export principle

Re-exports are managed through explicit `__all__` lists, and the boundary is
**contract-first**: the public surface leads with the interface protocols, then
the typed models that cross them. Two `__all__` lists matter:

- **`protocore/contracts/__init__.py`** — the contract surface, re-exported as
  the explicit `__all__` list (not every symbol the domain modules define). This
  is the list to import from when implementing an adapter. Some symbols defined
  in their contract modules are deliberately **not** in this `__all__` — e.g.
  `AttemptLedger` (`contracts/attempt_ledger.py`), `ConstantSpec` /
  `ConstantGroup` / `IConstantsRegistry` (`contracts/config.py`),
  `PendingInterrupt` / `InterruptResolution` (`contracts/interrupt.py`),
  `ITurnPolicy` / `TurnFlags` (`contracts/turn_policy.py`), `RequestManifest` /
  `IRequestManifestSink` (`contracts/observability.py`),
  `SNAPSHOT_SCHEMA_VERSION` (`contracts/snapshot.py`), `LLMTimeoutError`
  (`contracts/llm.py`), and `SkillNotFoundError` (`contracts/skills.py`) — import
  those from their named module.
- **`protocore/__init__.py`** (the top-level package) — a **curated subset** of
  the same surface for the common case. It re-exports the store/service
  interfaces a host reaches for most (`I*`) plus `Tool`, the core type system,
  `LoopConstants`, the lifecycle vocabulary, and a handful of runtime
  utilities.

> **Heads-up:** `IMemory`, `IWorkspace`, `IToolTransport`,
> `IResilienceClassifier`, `IWorkPool`, `IPromptTemplateProvider`, and
> `CacheObserverProtocol` are exported from `protocore.contracts` (and their
> named modules) but are **not** in the top-level `protocore` `__all__`.
> Import them from `protocore.contracts` (or the specific module) rather than
> the top-level package. `IProviderChain`, `IConstantsRegistry`,
> `ICoreConstantsProvider`, `IRequestManifestSink`, `IDelegationTool`,
> `IMemoryContentScanner`, `IRunToolErrorCounter`, `ITurnPolicy` and
> `SkillFileRef` are not in `protocore.contracts.__all__` either — import them
> from their own module.

`QueryEngine` is **not** re-exported at either level — import it directly from
`protocore.runtime.query_engine`; the resume entries (`resume`,
`resume_interrupts`, `resume_approved_tool`, `answer_pending`) come from `protocore.runtime`. See the
[Public API](architecture.md#public-api-protocore__init__py) section.

---

## Interface protocols (what the host provides)

These are the seams. Core declares the `Protocol` (or ABC); the host binds a
concrete adapter. Each is one line: the contract and what a host supplies.

| Protocol | Module | What the host provides |
|---|---|---|
| `ILLMProvider` | `contracts/llm.py` | LLM completions: `stream_with_tools`, `complete_structured` (post-loop JSON schema), `complete_text` (post-loop free-form document), and `count_tokens`; a universal LiteLLM/OpenAI-compatible adapter (OpenRouter / vLLM / OpenAI). |
| `IProviderChain` | `contracts/llm.py` | Ordered remaining providers plus a one-way `advance()` cursor. `QueryEngine` injects it as `provider_chain` for mid-stream failover; `None` leaves existing recovery untouched. Not in `protocore.contracts.__all__` — import from `protocore.contracts.llm`. |
| `IRequestTokenCounter` | `contracts/llm.py` | Optional provider capability: `async count_request_tokens(request) -> int \| None`, the prompt tokens the server renders the request to (messages through its chat template, tools, generation prompt). Asked only near a limit (`exact_token_count_margin_ratio`); `None` means "cannot count here", a raise falls back to the estimate with a warning. Looked up on the provider's class, so a provider without it is unaffected. |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` | Per-tenant `LoopConstants` (`async get(tenant_id)`), Postgres-backed with a Redis cache. |
| `ISessionStore` | `contracts/session.py` | Session / transcript persistence. |
| `IRunStore` | `contracts/run.py` | Run record create / list / read (durable row + hot record). |
| `IToolRegistry` | `contracts/tool_registry.py` | The concrete `ToolRegistry` ships in core; the host registers concrete `Tool`s and a `ToolVisibilityPolicy`. |
| `Tool` (ABC) / `@tool` | `contracts/tools.py`, `tools/decorator.py` | Concrete tool implementations bound to the canonical verb names. |
| `IToolTransport` | `contracts/resilience.py` | The tool / VM transport the resilience wrapper wraps; optional `rebuild()` hook. |
| `IMemory` | `contracts/memory.py` | A scope-aware FTS/BM25 memory store + an `IMemoryContentScanner` for injection-scanning. |
| `IWorkspace` | `contracts/workspace.py` | A durable byte store + FTS/BM25 manifest, atomic write, per-scope GC. |
| `ISkillStore` | `contracts/skills.py` | Skill-bundle storage and lookup **and** the multi-file API: `list_files` / `load_file` returning `SkillFileRef` rows (at minimum the canonical `SKILL.md` / `SKILL_ENTRY_PATH` entry). The **core loop never calls** `list_files` / `load_file` — it catalogs via `list` / `list_enabled_subset` and loads a triggered body via `load` / `list_subset`. Store reads key on `QueryEngineConfig.account_id`, not `tenant_id`. The catalog is rendered by `render_skills_catalog` (+ `derive_skill_index_budget_tokens`) in `runtime/skill_index.py` as `Skill(skill="{name}")` lines, not file paths. |
| `IHookManager` | `contracts/hooks.py` | Hooks whose executor is out of process (`invoke(event, payload, tenant_id)`), driven at the permission gate and around tool dispatch. |
| `ILifecycleRegistry` | `contracts/middleware.py` | The one lifecycle seam: observe / decide / transform / around / notify, with an owner, a scope and an idempotent disposer per registration. |
| `IEventStream` | `contracts/events.py` | Cross-pod durable event stream for SSE reconnect / replay. |
| `IBlobStore` | `contracts/blob.py` | Content-addressed blob storage used by Tier-1 compaction. |
| `ISearchIndex` | `contracts/search.py` | A generic lexical search index. |
| `ITodoStorage` | `contracts/todo.py` | Per-session todo persistence. |
| `IAgentDispatch` | `contracts/agent_dispatch.py` | Subagent dispatch / lookup. `dispatch` answers with a `SubagentHandle` — the child run is addressable from the moment it is launched, not only when it is over. A caller that wants the old blocking shape writes the wait itself: `await (await dispatch(task)).wait()`. |
| `IBackgroundTaskPool` | `contracts/background.py` | The session's background commands, as the loop reads them: `list`, `get`, `refresh`, `drain_wakes`, and the attachment declaration (`mark_session_attached` / `ensure_session_attached`) a resumed run checks before it trusts an empty wake list. |
| `IWorkPool` | `contracts/background.py` | Everything above, plus the handles that make a unit of work drivable: `launch(spec, start)` (the id is minted BEFORE the work starts and handed to `start`), `handle`, `stop`, `stop_session(scope, grace)` and `subscribe(on_terminal)`. A delegated child run is a record of `kind="agent"` in this same pool, so waiting on one and waiting on a background command are the same question asked of the same collaborator. |
| `IPromptTemplateProvider` | `contracts/prompts.py` | System-prompt template rendering. |
| `CacheObserverProtocol` | `contracts/observability.py` | A prompt-cache hit-rate sink, injected via `QueryEngineConfig.cache_observer`. |
| `IRequestManifestSink` | `contracts/observability.py` | Where a `RequestManifest` — the durable record of exactly what was sent to a provider — is kept, and for how long. The core computes the manifest and its SHA-256 id and hands it over; it owns no store and no retention policy. |
| `IConstantsRegistry` | `contracts/config.py` | Declaration of the host's own constant groups (`declare`), fail-closed name resolution (`resolve`), `defaults`, `coerce` and `repair`. |
| `ICoreConstantsProvider` | `contracts/config.py` | The `LoopConstants` snapshot in force for a scope, fresh as of the moment it is asked for. |
| `IResilienceClassifier` | `contracts/resilience.py` | Which neutral `ResilienceErrorClass` a failure message describes. The wordings belong to whatever the host put behind its tools, and a pattern for them in core is a copy of another codebase that goes stale without a build ever failing. A host that binds none keeps the neutral behaviour: every failure is told apart by its own text and none is recognised as a transport being down. |
| `IRunToolErrorCounter` | `contracts/run.py` | Durable per-run tool-error counting that survives a process change. |
| `IMemoryContentScanner` | `contracts/memory.py` | Injection-scanning of memory content on the way in. |
| `ITurnPolicy` | `contracts/turn_policy.py` | One product decision about a turn. Not an adapter the host must supply: the core ships its own set, and a host substitutes a policy into it **by name**. |

The **32 `Protocol`s** in `contracts/` are not one flat list of adapters: some
are stores the host must supply (`ILLMProvider`, `IRunStore`, `ISessionStore`,
`IBlobStore`, `ISearchIndex`, `ITodoStorage`, `IToolRegistry`, `ISkillStore`,
`IAgentDispatch`, `IEventStream`, `IHookManager`, `IMemory`, `IWorkspace`,
`IWorkPool`), some are optional seams that stay inert when nothing is bound
(`IToolTransport`, `IResilienceClassifier`, `CacheObserverProtocol`,
`IRequestManifestSink`, `IProviderChain`, `ILifecycleRegistry`), and some are
shapes the core hands *back* rather than asks for (`WorkHandle`,
`BackgroundTaskView`, `ITurnState`, `ClassifiedLike`).
`IToolSafetyPolicy` (`runtime/tool_permission.py`) is a runtime-registered
permission policy rather than a contract, and `Tool` is an ABC, not a
`Protocol` — the host subclasses it (or uses `@tool`).

Whichever kind it is, a host does not have to wait for a run to find out that
its object has the shape the core will call: `protocore.conformance` ships one
suite per contract in `SUITES`, and binding each to an adapter turns that into
a failing test in the host's own suite. See [`testing.md`](testing.md).

> `IBlobStore` is declared as an ABC; the rest of the store/service interfaces
> are `Protocol`s. Either way the rule is the same — the core depends only on the
> declared shape and never on a concrete implementation.

---

## The core type system

Every conversation primitive flows as one of these Pydantic models (the
"use `Message` models, never raw dicts" convention). All live in
`contracts/types.py` unless noted. Most are frozen value objects.

### Messages & content

- **`Message`** — the sole conversation primitive (role-scoped). Assistant turns
  carry `content_blocks` (text + tool_use + thinking interleaved).
- **`MessageRole`** (`StrEnum`) — `system` · `user` · `assistant` · `tool`.
- **`MessageOrigin`** (`StrEnum`) — `conversation` · `compaction` · `nudge` ·
  `wind_down` · `recovery` · `background_wake`; read as `Message.origin`.
  Compaction's records (ledger, summaries, floor digest) are user-role but
  `compaction` in origin, so a host that persists or shows history does not
  take them for the user's turn (see
  [compaction](compaction.md#13-its-records-are-marked-as-its-own)). Every
  message the loop writes for itself carries a non-conversation origin as well:
  a correction of the model's next step is a `nudge`, the notice that the run
  is wound down is `wind_down`, a resume after a cut answer, the tool-error
  breaker's turn or a synthesised salvage call is `recovery`, and the note that
  background tasks finished is `background_wake`. A host filters on the value
  instead of matching the wording of a template. The origin is derived from the
  message's tags and cannot be set: `Message(origin=...)` is accepted and
  ignored. It reads, in order, the compaction tags
  (`COMPACTION_SUMMARY_METADATA_KEY: True`), `ORIGIN_METADATA_KEY`
  (`"protocore.origin"`, a `MessageOrigin` value — for a host's own
  scaffolding), and the loop's `SYNTHETIC_RECOVERY_METADATA_KEY`, whose kinds
  `synthetic_recovery_origin()` maps; a kind it does not name reads as
  `recovery`.
- **`ContentBlock`** — a **union type**, not a class:
  `TextBlock | ThinkingBlock | ImageRefBlock | ToolUseBlock | ToolResultBlock`.
- **`ContentBlockKind`** (`StrEnum`) — the discriminant: `text` · `thinking` ·
  `image_ref` · `tool_use` · `tool_result`.
- **`TextBlock`** / **`ThinkingBlock`** — plain text and model reasoning (the
  latter usually stripped before persistence).
- **`ToolUseBlock`** — an assistant-emitted tool invocation (`tool_call_id`,
  `name`, `arguments_json`; arg bytes capped).
- **`ToolResultBlock`** — a tool-call result returned to the model
  (`tool_call_id`, `content`, `is_error`, `metadata`).
- **`ImageRefBlock`** — an image reference whose bytes live in `IBlobStore`.
  A user turn holds one block, its text, so the images sent with it ride in
  `metadata[IMAGE_REFS_METADATA_KEY]` (`"protocore.image_refs"`) as a list of
  `ImageRefBlock.model_dump(mode="json")` entries. `message_image_refs(message)`
  reads both places, blocks first, and skips an entry that does not validate.
  The core sends nothing itself: an adapter for a model that takes images
  loads each blob and sends it, one for a text-only model sends a marker or
  nothing. The prompt estimate prices every image, block or entry, at
  `token_count_image_tokens`.

### Tool calls & results

- **`ToolCall`** — an LLM-emitted invocation surfaced to `Tool.invoke`
  (`id`, `name`, `arguments`). Carries truncation flags
  (`truncated_by_output_cap`, `args_partial_truncated`) the loop uses to detect
  a mid-stream-truncated argument JSON.
- **`ToolResult`** — the result of a single invocation, stated once and
  projected three ways. `content` is the canonical value, complete whatever its
  size, and `is_error` says whether the call worked. `model_projection` is what
  the transcript carries in its place when the whole of it does not belong
  there (`model_content` is the property every `ToolResultBlock` is built
  from, so a tool that names no projection is unaffected); `ui_payload` rides
  the result event and never enters the transcript, so it costs no tokens and
  cannot change what the model decides; `canonical_ref` says where the whole
  value can be fetched back from once the transcript no longer holds it; and
  `path` says which workspace path the result is a view of, which is what lets
  a later write say the result is no longer true.
- **`ToolContext`** — the per-invocation context handed to a tool: the run's
  scope, its `metadata`, the `evidence` context when the tool produces evidence,
  and `run_state` — the live state of the run the call belongs to.
- **`RunScopedState`** (`contracts/run_state.py`) — everything one run carries
  while it executes: its cancel event, its shared-state lock, the tree's work
  ledger and concurrency budget, the streaks its caps are measured against, and
  `host` — an opaque compartment this package neither reads nor writes, so an
  embedder has one place to put what only it understands. A host composes it at
  run start and hands it to the engine; `to_snapshot()` states the part a resume
  in another process must be given back.
- **`ToolRoleMap`** (`contracts/tool_roles.py`) — what a host's tools DO, said
  once where they are registered: a `ToolRole` per tool (`reads_path`,
  `writes_path`, `runs_shell`, `never_delegated`, …) and, per
  `ToolArgumentSlot`, the argument spellings a value arrives under. The loop
  asks the map, never a tool's name, so an installation that names its writer
  something else keeps every behaviour that depends on knowing it is one.
- **`ToolDefinition`** — the registry entry (name, description, params schema,
  approval flag, category) a `@tool` function or `Tool` subclass produces.
- **`ToolParameterSchema`** — the JSON-Schema shape of a tool's parameters.
- **`ToolError`** (and `ToolInvocationError`, `ToolPolicyDenied`,
  in `contracts/tools.py`) — the tool error hierarchy.

### Runs, sessions, events

Three distinct run shapes — do not conflate them (see
[`architecture.md`](architecture.md)):

- **`RunStatus`** (`StrEnum`) — the **durable** run lifecycle mirrored in the
  persistent `runs.status` column: `queued` · `running` · `completed` ·
  `partial` · `error` · `cancelled` · `incomplete` · `paused`. `partial` is a
  functionally-terminal status for a run that finished its loop but accumulated
  tool-dispatch errors.
- **`Run`** — the durable run record (`id`, `tenant_id`, `session_id`,
  `status`, timestamps, optional detail-blob ref).
- **`RunState`** — the **ephemeral** hot working set (held in a Redis hash by the
  host): `current_turn`, token counters, `last_event_id`. (Distinct again from
  `LoopState`, the in-flight engine FSM in `runtime/loop_state.py`, which is
  *not* a contract type.)
- **`Session`** — the multi-turn conversation root (durable, never deleted).
- **`Event`** — the in-flight event envelope (`run_id`, `name`, `payload`)
  emitted via `IEventStream` / the in-process `EventBus`.
- **`StopReason`** (`StrEnum`) — why a turn terminated: `end_turn` · `tool_use` ·
  `max_tokens` · `max_turns` · `stop_sequence` · `error` · `cancelled`.
- **`ExecutionReport`** — the bounded per-run telemetry rollup (events,
  tool-call records, LLM-call records, warnings, subagent runs, artifacts, and an
  optional `AttemptLedger` snapshot), with structural caps from
  `protocore.constants`.

### LLM request / response

- **`LLMRequest`** — the request the loop assembles for `ILLMProvider`
  (`messages`, `tools`, `max_tokens`, `extra` — including the
  `cache_breakpoints` prompt-cache hints).
- **`LLMResponse`** — a non-streaming response shape returned by
  `complete_structured` and `complete_text`.
- **`LLMObservabilityContext`** — the per-call observability context attached to
  an LLM request. Its `cache_key` (set from `QueryEngineConfig.cache_key`) and
  `routing_key()` (the key, else the session, else the run) are what an adapter
  routes the provider's prompt cache by — a `prompt_cache_key` body field or a
  sticky-routing header — so a fork can be routed with its source.
- **`IProviderChain`** — not a request/response type; the failover cursor
  `QueryEngine` rebinds onto `self.llm` when a mid-stream provider fails.

### Provider failures and the run's outcome

- **`ProviderFailure`** (`contracts/llm.py`, frozen) — the verdict an adapter
  attaches to a provider error it raises:
  `LLMProviderError(message, classified=ProviderFailure(...))`. Fields:
  `reason` (a `ProviderFailureReason`), `retryable`, `should_fallback`,
  `retry_after_seconds`, `provider_message`. The loop reads it to choose between
  the next provider on the chain, another attempt on the same one, the
  wind-down and a terminal failure. An error raised without a verdict keeps the
  class default for `retryable` and never moves the chain.
- **`ProviderFailureReason`** (`StrEnum`) — `auth` · `auth_permanent` ·
  `billing` · `budget` · `connection` · `context_overflow` · `format_error` ·
  `image_too_large` · `llama_cpp_grammar_pattern` · `long_context_tier` ·
  `model_not_found` · `oauth_long_context_beta_forbidden` · `overloaded` ·
  `payload_too_large` · `provider_policy_blocked` · `rate_limit` ·
  `server_error` · `thinking_signature` · `timeout` · `unknown`. What the loop
  does with each is three public sets in the same module:
  `PROVIDER_ADVANCE_REASONS` (a sibling provider may serve it; includes
  `connection`), `PERMANENT_FAILURE_REASONS` (the same request will not succeed
  again) and `SURFACE_INDEPENDENT_FAILURE_REASONS` (narrowing the tool surface
  in a wind-down will not get past it). `budget` is a spend cap the **host**
  enforces: it is permanent, never moves the chain and never winds down.
- **`classify_transport_exception(exc)`** — the verdict for a failure to reach
  the endpoint at all (refused, reset, unreachable, unresolvable: `connection`;
  timed out: `timeout`), found on the exception or down its cause chain, or
  `None`. **`provider_failure_of(exc)`** reads the typed verdict back from any
  raised error, lifting a verdict object of an adapter's own type onto
  `ProviderFailure`.
- **`RunOutcome`** / **`RunCause`** (`contracts/outcome.py`) — every terminal
  `message_stop` carries `outcome` (`completed` · `failed` · `stopped` ·
  `wound_down`) and `cause` (the JSON form of a `RunCause`: `kind`, `detail`,
  and `failure`, the `ProviderFailure` when a provider failure ended the run;
  `None` for a run that simply completed). A run wound down because the
  provider failed (`provider_error`) or the model made no progress
  (`model_no_progress`) ends `failed` even when its closing turn wrote prose; a
  run wound down by a bound it was given (turns, tool calls, output tokens,
  wall clock) ends `wound_down`. Read back with `RunOutcome(payload["outcome"])`
  and `RunCause.model_validate(payload["cause"])`. The same stop lists, under
  `leftover_queued` (`LEFTOVER_QUEUED_FIELD`), the ids of the queued steer and
  follow-up items the engine still held and never placed.
- **`RecoveryBudget`** / **`RecoveryKind`** (`contracts/recovery.py`) — the
  one allowance every recovery layer of a run draws from: an in-place retry
  (`transient_retry`), a provider-chain advance (`chain_advance`), a re-ask
  after a round with nothing usable (`empty_response`), shrinking a request the
  window refused (`context_overflow`), and a host's own re-drive after an
  outage (`redrive`) or an overflow (`overflow_redrive`). Each layer keeps its
  own bound as well; the shared one stops them multiplying (two retries on each
  of three rungs, re-driven five times, was thirty requests for one outage).
  `recovery_budget_max_attempts` and `recovery_budget_max_seconds` bound a
  stretch; both default to `None`, which counts the draws without bounding
  anything, so recovery is unchanged until a host sets one. A provider round
  that comes back with words or tool calls ends the stretch (`settle()`). The
  budget lives in `RunScopedState.recovery_budget`, is read through
  `QueryEngine.recovery_budget`, starts afresh at `rearm()` unless the caller
  passes `rearm(recovery_budget=...)`, and travels in the snapshot.
- **`recoverable_after`** — every terminal `message_stop` carries it
  (`RECOVERABLE_AFTER_FIELD`): the seconds a host should wait before re-driving
  the run as a new run, or `None` when that is not worth it. Only a `failed`
  run qualifies. An overflow is recoverable at once (`0.0`: compact and
  re-drive) while fewer than `recovery_overflow_redrive_max_attempts`
  `overflow_redrive` draws were made in the stretch. An outage (a retryable
  verdict of `rate_limit`, `overloaded`, `server_error`, `timeout` or
  `connection`, or a rate limit, timeout or stall that carried no verdict) is
  recoverable after `recovery_redrive_backoff_base_seconds`, doubled per
  `redrive` already drawn, capped at `recovery_redrive_backoff_max_seconds`,
  and never sooner than the provider's `retry_after_seconds`. A permanent
  refusal, a spend cap (`budget`) or a crash is never recoverable, and nothing
  is once the budget refuses another draw. The host keeps the re-drive: it
  draws one `redrive` or `overflow_redrive` from `engine.recovery_budget` and
  hands the budget to the run it starts. `recoverable_after(outcome, cause,
  budget, rc)` is the function behind the field.

### Ingress, blobs, compaction

- **`AgentEnvelope`** — the single cross-component ingress contract
  (`kind`, `payload`, `metadata`; payload size capped). Parsed/serialised via
  `parse_envelope` / `serialize_envelope`.
- **`EnvelopeKind`** (`StrEnum`) — `task` · `control` · `result` · `error`.
- **`BlobMetadata`** — a blob index entry (`ref`, `content_type`, `size_bytes`,
  `sha256`).
- **`CompactionSourceRef`** — a pointer to a compacted tool-result blob, persisted
  as a wire-format placeholder during Tier-1 compaction.

### Verification & chunking

- **`VerificationLifecycle`** / **`VerificationDelivery`** /
  **`CandidateBundle`** / **`ReleaseDecision`** (`contracts/evidence.py`) —
  the candidate-verification lifecycle `QueryEngine` snapshots as
  `verification` and uses to gate public reader delivery. Re-exported from
  `protocore.contracts`.
- **`is_chunkable_content_mutation`** (`contracts/tool_chunking.py`) — the
  single predicate for the write→append→finalize truncation recovery, asked in
  terms of the `ToolRole`s a call carries rather than of tool names.
  Imported by the loop. Not in `protocore.contracts.__all__` — import from
  the named module.

### Interrupts, run state, and the snapshot schema

- **`PendingInterrupt`** / **`InterruptKind`** / **`InterruptResolution`**
  (`contracts/interrupt.py`) — what a paused run is waiting for, as a value
  rather than a latch. A run stops for a person in three different ways, and
  `InterruptKind` names them: `approval` (a gate parked the call; nothing ran),
  `question` (the tool ran far enough to ask, and the answer is its result) and
  `external_call` (the call was handed outside the run and its result arrives
  by another route). A single boolean could not tell them apart, so an answered
  question and an approved call arrived at the same door and the loop had to
  guess — one way runs a tool nobody approved, the other tells the model that a
  question it asked came back as a failure. A run holds however many interrupts
  are open at once, which is what lets a batch of three parked calls be
  answered in one act instead of three rounds of stop-ask-resume. The
  resolutions are `approve` (optionally with `updated_input`, the corrected
  arguments the call is then really run and recorded with), `deny`, `answer`
  and `abandon`; a map naming a wait that is not open, answering a kind that
  does not take that decision, or leaving an interrupt undecided is refused
  unless the caller says outright that it means to leave the rest parked.
- **`RunScopedState.to_snapshot()`** — the allowances a resumed run must
  not be handed twice: the tree's cumulative work ledger, the capacity of
  its concurrency budget, and the run's recovery budget (schema version 9). Live objects (an `asyncio.Event`, a semaphore, a
  lock) are per-process by nature and are rebuilt by whoever wires the resumed
  run, never restored from a payload.
- **`SNAPSHOT_SCHEMA_VERSION`** / **`SnapshotUpcaster`** /
  **`SnapshotSchemaError`** (`contracts/snapshot.py`) — the snapshot states its
  own schema version, and a reader that cannot recognise it refuses the payload
  outright rather than resuming a run with fields missing and budgets refilled.
  An older payload is brought forward instead where a chain of upcasters covers
  it — one step per version, each reading the shape one below and filling in
  what that version introduced. A version with no step is a refusal, because
  skipping one leaves its fields unset: the same silent half-restore. A payload
  with no version field is version 1.

### Hooks

- **`HookEvent`** (`StrEnum`) — the **21** lifecycle coordinates, the whole
  vocabulary of the seam in `contracts/middleware.py`: `run_start` ·
  `run_finalize` · `session_start` · `session_end` · `turn_start` · `turn_end` ·
  `context_transform` · `request_prepare` · `response_received` ·
  `request_error` · `user_prompt_submit` · `pre_tool_use` · `tool_execute` ·
  `post_tool_use` · `file_changed` · `pre_compact` · `compaction_commit` ·
  `compaction_rollback` · `post_compact` · `subagent_start` · `subagent_stop`.
- **`HookResult`** — a hook's verdict (allow / deny / modify, via
  `HookActionKind`).
- **`HookSpec`** — the declarative spec for a registered hook.

### Skills, subagents, todos

- **`SkillManifest`** / **`SkillIndexEntry`** / **`SkillBundle`** /
  **`SkillFileRef`** — the skill catalogue shapes. `SkillFileRef` is the
  multi-file bundle index row (`path`, `size_bytes`, `mime_type`,
  `content_hash`); fetch bytes via `ISkillStore.load_file`. Every bundle has
  at least `SKILL_ENTRY_PATH` (`SKILL.md`). A legacy single-file skill may
  synthesise that one row from `body_md`. `SkillFileRef` is **not** in
  `protocore.contracts.__all__` — import it from `protocore.contracts.skills`.
  The loop's catalog is `Skill(skill="{name}")` call shapes, not these paths.
- **`SubagentDef`** / **`SubagentTask`** / **`SubagentResult`** — the subagent
  dispatch shapes used by `IAgentDispatch`. Beyond the tool and skill lists, a
  definition also states how its child is DRIVEN — `model`, `max_turns`,
  `timeout_seconds`, `permission_mode`, `background` — and a task may override
  the last three for one call plus `notify_on_finish` and `expected_seconds`.
  All of them are declarations the host honours: nothing in the loop builds a
  child run, so nothing in the loop can pick its model or start its clock.
- **`WorkSpec`** / **`TaskRecord`** / **`WorkHandle`** / **`AgentRef`** — the
  pool shapes. `WorkSpec` is everything needed to mint a record before the work
  starts; `TaskRecord` is the record itself (`kind`, `status`, `owner_scope`,
  exit, error, durations, `agent`); `WorkHandle` is `identity()` / `wait()` /
  `stop(grace)` over one record, and `SubagentHandle` is that handle over a
  subagent result — there is no second handle type. `owner_scope` names the run
  that has to end the work when it is not the session's: a delegated run shares
  the session (so shares the workspace and the wake) and retires only what it
  started itself.
- **`IDelegationTool`** (`contracts/agent_dispatch.py`) — how the loop
  recognises a delegating tool. A tool declares the contract instead of
  carrying a flag attribute, so delegation is a stated capability rather than a
  duck-typed guess.
- **`Todo`** / **`TodoStatus`** (`StrEnum`) — per-session todo persistence shape.

---

## Implementing an adapter

To bind the core to a host:

1. Implement the interface protocols you need from `protocore.contracts` (you do
   not need all of them — memory is default-off (`memory_enabled = False`),
   and the concrete workspace tools live in the host).
2. Accept and return the core type-system models — never raw dicts at the
   boundary.
3. Inject configuration via `LoopConstants` (a frozen snapshot) and
   `ToolContext.metadata`; never hard-code tenant policy.
4. Construct a `QueryEngine` with your adapters and drive it with
   `async for evt in engine.run(message)`, or pick a stored run back up with
   `async for evt in resume(engine, snapshot)`.

### Reporting a provider failure

The loop never inspects a status code or a response body: the adapter does,
and says what it found by attaching a `ProviderFailure` to the error it raises.
Without one, a dead endpoint ends the run even when the next provider on the
chain is healthy, and a server's `Retry-After` is never seen.

```python
from protocore.contracts.llm import (
    LLMProviderError,
    LLMRateLimitError,
    ProviderFailure,
    ProviderFailureReason,
    classify_transport_exception,
)

try:
    response = await client.post(url, json=body)
except Exception as exc:
    failure = classify_transport_exception(exc)  # refused / reset / timed out
    if failure is None:
        raise
    raise LLMProviderError(str(exc), classified=failure) from exc

if response.status_code == 429:
    raise LLMRateLimitError(
        response.text,
        classified=ProviderFailure(
            reason=ProviderFailureReason.rate_limit,
            retryable=True,
            retry_after_seconds=parse_retry_after(response.headers),
            provider_message=response.text,
        ),
    )
```

- `retryable` on the verdict sets the error's own flag unless the raise passes
  `retryable=` explicitly, and the loop reads the error's flag first; keep the
  two in agreement.
- `retry_after_seconds` lengthens the backoff before the next attempt on the
  same endpoint, never past `llm_transient_error_retry_backoff_max_seconds`.
- `should_fallback=True` says the structured-output form of the request was
  refused and prompted JSON may be answered; only strategies with such a
  fallback read it.
- A spend cap the host enforces is reported as `reason=budget`,
  `retryable=False`: the run fails at once and its terminal `cause.failure`
  names the cap, so nobody has to match words in a message.
- `protocore.conformance.assert_adapter_failure(exc)` checks, in the adapter's
  own tests, that an error carries a verdict and that the two flags agree, and
  returns the verdict for further assertions.

The mechanics of extending the runtime — which seam to choose (protocol vs hook
vs RC toggle vs prompt section) and the hard "do not modify the loop structure"
rule — are covered in `extending.md` and [`architecture.md`](architecture.md).
The import boundary (core never imports a host) is enforced by
`tests/test_core_import_boundary.py`.
