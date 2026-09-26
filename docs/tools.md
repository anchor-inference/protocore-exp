# Tool surface

> Audience: an engineer adding or reasoning about agent-facing tools in the
> **pure core** (`protocore/`).
> Scope: the current core library (`protocore/`). The core owns
> tool *contracts, names, dispatch, gating, retrieval, and ordering*. It ships a
> few concrete tools whose entire surface **is** the protocol contract — ask-user
> (`tools/ask_user.py`) and memory (`tools/memory.py`) — but it registers **no**
> concrete backend-bound
> (sandbox / exec / read / write) tool itself; those production, backend-bound
> tools live in the sibling the host repo. (The `IWorkspace` **contract** lives
> in core, but its concrete tools are host-only — see the workspace
> subsystem in [`architecture.md`](architecture.md).)
> For the wider picture see [`architecture.md`](architecture.md).

The core deliberately keeps the agent-visible tool list **small and stable**: a
small universal pool keeps the prompt cheap for smaller local models, and a
deterministic ordering keeps the KV-prefix cache reusable across turns. For the
default backend-bound surface (sandbox / exec / read / write) the core registers no
concrete tool — it defines the *shape* and the machinery that runs one tool call
safely, and the host binds the backend-backed implementations. (The core does
ship its own protocol-surface tools — ask-user and memory — see the
[scope note](#tool-surface) above.)

---

## The `@tool` decorator

`tools/decorator.py` provides a lightweight in-core helper to turn an async
function into a `Tool` subclass. A Pydantic `TypeAdapter` derives the parameter
JSON Schema from the function's type hints; the special `context: ToolContext`
parameter is skipped. The decorated callable is **replaced** with the generated
`Tool` subclass.

`tool`, `ToolContext`, and `ToolResult` are re-exported at the package top
level:

```python
from protocore import tool, ToolContext, ToolResult


@tool(name="echo", description="Echo back the input.")
async def echo(context: ToolContext, text: str) -> ToolResult:
    return ToolResult(tool_call_id="...", content=text)
```

A result carries its canonical value and, beside it, projections for the model
and for the client — see [Extending the Core](./extending.md) where those
fields are worked through with an example.

The wrapped function must be `async` — `@tool` raises `TypeError` otherwise.
`echo` is now a `Tool` subclass whose `.definition` carries `name`,
`description`, and the schema built from the `text: str` hint. The
backend-backed default tools use this decorator but live in
the host package; the core itself registers none of them (though it
does ship the ask-user / memory protocol-surface tools). For richer
or stateful tools, implement the `Tool` ABC (`contracts/tools.py`) directly.

---

## Dispatch pipeline

`runtime/tool_dispatch.py` — `ToolDispatcher.dispatch(...)` is the **single core
entry point** for executing one `ToolCall` after the LLM has finished emitting
it. The dispatcher is agnostic to the tool implementation: it depends only on
`Tool.invoke`. It **never raises** on a tool error — every failure mode is
translated into a `tool_result(success=false)` block so the model can recover on
the next turn and the run never hard-crashes.

`dispatch()` is an async generator that yields `TurnEvent` envelopes and, as its
final item, a `DispatchOutcome`. The lifecycle:

1. **Registry lookup** — `DispatchErrorKind.unknown_tool` if the name is not
   registered.
2. **Schema validation** — the byte-cap / JSON-serialisability invariant on the
   input dict (tool-specific Pydantic validation lives in the host adapter).
3. **Permission gate** — fans out to `ToolPermissionGate.check(...)` (below). A
   `require_approval` verdict surfaces a `tool_call_pending` event and emits no
   tool result; the caller transitions the loop to `AWAITING`.
4. **Preconditions** — the tool's `ToolDefinition.preconditions` DAG is checked
   (below); an unsatisfied tool short-circuits to a failure when
   `LoopConstants.tool_preconditions_enabled` is set.
5. **Execute** — `tool.invoke(ctx)` wrapped in `asyncio.wait_for` honouring
   `rc.tool_timeout_seconds`.
6. **The `post_tool_use` coordinate** — fire-and-await; may rewrite the output.

`DispatchOutcome` (frozen) carries `success`, `content`, `is_error`,
`error_kind`, `approval_required` / `approval_token`, `ask_user_required` /
`ask_user_payload`, `duration_ms`, and a `metadata` bag. `DispatchErrorKind` is
the failure taxonomy: `validation | permission | execution | timeout |
rate_limit | unknown_tool | consecutive_error_cap`. The last is a guard — once
the per-run consecutive-identical-error streak exceeds
`tool_dispatch_consecutive_error_cap`, the dispatcher rewrites the failure into
this kind so a stuck loop terminates instead of burning iterations.

A tool may attach a machine-readable `structured_error` dict (e.g.
`{"finalization_recommended": True, "reason": ...}`) to a raised exception; the
dispatch except-branch forwards it on `DispatchOutcome.metadata` and the loop
surfaces a finalize hint to the model.

See the [Tool dispatch + gating](architecture.md#dispatch-roles-the-canonical-result-and-pairing-repair) section
for the event-emission contract.

---

## Permission gate

`runtime/tool_permission.py` — `ToolPermissionGate.check(...)` runs an async
pipeline and returns the **first non-allow** `ToolPermissionDecision`. The
decision's `outcome` is one of `allow` / `deny` / `require_approval`, with an
optional rewritten `modified_input` and `approval_token`. Each decision also
records the `PermissionStage` at which it was reached (for telemetry).

The `PermissionStage` StrEnum names the gate's four ordered stages, plus a
no-op default:

| Order | Stage (`PermissionStage`) | What it checks |
|---|---|---|
| 1 | `whitelist` | The `ToolVisibilityPolicy` (and any subagent narrowing whitelist) must permit the tool name — `blocked` always denies, `visible` (when non-empty) is a strict allow-list. |
| 2 | `safety_policy` | Per-side-effect-class checks via the `IToolSafetyPolicy` chain. |
| 3 | `rate_limit` | Host-only; the baseline is a no-op `allow`. Compose a Redis-backed bucket policy via `register_policy` to deny here. |
| 4 | `hook` | The `pre_tool_use` coordinate — the final, highest-leverage stage; it can flip `allow` → `deny` / `require_approval` / modify the args. Skipped when no hook manager is wired. |
| — | `default` | The decision's StrEnum default value, used for the implicit `allow` when no stage objected. |

A safety policy implements `IToolSafetyPolicy` — `applies_to(side_effect_class)`
plus `evaluate(tool, arguments, ctx)`. The default policy stack contains exactly
one policy:

- `ShellSafetyPolicyAdapter` — wraps `DefaultShellSafetyPolicy`, inspecting the
  argument the host declared as the shell command for tools carrying the
  `runs_shell` role. A tool whose command spelling is undeclared is sent for
  approval rather than run unexamined: the check cannot be skipped just because
  the core does not know where to look.

`HttpDnsAllowlistPolicy` and `WorkspacePathPolicy` are provided but **not** in
the default stack; the host stacks them on at runtime via `register_policy`,
which keeps the core API frozen. The gate is always on. The `pre_tool_use`
coordinate is the highest-leverage seam for an LLM-as-policy gate.

See the [Permission gate](architecture.md#dispatch-roles-the-canonical-result-and-pairing-repair) section for the
gate's place in the dispatch flow and the side-effect class map.

---

## The 3-layer effective surface

`runtime/tool_registry.py` — `ToolRegistry` implements the `IToolRegistry`
contract (`contracts/tool_registry.py`). Its `compute_effective_surface(...)` is
the per-turn filter that keeps the LLM's tool list small and relevant while
preserving a byte-deterministic ordering. It is called by the loop each turn and
applies three layers:

1. **Policy** — apply the `ToolVisibilityPolicy` (`visible` allow-list /
   `blocked` deny-list / `pinned` always-include), yielding the tenant's
   visible set.
2. **Clipping** — if `top_k is None` or the visible set is already `<= top_k`,
   return it sorted by name (no retrieval at all).
3. **Progressive discovery** — otherwise always include the `pinned`,
   `forced_pinned` and `always_load` tools, and fill the remaining slots with
   the best-ranked of the rest for the recent user `query` (see
   [Tool retrieval](#tool-retrieval)).

```python
def compute_effective_surface(
    self,
    tenant_id: str,
    policy: ToolVisibilityPolicy,
    *,
    query: str = "",
    top_k: int | None = None,
    retrieval: RetrievalSettings | None = None,
) -> Sequence[ToolDefinition]: ...
```

Whichever layer runs, the **final ordering is always name-ascending** — the
retrieval order drives *selection*, but the emitted list is sorted by name so
the LLM context stays byte-stable and the KV-prefix cache survives across turns.
The clip threshold is the RC `tool_retrieval_top_k` passed by the loop, and
`retrieval` is `RetrievalSettings.from_constants(rc)`.

`ToolRegistry.search(query, top_k=...)` — the path a `ToolSearch`-style tool
calls — differs in one respect: it returns tools in **rank order**, best first,
ties broken by name. A search result is read once by the model, and the order is
the only way it learns which hit fits best.

---

## Tool retrieval

`runtime/tool_retrieval.py` ranks tools lexically, in pure Python, with no
dependency beyond the standard library. The same engine serves the per-turn clip
and `search`.

**What is indexed.** Each tool is five fields: its name, its `search_hint`, the
first sentence of its description, the rest of the description, and its
parameter names and parameter descriptions. The hint and the parameters are for
finding the tool only; neither is added to the schema the model sees.

**How text is analysed** (`runtime/text_analysis.py`, the same steps for the
catalogue and the query):

- identifiers are split — CamelCase, `snake_case`, `kebab-case`, dotted paths,
  letter/digit boundaries — and the joined form is kept as well, so
  `BrowserOpen` matches both `browser open` and `browseropen`;
- text is case-folded and `ё` is spelled `е`;
- English and Russian stopwords are dropped, including conversational fillers
  ("please", "слушай", "короче", "плз");
- words are stemmed (`runtime/stemmers.py`): the Snowball Russian stemmer for
  Cyrillic, the original Porter stemmer for English. Tokens shorter than three
  letters are left alone.

**How a query is scored.** BM25F: a term's frequency in each field is
normalised by that field's average length, weighted, summed across fields and
saturated once. The weights and the BM25 parameters are constants:

| Constant | Default |
|---|---|
| `tool_retrieval_name_weight` | 1.0 |
| `tool_retrieval_search_hint_weight` | 1.0 |
| `tool_retrieval_summary_weight` (first sentence) | 1.0 |
| `tool_retrieval_description_weight` (the rest) | 0.6 |
| `tool_retrieval_parameters_weight` | 0.3 |
| `tool_retrieval_bm25_k1` | 1.5 |
| `tool_retrieval_bm25_b` | 0.3 |
| `tool_retrieval_lexicon_weight` | 0.5 |

They were chosen together by cross-validation over labelled English and Russian
queries against a catalogue of about 700 tools. With the lexicon turned off, a
hint weight of 2 does better than 1.

**Russian queries against English tools.** A Russian query shares no words with
an English description. The registry expands each Russian stem of the query to
the English stems it translates to, at `tool_retrieval_lexicon_weight` relative
to the query's own terms. The lexicon ships as package data
(`runtime/tool_retrieval_lexicon.json`, about two thousand English words with
their Russian equivalents); it is generic developer vocabulary translated from
the English of tool descriptions, never from queries. It is the one thing that
lets a Russian query find a third-party tool that will never carry a Russian
hint. A host passes its own with `ToolRegistry(lexicon=Lexicon.from_translations(...))`,
or turns expansion off with `lexicon=None` or a weight of 0.

**When nothing scores**, a second stage matches loose substrings and shared
prefixes (`normalized_fallback_match`), which still finds a partial tool name or
an inflection the stemmer does not reduce. It serves both `search` and the clip.

**Cost.** The analysed catalogue and the scoring constants are built once per
catalogue version and settings and cached on the registry instance; `register`
and `unregister` start a new version. At about 700 tools a build takes around
150 ms and a query about a third of a millisecond.

**A host ranker.** `ToolRegistry(retriever=...)` accepts an `IToolRetriever`
(`contracts/tool_retrieval.py`): a synchronous `rank(query, documents, limit)`
returning names, best first. Its ranking is fused with the lexical one by
reciprocal rank fusion (`reciprocal_rank_fusion`, `k` =
`tool_retrieval_fusion_rank_constant`). An embedding ranker is the intended use;
the core ships none.

### Writing a `search_hint`

Descriptions are written for the model, in English; the hint is written for
the person or model searching. Every host tool should carry one:

- **8–14 English synonyms** for the action and its object, not repeating the
  tool's name ("screenshot capture snap picture image page").
- **10–16 Russian words**: the dictionary form **and** the imperative of each
  verb (открыть открой, напомнить напомни), the nouns of the objects it acts on,
  and the slang and loanwords people actually type (скинуть скинь, глянуть
  глянь, коммит, пулреквест, напоминалка).
- **Avoid generic words** that would match other tools as well — "запрос",
  "задача", "сервер" without a qualifier pull queries towards the wrong tool.

---

## Tool preconditions

Preconditions enforce tool **ordering**: a tool that requires a prior
observation is masked until its precondition is satisfied, so the model cannot,
for example, mutate a record before reading the governing policy.

There are two systems; they never interact:

- **Runtime DAG** — `runtime/tool_preconditions.py` (`check_preconditions`,
  `resolve_precondition`, `record_satisfaction`, `compute_masked_tools`,
  `derive_satisfied_from_messages`). A tool's `preconditions` are
  read from its `ToolDefinition`; the satisfied set is replayed from the run's
  own messages, so it survives snapshot/resume without being persisted at all.
  This layer is consumed in
  the dispatch path (step 4 above) and is gated by
  `LoopConstants.tool_preconditions_enabled` (default `False`). It
  **blocks** a tool the model chose.
- **Run-level forcer** — `runtime/run_tool_preconditions.py` plus
  `QueryEngineConfig.tool_preconditions`. An ordered tuple of tools this run
  **must** call before the agent is free to answer; while an entry is
  outstanding the loop sets `LLMRequest.extra['forced_tool_choice']`. Empty
  (the default) is a no-op. This **forces** a tool the model did not choose.

See the [Tool preconditions](architecture.md#technology-inventory)
section for the precondition mechanism.

---

## What a tool declares: roles, and what its result is

**Roles, not names.** The runtime asks what a call DOES, never what it is
called. `contracts/tool_roles.py` is where that is said: `ToolRole` is the
capability (`reads_path`, `writes_path`, `appends_path`, `edits_path`,
`finalizes_path`, `searches_workspace`, `runs_shell`, `fetches_url`,
`delegates_work`, `records_plan`, `discovers_tools`, `asks_user`,
`never_delegated`), and `ToolRoleMap` — passed in as
`QueryEngineConfig.tool_roles` — is the host's declaration of which of ITS tool
names carry which of them. The map also carries the argument spellings that go
with those roles (`ToolArgumentSlot`): which key holds the shell command, which
holds the body of a write, which holds a terminal tool's answer. The runtime
reads raw arguments before any input model has resolved an alias, so it has to
be told them rather than guess.

Every comparison of a tool name against a string spelled inside the core used to
assume that every installation names its tools the way the first one did. A host
that called its shell tool something else lost the shell deny-patterns, its
large-file writes stopped converging, and nothing anywhere said so — the
comparison simply never matched. A role the map does not mention is now a
capability this installation does not have, and the feature that needs it says
so in a warning rather than going quietly inert.

The same map bounds a delegated run:
`runtime/child_capabilities.py::narrow_child_capabilities` computes what a child
may do from its parent and its `SubagentDef`, narrowing only. It is applied
twice — when the child's catalogue is resolved and again on each of the child's
calls — because the catalogue keeps a child from being shown what it may not
have, and the gate keeps it from having what it was not shown.

**One value, three audiences.** `ToolResult.content` is the canonical value,
complete whatever its size. The projections sit beside it: `model_projection`
is what the transcript carries in its place (the first page of a long listing,
`wrote 4.2 MB to <path>` for bytes the model has no use for reading back);
`ui_payload` rides the result event and never enters the transcript, so a whole
rendered table costs no tokens and cannot change what the model decides;
`canonical_ref` names a blob the whole value can be fetched back from — and when
a tool names none, compaction stores the value itself at the moment it first
needs the room and fills this in on the block it rewrites. A tool that names no
projection says the value is small enough to be its own, which is the common
case and costs it nothing.

**A record before the call.** Every dispatched call commits an `IntentRecord`
(`runtime/intent.py`) before the tool is touched, with its result ids reserved
and an explicit lifecycle — `RESERVED`, `PENDING_APPROVAL`, `DISPATCHED`,
`PAUSED_ASK_USER`, `SETTLED`. That is what lets a resumed run tell apart a call
that never started, one whose outcome is genuinely unknown, and one that is
waiting on an answer. Guessing costs correctness in the worst direction:
reporting an interrupted call as failed invites the model to repeat it, and a
repeated call with a side effect applies that effect twice.

---

## Extending the tool surface

- **Add a tool:** implement the `Tool` ABC (`contracts/tools.py`) or use `@tool`;
  register the resulting `ToolDefinition` with the `IToolRegistry`.
- **Control visibility:** set `visible` / `blocked` / `pinned` on a
  `ToolVisibilityPolicy`.
- **Add a safety check:** implement `IToolSafetyPolicy` and register it with
  `ToolPermissionGate.register_policy` — it evaluates after the defaults; the
  core gate stays frozen.
- **Declare what it does:** add the tool's `ToolRole`s and argument slots to the
  `ToolRoleMap` you pass as `QueryEngineConfig.tool_roles`. A tool the map does
  not describe still runs; what it loses is every behaviour that depends on
  knowing what kind of call it is.
- **Gate by observed state:** a rule keyed on an argument pattern plus
  something already observed in the run is a **host** evaluator bound at the
  lifecycle seam, not a core mechanism. The core's two precondition systems are
  the DAG and the run-level forcer above.

See [`extending.md`](extending.md) for the broader "pick your seam" guide and
the import-boundary rule.
