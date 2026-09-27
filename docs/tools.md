# Tool surface

> Audience: an engineer adding or reasoning about agent-facing tools in the
> **pure core** (`protocore/`).
> Scope: the current core library (`protocore/`). The core owns
> tool *contracts, names, dispatch, gating, retrieval, and ordering*. It ships a
> few concrete tools whose entire surface **is** the protocol contract — ask-user
> (`tools/ask_user.py`), memory (`tools/memory.py`) and tool search
> (`tools/tool_search.py`) — but it registers **no**
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
ship its own protocol-surface tools — ask-user, memory and tool search — see the
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
2. **Clipping** — if `top_k is None`, or the tools that are not pinned
   already number `<= top_k`, return the set sorted by name (no retrieval at
   all).
3. **Progressive discovery** — otherwise always include the `pinned`,
   `forced_pinned` and `always_load` tools, and add the `top_k` best-ranked of
   the rest for the recent user `query` (see
   [Tool retrieval](#tool-retrieval)). Pinned tools never count against
   `top_k`; they once did, and a floor as large as the default left the clip
   no room to retrieve anything.

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
The clip threshold is the RC `tool_retrieval_top_k` passed by the loop — `0`,
the default, passes `None` and turns the clip off (see
[why](#why-per-message-clipping-is-discouraged)) — and `retrieval` is
`RetrievalSettings.from_constants(rc)`.

The loop does not send this list as it is. `runtime/tool_deferral.py` builds the
request's tools from it: it leaves out any tool group the run
[holds back](#holding-tool-groups-back), and appends the tools the run has
discovered after it, in discovery order.

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
finding the tool only; neither is added to the schema the model sees. The first
sentence ends at a full stop, `!` or `?` followed by a space, but not at one
inside brackets or closing an abbreviation (`e.g.`, `i.e.`, `vs.`, `т.е.`,
`напр.`; `etc.` and `т.д.` only before a capital): the same sentence is the line
`ToolSearch` shows for the tool, and a line cut at "(e.g." tells the model
nothing.

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
hint, so it also carries the loanwords and slang a Russian-speaking developer
uses for the vocabulary of trackers, chat, calendars and deployments —
"пулреквест", "ишью", "таска", "смержи", "выкати", "созвон", "алерт" — which no
dictionary lists and which an MCP server's English is full of. A host passes its own with `ToolRegistry(lexicon=Lexicon.from_translations(...))`,
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

## Holding tool groups back

Advertising every tool is the best surface there is for as long as it fits:
measured on several models, a surface of about two hundred tools answered at
least as well as any scheme for finding tools on demand. It stops fitting in two
ways, and both are walls rather than slopes. The definitions can take enough of
the window that the conversation no longer has room, and some providers refuse a
request outright above a fixed number of tools (128 and 350 are both seen). A
host that connects a large MCP server reaches either on the first request.

So the loop can hold whole **groups** of tools back: leave them off `tools`,
name them in one block of the system prompt, and load them on request through
`ToolSearch`. Nothing of this happens while the surface fits — unless the host
asks for it: a group declared **lazy** is held back even then, because it is a
family the run rarely needs.

### Groups

A tool joins a group by its class attribute `tool_group` (read with `getattr`,
like `search_hint`), or by a name prefix the host declares:

```python
registry.declare_group("scheduling", "Timed and recurring jobs")
registry.declare_group(
    "github", "GitHub issues and pull requests", dynamic=True, prefix="Mcp_Github_"
)
registry.declare_group(
    "browser",
    "Drive a web browser",
    prefix="Browser",
    load="lazy",
    instructions="Ask the user before submitting a form.",
)
registry.register(ToolSearchTool(registry))
```

`ToolGroup(name, description, dynamic, prefix, load, instructions)` lives in
`contracts/tool_registry.py`, and `IToolRegistry` gains `declare_group` and
`tool_groups`. `declare_group(name, description, *, dynamic=False, prefix=None,
load="auto", instructions="")` refuses a load mode outside
`TOOL_GROUP_LOADS`; redeclaring a group replaces its description, load mode and
instructions, and the loop decides again on its next request. An explicit `tool_group` wins over a prefix; among prefixes the
longest wins. Membership never reaches the wire and is not part of the surface
digest. A **dynamic** group is one whose membership is not the host's own code —
an MCP server's proxies.

`undeclare_group(name)` forgets a group, and forgetting one that was never
declared is not an error. A host calls it when a group's tools are gone for good
— the proxies of an MCP server the operator switched off or removed — because a
declaration outlives its tools, and one left behind would claim the prefix of a
server added later under the same name. Neither declaring nor forgetting a group
changes the registry's catalogue generation: groups change no search result, and
the loop's deferral decision is keyed on the declared groups themselves.

### Load modes

A group's `load` says when it is advertised:

| `load` | Held back |
|---|---|
| `eager` | never for size; only a provider's hard limit on the number of tools pushes it off |
| `auto` (default) | by the rules below: dynamic groups at once, the others while the surface is over |
| `lazy` | whenever a discovery tool is admitted for the run, whatever the size |

`lazy` is for tool families a run rarely needs: a dozen browser tools used in
one session of a hundred cost their definitions on every request of the other
ninety-nine; held back, they cost one line of the catalogue and one call to
load. A run with no discovery tool treats `lazy` as `auto`, since a blind call
by exact name would then be the only way in, which costs more than the
definitions. A group only a tool's `tool_group` attribute names, never
declared, is `auto`.

`eager` beats the token budget, the dynamic rule and the room kept for loaded
tools, but not `max_advertised_tools` itself: a request over the provider's
limit is refused outright, which no load mode is worth. So an eager group is
held back only while the surface is still over that limit with every other
group held back, largest eager group first. A group that becomes eager leaves
the floor of an earlier decision (see below): the operator asked for it by
name.

`QueryEngineConfig.tool_group_loads` — a mapping of group name to load mode —
overrides the declarations for one run, and a value outside the three is
refused. A host maps operator settings and per-session exceptions through it
rather than redeclaring a group every session shares. It is part of what the
deferral decision is keyed on.

### When groups are held back

`tool_deferral_mode` is `"auto"` by default and `"off"` turns it off. In `auto`,
once per run — and again only if the registry's catalogue or groups, or the
host's visibility policy, change — the loop measures the surface it would
otherwise send. The policy counts because a host may switch tools on mid-run by
replacing it, registering nothing; the pins the run adds for tools it loaded do
not, so loading a tool never reopens the decision:

1. every **lazy** group is held back, largest first, when a discovery tool is
   admitted;
2. every **dynamic** group that is not eager is held back, largest first;
3. then, while the surface is over — its definitions above
   `tool_definitions_ratio` of the context window, or its count above
   `max_advertised_tools` with room left for `pinned_tool_max_count` loaded
   tools — the other groups go, largest first, eager ones last and only while
   the count is over the limit itself.

Never held back: the forced floor (`forced_pinned`), explicitly pinned tools,
`always_load` tools, the discovery tool itself, and any tool in no group. A host
that wants a tool deferrable says so by grouping it.

Without a discovery tool (role `discovers_tools`) registered and admitted, the
only way back for a held-back tool is a call by its exact name, so groups are
held back only when the surface is over `max_advertised_tools` — a provider
that refuses the request outright — dynamic groups first; the token budget is
not enforced then. The catalogue says to call by exact name instead of pointing
at a search the model does not have.

Held-back tools stay **admitted** by the visibility policy: they are only not
advertised. A model that calls one by its exact name is served (see
[below](#calls-of-tools-that-were-not-advertised)).

The decision is made once, not per request, because the catalogue it produces
sits at the head of the cached prompt: remade per request, it would move every
time the answer flipped. When a change of catalogue or policy makes it again,
the decision in force is its floor: a group held back stays held back while it
has tools, and only a dynamic group the run is newly admitted to, or a limit
the surface now exceeds, adds to it. Another session switching a server on, or
an unrelated tool switched off, leaves the catalogue as it was.

### The catalogue

When anything is held back, one `<system-reminder>` block follows the skill
catalogue in the system prompt: a sentence saying the tools can be loaded at any
time and how — `ToolSearch(group="<name>")` for a whole group, `select:` and
exact names for particular tools, or a description — a sentence saying that a
dedicated tool beats a workaround with a general one, and one line per held-back
group with its description and its **exact** tool names — or, for a group
declared by prefix with more than `tool_catalogue_max_listed_names` held-back
tools, the exact prefix and a count:

```text
- browser: Drive a web browser. Tools: BrowserClick, BrowserOpen
- scheduling: Timed and recurring jobs. Tools: IntentCreate, ScheduleCreate

Tools of connected servers:
- github: GitHub issues and pull requests. Tools: Mcp_Github_* (26 tools)
```

The host's own groups come first, and the `dynamic` groups — connected
servers — after them in a section of their own. Listed together in name order,
a dozen server lines stood ahead of the host's groups, and a model reading from
the top took them for the whole catalogue: it drove a server's browser, or a
shell, and never loaded the group made for the job. A sentence over the host's
groups naming those detours was tried too; models loaded groups more often
under it and did the task no better, so there is none.

Exact names matter: a model that has to guess a name gets its case wrong. The
second sentence matters more than it looks — a model that cannot see a tool
reaches for the nearest one it can (a service started with a shell command, two
edits instead of one multi-edit), and that, rather than a failed search, is the
usual way a held-back tool goes unused. The block is built from the decision, each
section in name order, so it is byte-identical on every request and after a resume. When
nothing is held back and no group carries rules, no block is emitted, and the
prompt is exactly what it would be without groups.

### Group rules

A group may carry `instructions`: rules for using its tools — which browser to
drive, what to ask before acting. They are given once per run, where the
group's tools first come in front of the model, and never for a group the run
does not touch:

- **In the catalogue**, for tools there from the start: a group left on the
  surface, and a held-back group some of whose tools the run begins with loaded
  (a host's seed, or a resumed snapshot). A held-back group's rules sit under
  its line, indented; the rules of a group on the surface follow the list. A
  block with rules and nothing held back has no header:

  ```text
  - browser: Drive a web browser. Tools: BrowserClick, BrowserOpen
    Rules for the browser tools:
    Ask the user before submitting a form.
  ```

  The catalogue is built once per run and is part of the cached prefix, so
  groups loaded mid-run are not added to it on every load; after a compaction,
  where the prefix starts over anyway, it is written again with the rules of
  every group still loaded, because the result that gave them may now be in
  the summary.
- **In the `ToolSearch` result** that loads the first tools of the group:
  `Rules for the <group> tools:` and the text, after the tool lines.
- **Instead of running a blind call** (see
  [below](#calls-of-tools-that-were-not-advertised)).

The loop keeps the set of groups whose rules were given
(`protocore.tool_group_rules_given` in the tool's metadata, and
`tool_group_rules_given` in the snapshot), so none is given twice.

### `ToolSearch`

`protocore/tools/tool_search.py` ships the discovery tool:
`ToolSearchTool(registry)`, named `ToolSearch`, carrying the role
`discovers_tools` in its own `tool_roles`, `always_load`, and concurrent-safe.
The loop advertises it **only while something is held back**; with the whole
catalogue on the surface it has nothing to find, and a model offered a search
it does not need spends turns on it.

- `query` in free text returns up to `tool_search_max_results` matches, best
  first, one line each — `Name(param1, param2*) — first sentence`, required
  parameters starred — and loads the first `tool_search_autoload_count` of them.
  A model that searches finds the tool it needs among the first three nearly
  every time.
- `select:Name1,Name2` loads exactly those tools (a name in the wrong case is
  still accepted). An unknown name is answered with its nearest admitted names.
  The names may also come as their own argument, `select`, a list or a
  comma-separated string: models reach for that spelling unprompted, and
  refusing it only costs them a turn. Names win over a description sent beside
  them.
- `group` loads every admitted tool of one group, named exactly (a name in the
  wrong case is still accepted); `select` also takes `group:<name>` entries
  beside tool names. An unknown group is answered with the groups the run may
  use. The result lists the group's tools one line each, like any load.
  Tools loaded as a whole group are **one entry** under `pinned_tool_max_count`
  and are unloaded together: a model that asked for a group cannot notice that
  eviction left half of it, and a group of twelve against a cap of fifteen
  would otherwise crowd out everything else.
- The first tools of a group with rules that a call loads bring the rules, once
  (see [Group rules](#group-rules)).
- The first lines of the result say which tools are now loaded and callable,
  and, apart from those, which of the requested or matched tools were **already
  in the tool list**. The loop tells the tool what the calling request
  advertised (`ADVERTISED_TOOLS_METADATA_KEY`, stamped after the operator's
  envelope is merged so it cannot be forged). Reporting a tool the model had
  all along as "Loaded" misleads: a model asked to use a skill searched for it,
  was told "Loaded: WebSearch", took the skill for loaded and never opened it.
  The description says so too — the tool loads tools only, and a skill is not
  one.
- The live visibility policy is read from `ToolContext.metadata`
  (`TOOL_VISIBILITY_POLICY_METADATA_KEY`, `protocore.tool_visibility_policy`),
  and a child run's declared tool set beside it (`TOOL_ALLOWLIST_METADATA_KEY`),
  so the search never lists, suggests or loads a tool the dispatch gate would
  refuse, and a blocked name is never offered as "close". The dispatcher
  assigns both on every call rather than setting them when absent, so a value
  already in the bag never stands in for the one the gate enforces. With no
  policy in the bag the search admits nothing.

The tool only ranks and reports. It names what it loaded under
`TOOLS_LOADED_METADATA_KEY` in its result's metadata — and the groups it loaded
whole under `TOOL_GROUPS_LOADED_METADATA_KEY`, the groups whose rules it gave
under `TOOL_GROUP_RULES_METADATA_KEY` — and the loop — which owns
the surface — loads them, believing that key only from a tool in the
`discovers_tools` role and only for names the policy (and a child's declared
tool set) admits. The tunables are read off the run's constants on
`ToolContext.run_state.rc`; an engine built without a host run state carries its
own constants there.

### Discovery order and the prompt cache

Loaded tools are **appended** to the end of the advertised list, in the order
they were discovered, and stay there. The base surface stays name-sorted and
does not move when a tool is loaded, and a loaded tool does not move when
another one is, so a prefix-caching provider re-reads only the tail: one loaded
schema costs on the order of a couple of thousand uncached tokens, not the whole
prefix. A tool the run searched for that is already on the surface is not loaded
again.

The run keeps loaded tools in discovery order with each one's last use
(`ContextManager.discover_tool`, `note_tool_used`). `pinned_tool_max_count` caps
them, but the cap is applied — least recently used first — only where the prompt
prefix starts over anyway: after compaction, at a turn boundary (`rearm`), and
when a run starts. Unloading a tool in the middle of a run would pull a schema
out from under a model that may be about to call it, and would cost the cache
exactly what the cap exists to save. The one exception is a provider limit:
when base plus loaded tools would exceed `max_advertised_tools`, the least
recently used loaded tools are left off that request (they stay loaded).

The loaded tools and the held-back groups travel in the snapshot
(`discovered_tools`, `deferred_tool_groups`), and a resumed run keeps the
decision instead of measuring again — as a floor: a dynamic group the snapshot
does not name (a server that connected since) is still held back, and the
limits are still enforced on top. A host that wants the next run of a session
to start with the same tools passes a list as `QueryEngineConfig.discovered_tools`.
`ContextManager.called_discovered_tool_names()` is the part of the loaded tools
the run actually called, and is the list to carry: a search loads its best few
matches whether or not the model wanted them. A seeded tool counts as called.
Each snapshot row carries `called`; a row without it is taken as called. A row
of a tool loaded as part of a whole group also carries `group`.

A group is carried whole with `QueryEngineConfig.loaded_tool_groups`: each group
named there is loaded as `ToolSearch(group=...)` would load it — only the tools
the run may call, one entry under `pinned_tool_max_count`, newer than the tools
seeded by name, its rules in the catalogue, announced as `tool_group_loaded` with
`via` `seed`. Passing a large group's tools one by one instead makes each of them
an entry, and the cap then pushes out the rest of the seed.
`ContextManager.loaded_tool_group_names()` is the groups the run holds whole. A
seeded group's tools do not count as called, so a host that carries only the
groups a run also called a tool of lets an unused group go after one run.

### Calls of tools that were not advertised

Dispatch checks the visibility policy, not the advertised list, so a call of a
held-back tool by its exact name is served, as it always was. It is also
**loaded**, so its schema is in front of the model from the next request, and
the loop emits `tool_unadvertised_call`.

The one exception is a tool whose group carries rules the run has not been
given. A call of it was written without the schema and without the rules, and
it would act on a guess about exactly what the rules are there to settle. So it
is **not run**: its **whole group** is loaded, as `ToolSearch(group=...)` would
load it (the tools the run may call, one entry under the cap), the group's
rules are marked given, and the call is answered — not as an error, since
nothing failed — with the rules, the names of the group's other tools now
callable, and the called tool's line: "BrowserOpen is loaded now; call it
again. It takes: BrowserOpen(url*) — …". The next call runs. The whole group,
because a job that starts with one of its tools usually needs another next,
and a model that had only the one it called went on without the rest; the
line, because a retry written from memory repeated the wrong arguments. A
second blind call of the same group in the same message waits too, and is
pointed at the first answer rather than given the rules twice. `tool_unadvertised_call`
carries `executed: false` for such a call. Models read an exact name from the
catalogue and call it without loading it often enough that refusing the call
outright would cost them; this costs one step, and only for groups with rules.

A name that is not registered at all is
answered with `unknown tool: 'X'. Did you mean: A, B, C?` — up to three
registered names the policy admits, compared case-insensitively.

A call made that way was written without the schema, and its arguments are a
guess. When it fails on them, the error ends with the tool's line in the
`ToolSearch` form — `It takes: Name(param1*, param2) — first sentence` — so the
retry is right the first time instead of after one more failure. "Fails on its
arguments" is read broadly, because a tool that checks its own arguments reports
a bad one as an error result, not an exception: a dispatch validation error, an
exception that is a `TypeError`, `ValueError` (which includes a pydantic
`ValidationError`) or `KeyError`, or an error result the tool counts as a
failure. A tool that was on the list never gets the line; its schema is already
in front of the model. Nor does a tool the policy or the child's declared set
refuses: the argument checks run before the gate, and a refused tool is never
"loaded now".

### Telemetry

- `tool_group_loaded` — `{group, via, tools}`, once per load that brought new
  tools of a group: `via` is `search` or `select` (a `ToolSearch` query or
  names), `group` (a whole-group load), `direct_call` (a blind call) or `seed`
  (the tools a host started the run with, announced beside the first
  advertisement). `tools` are the tools of the group that load added.
- `tool_surface_advertised` carries `tool_groups`: every group with a tool on
  the would-be surface, `{name, load, state}`, where `load` is the mode the run
  uses and `state` is `advertised`, `deferred` (held back, none of it loaded) or
  `loaded` (held back, some of its tools loaded onto this request).
  `tool_deferral_reasons` gains `lazy`.

### A runaway batch

`max_tool_calls_per_turn` (default 64) bounds the tool calls dispatched from one
model message. The calls past it are each answered with an error and never run,
so every call still has its result and the transcript stays valid. They do not
count as the tool failing — a thousand refused copies of one search would
otherwise trip the circuit breaker on it. More than a thousand parallel calls in
one message has been seen in practice.

### Why per-message clipping is discouraged

`tool_retrieval_top_k` still clips the surface to the tools that best match the
latest user message, but it is off by default and not recommended. It re-ranks
the surface on every user message, so the tool list — and with it the provider's
cached prefix — changes whenever the message does. It ranks the user's words,
not the model's: a Russian message against English descriptions leaves the right
tool out about half the time, and a model that cannot see the tool it needs
tends to invent a name for it. A model searching with `ToolSearch` writes its own
query, usually in English, and finds the tool. Hold groups back instead.

### Schemas as text

Delivering a loaded tool as text in a result, called through a generic
`CallTool(name, arguments)`, would keep the tool list constant. It is not
implemented. Measured against loading, it answered worse on weaker models, which
filled arguments without the provider's constrained decoding, and the cache it
saves on a prefix-caching provider is the few thousand tokens appending already
costs. Doing it properly would also mean rewriting each `CallTool` into the real
call before the permission gate, hooks, circuit breaker and preconditions see
it, and validating arguments against a schema the provider no longer enforces.

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
