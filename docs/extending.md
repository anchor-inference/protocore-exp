# Extending the Core

> Audience: an engineer adding capability to the **pure core** (`protocore/`) or
> wiring a new backend behind it. Scope: the core library (`protocore/`).

The core is a set of `Protocol`s and an immutable ReAct runtime. You extend it
**from the outside** — by implementing a contract, registering a hook, flipping a
`LoopConstants` toggle, or injecting a system-prompt section — **never by
editing the loop**. This page is a decision guide for picking the right seam,
followed by the two hard rules that bound every extension.

For the full per-subsystem detail behind each seam, see
[`architecture.md`](./architecture.md) — the
[extension-point table](./architecture.md#extension-points-the-protocols-the-host-implements),
the [conventions](./architecture.md#conventions), and the per-section
"Extension protocol" notes.

## Pick your seam

There are five seams. They are not interchangeable; each answers a different
question.

| You want to… | Use | Where |
|---|---|---|
| Provide a concrete backend the core only knows as an interface (an LLM, a store, a transport, memory, workspace, skills, …) | **A `Protocol` adapter** | a module in `contracts/`, implemented by the host |
| Observe, deny, transform, or wrap behaviour at a point in the run (tool calls, the provider exchange, session and compaction boundaries) without changing the loop | **A lifecycle registration** | `contracts/middleware.py`, dispatched by `hooks/manager.py` |
| Turn a capability on/off or tune a numeric/string value per tenant | **An `LoopConstants` toggle** | `contracts/runtime_constants.py` |
| Add static guidance/persona/orientation text to the system prompt | **A section of the system prompt** | `QueryEngineConfig.system_prompt_sections` |
| Replace one product decision about a turn — a budget, a nudge, what an empty answer earns | **A turn policy** | `contracts/turn_policy.py`, substituted into `QueryEngine.turn_policies` by name |

Decision order, fastest first:

1. **Is it just a value or an on/off switch?** → RC toggle. No new code path.
2. **Is it static text the model should always see?** → a system-prompt section.
3. **Is it a reaction at a point in the run (observe/decide/transform/around/notify)?** → a lifecycle registration.
4. **Is it a decision the turn driver makes — when a turn ends, what an empty
   answer earns, which ceiling applies?** → a turn policy, substituted by name.
5. **Is it a whole new backend / capability behind an interface?** → `Protocol`
   adapter (in the host).

If none of these fit, you are probably about to modify the loop — **stop** and
re-read [the hard rules](#hard-rules-do-not-cross-these).

---

### Implement a `Protocol` (the host adapter)

**Use when** the core needs a capability it deliberately does not own: talking to
an LLM, persisting runs/sessions/blobs, searching, storing memory or workspace
units, dispatching subagents, transporting tool calls, rendering prompt
templates. The core declares the *shape* as a `Protocol`; the concrete
implementation lives in the host (or any other consumer) and is
**injected** across the boundary.

There is **no single `protocols.py`** — each interface is in its own module under
`contracts/`. The principal interfaces (one row per seam, full list in the
[extension-point table](./architecture.md#extension-points-the-protocols-the-host-implements)):

| Protocol | Module |
|---|---|
| `ILLMProvider` / `IProviderChain` | `contracts/llm.py` |
| `RuntimeConstantsProvider` | `contracts/runtime_constants.py` |
| `ISessionStore` / `IRunStore` | `contracts/session.py` / `contracts/run.py` |
| `IBlobStore` / `ISearchIndex` / `ITodoStorage` | `contracts/blob.py` / `contracts/search.py` / `contracts/todo.py` |
| `IToolRegistry` / `Tool` (ABC) | `contracts/tool_registry.py` / `contracts/tools.py` |
| `IToolTransport` | `contracts/resilience.py` |
| `IMemory` / `IWorkspace` | `contracts/memory.py` / `contracts/workspace.py` |
| `ISkillStore` | `contracts/skills.py` |
| `IHookManager` | `contracts/hooks.py` |
| `IEventStream` | `contracts/events.py` |
| `IAgentDispatch` / `IDelegationTool` | `contracts/agent_dispatch.py` |
| `IBackgroundTaskPool` / `IWorkPool` | `contracts/background.py` |
| `IPromptTemplateProvider` | `contracts/prompts.py` |
| `IToolSafetyPolicy` | `runtime/tool_permission.py` |

**One pool holds both kinds of work.** A host that implements `IWorkPool`
implements it once and gets both: a background shell command and a delegated
child run are records of different `kind` in the same pool, addressed by the
same id, waited on and stopped through the same `WorkHandle`. Two things about
it are easy to get wrong. `launch` mints the record BEFORE the work starts and
hands the id to your `start` callable, so nothing you spawn is ever running and
unnamed. And `stop_session(scope, grace)` is keyed on ownership, not only on the
session: a record belongs to `scope` when its `owner_scope` is `scope` or its
`session_id` is, which is what lets a delegated run stop what it started without
touching its parent's commands. The grace is real time your handle is expected
to allow the work before forcing it.

A tool that delegates declares `IDelegationTool` rather than carrying a flag,
and the loop recognises delegation by that alone.

**Tools are a special case.** A concrete tool is the one piece you write *as*
code, but it is still registered through the `IToolRegistry`, not wired into the
loop. Implement the `Tool` ABC (`contracts/tools.py`) directly, or use the
`@tool` decorator (`tools/decorator.py`) for an async function:

```python
from typing import Any

from protocore import tool, ToolContext, ToolResult
from protocore import IToolRegistry  # the registry interface


@tool(name="echo", description="Echo back the input text.")
async def echo(context: ToolContext, text: str) -> ToolResult:
    # `context` (tenant_id / run_id / session_id / metadata) is injected by the
    # loop and is read-only. Non-`context` params become the JSON-Schema args.
    return ToolResult(tool_call_id=context.run_id, content=text)


def install(registry: IToolRegistry) -> None:
    registry.register(echo())  # the decorator returns a Tool subclass
```

**A result has one value and several audiences.** `ToolResult.content` is the
canonical value — whole, whatever its size. Three optional fields say how it
should be shown rather than what it is:

* `model_projection` — the text the transcript carries instead of `content`.
  Leave it unset and the value is its own projection, which is right for
  almost every tool. Set it when a shorter rendering is honest: the first page
  of a listing, `wrote 4.2 MB to <path>` for bytes the model gains nothing by
  reading back.
* `ui_payload` — structured detail for whoever is watching. It rides the
  result event and never enters the transcript, so it costs no tokens and
  cannot change what the model decides. Put the rendered table or the per-hunk
  diff here.
* `path` — the file the result is a view of. A result that names its file is
  one the runtime can notice has gone stale: a later call that rewrites that
  file lifts the pin on this result instead of keeping a superseded view in
  front of the model.

```python
return ToolResult(
    tool_call_id=context.run_id,
    content=whole_file,                       # the value
    model_projection=f"{len(lines)} lines",   # what the model reads
    ui_payload={"lines": lines},              # what the client renders
    path=path,                                # what it is a view of
)
```

The decorator derives the `ToolDefinition` (name, description, JSON-Schema
parameters) from the signature and docstring; the special `context: ToolContext`
parameter is skipped when building the schema. Adopt stable, domain-neutral
agent-visible verb names for cross-backend uniformity; see
[`tools.md`](./tools.md).

**Extra safety policies** are also a `Protocol` seam: implement
`IToolSafetyPolicy` and register it on the gate via `register_policy(...)`
(`runtime/tool_permission.py`) — the core gate API stays frozen while the policy
stack grows at runtime (this is how `HttpDnsAllowlistPolicy` and
`WorkspacePathPolicy` are added; they are **not** in the default stack).

**Rule of thumb:** if your code has a database driver, an HTTP client, a wire
format, or any I/O, it belongs in an adapter behind a `Protocol` — not in
`protocore/`.

---

### Use the lifecycle seam

**Use when** you want to react at a point in the run — deny a tool call, rewrite
what the model is shown, observe an event, wrap the work in your own timing or
transport — **without** changing the loop.

There is **one** seam, declared in `contracts/middleware.py`. A registration
names three things and gets a disposer back:

```python
from protocore import HookManager, RegistrationKind
from protocore.contracts.middleware import LifecycleDecision, LifecycleScope, LifecycleVerdict
from protocore.contracts.types import HookEvent

registry = HookManager()
dispose = registry.register(
    HookEvent.pre_tool_use,
    RegistrationKind.decide,
    lambda ctx: LifecycleDecision(verdict=LifecycleVerdict.deny, reason="not that one"),
    owner="my-policy",                       # named in every verdict and log line
    scope=LifecycleScope(tenant_id="acme"),  # optional; unset means "any"
    priority=50,                             # ascending, then registration order
    timeout_s=2.0,                           # awaitable handlers only
)
...
dispose()          # idempotent: the second call is a no-op returning False
```

The registry is handed to the engine at construction
(`QueryEngine(..., lifecycle_hooks=registry)`) and runs when
`typed_hooks_enabled` is on. That flag is the only switch over the seam: no
coordinate is hidden behind a second, unrelated one.

**Five kinds, and what each may do.**

| Kind | May change | On failure |
|---|---|---|
| `observe` | nothing | logged, run untouched |
| `decide` | the verdict (`allow` / `deny` / `require_approval` / `fail_run`) | **denies** |
| `transform` | the payload, which is then applied | **denies** |
| `around` | wraps the work; may skip it entirely | **denies** |
| `notify` | nothing, after the fact | logged, run untouched |

The split is deliberate. A seam that exists to answer whether something may
happen has not said yes when it crashes or times out, so `decide`, `transform`
and `around` fail **closed**. A seam that only watches must never be able to
change an outcome, so `observe` and `notify` fail **isolated** — the failure is
logged and recorded in `LifecycleOutcome.failures`, and nothing else moves.

**The coordinates** are the members of `HookEvent`, and every one of them is
dispatched by the loop: `run_start`, `turn_start`, `context_transform`,
`request_prepare`, `response_received`, `request_error`, `turn_end`,
`pre_tool_use`, `tool_execute`, `post_tool_use`, `pre_compact`, `compaction_commit`,
`compaction_rollback`, `post_compact`, `run_finalize`. `user_prompt_submit`,
`session_start`, `session_end`, `file_changed`, `subagent_start` and
`subagent_stop` reach the host's own `IHookManager` (below), which the loop
drives at the permission gate and the dispatcher.

A `transform` at `context_transform` is applied: it may return
`system_prompt_sections` (a list of strings) and `active_language`, and the
turn's provider request is rebuilt from what it returned. Fields it does not
name, and fields of the wrong shape, leave the bundle as it was.

**The out-of-process path.** `IHookManager` (`contracts/hooks.py`) is the
contract for hooks whose executor is not in this process — an HTTP endpoint, a
model asked to judge. The host supplies that adapter, the loop calls its 3-arg
`invoke(event, payload, tenant_id) -> HookResult`, and its `HookActionKind`
`ALLOW` / `DENY` / `MODIFY` map onto the verdicts above. It is driven at the
permission gate and around tool dispatch, and it fails closed there for the
same reason: an executor that could not be reached did not authorise anything.

`tool_execute` is the coordinate the tool dispatcher wraps its invocation in,
and the only place an `around` registration straddles a call: the handler is
entered before the tool runs and resumes after it returned, so one handler
holds the whole call — a timer, a circuit breaker, a recorder. A handler that
never awaits its `next` skips the tool, and that is read as a refusal rather
than as a silent success: the call comes back as a permission failure naming
the owner, because reporting an answer nothing produced is worse than saying
no. Raising, timing out, or answering unreadably refuses the same way, and the
tool's own failure is re-raised as itself, so a wrapper cannot turn a timeout
into a result by swallowing it.

The single most useful coordinate is `pre_tool_use`: it is the permission
gate's **final stage** (gate order: `whitelist` → `safety_policy` →
`rate_limit` → `hook`, in `runtime/tool_permission.py`), the highest-leverage
point where an otherwise-allowed call can be flipped to deny or parked for
approval. The gate as a whole runs before preconditions and execution
(`runtime/tool_dispatch.py`).

---

### Compose the run's state and declare what your tools do

**Use when** you are embedding the loop at all: these two are not optional
extras, they are the pair every host supplies.

`RunScopedState` (`contracts/run_state.py`) is what one run carries while it
executes. The host builds it at run start, puts its own per-run slots in the
`host` compartment — which this package never reads or writes — and hands it to
the engine. Every tool invocation then reaches it as `ToolContext.run_state`, and
a child run inherits from it by reference the things a whole tree shares.

```python
from protocore.contracts.run_state import RunScopedState

state = RunScopedState(
    rc=constants,                      # the constants snapshot for this run
    cancel_event=asyncio.Event(),      # what your cancel path sets
    host={"workspace": workspace},     # your own slots; the loop ignores them
)
engine.run_state = state
```

`ToolRoleMap` (`contracts/tool_roles.py`) says what your tools DO. The loop
never reads a tool's name to decide anything: it asks the map. Declare it once
where the tools are registered — a role per tool, and the argument spellings each
value arrives under, because the runtime sees arguments as the model emitted
them, before your tool's input model has resolved an alias.

```python
from protocore.contracts.tool_roles import ToolArgumentSlot, ToolRole, ToolRoleMap

roles = ToolRoleMap.declare(
    {
        "put_file": [ToolRole.writes_path],
        "open_file": [ToolRole.reads_path],
        "spawn": [ToolRole.delegates_work, ToolRole.never_delegated],
    },
    argument_aliases={ToolArgumentSlot.path: ["path", "file_path"]},
)
```

An undeclared role is a behaviour that goes quiet: a writer the map does not
name stops lifting stale pinned reads, and a shell tool whose command spelling is
undeclared sends its calls for approval rather than running them unexamined.

Say, too, who wrote what you put into the history. The core takes a plain
user-role message to be the operator's; a host that delivers its own events on
that channel, or the operator's replies on another, marks them with
`OPERATOR_WORDS_METADATA_KEY`, so compaction quotes the operator's words whole
in its ledger and compacts the rest like any other message (the table is in
[the compaction contract](compaction.md#who-wrote-a-message)):

```python
from protocore.contracts.types import OPERATOR_WORDS_METADATA_KEY, Message, MessageRole, TextBlock

answer = 'the operator answered request [a1b2] "Which port?": 8443'
note = Message(
    role=MessageRole.user,
    content_blocks=[TextBlock(text=f"2 events:\n- build 412 finished\n- {answer}")],
    metadata={OPERATOR_WORDS_METADATA_KEY: [answer]},  # True: all of it; False: none of it
)
```

---

### Add a `LoopConstants` toggle

**Use when** the change is a tunable value or an on/off switch — not new control
flow. Every tunable in the core flows through `LoopConstants`, a **frozen
Pydantic snapshot** with `model_config = ConfigDict(frozen=True, extra="forbid")`
(`contracts/runtime_constants.py`). **No inline magic numbers** — runtime code
reads from the snapshot, never a hard-coded literal.

New capabilities default **off** (or to a value that reproduces prior
behaviour), so a tenant opts in deliberately. Adding a tunable is **one edit**:
add the field, with its default, bounds and an operator-facing `description`, to
the model whose layer reads it. `LoopConstants` is that model for a threshold
the loop reads; a knob a surrounding layer reads belongs to that layer's own
model and its own constant group. Either way the group is reflected from the
model (`group_from_model`, `contracts/config.py`), so the operator catalogue
picks the field up without a second registration to keep in step.

Because `extra="forbid"` rejects unknown fields, **core and the host must
deploy paired** — a field added on one side without the other will reject the
snapshot. See [`runtime-constants.md`](./runtime-constants.md) for the full
model, the registry and the three visibility states a knob can have.

If your "feature" is really "let a tenant turn X on/off" or "let a tenant set the
threshold for Y", the field is the whole change — no other seam is needed.

---

### Inject a system-prompt section

**Use when** you want to add static text to the model's system prompt — a
persona, domain guidance, an orientation block — that should always be present
for a run. These are passed at engine construction via
`QueryEngineConfig.system_prompt_sections` (a `tuple[str, ...]`,
`runtime/query_engine.py`); the loop assembles them into the system message each
turn.

This is the right seam for *prompt content*. It is **not** the right seam for
behaviour that should react to events (use a lifecycle registration), for a value that should be
tenant-configurable (use an RC toggle), or for dynamic per-turn orientation read
from the environment (that is the `context_bootstrap_*` RC path, see the
[Hooks](./architecture.md#the-lifecycle-seam--injection--scratchpad--context_bootstrap)
section).

Several adjacent injection points on `QueryEngineConfig` are worth knowing —
they are **injected callables, maps and observers**, not prompt text:
`cache_observer` (`CacheObserverProtocol`, the prompt-cache hit-rate sink),
`request_manifest_sink` (`IRequestManifestSink` — where the record of what was
sent to a provider is kept), `resilience_classifier` (`IResilienceClassifier` —
which neutral failure class a message describes), `tool_roles` (`ToolRoleMap` —
what the host's tools DO, and the argument spellings they do it with) and the
two terminal-verify trigger callables (`pre_terminal_self_verify_trigger`,
`pre_dispatch_terminal_verify_trigger`). All default `None` / empty, so the
machinery behind each is inert until the host injects one — and, for the
RC-gated ones, flips the matching field too.

---

### Substitute a turn policy

**Use when** the change is a product decision the turn driver makes: that this
run has spent its budget, that an empty answer earns one more try, that a file
left half-written must be sealed before the run may finish. Before the seam
existed, each of those was a branch grown inside the driver, and there was no
way to change one without editing the loop.

A policy is an object with a `name`, the `coordinates` it wants to be consulted
at, and one `apply(turn)` that yields the events the loop forwards and writes
what happens next to `turn.outcome` — `proceed`, `restart_turn` or `end_turn`.
It reads and changes the run through `ITurnState`, a deliberately small
structural view: a policy that needs something not named there is reaching into
the loop's insides, and the review that adds the name is where that gets
noticed.

```python
engine.turn_policies = core_policies.merged_with(
    TurnPolicyRegistry((OurRunCeilingsPolicy(...),))
)
```

Two properties are the point of the seam. The **order is the core's**, declared
once in `TURN_POLICY_ORDER`, because the order matters where two policies meet
and an order assembled by whoever built the list would make that a coincidence;
a policy whose name is not in that tuple is refused at construction rather than
silently running last. And a set is **merged by name**, never put in place: your
policy displaces the core policy answering to the same name, and every bound
nobody named stays where it is. Replacing the whole registry would take the
core's own ceilings with it, which is exactly the failure the merge prevents.

---

## Hard rules (do not cross these)

Two rules bound every extension. Violating either is a process failure, not a
design choice.

### Do not modify the loop structure

The ReAct loop (`runtime/query.py` + `runtime/query_engine.py` +
`runtime/loop_state.py`) is **immutable**. It is the single consumer of every
other subsystem and the basis for cross-pod snapshot/resume. **Do not edit its
structure.** Customise only via the four seams above:

- **hooks** (deny/modify/observe at lifecycle points),
- **`QueryEngineConfig`** injected callables/observers,
- **`LoopConstants`** toggles,
- **`system_prompt_sections`**.

If a change seems to require new branches in the turn loop, that is a signal you have
picked the wrong seam — re-check the [decision order](#pick-your-seam). Recovery
behaviour, terminal classification, and finalization are all already
RC-gated; you toggle them, you do not rewrite them.

### Never import upward (the import boundary)

The core is the **root** of the dependency graph and must never import upward.
`protocore/` must never import any package whose name begins `protocore_` —
that underscore is what marks a sibling distribution sitting above the core:
adapters, a service layer, frontends, an execution backend, deployment tooling.

Add behaviour through **contracts / adapters / RC**, not by reaching upward. If
the core needs something from a higher layer, express that need as a new
`Protocol` and let the host inject the implementation.

This is enforced by the guard test
[`tests/test_core_import_boundary.py`](../tests/test_core_import_boundary.py).
It walks every `*.py` file under `protocore/`, parses the AST, and **fails CI**
if any top-level `import` / `from … import` references a forbidden package. Run
it (and the rest of the suite) before you push — see [`testing.md`](./testing.md).

---

## Related docs

- [`architecture.md`](./architecture.md) — full per-subsystem detail (the
  [extension-point table](./architecture.md#extension-points-the-protocols-the-host-implements),
  the [conventions](./architecture.md#conventions), per-section "Extension
  protocol" notes).
- [`contracts.md`](./contracts.md) — the interface surface you implement against.
- [`runtime-constants.md`](./runtime-constants.md) — the RC model and 3-edit rule.
- [`tools.md`](./tools.md) — the tool surface and the `@tool` decorator.
- [`testing.md`](./testing.md) — running the suite and the import-boundary guard.
