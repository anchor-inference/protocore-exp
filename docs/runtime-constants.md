# LoopConstants

`LoopConstants` is the single configuration surface that flows across the
core ↔ the host boundary. It is the mechanism behind the project's hard rule:
**no inline magic numbers.** Every runtime-tunable threshold — token-budget
fractions, operational caps, timeouts, feature kill-switches — is a typed field
on one frozen Pydantic snapshot that is default-safe and dashboard-configurable.
Runtime code reads from the snapshot; it never embeds a literal.

This page is the conceptual reference for that model. The operational day-to-day
(the dashboard flow, override precedence, and anti-patterns) lives with 
the host service and its administration dashboard, which read and persist the
per-tenant overrides this model describes. The deep architecture treatment is in
the [LoopConstants system](architecture.md#loopconstants-system) section of
[`architecture.md`](architecture.md).

## The model: frozen Pydantic, `extra="forbid"`

`LoopConstants` lives in `protocore/contracts/runtime_constants.py`. It is a
`pydantic.BaseModel` whose configuration is:

```python
class LoopConstants(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    ...
```

Two properties matter, and both are deliberate.

- **`frozen=True`** — a snapshot is immutable once constructed. The runtime
  binds one snapshot at engine construction — it is stored on
  `QueryEngineConfig.rc`, and the loop reads it from `engine.config.rc` —
  so a run's behaviour cannot drift mid-flight even if an operator edits a value
  while the run is in progress. There is no global state and no module-level
  cache; the snapshot is the authority for the lifetime of the engine. New
  values apply on the next run, never retroactively.

- **`extra="forbid"`** — an unknown field is a hard validation error, not a
  silently-ignored extra. The practical consequence: **core and the host must
  deploy paired.** The host builds each snapshot from per-tenant data and
  hands it to the core; if the host sends a field the deployed core model does
  not declare (or vice-versa), construction rejects rather than swallowing the
  mismatch. A core/the host version skew therefore fails loudly at the
  boundary instead of corrupting behaviour. Treat the core and its host as a single
  release unit when adding or renaming fields.

All fields are **canonical inputs only**. Formula-derived values (for example
token budgets and the compaction trigger in tokens) are *computed* from these
inputs elsewhere in the runtime — they are not stored as fields. Adding a
derived value as its own field would invite two sources of truth.

## The registry: groups, specs and who owns a knob

The snapshot is not the whole tunable surface — it is **one group of it**. A
deployment's knobs are a set of groups, each declared by the layer that
actually reads its values: the loop's own thresholds are the core's group, and
everything a surrounding layer reads is declared by that layer. `contracts/config.py`
owns the vocabulary both sides speak.

- **`ConstantSpec`** — one knob's descriptor: its wire `kind`
  (`int` / `float` / `bool` / `str` / `json`), its `default`, its
  operator-facing `description`, its bounds (`minimum` / `maximum`,
  `exclusive_*`, `allowed_values`, and `zero_means_unlimited` — the one piece of
  semantics a numeric bound cannot carry, a floor of zero that means "no ceiling
  at all" rather than "the smallest ceiling"), the `group` and `owner` it
  belongs to, and its visibility. Visibility has **three** states, not two:
  `editable=True` (a row an operator may set), `editable=False` (a row that is
  shown and refused on write) and `not_a_lever="<reason>"` (**no row at all** —
  a value the running system derives or owns outright, whose appearance in an
  editor would be an invitation to break the deployment).
- **`ConstantGroup`** — a set of specs with one `owner` and one `key`, plus the
  `invariants` relating its own constants. `group_from_model(model, key=...,
  owner=...)` reflects a declaring model into a group rather than restating it:
  a hand-written list of hundreds of names drifts from the model it describes on
  the first field anyone adds, and reflection cannot. `build_loop_group()` is
  that call for `LoopConstants` itself.
- **`IConstantsRegistry`** — declaration (`declare`), fail-closed resolution
  (`resolve` raises `UnknownConstantError` for a name no group declares; it
  never resolves to a default), `defaults`, `coerce` (a store keeps every value
  as text, and both the write path and the snapshot build must read it the same
  way) and `repair` (a bad stored row is reset by name and reported, because
  discarding the whole set would take every unrelated setting of the scope down
  with it). A name claimed by two **owning** groups is a `DuplicateConstantError`;
  a name claimed by an owning group and by a `provisional` one — the stand-in a
  layer keeps while it hands ownership over — goes to the owner, and the
  displacement is recorded.
- **`ICoreConstantsProvider`** — how the loop asks for the snapshot in force for
  one scope. The snapshot is always passed by value into a turn: no global
  state, no module-level cache, and freshness belongs entirely to the
  implementation.

Two facts about a knob deliberately live elsewhere, because a second
declaration of one fact drifts from the first: whether changing it requires a
restart, and which capability toggle it belongs to. Both are properties of the
place that consumes the value, and they are joined onto the catalogue as an
overlay rather than restated in the descriptor.

## The provider: `RuntimeConstantsProvider`

Core does not know how snapshots are built or where tenant overrides live — that
is the host's job, expressed through a Protocol that core defines and
the host implements:

```python
@runtime_checkable
class RuntimeConstantsProvider(Protocol):
    async def get(self, tenant_id: str) -> LoopConstants:
        """Return the latest snapshot for ``tenant_id``."""
        ...
```

The production implementation (in the host) reads per-tenant overrides from
Postgres, caches them in Redis, watches Redis pub/sub for invalidation, and
rebuilds a fresh frozen snapshot whenever an override changes. The core only
ever sees the result of `get(tenant_id)`: a ready-to-use, immutable snapshot.
This keeps the persistence and caching machinery entirely on the host side
of the boundary.

## Defaults and the in-memory provider

Every field carries a **default-safe, default-off** value defined inline on the
field. The default is the single source of truth for that value — that is
precisely *why* it lives on the field and not as a scattered literal. New
behavioural surfaces ship behind a boolean that defaults to `False` (or a cap
that defaults to a conservative value), so deploying the code does not change
behaviour until an operator opts in per tenant. Feature kill-switches default
`True` only when the feature is the established steady-state path and the switch
exists for incident rollback.

Read the live `Field(...)` default — do not infer it from older docs. Intent,
ledger, lanes, typed hooks, telemetry, manual compact and steer/follow-up are
default-**off**: `intent_settlement_enabled`, `usage_ledger_enabled`,
`lanes_enabled`, `typed_hooks_enabled`, `telemetry_spans_enabled`,
`compaction_manual_enabled`, `steer_follow_up_enabled`.

The snapshot carries only what the loop itself reads. Knobs that govern a
surface the host owns — authentication policy, session storage, the transport
to a provider — are declared by the host in its own model and reach the
operator through the same catalog; they are not fields of `LoopConstants` and
looking for them here will not find them.

For tests and the in-memory smoke runtime — anywhere there is no Postgres-backed
provider — core ships two helpers in `protocore/runtime/runtime_constants.py`:

- `default_runtime_constants(**overrides)` — returns a `LoopConstants`
  built entirely from field defaults, with optional keyword overrides for the
  fields a test needs to vary.
- `StaticRuntimeConstantsProvider` — a `RuntimeConstantsProvider` that returns
  one fixed snapshot for every `tenant_id`. Useful as the provider in a
  single-tenant smoke run or unit test.

Both are re-exported from the package top level (`from protocore import
default_runtime_constants, StaticRuntimeConstantsProvider`). Production pods do
**not** use these; they wire in the Postgres + Redis provider described above.

## Static caps in `constants.py`

`LoopConstants` is for values that should be tunable per tenant through the
dashboard. A small, separate set of values must **never** vary per scope: memory
safety ceilings and protocol identifiers. Those live as module-level constants
in `protocore/constants.py` — for example `MAX_TOOL_CALL_ARGUMENT_BYTES`,
`MAX_ARTIFACTS`, `MAX_STRUCTURED_JSON_CHARS`, `PROTOCOL_VERSION`, and
`DEFAULT_MODEL`. These are non-negotiable backstops enforced regardless of what
a snapshot says; they are not exposed for dashboard editing because making them
tunable would let a misconfiguration defeat a safety bound.

The decision rule:

- **Should an operator be able to tune it per tenant?** → `LoopConstants`
  field.
- **Is it a hard safety ceiling or a protocol/identity constant that must hold
  everywhere?** → `constants.py`.

A value belongs to exactly one of these. It is never both, and it is never an
inline literal in runtime logic.

## Adding a tunable

**Add the field to the model whose layer reads it.** For a threshold the loop
reads, that is `LoopConstants` in
`protocore/contracts/runtime_constants.py`, with a `Field(...)` declaration
carrying a default-safe/off value, validation bounds where applicable, and a
`description` written for an operator. For a knob a surrounding layer reads,
it is that layer's own model.

Nothing else in the core has to be told about it. The group is reflected from
the model by `group_from_model`, so the descriptor — type, bounds,
enumeration, default and the field's own words — comes from the declaration
itself, and the operator catalogue enumerates what the registry resolves. Where
a declaration cannot state something about itself (a unit, a category, the
reason a value is `not_a_lever`), a `SpecOverlay` supplies it at the point the
group is built.

Two mistakes the shape of the model prevents. Adding the field to the wrong
layer's model puts a knob in a group whose owner does not read it, and the
first thing a reviewer sees is an owner that makes no sense. Adding it to two
models is a `DuplicateConstantError` at declaration, not a silent race between
two defaults.

## How to read a field default

Each field declares its own default and (where numeric) its validation bounds.
To learn the current behaviour of a tunable, read its `Field(...)` definition in
`protocore/contracts/runtime_constants.py` — the `default=` is authoritative and
the `description=` explains the value's intent and history. That field
definition is the *only* place the value is written down; runtime code reads it
from the live snapshot:

```python
# Correct — read the tunable from the injected snapshot.
if turns >= rc.max_turns_per_run:
    stop()

# Wrong — an inline literal is a magic number. It bypasses the snapshot,
# cannot be tuned per tenant, and violates the no-magic-numbers rule.
if iteration >= 50:
    stop()
```

The literal in the "wrong" example is exactly what the model exists to
eliminate: the bound lives on the `max_turns_per_run` field, not in the
branch.

## Knobs that are tuned together

Most fields stand alone: one threshold, one effect. The tool-retrieval ranking
fields do not. `tool_retrieval_name_weight`,
`tool_retrieval_search_hint_weight`, `tool_retrieval_summary_weight`,
`tool_retrieval_description_weight`, `tool_retrieval_parameters_weight`,
`tool_retrieval_bm25_k1` and `tool_retrieval_bm25_b` were chosen as one set, by
cross-validation over labelled English and Russian queries, and moving one
changes what the others should be. `tool_retrieval_lexicon_weight` switches the
Russian-to-English query expansion (0 turns it off); with it off, a hint weight
of 2 is the better fit. `tool_retrieval_fusion_rank_constant` matters only when
a host supplies its own ranker. The loop passes all of them to the registry as
one `RetrievalSettings` value per call, and the registry keeps an index per
distinct value, so a tenant with its own weights gets its own index rather than
another tenant's. What the fields weigh is described in
[`tools.md`](tools.md#tool-retrieval).

The tool-deferral fields are a second such set. `tool_deferral_mode` decides
whether declared tool groups may be held back at all; `tool_definitions_ratio`
(default a quarter of the window, and now enforced) and `max_advertised_tools`
(0 for no limit) are the two limits a surface is held to; `pinned_tool_max_count`
caps the tools a run keeps loaded, and is also the headroom the count limit
leaves for them. `tool_search_max_results` and `tool_search_autoload_count` shape
a `ToolSearch` result, `tool_catalogue_max_listed_names` decides when a
prefix-declared group is listed by its prefix, and `max_tool_calls_per_turn`
bounds one model message's batch. `tool_retrieval_top_k` is `0` — the
per-message clip is off, and turning it on is not recommended. See
[`tools.md`](tools.md#holding-tool-groups-back).

## See also

- [`architecture.md`](architecture.md) — the LoopConstants system in the
  full core architecture, including where the snapshot is bound to the engine.
- `contracts.md` — the wider contract surface (the interface Protocols
  the host implements), of which `RuntimeConstantsProvider` is one.
