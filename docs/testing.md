# Testing the Core

How to test, lint, and type-check the pure core (`protocore`). All commands
assume [`uv`](https://docs.astral.sh/uv/) and are run from the repository root.

## Setup

Install the package together with its development dependencies (pytest,
pytest-asyncio, pytest-cov, pytest-xdist, ruff, mypy, bandit):

```bash
uv sync --extra dev
```

A host that only wants to run the conformance suites against its own adapters
installs far less — see [Conformance suites for a host](#conformance-suites-for-a-host).

## The four gates

The core is verified by four independent gates. Run all four before proposing a
change; CI runs the same commands.

```bash
uv run pytest .              # tests (+ coverage)
uv run ruff check .          # lint + import sorting
uv run mypy --strict         # static type-check
uv run bandit -r protocore -q -c pyproject.toml   # security scan
```

Notes:

- **Tests** — `pytest` discovers from `tests/` (configured by
  `testpaths = ["tests"]` in `pyproject.toml`). `asyncio_mode = "auto"`, so an
  `async def test_*` runs without a per-test `@pytest.mark.asyncio` marker.
  `pytest-xdist` is installed, so `-n 8 --dist loadscope` runs the suite in
  parallel; `--dist loadscope` keeps a module's tests on one worker, which
  `--dist load` does not.
- **Lint** — `ruff` is configured with `line-length = 120`,
  `target-version = "py312"`, and the `E,F,W,I,B,UP,RUF` rule sets.
- **Type-check** — run `mypy` with **no path argument**. A path on the command
  line replaces the `files` list configured in `pyproject.toml` and silently
  drops the test tree, so the check passes while checking less than it should.
- **Security scan** — `bandit` over the package, with the configuration in
  `pyproject.toml`. Never pipe a gate into `tail` or `head`: a pipeline reports
  the *last* command's status, so a failing gate reads as a pass.

## Test layout

Tests live at the **repository root under `tests/`**, *not* inside the
`protocore` package. The package ships no `test_*.py` files of its own — keeping
runtime code and test code physically separate.

```
tests/
  conftest.py                      # shared fixtures (e.g. tenant_id)
  test_core_import_boundary.py     # import-boundary guard (see below)
  test_constructible.py
  test_runtime_constants_sse.py
  unit/
    contracts/                     # contract types + protocol shape tests
    runtime/                       # loop, dispatch, context, resilience, ...
    tools/                         # @tool decorator, ask-user, memory
    prompts/
    test_*.py                      # token counting, chain parser, JSON utils, ...
```

The suite is fast and uses minimal mocking: the core has **no external service
dependencies** (no database, no HTTP, no Kubernetes — those live behind the
contracts that the host implements), so nothing needs to be stood up to run
the tests.

## Import-boundary guard test

`tests/test_core_import_boundary.py` enforces the central architectural
invariant: **the core never imports upward.** It AST-parses every `.py` file in
the `protocore` package tree and fails if any top-level import names a package
that sits above the core.

The rule is the namespace, not a list. Everything the core may import from
itself lives under the single `protocore` package; a host puts its adapters,
service layer, and frontends in sibling distributions whose import names all
begin `protocore_`. Naming the shape instead of enumerating today's siblings
means a package added tomorrow is caught on the day it appears.

In other words, importing any `protocore.*` module must pull in **zero**
symbols from the layers above. Outside-facing capabilities are
`Protocol`s the core defines and the host implements (see
[`contracts.md`](contracts.md)); you add behaviour through contracts, adapters,
or `LoopConstants`, never by importing upward (see
[`extending.md`](extending.md)). A violation fails CI immediately and reports
each offending `file: imports 'package'`.

## Conformance suites for a host

The core declares its dependencies as `Protocol`s and never sees the objects
that satisfy them until a run calls one. `protocore.conformance` moves that
moment earlier: a host imports the suite for a contract, binds it to its own
adapter with a factory fixture, and finds out in its own test run — rather than
in an agent's turn — that the adapter has the shape the core will call.

```bash
pip install "protocore[testing]"     # the suites need a test runner, nothing else
```

```python
from protocore.conformance import LLMProviderConformance

class TestOurProvider(LLMProviderConformance):
    @pytest.fixture
    def subject_factory(self):
        return lambda: OurProvider(...)
```

The factory, not an instance, is the parameter: a suite may want more than one
subject, and building each inside the test keeps one case's state out of the
next. What a suite checks is deliberately structural — presence, asynchrony,
and the parameters the core passes by name — because behaviour is the host's
own suite's job, and a core asserting it would be dictating storage semantics
it has no business knowing.

Two suites add behavioural cases because their contract's promise cannot be
read off a signature: `RequestManifestSinkConformance` (the handoff must return
promptly) and `ToolOutcomeStoreConformance` (the first recorded outcome wins, a
late start never unsettles a finished call, records are kept apart by run and
tenant, `discard` forgets only what it names). Their cases keep to run ids of
their own, so they pass against a store that every case shares.

`SUITES` is every suite the package publishes, one per contract, and the
package's own test module fails if a contract is declared without one — so the
catalogue cannot drift behind `contracts/`.

A suite a host imports but never binds **skips**, which means a conformance
directory can pass while asserting nothing. `bound_suites` is how a host tells
the two apart: point it at its own conformance module and compare the result
against the contracts it implements, so forgetting to bind one is a failure
rather than a silent skip.

## Coverage stance

Coverage is collected by `pytest-cov` against `source = ["protocore"]` with
`branch = true` (see `[tool.coverage.run]` / `[tool.coverage.report]` in
`pyproject.toml`; `__init__.py` files are omitted and Protocol-stub/`overload`
bodies are excluded from the denominator).

The core holds a **strict minimum coverage threshold** — most new capabilities
ship default-off, so their *enabled* paths are exercised by unit tests rather
than by a live run, and the threshold guards against silently shipping
unexercised branches. CI enforces it by running the suite with
`--cov-fail-under=90`; a change that drops coverage below 90% fails the
pipeline.

## What to run before a change

Run all four gates locally before proposing a change — they are exactly what CI
runs:

```bash
uv sync --extra dev          # once, to install the dev toolchain
uv run pytest .              # tests + coverage (must meet the threshold)
uv run ruff check .          # lint + import sorting
uv run mypy --strict         # strict static type-check
uv run bandit -r protocore -q -c pyproject.toml
```

The import-boundary guard (above) runs as part of `pytest`, so a reverse import
into a sibling package fails the test run immediately.
