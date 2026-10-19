"""How much a model can hold, found out rather than typed in.

Everything that sizes a run reads the context window: when to compact, how
large a reply may be, whether a fallback provider can take the conversation at
all. A number typed by hand is a question many operators cannot answer, and a
single default for every model is wrong in both directions: a model with a
million tokens compacted at an eighth of its room, a local server started with
8K refusing the first long turn.

:class:`ContextWindowResolver` takes the window from the first of these that
knows it:

1. **The server's own listing**, through a probe the host supplies (the
   reference adapter ships :class:`~protocore.providers.openai_compat.windows.ListingWindowProbe`
   for ``/models``, ``/models/{id}`` and llama.cpp's ``/props``).
2. **A catalogue** in the shape of models.dev's ``api.json``, a file the host
   downloads and passes in (:class:`WindowCatalogue`).
3. **A table** of model-family patterns the host keeps (:class:`WindowTable`).
4. **A refusal.** A server that rejected a request with "maximum context length
   is N tokens" has stated the real limit. It is remembered and, when smaller
   than what the sources above say, wins over all of them.

Nothing is bundled: which models exist and how large they are changes every
month, and a number shipped in a library is wrong long before anyone updates
it. When no source knows, the resolver says ``None`` and the host keeps the
number it has.

This module is pure Python and makes no request of its own; only the probe a
host passes in does.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import math
import re
import time
import tomllib
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from protocore.logging_utils import get_logger

if TYPE_CHECKING:
    from protocore.providers.chain import ChainRung

_logger = get_logger(__name__)

#: The range a context window is believed in. Anything outside it is another
#: quantity in the same field (a price, a page size) or a typo.
_SMALLEST_WINDOW = 1_000
_LARGEST_WINDOW = 100_000_000


class WindowSource(StrEnum):
    """Which source decided a window."""

    provider = "provider"
    catalogue = "catalogue"
    table = "table"
    refusal = "refusal"


@dataclass(frozen=True, slots=True)
class FoundWindow:
    """A window one source reported, with a few words on where exactly it came from."""

    tokens: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ResolvedWindow:
    """The window a model has, and which source decided it."""

    tokens: int
    source: WindowSource
    detail: str = ""


def plausible_window(value: Any) -> int | None:
    """A token count from JSON or text, or ``None`` for anything that is not a plausible window."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int | float) and math.isfinite(value) and _SMALLEST_WINDOW <= value <= _LARGEST_WINDOW:
        return int(value)
    return None


# Refusals -------------------------------------------------------------------

_STATED_WINDOW = (
    # OpenAI, vLLM, DeepSeek, OpenRouter: "This model's maximum context length is 131072 tokens."
    re.compile(r"maximum context length is (\d+)", re.IGNORECASE),
    # Anthropic: "prompt is too long: 210582 tokens > 200000 maximum"
    re.compile(r"tokens?\s*>\s*(\d+)\s*maximum", re.IGNORECASE),
    # llama.cpp: "the request exceeds the available context size (8192 tokens)"
    re.compile(r"available context size \((\d+) tokens?\)", re.IGNORECASE),
    # Google: "... exceeds the maximum number of tokens allowed (1048576)."
    re.compile(r"maximum number of tokens allowed \((\d+)\)", re.IGNORECASE),
    # LM Studio and others: "context length of only 4096 tokens"
    re.compile(r"context length of (?:only )?(\d+) tokens", re.IGNORECASE),
)


def stated_window(text: str) -> int | None:
    """The context window a refusal states, in any of the phrasings servers use; ``None`` without one.

    Wider than the parser the adapter's overflow error reads, which also needs
    the prompt and output sizes: here only the limit matters.
    """
    for pattern in _STATED_WINDOW:
        found = pattern.search(text or "")
        if found:
            return plausible_window(int(found.group(1)))
    return None


# Listings -------------------------------------------------------------------

#: Where each server puts the window in a model entry, most specific first.
#:
#: ``loaded_context_length`` (LM Studio) is the length the server actually
#: loaded, which can be less than ``max_context_length``, the most the model
#: could take; vLLM's ``max_model_len`` is likewise what the server was started
#: with. The rest name the model's published limit (OpenRouter, Together,
#: Groq, Mistral, Anthropic, Google, and models.dev's ``limit.context``).
ENTRY_FIELDS: tuple[tuple[str, ...], ...] = (
    ("loaded_context_length",),
    ("max_model_len",),
    ("context_length",),
    ("top_provider", "context_length"),
    ("context_window",),
    ("max_context_length",),
    ("max_input_tokens",),
    ("inputTokenLimit",),
    ("input_token_limit",),
    ("limit", "context"),
)


def window_from_entry(entry: Any) -> int | None:
    """The window a model-list entry states, whichever server's field it uses."""
    if not isinstance(entry, dict):
        return None
    for path in ENTRY_FIELDS:
        node: Any = entry
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        tokens = plausible_window(node)
        if tokens is not None:
            return tokens
    return None


def listing_rows(payload: Any) -> list[dict[str, Any]]:
    """The model entries of a listing: ``{"data": [...]}``, ``{"models": [...]}`` or a bare list."""
    if isinstance(payload, list):
        rows: Any = payload
    elif isinstance(payload, dict):
        rows = payload.get("data") if isinstance(payload.get("data"), list) else payload.get("models")
    else:
        rows = None
    return [row for row in rows or [] if isinstance(row, dict)]


def entry_id(entry: Mapping[str, Any]) -> str:
    """A listing entry's model id; Google writes ``name: "models/..."``, some caches ``slug``."""
    raw = entry.get("id") or entry.get("slug") or entry.get("model") or entry.get("name") or ""
    return str(raw).strip().removeprefix("models/")


def find_entry(rows: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
    """The entry for ``model``: the exact id first, then the same id in any case or under a ``vendor/`` prefix."""
    wanted = model.strip().removeprefix("models/")
    by_id = {entry_id(row): row for row in rows}
    if wanted in by_id:
        return by_id[wanted]
    lowered = wanted.lower()
    bare = lowered.rsplit("/", 1)[-1]
    for key, row in by_id.items():
        if key.lower() == lowered or key.lower().rsplit("/", 1)[-1] == bare:
            return row
    return None


def normalized_model(model: str) -> str:
    """A model id as tables are keyed: lower case, no ``vendor/`` prefix, no ``:tag``."""
    name = model.strip().lower().rsplit("/", 1)[-1]
    return name.split(":", 1)[0]


# Table and catalogue --------------------------------------------------------


class WindowTable:
    """Windows by model-family pattern, the host's own list.

    Keys are shell-style patterns over the normalised model id
    (:func:`normalized_model`), so ``"vendor/family-large-2:tag"`` is matched
    as ``family-large-2``. The most specific pattern wins: the one with the
    most characters that are not wildcards. A value that is not a plausible
    window is dropped.
    """

    def __init__(self, patterns: Mapping[str, int]) -> None:
        rows = [(str(key).lower(), tokens) for key, value in patterns.items()
                if (tokens := plausible_window(value)) is not None]
        self._rows = tuple(sorted(rows, key=lambda row: -len(row[0].replace("*", "").replace("?", ""))))

    @classmethod
    def from_toml(cls, path: Path) -> WindowTable:
        """A table from the ``[windows]`` section of a TOML file."""
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        windows = data.get("windows")
        return cls(windows if isinstance(windows, dict) else {})

    def lookup(self, model: str) -> int | None:
        name = normalized_model(model)
        if not name:
            return None
        for pattern, tokens in self._rows:
            if fnmatch.fnmatchcase(name, pattern):
                return tokens
        return None


def windows_from_catalogue(catalogue: Mapping[str, Any]) -> dict[str, int]:
    """``limit.context`` per model id from a models.dev-shaped catalogue, across every provider it lists.

    The same model is listed under many hosts, nearly always with the same
    number. Where hosts differ the most common value wins, and a tie goes to
    the smaller, the one every host can serve. Ids are kept as listed and
    also without their ``vendor/`` prefix, which is how most servers name them.
    """
    seen: dict[str, Counter[int]] = {}
    for provider in catalogue.values():
        models = provider.get("models") if isinstance(provider, dict) else None
        if not isinstance(models, dict):
            continue
        for model_id, model in models.items():
            tokens = window_from_entry({"limit": model.get("limit")}) if isinstance(model, dict) else None
            if tokens is None:
                continue
            lowered = str(model_id).lower()
            for key in {lowered, lowered.rsplit("/", 1)[-1]}:
                seen.setdefault(key, Counter())[tokens] += 1
    return {key: max(counts, key=lambda t: (counts[t], -t)) for key, counts in seen.items()}


class WindowCatalogue:
    """Windows by model id, from a catalogue a host downloaded (models.dev's ``api.json`` shape)."""

    def __init__(self, windows: Mapping[str, int]) -> None:
        self._windows = {str(key).lower(): value for key, value in windows.items()}

    @classmethod
    def from_mapping(cls, catalogue: Mapping[str, Any]) -> WindowCatalogue:
        return cls(windows_from_catalogue(catalogue))

    @classmethod
    def from_file(cls, path: Path) -> WindowCatalogue:
        """A catalogue from a JSON file in the models.dev shape."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(data if isinstance(data, dict) else {})

    def lookup(self, model: str) -> int | None:
        lowered = model.strip().lower()
        return (
            self._windows.get(lowered)
            or self._windows.get(lowered.rsplit("/", 1)[-1])
            or self._windows.get(normalized_model(model))
        )


# The order ------------------------------------------------------------------


def decide(
    *, provider: FoundWindow | None, catalogue: int | None, table: int | None, refused: int | None
) -> ResolvedWindow | None:
    """The window from what each source knows, in the order the module describes.

    ``None`` when nothing is known: the host keeps what it has rather than
    being handed a default that may be further from the truth than its own
    guess. A refusal wins only when it is smaller, because a server that
    refused at N cannot be sent more than N whatever its listing claims, while
    a larger one says nothing: the window was never reached.
    """
    base: ResolvedWindow | None = None
    if provider is not None:
        base = ResolvedWindow(provider.tokens, WindowSource.provider, provider.detail)
    elif catalogue is not None:
        base = ResolvedWindow(catalogue, WindowSource.catalogue)
    elif table is not None:
        base = ResolvedWindow(table, WindowSource.table)
    if refused is not None and (base is None or refused < base.tokens):
        return ResolvedWindow(refused, WindowSource.refusal)
    return base


WindowProbe = Callable[[str], Awaitable[FoundWindow | None]]
"""Asks a server how long a model's context is; ``None`` when it does not say.

A probe should be quiet: a server that does not answer the question is the
expected case, not an error. One that raises is logged and counted as a miss.
"""


class RefusalLike(Protocol):
    """What :meth:`ContextWindowResolver.observe_failure` reads from a refusal.

    The reference adapter's ``FailureObservation`` has this shape, so the
    method can be passed as its ``on_failure`` observer as it is.
    """

    @property
    def model(self) -> str: ...

    @property
    def body(self) -> str: ...


@dataclass(slots=True)
class _Cached:
    found: FoundWindow | None
    expires_at: float


class ContextWindowResolver:
    """The window of each model, from the sources in order, cached.

    ``probe`` asks a server; ``catalogue`` and ``table`` are the host's files.
    A probe's answer is kept for ``ttl_seconds`` and a probe that knew nothing
    for ``miss_ttl_seconds``, so a server that never says is not asked on every
    run while one that was down is asked again soon. Windows learned from
    refusals are kept for the resolver's lifetime: a host that wants them to
    survive a restart reads :meth:`learned` and seeds a new resolver with
    :meth:`remember_refusal`. ``clock`` is for tests.
    """

    def __init__(
        self,
        *,
        probe: WindowProbe | None = None,
        catalogue: WindowCatalogue | None = None,
        table: WindowTable | None = None,
        ttl_seconds: float = 24 * 3600.0,
        miss_ttl_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0 or miss_ttl_seconds <= 0:
            raise ValueError("ttl_seconds and miss_ttl_seconds must be positive")
        self._probe = probe
        self._catalogue = catalogue
        self._table = table
        self._ttl = ttl_seconds
        self._miss_ttl = miss_ttl_seconds
        self._clock = clock
        self._probed: dict[str, _Cached] = {}
        self._refused: dict[str, int] = {}
        self._resolved: dict[str, ResolvedWindow | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def resolve(self, model: str) -> ResolvedWindow | None:
        """The model's window, asking the probe when its cached answer is missing or stale."""
        async with self._locks.setdefault(model, asyncio.Lock()):
            cached = self._probed.get(model)
            if cached is None or cached.expires_at <= self._clock():
                found = await self._ask(model)
                ttl = self._ttl if found is not None else self._miss_ttl
                self._probed[model] = _Cached(found, self._clock() + ttl)
            return self._decide(model)

    def known(self, model: str) -> int | None:
        """The window last resolved or learned for ``model``, without asking anyone.

        For decisions that cannot wait for a request, such as a chain's fit
        check in the middle of a failure.
        """
        resolved = self._resolved.get(model)
        if resolved is not None:
            return resolved.tokens
        return self._refused.get(model)

    def rung_window(self, rung: ChainRung) -> int | None:
        """:meth:`known` for a chain rung's model; pass as ``ListProviderChain(window_of=...)``."""
        return self.known(rung.model)

    def learn_refusal(self, model: str, text: str) -> int | None:
        """Remember the window a refusal's text states; the window now held for ``model``."""
        tokens = stated_window(text)
        if tokens is not None:
            self.remember_refusal(model, tokens)
        return self.known(model)

    def remember_refusal(self, model: str, tokens: int) -> None:
        """Remember that a server refused ``model`` above ``tokens``; only a smaller value replaces one held."""
        held = self._refused.get(model)
        if held is None or tokens < held:
            self._refused[model] = tokens
            _logger.info("context window of %s learned from a refusal: %d tokens", model, tokens)
        self._decide(model)

    async def observe_failure(self, observation: RefusalLike) -> None:
        """Learn from a refusal; usable as the reference adapter's ``on_failure`` observer.

        Only a refusal the adapter classified as a context overflow is read,
        when it says how it classified it: a rate-limit message that happens to
        quote the window is not a statement that the prompt did not fit.
        """
        failure = getattr(observation, "failure", None)
        reason = getattr(failure, "reason", None)
        if failure is not None and reason != "context_overflow":
            return
        self.learn_refusal(observation.model, observation.body)

    def learned(self) -> dict[str, int]:
        """Every window learned from a refusal, by model."""
        return dict(self._refused)

    def forget(self, model: str) -> None:
        """Drop everything known about ``model``; the next :meth:`resolve` asks again."""
        self._probed.pop(model, None)
        self._refused.pop(model, None)
        self._resolved.pop(model, None)

    async def _ask(self, model: str) -> FoundWindow | None:
        if self._probe is None:
            return None
        try:
            found = await self._probe(model)
        except Exception as exc:
            _logger.warning("asking for the context window of %s failed: %s", model, exc)
            return None
        if found is None or plausible_window(found.tokens) is None:
            return None
        return found

    def _decide(self, model: str) -> ResolvedWindow | None:
        cached = self._probed.get(model)
        resolved = decide(
            provider=cached.found if cached is not None else None,
            catalogue=self._catalogue.lookup(model) if self._catalogue is not None else None,
            table=self._table.lookup(model) if self._table is not None else None,
            refused=self._refused.get(model),
        )
        if resolved is None:
            self._resolved.pop(model, None)
        else:
            self._resolved[model] = resolved
        return resolved


__all__ = [
    "ENTRY_FIELDS",
    "ContextWindowResolver",
    "FoundWindow",
    "RefusalLike",
    "ResolvedWindow",
    "WindowCatalogue",
    "WindowProbe",
    "WindowSource",
    "WindowTable",
    "decide",
    "entry_id",
    "find_entry",
    "listing_rows",
    "normalized_model",
    "plausible_window",
    "stated_window",
    "window_from_entry",
    "windows_from_catalogue",
]
