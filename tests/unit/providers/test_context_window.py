"""Finding out how much a model holds: listings, a catalogue, a table, and refusals."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from protocore.contracts.llm import ProviderFailure, ProviderFailureReason
from protocore.providers.chain import ChainRung, ListProviderChain
from protocore.providers.context_window import (
    ContextWindowResolver,
    FoundWindow,
    ResolvedWindow,
    WindowCatalogue,
    WindowSource,
    WindowTable,
    decide,
    find_entry,
    listing_rows,
    normalized_model,
    stated_window,
    window_from_entry,
    windows_from_catalogue,
)
from protocore.providers.openai_compat import OpenAICompatibleProvider
from protocore.providers.openai_compat.windows import ListingWindowProbe
from protocore.tests_support.adapters import InMemoryLLMProvider

from ._openai_support import FakeServer, collect, endpoint, user_request

# Refusal phrasings ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("This model's maximum context length is 131072 tokens. However, you requested 140000 tokens", 131_072),
        ("prompt is too long: 210582 tokens > 200000 maximum", 200_000),
        ("the request exceeds the available context size (8192 tokens), try increasing it", 8_192),
        ("The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).", 1_048_576),
        ("Trying to keep the first 5000 tokens when context the overflows. However, the model is loaded with "
         "context length of only 4096 tokens", 4_096),
    ],
)
def test_the_window_a_refusal_states_is_read_in_every_phrasing(text: str, tokens: int) -> None:
    assert stated_window(text) == tokens


def test_a_refusal_that_states_no_window_gives_nothing() -> None:
    assert stated_window("rate limit exceeded") is None
    assert stated_window("") is None
    # A number too small to be a window is a typo or another quantity.
    assert stated_window("maximum context length is 12 tokens") is None


# Listings -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "tokens"),
    [
        ({"id": "m", "max_model_len": 32_768}, 32_768),
        ({"id": "m", "context_length": 200_000}, 200_000),
        ({"id": "m", "top_provider": {"context_length": 64_000}}, 64_000),
        ({"id": "m", "context_window": 131_072}, 131_072),
        ({"id": "m", "max_input_tokens": 200_000}, 200_000),
        ({"name": "models/g", "inputTokenLimit": 1_048_576}, 1_048_576),
        ({"id": "m", "limit": {"context": 400_000}}, 400_000),
        ({"id": "m", "max_context_length": "65536"}, 65_536),
        # What a server loaded wins over what the model could take.
        ({"id": "m", "loaded_context_length": 8_192, "max_context_length": 131_072}, 8_192),
    ],
)
def test_each_vendor_field_of_a_listing_entry_is_read(entry: dict[str, Any], tokens: int) -> None:
    assert window_from_entry(entry) == tokens


def test_an_entry_without_a_window_gives_nothing() -> None:
    assert window_from_entry({"id": "m"}) is None
    assert window_from_entry({"id": "m", "context_length": True}) is None
    assert window_from_entry("not an entry") is None


def test_every_listing_shape_yields_its_rows() -> None:
    assert listing_rows({"data": [{"id": "a"}, 1]}) == [{"id": "a"}]
    assert listing_rows({"models": [{"name": "b"}]}) == [{"name": "b"}]
    assert listing_rows([{"id": "c"}]) == [{"id": "c"}]
    assert listing_rows("nothing") == []


def test_an_entry_is_found_by_id_with_or_without_its_vendor() -> None:
    rows = [{"id": "vendor/Model-A"}, {"name": "models/gemini-x"}, {"slug": "other"}]
    assert find_entry(rows, "vendor/Model-A") == rows[0]
    assert find_entry(rows, "model-a") == rows[0]
    assert find_entry(rows, "gemini-x") == rows[1]
    assert find_entry(rows, "missing") is None


def test_a_model_id_is_normalised_for_matching() -> None:
    assert normalized_model(" Vendor/Model-A:latest ") == "model-a"


# Table and catalogue --------------------------------------------------------------


def test_the_most_specific_table_pattern_wins() -> None:
    table = WindowTable({"family-*": 128_000, "family-large-*": 1_000_000, "bogus": 3})

    assert table.lookup("vendor/family-large-2:tag") == 1_000_000
    assert table.lookup("family-small") == 128_000
    assert table.lookup("other") is None
    assert table.lookup("") is None


def test_a_table_loads_from_a_toml_file(tmp_path: Path) -> None:
    path = tmp_path / "windows.toml"
    path.write_text('[windows]\n"family-*" = 32_000\n', encoding="utf-8")
    assert WindowTable.from_toml(path).lookup("family-a") == 32_000


def test_a_catalogue_takes_the_value_most_hosts_agree_on() -> None:
    catalogue = {
        "host-a": {"models": {"vendor/model-a": {"limit": {"context": 128_000}}}},
        "host-b": {"models": {"model-a": {"limit": {"context": 128_000}}, "model-b": {"limit": {"context": 64_000}}}},
        "host-c": {"models": {"model-a": {"limit": {"context": 256_000}},
                              "model-b": {"limit": {"context": 32_000}}}},
        "broken": "not a provider",
    }
    windows = windows_from_catalogue(catalogue)

    assert windows["model-a"] == 128_000
    assert windows["vendor/model-a"] == 128_000
    # A tie goes to the smaller, the one every host can serve.
    assert windows["model-b"] == 32_000
    assert WindowCatalogue(windows).lookup("Vendor/Model-B:free") == 32_000


def test_a_catalogue_loads_from_a_json_file(tmp_path: Path) -> None:
    path = tmp_path / "api.json"
    path.write_text(json.dumps({"host": {"models": {"model-a": {"limit": {"context": 100_000}}}}}), encoding="utf-8")
    assert WindowCatalogue.from_file(path).lookup("model-a") == 100_000


# The order ------------------------------------------------------------------------


def test_the_provider_wins_then_the_catalogue_then_the_table() -> None:
    assert decide(provider=FoundWindow(10_000, "listing"), catalogue=20_000, table=30_000, refused=None) == \
        ResolvedWindow(10_000, WindowSource.provider, "listing")
    assert decide(provider=None, catalogue=20_000, table=30_000, refused=None) == \
        ResolvedWindow(20_000, WindowSource.catalogue, "")
    assert decide(provider=None, catalogue=None, table=30_000, refused=None) == \
        ResolvedWindow(30_000, WindowSource.table, "")
    assert decide(provider=None, catalogue=None, table=None, refused=None) is None


def test_a_refusal_wins_only_when_it_is_smaller() -> None:
    assert decide(provider=FoundWindow(10_000), catalogue=None, table=None, refused=8_000).source is WindowSource.refusal
    resolved = decide(provider=FoundWindow(10_000), catalogue=None, table=None, refused=12_000)
    assert resolved is not None and resolved.source is WindowSource.provider


# The resolver ---------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class CountingProbe:
    def __init__(self, answers: dict[str, int | None], *, delay: float = 0.0, fail: bool = False) -> None:
        self.answers = answers
        self.calls: list[str] = []
        self.delay = delay
        self.fail = fail

    async def __call__(self, model: str) -> FoundWindow | None:
        self.calls.append(model)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("probe broke")
        tokens = self.answers.get(model)
        return FoundWindow(tokens, "probe") if tokens else None


async def test_the_resolver_asks_once_and_answers_from_its_cache() -> None:
    probe = CountingProbe({"model-a": 64_000})
    resolver = ContextWindowResolver(probe=probe)

    assert resolver.known("model-a") is None
    first = await resolver.resolve("model-a")
    second = await resolver.resolve("model-a")

    assert first == second == ResolvedWindow(64_000, WindowSource.provider, "probe")
    assert probe.calls == ["model-a"]
    assert resolver.known("model-a") == 64_000


async def test_concurrent_resolves_share_one_probe() -> None:
    probe = CountingProbe({"model-a": 64_000}, delay=0.01)
    resolver = ContextWindowResolver(probe=probe)

    results = await asyncio.gather(*(resolver.resolve("model-a") for _ in range(5)))

    assert {r.tokens for r in results if r is not None} == {64_000}
    assert probe.calls == ["model-a"]


async def test_answers_expire_and_misses_expire_sooner() -> None:
    clock = Clock()
    probe = CountingProbe({"model-a": 64_000})
    resolver = ContextWindowResolver(probe=probe, ttl_seconds=100, miss_ttl_seconds=10, clock=clock)

    assert await resolver.resolve("model-b") is None
    clock.now += 5
    assert await resolver.resolve("model-b") is None
    assert probe.calls == ["model-b"]
    clock.now += 10
    await resolver.resolve("model-b")
    assert probe.calls == ["model-b", "model-b"]

    await resolver.resolve("model-a")
    clock.now += 50
    await resolver.resolve("model-a")
    assert probe.calls.count("model-a") == 1
    clock.now += 60
    await resolver.resolve("model-a")
    assert probe.calls.count("model-a") == 2


async def test_a_broken_probe_falls_through_to_the_catalogue_and_table() -> None:
    resolver = ContextWindowResolver(
        probe=CountingProbe({}, fail=True),
        catalogue=WindowCatalogue({"model-a": 100_000}),
        table=WindowTable({"model-*": 50_000}),
    )

    assert await resolver.resolve("model-a") == ResolvedWindow(100_000, WindowSource.catalogue, "")
    assert await resolver.resolve("model-b") == ResolvedWindow(50_000, WindowSource.table, "")


@dataclass
class _Refusal:
    model: str
    body: str
    failure: ProviderFailure | None


async def test_a_refusal_teaches_the_resolver_a_smaller_window() -> None:
    resolver = ContextWindowResolver(probe=CountingProbe({"model-a": 131_072}))
    await resolver.resolve("model-a")

    overflow = ProviderFailure(reason=ProviderFailureReason.context_overflow, retryable=False)
    await resolver.observe_failure(_Refusal("model-a", "maximum context length is 32768 tokens", overflow))
    assert resolver.known("model-a") == 32_768
    resolved = await resolver.resolve("model-a")
    assert resolved == ResolvedWindow(32_768, WindowSource.refusal, "")

    # A larger stated window changes nothing; the smaller one already held.
    assert resolver.learn_refusal("model-a", "maximum context length is 65536 tokens") == 32_768
    assert resolver.known("model-a") == 32_768


async def test_a_refusal_for_another_reason_teaches_nothing() -> None:
    resolver = ContextWindowResolver()
    rate = ProviderFailure(reason=ProviderFailureReason.rate_limit, retryable=True)
    await resolver.observe_failure(_Refusal("model-a", "maximum context length is 32768 tokens", rate))
    assert resolver.known("model-a") is None


async def test_forgetting_a_model_asks_again() -> None:
    probe = CountingProbe({"model-a": 64_000})
    resolver = ContextWindowResolver(probe=probe)
    await resolver.resolve("model-a")
    resolver.remember_refusal("model-a", 10_000)
    resolver.forget("model-a")

    assert resolver.known("model-a") is None
    assert await resolver.resolve("model-a") == ResolvedWindow(64_000, WindowSource.provider, "probe")
    assert probe.calls == ["model-a", "model-a"]


def test_the_cache_lengths_must_be_positive() -> None:
    with pytest.raises(ValueError, match="ttl"):
        ContextWindowResolver(ttl_seconds=0)


async def test_a_learned_window_rules_a_chain_rung_out() -> None:
    resolver = ContextWindowResolver()
    rungs = [
        ChainRung(name="a", model="model-a", provider=InMemoryLLMProvider()),
        # Configured as large, but the server said otherwise.
        ChainRung(name="b", model="model-b", provider=InMemoryLLMProvider(), context_window=200_000,
                  max_output_tokens=4_000),
        ChainRung(name="c", model="model-c", provider=InMemoryLLMProvider()),
    ]
    chain = ListProviderChain(rungs, prompt_tokens=lambda: 50_000, window_of=resolver.rung_window)
    resolver.remember_refusal("model-b", 32_000)

    assert await chain.advance(reason="overloaded") is True
    assert chain.current_model_name() == "model-c"
    assert ("b", "context window 32000 cannot hold the prompt (50000 tokens) and the reply (4000)") in chain.attempted()


async def test_a_configured_window_smaller_than_the_learned_one_still_counts() -> None:
    resolver = ContextWindowResolver()
    resolver.remember_refusal("model-b", 1_000_000)
    rungs = [
        ChainRung(name="a", model="model-a", provider=InMemoryLLMProvider()),
        ChainRung(name="b", model="model-b", provider=InMemoryLLMProvider(), context_window=16_000),
    ]
    chain = ListProviderChain(rungs, prompt_tokens=lambda: 50_000, window_of=resolver.rung_window)

    assert await chain.advance(reason="overloaded") is False


# Asking the server ----------------------------------------------------------------


def _routes(routes: dict[str, httpx.Response | Exception]) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = routes.get(request.url.raw_path.decode().split("?")[0], httpx.Response(404))
        if isinstance(answer, Exception):
            raise answer
        return answer

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


async def test_the_probe_reads_the_model_listing() -> None:
    client, seen = _routes({"/v1/models": httpx.Response(200, json={"data": [
        {"id": "other", "max_model_len": 1_000}, {"id": "model-a", "max_model_len": 32_768}]})})
    probe = ListingWindowProbe("http://127.0.0.1:8000/v1", client=client, api_key="secret")

    assert await probe("model-a") == FoundWindow(32_768, "/models")
    assert seen[0].headers["authorization"] == "Bearer secret"


async def test_the_probe_tries_v1_and_then_the_model_entry() -> None:
    client, seen = _routes({
        "/v1/models": httpx.Response(200, json={"data": [{"id": "model-a"}]}),
        "/v1/models/vendor%2Fmodel-a": httpx.Response(200, json={"id": "model-a", "max_input_tokens": 200_000}),
    })
    probe = ListingWindowProbe("http://127.0.0.1:8000", client=client)

    assert await probe("vendor/model-a") == FoundWindow(200_000, "/models/{id}")
    assert [r.url.raw_path.decode() for r in seen] == ["/models", "/v1/models", "/v1/models/vendor%2Fmodel-a"]


async def test_the_probe_reads_a_llama_cpp_servers_props_first() -> None:
    client, seen = _routes({
        "/props": httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8_192}}),
        "/v1/models": httpx.Response(200, json={"data": [{"id": "model-a", "context_length": 131_072}]}),
    })
    probe = ListingWindowProbe("http://127.0.0.1:8080/v1", client=client, props=True)

    # The length the server was started with, not what the model could take.
    assert await probe("model-a") == FoundWindow(8_192, "/props")
    assert [r.url.path for r in seen] == ["/props"]


async def test_the_probe_reads_a_top_level_n_ctx() -> None:
    client, _ = _routes({"/props": httpx.Response(200, json={"n_ctx": 4_096})})
    assert await ListingWindowProbe("http://127.0.0.1:8080", client=client, props=True)("m") == \
        FoundWindow(4_096, "/props")


async def test_every_failure_of_the_probe_is_quiet() -> None:
    client, _ = _routes({
        "/props": httpx.Response(200, content=b"not json"),
        "/v1/models": httpx.ConnectError("refused"),
        "/models": httpx.Response(500),
    })
    probe = ListingWindowProbe("http://127.0.0.1:8080", client=client, props=True)

    assert await probe("model-a") is None


async def test_the_probe_does_not_follow_a_redirect_with_the_key() -> None:
    client, seen = _routes({
        "/v1/models": httpx.Response(302, headers={"location": "http://elsewhere.invalid/models"}),
    })
    probe = ListingWindowProbe("http://127.0.0.1:8000/v1", client=client, api_key="secret")

    assert await probe("model-a") is None
    assert all(r.url.host == "127.0.0.1" for r in seen)


async def test_the_probe_can_be_built_from_an_endpoint() -> None:
    client, seen = _routes({"/v1/models": httpx.Response(200, json={"data": [{"id": "m", "context_length": 9_000}]})})
    spec = endpoint(api_key="k", headers={"x-extra": "1"})

    probe = ListingWindowProbe.for_endpoint(spec, client=client)

    assert await probe("m") == FoundWindow(9_000, "/models")
    assert seen[0].headers["x-extra"] == "1"
    assert seen[0].headers["authorization"] == "Bearer k"


async def test_the_probe_owns_and_closes_a_client_it_made() -> None:
    probe = ListingWindowProbe("http://127.0.0.1:8000/v1")
    await probe.aclose()
    with pytest.raises(ValueError, match="base_url"):
        ListingWindowProbe("not a url")


async def test_the_adapters_failure_observer_feeds_the_resolver() -> None:
    resolver = ContextWindowResolver()
    server = FakeServer(httpx.Response(400, json={"error": {
        "message": "This model's maximum context length is 16384 tokens. However, you requested 20000 tokens "
                   "(18000 in the messages, 2000 in the completion).", "type": "invalid_request_error"}}))
    provider = OpenAICompatibleProvider(endpoint(), client=server.client(), on_failure=resolver.observe_failure)

    with pytest.raises(Exception, match="maximum context length"):
        await collect(provider.stream_with_tools(user_request()))

    assert resolver.known("model-a") == 16_384
