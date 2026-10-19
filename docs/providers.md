# Providers — the reference chat-completions adapter and the provider chain

> Audience: a host connecting the core to a model server, or writing its own
> adapter and wanting the reference to compare against.
> Code: `protocore/providers/chain.py` (the chain),
> `protocore/providers/openai_compat/` (the adapter: `spec.py` capabilities,
> `wire.py` request rendering, `stream.py` stream parsing, `failures.py`
> refusal reading, `tokenize.py` exact counts, `provider.py` the transport).

The loop talks to a model only through `ILLMProvider` and moves between
models only through `IProviderChain` (see [Contracts](contracts.md)). The
core ships one implementation of each so a host does not have to rewrite the
parts that are not about any one vendor:

| Module | What it is | Needs |
|---|---|---|
| `protocore.providers.chain` | `ListProviderChain`, a concrete `IProviderChain` | nothing extra |
| `protocore.providers.openai_compat` | `OpenAICompatibleProvider`, an `ILLMProvider` for any server that speaks the OpenAI chat-completions API | the `openai` extra |

```bash
pip install "protocore[openai]==2.0.0a25"   # adds httpx
```

Nothing in the core imports the adapter, so a host that brings its own loses
nothing without the extra. Both pass the packaged conformance suites
(`LLMProviderConformance`, `ProviderChainConformance`,
`RequestTokenCounterConformance`).

## A minimal endpoint

```python
from protocore.providers.openai_compat import (
    EndpointSpec,
    OpenAICompatibleProvider,
    ProviderCapabilities,
    ThinkingDialect,
)

provider = OpenAICompatibleProvider(
    EndpointSpec(
        name="local",
        base_url="http://127.0.0.1:8000/v1",
        capabilities=ProviderCapabilities(thinking=ThinkingDialect.chat_template),
    )
)
engine = QueryEngine(config=config, llm_provider=provider, ...)
```

`EndpointSpec` says where the server is and how to reach it: `base_url` (the
API root, usually ending in `/v1`), `api_key` (sent as a bearer token, kept out
of `repr`), `headers`, timeouts, the first-byte watchdog, the prompt-cache
routing field or header, a pinned `temperature`, `extra_body` (fields added to
every request; the adapter's own fields win), and the counting route. Pass an
`httpx.AsyncClient` as `client=` to share a pool or, in tests, to talk to an
`httpx.MockTransport`; a client the provider created is closed by `aclose()`.

## Capabilities

Servers agree on the core of the API and differ at the edges. Those edges are
data, one field of `ProviderCapabilities` per decision the adapter makes,
rather than a subclass per vendor. The defaults are the plain OpenAI API, and
**an unknown capability is treated as absent**: a field a server does not
know is a refused request, while leaving out one it would have accepted costs
at most some efficiency.

| Field | Default | What it decides |
|---|---|---|
| `thinking` | `openai` | how "think" and "do not think" are written (below) |
| `effort_names` | `{}` | the endpoint's words for the core's effort levels |
| `reasoning_off_effort` | `None` | the `reasoning_effort` meaning "off" under the `openai` dialect |
| `reasoning_replay` | `never` | whether `reasoning_content` goes back on later assistant turns |
| `replay_reasoning_details` | `False` | keep opaque `reasoning_details` and send them back |
| `forced_tool_choice`, `tool_choice_required` | `True` | which `tool_choice` forms the server takes |
| `schema_additional_properties` | `True` | whether a schema's `additionalProperties` reaches the wire |
| `repair_tool_pairs` | `True` | answer an unanswered call, drop a result without a call |
| `images` | `None` | image input and its limits (`ImageLimits`) |
| `cache_control` | `False` | turn the core's cache breakpoints into `cache_control` markers |
| `cache_prompt` | `False` | send `cache_prompt: true` (llama.cpp server) |
| `stream_usage` | `True` | ask for the usage chunk at the end of a stream |
| `max_tokens_field` | `max_tokens` | the body field of the output cap (`max_completion_tokens` for newer OpenAI reasoning models) |
| `json_schema_response` | `False` | `complete_structured` asks for `json_schema` rather than `json_object` |
| `repair_partial_arguments` | `True` | hand the loop the repaired prefix of a cut tool call |
| `text_tool_calls` | `()` | markups in which a model writes tool calls as text (below) |

### Thinking dialects

The adapter writes a thinking control only when the request carries one
(`extra["enable_thinking"]` with `extra["reasoning_effort"]`), and writes an
explicit "off" as off wherever the dialect can say it. Each dialect is a
public API convention:

| `ThinkingDialect` | On | Off |
|---|---|---|
| `none` | nothing | nothing |
| `openai` | `reasoning_effort` | `reasoning_off_effort` when set, else nothing |
| `chat_template` (vLLM, SGLang) | `chat_template_kwargs.enable_thinking: true` | `... false` |
| `llama_cpp` | `chat_template_kwargs` with `enable_thinking` and `reasoning_effort`, plus top-level `reasoning_effort` | `enable_thinking: false` |
| `deepseek` | `thinking: {"type": "enabled"}` and `reasoning_effort` | `thinking: {"type": "disabled"}` |
| `openrouter` | `reasoning: {"effort": ...}` | `reasoning: {"enabled": false}` |

DeepSeek accepts `low`, `high` and `max`, so its default effort names map the
core's `medium` to `high` and `xhigh` to `max`; `effort_names` overrides any
dialect's mapping.

### Reasoning replay

`ReasoningReplay.when_present` sends `reasoning_content` on the assistant turns
that have some. `required` (DeepSeek) does that for every assistant turn while
thinking is on, empty when a turn has none, and leaves out a turn with nothing
but reasoning (the partial a stream left when it died mid-thought), which the
server cannot use.

Opaque reasoning (`reasoning_details`, encrypted by the vendor) is kept on the
assistant message as `Message.reasoning_items`, in an envelope naming the
endpoint and model, and goes back only to that endpoint and model: encrypted
reasoning is readable only by the model that wrote it.

### Tool schemas and pairs

A tool's schema goes on the wire as written, including `additionalProperties`
when the tool states it. The core spells that flag `additional_properties` on
`ToolParameterSchema`, and a serialiser that copies only `properties` and
`required` silently drops it — the strict forced `plan` tool then becomes a
permissive one. Turn `schema_additional_properties` off only for a server that
refuses the keyword.

The API refuses a request in which a tool result has no call before it, or a
call has no result. A history cut by a cancel or a compaction can hold either,
so the adapter drops the first and answers the second with an explicit
`[no result: the call was interrupted before it returned]` — an error, never a
made-up success, so the model knows to decide whether to call again.

### Images and the prompt cache

With `images=ImageLimits(max_images=..., max_bytes=..., mime_types=...)` and an
`image_loader`, images are inlined as data URLs. One the endpoint would refuse
(over the count, which keeps the most recent; over the size; another format; a
load failure) is replaced by a note saying why, so a request never fails on a
picture and the model still knows one was attached.

`cache_control=True` places a `cache_control` marker (with `ttl` for the
`5m`/`1h` buckets) on the wire entry each of the core's `CacheBreakpoint`s
names. The request's `observability.routing_key()` — the configured
`QueryEngineConfig.cache_key`, else the session, else the run — goes into
`EndpointSpec.cache_key_field` (e.g. `prompt_cache_key` on the OpenAI API)
and/or `cache_key_header` (e.g. `x-grok-conv-id`, which xAI documents), so the
requests of one conversation reach the machine that holds its cache.

## The stream

`stream_with_tools` yields `ProviderDelta`s in the order the loop reads them:
text and thinking as they arrive, tool calls as they assemble, then a
`tool_use_stop` per call, `usage`, and `finish`.

- **Liveness.** An SSE comment, a role-only chunk and an empty delta each
  yield a `progress` delta. The core's idle watchdog counts from the last
  delta; without these a model that thinks silently was cut as a dead stream.
- **Tool calls** are assembled from fragments keyed by `index`; a server that
  numbers every call `0` is told apart by id, and arguments that arrive before
  the name are released with it. Arguments that do not parse were cut short:
  the stop carries the repaired prefix, `args_partial_truncated=True`, and
  `truncated_by_output_cap=True` when `finish_reason` was `length`.
- **Errors after the 200.** A chunk that is an `error` object is a refusal like
  any other, read through the same classifier.
- **Usage** is normalised: `input_tokens` is the whole prompt (cached part
  included), and the cache hit comes from `prompt_tokens_details.cached_tokens`,
  DeepSeek's `prompt_cache_hit_tokens` or `cache_read_input_tokens`; a reported
  `cost` becomes `cost_usd`.

### The first-byte watchdog

`first_byte_timeout_seconds` (off at `0`) bounds the wait for the response
headers and the first line of the body together: a server that answers its
headers at once and then says nothing is as stuck as one that never answers.
Any line counts, an SSE comment included. On a timeout the response is closed
and the same body is sent again, byte for byte, so the prompt cache sees the
identical prefix; this is safe because nothing came back, so nothing can be
repeated. A `progress` delta goes to the loop before each resend, so the two
windows are not added together by the idle watchdog. After
`first_byte_retries` resends the silence is raised as `LLMTimeoutError`
(`timeout`, retryable), which moves the chain first. A refusal is an answer
and is never resent. Leave it off for a local server that may spend minutes
reading a long prompt before its first byte.

### Tool calls written as text

Some models now and then write a tool call in their own markup inside the
reply instead of as a structured call. The server passes it on as text: the
markup is shown to the reader and the turn ends without the call. List the
markup in `ProviderCapabilities(text_tool_calls=(DSML,))` and the adapter
guards the text stream with a `TextToolCallGuard`
(`protocore.providers.text_tool_calls`, pure Python):

- text before a marker is released as it comes (only the few characters that
  might be the start of a marker split across chunks wait for the next one);
- from the marker on, text is held back, and a `progress` delta goes to the
  loop in place of each held piece so the idle watchdog does not cut the
  stream;
- at the end of the stream a complete block becomes ordinary tool calls (a
  `tool_use_start` and a `tool_use_stop` each, ids `call_<stream>_text<n>`)
  and a `stop` finish becomes `tool_use`; prose around the block is released;
- an incomplete block is released unchanged, and so is held text that passes
  the hold limit (4 000 characters) without looking like a call block — a
  model that merely mentions the markup keeps its reply.

`DSML` (`DsmlGrammar`) reads DeepSeek's `<｜DSML｜tool_calls>` markup in every
spelling of the bars; a parameter marked `string="true"` is text, any other is
read as JSON. Another markup is a class with `name`, `markers`,
`is_call_block(held)` and `parse(held) -> ParsedText | None`
(`TextToolCallGrammar`); the earliest marker of any listed grammar wins. The
guard is not tied to this adapter: any adapter can feed its own text deltas
through it.

## Failures

Every error the adapter raises carries a `ProviderFailure`
([Reporting a provider failure](contracts.md#reporting-a-provider-failure)), so
`assert_adapter_failure` accepts it:

| Response | Raised as | Reason | Retryable |
|---|---|---|---|
| 401, 403, an auth error type | `LLMProviderError` | `auth` | no |
| 402, `insufficient_quota` (even on a 429) | `LLMProviderError` | `billing` | no |
| 429, a rate-limit error type | `LLMRateLimitError` | `rate_limit` | yes, with the stated wait |
| 408, 504 | `LLMTimeoutError` | `timeout` | yes |
| 503, 529, an overload error type | `LLMProviderError` | `overloaded` | yes, with the stated wait |
| 404, "model does not exist" | `LLMProviderError` | `model_not_found` | no |
| a content-policy code | `LLMProviderError` | `provider_policy_blocked` | no |
| a context refusal | `LLMContextWindowExceeded` with the stated sizes | `context_overflow` | no (the loop compacts) |
| 413 otherwise | `LLMProviderError` | `payload_too_large` | no |
| any other 4xx | `LLMProviderError` | `format_error`, `should_fallback` when the request was constrained | no |
| 5xx | `LLMProviderError` | `server_error` | yes |
| connect refused, reset, broken mid-stream | `LLMProviderError` | `connection` | yes |
| a client timeout | `LLMTimeoutError` | `timeout` | yes |

The wait comes from `retry-after-ms`, then `Retry-After` (seconds or a date),
then, on a 429, the later of `x-ratelimit-reset-requests` and
`x-ratelimit-reset-tokens`; anything past a day is ignored and the loop clamps
it to its own ceiling. The context sizes are read from the wordings of vLLM and
OpenAI ("maximum context length is N tokens ... (P in the messages, O in the
completion)", "your messages resulted in P tokens", "you requested O output
tokens and your prompt contains at least P input tokens"), Anthropic-shaped
gateways ("prompt is too long: P tokens > N maximum") and llama.cpp
(`n_prompt_tokens` / `n_ctx` on the error, or "request (P tokens) exceeds the
available context size (N tokens)").

## Extending the adapter

Three points are meant to be overridden or passed in:

- `build_body(request, *, stream)` — call the base and add fields a deployment
  needs. The body is built once per call and reused for a watchdog resend.
- `request_headers(request)` — add headers.
- `on_usage(UsageObservation)` and `on_failure(FailureObservation)` — hear every
  usage report (for a ledger or a spend cap) and every refusal (for a hold, or
  to learn a model's real window from what the server said). A failing observer
  is logged and never hides the call's own result or error. A
  `FailureObservation` keeps only the retry headers, so it cannot store a
  credential a proxy echoed back.

## Spend budgets and prices

The loop budgets tokens; money is the host's to budget. The adapter gives that
decision one place to happen (`protocore.providers.spend`, pure Python):

```python
from protocore.providers.spend import ModelPricing, SpendRefused

prices = {"model-a": ModelPricing(input=0.30, output=1.20, cached_input=0.03)}

class DailyCap:
    async def reserve(self, estimate):          # before every call
        if spent_today() + (estimate.cost_usd or 0) > CAP:
            raise SpendRefused("daily cap reached", retry_after_seconds=until_midnight())
        return hold(estimate)                   # a reservation id, or None
    async def settle(self, reservation, settlement):   # the call reported usage
        book(reservation, settlement.cost_usd, complete=settlement.complete)
    async def release(self, reservation, *, outcome, detail):  # no usage came
        drop_or_keep(reservation, outcome)

provider = OpenAICompatibleProvider(spec, admission=DailyCap(), pricing=prices.get)
```

- `reserve(SpendEstimate)` is asked before each call (stream, text,
  structured) with the endpoint, model, purpose, the estimated prompt size,
  the output cap, the pricing's upper bound in `cost_usd` (when the model's
  prices are known) and the request's observability context. Raising
  `SpendRefused` refuses the call: nothing is sent, and the loop sees
  `LLMProviderError` with a `budget` `ProviderFailure` — every rung meets the
  same cap, so nothing retries it or moves past it. Returning `None` means
  the call needs no accounting.
- Exactly one of `settle` and `release` follows a reservation. `settle(id,
  SpendSettlement)` carries the normalised usage, the raw usage, `cost_usd`
  (the server's own `cost`, else the pricing's price) and `complete`, which is
  `False` when the server reported no token counts: the call may still have
  been billed, so a host usually keeps the reservation as spent. It runs
  before the `usage` delta is yielded, because the loop may stop reading at
  `finish`.
- `release(id, outcome=..., detail=...)` says the call ended without usage:
  `SpendEnd.refused` when the server answered with a 4xx or 5xx (usually not
  billed), `SpendEnd.interrupted` when it was sent and never finished — a
  broken stream, a first-byte timeout, a caller that stopped reading (output
  may already have been generated).

A failing `settle` or `release` is logged and never replaces the call's own
result or error. Override `estimate_spend(request, body, purpose)` to estimate
the prompt exactly; the default is the core's estimate of the messages and
tools as they go on the wire, which only has to err high.

`ModelPricing(input, output, cached_input=None, cache_write=None,
reasoning=None)` holds US dollars per million tokens. `cost(usage)` prices
fresh input, cache reads, cache writes, visible output and reasoning each at
its own rate, reading the keys the adapter reports (`input_tokens` is the
whole prompt, cache included; `output_tokens` includes `reasoning_tokens`).
An omitted cache price is the input price and an omitted reasoning price the
output price, so a missing number never makes tokens free.
`estimate(prompt_tokens=, max_output_tokens=)` is the upper bound: no cache
hit, the whole output cap spent at the dearer output rate. With `pricing=`
given, a usage report without a server-stated cost gets `cost_usd` from the
table, on the `usage` delta and on `LLMResponse.usage.response_cost_usd`.

## Exact token counts

With `EndpointSpec(tokenize=TokenizeDialect.vllm)` or
`TokenizeDialect.llama_cpp`, the provider implements
`IRequestTokenCounter.count_request_tokens`: it builds the body it would send
and counts it on the server, template, tools and thinking switch included.
vLLM counts the chat request on `POST /tokenize`; llama.cpp renders it on
`POST /apply-template` and counts the prompt on `POST /tokenize`. Both live at
the server root (`base_url` without `/v1`, or `tokenize_url`). A server without
the route answers 404 once and is not asked again; the count is then `None`,
and the core uses its own estimate, which `count_tokens` also returns.

## Finding a model's context window

Everything that sizes a run reads the window, and a number typed by hand is a
question many operators cannot answer. `ContextWindowResolver`
(`protocore.providers.context_window`, pure Python) takes it from the first
source that knows:

1. **the server's own listing**, through a probe;
2. **a catalogue** in the shape of models.dev's `api.json`, a file the host
   downloads and passes in (`WindowCatalogue.from_file(path)`);
3. **a table** of model-family patterns the host keeps
   (`WindowTable({"family-*": 128_000})` or `WindowTable.from_toml(path)`,
   reading a `[windows]` section; the most specific pattern wins, matched
   against the id in lower case without a `vendor/` prefix or `:tag`);
4. **a refusal**: a server that said "maximum context length is N tokens" has
   stated the real limit, and when N is smaller than what the sources above
   say it wins over all of them.

```python
from protocore.providers.context_window import ContextWindowResolver, WindowCatalogue
from protocore.providers.openai_compat import ListingWindowProbe

resolver = ContextWindowResolver(
    probe=ListingWindowProbe.for_endpoint(spec, props=is_llama_cpp),
    catalogue=WindowCatalogue.from_file(Path("models.json")),
)
provider = OpenAICompatibleProvider(spec, on_failure=resolver.observe_failure)
window = await resolver.resolve("model-a")      # ResolvedWindow(tokens, source, detail) or None
chain = ListProviderChain(rungs, window_of=resolver.rung_window)
```

Nothing is bundled: which models exist and how large they are changes every
month, and a number shipped in a library goes stale long before anyone
updates it. When no source knows, `resolve` returns `None` and the host keeps
the number it has.

- **The probe.** `ListingWindowProbe(base_url, *, client=None, api_key="",
  headers=None, props=False)` reads `GET {base}/models` (and `{base}/v1/models`
  for a base without `/v1`), then the model's own entry `GET .../models/{id}`;
  with `props=True` it first reads llama.cpp's `GET /props`, the length the
  server was started with rather than what the model was trained for. It
  reads the window from whichever field a server uses (`max_model_len`,
  `context_length`, `context_window`, `max_context_length`,
  `loaded_context_length`, `max_input_tokens`, `inputTokenLimit`,
  `limit.context`; `ENTRY_FIELDS` lists them in order). Every failure is quiet,
  and redirects are never followed, since the request may carry the key. Any
  async callable `model -> FoundWindow | None` is a probe.
- **The cache.** A probe's answer is kept for `ttl_seconds` (a day) and a miss
  for `miss_ttl_seconds` (ten minutes), so a server that never says is not
  asked on every run while one that was down is asked again soon. Concurrent
  `resolve` calls for one model share one probe. A probe that raises is
  logged and counts as a miss.
- **Refusals.** `observe_failure` has the shape of the adapter's `on_failure`
  observer and learns only from a refusal classified as `context_overflow`;
  `learn_refusal(model, text)` and `remember_refusal(model, tokens)` do it by
  hand. `stated_window(text)` reads the OpenAI/vLLM, Anthropic, llama.cpp,
  Google and LM Studio phrasings. A larger stated window never replaces a
  smaller one. Learned windows live as long as the resolver; `learned()` hands
  them to a host that persists them, and `forget(model)` starts over.
- **Without waiting.** `known(model)` answers from what was resolved or
  learned, never asking anyone, for decisions that cannot wait for a request.
  `rung_window` is that for a chain rung: `ListProviderChain(window_of=...)`
  uses it in the fit check (below).

## The provider chain

```python
from protocore.providers.chain import ChainRung, ListProviderChain

chain = ListProviderChain(
    [
        ChainRung(name="primary", model="model-a", provider=primary,
                  context_window=200_000, max_output_tokens=8_000),
        ChainRung(name="fallback", model="model-b", factory=build_fallback,
                  context_window=32_000, max_output_tokens=4_000),
    ],
    resting=lambda rung: holds.get(rung.name),
)
engine = QueryEngine(config=config, llm_provider=primary, provider_chain=chain, ...)
chain.bind_prompt_size(engine.prompt_tokens_now)
```

The walk is one way, under a lock, so two failures that arrive together cannot
both decide from the same position. On each `advance` a fallback is skipped,
with its reason recorded in `attempted()`, when:

- its window cannot hold the prompt the run is carrying plus the reply it asks
  for (demoting onto it trades a stalled stream for a context-window refusal);
- the `resting` check names it (an account that refused for a spent quota a
  minute ago);
- its `factory` raises (a missing key or a bad URL is a fact about that rung).

With `window_of=` (for instance a resolver's `rung_window`), the fit check
uses the smaller of the rung's configured `context_window` and the window
found out since: either one being right makes the larger a request the server
refuses.

A rung with a `factory` is built only when the run reaches it, so a run that
never leaves the primary opens no pool for the others. The primary is never
checked, and an unknown window or prompt size never rules a rung out. On resume
the engine re-walks the recorded number of advances and checks the model name;
a rung that was skipped then but fits now changes where they land, and the
engine refuses the snapshot rather than serve it from the wrong rung.
