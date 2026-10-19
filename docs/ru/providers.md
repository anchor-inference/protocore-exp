# Провайдеры — эталонный адаптер chat-completions и цепочка провайдеров

> Аудитория: хост, который подключает ядро к серверу модели, или пишет свой
> адаптер и хочет сверяться с эталоном.
> Код: `protocore/providers/chain.py` (цепочка),
> `protocore/providers/openai_compat/` (адаптер: `spec.py` возможности,
> `wire.py` сборка запроса, `stream.py` разбор потока, `failures.py` чтение
> отказов, `tokenize.py` точный подсчёт, `provider.py` транспорт).

Цикл говорит с моделью только через `ILLMProvider` и переходит между моделями
только через `IProviderChain` (см. [Контракты](contracts.md)). Ядро поставляет
по одной реализации каждого, чтобы хосту не приходилось заново писать то, что
не относится ни к одному конкретному вендору:

| Модуль | Что это | Что нужно |
|---|---|---|
| `protocore.providers.chain` | `ListProviderChain`, конкретная `IProviderChain` | ничего дополнительно |
| `protocore.providers.openai_compat` | `OpenAICompatibleProvider`, `ILLMProvider` для любого сервера с API OpenAI chat-completions | набор `openai` |

```bash
pip install "protocore[openai]==2.0.0a25"   # добавляет httpx
```

Ничто в ядре не импортирует адаптер, так что хост со своим адаптером без
этого набора ничего не теряет. Оба проходят поставляемые conformance-наборы
(`LLMProviderConformance`, `ProviderChainConformance`,
`RequestTokenCounterConformance`).

## Минимальный эндпоинт

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

`EndpointSpec` говорит, где сервер и как до него дойти: `base_url` (корень API,
обычно оканчивается на `/v1`), `api_key` (уходит как bearer-токен, не попадает
в `repr`), `headers`, таймауты, сторож первого байта, поле или заголовок ключа
маршрутизации кэша промптов, закреплённая `temperature`, `extra_body` (поля,
добавляемые в каждый запрос; собственные поля адаптера важнее) и маршрут
подсчёта. Передайте `httpx.AsyncClient` как `client=`, чтобы делить пул или, в
тестах, говорить с `httpx.MockTransport`; клиент, созданный самим провайдером,
закрывается `aclose()`.

## Возможности

Серверы согласны в основе API и расходятся на краях. Эти края — данные, по
одному полю `ProviderCapabilities` на каждое решение адаптера, а не подкласс на
вендора. По умолчанию — простой API OpenAI, и **неизвестная возможность
считается отсутствующей**: поле, которого сервер не знает, — отклонённый
запрос, а пропущенное поле, которое он бы принял, стоит в худшем случае
немного эффективности.

| Поле | По умолчанию | Что решает |
|---|---|---|
| `thinking` | `openai` | как пишется «думай» и «не думай» (ниже) |
| `effort_names` | `{}` | слова эндпоинта для уровней усилия ядра |
| `reasoning_off_effort` | `None` | `reasoning_effort`, означающий «выкл» в диалекте `openai` |
| `reasoning_replay` | `never` | возвращается ли `reasoning_content` в поздних ходах ассистента |
| `replay_reasoning_details` | `False` | хранить непрозрачные `reasoning_details` и отправлять обратно |
| `forced_tool_choice`, `tool_choice_required` | `True` | какие формы `tool_choice` принимает сервер |
| `schema_additional_properties` | `True` | доходит ли `additionalProperties` схемы до провода |
| `repair_tool_pairs` | `True` | ответить на вызов без ответа, выбросить результат без вызова |
| `images` | `None` | ввод изображений и его пределы (`ImageLimits`) |
| `cache_control` | `False` | превращать точки кэша ядра в маркеры `cache_control` |
| `cache_prompt` | `False` | отправлять `cache_prompt: true` (сервер llama.cpp) |
| `stream_usage` | `True` | просить чанк usage в конце потока |
| `max_tokens_field` | `max_tokens` | поле тела для предела вывода (`max_completion_tokens` у новых рассуждающих моделей OpenAI) |
| `json_schema_response` | `False` | `complete_structured` просит `json_schema`, а не `json_object` |
| `repair_partial_arguments` | `True` | отдавать циклу починенный префикс обрезанного вызова |

### Диалекты размышления

Адаптер пишет управление размышлением, только если оно есть в запросе
(`extra["enable_thinking"]` вместе с `extra["reasoning_effort"]`), и явное
«выкл» пишет как выкл везде, где диалект умеет это сказать. Каждый диалект —
публичное соглашение API:

| `ThinkingDialect` | Вкл | Выкл |
|---|---|---|
| `none` | ничего | ничего |
| `openai` | `reasoning_effort` | `reasoning_off_effort`, если задан, иначе ничего |
| `chat_template` (vLLM, SGLang) | `chat_template_kwargs.enable_thinking: true` | `... false` |
| `llama_cpp` | `chat_template_kwargs` с `enable_thinking` и `reasoning_effort`, плюс `reasoning_effort` верхнего уровня | `enable_thinking: false` |
| `deepseek` | `thinking: {"type": "enabled"}` и `reasoning_effort` | `thinking: {"type": "disabled"}` |
| `openrouter` | `reasoning: {"effort": ...}` | `reasoning: {"enabled": false}` |

DeepSeek принимает `low`, `high` и `max`, поэтому его имена по умолчанию
переводят `medium` ядра в `high`, а `xhigh` — в `max`; `effort_names`
переопределяет отображение любого диалекта.

### Возврат рассуждений

`ReasoningReplay.when_present` отправляет `reasoning_content` в тех ходах
ассистента, где он есть. `required` (DeepSeek) делает это для каждого хода
ассистента, пока размышление включено, — пустым, если у хода его нет, — и
выбрасывает ход, в котором нет ничего, кроме рассуждения (остаток потока,
умершего посреди мысли): сервер не может его использовать.

Непрозрачные рассуждения (`reasoning_details`, зашифрованные вендором) хранятся
в сообщении ассистента как `Message.reasoning_items`, в конверте с именем
эндпоинта и модели, и возвращаются только этому эндпоинту и этой модели:
зашифрованное рассуждение читает только модель, которая его написала.

### Схемы инструментов и пары

Схема инструмента уходит на провод как написана, включая `additionalProperties`,
если инструмент его задаёт. Ядро называет этот флаг `additional_properties` в
`ToolParameterSchema`, и сериализатор, копирующий только `properties` и
`required`, молча его теряет — строгий принудительный инструмент `plan`
становится нестрогим. Выключайте `schema_additional_properties` только для
сервера, который отвергает это ключевое слово.

API отвергает запрос, где у результата инструмента нет вызова перед ним или у
вызова нет результата. История, обрезанная отменой или компакцией, может
содержать и то и другое, поэтому адаптер выбрасывает первое и отвечает на
второе явным `[no result: the call was interrupted before it returned]` —
ошибкой, а не выдуманным успехом, чтобы модель решила, вызывать ли снова.

### Изображения и кэш промптов

С `images=ImageLimits(max_images=..., max_bytes=..., mime_types=...)` и
`image_loader` изображения встраиваются как data URL. То, что эндпоинт бы
отверг (сверх количества — остаются самые свежие; сверх размера; другой
формат; ошибка загрузки), заменяется заметкой с причиной, так что запрос
никогда не падает из-за картинки, а модель всё равно знает, что она была.

`cache_control=True` ставит маркер `cache_control` (с `ttl` для корзин
`5m`/`1h`) на запись провода, которую называет каждая `CacheBreakpoint` ядра.
`observability.routing_key()` запроса — заданный `QueryEngineConfig.cache_key`,
иначе сессия, иначе прогон — уходит в `EndpointSpec.cache_key_field` (например,
`prompt_cache_key` в API OpenAI) и/или `cache_key_header` (например,
`x-grok-conv-id`, который документирует xAI), чтобы запросы одного разговора
попадали на машину, где лежит его кэш.

## Поток

`stream_with_tools` отдаёт `ProviderDelta` в том порядке, в каком их читает
цикл: текст и размышление по мере прихода, вызовы инструментов по мере сборки,
затем `tool_use_stop` на каждый вызов, `usage` и `finish`.

- **Признаки жизни.** SSE-комментарий, чанк только с ролью и пустая дельта
  дают дельту `progress`. Сторож простоя ядра считает от последней дельты; без
  них модель, думающая молча, обрезалась как мёртвый поток.
- **Вызовы инструментов** собираются из фрагментов по `index`; сервер, который
  нумерует каждый вызов `0`, различается по id, а аргументы, пришедшие до
  имени, отдаются вместе с ним. Аргументы, которые не разбираются, были
  обрезаны: стоп несёт починенный префикс, `args_partial_truncated=True`, и
  `truncated_by_output_cap=True`, когда `finish_reason` был `length`.
- **Ошибки после 200.** Чанк, который является объектом `error`, — такой же
  отказ, как любой другой, и читается тем же классификатором.
- **Usage** нормализуется: `input_tokens` — весь промпт (включая кэшированную
  часть), а попадание в кэш берётся из `prompt_tokens_details.cached_tokens`,
  `prompt_cache_hit_tokens` у DeepSeek или `cache_read_input_tokens`;
  сообщённая `cost` становится `cost_usd`.

### Сторож первого байта

`first_byte_timeout_seconds` (выключен при `0`) ограничивает ожидание
заголовков ответа и первой строки тела вместе: сервер, который сразу ответил
заголовками и замолчал, застрял так же, как тот, что не ответил вовсе.
Считается любая строка, включая SSE-комментарий. По таймауту ответ закрывается,
и то же тело отправляется снова, байт в байт, чтобы кэш промптов увидел тот же
префикс; это безопасно, потому что ничего не вернулось и повторить нечего.
Перед каждой повторной отправкой циклу уходит дельта `progress`, чтобы сторож
простоя не сложил два окна. После `first_byte_retries` повторов молчание
поднимается как `LLMTimeoutError` (`timeout`, повторяемо), что сперва двигает
цепочку. Отказ — это ответ, и он никогда не отправляется повторно. Не
включайте сторож для локального сервера, который может минутами читать длинный
промпт до первого байта.

## Сбои

Каждая ошибка адаптера несёт `ProviderFailure`
([Как сообщить о сбое провайдера](contracts.md#как-сообщить-о-сбое-провайдера)),
так что `assert_adapter_failure` её принимает:

| Ответ | Поднимается как | Причина | Повторяемо |
|---|---|---|---|
| 401, 403, тип ошибки авторизации | `LLMProviderError` | `auth` | нет |
| 402, `insufficient_quota` (даже на 429) | `LLMProviderError` | `billing` | нет |
| 429, тип ошибки лимита | `LLMRateLimitError` | `rate_limit` | да, с указанным ожиданием |
| 408, 504 | `LLMTimeoutError` | `timeout` | да |
| 503, 529, тип ошибки перегрузки | `LLMProviderError` | `overloaded` | да, с указанным ожиданием |
| 404, «model does not exist» | `LLMProviderError` | `model_not_found` | нет |
| код политики контента | `LLMProviderError` | `provider_policy_blocked` | нет |
| отказ по окну контекста | `LLMContextWindowExceeded` с указанными размерами | `context_overflow` | нет (цикл компактирует) |
| прочие 413 | `LLMProviderError` | `payload_too_large` | нет |
| любой другой 4xx | `LLMProviderError` | `format_error`, `should_fallback` для ограниченного запроса | нет |
| 5xx | `LLMProviderError` | `server_error` | да |
| отказ в соединении, сброс, обрыв посреди потока | `LLMProviderError` | `connection` | да |
| таймаут клиента | `LLMTimeoutError` | `timeout` | да |

Ожидание берётся из `retry-after-ms`, затем `Retry-After` (секунды или дата),
затем, на 429, — позднее из `x-ratelimit-reset-requests` и
`x-ratelimit-reset-tokens`; всё дальше суток игнорируется, а цикл ограничивает
его собственным потолком. Размеры контекста читаются из формулировок vLLM и
OpenAI («maximum context length is N tokens ... (P in the messages, O in the
completion)», «your messages resulted in P tokens», «you requested O output
tokens and your prompt contains at least P input tokens»), шлюзов в форме
Anthropic («prompt is too long: P tokens > N maximum») и llama.cpp
(`n_prompt_tokens` / `n_ctx` в ошибке или «request (P tokens) exceeds the
available context size (N tokens)»).

## Расширение адаптера

Три точки предназначены для переопределения или передачи:

- `build_body(request, *, stream)` — вызовите базовый метод и добавьте поля,
  нужные развёртыванию. Тело строится один раз на вызов и переиспользуется при
  повторе сторожа.
- `request_headers(request)` — добавьте заголовки.
- `on_usage(UsageObservation)` и `on_failure(FailureObservation)` — слышат
  каждый отчёт об usage (для учёта или лимита трат) и каждый отказ (для паузы
  провайдера или чтобы узнать настоящее окно модели из слов сервера). Упавший
  наблюдатель логируется и никогда не скрывает собственный результат или
  ошибку вызова. `FailureObservation` хранит только заголовки повтора, так что
  не может сохранить учётные данные, которые вернул прокси.

## Точный подсчёт токенов

С `EndpointSpec(tokenize=TokenizeDialect.vllm)` или
`TokenizeDialect.llama_cpp` провайдер реализует
`IRequestTokenCounter.count_request_tokens`: строит тело, которое отправил бы,
и считает его на сервере, включая шаблон, инструменты и переключатель
размышления. vLLM считает chat-запрос на `POST /tokenize`; llama.cpp рендерит
его на `POST /apply-template` и считает промпт на `POST /tokenize`. Оба
маршрута живут в корне сервера (`base_url` без `/v1` или `tokenize_url`).
Сервер без маршрута один раз отвечает 404, и его больше не спрашивают; тогда
подсчёт — `None`, и ядро берёт собственную оценку, которую возвращает и
`count_tokens`.

## Цепочка провайдеров

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

Обход идёт в одну сторону и под замком, так что два сбоя, пришедшие вместе, не
могут оба решать из одной позиции. На каждом `advance` запасная ступень
пропускается, с причиной в `attempted()`, когда:

- её окно не вмещает промпт, который несёт прогон, плюс ответ, который она
  просит (понижение на неё меняет зависший поток на отказ по окну);
- её называет проверка `resting` (аккаунт, отказавший из-за исчерпанной квоты
  минуту назад);
- её `factory` бросает исключение (нет ключа или плохой URL — факт об этой
  ступени).

Ступень с `factory` строится только тогда, когда прогон до неё доходит, так что
прогон, не покинувший основную, не открывает пулов для остальных. Основная
ступень не проверяется никогда, а неизвестное окно или размер промпта никогда
не исключают ступень. При возобновлении движок заново проходит записанное
число продвижений и сверяет имя модели; ступень, пропущенная тогда, но
подходящая сейчас, меняет, куда они приводят, и движок отвергает снимок, а не
обслуживает его с неверной ступени.

> Перевод английского оригинала `docs/providers.md` (коммит `59caafb`). При изменении оригинала обновите перевод.
