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
| `text_tool_calls` | `()` | разметки, которыми модель пишет вызовы инструментов текстом (ниже) |

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

### Вызовы инструментов, написанные текстом

Некоторые модели время от времени пишут вызов инструмента своей разметкой
прямо в ответе, а не структурированным вызовом. Сервер передаёт его как
текст: разметку видит читатель, а ход заканчивается без вызова. Укажите
разметку в `ProviderCapabilities(text_tool_calls=(DSML,))`, и адаптер
пропускает текстовый поток через `TextToolCallGuard`
(`protocore.providers.text_tool_calls`, чистый Python):

- текст до маркера отдаётся сразу (ждут следующего куска только те несколько
  символов, что могут оказаться началом маркера, разрезанного между кусками);
- начиная с маркера текст придерживается, а вместо каждого придержанного
  куска циклу уходит дельта `progress`, чтобы сторож простоя не оборвал поток;
- в конце потока законченный блок становится обычными вызовами (по
  `tool_use_start` и `tool_use_stop` на каждый, id `call_<поток>_text<n>`), а
  завершение `stop` становится `tool_use`; текст вокруг блока отдаётся;
- незаконченный блок отдаётся без изменений, как и придержанный текст,
  который перерос предел (4 000 символов) и не похож на блок вызовов: модель,
  которая лишь упоминает разметку, сохраняет свой ответ.

`DSML` (`DsmlGrammar`) читает разметку DeepSeek `<｜DSML｜tool_calls>` при
любом написании черт; параметр с `string="true"` — текст, любой другой
читается как JSON. Другая разметка — класс с `name`, `markers`,
`is_call_block(held)` и `parse(held) -> ParsedText | None`
(`TextToolCallGrammar`); побеждает самый ранний маркер из перечисленных
грамматик. Страж не привязан к этому адаптеру: любой адаптер может
пропускать через него свои текстовые дельты.

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

## Бюджет расходов и цены

Цикл считает токены; деньги считает хост. Адаптер даёт этому решению одно
место (`protocore.providers.spend`, чистый Python):

```python
from protocore.providers.spend import ModelPricing, SpendRefused

prices = {"model-a": ModelPricing(input=0.30, output=1.20, cached_input=0.03)}

class DailyCap:
    async def reserve(self, estimate):          # перед каждым вызовом
        if spent_today() + (estimate.cost_usd or 0) > CAP:
            raise SpendRefused("daily cap reached", retry_after_seconds=until_midnight())
        return hold(estimate)                   # id резерва или None
    async def settle(self, reservation, settlement):   # вызов сообщил расход
        book(reservation, settlement.cost_usd, complete=settlement.complete)
    async def release(self, reservation, *, outcome, detail):  # расхода не пришло
        drop_or_keep(reservation, outcome)

provider = OpenAICompatibleProvider(spec, admission=DailyCap(), pricing=prices.get)
```

- `reserve(SpendEstimate)` спрашивается перед каждым вызовом (поток, текст,
  структурированный) с эндпоинтом, моделью, назначением, оценкой размера
  промпта, потолком вывода, верхней границей цены в `cost_usd` (когда цены
  модели известны) и контекстом наблюдаемости запроса. Исключение
  `SpendRefused` отказывает в вызове: ничего не отправляется, а цикл видит
  `LLMProviderError` с `ProviderFailure` причины `budget` — каждая ступень
  цепочки упирается в тот же потолок, поэтому ничто его не повторяет и не
  обходит. `None` означает, что вызов не нужно учитывать.
- За резервом следует ровно один из `settle` и `release`. `settle(id,
  SpendSettlement)` несёт нормализованный расход, сырой расход, `cost_usd`
  (собственный `cost` сервера, иначе цену по таблице) и `complete`, равный
  `False`, когда сервер не сообщил ни одного счётчика токенов: вызов мог быть
  оплачен, так что хост обычно оставляет резерв потраченным. Он выполняется до
  дельты `usage`, потому что цикл может перестать читать на `finish`.
- `release(id, outcome=..., detail=...)` сообщает, что вызов кончился без
  расхода: `SpendEnd.refused`, когда сервер ответил 4xx или 5xx (обычно не
  оплачивается), `SpendEnd.interrupted`, когда запрос ушёл и не закончился —
  оборванный поток, тайм-аут первого байта, вызывающий перестал читать (вывод
  мог уже быть сгенерирован).

Упавший `settle` или `release` пишется в лог и никогда не заменяет
собственный результат или ошибку вызова. Переопределите
`estimate_spend(request, body, purpose)`, чтобы оценивать промпт точно; по
умолчанию это оценка ядра для сообщений и инструментов в том виде, в каком
они уходят на провод, — ей достаточно ошибаться в большую сторону.

`ModelPricing(input, output, cached_input=None, cache_write=None,
reasoning=None)` хранит доллары США за миллион токенов. `cost(usage)`
оценивает свежий ввод, чтение кэша, запись в кэш, видимый вывод и
рассуждения каждый по своей ставке, читая ключи, которые сообщает адаптер
(`input_tokens` — весь промпт вместе с кэшем; `output_tokens` включает
`reasoning_tokens`). Не указанная цена кэша равна цене ввода, а не указанная
цена рассуждений — цене вывода, так что пропущенное число никогда не делает
токены бесплатными. `estimate(prompt_tokens=, max_output_tokens=)` — верхняя
граница: ни одного попадания в кэш, весь потолок вывода по более дорогой
ставке. С `pricing=` расход без цены от сервера получает `cost_usd` из
таблицы — в дельте `usage` и в `LLMResponse.usage.response_cost_usd`.

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

## Как узнать окно контекста модели

Окно читает всё, что задаёт размеры прогона, а число, введённое руками, —
вопрос, на который многие операторы ответить не могут. `ContextWindowResolver`
(`protocore.providers.context_window`, чистый Python) берёт его из первого
источника, который знает:

1. **собственный список моделей сервера** — через пробу;
2. **каталог** в форме `api.json` models.dev — файл, который хост скачивает
   и передаёт (`WindowCatalogue.from_file(path)`);
3. **таблица** шаблонов семейств моделей, которую ведёт хост
   (`WindowTable({"family-*": 128_000})` или `WindowTable.from_toml(path)`,
   читающий раздел `[windows]`; побеждает самый конкретный шаблон, а
   сравнивается id в нижнем регистре без префикса `vendor/` и `:tag`);
4. **отказ**: сервер, который сказал «maximum context length is N tokens»,
   назвал настоящий предел, и если N меньше, чем говорят источники выше, оно
   побеждает их все.

```python
from protocore.providers.context_window import ContextWindowResolver, WindowCatalogue
from protocore.providers.openai_compat import ListingWindowProbe

resolver = ContextWindowResolver(
    probe=ListingWindowProbe.for_endpoint(spec, props=is_llama_cpp),
    catalogue=WindowCatalogue.from_file(Path("models.json")),
)
provider = OpenAICompatibleProvider(spec, on_failure=resolver.observe_failure)
window = await resolver.resolve("model-a")      # ResolvedWindow(tokens, source, detail) или None
chain = ListProviderChain(rungs, window_of=resolver.rung_window)
```

Ничего не поставляется в комплекте: какие модели существуют и каков их
размер, меняется каждый месяц, и число, зашитое в библиотеку, устаревает
задолго до того, как его кто-нибудь обновит. Когда ни один источник не знает,
`resolve` возвращает `None`, и хост оставляет своё число.

- **Проба.** `ListingWindowProbe(base_url, *, client=None, api_key="",
  headers=None, props=False)` читает `GET {base}/models` (и `{base}/v1/models`
  для базы без `/v1`), затем собственную запись модели `GET .../models/{id}`;
  с `props=True` она сначала читает `GET /props` llama.cpp — длину, с которой
  запущен сервер, а не ту, на которой обучалась модель. Окно читается из того
  поля, которое использует сервер (`max_model_len`, `context_length`,
  `context_window`, `max_context_length`, `loaded_context_length`,
  `max_input_tokens`, `inputTokenLimit`, `limit.context`; `ENTRY_FIELDS`
  перечисляет их по порядку). Любой сбой тих, а перенаправления не
  выполняются, потому что запрос может нести ключ. Пробой служит любая
  асинхронная функция `model -> FoundWindow | None`.
- **Кэш.** Ответ пробы хранится `ttl_seconds` (сутки), а промах —
  `miss_ttl_seconds` (десять минут), так что сервер, который никогда не
  отвечает, не спрашивают на каждом прогоне, а тот, что был недоступен,
  спросят снова скоро. Одновременные `resolve` одной модели делят одну пробу.
  Проба, бросившая исключение, пишется в лог и считается промахом.
- **Отказы.** `observe_failure` имеет форму наблюдателя `on_failure`
  адаптера и учится только на отказе, классифицированном как
  `context_overflow`; `learn_refusal(model, text)` и
  `remember_refusal(model, tokens)` делают это вручную. `stated_window(text)`
  читает формулировки OpenAI/vLLM, Anthropic, llama.cpp, Google и LM Studio.
  Большее названное окно никогда не заменяет меньшее. Выученные окна живут,
  пока жив резолвер; `learned()` отдаёт их хосту, который их сохраняет, а
  `forget(model)` начинает заново.
- **Без ожидания.** `known(model)` отвечает из уже найденного или
  выученного, никого не спрашивая, — для решений, которые не могут ждать
  запроса. `rung_window` — то же для ступени цепочки:
  `ListProviderChain(window_of=...)` использует его в проверке вместимости
  (ниже).

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

С `window_of=` (например, `rung_window` резолвера) проверка вместимости берёт
меньшее из настроенного `context_window` ступени и окна, найденного с тех
пор: если право хоть одно из них, большее — это запрос, от которого сервер
откажется.

Ступень с `factory` строится только тогда, когда прогон до неё доходит, так что
прогон, не покинувший основную, не открывает пулов для остальных. Основная
ступень не проверяется никогда, а неизвестное окно или размер промпта никогда
не исключают ступень. При возобновлении движок заново проходит записанное
число продвижений и сверяет имя модели; ступень, пропущенная тогда, но
подходящая сейчас, меняет, куда они приводят, и движок отвергает снимок, а не
обслуживает его с неверной ступени.

> Перевод английского оригинала `docs/providers.md` (коммит `59caafb`). При изменении оригинала обновите перевод.
