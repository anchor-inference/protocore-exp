"""An exact prompt size from a server's counting route, for :class:`~protocore.contracts.llm.IRequestTokenCounter`.

The core sizes requests with an estimate and asks the provider for an exact
count only when the estimate is close to a limit. Self-hosted servers can give
that count for the request as they will render it, chat template and tool
definitions included:

* **vLLM** takes the chat ``messages`` and ``tools`` on ``POST /tokenize`` and
  answers with ``count``.
* **llama.cpp server** renders the messages through its template on
  ``POST /apply-template`` and counts the rendered prompt on
  ``POST /tokenize``.

Both mount the routes at the server root, beside ``/v1``. A server without the
route answers 404; the counter then stops asking for the life of the provider
and reports ``None``, which the core reads as "use the estimate".
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from protocore.contracts.llm import LLMProviderError, ProviderFailure, ProviderFailureReason
from protocore.logging_utils import get_logger
from protocore.providers.openai_compat.spec import TokenizeDialect

_logger = get_logger(__name__)

#: Statuses that mean the route is not there, as opposed to a request it refused.
_NO_ROUTE = frozenset({404, 405, 501})

#: Body fields that change what the template renders, passed on to the counter
#: so the count is of the prompt that will actually be sent.
_RENDERING_FIELDS = ("tools", "chat_template_kwargs")


class TokenizeCounter:
    """Counts a chat-completions body on the server that will serve it."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        root: str,
        dialect: TokenizeDialect,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._client = client
        self._root = root.rstrip("/")
        self._dialect = dialect
        self._headers = dict(headers or {})
        self._unavailable = False

    @property
    def available(self) -> bool:
        """``False`` once the server has said it has no counting route."""
        return not self._unavailable

    async def count(self, body: Mapping[str, Any]) -> int | None:
        """The prompt tokens ``body`` renders to, or ``None`` when the server cannot count.

        Raises :class:`~protocore.contracts.llm.LLMProviderError` when the
        server has the route and the count still failed; the core logs it and
        uses its estimate.
        """
        if self._unavailable:
            return None
        rendering = {key: body[key] for key in _RENDERING_FIELDS if body.get(key)}
        if self._dialect is TokenizeDialect.vllm:
            data = await self._post(
                "/tokenize",
                {"model": body.get("model"), "messages": body.get("messages") or [], "add_generation_prompt": True, **rendering},
            )
            if data is None:
                return None
            count = data.get("count")
            if isinstance(count, int) and not isinstance(count, bool):
                return count
            return _token_list_length(data)
        rendered = await self._post("/apply-template", {"messages": body.get("messages") or [], **rendering})
        if rendered is None:
            return None
        prompt = rendered.get("prompt")
        if not isinstance(prompt, str):
            raise self._failure("the template route returned no prompt")
        counted = await self._post("/tokenize", {"content": prompt, "add_special": True})
        if counted is None:
            return None
        return _token_list_length(counted)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        response = await self._client.post(self._root + path, json=payload, headers=self._headers)
        if response.status_code in _NO_ROUTE:
            self._unavailable = True
            _logger.info(
                "%s has no %s route (HTTP %d); exact request counts are off for it",
                self._root,
                path,
                response.status_code,
            )
            return None
        if response.status_code >= 400:
            raise self._failure(f"HTTP {response.status_code} from {path}: {response.text[:200]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise self._failure(f"{path} answered with something that is not JSON") from exc
        if not isinstance(data, dict):
            raise self._failure(f"{path} answered with a JSON {type(data).__name__}, not an object")
        return data

    def _failure(self, detail: str) -> LLMProviderError:
        return LLMProviderError(
            f"token count at {self._root}: {detail}",
            classified=ProviderFailure(
                reason=ProviderFailureReason.server_error, retryable=True, provider_message=detail
            ),
        )


def _token_list_length(data: Mapping[str, Any]) -> int | None:
    tokens = data.get("tokens")
    return len(tokens) if isinstance(tokens, list) else None


__all__ = ["TokenizeCounter"]
