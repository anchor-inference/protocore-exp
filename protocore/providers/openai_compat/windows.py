"""Asking a chat-completions server how long a model's context is.

:class:`ListingWindowProbe` is a :data:`~protocore.providers.context_window.WindowProbe`
for :class:`~protocore.providers.context_window.ContextWindowResolver`. It
reads what the server publishes: the model listing (``GET {base}/models``,
and ``{base}/v1/models`` for a base given without ``/v1``), the model's own
entry (``GET .../models/{id}``, which Anthropic-compatible servers answer with
``max_input_tokens``), and, for a llama.cpp server, ``GET /props``, the
length the server was actually started with.

Every failure is quiet. A 404 from a server that does not have the route is
the expected answer, not an error, and the resolver falls through to its other
sources.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from protocore.providers.context_window import (
    FoundWindow,
    find_entry,
    listing_rows,
    plausible_window,
    window_from_entry,
)
from protocore.providers.openai_compat.spec import EndpointSpec

#: Per request. A probe runs when a host configures a model or in a background
#: sweep, but whoever waits for it should not wait long for a server that is
#: not there.
_TIMEOUT = httpx.Timeout(8.0, connect=4.0)


class ListingWindowProbe:
    """Reads a model's window from what the server at ``base_url`` publishes.

    ``props=True`` asks llama.cpp's ``/props`` first, which reports the
    context the server was started with: the model may have been trained for
    more, but the server serves only this much. ``client`` is the HTTP client
    to use (one is made, and closed by :meth:`aclose`, when none is given).
    Redirects are never followed, because the request may carry the key and a
    redirect is no permission to send it to another origin.
    """

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.AsyncClient | None = None,
        api_key: str = "",
        headers: dict[str, str] | None = None,
        props: bool = False,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be an http(s) URL")
        self._base = base_url.rstrip("/")
        self._root = self._base.removesuffix("/v1")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=_TIMEOUT)
        self._headers = {"accept": "application/json", **(headers or {})}
        if api_key:
            self._headers["authorization"] = f"Bearer {api_key}"
        self._props = props

    @classmethod
    def for_endpoint(
        cls, endpoint: EndpointSpec, *, client: httpx.AsyncClient | None = None, props: bool = False
    ) -> ListingWindowProbe:
        """A probe for the server an :class:`EndpointSpec` describes, with its key and headers."""
        return cls(
            endpoint.base_url, client=client, api_key=endpoint.api_key, headers=dict(endpoint.headers), props=props
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __call__(self, model: str) -> FoundWindow | None:
        if self._props:
            found = await self._from_props()
            if found is not None:
                return found
        candidates = [self._base + "/models"]
        if not self._base.endswith("/v1"):
            candidates.append(self._base + "/v1/models")
        for url in candidates:
            rows = listing_rows(await self._get(url))
            if not rows:
                continue
            entry = find_entry(rows, model)
            tokens = window_from_entry(entry) if entry is not None else None
            if tokens is not None:
                return FoundWindow(tokens, "/models")
            tokens = window_from_entry(await self._get(url + "/" + quote(model, safe="")))
            if tokens is not None:
                return FoundWindow(tokens, "/models/{id}")
            # The listing answered and did not say: another listing at
            # another path is the same server and will not say either.
            return None
        return None

    async def _from_props(self) -> FoundWindow | None:
        props = await self._get(self._root + "/props")
        if not isinstance(props, dict):
            return None
        settings = props.get("default_generation_settings")
        nested = settings.get("n_ctx") if isinstance(settings, dict) else None
        tokens = plausible_window(nested) or plausible_window(props.get("n_ctx"))
        return FoundWindow(tokens, "/props") if tokens is not None else None

    async def _get(self, url: str) -> Any:
        try:
            response = await self._client.get(url, headers=self._headers, follow_redirects=False)
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            return response.json()
        except ValueError:
            return None


__all__ = ["ListingWindowProbe"]
