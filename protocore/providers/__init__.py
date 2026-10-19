"""Reference implementations of the provider contracts.

The contracts in :mod:`protocore.contracts.llm` are what the loop calls; this
package holds implementations a host can use as they are or subclass:

* :mod:`protocore.providers.chain` — :class:`~protocore.providers.chain.ListProviderChain`,
  an :class:`~protocore.contracts.llm.IProviderChain` over an ordered list of
  rungs. Pure Python, no extra dependency.
* :mod:`protocore.providers.openai_compat` — an
  :class:`~protocore.contracts.llm.ILLMProvider` for servers that speak the
  OpenAI chat-completions API. It needs ``httpx``, installed by the ``openai``
  extra (``pip install "protocore[openai]"``), and is imported only when asked
  for: nothing in the core imports it, so a host without the extra loses
  nothing but this module.
* :mod:`protocore.providers.text_tool_calls` — a guard that recovers tool calls
  a model wrote as reply text, with pluggable markups (DSML first). Pure
  Python; the reference adapter applies it per endpoint.
* :mod:`protocore.providers.spend` — :class:`~protocore.providers.spend.ModelPricing`
  and the spend admission hook a host enforces a budget through. Pure Python.
* :mod:`protocore.providers.context_window` —
  :class:`~protocore.providers.context_window.ContextWindowResolver`, a model's
  window from a server probe, a catalogue, a table and refusals. Pure Python;
  the HTTP probe is ``protocore.providers.openai_compat.ListingWindowProbe``.
"""
