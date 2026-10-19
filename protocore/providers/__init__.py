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
"""
