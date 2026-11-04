"""Per-engine token usage accumulator.

``TokenUsage`` slot. Simple monotonic counter; ``add_from_delta`` consumes
the ``ProviderDelta(kind=usage)`` envelope's payload.

Includes cache-stat fields — ``cache_creation_tokens`` and per-turn cache
counters so ``message_stop.tokens_used`` can surface a real-time hit
rate alongside the input/output totals.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TokenUsage:
    """Cumulative + this-turn usage breakdown."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    this_turn_input: int = 0
    this_turn_output: int = 0
    this_turn_cache_read: int = 0
    this_turn_cache_creation: int = 0

    def add(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_read_tokens += cache_read_tokens
        self.cache_creation_tokens += cache_creation_tokens
        self.this_turn_input += input_tokens
        self.this_turn_output += output_tokens
        self.this_turn_cache_read += cache_read_tokens
        self.this_turn_cache_creation += cache_creation_tokens

    def reset_turn(self) -> None:
        self.this_turn_input = 0
        self.this_turn_output = 0
        self.this_turn_cache_read = 0
        self.this_turn_cache_creation = 0

    def this_turn_total(self) -> int:
        return self.this_turn_input + self.this_turn_output

    def this_turn_cache_hit_rate(self) -> float:
        """Cache hit rate ∈ [0, 1] for this turn: ``cache_read / input``.

        ``input_tokens`` is the whole prompt with its cached part inside it,
        which is what every adapter normalises to (an OpenAI-shaped
        ``prompt_tokens`` already is; an Anthropic-shaped count is summed up
        to it). Adding the cache reads on top once more counted them twice,
        and a session that was 90 % cached reported 0.47. Clamped to 1 in
        case a provider reports more cached tokens than prompt tokens; 0 when
        no input was observed.
        """
        if self.this_turn_input <= 0:
            return 0.0
        return min(self.this_turn_cache_read / self.this_turn_input, 1.0)

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "this_turn_input": self.this_turn_input,
            "this_turn_output": self.this_turn_output,
            "this_turn_cache_read": self.this_turn_cache_read,
            "this_turn_cache_creation": self.this_turn_cache_creation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> TokenUsage:
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cache_read_tokens=int(data.get("cache_read_tokens", 0)),
            cache_creation_tokens=int(data.get("cache_creation_tokens", 0)),
            this_turn_input=int(data.get("this_turn_input", 0)),
            this_turn_output=int(data.get("this_turn_output", 0)),
            this_turn_cache_read=int(data.get("this_turn_cache_read", 0)),
            this_turn_cache_creation=int(data.get("this_turn_cache_creation", 0)),
        )


__all__ = ["TokenUsage"]
