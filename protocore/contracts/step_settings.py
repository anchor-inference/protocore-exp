"""The settings one model step actually ran with.

Settings change while a run is going: an operator switches the model or the
effort, a provider chain falls back to its next rung, a tool group's rules are
given, the surface grows. Each change takes effect at the next request, never
inside the one already in flight, so "which model answered this step" is a
question about the request, and the current settings answer it only until the
next change. A host that read them afterwards to label a step labelled it with
whatever had been set since.

So every request the loop builds for a model step leaves a
:class:`StepSettings` behind it: what the request was built with, a digest of
those values, and the names of the values that differ from the step before.
The digest says whether two steps ran alike; the changed names say when a
change took effect. Nothing secret is in it — model names, digests and limits,
never a prompt, a credential or a tool's arguments.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

#: The values a step's settings are compared and digested by, in a fixed order.
#: Identity of the step (``turn_id``, ``attempt``) and the manifest id are not
#: among them: they differ on every request by construction, and a digest that
#: always changes says nothing about whether the settings did.
SETTINGS_FIELDS: Final[tuple[str, ...]] = (
    "model",
    "provider_model",
    "provider_rung",
    "thinking_enabled",
    "reasoning_effort",
    "forced_tool",
    "tool_surface_digest",
    "tool_count",
    "tool_policy_digest",
    "rules_mark",
    "execution_profile",
    "run_mode",
    "max_output_tokens",
    "context_window",
    "max_turns_per_run",
    "tool_timeout_seconds",
)


class StepSettings(BaseModel):
    """What one request of one model step was built with."""

    model_config = ConfigDict(frozen=True)

    turn_id: str
    """The wire round the request belongs to, as every frame of it is stamped."""
    attempt: int = 1
    """Which request of the step this is. A second request in the same round —
    a reactive retry, a fallback rung — is a second attempt, and its record
    says what changed for it."""
    model: str
    """The model the request named."""
    provider_model: str | None = None
    """The model of the provider chain's current rung, when a chain is wired."""
    provider_rung: int = 0
    """How many times the provider chain has advanced in this run."""
    thinking_enabled: bool | None = None
    reasoning_effort: str | None = None
    forced_tool: str | None = None
    """The tool the request forced, when it forced one."""
    tool_surface_digest: str = ""
    """The digest of the tool definitions the request advertised."""
    tool_count: int = 0
    tool_policy_digest: str = ""
    """The digest of the visibility policy the step's calls are checked against."""
    rules_mark: str = ""
    """The run's mark for the tool-group rules it has given so far."""
    execution_profile: str = ""
    run_mode: str = ""
    max_output_tokens: int | None = None
    context_window: int = 0
    max_turns_per_run: int = 0
    tool_timeout_seconds: float = 0.0
    request_manifest_id: str | None = None
    """The manifest of the request, when the host keeps manifests."""
    digest: str = ""
    """SHA-256 over :data:`SETTINGS_FIELDS`, so two steps compare by one value."""
    changed: tuple[str, ...] = ()
    """The settings fields that differ from the step before, in
    :data:`SETTINGS_FIELDS` order. Empty for a run's first step, and for one
    resumed from a snapshot that recorded no step."""


def _settings_digest(values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        [values.get(name) for name in SETTINGS_FIELDS],
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def settle_step_settings(
    values: Mapping[str, Any],
    *,
    previous: StepSettings | None,
) -> StepSettings:
    """Build a step's record from its values and the record before it.

    ``values`` holds the identity and the settings fields; the digest, the
    attempt number and the changed names are worked out here so that every
    caller derives them the same way.
    """
    fields = {key: value for key, value in values.items() if key not in ("digest", "changed", "attempt")}
    draft = StepSettings(**fields)
    current = draft.model_dump(include=set(SETTINGS_FIELDS))
    changed: tuple[str, ...] = ()
    attempt = 1
    if previous is not None:
        before = previous.model_dump(include=set(SETTINGS_FIELDS))
        changed = tuple(name for name in SETTINGS_FIELDS if before.get(name) != current.get(name))
        if previous.turn_id == draft.turn_id:
            attempt = previous.attempt + 1
    return draft.model_copy(
        update={"digest": _settings_digest(current), "changed": changed, "attempt": attempt}
    )


__all__ = ["SETTINGS_FIELDS", "StepSettings", "settle_step_settings"]
