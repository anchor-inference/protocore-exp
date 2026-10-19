"""The scenario harness: a run described as a script, driven as a host drives it.

Every test under ``tests/scenarios`` states what the model says and what the
tools do, drives the engine through one of the four entry points a host
actually calls — ``QueryEngine.run``, ``QueryEngine.stop``,
``QueryEngine.resume_from_snapshot`` on a NEW instance, and
``resume_approved_tool`` — and then asserts on what is visible from outside
the engine:

* the ``LLMRequest`` objects the provider received (what the model was shown);
* the ``TurnEvent`` stream the caller iterated (what the reader saw);
* ``history_snapshot()`` and ``snapshot()`` (what survives the process).

No scenario reads a private attribute of the engine or imports a private name
from the runtime. That is the point of the suite: it is the part of the test
set that keeps working when the inside of the loop is rewritten, and
``tools/private_symbols.py`` measures whether it stayed that way.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from protocore.contracts.hooks import HookActionKind, HookResult
from protocore.contracts.llm import (
    ILLMProvider,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from protocore.contracts.runtime_constants import LoopConstants
from protocore.contracts.tools import Tool, ToolContext
from protocore.contracts.types import (
    HookEvent,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
)
from protocore.runtime.events import EventType, TurnEvent
from protocore.runtime.query_engine import QueryEngine, QueryEngineConfig
from protocore.runtime.turn_policies import TurnPolicyRegistry
from protocore.tests_support.adapters import (
    InMemoryBlobStore,
    InMemoryEventStream,
    InMemoryHookManager,
    InMemoryLLMProvider,
    InMemorySkillStore,
    InMemoryToolRegistry,
)
from tests._fixtures.delegation import DelegationContract

# ---------------------------------------------------------------------------
# Scripted tools
# ---------------------------------------------------------------------------


@dataclass
class ScriptedTool(Tool):
    """A tool whose every call is recorded and whose behaviour is scripted.

    ``invocations`` is the record a scenario asserts on — "this side effect
    happened exactly once" is the only way a test can tell a resumed run that
    re-executed a settled call from one that did not.
    """

    tool_name: str = "Note"
    description: str = "record a note"
    content: str = "ok"
    #: The file this tool's result is a view of, when it is a view of one.
    path: str | None = None
    is_error: bool = False
    delay_seconds: float = 0.0
    raises: BaseException | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    invocations: list[dict[str, Any]] = field(default_factory=list)
    #: The context each call was given, so a scenario can assert on what a tool
    #: is handed rather than on what the engine holds.
    contexts: list[ToolContext] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def name(self) -> str:
        return self.tool_name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.tool_name,
            description=self.description,
            parameters=ToolParameterSchema(properties={"v": {"type": "string"}}),
        )

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(arguments))
        self.contexts.append(context)
        self.started.set()
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.raises is not None:
            raise self.raises
        return ToolResult(
            tool_call_id="",
            content=self.content,
            is_error=self.is_error,
            metadata=dict(self.metadata),
            path=self.path,
        )


@dataclass
class DelegationTool(DelegationContract, ScriptedTool):
    """A delegation tool — one whose call spawns a whole nested run.

    Core identifies delegation by the contract a tool implements, never by the
    tool's name, so the scenario implements it and leaves the name to the host.
    """

    tool_name: str = "Delegate"
    description: str = "hand work to a subagent"
    is_concurrent_safe: bool = False

    async def invoke(self, context: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        self.invocations.append(dict(arguments))
        self.contexts.append(context)
        self.started.set()
        delay = float(arguments.get("delay", self.delay_seconds) or 0.0)
        if delay:
            await asyncio.sleep(delay)
        if arguments.get("fail"):
            raise RuntimeError(f"child failed: {arguments.get('v')}")
        if self.raises is not None:
            raise self.raises
        return ToolResult(
            tool_call_id="",
            content=f"{self.content}:{arguments.get('v', '')}",
            is_error=self.is_error,
        )


# ---------------------------------------------------------------------------
# Scripted providers beyond what the shipped double covers
# ---------------------------------------------------------------------------


class FailingProvider(ILLMProvider):
    """A provider that raises a scripted failure on scripted calls.

    ``failures`` maps a zero-based call index to the exception raised at that
    call; every other call streams ``text``.  ``partial_text`` is emitted as a
    delta BEFORE the failure, which is what makes the partial-into-the-next-
    provider question askable at all: the reader has already seen those
    characters when the stream dies. ``tool_calls`` makes a surviving round
    ask for tools instead of answering, so a scenario can carry a run past a
    provider failure and into the dispatch path.
    """

    def __init__(
        self,
        *,
        failures: dict[int, BaseException],
        text: str = "done",
        partial_text: str = "",
        tool_calls: Sequence[tuple[str, str, dict[str, Any]]] = (),
    ) -> None:
        self._failures = failures
        self._text = text
        self._partial = partial_text
        self._tool_calls = list(tool_calls)
        self._calls: list[LLMRequest] = []

    @property
    def calls(self) -> Sequence[LLMRequest]:
        return tuple(self._calls)

    async def stream_with_tools(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        index = len(self._calls)
        self._calls.append(request)
        yield LLMStreamEvent(name="message_start", payload={})
        yield LLMStreamEvent(name="content_block_start", payload={"index": 0, "kind": "text"})
        failure = self._failures.get(index)
        if failure is not None:
            if self._partial:
                yield LLMStreamEvent(
                    name="content_block_delta",
                    payload={"index": 0, "text": self._partial},
                )
            raise failure
        if self._tool_calls:
            yield LLMStreamEvent(name="content_block_stop", payload={"index": 0})
            for tool_call_id, tool_name, arguments in self._tool_calls:
                yield LLMStreamEvent(
                    name="tool_use_start",
                    payload={"tool_call_id": tool_call_id, "tool_name": tool_name},
                )
                yield LLMStreamEvent(
                    name="tool_use_stop",
                    payload={
                        "tool_call_id": tool_call_id,
                        "final_input": dict(arguments),
                    },
                )
            yield LLMStreamEvent(
                name="message_stop", payload={"stop_reason": StopReason.tool_use.value}
            )
            return
        yield LLMStreamEvent(
            name="content_block_delta", payload={"index": 0, "text": self._text}
        )
        yield LLMStreamEvent(name="content_block_stop", payload={"index": 0})
        yield LLMStreamEvent(
            name="message_stop", payload={"stop_reason": StopReason.end_turn.value}
        )

    async def complete_structured(
        self, request: LLMRequest, response_schema: dict[str, Any]
    ) -> LLMResponse:
        self._calls.append(request)
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    async def complete_text(self, request: LLMRequest) -> LLMResponse:
        self._calls.append(request)
        return LLMResponse(
            message=Message(role=MessageRole.assistant, content_blocks=[]),
            stop_reason=StopReason.end_turn,
        )

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return max(1, len(text) // 4) if text else 0


class ProviderChainDouble:
    """An ordered list of providers plus the one-way cursor over it.

    Matches ``IProviderChain``.  ``advance_reasons`` records what the run said
    when it stepped down, so a scenario can assert the demotion happened for
    the reason the loop classified rather than merely that it happened.
    """

    def __init__(self, rungs: Sequence[tuple[str, ILLMProvider]]) -> None:
        self._rungs = list(rungs)
        self._index = 0
        self.advance_reasons: list[str] = []

    def current(self) -> ILLMProvider:
        return self._rungs[self._index][1]

    def current_model_name(self) -> str:
        return self._rungs[self._index][0]

    async def advance(self, *, reason: str) -> bool:
        self.advance_reasons.append(reason)
        if self._index + 1 >= len(self._rungs):
            return False
        self._index += 1
        return True

    def attempted(self) -> Sequence[tuple[str, str]]:
        return tuple((name, "demoted") for name, _ in self._rungs[: self._index])


class Verdict:
    """The classification a provider adapter pins onto what it raises."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


def classified(exc: BaseException, reason: str) -> BaseException:
    """Attach an adapter verdict to ``exc`` and return it."""
    object.__setattr__(exc, "classified", Verdict(reason))
    return exc


# ---------------------------------------------------------------------------
# The scenario itself
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One scripted run and everything a test may look at afterwards."""

    engine: QueryEngine
    llm: InMemoryLLMProvider
    tools: InMemoryToolRegistry
    hooks: InMemoryHookManager
    #: Where the run's shed content goes. A scenario that asserts a value
    #: SURVIVED compaction has to read it back from here — a placeholder in
    #: the transcript proves the text is gone, not that the value is kept.
    blobs: InMemoryBlobStore = field(default_factory=InMemoryBlobStore)
    #: The provider the engine was constructed on — the scripted double above
    #: unless the scenario supplied its own. ``requests`` reads this one.
    provider: Any = None
    #: The provider the summariser talks to, when the scenario wired a separate
    #: one. A host may point compaction at a cheaper model than the run's, and
    #: a scenario that does the same can count summariser calls without
    #: separating them from the turn's own by inspection.
    summariser: Any = None
    events: list[TurnEvent] = field(default_factory=list)

    # -- driving the run, only through public entry points -------------------

    async def run(self, text: str | None = "go") -> list[TurnEvent]:
        """Drive one turn. ``text=None`` continues the existing history."""
        message = (
            None
            if text is None
            else Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])
        )
        produced = [evt async for evt in self.engine.run(message)]
        self.events.extend(produced)
        return produced

    async def run_and_stop_when(
        self,
        predicate: Callable[[], bool],
        *,
        text: str = "go",
        poll_seconds: float = 0.005,
        deadline_seconds: float = 5.0,
    ) -> list[TurnEvent]:
        """Drive one turn and call ``stop()`` from ANOTHER task once ``predicate``.

        This is the shape a host cancel has: the operator's request arrives on
        a different task than the one iterating the run, which is the only way
        ``stop()`` can interrupt an ``await`` that is already in flight.
        """
        produced: list[TurnEvent] = []

        async def drive() -> None:
            async for evt in self.engine.run(
                Message(role=MessageRole.user, content_blocks=[TextBlock(text=text)])
            ):
                produced.append(evt)

        driver = asyncio.create_task(drive())
        waited = 0.0
        while not predicate() and not driver.done() and waited < deadline_seconds:
            await asyncio.sleep(poll_seconds)
            waited += poll_seconds
        self.engine.stop()
        try:
            await driver
        except asyncio.CancelledError:
            pass
        self.events.extend(produced)
        return produced

    # -- what the model was shown -------------------------------------------

    @property
    def requests(self) -> Sequence[LLMRequest]:
        source = self.llm if self.provider is None else self.provider
        return source.calls

    def request_texts(self, index: int) -> list[str]:
        """Everything readable in the messages of request ``index``.

        Text blocks and tool-result bodies together: what the model is shown
        does not distinguish them, and neither should an assertion about what
        the model was shown.
        """
        readable: list[str] = []
        for message in self.requests[index].messages:
            for block in message.content_blocks:
                if isinstance(block, TextBlock):
                    readable.append(block.text)
                elif isinstance(block, ToolResultBlock):
                    readable.append(block.content)
        return readable

    def advertised_tool_names(self, index: int) -> list[str]:
        return [definition.name for definition in self.requests[index].tools]

    # -- what the reader saw -------------------------------------------------

    def event_types(self, produced: Iterable[TurnEvent] | None = None) -> list[EventType]:
        return [evt.type for evt in (self.events if produced is None else produced)]

    def events_of(self, kind: EventType) -> list[TurnEvent]:
        return [evt for evt in self.events if evt.type is kind]

    def state_reasons(self) -> list[str]:
        return [
            str(evt.payload.get("reason", ""))
            for evt in self.events_of(EventType.STATE_CHANGED)
        ]

    # -- what survives the process ------------------------------------------

    def history_texts(self) -> list[str]:
        return [
            block.text
            for message in self.engine.history_snapshot()
            for block in message.content_blocks
            if isinstance(block, TextBlock)
        ]

    def tool_results(self) -> list[ToolResultBlock]:
        return [
            block
            for message in self.engine.history_snapshot()
            for block in message.content_blocks
            if isinstance(block, ToolResultBlock)
        ]

    def tool_uses(self) -> list[ToolUseBlock]:
        return [
            block
            for message in self.engine.history_snapshot()
            for block in message.content_blocks
            if isinstance(block, ToolUseBlock)
        ]


def default_rc(**overrides: Any) -> LoopConstants:
    """Runtime constants for a scenario, small window unless told otherwise."""
    values: dict[str, Any] = {"model_context_window": 4_096}
    values.update(overrides)
    return LoopConstants(**values)


ScenarioFactory = Callable[..., Scenario]


@pytest.fixture
def scenario() -> ScenarioFactory:
    """Return ``build(**overrides) -> Scenario``.

    The engine is wired from the shipped in-memory adapters, exactly the way
    the host wires it from its real ones: everything the engine needs arrives
    through the constructor, and nothing is reached into afterwards.
    """

    def build(
        *,
        rc: LoopConstants | None = None,
        tools: Sequence[Tool] = (),
        model_name: str = "scenario-model",
        run_id: str = "run-scenario",
        session_id: str = "sess-scenario",
        tenant_id: str = "tenant-scenario",
        llm_provider: ILLMProvider | None = None,
        compaction_provider: ILLMProvider | None = None,
        provider_chain: Any | None = None,
        background_pool: Any | None = None,
        lifecycle_hooks: Any | None = None,
        expected_terminal_tool: str | None = None,
        blob_store: InMemoryBlobStore | None = None,
        turn_policies: TurnPolicyRegistry | None = None,
        event_stream: Any | None = None,
        tool_safety_policies: Any | None = None,
        **config_overrides: Any,
    ) -> Scenario:
        llm = InMemoryLLMProvider()
        registry = InMemoryToolRegistry()
        for tool in tools:
            registry.register(tool)
        hooks = InMemoryHookManager()
        blobs = blob_store or InMemoryBlobStore()
        engine = QueryEngine(
            config=QueryEngineConfig(
                run_id=run_id,
                tenant_id=tenant_id,
                account_id=tenant_id,
                session_id=session_id,
                model_name=model_name,
                rc=rc or default_rc(),
                expected_terminal_tool=expected_terminal_tool,
                **config_overrides,
            ),
            llm_provider=llm_provider or llm,
            compaction_provider=compaction_provider,
            tool_registry=registry,
            event_stream=event_stream or InMemoryEventStream(),
            hook_manager=hooks,
            skill_store=InMemorySkillStore(),
            blob_store=blobs,
            provider_chain=provider_chain,
            background_pool=background_pool,
            lifecycle_hooks=lifecycle_hooks,
            tool_safety_policies=tool_safety_policies,
        )
        # A run driven by a policy set of the caller's choosing. Passing none
        # keeps the core's own set, which is what every other scenario asserts
        # against; passing one is how a scenario states which policies were
        # installed rather than inferring it from the behaviour it sees.
        engine.turn_policies = turn_policies
        return Scenario(
            engine=engine,
            llm=llm,
            tools=registry,
            hooks=hooks,
            blobs=blobs,
            provider=llm_provider,
            summariser=compaction_provider,
        )

    return build


def require_approval(hooks: InMemoryHookManager, *, token: str = "tok-scenario") -> None:
    """Script the permission hook to demand approval for the next tool call."""
    hooks.queue_action(
        HookEvent.pre_tool_use,
        HookResult(
            action=HookActionKind.ALLOW,
            modifications={"requires_approval": True, "approval_token": token},
            reason="awaiting operator",
        ),
    )
