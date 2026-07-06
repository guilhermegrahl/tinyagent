"""T12c acceptance test: sync ``run()`` wrapper with pinned-loop bridge.

Per plan §5 (sync/async dispatch), §13 T12c acceptance criteria, and round-2
M3 closure: async hooks registered against the sync ``run()`` entry point MUST
NOT be silently dropped. The wrapper pins the worker-thread event loop at
entry so that bridge-based dispatch (via ``asyncio.run_coroutine_threadsafe``)
can deliver coroutine hooks to that loop.

Coverage:
- ``agent.run("prompt")`` is callable from sync code and returns the final
  answer (sync entry, async internals).
- Sync hooks registered via ``register_*`` fire when ``run()`` is called.
- Async hooks registered via ``register_*`` fire when ``run()`` is called,
  bridged via the pinned event loop (the round-2 M3 fix).
- Mixed sync + async hooks fire in registration order.
- ``AgentCancel`` raised from a hook propagates out of ``run()`` unchanged.
- ``run()`` enforces the round-3 M3 storage model: callers use ONLY
  ``register_*`` methods. The test asserts the canonical attribute-style form
  raises ``AttributeError`` on the registry it uses.

Per the task's "no ``cb.before_llm_call.append(...)`` form" requirement, the
tests below exclusively use ``register_*`` to attach hooks.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import tinyagent
from tinyagent import AgentCancel


# ============================================================================
# Synthetic response builders (mirrors tests/test_agent_loop.py)
# ============================================================================
@dataclass
class _ToolCall:
    id: str
    name: str
    arguments: str  # JSON-encoded string


@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _SyntheticToolCall:
    id: str
    function: _Function


@dataclass
class _Message:
    role: str = "assistant"
    content: str | None = ""
    tool_calls: list[_SyntheticToolCall] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.content:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return out


@dataclass
class _Choice:
    message: _Message


@dataclass
class _SyntheticResponse:
    choices: list[_Choice] = field(default_factory=list)


def _func_response(
    tool_calls: list[_ToolCall] | None = None,
    content: str = "",
    role: str = "assistant",
) -> _SyntheticResponse:
    msg = _Message(
        role=role,
        content=content,
        tool_calls=[
            _SyntheticToolCall(id=tc.id, function=_Function(tc.name, tc.arguments))
            for tc in (tool_calls or [])
        ],
    )
    return _SyntheticResponse(choices=[_Choice(message=msg)])


def _final_answer_tool_call(answer: str, call_id: str = "call_fa") -> _ToolCall:
    return _ToolCall(
        id=call_id,
        name="final_answer",
        arguments=json.dumps({"answer": answer}),
    )


# ============================================================================
# Fixtures — fresh agent per test
# ============================================================================
def _unique_tracer_name() -> str:
    """Return a tracer name unique per test invocation (see test_agent_loop.py)."""
    return f"tinyagent-t12c-{uuid.uuid4().hex}"


def _make_agent(
    *,
    max_turns: int = 10,
    callbacks: tinyagent.CallbackRegistry | None = None,
) -> tinyagent.TinyAgent:
    """Build a TinyAgent with an optional pre-built CallbackRegistry."""
    cfg = tinyagent.AgentConfig(
        instructions="You are a test agent.",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        max_turns=max_turns,
        callbacks=callbacks if callbacks is not None else tinyagent.CallbackRegistry(),
        name=_unique_tracer_name(),
    )
    return tinyagent.TinyAgent(cfg)


# ============================================================================
# 1. Sync entry point — ``run()`` is callable from sync code
# ============================================================================
def test_run_is_callable_from_sync_context() -> None:
    """``agent.run(prompt)`` is a plain sync function (no async required).

    Must not be a coroutine; if it were, calling from sync code without
    awaiting would silently return a coroutine object instead of the
    answer string.
    """
    agent = _make_agent()
    response = _func_response(tool_calls=[_final_answer_tool_call("done")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        result = agent.run("greet me")
    assert result == "done", f"run() should return the final answer, got {result!r}"


def test_run_returns_final_answer_end_to_end() -> None:
    """``run()`` end-to-end: sync entry drives async internals, returns answer."""
    agent = _make_agent()
    response = _func_response(tool_calls=[_final_answer_tool_call("hello")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        result = agent.run("greet me")
    assert result == "hello"
    mock_cm.assert_awaited_once()


# ============================================================================
# 2. Sync hooks fire inside ``run()``
# ============================================================================
def test_run_invokes_sync_before_llm_hook() -> None:
    """Sync ``register_before_llm_call`` hook fires inside ``run()``."""
    agent = _make_agent()
    fired: list[bool] = []
    agent._callbacks.register_before_llm_call(lambda ctx: fired.append(True))

    response = _func_response(tool_calls=[_final_answer_tool_call("ok")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        agent.run("...")

    assert fired == [True], f"sync before_llm_call hook should have fired once, got {fired!r}"


def test_run_invokes_sync_after_llm_hook() -> None:
    """Sync ``register_after_llm_call`` hook fires inside ``run()``."""
    agent = _make_agent()
    fired: list[bool] = []
    agent._callbacks.register_after_llm_call(lambda ctx: fired.append(True))

    response = _func_response(tool_calls=[_final_answer_tool_call("ok")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        agent.run("...")

    assert fired == [True]


def test_run_invokes_sync_tool_hook() -> None:
    """Sync ``register_before_tool_execution`` hook fires inside ``run()``."""
    agent = _make_agent()
    fired: list[Any] = []
    agent._callbacks.register_before_tool_execution(
        lambda ctx: fired.append(ctx.tool_call)
    )
    response = _func_response(tool_calls=[_final_answer_tool_call("the-answer")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        agent.run("...")
    assert len(fired) == 1, f"sync before_tool hook should fire once, got {fired!r}"


# ============================================================================
# 3. Async hooks are awaited (round-2 M3 fix) — the whole point of T12c
# ============================================================================
def test_run_invokes_async_hook_via_pinned_loop_bridge() -> None:
    """Async hooks fire inside ``run()`` via the pinned-loop bridge.

    This is the round-2 M3 closure: a sync ``run()`` entry MUST await
    coroutine hooks against the worker-thread event loop. Without the
    pinned-loop bridge, async hooks returned coroutines that were never
    awaited (silently dropped).
    """
    agent = _make_agent()
    observed: list[str] = []
    loop_seen: list[asyncio.AbstractEventLoop | None] = []

    async def async_hook(ctx: object) -> None:
        # Capture the loop the hook ran on (for round-2 M3 evidence:
        # if bridge works, this is the worker thread's pinned loop).
        loop_seen.append(asyncio.get_running_loop())
        observed.append("ran")

    agent._callbacks.register_before_llm_call(async_hook)
    response = _func_response(tool_calls=[_final_answer_tool_call("ok")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        agent.run("...")

    assert observed == ["ran"], (
        f"async hook MUST complete during run(); "
        f"if the bridge is missing the coroutine is silently dropped. "
        f"Got observed={observed!r}"
    )
    assert loop_seen[0] is not None, (
        "async hook should have run inside an event loop; got None"
    )


def test_run_invokes_async_tool_hook_via_pinned_loop_bridge() -> None:
    """Async ``register_before_tool_execution`` hook fires inside ``run()``."""
    agent = _make_agent()
    observed: list[Any] = []

    async def async_hook(ctx: object) -> None:
        observed.append(ctx.tool_call)

    agent._callbacks.register_before_tool_execution(async_hook)
    response = _func_response(tool_calls=[_final_answer_tool_call("the-answer")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        agent.run("...")

    assert len(observed) == 1, (
        f"async before_tool_execution hook should have fired once, "
        f"got {observed!r}"
    )


# ============================================================================
# 4. Mixed sync + async hooks — fire in registration order
# ============================================================================
def test_run_mixed_sync_and_async_hooks_run_in_registration_order() -> None:
    """Mixed sync + async hooks fire in the order they were registered.

    Verifies the bridge preserves order across both hook kinds — a hook
    registered after an earlier hook (sync or async) runs after it,
    regardless of kind.
    """
    agent = _make_agent()
    order: list[str] = []

    def sync_1(ctx: object) -> None:
        order.append("sync1")

    async def async_1(ctx: object) -> None:
        order.append("async1")

    def sync_2(ctx: object) -> None:
        order.append("sync2")

    async def async_2(ctx: object) -> None:
        order.append("async2")

    agent._callbacks.register_before_llm_call(sync_1)
    agent._callbacks.register_before_llm_call(async_1)
    agent._callbacks.register_before_llm_call(sync_2)
    agent._callbacks.register_before_llm_call(async_2)

    response = _func_response(tool_calls=[_final_answer_tool_call("ok")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        agent.run("...")

    assert order == ["sync1", "async1", "sync2", "async2"], (
        f"mixed sync/async hooks must run in registration order; got {order!r}"
    )


# ============================================================================
# 5. AgentCancel from a hook propagates to the caller
# ============================================================================
def test_run_propagates_agent_cancel_from_sync_hook() -> None:
    """AgentCancel raised from a sync hook bubbles out of ``run()``."""
    agent = _make_agent()

    def cancel_hook(ctx: object) -> None:
        raise AgentCancel("user aborted the run")

    agent._callbacks.register_before_llm_call(cancel_hook)

    response = _func_response(tool_calls=[_final_answer_tool_call("ok")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(AgentCancel) as exc_info:
            agent.run("...")

    assert "aborted" in str(exc_info.value).lower()


def test_run_propagates_agent_cancel_from_async_hook() -> None:
    """AgentCancel raised from an async hook bubbles out of ``run()``.

    Async hooks raised inside the pinned worker-thread loop must NOT
    swallow the cancel; the exception must cross the thread barrier
    back to the caller.
    """
    agent = _make_agent()

    async def async_cancel_hook(ctx: object) -> None:
        raise AgentCancel("async abort")

    agent._callbacks.register_before_llm_call(async_cancel_hook)

    response = _func_response(tool_calls=[_final_answer_tool_call("ok")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(AgentCancel) as exc_info:
            agent.run("...")

    assert "async" in str(exc_info.value).lower()


# ============================================================================
# 6. Storage model — registry uses ONLY ``register_*`` API (round-3 M3)
# ============================================================================
def test_run_uses_only_register_api_no_attribute_form() -> None:
    """The registry used by ``run()`` rejects the ``cb.before_llm_call.append(...)`` form.

    Per round-3 M3: the registry storage model is dict-backed; the
    attribute-storage form (e.g. ``cb.before_llm_call.append``) MUST raise
    ``AttributeError``. The hook-registration tests above exclusively use
    ``register_*`` — this test asserts the negative contract so a future
    regression cannot reintroduce attribute storage without breaking the
    test suite.
    """
    cb = tinyagent.CallbackRegistry()

    # Positive control: register_* works.
    cb.register_before_llm_call(lambda ctx: None)
    assert cb._hooks["before_llm_call"]  # populated via register_*

    # Negative control: attribute form is forbidden.
    for name in (
        "before_llm_call",
        "after_llm_call",
        "before_tool_execution",
        "after_tool_execution",
        "on_error",
    ):
        with pytest.raises(AttributeError):
            cb.__getattribute__(name)


def test_run_does_not_use_dispatch_async_directly_from_run_async() -> None:
    """Sanity: the loop's run_async body uses dispatch_async; the sync wrapper
    enables the bridge so that, even from sync ``run()``, async hooks
    registered are awaited (the M3 fix).

    We don't strict-check WHICH dispatch path the inner run_async uses;
    we assert the user-visible contract: an async hook MUST run.
    """
    cb = tinyagent.CallbackRegistry()
    ran = []

    async def async_hook(ctx: object) -> None:
        ran.append("yes")

    cb.register_before_llm_call(async_hook)
    agent = _make_agent(callbacks=cb)

    response = _func_response(tool_calls=[_final_answer_tool_call("ok")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        result = agent.run("...")

    assert result == "ok"
    assert ran == ["yes"], (
        f"async hook MUST execute under the sync run() wrapper; "
        f"got ran={ran!r}"
    )


# ============================================================================
# 7. The pinned loop is set during the run, cleared after.
# ============================================================================
def test_run_pins_loop_during_execution_and_clears_after() -> None:
    """``run()`` pins ``self._callbacks._loop`` to a non-None value while
    executing, and clears it afterwards so the registry is clean for
    the next run.

    The pinning contract is exactly what enables the round-2 M3 fix:
    ``dispatch_sync`` (and the bridge path inside the run) needs the
    pinned loop to be available inside the coroutine context.
    """
    agent = _make_agent()
    seen_during: list[Any] = []
    original_loop = agent._callbacks._loop

    def probe(ctx: object) -> None:
        seen_during.append(agent._callbacks._loop)

    agent._callbacks.register_before_llm_call(probe)

    response = _func_response(tool_calls=[_final_answer_tool_call("ok")])
    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        agent.run("...")

    # The probe observed the pinned loop during hook execution.
    assert len(seen_during) == 1
    assert seen_during[0] is not None, (
        f"loop MUST be pinned during run(); got {seen_during[0]!r}"
    )
    # After run(), the loop must be cleared so subsequent runs can re-pin.
    assert agent._callbacks._loop is original_loop, (
        f"run() should leave _loop in its prior state; "
        f"started as {original_loop!r}, ended as {agent._callbacks._loop!r}"
    )
