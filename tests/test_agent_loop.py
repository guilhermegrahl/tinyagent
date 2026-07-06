"""T12a acceptance tests: ReAct loop body, per-turn tool dispatch, ``final_answer`` short-circuit,
tool_choice retry.

Per plan §2 sections 14 + 15, §13 T12a acceptance criteria, §8 pseudocode, §0 C1 (hook
symmetry), §0 C3 (``final_answer`` hook symmetry — both hooks fire), round-3 M1
(``break`` after first ``final_answer`` to short-circuit the rest of the turn), round-3
M4 (empty-response retry under ``tool_choice=required`` once with ``tool_choice=auto``).

The loop body lives on ``TinyAgent``. These tests drive the loop with synthetic
``any_llm`` responses (mocked at the ``TinyAgent.call_model`` seam) and assert:

- The loop iterates EVERY ``tool_calls`` entry in the turn.
- The FIRST ``final_answer`` short-circuits the rest of the turn via ``break``
  — subsequent non-``final_answer`` tool calls in the same turn do NOT execute.
- BOTH ``before_tool_execution`` AND ``after_tool_execution`` fire on the
  ``final_answer`` tool call (round-3 M4 closure — no carve-out).
- An unknown tool returns a descriptive error string to the LLM (loop continues,
  no ``on_error``).
- Max-turns exceeded raises ``AgentError``.
- Trailing assistant text (no tool_calls) is returned (uncommon but possible).
- Under ``tool_choice=required`` an empty response triggers a single retry with
  ``tool_choice=auto`` (round-3 M4); the retry result is the one returned.

The tests mock ``TinyAgent.call_model`` rather than ``any_llm.acompletion`` so
the loop is tested in isolation from the OTel span seam. ``call_model`` is
patched on the class via ``patch.object(TinyAgent, "call_model")``.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import tinyagent
from tinyagent import AgentCancel, AgentError


# ============================================================================
# Synthetic response builders — model the any_llm response shape the loop reads
# ============================================================================
@dataclass
class _ToolCall:
    """Synthetic tool-call payload (test-side only)."""

    id: str
    name: str
    arguments: str  # JSON-encoded string


@dataclass
class _Function:
    """``.function`` part of a tool call — what ``tool_call.function.*`` returns."""

    name: str
    arguments: str


@dataclass
class _SyntheticToolCall:
    """Mirror of ``ChatCompletionMessageToolCall`` — the loop iterates these."""

    id: str
    function: _Function


@dataclass
class _Message:
    """Mirror of ``Choice.message`` — what the loop extracts per turn."""

    role: str = "assistant"
    content: str | None = ""
    tool_calls: list[_SyntheticToolCall] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        """Return the dict form the loop appends to ``messages``."""
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
    """Stand-in for any_llm ``ChatCompletion`` — what ``call_model`` returns."""

    choices: list[_Choice] = field(default_factory=list)


def _func_response(
    tool_calls: list[_ToolCall] | None = None,
    content: str = "",
    role: str = "assistant",
) -> _SyntheticResponse:
    """Build a single-response object the loop can consume."""
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
    """Build a _ToolCall whose ``function.name == "final_answer"``."""
    return _ToolCall(
        id=call_id,
        name="final_answer",
        arguments=json.dumps({"answer": answer}),
    )


def _func_call(name: str, call_id: str = "call_x", **args: Any) -> _ToolCall:
    """Build a _ToolCall for an arbitrary tool by name."""
    return _ToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps(args),
    )


# ============================================================================
# Test fixtures — fresh agent per test, with hooks captured for assertions
# ============================================================================
def _unique_tracer_name() -> str:
    """Return a tracer name that is unique per test invocation.

    Same rationale as ``tests/test_request_timeout.py``: prevents the
    ``_tracer_cache`` populated by ``test_otel_setup`` from leaking
    across tests.
    """
    return f"tinyagent-t12a-{uuid.uuid4().hex}"


def _make_agent(
    *,
    max_turns: int = 10,
) -> tinyagent.TinyAgent:
    """Build a TinyAgent ready for loop tests."""
    cfg = tinyagent.AgentConfig(
        instructions="You are a test agent.",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        max_turns=max_turns,
        callbacks=tinyagent.CallbackRegistry(),
        name=_unique_tracer_name(),
    )
    return tinyagent.TinyAgent(cfg)


# ============================================================================
# 1. Loop terminates on final_answer (M4 hook symmetry, M1 break semantics)
# ============================================================================
@pytest.mark.asyncio
async def test_final_answer_returns_answer_string() -> None:
    """The loop terminates and returns ``str(answer)`` when the LLM emits final_answer."""
    agent = _make_agent()
    response = _func_response(tool_calls=[_final_answer_tool_call("hello")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        result = await agent.run_async("greet me")

    assert result == "hello"
    mock_cm.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_answer_fires_both_before_and_after_tool_hooks() -> None:
    """BOTH ``before_tool_execution`` AND ``after_tool_execution`` fire on final_answer (C3/M4).

    Round-3 closure: there is NO carve-out for the termination tool. Both
    hooks fire with the same tool_call shape; ``after_tool_execution`` sees
    ``ctx.tool_result`` set to the captured answer.
    """
    agent = _make_agent()
    before_calls: list[Any] = []
    after_calls: list[Any] = []
    agent._callbacks.register_before_tool_execution(
        lambda ctx: before_calls.append(ctx.tool_call)
    )
    agent._callbacks.register_after_tool_execution(
        lambda ctx: after_calls.append(ctx.tool_call)
    )
    response = _func_response(tool_calls=[_final_answer_tool_call("the-answer")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        result = await agent.run_async("answer please")

    assert result == "the-answer"
    assert len(before_calls) == 1, (
        f"before_tool_execution should fire exactly once for final_answer, "
        f"got {len(before_calls)}: {before_calls!r}"
    )
    assert len(after_calls) == 1, (
        f"after_tool_execution should fire exactly once for final_answer, "
        f"got {len(after_calls)}: {after_calls!r}"
    )


@pytest.mark.asyncio
async def test_final_answer_after_hook_sees_answer_in_tool_result() -> None:
    """``after_tool_execution`` for ``final_answer`` sees ctx.tool_result == answer string."""
    agent = _make_agent()
    observed_results: list[Any] = []
    agent._callbacks.register_after_tool_execution(
        lambda ctx: observed_results.append(ctx.tool_result)
    )
    response = _func_response(tool_calls=[_final_answer_tool_call("the-answer")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        await agent.run_async("...")

    assert observed_results == ["the-answer"], (
        f"after_tool_execution for final_answer should see tool_result=the-answer, "
        f"got {observed_results!r}"
    )


# ============================================================================
# 2. Iterate ALL tool_calls; first final_answer short-circuits via break (round-3 M1)
# ============================================================================
@pytest.mark.asyncio
async def test_loop_iterates_all_tool_calls_in_one_turn() -> None:
    """The loop visits EVERY tool call in the turn (plan §8 rule (a))."""
    agent = _make_agent()

    def tool_a(x: int = 1) -> str:
        return f"a:{x}"

    def tool_b(y: str = "y") -> str:
        return f"b:{y}"

    agent._clients["tool_a"] = tinyagent._wrap_no_exception(tool_a)
    agent._clients["tool_b"] = tinyagent._wrap_no_exception(tool_b)

    response = _func_response(
        tool_calls=[
            _func_call("tool_a", call_id="c_a", x=2),
            _func_call("tool_b", call_id="c_b", y="hi"),
        ]
    )
    final_response = _func_response(tool_calls=[_final_answer_tool_call("done")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.side_effect = [response, final_response]
        result = await agent.run_async("two tools")

    assert result == "done"
    assert mock_cm.await_count == 2


@pytest.mark.asyncio
async def test_first_final_answer_short_circuits_remaining_tool_calls() -> None:
    """first final_answer break: subsequent non-final_answer tool calls do NOT execute (M1)."""
    agent = _make_agent()

    tool_a_invocations: list[int] = []

    def tool_a(x: int = 1) -> str:
        tool_a_invocations.append(x)
        return f"a:{x}"

    agent._clients["tool_a"] = tinyagent._wrap_no_exception(tool_a)

    # Turn contains [final_answer, tool_a]; the loop must break after
    # final_answer and NOT run tool_a.
    response = _func_response(
        tool_calls=[
            _final_answer_tool_call("the-answer", call_id="c_fa"),
            _func_call("tool_a", call_id="c_a", x=99),
        ]
    )

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        result = await agent.run_async("...")

    assert result == "the-answer"
    assert tool_a_invocations == [], (
        f"tool_a must NOT execute after final_answer short-circuit, "
        f"got invocations={tool_a_invocations!r}"
    )


@pytest.mark.asyncio
async def test_final_answer_break_skips_second_final_answer_in_same_turn() -> None:
    """Two final_answer calls in one turn — first wins via break; second NOT reached."""
    agent = _make_agent()
    after_count = {"n": 0}
    agent._callbacks.register_after_tool_execution(
        lambda ctx: after_count.__setitem__("n", after_count["n"] + 1)
    )

    response = _func_response(
        tool_calls=[
            _final_answer_tool_call("the-answer", call_id="c_fa1"),
            _final_answer_tool_call("ignored", call_id="c_fa2"),
        ]
    )

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        result = await agent.run_async("...")

    assert result == "the-answer"
    assert after_count["n"] == 1, (
        f"after_tool_execution should fire once for the chosen final_answer, "
        f"got {after_count['n']}"
    )


# ============================================================================
# 3. Unknown tool returns error string to LLM (loop continues, NO on_error)
# ============================================================================
@pytest.mark.asyncio
async def test_unknown_tool_returns_error_string_to_llm() -> None:
    """Unknown tool name: error string fed back, loop continues, no ``on_error`` fires."""
    agent = _make_agent()
    on_error_calls: list[Any] = []
    agent._callbacks.register_on_error(lambda ctx: on_error_calls.append(ctx))

    unknown_response = _func_response(
        tool_calls=[_func_call("does_not_exist", call_id="c_x")]
    )
    final_response = _func_response(tool_calls=[_final_answer_tool_call("recovered")])

    second_call_messages: list[dict[str, Any]] = []

    async def _side_effect(**kwargs: Any) -> Any:
        if "messages" in kwargs:
            second_call_messages.append({"messages": list(kwargs["messages"])})
        return None  # placeholder; we override below

    mock_cm = AsyncMock(side_effect=[unknown_response, final_response])

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        result = await agent.run_async("do a thing")

    assert result == "recovered"
    assert mock_cm.await_count == 2
    assert on_error_calls == [], (
        f"on_error must NOT fire for an unknown tool name (recoverable), "
        f"got {on_error_calls!r}"
    )


# ============================================================================
# 4. Max-turns exceeded → AgentError
# ============================================================================
@pytest.mark.asyncio
async def test_max_turns_exceeded_raises_agent_error() -> None:
    """If the loop iterates more than ``max_turns`` times, ``AgentError`` is raised."""
    agent = _make_agent(max_turns=3)

    def never_terminates() -> str:
        return "still going"

    agent._clients["never_terminates"] = tinyagent._wrap_no_exception(never_terminates)

    non_terminating_response = _func_response(
        tool_calls=[_func_call("never_terminates", call_id="c_n")]
    )

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.side_effect = [non_terminating_response] * 100
        with pytest.raises(AgentError) as exc_info:
            await agent.run_async("...")

    msg = str(exc_info.value).lower()
    assert "max_turns" in msg or "turns" in msg, (
        f"AgentError should mention max_turns, got: {exc_info.value!r}"
    )


# ============================================================================
# 5. Trailing-text fallback
# ============================================================================
@pytest.mark.asyncio
async def test_trailing_assistant_text_returns_as_answer() -> None:
    """When the LLM emits assistant text with no tool_calls, the loop returns that text."""
    agent = _make_agent()
    response = _func_response(content="this is the answer", tool_calls=[])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        result = await agent.run_async("...")

    assert result == "this is the answer"
    mock_cm.assert_awaited_once()


# ============================================================================
# 6. tool_choice retry (round-3 M4) — empty response under required triggers auto
# ============================================================================
@pytest.mark.asyncio
async def test_empty_response_under_required_triggers_auto_retry() -> None:
    """Round-3 M4: empty response under tool_choice=required retries once with auto."""
    agent = _make_agent()

    empty_response = _func_response(content="", tool_calls=[])
    final_response = _func_response(tool_calls=[_final_answer_tool_call("ok")])

    seen_tool_choices: list[str] = []

    async def _spy_call_model(**kwargs: Any) -> Any:
        seen_tool_choices.append(kwargs.get("tool_choice", "<unset>"))
        if len(seen_tool_choices) == 1:
            return empty_response
        return final_response

    with patch.object(tinyagent.TinyAgent, "call_model", new=AsyncMock(side_effect=_spy_call_model)):
        result = await agent.run_async("...")

    assert result == "ok"
    assert seen_tool_choices[:2] == ["required", "auto"], (
        f"expected first call with required and second retry with auto, "
        f"got tool_choices={seen_tool_choices!r}"
    )


@pytest.mark.asyncio
async def test_empty_after_retry_raises_agent_error() -> None:
    """If retry under auto is ALSO empty, AgentError is raised."""
    agent = _make_agent()
    empty_response = _func_response(content="", tool_calls=[])

    mock_cm = AsyncMock(return_value=empty_response)

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        with pytest.raises(AgentError):
            await agent.run_async("...")

    assert mock_cm.await_count == 2, (
        f"loop should make exactly 2 LLM calls (required, then auto), got {mock_cm.await_count}"
    )


@pytest.mark.asyncio
async def test_non_empty_retry_with_trailing_text_returns_without_again() -> None:
    """Retry under auto that returns trailing text (no tool_calls) returns without further retry."""
    agent = _make_agent()
    empty_response = _func_response(content="", tool_calls=[])
    trailing_response = _func_response(content="the answer", tool_calls=[])

    mock_cm = AsyncMock(side_effect=[empty_response, trailing_response])

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        result = await agent.run_async("...")

    assert result == "the answer"
    assert mock_cm.await_count == 2


# ============================================================================
# 7. Hook firing pattern — before/after_tool_execution per non-final tool call
# ============================================================================
@pytest.mark.asyncio
async def test_non_final_tool_call_fires_before_and_after_hooks() -> None:
    """For a non-final tool call, BOTH before and after hooks fire."""
    agent = _make_agent()

    def my_tool(value: int = 1) -> str:
        return f"v={value}"

    agent._clients["my_tool"] = tinyagent._wrap_no_exception(my_tool)

    fired: list[str] = []
    agent._callbacks.register_before_tool_execution(lambda ctx: fired.append("before"))
    agent._callbacks.register_after_tool_execution(lambda ctx: fired.append("after"))

    response = _func_response(tool_calls=[_func_call("my_tool", call_id="c1", value=42)])
    final = _func_response(tool_calls=[_final_answer_tool_call("ok")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.side_effect = [response, final]
        result = await agent.run_async("...")

    assert result == "ok"
    assert "before" in fired, f"before_tool_execution should fire for non-final tool, got {fired!r}"
    assert "after" in fired, f"after_tool_execution should fire for non-final tool, got {fired!r}"


# ============================================================================
# 8. on_error integration + AgentCancel mid-loop (T12d)
# ----------------------------------------------------------------------------
# Per plan §0 C1 round-1 closure, §13 T12d, §5 (error propagation contract):
#   - ``on_error`` fires on EVERY escaping exception EXCEPT ``AgentCancel``.
#   - ``on_error`` is observational — it cannot swallow. The original exception
#     is re-raised after ``on_error`` runs.
#   - ``AgentCancel`` raised from any hook terminates the loop and propagates
#     to the caller. ``on_error`` does NOT fire (user explicitly aborted).
#   - Multiple ``on_error`` hooks run in registration order.
# ============================================================================
@pytest.mark.asyncio
async def test_on_error_fires_when_before_llm_call_raises_generic_exception() -> None:
    """A hook raising a generic Exception causes ``on_error`` to fire (with the exception in ctx).

    The original exception is re-raised after ``on_error`` runs (not swallowed).
    Plan §0 C1 round-1 fix + §13 T12d acceptance criterion.
    """
    agent = _make_agent()
    boom = ValueError("synthetic hook failure")
    agent._callbacks.register_before_llm_call(lambda ctx: (_ for _ in ()).throw(boom))
    seen_errors: list[BaseException] = []
    agent._callbacks.register_on_error(lambda ctx: seen_errors.append(ctx.error))

    response = _func_response(tool_calls=[_final_answer_tool_call("never reached")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(ValueError) as exc_info:
            await agent.run_async("trigger before_llm_call failure")

    assert exc_info.value is boom, (
        f"the original exception must propagate unchanged; "
        f"got {exc_info.value!r}, expected {boom!r}"
    )
    assert len(seen_errors) == 1, (
        f"on_error should fire exactly once, got {len(seen_errors)}: {seen_errors!r}"
    )
    assert seen_errors[0] is boom, (
        f"on_error ctx.error should be the original exception; "
        f"got {seen_errors[0]!r}"
    )


@pytest.mark.asyncio
async def test_on_error_fires_when_after_llm_call_raises_generic_exception() -> None:
    """``on_error`` fires when ``after_llm_call`` raises a generic Exception."""
    agent = _make_agent()
    boom = RuntimeError("after_llm_call failed")
    agent._callbacks.register_after_llm_call(lambda ctx: (_ for _ in ()).throw(boom))
    seen_errors: list[BaseException] = []
    agent._callbacks.register_on_error(lambda ctx: seen_errors.append(ctx.error))

    response = _func_response(tool_calls=[_final_answer_tool_call("never reached")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(RuntimeError) as exc_info:
            await agent.run_async("trigger after_llm_call failure")

    assert exc_info.value is boom
    assert len(seen_errors) == 1
    assert seen_errors[0] is boom


@pytest.mark.asyncio
async def test_on_error_fires_when_before_tool_execution_raises_generic_exception() -> None:
    """``on_error`` fires when ``before_tool_execution`` raises a generic Exception."""
    agent = _make_agent()

    def my_tool(x: int = 1) -> str:
        return f"v={x}"

    agent._clients["my_tool"] = tinyagent._wrap_no_exception(my_tool)

    boom = KeyError("before_tool_execution failed")
    agent._callbacks.register_before_tool_execution(
        lambda ctx: (_ for _ in ()).throw(boom)
    )
    seen_errors: list[BaseException] = []
    agent._callbacks.register_on_error(lambda ctx: seen_errors.append(ctx.error))

    response = _func_response(tool_calls=[_func_call("my_tool", call_id="c1", x=42)])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(KeyError) as exc_info:
            await agent.run_async("trigger before_tool_execution failure")

    assert exc_info.value is boom
    assert len(seen_errors) == 1
    assert seen_errors[0] is boom


@pytest.mark.asyncio
async def test_on_error_fires_when_after_tool_execution_raises_generic_exception() -> None:
    """``on_error`` fires when ``after_tool_execution`` raises a generic Exception."""
    agent = _make_agent()

    def my_tool(x: int = 1) -> str:
        return f"v={x}"

    agent._clients["my_tool"] = tinyagent._wrap_no_exception(my_tool)

    boom = TypeError("after_tool_execution failed")
    agent._callbacks.register_after_tool_execution(
        lambda ctx: (_ for _ in ()).throw(boom)
    )
    seen_errors: list[BaseException] = []
    agent._callbacks.register_on_error(lambda ctx: seen_errors.append(ctx.error))

    response = _func_response(tool_calls=[_func_call("my_tool", call_id="c1", x=42)])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(TypeError) as exc_info:
            await agent.run_async("trigger after_tool_execution failure")

    assert exc_info.value is boom
    assert len(seen_errors) == 1
    assert seen_errors[0] is boom


@pytest.mark.asyncio
async def test_on_error_does_not_swallow_exception_original_propagates() -> None:
    """The original exception re-raises after ``on_error`` runs — observability only.

    Plan §5 contract: "``on_error`` is observability-only." Even if ``on_error``
    itself completes successfully, the original exception MUST propagate.
    """
    agent = _make_agent()
    boom = ValueError("in-loop failure")
    agent._callbacks.register_before_llm_call(lambda ctx: (_ for _ in ()).throw(boom))
    on_error_ran = {"n": 0}
    agent._callbacks.register_on_error(lambda ctx: on_error_ran.__setitem__("n", 1))

    response = _func_response(tool_calls=[_final_answer_tool_call("never")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(ValueError) as exc_info:
            await agent.run_async("...")

    assert exc_info.value is boom
    assert on_error_ran["n"] == 1, (
        f"on_error must run before the exception propagates; ran count={on_error_ran['n']}"
    )


@pytest.mark.asyncio
async def test_agent_cancel_terminates_loop_from_before_llm_call() -> None:
    """``AgentCancel`` from ``before_llm_call`` terminates the loop and propagates."""
    agent = _make_agent()
    agent._callbacks.register_before_llm_call(
        lambda ctx: (_ for _ in ()).throw(AgentCancel("aborted"))
    )
    on_error_calls: list[Any] = []
    agent._callbacks.register_on_error(lambda ctx: on_error_calls.append(ctx))

    response = _func_response(tool_calls=[_final_answer_tool_call("never reached")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(AgentCancel):
            await agent.run_async("abort before llm")

    assert on_error_calls == [], (
        f"on_error MUST NOT fire for AgentCancel (user-explicit abort); "
        f"got {on_error_calls!r}"
    )


@pytest.mark.asyncio
async def test_agent_cancel_terminates_loop_from_after_llm_call() -> None:
    """``AgentCancel`` from ``after_llm_call`` terminates the loop and propagates."""
    agent = _make_agent()
    agent._callbacks.register_after_llm_call(
        lambda ctx: (_ for _ in ()).throw(AgentCancel("aborted"))
    )
    on_error_calls: list[Any] = []
    agent._callbacks.register_on_error(lambda ctx: on_error_calls.append(ctx))

    response = _func_response(tool_calls=[_final_answer_tool_call("never reached")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(AgentCancel):
            await agent.run_async("abort after llm")

    assert on_error_calls == []


@pytest.mark.asyncio
async def test_agent_cancel_terminates_loop_from_before_tool_execution() -> None:
    """``AgentCancel`` from ``before_tool_execution`` terminates the loop and propagates."""
    agent = _make_agent()

    def my_tool(x: int = 1) -> str:
        return f"v={x}"

    agent._clients["my_tool"] = tinyagent._wrap_no_exception(my_tool)

    agent._callbacks.register_before_tool_execution(
        lambda ctx: (_ for _ in ()).throw(AgentCancel("aborted"))
    )
    on_error_calls: list[Any] = []
    agent._callbacks.register_on_error(lambda ctx: on_error_calls.append(ctx))

    response = _func_response(tool_calls=[_func_call("my_tool", call_id="c1", x=42)])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(AgentCancel):
            await agent.run_async("abort before tool")

    assert on_error_calls == []


@pytest.mark.asyncio
async def test_agent_cancel_terminates_loop_from_after_tool_execution() -> None:
    """``AgentCancel`` from ``after_tool_execution`` terminates the loop and propagates."""
    agent = _make_agent()

    def my_tool(x: int = 1) -> str:
        return f"v={x}"

    agent._clients["my_tool"] = tinyagent._wrap_no_exception(my_tool)

    agent._callbacks.register_after_tool_execution(
        lambda ctx: (_ for _ in ()).throw(AgentCancel("aborted"))
    )
    on_error_calls: list[Any] = []
    agent._callbacks.register_on_error(lambda ctx: on_error_calls.append(ctx))

    response = _func_response(tool_calls=[_func_call("my_tool", call_id="c1", x=42)])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(AgentCancel):
            await agent.run_async("abort after tool")

    assert on_error_calls == []


@pytest.mark.asyncio
async def test_agent_cancel_from_before_tool_execution_during_final_answer() -> None:
    """``AgentCancel`` from ``before_tool_execution`` on ``final_answer`` terminates the loop."""
    agent = _make_agent()
    agent._callbacks.register_before_tool_execution(
        lambda ctx: (_ for _ in ()).throw(AgentCancel("aborted during final_answer"))
    )
    on_error_calls: list[Any] = []
    agent._callbacks.register_on_error(lambda ctx: on_error_calls.append(ctx))

    response = _func_response(tool_calls=[_final_answer_tool_call("the-answer")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(AgentCancel):
            await agent.run_async("abort during final_answer")

    assert on_error_calls == []


@pytest.mark.asyncio
async def test_multiple_on_error_hooks_run_in_registration_order() -> None:
    """Multiple ``on_error`` hooks run in registration order (append-list semantics).

    Mirrors the contract for all other hook lists (round-3 M3 dict-backed
    `self._hooks[name]` with append semantics). On-error dispatch iterates
    `self._hooks["on_error"]` in order.
    """
    agent = _make_agent()
    boom = ValueError("boom")
    agent._callbacks.register_before_llm_call(lambda ctx: (_ for _ in ()).throw(boom))
    order: list[str] = []
    agent._callbacks.register_on_error(lambda ctx: order.append("first"))
    agent._callbacks.register_on_error(lambda ctx: order.append("second"))
    agent._callbacks.register_on_error(lambda ctx: order.append("third"))

    response = _func_response(tool_calls=[_final_answer_tool_call("never")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(ValueError):
            await agent.run_async("trigger on_error chain")

    assert order == ["first", "second", "third"], (
        f"on_error hooks must run in registration order, got {order!r}"
    )


@pytest.mark.asyncio
async def test_async_on_error_hook_runs() -> None:
    """An async ``on_error`` hook is awaited (not silently dropped).

    Sanity check that the dispatch_async path used by the loop also
    handles coroutine-returning hooks (round-3 M3 closure contract).
    """
    agent = _make_agent()
    boom = ValueError("boom")
    agent._callbacks.register_before_llm_call(lambda ctx: (_ for _ in ()).throw(boom))

    completed = {"n": 0}

    async def async_on_error(ctx: Any) -> None:
        # Yield to prove the coroutine was awaited.
        import asyncio as _asyncio

        await _asyncio.sleep(0)
        completed["n"] += 1

    agent._callbacks.register_on_error(async_on_error)

    response = _func_response(tool_calls=[_final_answer_tool_call("never")])

    with patch.object(tinyagent.TinyAgent, "call_model", new_callable=AsyncMock) as mock_cm:
        mock_cm.return_value = response
        with pytest.raises(ValueError):
            await agent.run_async("...")

    assert completed["n"] == 1, (
        f"async on_error hook must be awaited; ran count={completed['n']}"
    )
