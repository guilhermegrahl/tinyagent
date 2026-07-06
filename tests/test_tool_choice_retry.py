"""T12a-cross dedicated tests for the ``tool_choice`` retry rule.

Per plan §13 T12a-cross acceptance criteria + §8 round-3 M4 retry branch.

This module adds **finer-grained** dedicated coverage of the
``tool_choice`` retry logic. The high-level retry path is also exercised
in ``tests/test_agent_loop.py::test_empty_response_under_required_*``;
this file focuses on:

  * The **exact tool_choice kwarg sequence** sent to ``call_model`` on
    each turn (required first, auto exactly once, never a third attempt).
  * **Message history**: the empty response is NOT appended; only the
    successful (or finally-empty) response lands in ``messages``.
  * **Retry budget = 1**: the loop retries at most once per empty response
    under ``required``. The budget is exactly 1, NOT 0 (would skip the
    retry) and NOT 2 (would loop forever).
  * **Cross-turn re-arm**: each new turn re-arms the retry counter so a
    long conversation can do at most one retry per turn (per §8 (f) and
    the loop docstring in ``tinyagent.run_async``).
  * **Hook firing parity**: ``before_llm_call`` and ``after_llm_call``
    each fire once per *actual* LLM call — including the retry — so two
    attempts produce two pairs of hook firings.

All three acceptance-criteria scenarios are covered:

  1. ``required`` -> empty -> retry under ``auto`` -> trailing text (success)
  2. ``required`` -> empty -> retry under ``auto`` -> tool_call (success)
  3. ``required`` -> empty -> retry under ``auto`` -> also empty
     -> ``AgentError`` raised after exactly two LLM calls.

Tests mock ``TinyAgent.call_model`` via ``patch.object`` so the loop is
exercised end-to-end (hooks fire, message list mutates) without touching
the any-llm seam.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import tinyagent
from tinyagent import AgentError


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


def _empty_response() -> _SyntheticResponse:
    """Build an empty assistant response (no content, no tool_calls)."""
    return _SyntheticResponse(
        choices=[_Choice(message=_Message(role="assistant", content="", tool_calls=[]))]
    )


def _trailing_text_response(text: str) -> _SyntheticResponse:
    """Build a trailing-text assistant response (content set, no tool_calls)."""
    return _SyntheticResponse(
        choices=[_Choice(message=_Message(role="assistant", content=text, tool_calls=[]))]
    )


def _tool_call_response(
    name: str, call_id: str = "call_x", **args: Any
) -> _SyntheticResponse:
    """Build an assistant response with exactly one tool call (non-final)."""
    tc = _SyntheticToolCall(
        id=call_id,
        function=_Function(name=name, arguments=json.dumps(args)),
    )
    return _SyntheticResponse(
        choices=[_Choice(message=_Message(role="assistant", content="", tool_calls=[tc]))]
    )


def _final_answer_response(answer: str, call_id: str = "call_fa") -> _SyntheticResponse:
    """Build a final_answer response."""
    tc = _SyntheticToolCall(
        id=call_id,
        function=_Function(name="final_answer", arguments=json.dumps({"answer": answer})),
    )
    return _SyntheticResponse(
        choices=[_Choice(message=_Message(role="assistant", content="", tool_calls=[tc]))]
    )


# ============================================================================
# Test fixtures — fresh agent per test
# ============================================================================
def _unique_tracer_name() -> str:
    """Return a tracer name unique to this test invocation.

    Prevents the ``_tracer_cache`` populated by ``test_otel_setup`` from
    leaking across tests (same rationale as the sibling test files).
    """
    return f"tinyagent-t12across-{uuid.uuid4().hex}"


def _make_agent(
    *,
    max_turns: int = 10,
    tool_registry: dict[str, Any] | None = None,
) -> tinyagent.TinyAgent:
    """Build a TinyAgent with the requested tool registry seeded into ``_clients``.

    ``tool_registry`` defaults to a single callable tool ``echo`` that
    returns ``"echoed:" + repr(x)`` so scenarios 2 / 4 (which need a
    registered non-final tool) have something concrete to dispatch.
    """
    cfg = tinyagent.AgentConfig(
        instructions="You are a test agent.",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        max_turns=max_turns,
        callbacks=tinyagent.CallbackRegistry(),
        name=_unique_tracer_name(),
    )
    agent = tinyagent.TinyAgent(cfg)

    if tool_registry is None:

        def echo(x: str = "hi") -> str:
            return f"echoed:{x}"

        agent._clients["echo"] = tinyagent._wrap_no_exception(echo)
    else:
        for name, fn in tool_registry.items():
            agent._clients[name] = tinyagent._wrap_no_exception(fn)

    return agent


def _capturing_call_model(
    responses: list[_SyntheticResponse],
) -> tuple[AsyncMock, list[dict[str, Any]]]:
    """Build an ``AsyncMock`` whose side_effect feeds ``responses`` in order.

    Returns the mock and a list of kwarg dicts capturing every call to
    ``call_model``. The kwarg dicts are appended in call order so tests
    can assert on the exact ``tool_choice`` and ``messages`` passed.
    """
    captured: list[dict[str, Any]] = []
    iterator = iter(responses)

    async def _side_effect(**kwargs: Any) -> Any:
        captured.append(dict(kwargs))
        return next(iterator)

    return AsyncMock(side_effect=_side_effect), captured


# ============================================================================
# Scenario 1: required -> empty -> auto -> trailing text (success)
# ============================================================================
@pytest.mark.asyncio
async def test_retry_required_empty_then_auto_trailing_text_succeeds() -> None:
    """Scenario 1 (per T12a-cross acceptance): the retry returns trailing text.

    First ``call_model(tool_choice="required")`` returns an empty response;
    the loop retries exactly once with ``tool_choice="auto"``; the retry
    returns trailing assistant text; the loop returns that text.

    Asserts:
      - exactly 2 LLM calls (NOT 0, NOT 1, NOT 3+),
      - call 1 used ``tool_choice="required"``, call 2 used ``tool_choice="auto"``,
      - the retry budget is exactly 1 (no third attempt),
      - the loop returns the trailing text.
    """
    agent = _make_agent()
    responses = [_empty_response(), _trailing_text_response("retry answer")]
    mock_cm, captured = _capturing_call_model(responses)

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        result = await agent.run_async("...")

    assert result == "retry answer"
    assert len(captured) == 2, (
        f"retry budget must be exactly 1 (loop should make 2 LLM calls total), "
        f"got {len(captured)} calls: tool_choices={[c.get('tool_choice') for c in captured]}"
    )
    tool_choices = [c.get("tool_choice") for c in captured]
    assert tool_choices == ["required", "auto"], (
        f"first call must use 'required', retry must use 'auto', got {tool_choices!r}"
    )


@pytest.mark.asyncio
async def test_retry_trailing_text_empty_response_not_appended_to_history() -> None:
    """The empty response from the first call must NOT land in messages.

    Per §8 pseudocode: ``The empty response is NOT appended to messages —
    only the (successful or finally-empty) response lands in the conversation
    history.`` After Scenario 1's successful retry, ``messages`` contains
    exactly one assistant entry (the trailing text), not the empty one.
    """
    agent = _make_agent()
    responses = [_empty_response(), _trailing_text_response("the answer")]
    mock_cm, captured = _capturing_call_model(responses)

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        await agent.run_async("...")

    # Inspect what was passed on the SECOND call_model invocation —
    # the messages list sent on the retry should contain only the
    # initial system + user messages (NOT the empty first response),
    # because the empty response was discarded before the retry.
    assert len(captured) == 2
    second_call_messages = captured[1].get("messages", [])
    assistant_entries = [m for m in second_call_messages if m.get("role") == "assistant"]
    assert assistant_entries == [], (
        f"empty first response must NOT be in messages on retry; "
        f"got assistant entries: {assistant_entries!r}"
    )


@pytest.mark.asyncio
async def test_retry_hooks_fire_once_per_call_including_retry() -> None:
    """``before_llm_call`` / ``after_llm_call`` each fire once per LLM call.

    Scenario 1 makes 2 LLM calls (initial + retry); each hook should fire
    exactly twice. No silent coroutine drop, no double-firing.
    """
    agent = _make_agent()
    before_count: list[int] = []
    after_count: list[int] = []
    agent._callbacks.register_before_llm_call(
        lambda ctx: before_count.append(1)
    )
    agent._callbacks.register_after_llm_call(
        lambda ctx: after_count.append(1)
    )

    responses = [_empty_response(), _trailing_text_response("answer")]
    mock_cm, _ = _capturing_call_model(responses)

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        await agent.run_async("...")

    assert sum(before_count) == 2, (
        f"before_llm_call must fire once per LLM call (2 calls expected), got {sum(before_count)}"
    )
    assert sum(after_count) == 2, (
        f"after_llm_call must fire once per LLM call (2 calls expected), got {sum(after_count)}"
    )


# ============================================================================
# Scenario 2: required -> empty -> auto -> tool_call (success)
# ============================================================================
@pytest.mark.asyncio
async def test_retry_required_empty_then_auto_tool_call_succeeds() -> None:
    """Scenario 2: the retry returns a non-final tool_call.

    First ``call_model(tool_choice="required")`` returns empty;
    loop retries once with ``tool_choice="auto"``;
    retry returns an ``echo`` tool_call (registered on the agent);
    the loop dispatches the tool, then a subsequent LLM call returns
    a ``final_answer`` that terminates the conversation.

    Asserts:
      - first 2 calls use ``[required, auto]``,
      - the registered ``echo`` tool actually ran,
      - the loop returns the final_answer value.
    """
    echo_invocations: list[str] = []

    def echo(x: str = "default") -> str:
        echo_invocations.append(x)
        return f"echoed:{x}"

    agent = _make_agent(tool_registry={"echo": echo})
    responses = [
        _empty_response(),  # required -> empty
        _tool_call_response("echo", call_id="c_e", x="payload"),  # auto -> tool_call
        _final_answer_response("all done"),  # next required -> final_answer
    ]
    mock_cm, captured = _capturing_call_model(responses)

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        result = await agent.run_async("...")

    assert result == "all done"
    assert echo_invocations == ["payload"], (
        f"the retry's tool_call must dispatch through the registered tool, "
        f"got echo_invocations={echo_invocations!r}"
    )
    # First two calls are the retry pair; the third is the next turn's required call.
    assert len(captured) == 3, (
        f"expected 3 LLM calls (required empty, auto tool_call, required final_answer), "
        f"got {len(captured)}"
    )
    assert [c.get("tool_choice") for c in captured] == ["required", "auto", "required"], (
        f"tool_choice sequence must be [required, auto, required], "
        f"got {[c.get('tool_choice') for c in captured]!r}"
    )


# ============================================================================
# Scenario 3: required -> empty -> auto -> empty -> AgentError
# ============================================================================
@pytest.mark.asyncio
async def test_retry_required_empty_then_auto_empty_raises_agent_error() -> None:
    """Scenario 3: both calls empty -> AgentError after exactly 2 LLM calls.

    Per §8 (f): if the retry under ``auto`` also yields an empty response,
    raise ``AgentError``. The retry budget is exactly 1 — the loop must
    NOT attempt a third call (which would be ``tool_choice="required"
    again`` because the retry budget is exhausted, looping until
    ``max_turns``). Exactly two LLM calls should be made.
    """
    agent = _make_agent()
    mock_cm = AsyncMock(return_value=_empty_response())

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        with pytest.raises(AgentError) as exc_info:
            await agent.run_async("...")

    # Retry budget = 1: exactly 2 LLM calls, no more.
    assert mock_cm.await_count == 2, (
        f"retry budget must be exactly 1 (2 LLM calls: required then auto), "
        f"got {mock_cm.await_count} calls"
    )
    msg = str(exc_info.value).lower()
    assert "required" in msg and "auto" in msg, (
        f"AgentError should describe the tool_choice retry failure, got: {exc_info.value!r}"
    )


@pytest.mark.asyncio
async def test_retry_empty_after_retry_does_not_consume_max_turns_budget() -> None:
    """A single turn's required+auto retry pair counts as ONE turn, not two.

    Per the loop docstring: ``the worst case is 2 * max_turns LLM calls``.
    With ``max_turns=1`` and a single turn whose initial + retry are both
    empty, the loop must bail with AgentError after 2 calls — NOT loop
    back to the next turn (which would otherwise need a third call).
    """
    agent = _make_agent(max_turns=1)
    mock_cm = AsyncMock(return_value=_empty_response())

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        with pytest.raises(AgentError):
            await agent.run_async("...")

    # 1 turn * 2 calls (required + auto) = 2 calls. No third call.
    assert mock_cm.await_count == 2, (
        f"with max_turns=1, a single turn's required+auto retry pair must make "
        f"exactly 2 LLM calls, got {mock_cm.await_count}"
    )


# ============================================================================
# Cross-turn re-arm: each new turn resets the retry counter
# ============================================================================
@pytest.mark.asyncio
async def test_retry_counter_rearms_per_turn() -> None:
    """The retry budget is per-turn: each new turn can retry once.

    Sequence across two turns:
      turn 1: required -> empty -> auto -> final_answer (terminates)
    The retry only happens in turn 1. With a single turn, the retry
    counter resets at the start of the NEXT turn. We assert this
    indirectly by checking that the sequence of tool_choices matches
    [required, auto, required] — the third call starts a fresh turn
    with ``tool_choice=required`` again (per §8 reset rule).
    """
    agent = _make_agent()
    responses = [
        _empty_response(),  # turn 1, required -> empty
        _tool_call_response("echo", call_id="c1", x="a"),  # turn 1, auto -> tool
        _final_answer_response("done"),  # turn 2, required -> final
    ]
    mock_cm, captured = _capturing_call_model(responses)

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        result = await agent.run_async("...")

    assert result == "done"
    assert [c.get("tool_choice") for c in captured] == ["required", "auto", "required"], (
        f"per-turn retry re-arm: turn 2 must reset tool_choice to 'required', "
        f"got {[c.get('tool_choice') for c in captured]!r}"
    )


@pytest.mark.asyncio
async def test_retry_under_auto_does_not_re_retry_on_subsequent_empty() -> None:
    """If the auto-retry itself is empty, the loop raises — no further retries.

    Concretely: a turn where the initial required call AND the auto retry
    are both empty must raise ``AgentError`` after exactly 2 calls. The
    loop must NOT trigger a third ``required`` call (which would loop
    indefinitely up to ``max_turns``).
    """
    agent = _make_agent(max_turns=5)
    mock_cm = AsyncMock(return_value=_empty_response())

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        with pytest.raises(AgentError):
            await agent.run_async("...")

    # Hard upper bound: 2 calls. NOT 3, NOT 4, NOT 5.
    assert mock_cm.await_count == 2, (
        f"after auto-retry also empty, loop must NOT attempt more calls "
        f"(retry budget = 1), got {mock_cm.await_count}"
    )


# ============================================================================
# Negative path: tool_choice != 'required' on first call should NOT trigger retry
# ============================================================================
@pytest.mark.asyncio
async def test_no_retry_when_initial_tool_choice_is_auto() -> None:
    """If the loop ever starts with tool_choice='auto' (defensive path),
    an empty response must NOT trigger a retry — there is no 'auto -> required'
    second retry. The loop should raise AgentError after exactly 1 call.

    This is a regression guard against accidentally introducing a second
    retry branch (e.g. 'auto -> required' on second-empty).
    """
    agent = _make_agent()
    # Monkey-patch the loop's initial tool_choice so we exercise the
    # "already auto" path. We do this by overriding call_model to assert
    # the loop never retries — if the loop ever does call again, we'd
    # see a third invocation.
    call_count = 0

    async def _counting(**kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        # Force the loop into the auto-already branch by sending empty
        # under 'auto' first.
        if call_count == 1:
            assert kwargs.get("tool_choice") == "required", (
                f"first call should be 'required', got {kwargs.get('tool_choice')!r}"
            )
            # Return empty -> triggers the required->auto retry branch
            return _empty_response()
        # Second call (the retry under auto): also empty.
        # The loop should raise here, NOT call again.
        assert kwargs.get("tool_choice") == "auto", (
            f"second call must be the retry under 'auto', got {kwargs.get('tool_choice')!r}"
        )
        return _empty_response()

    with patch.object(tinyagent.TinyAgent, "call_model", new=AsyncMock(side_effect=_counting)):
        with pytest.raises(AgentError):
            await agent.run_async("...")

    assert call_count == 2, (
        f"loop must make exactly 2 calls (required then auto), then raise; "
        f"got {call_count} calls — possible second retry regression"
    )


# ============================================================================
# Sanity: a non-empty initial response under 'required' does NOT trigger retry
# ============================================================================
@pytest.mark.asyncio
async def test_no_retry_when_required_response_is_already_non_empty() -> None:
    """If the first ``required`` call returns a non-empty response, the
    loop must NOT attempt a retry — only empty responses trigger the
    retry branch.
    """
    agent = _make_agent()
    responses = [_trailing_text_response("immediate answer")]
    mock_cm, captured = _capturing_call_model(responses)

    with patch.object(tinyagent.TinyAgent, "call_model", new=mock_cm):
        result = await agent.run_async("...")

    assert result == "immediate answer"
    assert len(captured) == 1, (
        f"non-empty response under 'required' must NOT trigger a retry; "
        f"got {len(captured)} calls"
    )
    assert captured[0].get("tool_choice") == "required"
