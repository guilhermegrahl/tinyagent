"""T7 acceptance tests: AgentTrace / AgentSpan / TokenInfo / CostInfo + ToolCall TypedDict.

Per plan §2 sections 8 and 12, and §13 T7 acceptance criteria.

Covers:
- `ToolCall` is importable as a TypedDict (round-3 minor m6).
- `AgentSpan` can be constructed with the expected fields.
- `TokenInfo` and `CostInfo` carry the documented roll-up fields.
- `AgentTrace.tokens` sums input/output tokens across `call_llm` spans
  that HAVE token attributes (spans without tokens are skipped).
- `AgentTrace.cost` sums USD cost across `call_llm` spans that have the
  `gen_ai.usage.cost` attribute (round-3 M6: spans without are skipped —
  absence means "unknown", never "$0").
- `AgentTrace.cost` is zero when no spans have cost attrs.
"""
from __future__ import annotations

import pytest
from typing_extensions import is_typeddict

from tinyagent import (
    AgentSpan,
    AgentTrace,
    CostInfo,
    TokenInfo,
    ToolCall,
)


# Sentinel key for the cost attribute (matches upstream semconv constant).
COST_ATTR = "gen_ai.usage.cost"
INPUT_TOK_ATTR = "gen_ai.usage.input_tokens"
OUTPUT_TOK_ATTR = "gen_ai.usage.output_tokens"


# ---------------------------------------------------------------------
# ToolCall TypedDict (round-3 minor m6)
# ---------------------------------------------------------------------
def test_tool_call_is_typed_dict() -> None:
    """`ToolCall` is importable as a TypedDict from tinyagent."""
    assert is_typeddict(ToolCall), (
        "tinyagent.ToolCall must be a TypedDict (plan §2 section 8, round-3 m6)"
    )


def test_tool_call_constructable() -> None:
    """A ToolCall dict can be constructed with the documented keys."""
    tc: ToolCall = {
        "id": "call_abc123",
        "type": "function",
        "function": {"name": "final_answer", "arguments": '{"answer": "42"}'},
    }
    assert tc["id"] == "call_abc123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "final_answer"
    assert tc["function"]["arguments"] == '{"answer": "42"}'


# ---------------------------------------------------------------------
# AgentSpan dataclass
# ---------------------------------------------------------------------
def test_agent_span_constructable_with_required_fields() -> None:
    """`AgentSpan` accepts name/attributes/kind/parent_id/start_time."""
    span = AgentSpan(
        name="call_llm",
        attributes={
            INPUT_TOK_ATTR: 10,
            OUTPUT_TOK_ATTR: 20,
            COST_ATTR: 0.0001,
        },
        kind="internal",
        parent_id=None,
        start_time=0.0,
    )
    assert span.name == "call_llm"
    assert span.attributes[INPUT_TOK_ATTR] == 10
    assert span.attributes[OUTPUT_TOK_ATTR] == 20
    assert span.attributes[COST_ATTR] == 0.0001
    assert span.kind == "internal"
    assert span.parent_id is None
    assert span.start_time == 0.0


def test_agent_span_end_time_defaults_to_none() -> None:
    """`AgentSpan.end_time` is optional and defaults to None while span is open."""
    span = AgentSpan(
        name="call_llm",
        attributes={},
        kind="internal",
        parent_id="parent_1",
        start_time=1.0,
    )
    assert span.end_time is None


# ---------------------------------------------------------------------
# TokenInfo + CostInfo dataclasses
# ---------------------------------------------------------------------
def test_token_info_carries_input_and_output() -> None:
    """`TokenInfo` exposes `input_tokens` and `output_tokens` as ints."""
    info = TokenInfo(input_tokens=100, output_tokens=200)
    assert info.input_tokens == 100
    assert info.output_tokens == 200


def test_cost_info_carries_three_usd_fields() -> None:
    """`CostInfo` exposes input/output/total USD cost as floats."""
    info = CostInfo(
        input_cost_usd=0.001,
        output_cost_usd=0.002,
        total_cost_usd=0.003,
    )
    assert info.input_cost_usd == 0.001
    assert info.output_cost_usd == 0.002
    assert info.total_cost_usd == 0.003


# ---------------------------------------------------------------------
# AgentTrace tokens roll-up
# ---------------------------------------------------------------------
def _make_call_llm_span(
    *,
    name: str = "call_llm",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost: float | None = None,
) -> AgentSpan:
    """Helper: build a `call_llm` span with optional token / cost attrs."""
    attrs: dict[str, object] = {"gen_ai.operation.name": "chat"}
    if input_tokens is not None:
        attrs[INPUT_TOK_ATTR] = input_tokens
    if output_tokens is not None:
        attrs[OUTPUT_TOK_ATTR] = output_tokens
    if cost is not None:
        attrs[COST_ATTR] = cost
    return AgentSpan(
        name=name,
        attributes=attrs,
        kind="internal",
        parent_id=None,
        start_time=0.0,
    )


def test_agent_trace_tokens_sums_call_llm_spans() -> None:
    """`AgentTrace.tokens` sums tokens across `call_llm` spans that have attrs."""
    trace = AgentTrace(
        spans=[
            _make_call_llm_span(input_tokens=10, output_tokens=20),
            _make_call_llm_span(input_tokens=5, output_tokens=7),
        ]
    )
    tokens = trace.tokens
    assert tokens.input_tokens == 15
    assert tokens.output_tokens == 27


def test_agent_trace_tokens_skips_call_llm_spans_without_attrs() -> None:
    """`AgentTrace.tokens` skips call_llm spans that lack token attrs."""
    trace = AgentTrace(
        spans=[
            _make_call_llm_span(input_tokens=10, output_tokens=20),
            _make_call_llm_span(),  # no token attrs
            _make_call_llm_span(input_tokens=3, output_tokens=4),
        ]
    )
    tokens = trace.tokens
    assert tokens.input_tokens == 13
    assert tokens.output_tokens == 24


def test_agent_trace_tokens_ignores_non_call_llm_spans() -> None:
    """`AgentTrace.tokens` only sums `call_llm` spans — other names are ignored."""
    trace = AgentTrace(
        spans=[
            _make_call_llm_span(input_tokens=10, output_tokens=20),
            AgentSpan(
                name="execute_tool",
                attributes={INPUT_TOK_ATTR: 999, OUTPUT_TOK_ATTR: 999},
                kind="internal",
                parent_id=None,
                start_time=0.0,
            ),
        ]
    )
    tokens = trace.tokens
    assert tokens.input_tokens == 10
    assert tokens.output_tokens == 20


def test_agent_trace_tokens_zero_when_no_call_llm_spans() -> None:
    """`AgentTrace.tokens` is zero when there are no qualifying spans."""
    trace = AgentTrace(spans=[])
    tokens = trace.tokens
    assert tokens.input_tokens == 0
    assert tokens.output_tokens == 0


# ---------------------------------------------------------------------
# AgentTrace cost roll-up (round-3 M6 fix: skip spans without cost attr)
# ---------------------------------------------------------------------
def test_agent_trace_cost_sums_call_llm_spans_with_cost_attr() -> None:
    """`AgentTrace.cost` sums USD cost across spans that HAVE the cost attr."""
    trace = AgentTrace(
        spans=[
            _make_call_llm_span(cost=0.001),
            _make_call_llm_span(cost=0.002),
        ]
    )
    cost = trace.cost
    assert cost.input_cost_usd == pytest.approx(0.003)
    assert cost.output_cost_usd == 0.0
    assert cost.total_cost_usd == pytest.approx(0.003)


def test_agent_trace_cost_skips_call_llm_spans_without_cost_attr() -> None:
    """`AgentTrace.cost` skips call_llm spans that lack the cost attribute.

    This is the round-3 M6 fix: absence means "unknown", never "$0".
    """
    trace = AgentTrace(
        spans=[
            _make_call_llm_span(cost=0.005),     # known cost
            _make_call_llm_span(),               # unknown cost — must be skipped
            _make_call_llm_span(cost=0.010),     # known cost
        ]
    )
    cost = trace.cost
    assert cost.total_cost_usd == pytest.approx(0.015)
    assert cost.input_cost_usd == pytest.approx(0.015)
    assert cost.output_cost_usd == 0.0


def test_agent_trace_cost_zero_when_no_spans_have_cost_attrs() -> None:
    """`AgentTrace.cost` is zero when NO spans carry the cost attribute."""
    trace = AgentTrace(
        spans=[
            _make_call_llm_span(input_tokens=10),  # tokens only, no cost
            _make_call_llm_span(),
        ]
    )
    cost = trace.cost
    assert cost.input_cost_usd == 0.0
    assert cost.output_cost_usd == 0.0
    assert cost.total_cost_usd == 0.0


def test_agent_trace_cost_ignores_non_call_llm_spans() -> None:
    """`AgentTrace.cost` only sums `call_llm` spans."""
    trace = AgentTrace(
        spans=[
            _make_call_llm_span(cost=0.001),
            AgentSpan(
                name="execute_tool",
                attributes={COST_ATTR: 999.0},
                kind="internal",
                parent_id=None,
                start_time=0.0,
            ),
        ]
    )
    cost = trace.cost
    assert cost.total_cost_usd == pytest.approx(0.001)
