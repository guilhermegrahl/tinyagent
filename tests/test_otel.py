"""T9 acceptance tests: OpenTelemetry span generation around LLM and tool calls.

Per plan §13 T9, §2 section 13 (span generation), §0 C2 + cross-cutting
risk #8 (cost attribute omit rule — no ``(0.0, 0.0)`` fallback).

This module installs a TracerProvider wired to an InMemorySpanExporter at
module scope so ``_setup_tracing()`` returns a tracer that records into
the exporter. Per-test isolation comes from clearing the exporter and
the module tracer cache between tests.

Note on the library pattern (plan §6 / T8): ``_setup_tracing`` does NOT
call ``opentelemetry.trace.set_tracer_provider``. These tests therefore
install their own provider; the cache is cleared so the call picks up
the freshly-installed provider.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import tinyagent


# Attribute names — standard semconv + custom cost attr (§0 C2).
INPUT_TOK_ATTR = "gen_ai.usage.input_tokens"
OUTPUT_TOK_ATTR = "gen_ai.usage.output_tokens"
COST_ATTR = "gen_ai.usage.cost"
TOOL_NAME_ATTR = "gen_ai.tool.name"
TOOL_ARGS_ATTR = "gen_ai.tool.args"
TOOL_RESULT_ATTR = "gen_ai.tool.result"
OP_NAME_ATTR = "gen_ai.operation.name"
MODEL_ATTR = "gen_ai.request.model"


# ---------------------------------------------------------------------
# Stubs for any-llm's response shape (tests don't depend on any-llm).
# ---------------------------------------------------------------------
@dataclass
class _FakeUsage:
    """Minimal stand-in for an any-llm ``CompletionUsage``."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _FakeResponse:
    """Minimal stand-in for an any-llm ``ChatCompletion``."""

    usage: _FakeUsage


# ---------------------------------------------------------------------
# OTel wiring
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def otel_exporter() -> InMemorySpanExporter:
    """Install a TracerProvider with an InMemorySpanExporter (module scope).

    ``_setup_tracing`` is library-pattern (plan §6 / T8): it does NOT call
    ``trace.set_tracer_provider``. These tests install their own provider
    so that ``_setup_tracing`` returns a tracer that records into the
    exporter.

    The provider is a process-wide singleton and OTel refuses to
    replace it once set, so this fixture runs at module scope. Per-test
    isolation is provided by ``_reset_state`` clearing the exporter's
    accumulated spans and the module tracer cache before each test.
    """
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    trace.set_tracer_provider(provider)
    return exp


@pytest.fixture(autouse=True)
def _reset_state(otel_exporter: InMemorySpanExporter) -> Any:
    """Clear spans + tracer cache before each test for isolation."""
    otel_exporter.clear()
    tinyagent._setup_tracing_cache_clear()
    yield
    otel_exporter.clear()


@pytest.fixture
def tracer() -> Any:
    """TinyAgent tracer wired to the InMemorySpanExporter.

    Uses a per-session unique tracer name so we don't pollute the
    ``"tinyagent"`` cache (which ``test_otel_setup.py::test_setup_tracing_
    default_name_is_tinyagent`` asserts is hit once and only once).
    """
    name = f"tinyagent-test-{uuid.uuid4().hex}"
    return tinyagent._setup_tracing(name)


@pytest.fixture
def span_gen(tracer: Any) -> tinyagent._SpanGeneration:
    """``_SpanGeneration`` instance bound to a known model."""
    return tinyagent._SpanGeneration(tracer, model_id="openai:gpt-4o-mini")


# ---------------------------------------------------------------------
# call_llm span — token attributes
# ---------------------------------------------------------------------
def test_call_llm_span_has_input_and_output_tokens(
    span_gen: Any, otel_exporter: InMemorySpanExporter,
) -> None:
    """``call_llm`` span carries ``input_tokens`` and ``output_tokens`` from response.usage."""
    response = _FakeResponse(usage=_FakeUsage(prompt_tokens=123, completion_tokens=456))
    with span_gen.call_llm(response):
        pass

    spans = [s for s in otel_exporter.get_finished_spans() if s.name == "call_llm"]
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[INPUT_TOK_ATTR] == 123
    assert attrs[OUTPUT_TOK_ATTR] == 456
    assert attrs[OP_NAME_ATTR] == "chat"
    assert attrs[MODEL_ATTR] == "openai:gpt-4o-mini"


def test_call_llm_span_zero_tokens_when_usage_missing(
    span_gen: Any, otel_exporter: InMemorySpanExporter,
) -> None:
    """When response has no ``usage``, the token attrs are absent (not zero-defaulted)."""
    with span_gen.call_llm(object()):  # object() has no .usage
        pass

    spans = [s for s in otel_exporter.get_finished_spans() if s.name == "call_llm"]
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert INPUT_TOK_ATTR not in attrs
    assert OUTPUT_TOK_ATTR not in attrs


# ---------------------------------------------------------------------
# call_llm span — cost attribute presence / absence (round-1/2 fix)
# ---------------------------------------------------------------------
def test_call_llm_span_has_cost_when_estimate_cost_returns_number(
    span_gen: Any, otel_exporter: InMemorySpanExporter,
) -> None:
    """``gen_ai.usage.cost`` is written when ``_estimate_cost`` returns a non-None float."""
    response = _FakeResponse(usage=_FakeUsage(prompt_tokens=1000, completion_tokens=500))
    sentinel_cost = 0.000987
    with patch.object(tinyagent, "_estimate_cost", return_value=sentinel_cost):
        with span_gen.call_llm(response):
            pass

    spans = [s for s in otel_exporter.get_finished_spans() if s.name == "call_llm"]
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[COST_ATTR] == sentinel_cost


def test_call_llm_span_omits_cost_when_estimate_cost_returns_none(
    span_gen: Any, otel_exporter: InMemorySpanExporter,
) -> None:
    """``gen_ai.usage.cost`` is ABSENT (NOT ``0.0``) when ``_estimate_cost`` returns None.

    Per plan §0 C2 + cross-cutting risk #8: absence means "unknown",
    never "$0". A ``0.0`` placeholder would make "unknown price"
    indistinguishable from "actually free" in downstream dashboards.
    """
    response = _FakeResponse(usage=_FakeUsage(prompt_tokens=1000, completion_tokens=500))
    with patch.object(tinyagent, "_estimate_cost", return_value=None):
        with span_gen.call_llm(response):
            pass

    spans = [s for s in otel_exporter.get_finished_spans() if s.name == "call_llm"]
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert COST_ATTR not in attrs, (
        f"gen_ai.usage.cost must be OMITTED (not 0.0) when _estimate_cost returns "
        f"None; got attributes={attrs}"
    )


def test_call_llm_span_writes_zero_cost_when_estimate_returns_zero(
    span_gen: Any, otel_exporter: InMemorySpanExporter,
) -> None:
    """``gen_ai.usage.cost`` IS written when ``_estimate_cost`` returns exactly ``0.0``.

    This documents the boundary: zero is a valid *known* cost (free
    models, included-credit models). Only ``None`` triggers omission.
    """
    response = _FakeResponse(usage=_FakeUsage(prompt_tokens=100, completion_tokens=50))
    with patch.object(tinyagent, "_estimate_cost", return_value=0.0):
        with span_gen.call_llm(response):
            pass

    spans = [s for s in otel_exporter.get_finished_spans() if s.name == "call_llm"]
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[COST_ATTR] == 0.0


# ---------------------------------------------------------------------
# _compute_cost_attribute helper
# ---------------------------------------------------------------------
def test_compute_cost_attribute_passes_through_estimate_cost() -> None:
    """``_compute_cost_attribute`` delegates to ``_estimate_cost`` and returns its value verbatim.

    T13 plumbing: ``_compute_cost_attribute`` accepts a ``pricing`` kwarg
    so the per-instance override can flow through. When no override is
    supplied, the kwarg defaults to ``None``.
    """
    sentinel = 0.001234
    with patch.object(
        tinyagent, "_estimate_cost", return_value=sentinel
    ) as mock_estimate:
        result = tinyagent._compute_cost_attribute(
            "openai:gpt-4o-mini", 1000, 500
        )
    assert result is sentinel
    mock_estimate.assert_called_once_with(
        "openai:gpt-4o-mini", 1000, 500, pricing=None
    )


def test_compute_cost_attribute_returns_none_when_estimate_returns_none() -> None:
    """``_compute_cost_attribute`` returns None when ``_estimate_cost`` returns None."""
    with patch.object(tinyagent, "_estimate_cost", return_value=None):
        result = tinyagent._compute_cost_attribute("ollama:llama3", 100, 50)
    assert result is None


# ---------------------------------------------------------------------
# execute_tool span — name / args / result attributes
# ---------------------------------------------------------------------
def test_execute_tool_span_has_name_args_and_result_attrs(
    span_gen: Any, otel_exporter: InMemorySpanExporter,
) -> None:
    """``execute_tool`` span carries tool name, args, and result attributes."""
    with span_gen.execute_tool(
        tool_name="calculate",
        args={"expression": "2 + 2"},
        result="4",
    ):
        pass

    spans = [s for s in otel_exporter.get_finished_spans() if s.name == "execute_tool"]
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[TOOL_NAME_ATTR] == "calculate"
    assert attrs[OP_NAME_ATTR] == "execute_tool"
    assert "2 + 2" in str(attrs[TOOL_ARGS_ATTR])
    assert attrs[TOOL_RESULT_ATTR] == "4"


def test_execute_tool_span_truncates_huge_args(
    span_gen: Any, otel_exporter: InMemorySpanExporter,
) -> None:
    """Args larger than SPAN_LIMITS['tool_args'] are truncated, not silently dropped."""
    huge = "a" * 20_000  # SPAN_LIMITS['tool_args'] is 4096 by default
    with span_gen.execute_tool(tool_name="echo", args=huge, result="ok"):
        pass

    spans = [s for s in otel_exporter.get_finished_spans() if s.name == "execute_tool"]
    assert len(spans) == 1
    args_attr = dict(spans[0].attributes or {})[TOOL_ARGS_ATTR]
    assert isinstance(args_attr, str)
    assert len(args_attr) <= tinyagent.SPAN_LIMITS["tool_args"]
    assert "[truncated]" in args_attr


def test_execute_tool_span_truncates_huge_result(
    span_gen: Any, otel_exporter: InMemorySpanExporter,
) -> None:
    """Results larger than SPAN_LIMITS['tool_result'] are truncated."""
    huge = "b" * 20_000
    with span_gen.execute_tool(tool_name="echo", args="x", result=huge):
        pass

    spans = [s for s in otel_exporter.get_finished_spans() if s.name == "execute_tool"]
    assert len(spans) == 1
    result_attr = dict(spans[0].attributes or {})[TOOL_RESULT_ATTR]
    assert isinstance(result_attr, str)
    assert len(result_attr) <= tinyagent.SPAN_LIMITS["tool_result"]
    assert "[truncated]" in result_attr


def test_execute_tool_span_handles_non_string_args(
    span_gen: Any, otel_exporter: InMemorySpanExporter,
) -> None:
    """Non-string args (dicts, lists) are stringified into the span attr."""
    with span_gen.execute_tool(
        tool_name="http_get",
        args={"url": "https://example.com", "timeout": 10.0},
        result={"status": 200, "body": "ok"},
    ):
        pass

    spans = [s for s in otel_exporter.get_finished_spans() if s.name == "execute_tool"]
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    args_str = str(attrs[TOOL_ARGS_ATTR])
    assert "https://example.com" in args_str
    result_str = str(attrs[TOOL_RESULT_ATTR])
    assert '"status": 200' in result_str
    assert '"body": "ok"' in result_str


# ---------------------------------------------------------------------
# Span hierarchy: invoke_agent > call_llm / execute_tool
# ---------------------------------------------------------------------
def test_invoke_agent_span_is_parent_of_call_llm_and_execute_tool(
    span_gen: Any, otel_exporter: InMemorySpanExporter, tracer: Any,
) -> None:
    """A parent ``invoke_agent`` span contains the ``call_llm`` / ``execute_tool`` spans.

    Validated via the OTel parent linkage — the children's
    ``parent.span_id`` must equal the parent's ``context.span_id``.
    """
    with tracer.start_as_current_span("invoke_agent") as agent_span:
        response = _FakeResponse(usage=_FakeUsage(prompt_tokens=10, completion_tokens=20))
        with span_gen.call_llm(response):
            pass
        with span_gen.execute_tool(
            tool_name="calculate", args={"x": 1}, result="ok"
        ):
            pass

    spans = list(otel_exporter.get_finished_spans())
    invoke_agent = _find_only(spans, "invoke_agent")
    call_llm = _find_only(spans, "call_llm")
    execute_tool = _find_only(spans, "execute_tool")

    # invoke_agent is the root of this trace.
    assert invoke_agent.parent is None, "invoke_agent must be the root span"

    # Both children point at the invoke_agent span context.
    assert call_llm.parent is not None
    assert call_llm.parent.span_id == invoke_agent.context.span_id
    assert execute_tool.parent is not None
    assert execute_tool.parent.span_id == invoke_agent.context.span_id


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _find_only(spans: list[Any], name: str) -> Any:
    """Return the single span with the given name, asserting exactly one match."""
    matches = [s for s in spans if s.name == name]
    assert len(matches) == 1, (
        f"expected exactly one span named {name!r}, got {len(matches)}"
    )
    return matches[0]