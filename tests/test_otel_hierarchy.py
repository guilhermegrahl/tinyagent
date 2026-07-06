"""End-to-end OTel hierarchy test: invoke_agent > call_llm/execute_tool.

Per plan §6 (OTel hierarchy) + plan §2 sections 13 + 15: when ``run_async``
runs, the loop must open a parent ``invoke_agent`` span and emit
``call_llm`` / ``execute_tool`` spans as its children. The README's
"Tracing" section and the ``examples/tracing_otlp.py`` example both
document this hierarchy.

Per the post-PR review (round-3 pr-code-reviewer BLOCKER): the
``invoke_agent`` span was never wired — only ``call_llm`` and
``execute_tool`` (via ``_SpanGeneration``) emitted, and even those were
orphans because no parent span was opened. This test exercises the
full loop end-to-end and asserts the hierarchy.

The test also asserts ``agent.trace`` retrieval path: after ``run_async``,
``agent.trace`` is a populated ``AgentTrace`` with one ``AgentSpan`` per
OTel span emitted during the run.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import tinyagent


# ---------------------------------------------------------------------
# Synthetic response builders (mirror tests/test_agent_loop.py shape)
# ---------------------------------------------------------------------
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
class _SyntheticUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20


@dataclass
class _SyntheticResponse:
    """Stand-in for any_llm ChatCompletion — call_model returns this."""

    choices: list[_Choice] = field(default_factory=list)
    usage: _SyntheticUsage = field(default_factory=_SyntheticUsage)


def _func_response(
    tool_calls: list[_SyntheticToolCall] | None = None,
    content: str = "",
) -> _SyntheticResponse:
    msg = _Message(
        role="assistant",
        content=content,
        tool_calls=list(tool_calls or []),
    )
    return _SyntheticResponse(choices=[_Choice(message=msg)])


def _final_answer_tc(answer: str, call_id: str = "call_fa") -> _SyntheticToolCall:
    return _SyntheticToolCall(
        id=call_id,
        function=_Function("final_answer", json.dumps({"answer": answer})),
    )


def _func_call_tc(name: str, call_id: str, **args: Any) -> _SyntheticToolCall:
    return _SyntheticToolCall(
        id=call_id,
        function=_Function(name, json.dumps(args)),
    )


# ---------------------------------------------------------------------
# OTel wiring — attach a span processor to the existing provider
# ---------------------------------------------------------------------
# OTel's TracerProvider is a process-wide singleton. If
# ``tests/test_otel.py`` has already run, the provider is set with its
# own ``InMemorySpanExporter``. Calling ``set_tracer_provider`` again
# is a no-op (with a warning) — so we can't replace the provider.
# Instead, we ATTACH our own ``SimpleSpanProcessor`` to the existing
# provider via ``provider.add_span_processor(...)`` — OTel fan-outs
# spans to every registered processor.
#
# If no provider has been installed yet (this file runs in isolation
# before ``test_otel.py``), we install a fresh provider with our own
# exporter so the test is robust standalone.
@pytest.fixture(scope="module")
def hierarchy_exporter() -> InMemorySpanExporter:
    """InMemorySpanExporter that captures invoke_agent hierarchy spans."""
    exp = InMemorySpanExporter()
    existing_provider = trace.get_tracer_provider()
    # ProxyTracerProvider has no ``add_span_processor`` — only the real
    # TracerProvider does. Detect and fall back to installation.
    add_processor = getattr(existing_provider, "add_span_processor", None)
    if callable(add_processor):
        existing_provider.add_span_processor(SimpleSpanProcessor(exp))
    else:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exp))
        trace.set_tracer_provider(provider)
    return exp


@pytest.fixture(autouse=True)
def _reset_hierarchy_state(hierarchy_exporter: InMemorySpanExporter) -> Any:
    """Clear spans + tracer cache before each test in this module.

    Only the spans landed in OUR exporter are cleared here. Spans landed
    in ``test_otel.py``'s exporter (if attached) are unaffected — that
    module has its own autouse reset.
    """
    hierarchy_exporter.clear()
    tinyagent._setup_tracing_cache_clear()
    yield
    hierarchy_exporter.clear()


# ---------------------------------------------------------------------
# 1. invoke_agent parent + call_llm / execute_tool children
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_async_emits_invoke_agent_parent_with_call_llm_and_execute_tool_children(
    hierarchy_exporter: InMemorySpanExporter,
) -> None:
    """run_async opens a single invoke_agent span with call_llm + execute_tool children.

    Synthetic sequence:
      - Turn 1 LLM emits [my_tool(a), my_tool(b)] → 1 call_llm, 2 execute_tool
      - Turn 2 LLM emits [final_answer("done")]   → 1 call_llm

    Asserts:
      - Exactly 1 invoke_agent span
      - At least 2 call_llm spans (one per turn)
      - At least 2 execute_tool spans (one per my_tool call)
      - All call_llm / execute_tool spans have invoke_agent as their
        OTel parent (parent.span_id matches invoke_agent.context.span_id).
    """
    cfg = tinyagent.AgentConfig(
        instructions="test agent",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        max_turns=5,
        callbacks=tinyagent.CallbackRegistry(),
        name=f"tinyagent-otel-hier-{uuid.uuid4().hex}",
    )
    agent = tinyagent.TinyAgent(cfg)

    def my_tool(value: int = 1) -> str:
        return f"v={value}"

    agent._clients["my_tool"] = tinyagent._wrap_no_exception(my_tool)

    response_1 = _func_response(
        tool_calls=[
            _func_call_tc("my_tool", call_id="c1", value=1),
            _func_call_tc("my_tool", call_id="c2", value=2),
        ]
    )
    response_2 = _func_response(tool_calls=[_final_answer_tc("done")])

    with patch.object(
        tinyagent.any_llm, "acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.side_effect = [response_1, response_2]
        result = await agent.run_async("do two tools")

    assert result == "done"

    spans = hierarchy_exporter.get_finished_spans()
    invoke_agent_spans = [s for s in spans if s.name == "invoke_agent"]
    call_llm_spans = [s for s in spans if s.name == "call_llm"]
    execute_tool_spans = [s for s in spans if s.name == "execute_tool"]

    assert len(invoke_agent_spans) == 1, (
        f"expected exactly 1 invoke_agent span, got {len(invoke_agent_spans)}: "
        f"{[s.name for s in spans]}"
    )
    invoke_agent = invoke_agent_spans[0]

    assert invoke_agent.parent is None, (
        f"invoke_agent must be the root span; parent={invoke_agent.parent}"
    )

    assert len(call_llm_spans) >= 2, (
        f"expected at least 2 call_llm spans (one per turn), "
        f"got {len(call_llm_spans)}"
    )

    assert len(execute_tool_spans) >= 2, (
        f"expected at least 2 execute_tool spans for the two my_tool calls, "
        f"got {len(execute_tool_spans)}"
    )

    for span in call_llm_spans + execute_tool_spans:
        assert span.parent is not None, (
            f"span {span.name} must have a parent (invoke_agent); got parent=None"
        )
        assert span.parent.span_id == invoke_agent.context.span_id, (
            f"span {span.name} parent.span_id={span.parent.span_id} "
            f"does not match invoke_agent span_id={invoke_agent.context.span_id}"
        )


# ---------------------------------------------------------------------
# 2. agent.trace retrieval path
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_trace_returns_populated_agent_trace_after_run(
    hierarchy_exporter: InMemorySpanExporter,
) -> None:
    """agent.trace returns the AgentTrace populated during the most recent run."""
    cfg = tinyagent.AgentConfig(
        instructions="test agent",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        max_turns=5,
        callbacks=tinyagent.CallbackRegistry(),
        name=f"tinyagent-trace-{uuid.uuid4().hex}",
    )
    agent = tinyagent.TinyAgent(cfg)

    def my_tool(value: int = 1) -> str:
        return f"v={value}"

    agent._clients["my_tool"] = tinyagent._wrap_no_exception(my_tool)

    response_1 = _func_response(
        tool_calls=[_func_call_tc("my_tool", call_id="c1", value=1)]
    )
    response_2 = _func_response(tool_calls=[_final_answer_tc("done")])

    with patch.object(
        tinyagent.any_llm, "acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.side_effect = [response_1, response_2]
        await agent.run_async("...")

    trace_obj = agent.trace
    assert trace_obj is not None, "agent.trace must be non-None after run_async"
    assert isinstance(trace_obj, tinyagent.AgentTrace)

    span_names = [span.name for span in trace_obj.spans]
    assert span_names.count("invoke_agent") == 1, (
        f"agent.trace.spans should contain exactly 1 invoke_agent, "
        f"got names={span_names}"
    )
    assert span_names.count("call_llm") >= 1, (
        f"agent.trace.spans should contain at least 1 call_llm, "
        f"got names={span_names}"
    )
    assert span_names.count("execute_tool") >= 1, (
        f"agent.trace.spans should contain at least 1 execute_tool, "
        f"got names={span_names}"
    )


@pytest.mark.asyncio
async def test_agent_trace_is_reset_between_runs(
    hierarchy_exporter: InMemorySpanExporter,
) -> None:
    """A second run_async call resets agent.trace so the previous run's spans don't leak."""
    cfg = tinyagent.AgentConfig(
        instructions="test agent",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        max_turns=3,
        callbacks=tinyagent.CallbackRegistry(),
        name=f"tinyagent-trace-reset-{uuid.uuid4().hex}",
    )
    agent = tinyagent.TinyAgent(cfg)

    def my_tool(value: int = 1) -> str:
        return f"v={value}"

    agent._clients["my_tool"] = tinyagent._wrap_no_exception(my_tool)

    r1 = _func_response(tool_calls=[_func_call_tc("my_tool", call_id="c1", value=1)])
    r1_final = _func_response(tool_calls=[_final_answer_tc("first")])
    with patch.object(
        tinyagent.any_llm, "acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.side_effect = [r1, r1_final]
        await agent.run_async("run 1")
    first_run_span_count = len(agent.trace.spans)
    assert first_run_span_count >= 3

    r2 = _func_response(tool_calls=[_final_answer_tc("second")])
    with patch.object(
        tinyagent.any_llm, "acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.return_value = r2
        await agent.run_async("run 2")
    second_run_span_count = len(agent.trace.spans)
    assert second_run_span_count >= 1
    assert second_run_span_count < first_run_span_count + 3, (
        f"agent.trace should be reset between runs; "
        f"run1 had {first_run_span_count} spans, run2 had {second_run_span_count}"
    )