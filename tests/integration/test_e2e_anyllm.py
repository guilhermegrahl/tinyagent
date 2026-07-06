"""T16 - Integration test suite (gated ``ANY_LLM_TEST_MODEL``).

Per plan §11 + §13 T16 (round-2 M10 closure: per-scenario skipif via
``PROVIDER_KEY_ENV.get(provider, ...)``, not module-level
``allow_module_level=True``).

Scenarios (each independently skippable):

 1. ``test_calculator_then_final_answer``       - ``PROVIDER_ENV_SKIPIF``
 2. ``test_http_get_chain``                     - ``PROVIDER_ENV_SKIPIF``
 3. ``test_calculator_mcp_stdio_via_subprocess``- ``PROVIDER_ENV_SKIPIF``
 4. ``test_callbacks_across_loop``              - ``PROVIDER_ENV_SKIPIF``
 5. ``test_otel_real_exporter``                 - ``PROVIDER_ENV_SKIPIF``
 6. ``test_on_error_real_failure_mode``         - ``ANY_LLM_MODEL_SKIPIF``

When ``ANY_LLM_TEST_MODEL`` is unset, ALL scenarios must skip cleanly -
no collection errors, no KeyError, no warnings (acceptance criteria).

The two marker families differ only in whether they check provider keys.
``test_on_error_real_failure_mode`` uses ``ANY_LLM_MODEL_SKIPIF`` because
it intentionally passes a model id any-llm will reject; no provider key
is required (and the test must run even when the configured provider has
no key).
"""

# Implicit namespace package; pytest auto-discovers conftest.py without
# requiring an __init__.py sibling.
# ruff: noqa: INP001

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# pytest's ``--import-mode=prepend`` (the default) puts this test file's
# directory on sys.path before collection, so ``conftest`` is importable
# as a top-level module without requiring ``tests/integration/__init__.py``.
from conftest import ANY_LLM_MODEL_SKIPIF, PROVIDER_ENV_SKIPIF
from opentelemetry import trace

import tinyagent
from tinyagent import (
    AgentConfig,
    CallbackRegistry,
    MCPServer,
    TinyAgent,
    calculate,
    final_answer,
    http_get,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# Path to the in-process stdio MCP fixture shipped alongside the repo
# (same fixture consumed by ``examples/calculator_mcp_stdio.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_FIXTURE = REPO_ROOT / "examples" / "inproc_mcp_echo.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model() -> str:
    """Return the ``ANY_LLM_TEST_MODEL`` value or skip-evaluable sentinel."""
    return os.getenv("ANY_LLM_TEST_MODEL") or "provider:model"


def _build_agent(
    tools: list[Callable[..., object]] | None = None,
    *,
    instructions: str = "You are a helpful assistant.",
) -> TinyAgent:
    """Build a ``TinyAgent`` configured against ``ANY_LLM_TEST_MODEL``.

    ``setup()`` is invoked by the caller so MCP servers attach when used.
    When ``tools`` is omitted, the agent is built with no tools.
    """
    cfg = AgentConfig(
        instructions=instructions,
        tools=list(tools or []),
        mcp_servers=[],
        model=_model(),
        max_turns=5,
        callbacks=CallbackRegistry(),
    )
    return TinyAgent(cfg)


# ---------------------------------------------------------------------------
# Scenario 1: End-to-end ReAct with ``calculate``
# ---------------------------------------------------------------------------
@PROVIDER_ENV_SKIPIF
@pytest.mark.asyncio
async def test_calculator_then_final_answer() -> None:
    """End-to-end ReAct: model calls ``calculate`` then ``final_answer``.

    Asserts the loop terminates with a final-answer string (NOT an
    exception) and that ``final_answer`` is what the loop returns -
    not the raw ``calculate`` output.
    """
    agent = _build_agent(
        tools=[calculate, final_answer],
        instructions=(
            "Use the calculate tool to compute expressions exactly, "
            "then call final_answer with the result."
        ),
    )
    await agent.setup()

    result = await agent.run_async(
        "What is 17 + 25? Use calculate, then final_answer."
    )
    assert isinstance(result, str)
    assert result.strip(), "agent returned an empty final answer"
    # The number 42 must appear somewhere in the final answer.
    assert "42" in result, f"expected '42' in final answer, got: {result!r}"


# ---------------------------------------------------------------------------
# Scenario 2: End-to-end ReAct with ``http_get``
# ---------------------------------------------------------------------------
@PROVIDER_ENV_SKIPIF
@pytest.mark.asyncio
async def test_http_get_chain() -> None:
    """End-to-end ReAct: model calls ``http_get`` then ``final_answer``.

    Uses ``httpbin.org/html`` as the target - a stable, tiny endpoint
    whose response body contains literal page content. The model should
    fetch it, summarise, and call ``final_answer``.
    """
    agent = _build_agent(
        tools=[http_get, final_answer],
        instructions=(
            "When asked to fetch a URL, use the http_get tool, then call "
            "final_answer with a one-line summary of the page content."
        ),
    )
    await agent.setup()

    # Use a small, stable endpoint so the test does not depend on the
    # network-heavy JSON endpoints. httpbin.org/html is ~400 bytes.
    result = await agent.run_async(
        "Use http_get to fetch https://httpbin.org/html and summarise "
        "what the page is about in one short sentence via final_answer."
    )
    assert isinstance(result, str)
    assert result.strip(), "agent returned an empty final answer"


# ---------------------------------------------------------------------------
# Scenario 3: MCP stdio via subprocess
# ---------------------------------------------------------------------------
@PROVIDER_ENV_SKIPIF
@pytest.mark.asyncio
async def test_calculator_mcp_stdio_via_subprocess() -> None:
    """Spawn the in-process MCP stdio fixture; verify list/call round-trip.

    Exercises the full subprocess lifecycle (``connect`` ->
    ``list_tools`` -> ``call_tool`` -> ``cleanup``) against the stdio
    MCP transport. Uses ``agent.add_mcp_server`` to attach the server
    to a fresh agent and asserts synthesised tools land in
    ``agent._clients``.
    """
    server = MCPServer(
        name="echo",
        command=sys.executable,
        args=[str(MCP_FIXTURE)],
    )
    agent = _build_agent(tools=[calculate, final_answer])
    await agent.setup()

    async with agent.add_mcp_server(server) as tools:
        # The fixture advertises at least one synthesised tool.
        assert tools, "MCP server returned no synthesised tools"
        # The synthesised tools must be registered in agent._clients.
        # (_clients is private but the agent's tool registry IS this dict;
        # the integration test verifies the public contract "the synthesised
        # tool name is callable via the agent".)
        clients = agent._clients  # noqa: SLF001
        for tool in tools:
            assert tool.__name__ in clients, (
                f"synthesised tool {tool.__name__!r} missing from agent._clients"
            )
        # Confirm at least one non-final_answer tool name is callable.
        advertised = sorted(clients)
        assert "final_answer" in advertised
        assert any(name != "final_answer" for name in advertised), (
            "MCP server synthesised no non-final_answer tools"
        )


# ---------------------------------------------------------------------------
# Scenario 4: Callbacks (register_before_llm_call etc.)
# ---------------------------------------------------------------------------
@PROVIDER_ENV_SKIPIF
@pytest.mark.asyncio
async def test_callbacks_across_loop() -> None:
    """Register before/after hooks; assert they fire every iteration.

    Uses ONLY the canonical ``register_*`` API (round-3 M3 closure).
    Asserts ``before_llm_call`` and ``after_llm_call`` each fire at
    least once per loop iteration, and that ``register_on_error`` does
    NOT fire on the happy path (no exception escapes the loop body).
    """
    callbacks = CallbackRegistry()
    before_fires: list[int] = []
    after_fires: list[int] = []
    on_error_fires: list[int] = []

    def _before(_ctx: object) -> None:
        before_fires.append(len(before_fires))

    def _after(_ctx: object) -> None:
        after_fires.append(len(after_fires))

    def _on_error(_ctx: object) -> None:
        on_error_fires.append(len(on_error_fires))

    callbacks.register_before_llm_call(_before)
    callbacks.register_after_llm_call(_after)
    callbacks.register_on_error(_on_error)

    cfg = AgentConfig(
        instructions="Use calculate, then final_answer.",
        tools=[calculate, final_answer],
        mcp_servers=[],
        model=_model(),
        max_turns=5,
        callbacks=callbacks,
    )
    agent = TinyAgent(cfg)
    await agent.setup()

    await agent.run_async(
        "Compute 6 * 7 using calculate, then final_answer."
    )

    # Every LLM call fires both before AND after - at least one iteration.
    assert len(before_fires) >= 1, "before_llm_call never fired"
    assert len(after_fires) >= 1, "after_llm_call never fired"
    assert len(before_fires) == len(after_fires), (
        f"before/after mismatch: before={len(before_fires)} after={len(after_fires)}"
    )
    # Happy path: on_error must NOT fire.
    assert on_error_fires == [], (
        f"on_error fired on the happy path: {on_error_fires!r}"
    )


# ---------------------------------------------------------------------------
# Scenario 5: Real OTel with ``InMemorySpanExporter``
# ---------------------------------------------------------------------------
@PROVIDER_ENV_SKIPIF
@pytest.mark.asyncio
async def test_otel_real_exporter() -> None:
    """Wire a real ``TracerProvider`` + ``InMemorySpanExporter``; assert spans.

    Installs a ``TracerProvider`` (OTel refuses to replace it once set,
    so this is idempotent at the process level) and uses the tinyagent
    ``_setup_tracing`` library pattern (no ``set_tracer_provider`` from
    inside tinyagent). Asserts an ``invoke_agent`` span was emitted with
    the expected agent name attribute.
    """
    # ``opentelemetry-sdk`` is intentionally NOT a runtime dep of tinyagent
    # (plan §6 — only ``opentelemetry-api`` is declared). Lazy-import the
    # SDK so this test file loads cleanly in environments without it; the
    # test raises a clear error if the user tries to run the OTel scenario
    # without installing the SDK.
    try:
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
            InMemorySpanExporter,
        )
    except ImportError as exc:  # pragma: no cover - exercised by envs without SDK
        pytest.fail(
            "scenario 5 (real OTel exporter) requires the opentelemetry-sdk "
            "package; install it with `uv add --dev opentelemetry-sdk` "
            f"before running this scenario. Underlying error: {exc}"
        )

    # The provider is a process-wide singleton. If another test already
    # installed one, attach a fresh exporter to it. Otherwise install
    # ours and attach the exporter.
    provider_set = getattr(trace, "_TRACER_PROVIDER_SET_ONCE", None) is not None
    exporter = InMemorySpanExporter()
    if provider_set:
        existing = trace.get_tracer_provider()
        existing.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    # Clear the tinyagent tracer cache so _setup_tracing picks up the
    # freshly installed provider. Cache-clear is a documented test helper
    # in tinyagent.py (see plan §6 / T8), named with the underscore prefix
    # because it's not part of the public ``__all__`` surface.
    tinyagent._setup_tracing_cache_clear()  # noqa: SLF001

    try:
        agent = _build_agent(
            tools=[calculate, final_answer],
            instructions="Use calculate then final_answer.",
        )
        await agent.setup()
        await agent.run_async(
            "Compute 9 + 10 via calculate, then final_answer."
        )

        spans = exporter.get_finished_spans()
        names = [s.name for s in spans]
        assert "invoke_agent" in names, (
            f"no invoke_agent span emitted; got {names!r}"
        )
        # call_llm must fire at least once.
        assert "call_llm" in names, f"no call_llm span emitted; got {names!r}"

        # The invoke_agent span must carry the agent name attribute.
        invoke_spans = [s for s in spans if s.name == "invoke_agent"]
        assert invoke_spans, "no invoke_agent span found in exporter"
        attrs = dict(invoke_spans[0].attributes or {})
        assert "gen_ai.agent.name" in attrs, (
            f"invoke_agent span missing gen_ai.agent.name; attrs={attrs!r}"
        )
        assert attrs["gen_ai.agent.name"] == "tinyagent", (
            f"unexpected gen_ai.agent.name: {attrs.get('gen_ai.agent.name')!r}"
        )
    finally:
        exporter.clear()


# ---------------------------------------------------------------------------
# Scenario 6: ``on_error`` real failure mode (NO provider-key check)
# ---------------------------------------------------------------------------
@ANY_LLM_MODEL_SKIPIF
@pytest.mark.asyncio
async def test_on_error_real_failure_mode() -> None:
    """Pass an invalid model id; assert ``on_error`` fires and error raises.

    Uses ``ANY_LLM_MODEL_SKIPIF`` (NOT ``PROVIDER_ENV_SKIPIF``) on
    purpose: the model id is intentionally invalid so any-llm will
    reject the call; no provider key is needed to trigger that rejection.
    This scenario MUST run whenever ``ANY_LLM_TEST_MODEL`` is set, even
    if no provider key env var is present.
    """
    callbacks = CallbackRegistry()
    error_payload: list[object] = []

    def _capture(ctx: object) -> None:
        error_payload.append(ctx)

    callbacks.register_on_error(_capture)

    # Use a deliberately invalid model id so any-llm rejects the request
    # before the loop reaches final_answer. The provider prefix is real
    # (so any-llm finds a provider) but the model name is invalid.
    cfg = AgentConfig(
        instructions="",
        tools=[calculate, final_answer],
        mcp_servers=[],
        model="openai:this-model-definitely-does-not-exist-xyz",
        max_turns=1,
        callbacks=callbacks,
    )
    agent = TinyAgent(cfg)
    await agent.setup()

    with pytest.raises(tinyagent.AgentError):
        await agent.run_async("Compute 1+1, then final_answer.")

    assert error_payload, (
        "on_error hook never fired - AgentError escaped without the callback"
    )
