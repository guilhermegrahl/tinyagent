# SPDX-License-Identifier: Apache-2.0
"""Tracing demo — wire an OTLP ``TracerProvider`` so tinyagent spans ship out.

Demonstrates:

  1. Installing a process-wide ``TracerProvider`` with an OTLP HTTP
     exporter (the canonical OpenTelemetry pattern). The provider is
     set via ``opentelemetry.trace.set_tracer_provider`` BEFORE the
     ``TinyAgent`` is constructed — that way ``_setup_tracing``
     (called inside ``TinyAgent.__init__``) picks up the real provider
     instead of returning a NoOp tracer.
  2. Constructing a ``TinyAgent`` and running a tiny ``http_get`` task.
     Every ``invoke_agent`` / ``call_llm`` / ``execute_tool`` span is
     routed to the OTLP exporter.
  3. Registering an ``on_error`` hook (via the ``register_*`` API —
     plan §0 C5, round-3 M3) to log exceptions. This shows the fifth
     canonical hook in use.

Required environment
-------------------
- ``OPENAI_API_KEY`` (or any provider key listed in
  ``tinyagent.PROVIDER_KEY_ENV``).
- ``opentelemetry-exporter-otlp-proto-http`` installed in the active
  environment (NOT a hard runtime dependency — the example imports it
  lazily so the file still imports cleanly on hosts without it).
- An OTLP collector reachable at ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or
  the default ``http://localhost:4318/v1/traces``).

Run::

    pip install opentelemetry-exporter-otlp-proto-http
    OPENAI_API_KEY=sk-... \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
        python examples/tracing_otlp.py

The script is also importable — the test suite asserts a clean import.
"""
from __future__ import annotations

import asyncio
import sys
from typing import cast

import tinyagent
from tinyagent import (
    AgentConfig,
    CallbackRegistry,
    TinyAgent,
    final_answer,
    http_get,
)


def _install_tracer_provider() -> None:
    """Install an OTLP HTTP ``TracerProvider`` for the current process.

    The provider is set BEFORE the ``TinyAgent`` is constructed so
    ``tinyagent._setup_tracing`` (called inside ``TinyAgent.__init__``)
    resolves to the real provider instead of a NoOp tracer. This
    matches the library pattern documented in §0 C5 / cross-cutting
    risk #3: tinyagent itself MUST NOT call ``set_tracer_provider``,
    but the host application is responsible for installing one.

    Importing the OTLP exporter lazily keeps this example import-clean
    on hosts that don't have ``opentelemetry-exporter-otlp-proto-http``
    installed — the test suite runs without it.
    """
    try:
        from opentelemetry import trace  # noqa: PLC0415 - lazy for optional dep
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dep
        print(  # noqa: T201
            "[tracing] OTLP exporter not installed; skipping provider "
            f"setup ({exc}). Install "
            "`opentelemetry-exporter-otlp-proto-http` to enable.",
            file=sys.stderr,
        )
        return

    resource = Resource.create({"service.name": "tinyagent-tracing-demo"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    print(  # noqa: T201
        "[tracing] TracerProvider installed with OTLP HTTP exporter.",
        file=sys.stderr,
    )


def _build_callbacks() -> CallbackRegistry:
    """Build a callback registry that logs agent errors.

    Uses the ``register_*`` API exclusively (plan §0 C5 — round-3 M3).
    ``on_error`` is the fifth canonical hook and fires from inside
    ``agent.run_async`` whenever a recoverable exception is caught.
    """
    callbacks = CallbackRegistry()

    def _on_error(ctx: object) -> None:
        # The registry's signature is ``Callable[[object], Any]`` so we
        # narrow via cast — no getattr/hasattr (per project conventions).
        typed_ctx = cast("tinyagent.Context", ctx)
        # ``ctx.error`` is the exception that triggered the hook.
        print(f"[hook] on_error: {typed_ctx.error!r}", file=sys.stderr)  # noqa: T201

    callbacks.register_on_error(_on_error)
    return callbacks


async def amain() -> str:
    """Run a single-turn task so at least one span ships to the exporter."""
    config = AgentConfig(
        instructions=(
            "Fetch the page at the URL the user gives you via `http_get` "
            "and summarise the body via `final_answer`."
        ),
        tools=[http_get, final_answer],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        callbacks=_build_callbacks(),
        name="tinyagent-tracing-demo",
    )
    agent = TinyAgent(config)
    prompt = "Fetch http://example.com and summarise what the page says."
    return await agent.run_async(prompt)


if __name__ == "__main__":
    _install_tracer_provider()
    asyncio.run(amain())
