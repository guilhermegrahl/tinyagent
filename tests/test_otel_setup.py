"""T8 acceptance tests: _setup_tracing follows the library pattern.

Per plan §13 T8 and §6 (B1/B2 round-1 fix):
- Does NOT call opentelemetry.trace.set_tracer_provider
- Calls opentelemetry.trace.get_tracer(name) only
- Idempotent across multiple calls
- Returns a non-None tracer even when no TracerProvider is configured
  (OTel API returns a NoOp tracer in that case)
"""
from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

import tinyagent
from opentelemetry import trace


def _unique_name(prefix: str) -> str:
    """Return a per-test tracer name so the module-level cache cannot mask calls."""
    return f"{prefix}-{uuid.uuid4().hex}"


def test_setup_tracing_returns_non_none_tracer() -> None:
    """_setup_tracing() must return a non-None tracer."""
    tracer: Any = tinyagent._setup_tracing(_unique_name("non-none"))
    assert tracer is not None


def test_setup_tracing_is_idempotent() -> None:
    """Calling _setup_tracing twice with the same name returns the same object."""
    name = _unique_name("idempotent")
    first = tinyagent._setup_tracing(name)
    second = tinyagent._setup_tracing(name)
    assert first is second, "two calls with the same name must return the same object"


def test_setup_tracing_passes_custom_name_to_get_tracer() -> None:
    """_setup_tracing('custom-name') routes the name through opentelemetry.trace.get_tracer."""
    custom = _unique_name("custom-name")
    sentinel = object()
    with patch.object(trace, "get_tracer", return_value=sentinel) as mock_get_tracer:
        result = tinyagent._setup_tracing(custom)
    mock_get_tracer.assert_called_once_with(custom)
    assert result is sentinel


def test_setup_tracing_does_not_call_set_tracer_provider() -> None:
    """_setup_tracing MUST NOT call opentelemetry.trace.set_tracer_provider.

    Round-1 B1 blocker: set_tracer_provider mutates the process-wide
    TracerProvider singleton. A library must never do that; the host
    application is responsible for installing a provider.
    """
    name = _unique_name("no-set-provider")
    with patch.object(trace, "set_tracer_provider") as mock_set_provider:
        tinyagent._setup_tracing(name)
    mock_set_provider.assert_not_called()


def test_setup_tracing_does_not_crash_without_tracer_provider() -> None:
    """_setup_tracing returns a usable (NoOp) tracer when no provider is configured.

    OTel's get_tracer() returns a NoOp tracer when no TracerProvider has
    been installed. The function must not raise; the agent still runs
    without spans being emitted anywhere.
    """
    # Sanity: ensure the test runs in the un-configured state.
    tracer = tinyagent._setup_tracing(_unique_name("noop"))
    assert tracer is not None
    # NoOp tracers can be asked for spans without raising.
    with tracer.start_as_current_span("smoke"):
        pass


def test_setup_tracing_default_name_is_tinyagent() -> None:
    """Default tracer name is 'tinyagent' (plan §2 section 13 signature)."""
    default = "tinyagent"
    sentinel = object()
    with patch.object(trace, "get_tracer", return_value=sentinel) as mock_get_tracer:
        result = tinyagent._setup_tracing()
    mock_get_tracer.assert_called_once_with(default)
    assert result is sentinel
