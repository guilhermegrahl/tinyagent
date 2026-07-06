"""T11 acceptance tests: AgentConfig + TinyAgent.__init__ + call_model wiring.

Per plan §2 sections 14 + 15, §13 T11, §0 C2 + cross-cutting risk #8 (cost
attribute omit rule), §13 cross-cutting risk #14 (request_timeout_s).

The tests in this file cover the *unit* surface that T11 ships:
- AgentConfig (Pydantic model) construction, required fields, defaults
- TinyAgent.__init__ sets the expected attributes (tracer, callbacks, clients,
  mcp_servers); always populates ``final_answer`` in ``_clients``;
  builds an any-llm client via ``AnyLLM.create``
- call_model: forwards kwargs to ``any_llm.acompletion``; uses
  ``_SpanGeneration.call_llm`` to write token + cost attrs to a span.

The actual OTel attribute writing (token counts, model id, op name, cost
present-vs-absent) is asserted at the ``_SpanGeneration`` level by
``tests/test_otel.py`` using a real ``InMemorySpanExporter`` and a
real ``TracerProvider``. This file uses a lighter mocking approach
(``_SpanGeneration`` is patched) because T11 only needs to assert
that ``call_model`` USES ``_SpanGeneration`` correctly — the
attribute population is locked in T9 / test_otel.py.

Why the lighter approach: OTel's ``TracerProvider`` is a process-wide
singleton; ``test_otel.py`` installs one and refuses to be replaced.
If this file installed a competing provider, ``test_otel.py``'s
exporter would see no spans and its tests would fail. Mocking
``_SpanGeneration`` is the seam that decouples the two test modules.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import tinyagent


# ---------------------------------------------------------------------
# any-llm response / usage stubs (tests don't depend on any-llm at runtime)
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
# _SpanGeneration mock helper
# ---------------------------------------------------------------------
def _patched_span_gen() -> Any:
    """Return a context manager that patches ``_SpanGeneration``.

    Replaces the ``_SpanGeneration`` class with a ``MagicMock`` that
    records every ``call_llm(response)`` invocation. The patched
    ``call_llm`` returns a no-op context manager so call_model's
    ``with span_gen.call_llm(response): pass`` body works without
    raising.

    The returned context manager yields the mock so tests can inspect
    the recorded calls after ``call_model`` returns.
    """

    @contextmanager
    def _noop_cm() -> Any:
        yield

    span_gen_instance = MagicMock()
    span_gen_instance.call_llm = MagicMock(return_value=_noop_cm())

    span_gen_class = MagicMock(return_value=span_gen_instance)

    @contextmanager
    def _patch() -> Any:
        with patch.object(tinyagent, "_SpanGeneration", span_gen_class):
            yield span_gen_instance

    return _patch()


# ---------------------------------------------------------------------
# AgentConfig — required fields, defaults, types
# ---------------------------------------------------------------------
def test_agent_config_is_exported() -> None:
    """AgentConfig is part of the public API (plan §10 __all__)."""
    assert hasattr(tinyagent, "AgentConfig")
    cfg_cls = tinyagent.AgentConfig
    # Pydantic model classes have a model_fields attribute
    assert hasattr(cfg_cls, "model_fields"), (
        "AgentConfig must be a Pydantic model (have model_fields)"
    )


def test_agent_config_required_fields() -> None:
    """AgentConfig requires `instructions`, `tools`, `mcp_servers`, `model` (per §2 section 14)."""
    cfg = tinyagent.AgentConfig(
        instructions="Be terse.",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
    )
    assert cfg.instructions == "Be terse."
    assert cfg.tools == []
    assert cfg.mcp_servers == []
    assert cfg.model == "openai:gpt-4o-mini"


def test_agent_config_default_max_turns() -> None:
    """``max_turns`` defaults to DEFAULT_MAX_TURNS (=10)."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
    )
    assert cfg.max_turns == tinyagent.DEFAULT_MAX_TURNS
    assert cfg.max_turns == 10


def test_agent_config_default_keep_last_n() -> None:
    """``keep_last_n`` defaults to DEFAULT_KEEP_LAST_N (=10)."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
    )
    assert cfg.keep_last_n == tinyagent.DEFAULT_KEEP_LAST_N
    assert cfg.keep_last_n == 10


def test_agent_config_default_name() -> None:
    """``name`` defaults to "tinyagent"."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
    )
    assert cfg.name == "tinyagent"


def test_agent_config_default_description() -> None:
    """``description`` defaults to empty string."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
    )
    assert cfg.description == ""


def test_agent_config_default_request_timeout_s() -> None:
    """``request_timeout_s`` defaults to DEFAULT_REQUEST_TIMEOUT_S (120.0)."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
    )
    assert cfg.request_timeout_s == tinyagent.DEFAULT_REQUEST_TIMEOUT_S
    assert cfg.request_timeout_s == 120.0


def test_agent_config_default_callbacks_is_none() -> None:
    """``callbacks`` defaults to None; TinyAgent.__init__ will create one if so."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
    )
    assert cfg.callbacks is None


def test_agent_config_default_pricing_override_is_none() -> None:
    """``pricing_override`` defaults to None (per §2 section 14)."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
    )
    assert cfg.pricing_override is None


def test_agent_config_accepts_optional_overrides() -> None:
    """AgentConfig accepts explicit values for every optional field."""
    cb = tinyagent.CallbackRegistry()
    pricing = {"openai:gpt-4o": (1.0, 2.0)}
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
        max_turns=5,
        keep_last_n=3,
        callbacks=cb,
        pricing_override=pricing,
        name="agent-x",
        description="custom",
        request_timeout_s=10.0,
    )
    assert cfg.max_turns == 5
    assert cfg.keep_last_n == 3
    assert cfg.callbacks is cb
    assert cfg.pricing_override == pricing
    assert cfg.name == "agent-x"
    assert cfg.description == "custom"
    assert cfg.request_timeout_s == 10.0


# ---------------------------------------------------------------------
# TinyAgent.__init__ — attributes, callbacks, _clients, _mcp_servers, _llm
# ---------------------------------------------------------------------
def test_tinyagent_init_sets_tracer() -> None:
    """TinyAgent.__init__ stores a tracer on ``_tracer``."""
    cfg = tinyagent.AgentConfig(
        instructions="x", tools=[], mcp_servers=[], model="openai:gpt-4o-mini"
    )
    agent = tinyagent.TinyAgent(cfg)
    assert agent._tracer is not None
    # The tracer must be an OTel Tracer; calling start_as_current_span works
    # against both real and NoOp tracers, so we just exercise the API.
    with agent._tracer.start_as_current_span("probe"):
        pass


def test_tinyagent_init_sets_callbacks_to_default() -> None:
    """When ``config.callbacks is None``, ``_callbacks`` is a fresh CallbackRegistry."""
    cfg = tinyagent.AgentConfig(
        instructions="x", tools=[], mcp_servers=[], model="openai:gpt-4o-mini"
    )
    agent = tinyagent.TinyAgent(cfg)
    assert isinstance(agent._callbacks, tinyagent.CallbackRegistry)
    # No hooks registered — every list is empty.
    for name in (
        "before_llm_call",
        "after_llm_call",
        "before_tool_execution",
        "after_tool_execution",
        "on_error",
    ):
        assert agent._callbacks._hooks[name] == []


def test_tinyagent_init_uses_existing_callbacks() -> None:
    """When the user supplies a CallbackRegistry, the agent reuses it (not a new one)."""
    cb = tinyagent.CallbackRegistry()

    def hook(ctx: object) -> None:
        pass

    cb.register_before_llm_call(hook)
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        callbacks=cb,
    )
    agent = tinyagent.TinyAgent(cfg)
    assert agent._callbacks is cb
    # The pre-registered hook survived construction.
    assert agent._callbacks._hooks["before_llm_call"] == [hook]


def test_tinyagent_init_initializes_clients_dict() -> None:
    """``_clients`` is a dict; ``final_answer`` is always present after __init__."""
    cfg = tinyagent.AgentConfig(
        instructions="x", tools=[], mcp_servers=[], model="openai:gpt-4o-mini"
    )
    agent = tinyagent.TinyAgent(cfg)
    assert isinstance(agent._clients, dict)
    assert "final_answer" in agent._clients
    # The registered value is the top-level final_answer function.
    assert agent._clients["final_answer"] is tinyagent.final_answer


def test_tinyagent_init_stores_mcp_servers_list() -> None:
    """``_mcp_servers`` is initialised to a (copy of the) configured list."""
    cfg = tinyagent.AgentConfig(
        instructions="x", tools=[], mcp_servers=[], model="openai:gpt-4o-mini"
    )
    agent = tinyagent.TinyAgent(cfg)
    assert isinstance(agent._mcp_servers, list)
    assert agent._mcp_servers == []


def test_tinyagent_init_stores_config() -> None:
    """``__init__`` stores the AgentConfig as ``self.config``."""
    cfg = tinyagent.AgentConfig(
        instructions="x", tools=[], mcp_servers=[], model="openai:gpt-4o-mini"
    )
    agent = tinyagent.TinyAgent(cfg)
    assert agent.config is cfg


def test_tinyagent_init_calls_any_llm_create() -> None:
    """``__init__`` builds the any-llm client via ``AnyLLM.create`` (plan §2 section 15)."""
    cfg = tinyagent.AgentConfig(
        instructions="x", tools=[], mcp_servers=[], model="openai:gpt-4o-mini"
    )
    sentinel_client = object()  # any-llm returns an AnyLLM instance; we just need a sentinel
    with patch.object(
        tinyagent.any_llm.AnyLLM, "create", return_value=sentinel_client
    ) as mock_create:
        agent = tinyagent.TinyAgent(cfg)
    # AnyLLM.create must have been called exactly once with provider="openai"
    # (the first segment of the model string).
    assert mock_create.call_count == 1
    call_args, _call_kwargs = mock_create.call_args
    assert call_args[0] == "openai", (
        f"AnyLLM.create must be called with the provider name as the first "
        f"positional arg; got {call_args!r}"
    )
    # The created client is stored on the agent.
    assert agent._llm is sentinel_client


# ---------------------------------------------------------------------
# call_model — forwards kwargs to any_llm.acompletion
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_call_model_forwards_kwargs_to_any_llm_acompletion() -> None:
    """call_model forwards **completion_params verbatim to any_llm.acompletion."""
    cfg = tinyagent.AgentConfig(
        instructions="x", tools=[], mcp_servers=[], model="openai:gpt-4o-mini"
    )
    agent = tinyagent.TinyAgent(cfg)

    fake_response = _FakeResponse(usage=_FakeUsage(prompt_tokens=10, completion_tokens=20))

    with patch.object(
        tinyagent.any_llm, "acompletion", return_value=fake_response
    ) as mock_acompletion:
        with _patched_span_gen():
            result = await agent.call_model(
                model="openai:gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                tool_choice="auto",
            )

    assert result is fake_response
    mock_acompletion.assert_awaited_once_with(
        model="openai:gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        tool_choice="auto",
    )


# ---------------------------------------------------------------------
# call_model — opens call_llm span via _SpanGeneration (token + cost attrs
# are written by _SpanGeneration.call_llm; full attr coverage is in
# tests/test_otel.py — this file only asserts call_model USES the seam)
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_call_model_opens_call_llm_span_with_response() -> None:
    """call_model instantiates _SpanGeneration and calls .call_llm(response)."""
    cfg = tinyagent.AgentConfig(
        instructions="x", tools=[], mcp_servers=[], model="openai:gpt-4o-mini"
    )
    agent = tinyagent.TinyAgent(cfg)

    fake_response = _FakeResponse(usage=_FakeUsage(prompt_tokens=321, completion_tokens=654))

    with patch.object(tinyagent.any_llm, "acompletion", return_value=fake_response):
        with _patched_span_gen() as span_gen_instance:
            await agent.call_model(
                model="openai:gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )
            # Inside the context manager: _SpanGeneration is the
            # patched MagicMock; assert it was called with the agent's
            # tracer and the configured model id. T13 added the
            # ``pricing=`` kwarg for the per-instance override (None
            # here because the test config does not supply one).
            tinyagent._SpanGeneration.assert_called_once_with(
                agent._tracer, "openai:gpt-4o-mini", pricing=None
            )
            # And call_llm was called with the response from any_llm.acompletion.
            span_gen_instance.call_llm.assert_called_once_with(fake_response)


@pytest.mark.asyncio
async def test_call_model_returns_response_unmodified() -> None:
    """call_model returns the response from any_llm.acompletion verbatim.

    The response object flows from any_llm.acompletion through
    _SpanGeneration.call_llm (which extracts token counts and
    cost attrs but does not mutate the response) back to the caller.
    T12a reads ``response.choices[0].message`` etc. from this return
    value, so the round-trip must preserve the object.
    """
    cfg = tinyagent.AgentConfig(
        instructions="x", tools=[], mcp_servers=[], model="openai:gpt-4o-mini"
    )
    agent = tinyagent.TinyAgent(cfg)

    fake_response = _FakeResponse(usage=_FakeUsage(prompt_tokens=10, completion_tokens=20))

    with patch.object(tinyagent.any_llm, "acompletion", return_value=fake_response):
        with _patched_span_gen():
            result = await agent.call_model(
                model="openai:gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

    assert result is fake_response
