"""T13 acceptance tests: per-span cost writer + AgentConfig.pricing wiring.

Per plan §2 section 14, §13 T13 acceptance criteria, §0 C2 + M6
(canonical pricing rule — omit ``gen_ai.usage.cost`` when the price is
unknown, never fall back to ``0.0``).

This file covers the four contract clauses the task spec calls out:

1. Cost attribute is written on the span when ``_estimate_cost`` returns
   a number (already covered in T9 — verify still passes).
2. Cost attribute is OMITTED when ``_estimate_cost`` returns ``None``
   (already covered in T9 — verify still passes).
3. ``AgentConfig(pricing={"openai:gpt-4o": (1.0, 2.0)})`` causes that
   price to be used in ``_estimate_cost`` for this agent instance only
   (overriding ``DEFAULT_PRICING``).
4. After agent construction, module-level ``PRICING_OVERRIDE`` is
   restored to its pre-construction state (the override is
   instance-local, not module-global).

The T13 implementation plumbs the per-instance pricing override from
``AgentConfig.pricing`` through ``TinyAgent.__init__`` to
``_SpanGeneration.call_llm`` -> ``_compute_cost_attribute`` ->
``_estimate_cost(..., pricing=...)``. The override never leaks into the
module-level ``PRICING_OVERRIDE`` dict (that global remains a separate
escape hatch for users who want to mutate it directly).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import tinyagent


# ---------------------------------------------------------------------
# Test stubs (mirrors the stubs used in tests/test_agent_init.py)
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


def _patched_span_gen() -> Any:
    """Patch ``tinyagent._SpanGeneration`` so ``call_model`` runs without OTel."""

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
# 1. AgentConfig.pricing field exists, defaults to None, accepts dict
# ---------------------------------------------------------------------
def test_agent_config_pricing_field_defaults_to_none() -> None:
    """``AgentConfig.pricing`` defaults to ``None`` when not supplied."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
    )
    assert cfg.pricing is None


def test_agent_config_pricing_accepts_dict() -> None:
    """``AgentConfig.pricing`` accepts a dict[str, tuple[float, float]]."""
    pricing = {"openai:gpt-4o": (1.0, 2.0)}
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
        pricing=pricing,
    )
    assert cfg.pricing == pricing


# ---------------------------------------------------------------------
# 2. TinyAgent stores the per-instance override on self._pricing_override
# ---------------------------------------------------------------------
def test_tinyagent_stores_pricing_override_on_instance() -> None:
    """``TinyAgent.__init__`` copies ``config.pricing`` onto the instance."""
    pricing = {"openai:gpt-4o": (1.0, 2.0)}
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
        pricing=pricing,
    )
    agent = tinyagent.TinyAgent(cfg)
    # The agent owns its own copy: mutations to the original dict do NOT
    # leak into the agent's runtime.
    assert agent._pricing_override == pricing
    assert agent._pricing_override is not pricing, (
        "TinyAgent must defensively copy config.pricing; storing the same "
        "dict object would let caller mutations leak into the agent"
    )


def test_tinyagent_pricing_override_is_none_when_unset() -> None:
    """When ``config.pricing`` is ``None``, ``_pricing_override`` is also ``None``."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
    )
    agent = tinyagent.TinyAgent(cfg)
    assert agent._pricing_override is None


def test_tinyagent_pricing_override_is_defensive_copy() -> None:
    """Mutating the original ``AgentConfig.pricing`` dict after construction
    does NOT affect the agent's stored override."""
    pricing = {"openai:gpt-4o": (1.0, 2.0)}
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
        pricing=pricing,
    )
    agent = tinyagent.TinyAgent(cfg)
    # Mutate the original AFTER construction.
    pricing["openai:gpt-4o"] = (99.0, 99.0)
    # The agent's stored copy is unchanged.
    assert agent._pricing_override == {"openai:gpt-4o": (1.0, 2.0)}


# ---------------------------------------------------------------------
# 3. Per-instance pricing is used in _estimate_cost (overrides DEFAULT_PRICING)
# ---------------------------------------------------------------------
def test_per_instance_pricing_overrides_default_in_estimate_cost() -> None:
    """Calling ``_estimate_cost`` with the agent's override returns the override price.

    With ``pricing={"openai:gpt-4o": (1.0, 2.0)}`` and 1M input + 1M output,
    the override cost is ``1.0 + 2.0 = 3.0`` (vs the default ``12.50``).
    """
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
        pricing={"openai:gpt-4o": (1.0, 2.0)},
    )
    agent = tinyagent.TinyAgent(cfg)

    cost = tinyagent._estimate_cost(
        model_id="openai:gpt-4o",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        pricing=agent._pricing_override,
    )
    assert cost == pytest.approx(3.0, abs=1e-9)


def test_per_instance_pricing_returns_none_for_uncovered_models() -> None:
    """When the per-instance override does NOT cover a model, ``_estimate_cost``
    returns ``None`` (full-replacement semantics, not augmentation).

    Per plan §7: ``pricing or DEFAULT_PRICING`` — when ``pricing`` is supplied,
    it FULLY replaces the lookup table. Unknown models return ``None`` so the
    span writer omits ``gen_ai.usage.cost``.
    """
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
        pricing={"openai:gpt-4o": (1.0, 2.0)},  # only overrides gpt-4o
    )
    agent = tinyagent.TinyAgent(cfg)

    # ``anthropic:claude-3-5-sonnet`` is NOT in the per-instance pricing.
    cost = tinyagent._estimate_cost(
        model_id="anthropic:claude-3-5-sonnet",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        pricing=agent._pricing_override,
    )
    assert cost is None


def test_per_instance_pricing_handles_prefix_match() -> None:
    """Per-instance pricing also gets longest-prefix matching (per plan §7)."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
        pricing={"openai:gpt-4o": (1.0, 2.0)},
    )
    agent = tinyagent.TinyAgent(cfg)

    # ``openai:gpt-4o-2024-05-13`` matches the override key "openai:gpt-4o".
    cost = tinyagent._estimate_cost(
        model_id="openai:gpt-4o-2024-05-13",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        pricing=agent._pricing_override,
    )
    assert cost == pytest.approx(3.0, abs=1e-9)


def test_no_per_instance_pricing_falls_back_to_default_pricing() -> None:
    """When no per-instance override is supplied, ``_estimate_cost`` uses ``DEFAULT_PRICING``."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
    )
    agent = tinyagent.TinyAgent(cfg)
    assert agent._pricing_override is None

    cost = tinyagent._estimate_cost(
        model_id="openai:gpt-4o",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        pricing=agent._pricing_override,  # None -> falls back to DEFAULT_PRICING
    )
    # DEFAULT_PRICING: 2.50 + 10.00 = 12.50
    assert cost == pytest.approx(12.50, abs=1e-9)


@pytest.mark.asyncio
async def test_call_model_passes_per_instance_pricing_to_span_gen() -> None:
    """``call_model`` constructs ``_SpanGeneration`` with the per-instance pricing override.

    The pricing flows through this seam: ``call_model`` -> ``_SpanGeneration(
    tracer, model, pricing=...)`` -> ``call_llm`` -> ``_compute_cost_attribute
    (..., pricing=...)`` -> ``_estimate_cost(..., pricing=...)``. Asserting
    the constructor call (with the patched ``_SpanGeneration`` MagicMock)
    verifies the wiring at its source.
    """
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
        pricing={"openai:gpt-4o": (1.0, 2.0)},
    )
    agent = tinyagent.TinyAgent(cfg)

    fake_response = _FakeResponse(
        usage=_FakeUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    )

    with patch.object(tinyagent.any_llm, "acompletion", return_value=fake_response):
        with _patched_span_gen():
            await agent.call_model(
                model="openai:gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )
            # _SpanGeneration was instantiated with the per-instance
            # pricing override as the third kwarg. Post-PR-review
            # added ``trace_collector=`` for the ``agent.trace``
            # retrieval path — agent passes its current ``AgentTrace``
            # so the span helper can mirror spans in-process.
            tinyagent._SpanGeneration.assert_called_once_with(
                agent._tracer,
                "openai:gpt-4o",
                pricing={"openai:gpt-4o": (1.0, 2.0)},
                trace_collector=agent._trace,
            )


@pytest.mark.asyncio
async def test_call_model_passes_none_pricing_when_no_override() -> None:
    """When ``config.pricing`` is unset, ``call_model`` passes ``pricing=None``
    so ``_estimate_cost`` falls back to ``DEFAULT_PRICING``."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
    )
    agent = tinyagent.TinyAgent(cfg)

    fake_response = _FakeResponse(
        usage=_FakeUsage(prompt_tokens=10, completion_tokens=20)
    )

    with patch.object(tinyagent.any_llm, "acompletion", return_value=fake_response):
        with _patched_span_gen():
            await agent.call_model(
                model="openai:gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )
            tinyagent._SpanGeneration.assert_called_once_with(
                agent._tracer,
                "openai:gpt-4o",
                pricing=None,
                trace_collector=agent._trace,
            )


# ---------------------------------------------------------------------
# 4. Module-level PRICING_OVERRIDE is restored after construction
# ---------------------------------------------------------------------
def test_pricing_override_restored_after_construction() -> None:
    """Module-level ``PRICING_OVERRIDE`` is restored to its pre-construction state
    after ``TinyAgent.__init__`` returns.

    Per task spec: "After agent construction, module-level PRICING_OVERRIDE is
    restored to its pre-construction state (instance-local override)" — the
    override must NOT leak into the module-level dict.
    """
    # Snapshot pre-construction state.
    saved = dict(tinyagent.PRICING_OVERRIDE)
    try:
        cfg = tinyagent.AgentConfig(
            instructions="x",
            tools=[],
            mcp_servers=[],
            model="openai:gpt-4o",
            pricing={"openai:gpt-4o": (1.0, 2.0)},
        )
        tinyagent.TinyAgent(cfg)
        # After construction, module-level PRICING_OVERRIDE is unchanged.
        assert dict(tinyagent.PRICING_OVERRIDE) == saved
    finally:
        tinyagent.PRICING_OVERRIDE.clear()
        tinyagent.PRICING_OVERRIDE.update(saved)


def test_pricing_override_unchanged_when_no_pricing_config() -> None:
    """Even without a ``pricing`` config, construction does not modify
    module-level ``PRICING_OVERRIDE``."""
    saved = dict(tinyagent.PRICING_OVERRIDE)
    try:
        cfg = tinyagent.AgentConfig(
            instructions="x",
            tools=[],
            mcp_servers=[],
            model="openai:gpt-4o",
        )
        tinyagent.TinyAgent(cfg)
        assert dict(tinyagent.PRICING_OVERRIDE) == saved
    finally:
        tinyagent.PRICING_OVERRIDE.clear()
        tinyagent.PRICING_OVERRIDE.update(saved)


def test_pricing_override_unchanged_across_multiple_constructions() -> None:
    """``PRICING_OVERRIDE`` is restored after EACH agent construction; no leakage
    between successive agents with different per-instance pricing."""
    saved = dict(tinyagent.PRICING_OVERRIDE)
    try:
        # Agent 1: gpt-4o override.
        cfg1 = tinyagent.AgentConfig(
            instructions="x",
            tools=[],
            mcp_servers=[],
            model="openai:gpt-4o",
            pricing={"openai:gpt-4o": (1.0, 2.0)},
        )
        tinyagent.TinyAgent(cfg1)
        assert dict(tinyagent.PRICING_OVERRIDE) == saved

        # Agent 2: different model, different override.
        cfg2 = tinyagent.AgentConfig(
            instructions="x",
            tools=[],
            mcp_servers=[],
            model="anthropic:claude-3-5-sonnet",
            pricing={"anthropic:claude-3-5-sonnet": (5.0, 10.0)},
        )
        tinyagent.TinyAgent(cfg2)
        assert dict(tinyagent.PRICING_OVERRIDE) == saved
    finally:
        tinyagent.PRICING_OVERRIDE.clear()
        tinyagent.PRICING_OVERRIDE.update(saved)


def test_pricing_override_does_not_leak_into_other_estimate_cost_calls() -> None:
    """After agent construction, calling ``_estimate_cost`` (without ``pricing=``)
    uses ``DEFAULT_PRICING``, not the agent's per-instance override.

    This proves the override is instance-local: it does not pollute the
    default lookup path that other code uses.
    """
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o",
        pricing={"openai:gpt-4o": (1.0, 2.0)},
    )
    tinyagent.TinyAgent(cfg)

    # A subsequent _estimate_cost call without the per-instance pricing
    # arg uses the default table — the override does NOT leak.
    cost = tinyagent._estimate_cost(
        model_id="openai:gpt-4o",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    # DEFAULT_PRICING: 12.50 (NOT the override 3.0).
    assert cost == pytest.approx(12.50, abs=1e-9)


# ---------------------------------------------------------------------
# 5. T9 cost-attribute writer still honours the omit-when-None rule
#    (regression guards — the per-instance plumbing must not change the
#    "write iff non-None" invariant from §0 C2 + cross-cutting risk #8).
# ---------------------------------------------------------------------
def test_existing_t9_compute_cost_attribute_passes_through_estimate_cost() -> None:
    """``_compute_cost_attribute`` delegates to ``_estimate_cost`` and returns its value.

    T13 added a ``pricing`` kwarg to ``_compute_cost_attribute`` /
    ``_estimate_cost`` so the per-instance override can be plumbed
    through. The seam contract is preserved: ``_compute_cost_attribute``
    is still a thin pass-through over ``_estimate_cost`` and returns
    its value verbatim.
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


def test_existing_t9_compute_cost_attribute_returns_none_on_unknown() -> None:
    """``_compute_cost_attribute`` returns ``None`` when ``_estimate_cost`` returns ``None``."""
    with patch.object(tinyagent, "_estimate_cost", return_value=None):
        result = tinyagent._compute_cost_attribute("ollama:llama3", 100, 50)
    assert result is None


def test_existing_t9_span_writer_omits_cost_when_estimate_returns_none() -> None:
    """``_SpanGeneration.call_llm`` omits ``gen_ai.usage.cost`` when
    ``_compute_cost_attribute`` returns ``None`` (round-3 M6 closure).

    This is the canonical rule: absence means "unknown", never "$0".
    """
    response = _FakeResponse(usage=_FakeUsage(prompt_tokens=10, completion_tokens=20))
    with patch.object(tinyagent, "_compute_cost_attribute", return_value=None):
        captured_attrs: dict[str, Any] = {}

        # Capture attrs by patching the tracer's start_as_current_span.
        fake_span_cm = MagicMock()
        fake_span_cm.__enter__ = lambda self_: None
        fake_span_cm.__exit__ = lambda self_, *args: None

        def _capture_attrs(
            _name: str, attributes: dict[str, Any] | None = None
        ) -> Any:
            captured_attrs.update(attributes or {})
            return fake_span_cm

        tracer = MagicMock()
        tracer.start_as_current_span.side_effect = _capture_attrs

        span_gen = tinyagent._SpanGeneration(tracer, model_id="openai:gpt-4o-mini")
        with span_gen.call_llm(response):
            pass

    # gen_ai.usage.cost must NOT be present.
    assert "gen_ai.usage.cost" not in captured_attrs


def test_existing_t9_span_writer_writes_cost_when_estimate_returns_number() -> None:
    """``_SpanGeneration.call_llm`` writes ``gen_ai.usage.cost`` when
    ``_compute_cost_attribute`` returns a non-None float."""
    response = _FakeResponse(usage=_FakeUsage(prompt_tokens=10, completion_tokens=20))
    sentinel_cost = 0.000987
    with patch.object(
        tinyagent, "_compute_cost_attribute", return_value=sentinel_cost
    ):
        captured_attrs: dict[str, Any] = {}

        fake_span_cm = MagicMock()
        fake_span_cm.__enter__ = lambda self_: None
        fake_span_cm.__exit__ = lambda self_, *args: None

        def _capture_attrs(
            _name: str, attributes: dict[str, Any] | None = None
        ) -> Any:
            captured_attrs.update(attributes or {})
            return fake_span_cm

        tracer = MagicMock()
        tracer.start_as_current_span.side_effect = _capture_attrs

        span_gen = tinyagent._SpanGeneration(tracer, model_id="openai:gpt-4o-mini")
        with span_gen.call_llm(response):
            pass

    assert captured_attrs["gen_ai.usage.cost"] == sentinel_cost