"""T2 acceptance test: pricing table + longest-prefix lookup + override.

Per plan §13 T2 acceptance criteria. Covers:
- Exact match returns correct USD cost.
- Longest-prefix match (e.g., `openai:gpt-4o-2024-05-13` matches `openai:gpt-4o`).
- Unknown model returns None (NOT zero).
- Local provider (ollama) returns None regardless of model_id.
- Override dict wins over DEFAULT_PRICING.
- Tokens calculation: 1M input + 1M output at $2.50/$10.00 -> $12.50.
"""
from __future__ import annotations

import importlib
import sys

import pytest

# Per spec: LOCAL_PROVIDERS must include ollama, vllm, and local. (T2 contract.)
EXPECTED_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm", "local"})


@pytest.fixture
def tinyagent() -> object:
    """Reload tinyagent freshly per test so module-level state is isolated."""
    sys.modules.pop("tinyagent", None)
    return importlib.import_module("tinyagent")


# ---------------------------------------------------------------------
# Constant surface
# ---------------------------------------------------------------------
def test_default_pricing_is_dict_of_tuples(tinyagent: object) -> None:
    """DEFAULT_PRICING is dict[str, tuple[float, float]] with the spec's 9 entries."""
    pricing = tinyagent.DEFAULT_PRICING  # type: ignore[attr-defined]
    assert isinstance(pricing, dict)
    # T2 spec lists exactly 9 well-known model strings ("~10" in the wording).
    assert len(pricing) >= 9
    for key, value in pricing.items():
        assert isinstance(key, str)
        assert isinstance(value, tuple)
        assert len(value) == 2
        in_price, out_price = value
        assert isinstance(in_price, float)
        assert isinstance(out_price, float)
        assert in_price > 0.0
        assert out_price > 0.0


def test_default_pricing_contains_expected_models(tinyagent: object) -> None:
    """DEFAULT_PRICING ships entries for the canonical well-known model list."""
    pricing = tinyagent.DEFAULT_PRICING  # type: ignore[attr-defined]
    required_keys = {
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
        "openai:gpt-4.1",
        "openai:gpt-4.1-mini",
        "anthropic:claude-3-5-sonnet",
        "anthropic:claude-3-5-haiku",
        "anthropic:claude-opus-4",
        "mistral:mistral-large",
        "groq:llama-3.1-70b",
    }
    assert required_keys <= set(pricing.keys())


def test_local_providers_frozenset(tinyagent: object) -> None:
    """LOCAL_PROVIDERS is a frozenset containing ollama, vllm, local."""
    local = tinyagent.LOCAL_PROVIDERS  # type: ignore[attr-defined]
    assert isinstance(local, frozenset)
    assert local == EXPECTED_LOCAL_PROVIDERS


def test_pricing_override_is_dict(tinyagent: object) -> None:
    """PRICING_OVERRIDE is a dict (empty by default; user sets entries)."""
    override = tinyagent.PRICING_OVERRIDE  # type: ignore[attr-defined]
    assert isinstance(override, dict)


def test_estimate_cost_is_callable(tinyagent: object) -> None:
    """`_estimate_cost` is exposed at module level (private-but-callable)."""
    func = tinyagent._estimate_cost  # type: ignore[attr-defined]
    assert callable(func)


# ---------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------
def test_exact_match_openai_gpt4o(tinyagent: object) -> None:
    """`openai:gpt-4o` -> $2.50 in, $10.00 out per 1M tokens (exact match)."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="openai:gpt-4o",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == pytest.approx(12.50, abs=1e-9)


def test_exact_match_anthropic_opus(tinyagent: object) -> None:
    """`anthropic:claude-opus-4` -> $15.00 in, $75.00 out per 1M tokens."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="anthropic:claude-opus-4",
        prompt_tokens=2_000_000,
        completion_tokens=500_000,
    )
    # 2.0 * 15.0 + 0.5 * 75.0 = 30.0 + 37.5 = 67.5
    assert cost == pytest.approx(67.5, abs=1e-9)


# ---------------------------------------------------------------------
# Longest-prefix match
# ---------------------------------------------------------------------
def test_longest_prefix_dated_openai_gpt4o(tinyagent: object) -> None:
    """`openai:gpt-4o-2024-05-13` matches `openai:gpt-4o` via longest-prefix."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="openai:gpt-4o-2024-05-13",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == pytest.approx(12.50, abs=1e-9)


def test_longest_prefix_mini_beats_bare(tinyagent: object) -> None:
    """`openai:gpt-4o-mini-2024-07-18` matches `openai:gpt-4o-mini` (NOT `gpt-4o`)."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="openai:gpt-4o-mini-2024-07-18",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    # mini: 0.15 + 0.60 = 0.75
    assert cost == pytest.approx(0.75, abs=1e-9)


def test_longest_prefix_anthropic_sonnet(tinyagent: object) -> None:
    """`anthropic:claude-3-5-sonnet-20241022` matches `anthropic:claude-3-5-sonnet`."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="anthropic:claude-3-5-sonnet-20241022",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    # 1.0 * 3.00 + 0.0 * 15.0 = 3.0
    assert cost == pytest.approx(3.0, abs=1e-9)


# ---------------------------------------------------------------------
# Unknown model -> None (NOT zero)
# ---------------------------------------------------------------------
def test_unknown_model_returns_none(tinyagent: object) -> None:
    """A model id with no prefix-match in DEFAULT_PRICING returns None (not 0.0)."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="openai:totally-unknown-model-9000",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost is None


def test_unregistered_provider_returns_none(tinyagent: object) -> None:
    """A provider never in DEFAULT_PRICING returns None."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="madeupprovider:flamingo-xl",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost is None


# ---------------------------------------------------------------------
# Local provider -> None regardless of model_id
# ---------------------------------------------------------------------
def test_local_provider_ollama_returns_none(tinyagent: object) -> None:
    """`ollama:llama3` returns None — local provider never gets a cost attribute."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="ollama:llama3",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost is None


def test_local_provider_vllm_returns_none(tinyagent: object) -> None:
    """`vllm:custom-model` returns None — local provider never gets a cost attribute."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="vllm:custom-model",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost is None


def test_local_provider_bare_local_returns_none(tinyagent: object) -> None:
    """`local:foo` returns None — `local` is in LOCAL_PROVIDERS."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="local:foo",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost is None


# ---------------------------------------------------------------------
# Override dict wins over DEFAULT_PRICING
# ---------------------------------------------------------------------
def test_override_dict_wins_over_default(tinyagent: object) -> None:
    """An entry in PRICING_OVERRIDE replaces the DEFAULT_PRICING lookup for that key."""
    override = tinyagent.PRICING_OVERRIDE  # type: ignore[attr-defined]
    # Clear any prior test pollution (best-effort; tests should be self-contained).
    override.clear()
    override["openai:gpt-4o"] = (0.0, 0.0)
    try:
        cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
            model_id="openai:gpt-4o",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert cost == pytest.approx(0.0, abs=1e-9)
    finally:
        override.clear()


def test_override_dict_handles_prefix_match(tinyagent: object) -> None:
    """Override dict also gets longest-prefix matching (not just exact)."""
    override = tinyagent.PRICING_OVERRIDE  # type: ignore[attr-defined]
    override.clear()
    override["openai:gpt-4o"] = (1.0, 2.0)
    try:
        cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
            model_id="openai:gpt-4o-2024-05-13",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        # override: 1.0 + 2.0 = 3.0
        assert cost == pytest.approx(3.0, abs=1e-9)
    finally:
        override.clear()


# ---------------------------------------------------------------------
# Token math
# ---------------------------------------------------------------------
def test_one_million_in_one_million_out_at_2_50_10(tinyagent: object) -> None:
    """1M input + 1M output at $2.50/$10.00 -> $12.50 (canonical spec example)."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="openai:gpt-4o",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert cost == pytest.approx(12.50, abs=1e-9)


def test_zero_tokens_returns_zero(tinyagent: object) -> None:
    """Zero tokens -> zero cost (math edge case, not an unknown model)."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="openai:gpt-4o",
        prompt_tokens=0,
        completion_tokens=0,
    )
    assert cost == pytest.approx(0.0, abs=1e-9)


def test_sub_million_tokens_proportional(tinyagent: object) -> None:
    """100k input + 50k output at gpt-4o pricing -> 0.1*2.5 + 0.05*10.0 = 0.75."""
    cost = tinyagent._estimate_cost(  # type: ignore[attr-defined]
        model_id="openai:gpt-4o",
        prompt_tokens=100_000,
        completion_tokens=50_000,
    )
    assert cost == pytest.approx(0.75, abs=1e-9)
