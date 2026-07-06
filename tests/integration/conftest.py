"""Integration-test skipif helpers (plan §11, §13 T16).

The integration suite is gated by ``ANY_LLM_TEST_MODEL`` (e.g. ``openai:gpt-4o-mini``).
When unset, every scenario in ``test_e2e_anyllm.py`` MUST skip cleanly — no
collection errors, no KeyError, no warnings.

Two markers are exposed:

- ``PROVIDER_ENV_SKIPIF`` — skip if ``ANY_LLM_TEST_MODEL`` is unset OR the
  current provider's required env vars are missing.
- ``ANY_LLM_MODEL_SKIPIF`` — skip if only ``ANY_LLM_TEST_MODEL`` is unset
  (no provider-key check). Used by ``test_on_error_real_failure_mode``,
  which intentionally uses an invalid model id.

Hard rule (round-2 M10 closure): ``PROVIDER_KEY_ENV`` is ALWAYS accessed via
``.get(provider, ...)`` — never ``[provider]``. The previous subscript
KeyError'd for ollama / vertex (no key required).
"""

# Implicit namespace package; pytest auto-discovers conftest.py without
# requiring an __init__.py sibling.
# ruff: noqa: INP001

from __future__ import annotations

import os

import pytest

from tinyagent import PROVIDER_EXTRA_ENV, PROVIDER_KEY_ENV


def _resolve_provider_env() -> tuple[str, list[str]] | None:
    """Return ``(provider, required_env_keys)`` or ``None`` if model unset.

    ``PROVIDER_KEY_ENV`` maps provider -> single env-var name (str) while
    ``PROVIDER_EXTRA_ENV`` maps provider -> tuple of extra names. The plan's
    ``list(PROVIDER_KEY_ENV.get(provider, ()))`` form iterates a string as
    chars (returning e.g. ``['O', 'P', 'E', 'N', ...]``); that ALWAYS
    reports "missing" and skips the test even when the real key is set.
    Normalise both lookups to ``list[str]`` here so the skipif actually
    reflects the env state.

    ``.get(provider, ...)`` (NEVER ``[provider]``) — round-2 M10 closure.
    """
    model = os.getenv("ANY_LLM_TEST_MODEL")
    if not model:
        return None
    provider, _, _ = model.partition(":")
    key = PROVIDER_KEY_ENV.get(provider)
    keys = [key] if key else []
    extras = list(PROVIDER_EXTRA_ENV.get(provider, ()))
    return provider, [k for k in keys + extras if k]


def _skipif_missing_env() -> pytest.MarkDecorator:
    """Build the per-scenario ``PROVIDER_ENV_SKIPIF`` marker from the env."""
    resolved = _resolve_provider_env()
    if resolved is None:
        return pytest.mark.skipif(condition=True, reason="ANY_LLM_TEST_MODEL not set")
    provider, required = resolved
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        return pytest.mark.skipif(
            condition=True,
            reason=f"{provider} requires env vars: {missing}",
        )
    return pytest.mark.skipif(condition=False, reason="")


# Pre-computed markers — each scenario imports the one it needs.
PROVIDER_ENV_SKIPIF: pytest.MarkDecorator = _skipif_missing_env()
ANY_LLM_MODEL_SKIPIF: pytest.MarkDecorator = pytest.mark.skipif(
    not os.getenv("ANY_LLM_TEST_MODEL"),
    reason="set ANY_LLM_TEST_MODEL=provider:model to run",
)
