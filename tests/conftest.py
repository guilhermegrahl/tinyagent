"""Shared pytest configuration for the tinyagent test suite.

T1 only ships the import test. The fixtures below (skipif helpers for the
integration suite, ANY_LLM_TEST_MODEL derivation) are added now so downstream
tasks (T6–T16) inherit a consistent test-runner contract.
"""
from __future__ import annotations

import os
from typing import Any

import pytest


# ---------------------------------------------------------------------
# Integration skipif helpers (plan §11 — populated by T16, scaffolded here)
# ---------------------------------------------------------------------
def _resolve_provider_env() -> tuple[str, list[str]] | None:
    """Return (provider, required_env_keys) or None if ANY_LLM_TEST_MODEL unset.

    Hard requirement: must NOT raise KeyError for providers not in
    PROVIDER_KEY_ENV (e.g. ollama, vertex). `.get(..., ())` is the single
    mechanism that prevents the KeyError the round-1 skipif hit.
    """
    from tinyagent import PROVIDER_EXTRA_ENV, PROVIDER_KEY_ENV

    model = os.getenv("ANY_LLM_TEST_MODEL")
    if not model:
        return None
    provider, _, _ = model.partition(":")
    keys = list(PROVIDER_KEY_ENV.get(provider, ()))
    extras = list(PROVIDER_EXTRA_ENV.get(provider, ()))
    return provider, [k for k in keys + extras if k]


def _skipif_missing_env() -> pytest.MarkDecorator:
    """Build a per-scenario skipif marker from the current env."""
    resolved = _resolve_provider_env()
    if resolved is None:
        return pytest.mark.skipif(True, reason="ANY_LLM_TEST_MODEL not set")
    provider, required = resolved
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        return pytest.mark.skipif(
            True, reason=f"{provider} requires env vars: {missing}"
        )
    return pytest.mark.skipif(False, reason="")


# Pre-computed markers — each scenario imports the one it needs.
PROVIDER_ENV_SKIPIF: pytest.MarkDecorator = _skipif_missing_env()
ANY_LLM_MODEL_SKIPIF: pytest.MarkDecorator = pytest.mark.skipif(
    not os.getenv("ANY_LLM_TEST_MODEL"),
    reason="set ANY_LLM_TEST_MODEL=provider:model to run",
)


@pytest.fixture
def tinyagent_module() -> Any:
    """Return the imported tinyagent module (cached across tests)."""
    import tinyagent

    return tinyagent
