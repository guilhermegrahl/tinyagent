"""T11 acceptance tests: TinyAgent.call_model honors request_timeout_s.

Per plan §13 cross-cutting risk #14 and §9 request_timeout section:

    AgentConfig.request_timeout_s default 120.0. ``call_model`` wraps
    ``self.llm.acompletion(...)`` in ``asyncio.wait_for(..., timeout=...)``.
    ``asyncio.TimeoutError`` is caught by the loop's exception arm, fired
    through ``on_error``, and re-raised wrapped in ``AgentError``.

T11 covers the *direct* ``call_model`` timeout surface: when
``any_llm.acompletion`` takes longer than ``request_timeout_s``, the call
must raise ``asyncio.TimeoutError`` (NOT hang forever, NOT silently swallow).

The T12a error-path tests (``tests/test_agent_loop.py``) cover the loop
integration (timeout -> on_error fires -> re-raise wrapped in ``AgentError``).
This file focuses on the unit-level behavior of ``call_model`` itself.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

import tinyagent


class _NeverCompletes:
    """Awaitable that never completes (simulates a hanging LLM request).

    ``await _NeverCompletes()`` blocks forever; ``asyncio.wait_for`` is the
    only thing that can interrupt it.
    """

    def __await__(self) -> Any:
        # Yield control forever; never return. The test relies on
        # ``asyncio.wait_for`` to cancel this coroutine.
        yield from asyncio.sleep(3600).__await__()
        return None  # unreachable; satisfies mypy


def _unique_tracer_name() -> str:
    """Return a tracer name that is unique per test invocation.

    ``test_otel_setup.py`` patches ``trace.get_tracer`` and stores a
    plain ``object()`` sentinel in the module-level ``_tracer_cache``
    under the names it tests with. After that module runs, any agent
    that requests a tracer with one of those names gets the
    object sentinel back — and ``object`` has no
    ``start_as_current_span`` method, so the span-emission path in
    ``call_model`` blows up. The cleanest fix is to give each test
    agent a unique tracer name so the cache miss path runs against
    the real OTel API every time.
    """
    return f"tinyagent-t11-{uuid.uuid4().hex}"


@pytest.mark.asyncio
async def test_call_model_raises_timeout_error_on_slow_completion() -> None:
    """When any_llm.acompletion hangs, call_model raises TimeoutError after request_timeout_s.

    Per cross-cutting risk #14, ``asyncio.wait_for`` is the timeout
    mechanism; the resulting exception is ``asyncio.TimeoutError`` (NOT
    ``asyncio.CancelledError``), which the loop's exception arm later
    fires through ``on_error`` and wraps in ``AgentError`` (T12a scope).
    """
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        name=_unique_tracer_name(),
        request_timeout_s=0.1,  # tight bound so the test is fast
    )
    agent = tinyagent.TinyAgent(cfg)

    async def _hanging_acompletion(**_kwargs: Any) -> Any:
        # Simulate a provider that never returns; the wait_for timeout
        # is the only thing that can break us out of this await.
        await _NeverCompletes()
        return None  # unreachable; satisfies mypy

    with patch.object(
        tinyagent.any_llm, "acompletion", side_effect=_hanging_acompletion
    ):
        with pytest.raises(asyncio.TimeoutError):
            await agent.call_model(
                model="openai:gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )


@pytest.mark.asyncio
async def test_call_model_does_not_swallow_timeout() -> None:
    """A timeout must propagate out of call_model — never silently logged and dropped."""
    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        name=_unique_tracer_name(),
        request_timeout_s=0.05,
    )
    agent = tinyagent.TinyAgent(cfg)

    async def _hanging(**_kwargs: Any) -> Any:
        await _NeverCompletes()
        return None  # unreachable; satisfies mypy

    raised: BaseException | None = None
    with patch.object(tinyagent.any_llm, "acompletion", side_effect=_hanging):
        try:
            await agent.call_model(
                model="openai:gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )
        except BaseException as exc:  # noqa: BLE001 — verifying the contract
            raised = exc

    assert raised is not None, "call_model must raise on timeout; it silently returned"
    assert isinstance(raised, asyncio.TimeoutError), (
        f"expected asyncio.TimeoutError, got {type(raised).__name__}: {raised!r}"
    )


@pytest.mark.asyncio
async def test_call_model_completes_within_default_timeout() -> None:
    """A fast completion within the default timeout returns the response normally."""

    @dataclass
    class _Resp:
        usage: Any = None

    cfg = tinyagent.AgentConfig(
        instructions="x",
        tools=[],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        name=_unique_tracer_name(),
        # request_timeout_s left at the default (120.0) — must NOT fire on a fast call.
    )
    agent = tinyagent.TinyAgent(cfg)

    async def _fast_completion(**_kwargs: Any) -> Any:
        return _Resp(usage=None)

    with patch.object(
        tinyagent.any_llm, "acompletion", side_effect=_fast_completion
    ):
        result = await agent.call_model(
            model="openai:gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert isinstance(result, _Resp)
