# SPDX-License-Identifier: Apache-2.0
"""Calculator MCP stdio example — wire a stdio MCP server into a TinyAgent.

Demonstrates the canonical multi-turn setup:

  1. Spawn a stdio MCP server that exposes ``add`` and ``multiply`` tools
     (the server fixture is ``examples/inproc_mcp_echo.py`` — it ships
     with the repo and exposes two tools via the mcp library's stdio
     transport).
  2. Attach it to a ``TinyAgent`` via ``agent.add_mcp_server(server)`` so
     the agent's tool dispatcher picks up the synthesised callables.
  3. Register a ``before_tool_execution`` hook (via the ``register_*``
     API — plan §0 C5, round-3 M3) to log every tool call before it
     runs. This proves the canonical callback surface is in use.
  4. Run a multi-turn task that asks the agent to compute an expression
     using the MCP tools and the built-in ``final_answer`` terminator.

Required environment
-------------------
- ``OPENAI_API_KEY`` (or any provider key listed in
  ``tinyagent.PROVIDER_KEY_ENV``). The default model is
  ``openai:gpt-4o-mini``; override ``config.model`` for another
  provider.

Run::

    OPENAI_API_KEY=sk-... python examples/calculator_mcp_stdio.py

The script is also importable — the test suite asserts a clean import.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import tinyagent
from tinyagent import (
    AgentConfig,
    CallbackRegistry,
    MCPServer,
    TinyAgent,
    calculate,
    final_answer,
)

if TYPE_CHECKING:
    from typing import Any

# Path to the fixture server shipped alongside the example. The fixture
# exposes two MCP tools (``add`` and ``multiply``) — the same pattern
# lives in ``examples/inproc_mcp_echo.py``.
_FIXTURE_SERVER = str(Path(__file__).with_name("inproc_mcp_echo.py"))


def _build_callbacks() -> CallbackRegistry:
    """Build a callback registry that logs every tool dispatch.

    Uses the ``register_*`` API exclusively (plan §0 C5 — round-3 M3
    closure). The hook receives a ``Context`` and appends a one-line
    trace to ``stderr`` so the example is observable from the terminal.
    """
    callbacks = CallbackRegistry()

    def _before_tool(ctx: object) -> None:
        # ``ctx.tool_call`` carries the raw assistant tool_call dict.
        # We only log the tool name to keep the example minimal.
        # The registry's signature is ``Callable[[object], Any]`` so we
        # narrow via cast — no getattr/hasattr (per project conventions).
        typed_ctx = cast("tinyagent.Context", ctx)
        tool_call: dict[str, Any] = typed_ctx.tool_call
        tool_name = tool_call.get("function", {}).get("name", "<unknown>")
        print(f"[hook] before_tool_execution: {tool_name}", file=sys.stderr)  # noqa: T201

    callbacks.register_before_tool_execution(_before_tool)
    return callbacks


async def amain() -> str:
    """Run the multi-turn task end-to-end.

    ``add_mcp_server`` is an async context manager (T14 / plan §10 Form
    A) so the example is structured around the canonical ``async with``
    spelling.
    """
    config = AgentConfig(
        instructions=(
            "You have access to MCP tools `add` and `multiply`. Use them "
            "to compute the user's arithmetic request and respond via "
            "`final_answer`."
        ),
        tools=[calculate, final_answer],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        callbacks=_build_callbacks(),
    )
    agent = TinyAgent(config)
    server = MCPServer(
        name="calc",
        command=sys.executable,
        args=[_FIXTURE_SERVER],
    )
    prompt = "Compute (17 * 23) + 4 using the MCP tools, then call final_answer."
    async with agent.add_mcp_server(server):
        return await agent.run_async(prompt)


if __name__ == "__main__":
    # ``asyncio.run`` is the canonical entry point for an async ``amain``
    # in a script. The test suite only imports this module, so the body
    # below does not affect the import-time contract.
    asyncio.run(amain())
