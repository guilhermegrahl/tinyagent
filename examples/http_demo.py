# SPDX-License-Identifier: Apache-2.0
"""HTTP demo — fetch a URL via the built-in ``http_get`` tool.

Demonstrates:

  1. Defining a custom tool with ``@tool`` (the OpenAI-compatible JSON
     schema is generated from the function signature — see
     ``tinyagent._build_json_schema``).
  2. Using the built-in ``http_get`` tool (also ``@tool``-decorated) to
     fetch a public web page.
  3. Registering an ``after_tool_execution`` hook (via the ``register_*``
     API — plan §0 C5, round-3 M3) to log the tool name and a short
     snippet of the result on every dispatch.

The example agent runs a single turn: the user prompt is "fetch the
homepage of example.com", and the agent is expected to call ``http_get``
followed by ``final_answer``.

Required environment
-------------------
- ``OPENAI_API_KEY`` (or any provider key listed in
  ``tinyagent.PROVIDER_KEY_ENV``). The default model is
  ``openai:gpt-4o-mini``.

Run::

    OPENAI_API_KEY=sk-... python examples/http_demo.py

The script is also importable — the test suite asserts a clean import.
"""
from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, cast

import tinyagent
from tinyagent import (
    AgentConfig,
    CallbackRegistry,
    TinyAgent,
    final_answer,
    http_get,
    tool,
)

if TYPE_CHECKING:
    from typing import Any


@tool
def echo(message: str) -> str:
    """Echo the input back to the caller — minimal custom ``@tool`` example.

    ``@tool`` (no parens) inspects the function signature, builds an
    OpenAI-compatible JSON schema, and returns the original callable
    with a ``tool_schema`` attribute attached. The agent loop reads
    ``tool_schema`` when advertising tools to the LLM.
    """
    return message


def _build_callbacks() -> CallbackRegistry:
    """Build a callback registry that logs every tool result.

    Uses the ``register_*`` API exclusively (plan §0 C5 — round-3 M3).
    The hook prints the tool name and a short preview of the result to
    ``stderr`` so the example is observable from the terminal.
    """
    callbacks = CallbackRegistry()

    def _after_tool(ctx: object) -> None:
        # The registry's signature is ``Callable[[object], Any]`` so we
        # narrow via cast — no getattr/hasattr (per project conventions).
        typed_ctx = cast("tinyagent.Context", ctx)
        tool_call: dict[str, Any] = typed_ctx.tool_call
        tool_name = tool_call.get("function", {}).get("name", "<unknown>")
        result = typed_ctx.tool_result or ""
        preview = result[:80] + ("..." if len(result) > 80 else "")  # noqa: PLR2004
        print(f"[hook] after_tool_execution: {tool_name} -> {preview!r}", file=sys.stderr)  # noqa: T201

    callbacks.register_after_tool_execution(_after_tool)
    return callbacks


async def amain() -> str:
    """Run the single-turn HTTP fetch task."""
    config = AgentConfig(
        instructions=(
            "You can fetch web pages via the `http_get` tool. After "
            "calling http_get, summarise the result via `final_answer`."
        ),
        tools=[echo, http_get, final_answer],
        mcp_servers=[],
        model="openai:gpt-4o-mini",
        callbacks=_build_callbacks(),
    )
    agent = TinyAgent(config)
    prompt = "Fetch http://example.com and summarise what the page says."
    return await agent.run_async(prompt)


if __name__ == "__main__":
    asyncio.run(amain())
