# SPDX-License-Identifier: Apache-2.0
#
# Forked from https://github.com/mozilla-ai/tinyagent (Apache-2.0, Copyright 2026 Mozilla.ai).
# Modifications: single-file packaging; canon 5-hook callbacks + on_error; stdio-only MCP;
# custom gen_ai.usage.cost attribute; safe calculate() via simpleeval; library-pattern OTel.
#
# T1 bootstrap stub. Bodies are placeholders. Downstream tasks (T2-T14) fill them in
# per plan §13 task breakdown. The section headers below match plan §2.
#
# Stub note: heavy third-party imports (any_llm, mcp, opentelemetry, pydantic,
# simpleeval, httpx) live under TYPE_CHECKING for T1 so `import tinyagent`
# resolves in a fresh venv without the dependency tree present yet. T11+ will
# promote them to runtime imports as their respective modules land.


# =====================================================================
# Section 1 - Module docstring (single-file attribution, Apache-2.0 notice)
# =====================================================================
"""tinyagent: a single-file ReAct agent forked from mozilla-ai/tinyagent.

A pip-installable Python package whose runtime source is this one file.
Wraps any-llm for multi-provider LLM access, native MCP over stdio for tools,
OpenTelemetry for tracing, and a canonical 5-hook callback registry for
guardrails.

License: Apache-2.0 (see LICENSE; NOTICE preserves upstream Mozilla.ai attribution).
"""


# =====================================================================
# Section 2 - Imports
# =====================================================================
from __future__ import annotations

import asyncio  # noqa: F401  # populated in T11+
import contextlib  # noqa: F401
import dataclasses  # noqa: F401
import json  # noqa: F401
import logging
import os  # noqa: F401
import re  # noqa: F401
import uuid  # noqa: F401
import warnings  # noqa: F401
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypedDict,
)

from typing_extensions import TypeAlias  # used at type hints below

if TYPE_CHECKING:
    import any_llm
    import httpx
    import mcp
    import pydantic
    import simpleeval
    from opentelemetry import trace
    from mcp import ClientSession, StdioServerParameters  # noqa: F401
    from mcp.client.stdio import stdio_client  # noqa: F401
    from mcp.types import Tool as _MCPToolType
    from pydantic import BaseModel, ConfigDict, Field  # noqa: F401


# =====================================================================
# Section 3 - __all__ (CANONICAL — per plan §10)
# =====================================================================
__all__ = [
    # core
    "TinyAgent",
    "AgentConfig",
    "tool",
    # MCP
    "MCPServer",
    "add_mcp_server",
    "MCPTool",
    # callbacks
    "CallbackRegistry",
    "Context",
    "ToolCall",
    # tracing
    "AgentTrace",
    "AgentSpan",
    "TokenInfo",
    "CostInfo",
    # exceptions
    "AgentError",
    "AgentCancel",
    "ToolNotFoundError",
    # example tools
    "calculate",
    "http_get",
    "final_answer",
    # test-helper exports
    "PROVIDER_KEY_ENV",
    "PROVIDER_EXTRA_ENV",
    # misc
    "__version__",
]


# =====================================================================
# Section 4 - Constants
# =====================================================================
__version__: str = "0.1.0"

DEFAULT_MAX_TURNS: int = 10
DEFAULT_KEEP_LAST_N: int = 10
DEFAULT_REQUEST_TIMEOUT_S: float = 120.0
SPAN_LIMITS: dict[str, int] = {}  # populated in T9

# LOCAL_PROVIDERS: providers whose cost is never recorded on the span (M6 round-2).
# T2 spec: must include ollama, vllm, and local — these are self-hosted providers
# with no per-token pricing; the cost attribute is omitted (None), not $0.00.
LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "vllm", "local"})

# PROVIDER_KEY_ENV: any-llm env var lookup per provider. Always access via
# PROVIDER_KEY_ENV.get(provider, ()) so ollama / vertex (no key required) don't
# KeyError (M10 round-2 closure).
PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "azure": "AZURE_API_KEY",
    "huggingface": "HF_TOKEN",
    "gemini": "GEMINI_API_KEY",
}

PROVIDER_EXTRA_ENV: dict[str, tuple[str, ...]] = {
    "vertex": ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"),
    "gemini": ("GOOGLE_API_KEY",),
}


# =====================================================================
# Section 5 - Pricing table (DEFAULT_PRICING) and lookup rules
# =====================================================================
# DEFAULT_PRICING: per-1M-token USD, (input, output). Sourced from published
# provider pricing as of 2026-07-06. Maintained via PR, not runtime. See
# plan §2 section 5 and §7 for the canonical lookup algorithm.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "openai:gpt-4o": (2.50, 10.00),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai:gpt-4.1": (2.00, 8.00),
    "openai:gpt-4.1-mini": (0.40, 1.60),
    "anthropic:claude-3-5-sonnet": (3.00, 15.00),
    "anthropic:claude-3-5-haiku": (0.80, 4.00),
    "anthropic:claude-opus-4": (15.0, 75.0),
    "mistral:mistral-large": (2.0, 6.0),
    "groq:llama-3.1-70b": (0.59, 0.79),
}

# PRICING_OVERRIDE: user-supplied per-call table that wins over DEFAULT_PRICING.
# Users mutate this dict directly (e.g. `tinyagent.PRICING_OVERRIDE["openai:gpt-4o"] = (0.0, 0.0)`)
# to override pricing for one model. The override dict is matched longest-prefix
# the same way DEFAULT_PRICING is, so partial keys like "openai:gpt-4o" cover
# dated variants. Empty by default; populated by the user at runtime. T13 wires
# an AgentConfig.pricing dict that copies into this module-level override per
# agent instance.
PRICING_OVERRIDE: dict[str, tuple[float, float]] = {}


def _estimate_cost(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Estimate USD cost for an LLM call. Returns None when price is unknown.

    Lookup order (CANONICAL — plan §7):
      1. Local provider? -> None (no per-token price for self-hosted).
      2. Longest-prefix match in PRICING_OVERRIDE (wins over DEFAULT_PRICING).
      3. Longest-prefix match in DEFAULT_PRICING.
      4. Otherwise -> None. NEVER return 0.00 for an unknown model.

    Pricing tuples are (input_usd_per_1m, output_usd_per_1m). USD formula:
        cost = (prompt_tokens / 1_000_000) * input_price
             + (completion_tokens / 1_000_000) * output_price

    Args:
        model_id: The full model string, e.g. "openai:gpt-4o-2024-05-13".
        prompt_tokens: Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.

    Returns:
        USD cost as a float, or None when price is unknown.
    """
    # 1. Local provider short-circuit.
    provider = model_id.partition(":")[0]
    if provider in LOCAL_PROVIDERS:
        return None

    # 2+3. Longest-prefix match against override first, then defaults.
    # Sort keys descending by length so the longest match wins deterministically.
    def _longest_match(table: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
        best_key: str | None = None
        for key in table:
            if model_id.startswith(key) and (best_key is None or len(key) > len(best_key)):
                best_key = key
        if best_key is None:
            return None
        return table[best_key]

    pricing_tuple = _longest_match(PRICING_OVERRIDE) or _longest_match(DEFAULT_PRICING)
    if pricing_tuple is None:
        return None

    input_price_per_1m, output_price_per_1m = pricing_tuple
    return (prompt_tokens / 1_000_000) * input_price_per_1m + (
        completion_tokens / 1_000_000
    ) * output_price_per_1m


# =====================================================================
# Section 6 - Exceptions (CANONICAL — every exception the library raises)
# =====================================================================
class AgentError(Exception):
    """Base class for every exception this library raises.

    Callers can catch `AgentError` to handle every internal failure mode.
    """


class AgentCancel(AgentError):
    """Raised by hooks (canonical 5-hook set) to terminate the agent loop."""


class ToolNotFoundError(AgentError):
    """Raised internally when a tool call targets an unregistered name."""


class MCPConnectionError(AgentError):
    """MCP subprocess death / EOF on stdin (M8 round-2 closure)."""


class MCPProtocolError(AgentError):
    """MCP invalid UTF-8 / malformed JSON-RPC frame (M8 round-2 closure)."""


# =====================================================================
# Section 7 - Callback Registry
# =====================================================================
class CallbackRegistry:
    """Registry of hook callables for the canonical 5-hook set.

    Stub for T1; full implementation lands in T6 (round-3 M3 storage model).
    """


# =====================================================================
# Section 8 - Context type and ToolCall TypedDict
# =====================================================================
class ToolCall(TypedDict):
    """TypedDict shape for ctx.tool_call (round-3 minor m6).

    Mirrors the relevant OpenAI tool-call message part. Any-llm returns
    ChatCompletionMessageToolCall objects with .function.{name, arguments};
    we adopt a TypedDict so users can write `tc: ToolCall = {...}` for hints.
    """

    id: str
    type: str  # Literal["function"] — enforced by callers
    function: dict[str, Any]


class Context:
    """Stub for T1; full Context lands in T7 (SimpleNamespace-like)."""


# =====================================================================
# Section 9 - Tool helpers (@tool decorator + wrappers)
# ====================================================================
def tool(fn: Callable[..., Any] | None = None, **kwargs: Any) -> Callable[..., Any]:
    """Stub decorator. Full @tool implementation lands in T4.

    Accepts either `@tool` (no-arg) or `@tool(...)` (kwarg) call forms.
    """


def _wrap_no_exception(callable_: Callable[..., Any]) -> Callable[..., Any]:
    """Stub. Full implementation lands in T4 (lifted from upstream wrappers.py)."""


def _cast_argument(value: Any, param_annotation: Any) -> Any:
    """Stub. Full implementation lands in T4 (lifted from upstream utils/cast.py)."""


# =====================================================================
# Section 10 - Example tools (shipped, importable from top-level)
# =====================================================================
def final_answer(answer: str) -> str:
    """Bare termination tool: model calls this to end the loop cleanly."""
    raise NotImplementedError


def calculate(expression: str) -> str:
    """Stub: safe expression evaluator (simpleeval) lands in T5."""
    raise NotImplementedError


async def http_get(url: str, timeout: float = 10.0) -> str:
    """Stub: httpx.AsyncClient GET lands in T5."""
    raise NotImplementedError


# =====================================================================
# Section 11 - MCP stdio client
# =====================================================================
class MCPServer:
    """Stub: stdio-only MCP server config + lifecycle lands in T10."""


def _create_tool_function(server: Any, tool: Any) -> Callable[..., Any]:
    """Stub: synthesises a callable + schema for an MCP tool. Lands in T10."""


# =====================================================================
# Section 12 - AgentTrace / AgentSpan / TokenInfo / CostInfo
# =====================================================================
class AgentSpan:
    """Stub: full implementation lands in T7."""


class TokenInfo:
    """Stub: input_tokens + output_tokens roll-up field."""


class CostInfo:
    """Stub: USD cost roll-up field."""


class AgentTrace:
    """Stub: spans list + tokens/cost roll-ups land in T7."""


# =====================================================================
# Section 13 - OpenTelemetry setup (library pattern, idempotent)
# =====================================================================
def _setup_tracing(name: str = "tinyagent") -> Any:
    """Acquire the named tracer (library pattern).

    Stub: full implementation lands in T8. Idempotent, does NOT call
    trace.set_tracer_provider, does NOT configure exporters.
    """
    raise NotImplementedError


# =====================================================================
# Section 14 - AgentConfig (Pydantic)
# =====================================================================
class AgentConfig:
    """Stub: full Pydantic model lands in T11."""


# =====================================================================
# Section 15 - TinyAgent class
# =====================================================================
class TinyAgent:
    """Stub: full TinyAgent lands in T11 + T12a-d + T13 + T14."""

    def __init__(self, config: Any) -> None:
        """Stub. Real signature: (config: AgentConfig)."""

    async def add_mcp_server(self, server: Any) -> Any:
        """Stub: returns an async context manager of synthesised tools. Lands in T14."""
        raise NotImplementedError


# Re-export `add_mcp_server` at module level for `from tinyagent import add_mcp_server`.
add_mcp_server = TinyAgent.add_mcp_server  # type: ignore[attr-defined]


# =====================================================================
# Section 16 - Module-level tinyagent.io logger (basicConfig on import)
# =====================================================================
logger = logging.getLogger("tinyagent.io")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] tinyagent.io: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# =====================================================================
# Section 17 - Footer: MCPTool runtime binding + __all__ re-affirmation
# =====================================================================
# MCPTool is re-exported from mcp.types at runtime when available. For T1 we
# ship a stub so `from tinyagent import MCPTool` resolves without forcing the
# mcp.types.Tool import into the module's top-level imports (the heavy
# mcp.types.Tool is referenced under TYPE_CHECKING above for type checkers).
try:
    from mcp.types import Tool as _ImportedMCPTool  # type: ignore[attr-defined]
    MCPTool: Any = _ImportedMCPTool
except ImportError:  # pragma: no cover - guard for future mcp API drift
    class MCPTool:  # type: ignore[no-redef,misc]
        """Fallback when mcp.types.Tool cannot be imported at runtime.

        Populated for real in T10 once mcp integration lands.
        """

        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)


# Reaffirm __all__ is the canonical list (round-3 §0 C5/C6 contract guards).
# Downstream tasks MUST NOT add a symbol to this module without adding it here
# AND to the canonical list in plan §10.
__all__ = [
    "TinyAgent",
    "AgentConfig",
    "tool",
    "MCPServer",
    "add_mcp_server",
    "MCPTool",
    "CallbackRegistry",
    "Context",
    "ToolCall",
    "AgentTrace",
    "AgentSpan",
    "TokenInfo",
    "CostInfo",
    "AgentError",
    "AgentCancel",
    "ToolNotFoundError",
    "calculate",
    "http_get",
    "final_answer",
    "PROVIDER_KEY_ENV",
    "PROVIDER_EXTRA_ENV",
    "__version__",
]
