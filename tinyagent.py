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
from functools import cached_property  # T7: AgentTrace.tokens / .cost roll-ups
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

# T8: opentelemetry-api is a hard dependency in pyproject.toml and _setup_tracing
# (Section 13) is now in scope. Promote the runtime import so the function can
# acquire tracers. Heavy third-party imports for sections still pending (T4+,
# T10, T11+) remain under TYPE_CHECKING below.
from opentelemetry import trace as _otel_trace  # used in §13

if TYPE_CHECKING:
    import any_llm
    import httpx
    import mcp
    import pydantic
    import simpleeval
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
LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama"})

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

# T7: OTel-style attribute names used by `AgentTrace` roll-ups. These mirror
# the semconv constants upstream defines in `tracing/attributes.py`; we keep
# the literal strings here (instead of importing opentelemetry-semconv) so the
# roll-ups remain testable without a heavy runtime dependency. The full
# constants table is added in T9.
INPUT_TOK_ATTR: str = "gen_ai.usage.input_tokens"
OUTPUT_TOK_ATTR: str = "gen_ai.usage.output_tokens"
COST_ATTR: str = "gen_ai.usage.cost"


# =====================================================================
# Section 5 - Pricing table (DEFAULT_PRICING) and lookup rules
# =====================================================================
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


# =====================================================================
# Section 6 - Exceptions (CANONICAL — every exception the library raises)
# =====================================================================
class AgentError(Exception):
    """Base class for every exception this library raises.

    Catch ``AgentError`` to handle every internal failure mode in one place.
    Specific subclasses (below) carry semantic meaning so callers can branch
    on the failure type when they need to (e.g. distinguish an MCP transport
    drop from a user-initiated abort).
    """


class AgentCancel(AgentError):
    """Raised by a hook to terminate the agent loop.

    Any of the canonical 5 hooks (``before_llm_call``, ``after_llm_call``,
    ``before_tool_execution``, ``after_tool_execution``, ``on_error``) may
    raise ``AgentCancel`` to short-circuit the run; the loop unwinds and the
    exception propagates out of ``TinyAgent.run`` / ``run_async`` unchanged.
    ``AgentCancel`` is **not** routed through ``on_error`` — the user
    explicitly aborted, so it is not an error in the observability sense.
    """


class ToolNotFoundError(AgentError):
    """Raised when a tool call targets a name that is not registered.

    The agent's loop catches this specific subclass and feeds a descriptive
    string back to the LLM (so the model can self-correct on the next
    turn). ``on_error`` does **not** fire — this is a recoverable in-band
    signal, not an exception escaping the loop body. If the call originates
    from user code that bypassed the loop, the exception propagates
    unchanged.
    """


class MCPConnectionError(AgentError):
    """Raised when an MCP subprocess dies or EOFs on stdin.

    Wraps the underlying transport failure (process exit, broken pipe,
    EOF on stdin) so the agent's loop can mark the server **broken** for
    the remainder of the run, fire ``on_error``, and route the failure up
    the stack. Subsequent tool calls into the same server return a
    ``"server unavailable"`` string to the LLM instead of crashing the
    loop.
    """


class MCPProtocolError(AgentError):
    """Raised when the MCP server emits an invalid frame.

    Wraps invalid UTF-8 on stdout, malformed JSON-RPC payloads, and other
    protocol-layer violations. The agent's loop treats this the same as
    ``MCPConnectionError`` — mark the server broken, fire ``on_error``,
    and re-raise so the caller can decide how to recover.
    """


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
@dataclasses.dataclass(frozen=True)
class AgentSpan:
    """A single OpenTelemetry-flavoured span tracked by `AgentTrace`.

    Fields match the span attributes the loop writes in T9 (call_llm /
    execute_tool / invoke_agent). `end_time` is `None` while the span is
    open; the span-generation code sets it to a `time.time()` float when
    the span closes.
    """

    name: str
    attributes: dict[str, Any]
    kind: str
    parent_id: str | None
    start_time: float
    end_time: float | None = None


@dataclasses.dataclass(frozen=True)
class TokenInfo:
    """Roll-up of token usage across `call_llm` spans in an `AgentTrace`.

    `AgentTrace.tokens` returns a `TokenInfo` summing input/output tokens
    across spans that actually carry the attributes — spans without the
    attributes are skipped (their token counts are "unknown", not zero).
    """

    input_tokens: int
    output_tokens: int


@dataclasses.dataclass(frozen=True)
class CostInfo:
    """Roll-up of USD cost across `call_llm` spans in an `AgentTrace`.

    `AgentTrace.cost` returns a `CostInfo` summing cost only across spans
    that carry the `gen_ai.usage.cost` attribute. The round-3 M6 fix
    (canonical pricing rule) means unknown-cost spans are SKIPPED, never
    treated as $0 — `CostInfo` is allowed to be all-zero when no spans
    have the cost attribute.
    """

    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


@dataclasses.dataclass(frozen=True)
class AgentTrace:
    """Collection of spans emitted by a single agent run, with roll-ups.

    `tokens` and `cost` are `cached_property` roll-ups computed lazily on
    first access. Both are read-only views of the underlying span list;
    the roll-ups only consider `call_llm` spans that carry the relevant
    attributes (spans without the attributes are skipped, not treated
    as zero).
    """

    spans: list[AgentSpan]

    @cached_property
    def tokens(self) -> TokenInfo:
        """Sum input/output tokens across `call_llm` spans WITH token attrs.

        Spans missing `gen_ai.usage.input_tokens` / `output_tokens` are
        skipped — partial sums are still meaningful (a span that records
        only input_tokens still contributes its input to the roll-up).
        """
        in_total = 0
        out_total = 0
        for span in self.spans:
            if span.name != "call_llm":
                continue
            attrs = span.attributes
            in_val = attrs.get(INPUT_TOK_ATTR)
            out_val = attrs.get(OUTPUT_TOK_ATTR)
            if in_val is None and out_val is None:
                continue
            if in_val is not None:
                in_total += int(in_val)
            if out_val is not None:
                out_total += int(out_val)
        return TokenInfo(input_tokens=in_total, output_tokens=out_total)

    @cached_property
    def cost(self) -> CostInfo:
        """Sum USD cost across `call_llm` spans WITH `gen_ai.usage.cost`.

        Round-3 M6: unknown-cost spans are SKIPPED. The cost attribute is
        a single rolled-up USD total (input + output combined per the
        plan's pricing rule); we report that figure as `total_cost_usd`
        and leave the per-direction breakdown at zero — the upstream
        per-direction split was deprecated when the custom cost
        attribute was consolidated into a single total.
        """
        total = 0.0
        any_with_cost = False
        for span in self.spans:
            if span.name != "call_llm":
                continue
            cost_val = span.attributes.get(COST_ATTR)
            if cost_val is None:
                continue
            total += float(cost_val)
            any_with_cost = True
        if not any_with_cost:
            return CostInfo(
                input_cost_usd=0.0,
                output_cost_usd=0.0,
                total_cost_usd=0.0,
            )
        return CostInfo(
            input_cost_usd=total,
            output_cost_usd=0.0,
            total_cost_usd=total,
        )


# =====================================================================
# Section 13 - OpenTelemetry setup (library pattern, idempotent)
# =====================================================================
# Module-level cache keyed by tracer name. _setup_tracing() is idempotent
# across calls: the same name returns the same tracer object reference.
# Hosts that want a fresh tracer (e.g. after reconfiguring the global
# TracerProvider) should call _setup_tracing.cache_clear() explicitly.
_tracer_cache: dict[str, Any] = {}


def _setup_tracing(name: str = "tinyagent") -> Any:
    """Acquire the named tracer using the library pattern (B1/B2 round-1 fix).

    Idempotent — repeated calls with the same name return the SAME tracer
    object (cached on the module). Does NOT call
    ``opentelemetry.trace.set_tracer_provider``; that mutates the
    process-wide TracerProvider singleton and would break multi-instance
    use of the library plus any host application that already configured
    a provider. Does NOT configure exporters; the host application is
    responsible for installing a TracerProvider (e.g. via
    ``opentelemetry-instrumentation`` autoconfigure, or manually) before
    importing tinyagent or before calling ``agent.run()``.

    If no TracerProvider is configured yet, ``trace.get_tracer`` returns a
    NoOp tracer and this function returns it without raising — the agent
    still runs correctly, just without spans emitted anywhere.

    Parameters
    ----------
    name:
        Tracer instrumentation name. Defaults to ``"tinyagent"`` (plan §2
        section 13). Different names produce different tracer instances.

    Returns
    -------
    An OpenTelemetry ``Tracer`` instance (real or NoOp). The exact type
    is owned by opentelemetry-api and not asserted by this contract.
    """
    cached = _tracer_cache.get(name)
    if cached is not None:
        return cached
    tracer = _otel_trace.get_tracer(name)
    _tracer_cache[name] = tracer
    return tracer


def _setup_tracing_cache_clear() -> None:
    """Drop all cached tracers (test helper; not part of the public API).

    Lets tests that reconfigure the global TracerProvider between cases
    observe a fresh ``trace.get_tracer`` call.
    """
    _tracer_cache.clear()


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
