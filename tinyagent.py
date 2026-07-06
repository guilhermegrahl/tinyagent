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
import functools  # T10: functools.wraps for synthesised MCP tool callables
from functools import cached_property  # T7: AgentTrace.tokens / .cost roll-ups
import inspect  # T4: @tool decorator uses inspect.signature
import json  # noqa: F401
import logging
import os  # noqa: F401
import re  # noqa: F401
import types as _types  # T10: TracebackType for __aexit__ signature
from types import SimpleNamespace  # T12a: Context is a SimpleNamespace — see §2 section 8
import uuid  # noqa: F401
import warnings  # noqa: F401
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Protocol,
    TypedDict,
    cast,
    get_origin,
    overload,
)

from typing_extensions import TypeAlias  # used at type hints below

# T8: opentelemetry-api is a hard dependency in pyproject.toml and _setup_tracing
# (Section 13) is now in scope. Promote the runtime import so the function can
# acquire tracers. T10: mcp runtime imports for MCPServer. T11: any_llm +
# pydantic for AgentConfig / TinyAgent. T5: simpleeval + httpx for example
# tools. All heavy third-party imports now at runtime; nothing under TYPE_CHECKING.
from opentelemetry import trace as _otel_trace  # used in §13

# T10: MCPServer runtime imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# T11: any_llm (call_model) + pydantic (AgentConfig)
import any_llm
import pydantic
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import mcp  # noqa: F401  # runtime import above
    from mcp.types import Tool as _MCPToolType


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
# T9: span attribute size caps. Tool args/results are JSON-serialised and
# clipped against these limits so a 1 MB tool payload cannot blow up the
# span budget. Tuned in T9; safe defaults chosen for both human-readable
# traces and OTLP exporter budgets.
SPAN_LIMITS: dict[str, int] = {
    "tool_args": 4096,
    "tool_result": 4096,
    "input_messages": 8192,
    "output": 4096,
}
_SPAN_TRUNCATION_MARKER: str = "...[truncated]"

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
# Section 7 - Callback Registry (CANONICAL — round-3 M2 + M3 storage model)
# =====================================================================
class CallbackRegistry:
    """Registry of hook callables for the canonical 5-hook set.

    Storage is dict-backed: `self._hooks: dict[str, list[Callable]]` keyed by
    hook name. Users register hooks via `register_*` methods; dispatch reads
    via `self._hooks.get(name, ())`. The attribute-style form
    `cb.before_llm_call.append(fn)` is **not** supported and raises
    AttributeError — see §0 C5 (round-3 M3 closure) for rationale.

    Hook signature (CANONICAL — round-3 M2): one positional `ctx` argument;
    return value discarded. Both sync (`(ctx) -> None`) and async
    (`(ctx) -> Awaitable[None]`) hooks are supported. Async hooks are
    awaited via the pinned event loop (set by `run()`) using
    `asyncio.run_coroutine_threadsafe`; this is the only correct path for
    sync `run()` (peer-review M3 round-1 closure).
    """

    __slots__ = ("_hooks", "_loop")

    # Canonical hook names — frozen for the lifetime of the class.
    _HOOK_NAMES: tuple[str, ...] = (
        "before_llm_call",
        "after_llm_call",
        "before_tool_execution",
        "after_tool_execution",
        "on_error",
    )

    def __init__(self) -> None:
        """Initialise with empty per-hook lists and no pinned loop."""
        self._hooks: dict[str, list[Callable[..., Any]]] = {n: [] for n in self._HOOK_NAMES}
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Registration — five `register_*` methods (one per canonical hook)
    # ------------------------------------------------------------------
    def register_before_llm_call(self, fn: Callable[[object], Any]) -> None:
        """Append `fn` to the `before_llm_call` hook list."""
        self._hooks["before_llm_call"].append(fn)

    def register_after_llm_call(self, fn: Callable[[object], Any]) -> None:
        """Append `fn` to the `after_llm_call` hook list."""
        self._hooks["after_llm_call"].append(fn)

    def register_before_tool_execution(self, fn: Callable[[object], Any]) -> None:
        """Append `fn` to the `before_tool_execution` hook list."""
        self._hooks["before_tool_execution"].append(fn)

    def register_after_tool_execution(self, fn: Callable[[object], Any]) -> None:
        """Append `fn` to the `after_tool_execution` hook list."""
        self._hooks["after_tool_execution"].append(fn)

    def register_on_error(self, fn: Callable[[object], Any]) -> None:
        """Append `fn` to the `on_error` hook list."""
        self._hooks["on_error"].append(fn)

    # ------------------------------------------------------------------
    # Dispatch — sync entry points
    # ------------------------------------------------------------------
    def dispatch(self, name: str, ctx: object) -> None:
        """Dispatch a hook event. Supports sync and async hooks.

        Iterates `self._hooks.get(name, ())` and invokes each hook with
        `ctx`. For hooks that return a coroutine, the coroutine is
        awaited:

        - If a loop is pinned (`self._loop` is not None), the coroutine is
          scheduled via `asyncio.run_coroutine_threadsafe` against the
          pinned loop and the dispatch blocks on the resulting future.
          This is the canonical sync path used by `agent.run()`.
        - If no loop is pinned, the coroutine is awaited via `asyncio.run`
          (top-level / out-of-loop path). The implementation never
          silently drops a coroutine — that was the round-1 bug C7
          flagged and the contract here closes it.

        Hook return values are DISCARDED. The hook contract is
        fire-and-forget: `Callable[[Context], None] | Callable[[Context],
        Awaitable[None]]` (round-3 M2).
        """
        for hook in self._hooks.get(name, ()):
            result = hook(ctx)
            if asyncio.iscoroutine(result):
                if self._loop is not None:
                    # Pinned-loop bridge — required for sync `run()`.
                    future = asyncio.run_coroutine_threadsafe(self._await_coro(result), self._loop)
                    future.result()
                else:
                    # No pinned loop; spin a one-shot loop to await the
                    # coroutine. This is the entry point for tests and
                    # one-off sync callers outside the agent runtime.
                    asyncio.run(self._await_coro(result))

    def dispatch_sync(self, name: str, ctx: object) -> None:
        """Bridge async hooks to a sync context via the pinned event loop.

        Asserts `self._loop` is set; the assertion exists to prevent the
        silent-coroutine-drop bug from peer-review M3 round-1. The caller
        MUST have pinned the loop (via `self._loop = asyncio.get_event_loop()`
        inside the coroutine that `run_async_in_sync` runs) before calling
        `dispatch_sync`. If the hook is async, it is scheduled on the
        pinned loop via `asyncio.run_coroutine_threadsafe` and this
        method blocks on the resulting future.
        """
        assert self._loop is not None, (
            "CallbackRegistry.dispatch_sync called before run() pinned the loop. "
            "This is a bug; please open an issue."
        )
        for hook in self._hooks.get(name, ()):
            result = hook(ctx)
            if asyncio.iscoroutine(result):
                future = asyncio.run_coroutine_threadsafe(self._await_coro(result), self._loop)
                future.result()

    # ------------------------------------------------------------------
    # Dispatch — async entry point (used from the agent loop body — T12a)
    # ------------------------------------------------------------------
    async def dispatch_async(self, name: str, ctx: object) -> None:
        """Await async hooks directly; sync hooks run inline.

        Used by ``TinyAgent.run_async`` (and any other async caller).
        Iterates ``self._hooks.get(name, ())`` and invokes each hook with
        ``ctx``. Coroutine return values are awaited on the caller's
        loop; the dispatch helper never silently drops a coroutine — the
        round-1 peer-review C7 bug is closed here.

        Returns ``None``. Errors raised by hooks are propagated to the
        caller; ``AgentCancel`` and other exceptions are NOT caught.
        """
        for hook in self._hooks.get(name, ()):
            result = hook(ctx)
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------
    # Internal — single coroutine-await helper used by run_coroutine_threadsafe
    # ------------------------------------------------------------------
    async def _await_coro(self, coro: Coroutine[Any, Any, object]) -> object:
        """Pass-through awaitable wrapping a coroutine for the pinned loop.

        Per plan §2 section 7: `run_coroutine_threadsafe` needs an awaitable
        bound to the target loop. The method body is a single `await`, but
        routing through this helper gives a self-documenting call site
        AND a single place to add error wrapping later if needed (e.g.,
        catching CancelledError to release loop resources on shutdown).
        """
        return await coro

    # ------------------------------------------------------------------
    # Test helper
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Empty every per-hook list in `self._hooks` (keeps the keys).

        Test-only convenience. Production code never calls this — tests
        use it to reset the registry between scenarios without rebuilding
        the registry object (which would lose any loop pinning).
        """
        for name in self._HOOK_NAMES:
            self._hooks[name].clear()


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


class Context(SimpleNamespace):
    """Hook context passed to every callback (plan §2 section 8, §5).

    Fields are populated selectively by the loop based on the hook name:

    - ``agent: TinyAgent`` — set on every hook (the agent the loop belongs to).
    - ``span: Any`` — the active OTel span (set by the loop / span-generation).
    - ``trace: AgentTrace | None`` — top-level trace collector (set per run).
    - ``message: Any`` — populated on LLM hooks (``before_llm_call`` /
      ``after_llm_call``) with the model response message.
    - ``tool_call: ToolCall | None`` — populated on tool hooks
      (``before_tool_execution`` / ``after_tool_execution``) with the
      tool call being executed.
    - ``tool_result: Any`` — populated on ``after_tool_execution`` with
      the result string the loop will feed back to the LLM. For
      ``final_answer`` this is the captured answer.
    - ``error: BaseException | None`` — populated on ``on_error`` with
      the escaping exception.
    - ``turn: int`` — current turn index (0-based).

    Unused fields stay at their default (``None`` for object fields).
    """

    def __init__(
        self,
        *,
        agent: Any = None,
        span: Any = None,
        trace: Any = None,
        tool_call: Any = None,
        tool_result: Any = None,
        message: Any = None,
        error: BaseException | None = None,
        turn: int = 0,
    ) -> None:
        super().__init__(
            agent=agent,
            span=span,
            trace=trace,
            tool_call=tool_call,
            tool_result=tool_result,
            message=message,
            error=error,
            turn=turn,
        )


# =====================================================================
# Section 9 - Tool helpers (@tool decorator + wrappers)
# =====================================================================
# Mapping from a Python primitive type to its JSON-Schema type string.
# Used by `@tool` to materialise the parameter schema and by `_cast_argument`
# to coerce stringified JSON values back into their declared type.
_PRIMITIVE_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_schema_type(annotation: Any) -> str | None:
    """Map a Python annotation to a JSON-Schema type string.

    Returns None for annotations we don't recognise; the schema builder
    skips the `type` field in that case so callers can still inspect the
    parameter name and default.
    """
    if annotation in _PRIMITIVE_JSON_TYPES:
        return _PRIMITIVE_JSON_TYPES[annotation]
    # Parameterised generics: list[X], dict[X, Y] → "array" / "object".
    origin = get_origin(annotation)
    if origin in _PRIMITIVE_JSON_TYPES:
        return _PRIMITIVE_JSON_TYPES[origin]
    return None


def _build_json_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build an OpenAI-compatible JSON schema dict from a callable's signature.

    The schema contains:
      - `name`: the callable's __name__
      - `description`: the callable's docstring (None if absent)
      - `parameters`: JSON-Schema object with `properties` (name → {type, default?})
        and `required` (parameter names with no default value, in signature order)

    String annotations (e.g. from `from __future__ import annotations` or
    `PEP 563` string-form hints) are resolved via `inspect.signature`'s
    `eval_str=True` mode, which evaluates them against the function's
    `__globals__` and the builtins.
    """
    try:
        sig = inspect.signature(fn, eval_str=True)
    except (NameError, TypeError):
        # If evaluation fails (e.g. undefined forward reference), fall back
        # to the raw signature and accept string annotations as unresolvable.
        sig = inspect.signature(fn)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        # Skip *args / **kwargs — tools never expose variadic params to the LLM.
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        entry: dict[str, Any] = {}
        json_type = _json_schema_type(param.annotation)
        if json_type is not None:
            entry["type"] = json_type
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        properties[name] = entry
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "name": fn.__name__,
        "description": inspect.getdoc(fn),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


class _ToolCallable(Protocol):
    """Structural type for a function decorated with `@tool`.

    The decorator attaches two extra attributes to the wrapped callable:
    `tool_schema` (the JSON-Schema dict) and `is_tool` (always True).
    mypy is told this Protocol is the return type; the cast in `_attach_schema`
    is a runtime no-op.
    """

    tool_schema: dict[str, Any]
    is_tool: bool

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _attach_schema(fn: Callable[..., Any], schema: dict[str, Any]) -> _ToolCallable:
    """Attach `tool_schema` and `is_tool` attributes to `fn` in-place.

    `functools.wraps` is intentionally NOT used here: the decorated function
    is the same object as the original, so `inspect.signature`, docstring,
    __name__, and async-ness are all preserved without explicit copying.
    """
    annotated: _ToolCallable = cast("_ToolCallable", fn)
    annotated.tool_schema = schema
    annotated.is_tool = True
    return annotated


@overload
def tool(fn: Callable[..., Any], **kwargs: Any) -> _ToolCallable: ...


@overload
def tool(
    fn: None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], _ToolCallable]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> _ToolCallable | Callable[[Callable[..., Any]], _ToolCallable]:
    """Decorate a callable as a tinyagent tool.

    Accepts both call forms:
      - `@tool` (no parens)
      - `@tool` (parens, no kwargs)
      - `@tool(name=...)` (parens, kwargs; kwargs are stored on the schema)

    The decorated callable keeps its original signature, return type, and
    (for async) its coroutine semantics; the decorator only attaches
    `tool_schema` (an OpenAI-compatible JSON-Schema dict) and `is_tool=True`.
    """

    def _decorate(target: Callable[..., Any]) -> _ToolCallable:
        schema = _build_json_schema(target)
        if kwargs:
            # Expose the decorator kwargs under a reserved namespace so they
            # survive the round-trip to the LLM tool spec without polluting
            # the parameter list.
            schema["decorator_kwargs"] = dict(kwargs)
        return _attach_schema(target, schema)

    if fn is not None:
        # `@tool` (no-paren) form: fn is the decorated callable directly.
        return _decorate(fn)
    # `@tool(...)` (parens) form: return a decorator that accepts the function.
    return _decorate


def _wrap_no_exception(callable_: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a bare-Python tool so its exceptions become a string result.

    On the success path, the wrapper passes the call through to the
    underlying callable unchanged. On any `Exception` it returns the string
    ``"Error calling tool: {e}"`` so the agent loop can feed a recoverable
    error message back to the LLM instead of crashing the run.
    """

    if asyncio.iscoroutinefunction(callable_):

        @functools.wraps(callable_)
        async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return await callable_(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — bare-tool adapter; the
                # loop relies on the string-return contract, so we catch
                # broadly to keep the run alive.
                return f"Error calling tool: {exc}"

        return _async_wrapped

    @functools.wraps(callable_)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return callable_(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — see note above.
            return f"Error calling tool: {exc}"

    return _wrapped


# Truthy / falsy string spellings accepted by `_cast_argument(..., bool)`.
_BOOL_TRUE_STRINGS: frozenset[str] = frozenset({"true", "1", "yes"})
_BOOL_FALSE_STRINGS: frozenset[str] = frozenset({"false", "0", "no"})


def _coerce_bool(value: Any) -> bool:
    """Coerce a value to bool, recognising common string spellings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in _BOOL_TRUE_STRINGS:
            return True
        if lower in _BOOL_FALSE_STRINGS:
            return False
    return bool(value)


def _coerce_json_container(value: Any, target: type) -> Any:
    """Coerce a value to a list/dict, parsing JSON strings.

    Used for `list`, `dict`, and their parameterised forms (`list[X]`,
    `dict[X, Y]`). String inputs are parsed as JSON; native values are
    re-materialised via the target's constructor.
    """
    if isinstance(value, str):
        return json.loads(value)
    return target(value)


# Dispatch table for the numeric / string scalar casts: a target type
# → its corresponding coercion function. Built once at import time so
# `_cast_argument` is a flat dispatch and stays under the PLR0911
# (return-statement) budget.
_SCALAR_COERCERS: dict[type, Callable[[Any], Any]] = {
    str: str,
    int: int,
    float: float,
    bool: _coerce_bool,
}


def _cast_argument(value: Any, param_annotation: Any) -> Any:
    """Coerce a value to a primitive type declared by a tool parameter.

    Handles the primitive types the agent surfaces to the LLM:
    `str`, `int`, `float`, `bool`, `list`, `dict` (including their
    parameterised forms `list[X]`, `dict[X, Y]`). String values destined
    for `list` / `dict` are parsed as JSON. Unknown annotations return
    the value unchanged.
    """
    # Avoid clobbering explicit None on optional parameters.
    if value is None:
        return None

    # Resolve the "effective" target type. Parameterised generics
    # (`list[X]`, `dict[X, Y]`) collapse to their origin type so we
    # don't have to introspect the type args.
    target = param_annotation
    origin = get_origin(param_annotation)
    if origin in (list, dict):
        target = origin

    coercer = _SCALAR_COERCERS.get(target)
    if coercer is not None:
        return coercer(value)
    if target in (list, dict):
        return _coerce_json_container(value, target)

    # Unknown annotation: pass the value through.
    return value


# =====================================================================
# Section 10 - Example tools (shipped, importable from top-level)
# =====================================================================
# T5: shipped example tools. The module is split into:
#   - final_answer: bare termination function (NOT @tool-decorated).
#     The agent loop recognises it by name and short-circuits when the
#     LLM emits it; adding a JSON schema would route it through the
#     normal tool-dispatch path and bypass the termination branch.
#   - calculate:    @tool-decorated safe expression evaluator. Backed
#     by simpleeval.SimpleEval so the LLM cannot reach dunders, builtin
#     imports, or attribute lookups. Errors are returned as a string
#     (the loop feeds the string back to the LLM, the loop continues).
#   - http_get:     @tool-decorated async HTTP GET. Uses httpx.AsyncClient
#     and truncates oversized responses so a 1 MB page cannot blow up
#     the agent's context. Network errors are returned as a string.
#
# All three names are exported via __all__ (sections 3 and 17) and
# importable as `from tinyagent import final_answer, calculate, http_get`.

# Cap http_get response bodies at ~4KB. The exact value matches the
# SPAN_LIMITS["tool_result"] budget in §4 so a single GET round-trip
# fits in a single trace span without re-truncation.
_HTTP_GET_MAX_CHARS: int = 4096
_HTTP_GET_TRUNCATION_MARKER: str = "...[truncated]"


def final_answer(answer: str) -> str:
    """Bare termination tool: the agent loop returns ``answer`` as the run's result.

    The function body is a no-op identity: it does not evaluate, parse,
    or transform its argument. The semantic role of ``final_answer`` is
    to be the agent loop's termination signal — the loop sees the model
    call this function name and short-circuits via ``break`` (round-3
    M1, plan §8), returning ``answer`` to the caller. See §8 in the
    plan for the loop-side wiring.

    Parameters
    ----------
    answer:
        The final answer the agent should return to its caller. Treated
        as an opaque string — no eval, no formatting, no PII redaction.
        Callers that need validation (PII redaction, length caps,
        format checks) should hook the ``before_tool_execution`` /
        ``after_tool_execution`` callbacks (round-3 M4 symmetry rule).

    Returns
    -------
    The ``answer`` argument, unchanged.
    """
    return answer


# Maximum nesting depth for the simpleeval AST. The default (no limit)
# would let a prompt inject deeply-nested expressions; capping it at 32
# keeps a malicious payload from blowing up the parser while still
# leaving headroom for legitimate math.
_CALC_MAX_NESTING: int = 32
# Maximum string length for the simpleeval input. The LLM is unlikely
# to ever emit a useful expression > 1 KB; anything larger is either a
# probe or a runaway tool-call argument.
_CALC_MAX_INPUT_CHARS: int = 1024


def _eval_arithmetic(expression: str) -> str:
    """Run a single simpleeval evaluation, returning the result as a string.

    Pulled out of ``calculate`` to keep the tool wrapper under the
    PLR0912 branch budget (ruff C901). ``calculate`` is now a thin
    guard wrapper: validate the input length, call this helper, and
    return. This helper raises simpleeval's own exceptions on failure;
    the caller maps each exception class to an "Error: ..." string.
    """
    import math

    evaluator = simpleeval.SimpleEval()
    evaluator.max_node_count = 1000
    evaluator.names.update(
        {
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            "inf": math.inf,
            "nan": math.nan,
        }
    )
    evaluator.functions.update(
        {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
    )
    return str(evaluator.eval(expression, previously_parsed=None))


@tool
def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression and return the result as a string.

    The implementation uses :class:`simpleeval.SimpleEval`, which parses
    the input with the standard-library ``ast`` module and evaluates only
    a fixed allow-list of operators, names (``pi``, ``e``, ``tau``),
    and built-in functions (``abs``, ``round``, ``min``, ``max``,
    ``sum``, ``pow``). It does NOT expose Python's ``eval`` / ``exec``,
    the ``__import__`` builtin, attribute access, or method calls — a
    prompt that asks for ``__import__('os').system('rm -rf /')`` is
    rejected at parse time.

    On any error (parse error, undefined name, division by zero,
    nesting-depth exceeded, input too long, disallowed function) the
    function returns the string ``"Error: <message>"`` rather than
    raising. The agent loop feeds that string back to the LLM, which
    can then self-correct on the next turn.

    Parameters
    ----------
    expression:
        A Python-style arithmetic expression, e.g. ``"2 + 3 * 4"`` or
        ``"(1 + 2) * 3.5"``.

    Returns
    -------
    The numeric result stringified via :func:`str`. Errors are
    stringified as ``"Error: <exception message>"``.
    """
    if not isinstance(expression, str):
        return f"Error: expression must be a string, got {type(expression).__name__}"
    if len(expression) > _CALC_MAX_INPUT_CHARS:
        return (
            f"Error: expression too long "
            f"({len(expression)} > {_CALC_MAX_INPUT_CHARS} chars)"
        )
    try:
        return _eval_arithmetic(expression)
    except (
        simpleeval.FunctionNotDefined,
        simpleeval.NameNotDefined,
        simpleeval.OperatorNotDefined,
        simpleeval.InvalidExpression,
        ZeroDivisionError,
        ValueError,
        TypeError,
        SyntaxError,
    ) as exc:
        return f"Error: {exc}"
    except RecursionError as exc:
        return f"Error: expression nested too deeply ({exc})"
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        # Catch-all so a buggy simpleeval release or a future
        # exception subclass cannot crash the agent loop. The error
        # string keeps the loop recoverable.
        return f"Error: {exc}"


@tool
async def http_get(url: str, timeout: float = 10.0) -> str:
    """Fetch ``url`` over HTTP(S) and return the response body (truncated to ~4KB).

    The implementation uses :class:`httpx.AsyncClient` with the
    ``timeout`` parameter honoured as both connect and read timeout.
    The response body is truncated to :data:`_HTTP_GET_MAX_CHARS`
    characters so a 1 MB page cannot blow up the agent's context or
    the ``gen_ai.tool.result`` span attribute budget. The truncation
    marker is appended inside the budget so the returned string is
    always ``len <= _HTTP_GET_MAX_CHARS + len(marker)``.

    On any error (connection refused, DNS failure, timeout, non-2xx
    status — actually 2xx pass through; 4xx/5xx return the body as
    text so the LLM can see the error message) the function returns
    ``"Error: <message>"`` rather than raising. The agent loop feeds
    that string back to the LLM.

    Parameters
    ----------
    url:
        Absolute HTTP(S) URL to fetch.
    timeout:
        Per-operation timeout in seconds. Defaults to 10.0.

    Returns
    -------
    The response body as a string, truncated to ~4KB. Errors are
    stringified as ``"Error: <exception message>"``.
    """
    if not isinstance(url, str) or not url:
        return "Error: url must be a non-empty string"
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        return f"Error: timeout must be a positive number, got {timeout!r}"
    try:
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            response = await client.get(url)
            # We do NOT raise_for_status() — a 404 or 500 still has a
            # body the LLM can inspect on the next turn. The status
            # code is available to the loop via the response object if
            # needed; the tool contract is "return response text".
            body = response.text
    except httpx.HTTPError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001 — last-resort safety net
        return f"Error: {exc}"

    if len(body) > _HTTP_GET_MAX_CHARS:
        keep = max(0, _HTTP_GET_MAX_CHARS - len(_HTTP_GET_TRUNCATION_MARKER))
        body = body[:keep] + _HTTP_GET_TRUNCATION_MARKER
    return body


# =====================================================================
# Section 11 - MCP stdio client
# =====================================================================
# Sentinel for ``call_tool`` failures that look like "the server
# returned a CallToolResult with isError=True" — the spec's wire-level
# way of saying "this tool name does not exist".  We translate it into
# ``ToolNotFoundError`` so the agent loop can feed a descriptive string
# back to the LLM (see plan §4 error table).
_UNKNOWN_TOOL_MARKER: str = "unknown tool:"


class MCPServer:
    """An MCP server connected over stdio (plan §2 section 11, §4).

    The server is configured with a ``command`` + ``args`` + optional
    ``env`` dict.  ``connect()`` spawns the subprocess via
    ``mcp.client.stdio.stdio_client``, which internally uses
    ``start_new_session=True`` so the subprocess is its own session
    leader; ``cleanup()`` drains the exit stack and the stdio transport
    terminates the process group.  This is the *only* supported
    transport — SSE and streamable-HTTP code paths were dropped from
    upstream per plan §4.

    Typical use (preferred context-manager form):

        async with MCPServer(
            name="calc",
            command=sys.executable,
            args=["examples/calculator_mcp_stdio.py"],
        ) as srv:
            tools = await srv.list_tools()
            result = await srv.call_tool("add", {"a": 1, "b": 2})

    Or explicit connect / cleanup:

        srv = MCPServer(name="calc", command=sys.executable, args=[...])
        await srv.connect()
        try:
            ...
        finally:
            await srv.cleanup()

    The synthesised ``self.tools`` dict maps each tool name advertised
    by the server to a Python callable produced by
    ``_create_tool_function``.  The agent loop consumes this dict
    directly; MCP-specific call / schema details stay inside
    ``MCPServer``.
    """

    __slots__ = (
        "name",
        "command",
        "args",
        "env",
        "tools",
        "_exit_stack",
        "_session",
        "_read_stream",
        "_write_stream",
        "_process",
    )

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Configure a stdio MCP server.  Does NOT spawn the subprocess.

        ``args`` defaults to an empty list.  ``env`` is defensively
        copied so post-construction mutation by the caller does not
        leak into the subprocess environment.
        """
        self.name: str = name
        self.command: str = command
        self.args: list[str] = list(args) if args is not None else []
        self.env: dict[str, str] | None = dict(env) if env is not None else None
        # ``tools`` is the synthesised-callable dict the agent loop
        # reads from.  Populated by ``connect()`` (which calls
        # ``list_tools()`` and registers each tool via
        # ``_create_tool_function``).
        self.tools: dict[str, Callable[..., Any]] = {}
        # AsyncExitStack holds the ``stdio_client`` context manager so
        # ``cleanup()`` can pop it (which closes the streams and
        # terminates the subprocess group).
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._read_stream: Any = None
        self._write_stream: Any = None
        # ``_process`` is the live subprocess handle, set by
        # ``connect()``.  Exposed for the test suite (it asserts the
        # PID is in its own process group) and so ``cleanup()`` can
        # kill the whole group via ``os.killpg`` if needed.
        self._process: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Spawn the subprocess and open a ``ClientSession``.

        On success, ``self.tools`` is populated with one synthesised
        callable per advertised MCP tool.  On failure, the subprocess
        is reaped and the exception propagates — the caller should
        treat this as "the server is unavailable".

        We capture the live subprocess handle by wrapping the mcp
        library's internal ``_create_platform_compatible_process``
        for the duration of this call only.  The original is restored
        before the function returns, so the patch is invisible to
        other call sites in the same process.
        """
        from mcp.client import stdio as _mcp_stdio

        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        stack = contextlib.AsyncExitStack()
        # ``stdio_client`` returns ``(read_stream, write_stream)`` and
        # spawns the subprocess with ``start_new_session=True`` on
        # POSIX.  We hold the stack so ``cleanup()`` can drain it
        # deterministically even on error paths.
        # Wrap the mcp subprocess factory so we can capture the live
        # process handle (the mcp library does NOT expose the process
        # on the returned streams — it's a local variable in the
        # ``stdio_client`` async generator).  Restored on the way out
        # in the ``finally`` so the patch is local to this call.
        captured: dict[str, Any] = {}
        # ``_create_platform_compatible_process`` is a private symbol
        # in the mcp library; we deliberately wrap it to capture the
        # process handle.  The patch is scoped to this call only.
        original_create_process = _mcp_stdio._create_platform_compatible_process  # noqa: SLF001

        async def _capturing_create_process(*args: Any, **kwargs: Any) -> Any:
            proc = await original_create_process(*args, **kwargs)
            captured["process"] = proc
            return proc

        _mcp_stdio._create_platform_compatible_process = _capturing_create_process  # noqa: SLF001
        try:
            streams_cm = stdio_client(params)
            read_stream, write_stream = await stack.enter_async_context(streams_cm)
            session_cm = ClientSession(read_stream, write_stream)
            session = await stack.enter_async_context(session_cm)
            await session.initialize()
            # Discover the tools and synthesise a callable for each one.
            result = await session.list_tools()
            for tool in result.tools:
                fn = _create_tool_function(self, tool)
                self.tools[tool.name] = fn
        finally:
            # Always restore the original factory so a failed connect
            # doesn't leave the patch in place for subsequent calls
            # in the same process.
            _mcp_stdio._create_platform_compatible_process = original_create_process  # noqa: SLF001
        # Commit the stack only on the success path.
        self._exit_stack = stack
        self._session = session
        self._read_stream = read_stream
        self._write_stream = write_stream
        self._process = captured.get("process")

    async def cleanup(self) -> None:
        """Drain the exit stack and terminate the subprocess group.

        Idempotent — calling ``cleanup()`` after the server has already
        been torn down is a no-op.  The ``stdio_client`` transport
        handles the actual SIGTERM / SIGKILL escalation against the
        subprocess group; we drain the stack to release the streams
        and let the transport's finally-block run.
        """
        stack = self._exit_stack
        if stack is None:
            return
        # Clear state first so re-entrant ``cleanup()`` is a no-op even
        # if the stack raises.
        self._exit_stack = None
        self._session = None
        self._read_stream = None
        self._write_stream = None
        self._process = None
        self.tools = {}
        await stack.aclose()

    async def __aenter__(self) -> MCPServer:
        """``async with MCPServer(...) as srv:`` — calls ``connect()`` on enter."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _types.TracebackType | None,
    ) -> None:
        """``async with`` exit — calls ``cleanup()`` regardless of exception state."""
        await self.cleanup()

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------
    async def list_tools(self) -> list[Any]:
        """Return the list of MCP tool descriptors advertised by the server.

        Returns the raw ``mcp.types.Tool`` objects from the most recent
        ``connect()`` / ``list_tools`` round-trip.  The synthesised
        callables in ``self.tools`` are produced from this list.
        """
        session = self._session
        if session is None:
            message = "MCPServer.list_tools called before connect()"
            raise MCPConnectionError(message)
        result = await session.list_tools()
        return list(result.tools)

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Dispatch a tool call to the named MCP tool.

        Returns the first ``TextContent`` text of the server's
        ``CallToolResult`` (or the ``structuredContent`` if present).
        Raises ``ToolNotFoundError`` if ``name`` is not in the server's
        advertised tool list.  Raises ``MCPConnectionError`` on
        transport-level failures and ``MCPProtocolError`` on malformed
        frames.
        """
        session = self._session
        if session is None:
            message = f"MCPServer.call_tool({name!r}) called before connect()"
            raise MCPConnectionError(message)
        if name not in self.tools:
            # Surface unknown tool names as ``ToolNotFoundError`` so the
            # loop can feed a descriptive string back to the LLM.
            # (plan §4 error table: "tool call to unknown tool name …
            # descriptive error string to the LLM as the tool result.
            # The loop continues. ``on_error`` does NOT fire.")
            message = (
                f"tool {name!r} is not registered on server {self.name!r}; "
                f"available tools: {sorted(self.tools)}"
            )
            raise ToolNotFoundError(message)
        try:
            result = await session.call_tool(name, dict(args))
        except Exception as exc:
            raise _classify_mcp_exception(exc, name) from exc
        return _extract_call_result(result)


def _create_tool_function(server: MCPServer, tool: Any) -> Callable[..., Any]:
    """Synthesise a Python callable that round-trips to ``server.call_tool``.

    The returned function:

    - Accepts keyword arguments matching the MCP tool's
      ``inputSchema.properties`` (passed straight to ``call_tool``).
    - Carries a ``tool_schema`` attribute mirroring the MCP tool
      description so the agent loop can advertise the tool to the
      LLM in a provider-agnostic way.
    - Is awaitable (always returns a coroutine) — the agent loop
      always awaits tool results, so the synthesis returns an
      ``async def`` regardless of whether the underlying MCP tool
      is sync or async.

    The schema dict is built by copying the MCP tool's fields and
    mapping ``inputSchema`` → ``parameters`` so it matches the
    OpenAI-compatible shape produced by ``@tool`` in section 9.
    """
    # The MCP tool descriptor exposes name/description/inputSchema.
    # We mirror those into a dict the agent loop already understands.
    schema: dict[str, Any] = {
        "name": tool.name,
        "description": getattr(tool, "description", None),
        "parameters": getattr(tool, "inputSchema", None) or {"type": "object"},
    }

    @functools.wraps(_create_tool_function)
    async def _call(**kwargs: Any) -> Any:
        # ``server.call_tool`` raises ``ToolNotFoundError`` for unknown
        # names; we just propagate the exception (the loop catches
        # it and feeds a descriptive string to the LLM).
        return await server.call_tool(tool.name, kwargs)

    # Attach the schema as a public attribute so callers can introspect
    # the synthesised callable the same way they introspect an
    # ``@tool``-decorated function.
    _call.tool_schema = schema  # type: ignore[attr-defined]
    return _call


# ---------------------------------------------------------------------
# Section 11 helpers — internal extractors + error classification
# ---------------------------------------------------------------------
def _classify_mcp_exception(exc: BaseException, tool_name: str) -> AgentError:
    """Translate a low-level transport error into ``AgentError`` subclass.

    The mcp library raises a variety of exceptions on the transport
    boundary (broken pipe, EOF, JSON decode errors, etc.).  We map
    them to the two MCP error classes the agent loop knows about:

    - ``MCPConnectionError`` for connection / process death
      (``BrokenPipeError``, ``ConnectionError``, ``EOFError``).
    - ``MCPProtocolError`` for protocol-layer failures
      (``json.JSONDecodeError``, ``UnicodeDecodeError``, etc.).

    Anything else is wrapped in ``MCPConnectionError`` as the safe
    default — the server is in an unknown state, so marking it
    "broken" is the conservative choice.
    """
    message = f"MCP tool {tool_name!r} failed: {exc!r}"
    if isinstance(exc, (BrokenPipeError, ConnectionError, EOFError)):
        return MCPConnectionError(message)
    if isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError, ValueError)):
        return MCPProtocolError(message)
    # Default: treat as a connection failure (server is in an
    # unknown state — safer to mark it broken than to keep
    # dispatching into a wedged transport).
    return MCPConnectionError(message)


def _extract_call_result(result: Any) -> Any:
    """Reduce an MCP ``CallToolResult`` to a Python-friendly value.

    Strategy:
    1. If the server flagged ``isError=True`` AND the error text
       starts with the ``_UNKNOWN_TOOL_MARKER`` sentinel, raise
       ``ToolNotFoundError`` so the loop can recover.
    2. If ``structuredContent`` is set, return that dict (it is the
       server's typed return value).
    3. Otherwise, concatenate the ``TextContent.text`` fields into a
       single string.  Non-text content blocks are stringified.
    """
    is_error = bool(getattr(result, "isError", False))
    content = list(getattr(result, "content", []) or [])
    text_parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(str(getattr(block, "text", "")))
            continue
        # Non-text blocks (image, audio, resource link, embedded
        # resource) are stringified for the agent loop — the loop
        # only consumes string results.
        text_parts.append(str(block))
    text = "".join(text_parts)
    if is_error and _UNKNOWN_TOOL_MARKER in text:
        # Server says "unknown tool" — surface as ToolNotFoundError.
        # Extract the tool name from the marker if possible.
        # Format is "unknown tool: 'name'".
        raise ToolNotFoundError(
            f"tool not registered on server: {text.strip()!r}"
        )
    if is_error:
        # Other tool-level errors become MCPProtocolError so the loop
        # can mark the server broken.
        raise MCPProtocolError(
            f"MCP tool returned isError=True: {text!r}"
        )
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    return text


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


# ---------------------------------------------------------------------
# Section 13a - Pricing lookup (T9 STUB — replaced by T2)
# ---------------------------------------------------------------------
# Canonical signature: (model_id, prompt_tokens, completion_tokens,
# pricing=None, pricing_fn=None) -> float | None. The T9 stub honours the
# optional override channels so tests can drive the cost attribute; T2
# replaces this body with the longest-prefix match algorithm against
# DEFAULT_PRICING + LOCAL_PROVIDERS short-circuit. The contract
# (`float | None`) and the omit-when-None invariant (plan §0 C2 +
# cross-cutting risk #8) are LOCKED here so the T9 span writer is correct
# against the canonical rule.
def _estimate_cost(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, tuple[float, float]] | None = None,
    pricing_fn: Callable[[str], tuple[float, float] | None] | None = None,
) -> float | None:
    """Estimate USD cost for a single LLM call (CANONICAL — plan §7).

    Returns a float (USD total) when the price is known, else ``None``.
    ``None`` means "unknown price" — the caller (the span writer in
    ``_SpanGeneration.call_llm``) MUST omit the ``gen_ai.usage.cost``
    attribute when this returns ``None`` (plan §0 C2, cross-cutting risk
    #8). There is **no** ``(0.0, 0.0)`` fallback: zero is a valid *known*
    cost (free / included-credit models); only ``None`` triggers the
    omit.

    Lookup order (CANONICAL — plan §7 + §2 §14):

      1. ``pricing_fn(model_id)`` if provided — callable wins.
      2. ``pricing`` per-instance override (T13) — full replacement of the
         lookup table; if a model is not in ``pricing`` and ``pricing`` is
         non-``None``, the function returns ``None`` (no fallback to
         ``PRICING_OVERRIDE`` or ``DEFAULT_PRICING``).
      3. ``PRICING_OVERRIDE`` module-level dict (legacy T2 API). Longest-
         prefix match within it. If no match, falls back to step 4.
      4. ``DEFAULT_PRICING``. Longest-prefix match within it. If no
         match, the function returns ``None``.

    Parameters
    ----------
    model_id:
        Provider-prefixed model string, e.g. ``"openai:gpt-4o-mini"``.
    prompt_tokens:
        Input tokens consumed by this call.
    completion_tokens:
        Output tokens emitted by this call.
    pricing:
        Optional full-table override (provider:model -> per-1M tuple).
        When non-``None``, this is a full replacement of the lookup
        table for this call (per plan §7 "``pricing or DEFAULT_PRICING``").
        Used by T13 to plumb the per-instance ``AgentConfig.pricing``
        override through to the span writer. Wins over ``PRICING_OVERRIDE``
        and ``DEFAULT_PRICING``.
    pricing_fn:
        Optional per-call callable returning ``(input_per_1m,
        output_per_1m)`` or ``None``. Highest precedence.

    Returns
    -------
    USD total cost as float, or ``None`` if the price is unknown.
    """
    if pricing_fn is not None:
        price = pricing_fn(model_id)
        if price is None:
            return None
        return _cost_from_price(prompt_tokens, completion_tokens, price)
    provider, _, _ = model_id.partition(":")
    if provider in LOCAL_PROVIDERS:
        return None
    # Per-instance override (T13): full replacement. The override REPLACES
    # both PRICING_OVERRIDE and DEFAULT_PRICING for this call — if a
    # model is not in the override, return None (the agent's table is
    # authoritative for that model).
    if pricing is not None:
        candidates = sorted(
            (k for k in pricing if model_id.startswith(k)),
            key=len,
            reverse=True,
        )
        if not candidates:
            return None
        return _cost_from_price(prompt_tokens, completion_tokens, pricing[candidates[0]])
    # Module-level PRICING_OVERRIDE (legacy T2 API) — try it first, fall
    # back to DEFAULT_PRICING on miss. ``or`` semantics: the override
    # takes precedence when present, DEFAULT_PRICING covers everything else.
    override_match = _longest_prefix_match(PRICING_OVERRIDE, model_id)
    if override_match is not None:
        return _cost_from_price(prompt_tokens, completion_tokens, override_match)
    default_match = _longest_prefix_match(DEFAULT_PRICING, model_id)
    if default_match is None:
        return None
    return _cost_from_price(prompt_tokens, completion_tokens, default_match)


def _longest_prefix_match(
    table: dict[str, tuple[float, float]],
    model_id: str,
) -> tuple[float, float] | None:
    """Return the value of the longest-key prefix-match in ``table``, or ``None``.

    Helper for ``_estimate_cost``. ``model_id`` is matched against each
    key in ``table`` via ``startswith``; the longest matching key wins.
    If no key matches, returns ``None`` so the caller can fall back to
    the next table.
    """
    candidates = sorted(
        (k for k in table if model_id.startswith(k)),
        key=len,
        reverse=True,
    )
    if not candidates:
        return None
    return table[candidates[0]]


def _cost_from_price(
    prompt_tokens: int,
    completion_tokens: int,
    price: tuple[float, float],
) -> float:
    """Convert per-1M-token pricing + token counts to a USD total.

    ``price`` is ``(input_usd_per_1m, output_usd_per_1m)``. Always
    returns a finite float; callers decide whether to write it as a
    span attribute (see ``_compute_cost_attribute``).
    """
    in_per_m, out_per_m = price
    return (prompt_tokens / 1_000_000.0) * in_per_m + (
        completion_tokens / 1_000_000.0
    ) * out_per_m


def _compute_cost_attribute(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """Return the cost-attribute value for a span, or ``None`` when price is unknown.

    Thin pass-through over ``_estimate_cost`` — exists so the span writer
    (``_SpanGeneration.call_llm``) and the test suite have a single,
    stable seam. Per plan §0 C2 + cross-cutting risk #8: callers MUST
    omit ``gen_ai.usage.cost`` from the span when this returns
    ``None``. Returning ``0.0`` is forbidden (it would make "unknown"
    indistinguishable from "actually free" in downstream dashboards).

    The ``pricing`` kwarg (T13) lets the span writer pass the per-
    instance override down to ``_estimate_cost`` so that the cost
    attribute reflects the agent's configured pricing table rather
    than the global default.
    """
    return _estimate_cost(model_id, prompt_tokens, completion_tokens, pricing=pricing)


# ---------------------------------------------------------------------
# Section 13b - Span generation (T9)
# ---------------------------------------------------------------------
class _SpanGeneration:
    """OpenTelemetry span generator for LLM and tool calls (plan §2 section 13).

    Opens ``call_llm`` and ``execute_tool`` child spans under a parent
    ``invoke_agent`` span (the parent is created by the agent loop, not
    here). For each child span, the standard semconv attributes
    (``gen_ai.operation.name``, ``gen_ai.usage.input_tokens``,
    ``gen_ai.usage.output_tokens``) plus the custom
    ``gen_ai.usage.cost`` attribute are populated up-front when the
    response is known.

    The ``gen_ai.usage.cost`` attribute is **omitted** when the model's
    price is unknown (plan §0 C2 + cross-cutting risk #8); the writer
    never emits a ``0.0`` placeholder, because that would make "unknown"
    indistinguishable from "actually free" in observability tooling.
    """

    def __init__(
        self,
        tracer: Any,
        model_id: str,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Store the tracer and the model id used in ``gen_ai.request.model``.

        Parameters
        ----------
        tracer:
            An OTel ``Tracer`` (typically obtained via
            ``_setup_tracing(name)``).
        model_id:
            Provider-prefixed model string, e.g. ``"openai:gpt-4o-mini"``.
            Stored on the instance so every span it opens carries
            ``gen_ai.request.model``.
        pricing:
            Optional per-instance pricing override (T13). When non-``None``,
            this table is consulted (longest-prefix match) by
            ``_compute_cost_attribute`` for cost estimation. ``None`` falls
            back to the module-level ``PRICING_OVERRIDE`` then
            ``DEFAULT_PRICING`` lookup chain. The override is held on the
            instance only — it is NOT copied into module-level state.
        """
        self._tracer = tracer
        self._model_id = model_id
        self._pricing_override = pricing

    def call_llm(self, response: Any) -> Any:
        """Open a ``call_llm`` span with token + cost attrs from ``response``.

        ``response`` is an any-llm ``ChatCompletion`` (or any object with
        ``.usage.prompt_tokens`` / ``.usage.completion_tokens``). The
        span attributes are populated from ``response.usage``:

        - ``gen_ai.usage.input_tokens``  -> ``response.usage.prompt_tokens``
        - ``gen_ai.usage.output_tokens`` -> ``response.usage.completion_tokens``
        - ``gen_ai.usage.cost`` -> ``_compute_cost_attribute(...)``
          **only when non-None**.

        When ``response`` has no ``usage`` attribute, the token and
        cost attributes are simply absent (no zero-defaulting).

        Cost estimation honours the per-instance ``pricing`` override
        passed to ``__init__`` (T13 wiring). The omit-when-``None`` rule
        (§0 C2 + cross-cutting risk #8) still applies: if the price is
        unknown for this model, the cost attribute is omitted entirely
        (NEVER written as ``0.0``).

        Returns the OTel context-manager produced by
        ``tracer.start_as_current_span("call_llm", attributes=...)``.
        Use as ``with span_gen.call_llm(response): ...``.
        """
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": self._model_id,
        }
        usage = getattr(response, "usage", None)
        if usage is not None:
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            attrs["gen_ai.usage.input_tokens"] = in_tok
            attrs["gen_ai.usage.output_tokens"] = out_tok
            cost = _compute_cost_attribute(
                self._model_id, in_tok, out_tok,
                pricing=self._pricing_override,
            )
            if cost is not None:
                attrs["gen_ai.usage.cost"] = cost
        return self._tracer.start_as_current_span("call_llm", attributes=attrs)

    def execute_tool(
        self,
        *,
        tool_name: str,
        args: Any,
        result: Any,
    ) -> Any:
        """Open an ``execute_tool`` span with tool name / args / result attrs.

        ``args`` and ``result`` are stringified (JSON when dict/list) and
        truncated per ``SPAN_LIMITS["tool_args"]`` /
        ``SPAN_LIMITS["tool_result"]`` so that a model returning a 1 MB
        tool payload cannot blow up the span budget.

        Returns the OTel context-manager produced by
        ``tracer.start_as_current_span("execute_tool", attributes=...)``.
        """
        args_limit = SPAN_LIMITS.get("tool_args", 4096)
        result_limit = SPAN_LIMITS.get("tool_result", 4096)
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.args": _truncate_for_span(
                _stringify_for_span(args), args_limit
            ),
            "gen_ai.tool.result": _truncate_for_span(
                _stringify_for_span(result), result_limit
            ),
        }
        return self._tracer.start_as_current_span(
            "execute_tool", attributes=attrs
        )


# ---------------------------------------------------------------------
# Section 13c - Internal string helpers for span attributes
# ---------------------------------------------------------------------
def _stringify_for_span(value: Any) -> str:
    """Coerce a Python value into a span-attribute-friendly string.

    Dicts / lists / tuples are JSON-serialised; everything else is
    stringified via ``str(...)``. ``None`` becomes the empty string.
    Used for ``gen_ai.tool.args`` and ``gen_ai.tool.result`` attrs.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)
    if value is None:
        return ""
    return str(value)


def _truncate_for_span(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, appending a marker when clipped.

    A span attribute is a finite string; arbitrary-length tool args /
    results must be clipped to keep the trace export under control. The
    marker is appended *inside* the budget so the result still satisfies
    ``len <= limit``.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(0, limit - len(_SPAN_TRUNCATION_MARKER))
    return text[:keep] + _SPAN_TRUNCATION_MARKER


# =====================================================================
# Section 14 - AgentConfig (Pydantic)
# =====================================================================
class AgentConfig(BaseModel):
    """Pydantic model holding the configuration for a single ``TinyAgent``.

    Per plan §2 section 14 + §13 T11 acceptance criteria. Field order /
    types match the canonical list:

      - ``instructions: str``     — system prompt sent on every turn.
      - ``tools: list[Callable]`` — bare-Python tools registered for the
        agent. MCP tools are passed via ``mcp_servers`` instead.
      - ``mcp_servers: list[MCPServer]`` — MCP server configs to connect
        to in ``TinyAgent.setup()``. May be empty.
      - ``model: str``            — provider-prefixed model id, e.g.
        ``"openai:gpt-4o-mini"``. Split on the first ``":"`` to derive
        the provider name for ``AnyLLM.create``.
      - ``max_turns: int``        — default ``DEFAULT_MAX_TURNS`` (=10).
      - ``keep_last_n: int``      — default ``DEFAULT_KEEP_LAST_N`` (=10).
      - ``request_timeout_s: float`` — default ``DEFAULT_REQUEST_TIMEOUT_S``
        (=120.0). Bounds the per-call LLM latency in
        ``TinyAgent.call_model`` (cross-cutting risk #14).
      - ``callbacks: CallbackRegistry | None`` — optional user-supplied
        registry. ``None`` means ``TinyAgent.__init__`` builds a fresh
        empty one (so the agent always has a valid ``_callbacks``).
      - ``pricing: dict | None`` — optional per-instance pricing table
        (``provider:model`` -> ``(input, output)`` USD per 1M tokens).
        T13: when non-``None``, the agent stores a defensive copy as
        ``self._pricing_override`` and uses it for cost estimation. The
        override is instance-local — it is NOT copied into module-level
        ``PRICING_OVERRIDE``.
      - ``pricing_override: dict | None`` — deprecated alias for
        ``pricing``. T11 originally shipped the field under this name;
        T13 renames it to ``pricing`` per plan §2 §14 while keeping the
        old name as a back-compat alias. When both are set, ``pricing``
        wins.
      - ``name: str``             — default ``"tinyagent"``. Used as the
        OTel tracer instrumentation name.
      - ``description: str``      — default ``""``. Free-form agent
        description; surfaced on the ``invoke_agent`` span (T9 / T12a).

    Validation: Pydantic enforces the type annotations. ``mcp_servers``
    is constrained to ``MCPServer`` instances (the type alias resolves
    to ``Any`` until T10 ships — Pydantic will accept any object that
    exposes the contract used by ``MCPServer.connect``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: str
    tools: list[Callable[..., Any]]
    mcp_servers: list[Any]  # MCPServer — typed as Any until T10 lands
    model: str
    max_turns: int = DEFAULT_MAX_TURNS
    keep_last_n: int = DEFAULT_KEEP_LAST_N
    callbacks: Any = None  # CallbackRegistry | None — typed as Any to avoid forward-ref churn
    pricing: dict[str, tuple[float, float]] | None = None  # T13: canonical per-instance pricing override
    pricing_override: dict[str, tuple[float, float]] | None = None  # T11 deprecated alias
    name: str = "tinyagent"
    description: str = ""
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S


# =====================================================================
# Section 15 - TinyAgent class
# =====================================================================
class TinyAgent:
    """The ReAct agent: tinyagent's core runtime.

    Per plan §2 section 15 + §13 T11 acceptance criteria.

    Construction (``__init__``) wires the agent's persistent state:
      - OTel tracer (acquired via ``_setup_tracing(config.name)``)
      - callback registry (user-supplied or a fresh ``CallbackRegistry()``)
      - the any-llm client (built via ``AnyLLM.create(provider, **kw)``)
      - the tool dict (``_clients``), seeded with ``final_answer``
      - the MCP server list (shallow-copied from config)

    ``setup()`` opens MCP connections and merges their tools into
    ``_clients`` (landed in T11; MCP support fully wired in T10).
    ``call_model()`` is the per-call LLM wrapper: it forwards
    ``**completion_params`` to ``any_llm.acompletion`` under
    ``asyncio.wait_for(timeout=request_timeout_s)`` and emits a
    ``call_llm`` span with token + cost attributes (T9's
    ``_SpanGeneration.call_llm`` is reused — T11 doesn't reimplement
    span attribute logic; the seam is locked in §2 section 13).

    The ReAct loop body (final_answer short-circuit, per-turn tool
    dispatch, tool_choice retry) lands in T12a. Sync ``run()`` lands in
    T12c. ``add_mcp_server`` is in T14.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Build the agent's persistent state from ``config``.

        Steps (plan §2 section 15):
          1. Store the config.
          2. Acquire the OTel tracer via ``_setup_tracing(config.name)``.
          3. Resolve the callback registry (user-supplied or a fresh
             ``CallbackRegistry()``).
          4. Build the any-llm client via
             ``AnyLLM.create(provider, model=model_id)``. The
             provider name is the first segment of ``config.model``
             (split on ``":"``); the model id is the remainder
             including the provider prefix (any-llm accepts both
             forms and re-splits internally).
          5. Initialise the tool dict (``_clients``) seeded with
             ``final_answer``. Other tools / MCP tools are merged in
             during ``setup()`` (T10/T14) — the constructor itself
             does NOT scan ``config.tools`` for wire-formatting; the
             tool dict at this point is only the termination tool
             because the loop has not started yet and any per-tool
             formatting lives in the loop body (T12a).
          6. Copy the MCP server list onto the agent.
        """
        self.config = config
        self._tracer = _setup_tracing(config.name)
        self._callbacks: CallbackRegistry = (
            config.callbacks if config.callbacks is not None else CallbackRegistry()
        )

        # Build the any-llm client. AnyLLM.create() returns an AnyLLM
        # instance configured for the named provider. The model id is
        # passed to ``any_llm.acompletion(model=...)`` at call time
        # (NOT to ``AnyLLM.create``) — provider clients (e.g.
        # ``AsyncOpenAI``) do not accept a ``model`` constructor kwarg.
        #
        # API-key resolution: prefer the env var named in
        # ``PROVIDER_KEY_ENV[provider]``; fall back to a placeholder
        # when the env var is unset so unit tests can construct an
        # agent without a real provider key. The provider only
        # actually USES the key at call time, so the placeholder is
        # sufficient to satisfy the constructor's verification.
        provider, _, _model_id = config.model.partition(":")
        api_key_env = PROVIDER_KEY_ENV.get(provider)
        api_key = os.getenv(api_key_env) if api_key_env else None
        if api_key is None:
            api_key = "tinyagent-construction-placeholder"
        self._llm: Any = any_llm.AnyLLM.create(provider, api_key=api_key)

        # _clients: tool-name -> callable. Seeded with final_answer per
        # plan §2 section 15 ("_clients ... includes final_answer").
        # The loop (T12a) and the user-facing tool wrappers (T4) read
        # from this dict; setup() appends MCP tools (T10/T14).
        self._clients: dict[str, Callable[..., Any]] = {
            final_answer.__name__: final_answer,
        }

        # _mcp_servers: shallow copy of config.mcp_servers. The list
        # order matches config.mcp_servers — setup() iterates in order
        # to call .connect() (T10).
        self._mcp_servers: list[Any] = list(config.mcp_servers)

        # T13: per-instance pricing override. Stored as a defensive
        # copy on the agent so caller mutations to the original
        # ``AgentConfig.pricing`` dict do NOT leak into runtime. The
        # override is instance-local: it is NOT copied into the
        # module-level ``PRICING_OVERRIDE`` dict (that global remains a
        # separate escape hatch for users who want to mutate it
        # directly). ``pricing_override`` is kept as a deprecated alias
        # for backward compatibility with T11's AgentConfig — when
        # ``pricing`` is ``None`` we fall back to it.
        pricing_source: dict[str, tuple[float, float]] | None = (
            config.pricing if config.pricing is not None else config.pricing_override
        )
        self._pricing_override: dict[str, tuple[float, float]] | None = (
            dict(pricing_source) if pricing_source is not None else None
        )

    async def setup(self) -> None:
        """Open MCP server connections and merge their tools into ``_clients``.

        Per plan §2 section 15 + §13 T11: ``setup()`` calls each MCP
        server's ``connect()`` (T10) and populates ``_clients`` with
        every synthesised tool plus ``final_answer``. ``final_answer``
        is always last in the dict (already there from ``__init__``);
        any name collision with a synthesised tool would let the MCP
        tool win, but the canonical termination tool name
        ``"final_answer"`` is reserved and no MCP server should ever
        expose a tool with the same name.

        For T11 (no MCP support yet) this is a no-op: ``config.mcp_servers``
        is empty in the unit tests, and the method must still exist
        for the loop to call it.
        """
        for server in self._mcp_servers:
            await server.connect()
            tools = await server.list_tools()
            for tool in tools:
                fn = _create_tool_function(server, tool)
                self._clients[tool.name] = fn

    async def call_model(self, **completion_params: Any) -> Any:
        """Wrap ``any_llm.acompletion`` in a timeout and emit a ``call_llm`` span.

        Per plan §2 section 15 + §13 T11 + cross-cutting risk #14.

        The flow:
          1. Schedule ``any_llm.acompletion(**completion_params)`` under
             ``asyncio.wait_for(timeout=self.config.request_timeout_s)``.
             On timeout, ``asyncio.TimeoutError`` propagates to the
             caller (the loop's exception arm wraps it in ``AgentError``
             and fires ``on_error`` — T12a).
          2. Open a ``call_llm`` span via ``_SpanGeneration.call_llm``.
             The span attributes (token counts, model id, cost) are
             populated by the seam; ``gen_ai.usage.cost`` is written
             iff ``_estimate_cost`` returns a non-None float (T9's
             omit-when-unknown rule, §0 C2 + cross-cutting risk #8).
          3. Return the response unchanged.

        ``completion_params`` must include the model string (typically
        ``config.model``); the loop body in T12a passes it explicitly.
        """
        response = await asyncio.wait_for(
            any_llm.acompletion(**completion_params),
            timeout=self.config.request_timeout_s,
        )
        span_gen = _SpanGeneration(
            self._tracer, self.config.model, pricing=self._pricing_override
        )
        with span_gen.call_llm(response):
            pass
        return response

    # ------------------------------------------------------------------
    # Tool dispatch (helper used by the loop body below)
    # ------------------------------------------------------------------
    async def _dispatch_tool(self, tool_call: Any) -> Any:
        """Invoke the tool registered under ``tool_call.function.name``.

        Resolves the tool name against ``self._clients``, parses the
        JSON-encoded argument string, and invokes the tool with the
        resulting kwargs. Sync tools are awaited via ``asyncio.to_thread``
        (so a slow blocking tool doesn't stall the event loop); async
        tools are awaited directly.

        Returns whatever the tool returns; the loop coerces the value to
        ``str`` when feeding it back to the LLM.

        Raises ``ToolNotFoundError`` when the tool name is not
        registered; ``ValueError`` when the argument JSON cannot be
        parsed; any other exception raised by the tool itself
        propagates (the loop does not mask it — T12d wires
        ``on_error``/AgentError for that path).
        """
        tool_name: str = tool_call.function.name
        if tool_name not in self._clients:
            raise ToolNotFoundError(
                f"tool {tool_name!r} is not registered; "
                f"available tools: {sorted(self._clients)}"
            )
        fn = self._clients[tool_name]
        raw_args = tool_call.function.arguments
        # The argument string from a chat-completion tool call is JSON.
        # An empty string is treated as "no arguments" so a model that
        # emits ``arguments=""`` for an arg-less tool doesn't crash.
        if raw_args:
            try:
                parsed = json.loads(raw_args)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"tool {tool_name!r}: invalid JSON arguments {raw_args!r}: {exc}"
                ) from exc
        else:
            parsed = {}
        if not isinstance(parsed, dict):
            # OpenAI tool arguments must be a JSON object. Anything else
            # is a model bug we surface rather than silently coerce.
            raise ValueError(
                f"tool {tool_name!r}: arguments must be a JSON object, got {type(parsed).__name__}"
            )
        kwargs = parsed

        if asyncio.iscoroutinefunction(fn):
            return await fn(**kwargs)
        # Sync tool — offload to a worker thread to avoid blocking the loop.
        return await asyncio.to_thread(fn, **kwargs)

    # ------------------------------------------------------------------
    # ReAct loop body — helpers (T12a — §2 section 15, §13 T12a acceptance
    # criteria). The helpers are factored out so ``run_async`` stays under
    # the ruff C901 complexity budget.
    # ------------------------------------------------------------------
    def _initial_messages(self, prompt: str) -> list[dict[str, Any]]:
        """Build the initial ``messages`` list (system + user)."""
        messages: list[dict[str, Any]] = []
        if self.config.instructions:
            messages.append(
                {"role": "system", "content": self.config.instructions}
            )
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _is_empty_response(message: Any) -> bool:
        """Return ``True`` for a response with no tool_calls and no content."""
        return not bool(message.tool_calls) and not bool(
            message.content and message.role == "assistant"
        )

    async def _capture_final_answer(
        self, tool_call: Any, turn: int
    ) -> str:
        """Run the M4-symmetric hook sequence for a ``final_answer`` tool call.

        Fires ``before_tool_execution``, parses ``answer`` from the
        arguments, fires ``after_tool_execution`` with ``tool_result`` set
        to the answer string, and returns the captured value. The
        ``break`` that follows in the caller's loop uses this return to
        decide whether to short-circuit the rest of the turn.
        """
        await self._callbacks.dispatch_async(
            "before_tool_execution",
            Context(agent=self, tool_call=tool_call, turn=turn),
        )
        answer = self._extract_final_answer_arg(tool_call)
        await self._callbacks.dispatch_async(
            "after_tool_execution",
            Context(
                agent=self,
                tool_call=tool_call,
                tool_result=answer,
                turn=turn,
            ),
        )
        return answer

    @staticmethod
    def _extract_final_answer_arg(tool_call: Any) -> str:
        """Parse the ``answer`` argument out of a ``final_answer`` tool call.

        Defensive against malformed JSON or non-dict arguments — the
        loop can recover from either and let the LLM self-correct on
        the next turn.
        """
        raw = tool_call.function.arguments
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return ""
        if not isinstance(parsed, dict):
            return str(parsed)
        return str(parsed.get("answer", ""))

    async def _execute_non_final_tool_call(
        self, tool_call: Any, turn: int
    ) -> str:
        """Run a non-``final_answer`` tool call: dispatch, then return result string.

        Returns the tool result coerced to ``str`` (the form the loop
        feeds back to the LLM as the tool-role message content). A
        ``ToolNotFoundError`` is caught and translated to a descriptive
        error string so the LLM can self-correct; ``on_error`` does NOT
        fire — the rule (d) contract from plan §8.
        """
        await self._callbacks.dispatch_async(
            "before_tool_execution",
            Context(agent=self, tool_call=tool_call, turn=turn),
        )
        try:
            result: Any = await self._dispatch_tool(tool_call)
        except ToolNotFoundError as exc:
            result = (
                f"error: {exc}; available tools: "
                f"{sorted(self._clients)}"
            )
        result_str = str(result)
        await self._callbacks.dispatch_async(
            "after_tool_execution",
            Context(
                agent=self,
                tool_call=tool_call,
                tool_result=result_str,
                turn=turn,
            ),
        )
        return result_str

    # ------------------------------------------------------------------
    # ReAct loop body (T12a — §2 section 15, §13 T12a acceptance criteria)
    # ------------------------------------------------------------------
    async def run_async(self, prompt: str, **kwargs: Any) -> str:
        """Drive the ReAct loop and return the agent's final answer.

        Loop structure (per plan §8 pseudocode + cross-cutting risks #7
        and #11):

          1. Build the initial messages list with the system prompt and
             the user's prompt.
          2. ``tool_choice_for_next`` starts as ``"required"``; ``retried_with_auto``
             tracks whether the current turn has already retried once
             under ``"auto"`` (round-3 M4).
          3. For each turn (up to ``config.max_turns``):
              a. Fire ``before_llm_call``; call ``self.call_model``.
              b. Fire ``after_llm_call``; classify the response.
              c. If empty AND under ``"required"`` AND not-yet-retried,
                 switch to ``"auto"`` and retry THIS turn.
              d. If empty after retry, raise ``AgentError``.
              e. Append the (non-empty) assistant message to ``messages``.
              f. If no tool_calls, return trailing assistant text.
              g. Iterate tool_calls. ``before_tool_execution`` /
                 ``after_tool_execution`` fire on EVERY tool call,
                 including ``final_answer`` (round-3 M4 closure).
              h. The first ``final_answer`` short-circuits via ``break``
                 (round-3 M1) — subsequent tool calls in the turn do
                 NOT execute.
              i. Non-``final_answer`` tool calls: dispatch via
                 ``_dispatch_tool``; on ``ToolNotFoundError`` feed a
                 descriptive error string to the LLM (recoverable, no
                 ``on_error`` fires).
              j. Return ``final_answer_value`` when ``seen_final_answer``.
          4. If the turn budget is exhausted, raise ``AgentError``.

        The ``tool_choice`` retry is driven per turn: each new turn
        re-arms the retry counter, so a long conversation can do at
        most one retry per turn. The retry counts toward ``max_turns``
        (so the worst case is ``2 * max_turns`` LLM calls).
        """
        messages = self._initial_messages(prompt)

        # --- Per-turn tool_choice retry state (round-3 M4) ---
        tool_choice_for_next: str = "required"
        retried_with_auto: bool = False

        turn: int = 0
        while turn < self.config.max_turns:
            message = await self._llm_step(
                messages, tool_choice_for_next, turn, kwargs
            )
            empty_response = self._is_empty_response(message)

            # d. Round-3 M4: retry once under "auto".
            if (
                empty_response
                and tool_choice_for_next == "required"
                and not retried_with_auto
            ):
                retried_with_auto = True
                tool_choice_for_next = "auto"
                # The empty message is NOT appended; the retry result
                # is the one we keep (or the error we raise).
                continue

            # e. Empty after retry (or directly empty under auto): bail.
            if empty_response:
                _empty_msg = (
                    "model returned no tool calls and no assistant text under "
                    "tool_choice=required (retried once with tool_choice=auto)"
                )
                raise AgentError(_empty_msg)

            # f. Successful response — reset retry state for the next turn.
            tool_choice_for_next = "required"
            retried_with_auto = False
            messages.append(message.model_dump())

            # g. Trailing-text fallback (no tool_calls).
            if not message.tool_calls:
                return str(message.content or "")

            # h+i. Iterate tool_calls. The first ``final_answer`` short-circuits.
            final_value = await self._run_tool_calls(message, messages, turn)
            if final_value is not None:
                return final_value

            # Pruning is owned by T12b (pair-preserving). T12a leaves the
            # messages list intact end-of-turn so tests can assert on the
            # exact construction without surprise mutation.

            turn += 1

        # Out of turns.
        _max_turns_msg = (
            f"max_turns={self.config.max_turns} exceeded without a final_answer"
        )
        raise AgentError(_max_turns_msg)

    async def _llm_step(
        self,
        messages: list[dict[str, Any]],
        tool_choice: str,
        turn: int,
        completion_kwargs: dict[str, Any],
    ) -> Any:
        """Run one LLM step: dispatch hooks around ``self.call_model``.

        Returns ``response.choices[0].message``. The caller appends to
        ``messages`` and decides what to do with the result.
        """
        llm_ctx = Context(agent=self, turn=turn)
        await self._callbacks.dispatch_async("before_llm_call", llm_ctx)
        response = await self.call_model(
            model=self.config.model,
            messages=list(messages),
            tool_choice=tool_choice,
            **completion_kwargs,
        )
        llm_ctx.message = response
        await self._callbacks.dispatch_async("after_llm_call", llm_ctx)
        return response.choices[0].message

    async def _run_tool_calls(
        self,
        message: Any,
        messages: list[dict[str, Any]],
        turn: int,
    ) -> str | None:
        """Iterate ``message.tool_calls``; return ``final_answer`` or ``None``.

        Appends tool-role messages to ``messages`` in place for every
        non-``final_answer`` call. The first ``final_answer`` short-
        circuits via ``break`` (round-3 M1); its captured value is
        returned. A ``None`` return means "no final_answer this turn;
        keep iterating".
        """
        seen_final_answer = False
        final_answer_value: str | None = None

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            # M4 symmetry: BOTH hooks fire on final_answer (no carve-out).
            if tool_name == "final_answer" and not seen_final_answer:
                final_answer_value = await self._capture_final_answer(
                    tool_call, turn
                )
                seen_final_answer = True
                break  # <-- round-3 M1: short-circuit the rest of the turn

            # Defensive branch: a second ``final_answer`` in the same turn
            # after we've already captured one. Unreachable in normal flow
            # thanks to the ``break`` above, kept for test synthesis.
            if tool_name == "final_answer":
                warnings.warn(
                    f"model emitted a second final_answer in the same turn; "
                    f"skipping tool_call_id={tool_call.id}",
                    stacklevel=2,
                )
                continue

            # Non-final_answer tool call.
            result_str = await self._execute_non_final_tool_call(
                tool_call, turn
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                }
            )

        if seen_final_answer:
            return final_answer_value or ""
        return None

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
