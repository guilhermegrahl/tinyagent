# tinyagent — Implementation Plan

> Run: `run-20260706-120000-abc123`
> Spec: `clarified-requirements.md` (locked) · Research: `research.md`
> Revision: round 3 (final, post peer review)

---

## 0. Conflict Resolutions

### C1. Callback hook set — **Option A: collapse to canonical 5, add `on_error`**

**Decision.** Drop the two agent-level brackets (`before_agent_invocation`, `after_agent_invocation`) and add an `on_error` hook. Final set: `before_llm_call`, `after_llm_call`, `before_tool_execution`, `after_tool_execution`, `on_error`. (Upstream has 6; spec asks for 5.)

**Trade-off rejected.** Option B (keep upstream 6 + add `on_error`) preserves upstream semantics at the cost of spec drift. Option C (keep upstream 6, no `on_error`) silently drops `on_error`, making guardrail-as-callback require a try/except around `agent.run()`.

**Justification.** Single-file runtime + spec-driven API = Option A. `on_error` lets users implement dead-letter / logging / circuit-breaker handlers without external try/except. The agent-level brackets are weaker than OTel spans (which we already emit as `invoke_agent`) — duplicate bracket coverage is redundant.

**Anchor.** `research.md` §D, §I; conflict C1.

**Symmetry rule.** Each of the 5 hooks fires on **every** matching event in the loop. No "first iteration only" special cases. (Resolves peer-review issue M7.)

### C2. OTel cost attribute — **Option (b): ship standard `gen_ai.usage.input_tokens`/`output_tokens` AND custom `gen_ai.usage.cost`**

**Decision.** Write all three attributes on every `call_llm` span:
- `gen_ai.usage.input_tokens` — semconv standard
- `gen_ai.usage.output_tokens` — semconv standard
- `gen_ai.usage.cost` — **custom, non-standard**, USD total (input + output), kept as a single rolled-up number

**Pricing rule (CANONICAL — resolves M6 round-2).** `_estimate_cost` returns `float | None`. `gen_ai.usage.cost` is written **iff** the returned value is non-`None`. Local providers (`LOCAL_PROVIDERS`) and unknown models both return `None` (no `(0.0, 0.0)` fallback). `AgentTrace.cost` roll-up sums only the spans that have the attribute; absence is treated as "unknown", never as `$0`. Documented in §2 (section 5), §7 (algorithm), §13 T13 (test), and §13 cross-cutting risks #8.

Document the deviation in README. If a strict OTel collector drops unknown attributes, cost still shows up in the returned `AgentTrace.cost` roll-up.

**Trade-off rejected.** (a) Drop cost from spans entirely (loses the only way to discover cost in standard observability back-ends). (c) Drop custom cost entirely (spec explicitly names it).

**Justification.** Cost is a single USD number per LLM span; splitting it into input/output costs requires a pricing table lookup at runtime and the per-direction breakdown is over-engineering for v0.1.0. The community has not standardized cost — chosing `gen_ai.usage.cost` is a forward bet and easy to rename later.

**Anchor.** `research.md` §G, conflict C2.

### C3. `final_answer` hook symmetry — **Option (a): fire BOTH `before_tool_execution` AND `after_tool_execution`**

**Decision (resolves M4 round-2).** The `final_answer` tool call fires **both** `before_tool_execution` and `after_tool_execution`, with **no carve-out** from the symmetry rule. `before_tool_execution` sees the raw `tool_call` (args not yet parsed); `after_tool_execution` sees `ctx.tool_result` set to the captured answer string. The loop's termination logic (set `seen_final_answer`, return the captured value) runs **after** `after_tool_execution`. `AgentCancel` raised from either hook still terminates the loop.

**Trade-off rejected.** Option (b) — carve `final_answer` out of `before_tool_execution` because it's a "termination signal not an external action" — is intellectually honest about the special role but introduces an asymmetric rule that surprises users registering `before_tool_execution` to inspect/log/sanitize final answers. Option (a) keeps the symmetry rule clean (every hook fires on every relevant event) and gives users a single, predictable rule: hook before any tool call, hook after, regardless of whether the call is the loop terminator.

**Anchor.** Peer-review M4 (round-1 partial fix + round-2 new major). See §5 (hook table), §8 (pseudocode), §13 T12a (acceptance criteria), §13 cross-cutting risks #7.

### C4. Integration-test skipif — **per-scenario markers via `conftest.py` (resolves M10 round-2 + structural minor)**

**Decision.** The integration suite's skipif is split into a `tests/integration/conftest.py` helper and per-scenario markers on each test function. Two markers are exposed:
- `PROVIDER_ENV_SKIPIF` — skips a scenario if `ANY_LLM_TEST_MODEL` is unset OR the current provider's required env vars are missing.
- `ANY_LLM_MODEL_SKIPIF` — skips a scenario if only `ANY_LLM_TEST_MODEL` is unset.

The `test_on_error_real_failure_mode` scenario uses `ANY_LLM_MODEL_SKIPIF` (NOT `PROVIDER_ENV_SKIPIF`) because it intentionally uses an invalid model id and has no provider-key requirement. All other scenarios use `PROVIDER_ENV_SKIPIF`.

**Hard rule.** `PROVIDER_KEY_ENV` is ALWAYS accessed via `.get(provider, ())`. The previous `[PROVIDER]` subscript KeyError'd for ollama and vertex — fixed by using `.get()`.

**Trade-off rejected.** Module-level `pytest.skip(..., allow_module_level=True)` runs at collection time and skips the whole module, defeating per-scenario skipif for tests with different env requirements.

**Anchor.** Peer-review M10 (round-1 partial fix + round-2 new major). See §11 (conftest + scenarios), §13 T16, §13 cross-cutting risks #10.

### C5. CallbackRegistry storage model — **dict-backed `register_*` methods, no attribute storage (resolves round-3 M3)**

**Decision.** `CallbackRegistry` uses **one** storage model, end-to-end:

- Internal storage: `self._hooks: dict[str, list[Callable]]` keyed by canonical hook name. The `_loop` slot is set by `run()` for the sync bridge (§5). `__slots__ = ("_hooks", "_loop")` stays — the dict lives on the heap, the slots pin the registry.
- User-facing API: five `register_*` methods (`register_before_llm_call`, `register_after_llm_call`, `register_before_tool_execution`, `register_after_tool_execution`, `register_on_error`) — each does `self._hooks[name].append(fn)`. Append-lists semantics, additive registration.
- Dispatch internals: `dispatch_sync` and `dispatch_async` iterate `self._hooks.get(name, ())` — direct dict lookup, NEVER `getattr(self, name)`.
- Old `cb.before_llm_call.append(fn)` API form is **dropped**. Users who want list-style bulk registration call `register_*` per function. This keeps the validation surface tiny (one method per hook name, name is statically known) and makes the sync/async dispatch metadata (which lives on the hook list, not on a per-attribute cache) trivial to maintain.

**Trade-off rejected.** The attribute-storage form (`cb.before_llm_call.append(fn)` with `__slots__ = ("_loop", "before_llm_call", "after_llm_call", ...)` and `getattr(self, name)` dispatch) is shorter at the call site but loses the symmetry between user-facing methods (which become magic attribute writes) and the internal register_* contract. The dict form is one canonical mechanism: `register_*` writes to `self._hooks[name]`; dispatch reads from `self._hooks[name]`; tests assert both directions. No attribute lookup surprises.

**Anchor.** Round-3 peer-review M3 (api-consistency, lines 143-147 vs §5). See §2 section 7, §5 (CB), §6/T6 (`tests/test_callback_registry.py`), §11 (test inventory), §13 cross-cutting risks #12, README example in T15c.

### C6. Package layout — **flat layout, `tinyagent.py` at repo root, no `src/` directory (resolves round-3 minor m5 + m7)**

**Decision.** `tinyagent.py` lives at the **repo root**, NOT under `src/`. The setuptools config is `[tool.setuptools] py-modules = ["tinyagent"]` which by default looks for `tinyagent.py` at the project root. There is no `src/__init__.py`, no `src/tinyagent/__init__.py`, and no `package-dir` mapping. Single source of truth, matches the "literally one Python file at the heart of a pip-installable package" spec wording, and removes the §3 contradiction that listed `src/__init__.py` (which would shadow the flat-module install).

**Trade-off rejected.** Keeping `src/tinyagent.py` + adding `[tool.setuptools] package-dir = {"": "src"}` is one valid alternative (many pyproject canonical projects use it) but adds a moving part for no benefit at this scale. The flat layout is canonical for tiny single-file packages (see e.g. boltons, requests' shim, halo) and tests install faster.

**Anchor.** Round-3 peer-review minor `m5_src_init` (T1 file list) + `m7_package_dir` (pyproject.toml). See §3 (package layout), §13 T1 (drop `src/__init__.py` from files), §13 T15a (pyproject.toml finalization).

---

## 1. Anchor files to lift from upstream (priority)

| Priority | Upstream file | Lines | Use |
|---|---|---|---|
| P0 | `src/tinyagent/agent.py` | 227-229, 265-282, 389-592, 624-625 | `final_answer` shim, any-llm init, run loops, `call_model` |
| P0 | `src/tinyagent/tools/mcp/mcp_client.py` | 51-86, 88-133, 135-201 | stdio-only `connect()`, `list_tools()`, `_create_tool_function()` |
| P0 | `src/tinyagent/tools/wrappers.py` | 58-82 | `_wrap_tools` (callable -> JSON schema) |
| P0 | `src/tinyagent/callbacks/base.py` | 10-47 | Re-shape to 5 hooks — drop agent brackets, add `on_error` |
| P0 | `src/tinyagent/callbacks/wrapper.py` | 29-97 | Monkey-patch dispatcher around `call_model` + `call_tool` |
| P1 | `src/tinyagent/config.py` | 11-40 | `MCPStdio` only |
| P1 | `src/tinyagent/tracing/agent_trace.py` | 79-205, 286-316 | `AgentTrace`, `AgentSpan`, roll-up props |
| P1 | `src/tinyagent/tracing/attributes.py` | full | Semconv constants; deprecate `input_cost`/`output_cost` |
| P1 | `src/tinyagent/callbacks/span_generation.py` | 22-193 | Open `call_llm`/`execute_tool` spans, extract tokens (fix line 180 redundant assignment on adoption) |
| P1 | `src/tinyagent/callbacks/span_end.py` | 6-19 | End span + append to trace |
| P2 | `src/tinyagent/callbacks/context.py` | full | `Context` dataclass |
| DROP | `serving/a2a/`, `serving/mcp/`, `evaluation/` | all | Out of scope |
| DROP | `tools/a2a.py`, `composio.py`, `web_browsing.py`, `user_interaction.py`, `final_output.py` | all | Vendor / out of scope |

---

## 2. Single-file `tinyagent.py` outline

Sections in file order (target ~1,800–2,400 LOC, includes docstrings + type hints):

```
 1.  Module docstring (single-file attribution, Apache-2.0 notice)
 2.  Imports
     - stdlib: asyncio, contextlib, dataclasses, json, os, re, typing, uuid, warnings
     - typing extensions: TypedDict        # for ToolCall shape (§8 — round-3 minor m6)
     - third-party: any_llm, mcp (client/session/stdio), opentelemetry (api/trace only),
                    pydantic (BaseModel, Field, ConfigDict), simpleeval, httpx
 3.  __all__ (CANONICAL — see §10 for rationale; this list is authoritative):
       ["TinyAgent", "AgentConfig", "tool",
        "MCPServer", "add_mcp_server", "MCPTool",
        "CallbackRegistry", "Context", "ToolCall",
        "AgentTrace", "AgentSpan", "TokenInfo", "CostInfo",
        "AgentError", "AgentCancel", "ToolNotFoundError",
        "calculate", "http_get", "final_answer",
        "PROVIDER_KEY_ENV", "PROVIDER_EXTRA_ENV",   # test-helper exports
        "__version__"]
 4.  Constants
     - __version__ = "0.1.0"
     - DEFAULT_MAX_TURNS = 10
     - DEFAULT_KEEP_LAST_N = 10
     - DEFAULT_REQUEST_TIMEOUT_S = 120.0
     - SPAN_LIMITS (max message length in span attrs)
     - LOCAL_PROVIDERS = frozenset({"ollama"})        # always zero-priced, never write cost attr
     - PROVIDER_KEY_ENV = {                          # any-llm env var lookup (per provider)
         # Source: any-llm Each provider class declares ENV_API_KEY_NAME;
         # the base class reads it via os.getenv. Default rule is <PROVIDER>_API_KEY;
         # the exceptions below are hardcoded in any-llm and must be honored.
           "openai": "OPENAI_API_KEY",
           "anthropic": "ANTHROPIC_API_KEY",
           "mistral": "MISTRAL_API_KEY",
           "groq": "GROQ_API_KEY",
           "azure": "AZURE_API_KEY",                  # any-llm uses AZURE_API_KEY, not AZURE_OPENAI_API_KEY
           "huggingface": "HF_TOKEN",                 # any-llm uses HF_TOKEN, not HUGGINGFACE_API_KEY
           "gemini": "GEMINI_API_KEY",                # any-llm also accepts GOOGLE_API_KEY; we try GEMINI first
           # vertexai / ollama: no API key required
           #   vertex: needs GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION
           #   ollama: needs OLLAMA_HOST (or default localhost)
       }
     - PROVIDER_EXTRA_ENV = {
           "vertex": ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"),
           "gemini":  ("GOOGLE_API_KEY",),            # fallback
       }
     - GenAI attribute constants (subset from upstream attributes.py)
 5.  Pricing table — DEFAULT_PRICING (per 1M tokens, USD) for {openai, anthropic, mistral, groq};
     lookup rule (CANONICAL — single source of truth, see §7 for the full algorithm):
     longest-prefix match against pricing/pricing_fn/DEFAULT_PRICING; if no match,
     return None (omit gen_ai.usage.cost attribute). LOCAL_PROVIDERS and unknown
     models both return None — there is NO `(0.0, 0.0)` fallback.
 6.  Exceptions (CANONICAL — every exception the library raises is declared here)
     - class AgentError(Exception)                    # base; catch-all for callers
     - class AgentCancel(AgentError)                  # raised by hooks to terminate the loop
     - class ToolNotFoundError(AgentError)            # raised internally when tool name is unknown
     - class MCPConnectionError(AgentError)           # MCP subprocess death / EOF on stdin (M8)
     - class MCPProtocolError(AgentError)             # MCP invalid UTF-8 / malformed JSON-RPC (M8)
 7.  Callback Registry (CANONICAL — matches §5. Round-3 M2 + M3.)
     - class CallbackRegistry
         - __slots__ = ("_hooks", "_loop")
         - _HOOK_NAMES = ("before_llm_call", "after_llm_call",
                          "before_tool_execution", "after_tool_execution",
                          "on_error")
         - self._hooks: dict[str, list[Callable]] = {n: [] for n in _HOOK_NAMES}
         - self._loop: asyncio.AbstractEventLoop | None = None  # pinned by sync run() (M3)
         - register_before_llm_call(fn), register_after_llm_call(fn),
           register_before_tool_execution(fn), register_after_tool_execution(fn),
           register_on_error(fn)
             # Each does `self._hooks[name].append(fn)`. Append-lists semantics.
         - dispatch_sync(name, ctx) -> None       # awaits async hooks via
                                                 # asyncio.run_coroutine_threadsafe against
                                                 # the pinned event loop (see §5).
                                                 # Iterates: for hook in self._hooks[name]
         - dispatch_async(name, ctx) -> None      # awaits async hooks directly.
                                                 # Iterates: for hook in self._hooks[name]
         - async def _await_coro(self, coro) -> Any
             # pass-through used by dispatch_sync to wrap a coroutine result so
             # run_coroutine_threadsafe gets an awaitable; a bare awaitable works
             # but the explicit method makes the call site self-documenting and
             # gives a single place to add error wrapping later if needed.
         - clear() (test helper) — empties every list in self._hooks
         - Hook signature (CANONICAL — single source of truth, matches §5):
             Sync hook:   Callable[[Context], None]
             Async hook:  Callable[[Context], Awaitable[None]]
             Return value is DISCARDED by dispatch. No **kwargs. No Context return.
             The §2 outline's earlier "(ctx, **kwargs) -> Context | None" wording
             was leftover upstream boilerplate (research §D); dispatch never
             supplies kwargs and discards the return value, so the contract
             collapses to "positional ctx; fire-and-forget".
 8.  Context type (CANONICAL — matches §5. Round-3 minor m6 resolved.)
     - class ToolCall(TypedDict):
         # Shape mirrors the relevant OpenAI tool-call message part. Any-llm
         # returns ChatCompletionMessageToolCall objects with .function.{name,
         # arguments}; we adopt a TypedDict for the ctx.tool_call shape and
         # store the dict (already produced by .model_dump() in the loop).
             id: str
             type: Literal["function"]
             function: dict  # {name: str, arguments: str (JSON-encoded)}
     - Context = SimpleNamespace(span, trace, agent,
                                 tool_call: ToolCall | None,
                                 tool_result: Any,
                                 message: Any,
                                 error: BaseException | None,
                                 turn: int)
     Used by all 5 hooks. Each hook populates only the relevant fields;
     unused fields remain None. ToolCall is declared as a TypedDict so the
     public import (`from tinyagent import ToolCall`) gives users a type hint
     surface — it's added to __all__.
 9.  Tool helpers
     - @tool decorator — wraps a sync/async callable, captures JSON schema via inspect.signature
     - _wrap_no_exception(callable) — bare-Python-tool adapter
     - _cast_argument(value, param_annotation) — restored from upstream utils/cast.py
10.  Example tools (shipped, importable from top-level)
     - def final_answer(answer: str) -> str  — bare function, no eval
     - @tool def calculate(expression: str) -> str  — uses simpleeval.SimpleEval with safe operators
     - @tool async def http_get(url: str, timeout: float = 10.0) -> str  — httpx.AsyncClient
11.  MCP stdio client
     - class MCPServer (Pydantic-like config: name, command, args, env)
         - async connect() — launches stdio_client, enters ClientSession, lists tools
         - async list_tools() -> List[MCPTool]
         - async call_tool(name, args) -> Any        # raises ToolNotFoundError on missing tool
         - async cleanup() — drains exit_stack, kills subprocess
         - __aenter__ / __aexit__                    # context-manager form
         - Popen spawned with start_new_session=True so we can kill the process group on cancel
     - _create_tool_function(server, tool: MCPTool) -> Callable  — synthesizes callable + schema
12.  AgentTrace / AgentSpan / TokenInfo / CostInfo  (subset of upstream tracing/agent_trace.py)
     - AgentSpan: name, attributes dict, kind, context_token, parent_id, start/end times
     - TokenInfo: input_tokens, output_tokens
     - CostInfo: input_cost_usd, output_cost_usd, total_cost_usd
     - AgentTrace: spans list, tokens cached_property, cost cached_property
13.  OpenTelemetry setup
     - _setup_tracing(name="tinyagent") -> Tracer
       - IDEMPOTENT and PASSIVE (see §6 below)
       - Does NOT call trace.set_tracer_provider(...) by default
       - Acquires tracer via trace.get_tracer(name) only
       - No exporter wiring in the library — user configures their own provider
14.  AgentConfig (Pydantic)
     - model: str
     - instructions: str
     - tools: list[Callable]
     - mcp_servers: list[MCPServer]
     - max_turns: int = 10
     - keep_last_n: int = 10
     - request_timeout_s: float = 120.0
     - callbacks: CallbackRegistry = CallbackRegistry()
     - pricing: dict[str, tuple[float, float]] | None = None
     - pricing_fn: Callable[[str], tuple[float, float] | None] | None = None
     - name: str = "tinyagent"
     - description: str | None = None
     - tool_choice_required: bool = True          # fallback to "auto" if provider rejects
15.  TinyAgent class
     - __init__(config: AgentConfig)
     - _tracer, _callbacks, _clients: dict[str, Callable], _mcp_servers: list[MCPServer]
     - async def setup() — opens MCP servers, builds tool dict, attaches final_answer to all
                            providers (not upstream's conditional)
     - async def call_model(**completion_params) — wraps any_llm.acompletion wrapped in
                            asyncio.wait_for(timeout=request_timeout_s); extracts token usage;
                            sets span attrs; computes cost (omits when None)
     - def run(prompt, **kwargs) — sync wrapper via any_llm.utils.aio.run_async_in_sync;
                            pins the loop and bridges sync/async callbacks (§5)
     - async def run_async(prompt, **kwargs):
         - opens invoke_agent span
         - tool_choice_for_next = "required"
         - retried_with_auto = False
         - while turn < max_turns:
              fire before_llm_call(ctx)
              response = await call_model(tool_choice=tool_choice_for_next)
              fire after_llm_call(ctx)
              message = response.choices[0].message
              empty_response = (
                  not message.tool_calls
                  and not (message.content and message.role == "assistant")
              )
              # Round-3 M4 retry: empty response under "required" -> retry once with "auto".
              # Retry re-arms each turn (per-turn budget of 1 retry). Counts toward
              # max_turns via the outer iteration counter.
              # The empty response is NOT appended to messages — only the (successful
              # or finally-empty) response lands in the conversation history.
              if empty_response and tool_choice_for_next == "required" and not retried_with_auto:
                  retried_with_auto = True
                  tool_choice_for_next = "auto"
                  continue
              if empty_response:
                  # Already retried OR tool_choice was already "auto" and still empty
                  raise AgentError(
                      "model returned no tool calls and no assistant text under "
                      "tool_choice=required (retried once with tool_choice=auto)"
                  )
              tool_choice_for_next = "required"   # reset for next turn
              retried_with_auto = False
              messages.append(message.model_dump())
              # Trailing-text fallback (uncommon for tool_choice=required but possible):
              if not message.tool_calls:
                  return str(message.content)
              # Iterate ALL tool_calls; first final_answer short-circuits via break (round-3 M1).
              seen_final_answer = False
              final_answer_value: str | None = None
              for tool_call in message.tool_calls:
                  tool_name = tool_call.function.name
                  if tool_name == "final_answer" and not seen_final_answer:
                      # M4 symmetry: BOTH hooks fire.
                      fire_before_tool_execution(ctx_for(tool_call))
                      args = json.loads(tool_call.function.arguments)
                      final_answer_value = str(args.get("answer", ""))
                      fire_after_tool_execution(ctx_for(tool_call, result=final_answer_value))
                      seen_final_answer = True
                      break                         # <-- round-3 M1 fix; short-circuit
                  if tool_name == "final_answer":   # unreachable after break; defensive
                      continue
                  fire_before_tool_execution(ctx)
                  try:
                      result = await self._dispatch_tool(tool_call)
                  except ToolNotFoundError as e:
                      result = f"error: {e}; available tools: {sorted(self._clients)}"
                  except asyncio.CancelledError:
                      await self._cleanup_mcp_servers()
                      raise
                  except (MCPConnectionError, MCPProtocolError):
                      fire_on_error(ctx); raise AgentError(...) from e
                  fire_after_tool_execution(ctx_for(tool_call, result=result))
                  append_tool_message(tool_call.id, str(result))
              if seen_final_answer:
                  close_invoke_agent_span(status=OK)
                  return final_answer_value
              prune messages with _prune_messages_keeping_pairs()
         - on exception (loop-top): fire on_error(ctx); re-raise wrapped in AgentError
     - _prune_messages_keeping_pairs(messages, keep_last_n) — see §9
     - _dispatch_tool(self, tool_call) -> Any
         # resolves tool_call.function.name against self._clients and invokes
         # the registered callable with parsed args. Raises:
         #   - ToolNotFoundError   when no client is registered for the name
         #   - MCPConnectionError  when the backing MCP server is broken (M8)
         #   - MCPProtocolError    when the MCP server returns a malformed frame
         #   - asyncio.CancelledError on cancel (re-raised after cleanup)
         # The loop body in §8 catches ToolNotFoundError and feeds a descriptive
         # string back to the LLM (recoverable, no on_error). MCP*Error and
         # CancelledError propagate to the loop's exception arm.
     - _estimate_cost(usage, model_str) -> float | None
         # CANONICAL return type: `float | None`. None means "unknown price";
         # the span writer omits gen_ai.usage.cost when None. There is NO
         # `(0.0, 0.0)` fallback. See §7 for the full algorithm.
     - async def cleanup_async()
     - async def add_mcp_server(server: MCPServer) -> AsyncContextManager[List[Callable]]
       — public method registered in __all__
     (Note: `_return_final_answer` is NOT a method on the class — the §8
      pseudocode inlines the answer capture directly to keep the loop body
      linear. The earlier round-1 declaration was dead code and is removed.)
16.  Module-level: tinyagent.io logger, basicConfig on import
17.  Footer: __all__ re-affirmation; type-check with TYPE_CHECKING guards
```

---

## 3. Package layout (single-file at the core)

```
/Users/guilhermegrahl/tinyagent/
├── LICENSE                          # Apache-2.0 verbatim (Section 1-9 + Appendix),
│                                    # copyright "2026 Mozilla.ai" + "2026 tinyagent authors"
├── NOTICE                           # "This product includes software developed at
│                                    # Mozilla.ai (https://mozilla.ai/)" — customary, not §4 mandate
├── README.md                        # install, config, tool registration, MCP stdio,
│                                    # tracing, callbacks, end-to-end example
├── pyproject.toml                   # [project] name="tinyagent", version="0.1.0",
│                                    # dependencies pinned, license, python=>=3.11
│                                    # [tool.setuptools] py-modules=["tinyagent"]
├── tinyagent.py                     # the one runtime file (~2000 LOC) — at repo root
├── tests/
│   ├── conftest.py                  # fixtures: tmp config, ANY_LLM_TEST_MODEL skipif
│   ├── test_imports.py
│   ├── test_callback_registry.py    # sync/async dispatch via pinned loop, error propagation
│   ├── test_tool_decorator.py       # schema extraction, sync+async callables
│   ├── test_pricing.py              # default table, longest-prefix, override, None semantics
│   ├── test_prune.py                # keep-last-N PAIRED semantics
│   ├── test_final_answer.py         # termination, multiple final_answer, first-wins,
│   │                                # break short-circuits remaining tool calls
│   ├── test_tool_choice_retry.py    # empty tool_calls under required -> retry auto once
│   ├── test_example_tools.py        # calculate via simpleeval, http_get mocked
│   ├── test_exceptions.py           # AgentError/AgentCancel/ToolNotFoundError
│   ├── test_mcp_stdio.py            # spawn in-process stdio server; subprocess lifecycle;
│   │                                # tool_not_found; cancel cleanup
│   ├── test_otel.py                 # InMemorySpanExporter; asserts hierarchy + attrs;
│   │                                # cost present when known, ABSENT when unknown
│   ├── test_agent_loop.py           # loop body: terminate on final_answer, max_turns cap,
│   │                                # on_error fires; multiple final_answer in one turn;
│   │                                # unknown tool returns string to LLM; tool_choice fallback
│   ├── test_agent_loop_sync.py      # sync run() via pinned loop; async hook bridged
│   ├── test_request_timeout.py      # asyncio.TimeoutError surfaces via on_error
│   └── test_examples_run.py         # smoke-run each examples/*.py against mocked LLM/MCP
├── tests/integration/
│   ├── conftest.py                  # integration skipif marker; env-var resolution helper
│   └── test_e2e_anyllm.py           # parametrized: calculator-mcp-stdio, http_get,
│                                    # callbacks, OTel with InMemorySpanExporter, on_error;
│                                    # skipif per-provider env-var requirements
├── examples/
│   ├── calculator_mcp_stdio.py      # runnable stdio MCP server (calc tool) — used in README
│   ├── http_demo.py                 # http_get + calculate + final_answer demo
│   └── tracing_otlp.py              # OTel env-var demo (user wires their own provider)
├── docs/
│   └── decisions.md                 # C1-C6 from §0 + research-C1..C10 archived for traceability
└── .gitignore / .harness/           # existing
```

`pyproject.toml` package config uses **flat-module** install:

```toml
[tool.setuptools]
py-modules = ["tinyagent"]
# tinyagent.py is at the project root (NOT under src/). setuptools looks here
# by default for py-modules entries — no package-dir mapping needed.
# Resolves round-3 minor m7. See §0 C6.
```

There is NO `src/` directory, NO `src/__init__.py`, NO `src/tinyagent/` package tree. This matches the spec ("literally one Python file at the heart of a pip-installable package") and removes the original m5 contradiction where T1 listed `src/__init__.py` against a `py-modules` install (the package would shadow the module).

Tool references source via `from tinyagent import ...` both in dev and installed form.

**Dependency surface** (tightened to avoid research C5 and C6):

```
any-llm-sdk>=1.16,<1.20      # tighter pin than upstream (C5)
mcp==1.28.1                  # exact pin per risk C6
opentelemetry-api            # we only need api, not sdk (see §6)
simpleeval                   # for safe calculate tool (replaces eval)
httpx                        # for http_get tool
pydantic>=2
# opentelemetry-sdk, opentelemetry-exporter-otlp-proto-{grpc,http}
# are NOT declared; users install what their exporter needs (see §6)
```

---

## 4. MCP stdio-only strategy

**Drop from upstream.**
- `MCPSse` and `MCPStreamableHttp` Pydantic classes
- `mcp.client.sse.sse_client`, `mcp.client.streamable_http.streamablehttp_client` imports
- The two `isinstance(self.config, MCPSse/MCPStreamableHttp)` branches in `MCPClient.connect()` (upstream `mcp_client.py:51-86`)

**Keep.**
- `MCPStdio` config shape: `name`, `command`, `args: list[str]`, `env: dict[str, str]`
- The stdio branch body (StdioServerParameters → `stdio_client` → `ClientSession.initialize()`)
- `list_tools()` / `call_tool()` / `_create_tool_function()` synthesis

**Simplification.** Rename to `MCPServer` (per spec), single class. No abstract dispatch — `connect()` IS the stdio path. Constructor takes `command: str`, `args: list[str] = []`, `env: dict[str, str] | None = None`, `name: str | None = None`. Public API entry: `agent.add_mcp_server(server)` returns an async context manager that yields the synthesised tool list.

### Error handling (per peer-review issue M8)

| Failure | Behavior |
|---|---|
| Subprocess spawn fails (executable not found, permission denied) | `AgentError` raised during `setup()` before any LLM call. `on_error` does not fire (no span context yet). Documented. |
| Subprocess dies mid-conversation (EOF on stdout, nonzero exit) | `MCPServer.call_tool()` raises `MCPConnectionError` (a subclass of `AgentError`); `on_error` fires; the server is marked **broken** for the remainder of the run; subsequent tool calls into that server return a "server unavailable" string to the LLM instead of crashing. |
| Invalid UTF-8 from server stdout (research C6 #2873) | Wrapped in `MCPProtocolError`; `on_error` fires; same broken-server marking. |
| `asyncio.CancelledError` during a `call_tool` (research C6 #2610) | Cancel the in-flight task; ensure subprocess group is killed via `start_new_session=True`; cleanup `exit_stack`; re-raise `CancelledError` after cleanup completes (never silently swallow). |
| EOF on stdin mid-call (research C6 #2678) | Treated as subprocess death: `MCPConnectionError`; broken-server marking; `on_error` fires. |
| Tool call to unknown tool name (e.g. MCP server crashed between `list_tools` and `call_tool`) | The agent's loop catches `ToolNotFoundError` from the dispatch step and feeds the LLM a string result of the form `"error: tool 'foo' is not registered; available tools: [a, b, c]"`. The loop **continues** — the LLM can self-correct on the next turn. `on_error` does NOT fire (this is a recoverable error). |

The MCP subprocess is launched with `subprocess.Popen(..., start_new_session=True)` so `os.killpg(os.getpgid(pid), SIGTERM)` works on cancel.

---

## 5. Callback registry design

### Registration API (CANONICAL — round-3 M3)

```python
from tinyagent import CallbackRegistry

cb = CallbackRegistry()
cb.register_before_llm_call(my_fn)          # appends to internal list
cb.register_before_llm_call(my_async_fn)
cb.clear()                                  # optional: empties all lists (test helper)
```

Five `register_*` methods, one per canonical hook. Each does `self._hooks[name].append(fn)`. Internally `self._hooks` is `dict[str, list[Callable]]` keyed by hook name. Append-lists semantics, additive registration, no magic attribute writes. The old attribute-style form `cb.before_llm_call.append(fn)` is **dropped** — see §0 C5 for the rationale.

### Signatures (CANONICAL — round-3 M2)

All five hooks share a uniform callable shape:

```python
Hook = Callable[[Context], None] | Callable[[Context], Awaitable[None]]
# Sync:  (ctx) -> None
# Async: (ctx) -> Awaitable[None]
# Return value is DISCARDED. No **kwargs. No Context return.
# The earlier §2 wording "(ctx, **kwargs) -> Context | None" was leftover
# upstream boilerplate (research §D); dispatch never supplies kwargs and
# discards the return value, so the contract collapses to "positional ctx;
# fire-and-forget".

# Context type (CANONICAL — see §2 for the source-of-truth declaration):
Context = SimpleNamespace(
    span: Span | None,
    trace: AgentTrace | None,
    agent: TinyAgent,
    tool_call: ToolCall | None,         # ToolCall is a TypedDict (§2 section 8)
    tool_result: Any,
    message: Any,
    error: BaseException | None,
    turn: int,
)
```

| Hook | Receives `Context` field | Fires on |
|---|---|---|
| `before_llm_call` | fresh ctx, `agent` set, `span` opening, `turn` set | **Every** LLM call iteration (no "first iteration only") |
| `after_llm_call` | ctx with `message` populated | **Every** LLM call iteration |
| `before_tool_execution` | ctx with `tool_call` populated | **Every** tool invocation, **including `final_answer`** (M4 — see §8) |
| `after_tool_execution` | ctx with `tool_result` populated | **Every** tool invocation, **including `final_answer`** (M4 — see §8) |
| `on_error` | ctx with `error` populated | Any exception escaping the loop body, after broken-server marking |

**Symmetry rule (resolves M4 + M7):** every hook fires on every relevant event, with no carve-outs for the termination tool. `final_answer` is a normal tool call from the hook system's perspective: `before_tool_execution` fires before the args are captured (so users can inspect / redact the proposed answer) and `after_tool_execution` fires after the answer is captured into `ctx.tool_result`. The loop's termination logic (set `seen_final_answer`, return the captured value) is separate from hook firing and runs **after** `after_tool_execution`. No first-iteration specials. If a user wants one-shot behaviour, they check `ctx.turn == 0` themselves.

### Sync/async dispatch (resolves M3)

The hard problem: when the user calls `agent.run(...)` (sync), `any_llm.utils.aio.run_async_in_sync` runs the inner coroutine on a worker thread that owns a private event loop. Async hooks registered against the registry will return coroutines from inside the worker-thread loop — those coroutines must be awaited against **that** loop, not the calling thread's loop.

**Rule.** The sync `run()` wrapper pins the worker-thread's loop at entry, stores it on the registry via `self._loop = asyncio.get_event_loop()` (called from inside the coroutine that `run_async_in_sync` runs), and exposes two dispatch helpers on `CallbackRegistry`:

```python
class CallbackRegistry:
    __slots__ = ("_hooks", "_loop")
    _HOOK_NAMES = ("before_llm_call", "after_llm_call",
                   "before_tool_execution", "after_tool_execution",
                   "on_error")

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable]] = {n: [] for n in self._HOOK_NAMES}
        self._loop: asyncio.AbstractEventLoop | None = None

    def register_before_llm_call(self, fn): self._hooks["before_llm_call"].append(fn)
    # ... one register_* method per hook ...

    def clear(self) -> None:
        for n in self._HOOK_NAMES:
            self._hooks[n].clear()

    def dispatch_sync(self, name: str, ctx: Context) -> None:
        """Used from sync run() — runs coroutine hooks via the pinned worker-thread loop."""
        assert self._loop is not None, (
            "CallbackRegistry.dispatch_sync called before run() pinned the loop. "
            "This is a bug; please open an issue."
        )
        for hook in self._hooks.get(name, ()):                # <-- dict lookup, NEVER getattr
            result = hook(ctx)
            if asyncio.iscoroutine(result):
                # Schedule on the pinned loop and wait for completion.
                # This blocks the worker thread; that is intentional and required
                # because sync run() is itself blocking.
                future = asyncio.run_coroutine_threadsafe(self._await_coro(result), self._loop)
                future.result()

    async def dispatch_async(self, name: str, ctx: Context) -> None:
        """Used from async run_async() — awaits coroutine hooks directly."""
        for hook in self._hooks.get(name, ()):                # <-- dict lookup, NEVER getattr
            result = hook(ctx)
            if asyncio.iscoroutine(result):
                await result
```

**Contract.** Sync hooks always work. Async hooks always work. The dispatch helper is selected at the top of `run()` / `run_async()` — there is no "log and skip" path. TDD test asserts both directions in `tests/test_agent_loop_sync.py`.

### AgentCancel semantics (resolves m12)

**Decision.** `AgentCancel` raised from **any** hook (including `before_tool_execution`) **terminates the entire loop** with `AgentCancel` propagating out of `run()` / `run_async()`. We deliberately do NOT support "skip just this tool and continue" — that mode is too easy to misuse silently. If a user wants per-tool skip, they return a sentinel from the hook and have the tool function return a string to the LLM themselves.

This matches upstream semantics and makes the abort path single-purpose.

### Error propagation (resolves m15)

- Any hook raising `AgentCancel` → loop terminates with `AgentCancel`. Caught at the loop top, NOT passed through `on_error` (the user explicitly aborted).
- Any other exception escaping the loop body → `on_error(ctx)` fires (ctx.error populated), then the exception is **re-raised unchanged** wrapped in `AgentError`. `on_error` **cannot swallow**.
- **Rationale.** Spec criterion #4 motivates callbacks as guardrails (the LLM/tool-side error path), not as exception handlers. The dead-letter / logging / circuit-breaker pattern fires `on_error` and lets the calling code decide. Users who want swallow-and-recover wrap `agent.run()` themselves; we don't make that implicit because it would hide bugs in multi-tool loops.

`on_error` is observability-only. Documented in the README with a one-liner: "callbacks observe; the agent re-raises".

---

## 6. OTel setup — **library pattern, idempotent, no exporter wiring** (resolves B1, B2)

### Decision

`tinyagent` is a **library**. Per the OpenTelemetry library-author guidance, a library must NOT call `trace.set_tracer_provider(...)` and must NOT configure exporters — those are the **host application's** responsibility. tinyagent simply acquires a tracer by name and emits spans; if no provider is configured, the spans go to a no-op tracer and the cost of tracing is zero.

```python
# tinyagent.py section 13
def _setup_tracing(name: str = "tinyagent") -> Tracer:
    """Acquire the named tracer.

    Idempotent. Does NOT call trace.set_tracer_provider(). Does NOT configure
    any exporter. The host application is responsible for installing a
    TracerProvider (e.g. via opentelemetry-instrumentation autoconfigure, or
    manually) BEFORE importing tinyagent or before calling agent.run().

    If no provider has been configured yet, this returns a NoOp tracer; the
    agent still runs correctly, just without spans emitted anywhere.
    """
    return trace.get_tracer(name)
```

`_setup_tracing` is called **once** in `TinyAgent.__init__`, NOT at module import. There is no module-level side effect. Multiple `TinyAgent` instances in the same process share whatever provider the host configured; the default no-op tracer is preserved when the user does nothing.

**Rationale (resolves B1).** `TracerProvider` is a process-wide singleton. The plan's earlier `_setup_tracing` that called `set_tracer_provider` at module import silently breaks multi-instance use and any host that already configured a provider. The library pattern (just acquire a tracer) is the OTel-correct approach.

**Rationale (resolves B2).** `_build_exporter()` was referenced but never defined; `OTEL_TRACES_EXPORTER` is honored by `opentelemetry-instrumentation` autoconfigure, not by the bare SDK. We pick **Option C**: ship **no** exporter wiring. Users wire their own provider via the standard `opentelemetry-instrumentation` autoconfigure entry point or by constructing a `TracerProvider` themselves. Documented in README and `examples/tracing_otlp.py`.

### Span hierarchy

```
invoke_agent {gen_ai.agent.name=tinyagent}
├── call_llm {gen_ai.operation.name=chat, gen_ai.request.model=openai:gpt-4o-mini,
│             gen_ai.usage.input_tokens, gen_ai.usage.output_tokens,
│             [gen_ai.usage.cost if known]}
├── execute_tool {gen_ai.operation.name=execute_tool, gen_ai.tool.name=calculate,
│                gen_ai.tool.description, gen_ai.tool.args}
├── call_llm
└── execute_tool {gen_ai.tool.name=final_answer}   <- loop exits after this
```

### Attribute naming

Standard semconv keys (from upstream `tracing/attributes.py`):
- `gen_ai.operation.name` = `"chat"` | `"execute_tool"`
- `gen_ai.agent.name`, `gen_ai.agent.description`
- `gen_ai.request.model` = full model string `"provider:model"`
- `gen_ai.tool.name`, `gen_ai.tool.description`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` — standard
- `gen_ai.usage.cost` — custom, USD float, total — **omitted when no price known**

TinyAgent-local:
- `tinyagent.version`
- `gen_ai.input.messages` (truncated to SPAN_LIMITS)
- `gen_ai.output`
- `gen_ai.tool.args`

### User-side exporter setup (README guidance)

Document a one-liner the user pastes at the top of their app:

```python
# User's app code, NOT tinyagent's.
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# Now import tinyagent and use it.
from tinyagent import TinyAgent, calculate, final_answer
```

Or point users at `opentelemetry-instrumentation` autoconfigure and the standard `OTEL_*` env vars.

---

## 7. Pricing table

### Built-in defaults (`DEFAULT_PRICING`)

Per 1M tokens, USD. Source: published provider pricing as of 2026-07-06. Updated by maintainer via PR, not runtime:

```python
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # provider:model_substring  ->  (input_usd_per_1m, output_usd_per_1m)
    "openai:gpt-4o":                 (2.50, 10.00),
    "openai:gpt-4o-mini":            (0.15, 0.60),
    "openai:gpt-4.1":                (2.00, 8.00),
    "openai:gpt-4.1-mini":           (0.40, 1.60),
    "anthropic:claude-3-5-sonnet":   (3.00, 15.00),
    "anthropic:claude-3-5-haiku":    (0.80, 4.00),
    "anthropic:claude-opus-4":       (15.0, 75.0),
    "mistral:mistral-large":         (2.0, 6.0),
    "groq:llama-3.1-70b":            (0.59, 0.79),
}

LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama"})
```

### Lookup algorithm (resolves M6, m13)

Single, unambiguous rule:

```
def _estimate_cost(model_str: str,
                   usage: CompletionUsage,
                   pricing: dict | None,
                   pricing_fn) -> float | None:
    1. If pricing_fn(model_str) returns a tuple, use it.            # callable wins
    2. Else, in `pricing or DEFAULT_PRICING`, find the longest
       prefix-match key for model_str. If found, use that tuple.
    3. Else, if provider-prefix of model_str is in LOCAL_PROVIDERS:
       return None.                                                 # omit attribute (M6: omit)
    4. Else, return None.                                            # unknown model -> omit
    5. With the tuple (input_per_m, output_per_m):
       cost = (usage.input_tokens / 1_000_000) * input_per_m
            + (usage.output_tokens / 1_000_000) * output_per_m
       return cost
```

The span writer sets `gen_ai.usage.cost` **iff** `_estimate_cost` returned a non-None float. Unknown and local-provider models get **no cost attribute**. `AgentTrace.cost` roll-up sums only the spans that have the attribute; absence is treated as "unknown", never as `$0`.

**The earlier contradiction (fall through to `(0.0, 0.0)` vs. omit) is resolved in favour of "omit" with a single canonical rule.**

### Override mechanism

Two channels, in priority order:
1. **Per-call callable:** `AgentConfig(pricing_fn=lambda m: (inp, out))` — invoked per LLM call. Highest precedence.
2. **Per-call dict:** `AgentConfig(pricing={"openai:gpt-4o": (0,0)})` — full override of the lookup table for this agent instance.
3. Built-in `DEFAULT_PRICING` is the fallback.

### `ollama` and other local providers (resolves m13)

Local providers are explicit, not a wildcard key. The `LOCAL_PROVIDERS` set names them. The lookup never accidentally matches `"ollama:0"` against a real future `ollama:0` model — if one ever ships, the maintainer adds an explicit entry to `DEFAULT_PRICING`.

---

## 8. `final_answer` handling

### Loop termination

`final_answer` is auto-attached to `self._clients` for **all** providers (upstream restricts to non-OpenAI — fork lifts that restriction per risk C8).

### Per-turn tool execution (resolves M4 + round-3 M1 + M4)

Each turn, the loop iterates **every** entry in `message.tool_calls`. The first `final_answer` short-circuits via `break` (round-3 M1 fix). At the top of each turn, the empty-response retry hook (round-3 M4) runs before iteration.

```python
# Round-3 M4: tool_choice retry state. Re-armed each turn (each turn can
# retry once). Retry consumes the current iteration's call_model call but
# the empty response is NOT appended to messages before the retry — only
# the successful (or finally-empty) response is appended.
tool_choice_for_next = "required"
retried_with_auto = False

while turn < max_turns:
    fire_before_llm_call(ctx)
    response = await call_model(tool_choice=tool_choice_for_next)
    fire_after_llm_call(ctx)
    message = response.choices[0].message

    empty_response = (
        not message.tool_calls
        and not (message.content and message.role == "assistant")
    )

    # --- Round-3 M4 retry branch ---
    if empty_response and tool_choice_for_next == "required" and not retried_with_auto:
        retried_with_auto = True
        tool_choice_for_next = "auto"
        # Do NOT append the empty message to messages; the retry result
        # is the one that lands in the conversation history.
        continue                                                # retry this turn
    if empty_response:
        raise AgentError(
            "model returned no tool calls and no assistant text under "
            "tool_choice=required (retried once with tool_choice=auto)"
        )

    # Successful response — reset retry state for the next turn
    tool_choice_for_next = "required"
    retried_with_auto = False
    messages.append(message.model_dump())

    # --- Trailing-text fallback (no tool_calls) ---
    if not message.tool_calls:
        return str(message.content)

    # --- Iterate tool_calls; first final_answer short-circuits via break ---
    seen_final_answer = False
    final_answer_value: str | None = None

    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name

        if tool_name == "final_answer" and not seen_final_answer:
            # M4 (symmetry): fires BOTH before/after_tool_execution hooks,
            # same as every other tool call. before_tool_execution lets the
            # user inspect/redact the proposed answer before it is captured
            # into final_answer_value (e.g. PII redaction, length cap, format
            # check). AgentCancel raised from either hook still terminates
            # the loop — that contract is independent of hook firing.
            before_ctx = make_ctx(tool_call=tool_call, ...)
            fire_before_tool_execution(before_ctx)
            args = json.loads(tool_call.function.arguments)
            final_answer_value = str(args.get("answer", ""))
            fire_after_tool_execution(ctx_for(tool_call, result=final_answer_value))
            seen_final_answer = True
            break                                       # <-- round-3 M1 fix

        if tool_name == "final_answer":                 # unreachable after break; defensive
            warnings.warn(
                f"model emitted a second final_answer in the same turn; "
                f"skipping tool_call_id={tool_call.id}",
                stacklevel=2,
            )
            continue

        # Non-final_answer tool call
        ctx = make_ctx(tool_call=tool_call, ...)
        fire_before_tool_execution(ctx)
        try:
            result = await self._dispatch_tool(tool_call)
        except ToolNotFoundError as e:
            # M4: feed a string result back to the LLM so it can self-correct
            result = f"error: {e}; available tools: {sorted(self._clients)}"
        except asyncio.CancelledError:
            await self._cleanup_mcp_servers()
            raise
        fire_after_tool_execution(ctx_for(tool_call, result=result))
        append_tool_message(tool_call.id, str(result))

    if seen_final_answer:
        close_invoke_agent_span(status=OK)
        return final_answer_value

    prune messages with _prune_messages_keeping_pairs()
```

**Rules (M4 round-2 + round-3 M1 + M4):**
- (a) The loop iterates **all** `tool_calls` in the turn.
- (b) If `final_answer` is one of them, take the **first** `final_answer`'s result as the answer and short-circuit the rest of the turn via `break`. **No subsequent tool call in the same turn executes** — this is the round-3 M1 fix, replacing the previous `continue` that let non-`final_answer` tool calls slip through after the first `final_answer`.
- (c) **Both `before_tool_execution` and `after_tool_execution` fire on `final_answer`** — the termination tool is not carved out from hook symmetry. `before_tool_execution` sees the raw `tool_call` (args not yet parsed); `after_tool_execution` sees `ctx.tool_result` set to the captured answer string. The captured answer is what the loop ultimately returns. Raising `AgentCancel` from either hook still terminates the loop — that contract is independent.
- (d) Tool call to a non-existent tool → `ToolNotFoundError` is raised by `_dispatch_tool`; the loop catches it and returns a descriptive error string to the LLM as the tool result. The loop **continues** (recoverable). `on_error` does NOT fire (this is a recoverable in-band signal, not an exception escaping the loop body).
- (e) Multiple `final_answer` calls in one turn → **first wins**; the `break` in rule (b) means we never visit a second `final_answer`. The defensive `if tool_name == "final_answer": warnings.warn + continue` branch is kept for unit-test scenarios that synthesize sequential `final_answer` tools without using `break`; it is unreachable in normal execution.
- (f) [Round-3 M4] If the LLM returns an empty response (no `tool_calls`, no assistant content) under `tool_choice="required"`, retry **once** with `tool_choice="auto"`. Re-arms every turn. If the retry also yields an empty response, raise `AgentError` (caught at loop top, fired through `on_error`). Retry **counts toward `max_turns`** as one LLM call within the same iteration (not a separate turn) — bounding reasoning: the worst case is `2 × max_turns` model calls, which matches the spec criterion of "retry once" semantically.

### Trailing text fallback (unchanged)

If the model emits `assistant` text **with no tool_calls** and the loop has not yet returned, accept it as a final answer and return `str(content)`. Same as upstream `agent.py:585-588`. The retry branch above also covers the edge case where a `tool_choice="required"` response **has** trailing assistant text but **no** tool calls — the retry is skipped (response is not "empty"), and the trailing-text branch returns.

---

## 9. Loop parameters

- `max_turns` default `= 10` (overridable via `AgentConfig.max_turns`)
- After each turn, prune `messages` to `system + last keep_last_n` **PAIRS** where `keep_last_n` default = 10
- `max_turns` exceeded → `AgentError("max_turns=10 exceeded")` — surfaced via `on_error`, not silently truncated.

### `request_timeout` (resolves m14)

`AgentConfig.request_timeout_s` default `120.0`. `call_model` wraps `self.llm.acompletion(...)` in `asyncio.wait_for(..., timeout=request_timeout_s)`. `asyncio.TimeoutError` is caught by the loop's exception arm, fired through `on_error`, and re-raised wrapped in `AgentError`. TDD test in `tests/test_request_timeout.py` injects a hanging coroutine and asserts the timeout fires.

### Pruning algorithm (resolves M5)

Keep-last-N must preserve **assistant-tool message pairing** for providers that enforce strict ordering (OpenAI, Anthropic, Google).

```python
def _prune_messages_keeping_pairs(messages: list[dict], keep_last_n: int) -> list[dict]:
    """Prune to system + last keep_last_n tool-pair units.

    A "unit" is one assistant message plus ALL its tool-role follow-ups.
    The system message (if present, always first) is never pruned.
    """
    if not messages:
        return messages
    has_system = messages[0].get("role") == "system"
    body = messages[1:] if has_system else list(messages)

    # Walk body from the right; collect units of (assistant, [tool, tool, ...]).
    units: list[list[dict]] = []
    i = len(body) - 1
    while i >= 0 and len(units) < keep_last_n:
        msg = body[i]
        if msg.get("role") == "tool":
            # Walk leftward collecting the assistant and any preceding tools with
            # the same tool_call_id cluster.
            tool_ids: set[str] = set()
            cluster: list[dict] = []
            j = i
            while j >= 0 and body[j].get("role") == "tool":
                cluster.append(body[j])
                tool_ids.add(body[j]["tool_call_id"])
                j -= 1
            # j now points at the assistant message or earlier; capture it too.
            if j >= 0 and body[j].get("role") == "assistant" and body[j].get("tool_calls"):
                cluster.append(body[j])
                j -= 1
            cluster.reverse()
            units.append(cluster)
            i = j
        elif msg.get("role") == "assistant":
            # assistant with no tool_calls — counts as its own unit
            units.append([msg])
            i -= 1
        else:
            # user message — keep individually, counts as a unit
            units.append([msg])
            i -= 1

    units.reverse()
    pruned_body = [m for unit in units for m in unit]
    return ([messages[0]] if has_system else []) + pruned_body
```

**Invariant.** For every tool-role message that survives pruning, its parent assistant message (with matching `tool_call_id`) also survives. The algorithm walks pairs as a unit; you cannot end up with an orphaned tool message.

**Observability.** If pruning actually dropped anything, the agent emits a debug log line: `tinyagent.prune: dropped {N} earlier units, {M} kept`. Tests assert the invariant directly with a synthetic provider that rejects malformed pairings.

---

## 10. Public API surface (final `__all__`)

```python
__all__ = [
    # core
    "TinyAgent",
    "AgentConfig",
    "tool",
    # MCP
    "MCPServer",
    "add_mcp_server",        # method-name exported for from tinyagent import add_mcp_server
                             # NOTE: it's a method on TinyAgent, but re-exported at module
                             # level for `from tinyagent import add_mcp_server` discoverability.
                             # The import resolves to TinyAgent.add_mcp_server.
    "MCPTool",               # re-exported from mcp.types for type hints
    # callbacks
    "CallbackRegistry",
    "Context",
    "ToolCall",              # TypedDict shape for ctx.tool_call (round-3 minor m6)
    # tracing
    "AgentTrace",
    "AgentSpan",
    "TokenInfo",
    "CostInfo",
    # exceptions
    "AgentError",
    "AgentCancel",
    "ToolNotFoundError",
    # example tools (shipped)
    "calculate",
    "http_get",
    "final_answer",
    # test-helper exports (so integration tests can import without private-name access)
    "PROVIDER_KEY_ENV",
    "PROVIDER_EXTRA_ENV",
    # misc
    "__version__",
]
```

Importable as a single module: `from tinyagent import TinyAgent, tool, MCPServer, ...`

### `add_mcp_server` idiom (resolves m11)

Two equivalent forms, both supported and tested:

```python
# Form A — context-manager form (preferred for multi-server sessions)
async with await agent.add_mcp_server(MCPServer(command="python", args=["calc_server.py"])) as tools:
    # tools is List[Callable]; agent._clients is now populated
    result = await agent.run_async("What is 2+2?")
# MCP server cleaned up on exit.

# Form B — explicit register / cleanup
await agent.add_mcp_server(server).__aenter__()
try:
    result = await agent.run_async("...")
finally:
    await agent.cleanup_async()
```

`add_mcp_server(server)` returns an async context manager. The double `await ... as` is the standard "async CM returning context" pattern. The method is exported in `__all__` so `from tinyagent import add_mcp_server` works for type-hint discoverability (the symbol resolves to `TinyAgent.add_mcp_server`).

---

## 11. Testing plan

### Unit tests (always run)

| Test file | Covers |
|---|---|
| `test_imports.py` | `import tinyagent; from tinyagent import TinyAgent, ...` — no name errors, `__all__` matches §10 list (including `ToolCall`) |
| `test_callback_registry.py` | `register_*` methods (round-3 M3), `dispatch_sync` via pinned loop, `dispatch_async`, AgentCancel propagation, error propagation; **dict-backed storage** (`self._hooks[name]`); negative test that the attribute-style `cb.before_llm_call.append(fn)` form raises `AttributeError` (regression against reintroducing that API) |
| `test_agent_loop_sync.py` | sync `run()` with both sync and async hooks bridged via the pinned loop; uses ONLY `register_*` API (round-3 M2/M3) |
| `test_tool_decorator.py` | sync/async callables, JSON schema extraction, defaults handling |
| `test_pricing.py` | longest-prefix match, override callable, unknown → None, local provider → None |
| `test_prune.py` | keep-last-N pairs invariant, system message preservation, empty history no-op |
| `test_final_answer.py` | string return, single final_answer terminates, multiple final_answer (first wins), BEFORE+AFTER hooks fire on final_answer (M4 symmetry), **after first final_answer, subsequent non-final_answer tool calls in the same turn are NOT executed (round-3 M1 `break` short-circuit)** |
| `test_tool_choice_retry.py` | under `tool_choice="required"` an LLM response with empty `tool_calls` and no trailing assistant text triggers a single retry with `tool_choice="auto"` (round-3 M4); if the retry also returns empty, `AgentError` is raised via `on_error`; retry re-arms every turn |
| `test_example_tools.py` | calculate uses simpleeval (no raw eval), http_get mocked, final_answer round-trip |
| `test_exceptions.py` | AgentError / AgentCancel / ToolNotFoundError / MCPConnectionError / MCPProtocolError subclassing (M8) |
| `test_mcp_stdio.py` | spawn in-process stdio MCP server; connect/list_tools/call_tool; tool_not_found string to LLM; cancel cleanup |
| `test_otel.py` | InMemorySpanExporter; assert hierarchy; cost attribute present when known, absent when unknown; no provider-set at module import |
| `test_otel_setup.py` | (T8) calling `_setup_tracing` does NOT call `set_tracer_provider`; returns NoOp tracer when none configured; idempotent |
| `test_agent_init.py` | (T11) `AgentConfig` defaults, `TinyAgent.__init__` builds the right any-llm client, `_clients` includes `final_answer`, `request_timeout_s` is honoured by `call_model` |
| `test_agent_trace.py` | (T7) `AgentTrace.tokens` / `AgentTrace.cost` roll-up — sums only spans WITH cost attr; absent spans are skipped |
| `test_agent_loop.py` | mock LLM sequences; final_answer terminates; max_turns cap; on_error fires and re-raises; AgentCancel mid-loop; tool_choice retry fallback when zero tool_calls; **MCP exception classes propagate via on_error** (M8); **first final_answer short-circuits remaining tool calls via break** (round-3 M1) |
| `test_request_timeout.py` | `asyncio.wait_for` triggers on hanging coroutine; `on_error` fires; re-raise |
| `test_examples_run.py` | smoke-run each `examples/*.py` against mocked LLM/MCP; README example uses `register_*` (round-3 M3) |

### Integration test (gated, expanded per M10)

The integration suite is split between a `conftest.py` helper and a parametrized test file. The skipif logic is **per-scenario**, not per-module — so a missing provider key skips only the scenarios that actually need that provider, not the whole file. The `test_on_error_real_failure_mode` scenario is explicitly exempted from the provider-key check because it intentionally uses an invalid model id to trigger `on_error`.

`tests/integration/conftest.py`:

```python
import os
import pytest
from tinyagent import PROVIDER_KEY_ENV, PROVIDER_EXTRA_ENV


def _resolve_provider_env() -> tuple[str, list[str]] | None:
    """Return (provider, required_env_keys) or None if ANY_LLM_TEST_MODEL is unset.

    Hard requirement: must NOT raise KeyError for providers not in
    PROVIDER_KEY_ENV (e.g. ollama, vertex). `.get(..., ())` is the single
    mechanism that prevents the KeyError the round-1 skipif hit.
    """
    model = os.getenv("ANY_LLM_TEST_MODEL")
    if not model:
        return None
    provider, _, _ = model.partition(":")
    # .get() with empty-tuple default handles ollama / vertex (no key required).
    keys = list(PROVIDER_KEY_ENV.get(provider, ()))
    extras = list(PROVIDER_EXTRA_ENV.get(provider, ()))
    # Filter out any falsy entries (defensive — keeps a missing/None sentinel
    # from short-circuiting the env-var check).
    return provider, [k for k in keys + extras if k]


def _skipif_missing_env() -> pytest.MarkDecorator:
    """Build a per-scenario skipif marker from the current env."""
    resolved = _resolve_provider_env()
    if resolved is None:
        return pytest.mark.skipif(True, reason="ANY_LLM_TEST_MODEL not set")
    provider, required = resolved
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        return pytest.mark.skipif(True, reason=f"{provider} requires env vars: {missing}")
    return pytest.mark.skipif(False, reason="")


# Pre-computed markers — each scenario imports the one it needs.
PROVIDER_ENV_SKIPIF = _skipif_missing_env()
ANY_LLM_MODEL_SKIPIF = pytest.mark.skipif(
    not os.getenv("ANY_LLM_TEST_MODEL"),
    reason="set ANY_LLM_TEST_MODEL=provider:model to run",
)
```

`tests/integration/test_e2e_anyllm.py`:

```python
import os
import pytest
from .conftest import PROVIDER_ENV_SKIPIF, ANY_LLM_MODEL_SKIPIF


@PROVIDER_ENV_SKIPIF
def test_calculator_then_final_answer():
    """Bare calculate + final_answer end-to-end on the real provider."""


@PROVIDER_ENV_SKIPIF
def test_calculator_mcp_stdio_via_subprocess():
    """Spawn examples/calculator_mcp_stdio.py; verify tool list + call_tool."""


@PROVIDER_ENV_SKIPIF
def test_http_get_chain():
    """http_get hits a real network endpoint (also skipped without network)."""


@PROVIDER_ENV_SKIPIF
def test_callbacks_across_loop():
    """before/after hooks fire on every iteration; on_error fires on injected failure."""


@PROVIDER_ENV_SKIPIF
def test_otel_real_exporter():
    """Wire InMemorySpanExporter to a real TracerProvider; assert attributes including cost."""


@ANY_LLM_MODEL_SKIPIF          # <-- NOT PROVIDER_ENV_SKIPIF: intentionally
def test_on_error_real_failure_mode():    # uses an invalid model id, no key needed
    """Pass a model string any-llm will reject; assert on_error fires."""
```

**Why per-scenario skipif (resolves M10 + the structural-minor at §11 line 745-754).** A module-level `pytest.skip(..., allow_module_level=True)` runs at *collection* time, before any test is parametrized. That means a missing provider key would skip the entire module — including `test_on_error_real_failure_mode`, which uses an intentionally invalid model id and has no key requirement. The fixture-based per-scenario markers above fix both bugs: (a) `PROVIDER_KEY_ENV.get(provider, ())` ensures no KeyError for ollama / vertex; (b) the per-scenario `@PROVIDER_ENV_SKIPIF` / `@ANY_LLM_MODEL_SKIPIF` markers let pytest skip individual scenarios without dragging siblings.

Scenarios (parametrized so each scenario is independently skippable):

| Scenario | Skipif marker | Covers |
|---|---|---|
| `test_calculator_then_final_answer` | `PROVIDER_ENV_SKIPIF` | bare `calculate` + `final_answer` |
| `test_calculator_mcp_stdio_via_subprocess` | `PROVIDER_ENV_SKIPIF` | spawn `examples/calculator_mcp_stdio.py` subprocess; tool list loaded; call_tool round-trip |
| `test_http_get_chain` | `PROVIDER_ENV_SKIPIF` | `http_get` real network call (also skipped without network) |
| `test_callbacks_across_loop` | `PROVIDER_ENV_SKIPIF` | before/after hooks fire on every iteration; `on_error` fires on injected failure |
| `test_otel_real_exporter` | `PROVIDER_ENV_SKIPIF` | wire `InMemorySpanExporter` to a real `TracerProvider`; assert attributes including cost |
| `test_on_error_real_failure_mode` | `ANY_LLM_MODEL_SKIPIF` (NOT provider-key) | pass a model string that any-llm will reject (e.g. invalid model id); assert `on_error` fires and the exception propagates |

### End-to-end README example

README's "Quickstart" runs the calculator MCP stdio server as a subprocess, then:
```bash
ANY_LLM_TEST_MODEL=openai:gpt-4o-mini \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
python -m tinyagent.examples.http_demo
```
Output: trace pushed to OTLP collector (via the user's wired provider) + final answer printed + `AgentTrace` JSON serialized to `trace.json`.

---

## 12. License & attribution

### `LICENSE`
- Apache-2.0 verbatim text (Sections 1–9 + Appendix).
- Copyright header lines:
  ```
                              Apache License
                          Version 2.0, January 2004
                            http://www.apache.org/licenses/
  ```
  Body unchanged from upstream.

### Upstream notice (preserved verbatim)
- `NOTICE` file at repo root:
  ```
  This product includes software developed at Mozilla.ai (https://mozilla.ai/).
  Original source: https://github.com/mozilla-ai/tinyagent (Apache-2.0).
  ```
  Apache-2.0 §4(c) requires preservation of copyright/attribution notices.

### `tinyagent.py` header (every modified source file)
```python
# SPDX-License-Identifier: Apache-2.0
#
# Forked from https://github.com/mozilla-ai/tinyagent (Apache-2.0, Copyright 2026 Mozilla.ai).
# Modifications: single-file packaging; canon 5-hook callbacks + on_error; stdio-only MCP;
# custom gen_ai.usage.cost attribute; safe calculate() via simpleeval; library-pattern OTel.
```

### `pyproject.toml`
- `license = {text = "Apache-2.0"}` — required by spec §7
- `version = "0.1.0"` — required by spec §7
- `requires-python = ">=3.11"`

---

## 13. Task breakdown (TDD-ordered, T12 split per M9)

Each task's test must be written before implementation per TDD rules. Tests live under `tests/`, implementation under `tinyagent.py` at the repo root (NOT under `src/` — flat layout per §0 C6, round-3 m7). No task exceeds ~200 LOC of new code; the loop is split into focused single-responsibility tasks (T12a-d + T12a-cross).

### Order

1. **T1 — Repo bootstrap**
   - Title: `bootstrap repo skeleton (LICENSE, NOTICE, pyproject.toml, tests/) — FLAT layout, no src/`
   - Scope: ~80 LOC (mostly config).
   - Test: `tests/test_imports.py::test_import_top_level` — asserts `import tinyagent; from tinyagent import TinyAgent, ToolCall, ...` and `__all__` contents.
   - Files: `pyproject.toml`, `LICENSE`, `NOTICE`, `tinyagent.py` (empty stub at repo root), `tests/conftest.py`, `tests/test_imports.py`. **Does NOT include `src/__init__.py`** — `tinyagent.py` lives at the repo root and the install is `[tool.setuptools] py-modules = ["tinyagent"]` (no `src/` directory). Resolves round-3 minor m5 + m7. See §3 and §0 C6.

2. **T2 — Pricing table + override**
   - Title: `pricing table + longest-prefix lookup + override channels`
   - Scope: ~120 LOC.
   - Test: `tests/test_pricing.py` — table lookup, longest-prefix, override callable wins, unknown → None, local provider → None.
   - Files: `tinyagent.py` (section 5), `tests/test_pricing.py`.

3. **T3 — Exceptions**
   - Title: `AgentError, AgentCancel, ToolNotFoundError, MCPConnectionError, MCPProtocolError`
   - Scope: ~50 LOC.
   - Test: `tests/test_exceptions.py` — subclass relationships; both MCP classes are subclasses of `AgentError` (M8 round-1 closure); `ToolNotFoundError` is a subclass of `AgentError`.
   - Files: `tinyagent.py` (section 6), `tests/test_exceptions.py`.

4. **T4 — `tool` decorator + `_wrap_no_exception` + `_cast_argument`**
   - Title: `@tool decorator + cast helpers`
   - Scope: ~120 LOC.
   - Test: `tests/test_tool_decorator.py`.
   - Files: `tinyagent.py` (section 9), `tests/test_tool_decorator.py`.

5. **T5 — Example tools (`final_answer`, `calculate`, `http_get`)**
   - Title: `shipped example tools (calculate via simpleeval)`
   - Scope: ~120 LOC.
   - Test: `tests/test_example_tools.py` — calculate uses simpleeval, http_get mocked, final_answer round-trip.
   - Files: `tinyagent.py` (section 10), `tests/test_example_tools.py`.

6. **T6 — CallbackRegistry (5 hooks, sync+async dispatch, dict-backed `register_*` methods)**
   - Title: `CallbackRegistry with pinned-loop sync/async dispatch — round-3 M3 storage model`
   - Scope: ~150 LOC.
   - Test: `tests/test_callback_registry.py` — `register_before_llm_call(fn)` writes to `self._hooks["before_llm_call"]`; `dispatch_sync` via `run_coroutine_threadsafe` reads via `self._hooks.get(name, ())` (NOT `getattr`); `dispatch_async` direct dict lookup; AgentCancel propagation; error propagation; **negative regression test that the attribute-style `cb.before_llm_call.append(fn)` form raises `AttributeError`** (regression guard against reverting to that API). Hook signature is `(ctx: Context) -> None | Awaitable[None]` (round-3 M2: no kwargs, return discarded).
   - Files: `tinyagent.py` (section 7), `tests/test_callback_registry.py`.

7. **T7 — Context dataclass + ToolCall TypedDict + AgentTrace/AgentSpan/TokenInfo/CostInfo**
   - Title: `tracing dataclasses + roll-up properties + ToolCall TypedDict (round-3 minor m6)`
   - Scope: ~200 LOC.
   - Test: `tests/test_agent_trace.py` — roll-up properties, cost math (sums only spans WITH cost attr). Also: `from tinyagent import ToolCall; tc: ToolCall = {"id": "x", "type": "function", "function": {"name": "...", "arguments": "..."}}` validates the TypedDict shape and import.
   - Files: `tinyagent.py` (sections 8, 12), `tests/test_agent_trace.py`.

8. **T8 — OTel setup module + attribute constants (library pattern)**
   - Title: `library-pattern OTel: passive tracer acquisition, no provider/exporter wiring`
   - Scope: ~60 LOC.
   - Test: `tests/test_otel_setup.py` — calling `_setup_tracing` does NOT call `set_tracer_provider`; returns NoOp tracer when none configured; idempotent across multiple calls.
   - Files: `tinyagent.py` (section 13), `tests/test_otel_setup.py`.

9. **T9 — OpenTelemetry span generation around LLM and tool calls**
   - Title: `span generation (call_llm + execute_tool) + cost-attribute writer`
   - Scope: ~250 LOC.
   - Test: `tests/test_otel.py` — InMemorySpanExporter asserts hierarchy + attrs incl. `gen_ai.usage.cost` when known, ABSENT when unknown.
   - Files: `tinyagent.py` (extends section 14, span helpers in section 13).

10. **T10 — MCP stdio client (`MCPServer`)**
    - Title: `MCPServer (stdio-only) with subprocess lifecycle + error handling`
    - Scope: ~250 LOC.
    - Test: `tests/test_mcp_stdio.py` — in-process echo/calc stdio server; tool_not_found string; subprocess cancel cleanup; broken-server marking after EOF.
    - Files: `tinyagent.py` (section 11), helper fixture `examples/inproc_mcp_echo.py` for tests.
    - Dependency: T4 (tool wrappers).

11. **T11 — `TinyAgent` core: AgentConfig + any-llm init + `call_model` with timeout**
    - Title: `AgentConfig + TinyAgent.__init__ + call_model (with request_timeout_s)`
    - Scope: ~200 LOC.
    - Test: `tests/test_agent_init.py` (mocked AnyLLM.create, completion params); `tests/test_request_timeout.py` (timeout fires on hanging coroutine).
    - Files: `tinyagent.py` (sections 14, 15, parts 1–3).

12. **T12a — ReAct loop body + per-turn tool dispatch + `final_answer` short-circuit + tool_choice retry**
    - Title: `loop body: break after first final_answer (M1), retry tool_choice on empty response (M4)`
    - Scope: ~180 LOC.
    - Test: `tests/test_agent_loop.py` — mock LLM sequences; final_answer terminates; multiple final_answer (first wins, no follow-on tool calls execute — round-3 M1 `break` short-circuit); **BOTH `before_tool_execution` and `after_tool_execution` fire on the `final_answer` tool call (M4 round-2 closure — see §8 (c))**; **after first final_answer, subsequent non-final_answer tool calls in the same turn are NOT executed (round-3 M1 `break`)**; unknown tool returns string to LLM (loop continues); trailing text fallback; **tool_choice_required fallback: empty response under `required` -> retry once with `auto` -> if empty, AgentError (round-3 M4)**.
    - Files: `tinyagent.py` (extends section 15, loop body only — break + retry branches from §8).
    - Dependency: T2, T5, T6, T7, T9, T11.

13. **T12a-cross — dedicated tool_choice retry test (round-3 M4 cross-cut)**
    - Title: `tests/test_tool_choice_retry.py — empty response under required -> auto retry -> empty -> AgentError`
    - Scope: ~60 LOC.
    - Test: `tests/test_tool_choice_retry.py` — synthetic LLM that returns `tool_calls=[]`, no assistant content, on the first call under `required`; asserts the registry of tool_choice calls shows the retry under `auto` was issued exactly once; if the second call also returns empty, asserts `AgentError` is raised. Also: a test that a non-empty retry response (e.g. trailing assistant text) is **not** flagged as empty and the trailing-text branch returns normally.
    - Files: `tests/test_tool_choice_retry.py`.
    - Dependency: T12a. (Implementation is in T12a; the test is its own cross-cut task for clarity.)

14. **T12b — Keep-last-N pruning with pair-preserving algorithm**
    - Title: `_prune_messages_keeping_pairs: pair-preserving pruning`
    - Scope: ~100 LOC (the algorithm body; the algorithm is documented in §9).
    - Test: `tests/test_prune.py` — invariant: every surviving tool message has its parent assistant; synthetic provider rejects malformed pairings and we confirm it accepts the pruned set.
    - Files: `tinyagent.py` (extends section 15, `_prune_messages_keeping_pairs`).

15. **T12c — Sync `run()` wrapper with pinned-loop bridge**
    - Title: `sync run() via run_async_in_sync + pinned loop for async hooks — dispatch_sync via self._hooks[name] (round-3 M3)`
    - Scope: ~80 LOC.
    - Test: `tests/test_agent_loop_sync.py` — sync `run()` invokes async hooks via pinned loop; sync hooks work; AgentCancel propagates; mixed hook set runs in order. Registration uses ONLY `register_*` methods (round-3 M3); the test asserts no attribute-style registration is used anywhere.
    - Files: `tinyagent.py` (extends section 15, `run()`).

16. **T12d — `on_error` integration + AgentCancel mid-loop**
    - Title: `on_error hook fires on every escaping exception; AgentCancel terminates loop`
    - Scope: ~80 LOC.
    - Test: extend `tests/test_agent_loop.py::test_on_error_fires_and_reraises` and `test_agent_cancel_mid_loop`.
    - Files: `tinyagent.py` (extends section 15, error branches).

17. **T13 — Pricing override wiring at the call site (per-span writer)**
    - Title: `per-span cost writer: write iff known`
    - Scope: ~60 LOC.
    - Test: `tests/test_pricing_override.py` — cost attr written when known, OMITTED when unknown/local; AgentTrace.cost roll-up skips absent.
    - Files: `tinyagent.py` (extends section 15, `_estimate_cost` integration).
    - Dependency: T9, T11.

18. **T14 — `add_mcp_server` public method + `MCPServer` ergonomic helpers**
    - Title: `add_mcp_server public async-CM method + __all__ export`
    - Scope: ~80 LOC.
    - Test: extend `tests/test_mcp_stdio.py::test_add_mcp_server_async_cm` (both forms from §10).
    - Files: `tinyagent.py` (extends section 15).
    - Dependency: T10.

19. **T15a — pyproject finalization + license metadata + dependency pins**
    - Title: `pyproject.toml: dependency pins + license + Python version — flat py-modules layout (round-3 m7)`
    - Scope: ~50 LOC.
    - Test: `tests/test_examples_run.py::test_pip_install_smoke` — `pip install -e .` in a fresh venv, then `import tinyagent` succeeds AND `tinyagent.py` is the package module (no `src/` shadowing). Confirms py-modules resolution works against the repo-root file.
    - Files: `pyproject.toml` (final), `tests/test_examples_run.py`. The pyproject.toml locks `[tool.setuptools] py-modules = ["tinyagent"]` with `tinyagent.py` at the repo root — no `package-dir`, no `packages.find`, no `src/`. Resolves round-3 minor m7.

20. **T15b — Example scripts**
    - Title: `examples/*.py: calculator_mcp_stdio.py, http_demo.py, tracing_otlp.py`
    - Scope: ~150 LOC.
    - Test: `tests/test_examples_run.py::test_each_example_runs_under_mocked_llm`. Example code uses `register_*` callback API (round-3 M3); a smoke scan asserts no `cb.before_llm_call.append(...)` form anywhere in the `examples/` tree.
    - Files: `examples/calculator_mcp_stdio.py`, `examples/http_demo.py`, `examples/tracing_otlp.py`.

21. **T15c — README + docs/decisions.md**
    - Title: `README: install, config, callbacks, MCP stdio, OTel user wiring, end-to-end example`
    - Scope: prose, no LOC budget.
    - Files: `README.md`, `docs/decisions.md`. The README's callback example uses `cb.register_before_llm_call(...)` (round-3 M3); docs/decisions.md cross-links C5 and C6 from §0.

22. **T16 — Integration test suite (gated `ANY_LLM_TEST_MODEL`)**
    - Title: `parametrized integration tests across scenarios, per-scenario skipif via conftest helpers`
    - Scope: ~200 LOC.
    - Test: `tests/integration/test_e2e_anyllm.py` — all six scenarios from §11; per-scenario skipif via `PROVIDER_ENV_SKIPIF` / `ANY_LLM_MODEL_SKIPIF` markers built in `conftest.py` (M10 round-1 closure). One scenario asserts the tool_choice retry on a real provider scenario sketch — the actual retry behavior is unit-tested in T12a-cross, the integration scenario stays focused on the end-to-end e2e path.
    - Files: `tests/integration/conftest.py`, `tests/integration/test_e2e_anyllm.py`.
    - Dependency: T1–T15.

### Cross-cutting risks the implementation phase MUST respect

1. **`mcp` stdio lifecycle bugs** (research C6) — pin `mcp==1.28.1` exactly. If the package breaks at runtime, do NOT loosen the pin without a reproducer. Subprocess must use `start_new_session=True` for clean SIGTERM via `os.killpg`.
2. **OpenAI `tool_choice="required"`** (research C8) — fork always auto-attaches `final_answer`. The T12a task includes an explicit fallback: if `tool_choice="required"` produces a response with **zero tool_calls AND no `final_answer` AND no trailing assistant text**, retry **once** with `tool_choice="auto"`. If that also fails, raise `AgentError` with a clear message.
3. **OTel library pattern** (peer-review B1/B2) — `_setup_tracing` MUST NOT call `set_tracer_provider`. It MUST be idempotent. Acquire tracer via `trace.get_tracer(name)` only. Module import has no OTel side effects. Tests in T8 enforce this.
4. **Callback sync/async bridge** (peer-review M3) — async hooks registered against `agent.run()` (sync) MUST be awaited via `run_coroutine_threadsafe` against the pinned worker-thread loop. No silent coroutine drop. Tests in T6 and T12c enforce this.
5. **Pair-preserving pruning** (peer-review M5) — pruning walks units of (assistant + its tool messages). Invariant: every surviving tool message has its parent assistant. T12b test injects a synthetic provider that rejects malformed pairings and confirms the pruned set passes.
6. **Async/sync callback pitfall** (research C7) — sync `run` wraps via `any_llm.utils.aio.run_async_in_sync`; user-registered async hooks MUST be supported. TDD test in T6 covers this.
7. **`final_answer` hook symmetry** (peer-review M4 round-2 closure) — `final_answer` MUST fire BOTH `before_tool_execution` AND `after_tool_execution`, with no carve-out. `before_tool_execution` sees the raw `tool_call` (args not yet parsed); `after_tool_execution` sees `ctx.tool_result` set to the captured answer. The loop's termination logic (set `seen_final_answer`, return the captured value) runs AFTER `after_tool_execution`. Tests in T12a enforce this; agent_cancel raised from either hook still terminates the loop.
8. **Pricing rule is single and canonical** (peer-review M6 round-2 closure) — `_estimate_cost` returns `float | None` and ONLY `float | None`. There is NO `(0.0, 0.0)` fallback. `gen_ai.usage.cost` is written iff the returned value is non-None. `AgentTrace.cost` roll-up sums only the spans that have the attribute. Local providers (`LOCAL_PROVIDERS`) and unknown models BOTH return None — they are indistinguishable at the span layer.
9. **MCP exception classes are declared once** (peer-review M8 round-2 closure) — `MCPConnectionError` and `MCPProtocolError` are declared in §2 (section 6) as subclasses of `AgentError`. §4's error-handling table references them by name. T3's test_exceptions.py covers the subclass relationships.
10. **Integration-test skipif uses `.get()` + per-scenario markers** (peer-review M10 round-2 closure) — `tests/integration/conftest.py` builds `PROVIDER_ENV_SKIPIF` / `ANY_LLM_MODEL_SKIPIF` markers using `PROVIDER_KEY_ENV.get(provider, ())`. NEVER subscript `[provider]`. Per-scenario markers (not module-level `allow_module_level=True`) ensure one missing key skips only its dependent scenarios, not the whole file — so `test_on_error_real_failure_mode` (which uses an invalid model id and needs no key) still runs even when no provider key is set.
11. **`final_answer` short-circuits via `break` not `continue`** (round-3 M1) — §8 pseudocode and T12a implementation MUST use `break` after the first `final_answer` capture so subsequent non-`final_answer` tool calls in the same turn do NOT execute. The previous `continue` (round-2 line 635) is gone. T12a acceptance test feeds the LLM a turn containing `[final_answer, calculate]` and asserts only `final_answer` ran. See §8 (b).
12. **CallbackRegistry uses dict-backed `register_*` storage, NOT attribute storage** (round-3 M3) — `self._hooks: dict[str, list[Callable]]`; users call `register_before_llm_call(fn)` etc.; dispatch reads via `self._hooks.get(name, ())`. The attribute form `cb.before_llm_call.append(fn)` MUST raise `AttributeError` — T6 has a negative regression test for this. No `getattr(self, name)` anywhere in dispatch. Hook signature is `Callable[[Context], None | Awaitable[None]]` (no kwargs, return discarded — round-3 M2). See §0 C5, §2 section 7, §5.
13. **Package layout is flat — no `src/`, `tinyagent.py` at repo root** (round-3 m5 + m7) — `[tool.setuptools] py-modules = ["tinyagent"]` with `tinyagent.py` at the project root. No `src/` directory. No `package-dir` mapping. T1 does NOT create `src/__init__.py`. T15a locks the flat pyproject.toml. See §0 C6, §3.

### Self-review against research.md

Checked against `research.md` §A–§I and conflicts research-C1 through research-C10. (Note: §0 C1–C6 above are architect decisions for this plan; research-C1–C10 are upstream risks. The labels overlap intentionally — both enumerations track the same incremental set of decisions, with §0's C5/C6 added in round 3 for round-3 M3 and m5+m7.)
- research-C1 (hook count) → resolved (Option A in §0 C1); symmetry rule enforced (M7 + M4 round-2)
- research-C2 (`gen_ai.usage.cost` not in semconv) → resolved (Option b in §0 C2: emit standard token attrs + custom `gen_ai.usage.cost` total); consistent omit-when-unknown (M6 round-2 canonical)
- research-C3 (MCP transports beyond stdio) → addressed (drop SSE + streamable-http)
- research-C4 (test gating env var mismatch) → addressed (`ANY_LLM_TEST_MODEL` derivation; per-provider `PROVIDER_KEY_ENV` with `.get()` to avoid KeyError — M10 round-2)
- research-C5 (any-llm pin dangerous) → addressed (`pyproject.toml` pin `any-llm-sdk>=1.16,<1.20`)
- research-C6 (mcp stdio bugs live) → addressed (pin `mcp==1.28.1` exact, `start_new_session=True`, full error table in §4, both MCP exception classes declared — M8 round-2)
- research-C7 (sync/async pitfall) → addressed (sync callback support in T6 + T12c; async callback bridge for sync `run()`)
- research-C8 (`tool_choice="required"`) → addressed (`final_answer` always registered; **empty-response retry with `tool_choice="auto"` once per turn in T12a — round-3 M4**; both hooks fire on `final_answer` — M4 round-2)
- research-C9 (tool safety) → addressed (`simpleeval` for `calculate`)
- research-C10 (cosmetic upstream bug) → addressed (line 180 redundant assignment fixed in T9 adoption)
- §0 C5 (round-3 CallbackRegistry storage) → resolved; `register_*` methods write to `self._hooks[name]`; dispatch reads via `self._hooks.get(name, ())`; attribute-style form raises `AttributeError` (T6 negative test). See cross-cutting risk #12.
- §0 C6 (round-3 package layout) → resolved; flat layout, `tinyagent.py` at repo root, no `src/`, no `package-dir` mapping. See cross-cutting risk #13.

**No hidden constraints uncovered.** Open question #5 (single-file vs split layout) — locked to single file per user spec §2. Open question #4 (Pydantic `output_type`) — deferred to v0.2.0 per §8. Open question #7/#8 (pricing format/units) — resolved §7. Open question #1 (hook count 5 vs 6) — resolved §0 C1 (Option A). Open question #2 (cost attribute name) — resolved §0 C2 (Option b). Open question #3 (test gating env var) — resolved §11 + §0 C4. Open question #6 (`calculate` safety) — resolved §10 (simpleeval). All seven open questions are now closed.

**Anchor files identified**: 10 above, prioritized.