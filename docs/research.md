# tinyagent Fork — Research Synthesis

> Source spec: `clarified-requirements.md` (Option C — fork mozilla-ai/tinyagent, lift core into a single-file Apache-2.0 package).
> Upstream investigated: `mozilla-ai/tinyagent` HEAD `e667858a`, latest release v0.1.1 (2026-04-29).
> Research date: 2026-07-06.

---

## Findings

### A. Upstream package layout

Import name `tinyagent`. Source under `src/tinyagent/`. Top-level structure:

```
src/tinyagent/
├── __init__.py              # public API re-exports: TinyAgent, AgentConfig, AgentTrace, AgentCancel, AgentRunError
├── agent.py                 # TinyAgent class, ReAct loop, any-llm wiring
├── config.py                # Pydantic AgentConfig + MCPStdio / MCPSse / MCPStreamableHttp
├── logging.py
├── py.typed
├── callbacks/
│   ├── base.py              # Callback base — 6 lifecycle hooks
│   ├── context.py           # Context dataclass (current_span, trace, tracer, shared)
│   ├── span_end.py          # SpanEndCallback — ends span + appends to AgentTrace
│   ├── span_generation.py   # _SpanGeneration — opens call_llm / execute_tool spans
│   ├── span_print.py        # ConsolePrintSpan — rich console printer
│   └── wrapper.py           # _TinyAgentWrapper — monkey-patches call_model + tool.call_tool
├── evaluation/              # LlmJudge, AgentJudge, schemas, tools
├── serving/
│   ├── a2a/                 # Agent2Agent serving (a2a extra) — EXCLUDED per user spec
│   └── mcp/                 # MCP serving
├── testing/helpers.py       # LLM_IMPORT_PATH, DEFAULT_SMALL_MODEL_ID, wait_for_server_async
├── tools/
│   ├── a2a.py, composio.py
│   ├── final_output.py      # prepare_final_output() — Pydantic-validated variant
│   ├── user_interaction.py, web_browsing.py
│   ├── wrappers.py          # _wrap_tools, _wrap_no_exception, verify_callable
│   └── mcp/mcp_client.py    # MCPClient — stdio / sse / streamable-http dispatch
├── tracing/
│   ├── agent_trace.py       # AgentTrace, AgentSpan, TokenInfo, CostInfo, AgentMessage
│   ├── attributes.py        # GenAI + TinyAgentAttributes constants
│   └── otel_types.py        # Pydantic wrappers for OTEL types
└── utils/cast.py            # safe_cast_argument
```

### B. ReAct loop (core anchor)

**File:** `src/tinyagent/agent.py`

- `_run_async(self, prompt, **kwargs)` at lines **499-592** — the actual `while True` loop.
- `run_async(...)` at lines **389-497** — outer entry; opens `invoke_agent` span, manages callbacks and lock.
- `run(...)` is a sync wrapper that delegates via `any_llm.utils.aio.run_async_in_sync` (import at line 11).
- Loop body shape:
  ```python
  while True:
      response = await self.call_model(**completion_params)
      message = response.choices[0].message
      messages.append(message.model_dump())
      if message.tool_calls:
          for tool_call in message.tool_calls:
              tool_name = tool_call.function.name
              ... dispatch via self.clients[tool_name].call_tool(...)
              if tool_name == "final_answer":
                  return str(result)         # terminates the loop
      elif message.role == "assistant" and message.content:
          return str(message.content)       # terminates on plain assistant text
  ```
- `tool_choice="required"` (agent.py:280) — every turn must produce at least one tool call.
- Termination in upstream = model calls the `final_answer` tool, OR model emits a plain assistant message (OpenAI structured-output path only).

### C. final_answer — TWO mechanisms coexist upstream

1. **Bare function** at `src/tinyagent/agent.py:227-229`:
   ```python
   def final_answer(answer: str) -> str:
       """Return the final answer to the user."""
       return answer
   ```
   Auto-appended to `self.config.tools` for **non-OpenAI providers** at lines 283-284. Loop exits at line 580 when `tool_name == "final_answer"`.

2. **Pydantic-schema variant** in `src/tinyagent/tools/final_output.py`: function `prepare_final_output(output_type, instructions=None)` returns a tool internally named `"final_output"` (not `final_answer`) that validates against `output_type.model_json_schema()`. Used when `AgentConfig.output_type` is set (a Pydantic model). Not auto-registered.

**User spec mentions `final_answer` only.** The Pydantic variant is opt-in via `output_type` — for the single-file fork, keep it simple: just `final_answer` and optionally accept an `output_type` Pydantic model that re-uses `_return_output_type` logic at `agent.py:594-622`.

### D. Callback system (CRITICAL FINDING — conflicts with user spec)

**File:** `src/tinyagent/callbacks/base.py`, class `Callback` at line 10.

**Six** hooks total, **not the canonical five**:

| Hook | Line | Signature |
|---|---|---|
| `before_agent_invocation` | 25 | `(self, context: Context, *args, **kwargs) -> Context` |
| `before_llm_call` | 29 | same shape |
| `before_tool_execution` | 33 | same shape |
| `after_agent_invocation` | 37 | same shape |
| `after_llm_call` | 41 | same shape |
| `after_tool_execution` | 45 | same shape |

- **No `on_error` hook.** Errors propagate via Python exceptions: `AgentRunError` (`agent.py:107`) for fatal, `AgentCancel` (`agent.py:48-104`) for cancellation. Cancellation works by raising `AgentCancel` from any hook; framework catches it at `agent.py:469`.
- Each hook receives a `Context` (threaded through) supporting sync or async implementations (`asyncio.iscoroutinefunction` check in `wrapper.py:38,47,62,71`).
- Dispatch via `_TinyAgentWrapper` (`callbacks/wrapper.py`) monkey-patches `agent.call_model` (lines 32-54) and each `agent.clients[k].call_tool` (lines 56-76). Context is keyed by `get_current_span().get_span_context().trace_id` (lines 33-35, 57-59).
- Built-in callbacks auto-injected: `ConsolePrintSpan`, `_SpanGeneration`, `SpanEndCallback`.

**User's confirmed spec (locked decision #5) names the canonical 5-hook set: `before_llm_call`, `after_llm_call`, `before_tool_execution`, `after_tool_execution`, `on_error`. Upstream has 6 different hooks with NO `on_error`.** See Conflicts section.

### E. any-llm integration

**Library:** `any-llm-sdk` (renamed from `any-llm` at v1.0; latest 1.19.0, 2026-06-26).

- Imports at `agent.py:9-10`:
  ```python
  from any_llm import AnyLLM, LLMProvider
  from any_llm.utils.aio import run_async_in_sync
  ```
- Construction at `agent.py:265-282`: `AnyLLM.create(provider_name, **llm_kwargs)` then `self.completion_params = {"model": model_id, "tools": [], "tool_choice": "required", ...}`.
- Call site — `call_model` at `agent.py:624-625`:
  ```python
  async def call_model(self, **completion_params: Any) -> ChatCompletion:
      return await self.llm.acompletion(**completion_params)
  ```
- Model string format: **`"openai:gpt-4o"`** (colon, **not** LiteLLM-style slash). Provider may be split out: `provider="openai", model="gpt-4o"`.
- Response is OpenAI-compatible: `response.choices[0].message.tool_calls[i].function.{name, arguments}`. `arguments` is a JSON-encoded string — parsed via `json.loads` at `agent.py:557`.
- Token usage: `response.usage.prompt_tokens` / `.completion_tokens` / `.total_tokens` (CompletionUsage re-export from openai.types).
- Pricing: **NOT exposed** by any-llm. Fork must bring its own pricing table.

### F. MCP integration

**Library:** `mcp` (official Anthropic MCP Python SDK; current stable 1.28.1, 2026-06-26). **No** `langchain_mcp_adapters`.

- Imports at `tools/mcp/mcp_client.py:21-28`:
  ```python
  from mcp import ClientSession, StdioServerParameters
  from mcp.client.sse import sse_client
  from mcp.client.stdio import stdio_client
  from mcp.client.streamable_http import streamablehttp_client
  ```
- Three Pydantic config classes in `config.py`:
  - `MCPStdio` (lines 11-40)
  - `MCPSse` (lines 43-71) — **deprecated**; emits `DeprecationWarning("SSE is deprecated in the MCP specification in favor of Streamable HTTP as of version 2025-03-26")` via a `@model_validator(mode="before")` named `sse_deprecation`.
  - `MCPStreamableHttp` (lines 74-92)
- Dispatch in `MCPClient.connect()` at `mcp_client.py:51-86` via `isinstance(self.config, ...)` branches.
- stdio pattern (canonical):
  ```python
  server_params = StdioServerParameters(command=..., args=[...], env={**os.environ})
  read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
  async with ClientSession(read, write) as session:
      await session.initialize()
      tools = await session.list_tools()      # .tools -> List[MCPTool] with .name, .description, JSON schema
      result = await session.call_tool("add", {"a": 1, "b": 2})  # CallToolResult w/ .content (TextContent/ImageContent), .isError
  ```
- Tool registration: `tools/wrappers.py:_wrap_tools(tools)` (lines 58-82) iterates `AgentConfig.tools`, detects `MCPParams`, calls `MCPClient.connect()`, `list_tools()`, synthesises Python callables via `_create_tool_function()` (`mcp_client.py:135-201`) with `inspect`-derived schemas.
- Cleanup: `cleanup_async` (`agent.py:360-368`) drains all MCP clients.

**User spec locks MCP transports to stdio only.** The SSE and streamable-http branches in `connect()` and `MCPSse`/`MCPStreamableHttp` config classes should be dropped.

### G. OpenTelemetry tracing

- Tracer obtained at `agent.py:258`: `self._tracer = otel_trace.get_tracer("tinyagent")`.
- Span hierarchy (parent -> children):
  ```
  invoke_agent {config.name}                     <- agent.py:412-414 (root)
  +-- call_llm {model_id}                        <- span_generation.py:52
  +-- execute_tool {tool_name}                   <- span_generation.py:102
  +-- call_llm {model_id}                        <- (next iteration)
  +-- execute_tool {tool_name}
  ```
  Children inherit trace ID via nested OTel contexts; trace stored as flat `AgentTrace.spans: list[AgentSpan]` (`agent_trace.py:190`).
- Span attribute constants in `src/tinyagent/tracing/attributes.py`:

  | Constant | Key | Source |
  |---|---|---|
  | `OPERATION_NAME` | `gen_ai.operation.name` | semconv |
  | `AGENT_NAME` | `gen_ai.agent.name` | semconv |
  | `AGENT_DESCRIPTION` | `gen_ai.agent.description` | semconv |
  | `REQUEST_MODEL` | `gen_ai.request.model` | semconv |
  | `OUTPUT_TYPE` | `gen_ai.output.type` | semconv |
  | `TOOL_NAME` | `gen_ai.tool.name` | semconv |
  | `TOOL_DESCRIPTION` | `gen_ai.tool.description` | semconv |
  | `USAGE_INPUT_TOKENS` | `gen_ai.usage.input_tokens` | **semconv standardized** |
  | `USAGE_OUTPUT_TOKENS` | `gen_ai.usage.output_tokens` | **semconv standardized** |
  | `INPUT_MESSAGES` | `gen_ai.input.messages` | tinyagent-local |
  | `OUTPUT` | `gen_ai.output` | tinyagent-local |
  | `TOOL_ARGS` | `gen_ai.tool.args` | tinyagent-local |
  | `USAGE_INPUT_COST` | `gen_ai.usage.input_cost` | **tinyagent-local — NOT in semconv; never populated** |
  | `USAGE_OUTPUT_COST` | `gen_ai.usage.output_cost` | **tinyagent-local — NOT in semconv; never populated** |
  | `TinyAgentAttributes.VERSION` | `tinyagent.version` | tinyagent-local |

- Token extraction at `span_generation.py:173-182`:
  ```python
  if token_usage := getattr(response, "usage", None):
      if token_usage:
          input_tokens = token_usage.prompt_tokens
          input_tokens = token_usage.prompt_tokens        # <- line 180 redundant assignment (cosmetic bug)
          output_tokens = token_usage.completion_tokens
  ```
- Trace roll-up: `AgentTrace.tokens` and `AgentTrace.cost` are `@cached_property` (lines 296-316) summing across `is_llm_call()` spans.
- **`gen_ai.usage.cost` does NOT exist in the OTel semconv.** See Conflicts section. Upstream defines `gen_ai.usage.input_cost` / `output_cost` as **custom** attributes, but the code never writes them — no writer found in `span_generation.py`, `span_end.py`, or `agent.py`.

### H. License & attribution

- **LICENSE**: Apache-2.0. Copyright 2026 Mozilla.ai. Full Apache text body (Sections 1-9 + Appendix) verbatim.
- **NOTICE file**: **absent upstream.** §4(d) obligation is inert (only-if-it-exists rule).
- **Apache-2.0 fork checklist** (Sections 4(a)-(d)):
  - (a) Ship `LICENSE` verbatim — required.
  - (b) Mark modified files with prominent change notices — required.
  - (c) Preserve upstream copyright/patent/trademark/attribution notices in source form — required.
  - (d) NOTICE preservation — not triggered (upstream has none).
- Per-file SPDX headers (`SPDX-License-Identifier: Apache-2.0`) are **best practice, not §4 mandate.**
- §6 (Trademarks): "Mozilla.ai" name/logo not licensed. Customary "forked from mozilla-ai/tinyagent" attribution in README is fine.

### I. Conventions to mirror (from conventions leaf)

- **Source layout:** src-layout (`src/tinyagent/...`) — but user wants single-file. The single-file variant uses `[tool.setuptools] py-modules = ["tinyagent"]` instead of `packages.find`.
- **Build backend:** setuptools + setuptools_scm (dynamic version). For our fork we can hardcode `version = "0.1.0"`.
- **Lint:** ruff with `extend-select = ["ALL"]` + per-file-ignores. No black/isort.
- **Types:** mypy `strict = true`. Source uses `from __future__ import annotations` liberally.
- **Pydantic:** `ConfigDict(frozen=True, extra="forbid")` on config; `extra="forbid"` on tracing models.
- **Imports:** absolute; `TYPE_CHECKING` guard for type-only imports.
- **Naming:** PascalCase classes, snake_case functions/modules, UPPER_SNAKE constants, `_` prefix for privates.
- **Tests:** pytest + pytest-asyncio (`auto` mode), `unittest.mock.AsyncMock` for LLM/MCP, fixture-per-directory conftest, session-scoped async fixtures for subprocess MCP servers.
- **Integration gating:** `pytest.mark.skipif(not _has_<X>())` over env vars. **Upstream uses `MISTRAL_API_KEY` / `OPENAI_API_KEY`; user spec wants `ANY_LLM_TEST_MODEL`.**
- **Docstrings:** Google-style (Args / Returns / Raises / Example); ruff D rules enforced in src, disabled in tests.
- **Async:** async-first I/O; sync wrapper via `any_llm.utils.aio.run_async_in_sync`. Notebook detection via `hasattr(builtins, "__IPYTHON__")`.

---

## Open Questions

1. **Hook surface (5 vs 6 vs 7):** User locked decision is the canonical 5-hook set including `on_error`. Upstream has 6 (no `on_error`, has agent-level brackets). Should we (a) keep the upstream 6 + add `on_error` as a 7th hook, (b) keep upstream 6 and document deviation from spec, or (c) collapse to the canonical 5 by dropping the agent-level brackets? Architect must choose.
2. **Cost attribute name:** User spec says `gen_ai.usage.cost`. OTel semconv has no such attribute. Options: (a) emit a custom `tinyagent.cost.usd` attribute (or `gen_ai.usage.input_cost`/`output_cost` matching upstream) and document the deviation; (b) drop cost entirely from spans and surface it only in the `AgentTrace.cost` roll-up; (c) push back to user — confirm whether `gen_ai.usage.cost` was intentional or shorthand for "some cost field somewhere."
3. **Test gating env var:** User locked decision is `ANY_LLM_TEST_MODEL`. Upstream uses `MISTRAL_API_KEY` / `OPENAI_API_KEY` separately. The `ANY_LLM_TEST_MODEL` shape (e.g. `"openai:gpt-4o-mini"`) is more general — gate on the env var's presence + parse to provider for key lookup. Architect should confirm.
4. **final_answer scope:** User mentions `final_answer` as a single tool. Upstream has TWO termination tools (`final_answer` for non-OpenAI + `final_output` for Pydantic). Should we ship only `final_answer`, or keep `output_type` support for typed returns?
5. **Single-file packaging:** Spec is "literally one Python file." Upstream is ~500 LOC across many modules. Compression into one file is feasible but loses `src/` layout, separate `tracing/attributes.py` constants, etc. Architect should confirm: literal one file (`tinyagent.py`) vs minimal-but-split (a few files in a flat layout).
6. **Tool safety:** User spec ships `calculate` (safe expression evaluator) — upstream uses `eval()` (per `agent.md` §3 risk #5). Must replace with `simpleeval` or `RestrictedPython` from day one.
7. **Built-in pricing table:** Locked decision #2 says "Built-in pricing table for cost estimation, overridable via callback/config." Upstream has the roll-up but no writer. Architect must decide the table format and how `OVERRIDE` works.
8. **Pricing currency / units:** USD? Per-million tokens? Per-thousand? Align with provider conventions.

---

## Conflicts / Risks

### C1. Hook count deviates from user spec — HIGH RISK

User locked decision #5: canonical 5-hook set (`before_llm_call`, `after_llm_call`, `before_tool_execution`, `after_tool_execution`, `on_error`). **Upstream has 6 hooks (adds `before_agent_invocation`/`after_agent_invocation`) and NO `on_error`.** Three options for the architect:

- **Option A (rebase to user spec):** drop the two agent-level brackets and add an `on_error` hook. Looser match to upstream semantics; cleaner API per spec.
- **Option B (extend upstream to user spec):** keep the 6 upstream hooks AND add `on_error` as a 7th. Most faithful to upstream; mild deviation from the spec's "canonical 5."
- **Option C (deviate, document):** keep the upstream 6 as-is and surface errors via Python exceptions only. Document the deviation.

Recommend **Option A** — single-file package, clean spec-driven API, `on_error` lets callbacks implement dead-letter-style handlers without try/except around each `agent.run()`.

### C2. `gen_ai.usage.cost` is not a semconv attribute — HIGH RISK

User spec (success criterion #3) names `gen_ai.usage.cost`. The OTel GenAI semantic conventions define **only token-counting attributes** under `gen_ai.usage.*`:

```
gen_ai.usage.input_tokens
gen_ai.usage.output_tokens
gen_ai.usage.cache_read.input_tokens
gen_ai.usage.cache_creation.input_tokens
gen_ai.usage.reasoning.output_tokens
```
(Plus the deprecated `completion_tokens` / `prompt_tokens` aliases.)

There is **no** `gen_ai.usage.cost`. Upstream uses `gen_ai.usage.input_cost` / `output_cost` as **custom** attributes (tinyagent-local) but never writes them. Implications:

- A strict OTel collector schema validator will drop any cost attribute.
- The community has not converged on a stable cost attribute name; choosing one now risks a namespace collision when OTel eventually standardizes it.
- **Recommendation:** emit `gen_ai.usage.input_tokens` / `output_tokens` (standardized), and surface cost via either a custom attribute (`tinyagent.cost.usd` or `gen_ai.usage.input_cost`/`output_cost`) or via the `AgentTrace.cost` roll-up returned to the caller. Document the deviation explicitly in the README. Confirm with the user.

### C3. MCP transports > stdio — LOW RISK

User locked decision #3: stdio only. Upstream ships three (stdio, SSE-deprecated, streamable-http). Drop the SSE and streamable-http branches in `connect()` and remove `MCPSse` / `MCPStreamableHttp` config classes. This is a simplification, not a conflict — but a structural reduction in `config.py` and `mcp_client.py`.

### C4. Test gating env var mismatch — LOW RISK

User wants `ANY_LLM_TEST_MODEL` (e.g. `"openai:gpt-4o-mini"`). Upstream gates on `MISTRAL_API_KEY` / `OPENAI_API_KEY` separately. Implementation note: `ANY_LLM_TEST_MODEL` is a *better* shape because it carries provider+model in one string — easy to derive the required API key env var via `provider.split(":")[0].upper() + "_API_KEY"`. Confirm the gating pattern.

### C5. Upstream's any-llm pin is dangerous — MEDIUM RISK

Upstream `pyproject.toml` declares `any-llm-sdk>=1.0,<2`. Known 1.x breakages:
- **1.16.0:** `platform` provider **removed**.
- **1.15.0:** header `X-AnyLLM-Key` renamed to `AnyLLM-Key`.
- **1.13.0:** `anthropic` SDK now a **required** dependency (was optional).
- Package renamed from `any-llm` to `any-llm-sdk` at the 1.0 boundary.

Floating `>=1.0,<2` means a `pip install --upgrade` today can break a working install tomorrow. **Pin tighter** in the fork, e.g. `any-llm-sdk>=1.16,<1.20` once validated, and document.

### C6. MCP stdio bugs are live — MEDIUM RISK

Upstream pins `mcp>=1.5.0` (no upper bound). Open issues as of 2026-07-06:
- **#2678** — stdio silently drops in-flight tool responses on stdin EOF.
- **#2610** — `RequestResponder.__exit__` leaks `CancelledError`, kills receive loop.
- **#2873** — invalid UTF-8 from server stdout **abandoned** (maintainers won't fix).
- **#2839, #2840, #2880, #2915, #2958, #3060** — adjacent stdio lifecycle fixes.

**Pin an exact tested version** (e.g. `mcp==1.28.1`) for the fork, not a `>=` range. Track upstream `v2.x` migration separately.

### C7. Async/sync pitfall — MEDIUM RISK

Upstream's main agent loop is async-first (`run_async`); sync `run` delegates via `any_llm.utils.aio.run_async_in_sync`. Tools accept both sync and async callables. The user spec says "A user can register a `before_tool_execution` callback that raises, and the loop halts immediately" — this works in upstream via `AgentCancel` exception, but the user spec wording suggests an exception raised in a sync callback. Architect must confirm whether sync callback invocation is supported.

### C8. Upstream drops features the fork needs — LOW RISK

- Upstream auto-injects `final_answer` only for non-OpenAI providers. For OpenAI, it relies on `output_type` Pydantic schema. The fork needs `final_answer` working on **all** providers (per user spec) — guard against upstream's OpenAI-specific behaviour by always registering `final_answer`.
- Upstream requires `tool_choice="required"` — every turn must produce a tool call. For providers that don't honor `tool_choice`, this can fail. Architect may need a fallback.

### C9. Tool safety — MEDIUM RISK

Upstream uses `eval()`-style reflection in tool wrappers (per `agent.md` §3 risk #5, mozilla tinyagent inherits the unsafe pattern via `safe_cast_argument`). The user spec explicitly ships a `calculate` tool that must be safe. **Use `simpleeval` or `RestrictedPython` — not raw eval.**

### C10. Cosmetic upstream bug — LOW RISK

`callbacks/span_generation.py:179-180` reads `token_usage.prompt_tokens` into `input_tokens` twice. Functional impact: zero (the second assignment is identical), but it's a smell. Fix on adoption.

---

## Recommended anchor files for the next phase

For the architect, the upstream files to lift from (or model after) — in priority order:

| Priority | File | Lines | Why |
|---|---|---|---|
| **P0** | `src/tinyagent/agent.py` | 227-229, 265-282, 389-497, 499-592, 594-622, 624-625 | `final_answer` function, any-llm init, outer `run_async`, `_run_async` loop, `_return_output_type`, `call_model`. The whole agent is here. |
| **P0** | `src/tinyagent/callbacks/base.py` | 10-47 | Six-hook `Callback` base. Architect decides whether to keep 6, drop to 5, or extend to 7 with `on_error`. |
| **P0** | `src/tinyagent/callbacks/wrapper.py` | 29-97 | `_TinyAgentWrapper.wrap`/`unwrap` — monkey-patch machinery that fires hooks around `call_model` and `call_tool`. |
| **P0** | `src/tinyagent/tools/wrappers.py` | 58-82, 135-201 (mcp_client) | `_wrap_tools` for callable -> JSON schema; `_create_tool_function` for MCP tool callable synthesis. |
| **P0** | `src/tinyagent/tools/mcp/mcp_client.py` | 51-86, 88-133 | `connect()` (drop SSE+streamable-http branches); `list_tools()`; `_create_tool_function()`. |
| **P1** | `src/tinyagent/config.py` | 11-92 | `MCPStdio` only (drop `MCPSse`, `MCPStreamableHttp`). Pydantic pattern to mirror for `AgentConfig`. |
| **P1** | `src/tinyagent/tracing/agent_trace.py` | 79, 155-167, 190, 200-205, 286-316 | `AgentSpan`, `AgentTrace`, `TokenInfo`, `CostInfo`, `@cached_property` roll-up. |
| **P1** | `src/tinyagent/tracing/attributes.py` | full file | Span attribute constants — adopt semconv keys, drop or namespace the cost keys. |
| **P1** | `src/tinyagent/callbacks/span_generation.py` | 22-193, esp. 173-182 | Open `call_llm` / `execute_tool` spans; token extraction (fix the line 180 redundant assignment on adoption). |
| **P1** | `src/tinyagent/callbacks/span_end.py` | 6-19 | `SpanEndCallback` — ends span + appends to `AgentTrace`. |
| **P2** | `src/tinyagent/callbacks/context.py` | full | `Context` dataclass (current_span, trace, tracer, shared). |
| **P2** | `src/tinyagent/agent.py` | 48-104 | `AgentCancel` exception class — used for callback-driven termination. |
| **P3** | `src/tinyagent/utils/cast.py` | full | `safe_cast_argument` — type-coerce tool args from JSON. |
| **P3** | `src/tinyagent/tools/final_output.py` | 8-65 | `prepare_final_output` Pydantic-schema helper — only if Architect decides to keep `output_type` support. |
| **P3** | `src/tinyagent/callbacks/span_print.py` | full | `ConsolePrintSpan` rich printer — useful default for examples, not strictly needed. |
| **DROP** | `src/tinyagent/serving/a2a/` | all | Excluded by user spec. |
| **DROP** | `src/tinyagent/serving/mcp/` | all | Excluded by user spec. |
| **DROP** | `src/tinyagent/evaluation/` | all | Not in user spec. |
| **DROP** | `src/tinyagent/tools/a2a.py`, `composio.py`, `web_browsing.py`, `user_interaction.py` | all | Vendor integrations out of scope. |
| **DROP** | SSE / streamable-http branches in `mcp_client.py` | — | User spec is stdio only. |

**Target dependency surface** for the fork:

```
any-llm-sdk>=1.16,<1.20      # tighter pin than upstream
mcp==1.28.1                  # exact pin per risk C6
opentelemetry-api
opentelemetry-sdk
pydantic>=2
httpx                        # for http_get tool
simpleeval                   # for safe calculate tool (replaces eval)
```

---

## Cross-Verification

Three load-bearing claims from this synthesis were independently re-verified by fresh haiku-tier leaf delegates that were blind to my reasoning chain. Each received only the specific falsifiable claim and a bounded scope to check. Verdicts below.

### V1. Callback hooks — CONFIRMED

**Claim:** mozilla-ai/tinyagent's `Callback` base class defines exactly these six hook methods in this order, each `(self, context: Context, *args, **kwargs) -> Context`: `before_agent_invocation`, `before_llm_call`, `before_tool_execution`, `after_agent_invocation`, `after_llm_call`, `after_tool_execution`. No `on_error` method exists.

**Verifier source:** https://raw.githubusercontent.com/mozilla-ai/tinyagent/main/src/tinyagent/callbacks/base.py

**Verifier findings:**
- All six methods present at lines 25, 29, 33, 37, 41, 45 — verbatim signatures match the claim.
- Order matches exactly.
- Total hook count: 6.
- **No `on_error` method anywhere in the class body** (class body ends at line 47).

**Verdict: CONFIRMED.** The user spec's canonical 5-hook set deviates from upstream; architect must reconcile (see Conflict C1).

### V2. `gen_ai.usage.cost` in OTel semconv — CONFIRMED (not present)

**Claim:** The OpenTelemetry GenAI semantic conventions define NO `gen_ai.usage.cost` attribute. Standardized `gen_ai.usage.*` attributes are limited to token counting.

**Verifier source:** https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/ and https://github.com/open-telemetry/semantic-conventions-genai (canonical `model/gen-ai/registry.yaml`).

**Verifier findings:** All `gen_ai.usage.*` attributes enumerated verbatim:
1. `gen_ai.usage.input_tokens`
2. `gen_ai.usage.output_tokens`
3. `gen_ai.usage.cache_read.input_tokens`
4. `gen_ai.usage.cache_creation.input_tokens`
5. `gen_ai.usage.reasoning.output_tokens`

Plus deprecated aliases (`completion_tokens` -> `output_tokens`, `prompt_tokens` -> `input_tokens`).

**No attribute containing the substring `cost` exists anywhere in the `gen_ai.usage.*` namespace.**

**Verdict: CONFIRMED.** The user spec's `gen_ai.usage.cost` is not a real OTel attribute. Cost must be a custom attribute (see Conflict C2).

### V3. MCP transports and SSE deprecation — CONFIRMED

**Claim:** mozilla-ai/tinyagent supports three MCP transports via `MCPStdio`, `MCPSse`, `MCPStreamableHttp` Pydantic classes. `MCPClient.connect()` dispatches via `isinstance` checks. `MCPSse` carries a deprecation warning.

**Verifier source:** https://raw.githubusercontent.com/mozilla-ai/tinyagent/main/src/tinyagent/config.py and https://raw.githubusercontent.com/mozilla-ai/tinyagent/main/src/tinyagent/tools/mcp/mcp_client.py.

**Verifier findings:**
- All three config classes present in `config.py`: `MCPStdio`, `MCPSse`, `MCPStreamableHttp`.
- `MCPSse` has a `@model_validator(mode="before")` named `sse_deprecation` that emits `warnings.warn("SSE is deprecated in the MCP specification in favor of Streamable HTTP as of version 2025-03-26", DeprecationWarning, stacklevel=2)`.
- `MCPClient.connect()` has exactly three `isinstance` branches (stdio, sse, streamable_http) plus an `else: raise ValueError` fallback.

**Verdict: CONFIRMED.** Dropping the SSE and streamable-http branches is a clean simplification (Conflict C3).

---

*Synthesis complete. Ready for architect hand-off.*