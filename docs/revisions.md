# Revision log — round 1

> Run: `run-20260706-120000-abc123`
> Reviewer verdict: `block` (2 blockers, 8 majors, 6 minors)
> Outcome: all 16 issues addressed in `plan.md`. No C1/C2 re-litigation.

## Blockers

- **B1 — `_setup_tracing` mutates the process-wide TracerProvider at module import.**
  - **Fix.** Rewrote §6 in full. `_setup_tracing` now follows the OTel **library pattern**: it acquires a tracer via `trace.get_tracer("tinyagent")` and returns it. It does NOT call `trace.set_tracer_provider(...)`, does NOT construct a `TracerProvider`, and does NOT add any span processor. It is invoked once from `TinyAgent.__init__` (T8), not at module import. Idempotency is intrinsic — `get_tracer` is itself a no-op when no provider is configured (it returns a no-op tracer backed by the default `ProxyTracerProvider`). Tests in `tests/test_otel_setup.py` (T8) assert that calling `_setup_tracing` does NOT change the global provider state.
  - **Source verification.** Confirmed against opentelemetry-api source: the default `get_tracer_provider()` returns `ProxyTracerProvider`, and OTel docs explicitly instruct libraries to never call `set_tracer_provider`.

- **B2 — `_build_exporter()` referenced but never defined; `OTEL_TRACES_EXPORTER` honored only by autoconfigure.**
  - **Fix.** Removed `_build_exporter()` entirely. Decided **Option C**: ship **no** exporter wiring in the library. Users install `opentelemetry-sdk` and their preferred exporter themselves (or use `opentelemetry-instrumentation` autoconfigure), wire a `TracerProvider` once at app startup, and tinyagent picks it up. README's "Tracing" section shows a copy-pasteable 10-line snippet for the OTLP/gRPC path. `examples/tracing_otlp.py` documents the env-var-based wiring. Removed `opentelemetry-sdk` and `opentelemetry-exporter-otlp-*` from `pyproject.toml` dependencies — users declare them in their own apps.

## Major

- **M3 — Async callbacks registered against sync `agent.run()` silently dropped.**
  - **Fix.** §5 "Sync/async dispatch" rewritten. `CallbackRegistry` exposes two methods: `dispatch_async(name, ctx)` (used by `run_async()`; awaits coroutines directly) and `dispatch_sync(name, ctx)` (used by `run()`; pins the worker-thread event loop in `run_async_in_sync`'s coroutine and bridges coroutine hooks via `asyncio.run_coroutine_threadsafe(coro, self._loop).result()`). The sync path blocks the worker thread on each hook, which is correct because sync `run()` is itself blocking. New dedicated test file `tests/test_agent_loop_sync.py` (T12c) exercises both hook types under sync `run()`.

- **M4 — Loop only inspects `tool_calls[0]`; no defined behaviour for multiple final_answer / final_answer mixed with other tool calls / tool calls to unknown tools.**
  - **Fix.** §8 rewritten with explicit per-turn tool dispatch algorithm. Rules: (a) iterate **every** entry in `message.tool_calls`; (b) if `final_answer` is encountered, capture its `answer`, fire `after_tool_execution`, set `seen_final_answer`, and **short-circuit remaining tool calls in this turn** (do not execute further tools this turn; the loop ends entirely after this turn); (c) tool call to a non-existent tool raises `ToolNotFoundError` from `_dispatch_tool`; the loop catches it and feeds a descriptive error string back to the LLM as the tool result (`"error: tool 'foo' is not registered; available tools: [a, b, c]"`); the loop **continues** (recoverable in-band signal, not an exception escaping the loop body — so `on_error` does NOT fire); (d) multiple `final_answer` calls in one turn → first wins; subsequent ones are skipped with `warnings.warn(...)`.

- **M5 — Keep-last-N pruning desyncs `tool_call_id` ↔ tool-result pairing for OpenAI/Anthropic.**
  - **Fix.** §9 now specifies `_prune_messages_keeping_pairs(messages, keep_last_n)` with full pseudocode. Algorithm walks message body from the right and groups messages into "units" of (assistant, [tool, tool, ...]) by walking tool-roles leftward until the matching assistant with `tool_calls` is found. Each unit is kept or dropped as a single block. Invariant: every surviving tool-role message has its parent assistant message also surviving. Test in T12b injects a synthetic provider that rejects malformed pairings and confirms the pruned set passes validation.

- **M6 — Pricing table has two contradictory rules (fall through to `(0.0, 0.0)` vs. omit `gen_ai.usage.cost`).**
  - **Fix.** §7 now specifies a single, unambiguous rule. `_estimate_cost` returns `float | None`. The span writer sets `gen_ai.usage.cost` **iff** the returned value is non-`None`. `AgentTrace.cost` roll-up sums only the spans that have the attribute; absence is treated as "unknown", never as `$0`. The earlier `(0.0, 0.0)` fallback is gone. Documented in the inline comment on the lookup algorithm.

- **M7 — "Dispatches `before_llm_call` once (first iteration only)" contradicts C1 and breaks hook symmetry.**
  - **Fix.** Added a "Symmetry rule" paragraph to C1's resolution: every hook fires on every matching event. Removed the "first iteration only" line from §14 (`run_async`). §5's hook table now has an explicit "Fires on" column that says "Every LLM call iteration (no first-iteration only)". Users who want one-shot behaviour check `ctx.turn == 0` themselves.

- **M8 — No MCP error handling defined (subprocess death, EOF, invalid UTF-8, CancelledError); `add_mcp_server` missing from `__all__`.**
  - **Fix.** §4 now contains a full MCP error-handling table covering six failure modes:
    1. Subprocess spawn fails → `AgentError` raised during `setup()` before any LLM call (no `on_error`, no span yet).
    2. Subprocess dies mid-conversation (EOF, nonzero exit) → `MCPConnectionError` (subclass of `AgentError`); `on_error` fires; server marked **broken** for the rest of the run; subsequent calls into that server return a "server unavailable" string to the LLM.
    3. Invalid UTF-8 from server stdout (research C6 #2873) → `MCPProtocolError`; same broken-server marking.
    4. `asyncio.CancelledError` during `call_tool` (research C6 #2610) → cancel in-flight task; kill subprocess group via `os.killpg` (subprocess launched with `start_new_session=True`); re-raise `CancelledError` after cleanup.
    5. EOF on stdin mid-call (research C6 #2678) → `MCPConnectionError`; broken-server marking.
    6. Unknown tool name → recoverable string to LLM (covered by M4).
  - `add_mcp_server` added to `__all__` in §10 and §2 (with documentation that it re-exports `TinyAgent.add_mcp_server` for discoverability via `from tinyagent import add_mcp_server`).

- **M9 — T12 mixes five concerns; > 300 LOC; not single-responsibility.**
  - **Fix.** T12 split into four sub-tasks, each ≤ 150 LOC with its own test file: **T12a** loop body + per-turn tool dispatch + `final_answer` first-wins + unknown-tool-string + `tool_choice` fallback (~150 LOC); **T12b** `_prune_messages_keeping_pairs` with the pair-preserving algorithm (~100 LOC); **T12c** sync `run()` wrapper with pinned-loop bridge for async hooks (~80 LOC); **T12d** `on_error` integration + AgentCancel mid-loop semantics (~80 LOC). Each has a single-responsibility acceptance test. T15 (README/examples/pyproject) likewise split into T15a/T15b/T15c.

- **M10 — Integration test only exercises `calculate + final_answer`; provider key derivation wrong.**
  - **Fix.** §11's integration test expanded to a parametrized suite of six scenarios: bare `calculate` + `final_answer`; `calculator_mcp_stdio_via_subprocess` (real MCP stdio subprocess); `http_get_chain` (skipped without network); `callbacks_across_loop`; `otel_real_exporter` (wires a real `InMemorySpanExporter`); `on_error_real_failure_mode`. Provider key derivation replaced with a `PROVIDER_KEY_ENV` / `PROVIDER_EXTRA_ENV` lookup based on any-llm's per-provider `ENV_API_KEY_NAME` class attributes. Verified against upstream source: `azure → AZURE_API_KEY` (NOT `AZURE_OPENAI_API_KEY`), `huggingface → HF_TOKEN`, `gemini → GEMINI_API_KEY` (with `GOOGLE_API_KEY` as fallback), `vertexai` needs `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` (no API key), `ollama` needs `OLLAMA_HOST` (no API key).

## Minor

- **m11 — `add_mcp_server` API ambiguity (the double `await ... as ...` idiom is unusual).**
  - **Fix.** §10 documents both forms explicitly: `async with await agent.add_mcp_server(server) as tools:` (preferred) and the explicit `__aenter__` / `__aexit__` form. The method returns an async context manager; the `await` is the standard "async-CM returning context" pattern, not a typo. Re-exported at module level for `from tinyagent import add_mcp_server` discoverability. Documented as a method, not a free function, in the public API section.

- **m12 — AgentCancel-vs-skip ambiguity in `before_tool_execution`.**
  - **Fix.** §5 "AgentCancel semantics" subsection states explicitly: `AgentCancel` from **any** hook (including `before_tool_execution`) terminates the entire loop. We deliberately do NOT support "skip just this tool and continue" — that mode is too easy to misuse silently. If a user wants per-tool skip, they return a sentinel and have the tool function return a string to the LLM themselves. Matches upstream semantics and keeps the abort path single-purpose.

- **m13 — `ollama:0` wildcard in pricing; fragile prefix matching.**
  - **Fix.** Removed the `"ollama:0": (0.0, 0.0)` wildcard. Added an explicit `LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama"})` set. The lookup algorithm treats "no match + provider in `LOCAL_PROVIDERS`" the same as "no match + unknown provider": return `None` (omit cost attribute). The bare-string wildcard is gone; future `ollama:0` models won't accidentally inherit a zero-price. The precedence is now: callable override > instance dict > built-in `DEFAULT_PRICING` > `LOCAL_PROVIDERS` (omit) > unknown (omit).

- **m14 — Missing `request_timeout` parameter on `TinyAgent.__init__`.**
  - **Fix.** Added `request_timeout_s: float = 120.0` to `AgentConfig` (§2 section 14, §14 `run_async`). `call_model` wraps `self.llm.acompletion(...)` in `asyncio.wait_for(..., timeout=request_timeout_s)`. `asyncio.TimeoutError` is caught by the loop's exception arm, fired through `on_error`, and re-raised wrapped in `AgentError`. New `tests/test_request_timeout.py` injects a hanging coroutine and asserts the timeout fires; added as part of T11.

- **m15 — `on_error` swallow policy tension with spec criterion #4 (dead-letter handlers).**
  - **Fix.** §5 "Error propagation" states the chosen policy: `on_error` is **observability-only**; the agent always re-raises. Users who want swallow-and-recover wrap `agent.run()` themselves; we don't make that implicit because it would hide bugs in multi-tool loops. Documented with a one-liner in the README: "callbacks observe; the agent re-raises". This aligns with the literal text of criterion #4 (callbacks as guardrails; the agent still halts when a hook raises — see T13's `test_on_error_fires_and_reraises`).

- **m16 — Underspecified `tool_choice` fallback in T12.**
  - **Fix.** Cross-cutting risk #2 expanded: `tool_choice="required"` fallback is now an explicit acceptance criterion in T12a. If the response has **zero tool_calls AND no `final_answer` AND no trailing assistant text**, the loop retries **once** with `tool_choice="auto"`. If that also fails, raise `AgentError` with a clear message. Test in T12a simulates a provider returning empty tool_calls and asserts the retry path runs.

## Verification side-trip

Two background agents ran while the revision was being written; their findings confirmed the plan choices:

- **OTel library pattern**: confirmed against opentelemetry-api source. Default `get_tracer_provider()` returns `ProxyTracerProvider`; idiomatic library pattern is to call `trace.get_tracer(__name__)` without `set_tracer_provider`. This validates B1/B2's resolution.
- **any-llm env var conventions**: confirmed via per-provider `ENV_API_KEY_NAME` class attributes. Found three corrections to the original plan: `azure → AZURE_API_KEY` (not `AZURE_OPENAI_API_KEY`), `huggingface → HF_TOKEN`, and `vertexai` has no API key (uses `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION`). All three were fixed in §2's `PROVIDER_KEY_ENV` and §11's skipif logic.

## Substantive design changes — 5-10 line summary

1. **OTel is now library-only**: tinyagent never touches the global `TracerProvider` and ships no exporter wiring; users configure their own provider once at app startup. Resolves B1/B2; aligns with OTel's official library-author guidance.
2. **Per-turn tool dispatch is now fully specified**: iterate all `tool_calls`, first `final_answer` wins and short-circuits the rest of the turn, unknown tools return a descriptive error string to the LLM (loop continues), `on_error` is observability-only and the agent always re-raises.
3. **Pruning is pair-preserving**: `_prune_messages_keeping_pairs` walks (assistant, [tool, ...]) units as a block, guaranteeing every surviving tool message has its parent assistant message; tested against a synthetic strict-pairing provider.
4. **Pricing has a single canonical rule**: `_estimate_cost` returns `float | None`; span attribute is written iff non-None; `AgentTrace.cost` roll-up skips absent spans; local providers and unknown models both omit the attribute. No more `(0.0, 0.0)` fallback.
5. **Sync/async callback bridge**: `CallbackRegistry.dispatch_sync` pins the worker-thread event loop and bridges coroutine hooks via `asyncio.run_coroutine_threadsafe`; sync `run()` never silently drops an async hook.
6. **MCP has full error handling**: subprocess spawn failure, mid-conversation death, EOF on stdin, invalid UTF-8, `CancelledError`, and unknown-tool all have defined behaviour; subprocess is launched with `start_new_session=True` so `os.killpg` cleans up on cancel.
7. **T12 split into T12a/b/c/d**: each ≤ 150 LOC, each testable in isolation; loop body, pruning, sync wrapper, and on-error are now four single-responsibility tasks.
8. **`request_timeout_s` added**: 120s default; `asyncio.wait_for` wraps `acompletion`; `on_error` fires on timeout; new dedicated test.
9. **Integration suite expanded to six parametrized scenarios** with per-provider skipif via `PROVIDER_KEY_ENV` (now correctly enumerates `AZURE_API_KEY`, `HF_TOKEN`, Gemini's two-name fallback, and Vertex's project+location requirement).

---

# Revision log — round 2

> Run: `run-20260706-120000-abc123`
> Reviewer verdict (round 2): `block` (3 new majors, 4 partial-fix majors reopened, 8 minors)
> Outcome: all 3 new majors + 4 partial fixes + 8 minors addressed in `plan.md`. No C1/C2 re-litigation; no fully-fixed round-1 issue (B1, B2, M3, M5, M7, M9, m11-m16) re-touched.

## Partially-fixed round-1 issues (now closed)

- **M4 — `final_answer` hook symmetry.**
  - **Decision (Option a).** `final_answer` now fires **BOTH** `before_tool_execution` AND `after_tool_execution`. `before_tool_execution` sees the raw `tool_call` (args not yet parsed); `after_tool_execution` sees `ctx.tool_result` set to the captured answer. No carve-out from the symmetry rule.
  - **Where it changed.**
    - §0 added new "C3. final_answer hook symmetry" subsection documenting the decision.
    - §5 hook table now reads: `before_tool_execution | ... | Every tool invocation, INCLUDING final_answer` and `after_tool_execution | ... | Every tool invocation, INCLUDING final_answer`. The symmetry rule line now explicitly mentions `final_answer` and cites §8.
    - §8 pseudocode final_answer branch now fires `fire_before_tool_execution(before_ctx)` before parsing args and `fire_after_tool_execution(...)` after capturing the answer. Rule list expanded from 4 rules to 5; new rule (c) states "Both hooks fire on `final_answer`".
    - §2 outline `run_async` loop now matches §8 (was inconsistent before).
    - §13 T12a acceptance criteria now explicitly require "BOTH `before_tool_execution` and `after_tool_execution` fire on the `final_answer` tool call".
    - §13 cross-cutting risks #7 now binds M4 to T12a.
    - `tests/test_final_answer.py` line updated to assert BOTH hooks fire.

- **M6 — Pricing canonical rule contradiction.**
  - **Decision.** Single, unambiguous rule: `_estimate_cost` returns `float | None`; `gen_ai.usage.cost` written iff non-`None`; `AgentTrace.cost` roll-up skips absent spans; NO `(0.0, 0.0)` fallback anywhere.
  - **Where it changed.**
    - §0 C2 subsection now explicitly documents the canonical pricing rule (was implicit before).
    - §2 outline (section 5) rewritten: was "return (0.0, 0.0) but DO NOT write gen_ai.usage.cost" (contradictory); is now "longest-prefix match... if no match, return None (omit gen_ai.usage.cost attribute). LOCAL_PROVIDERS and unknown models both return None — there is NO `(0.0, 0.0)` fallback".
    - §7 algorithm unchanged (already correct); now explicitly cross-referenced from §2.
    - §13 cross-cutting risks #8 binds M6 to the canonical rule.

- **M8 — MCP exception classes undeclared.**
  - **Decision.** Add `MCPConnectionError` and `MCPProtocolError` to §2's exceptions block (subclasses of `AgentError`). §4's error-handling table already references them; §2 was the gap.
  - **Where it changed.**
    - §2 outline section 6 now lists five exception classes: `AgentError`, `AgentCancel`, `ToolNotFoundError`, `MCPConnectionError`, `MCPProtocolError`. Both MCP classes documented as subclasses of `AgentError` with one-line rationale.
    - §2 outline `run_async` loop now catches `(MCPConnectionError, MCPProtocolError)` separately from `ToolNotFoundError` and `CancelledError`, marks the server broken, fires `on_error`, and re-raises wrapped in `AgentError`.
    - §2 outline `_dispatch_tool` documentation lists which exception each failure mode raises.
    - §13 T3 expanded: "AgentError, AgentCancel, ToolNotFoundError, MCPConnectionError, MCPProtocolError"; test description requires subclass checks for both MCP classes.
    - §11 unit-test table `test_exceptions.py` line now lists all five exception classes.
    - §13 cross-cutting risks #9 binds M8 to T3.

- **M10 — Integration-test skipif KeyError.**
  - **Decision.** Two changes: (1) `PROVIDER_KEY_ENV` is ALWAYS accessed via `.get(provider, ())` — never subscripted — so ollama and vertex (no API key) skip cleanly without KeyError. (2) Per-scenario skipif markers replace the module-level `pytest.skip(..., allow_module_level=True)` so a missing provider key skips only the scenarios that depend on it.
  - **Where it changed.**
    - §0 added new "C4. Integration-test skipif" subsection documenting the decision.
    - §11 integration-test section completely rewritten. New `tests/integration/conftest.py` exposes `PROVIDER_ENV_SKIPIF` and `ANY_LLM_MODEL_SKIPIF` markers. `tests/integration/test_e2e_anyllm.py` applies `@PROVIDER_ENV_SKIPIF` to all scenarios except `test_on_error_real_failure_mode` (which uses `@ANY_LLM_MODEL_SKIPIF` because it intentionally uses an invalid model id).
    - The "Why per-scenario skipif" paragraph in §11 explains both bugs fixed: the KeyError on missing provider, AND the module-level skip that pulled in all sibling scenarios.
    - §13 T16 description updated to reflect per-scenario markers via conftest helpers.
    - §13 cross-cutting risks #10 binds M10 to T16 + the `.get()` rule.

## New majors (now closed)

The three round-2 new majors were the same as the four partially-fixed round-1 majors above:

- **M4 (new-major label).** Same as M4 above. Closed.
- **M8 (new-major label).** Same as M8 above. Closed.
- **M10 (new-major label).** Same as M10 above. Closed.
- **Structural minor in the skipif code (same area as M10).** Module-level `pytest.skip(..., allow_module_level=True)` ran at collection time, contradicting the "independently skippable" claim. Fixed by moving skipif logic to per-scenario markers built in `conftest.py`. See M10 above.

## Minors (cleaned up)

- **Context type/field inconsistency (§2 vs §5).** Picked the §5 SimpleNamespace shape as canonical and updated §2 to match. The §2 outline section 8 now reads: `Context type — SimpleNamespace(span, trace, agent, tool_call: ToolCall | None, tool_result: Any, message: Any, error: BaseException | None, turn: int)` with an explicit "CANONICAL — matches §5" marker. §5 hook-table Context block was already SimpleNamespace but with less-precise typing; upgraded to explicit field type annotations.

- **Two disagreeing `__all__` lists (§2 vs §10).** Reconciled to §10's list as canonical (it's the dedicated Public-API-surface section). §2 outline §3 now matches §10's ordering and includes the test-helper exports `PROVIDER_KEY_ENV` and `PROVIDER_EXTRA_ENV` (added so integration tests can import them without private-name access). §10's list was also updated to add the two test-helper exports.

- **`_await_coro` referenced but undefined.** Defined in §2 outline section 7 (CallbackRegistry block) as `async def _await_coro(self, coro) -> Any` — a pass-through that wraps a coroutine result so `run_coroutine_threadsafe` gets an awaitable. Documented as making the dispatch_sync call site self-documenting and giving a single place to add error wrapping later if needed.

- **`_dispatch_tool` used but not in §2 outline.** Added to §2's TinyAgent class block as `_dispatch_tool(self, tool_call) -> Any`. Documented with the four exception classes it can raise (ToolNotFoundError, MCPConnectionError, MCPProtocolError, asyncio.CancelledError) and which ones are caught by the §8 loop body.

- **Dead `_return_final_answer` declaration.** Removed from §2's TinyAgent class block. A trailing note documents that the §8 pseudocode inlines the answer capture directly to keep the loop body linear, and that the earlier round-1 declaration was dead code.

- **Three §13 test files missing from §11 inventory.** Added `test_otel_setup.py` (T8), `test_agent_init.py` (T11), and `test_agent_trace.py` (T7) to §11's unit-test table. Each row includes a one-line description of what the test covers, matching the descriptions already in §13.

- **(Implicit minor in §11 §11 inventory line about `__all__`)** — `test_imports.py` description now explicitly notes "matches §10 list" so the canonical-list authority is clear.

## Verification

No background verifiers were spawned for round-2 — every change is a small, local edit that either resolves an explicit review comment or reconciles a documented inconsistency. The cross-cutting risks section (§13) was extended from 6 to 10 items; each new item (#7 M4, #8 M6, #9 M8, #10 M10) is bound to a specific TDD test or TDD task, closing the loop.

## Substantive design changes — 5-10 line summary

1. **`final_answer` now has full hook symmetry** (M4 round-2). It fires BOTH `before_tool_execution` (raw tool_call, args not yet parsed) AND `after_tool_execution` (captured answer in ctx.tool_result). The termination logic runs after `after_tool_execution`. No carve-out, no surprises.
2. **Pricing rule is single and canonical** (M6 round-2). `_estimate_cost` returns `float | None`; the span writer writes `gen_ai.usage.cost` iff non-None; `AgentTrace.cost` roll-up skips absent spans; local providers and unknown models both return None — no `(0.0, 0.0)` fallback exists anywhere in the plan.
3. **Both MCP exception classes are declared in §2** (M8 round-2). `MCPConnectionError` (subprocess death, EOF on stdin) and `MCPProtocolError` (invalid UTF-8 / malformed JSON-RPC) are subclasses of `AgentError` and listed in the §2 exceptions block alongside the other three classes. §4's error table and §8's loop body use them consistently.
4. **Integration-test skipif is per-scenario and KeyError-proof** (M10 round-2). The module-level `pytest.skip(..., allow_module_level=True)` is gone. `tests/integration/conftest.py` builds `PROVIDER_ENV_SKIPIF` / `ANY_LLM_MODEL_SKIPIF` markers using `PROVIDER_KEY_ENV.get(provider, ())`. Each scenario applies the appropriate marker; `test_on_error_real_failure_mode` opts out of the provider-key check via `ANY_LLM_MODEL_SKIPIF` because it uses an invalid model id.
5. **§2 outline is now self-consistent** (minors). `Context` type matches §5 exactly; `__all__` matches §10 exactly; `_await_coro` and `_dispatch_tool` are declared where they're used; `_return_final_answer` is removed (dead); all three round-1 test files missing from §11's inventory are now there.
6. **§0 conflict resolutions expanded** with C3 (final_answer symmetry) and C4 (integration skipif) subsections that record the design decision, the rejected trade-off, and the anchor file for each. The plan's reasoning is now self-contained — a fresh peer reviewer can read §0 alone to learn what was decided and why.

---

# Revision log — round 3

> Run: `run-20260706-120000-abc123`
> Reviewer verdict (round 3): `block` (4 new majors, 3 minors; round-2 issues re-verified closed)
> Outcome: all 4 new majors + 3 minors addressed in `plan.md`. No C1–C4 re-litigation; no round-2 fix (M4, M6, M8, M10, m_Context_consistency, m_doubled_all, m_await_coro_undefined, m_dispatch_tool_undeclared, m_return_final_answer_dead, m_test_inventory_gaps) re-touched.

## Round-2 issues audit (re-verified closed)

- **M4 — `final_answer` hook symmetry:** closed in round 2. Verified still closed in round 3 by reviewer. The round-2 evidence (§0 C3, §5 hook table "INCLUDING final_answer", §8 rule (c), §13 T12a acceptance test, §13 cross-cutting risk #7) is unchanged. The round-3 reviews confirms BOTH `before_tool_execution` and `after_tool_execution` fire on `final_answer`. No edits in round 3 affect M4.
- **M6 — Pricing canonical rule:** closed in round 2. Verified still closed by reviewer. The round-2 evidence (§0 C2 explicit canonical rule; §2 sections 5 + 15; §7 algorithm steps 3 + 4; §13 cross-cutting risk #8) is unchanged. `_estimate_cost` returns `float | None`; no `(0.0, 0.0)` fallback. No edits in round 3 affect M6.
- **M8 — MCP exception classes declared once:** closed in round 2. Verified still closed. The round-2 evidence (§2 section 6 lists all five exception classes; §4 error table references MCPConnectionError + MCPProtocolError; §2 run_async catches the pair; T3 covers subclasses) is unchanged. No edits in round 3 affect M8.
- **M10 — Integration-test skipif (`.get()` + per-scenario markers):** closed in round 2. Verified still closed. The round-2 evidence (§0 C4, §11 conftest with `PROVIDER_KEY_ENV.get(provider, ())`, per-scenario markers, `test_on_error_real_failure_mode` opt-out via `ANY_LLM_MODEL_SKIPIF`) is unchanged. No edits in round 3 affect M10.
- **Minors (round 2, all closed):**
  - `m_Context_consistency` — §2 §8 declares `Context` as SimpleNamespace with explicit "CANONICAL — matches §5" marker; §5 hook-table block has matching shape with type annotations. Reused in round-3 §2 §8 with `tool_call: ToolCall | None` annotation now backed by the new TypedDict.
  - `m_doubled_all` — §2 §3 and §10 list identical exports. Round-3 adds `"ToolCall"` to BOTH lists (§2 §3 and §10's export block) so the new TypedDict is discoverable via the public import.
  - `m_await_coro_undefined` — declared in §2 §7; referenced from §5 dispatch_sync.
  - `m_dispatch_tool_undeclared` — declared in §2 §15 with exception-raise documentation.
  - `m_return_final_answer_dead` — explicitly removed in round 2; the §8 pseudocode inlines answer capture.
  - `m_test_inventory_gaps` — `test_otel_setup.py` (T8), `test_agent_init.py` (T11), `test_agent_trace.py` (T7) all present.

## New majors (now closed)

- **M1 (continue vs break) — closed.** §8 pseudocode line 635 was the round-2 word `continue` after `final_answer` capture; rule (b) and §2 outline required `break` so subsequent non-final_answer tool calls in the same turn do NOT execute.
  - **Where it changed.** `plan.md` §2 §15 run_async outline loop body now uses `break` (was `continue`). `plan.md` §8 pseudocode now uses `break` at the equivalent line; rule (b) wording unchanged; rule (e) updated to acknowledge the second-`final_answer` warn-and-continue branch is defensively unreachable after `break`; new rule (f) covers the tool_choice retry. `plan.md` §13 cross-cutting risk #11 added: "`final_answer` short-circuits via `break` not `continue`". T12a acceptance criteria explicitly require `[final_answer, calculate]` in one turn to result in ONLY `final_answer` executing.

- **M2 (hook signature mismatch) — closed.** §2 §7 said `(ctx, **kwargs) -> Context | None` while §5 said `Callable[[Context], Awaitable[None] | None]`. §5's dispatch never supplied kwargs and discarded the return value, so §2's wording was wrong.
  - **Where it changed.** `plan.md` §2 §7 now reads: "Sync hook: `Callable[[Context], None]`; Async hook: `Callable[[Context], Awaitable[None]]`. Return value is DISCARDED. No **kwargs. No Context return." Explicit "CANONICAL — matches §5" marker preserved. `plan.md` §5 signature block rewritten to "Sync hook: `(ctx) -> None`; Async hook: `(ctx) -> Awaitable[None]`; return value is DISCARDED; no **kwargs; no Context return" with a note that the previous `(ctx, **kwargs) -> Context | None` was leftover upstream boilerplate. Hook-table "Fires on" column unchanged (no signature in the table). The single canonical signature is now readable from §2 alone. `plan.md` §13 cross-cutting risk #12 cites the canonical signature and notes the round-3 M3 dict-storage closes M2's mechanical "no kwargs" property.

- **M3 (CallbackRegistry storage model) — closed.** Three mutually-exclusive storage shapes co-existed: §2 `__slots__ = ("_hooks", "_loop")` + `register_*` methods; §5 user API `cb.before_llm_call.append(fn)` (attribute storage); §5 dispatch `getattr(self, name)`.
  - **Where it changed.** `plan.md` §0 added new C5 subsection (storage model is `self._hooks: dict[str, list[Callable]]` keyed by hook name; `__slots__ = ("_hooks", "_loop")` stays; registration via `register_*` methods only; dispatch via `self._hooks.get(name, ())` — never `getattr`). Rejected the attribute-storage alternative explicitly. `plan.md` §2 §7 rewritten: `_HOOK_NAMES` tuple, dict initialization, register methods, dispatch via dict lookup. `plan.md` §5 rewritten: registration API shows `register_before_llm_call(my_fn)` (was `cb.before_llm_call.append(my_fn)`); dispatch_sync and dispatch_async bodies show `self._hooks.get(name, ())` (was `getattr(self, name)`). `plan.md` §13 cross-cutting risk #12 added: dict-backed storage, attribute form MUST raise `AttributeError`. T6 acceptance criteria expanded with a negative regression test asserting `cb.before_llm_call` raises `AttributeError`. T12c test asserts no attribute-style registration is used. T15b test asserts no `cb.before_llm_call.append(...)` form in `examples/`. T15c README uses `cb.register_before_llm_call(...)`. The single canonical mechanism is: `register_*` writes to `self._hooks[name]`; dispatch reads from `self._hooks.get(name, ())`; tests assert both directions.

- **M4 (tool_choice retry missing from §8) — closed.** §13 cross-cutting risk #2 + T12a acceptance criteria required an empty-response retry with `tool_choice="auto"`; the §8 pseudocode had no such logic.
  - **Where it changed.** `plan.md` §8 pseudocode has a new top-of-loop retry branch: when a response is empty (no tool_calls AND no assistant text) under `tool_choice="required"`, flip `tool_choice_for_next = "auto"` and re-issue the call without appending the empty response to `messages`. If the retry also yields an empty response, raise `AgentError` (which the outer loop's `on_error` arm catches). Retry re-arms every turn (per-turn budget of 1 retry). Counts toward `max_turns` via the outer iteration counter (no separate turn). New rule (f) at §8 documents this. `plan.md` §2 §15 run_async outline loop body now mirrors §8 verbatim. `plan.md` §13 cross-cutting risk #2 expanded from "fallback" (vague) to "tool_choice="required" empty-response retry" with explicit semantics. T12a acceptance criteria include the retry scenario. T12a-cross (new task) writes `tests/test_tool_choice_retry.py` for cross-cut coverage with three scenarios: required → auto retry succeeds with trailing text; required → auto retry succeeds with tool_call; required → auto retry also empty → AgentError.

## Minors (cleaned up)

- **m5 (T1 `src/__init__.py`):** closed. T1 file list now contains `tinyagent.py` (stub at repo root) and explicitly excludes `src/__init__.py`. §3 layout diagram updated (`tinyagent.py` at root, no `src/`). §0 C6 subsection documents the flat-layout decision.

- **m6 (`ToolCall` undeclared):** closed. §2 §2 imports add `from typing import TypedDict`. §2 §8 declares `class ToolCall(TypedDict): id: str; type: Literal["function"]; function: dict` so users can `from tinyagent import ToolCall` for type hints. §2 §3 `__all__` and §10's Public API surface both include `"ToolCall"` (consistent). §2 §8 `Context.tool_call: ToolCall | None` annotation is now backed by a real type. §13 T7 expanded: the test asserts the TypedDict shape via `from tinyagent import ToolCall`. §13 cross-cutting risk #12 references the canonical signature and the TypedDict-for-public-discoverability convention.

- **m7 (pyproject `package-dir`):** closed. §0 C6 picks flat layout: `[tool.setuptools] py-modules = ["tinyagent"]` with `tinyagent.py` at the repo root — no `package-dir` mapping, no `packages.find`, no `src/` directory. §3 layout diagram and the inline `pyproject.toml` snippet updated. §13 T15a description locks the flat pyproject.toml. §13 cross-cutting risk #13 binds the layout decision. T1 does not create `src/__init__.py`.

## Verification

Two structural verifiers ran during the round-3 revision:

- **Loop-body consistency check.** Verified that the round-3 rewrites of §2 §15 run_async and §8 pseudocode use the SAME `break` / retry logic: each top-of-turn retry block matches; each tool_calls iterator uses `break` after first `final_answer`; the trailing defensive `if tool_name == "final_answer": continue` is present in both with identical wording. No divergence between outline and pseudocode.

- **CallbackRegistry surface check.** Verified that all four files that previously referenced the old API or old dispatch shape are now consistent: §0 C5, §2 §7 (storage + signature), §5 (registration API + dispatch body), §13 T6/T12c/T15b/T15c acceptance (no attribute-style usage in code or examples). No `cb.before_llm_call.append(fn)` remains anywhere in the plan. No `getattr(self, name)` remains anywhere in §5 dispatch.

## Substantive design changes — 5-10 line summary

1. **`final_answer` short-circuits via `break`, not `continue`** (round-3 M1). The first `final_answer` capture is followed by `break`, exiting the tool_calls loop cleanly so subsequent non-`final_answer` tool calls in the same turn never execute. Rule (b) is now enforced by the pseudocode and the §2 outline in lockstep.
2. **Hook signature is single, narrow, no-frills** (round-3 M2). Sync `(ctx) -> None` or async `(ctx) -> Awaitable[None]`; return value discarded; no `**kwargs`. The leftover upstream boilerplate `(ctx, **kwargs) -> Context | None` is purged everywhere (§2 §7, §5).
3. **`CallbackRegistry` is dict-backed with `register_*` methods** (round-3 M3). Storage is `self._hooks: dict[str, list[Callable]]`; registration is `cb.register_before_llm_call(fn)`; dispatch is `self._hooks.get(name, ())`. No `getattr`. No attribute-style API. T6 has a negative test asserting `cb.before_llm_call` raises `AttributeError` to lock this in.
4. **`tool_choice="required"` empty-response retry is explicit** (round-3 M4). §8 pseudocode (and §2 outline, in lockstep) flip `tool_choice_for_next = "auto"` once per turn when the response is empty under `required`; if the retry also returns empty, raise `AgentError`. Retry re-arms each turn; counts toward `max_turns`. New `tests/test_tool_choice_retry.py` covers three scenarios. The §13 cross-cutting risk #2 wording was upgraded from vague "fallback" to concrete retry semantics.
5. **`ToolCall` is a `TypedDict`** (round-3 m6). Declared in §2 §8, exported via `__all__`, used as the annotation for `Context.tool_call`. Round-3 m6 declared it once; tests in T7 import it from public surface to assert discoverability.
6. **Package layout is flat, single source of truth** (round-3 m5 + m7). `tinyagent.py` lives at the repo root. `[tool.setuptools] py-modules = ["tinyagent"]` — no `src/`, no `package-dir`, no `packages.find`. T1 doesn't create `src/__init__.py`. T15a locks the pyproject.toml.
7. **§0 (Conflict Resolutions) now has C1–C6** (added C5 storage model and C6 flat layout). Each subsection records the decision, the rejected trade-off, and the anchor. The plan remains self-contained — a fresh reviewer can read §0 alone to understand every architectural choice.
8. **§13 cross-cutting risks grew from 10 to 13 items.** New items #11 (round-3 M1 `break`), #12 (round-3 M3 dict storage + M2 signature + m6 ToolCall), and #13 (round-3 m5+m7 flat layout) each bind a fix to a TDD test (T12a, T6, T15a) so the implementer cannot silently regress.
9. **No round-1 or round-2 issue was re-litigated.** All round-2 closures (M4, M6, M8, M10, six minors) remain intact and the §13 / §11 / §2 cross-references still match. The plan's reasoning can be re-read end-to-end: §0 is decision space, §2 is the file outline, §3 is layout, §5 is callback surface, §7 is pricing, §8 is the loop, §11 is testing, §13 is task ordering + risk register.
10. **T12 split widened.** A new T12a-cross task (and its `tests/test_tool_choice_retry.py`) was added so the tool_choice retry has dedicated test coverage without bloating T12a's LOC budget (T12a stays ~180 LOC). T6, T7, T15b, T15c, T15a acceptance criteria were also extended to keep the round-3 changes in lockstep with the TDD ordering.