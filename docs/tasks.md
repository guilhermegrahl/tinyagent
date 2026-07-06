# Run Summary — tinyagent (run-20260706-120000-abc123)

> Plan source: `plan.md` §13 (TDD-ordered task breakdown)
> Spec source: `clarified-requirements.md` (locked)

## Total task count: 22

The plan's §13 enumerates 22 individual work items (T1–T16 inclusive, with T12 split into T12a / T12a-cross / T12b / T12c / T12d per peer-review M9, and T15 split into T15a / T15b / T15c).

## Per-task one-line summary

| # | ID | Title (subject) | Key acceptance criterion | Test file |
|---|---|---|---|---|
| 1 | T1 | Repo bootstrap (flat layout, no `src/`) | `import tinyagent` resolves to repo-root `tinyagent.py`; `__all__` matches §10; `[tool.setuptools] py-modules=["tinyagent"]` | `tests/test_imports.py` |
| 2 | T2 | Pricing table + longest-prefix lookup | `_estimate_cost` returns `float \| None`; longest-prefix match; override callable wins; unknown / local → `None` | `tests/test_pricing.py` |
| 3 | T3 | Exception hierarchy | `AgentCancel`, `ToolNotFoundError`, `MCPConnectionError`, `MCPProtocolError` all subclass `AgentError` (M8) | `tests/test_exceptions.py` |
| 4 | T4 | `@tool` decorator + cast helpers | JSON schema from `inspect.signature`; sync + async callables; defaults handled | `tests/test_tool_decorator.py` |
| 5 | T5 | Shipped example tools | `calculate` uses `simpleeval` (no raw `eval`); `http_get` round-trip; `final_answer` round-trip | `tests/test_example_tools.py` |
| 6 | T6 | `CallbackRegistry` (5 hooks, pinned-loop bridge) | `register_*` writes to `self._hooks[name]`; `dispatch_sync` via `run_coroutine_threadsafe`; `dispatch_async` direct; `cb.before_llm_call.append(fn)` raises `AttributeError` (round-3 M3 regression) | `tests/test_callback_registry.py` |
| 7 | T7 | Tracing dataclasses + `ToolCall` TypedDict | `AgentTrace.tokens` / `AgentTrace.cost` sum only spans WITH cost attr; `ToolCall` importable as TypedDict | `tests/test_agent_trace.py` |
| 8 | T8 | OTel setup (library pattern) | `_setup_tracing` does NOT call `set_tracer_provider`; idempotent; returns NoOp tracer when none configured | `tests/test_otel_setup.py` |
| 9 | T9 | Span generation + cost-attribute writer | `call_llm` / `execute_tool` spans; `gen_ai.usage.cost` written iff `_estimate_cost` non-`None`; ABSENT otherwise | `tests/test_otel.py` |
| 10 | T10 | `MCPServer` (stdio-only) | In-process stdio server; tool_not_found returns string; `start_new_session=True`; broken-server marking on EOF; cancel cleanup kills process group | `tests/test_mcp_stdio.py` |
| 11 | T11 | `AgentConfig` + `TinyAgent.__init__` + `call_model` | any-llm client built; `_clients` includes `final_answer`; `request_timeout_s` wraps `asyncio.wait_for` | `tests/test_agent_init.py`, `tests/test_request_timeout.py` |
| 12 | T12a | ReAct loop body (`break` after first `final_answer`, `tool_choice` retry) | `final_answer` short-circuits via `break`; BOTH hooks fire on `final_answer` (M4); empty `tool_calls` under `required` retries once with `auto` (round-3 M4); unknown tool returns string to LLM (loop continues); trailing-text fallback | `tests/test_agent_loop.py` |
| 13 | T12a-cross | Dedicated `tool_choice` retry test | Synthetic LLM returns `tool_calls=[]` under `required`; retry under `auto` issued exactly once; second empty → `AgentError`; non-empty retry takes trailing-text branch | `tests/test_tool_choice_retry.py` |
| 14 | T12b | Pair-preserving prune | Every surviving tool message has its parent assistant; system preserved; empty history no-op | `tests/test_prune.py` |
| 15 | T12c | Sync `run()` wrapper with pinned-loop bridge | Sync `run()` invokes async hooks via pinned loop; sync hooks work; `AgentCancel` propagates; mixed set runs in order; uses ONLY `register_*` API | `tests/test_agent_loop_sync.py` |
| 16 | T12d | `on_error` integration + `AgentCancel` mid-loop | `on_error` fires on every escaping exception; cannot swallow; `AgentCancel` raised from any hook terminates the loop | extend `tests/test_agent_loop.py` |
| 17 | T13 | Per-span cost writer | Cost attribute present when known, OMITTED when unknown / local; `AgentTrace.cost` roll-up skips absent spans | `tests/test_pricing_override.py` |
| 18 | T14 | `add_mcp_server` public async-CM method | Both context-manager form AND explicit register/cleanup form work; `__all__` exports symbol | extend `tests/test_mcp_stdio.py` |
| 19 | T15a | `pyproject.toml` finalization | `pip install -e .` succeeds in fresh venv; `tinyagent.py` resolves as the package module (flat layout — no `src/` shadowing) | `tests/test_examples_run.py::test_pip_install_smoke` |
| 20 | T15b | Example scripts (`calculator_mcp_stdio.py`, `http_demo.py`, `tracing_otlp.py`) | Each example imports cleanly; uses `register_*` API (no `cb.before_llm_call.append(...)` form) | `tests/test_examples_run.py::test_each_example_runs_under_mocked_llm` |
| 21 | T15c | README + `docs/decisions.md` | README callback example uses `cb.register_before_llm_call(...)`; decisions.md cross-links C5 + C6 | n/a (prose) |
| 22 | T16 | Integration test suite (gated `ANY_LLM_TEST_MODEL`) | All six scenarios from §11; per-scenario skipif via `PROVIDER_ENV_SKIPIF` / `ANY_LLM_MODEL_SKIPIF`; `test_on_error_real_failure_mode` uses `ANY_LLM_MODEL_SKIPIF` (NOT provider-key); `PROVIDER_KEY_ENV.get(provider, ())` (no KeyError) | `tests/integration/test_e2e_anyllm.py` |

## Dependency graph

Linear backbone T1 → T2/T3/T4/T5/T6/T7/T8 fan-out → T10/T11 join → T12a → T12b → T12c → T12d / T12a-cross → T13/T14 → T15a → T15b → T15c → T16.

```
T1 (bootstrap) ─┬─→ T2 (pricing table) ───────────────────────────────────────────┐
                ├─→ T3 (exceptions) ─────────────────────┬──────────────────────┐│
                ├─→ T4 (@tool) ─→ T5 (example tools) ────┤──────────────────────┐││
                │                                       └─→ T10 (MCPServer) ──→ T14 (add_mcp_server) ┐
                ├─→ T6 (CallbackRegistry) ──────────────┬─→ T11 (TinyAgent core) ─┐                    │
                │                                       │                          │                    │
                ├─→ T7 (Context/AgentTrace) ────────────┤                          │                    │
                │                                                                      │                    │
                ├─→ T8 (OTel setup) ─→ T9 (span generation) ─────────────────────────┤                    │
                │                                                                      │                    │
                │                                                                      ▼                    │
                │                                                          T12a (loop body)               │
                │                                                                  │                         │
                │                                                                  ├──→ T12a-cross (retry test)
                │                                                                  ├──→ T12b (prune) ──→─────┐
                │                                                                  │                         │
                │                                                                  ▼                         ▼
                │                                                          T12c (sync run() bridge) ──→ T12d (on_error)
                │                                                                  │                         ▲
                │                                                                  ▼                         │
                │                                                          T13 (per-span cost writer) ─────┘
                │                                                                                            ▲
                └────────────────────────────────────────────────────────────────────────────────────────────┴──→ T15a (pyproject)
                                                                                                                       │
                                                                                                                       ▼
                                                                                                                  T15b (examples) ─→ T15c (README) ─→ T16 (integration)
```

### Per-task `addBlockedBy` (parent list)

| Task | Blocked by |
|---|---|
| T1 | — |
| T2 | T1 |
| T3 | T1 |
| T4 | T1 |
| T5 | T4 |
| T6 | T1 |
| T7 | T1 |
| T8 | T1 |
| T9 | T8 |
| T10 | T4 |
| T11 | T6, T7, T9 |
| T12a | T2, T3, T5, T6, T7, T9, T11 |
| T12a-cross | T12a |
| T12b | T12a |
| T12c | T11, T12a, T12b |
| T12d | T12c |
| T13 | T9, T11, T12a |
| T14 | T10, T12a |
| T15a | T1, T13, T14 |
| T15b | T15a, T13, T14 |
| T15c | T15b |
| T16 | T15a, T15b, T15c |

## Parallel opportunities

The plan encodes several explicit parallelizable bands. Up to 3-4 implementer agents can work concurrently on these slices while respecting the linear backbone:

- **Band A (after T1):** T2, T3, T4, T6, T7, T8 all run in parallel — they touch disjoint sections of `tinyagent.py` (sections 5, 6, 9, 7, 8, 13 respectively) and write independent test files. Stubs at the import surface only.
- **Band B (after T4):** T5 starts in parallel with Band A's tail. T5 builds on T4 but doesn't depend on T2/T3/T6/T7/T8.
- **Band C (after T8):** T9 starts; depends on T8 only.
- **Band D (after T4):** T10 starts; depends on T4 only. Long pole (~250 LOC) — kick off early.
- **Junction at T11:** T11 (TinyAgent core) joins T6, T7, T9 — they all must land before T11 begins.
- **T12a is the merge point** — T12a unblocks T12a-cross, T12b, T13, T14. Those four can proceed in parallel:
  - T12a-cross: test-only, no implementation coupling
  - T12b: pruning algorithm, independent
  - T13: cost-attribute wiring at call site
  - T14: `add_mcp_server` ergonomic helper
- **T15a finalizes pyproject** — needs T13, T14 done (both touch `tinyagent.py` symbols that `py_modules` picks up).
- **T15b / T15c / T16 are tail:** T15b builds examples; T15c writes docs; T16 writes integration tests. T15c can start as soon as T15a is in (docs are prose). T16 needs T15b done (integration tests run examples).

## Critical path

The longest dependency chain (sum of estimated LOC for each task):

```
T1 (80) → T4 (120) → T5 (120) → T11 (200) → T12a (180) → T12b (100) → T12c (80) → T15a (50) → T15b (150) → T15c (prose) → T16 (200)
≈ 1280 LOC of implementation, plus prose + tests
```

**Critical path length: 11 tasks long, ~1280 LOC of implementation code** (T1, T4, T5, T11, T12a, T12b, T12c, T15a, T15b, T15c, T16).

**Long-pole tasks (>200 LOC):** T7 (~200), T9 (~250), T10 (~250), T16 (~200). These are the large ones — they should be staffed by senior implementers and T10 should be kicked off first in Band B since it's on the critical path's MCP branch.

## Estimated total work units

- **22 tasks total**
- **6 are large** (>150 LOC each): T7 (200), T9 (250), T10 (250), T11 (200), T12a (180), T16 (200)
- **8 are medium** (50-150 LOC): T1, T2, T3, T4, T5, T6, T12b, T12c, T12d, T13, T14, T15a, T15b
- **1 is prose** (no LOC budget): T15c
- **Rough order of magnitude:** 3-4 implementer agents working in parallel can complete the run in ~6-8 hours assuming each LOC takes ~30-60s including test-write/fix-up. Serial execution: ~12-15 hours.

## Tool-limitation note

The `TaskCreate` tool referenced in the task-tracker agent instructions is not available in this environment, and the `Bash` tool is blocked by the repo-boundary guard. This `run-summary.md` is the file-based handoff that downstream agents (e.g. the harness-loop or implementer dispatcher) should consume to schedule the work. The dependency list and per-task test-file references above are the authoritative graph for downstream dispatch.
