# Design decisions

This fork of [mozilla-ai/tinyagent](https://github.com/mozilla-ai/tinyagent)
diverges from upstream in six deliberate ways. Each was a genuine conflict —
two or more defensible options — resolved during planning. This document
records the decision, the trade-off that was rejected, and an anchor back to
the authoritative source.

The authoritative source for every decision below is **§0 (Conflict
Resolutions)** of the implementation plan at
`.harness/run-20260706-120000-abc123/plan.md`. Section references such as
"§5" or "§13 T12a" point into that same plan.

---

## C1 — Callback hook set

**Decision.** Collapse to the canonical **five** hooks and add `on_error`:
`before_llm_call`, `after_llm_call`, `before_tool_execution`,
`after_tool_execution`, `on_error`. Upstream ships six hooks (including two
agent-level brackets, `before_agent_invocation` / `after_agent_invocation`);
those two brackets are dropped. Each of the five fires on **every** matching
event in the loop — no "first iteration only" special cases (symmetry rule).

**Rejected trade-off.** *Option B* — keep upstream's six brackets and add
`on_error` — preserves upstream semantics but drifts from the spec's request
for five hooks. *Option C* — keep the six brackets with no `on_error` —
silently drops the error hook, forcing guardrail-as-callback users to wrap
`agent.run()` in their own `try/except`. Option A wins because the agent-level
brackets are strictly weaker than the OTel `invoke_agent` span we already emit,
so bracket coverage is redundant, while `on_error` unlocks dead-letter /
logging / circuit-breaker handlers with no external try/except.

**Anchor.** plan §0 **C1**; `research.md` §D, §I. Symmetry rule resolves
peer-review issue M7.

---

## C2 — OTel cost attribute

**Decision.** Write all three attributes on every `call_llm` span:
`gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` (semconv standard)
plus `gen_ai.usage.cost` — a **custom, non-standard** USD total (input +
output) kept as one rolled-up number. The cost attribute is written **iff**
`_estimate_cost` returns a non-`None` float. Local providers
(`LOCAL_PROVIDERS`) and unknown models both return `None`, so their spans carry
**no** cost attribute — there is no `(0.0, 0.0)` fallback, and absence means
"unknown", never `$0`. `AgentTrace.cost` sums only the spans that actually
carry the attribute.

**Rejected trade-off.** *(a)* Drop cost from spans entirely — loses the only
way to surface cost in standard observability back-ends. *(c)* Drop the custom
cost attribute entirely — but the spec explicitly names it. Shipping the
standard token attributes *alongside* the custom cost total means a strict OTel
collector that drops unknown attributes still keeps the tokens, and cost is
still available via the returned `AgentTrace.cost` roll-up. Splitting cost into
per-direction input/output costs was rejected as over-engineering for v0.1.0.

**Anchor.** plan §0 **C2**; `research.md` §G. Canonical pricing rule detailed in
plan §7; cross-cutting risk #8.

---

## C3 — `final_answer` hook symmetry

**Decision.** The `final_answer` tool call fires **both**
`before_tool_execution` **and** `after_tool_execution`, with **no carve-out**
from the symmetry rule. `before_tool_execution` sees the raw `tool_call` (args
not yet parsed); `after_tool_execution` sees `ctx.tool_result` set to the
captured answer string. The loop's termination logic (set `seen_final_answer`,
return the captured value) runs **after** `after_tool_execution`. `AgentCancel`
raised from either hook still terminates the loop.

**Rejected trade-off.** *Option (b)* — carve `final_answer` out of
`before_tool_execution` because it is "a termination signal, not an external
action" — is intellectually honest about its special role but introduces an
asymmetric rule that surprises users who register `before_tool_execution` to
inspect, log, or sanitize final answers. Option (a) keeps a single predictable
rule: hook before any tool call, hook after, regardless of whether the call is
the loop terminator.

**Anchor.** plan §0 **C3**; peer-review M4 (round-1 partial fix + round-2 new
major). See plan §5 (hook table), §8 (pseudocode), §13 T12a; cross-cutting
risk #7.

---

## C4 — Integration-test skipif

**Decision.** The integration suite's skipif logic lives in
`tests/integration/conftest.py` and is applied **per-scenario** via markers, not
at module level. Two markers are exposed: `PROVIDER_ENV_SKIPIF` (skips a
scenario when `ANY_LLM_TEST_MODEL` is unset **or** the current provider's
required env vars are missing) and `ANY_LLM_MODEL_SKIPIF` (skips only when
`ANY_LLM_TEST_MODEL` is unset). The `test_on_error_real_failure_mode` scenario
uses `ANY_LLM_MODEL_SKIPIF` because it intentionally uses an invalid model id
and needs no provider key. **Hard rule:** `PROVIDER_KEY_ENV` is *always*
accessed via `.get(provider, ())` — the previous `[provider]` subscript raised
`KeyError` for `ollama` and `vertex`.

**Rejected trade-off.** A module-level
`pytest.skip(..., allow_module_level=True)` runs at collection time and skips
the *whole* module. That defeats per-scenario skipif for tests with different
env requirements — a single missing provider key would drag down siblings,
including the key-less `test_on_error_real_failure_mode`.

**Anchor.** plan §0 **C4**; peer-review M10 (round-1 partial + round-2 new
major). See plan §11 (conftest + scenarios), §13 T16; cross-cutting risk #10.

---

## C5 — CallbackRegistry storage model

**Decision.** `CallbackRegistry` uses **one** storage model end-to-end:
dict-backed lists keyed by canonical hook name (`self._hooks: dict[str,
list[Callable]]`, with `__slots__ = ("_hooks", "_loop")`). The user-facing API
is five `register_*` methods — `register_before_llm_call`,
`register_after_llm_call`, `register_before_tool_execution`,
`register_after_tool_execution`, `register_on_error` — each doing
`self._hooks[name].append(fn)` with additive, append-list semantics. Dispatch
(`dispatch_sync` / `dispatch_async`) iterates `self._hooks.get(name, ())` — a
direct dict lookup, **never** `getattr(self, name)`. The old
`cb.before_llm_call.append(fn)` attribute form is **dropped** and raises
`AttributeError` (guarded by a regression test).

**Rejected trade-off.** The attribute-storage form
(`cb.before_llm_call.append(fn)` with per-hook slots and `getattr`-based
dispatch) is shorter at the call site but loses the symmetry between the
user-facing methods and the internal contract — user registration becomes magic
attribute writes, and sync/async dispatch metadata scatters across per-attribute
caches. The dict form is one canonical mechanism: `register_*` writes to
`self._hooks[name]`, dispatch reads from `self._hooks[name]`, and tests assert
both directions.

**Anchor.** plan §0 **C5**; round-3 peer-review M3. See plan §2 section 7, §5
(CB), §6/T6 (`tests/test_callback_registry.py`), §11; cross-cutting risk #12.
The README callback example uses the `register_*` form accordingly.

---

## C6 — Package layout

**Decision.** `tinyagent.py` lives at the **repo root** (flat layout), **not**
under `src/`. The setuptools config is `[tool.setuptools] py-modules =
["tinyagent"]`, which by default looks for `tinyagent.py` at the project root.
There is no `src/` directory, no `src/__init__.py`, no `src/tinyagent/`, and no
`package-dir` mapping. This is a single source of truth that matches the spec
wording — "literally one Python file at the heart of a pip-installable package"
— and removes the earlier contradiction where a listed `src/__init__.py` would
have shadowed the flat-module install.

**Rejected trade-off.** Keeping `src/tinyagent.py` + adding `[tool.setuptools]
package-dir = {"": "src"}` is a valid canonical layout used by many projects,
but it adds a moving part for no benefit at this scale. The flat layout is
canonical for tiny single-file packages (e.g. boltons, halo) and installs
faster in tests.

**Anchor.** plan §0 **C6**; round-3 peer-review minors `m5_src_init` (T1 file
list) + `m7_package_dir` (pyproject.toml). See plan §3 (package layout), §13 T1,
§13 T15a; cross-cutting risk #13.

---

## Traceability

The upstream risk register (`research.md` conflicts research-C1 … research-C10)
is archived in the plan's **§13 "Self-review against research.md"** section,
which maps each upstream risk to its resolution. Decisions **C1–C4** track the
architect's original conflict set; **C5** and **C6** were added in round 3 to
close peer-review M3 and minors m5 + m7 respectively.
