## Status

confirmed_by_user: true
confirmed_at: 2026-07-06T12:00:00Z

## Locked decisions (from user confirmation)

1. **Implementation approach:** Fork and re-package github.com/mozilla-ai/tinyagent; lift its core into a clean distribution under our own Apache-2.0 name.
2. **"Single-file install via pip" means:** Literally one Python file. The package's runtime source is a single `tinyagent.py` (or equivalent) that `pip install` resolves to. Repo may contain supporting files (pyproject.toml, LICENSE, README, tests, examples) but the runtime is one file.
3. **MCP transports:** stdio only.
4. **OTel export destination:** Pluggable via standard OTel SDK env vars — `OTEL_EXPORTER_OTLP_ENDPOINT` and friends. Whatever exporter the user configures works (Console, OTLP/HTTP, OTLP/gRPC, etc.).
5. **Callback hook signatures:** Canonical 5-hook set — `before_llm_call`, `after_llm_call`, `before_tool_execution`, `after_tool_execution`, `on_error`.
6. **A2A serving:** No, skip it.
7. **any-llm provider coverage:** Fully open — no opinion. Users can pass any provider/model string; tests cover a reference trio but the package makes no opinion.
8. **Repository state:** /Users/guilhermegrahl/tinyagent is the target. Build in place.
9. **Testing strategy:** Unit tests + integration test gated by `ANY_LLM_TEST_MODEL` env var (runs when credentials are present, skipped otherwise).
10. **Example tools to ship:** `calculate` (safe expression evaluator), `http_get`, `final_answer`.

## Preliminary spec

### Scope

Build a self-contained ReAct agent package named `tinyagent` (working repo at
`/Users/guilhermegrahl/tinyagent`) that follows the mozilla-ai/tinyagent
architecture. The agent:

- Uses **any-llm** as the model provider abstraction (multi-provider: OpenAI,
  Anthropic, Ollama, etc., without vendor lock-in).
- Supports **native MCP tools** (Model Context Protocol) over stdio. The agent
  loads tool definitions from one or more stdio MCP servers at runtime.
- Emits **OpenTelemetry traces** via the standard OTel SDK (pluggable exporter
  configured by `OTEL_EXPORTER_OTLP_ENDPOINT` and friends), covering every LLM
  call and tool execution, including token counts (prompt / completion / total)
  and an estimated cost attribute when the provider/model pricing is known.
- Exposes a **callback system** with the canonical 5-hook set
  (`before_llm_call`, `after_llm_call`, `before_tool_execution`,
  `after_tool_execution`, `on_error`) that lets users register guardrails.
- Defines an optional **`final_answer`** tool that the model can call to end
  the loop cleanly with a structured final answer.
- Ships as a **single Python file** at the heart of a pip-installable package.
- Targets **Python 3.11+**, ships under **Apache-2.0**, installs with a
  single `pip install tinyagent` command.

The agent loop is the canonical ReAct pattern: `Thought → Action → PAUSE →
Observation → … → Answer`, but with native function calling (no text parsing).

A2A serving is explicitly out of scope.

### Success criteria

1. **Single-command install.** `pip install tinyagent` (from this repo or PyPI)
   succeeds in a clean venv on Python 3.11, 3.12, and 3.13.
2. **ReAct loop works end-to-end.** A runnable example with at least one local
   stdio MCP server (e.g. a calculator MCP server) and one user-defined Python
   tool completes a multi-turn task via native function calling, terminating
   via `final_answer`.
3. **Tracing is real.** Running the example with `OTEL_EXPORTER_OTLP_ENDPOINT`
   set (or whatever exporter the user wires) produces traces that include:
   - One span per LLM call with `gen_ai.usage.input_tokens`,
     `gen_ai.usage.output_tokens`, and (when known) `gen_ai.usage.cost`.
   - One span per tool execution with tool name, arguments, and result.
   - One parent "agent run" span covering the whole interaction.
4. **Callbacks are usable as guardrails.** A user can register a
   `before_tool_execution` callback that raises, and the loop halts immediately
   without losing trace context.
5. **`final_answer` works.** When the model calls `final_answer`, the loop
   exits and returns the answer as a typed object.
6. **Multi-provider.** Switching providers/models requires only a one-line
   config change (the any-llm model string).
7. **License + version surface.** `LICENSE` contains Apache 2.0 text.
   `pyproject.toml` declares the license. Package version is `0.1.0`.
8. **README + runnable example.** A README documents install, configuration,
   tool registration, MCP stdio setup, tracing setup, and callback registration,
   with at least one end-to-end runnable example using `calculate` + `http_get`
   + `final_answer`.
9. **Single-file runtime.** The package's importable source is a single Python
   file. Repo may contain tests, examples, docs, build config.

### Blast radius

- **New repo (target).** /Users/guilhermegrahl/tinyagent is currently a clean
  repo with one init commit (`fb4acfc`). The package will be created in place.
  Nothing in this repo is touched elsewhere — rollback is `git reset --hard
  fb4acfc`.

### Assumptions

- Native function calling (not text parsing) — the model emits actions via the
  provider's tool-use API and we feed observations back as tool messages.
- Three exit conditions active together: `final_answer` tool call, `max_turns`
  hard cap (default 10), and empty tool-calls → trailing text answer.
- The fork preserves Apache-2.0 attribution to upstream
  `mozilla-ai/tinyagent`. LICENSE retains their notice where required.
- any-llm is unpinned — we add it as a dependency and document the trio
  (OpenAI / Anthropic / Ollama) in the README; tests gate on
  `ANY_LLM_TEST_MODEL`.

### Interpretation & additions

- Keep-last-N-turns pruning (default N=10) to avoid context overflow.
- Built-in pricing table for cost estimation, overridable via callback/config.
- `tinyagent` package surface: `TinyAgent`, `tool`, `MCPServer` (stdio),
  `CallbackRegistry`, `AgentTrace`.
- README example wires a calculator MCP server (stdio) for the demo walkthrough.

---

*Confirmed by user 2026-07-06. Source: `/Users/guilhermegrahl/rig/agent.md` §2 "Option C".*