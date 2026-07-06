# tinyagent

A single-file [ReAct](https://arxiv.org/abs/2210.03629) agent forked from
[mozilla-ai/tinyagent](https://github.com/mozilla-ai/tinyagent) (Apache-2.0).

- **Single file at the core.** The package's runtime source is one Python file
  (`tinyagent.py`) at the repo root — vendor it, read it, patch it.
- **Any-llm provider coverage.** OpenAI, Anthropic, Mistral, Groq, Azure,
  Hugging Face, Gemini, Vertex, Ollama — switch with a one-line config change.
- **Native MCP tools over stdio.** Attach drop-in servers via `mcp` 1.28.1.
- **OpenTelemetry tracing.** Standard semconv token attributes plus a custom
  cost attribute. The library is *passive* — the host application wires the
  exporter.
- **Canonical 5-hook callback surface.** `before_llm_call`, `after_llm_call`,
  `before_tool_execution`, `after_tool_execution`, `on_error`, registered via
  `register_*` methods.
- **Apache-2.0.** See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Python 3.11+ is required.

## Install

```bash
pip install tinyagent          # from PyPI (future)
pip install -e .               # from this checkout
```

## Quickstart

A complete, single-file example using `TinyAgent` + `AgentConfig` +
`final_answer`:

```python
from tinyagent import TinyAgent, AgentConfig, calculate, final_answer

config = AgentConfig(
    model="openai:gpt-4o-mini",
    instructions="You are a helpful assistant. Answer via final_answer.",
    tools=[calculate, final_answer],
    mcp_servers=[],
)

agent = TinyAgent(config)
answer = agent.run("What is (17 * 23) + 4?")
print(answer)
```

`agent.run(...)` is the synchronous entry point; it drives the full ReAct loop
(LLM call → tool execution → repeat) until the model calls `final_answer` (or
returns trailing assistant text). Inside an event loop, use the async form:

```python
answer = await agent.run_async("What is (17 * 23) + 4?")
```

`final_answer` is auto-attached to every agent, but listing it in `tools`
documents intent and keeps the example self-describing. `calculate` uses
[`simpleeval`](https://pypi.org/project/simpleeval/) for safe arithmetic
(no raw `eval`).

## Callbacks

`tinyagent` exposes exactly **five** callback hooks. Register handlers with the
`register_*` methods on a `CallbackRegistry` and pass the registry through
`AgentConfig`:

```python
from tinyagent import TinyAgent, AgentConfig, CallbackRegistry, final_answer

cb = CallbackRegistry()

def log_llm_call(ctx):
    print(f"[turn {ctx.turn}] calling the model")

async def audit_tool(ctx):                       # async hooks work too
    name = ctx.tool_call["function"]["name"]
    print(f"about to run tool: {name}")

cb.register_before_llm_call(log_llm_call)
cb.register_before_tool_execution(audit_tool)

config = AgentConfig(
    model="openai:gpt-4o-mini",
    instructions="Answer via final_answer.",
    tools=[final_answer],
    mcp_servers=[],
    callbacks=cb,
)
agent = TinyAgent(config)
agent.run("Say hello.")
```

The full hook set:

| Hook | Fires on | `Context` field populated |
|---|---|---|
| `register_before_llm_call` | every LLM call | `turn`, `span` |
| `register_after_llm_call` | every LLM call | `message` |
| `register_before_tool_execution` | every tool call, **including `final_answer`** | `tool_call` (raw, args not yet parsed) |
| `register_after_tool_execution` | every tool call, **including `final_answer`** | `tool_result` |
| `register_on_error` | any exception escaping the loop body | `error` |

Notes:

- **`register_*` is the only registration API.** The attribute-style form
  `cb.before_llm_call.append(fn)` is **not** supported and raises
  `AttributeError`. See [decision C5](docs/decisions.md#c5--callbackregistry-storage-model).
- **`final_answer` fires both tool hooks.** There is no carve-out for the
  termination tool — `before_tool_execution` sees the raw call and
  `after_tool_execution` sees the captured answer. See
  [decision C3](docs/decisions.md#c3--final_answer-hook-symmetry).
- **Both sync and async handlers are supported.** They work identically under
  `agent.run()` (sync) and `agent.run_async()` (async).
- **Callbacks observe; the agent re-raises.** `on_error` is observability-only —
  it cannot swallow an exception. To abort the loop from a hook, raise
  `AgentCancel`, which propagates out of `run()` / `run_async()`.

## MCP stdio tools

Register a stdio MCP server with `agent.add_mcp_server(server)`, which returns
an async context manager. The synthesised tool callables are attached to the
agent's dispatcher for the lifetime of the context:

```python
import sys
from tinyagent import TinyAgent, AgentConfig, MCPServer, final_answer

config = AgentConfig(
    model="openai:gpt-4o-mini",
    instructions="Use the MCP tools, then call final_answer.",
    tools=[final_answer],
    mcp_servers=[],
)
agent = TinyAgent(config)

server = MCPServer(
    name="calc",
    command=sys.executable,
    args=["examples/inproc_mcp_echo.py"],   # your stdio MCP server script
)

async def main():
    async with agent.add_mcp_server(server):
        return await agent.run_async("Compute (17 * 23) + 4 with the MCP tools.")
```

Only the **stdio** transport is supported (SSE and streamable-HTTP are dropped).
The subprocess is spawned with `start_new_session=True` so it can be cleanly
killed on cancellation. See
[`examples/calculator_mcp_stdio.py`](examples/calculator_mcp_stdio.py) for a
runnable end-to-end script.

## Tracing (OpenTelemetry)

`tinyagent` is a **library**: it acquires a named tracer and emits spans, but it
never calls `trace.set_tracer_provider(...)` and never wires an exporter. If no
provider is configured, spans go to a no-op tracer at zero cost. The **host
application** installs the provider — do this *before* constructing the agent:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# Now build and run the agent — its spans route to your exporter.
from tinyagent import TinyAgent, AgentConfig, http_get, final_answer
```

`tinyagent` declares only `opentelemetry-api`; install whichever exporter your
back-end needs (`opentelemetry-sdk`, `opentelemetry-exporter-otlp-*`). You can
also use the standard `opentelemetry-instrumentation` autoconfigure entry point
and `OTEL_*` env vars instead of wiring a provider by hand.

Span hierarchy:

```
invoke_agent
├── call_llm     {gen_ai.usage.input_tokens, gen_ai.usage.output_tokens,
│                 gen_ai.usage.cost (only when a price is known)}
├── execute_tool {gen_ai.tool.name, gen_ai.tool.args}
└── execute_tool {gen_ai.tool.name=final_answer}
```

`gen_ai.usage.cost` is a **custom, non-standard** USD total shipped alongside
the standard token attributes. It is *omitted* when the model's price is unknown
or the provider is local — cost is never reported as `$0`. See
[decision C2](docs/decisions.md#c2--otel-cost-attribute). A runnable OTLP demo
lives in [`examples/tracing_otlp.py`](examples/tracing_otlp.py).

## Multi-provider

Switching providers is a **one-line config change** — set `model` to a
`provider:model` string and supply the matching API key in the environment:

```python
config = AgentConfig(model="openai:gpt-4o-mini",         ...)  # OpenAI
config = AgentConfig(model="anthropic:claude-3-5-sonnet", ...) # Anthropic
config = AgentConfig(model="groq:llama-3.1-70b",          ...) # Groq
config = AgentConfig(model="ollama:llama3",               ...) # local Ollama
```

The provider prefix (before the first `:`) selects the any-llm backend. The
required environment variable per provider is exported as
`tinyagent.PROVIDER_KEY_ENV` (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GROQ_API_KEY`); local providers such as `ollama` need no key. Provider-specific
extras (e.g. Vertex's `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`) are in
`tinyagent.PROVIDER_EXTRA_ENV`.

## Examples

Runnable scripts live in [`examples/`](examples/):

- [`calculator_mcp_stdio.py`](examples/calculator_mcp_stdio.py) — attach a stdio
  MCP server and run a multi-turn arithmetic task.
- [`http_demo.py`](examples/http_demo.py) — custom `@tool` + built-in `http_get`
  + `final_answer`.
- [`tracing_otlp.py`](examples/tracing_otlp.py) — wire an OTLP exporter and ship
  spans.

## Design decisions

The six conflict resolutions that shaped this fork (callback hook set, cost
attribute, `final_answer` symmetry, integration-test gating, callback storage
model, and package layout) are documented in
[`docs/decisions.md`](docs/decisions.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for the full text
and upstream attribution.
