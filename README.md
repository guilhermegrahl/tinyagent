# tinyagent

A single-file ReAct agent forked from [mozilla-ai/tinyagent](https://github.com/mozilla-ai/tinyagent) (Apache-2.0).

- **Single file at the core.** The package's runtime source is one Python file (`tinyagent.py`) at the repo root.
- **Any-llm provider coverage.** OpenAI, Anthropic, Mistral, Groq, Azure, Hugging Face, Gemini, Vertex, Ollama — switch with one config change.
- **Native MCP tools over stdio.** Drop-in servers via `mcp` 1.28.1.
- **OpenTelemetry tracing.** Standard semconv tokens + custom cost attribute; the library is passive — the host application wires the exporter.
- **Canonical 5-hook callback surface.** `before_llm_call`, `after_llm_call`, `before_tool_execution`, `after_tool_execution`, `on_error`.
- **Apache-2.0.** See `LICENSE` and `NOTICE`.

## Status

T1 bootstrap complete (this commit). The package skeleton is in place; the
remaining task breakdown (T2–T14) is documented in
`.harness/run-20260706-120000-abc123/plan.md`.

## Install

```bash
pip install tinyagent          # from PyPI (future)
pip install -e .               # from this checkout
```

Python 3.11+ is required.

## Quickstart

```python
from tinyagent import TinyAgent, AgentConfig, calculate, final_answer

config = AgentConfig(
    model="openai:gpt-4o-mini",
    instructions="You are a helpful assistant.",
    tools=[calculate, final_answer],
)

agent = TinyAgent(config)
answer = agent.run("What is 2 + 2?")
print(answer)
```

The full ReAct loop with MCP stdio + OTel tracing + callbacks lands in T11–T15.

## Project layout

```
.
├── LICENSE                  # Apache-2.0 verbatim
├── NOTICE                   # Mozilla.ai upstream attribution
├── pyproject.toml           # flat py-modules install
├── tinyagent.py             # the one runtime file
├── tests/                   # pytest suite
├── examples/                # runnable examples
└── docs/                    # design decisions and notes
```

## License

Apache-2.0. See `LICENSE` and `NOTICE` for the full text and upstream
attribution.
