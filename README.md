# bestie-bench

Benchmarking harness for the **Bestie** AI agent — evaluating **tool calling**, **advice quality**, and **empathy**.

Built on [DeepEval](https://github.com/confident-ai/deepeval) with a model-agnostic design. Swap in any OpenAI- or Anthropic-compatible model.

## Quick Start

```bash
# Install
pip install -e .

# Install default stubs (optional — enables fast, reproducible runs)
bestie-bench install-stubs

# Run with stubs
bestie-bench run --provider openai --model gpt-4o --stubs-dir stubs --runs-per-case 3

# View results
bestie-bench report
bestie-bench compare --limit 5
```

## Tool Calling — Bestie's Real Tools

Bestie uses 7 real tools (mirrored from `apps/web/app/api/chat/tools.ts`):

| Tool | Description |
|------|-------------|
| `get_weather` | Current weather by location |
| `get_journal_entries` | Read recent journal entries |
| `internet_lookup` | Web search via Tavily |
| `schedule_event` | Create a reminder |
| `get_reminders` | List active reminders |
| `update_reminder` | Edit an existing reminder |
| `cancel_reminder` | Cancel one or more reminders |

## Stubbing

Stubs intercept tool calls and return pre-recorded responses. This makes runs:
- **Fast** — no external API calls
- **Reproducible** — same inputs, same outputs every time
- **Deterministic** — no flaky weather or search APIs

```bash
# Install default stubs to stubs/ directory
bestie-bench install-stubs

# Edit stubs (optional — customize responses)
vim stubs/get_weather/London.json

# Run with stubs
bestie-bench run --stubs-dir stubs --verbose
```

Stubs are JSON files at `stubs/<tool_name>/<stub_key>.json`:
```json
{
  "condition": "Overcast",
  "feelsLike": "11°C",
  "humidity": "80%",
  "location": "London, England",
  "temperature": "13°C",
  "wind": "18 km/h"
}
```

## Test Fixtures

Fixtures are YAML files in `fixtures/<axis>/`:

```bash
fixtures/tool_calling/reminders.yaml   # schedule/get/update/cancel reminders
fixtures/tool_calling/weather.yaml     # get_weather
fixtures/tool_calling/journal.yaml      # get_journal_entries
fixtures/tool_calling/internet.yaml     # internet_lookup
fixtures/advice/productivity.yaml       # advice quality
fixtures/empathy/relationships.yaml     # empathy + emotional intelligence
fixtures/empathy/milestones.yaml       # empathy in life transitions
```

## Architecture

```
bestie-bench/
├── fixtures/                   # Test cases (YAML)
│   ├── tool_calling/          # 16 cases covering all 7 Bestie tools
│   ├── advice/
│   └── empathy/
├── stubs/                      # Pre-recorded tool responses (optional)
│   ├── get_weather/           # Stub files per tool
│   ├── schedule_event/
│   └── ...
├── src/bestie_bench/
│   ├── harness.py              # Core eval runner + agent loop
│   ├── stubs/registry.py       # StubRegistry + install_stubs()
│   ├── cases/                 # TestCase models + YAML loader
│   ├── metrics/               # DeepEval wrappers per axis
│   ├── models/                # Unified model client
│   └── reporters/              # Rich tables + JSON output
└── results/                    # JSON outputs per run
```

## CLI Reference

```bash
# Run benchmark
bestie-bench run [options]

# Install/edit stubs
bestie-bench install-stubs [--stubs-dir stubs] [--force]

# View results
bestie-bench report [--results-dir results]

# Compare runs
bestie-bench compare [--results-dir results] [--limit 10]

# List system prompts
bestie-bench prompts
```

## Scoring

| Axis | Metric | Method |
|------|--------|--------|
| Tool calling | `ToolCorrectnessMetric` + `ArgumentCorrectnessMetric` | Deterministic — compare expected vs actual tool calls |
| Advice | `GEval` with custom rubric | LLM-as-judge (GPT-4o by default) |
| Empathy | `GEval` with custom rubric | LLM-as-judge |

Each tool call test case runs the full **agent loop**: model → tool calls → stubbed responses → model → final text.
