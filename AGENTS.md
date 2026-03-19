# AGENTS.md — bestie-bench

## Project Overview

bestie-bench is a benchmarking harness for evaluating AI agents across three axes:
- **Tool calling** — deterministic checks (correct tool + arguments)
- **Advice quality** — LLM-judged (practicality, soundness, calibration)
- **Empathy** — LLM-judged (emotional intelligence, warmth, non-judgment)

## Directory Structure

```
bestie-bench/
├── fixtures/                   # Test cases (YAML, 16 tool_calling cases)
│   ├── tool_calling/          # Reminders, weather, journal, internet
│   ├── advice/
│   └── empathy/
├── stubs/                      # Pre-recorded tool responses (optional)
│   ├── get_weather/           # Edinburgh.json, London.json, etc.
│   ├── schedule_event/
│   ├── get_reminders/
│   └── ...
├── src/bestie_bench/
│   ├── harness.py              # Harness class + HarnessConfig dataclass
│   ├── cli.py                 # Click CLI: run, install-stubs, report, compare
│   ├── stubs/registry.py      # StubRegistry, install_stubs(), stubbed_harness()
│   ├── cases/                 # TestCase models + YAML loader
│   │   ├── models.py         # Axis, TestCase, TestResult, AggregateResult
│   │   └── loader.py         # load_fixtures(), iter_fixtures()
│   ├── metrics/               # DeepEval wrappers per axis
│   │   ├── tool_calling.py   # evaluate_tool_case()
│   │   ├── advice.py         # build_advice_metric() + GEval
│   │   └── empathy.py        # build_empathy_metric() + GEval
│   ├── models/               # Unified model client
│   │   └── client.py        # ModelClient ABC + OpenAI/Anthropic impl
│   └── reporters/            # Result aggregation + display
│       └── summary.py        # Rich tables + JSON loader
└── results/                   # JSON outputs per run
```

## Bestie's Real Tools (from apps/web/app/api/chat/tool-metadata.ts)

| Tool | Input schema |
|------|-------------|
| `get_weather` | `{ location?: string }` |
| `get_journal_entries` | `{ timeframe: string }` |
| `internet_lookup` | `{ query: string }` |
| `schedule_event` | `{ when: string, prompt: string, silent?: bool, fallbackReply?: string }` |
| `get_reminders` | `{}` |
| `update_reminder` | `{ index: number, when?: string, prompt?: string }` |
| `cancel_reminder` | `{ indexes: number[] }` |

## Key Commands

```bash
pip install -e .                          # Install
bestie-bench install-stubs                 # Populate stubs/ directory
bestie-bench run --stubs-dir stubs        # Run with stubs
bestie-bench report                       # View latest result
bestie-bench compare --limit 5            # Compare runs
```

## Core Flow

1. `Harness.run(config)` loads fixtures via `load_fixtures()`
2. For tool_calling: runs full **agent loop** — model calls tools → stubs return responses → model produces final text
3. For advice/empathy: single chat call, scored via GEval LLM-judge
4. Scores are aggregated per case (mean, std, pass rate) and per axis
5. Results saved as JSON to `results/`, printed as Rich tables

## HarnessConfig

```python
config = HarnessConfig(
    system_prompt_name="bestie-v1",
    runs_per_case=3,
    temperature=0.7,
    judge_model="gpt-4o",
    max_agent_turns=10,
    stub_registry=StubRegistry(stubs_dir="stubs"),
)
result = harness.run(config=config)
```

## Stubbing

`StubRegistry.get(tool_name, case_id, arguments)` returns a pre-recorded response dict, or None if no stub found.

Stubs are loaded from:
1. `stubs/<tool_name>/<case_id>.json` (case-specific)
2. `stubs/<tool_name>/default.json` (tool-level default)
3. Built-in `DEFAULT_STUBS` dict (always available as fallback)

## Adding Test Cases

Add YAML to `fixtures/<axis>/`:
```yaml
- id: tc_reminder_cancel_001
  description: Cancel a specific reminder
  user_input: "Actually cancel reminder number 2"
  expected_tools:
    - name: cancel_reminder
      arguments:
        indexes: [2]
```

## Adding New Tools

1. Add fixture cases with `expected_tools` for the new tool
2. If stubbing needed: `bestie-bench install-stubs` and edit `stubs/<tool_name>/`
3. If tool is external (needs real API): run without `--stubs-dir`
