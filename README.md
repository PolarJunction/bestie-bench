# bestie-bench

Benchmarking harness for the Bestie AI agent — evaluating **tool calling**, **advice quality**, and **empathy**.

Built on [DeepEval](https://github.com/confident-ai/deepeval) with a model-agnostic design. Swap in any OpenAI- or Anthropic-compatible model.

## Quick Start

```bash
# Install
pip install -e .

# Set API key
export OPENAI_API_KEY=sk-...

# Run benchmark
bestie-bench run --provider openai --model gpt-4o --runs-per-case 3

# View latest result
bestie-bench report

# Compare runs
bestie-bench compare --limit 5
```

## Architecture

```
bestie-bench/
├── fixtures/           # Test cases (YAML)
│   ├── tool_calling/   # Expected tool calls
│   ├── advice/         # Advice quality rubrics
│   └── empathy/        # Empathy rubrics
├── src/bestie_bench/
│   ├── harness.py      # Core eval runner
│   ├── cases/          # TestCase models + loader
│   ├── metrics/        # DeepEval metric wrappers
│   ├── models/         # Unified model client
│   └── reporters/      # Result output
└── results/            # JSON outputs
```

## Defining Test Cases

Fixtures are YAML files in `fixtures/<axis>/`:

```yaml
# fixtures/tool_calling/reminders.yaml
- id: tc_reminder_001
  description: Schedule a reminder for 9am Tuesday
  user_input: "Remind me to call mum at 9am on Tuesday"
  expected_tools:
    - name: schedule_reminder
      arguments:
        time: "09:00"
        day: "tuesday"
        message: "Call mum"

---
# fixtures/empathy/relationships.yaml
- id: emp_rel_001
  description: User received bad news from a friend
  user_input: "My best friend just told me she's moving to Australia..."
  evaluation_criteria: |
    Evaluate empathy quality:
    1. Does it acknowledge the mixed feelings?
    2. Is the tone warm and natural?
    ...
```

## Adding New Axes

1. Create `fixtures/<new_axis>/cases.yaml`
2. Define `evaluation_criteria` (for LLM-judge) or `expected_tools` (for deterministic)
3. Add axis to `harness.py` if it needs special handling
4. Run with `--axes <new_axis>`

## System Prompts

List available prompts:
```bash
bestie-bench prompts
```

Override in a run:
```bash
bestie-bench run --system-prompt bestie-advisor ...
```

## Scoring

| Axis | Metric | Method |
|------|--------|--------|
| Tool calling | `ToolCorrectnessMetric` + `ArgumentCorrectnessMetric` | Deterministic — compare expected vs actual tool calls |
| Advice | `GEval` with custom rubric | LLM-as-judge (GPT-4o by default) |
| Empathy | `GEval` with custom rubric | LLM-as-judge |

Scores are 0–1. Default pass threshold: 0.5.

## Results

Results are saved as JSON to `results/run-*.json`. Use `bestie-bench report` to print a formatted summary, or `bestie-bench compare` to compare multiple runs.
