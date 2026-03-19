# CLAUDE.md — bestie-bench

You are an expert Python developer and ML engineer working on the bestie-bench benchmarking harness.

## Tech Stack

- **Python 3.11+** with type annotations
- **DeepEval** — LLM evaluation framework (LLM-as-judge, tool call metrics)
- **httpx** — HTTP client for model API calls
- **Pytest** — testing
- **Ruff** — linting/formatting
- **Click** — CLI framework
- **Rich** — terminal output

## Key Patterns

| Pattern | Location | Notes |
|---------|---------|-------|
| Test case fixtures | `fixtures/<axis>/*.yaml` | YAML with `cases` list |
| Metric definition | `src/bestie_bench/metrics/` | GEval wrappers per axis |
| Model abstraction | `src/bestie_bench/models/client.py` | Factory `make_client()` |
| Harness | `src/bestie_bench/harness.py` | `Harness.run()` is main entry |
| CLI | `src/bestie_bench/cli.py` | Click commands |

## Commands

```bash
pip install -e .               # Install in dev mode
pip install -e ".[dev]"         # With dev dependencies
ruff check .                    # Lint
ruff format .                   # Format
mypy src/                       # Type check
pytest tests/                   # Run tests
bestie-bench run --verbose      # Run benchmark
```

## Design Decisions

- **Fixtures over code** — test cases are YAML, not Python, so non-engineers can add cases
- **Model-agnostic** — any OpenAI/Anthropic-compatible API via `ModelClient` abstraction
- **GEval for subjectivity** — advice/empathy use LLM-as-judge; tool calling is deterministic
- **N runs per case** — captures variance from temperature > 0

## Adding a Metric

1. Create `src/bestie_bench/metrics/<axis>.py` with a `build_<metric>()` function
2. Add `evaluate_<axis>_case()` helper
3. Wire it into `harness.py` `_run_case()` match block
4. Add fixture examples to `fixtures/<axis>/`
