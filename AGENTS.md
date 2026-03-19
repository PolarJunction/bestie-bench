# AGENTS.md — bestie-bench

## Project Overview

bestie-bench is a benchmarking harness for evaluating AI agents across three axes:
- **Tool calling** — deterministic checks (correct tool + arguments)
- **Advice quality** — LLM-judged (practicality, soundness, calibration)
- **Empathy** — LLM-judged (emotional intelligence, warmth, non-judgment)

## Directory Structure

```
bestie-bench/
├── src/bestie_bench/
│   ├── harness.py      # Core runner (Harness class)
│   ├── cli.py          # Click CLI entrypoint
│   ├── cases/          # TestCase models + YAML loader
│   │   ├── models.py  # Dataclasses: TestCase, TestResult, etc.
│   │   └── loader.py  # load_fixtures(), iter_fixtures()
│   ├── metrics/        # DeepEval wrappers per axis
│   │   ├── tool_calling.py
│   │   ├── advice.py
│   │   └── empathy.py
│   ├── models/        # Unified model client
│   │   └── client.py  # ModelClient ABC + OpenAI/Anthropic impl
│   └── reporters/     # Result aggregation + display
│       └── summary.py # Rich tables + JSON loader
├── fixtures/          # Test case definitions
│   ├── tool_calling/
│   ├── advice/
│   └── empathy/
└── results/           # JSON outputs per run
```

## Key Commands

```bash
pip install -e . && bestie-bench run --verbose
bestie-bench prompts
bestie-bench report
bestie-bench compare --limit 5
```

## Core Flow

1. `Harness.run()` loads fixtures via `load_fixtures()`
2. For each case × N runs: calls `ModelClient.chat()` with system prompt + user input
3. Scores via DeepEval metrics (deterministic for tool calling, GEval for subjective)
4. Aggregates results → saves JSON → prints Rich table

## Model Abstraction

`make_client(provider, model, **kwargs)` returns `ModelClient`:
- `OpenAIClient` — OpenAI + any OpenAI-compatible API
- `AnthropicClient` — Anthropic Messages API

Tools are passed as OpenAI-style function definitions and translated for Anthropic.

## Adding Test Cases

Add YAML files to `fixtures/<axis>/`. Each case needs:
- `id`: unique identifier
- `user_input`: the prompt to send
- `expected_tools` (tool_calling) OR `evaluation_criteria` (advice/empathy)
- `description`: human-readable summary

## System Prompts

Defined in `harness.py` `SYSTEM_PROMPTS` dict. Currently:
- `bestie-v1` — warm, empathetic companion
- `bestie-advisor` — pragmatic, well-reasoned advice
- `bestie-cheerleader` — enthusiastic, supportive
