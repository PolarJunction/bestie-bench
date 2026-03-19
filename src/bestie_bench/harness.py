"""Core evaluation harness for bestie-bench.

Orchestrates: loading fixtures → running model → scoring → aggregating results.
Supports stubbing for deterministic, fast runs.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bestie_bench.cases import (
    Axis,
    BenchmarkResult,
    AxisSummary,
    AggregateResult,
    TestCase,
    TestResult,
    load_fixtures,
)
from bestie_bench.metrics.advice import evaluate_advice_case
from bestie_bench.metrics.empathy import evaluate_empathy_case
from bestie_bench.metrics.tool_calling import evaluate_tool_case
from bestie_bench.models.client import ModelClient, ModelResponse, ToolCallResult

try:
    from bestie_bench.stubs.registry import StubRegistry
except ImportError:
    StubRegistry = None  # type: ignore


# ---------------------------------------------------------------------------
# System prompt registry
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "bestie-v1": (
        "You are Bestie, an AI companion and trusted friend. "
        "You are warm, empathetic, and genuinely care about the user's wellbeing. "
        "You give practical advice when asked, but always lead with empathy. "
        "You are available 24/7 and remember details from our conversations. "
        "Your personality is supportive but not sycophantic — you can gently disagree "
        "or offer a different perspective when it serves the user."
    ),
    "bestie-advisor": (
        "You are Bestie, a wise and pragmatic AI advisor. "
        "You focus on giving actionable, well-reasoned advice. "
        "You acknowledge uncertainty when you aren't confident, "
        "and you always prioritize the user's safety and wellbeing. "
        "Be direct but kind."
    ),
    "bestie-cheerleader": (
        "You are Bestie, an enthusiastic and supportive AI companion. "
        "You are optimistic and focus on the positive while remaining realistic. "
        "You celebrate wins, lift spirits during lows, "
        "and always remind the user of their strengths and progress."
    ),
}


# ---------------------------------------------------------------------------
# Bestie tool definitions (mirrors apps/web/app/api/chat/tool-metadata.ts)
# ---------------------------------------------------------------------------

BESTIE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Optional city or area. Omit if user location is obvious from context.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_journal_entries",
            "description": "Read recent journal entries from the past week.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {
                        "type": "string",
                        "description": "Natural-language recent time range like 'yesterday', 'last 3 days', or 'March 12 to March 14'",
                    },
                },
                "required": ["timeframe"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "internet_lookup",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific live-search query to look up.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_event",
            "description": "Create a new reminder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "when": {
                        "type": "string",
                        "description": "Natural-language timing like 'tomorrow at 2pm', 'in 30 minutes', or 'every weekday at 8am'.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": 'Reminder message formatted as "Remind the user to..."',
                    },
                    "silent": {
                        "type": "boolean",
                        "description": "Optional. Use bare true or false. Omit for normal visible reminders.",
                    },
                    "fallbackReply": {
                        "type": "string",
                        "description": "Optional short reply hint for quiet proactive reminders.",
                    },
                },
                "required": ["when", "prompt"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_reminders",
            "description": "List active reminders.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_reminder",
            "description": "Edit an existing reminder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "number",
                        "description": "Reminder number from the list (e.g., 1, 2, 3)",
                    },
                    "when": {
                        "type": "string",
                        "description": "New timing like 'tomorrow at 2pm'.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "New reminder text.",
                    },
                },
                "required": ["index"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Cancel one or more reminders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indexes": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Reminder numbers to cancel (e.g., [1] or [1, 2, 3])",
                    },
                },
                "required": ["indexes"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """Configuration for a benchmark run."""

    system_prompt_name: str = "bestie-v1"
    runs_per_case: int = 3
    temperature: float = 0.7
    judge_model: str = "gpt-4o"
    max_agent_turns: int = 10
    stub_registry: "StubRegistry | None" = None


class Harness:
    """Bestie evaluation harness.

    Usage:
        registry = StubRegistry(stubs_dir=Path("stubs"))
        harness = Harness(client, fixtures_dir=Path("fixtures"), stub_registry=registry)
        result = harness.run(HarnessConfig(system_prompt_name="bestie-v1", runs_per_case=3))
    """

    def __init__(
        self,
        client: ModelClient,
        fixtures_dir: Path | str = "fixtures",
        results_dir: Path | str = "results",
        stub_registry: "StubRegistry | None" = None,
    ):
        self.client = client
        self.fixtures_dir = Path(fixtures_dir)
        self.results_dir = Path(results_dir)
        self.stub_registry = stub_registry
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        config: HarnessConfig | None = None,
        axes: list[Axis] | None = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """Run the full benchmark."""
        config = config or HarnessConfig()
        axes = axes or [Axis.TOOL_CALLING, Axis.ADVICE, Axis.EMPATHY]
        system_prompt = SYSTEM_PROMPTS.get(
            config.system_prompt_name, config.system_prompt_name
        )

        cases_by_axis = load_fixtures(self.fixtures_dir)

        t0 = time.perf_counter()
        all_results: dict[Axis, list[TestResult]] = {axis: [] for axis in axes}
        errors: list[str] = []

        total_cases = sum(len(cases_by_axis[axis]) for axis in axes)
        case_num = 0

        for axis in axes:
            cases = cases_by_axis[axis]
            if not cases:
                continue

            if verbose:
                print(f"\n{'='*60}")
                print(f"Axis: {axis.value}")
                print(f"{'='*60}")

            for case in cases:
                case_num += 1
                if verbose:
                    print(f"\n[{case_num}/{total_cases}] {case.id}: {case.description}")

                for run in range(config.runs_per_case):
                    try:
                        result = self._run_case(
                            case=case,
                            system_prompt=system_prompt,
                            config=config,
                        )
                        all_results[axis].append(result)
                        if verbose:
                            score_str = ", ".join(
                                f"{k}={v:.2f}" for k, v in result.scores.items()
                            )
                            status = "✓" if result.passed else "✗"
                            print(
                                f"  run {run}: {status} {score_str} ({result.latency_ms:.0f}ms)"
                            )
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{case.id} run {run}: {exc}")
                        if verbose:
                            print(f"  run {run}: ERROR — {exc}")

        total_duration_ms = (time.perf_counter() - t0) * 1000

        axis_summaries = {}
        for axis in axes:
            axis_summaries[axis] = self._aggregate_axis(all_results[axis])

        result = BenchmarkResult(
            model=self.client.name(),
            system_prompt_name=config.system_prompt_name,
            temperature=config.temperature,
            num_runs_per_case=config.runs_per_case,
            axes=axis_summaries,
            total_duration_ms=total_duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            errors=errors,
        )

        self._save_result(result)
        return result

    def _run_case(
        self,
        case: TestCase,
        system_prompt: str,
        config: HarnessConfig,
    ) -> TestResult:
        """Run a single test case once.

        For tool_calling: runs the full agent loop (model → tools → model → ...)
        For advice/empathy: single chat call with no tools.
        """
        if case.axis == Axis.TOOL_CALLING:
            return self._run_tool_case(case, system_prompt, config)
        else:
            return self._run_single_turn(case, system_prompt, config)

    def _run_single_turn(
        self,
        case: TestCase,
        system_prompt: str,
        config: HarnessConfig,
    ) -> TestResult:
        """Run a single chat turn (no tool calling)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case.user_input},
        ]

        response = self.client.chat(
            messages=messages,
            tools=None,
            temperature=config.temperature,
        )

        scores: dict[str, float] = {}
        passed = True

        if case.axis == Axis.ADVICE:
            advice_score, _ = evaluate_advice_case(
                case, response.content, config.judge_model
            )
            scores["advice_quality"] = advice_score
            passed = advice_score >= 0.5

        elif case.axis == Axis.EMPATHY:
            empathy_score, _ = evaluate_empathy_case(
                case, response.content, config.judge_model
            )
            scores["empathy"] = empathy_score
            passed = empathy_score >= 0.5

        return TestResult(
            case_id=case.id,
            axis=case.axis,
            scores=scores,
            passed=passed,
            model_response=response.content,
            latency_ms=response.latency_ms,
            model=response.model,
            temperature=config.temperature,
        )

    def _run_tool_case(
        self,
        case: TestCase,
        system_prompt: str,
        config: HarnessConfig,
    ) -> TestResult:
        """Run a tool-calling test case with full agent loop."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case.user_input},
        ]

        all_tool_calls: list[dict[str, Any]] = []
        total_latency_ms = 0.0
        final_content = ""

        for _turn in range(config.max_agent_turns):
            response = self.client.chat(
                messages=messages,
                tools=BESTIE_TOOLS,
                temperature=config.temperature,
            )

            total_latency_ms += response.latency_ms

            if not response.tool_calls:
                # No more tools — model has responded
                final_content = response.content
                break

            # Collect tool calls for scoring
            for tc in response.tool_calls:
                all_tool_calls.append({"name": tc.name, "arguments": tc.arguments})

            # Build tool results (stubbed or real)
            tool_results = self._execute_tool_calls(
                case.id, response.tool_calls
            )

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            })

            # Add tool result messages
            for tc_result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_result["tool_call_id"],
                    "content": json.dumps(tc_result["response"]),
                })

        # Score
        tool_score, arg_score, _ = evaluate_tool_case(
            case, final_content, all_tool_calls, include_args=True
        )
        scores = {
            "tool_correctness": tool_score,
            "argument_correctness": arg_score,
        }
        passed = tool_score >= 0.5 and arg_score >= 0.5

        return TestResult(
            case_id=case.id,
            axis=case.axis,
            scores=scores,
            passed=passed,
            tools_called=all_tool_calls,
            model_response=final_content,
            latency_ms=total_latency_ms,
            model=self.client.name(),
            temperature=config.temperature,
        )

    def _execute_tool_calls(
        self,
        case_id: str,
        tool_calls: list[ToolCallResult],
    ) -> list[dict[str, Any]]:
        """Execute tool calls, using stubs when available.

        Returns a list of tool result dicts ready to append to messages.
        """
        results = []
        for tc in tool_calls:
            response = self._get_tool_response(case_id, tc)
            results.append({
                "tool_call_id": tc.id or f"call_{tc.name}",
                "response": response,
            })
        return results

    def _get_tool_response(
        self,
        case_id: str,
        tool_call: ToolCallResult,
    ) -> dict[str, Any]:
        """Get tool response — stubbed if registry is available, otherwise real."""
        tool_name = tool_call.name

        # Try stub first
        if self.stub_registry:
            stubbed = self.stub_registry.get(
                tool_name=tool_name,
                case_id=case_id,
                arguments=tool_call.arguments,
            )
            if stubbed is not None:
                return stubbed

        # No stub available — return a generic error response
        # In a real run this would call the actual API
        return {
            "error": f"No stub found for {tool_name} (case={case_id}). "
                     f"Run 'bestie-bench install-stubs' to set up stubs, "
                     f"or run with --no-stubs to use real APIs."
        }

    def _aggregate_axis(self, results: list[TestResult]) -> AxisSummary:
        """Aggregate results for an axis."""
        if not results:
            return AxisSummary(
                axis=Axis.TOOL_CALLING,
                overall_mean_score=0.0,
                overall_pass_rate=0.0,
                num_cases=0,
                num_runs=0,
            )

        import statistics
        from collections import defaultdict

        by_case: dict[str, list[TestResult]] = defaultdict(list)
        for r in results:
            by_case[r.case_id].append(r)

        case_aggregates: list[AggregateResult] = []

        for case_id, case_results in by_case.items():
            first = case_results[0]
            all_scores = []
            for r in case_results:
                all_scores.extend(r.scores.values())

            mean = statistics.mean(all_scores) if all_scores else 0.0
            std = statistics.stdev(all_scores) if len(all_scores) > 1 else None
            pass_rate = sum(1 for r in case_results if r.passed) / len(case_results)

            case_aggregates.append(AggregateResult(
                case_id=case_id,
                axis=first.axis,
                description=first.axis.value,
                mean_score=mean,
                std_score=std,
                min_score=min(all_scores) if all_scores else 0.0,
                max_score=max(all_scores) if all_scores else 0.0,
                pass_rate=pass_rate,
                num_runs=len(case_results),
            ))

        all_scores_flat = [s for r in results for s in r.scores.values()]
        overall_mean = statistics.mean(all_scores_flat) if all_scores_flat else 0.0
        overall_pass = (
            sum(1 for r in results if r.passed) / len(results) if results else 0.0
        )

        return AxisSummary(
            axis=first.axis,
            overall_mean_score=overall_mean,
            overall_pass_rate=overall_pass,
            num_cases=len(case_aggregates),
            num_runs=len(results),
            case_results=case_aggregates,
        )

    def _save_result(self, result: BenchmarkResult) -> Path:
        """Save result to disk."""
        run_id = str(uuid.uuid4())[:8]
        safe_model = result.model.replace("/", "-").replace(":", "-")
        filename = f"run-{safe_model}-{run_id}.json"
        path = self.results_dir / filename

        data = {
            "model": result.model,
            "system_prompt_name": result.system_prompt_name,
            "temperature": result.temperature,
            "num_runs_per_case": result.num_runs_per_case,
            "total_duration_ms": result.total_duration_ms,
            "timestamp": result.timestamp,
            "errors": result.errors,
            "axes": {},
        }

        for axis, summary in result.axes.items():
            data["axes"][axis.value] = {
                "overall_mean_score": summary.overall_mean_score,
                "overall_pass_rate": summary.overall_pass_rate,
                "num_cases": summary.num_cases,
                "num_runs": summary.num_runs,
                "cases": [
                    {
                        "case_id": cr.case_id,
                        "mean_score": cr.mean_score,
                        "std_score": cr.std_score,
                        "min_score": cr.min_score,
                        "max_score": cr.max_score,
                        "pass_rate": cr.pass_rate,
                        "num_runs": cr.num_runs,
                    }
                    for cr in summary.case_results
                ],
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path
