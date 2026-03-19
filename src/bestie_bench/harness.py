"""Core evaluation harness for bestie-bench.

Orchestrates: loading fixtures → running model → scoring → aggregating results.
"""

from __future__ import annotations

import json
import time
import uuid
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
from bestie_bench.models.client import ModelClient, ModelResponse


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
# Tool definitions for Bestie
# ---------------------------------------------------------------------------

BESTIE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name or coordinates"},
                    "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_reminder",
            "description": "Schedule a reminder or alarm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {"type": "string", "description": "Time in 24h format, e.g. '09:00'"},
                    "day": {"type": "string", "description": "Day of week, e.g. 'Tuesday' or 'today'"},
                    "message": {"type": "string", "description": "Reminder message"},
                    "repeat": {"type": "string", "enum": ["none", "daily", "weekly"], "default": "none"},
                },
                "required": ["time", "day", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_outfit",
            "description": "Analyze a photo of an outfit and suggest improvements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "URL of the outfit photo"},
                    "occasion": {"type": "string", "description": "Occasion or context for the outfit"},
                },
                "required": ["image_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_goal",
            "description": "Set or update a personal goal with tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Goal title"},
                    "target_date": {"type": "string", "description": "Target completion date (YYYY-MM-DD)"},
                    "milestones": {"type": "array", "items": {"type": "string"}, "description": "Milestone labels"},
                },
                "required": ["title", "target_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


class Harness:
    """Bestie evaluation harness.

    Usage:
        harness = Harness(model_client, fixtures_dir=Path("fixtures"))
        result = harness.run(
            system_prompt_name="bestie-v1",
            runs_per_case=5,
            temperature=0.7,
            judge_model="gpt-4o",
        )
    """

    def __init__(
        self,
        client: ModelClient,
        fixtures_dir: Path | str = "fixtures",
        results_dir: Path | str = "results",
    ):
        self.client = client
        self.fixtures_dir = Path(fixtures_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        system_prompt_name: str = "bestie-v1",
        runs_per_case: int = 3,
        temperature: float = 0.7,
        judge_model: str = "gpt-4o",
        axes: list[Axis] | None = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """Run the full benchmark.

        Args:
            system_prompt_name: Key into SYSTEM_PROMPTS registry.
            runs_per_case: Number of times to run each test case (for variance).
            temperature: Temperature for model calls.
            judge_model: Model to use for GEval LLM-judge scoring.
            axes: Which axes to run. Defaults to all three.
            verbose: Print progress.

        Returns:
            BenchmarkResult with all scores and aggregations.
        """
        system_prompt = SYSTEM_PROMPTS.get(system_prompt_name, system_prompt_name)
        axes = axes or [Axis.TOOL_CALLING, Axis.ADVICE, Axis.EMPATHY]

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

                for run in range(runs_per_case):
                    try:
                        result = self._run_case(
                            case=case,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            judge_model=judge_model,
                            run=run,
                        )
                        all_results[axis].append(result)
                        if verbose:
                            score_str = ", ".join(f"{k}={v:.2f}" for k, v in result.scores.items())
                            status = "✓" if result.passed else "✗"
                            print(f"  run {run}: {status} {score_str} ({result.latency_ms:.0f}ms)")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{case.id} run {run}: {exc}")
                        if verbose:
                            print(f"  run {run}: ERROR — {exc}")

        total_duration_ms = (time.perf_counter() - t0) * 1000

        # Aggregate
        axis_summaries = {}
        for axis in axes:
            axis_summaries[axis] = self._aggregate_axis(all_results[axis])

        result = BenchmarkResult(
            model=self.client.name(),
            system_prompt_name=system_prompt_name,
            temperature=temperature,
            num_runs_per_case=runs_per_case,
            axes=axis_summaries,
            total_duration_ms=total_duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            errors=errors,
        )

        # Save
        self._save_result(result)

        return result

    def _run_case(
        self,
        case: TestCase,
        system_prompt: str,
        temperature: float,
        judge_model: str,
        run: int,
    ) -> TestResult:
        """Run a single test case once."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case.user_input},
        ]

        tools = BESTIE_TOOLS if case.axis == Axis.TOOL_CALLING else None

        response = self.client.chat(
            messages=messages,
            tools=tools,
            temperature=temperature,
        )

        tool_call_dicts = [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in response.tool_calls
        ]

        scores: dict[str, float] = {}
        passed = True

        if case.axis == Axis.TOOL_CALLING:
            tool_score, arg_score, _ = evaluate_tool_case(
                case, response.content, tool_call_dicts, include_args=True
            )
            scores["tool_correctness"] = tool_score
            scores["argument_correctness"] = arg_score
            passed = tool_score >= 0.5 and arg_score >= 0.5

        elif case.axis == Axis.ADVICE:
            advice_score, _ = evaluate_advice_case(case, response.content, judge_model)
            scores["advice_quality"] = advice_score
            passed = advice_score >= 0.5

        elif case.axis == Axis.EMPATHY:
            empathy_score, _ = evaluate_empathy_case(case, response.content, judge_model)
            scores["empathy"] = empathy_score
            passed = empathy_score >= 0.5

        return TestResult(
            case_id=case.id,
            axis=case.axis,
            scores=scores,
            passed=passed,
            tools_called=tool_call_dicts,
            model_response=response.content,
            latency_ms=response.latency_ms,
            model=response.model,
            temperature=temperature,
        )

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

        from collections import defaultdict
        import statistics

        by_case: dict[str, list[TestResult]] = defaultdict(list)
        for r in results:
            by_case[r.case_id].append(r)

        case_aggregates: list[AggregateResult] = []

        for case_id, case_results in by_case.items():
            first = case_results[0]
            # Flatten all scores for this case
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
        overall_pass = sum(1 for r in results if r.passed) / len(results) if results else 0.0

        return AxisSummary(
            axis=first.axis,
            overall_mean_score=overall_mean,
            overall_pass_rate=overall_pass,
            num_cases=len(case_aggregates),
            num_runs=len(results),
            case_results=case_aggregates,
        )

    def _save_result(self, result: BenchmarkResult) -> Path:
        """Save a result to disk."""
        run_id = str(uuid.uuid4())[:8]
        filename = f"run-{result.model.replace('/', '-')}-{run_id}.json"
        path = self.results_dir / filename

        # Convert dataclasses to dict for JSON serialization
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
