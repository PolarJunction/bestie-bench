"""Result aggregation and reporting for bestie-bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from bestie_bench.cases import BenchmarkResult, Axis


console = Console()


def load_result(path: Path) -> BenchmarkResult:
    """Load a BenchmarkResult from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    axes = {}
    for axis_str, summary in data.get("axes", {}).items():
        from dataclasses import dataclass, field
        from bestie_bench.cases import AggregateResult, AxisSummary

        case_results = [
            AggregateResult(
                case_id=c["case_id"],
                axis=Axis(axis_str),
                description="",
                mean_score=c["mean_score"],
                std_score=c["std_score"],
                min_score=c["min_score"],
                max_score=c["max_score"],
                pass_rate=c["pass_rate"],
                num_runs=c["num_runs"],
            )
            for c in summary.get("cases", [])
        ]

        axes[Axis(axis_str)] = AxisSummary(
            axis=Axis(axis_str),
            overall_mean_score=summary["overall_mean_score"],
            overall_pass_rate=summary["overall_pass_rate"],
            num_cases=summary["num_cases"],
            num_runs=summary["num_runs"],
            case_results=case_results,
        )

    return BenchmarkResult(
        model=data["model"],
        system_prompt_name=data["system_prompt_name"],
        temperature=data["temperature"],
        num_runs_per_case=data["num_runs_per_case"],
        axes=axes,
        total_duration_ms=data["total_duration_ms"],
        timestamp=data["timestamp"],
        errors=data.get("errors", []),
    )


def iter_results(results_dir: Path) -> Iterator[tuple[Path, BenchmarkResult]]:
    """Iterate over all result files in a directory."""
    for path in sorted(results_dir.glob("run-*.json")):
        try:
            yield path, load_result(path)
        except Exception:
            continue


def print_benchmark_result(result: BenchmarkResult) -> None:
    """Print a benchmark result to the console."""
    header = Text()
    header.append(f"Model: ", style="bold cyan")
    header.append(result.model)
    header.append(f"\nSystem prompt: ", style="bold cyan")
    header.append(result.system_prompt_name)
    header.append(f"\nTemperature: ", style="bold cyan")
    header.append(str(result.temperature))
    header.append(f"\nRuns/case: ", style="bold cyan")
    header.append(str(result.num_runs_per_case))
    header.append(f"\nDuration: ", style="bold cyan")
    header.append(f"{result.total_duration_ms / 1000:.1f}s")

    console.print(Panel(header, title="Benchmark Result", border_style="cyan"))

    for axis in [Axis.TOOL_CALLING, Axis.ADVICE, Axis.EMPATHY]:
        summary = result.axes.get(axis)
        if not summary:
            continue

        color = _axis_color(axis)
        table = Table(title=f"{axis.value.upper()} — {summary.num_cases} cases, {summary.num_runs} runs", border_style=color)
        table.add_column("Case", style="dim")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Pass %", justify="right")

        for cr in summary.case_results:
            pass_pct = f"{cr.pass_rate * 100:.0f}%"
            mean_str = f"{cr.mean_score:.3f}"
            std_str = f"{cr.std_score:.3f}" if cr.std_score is not None else "—"
            table.add_row(
                cr.case_id,
                mean_str,
                std_str,
                f"{cr.min_score:.3f}",
                f"{cr.max_score:.3f}",
                pass_pct,
            )

        # Summary row
        table.add_section()
        table.add_row(
            "[bold]OVERALL[/bold]",
            f"[bold]{summary.overall_mean_score:.3f}[/bold]",
            "—",
            "—",
            "—",
            f"[bold]{summary.overall_pass_rate * 100:.0f}%[/bold]",
            style=color,
        )

        console.print(table)

    if result.errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for err in result.errors:
            console.print(f"  • {err}")


def compare_results(results: list[BenchmarkResult]) -> None:
    """Compare multiple benchmark results side by side."""
    table = Table(
        title="Model Comparison",
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Model", style="cyan bold")
    table.add_column("System Prompt", style="dim")
    table.add_column("Tool Calling", justify="right")
    table.add_column("Advice", justify="right")
    table.add_column("Empathy", justify="right")
    table.add_column("Duration", justify="right")

    for result in results:
        tc = result.axes.get(Axis.TOOL_CALLING)
        adv = result.axes.get(Axis.ADVICE)
        emp = result.axes.get(Axis.EMPATHY)

        table.add_row(
            result.model,
            result.system_prompt_name,
            f"{tc.overall_mean_score:.3f}" if tc else "—",
            f"{adv.overall_mean_score:.3f}" if adv else "—",
            f"{emp.overall_mean_score:.3f}" if emp else "—",
            f"{result.total_duration_ms / 1000:.1f}s",
        )

    console.print(table)


def _axis_color(axis: Axis) -> str:
    return {
        Axis.TOOL_CALLING: "green",
        Axis.ADVICE: "yellow",
        Axis.EMPATHY: "magenta",
    }.get(axis, "white")
