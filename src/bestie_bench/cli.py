"""CLI for bestie-bench.

Usage:
    bestie-bench run --provider openai --model gpt-4o --fixtures fixtures
    bestie-bench install-stubs
    bestie-bench report --results-dir results
    bestie-bench compare --results-dir results
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from bestie_bench import Harness, make_client, Axis
from bestie_bench.harness import SYSTEM_PROMPTS, HarnessConfig
from bestie_bench.reporters import (
    print_benchmark_result,
    compare_results,
    iter_results,
    load_result,
)

console = Console()


@click.group()
def cli() -> None:
    """bestie-bench — Benchmarking harness for Bestie AI agent."""
    pass


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "openai-compatible"]),
    default="openai",
    help="Model provider",
)
@click.option(
    "--model",
    default="gpt-4o",
    help="Model name",
)
@click.option(
    "--base-url",
    default=None,
    help="Base URL for OpenAI-compatible APIs",
)
@click.option(
    "--api-key",
    default=None,
    help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)",
)
@click.option(
    "--fixtures-dir",
    type=click.Path(path_type=Path),
    default="fixtures",
    help="Directory containing fixture YAML files",
)
@click.option(
    "--stubs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing stub JSON files (enables stubbing)",
)
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default="results",
    help="Directory to write results JSON",
)
@click.option(
    "--system-prompt",
    default="bestie-v1",
    type=click.Choice(list(SYSTEM_PROMPTS.keys())),
    help="System prompt to use",
)
@click.option(
    "--runs-per-case",
    default=3,
    type=int,
    help="Number of runs per test case",
)
@click.option(
    "--temperature",
    default=0.7,
    type=float,
    help="Sampling temperature",
)
@click.option(
    "--judge-model",
    default="gpt-4o",
    help="Model to use for LLM-judge scoring",
)
@click.option(
    "--max-agent-turns",
    default=10,
    type=int,
    help="Max agent loop turns per tool-calling case",
)
@click.option(
    "--axes",
    multiple=True,
    type=click.Choice(["tool_calling", "advice", "empathy"]),
    help="Which axes to run (default: all)",
)
@click.option("--verbose", is_flag=True, help="Print progress")
def run(
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    fixtures_dir: Path,
    stubs_dir: Path | None,
    results_dir: Path,
    system_prompt: str,
    runs_per_case: int,
    temperature: float,
    judge_model: str,
    max_agent_turns: int,
    axes: tuple[str, ...],
    verbose: bool,
) -> None:
    """Run the benchmark."""
    # Build model client
    kwargs: dict = {"model": model}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    try:
        client = make_client(provider, **kwargs)
    except Exception as exc:
        console.print(f"[red]Failed to create client: {exc}[/red]")
        sys.exit(1)

    # Build stub registry
    stub_registry = None
    if stubs_dir:
        try:
            from bestie_bench.stubs.registry import StubRegistry
            stub_registry = StubRegistry(stubs_dir=stubs_dir)
            console.print(f"[dim]Stubbing enabled from: {stubs_dir}[/dim]")
        except ImportError:
            console.print("[yellow]Stubbing requested but stubs module not available[/yellow]")

    harness = Harness(
        client=client,
        fixtures_dir=fixtures_dir,
        results_dir=results_dir,
        stub_registry=stub_registry,
    )

    config = HarnessConfig(
        system_prompt_name=system_prompt,
        runs_per_case=runs_per_case,
        temperature=temperature,
        judge_model=judge_model,
        max_agent_turns=max_agent_turns,
        stub_registry=stub_registry,
    )

    axis_list = [Axis(a) for a in axes] if axes else None

    console.print(f"\n[cyan]Starting benchmark[/cyan]")
    console.print(f"  Model: {client.name()}")
    console.print(f"  System prompt: {system_prompt}")
    console.print(f"  Runs/case: {runs_per_case}")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  Judge model: {judge_model}")
    console.print(f"  Fixtures: {fixtures_dir}")
    if stubs_dir:
        console.print(f"  Stubs: {stubs_dir}")
    console.print()

    result = harness.run(config=config, axes=axis_list, verbose=verbose)

    console.print()
    print_benchmark_result(result)


@cli.command()
@click.option(
    "--stubs-dir",
    type=click.Path(path_type=Path),
    default="stubs",
    help="Directory to install stub files to",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing stub files",
)
def install_stubs(stubs_dir: Path, force: bool) -> None:
    """Install default stub files to a directory for customization.

    Run this once to populate stubs/ with JSON files for each tool.
    Edit the files to change the stubbed responses.
    """
    try:
        from bestie_bench.stubs.registry import install_stubs as _install
    except ImportError as exc:
        console.print(f"[red]Stubs module not available: {exc}[/red]")
        sys.exit(1)

    console.print(f"Installing stubs to [cyan]{stubs_dir}[/cyan]...")
    _install(stubs_dir, force=force)
    console.print("\n[green]Done.[/green] Stub files installed:")
    console.print("  Edit any .json file to change the stubbed response.")
    console.print("  Run with [cyan]--stubs-dir stubs[/cyan] to use them.")


@cli.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default="results",
    help="Directory containing result JSON files",
)
def report(results_dir: Path) -> None:
    """Print a summary of the most recent benchmark result."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        console.print("[red]Results directory not found[/red]")
        sys.exit(1)

    result_files = sorted(results_dir.glob("run-*.json"))
    if not result_files:
        console.print("[yellow]No result files found[/yellow]")
        sys.exit(1)

    latest = result_files[-1]
    result = load_result(latest)
    console.print(f"\n[dim]Loaded from {latest.name}[/dim]\n")
    print_benchmark_result(result)


@cli.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default="results",
    help="Directory containing result JSON files",
)
@click.option(
    "--limit",
    default=10,
    type=int,
    help="Maximum number of results to compare",
)
def compare(results_dir: Path, limit: int) -> None:
    """Compare multiple benchmark results."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        console.print("[red]Results directory not found[/red]")
        sys.exit(1)

    results = []
    for _, result in list(iter_results(results_dir))[:limit]:
        results.append(result)

    if not results:
        console.print("[yellow]No result files found[/yellow]")
        sys.exit(1)

    compare_results(results)


@cli.command()
def prompts() -> None:
    """List available system prompts."""
    console.print("[bold cyan]Available system prompts:[/bold cyan]\n")
    for name, prompt in SYSTEM_PROMPTS.items():
        short = prompt[:100] + "..." if len(prompt) > 100 else prompt
        console.print(f"  [green]{name}[/green]")
        console.print(f"    {short}\n")


def main() -> None:
    cli(auto_envvar_prefix="BESTIE_BENCH")


if __name__ == "__main__":
    main()
