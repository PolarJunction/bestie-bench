"""CLI for bestie-bench.

Usage:
    bestie-bench run                          # uses bestie-bench.toml if present
    bestie-bench run --provider openai --model gpt-4o
    bestie-bench install-stubs
    bestie-bench report --results-dir results
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from rich.console import Console

from bestie_bench import Harness, make_client, Axis
from bestie_bench.harness import SYSTEM_PROMPTS, HarnessConfig

console = Console()


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

try:
    import tomllib
    HAVE_TOMLLIB = True
except ImportError:
    import tomli as tomllib  # type: ignore
    HAVE_TOMLLIB = False


def load_config(config_path: Path | None = None) -> dict:
    """Load config from TOML file.

    Searches for bestie-bench.toml in:
    1. Explicit path via --config
    2. Current directory
    3. User's home directory
    """
    if config_path and config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    search_paths = [
        Path("bestie-bench.toml"),
        Path.home() / ".config" / "bestie-bench.toml",
    ]

    for path in search_paths:
        if path.exists():
            with open(path, "rb") as f:
                return tomllib.load(f)

    return {}


def config_val(config: dict, *keys: str, default=None):
    """Get a nested config value."""
    val = config
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """bestie-bench — Benchmarking harness for Bestie AI agent."""
    pass


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--provider",
    type=click.Choice(["openai", "anthropic", "minimax", "openai-compatible"]),
    default=None,
    help="Model provider (overrides config file)",
)
@click.option(
    "--model",
    default=None,
    help="Model name (overrides config file)",
)
@click.option(
    "--base-url",
    default=None,
    help="Base URL for OpenAI-compatible APIs",
)
@click.option(
    "--api-key",
    default=None,
    help="API key (overrides BESTIE_BENCH_API_KEY env var and config file)",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to bestie-bench.toml config file",
)
@click.option(
    "--fixtures-dir",
    type=click.Path(path_type=Path),
    default=None,
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
    default=None,
    help="Directory to write results JSON",
)
@click.option(
    "--system-prompt",
    default=None,
    type=click.Choice(list(SYSTEM_PROMPTS.keys())),
    help="System prompt to use",
)
@click.option(
    "--runs-per-case",
    default=None,
    type=int,
    help="Number of runs per test case",
)
@click.option(
    "--temperature",
    default=None,
    type=float,
    help="Sampling temperature",
)
@click.option(
    "--judge-model",
    default=None,
    help="Model to use for LLM-judge scoring",
)
@click.option(
    "--max-agent-turns",
    default=None,
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
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    config: Path | None,
    fixtures_dir: Path | None,
    stubs_dir: Path | None,
    results_dir: Path | None,
    system_prompt: str | None,
    runs_per_case: int | None,
    temperature: float | None,
    judge_model: str | None,
    max_agent_turns: int | None,
    axes: tuple[str, ...],
    verbose: bool,
) -> None:
    """Run the benchmark.

    If bestie-bench.toml exists in the current directory, values from it
    are used as defaults. CLI flags always override config file values.
    """
    cfg = load_config(config)

    # Resolve values: CLI flag > env var > config file
    provider = provider or os.environ.get("BESTIE_BENCH_PROVIDER") or config_val(cfg, "default", "provider")
    model = model or os.environ.get("BESTIE_BENCH_MODEL") or config_val(cfg, "default", "model")
    api_key = api_key or os.environ.get("BESTIE_BENCH_API_KEY") or config_val(cfg, "default", "api_key")
    judge_model = judge_model or os.environ.get("BESTIE_BENCH_JUDGE_MODEL") or config_val(cfg, "default", "judge_model")
    temperature = temperature if temperature is not None else float(os.environ.get("BESTIE_BENCH_TEMPERATURE") or config_val(cfg, "default", "temperature", default=0.7))
    runs_per_case = runs_per_case if runs_per_case is not None else int(os.environ.get("BESTIE_BENCH_RUNS_PER_CASE") or config_val(cfg, "default", "runs_per_case", default=3))
    fixtures_dir = fixtures_dir or Path(os.environ.get("BESTIE_BENCH_FIXTURES_DIR") or config_val(cfg, "default", "fixtures_dir", default="fixtures"))
    stubs_dir_cfg = os.environ.get("BESTIE_BENCH_STUBS_DIR") or config_val(cfg, "default", "stubs_dir")
    stubs_dir = stubs_dir or (Path(stubs_dir_cfg) if stubs_dir_cfg else None)
    results_dir = results_dir or Path(os.environ.get("BESTIE_BENCH_RESULTS_DIR") or config_val(cfg, "default", "results_dir", default="results"))
    system_prompt = system_prompt or os.environ.get("BESTIE_BENCH_SYSTEM_PROMPT") or config_val(cfg, "default", "system_prompt", default="bestie-v1")
    judge_model = judge_model or config_val(cfg, "default", "judge_model", default="gpt-4o")
    max_agent_turns = max_agent_turns if max_agent_turns is not None else int(os.environ.get("BESTIE_BENCH_MAX_AGENT_TURNS") or config_val(cfg, "default", "max_agent_turns", default=10))

    if not provider or not model:
        console.print("[red]--provider and --model are required (or set in bestie-bench.toml)[/red]")
        sys.exit(1)

    if not api_key:
        console.print("[red]API key not set. Set BESTIE_BENCH_API_KEY env var, or api_key in bestie-bench.toml[/red]")
        sys.exit(1)

    # Build model client
    kwargs: dict = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    try:
        client = make_client(provider, **kwargs)
    except Exception as exc:
        console.print(f"[red]Failed to create client: {exc}[/red]")
        sys.exit(1)

    # Build stub registry
    stub_registry = None
    if stubs_dir and stubs_dir.exists():
        try:
            from bestie_bench.stubs.registry import StubRegistry
            stub_registry = StubRegistry(stubs_dir=stubs_dir)
            console.print(f"[dim]Stubbing enabled from: {stubs_dir}[/dim]")
        except ImportError:
            console.print("[yellow]Stubs requested but module not available[/yellow]")

    harness = Harness(
        client=client,
        fixtures_dir=fixtures_dir,
        results_dir=results_dir,
        stub_registry=stub_registry,
    )

    cfg_harness = HarnessConfig(
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

    from bestie_bench.reporters import print_benchmark_result

    result = harness.run(config=cfg_harness, axes=axis_list, verbose=verbose)
    console.print()
    print_benchmark_result(result)


# ---------------------------------------------------------------------------
# install-stubs command
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--stubs-dir",
    type=click.Path(path_type=Path),
    default="stubs",
    help="Directory to install stub files to",
)
@click.option("--force", is_flag=True, help="Overwrite existing stub files")
def install_stubs(stubs_dir: Path, force: bool) -> None:
    """Install default stub files to a directory for customization."""
    try:
        from bestie_bench.stubs.registry import install_stubs as _install
    except ImportError as exc:
        console.print(f"[red]Stubs module not available: {exc}[/red]")
        sys.exit(1)

    console.print(f"Installing stubs to [cyan]{stubs_dir}[/cyan]...")
    _install(stubs_dir, force=force)
    console.print("\n[green]Done.[/green] Stub files installed.")
    console.print("  Edit any .json file to change the stubbed response.")
    console.print("  Run with [cyan]--stubs-dir stubs[/cyan] to use them.")


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default="results",
    help="Directory containing result JSON files",
)
def report(results_dir: Path) -> None:
    """Print a summary of the most recent benchmark result."""
    from bestie_bench.reporters import load_result, print_benchmark_result

    results_dir = Path(results_dir)
    if not results_dir.exists():
        console.print("[red]Results directory not found[/red]")
        sys.exit(1)

    result_files = sorted(results_dir.glob("run-*.json"))
    if not result_files:
        console.print("[yellow]No result files found[/yellow]")
        sys.exit(1)

    latest = result_files[-1]
    result_obj = load_result(latest)
    console.print(f"\n[dim]Loaded from {latest.name}[/dim]\n")
    print_benchmark_result(result_obj)


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default="results",
    help="Directory containing result JSON files",
)
@click.option("--limit", default=10, type=int, help="Maximum number of results to compare")
def compare(results_dir: Path, limit: int) -> None:
    """Compare multiple benchmark results."""
    from bestie_bench.reporters import iter_results, compare_results

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


# ---------------------------------------------------------------------------
# prompts command
# ---------------------------------------------------------------------------


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
