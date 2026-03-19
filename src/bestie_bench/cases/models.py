"""Test case models for bestie-bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Axis(Enum):
    """Evaluation axis."""

    TOOL_CALLING = "tool_calling"
    ADVICE = "advice"
    EMPATHY = "empathy"


class Difficulty(Enum):
    """Case difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ToolCall:
    """Expected tool call."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    optional_arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    """A single test case."""

    id: str
    axis: Axis
    description: str
    user_input: str
    expected_tools: list[ToolCall] = field(default_factory=list)
    # For subjective axes — criteria fed to GEval
    evaluation_criteria: str | None = None
    evaluation_steps: list[str] = field(default_factory=list)
    # Optional: reference best response for comparison
    reference_response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.axis, str):
            self.axis = Axis(self.axis)


@dataclass
class TestResult:
    """Result of a single test case evaluation."""

    case_id: str
    axis: Axis
    # Raw scores from DeepEval metrics (0.0 - 1.0)
    scores: dict[str, float]
    # Pass/fail per metric
    passed: bool
    # Tool calls made (for tool_calling axis)
    tools_called: list[dict[str, Any]] = field(default_factory=list)
    # Model's raw response
    model_response: str = ""
    # Execution metadata
    latency_ms: float = 0.0
    model: str = ""
    temperature: float = 0.0
    error: str | None = None


@dataclass
class AggregateResult:
    """Aggregated results across multiple runs of the same test case."""

    case_id: str
    axis: Axis
    description: str
    # Statistics across runs
    mean_score: float
    std_score: float | None = None
    min_score: float = 0.0
    max_score: float = 0.0
    pass_rate: float = 0.0
    num_runs: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AxisSummary:
    """Summary of all results for an evaluation axis."""

    axis: Axis
    overall_mean_score: float
    overall_pass_rate: float
    num_cases: int
    num_runs: int
    case_results: list[AggregateResult] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Full benchmark result."""

    model: str
    system_prompt_name: str
    temperature: float
    num_runs_per_case: int
    axes: dict[Axis, AxisSummary]
    total_duration_ms: float
    timestamp: str
    errors: list[str] = field(default_factory=list)
