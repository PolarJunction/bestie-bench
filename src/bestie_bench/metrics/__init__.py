"""Metrics modules for bestie-bench evaluation axes."""

from bestie_bench.metrics.advice import (
    DEFAULT_ADVICE_CRITERIA,
    DEFAULT_ADVICE_STEPS,
    build_advice_metric,
    evaluate_advice_case,
)
from bestie_bench.metrics.empathy import (
    DEFAULT_EMPATHY_CRITERIA,
    DEFAULT_EMPATHY_STEPS,
    build_empathy_metric,
    evaluate_empathy_case,
)
from bestie_bench.metrics.tool_calling import (
    build_tool_correctness_metric,
    build_argument_correctness_metric,
    evaluate_tool_case,
)

__all__ = [
    # Tool calling
    "build_tool_correctness_metric",
    "build_argument_correctness_metric",
    "evaluate_tool_case",
    # Advice
    "build_advice_metric",
    "evaluate_advice_case",
    "DEFAULT_ADVICE_CRITERIA",
    "DEFAULT_ADVICE_STEPS",
    # Empathy
    "build_empathy_metric",
    "evaluate_empathy_case",
    "DEFAULT_EMPATHY_CRITERIA",
    "DEFAULT_EMPATHY_STEPS",
]
