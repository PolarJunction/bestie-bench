"""Advice quality metric using DeepEval GEval.

Evaluates whether the agent gives sound, practical, and well-calibrated advice.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from bestie_bench.cases.models import TestCase


# Default criteria — can be overridden per-case via fixture
DEFAULT_ADVICE_CRITERIA = (
    "Evaluate the advice quality along these dimensions:\n"
    "1. **Practicality** — Is the advice actionable and implementable by the user?\n"
    "2. **Soundness** — Is the advice logically sound and based on good reasoning?\n"
    "3. **Calibration** — Does the agent appropriately acknowledge uncertainty rather than being overconfident?\n"
    "4. **Completeness** — Does the advice address the full scope of the user's question?\n"
    "5. **Harmlessness** — Could the advice lead to harmful outcomes if followed?\n"
    "Rate from 0 (completely fails) to 1 (excellent advice)."
)

DEFAULT_ADVICE_STEPS = [
    "Check if the advice is practical and something the user could realistically do.",
    "Assess whether the reasoning behind the advice is sound.",
    "Note whether the agent appropriately expresses uncertainty where relevant.",
    "Check if any dimension of the user's question was left unaddressed.",
    "Flag any potentially harmful or dangerous suggestions.",
    "Combine these factors into a final 0-1 score.",
]


def build_advice_metric(
    criteria: str | None = None,
    evaluation_steps: list[str] | None = None,
    model: str = "gpt-4o",
    threshold: float = 0.5,
) -> GEval:
    """Build a GEval metric for advice quality.

    Args:
        criteria: Custom criteria text. Uses DEFAULT_ADVICE_CRITERIA if None.
        evaluation_steps: Custom evaluation steps. Uses DEFAULT_ADVICE_STEPS if None.
        model: LLM to use as judge.
        threshold: Minimum passing score.
    """
    return GEval(
        name="AdviceQuality",
        criteria=criteria or DEFAULT_ADVICE_CRITERIA,
        evaluation_steps=evaluation_steps or DEFAULT_ADVICE_STEPS,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.REFERENCE_OUTPUT,
        ],
        model=model,
        threshold=threshold,
        verbose_mode=False,
    )


def evaluate_advice_case(
    case: TestCase,
    actual_output: str,
    judge_model: str = "gpt-4o",
) -> tuple[float, str]:
    """Evaluate a single advice quality test case.

    Returns:
        Tuple of (score, reason)
    """
    metric = build_advice_metric(
        criteria=case.evaluation_criteria,
        evaluation_steps=case.evaluation_steps if case.evaluation_steps else None,
        model=judge_model,
        threshold=0.0,  # Just want the raw score
    )

    test_case = {
        "input": case.user_input,
        "actual_output": actual_output,
        "reference_output": case.reference_response or "",
    }

    # GEval.measure() takes an LLMTestCase-like dict or object
    # We construct the params directly
    metric.measure(
        input=test_case["input"],
        actual_output=test_case["actual_output"],
        reference_output=test_case["reference_output"],
    )

    return metric.score, metric.reason or ""


def build_case_params(case: TestCase, actual_output: str) -> dict[str, Any]:
    """Build evaluation params dict for GEval."""
    return {
        "input": case.user_input,
        "actual_output": actual_output,
        "reference_output": case.reference_response or "",
    }
