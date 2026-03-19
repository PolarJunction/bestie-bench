"""Empathy metric using DeepEval GEval.

Evaluates whether the agent responds with emotional intelligence,
warmth, and appropriate validation of feelings.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from bestie_bench.cases.models import TestCase


# Default empathy criteria
DEFAULT_EMPATHY_CRITERIA = (
    "Evaluate the empathetic quality along these dimensions:\n"
    "1. **Emotional Acknowledgment** — Does the response acknowledge and validate the user's feelings?\n"
    "2. **Tone** — Is the tone warm, supportive, and appropriate for the emotional context?\n"
    "3. **Non-judgment** — Does the agent avoid being preachy, dismissive, or morally superiority?\n"
    "4. **Toxic Positivity Check** — Does the agent avoid empty platitudes ('Everything will be fine!')?\n"
    "5. **Personalization** — Does the response feel tailored to this specific situation and person?\n"
    "Rate from 0 (completely fails) to 1 (deeply empathetic)."
)

DEFAULT_EMPATHY_STEPS = [
    "Check whether the response explicitly names or acknowledges the user's emotional state.",
    "Assess the warmth and naturalness of the tone.",
    "Note any signs of judgment, dismissiveness, or condescension.",
    "Flag any instances of toxic positivity (forced optimism, platitudes).",
    "Check if the response feels generic/copy-pasted or personalized to this specific situation.",
    "Combine these factors into a final 0-1 empathy score.",
]


def build_empathy_metric(
    criteria: str | None = None,
    evaluation_steps: list[str] | None = None,
    model: str = "gpt-4o",
    threshold: float = 0.5,
) -> GEval:
    """Build a GEval metric for empathy.

    Args:
        criteria: Custom criteria text. Uses DEFAULT_EMPATHY_CRITERIA if None.
        evaluation_steps: Custom evaluation steps. Uses DEFAULT_EMPATHY_STEPS if None.
        model: LLM to use as judge.
        threshold: Minimum passing score.
    """
    return GEval(
        name="Empathy",
        criteria=criteria or DEFAULT_EMPATHY_CRITERIA,
        evaluation_steps=evaluation_steps or DEFAULT_EMPATHY_STEPS,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=model,
        threshold=threshold,
        verbose_mode=False,
    )


def evaluate_empathy_case(
    case: TestCase,
    actual_output: str,
    judge_model: str = "gpt-4o",
) -> tuple[float, str]:
    """Evaluate a single empathy test case.

    Returns:
        Tuple of (score, reason)
    """
    metric = build_empathy_metric(
        criteria=case.evaluation_criteria,
        evaluation_steps=case.evaluation_steps if case.evaluation_steps else None,
        model=judge_model,
        threshold=0.0,
    )

    test_case = {
        "input": case.user_input,
        "actual_output": actual_output,
    }

    metric.measure(
        input=test_case["input"],
        actual_output=test_case["actual_output"],
    )

    return metric.score, metric.reason or ""


def build_case_params(case: TestCase, actual_output: str) -> dict[str, Any]:
    """Build evaluation params dict for GEval."""
    return {
        "input": case.user_input,
        "actual_output": actual_output,
    }
