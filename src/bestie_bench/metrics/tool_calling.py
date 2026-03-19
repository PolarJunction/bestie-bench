"""Tool calling metrics using DeepEval.

Provides ToolCorrectnessMetric and ArgumentCorrectnessMetric wrapped
for use in the bestie-bench harness.
"""

from __future__ import annotations

from typing import Any

from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall

from bestie_bench.cases.models import TestCase, TestResult


def build_tool_correctness_metric(
    include_input_params: bool = True,
    include_output: bool = False,
    threshold: float = 0.5,
) -> ToolCorrectnessMetric:
    """Build a ToolCorrectnessMetric for tool calling evaluation.

    Args:
        include_input_params: Require input params to match expected.
        include_output: Require tool output to match (not applicable for harnesses).
        threshold: Minimum passing score (0-1).
    """
    evaluation_params = [ToolCallParams.TOOL]
    if include_input_params:
        evaluation_params.append(ToolCallParams.INPUT_PARAMETERS)

    return ToolCorrectnessMetric(
        threshold=threshold,
        evaluation_params=evaluation_params,
        should_consider_ordering=False,
        strict_mode=False,
    )


def build_argument_correctness_metric(
    threshold: float = 0.5,
) -> ArgumentCorrectnessMetric:
    """Build an ArgumentCorrectnessMetric for tool argument validation.

    Args:
        threshold: Minimum passing score (0-1).
    """
    return ArgumentCorrectnessMetric(
        threshold=threshold,
        strict_mode=False,
    )


def create_tool_test_case(
    case: TestCase,
    actual_output: str,
    tools_called: list[dict[str, Any]],
) -> LLMTestCase:
    """Create a DeepEval LLMTestCase for tool calling.

    Args:
        case: The test case definition.
        actual_output: The model's text response.
        tools_called: List of tool calls made, each as dict with name/arguments.
    """
    expected_tools = [
        ToolCall(name=t.name, args=t.arguments)
        for t in case.expected_tools
    ]

    called_tools = [
        ToolCall(name=tc["name"], args=tc.get("arguments", {}))
        for tc in tools_called
    ]

    return LLMTestCase(
        input=case.user_input,
        actual_output=actual_output,
        expected_tools=expected_tools,
        tools_called=called_tools,
    )


def evaluate_tool_case(
    case: TestCase,
    actual_output: str,
    tools_called: list[dict[str, Any]],
    include_args: bool = True,
) -> tuple[float, float, list[str]]:
    """Evaluate a single tool calling test case.

    Returns:
        Tuple of (tool_score, argument_score, reasons)
    """
    test_case = create_tool_test_case(case, actual_output, tools_called)

    tool_metric = ToolCorrectnessMetric(
        threshold=0.0,  # We just want the score, not pass/fail
        evaluation_params=[ToolCallParams.TOOL, ToolCallParams.INPUT_PARAMETERS] if include_args else [ToolCallParams.TOOL],
        should_consider_ordering=False,
        strict_mode=False,
        verbose_mode=False,
    )

    arg_metric = ArgumentCorrectnessMetric(
        threshold=0.0,
        strict_mode=False,
        verbose_mode=False,
    )

    tool_metric.measure(test_case)
    arg_metric.measure(test_case)

    return (
        tool_metric.score,
        arg_metric.score,
        [tool_metric.reason or "", arg_metric.reason or ""],
    )


# DeepEval imports
from deepeval.test_case.llm_test_case import ToolCallParams  # noqa: E402
