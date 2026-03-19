"""Tests for bestie-bench."""

from bestie_bench.cases.models import Axis, TestCase, ToolCall


def test_axis_enum() -> None:
    assert Axis.TOOL_CALLING.value == "tool_calling"
    assert Axis.ADVICE.value == "advice"
    assert Axis.EMPATHY.value == "empathy"


def test_test_case_creation() -> None:
    tc = TestCase(
        id="test_001",
        axis=Axis.TOOL_CALLING,
        description="Schedule a reminder",
        user_input="Remind me to call mum at 9am Tuesday",
        expected_tools=[
            ToolCall(name="schedule_reminder", arguments={"time": "09:00", "day": "tuesday"}),
        ],
    )
    assert tc.id == "test_001"
    assert tc.axis == Axis.TOOL_CALLING
    assert len(tc.expected_tools) == 1
    assert tc.expected_tools[0].name == "schedule_reminder"


def test_test_case_axis_parsing() -> None:
    """Test that string axis values are correctly parsed."""
    tc = TestCase(
        id="test_002",
        axis="empathy",
        description="Test empathy",
        user_input="I'm feeling down",
    )
    assert tc.axis == Axis.EMPATHY
