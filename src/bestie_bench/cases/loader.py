"""Load test cases from YAML fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import yaml

from bestie_bench.cases.models import Axis, TestCase, ToolCall


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def parse_tool_call(data: dict) -> ToolCall:
    """Parse a tool call from fixture data."""
    return ToolCall(
        name=data["name"],
        arguments=data.get("arguments", {}),
        optional_arguments=data.get("optional_arguments", {}),
    )


def parse_test_case(data: dict, case_id: str) -> TestCase:
    """Parse a test case from fixture data."""
    axis = Axis(data.get("axis", "tool_calling"))
    expected_tools = [
        parse_tool_call(t) for t in data.get("expected_tools", [])
    ]

    evaluation_criteria = data.get("evaluation_criteria")
    evaluation_steps = data.get("evaluation_steps", [])

    # For advice/empathy, criteria can be defined in fixture or defaulted
    if not evaluation_criteria and axis in (Axis.ADVICE, Axis.EMPATHY):
        evaluation_criteria = data.get("rubric", "")

    return TestCase(
        id=case_id,
        axis=axis,
        description=data.get("description", ""),
        user_input=data.get("user_input", ""),
        expected_tools=expected_tools,
        evaluation_criteria=evaluation_criteria,
        evaluation_steps=evaluation_steps,
        reference_response=data.get("reference_response"),
        metadata=data.get("metadata", {}),
    )


def load_case_file(path: Path) -> list[TestCase]:
    """Load all test cases from a single fixture file."""
    data = load_yaml(path) if path.suffix in (".yaml", ".yml") else load_json(path)

    # Support both single case and list of cases
    if isinstance(data, dict) and "cases" in data:
        cases_data = data["cases"]
    elif isinstance(data, list):
        cases_data = data
    else:
        cases_data = [data]

    filename_stem = path.stem
    cases = []
    for i, case_data in enumerate(cases_data):
        case_id = case_data.get("id", f"{filename_stem}_{i:03d}")
        cases.append(parse_test_case(case_data, case_id))

    return cases


def load_fixtures(fixtures_dir: Path) -> dict[Axis, list[TestCase]]:
    """Load all fixtures from a directory, grouped by axis."""
    fixtures_dir = Path(fixtures_dir)

    cases_by_axis: dict[Axis, list[TestCase]] = {
        Axis.TOOL_CALLING: [],
        Axis.ADVICE: [],
        Axis.EMPATHY: [],
    }

    for axis_dir in ("tool_calling", "advice", "empathy"):
        axis_path = fixtures_dir / axis_dir
        if not axis_path.exists():
            continue

        for file_path in sorted(axis_path.glob("*.yaml")) + sorted(axis_path.glob("*.yml")) + sorted(axis_path.glob("*.json")):
            cases = load_case_file(file_path)
            axis = Axis(axis_dir)
            cases_by_axis[axis].extend(cases)

    return cases_by_axis


def load_fixtures_for_axis(fixtures_dir: Path, axis: Axis) -> list[TestCase]:
    """Load fixtures for a specific axis only."""
    fixtures_dir = Path(fixtures_dir) / axis.value
    if not fixtures_dir.exists():
        return []
    cases = []
    for file_path in sorted(fixtures_dir.glob("*.yaml")) + sorted(fixtures_dir.glob("*.yml")) + sorted(fixtures_dir.glob("*.json")):
        cases.extend(load_case_file(file_path))
    return cases


def iter_fixtures(fixtures_dir: Path) -> Iterator[tuple[Path, TestCase]]:
    """Iterate over all fixtures, yielding (file_path, test_case)."""
    fixtures_dir = Path(fixtures_dir)

    for axis in ("tool_calling", "advice", "empathy"):
        axis_path = fixtures_dir / axis
        if not axis_path.exists():
            continue

        for file_path in sorted(axis_path.glob("*.yaml")) + sorted(axis_path.glob("*.yml")) + sorted(axis_path.glob("*.json")):
            for case in load_case_file(file_path):
                yield file_path, case
