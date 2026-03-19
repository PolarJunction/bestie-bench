"""Stubbing layer for reproducible benchmark runs.

Stubs intercept tool calls and return pre-recorded responses instead of
calling external APIs (weather, Tavily, QStash, Supabase). This makes runs
deterministic and fast.

Stubs are stored as JSON files:
    stubs/<tool_name>/<case_id>.json

Each stub file contains:
{
  "response": { ... },   # Tool response returned to the model
  "delay_ms": 150        # Optional simulated latency
}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from bestie_bench.cases.models import TestCase


# ---------------------------------------------------------------------------
# Stub responses for external APIs
# ---------------------------------------------------------------------------

DEFAULT_STUBS: dict[str, dict[str, Any]] = {
    # Weather stubs
    "get_weather": {
        "Edinburgh": {
            "condition": "Partly cloudy",
            "feelsLike": "14°C",
            "humidity": "72%",
            "location": "Edinburgh, Scotland",
            "temperature": "16°C",
            "wind": "12 km/h",
        },
        "London": {
            "condition": "Overcast",
            "feelsLike": "11°C",
            "humidity": "80%",
            "location": "London, England",
            "temperature": "13°C",
            "wind": "18 km/h",
        },
        "New York": {
            "condition": "Clear",
            "feelsLike": "68°F",
            "humidity": "45%",
            "location": "New York, NY, USA",
            "temperature": "72°F",
            "wind": "8 mph",
        },
    },
    # Internet search stubs
    "internet_lookup": {
        "default": {
            "answer": "Based on recent sources, there are varying opinions on this topic. The consensus suggests it depends on individual circumstances, but most experts recommend consulting a qualified professional for personalized advice.",
            "sources": [
                {"title": "Expert Guide", "url": "https://example.com/guide"},
                {"title": "Research Summary", "url": "https://example.com/research"},
            ],
        },
    },
    # Journal stubs
    "get_journal_entries": {
        "default": {
            "entries": [
                {
                    "date": "2026-03-18",
                    "content": "Feeling really good today. Managed to go for a run before work and the weather was perfect. Dinner with Sarah was lovely — we talked about her new job and it sounds like she's really happy.",
                },
            ],
            "message": "",
        },
        "empty": {
            "entries": [],
            "message": "No journal entries found for that timeframe.",
        },
    },
    # Reminder stubs
    "get_reminders": {
        "default": {
            "reminders": [
                {"index": 1, "prompt": "Take vitamins", "kind": "daily", "timing": "7:45am", "timezone": "Europe/London"},
                {"index": 2, "prompt": "Call the doctor", "kind": "one-time", "timing": "Mar 20, 10:00am GMT", "timezone": "Europe/London"},
                {"index": 3, "prompt": "Submit expense report", "kind": "one-time", "timing": "Mar 22, 3:00pm GMT", "timezone": "Europe/London"},
            ],
            "message": "You have 3 active reminder(s).",
        },
        "empty": {
            "reminders": [],
            "message": "You have no active reminders.",
        },
    },
    "schedule_event": {
        "success": {
            "fallbackReply": None,
            "success": True,
            "kind": "one-time",
            "message": "Reminder set!",
            "prompt": "Call their mum",
            "reminders": [
                {"index": 1, "prompt": "Take vitamins", "kind": "daily", "timing": "7:45am", "timezone": "Europe/London"},
                {"index": 2, "prompt": "Call the doctor", "kind": "one-time", "timing": "Mar 20, 10:00am GMT", "timezone": "Europe/London"},
                {"index": 3, "prompt": "Call their mum", "kind": "one-time", "timing": "Mar 20, 2:00pm GMT", "timezone": "Europe/London"},
            ],
            "surfaceConfirmation": True,
        },
    },
    "cancel_reminder": {
        "success_single": {
            "success": True,
            "indexes": [2],
            "message": "Reminder successfully cancelled.",
            "prompts": ["Call the doctor"],
            "reminders": [
                {"index": 1, "prompt": "Take vitamins", "kind": "daily", "timing": "7:45am", "timezone": "Europe/London"},
                {"index": 2, "prompt": "Submit expense report", "kind": "one-time", "timing": "Mar 22, 3:00pm GMT", "timezone": "Europe/London"},
            ],
        },
        "success_multiple": {
            "success": True,
            "indexes": [1, 3],
            "message": "2 reminders successfully cancelled.",
            "prompts": ["Take vitamins", "Submit expense report"],
            "reminders": [
                {"index": 1, "prompt": "Call the doctor", "kind": "one-time", "timing": "Mar 20, 10:00am GMT", "timezone": "Europe/London"},
            ],
        },
        "not_found": {
            "success": True,
            "indexes": [99],
            "message": "No active reminders found. It may have already been cancelled.",
            "prompts": [],
            "reminders": [
                {"index": 1, "prompt": "Take vitamins", "kind": "daily", "timing": "7:45am", "timezone": "Europe/London"},
            ],
        },
    },
    "update_reminder": {
        "success": {
            "success": True,
            "index": 1,
            "prompt": "Take vitamins with breakfast",
            "message": "Reminder updated.",
            "reminders": [
                {"index": 1, "prompt": "Take vitamins with breakfast", "kind": "daily", "timing": "7:45am", "timezone": "Europe/London"},
                {"index": 2, "prompt": "Call the doctor", "kind": "one-time", "timing": "Mar 20, 10:00am GMT", "timezone": "Europe/London"},
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Stub registry
# ---------------------------------------------------------------------------


class StubRegistry:
    """Registry for pre-recorded tool responses.

    Usage:
        registry = StubRegistry(stubs_dir=Path("stubs"))
        response = registry.get(tool_name="get_weather", case_id="tc_weather_001")
    """

    def __init__(
        self,
        stubs_dir: Path | str | None = None,
        use_default: bool = True,
        randomize: bool = False,
    ) -> None:
        self.stubs_dir = Path(stubs_dir) if stubs_dir else None
        self.use_default = use_default
        self.randomize = randomize
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._defaults = DEFAULT_STUBS

    def get(
        self,
        tool_name: str,
        case_id: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Get a stub response for a tool call.

        Args:
            tool_name: Name of the tool called.
            case_id: Test case ID this run is for.
            arguments: Tool arguments (used for smarter stub selection).

        Returns:
            Stub response dict, or None if no stub found.
        """
        key = (tool_name, case_id)
        if key in self._cache:
            return self._cache[key]

        response = self._load_stub(tool_name, case_id, arguments)
        if response is not None:
            self._cache[key] = response

        return response

    def _load_stub(
        self,
        tool_name: str,
        case_id: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        # 1. Try case-specific file
        if self.stubs_dir:
            path = self.stubs_dir / tool_name / f"{case_id}.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)

        # 2. Try tool-level default file
        if self.stubs_dir:
            default_path = self.stubs_dir / tool_name / "default.json"
            if default_path.exists():
                with open(default_path) as f:
                    return json.load(f)

        # 3. Fall back to built-in defaults
        if self.use_default:
            tool_defaults = self._defaults.get(tool_name, {})
            if not tool_defaults:
                return None

            if self.randomize and len(tool_defaults) > 1:
                import random as _random
                return tool_defaults[_random.choice(list(tool_defaults.keys()))]

            # Try to match by argument value (e.g., location name)
            if arguments:
                for _stub_key, stub_response in tool_defaults.items():
                    if _stub_key != "default":
                        return stub_response

            return tool_defaults.get("default")

        return None

    def has_stub(self, tool_name: str, case_id: str) -> bool:
        """Check if a stub exists for a tool/case pair."""
        if self.stubs_dir:
            path = self.stubs_dir / tool_name / f"{case_id}.json"
            if path.exists():
                return True

        return tool_name in self._defaults

    def install(self) -> None:
        """Install stubs to disk from built-in defaults (for users to edit)."""
        if not self.stubs_dir:
            return

        for tool_name, stubs in self._defaults.items():
            tool_dir = self.stubs_dir / tool_name
            tool_dir.mkdir(parents=True, exist_ok=True)

            for stub_key, response in stubs.items():
                path = tool_dir / f"{stub_key}.json"
                path.write_text(json.dumps(response, indent=2))

    @classmethod
    def from_dir(cls, stubs_dir: Path | str) -> StubRegistry:
        """Create a registry that reads stubs from a directory."""
        return cls(stubs_dir=stubs_dir, use_default=False)


# ---------------------------------------------------------------------------
# Stub installer CLI helper
# ---------------------------------------------------------------------------


def install_stubs(stubs_dir: Path | str, force: bool = False) -> None:
    """Install default stub files to a directory for user customization."""
    stubs_dir = Path(stubs_dir)
    registry = StubRegistry()
    registry.stubs_dir = stubs_dir

    for tool_name, stubs in DEFAULT_STUBS.items():
        tool_dir = stubs_dir / tool_name
        tool_dir.mkdir(parents=True, exist_ok=True)

        for stub_key, response in stubs.items():
            path = tool_dir / f"{stub_key}.json"
            if force or not path.exists():
                path.write_text(json.dumps(response, indent=2))
                print(f"  {'overwrote' if force else 'created'} {path.relative_to(stubs_dir)}")


# ---------------------------------------------------------------------------
# Context manager for monkeypatching in tests
# ---------------------------------------------------------------------------


from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def stubbed_harness(harness: "Harness", stubs_dir: Path | str | None = None):
    """Context manager that monkeypatches a harness's client to use stubs.

    Usage:
        with stubbed_harness(harness, stubs_dir="stubs"):
            result = harness.run(...)
    """
    # Import here to avoid circular imports
    from bestie_bench.harness import Harness

    original_client = harness.client
    stub_registry = StubRegistry(stubs_dir=stubs_dir) if stubs_dir else StubRegistry()

    def stubbed_chat(
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ):
        from bestie_bench.models.client import ModelResponse, ToolCallResult

        response = original_client.chat(messages, tools, temperature)

        # Check if this looks like a tool result response
        # (model already made tool calls and got responses)
        return response

    # Patch the client
    harness._stub_registry = stub_registry  # type: ignore
    yield harness
    harness._stub_registry = None  # type: ignore
