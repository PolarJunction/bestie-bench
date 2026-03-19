"""Unified model client for bestie-bench.

Supports OpenAI-compatible APIs (OpenAI, Anthropic via protocol, local models).
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


@dataclass
class ToolCallResult:
    """A single tool call made by the model."""

    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass
class ModelResponse:
    """Response from a model."""

    content: str
    tool_calls: list[ToolCallResult]
    # Raw provider-specific response for debugging
    raw: dict[str, Any]
    latency_ms: float
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------


class ModelClient(ABC):
    """Abstract model client."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable model/provider name."""
        ...


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------


class OpenAIClient(ModelClient):
    """OpenAI API client (also works for OpenAI-compatible APIs)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def name(self) -> str:
        return f"openai/{self.model}"

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """Send a chat completion request."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        t0 = time.perf_counter()
        resp = self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]

        tool_calls = []
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                tool_calls.append(
                    ToolCallResult(
                        id=tc.get("id"),
                        name=tc["function"]["name"],
                        arguments=parse_json_args(tc["function"]["arguments"]),
                    )
                )

        content = message.get("content") or ""
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
            latency_ms=latency_ms,
            model=self.name(),
            input_tokens=data.get("usage", {}).get("prompt_tokens"),
            output_tokens=data.get("usage", {}).get("completion_tokens"),
        )


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------


class AnthropicClient(ModelClient):
    """Anthropic API client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def name(self) -> str:
        return f"anthropic/{self.model}"

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> ModelResponse:
        """Send a messages API request."""
        # Anthropic uses system message separately
        system = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _anthropic_tools(tools)

        t0 = time.perf_counter()
        resp = self._client.post("/messages", json=payload)
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000

        data = resp.json()
        content = data["content"][0] if data["content"] else ""

        # Handle text blocks
        if hasattr(content, "type"):
            text = content.text if content.type == "text" else ""
        else:
            text = str(content)

        # Handle tool use blocks
        tool_calls = []
        for block in data.get("content", []):
            if hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append(
                    ToolCallResult(
                        id=block.id,
                        name=block.name,
                        arguments=parse_json_args(block.input),
                    )
                )

        return ModelResponse(
            content=text,
            tool_calls=tool_calls,
            raw=data,
            latency_ms=latency_ms,
            model=self.name(),
            input_tokens=data.get("usage", {}).get("input_tokens"),
            output_tokens=data.get("usage", {}).get("output_tokens"),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_client(provider: str, model: str, **kwargs: Any) -> ModelClient:
    """Create a model client from a provider string."""
    match provider.lower():
        case "openai" | "openai-compatible":
            return OpenAIClient(model=model, **kwargs)
        case "anthropic":
            return AnthropicClient(model=model, **kwargs)
        case _:
            msg = f"Unknown provider: {provider}. Supported: openai, anthropic"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_json_args(args_str: str | dict) -> dict[str, Any]:
    """Parse JSON arguments from a string or dict."""
    if isinstance(args_str, dict):
        return args_str
    try:
        import json
        return json.loads(args_str)
    except json.JSONDecodeError:
        return {}


def _anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tools to Anthropic format."""
    anthropic_tools = []
    for tool in tools:
        anthropic_tools.append({
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"]["parameters"],
        })
    return anthropic_tools
