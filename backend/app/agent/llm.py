"""Streaming chat-completions client for Groq (OpenAI-compatible), over plain httpx.

LLMClient.stream() yields TextDelta events as tokens arrive and ends with one Completion
holding the assembled tool calls. Closing the iterator (or cancelling its task) closes the
HTTP response, so an abandoned generation stops consuming tokens.
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, as produced by the model

    def parsed_arguments(self) -> dict[str, Any]:
        """The arguments as a dict; malformed JSON becomes {} (validation then rejects it)."""
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


LLMEvent = TextDelta | Completion


class LLMError(Exception):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after_s = retry_after_s  # from the provider's retry-after header, if any


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


class LLMClient(Protocol):
    def stream(
        self, messages: list[Message], tools: list[dict[str, Any]], *, stop: list[str] | None = None
    ) -> AsyncIterator[LLMEvent]: ...


class _ToolCallBuilder:
    """Tool calls arrive as fragments keyed by index; arguments are concatenated text."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, str]] = {}

    def add(self, fragments: list[dict[str, Any]]) -> None:
        for frag in fragments:
            call = self._calls.setdefault(frag.get("index", 0), {"id": "", "name": "", "args": ""})
            call["id"] = frag.get("id") or call["id"]
            fn = frag.get("function") or {}
            call["name"] = fn.get("name") or call["name"]
            call["args"] += fn.get("arguments") or ""

    def build(self) -> list[ToolCall]:
        return [
            ToolCall(id=c["id"] or f"call_{i}", name=c["name"], arguments=c["args"])
            for i, c in sorted(self._calls.items())
        ]


class GroqClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.2,
        timeout_s: float = 20.0,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,  # tests pass a mock; production uses the default
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def stream(
        self, messages: list[Message], tools: list[dict[str, Any]], *, stop: list[str] | None = None
    ) -> AsyncIterator[LLMEvent]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.max_completion_tokens:
            body["max_completion_tokens"] = self.max_completion_tokens
        if self.reasoning_effort:
            # GPT-OSS: reasoning still costs tokens, but we don't need its text back.
            body["reasoning_effort"] = self.reasoning_effort
            body["include_reasoning"] = False
        if stop:
            body["stop"] = stop
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        text: list[str] = []
        calls = _ToolCallBuilder()
        finish_reason = None
        usage: dict[str, Any] = {}
        try:
            async with self._http.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode(errors="replace")[:500]
                    retryable = response.status_code == 429 or response.status_code >= 500
                    raise LLMError(
                        f"LLM HTTP {response.status_code}: {detail}",
                        retryable=retryable,
                        status=response.status_code,
                        retry_after_s=_retry_after(response),
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            text.append(delta["content"])
                            yield TextDelta(delta["content"])
                        if delta.get("tool_calls"):
                            calls.add(delta["tool_calls"])
                        finish_reason = choice.get("finish_reason") or finish_reason
        except httpx.TimeoutException as exc:
            raise LLMError("LLM request timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM transport error: {type(exc).__name__}", retryable=True) from exc
        yield Completion(
            text="".join(text), tool_calls=calls.build(), finish_reason=finish_reason, usage=usage
        )
