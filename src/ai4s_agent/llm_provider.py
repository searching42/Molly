from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from ai4s_agent.schemas import LLMInvocationRecord, LLMProviderConfig


class LLMProviderError(ValueError):
    pass


class LLMProvider(Protocol):
    def complete_json(self, *, messages: list[dict[str, str]], prompt_version: str) -> LLMInvocationRecord:
        ...


Transport = Callable[[str, dict[str, object], dict[str, str], int], dict[str, object]]


class StubLLMProvider:
    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        model: str = "stub",
        response_id: str = "stub",
    ) -> None:
        self.response = response or {}
        self.model = model
        self.response_id = response_id

    def complete_json(self, *, messages: list[dict[str, str]], prompt_version: str) -> LLMInvocationRecord:
        return LLMInvocationRecord(
            provider="stub",
            model=self.model,
            prompt_version=prompt_version,
            response_id=self.response_id,
            raw_response={"messages": messages, "response": self.response},
            parsed_output=self.response,
        )


class OpenAICompatibleProvider:
    def __init__(self, *, config: LLMProviderConfig, transport: Transport | None = None) -> None:
        if not config.endpoint.strip():
            raise LLMProviderError("endpoint is required for openai_compatible provider")
        self.config = config
        self.transport = transport or _default_transport

    def complete_json(self, *, messages: list[dict[str, str]], prompt_version: str) -> LLMInvocationRecord:
        payload: dict[str, object] = {
            "model": self.config.model or "default",
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = self.config.endpoint.rstrip("/") + "/chat/completions"
        try:
            raw = self.transport(url, payload, headers, self.config.timeout_sec)
        except OSError as exc:
            raise LLMProviderError(f"OpenAI-compatible request failed: {exc}") from exc
        parsed_output = _parse_chat_completion_json(raw)
        return LLMInvocationRecord(
            provider="openai_compatible",
            model=self.config.model,
            prompt_version=prompt_version,
            response_id=str(raw.get("id") or ""),
            raw_response=_json_safe_raw(raw),
            parsed_output=parsed_output,
        )


def create_llm_provider(config: LLMProviderConfig, *, transport: Transport | None = None) -> LLMProvider:
    provider = config.provider.strip().lower().replace("-", "_")
    if provider == "stub":
        return StubLLMProvider(response=config.stub_response, model=config.model or "stub", response_id="stub")
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(config=config, transport=transport)
    raise LLMProviderError(f"unknown LLM provider: {config.provider}")


def _default_transport(url: str, payload: dict[str, object], headers: dict[str, str], timeout_sec: int) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    if not isinstance(loaded, dict):
        raise LLMProviderError("OpenAI-compatible response must be a JSON object")
    return loaded


def _parse_chat_completion_json(raw: dict[str, object]) -> dict[str, Any]:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMProviderError("OpenAI-compatible response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMProviderError("OpenAI-compatible choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMProviderError("OpenAI-compatible choice missing message")
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise LLMProviderError("OpenAI-compatible message content must be JSON text")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"OpenAI-compatible message content is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMProviderError("OpenAI-compatible message content must decode to a JSON object")
    return parsed


def _json_safe_raw(raw: dict[str, object]) -> dict[str, Any]:
    return json.loads(json.dumps(raw, ensure_ascii=False, default=str))
