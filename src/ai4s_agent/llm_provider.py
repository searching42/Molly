from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ValidationError

from ai4s_agent.schemas import LLMInvocationRecord, LLMProviderConfig


class LLMProviderError(ValueError):
    pass


class LLMResponseValidationError(LLMProviderError):
    pass


ResponseModel = type[BaseModel]


class LLMProvider(Protocol):
    def complete_text(
        self,
        *,
        messages: list[dict[str, str]],
        prompt_version: str,
    ) -> str:
        ...

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        prompt_version: str,
        response_model: ResponseModel | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMInvocationRecord:
        ...

    def stream_text(
        self,
        *,
        messages: list[dict[str, str]],
        prompt_version: str,
    ) -> Iterator[str]:
        ...

    def close(self) -> None:
        ...


Transport = Callable[[str, dict[str, object], dict[str, str], int], dict[str, object]]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


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

    def complete_text(
        self,
        *,
        messages: list[dict[str, str]],
        prompt_version: str,
    ) -> str:
        del messages, prompt_version
        for key in ("text", "reply", "content"):
            value = self.response.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(self.response, ensure_ascii=False)

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        prompt_version: str,
        response_model: ResponseModel | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMInvocationRecord:
        parsed = _validate_structured_output(
            self.response,
            response_model=response_model,
            response_schema=response_schema,
        )
        return LLMInvocationRecord(
            provider="stub",
            model=self.model,
            prompt_version=prompt_version,
            response_id=self.response_id,
            raw_response={"messages": messages, "response": self.response},
            parsed_output=parsed,
        )

    def stream_text(
        self,
        *,
        messages: list[dict[str, str]],
        prompt_version: str,
    ) -> Iterator[str]:
        yield self.complete_text(messages=messages, prompt_version=prompt_version)

    def close(self) -> None:
        return None


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        config: LLMProviderConfig,
        transport: Transport | None = None,
        client: httpx.Client | None = None,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        _validate_endpoint(config.endpoint)
        if transport is not None and client is not None:
            raise LLMProviderError("transport and client are mutually exclusive")
        self.config = config
        self.transport = transport
        self._clock = clock
        self._sleeper = sleeper
        self._closed = False
        self._owns_client = transport is None and client is None
        self._client = (
            client
            if transport is None and client is not None
            else httpx.Client(follow_redirects=False) if transport is None else None
        )

    def complete_text(
        self,
        *,
        messages: list[dict[str, str]],
        prompt_version: str,
    ) -> str:
        del prompt_version
        deadline = self._clock() + float(self.config.total_timeout_sec)
        payload = self._payload(messages=messages)
        raw = self._request_raw(payload, deadline=deadline)
        text = _extract_completion_text(raw)
        self._check_deadline(deadline)
        return text

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        prompt_version: str,
        response_model: ResponseModel | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMInvocationRecord:
        _validate_schema_arguments(response_model=response_model, response_schema=response_schema)
        deadline = self._clock() + float(self.config.total_timeout_sec)
        payload = self._payload(
            messages=messages,
            json_mode=True,
            response_model=response_model,
            response_schema=response_schema,
        )
        raw = self._request_raw(payload, deadline=deadline)
        parsed_output = _parse_chat_completion_json(raw)
        parsed_output = _validate_structured_output(
            parsed_output,
            response_model=response_model,
            response_schema=response_schema,
        )
        self._check_deadline(deadline)
        return LLMInvocationRecord(
            provider="openai_compatible",
            # Bind the effective model sent on the wire, including the
            # OpenAI-compatible default used for an empty configuration.
            model=str(payload["model"]),
            prompt_version=prompt_version,
            response_id=_sanitize_text(raw.get("id"), self.config.api_key),
            raw_response=_json_safe_raw(raw, self.config.api_key),
            parsed_output=parsed_output,
        )

    def stream_text(
        self,
        *,
        messages: list[dict[str, str]],
        prompt_version: str,
    ) -> Iterator[str]:
        del prompt_version
        self._ensure_open()
        if self.transport is not None:
            yield self.complete_text(messages=messages, prompt_version="stream.compat")
            return
        if self._client is None:
            raise LLMProviderError("OpenAI-compatible HTTP client is unavailable")
        payload = self._payload(messages=messages)
        payload["stream"] = True
        deadline = self._clock() + float(self.config.total_timeout_sec)
        attempts = 0
        yielded = False
        while True:
            try:
                timeout = self._httpx_timeout(deadline)
                with self._client.stream(
                    "POST",
                    self._url,
                    json=payload,
                    headers=self._stream_headers,
                    timeout=timeout,
                ) as response:
                    self._raise_for_status(response)
                    for line in response.iter_lines():
                        self._check_deadline(deadline)
                        chunk = _parse_sse_text_delta(line)
                        if chunk is _SSE_DONE:
                            return
                        if chunk is None:
                            continue
                        yielded = True
                        yield chunk
                    return
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                if yielded or attempts >= self.config.max_connect_retries:
                    raise self._request_error(exc) from exc
                self._retry_pause(attempts=attempts, deadline=deadline)
                attempts += 1
            except LLMProviderError:
                raise
            except httpx.HTTPError as exc:
                raise self._request_error(exc) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client and self._client is not None:
            self._client.close()

    def __enter__(self) -> OpenAICompatibleProvider:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    @property
    def _url(self) -> str:
        return self.config.endpoint.rstrip("/") + "/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        return self._headers_for(accept="application/json")

    @property
    def _stream_headers(self) -> dict[str, str]:
        return self._headers_for(accept="text/event-stream")

    def _headers_for(self, *, accept: str) -> dict[str, str]:
        headers = {"Accept": accept, "Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _payload(
        self,
        *,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        response_model: ResponseModel | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.config.model or "default",
            "messages": messages,
        }
        if response_model is not None:
            payload["response_format"] = _json_schema_response_format(
                name=response_model.__name__,
                schema=response_model.model_json_schema(),
            )
        elif response_schema is not None:
            payload["response_format"] = _json_schema_response_format(
                name="molly_response",
                schema=response_schema,
            )
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _request_raw(
        self,
        payload: dict[str, object],
        *,
        deadline: float,
    ) -> dict[str, object]:
        self._ensure_open()
        if self.transport is not None:
            try:
                raw = self.transport(
                    self._url,
                    payload,
                    self._headers,
                    self.config.timeout_sec,
                )
            except OSError as exc:
                raise self._request_error(exc) from exc
            self._check_deadline(deadline)
            if not isinstance(raw, dict):
                raise LLMProviderError("OpenAI-compatible response must be a JSON object")
            return raw
        if self._client is None:
            raise LLMProviderError("OpenAI-compatible HTTP client is unavailable")

        attempts = 0
        while True:
            try:
                response = self._client.post(
                    self._url,
                    json=payload,
                    headers=self._headers,
                    timeout=self._httpx_timeout(deadline),
                )
                self._check_deadline(deadline)
                self._raise_for_status(response)
                try:
                    loaded = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise LLMProviderError(
                        "OpenAI-compatible response body is not valid JSON"
                    ) from exc
                if not isinstance(loaded, dict):
                    raise LLMProviderError(
                        "OpenAI-compatible response must be a JSON object"
                    )
                return loaded
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                if attempts >= self.config.max_connect_retries:
                    raise self._request_error(exc) from exc
                self._retry_pause(attempts=attempts, deadline=deadline)
                attempts += 1
            except LLMProviderError:
                raise
            except httpx.HTTPError as exc:
                raise self._request_error(exc) from exc

    def _httpx_timeout(self, deadline: float) -> httpx.Timeout:
        remaining = self._remaining(deadline)
        return httpx.Timeout(
            connect=min(float(self.config.connect_timeout_sec), remaining),
            read=min(float(self.config.timeout_sec), remaining),
            write=min(float(self.config.write_timeout_sec), remaining),
            pool=min(float(self.config.pool_timeout_sec), remaining),
        )

    def _retry_pause(self, *, attempts: int, deadline: float) -> None:
        remaining = self._remaining(deadline)
        delay = min(float(self.config.retry_backoff_sec) * (2**attempts), remaining)
        if delay > 0:
            self._sleeper(delay)
        self._check_deadline(deadline)

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise LLMProviderError("OpenAI-compatible request exceeded total deadline")
        return remaining

    def _check_deadline(self, deadline: float) -> None:
        if self._clock() >= deadline:
            raise LLMProviderError("OpenAI-compatible request exceeded total deadline")

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 300:
            return
        request_id = response.headers.get("x-request-id") or response.headers.get(
            "x-correlation-id"
        )
        suffix = f" (request_id={_sanitize_text(request_id, self.config.api_key)})" if request_id else ""
        raise LLMProviderError(
            f"OpenAI-compatible request returned HTTP {response.status_code}{suffix}"
        )

    def _request_error(self, exc: BaseException) -> LLMProviderError:
        detail = _sanitize_text(str(exc), self.config.api_key)
        return LLMProviderError(f"OpenAI-compatible request failed: {detail}")

    def _ensure_open(self) -> None:
        if self._closed:
            raise LLMProviderError("OpenAI-compatible provider is closed")


@dataclass
class _ProviderEntry:
    fingerprint: str
    provider: LLMProvider
    leases: int = 0
    retired: bool = False


ProviderFactory = Callable[[LLMProviderConfig], LLMProvider]


class LLMProviderManager:
    """Lease one shared settings provider without closing in-flight requests."""

    def __init__(self, *, provider_factory: ProviderFactory | None = None) -> None:
        self._lock = threading.RLock()
        self._entry: _ProviderEntry | None = None
        self._retired: list[_ProviderEntry] = []
        self._provider_factory = provider_factory
        self._closed = False

    @contextmanager
    def lease(self, config: LLMProviderConfig) -> Iterator[LLMProvider]:
        fingerprint = _config_fingerprint(config)
        with self._lock:
            if self._closed:
                raise LLMProviderError("LLM provider manager is closed")
            if self._entry is None or self._entry.fingerprint != fingerprint:
                self._retire_current_locked()
                factory = self._provider_factory or create_llm_provider
                self._entry = _ProviderEntry(
                    fingerprint=fingerprint,
                    provider=factory(config),
                )
            entry = self._entry
            entry.leases += 1
        try:
            yield entry.provider
        finally:
            with self._lock:
                entry.leases -= 1
                self._close_if_unused_locked(entry)

    def invalidate(self) -> None:
        with self._lock:
            self._retire_current_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._retire_current_locked()

    def _retire_current_locked(self) -> None:
        if self._entry is None:
            return
        entry = self._entry
        self._entry = None
        entry.retired = True
        self._retired.append(entry)
        self._close_if_unused_locked(entry)

    def _close_if_unused_locked(self, entry: _ProviderEntry) -> None:
        if not entry.retired or entry.leases:
            return
        entry.provider.close()
        if entry in self._retired:
            self._retired.remove(entry)


def create_llm_provider(
    config: LLMProviderConfig,
    *,
    transport: Transport | None = None,
    client: httpx.Client | None = None,
) -> LLMProvider:
    provider = config.provider.strip().lower().replace("-", "_")
    if provider == "stub":
        return StubLLMProvider(
            response=config.stub_response,
            model=config.model or "stub",
            response_id="stub",
        )
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(
            config=config,
            transport=transport,
            client=client,
        )
    raise LLMProviderError(f"unknown LLM provider: {config.provider}")


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlparse(str(endpoint or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMProviderError("endpoint is required for openai_compatible provider")
    if parsed.username or parsed.password:
        raise LLMProviderError("OpenAI-compatible endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise LLMProviderError("OpenAI-compatible endpoint must not contain query or fragment")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname or ""):
        raise LLMProviderError("non-loopback OpenAI-compatible endpoints must use https")


def _is_loopback_host(host: str) -> bool:
    import ipaddress

    clean = str(host or "").strip().lower()
    if clean == "localhost":
        return True
    try:
        return ipaddress.ip_address(clean).is_loopback
    except ValueError:
        return False


def _json_schema_response_format(*, name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64] or "molly_response",
            "strict": True,
            "schema": schema,
        },
    }


def _validate_schema_arguments(
    *,
    response_model: ResponseModel | None,
    response_schema: dict[str, Any] | None,
) -> None:
    if response_model is not None and response_schema is not None:
        raise LLMProviderError("response_model and response_schema are mutually exclusive")
    if response_schema is not None:
        try:
            Draft202012Validator.check_schema(response_schema)
        except SchemaError as exc:
            raise LLMResponseValidationError(
                f"invalid response JSON Schema: {_sanitize_text(exc.message)}"
            ) from exc


def _validate_structured_output(
    payload: dict[str, Any],
    *,
    response_model: ResponseModel | None,
    response_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    _validate_schema_arguments(response_model=response_model, response_schema=response_schema)
    if response_model is not None:
        try:
            validated = response_model.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors(include_url=False, include_input=False)[0]
            location = ".".join(str(item) for item in first.get("loc", ())) or "response"
            raise LLMResponseValidationError(
                f"structured response failed Pydantic validation at {location}: "
                f"{_sanitize_text(str(first.get('msg') or 'invalid value'))}"
            ) from exc
        dumped = validated.model_dump(mode="json")
        if not isinstance(dumped, dict):
            raise LLMResponseValidationError("Pydantic response model must serialize to an object")
        return dumped
    if response_schema is not None:
        validator = Draft202012Validator(response_schema)
        error = next(iter(validator.iter_errors(payload)), None)
        if error is not None:
            location = ".".join(str(item) for item in error.absolute_path) or "response"
            constraint = str(error.validator or "schema")
            raise LLMResponseValidationError(
                f"structured response failed JSON Schema validation at {location}: "
                f"{constraint} constraint failed"
            ) from error
    return payload


def _parse_chat_completion_json(raw: dict[str, object]) -> dict[str, Any]:
    message = _first_message(raw)
    candidates: list[tuple[str, Any]] = [
        ("content", message.get("content")),
        ("reasoning_content", message.get("reasoning_content")),
    ]
    failures: list[str] = []
    for label, content in candidates:
        if isinstance(content, dict):
            return content
        text = _content_to_text(content)
        if not text:
            failures.append(f"{label}: empty")
            continue
        try:
            return _decode_json_object(text)
        except LLMProviderError as exc:
            failures.append(f"{label}: {exc}")
    raise LLMProviderError(
        "OpenAI-compatible response did not contain a JSON object "
        f"({' ; '.join(failures)})"
    )


def _extract_completion_text(raw: dict[str, object]) -> str:
    message = _first_message(raw)
    text = _content_to_text(message.get("content"))
    if not text:
        raise LLMProviderError("OpenAI-compatible response missing text content")
    return text


def _first_message(raw: dict[str, object]) -> dict[str, Any]:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMProviderError("OpenAI-compatible response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMProviderError("OpenAI-compatible choice must be an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMProviderError("OpenAI-compatible choice missing message")
    return message


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for block in content:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            chunks.append(str(block["text"]))
    return "".join(chunks)


def _decode_json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines and lines[-1].strip() == "```":
            clean = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    decoder = json.JSONDecoder()
    for index, char in enumerate(clean):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(clean[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    raise LLMProviderError("content is not valid JSON object text")


_SSE_DONE = object()


def _parse_sse_text_delta(line: str) -> str | None | object:
    clean = str(line or "").strip()
    if not clean or clean.startswith(":"):
        return None
    if clean.startswith("data:"):
        clean = clean[5:].strip()
    elif not clean.startswith("{"):
        return None
    if clean == "[DONE]":
        return _SSE_DONE
    try:
        event = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise LLMProviderError("OpenAI-compatible stream emitted invalid JSON") from exc
    if not isinstance(event, dict):
        return None
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return None
    text = _content_to_text(delta.get("content"))
    return text or None


def _json_safe_raw(raw: dict[str, object], api_key: str = "") -> dict[str, Any]:
    safe = json.loads(json.dumps(raw, ensure_ascii=False, default=str))
    return _redact_value(safe, api_key)


def _redact_value(value: Any, api_key: str = "") -> Any:
    if isinstance(value, str):
        return _redact_text(value, api_key)
    if isinstance(value, list):
        return [_redact_value(item, api_key) for item in value]
    if not isinstance(value, dict):
        return value
    redacted: dict[str, Any] = {}
    sensitive = {
        "authorization",
        "api_key",
        "api-key",
        "x-api-key",
        "access_token",
        "refresh_token",
        "reasoning_content",
    }
    for key, item in value.items():
        redacted[str(key)] = (
            "[REDACTED]"
            if str(key).lower() in sensitive
            else _redact_value(item, api_key)
        )
    return redacted


def _sanitize_text(value: Any, api_key: str = "") -> str:
    return _redact_text(str(value or ""), api_key)[:600]


def _redact_text(value: str, api_key: str = "") -> str:
    clean = str(value or "")
    if api_key:
        clean = clean.replace(api_key, "[REDACTED]")
    clean = re.sub(r"(?i)bearer\s+[a-z0-9._~+/-]+", "Bearer [REDACTED]", clean)
    clean = re.sub(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|token)=)[^&\s]+",
        r"\1[REDACTED]",
        clean,
    )
    return clean


def _config_fingerprint(config: LLMProviderConfig) -> str:
    payload = config.model_dump(mode="json")
    api_key = str(payload.pop("api_key", ""))
    payload["api_key_sha256"] = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
