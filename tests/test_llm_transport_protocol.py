from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from threading import Event, Thread

import httpx
import pytest
from pydantic import BaseModel

import ai4s_agent.routes.agents as agent_routes
from ai4s_agent.app import create_app
from ai4s_agent.llm_provider import (
    LLMProviderError,
    LLMProviderManager,
    LLMResponseValidationError,
    OpenAICompatibleProvider,
    StubLLMProvider,
    _config_fingerprint,
)
from ai4s_agent.schemas import LLMProviderConfig


class _Answer(BaseModel):
    answer: str
    score: int


def _config(**overrides) -> LLMProviderConfig:
    payload = {
        "provider": "openai_compatible",
        "endpoint": "https://llm.example.test/v1",
        "api_key": "secret-token",
        "model": "decision-model",
        "retry_backoff_sec": 0,
        **overrides,
    }
    return LLMProviderConfig(**payload)


def _completion(content, *, extra_message: dict | None = None) -> dict:
    message = {"content": content, **(extra_message or {})}
    return {"id": "chatcmpl-test", "choices": [{"message": message}]}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


class _ChunkedSSEStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes], *, error_factory=None, request=None) -> None:
        self.chunks = chunks
        self.error_factory = error_factory
        self.request = request

    def __iter__(self):
        yield from self.chunks
        if self.error_factory is not None:
            raise self.error_factory(self.request)

    def close(self) -> None:
        return None


def _sse_body(
    events: list[dict],
    *,
    include_done: bool = True,
    split_at: int | None = None,
) -> list[bytes]:
    parts = [b": keep-alive\n\n\n"]
    parts.extend(
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
        for event in events
    )
    if include_done:
        parts.append(b"data: [DONE]\n\n")
    body = b"".join(parts)
    if split_at is None or split_at <= 0:
        return [body]
    return [body[index : index + split_at] for index in range(0, len(body), split_at)]


def _structured_sse_response(
    request: httpx.Request,
    events: list[dict],
    *,
    include_done: bool = True,
    split_at: int | None = None,
    error_factory=None,
) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_ChunkedSSEStream(
            _sse_body(events, include_done=include_done, split_at=split_at),
            error_factory=error_factory,
            request=request,
        ),
    )


def test_complete_text_reuses_persistent_client_and_supports_content_blocks() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json=_completion(
                [
                    {"type": "text", "text": "first "},
                    {"type": "text", "text": "answer"},
                ]
            ),
        )

    client = _client(handler)
    provider = OpenAICompatibleProvider(config=_config(), client=client)
    first = provider.complete_text(
        messages=[{"role": "user", "content": "hello"}],
        prompt_version="conversation.v1",
    )
    second = provider.complete_text(
        messages=[{"role": "user", "content": "again"}],
        prompt_version="conversation.v1",
    )

    assert first == "first answer"
    assert second == "first answer"
    assert len(calls) == 2
    assert calls[0].headers["authorization"] == "Bearer secret-token"
    assert "response_format" not in json.loads(calls[0].content)
    assert not client.is_closed
    provider.close()
    assert not client.is_closed  # injected clients retain ownership


def test_owned_client_lifecycle_is_explicit_and_idempotent() -> None:
    provider = OpenAICompatibleProvider(config=_config())
    owned_client = provider._client
    assert owned_client is not None and not owned_client.is_closed
    provider.close()
    provider.close()
    assert owned_client.is_closed
    with pytest.raises(LLMProviderError, match="provider is closed"):
        provider.complete_text(messages=[], prompt_version="closed.v1")


def test_complete_json_validates_pydantic_model_and_sends_generated_schema() -> None:
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"answer":"ok","score":3}'))

    provider = OpenAICompatibleProvider(config=_config(), client=_client(handler))
    result = provider.complete_json(
        messages=[{"role": "user", "content": "structured"}],
        prompt_version="structured.v1",
        response_model=_Answer,
    )

    assert result.parsed_output == {"answer": "ok", "score": 3}
    response_format = captured_payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["required"] == ["answer", "score"]


def test_complete_json_uses_explicit_json_object_capability() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"answer":"ok","score":3}'))

    provider = OpenAICompatibleProvider(
        config=_config(
            endpoint="https://api.deepseek.com",
            model="deepseek-v4-flash",
            capabilities={"structured_output_mode": "json_object_local_validation"},
        ),
        client=_client(handler),
    )
    result = provider.complete_json(
        messages=[{"role": "user", "content": "structured"}],
        prompt_version="structured.v1",
        response_model=_Answer,
    )

    assert result.parsed_output == {"answer": "ok", "score": 3}
    assert len(calls) == 1
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["temperature"] == 0


def test_structured_output_transport_defaults_to_buffered_and_preserves_payload() -> None:
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"answer":"ok","score":3}'))

    config = _config(capabilities={"structured_output_mode": "json_object_local_validation"})
    assert config.structured_output_transport == "buffered"
    provider = OpenAICompatibleProvider(config=config, client=_client(handler))
    provider.complete_json(
        messages=[{"role": "user", "content": "structured"}],
        prompt_version="structured.v1",
        response_model=_Answer,
    )

    assert "stream" not in captured_payload
    assert captured_payload["response_format"] == {"type": "json_object"}
    assert captured_payload["temperature"] == 0


def test_complete_json_sse_reconstructs_content_and_discards_reasoning() -> None:
    captured: dict[str, object] = {}
    final_content = '{"answer":"ok","score":3}'
    events = [
        {
            "id": "stream-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "private reasoning marker"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "stream-1",
            "choices": [{"index": 0, "delta": {"content": '{"answer":'}}],
        },
        {
            "id": "stream-1",
            "choices": [{"index": 0, "delta": {"content": '"ok","score":'}}],
        },
        {
            "id": "stream-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "3}"},
                    "finish_reason": "stop",
                }
            ],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return _structured_sse_response(request, events, split_at=5)

    provider = OpenAICompatibleProvider(
        config=_config(
            capabilities={"structured_output_mode": "json_object_local_validation"},
            structured_output_transport="sse_stream",
        ),
        client=_client(handler),
    )
    result = provider.complete_json(
        messages=[{"role": "user", "content": "structured"}],
        prompt_version="structured.sse.v1",
        response_model=_Answer,
    )

    request = captured["request"]
    assert isinstance(request, httpx.Request)
    payload = json.loads(request.content)
    assert request.headers["accept"] == "text/event-stream"
    assert payload["stream"] is True
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["temperature"] == 0
    assert result.parsed_output == {"answer": "ok", "score": 3}
    assert result.response_id == "stream-1"
    assert result.raw_response["choices"][0]["message"]["content"] == final_content
    serialized = json.dumps(result.raw_response)
    assert "private reasoning marker" not in serialized


def test_complete_json_sse_preserves_native_schema_and_response_schema_validation() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return _structured_sse_response(
            request,
            [
                {
                    "id": "schema-stream",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": '{"answer":"ok"}'},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ],
        )

    provider = OpenAICompatibleProvider(
        config=_config(
            capabilities={"structured_output_mode": "native_json_schema"},
            structured_output_transport="sse_stream",
        ),
        client=_client(handler),
    )
    result = provider.complete_json(
        messages=[],
        prompt_version="schema.sse.v1",
        response_schema=schema,
    )

    assert result.parsed_output == {"answer": "ok"}
    assert captured_payload["stream"] is True
    assert captured_payload["response_format"]["type"] == "json_schema"
    assert captured_payload["response_format"]["json_schema"]["strict"] is True


def test_structured_output_transport_is_fingerprinted_and_injected_transport_fails_closed() -> None:
    buffered = _config(structured_output_transport="buffered")
    sse = _config(structured_output_transport="sse_stream")
    assert _config_fingerprint(buffered) == _config_fingerprint(
        _config(structured_output_transport="buffered")
    )
    assert _config_fingerprint(buffered) != _config_fingerprint(sse)

    calls = 0

    def transport(*_args) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _completion('{"answer":"ok","score":3}')

    provider = OpenAICompatibleProvider(config=sse, transport=transport)
    with pytest.raises(
        LLMProviderError,
        match="structured SSE transport is unavailable with buffered injected transport",
    ):
        provider.complete_json(messages=[], prompt_version="sse.v1", response_model=_Answer)
    assert calls == 0


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("missing_done", r"before \[DONE\]"),
        ("done_without_stop", "did not finish with stop"),
        ("length", "did not finish with stop"),
        ("content_filter", "did not finish with stop"),
        ("empty", "empty content"),
        ("invalid_json", "not valid JSON"),
        ("malformed_event", "emitted invalid JSON"),
    ],
)
def test_complete_json_sse_fails_closed_for_incomplete_or_invalid_stream(
    case: str,
    expected_message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if case == "malformed_event":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_ChunkedSSEStream([b"data: {not-json}\n\n"], request=request),
            )
        if case == "missing_done":
            events = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": '{"answer":"ok","score":3}'},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ]
            return _structured_sse_response(request, events, include_done=False)
        if case == "done_without_stop":
            events = [{"choices": [{"index": 0, "delta": {"content": "{}"}}]}]
        elif case in {"length", "content_filter"}:
            events = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "{}"},
                            "finish_reason": case,
                        }
                    ]
                }
            ]
        elif case == "empty":
            events = [
                {
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ]
                }
            ]
        else:
            events = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "not-json"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ]
        return _structured_sse_response(request, events)

    provider = OpenAICompatibleProvider(
        config=_config(structured_output_transport="sse_stream"),
        client=_client(handler),
    )
    with pytest.raises(LLMProviderError, match=expected_message):
        provider.complete_json(messages=[], prompt_version="sse.failure.v1")


@pytest.mark.parametrize("error_type", [httpx.RemoteProtocolError, httpx.ReadError])
def test_complete_json_sse_does_not_retry_after_stream_material(
    error_type: type[httpx.HTTPError],
) -> None:
    calls = 0
    reasoning_marker = "private reasoning must not escape"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedSSEStream(
                [
                    (
                        'data: '
                        + json.dumps(
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "reasoning_content": reasoning_marker,
                                            "content": "{",
                                        },
                                    }
                                ]
                            }
                        )
                        + "\n\n"
                    ).encode()
                ],
                error_factory=lambda request: error_type("stream interrupted", request=request),
                request=request,
            ),
        )

    provider = OpenAICompatibleProvider(
        config=_config(
            max_connect_retries=1,
            structured_output_transport="sse_stream",
        ),
        client=_client(handler),
    )
    with pytest.raises(LLMProviderError, match="stream interrupted") as exc_info:
        provider.complete_json(messages=[], prompt_version="sse.midstream.v1")
    assert calls == 1
    assert reasoning_marker not in str(exc_info.value)


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout])
def test_complete_json_sse_retries_pre_stream_connect_failures(
    error_type: type[httpx.HTTPError],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_type("before stream response", request=request)
        return _structured_sse_response(
            request,
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": '{"answer":"ok","score":3}'},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ],
        )

    provider = OpenAICompatibleProvider(
        config=_config(
            max_connect_retries=1,
            structured_output_transport="sse_stream",
        ),
        client=_client(handler),
    )
    result = provider.complete_json(
        messages=[],
        prompt_version="sse.retry.v1",
        response_model=_Answer,
    )
    assert result.parsed_output == {"answer": "ok", "score": 3}
    assert calls == 2


@pytest.mark.parametrize(
    ("events", "expected_message"),
    [
        (
            [
                {
                    "choices": [
                        {"index": 0, "delta": {"content": "{}"}},
                        {"index": 1, "delta": {"content": "{}"}},
                    ]
                }
            ],
            "multiple content choices",
        ),
        (
            [
                {"id": "one", "choices": []},
                {"id": "two", "choices": []},
            ],
            "conflicting response IDs",
        ),
    ],
)
def test_complete_json_sse_rejects_ambiguous_stream_metadata(
    events: list[dict],
    expected_message: str,
) -> None:
    provider = OpenAICompatibleProvider(
        config=_config(structured_output_transport="sse_stream"),
        client=_client(lambda request: _structured_sse_response(request, events)),
    )
    with pytest.raises(LLMProviderError, match=expected_message):
        provider.complete_json(messages=[], prompt_version="sse.metadata.v1")


def test_complete_json_sse_reuses_existing_pydantic_validation_without_echoing_content() -> None:
    sensitive = "sensitive-output"

    def handler(request: httpx.Request) -> httpx.Response:
        return _structured_sse_response(
            request,
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": json.dumps(
                                    {"answer": sensitive, "score": "not-an-integer"}
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ],
        )

    provider = OpenAICompatibleProvider(
        config=_config(structured_output_transport="sse_stream"),
        client=_client(handler),
    )
    with pytest.raises(LLMResponseValidationError) as exc_info:
        provider.complete_json(
            messages=[],
            prompt_version="sse.validation.v1",
            response_model=_Answer,
        )
    assert sensitive not in str(exc_info.value)


def test_complete_json_sse_enforces_total_deadline_during_stream() -> None:
    clock_values = iter([0.0, 0.0, 6.0])
    provider = OpenAICompatibleProvider(
        config=_config(
            total_timeout_sec=5,
            structured_output_transport="sse_stream",
        ),
        client=_client(
            lambda request: _structured_sse_response(
                request,
                [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "{}"},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                ],
            )
        ),
        clock=lambda: next(clock_values),
    )
    with pytest.raises(LLMProviderError, match="total deadline"):
        provider.complete_json(messages=[], prompt_version="sse.deadline.v1")


def test_provider_hostname_does_not_override_explicit_native_capability() -> None:
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"answer":"ok","score":3}'))

    provider = OpenAICompatibleProvider(
        config=_config(
            endpoint="https://api.deepseek.com",
            capabilities={"structured_output_mode": "native_json_schema"},
        ),
        client=_client(handler),
    )
    provider.complete_json(
        messages=[{"role": "user", "content": "structured"}],
        prompt_version="structured.v1",
        response_model=_Answer,
    )

    assert captured_payload["response_format"]["type"] == "json_schema"
    assert "temperature" not in captured_payload


def test_complete_json_fails_when_native_schema_format_is_unavailable() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "This response_format type is unavailable now"
                    }
                },
            )
        return httpx.Response(200, json=_completion('{"answer":"ok","score":3}'))

    provider = OpenAICompatibleProvider(
        config=_config(capabilities={"structured_output_mode": "native_json_schema"}),
        client=_client(handler),
    )
    with pytest.raises(LLMProviderError, match="requested structured response format"):
        provider.complete_json(
            messages=[{"role": "user", "content": "structured"}],
            prompt_version="structured.v1",
            response_model=_Answer,
        )

    assert len(calls) == 1
    assert calls[0]["response_format"]["type"] == "json_schema"


def test_complete_json_rejects_invalid_pydantic_output_without_echoing_input() -> None:
    client = _client(
        lambda _request: httpx.Response(
            200,
            json=_completion('{"answer":"sensitive-output","score":"not-an-integer"}'),
        )
    )
    provider = OpenAICompatibleProvider(config=_config(), client=client)
    with pytest.raises(LLMResponseValidationError) as exc_info:
        provider.complete_json(
            messages=[],
            prompt_version="structured.v1",
            response_model=_Answer,
        )
    assert "score" in str(exc_info.value)
    assert "sensitive-output" not in str(exc_info.value)


def test_complete_json_validates_json_schema_and_rejects_bad_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    provider = OpenAICompatibleProvider(
        config=_config(),
        client=_client(lambda _request: httpx.Response(200, json=_completion('{"answer":"ok"}'))),
    )
    result = provider.complete_json(
        messages=[],
        prompt_version="schema.v1",
        response_schema=schema,
    )
    assert result.parsed_output == {"answer": "ok"}

    invalid_output_provider = OpenAICompatibleProvider(
        config=_config(),
        client=_client(
            lambda _request: httpx.Response(
                200,
                json=_completion('{"answer":{"secret":"sensitive-output"}}'),
            )
        ),
    )
    with pytest.raises(LLMResponseValidationError, match="JSON Schema validation") as exc_info:
        invalid_output_provider.complete_json(
            messages=[],
            prompt_version="schema.v1",
            response_schema=schema,
        )
    assert "sensitive-output" not in str(exc_info.value)
    with pytest.raises(LLMResponseValidationError, match="invalid response JSON Schema"):
        provider.complete_json(
            messages=[],
            prompt_version="schema.v1",
            response_schema={"type": "not-a-json-schema-type"},
        )
    with pytest.raises(LLMProviderError, match="mutually exclusive"):
        provider.complete_json(
            messages=[],
            prompt_version="schema.v1",
            response_model=_Answer,
            response_schema=schema,
        )


def test_json_protocol_accepts_fenced_or_reasoning_fallback_and_redacts_reasoning() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json=_completion("```json\n{\"answer\":\"fenced\"}\n```"))
        return httpx.Response(
            200,
            json=_completion(
                "not-json",
                extra_message={"reasoning_content": 'analysis then {"answer":"fallback"}'},
            ),
        )

    provider = OpenAICompatibleProvider(config=_config(), client=_client(handler))
    fenced = provider.complete_json(messages=[], prompt_version="json.v1")
    fallback = provider.complete_json(messages=[], prompt_version="json.v1")
    assert fenced.parsed_output == {"answer": "fenced"}
    assert fallback.parsed_output == {"answer": "fallback"}
    assert fallback.raw_response["choices"][0]["message"]["reasoning_content"] == "[REDACTED]"


def test_stream_text_parses_openai_sse_deltas() -> None:
    stream_body = "\n\n".join(
        [
            'data: {"choices":[{"delta":{"content":"hello "}}]}',
            'data: {"choices":[{"delta":{"content":[{"type":"text","text":"world"}]}}]}',
            "data: [DONE]",
        ]
    )
    provider = OpenAICompatibleProvider(
        config=_config(),
        client=_client(lambda _request: httpx.Response(200, text=stream_body)),
    )
    chunks = list(provider.stream_text(messages=[], prompt_version="stream.v1"))
    assert chunks == ["hello ", "world"]


def test_stream_uses_sse_accept_header_and_stops_immediately_at_done() -> None:
    requests: list[httpx.Request] = []
    stream_closed = False

    class DoneThenFailStream(httpx.SyncByteStream):
        def __init__(self, request: httpx.Request) -> None:
            self.request = request

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"final"}}]}\n\n'
            yield b"data: [DONE]\n\n"
            raise httpx.ReadError("must not read after DONE", request=self.request)

        def close(self) -> None:
            nonlocal stream_closed
            stream_closed = True

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, stream=DoneThenFailStream(request))

    provider = OpenAICompatibleProvider(config=_config(), client=_client(handler))
    assert list(provider.stream_text(messages=[], prompt_version="stream.v1")) == ["final"]
    assert requests[0].headers["accept"] == "text/event-stream"
    assert stream_closed is True


def test_stream_does_not_retry_after_first_text_delta() -> None:
    calls = 0

    class BrokenStream(httpx.SyncByteStream):
        def __init__(self, request: httpx.Request) -> None:
            self.request = request

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            raise httpx.ReadError("stream interrupted", request=self.request)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=BrokenStream(request))

    provider = OpenAICompatibleProvider(config=_config(), client=_client(handler))
    stream = provider.stream_text(messages=[], prompt_version="stream.v1")
    assert next(stream) == "first"
    with pytest.raises(LLMProviderError, match="stream interrupted"):
        next(stream)
    assert calls == 1


def test_only_pre_delivery_connect_failures_are_retried() -> None:
    connect_calls = 0

    def connect_then_succeed(request: httpx.Request) -> httpx.Response:
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == 1:
            raise httpx.ConnectError("DNS unavailable", request=request)
        return httpx.Response(200, json=_completion("connected"))

    provider = OpenAICompatibleProvider(config=_config(), client=_client(connect_then_succeed))
    assert provider.complete_text(messages=[], prompt_version="retry.v1") == "connected"
    assert connect_calls == 2

    read_calls = 0

    def read_timeout(request: httpx.Request) -> httpx.Response:
        nonlocal read_calls
        read_calls += 1
        raise httpx.ReadTimeout("response read timed out", request=request)

    provider = OpenAICompatibleProvider(config=_config(), client=_client(read_timeout))
    with pytest.raises(LLMProviderError, match="response read timed out"):
        provider.complete_text(messages=[], prompt_version="retry.v1")
    assert read_calls == 1

    status_calls = 0

    def unavailable(_request: httpx.Request) -> httpx.Response:
        nonlocal status_calls
        status_calls += 1
        return httpx.Response(503, text="secret provider body")

    provider = OpenAICompatibleProvider(config=_config(), client=_client(unavailable))
    with pytest.raises(LLMProviderError, match="HTTP 503") as exc_info:
        provider.complete_text(messages=[], prompt_version="retry.v1")
    assert status_calls == 1
    assert "secret provider body" not in str(exc_info.value)

    redirect_calls = 0

    def redirect(_request: httpx.Request) -> httpx.Response:
        nonlocal redirect_calls
        redirect_calls += 1
        return httpx.Response(302, headers={"Location": "https://evil.example/collect"})

    provider = OpenAICompatibleProvider(config=_config(), client=_client(redirect))
    with pytest.raises(LLMProviderError, match="HTTP 302"):
        provider.complete_text(messages=[], prompt_version="redirect.v1")
    assert redirect_calls == 1


def test_network_timeouts_are_split_and_total_deadline_is_enforced() -> None:
    observed_timeout: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed_timeout.update(request.extensions["timeout"])
        return httpx.Response(200, json=_completion("ok"))

    provider = OpenAICompatibleProvider(
        config=_config(
            timeout_sec=120,
            connect_timeout_sec=7,
            write_timeout_sec=21,
            pool_timeout_sec=4,
            total_timeout_sec=240,
        ),
        client=_client(handler),
        clock=lambda: 0,
    )
    assert provider.complete_text(messages=[], prompt_version="timeout.v1") == "ok"
    assert observed_timeout == {"connect": 7.0, "read": 120.0, "write": 21.0, "pool": 4.0}

    clock_values = iter([0.0, 0.0, 6.0])
    deadline_provider = OpenAICompatibleProvider(
        config=_config(total_timeout_sec=5),
        client=_client(lambda _request: httpx.Response(200, json=_completion("late"))),
        clock=lambda: next(clock_values),
    )
    with pytest.raises(LLMProviderError, match="total deadline"):
        deadline_provider.complete_text(messages=[], prompt_version="timeout.v1")


def test_headers_raw_response_and_errors_are_redacted() -> None:
    def echo_sensitive(_request: httpx.Request) -> httpx.Response:
        raw = _completion('{"answer":"ok"}')
        raw["debug"] = {
            "Authorization": "Bearer echoed-secret",
            "api_key": "echoed-secret",
        }
        raw["id"] = "secret-token"
        raw["ordinary_debug"] = "Authorization: Bearer secret-token"
        raw["nested"] = ["prefix secret-token suffix"]
        raw["choices"][0]["message"]["reasoning_content"] = "private chain of thought"
        return httpx.Response(200, json=raw)

    provider = OpenAICompatibleProvider(config=_config(), client=_client(echo_sensitive))
    result = provider.complete_json(messages=[], prompt_version="redaction.v1")
    serialized = json.dumps(result.raw_response)
    assert "echoed-secret" not in serialized
    assert "secret-token" not in serialized
    assert "private chain of thought" not in serialized
    assert result.response_id == "[REDACTED]"
    assert serialized.count("[REDACTED]") >= 6

    def leak_in_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Bearer secret-token https://host.test/?api_key=secret-token",
            request=request,
        )

    provider = OpenAICompatibleProvider(
        config=_config(max_connect_retries=0),
        client=_client(leak_in_error),
    )
    with pytest.raises(LLMProviderError) as exc_info:
        provider.complete_text(messages=[], prompt_version="redaction.v1")
    assert "secret-token" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_provider_manager_reuses_leases_and_defers_retired_client_close() -> None:
    manager = LLMProviderManager()
    with manager.lease(_config()) as first:
        with manager.lease(_config()) as same:
            assert same is first
        with manager.lease(_config(model="other-model")) as replacement:
            assert replacement is not first
            assert isinstance(first, OpenAICompatibleProvider)
            assert first._closed is False
            assert isinstance(replacement, OpenAICompatibleProvider)
        assert first._closed is False
    assert first._closed is True
    assert replacement._closed is False
    manager.invalidate()
    assert replacement._closed is True


def test_settings_change_retires_but_does_not_interrupt_leased_provider(tmp_path) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    manager = app.extensions["llm_provider_manager"]
    with manager.lease(_config()) as provider:
        assert isinstance(provider, OpenAICompatibleProvider)
        response = app.test_client().patch(
            "/api/settings/llm",
            json={
                "endpoint": "https://new.example.test/v1",
                "model": "new-model",
                "api_key_source": "file",
                "api_key": "new-secret",
            },
        )
        assert response.status_code == 200
        assert provider._closed is False
    assert provider._closed is True


def test_active_stream_survives_concurrent_manager_invalidation() -> None:
    stream_started = Event()
    allow_done = Event()
    provider_holder: list[OpenAICompatibleProvider] = []

    class BlockingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"active"}}]}\n\n'
            stream_started.set()
            assert allow_done.wait(timeout=5)
            yield b"data: [DONE]\n\n"

    def factory(config: LLMProviderConfig):
        provider = OpenAICompatibleProvider(
            config=config,
            client=_client(lambda _request: httpx.Response(200, stream=BlockingStream())),
        )
        provider._owns_client = True
        provider_holder.append(provider)
        return provider

    manager = LLMProviderManager(provider_factory=factory)
    chunks: list[str] = []
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            with manager.lease(_config()) as provider:
                chunks.extend(provider.stream_text(messages=[], prompt_version="stream.v1"))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = Thread(target=consume)
    thread.start()
    assert stream_started.wait(timeout=5)
    manager.invalidate()
    assert provider_holder[0]._closed is False
    allow_done.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert chunks == ["active"]
    assert provider_holder[0]._closed is True


def test_agent_plan_route_uses_server_settings_without_browser_secret(tmp_path, monkeypatch) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    client = app.test_client()
    assert client.patch(
        "/api/settings/llm",
        json={
            "endpoint": "https://llm.example.test/v1",
            "model": "saved-model",
            "api_key_source": "file",
            "api_key": "server-only-secret",
            "connect_timeout_sec": 3,
            "write_timeout_sec": 9,
            "pool_timeout_sec": 2,
            "total_timeout_sec": 180,
            "max_connect_retries": 2,
            "retry_backoff_sec": 0.5,
            "external_llm_data_sharing_enabled": True,
        },
    ).status_code == 200

    seen_configs: list[LLMProviderConfig] = []
    manager = app.extensions["llm_provider_manager"]

    @contextmanager
    def fake_lease(config: LLMProviderConfig):
        seen_configs.append(config)
        yield StubLLMProvider(response={"requested_tasks": ["render_report"]})

    monkeypatch.setattr(manager, "lease", fake_lease)
    response = client.post(
        "/api/agent/plan-proposal",
        json={
            "run_id": "run-saved-settings",
            "goal": "Render a report from existing predictions.",
            "available_artifacts": ["candidate_predictions"],
        },
    )
    assert response.status_code == 200
    assert response.json["proposal"]["planner_backend"] == "stub"
    assert seen_configs[0].model == "saved-model"
    assert seen_configs[0].api_key == "server-only-secret"
    assert seen_configs[0].connect_timeout_sec == 3
    assert seen_configs[0].total_timeout_sec == 180
    assert seen_configs[0].max_connect_retries == 2
    assert "server-only-secret" not in response.get_data(as_text=True)


def test_invalid_project_id_does_not_create_inline_provider(tmp_path, monkeypatch) -> None:
    app = create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "config",
    )
    created_configs: list[LLMProviderConfig] = []

    def record_creation(config: LLMProviderConfig):
        created_configs.append(config)
        return StubLLMProvider(response={"requested_tasks": ["render_report"]})

    monkeypatch.setattr(agent_routes, "create_llm_provider", record_creation)
    response = app.test_client().post(
        "/api/agent/plan-proposal",
        json={
            "run_id": "run-invalid-project",
            "goal": "Render a report.",
            "project_id": "../escape",
            "llm_provider": {
                "provider": "openai_compatible",
                "endpoint": "https://llm.example.test/v1",
                "api_key": "inline-secret",
                "model": "inline-model",
            },
        },
    )

    assert response.status_code == 400
    assert "project_id escapes memory directory" in response.json["error"]
    assert created_configs == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://remote.example.test/v1",
        "https://user:password@llm.example.test/v1",
        "https://llm.example.test/v1?api_key=query-secret",
    ],
)
def test_provider_rejects_unsafe_endpoint_even_outside_settings_api(endpoint: str) -> None:
    with pytest.raises(LLMProviderError):
        OpenAICompatibleProvider(config=_config(endpoint=endpoint))
