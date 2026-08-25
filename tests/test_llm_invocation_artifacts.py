from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_llm_context_mapping import (
    build_oled_llm_paper_mapping_request,
    run_oled_llm_context_mapping,
)
from ai4s_agent.domains.oled_mineru_candidates import OledMineruCandidateType
from ai4s_agent.domains.oled_mineru_semantic_mapping import OledSemanticMappingPacket
from ai4s_agent.llm_invocation_artifacts import (
    ExactLLMInvocationArtifactError,
    ExactLLMInvocationArtifactStore,
    FrozenLLMInvocation,
    canonical_json_bytes,
    replay_frozen_invocation,
)
from ai4s_agent.llm_provider import OpenAICompatibleProvider, StubLLMProvider
from ai4s_agent.schemas import LLMProviderConfig


class _Answer(BaseModel):
    ok: bool


def _frozen(*, payload: dict | None = None, model: str = "model-a") -> FrozenLLMInvocation:
    return FrozenLLMInvocation.from_payload(
        provider="test-provider",
        model=model,
        prompt_version="prompt.v1",
        request_digest="request-digest",
        structured_output_mode="json_object_local_validation",
        structured_output_transport="buffered",
        payload=payload
        or {
            "model": model,
            "messages": [{"role": "user", "content": '{"ok":true}'}],
            "response_format": {"type": "json_object"},
        },
    )


def test_same_semantic_invocation_has_same_canonical_bytes_and_digest() -> None:
    first = _frozen(
        payload={
            "messages": [{"role": "user", "content": "same"}],
            "model": "model-a",
            "temperature": 0,
        }
    )
    second = _frozen(
        payload={
            "temperature": 0,
            "model": "model-a",
            "messages": [{"role": "user", "content": "same"}],
        }
    )

    assert first.canonical_bytes == second.canonical_bytes
    assert first.invocation_digest == second.invocation_digest


def test_nonsemantic_runtime_metadata_is_not_part_of_digest() -> None:
    first = _frozen()
    second = _frozen()

    assert first.invocation_digest == second.invocation_digest
    assert first.canonical_bytes == canonical_json_bytes(
        json.loads(first.canonical_bytes.decode("utf-8"))
    )


@pytest.mark.parametrize(
    "change",
    [
        {"model": "model-b"},
        {"prompt_version": "prompt.v2"},
        {"request_digest": "other-request"},
        {"structured_output_transport": "sse_stream"},
    ],
)
def test_semantic_invocation_change_changes_digest(change: dict[str, str]) -> None:
    base = _frozen()
    payload = base.provider_payload()
    changed = FrozenLLMInvocation.from_payload(
        provider=change.get("provider", base.provider),
        model=change.get("model", base.model),
        prompt_version=change.get("prompt_version", base.prompt_version),
        request_digest=change.get("request_digest", base.request_digest),
        structured_output_mode=change.get(
            "structured_output_mode", base.structured_output_mode
        ),
        structured_output_transport=change.get(
            "structured_output_transport", base.structured_output_transport
        ),
        payload=payload,
    )

    assert changed.invocation_digest != base.invocation_digest


def test_persist_reread_recomputes_exact_digest_and_is_idempotent(tmp_path: Path) -> None:
    store = ExactLLMInvocationArtifactStore(tmp_path / "private" / "invocations")
    frozen = _frozen()

    first = store.persist_and_verify(frozen)
    second = store.persist_and_verify(frozen)
    loaded = store.load(frozen.invocation_digest)

    assert first.canonical_bytes == frozen.canonical_bytes
    assert second.canonical_bytes == frozen.canonical_bytes
    assert loaded.canonical_bytes == frozen.canonical_bytes
    assert loaded.invocation_digest == frozen.invocation_digest
    assert (tmp_path / "private" / "invocations" / frozen.invocation_digest / "manifest.json").is_file()


def test_tampered_payload_or_manifest_fails_closed(tmp_path: Path) -> None:
    store = ExactLLMInvocationArtifactStore(tmp_path / "invocations")
    frozen = _frozen()
    store.persist_and_verify(frozen)
    artifact_dir = tmp_path / "invocations" / frozen.invocation_digest

    (artifact_dir / "payload.json").write_bytes(
        (artifact_dir / "payload.json").read_bytes() + b" "
    )
    with pytest.raises(ExactLLMInvocationArtifactError):
        store.load(frozen.invocation_digest)


def test_manifest_tampering_is_not_silently_overwritten(tmp_path: Path) -> None:
    store = ExactLLMInvocationArtifactStore(tmp_path / "invocations")
    frozen = _frozen()
    store.persist_and_verify(frozen)
    manifest_path = tmp_path / "invocations" / frozen.invocation_digest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provider"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExactLLMInvocationArtifactError):
        store.persist_and_verify(frozen)


def test_openai_provider_sends_the_frozen_payload_after_source_mutation(tmp_path: Path) -> None:
    sent: list[dict] = []

    def transport(_url: str, payload: dict, _headers: dict[str, str], _timeout: int) -> dict:
        sent.append(payload)
        return {
            "id": "response-1",
            "choices": [
                {
                    "message": {"content": '{"ok":true}'},
                    "finish_reason": "stop",
                }
            ],
        }

    provider = OpenAICompatibleProvider(
        config=LLMProviderConfig(
            provider="openai_compatible",
            endpoint="http://127.0.0.1",
            api_key="secret-api-key",
            model="model-a",
        ),
        transport=transport,
    )
    messages = [{"role": "user", "content": "frozen content"}]
    frozen = provider.prepare_json_invocation(
        messages=messages,
        prompt_version="prompt.v1",
        response_model=_Answer,
        request_digest="request-digest",
    )
    store = ExactLLMInvocationArtifactStore(tmp_path / "private")
    loaded = store.persist_and_verify(frozen)
    messages[0]["content"] = "mutated after freeze"

    result = provider.complete_json(
        messages=messages,
        prompt_version="prompt.v1",
        response_model=_Answer,
        frozen_invocation=loaded,
    )

    assert result.parsed_output == {"ok": True}
    assert sent == [loaded.provider_payload()]
    assert "Authorization" not in sent[0]
    payload_text = (
        tmp_path / "private" / frozen.invocation_digest / "payload.json"
    ).read_text(encoding="utf-8")
    manifest_text = (
        tmp_path / "private" / frozen.invocation_digest / "manifest.json"
    ).read_text(encoding="utf-8")
    assert "secret-api-key" not in payload_text
    assert "secret-api-key" not in manifest_text
    provider.close()


def test_sse_provider_sends_the_same_frozen_payload(tmp_path: Path) -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content.decode("utf-8")))
        stream = (
            'data: {"id":"response-1","choices":[{"index":0,"delta":{"content":"{\\"ok\\":true}"},"finish_reason":null}]}'
            "\n\n"
            'data: {"id":"response-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            "\n\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream.encode("utf-8"),
            request=request,
        )

    config = LLMProviderConfig(
        provider="openai_compatible",
        endpoint="https://api.example.test",
        model="model-a",
        structured_output_transport="sse_stream",
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(config=config, client=client)
    store = ExactLLMInvocationArtifactStore(tmp_path / "private")
    frozen = provider.prepare_json_invocation(
        messages=[{"role": "user", "content": "sse exact"}],
        prompt_version="prompt.v1",
        response_model=_Answer,
        request_digest="request-digest",
    )
    loaded = store.persist_and_verify(frozen)

    result = provider.complete_json(
        messages=[{"role": "user", "content": "mutated"}],
        prompt_version="prompt.v1",
        response_model=_Answer,
        frozen_invocation=loaded,
    )

    assert result.parsed_output == {"ok": True}
    assert sent == [loaded.provider_payload()]
    assert sent[0]["stream"] is True
    provider.close()
    client.close()


def test_exact_replay_uses_artifact_without_rebuilding_upstream(tmp_path: Path) -> None:
    sent: list[dict] = []

    def transport(_url: str, payload: dict, _headers: dict[str, str], _timeout: int) -> dict:
        sent.append(payload)
        return {
            "choices": [
                {
                    "message": {"content": '{"ok":true}'},
                    "finish_reason": "stop",
                }
            ]
        }

    provider = OpenAICompatibleProvider(
        config=LLMProviderConfig(
            provider="openai_compatible",
            endpoint="http://127.0.0.1",
            model="model-a",
        ),
        transport=transport,
    )
    store = ExactLLMInvocationArtifactStore(tmp_path / "private")
    frozen = provider.prepare_json_invocation(
        messages=[{"role": "user", "content": "exact"}],
        prompt_version="prompt.v1",
        response_model=_Answer,
        request_digest="request-digest",
    )
    store.persist_and_verify(frozen)

    replay_frozen_invocation(
        provider,
        store=store,
        reference=frozen.invocation_digest,
        response_model=_Answer,
    )

    assert sent == [frozen.provider_payload()]
    provider.close()


def test_persistence_failure_blocks_provider_call() -> None:
    class FailingStore:
        def persist_and_verify(self, _frozen: FrozenLLMInvocation) -> FrozenLLMInvocation:
            raise ExactLLMInvocationArtifactError("simulated persistence failure")

    class CountingProvider(StubLLMProvider):
        calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            return super().complete_json(**kwargs)

    # The request is intentionally tiny; this test only exercises the gate.
    request = build_oled_llm_paper_mapping_request(
        [
            OledSemanticMappingPacket(
                packet_id="packet-1",
                source_candidate_hash="source-1",
                source_evidence_anchor="paper:p1",
                source_candidate_type=OledMineruCandidateType.TEXT,
                paper_id="paper",
                raw_text="source",
                allowed_layers=[layer.value for layer in OledCausalLayer],
            )
        ],
        parsed_document={"paper_id": "paper", "elements": [{"text": "source"}]},
    )
    provider = CountingProvider(response={})
    result = run_oled_llm_context_mapping(
        request,
        provider=provider,
        invocation_artifact_store=FailingStore(),
    )

    assert result.status == "provider_error"
    assert result.findings[0].code == "llm_invocation_artifact_error"
    assert result.metadata["llm_called"] is False
    assert provider.calls == 0


def test_binding_failure_report_is_bound_to_verified_invocation(tmp_path: Path) -> None:
    request = build_oled_llm_paper_mapping_request(
        [
            OledSemanticMappingPacket(
                packet_id="packet-1",
                source_candidate_hash="source-1",
                source_evidence_anchor="paper:p1",
                source_candidate_type=OledMineruCandidateType.TEXT,
                paper_id="paper",
                raw_text="source",
                allowed_layers=[layer.value for layer in OledCausalLayer],
            )
        ],
        parsed_document={"paper_id": "paper", "elements": [{"text": "source"}]},
    )
    provider = StubLLMProvider(
        response={"paper_id": "wrong-paper", "packet_results": [], "response_notes": []}
    )
    store = ExactLLMInvocationArtifactStore(tmp_path / "private")

    result = run_oled_llm_context_mapping(
        request,
        provider=provider,
        invocation_artifact_store=store,
    )

    failure = result.metadata["response_binding_failure"]
    summary = failure["invocation_artifact"]
    assert failure["binding_error_code"] == "PAPER_ID_MISMATCH"
    assert summary["status"] == "verified"
    loaded = store.load(summary["invocation_digest"])
    assert loaded.invocation_digest == summary["invocation_digest"]


def test_synthetic_artifact_manifest_does_not_persist_prompt_content(tmp_path: Path) -> None:
    frozen = _frozen(
        payload={
            "model": "model-a",
            "messages": [
                {
                    "role": "user",
                    "content": "private unpublished document content",
                }
            ],
            "response_format": {"type": "json_object"},
        }
    )
    store = ExactLLMInvocationArtifactStore(tmp_path / "private")
    store.persist_and_verify(frozen)
    manifest = (
        tmp_path / "private" / frozen.invocation_digest / "manifest.json"
    ).read_text(encoding="utf-8")

    assert "private unpublished document content" not in manifest
    assert "Authorization" not in manifest
    assert "private reasoning content" not in manifest


def test_artifact_rejects_credentials_and_headers_in_provider_payload() -> None:
    with pytest.raises(ExactLLMInvocationArtifactError):
        FrozenLLMInvocation.from_payload(
            provider="test-provider",
            model="model-a",
            prompt_version="prompt.v1",
            request_digest="request-digest",
            structured_output_mode="json_object_local_validation",
            structured_output_transport="buffered",
            payload={"headers": {"Authorization": "Bearer secret"}},
        )
