from __future__ import annotations

import json
from pathlib import Path

import pytest

import ai4s_agent.adapters.br2_contextual_mapping as br2_adapter
from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMContextMappingResult,
    OledLLMPacketMappingProposal,
    OledLLMPaperMappingRequest,
    OledSemanticMappingPacket,
    build_oled_llm_paper_mapping_request,
    canonical_oled_llm_paper_mapping_request_bytes,
)
from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruCandidate,
    OledMineruCandidateType,
    OledMineruSourceFormat,
    OledMineruTableParseStatus,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSchemaCandidate,
    OledSchemaCandidateStatus,
    OledSchemaCandidateType,
    OledSchemaEvidenceRef,
    OledSemanticMappingFinding,
)
from ai4s_agent.oled_llm_context_request import (
    build_frozen_oled_llm_provider_invocation_manifest,
    compute_oled_source_candidates_digest,
    freeze_oled_llm_paper_mapping_request,
    load_frozen_oled_llm_paper_mapping_request,
    load_frozen_oled_llm_provider_invocation_manifest,
    persist_frozen_oled_llm_paper_mapping_request,
    persist_frozen_oled_llm_provider_invocation_manifest,
    recover_oled_llm_paper_mapping_request,
    verify_oled_br2_replay_binding,
)
from ai4s_agent.schemas import LLMInvocationRecord


def _source_and_packet() -> tuple[OledMineruCandidate, OledSemanticMappingPacket]:
    source = OledMineruCandidate(
        paper_id="paper-replay",
        source_format=OledMineruSourceFormat.MINERU_LIKE,
        candidate_type=OledMineruCandidateType.TABLE,
        page_index=3,
        block_index=0,
        block_id="table-0",
        raw_text="Emitter | PLQY (%)\nMolecule-A | 82",
        caption="Photophysical properties",
        table_headers=["Emitter", "PLQY (%)"],
        table_rows=[{"Emitter": "Molecule-A", "PLQY (%)": "82"}],
        table_parse_status=OledMineruTableParseStatus.PARSED,
        evidence_anchor="paper-replay:p3:table-0",
        candidate_hash="replay-source-hash",
    )
    packet = OledSemanticMappingPacket(
        packet_id="packet-replay-0",
        source_candidate_hash=source.candidate_hash,
        source_evidence_anchor=source.evidence_anchor,
        source_candidate_type=source.candidate_type,
        paper_id=source.paper_id,
        caption=source.caption,
        raw_text=source.raw_text,
        table_headers=source.table_headers,
        table_rows=source.table_rows,
        allowed_property_ids=["plqy"],
        allowed_layers=[layer.value for layer in OledCausalLayer],
    )
    return source, packet


def _request_and_source() -> tuple[OledLLMPaperMappingRequest, OledMineruCandidate]:
    source, packet = _source_and_packet()
    request = build_oled_llm_paper_mapping_request(
        [packet],
        parsed_document={
            "paper_id": source.paper_id,
            "elements": [
                {
                    "element_id": "paper-replay:p1:text-0",
                    "page": 1,
                    "type": "paragraph",
                    "text": "Molecule-A is the emitter.",
                    "source_hash": "replay-context-hash",
                }
            ],
            "tables": [
                {
                    "table_id": "paper-replay:p3:table-0",
                    "page": 3,
                    "caption": source.caption,
                    "headers": source.table_headers,
                    "rows": source.table_rows,
                }
            ],
        },
    )
    candidate = OledSchemaCandidate(
        candidate_id="deterministic-replay-candidate",
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        status=OledSchemaCandidateStatus.PROPOSED,
        source_paper_id=source.paper_id,
        source_candidate_hash=source.candidate_hash,
        source_evidence_anchor=source.evidence_anchor,
        target_layer=OledCausalLayer.INTERACTION,
        property_id="plqy",
        property_label="Photoluminescence quantum yield",
        value=82,
        unit="%",
        reported_value_text="82",
        reported_decimal_places=0,
        material_name="Molecule-A",
        evidence_refs=[
            OledSchemaEvidenceRef(
                source_candidate_hash=source.candidate_hash,
                source_evidence_anchor=source.evidence_anchor,
                source_candidate_type="table",
                row_index=0,
                column_name="PLQY (%)",
                cell_value="82",
            )
        ],
        confidence_score=0.8,
    )
    return request.model_copy(update={"deterministic_schema_candidates": [candidate]}), source


def _invocation() -> LLMInvocationRecord:
    return LLMInvocationRecord(
        provider="stub",
        model="replay-test",
        prompt_version="oled.contextual_semantic_mapping.test",
        response_id="response-test",
        raw_response={"messages": [], "response": {"paper_id": "paper-replay"}},
        parsed_output={"paper_id": "paper-replay"},
    )


def _mapping_result(request: OledLLMPaperMappingRequest) -> OledLLMContextMappingResult:
    packet = request.packets[0]
    return OledLLMContextMappingResult(
        paper_id=request.paper_id,
        status="ready_for_human_review",
        request_digest=request.request_digest,
        packet_results=[
            OledLLMPacketMappingProposal(
                packet_id=packet.packet_id,
                action="keep_deterministic",
                scope_classification="property_bearing",
                rationale_summary="The deterministic property candidate is retained.",
            )
        ],
        llm_invocation=_invocation(),
        metadata={"llm_called": True},
    )


def test_request_canonical_bytes_and_digest_are_deterministic() -> None:
    request, _source = _request_and_source()
    reloaded = OledLLMPaperMappingRequest.model_validate_json(request.model_dump_json())

    assert canonical_oled_llm_paper_mapping_request_bytes(request) == (
        canonical_oled_llm_paper_mapping_request_bytes(reloaded)
    )
    assert request.request_digest == reloaded.request_digest

    second_packet = request.packets[0].model_copy(
        update={
            "packet_id": "packet-replay-1",
            "source_candidate_hash": "replay-source-hash-1",
            "source_evidence_anchor": "paper-replay:p3:table-1",
        }
    )
    ordered = request.model_copy(update={"packets": [request.packets[0], second_packet]})
    reordered = request.model_copy(update={"packets": [second_packet, request.packets[0]]})
    assert reordered.request_digest != ordered.request_digest


def test_request_runtime_metadata_is_not_domain_identity() -> None:
    request, _source = _request_and_source()
    changed_metadata = {
        **request.metadata,
        "path": "/private/another-run/request.json",
        "timestamp": "2099-01-01T00:00:00Z",
        "trace_id": "different-trace",
        "pid": 99999,
    }
    changed = request.model_copy(update={"metadata": changed_metadata})

    assert changed.request_digest == request.request_digest
    assert canonical_oled_llm_paper_mapping_request_bytes(changed) == (
        canonical_oled_llm_paper_mapping_request_bytes(request)
    )


def test_request_canonicalizer_materializes_absent_optional_fields_as_null() -> None:
    request, _source = _request_and_source()
    payload = request.model_dump(mode="json")
    packet_payload = payload["packets"][0]
    packet_payload.pop("nearby_text_before")
    packet_payload.pop("nearby_text_after")
    reloaded_from_absent = OledLLMPaperMappingRequest.model_validate(payload)
    reloaded_from_null = request.model_copy(
        update={
            "packets": [
                request.packets[0].model_copy(
                    update={"nearby_text_before": None, "nearby_text_after": None}
                )
            ]
        }
    )

    assert reloaded_from_absent.request_digest == reloaded_from_null.request_digest
    assert b'"nearby_text_before":null' in canonical_oled_llm_paper_mapping_request_bytes(
        reloaded_from_null
    )


@pytest.mark.parametrize(
    "field",
    [
        "packets",
        "document_context",
        "ontology",
        "deterministic_schema_candidates",
        "deterministic_findings",
        "instructions",
    ],
)
def test_request_digest_covers_candidate_assembly_request_fields(field: str) -> None:
    request, _source = _request_and_source()
    if field == "packets":
        changed = request.model_copy(
            update={
                "packets": [
                    request.packets[0].model_copy(
                        update={"table_rows": [{"Emitter": "Molecule-B", "PLQY (%)": "82"}]}
                    )
                ]
            }
        )
    elif field == "document_context":
        changed = request.model_copy(
            update={
                "document_context": [
                    request.document_context[0].model_copy(update={"text": "changed source text"})
                ]
            }
        )
    elif field == "ontology":
        changed = request.model_copy(
            update={
                "ontology": [
                    request.ontology[0].model_copy(
                        update={"name": request.ontology[0].name + " changed"}
                    ),
                    *request.ontology[1:],
                ]
            }
        )
    elif field == "deterministic_schema_candidates":
        changed = request.model_copy(update={"deterministic_schema_candidates": []})
    elif field == "deterministic_findings":
        changed = request.model_copy(
            update={
                "deterministic_findings": [
                    OledSemanticMappingFinding(
                        code="coverage-test",
                        message="identity coverage test",
                    )
                ]
            }
        )
    else:
        changed = request.model_copy(
            update={"instructions": [*request.instructions, "identity coverage test"]}
        )

    assert changed.request_digest != request.request_digest


def test_frozen_request_persist_reread_recomputes_exact_identity(tmp_path: Path) -> None:
    request, source = _request_and_source()
    artifact = freeze_oled_llm_paper_mapping_request(
        request=request,
        source_candidates=[source],
        run_id="replay-run",
        generated_at="2026-08-25T00:00:00Z",
    )
    path = tmp_path / "frozen_domain_mapping_request.json"
    persisted = persist_frozen_oled_llm_paper_mapping_request(path, artifact)
    loaded = load_frozen_oled_llm_paper_mapping_request(path)

    assert persisted.request_digest == request.request_digest
    assert loaded.request_digest == request.request_digest
    assert loaded.request.request_digest == request.request_digest
    assert loaded.source_candidates_digest == compute_oled_source_candidates_digest([source])
    assert loaded.metadata["historically_persisted"] is False

    replayed = persist_frozen_oled_llm_paper_mapping_request(path, artifact)
    assert replayed == persisted

    with pytest.raises(ValueError, match="different identity"):
        persist_frozen_oled_llm_paper_mapping_request(
            path,
            artifact.model_copy(update={"run_id": "different-run"}),
        )


def test_tampered_frozen_request_fails_closed_before_candidate_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, source = _request_and_source()
    artifact = freeze_oled_llm_paper_mapping_request(
        request=request,
        source_candidates=[source],
        run_id="replay-run",
    )
    request_path = tmp_path / "frozen.json"
    persist_frozen_oled_llm_paper_mapping_request(request_path, artifact)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    payload["request"]["document_context"][0]["text"] = "tampered source text"
    request_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="request_digest"):
        load_frozen_oled_llm_paper_mapping_request(request_path)

    called = False

    def must_not_assemble(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("candidate assembly was reached after artifact tampering")

    monkeypatch.setattr(br2_adapter, "build_oled_br2_candidate_raw_dataset", must_not_assemble)
    result = br2_adapter.prepare_oled_candidate_raw_dataset_adapter(
        {
            "frozen_domain_mapping_request_path": str(request_path),
            "contextual_mapping_result_path": str(tmp_path / "missing-result.json"),
            "provider_invocation_manifest_path": str(tmp_path / "missing-manifest.json"),
            "output_root": str(tmp_path / "output"),
        }
    )
    assert result["status"] == "failed"
    assert called is False


def test_provider_and_mapping_result_linkage_is_exact() -> None:
    request, _source = _request_and_source()
    mapping_result = _mapping_result(request)
    manifest = build_frozen_oled_llm_provider_invocation_manifest(
        request_digest=request.request_digest,
        invocation=mapping_result.llm_invocation,
        generated_at="2026-08-25T00:00:00Z",
    )
    manifest_payload = manifest.model_dump(mode="json")
    assert "raw_response" not in json.dumps(manifest_payload)
    assert "response_id" not in manifest_payload
    artifact = freeze_oled_llm_paper_mapping_request(
        request=request,
        source_candidates=[_source_and_packet()[0]],
        run_id="replay-run",
    )

    verify_oled_br2_replay_binding(
        domain_request=artifact,
        mapping_result=mapping_result,
        invocation_manifest=manifest,
    )

    with pytest.raises(ValueError, match="provider invocation request_digest"):
        verify_oled_br2_replay_binding(
            domain_request=artifact,
            mapping_result=mapping_result,
            invocation_manifest=manifest.model_copy(
                update={"request_digest": "0" * 64}
            ),
        )

    with pytest.raises(ValueError, match="provider invocation provider"):
        verify_oled_br2_replay_binding(
            domain_request=artifact,
            mapping_result=mapping_result,
            invocation_manifest=manifest.model_copy(update={"provider": "tampered"}),
        )


def test_candidate_assembly_replay_uses_only_verified_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, source = _request_and_source()
    mapping_result = _mapping_result(request)
    domain_path = tmp_path / "frozen_domain_mapping_request.json"
    persist_frozen_oled_llm_paper_mapping_request(
        domain_path,
        freeze_oled_llm_paper_mapping_request(
            request=request,
            source_candidates=[source],
            run_id="replay-run",
        ),
    )
    result_path = tmp_path / "mapping_result.json"
    manifest_path = tmp_path / "provider_invocation_manifest.json"
    write_json(result_path, mapping_result.model_dump(mode="json"))
    persisted_manifest = persist_frozen_oled_llm_provider_invocation_manifest(
        manifest_path,
        build_frozen_oled_llm_provider_invocation_manifest(
            request_digest=request.request_digest,
            invocation=mapping_result.llm_invocation,
        ),
    )
    assert load_frozen_oled_llm_provider_invocation_manifest(
        manifest_path
    ).invocation_digest == persisted_manifest.invocation_digest

    provider_calls = 0

    def no_provider(*args: object, **kwargs: object) -> None:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("replay unexpectedly called a provider")

    monkeypatch.setattr(br2_adapter, "create_llm_provider", no_provider)
    result = br2_adapter.prepare_oled_candidate_raw_dataset_adapter(
        {
            "frozen_domain_mapping_request_path": str(domain_path),
            "contextual_mapping_result_path": str(result_path),
            "provider_invocation_manifest_path": str(manifest_path),
            "output_root": str(tmp_path / "candidate-output"),
        }
    )

    assert result["status"] == "success"
    assert provider_calls == 0
    package = json.loads(
        Path(result["outputs"]["candidate_raw_dataset"]).read_text(encoding="utf-8")
    )
    assert package["candidate_records"]
    assert package["metadata"]["request_digest"] == request.request_digest


def test_arbitrary_rebuilt_request_with_different_digest_is_rejected() -> None:
    request, source = _request_and_source()
    mapping_result = _mapping_result(request)
    rebuilt = request.model_copy(
        update={"instructions": [*request.instructions, "rebuilt differently"]}
    )
    assert rebuilt.request_digest != request.request_digest

    from ai4s_agent.domains.oled_br2_candidate_raw_dataset import (
        build_oled_br2_candidate_raw_dataset,
    )

    with pytest.raises(ValueError, match="mapping result request_digest"):
        build_oled_br2_candidate_raw_dataset(rebuilt, mapping_result, [source])


def test_recovery_requires_all_historical_identity_links() -> None:
    request, source = _request_and_source()
    recovered = recover_oled_llm_paper_mapping_request(
        rebuilt_request=request,
        source_candidates=[source],
        recorded_request_digest=request.request_digest,
        provider_request_digest=request.request_digest,
        mapping_result_request_digest=request.request_digest,
        run_id="v8-recovery",
    )

    assert recovered.request_digest == request.request_digest
    assert recovered.metadata["recovery_mode"] == "deterministic_rebuild_digest_verified"
    assert recovered.metadata["historically_persisted"] is False

    with pytest.raises(ValueError, match="recorded digest"):
        recover_oled_llm_paper_mapping_request(
            rebuilt_request=request,
            source_candidates=[source],
            recorded_request_digest="0" * 64,
            provider_request_digest=request.request_digest,
            mapping_result_request_digest=request.request_digest,
            run_id="v8-recovery",
        )


def test_source_evidence_tampering_changes_its_binding_digest() -> None:
    source, _packet = _source_and_packet()
    changed = source.model_copy(
        update={"table_rows": [{"Emitter": "Molecule-A", "PLQY (%)": "83"}]}
    )

    assert compute_oled_source_candidates_digest([changed]) != (
        compute_oled_source_candidates_digest([source])
    )
