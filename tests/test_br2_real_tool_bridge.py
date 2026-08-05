from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.pr_fast]

from ai4s_agent.adapters.br2_real_tool_bridge import (
    OledBr2ExternalLLMContentAuthorization,
    extract_oled_evidence_bridge_adapter,
    prepare_oled_candidate_raw_dataset_bridge_adapter,
)
from ai4s_agent.domains.oled_br2_candidate_raw_dataset import (
    build_oled_br2_candidate_raw_dataset,
)
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMPaperMappingRequest,
    run_oled_llm_context_mapping,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSchemaCandidate,
    OledSchemaCandidateType,
)
from ai4s_agent.harness_tracing import _validate_attribute
from ai4s_agent.llm_provider import StubLLMProvider
from ai4s_agent.planner import br2_real_tool_observability_smoke_task_registry_v1
from ai4s_agent.schemas import ParsedDocument, ParsedDocumentElement, ParsedTable


def _parsed_document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="paper-br2-contract",
        source_path="redacted-source.pdf",
        parser_backend="mineru_test_contract",
        metadata={"source_hash": "sha256:" + "1" * 64},
        elements=[
            ParsedDocumentElement(
                element_id="el:p1:context",
                page=1,
                type="paragraph",
                text="The experimental context is reported in the photophysical table.",
                markdown="The experimental context is reported in the photophysical table.",
                source_hash="sha256:" + "2" * 64,
            )
        ],
        tables=[
            ParsedTable(
                table_id="table:p13:photophysical",
                caption="Photophysical properties and energy levels.",
                headers=["Emitter", "PLQY (%)", "HOMO (eV)"],
                rows=[
                    {
                        "Emitter": "Molecule-A",
                        "PLQY (%)": "82",
                        "HOMO (eV)": "-5.40",
                    }
                ],
                page=13,
                markdown=(
                    "| Emitter | PLQY (%) | HOMO (eV) |\n"
                    "| --- | --- | --- |\n"
                    "| Molecule-A | 82 | -5.40 |"
                ),
            )
        ],
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _valid_response(packets: list[dict[str, object]]) -> dict[str, object]:
    packet = packets[0]
    packet_id = str(packet["packet_id"])
    source_hash = str(packet["source_candidate_hash"])
    anchor = str(packet["source_evidence_anchor"])
    return {
        "paper_id": "paper-br2-contract",
        "packet_results": [
            {
                "packet_id": packet_id,
                "action": "supplement",
                "scope_classification": "property_bearing",
                "candidate_proposals": [
                    {
                        "candidate_type": "property_observation",
                        "target_layer": "interaction",
                        "property_id": "plqy",
                        "property_label": "Photoluminescence quantum yield",
                        "value": 82,
                        "unit": "%",
                        "reported_value_text": "82",
                        "reported_decimal_places": 0,
                        "material_name": "Molecule-A",
                        "material_role": "emitter",
                        "evidence_refs": [
                            {
                                "source_candidate_hash": source_hash,
                                "source_evidence_anchor": anchor,
                                "source_candidate_type": "table",
                                "row_index": 0,
                                "column_name": "PLQY (%)",
                                "cell_value": "82",
                            }
                        ],
                        "confidence_score": 0.91,
                        "rationale": "The exact table row binds the emitter and reported PLQY.",
                        "reason_codes": ["contract_test"],
                    }
                ],
                "ontology_extension_proposals": [],
                "source_check_questions": [],
                "rationale_summary": "The table provides a known property with exact row evidence.",
            }
        ],
        "response_notes": [],
    }


def test_br2_registry_is_narrow_and_reuses_existing_authority() -> None:
    registry = br2_real_tool_observability_smoke_task_registry_v1()
    task_ids = {task.task_id for task in registry.list_tasks()}
    assert task_ids == {
        "parse_document",
        "extract_oled_evidence",
        "map_oled_contextual_semantics",
        "prepare_oled_candidate_raw_dataset",
        "await_oled_candidate_confirmation",
    }
    assert not task_ids.intersection(
        {"train_model", "generate_candidates", "predict_candidates", "filter_rank"}
    )
    assert registry.get("await_oled_candidate_confirmation").gates == [
        "gate_3_train_config"
    ]


def test_stub_contract_runs_extraction_mapping_and_candidate_package(tmp_path: Path) -> None:
    run_id = "run-br2-contract"
    parsed_path = tmp_path / "parsed_document.json"
    _write_json(parsed_path, _parsed_document().model_dump(mode="json"))
    extraction_root = tmp_path / "extraction"
    extraction = extract_oled_evidence_bridge_adapter(
        {
            "project_id": "project-br2-contract",
            "run_id": run_id,
            "task_id": "extract_oled_evidence",
            "parsed_document_path": str(parsed_path),
            "output_root": str(extraction_root),
        }
    )
    assert extraction["status"] == "success"
    outputs = extraction["outputs"]
    candidates_payload = json.loads(Path(outputs["oled_mineru_candidates"]).read_text())
    packets_payload = json.loads(
        Path(outputs["oled_semantic_mapping_packets"]).read_text()
    )
    assert candidates_payload["candidates"]
    assert packets_payload["packets"]
    result = run_oled_llm_context_mapping(
        OledLLMPaperMappingRequest.model_validate(
            json.loads(Path(outputs["oled_llm_context_request"]).read_text())["request"]
        ),
        provider=StubLLMProvider(
            response=_valid_response(packets_payload["packets"]),
            model="stub-contextual-provider",
        ),
    )
    assert result.status == "ready_for_human_review"
    assert result.metadata["automatic_candidate_merge"] is False

    mapping_result_path = tmp_path / "mapping_result.json"
    _write_json(mapping_result_path, result.model_dump(mode="json"))
    package = prepare_oled_candidate_raw_dataset_bridge_adapter(
        {
            "project_id": "project-br2-contract",
            "run_id": run_id,
            "task_id": "prepare_oled_candidate_raw_dataset",
            "parsed_document_path": str(parsed_path),
            "oled_mineru_candidates_path": outputs["oled_mineru_candidates"],
            "oled_deterministic_schema_candidates_path": outputs[
                "oled_deterministic_schema_candidates"
            ],
            "oled_contextual_mapping_result_path": str(mapping_result_path),
            "oled_llm_context_request_path": outputs["oled_llm_context_request"],
            "output_root": str(tmp_path / "candidate_dataset"),
        }
    )
    assert package["status"] == "success"
    dataset = json.loads(Path(package["outputs"]["candidate_raw_dataset"]).read_text())
    snapshot = json.loads(Path(package["outputs"]["review_snapshot"]).read_text())
    assert dataset["rows"]
    assert all(row["evidence_anchor"] and row["evidence_refs"] for row in dataset["rows"])
    assert all(row["causal_layer"] in {"molecule", "interaction"} for row in dataset["rows"])
    assert snapshot["review_status"] == "needs_user_confirmation"
    assert snapshot["confirmed_dataset_created"] is False
    assert snapshot["downstream_dispatch"] == {
        "training": False,
        "generation": False,
        "prediction": False,
        "ranking": False,
    }


def test_invalid_contextual_response_fails_closed() -> None:
    from ai4s_agent.domains.oled_llm_context_mapping import build_oled_llm_paper_mapping_request
    from ai4s_agent.domains.oled_mineru_semantic_mapping import (
        OledSemanticMappingPacket,
    )

    packet = OledSemanticMappingPacket(
        packet_id="packet:invalid",
        source_candidate_hash="source-invalid",
        source_evidence_anchor="anchor-invalid",
        source_candidate_type="table",
        paper_id="paper-invalid",
        table_headers=["Emitter", "PLQY (%)"],
        table_rows=[{"Emitter": "Molecule-A", "PLQY (%)": "82"}],
        allowed_property_ids=["plqy"],
        allowed_layers=[layer.value for layer in OledCausalLayer],
    )
    request = build_oled_llm_paper_mapping_request(
        [packet],
        parsed_document={
            "paper_id": "paper-invalid",
            "elements": [{"element_id": "e1", "page": 1, "text": "context"}],
        },
    )
    result = run_oled_llm_context_mapping(
        request,
        provider=StubLLMProvider(
            response={"paper_id": "paper-invalid", "packet_results": []}
        ),
    )
    assert result.status == "invalid_response"
    assert result.schema_candidates == []


def test_device_only_candidate_is_excluded_from_candidate_dataset() -> None:
    source = []
    deterministic = OledSchemaCandidate(
        candidate_id="device-only",
        candidate_type=OledSchemaCandidateType.DEVICE_STRUCTURE,
        source_paper_id="paper-device",
        source_candidate_hash="source-device",
        source_evidence_anchor="anchor-device",
        target_layer=OledCausalLayer.DEVICE,
        device_stack=["ITO", "EML"],
        evidence_refs=[
            {
                "source_candidate_hash": "source-device",
                "source_evidence_anchor": "anchor-device",
                "source_candidate_type": "text",
            }
        ],
    )
    result = type("MappingResult", (), {
        "paper_id": "paper-device",
        "status": "ready_for_human_review",
        "schema_candidates": [],
    })()
    with pytest.raises(ValueError, match="no molecule/interaction property rows"):
        build_oled_br2_candidate_raw_dataset(
            paper_id="paper-device",
            deterministic_candidates=[deterministic],
            contextual_result=result,
            source_candidates=source,
            request_digest="sha256:" + "3" * 64,
            response_digest="sha256:" + "4" * 64,
        )


def test_external_authorization_rejects_raw_pdf_and_downstream_scope() -> None:
    with pytest.raises(ValueError, match="raw PDF"):
        OledBr2ExternalLLMContentAuthorization(
            authorization_id="auth-1",
            run_id="run-1",
            paper_id="paper-1",
            provider_class="openai_compatible",
            model="deepseek-v4-flash-ascend1",
            raw_pdf_allowed=True,
        )


def test_contextual_llm_observability_accepts_prompt_and_response_identifiers() -> None:
    assert _validate_attribute("prompt_version", "oled.contextual_semantic_mapping.v5")[1]
    assert _validate_attribute("response_id", "resp-20260805-1")[1]
