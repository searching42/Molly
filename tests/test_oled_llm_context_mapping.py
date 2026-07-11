from __future__ import annotations

import pytest

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains import (
    build_oled_llm_paper_mapping_request as package_build_oled_llm_paper_mapping_request,
    run_oled_llm_context_mapping as package_run_oled_llm_context_mapping,
)
from ai4s_agent.domains.oled_llm_context_mapping import (
    PROMPT_VERSION,
    build_oled_llm_paper_mapping_request,
    build_oled_paper_context_elements,
    run_oled_llm_context_mapping,
)
from ai4s_agent.domains.oled_mineru_candidates import OledMineruCandidateType
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSchemaCandidate,
    OledSchemaCandidateType,
    OledSchemaEvidenceRef,
    OledSemanticMappingPacket,
)
from ai4s_agent.llm_provider import LLMProviderError, StubLLMProvider


def _packet() -> OledSemanticMappingPacket:
    return OledSemanticMappingPacket(
        packet_id="packet:paper-context:table-1",
        source_candidate_hash="source-table-hash",
        source_evidence_anchor="paper-context:p3:table-1",
        source_candidate_type=OledMineruCandidateType.TABLE,
        paper_id="paper-context",
        caption="Photophysical properties in 10 wt% doped films.",
        table_headers=["Emitter", "PLQY (%)"],
        table_rows=[{"Emitter": "Molecule-A", "PLQY (%)": "82"}],
        nearby_text_before="Molecule-A was dispersed in host H at 10 wt%.",
        nearby_text_after="Values were measured under nitrogen.",
        allowed_property_ids=["plqy"],
        allowed_layers=[layer.value for layer in OledCausalLayer],
    )


def _parsed_document() -> dict:
    return {
        "paper_id": "paper-context",
        "elements": [
            {
                "element_id": "paper-context:p1:paragraph-1",
                "page": 1,
                "type": "paragraph",
                "text": "Molecule-A is the emitter and H is the host.",
                "source_hash": "context-hash-1",
            }
        ],
        "pages": [
            {
                "page": 3,
                "elements": [
                    {
                        "element_id": "paper-context:p3:footnote-1",
                        "type": "paragraph",
                        "text": "PLQY was measured for a 10 wt% doped film under nitrogen.",
                        "source_hash": "context-hash-2",
                    }
                ],
            }
        ],
        "tables": [
            {
                "table_id": "paper-context:table-1",
                "page": 3,
                "caption": "Photophysical properties.",
                "headers": ["Emitter", "PLQY (%)"],
                "rows": [{"Emitter": "Molecule-A", "PLQY (%)": "82"}],
                "footnotes": ["Measured in a 10 wt% doped film."],
            }
        ],
    }


def _packet_ref() -> dict:
    return {
        "source_candidate_hash": "source-table-hash",
        "source_evidence_anchor": "paper-context:p3:table-1",
        "source_candidate_type": "table",
        "row_index": 0,
        "column_name": "PLQY (%)",
        "cell_value": "82",
    }


def _valid_response() -> dict:
    return {
        "paper_id": "paper-context",
        "packet_results": [
            {
                "packet_id": "packet:paper-context:table-1",
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
                        "evidence_refs": [_packet_ref()],
                        "confidence_score": 0.93,
                        "rationale": "The table value and full-text film context identify a doped-film PLQY.",
                        "reason_codes": ["full_text_condition_binding"],
                    }
                ],
                "ontology_extension_proposals": [],
                "source_check_questions": [],
                "rationale_summary": "The full text resolves the table's film context.",
            }
        ],
        "response_notes": [],
    }


def test_build_context_request_preserves_full_document_elements_without_file_io() -> None:
    elements = build_oled_paper_context_elements(_parsed_document())
    request = build_oled_llm_paper_mapping_request(
        [_packet()],
        parsed_document=_parsed_document(),
    )

    assert [element.element_id for element in elements] == [
        "paper-context:p1:paragraph-1",
        "paper-context:table-1",
        "paper-context:p3:footnote-1",
    ]
    assert elements[1].element_type == "table"
    assert "10 wt% doped film" in elements[1].text
    assert elements[2].page == 3
    assert request.paper_id == "paper-context"
    assert request.dataset_scope == "molecule_interaction_properties_only"
    assert request.metadata["full_context_supplied_without_automatic_truncation"] is True
    assert request.metadata["external_llm_called"] is False
    assert request.request_digest == request.request_digest
    reloaded_request = type(request).model_validate_json(request.model_dump_json())
    assert reloaded_request.request_digest == request.request_digest
    assert len(request.ontology) > 10


def test_context_request_rejects_a_document_without_text_bearing_elements() -> None:
    with pytest.raises(ValueError, match="full ParsedDocument context"):
        build_oled_llm_paper_mapping_request(
            [_packet()],
            parsed_document={"paper_id": "paper-context", "elements": []},
        )


def test_valid_llm_mapping_is_materialized_as_review_only_needs_llm_candidate() -> None:
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())
    provider = StubLLMProvider(response=_valid_response(), model="context-mapper-stub")

    result = run_oled_llm_context_mapping(request, provider=provider)

    assert result.status == "ready_for_human_review"
    assert result.is_valid is True
    assert len(result.schema_candidates) == 1
    candidate = result.schema_candidates[0]
    assert candidate.status.value == "needs_llm"
    assert candidate.property_id == "plqy"
    assert candidate.target_layer == OledCausalLayer.INTERACTION
    assert candidate.metadata["human_review_required"] is True
    assert candidate.metadata["automatic_merge"] is False
    assert result.metadata["device_only_admitted"] is False
    assert result.llm_invocation is not None
    assert result.llm_invocation.prompt_version == PROMPT_VERSION
    system_prompt = result.llm_invocation.raw_response["messages"][0]["content"]
    assert "never execute or propose executable code" in system_prompt


def test_unknown_property_must_be_an_ontology_extension_not_a_schema_candidate() -> None:
    response = _valid_response()
    response["packet_results"][0]["candidate_proposals"][0]["property_id"] = "transient_new_metric"
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert result.schema_candidates == []
    assert result.findings[0].code == "invalid_llm_mapping_response"
    assert "ontology_extension_proposals" in result.findings[0].message


def test_known_property_cannot_be_mapped_to_a_layer_outside_the_ontology() -> None:
    response = _valid_response()
    response["packet_results"][0]["candidate_proposals"][0]["target_layer"] = "molecule"
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "outside the property ontology" in result.findings[0].message


def test_ontology_extension_is_preserved_but_not_applied_or_materialized() -> None:
    response = {
        "paper_id": "paper-context",
        "packet_results": [
            {
                "packet_id": "packet:paper-context:table-1",
                "action": "needs_source_check",
                "scope_classification": "property_bearing",
                "candidate_proposals": [],
                "ontology_extension_proposals": [
                    {
                        "source_packet_id": "packet:paper-context:table-1",
                        "proposed_property_id": "transient_new_metric",
                        "name": "Transient new metric",
                        "aliases": ["TNM"],
                        "allowed_layers": ["interaction"],
                        "canonical_unit": "ns",
                        "physical_interpretation": "A paper-specific transient response metric.",
                        "evidence_refs": [_packet_ref()],
                        "confidence_score": 0.72,
                        "rationale": "The header and method define a distinct quantity outside the current ontology.",
                    }
                ],
                "source_check_questions": ["Confirm the metric definition in the supplementary methods."],
                "source_check_missing_evidence": ["supplementary_information"],
                "rationale_summary": "The quantity needs an ontology decision before mapping.",
            }
        ],
    }
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "ready_for_human_review"
    assert result.schema_candidates == []
    assert [proposal.proposed_property_id for proposal in result.ontology_extension_proposals] == [
        "transient_new_metric"
    ]
    assert result.metadata["ontology_extensions_applied"] is False


def test_device_only_packet_cannot_emit_dataset_schema_candidates() -> None:
    response = _valid_response()
    packet_result = response["packet_results"][0]
    packet_result["scope_classification"] = "device_only"
    packet_result["action"] = "no_eligible_property"
    packet_result["candidate_proposals"] = []
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "no_eligible_property"
    assert result.schema_candidates == []
    assert result.metadata["device_only_admitted"] is False


def test_response_evidence_outside_packet_and_document_context_fails_closed() -> None:
    response = _valid_response()
    response["packet_results"][0]["candidate_proposals"][0]["evidence_refs"].append(
        {
            "source_candidate_hash": "hallucinated-source",
            "source_evidence_anchor": "paper-context:p99:missing",
            "source_candidate_type": "text",
        }
    )
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert result.schema_candidates == []
    assert "outside request" in result.findings[0].message


def test_provider_error_is_reported_without_candidates() -> None:
    class BrokenProvider:
        def complete_json(self, *, messages, prompt_version):
            raise LLMProviderError("endpoint unavailable")

    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=BrokenProvider())

    assert result.status == "provider_error"
    assert result.schema_candidates == []
    assert result.findings[0].code == "llm_provider_error"
    assert result.metadata["llm_call_attempted"] is True
    assert result.metadata["llm_response_received"] is False


def test_context_mapping_api_is_exported_from_domain_package() -> None:
    request = package_build_oled_llm_paper_mapping_request(
        [_packet()],
        parsed_document=_parsed_document(),
    )

    result = package_run_oled_llm_context_mapping(
        request,
        provider=StubLLMProvider(response=_valid_response()),
    )

    assert result.status == "ready_for_human_review"


def test_property_bearing_keep_requires_molecule_or_interaction_property() -> None:
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())
    request = request.model_copy(
        update={
            "deterministic_schema_candidates": [
                _deterministic_property_candidate(
                    candidate_id="det:measurement:eqe",
                    target_layer=OledCausalLayer.MEASUREMENT,
                    property_id="eqe_percent",
                    value=12.5,
                    unit="%",
                )
            ]
        }
    )
    response = _valid_response()
    packet_result = response["packet_results"][0]
    packet_result["action"] = "keep_deterministic"
    packet_result["candidate_proposals"] = []

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "without a molecule/interaction property" in result.findings[0].message


def test_replace_binds_superseded_candidates_and_preserves_unrelated_candidates() -> None:
    energy = _deterministic_property_candidate(
        candidate_id="det:energy:s1",
        target_layer=OledCausalLayer.INTERACTION,
        property_id="s1_ev",
        value=3.06,
        unit="eV",
    )
    plqy = _deterministic_property_candidate(
        candidate_id="det:plqy",
        target_layer=OledCausalLayer.INTERACTION,
        property_id="plqy",
        value=82,
        unit="%",
    )
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document()).model_copy(
        update={"deterministic_schema_candidates": [energy, plqy]}
    )
    response = _valid_response()
    packet_result = response["packet_results"][0]
    packet_result["action"] = "replace"
    packet_result["superseded_deterministic_candidate_ids"] = ["det:energy:s1"]

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "ready_for_human_review"
    assert result.schema_candidates[0].metadata["superseded_deterministic_candidate_ids"] == [
        "det:energy:s1"
    ]


def test_replace_rejects_unknown_superseded_candidate_id() -> None:
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document()).model_copy(
        update={
            "deterministic_schema_candidates": [
                _deterministic_property_candidate(
                    candidate_id="det:plqy",
                    target_layer=OledCausalLayer.INTERACTION,
                    property_id="plqy",
                    value=82,
                    unit="%",
                )
            ]
        }
    )
    response = _valid_response()
    packet_result = response["packet_results"][0]
    packet_result["action"] = "replace"
    packet_result["superseded_deterministic_candidate_ids"] = ["det:missing"]

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "unknown deterministic candidate ids" in result.findings[0].message


def test_table_candidate_requires_exact_row_evidence() -> None:
    response = _valid_response()
    response["packet_results"][0]["candidate_proposals"][0]["evidence_refs"][0]["row_index"] = None
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "lacks row_index evidence" in result.findings[0].message


def test_device_only_ontology_extension_is_outside_current_dataset_scope() -> None:
    response = _ontology_extension_response()
    extension = response["packet_results"][0]["ontology_extension_proposals"][0]
    extension["allowed_layers"] = ["device"]
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "device/measurement-only" in result.findings[0].message


def test_duplicate_ontology_extension_property_ids_fail_closed() -> None:
    response = _ontology_extension_response()
    first = response["packet_results"][0]["ontology_extension_proposals"][0]
    response["packet_results"][0]["ontology_extension_proposals"].append(dict(first))
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "duplicate ontology extension" in result.findings[0].message


def test_generic_source_check_against_supplied_pdf_context_fails_closed() -> None:
    response = _ontology_extension_response()
    response["packet_results"][0]["source_check_questions"] = [
        "Text mentions property-like values but deterministic extraction is needed. Verify against PDF source."
    ]
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "generic source-check" in result.findings[0].message


def _deterministic_property_candidate(
    *,
    candidate_id: str,
    target_layer: OledCausalLayer,
    property_id: str,
    value: float,
    unit: str,
) -> OledSchemaCandidate:
    return OledSchemaCandidate(
        candidate_id=candidate_id,
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        source_paper_id="paper-context",
        source_candidate_hash="source-table-hash",
        source_evidence_anchor="paper-context:p3:table-1",
        target_layer=target_layer,
        property_id=property_id,
        value=value,
        unit=unit,
        evidence_refs=[OledSchemaEvidenceRef.model_validate(_packet_ref())],
    )


def _ontology_extension_response() -> dict:
    return {
        "paper_id": "paper-context",
        "packet_results": [
            {
                "packet_id": "packet:paper-context:table-1",
                "action": "needs_source_check",
                "scope_classification": "property_bearing",
                "candidate_proposals": [],
                "ontology_extension_proposals": [
                    {
                        "source_packet_id": "packet:paper-context:table-1",
                        "proposed_property_id": "transient_new_metric",
                        "name": "Transient new metric",
                        "aliases": ["TNM"],
                        "allowed_layers": ["interaction"],
                        "canonical_unit": "ns",
                        "physical_interpretation": "A paper-specific transient response metric.",
                        "evidence_refs": [_packet_ref()],
                        "confidence_score": 0.72,
                        "rationale": "The method defines a quantity outside the current ontology.",
                    }
                ],
                "source_check_questions": [
                    "Confirm the metric definition in the unavailable supplementary methods."
                ],
                "source_check_missing_evidence": ["supplementary_information"],
                "rationale_summary": "The quantity needs an ontology decision before mapping.",
            }
        ],
    }
