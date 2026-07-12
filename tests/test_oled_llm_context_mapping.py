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
                        "reported_value_text": "82",
                        "reported_decimal_places": 0,
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


def _photophysical_packet() -> OledSemanticMappingPacket:
    return _packet().model_copy(
        update={
            "caption": "Prompt lifetime measured for a 10 wt% doped film.",
            "table_headers": ["Emitter", "Prompt lifetime (ns)"],
            "table_rows": [{"Emitter": "Molecule-A", "Prompt lifetime (ns)": "13.20"}],
            "allowed_property_ids": ["prompt_lifetime_ns"],
        }
    )


def _photophysical_response(*, include_context: bool = True) -> dict:
    proposal = {
        "candidate_type": "property_observation",
        "target_layer": "interaction",
        "property_id": "prompt_lifetime_ns",
        "property_label": "Prompt emission-decay lifetime",
        "value": 13.2,
        "unit": "ns",
        "reported_value_text": "13.20",
        "reported_decimal_places": 2,
        "material_name": "Molecule-A in Host-H",
        "evidence_refs": [
            {
                **_packet_ref(),
                "column_name": "Prompt lifetime (ns)",
                "cell_value": "13.20",
            }
        ],
        "confidence_score": 0.91,
        "rationale": "The table and full context bind the prompt lifetime to the doped film.",
    }
    if include_context:
        proposal["comparison_context"] = {
            "measurement_temperature": None,
            "host_material": "Host-H",
            "dopant_concentration": 10,
            "dopant_concentration_unit": "wt%",
            "sample_form": "doped film",
            "excitation_wavelength": None,
            "lifetime_fit_method": None,
        }
    return {
        "paper_id": "paper-context",
        "packet_results": [
            {
                "packet_id": "packet:paper-context:table-1",
                "action": "supplement",
                "scope_classification": "property_bearing",
                "candidate_proposals": [proposal],
                "ontology_extension_proposals": [],
                "source_check_questions": [],
                "rationale_summary": "The property and comparison context are source-bound.",
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
    assert candidate.reported_value_text == "82"
    assert candidate.reported_decimal_places == 0
    assert candidate.metadata["human_review_required"] is True
    assert candidate.metadata["automatic_merge"] is False
    assert result.metadata["device_only_admitted"] is False
    assert result.llm_invocation is not None
    assert result.llm_invocation.prompt_version == PROMPT_VERSION
    system_prompt = result.llm_invocation.raw_response["messages"][0]["content"]
    assert "never execute or propose executable code" in system_prompt


def test_v5_required_photophysical_context_is_materialized_with_explicit_missingness() -> None:
    request = build_oled_llm_paper_mapping_request(
        [_photophysical_packet()], parsed_document=_parsed_document()
    )

    result = run_oled_llm_context_mapping(
        request,
        provider=StubLLMProvider(response=_photophysical_response()),
    )

    assert result.status == "ready_for_human_review"
    context = result.schema_candidates[0].comparison_context
    assert context is not None
    assert context.host_material == "Host-H"
    assert context.dopant_concentration == 10
    assert context.dopant_concentration_unit == "wt%"
    assert context.measurement_temperature is None
    assert "measurement_temperature" in context.model_fields_set
    assert "lifetime_fit_method" in context.model_fields_set


def test_v5_required_photophysical_context_rejects_missing_context_object() -> None:
    request = build_oled_llm_paper_mapping_request(
        [_photophysical_packet()], parsed_document=_parsed_document()
    )

    result = run_oled_llm_context_mapping(
        request,
        provider=StubLLMProvider(response=_photophysical_response(include_context=False)),
    )

    assert result.status == "invalid_response"
    assert "lacks required comparison_context" in result.findings[0].message


def test_v5_required_photophysical_context_rejects_omitted_fields_instead_of_nulls() -> None:
    response = _photophysical_response()
    response["packet_results"][0]["candidate_proposals"][0]["comparison_context"].pop(
        "lifetime_fit_method"
    )
    request = build_oled_llm_paper_mapping_request(
        [_photophysical_packet()], parsed_document=_parsed_document()
    )

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "use explicit null" in result.findings[0].message


def test_legacy_request_without_comparison_context_contract_remains_readable() -> None:
    request = build_oled_llm_paper_mapping_request(
        [_photophysical_packet()], parsed_document=_parsed_document()
    )
    legacy_metadata = dict(request.metadata)
    legacy_metadata.pop("comparison_context_contract_required")
    legacy_metadata.pop("comparison_context_contract_version")
    request = request.model_copy(update={"metadata": legacy_metadata})

    result = run_oled_llm_context_mapping(
        request,
        provider=StubLLMProvider(response=_photophysical_response(include_context=False)),
    )

    assert result.status == "ready_for_human_review"
    assert result.schema_candidates[0].comparison_context is None


def test_unknown_property_must_be_an_ontology_extension_not_a_schema_candidate() -> None:
    response = _valid_response()
    response["packet_results"][0]["candidate_proposals"][0]["property_id"] = "transient_new_metric"
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert result.schema_candidates == []
    assert result.findings[0].code == "invalid_llm_mapping_response"
    assert "ontology_extension_proposals" in result.findings[0].message


def test_v4_numeric_candidate_requires_reported_source_lexeme() -> None:
    response = _valid_response()
    proposal = response["packet_results"][0]["candidate_proposals"][0]
    proposal.pop("reported_value_text")
    proposal.pop("reported_decimal_places")
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "lacks required reported value fields" in result.findings[0].message


def test_v4_numeric_string_candidate_requires_reported_source_lexeme() -> None:
    response = _valid_response()
    proposal = response["packet_results"][0]["candidate_proposals"][0]
    proposal["value"] = "82.0"
    proposal.pop("reported_value_text")
    proposal.pop("reported_decimal_places")
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "lacks required reported value fields" in result.findings[0].message


def test_legacy_request_without_reported_value_contract_remains_readable() -> None:
    response = _valid_response()
    proposal = response["packet_results"][0]["candidate_proposals"][0]
    proposal.pop("reported_value_text")
    proposal.pop("reported_decimal_places")
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())
    legacy_metadata = dict(request.metadata)
    legacy_metadata.pop("reported_value_contract_required")
    legacy_metadata.pop("reported_value_contract_version")
    request = request.model_copy(update={"metadata": legacy_metadata})

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "ready_for_human_review"
    assert result.schema_candidates[0].reported_value_text is None


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
                "action": "needs_ontology_review",
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
                "source_check_questions": [],
                "source_check_missing_evidence": [],
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
    assert "without a molecule/interaction property" in result.findings[0].message


def test_duplicate_ontology_extension_property_ids_fail_closed() -> None:
    response = _ontology_extension_response()
    first = response["packet_results"][0]["ontology_extension_proposals"][0]
    response["packet_results"][0]["ontology_extension_proposals"].append(dict(first))
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "duplicate ontology extension" in result.findings[0].message


def test_generic_source_check_against_supplied_pdf_context_fails_closed() -> None:
    response = _source_check_response()
    response["packet_results"][0]["source_check_questions"] = [
        "Text mentions property-like values but deterministic extraction is needed. Verify against PDF source."
    ]
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "generic source-check" in result.findings[0].message


def test_supplement_can_include_known_candidates_and_ontology_extensions() -> None:
    response = _valid_response()
    extension = _ontology_extension_response()["packet_results"][0]["ontology_extension_proposals"][0]
    response["packet_results"][0]["ontology_extension_proposals"] = [extension]
    request = build_oled_llm_paper_mapping_request([_packet()], parsed_document=_parsed_document())

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "ready_for_human_review"
    assert len(result.schema_candidates) == 1
    assert len(result.ontology_extension_proposals) == 1


def test_explicit_ev_property_signals_require_structured_exclusion_reason() -> None:
    packet = _explicit_property_text_packet()
    request = build_oled_llm_paper_mapping_request([packet], parsed_document=_parsed_document())
    response = _no_eligible_text_response(packet.packet_id)

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "invalid_response"
    assert "explicit property signals" in result.findings[0].message
    assert "homo_ev" in result.findings[0].message
    assert "lumo_ev" in result.findings[0].message


def test_explicit_ev_property_signals_can_be_audited_as_external_background() -> None:
    packet = _explicit_property_text_packet()
    request = build_oled_llm_paper_mapping_request([packet], parsed_document=_parsed_document())
    response = _no_eligible_text_response(packet.packet_id)
    response["packet_results"][0][
        "explicit_property_exclusion_reason"
    ] = "background_or_external_reference"

    result = run_oled_llm_context_mapping(request, provider=StubLLMProvider(response=response))

    assert result.status == "no_eligible_property"
    assert result.is_valid is True


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
                "action": "needs_ontology_review",
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
                "source_check_questions": [],
                "source_check_missing_evidence": [],
                "rationale_summary": "The quantity needs an ontology decision before mapping.",
            }
        ],
    }


def _source_check_response() -> dict:
    return {
        "paper_id": "paper-context",
        "packet_results": [
            {
                "packet_id": "packet:paper-context:table-1",
                "action": "needs_source_check",
                "scope_classification": "property_bearing",
                "candidate_proposals": [],
                "ontology_extension_proposals": [],
                "source_check_questions": [
                    "Inspect the unavailable supplementary table to resolve material assignments."
                ],
                "source_check_missing_evidence": ["supplementary_information"],
                "rationale_summary": "The supplied full text points to an unavailable supplementary table.",
            }
        ],
    }


def _explicit_property_text_packet() -> OledSemanticMappingPacket:
    return _packet().model_copy(
        update={
            "packet_id": "packet:paper-context:text-energy",
            "source_candidate_hash": "source-text-energy-hash",
            "source_evidence_anchor": "paper-context:p2:text-energy",
            "source_candidate_type": OledMineruCandidateType.TEXT,
            "caption": None,
            "raw_text": (
                "The HOMO energy level is -5.30 eV and the LUMO energy level is -2.76 eV."
            ),
            "table_headers": [],
            "table_rows": [],
        }
    )


def _no_eligible_text_response(packet_id: str) -> dict:
    return {
        "paper_id": "paper-context",
        "packet_results": [
            {
                "packet_id": packet_id,
                "action": "no_eligible_property",
                "scope_classification": "no_eligible_property",
                "candidate_proposals": [],
                "ontology_extension_proposals": [],
                "source_check_questions": [],
                "source_check_missing_evidence": [],
                "rationale_summary": "No eligible property.",
            }
        ],
    }
