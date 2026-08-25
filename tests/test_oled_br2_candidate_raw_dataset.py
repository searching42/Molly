from __future__ import annotations

import pytest

from ai4s_agent.domains.oled_br2_candidate_raw_dataset import (
    build_oled_br2_candidate_raw_dataset,
)
from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import OledMeasurementCondition
from ai4s_agent.domains.oled_llm_context_mapping import (
    OledLLMContextMappingResult,
    OledLLMPacketMappingProposal,
    OledLLMSchemaCandidateProposal,
    build_oled_llm_paper_mapping_request,
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
    OledSemanticMappingPacket,
)
from ai4s_agent.domains.oled_property_ontology import DEFAULT_OLED_PROPERTY_ONTOLOGY


def _source_and_packet(index: int) -> tuple[OledMineruCandidate, OledSemanticMappingPacket]:
    source_hash = f"source-hash-{index}"
    anchor = f"paper:p3:table-{index}"
    source = OledMineruCandidate(
        paper_id="paper",
        source_format=OledMineruSourceFormat.MINERU_LIKE,
        candidate_type=OledMineruCandidateType.TABLE,
        page_index=3,
        block_index=index,
        block_id=f"table-{index}",
        raw_text=f"Emitter | PLQY (%)\nMolecule-A | 82\nTable {index}",
        caption=f"Photophysical properties {index}",
        table_headers=["Emitter", "PLQY (%)"],
        table_rows=[{"Emitter": "Molecule-A", "PLQY (%)": "82"}],
        table_parse_status=OledMineruTableParseStatus.PARSED,
        evidence_anchor=anchor,
        candidate_hash=source_hash,
    )
    packet = OledSemanticMappingPacket(
        packet_id=f"packet-{index}",
        source_candidate_hash=source_hash,
        source_evidence_anchor=anchor,
        source_candidate_type=OledMineruCandidateType.TABLE,
        paper_id="paper",
        caption=f"Photophysical properties {index}",
        raw_text=source.raw_text,
        table_headers=source.table_headers,
        table_rows=source.table_rows,
        allowed_property_ids=["plqy"],
        allowed_layers=[layer.value for layer in OledCausalLayer],
    )
    return source, packet


def _evidence_ref(source: OledMineruCandidate) -> OledSchemaEvidenceRef:
    return OledSchemaEvidenceRef(
        source_candidate_hash=source.candidate_hash,
        source_evidence_anchor=source.evidence_anchor,
        source_candidate_type="table",
        row_index=0,
        column_name="PLQY (%)",
        cell_value="82",
    )


def _schema_candidate(
    source: OledMineruCandidate,
    candidate_id: str,
    *,
    value: int = 50,
    status: OledSchemaCandidateStatus = OledSchemaCandidateStatus.PROPOSED,
) -> OledSchemaCandidate:
    return OledSchemaCandidate(
        candidate_id=candidate_id,
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        status=status,
        source_paper_id="paper",
        source_candidate_hash=source.candidate_hash,
        source_evidence_anchor=source.evidence_anchor,
        target_layer=OledCausalLayer.INTERACTION,
        property_id="plqy",
        property_label="Photoluminescence quantum yield",
        value=value,
        unit="%",
        reported_value_text=str(value),
        reported_decimal_places=0,
        material_name="Molecule-A",
        evidence_refs=[_evidence_ref(source)],
        confidence_score=0.8,
        metadata={"source_packet_id": f"packet-{source.block_index}"},
    )


def _proposal(candidate: OledSchemaCandidate) -> OledLLMSchemaCandidateProposal:
    return OledLLMSchemaCandidateProposal(
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        target_layer=OledCausalLayer.INTERACTION,
        property_id="plqy",
        property_label="Photoluminescence quantum yield",
        value=candidate.value,
        unit="%",
        reported_value_text=candidate.reported_value_text,
        reported_decimal_places=0,
        material_name="Molecule-A",
        evidence_refs=list(candidate.evidence_refs),
        confidence_score=0.9,
        rationale="The table binds the value to the emitter row.",
    )


def _request(
    packets: list[OledSemanticMappingPacket],
    deterministic: list[OledSchemaCandidate],
):
    request = build_oled_llm_paper_mapping_request(
        packets,
        parsed_document={
            "paper_id": "paper",
            "elements": [
                {
                    "element_id": "paper:p1:paragraph-1",
                    "page": 1,
                    "type": "paragraph",
                    "text": "Molecule-A is the emitter.",
                    "source_hash": "context-hash",
                }
            ],
            "tables": [
                {
                    "table_id": packet.source_evidence_anchor.replace(":", "-"),
                    "page": 3,
                    "caption": packet.caption,
                    "headers": list(packet.table_headers),
                    "rows": [dict(row) for row in packet.table_rows],
                    "footnotes": [],
                }
                for packet in packets
                if packet.source_candidate_type == OledMineruCandidateType.TABLE
            ],
        },
    )
    return request.model_copy(update={"deterministic_schema_candidates": deterministic})


def _result(request, packet_results, schema_candidates):
    return OledLLMContextMappingResult(
        paper_id=request.paper_id,
        status="ready_for_human_review",
        request_digest=request.request_digest,
        packet_results=packet_results,
        schema_candidates=schema_candidates,
        metadata={"llm_called": True},
    )


def test_action_aware_reducer_does_not_unconditionally_concat_or_clear() -> None:
    sources_packets = [_source_and_packet(index) for index in range(3)]
    sources = [source for source, _ in sources_packets]
    packets = [packet for _, packet in sources_packets]
    deterministic = [_schema_candidate(source, f"det-{index}") for index, source in enumerate(sources)]
    request = _request(packets, deterministic)
    supplement_candidate = _schema_candidate(
        sources[1], "llm-supplement", value=61, status=OledSchemaCandidateStatus.NEEDS_LLM
    )
    replace_candidate = _schema_candidate(
        sources[2], "llm-replace", value=72, status=OledSchemaCandidateStatus.NEEDS_LLM
    )
    packet_results = [
        OledLLMPacketMappingProposal(
            packet_id="packet-0",
            action="keep_deterministic",
            scope_classification="property_bearing",
            rationale_summary="Existing deterministic candidate is retained.",
        ),
        OledLLMPacketMappingProposal(
            packet_id="packet-1",
            action="supplement",
            scope_classification="property_bearing",
            candidate_proposals=[_proposal(supplement_candidate)],
            rationale_summary="The context supplies a second proposal.",
        ),
        OledLLMPacketMappingProposal(
            packet_id="packet-2",
            action="replace",
            scope_classification="property_bearing",
            candidate_proposals=[_proposal(replace_candidate)],
            superseded_deterministic_candidate_ids=["det-2"],
            rationale_summary="Only the named deterministic candidate is superseded.",
        ),
    ]
    mapping = _result(request, packet_results, [supplement_candidate, replace_candidate])

    package, _review = build_oled_br2_candidate_raw_dataset(request, mapping, sources)

    dispositions = {item.packet_id: item for item in package.packet_dispositions}
    assert dispositions["packet-0"].selected_candidate_ids == ["det-0"]
    assert set(dispositions["packet-1"].selected_candidate_ids) == {"det-1", "llm-supplement"}
    assert dispositions["packet-2"].selected_candidate_ids == ["llm-replace"]
    assert all(record.status.value in {"compiled", "partial", "needs_review"} for record in package.candidate_records)
    origins = {
        observation.metadata["origin"]
        for record in package.candidate_records
        if record.layered_record is not None
        for layer in (record.layered_record.molecule, record.layered_record.interaction)
        if layer is not None
        for observation in layer.properties
    }
    assert origins == {"deterministic", "llm_proposal"}
    assert package.confirmed is False
    assert package.gold_records_created is False
    assert package.ontology_mutated is False
    assert package.human_confirmation_required is True


def test_foreign_source_candidate_or_evidence_cannot_enter_review_package() -> None:
    source, packet = _source_and_packet(0)
    request = _request([packet], [])
    foreign = _schema_candidate(source, "foreign-paper-candidate").model_copy(
        update={"source_candidate_hash": "not-in-source-roster"}
    )
    mapping = _result(
        request,
        [
            OledLLMPacketMappingProposal(
                packet_id=packet.packet_id,
                action="supplement",
                scope_classification="property_bearing",
                candidate_proposals=[_proposal(foreign)],
                rationale_summary="malformed source binding",
            )
        ],
        [foreign],
    )

    with pytest.raises(ValueError, match="unknown source candidate"):
        build_oled_br2_candidate_raw_dataset(request, mapping, [source])


def test_needs_source_check_stays_unresolved_and_device_only_is_excluded() -> None:
    source, packet = _source_and_packet(0)
    device_source, device_packet = _source_and_packet(1)
    request = _request([packet, device_packet], [])
    mapping = _result(
        request,
        [
            OledLLMPacketMappingProposal(
                packet_id=packet.packet_id,
                action="needs_source_check",
                scope_classification="property_bearing",
                source_check_questions=["Is the host concentration reported in the SI?"],
                source_check_missing_evidence=["supplementary_information"],
                rationale_summary="A source check is required.",
            ),
            OledLLMPacketMappingProposal(
                packet_id=device_packet.packet_id,
                action="no_eligible_property",
                scope_classification="device_only",
                explicit_property_exclusion_reason=None,
                rationale_summary="Device-only performance is out of scope.",
            ),
        ],
        [],
    )

    package, review = build_oled_br2_candidate_raw_dataset(
        request,
        mapping,
        [source, device_source],
    )

    assert package.candidate_records == []
    assert {item.kind for item in package.unresolved_items} == {"source_check", "device_only_excluded"}
    assert review.source_check_count == 1
    assert review.device_only_excluded_count == 1


def test_ontology_review_isolated_without_mutating_default_ontology() -> None:
    source, packet = _source_and_packet(0)
    request = _request([packet], [])
    extension = {
        "source_packet_id": packet.packet_id,
        "proposed_property_id": "new_review_metric",
        "name": "New review metric",
        "allowed_layers": ["interaction"],
        "canonical_unit": "arb",
        "physical_interpretation": "A review-only metric.",
        "evidence_refs": [_evidence_ref(source).model_dump(mode="json")],
        "confidence_score": 0.7,
        "rationale": "The source describes a property absent from the ontology.",
    }
    proposal = OledLLMPacketMappingProposal(
        packet_id=packet.packet_id,
        action="needs_ontology_review",
        scope_classification="property_bearing",
        ontology_extension_proposals=[extension],
        rationale_summary="The ontology needs review.",
    )
    mapping = _result(request, [proposal], [])
    before = [item.property_id for item in DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties()]

    package, review = build_oled_br2_candidate_raw_dataset(request, mapping, [source])

    assert package.candidate_records == []
    assert len(package.ontology_review_proposals) == 1
    assert review.ontology_review_count == 1
    assert [item.property_id for item in DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties()] == before


def test_measurement_context_survives_layered_compilation() -> None:
    source, packet = _source_and_packet(0)
    request = _request([packet], [])
    candidate = _schema_candidate(source, "llm-context", value=82, status=OledSchemaCandidateStatus.NEEDS_LLM).model_copy(
        update={
            "comparison_context": OledMeasurementCondition(
                host_material="Host-H",
                dopant_concentration=10,
                dopant_concentration_unit="wt%",
                sample_form="doped film",
                atmosphere="nitrogen",
            )
        }
    )
    mapping = _result(
        request,
        [
            OledLLMPacketMappingProposal(
                packet_id=packet.packet_id,
                action="supplement",
                scope_classification="property_bearing",
                candidate_proposals=[_proposal(candidate)],
                rationale_summary="Context is retained.",
            )
        ],
        [candidate],
    )

    package, review = build_oled_br2_candidate_raw_dataset(request, mapping, [source])

    assert review.property_observation_count == 1
    observation = package.candidate_records[0].layered_record.interaction.properties[0]
    assert observation.condition is not None
    assert observation.condition.host_material == "Host-H"
    assert observation.condition.dopant_concentration == 10
    assert observation.condition.dopant_concentration_unit == "wt%"
    assert observation.evidence_sources
