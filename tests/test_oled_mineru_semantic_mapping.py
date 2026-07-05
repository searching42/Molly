from __future__ import annotations

from ai4s_agent.domains import (
    OledCausalLayer,
    OledMineruCandidateType,
    OledMineruTableParseStatus,
    OledSemanticMapperKind,
    OledSemanticMappingFinding,
    OledSemanticMappingPacket,
    OledSemanticMappingReport,
    OledSchemaCandidate,
    OledSchemaCandidateStatus,
    OledSchemaCandidateType,
    OledSchemaEvidenceRef,
    build_oled_semantic_mapping_packets as package_build_oled_semantic_mapping_packets,
    map_oled_mineru_candidates_to_schema_candidates as package_map_oled_mineru_candidates_to_schema_candidates,
    validate_oled_schema_candidates as package_validate_oled_schema_candidates,
)
from ai4s_agent.domains.oled_mineru_candidates import extract_oled_mineru_candidates_from_document
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    build_oled_semantic_mapping_packets,
    map_oled_mineru_candidates_to_schema_candidates,
    validate_oled_schema_candidates,
)


def _first_candidate(parsed_document, *, paper_id: str = "paper-semantic", md_text: str | None = None):
    return extract_oled_mineru_candidates_from_document(
        parsed_document,
        paper_id=paper_id,
        md_text=md_text,
        include_irrelevant=True,
    )[0]


def test_packet_builder_includes_local_candidate_context_and_no_invention_instructions() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Table 1. OLED device performance.",
                "table_body": (
                    "<table><tr><th>Emitter</th><th>Max EQE (%)</th></tr>"
                    "<tr><td>D1</td><td>22.1</td></tr></table>"
                ),
                "page_idx": 3,
            }
        ],
        md_text=(
            "Before local context mentions a doped device.\n"
            "Table 1. OLED device performance.\n"
            "After local context mentions 100 cd m-2."
        ),
    )

    packet = build_oled_semantic_mapping_packets([candidate])[0]

    assert isinstance(packet, OledSemanticMappingPacket)
    assert packet.source_candidate_hash == candidate.candidate_hash
    assert packet.source_evidence_anchor == candidate.evidence_anchor
    assert packet.source_candidate_type == OledMineruCandidateType.TABLE
    assert packet.paper_id == "paper-semantic"
    assert packet.caption == "Table 1. OLED device performance."
    assert packet.table_headers == ["Emitter", "Max EQE (%)"]
    assert packet.table_rows == [{"Emitter": "D1", "Max EQE (%)": "22.1"}]
    assert "Before local context" in (packet.nearby_text_before or "")
    assert "After local context" in (packet.nearby_text_after or "")
    assert not hasattr(packet, "full_paper_text")
    assert "eqe_percent" in packet.allowed_property_ids
    assert {layer.value for layer in OledCausalLayer}.issubset(set(packet.allowed_layers))
    assert any("Do not invent values" in instruction for instruction in packet.instructions)
    assert packet.expected_output_schema["model"] == "OledSchemaCandidate"


def test_rule_mapper_handles_table_1_like_eml_components() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Table 1. Photophysical properties of EML components in host-dopant films.",
                "table_body": (
                    "| EL colour | Host (S1, T1) (eV) | Assistant dopant (S1, T1) (eV) | "
                    "Assistant dopant concentration (wt%) | ΔEST (eV) | Emitter dopant | "
                    "Emitter dopant concentration (wt%) | ΦPL (%) |\n"
                    "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
                    "| Blue | mCBP | TADF-A | 20 | 0.08 | Emitter-B | 2 | 82 |\n"
                    "| Green | DPEPO | TADF-G | 10 | 0.12 | Emitter-G | 3 | 76 |"
                ),
            }
        ],
        paper_id="paper-table1",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])

    assert report.mapper_kind == OledSemanticMapperKind.RULE_BASED
    assert report.source_candidate_count == 1
    material_roles = [
        schema_candidate
        for schema_candidate in report.schema_candidates
        if schema_candidate.candidate_type == OledSchemaCandidateType.MATERIAL_ROLE
    ]
    properties = [
        schema_candidate
        for schema_candidate in report.schema_candidates
        if schema_candidate.candidate_type == OledSchemaCandidateType.PROPERTY_OBSERVATION
    ]

    assert {candidate.material_role for candidate in material_roles} >= {
        "host",
        "assistant_dopant",
        "emitter_dopant",
    }
    assert ("host", "mCBP") in {(candidate.material_role, candidate.material_name) for candidate in material_roles}
    assert {"delta_e_st_ev", "plqy", "doping_ratio_percent"}.issubset(
        {candidate.property_id for candidate in properties}
    )
    assert all(candidate.source_candidate_hash == candidate.evidence_refs[0].source_candidate_hash for candidate in properties)

    plqy = next(candidate for candidate in properties if candidate.property_id == "plqy")
    assert plqy.target_layer == OledCausalLayer.INTERACTION
    assert plqy.unit == "%"
    assert plqy.evidence_refs[0].row_index == 0
    assert plqy.evidence_refs[0].column_name == "ΦPL (%)"
    assert plqy.evidence_refs[0].cell_value == "82"

    concentration = next(
        candidate
        for candidate in properties
        if candidate.property_id == "doping_ratio_percent"
        and candidate.evidence_refs[0].column_name == "Assistant dopant concentration (wt%)"
    )
    assert concentration.unit == "wt%"
    assert concentration.target_layer == OledCausalLayer.INTERACTION


def test_rule_mapper_handles_table_2_like_device_performance_conditions() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Table 2. Device performance of OLEDs.",
                "table_body": (
                    "| Device | Turn on voltage (V) | Max EQE (%) | Max CE | Max PE | "
                    "Performance at 1,000 cd m^-2 EQE (%) |\n"
                    "| --- | --- | --- | --- | --- | --- |\n"
                    "| B1 | 3.2 | 24.6 | 45.1 | 39.5 | 18.2 |\n"
                    "| G1 | 2.8 | 29.4 | 60.2 | 51.8 | 22.0 |"
                ),
            }
        ],
        paper_id="paper-table2",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    eqe_candidates = [
        schema_candidate
        for schema_candidate in report.schema_candidates
        if schema_candidate.property_id == "eqe_percent"
    ]

    assert len(eqe_candidates) == 4
    assert all(candidate.target_layer == OledCausalLayer.MEASUREMENT for candidate in eqe_candidates)
    max_candidate = next(
        candidate for candidate in eqe_candidates if candidate.evidence_refs[0].column_name == "Max EQE (%)"
    )
    assert max_candidate.metadata["is_max_reported"] is True
    conditioned_candidate = next(
        candidate
        for candidate in eqe_candidates
        if candidate.evidence_refs[0].column_name == "Performance at 1,000 cd m^-2 EQE (%)"
    )
    assert conditioned_candidate.metadata["condition_context_text"] == "1,000 cd m^-2"
    assert conditioned_candidate.metadata["condition_field"] == "luminance"
    assert conditioned_candidate.metadata["condition_value"] == 1000.0
    assert conditioned_candidate.metadata["condition_unit"] == "cd/m^2"
    assert not any(type(candidate).__name__ == "OledLayeredRecord" for candidate in report.schema_candidates)


def test_text_derived_device_structure_candidate_splits_stack_conservatively() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "text",
                "text": (
                    "Blue OLEDs with the structure ITO/a-NPB (35 nm)/mCP:Emitter-B/"
                    "TPBi/LiF/Al were fabricated."
                ),
                "page_idx": 6,
            }
        ],
        paper_id="paper-device-text",
    )

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])
    device_candidate = next(
        candidate
        for candidate in report.schema_candidates
        if candidate.candidate_type == OledSchemaCandidateType.DEVICE_STRUCTURE
    )

    assert device_candidate.device_stack == ["ITO", "a-NPB (35 nm)", "mCP:Emitter-B", "TPBi", "LiF", "Al"]
    assert device_candidate.evidence_refs == [
        OledSchemaEvidenceRef(
            source_candidate_hash=candidate.candidate_hash,
            source_evidence_anchor=candidate.evidence_anchor,
            source_candidate_type="text",
            field_name="raw_text",
        )
    ]
    assert "device_structure_pattern" in device_candidate.reason_codes


def test_unsupported_tables_emit_finding_and_packet_but_no_row_level_candidates() -> None:
    candidate = _first_candidate(
        [
            {
                "type": "table",
                "table_caption": "Table S1. OLED EQE values.",
                "table_body": (
                    "<table><tr><th rowspan=\"2\">Emitter</th><th>EQE</th></tr>"
                    "<tr><td>22.1</td></tr></table>"
                ),
            }
        ],
        paper_id="paper-complex",
    )
    assert candidate.table_parse_status == OledMineruTableParseStatus.COMPLEX_UNSUPPORTED

    report = map_oled_mineru_candidates_to_schema_candidates([candidate])

    assert report.schema_candidates == []
    assert len(report.packets) == 1
    assert report.packets[0].table_rows == []
    assert "unsupported_table_structure" in report.warning_codes


def test_validation_checks_intermediate_candidate_requirements() -> None:
    source_ref = OledSchemaEvidenceRef(
        source_candidate_hash="abc123",
        source_evidence_anchor="paper:p1:b2:table",
        source_candidate_type="table",
        row_index=0,
        column_name="EQE (%)",
        cell_value="22.1",
    )
    valid_property = OledSchemaCandidate(
        candidate_id="schema:abc123:row-0:eqe_percent",
        candidate_type=OledSchemaCandidateType.PROPERTY_OBSERVATION,
        source_paper_id="paper-validation",
        source_candidate_hash="abc123",
        source_evidence_anchor="paper:p1:b2:table",
        target_layer=OledCausalLayer.MEASUREMENT,
        property_id="eqe_percent",
        property_label="External quantum efficiency",
        value=22.1,
        unit="%",
        evidence_refs=[source_ref],
    )
    valid_role = OledSchemaCandidate(
        candidate_id="schema:abc123:row-0:material-host",
        candidate_type=OledSchemaCandidateType.MATERIAL_ROLE,
        source_paper_id="paper-validation",
        source_candidate_hash="abc123",
        source_evidence_anchor="paper:p1:b2:table",
        material_role="host",
        material_name="mCBP",
        evidence_refs=[
            source_ref.model_copy(update={"column_name": "Host", "cell_value": "mCBP"}),
        ],
    )
    missing_evidence = valid_property.model_copy(
        update={
            "candidate_id": "schema:bad:missing-evidence",
            "evidence_refs": [],
        }
    )

    valid_report = validate_oled_schema_candidates([valid_property, valid_role])
    invalid_report = validate_oled_schema_candidates([missing_evidence])

    assert valid_report.is_valid is True
    assert invalid_report.is_valid is False
    assert "missing_evidence_ref" in invalid_report.error_codes


def test_llm_packet_mode_returns_packets_without_schema_candidates_or_llm_calls() -> None:
    candidate = _first_candidate(
        [{"type": "text", "text": "OLED EQE values are summarized in the next table.", "page_idx": 1}],
        paper_id="paper-llm-packet",
    )

    report = map_oled_mineru_candidates_to_schema_candidates(
        [candidate],
        mapper_kind=OledSemanticMapperKind.LLM_PACKET,
    )

    assert report.mapper_kind == OledSemanticMapperKind.LLM_PACKET
    assert report.schema_candidates == []
    assert len(report.packets) == 1
    assert report.metadata["llm_called"] is False


def test_public_semantic_mapping_api_is_exported_from_domain_package() -> None:
    candidate = _first_candidate(
        [{"type": "text", "text": "OLED EQE values are summarized in the next table.", "page_idx": 1}],
        paper_id="paper-package",
    )

    packets = package_build_oled_semantic_mapping_packets([candidate])
    report = package_map_oled_mineru_candidates_to_schema_candidates([candidate])
    validation_report = package_validate_oled_schema_candidates(report.schema_candidates)

    assert OledSemanticMapperKind.RULE_BASED.value == "rule_based"
    assert OledSchemaCandidateType.PROPERTY_OBSERVATION.value == "property_observation"
    assert OledSchemaCandidateStatus.PROPOSED.value == "proposed"
    assert isinstance(packets[0], OledSemanticMappingPacket)
    assert isinstance(report, OledSemanticMappingReport)
    assert isinstance(validation_report, OledSemanticMappingReport)
    assert isinstance(OledSemanticMappingFinding(code="x", message="y"), OledSemanticMappingFinding)
