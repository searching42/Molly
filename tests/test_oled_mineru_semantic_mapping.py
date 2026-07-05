from __future__ import annotations

from ai4s_agent.domains import (
    OledCausalLayer,
    OledMineruCandidateType,
    OledSemanticMapperKind,
    OledSemanticMappingPacket,
    OledSchemaCandidate,
    OledSchemaCandidateEvidenceRef,
    OledSchemaCandidateSummary,
    OledSchemaCandidateType,
    OledSchemaCandidateValidationReport,
    build_oled_semantic_mapping_packets as package_build_oled_semantic_mapping_packets,
    map_oled_mineru_table_candidates as package_map_oled_mineru_table_candidates,
    summarize_oled_schema_candidates as package_summarize_oled_schema_candidates,
    validate_oled_schema_candidates as package_validate_oled_schema_candidates,
)
from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruTableParseStatus,
    extract_oled_mineru_candidates_from_document,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    build_oled_semantic_mapping_packets,
    map_oled_mineru_table_candidates,
    summarize_oled_schema_candidates,
    validate_oled_schema_candidates,
)


def _table_candidate():
    return extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "table",
                "table_caption": "Table 1. OLED device performance for doped films.",
                "table_body": (
                    "<table><tr><th>Emitter</th><th>Host</th><th>EQE (%)</th>"
                    "<th>HOMO level (eV)</th><th>PLQY</th></tr>"
                    "<tr><td>D1</td><td>mCBP</td><td>22.1</td><td>-5.4</td><td>0.80</td></tr></table>"
                ),
                "page_idx": 4,
            }
        ],
        paper_id="paper-semantic",
        source_path="/absolute/local/path/paper-semantic_content_list.json",
    )[0]


def test_rule_based_table_mapper_emits_property_and_role_schema_candidates() -> None:
    mineru_candidate = _table_candidate()

    schema_candidates = map_oled_mineru_table_candidates([mineru_candidate])

    property_candidates = [
        candidate
        for candidate in schema_candidates
        if candidate.candidate_type == OledSchemaCandidateType.PROPERTY_VALUE
    ]
    role_candidates = [
        candidate for candidate in schema_candidates if candidate.candidate_type == OledSchemaCandidateType.ENTITY_ROLE
    ]

    assert {candidate.property_id for candidate in property_candidates} == {
        "eqe_percent",
        "homo_ev",
        "plqy",
    }
    assert {(candidate.role, candidate.entity_label) for candidate in role_candidates} == {
        ("emitter", "D1"),
        ("host", "mCBP"),
    }

    eqe = next(candidate for candidate in property_candidates if candidate.property_id == "eqe_percent")
    assert eqe.mapper_kind == OledSemanticMapperKind.RULE_BASED
    assert eqe.target_layer == OledCausalLayer.MEASUREMENT
    assert eqe.property_label == "External quantum efficiency"
    assert eqe.raw_value == "22.1"
    assert eqe.value == 22.1
    assert eqe.unit == "%"
    assert eqe.source_candidate_hash == mineru_candidate.candidate_hash
    assert eqe.evidence_anchor == mineru_candidate.evidence_anchor
    assert eqe.evidence_refs == [
        OledSchemaCandidateEvidenceRef(
            source_candidate_hash=mineru_candidate.candidate_hash,
            evidence_anchor=mineru_candidate.evidence_anchor,
            paper_id="paper-semantic",
            source_candidate_type=OledMineruCandidateType.TABLE,
            row_index=0,
            column_name="EQE (%)",
            cell_value="22.1",
        )
    ]


def test_rule_based_mapper_skips_non_parsed_or_non_table_evidence_by_default() -> None:
    candidates = extract_oled_mineru_candidates_from_document(
        [
            {"type": "text", "text": "OLED EQE was 22%.", "page_idx": 1},
            {
                "type": "table",
                "table_caption": "Table S1. OLED EQE.",
                "table_body": "<table><tr><th rowspan=\"2\">Emitter</th><th>EQE</th></tr><tr><td>22</td></tr></table>",
            },
        ],
        paper_id="paper-skip",
    )

    assert [candidate.table_parse_status for candidate in candidates] == [
        OledMineruTableParseStatus.NOT_TABLE,
        OledMineruTableParseStatus.COMPLEX_UNSUPPORTED,
    ]
    assert map_oled_mineru_table_candidates(candidates) == []


def test_llm_ready_packets_retain_source_evidence_without_calling_llm() -> None:
    mineru_candidate = _table_candidate()

    packets = build_oled_semantic_mapping_packets([mineru_candidate])

    assert len(packets) == 1
    packet = packets[0]
    assert packet.mapper_kind == OledSemanticMapperKind.LLM_PACKET
    assert packet.paper_id == "paper-semantic"
    assert packet.source_candidate_hash == mineru_candidate.candidate_hash
    assert packet.evidence_anchor == mineru_candidate.evidence_anchor
    assert packet.candidate_type == OledMineruCandidateType.TABLE
    assert packet.table_headers == ["Emitter", "Host", "EQE (%)", "HOMO level (eV)", "PLQY"]
    assert packet.table_rows[0]["EQE (%)"] == "22.1"
    assert packet.proposed_schema_candidates == []
    assert "Do not create final OledLayeredRecord objects" in packet.instructions


def test_schema_candidate_hashes_and_ids_are_deterministic_and_path_independent() -> None:
    first = _table_candidate()
    second = extract_oled_mineru_candidates_from_document(
        [
            {
                "type": "table",
                "table_caption": "Table 1. OLED device performance for doped films.",
                "table_body": (
                    "<table><tr><th>Emitter</th><th>Host</th><th>EQE (%)</th>"
                    "<th>HOMO level (eV)</th><th>PLQY</th></tr>"
                    "<tr><td>D1</td><td>mCBP</td><td>22.1</td><td>-5.4</td><td>0.80</td></tr></table>"
                ),
                "page_idx": 4,
            }
        ],
        paper_id="paper-semantic",
        source_path="/other/machine/path/paper-semantic_content_list.json",
    )[0]

    first_candidates = map_oled_mineru_table_candidates([first])
    second_candidates = map_oled_mineru_table_candidates([second])

    assert [candidate.candidate_id for candidate in first_candidates] == [
        candidate.candidate_id for candidate in second_candidates
    ]
    assert [candidate.candidate_hash for candidate in first_candidates] == [
        candidate.candidate_hash for candidate in second_candidates
    ]


def test_validation_reports_missing_evidence_and_unknown_property_ids() -> None:
    mineru_candidate = _table_candidate()
    valid_candidate = next(
        candidate
        for candidate in map_oled_mineru_table_candidates([mineru_candidate])
        if candidate.property_id == "eqe_percent"
    )
    invalid_candidate = valid_candidate.model_copy(
        update={
            "candidate_id": "schema:manual:bad",
            "property_id": "unknown_property",
            "evidence_refs": [],
        }
    )

    report = validate_oled_schema_candidates([valid_candidate, invalid_candidate])

    assert isinstance(report, OledSchemaCandidateValidationReport)
    assert report.is_valid is False
    assert "missing_evidence_ref" in report.error_codes
    assert "unknown_property_id" in report.error_codes
    assert validate_oled_schema_candidates([valid_candidate]).is_valid is True


def test_summary_counts_schema_candidates_by_type_layer_mapper_and_property() -> None:
    schema_candidates = map_oled_mineru_table_candidates([_table_candidate()])

    summary = summarize_oled_schema_candidates(schema_candidates)

    assert summary.total_candidates == 5
    assert summary.candidates_by_mapper[OledSemanticMapperKind.RULE_BASED] == 5
    assert summary.candidates_by_type[OledSchemaCandidateType.PROPERTY_VALUE] == 3
    assert summary.candidates_by_type[OledSchemaCandidateType.ENTITY_ROLE] == 2
    assert summary.candidates_by_layer[OledCausalLayer.MEASUREMENT] == 2
    assert summary.candidates_by_layer[OledCausalLayer.MOLECULE] == 1
    assert summary.candidates_by_property["eqe_percent"] == 1
    assert summary.evidence_candidate_count == 1


def test_public_semantic_mapping_api_is_exported_from_domain_package() -> None:
    mineru_candidate = _table_candidate()
    schema_candidates = package_map_oled_mineru_table_candidates([mineru_candidate])
    packets = package_build_oled_semantic_mapping_packets([mineru_candidate])
    summary = package_summarize_oled_schema_candidates(schema_candidates)
    report = package_validate_oled_schema_candidates(schema_candidates)

    assert isinstance(schema_candidates[0], OledSchemaCandidate)
    assert isinstance(packets[0], OledSemanticMappingPacket)
    assert isinstance(summary, OledSchemaCandidateSummary)
    assert isinstance(report, OledSchemaCandidateValidationReport)
    assert report.is_valid is True
