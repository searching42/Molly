from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledAppliedReviewCorrection,
    OledCorrectionApplicationStatus,
    OledGoldDatasetRecord,
    OledMineruReviewPacket,
    OledReviewedExtractionCandidate,
    OledReviewedExtractionStatus,
    OledReviewedGoldCandidate,
    OledReviewedGoldCandidateStatus,
    OledReviewedGoldConversionFinding,
    OledReviewedGoldConversionPolicy,
    OledReviewedGoldConversionReport,
    OledReviewCorrectionType,
    OledReviewDecision,
    OledReviewPacketMaterialRole,
    OledReviewPacketProperty,
    OledReviewPacketSourceRef,
    build_gold_record_candidate_from_reviewed_extraction as package_build_gold_record_candidate_from_reviewed_extraction,
    convert_reviewed_extractions_to_gold_candidates as package_convert_reviewed_extractions_to_gold_candidates,
    effective_reviewed_extraction_packet as package_effective_reviewed_extraction_packet,
    load_oled_reviewed_extraction_candidates_jsonl as package_load_oled_reviewed_extraction_candidates_jsonl,
    write_oled_reviewed_gold_candidates_jsonl as package_write_oled_reviewed_gold_candidates_jsonl,
    write_oled_reviewed_gold_conversion_report_json as package_write_oled_reviewed_gold_conversion_report_json,
)
from ai4s_agent.domains.oled_reviewed_gold_candidates import (
    build_gold_record_candidate_from_reviewed_extraction,
    convert_reviewed_extractions_to_gold_candidates,
    effective_reviewed_extraction_packet,
    load_oled_reviewed_extraction_candidates_jsonl,
    main,
    write_oled_reviewed_gold_candidates_jsonl,
    write_oled_reviewed_gold_conversion_report_json,
)


def _packet(
    packet_id: str,
    *,
    property_label: str = "PLQY",
    property_id: str | None = "plqy",
    value: float | int | str | None = 82,
    unit: str | None = "%",
    reported_value_text: str | None = "82",
    reported_decimal_places: int | None = 0,
    layer: str = "interaction",
    anchors: list[str] | None = None,
) -> OledMineruReviewPacket:
    evidence_anchors = anchors if anchors is not None else ["paper:p2:b0:table"]
    return OledMineruReviewPacket(
        packet_id=packet_id,
        paper_id="paper-gold",
        source_label="synthetic",
        compiled_record_id=packet_id.replace("review:", ""),
        compiled_status="partial",
        source_candidate_hashes=["hash-table"],
        source_evidence_anchors=evidence_anchors,
        material_roles=[
            OledReviewPacketMaterialRole(
                role="host",
                material_name="mCBP",
                evidence_refs=[
                    OledReviewPacketSourceRef(
                        source_candidate_hash="hash-table",
                        source_evidence_anchor=evidence_anchors[0] if evidence_anchors else "paper:p2:b0:table",
                        source_candidate_type="table",
                        row_index=0,
                        column_name="Host",
                    )
                ],
            ),
            OledReviewPacketMaterialRole(
                role="emitter_dopant",
                material_name="D1",
                evidence_refs=[
                    OledReviewPacketSourceRef(
                        source_candidate_hash="hash-table",
                        source_evidence_anchor=evidence_anchors[0] if evidence_anchors else "paper:p2:b0:table",
                        source_candidate_type="table",
                        row_index=0,
                        column_name="Emitter dopant",
                    )
                ],
            ),
        ],
        properties=[
            OledReviewPacketProperty(
                layer=layer,
                property_id=property_id,
                property_label=property_label,
                value=value,
                unit=unit,
                reported_value_text=reported_value_text,
                reported_decimal_places=reported_decimal_places,
                evidence_refs=[
                    OledReviewPacketSourceRef(
                        source_candidate_hash="hash-table",
                        source_evidence_anchor=evidence_anchors[0] if evidence_anchors else "paper:p2:b0:table",
                        source_candidate_type="table",
                        row_index=0,
                        column_name="PLQY (%)",
                    )
                ],
                confidence_score=0.8,
            )
        ],
        device_stack=["ITO", "HTL", "EML", "ETL", "Al"],
        review_decision=OledReviewDecision.ACCEPT,
        reviewer_notes="Checked.",
        metadata={
            "review_packet_only": True,
            "gold_record_created": False,
            "curated_dataset_written": False,
        },
    )


def _reviewed_candidate(
    candidate_id: str,
    *,
    status: OledReviewedExtractionStatus = OledReviewedExtractionStatus.ACCEPTED,
    packet: OledMineruReviewPacket | None = None,
    corrected_packet: OledMineruReviewPacket | None = None,
    schema_errors: list[str] | None = None,
    adjudication_errors: list[str] | None = None,
    anchors: list[str] | None = None,
    applied_corrections: list[OledAppliedReviewCorrection] | None = None,
) -> OledReviewedExtractionCandidate:
    packet_snapshot = packet or _packet(candidate_id.replace("reviewed-extraction:", "review:"))
    evidence_anchors = anchors if anchors is not None else list(packet_snapshot.source_evidence_anchors)
    return OledReviewedExtractionCandidate(
        candidate_id=candidate_id,
        source_packet_id=packet_snapshot.packet_id,
        source_compiled_record_id=packet_snapshot.compiled_record_id,
        paper_id=packet_snapshot.paper_id,
        source_label=packet_snapshot.source_label,
        status=status,
        review_decision=packet_snapshot.review_decision,
        packet_snapshot=packet_snapshot,
        corrected_packet_snapshot=corrected_packet,
        applied_corrections=applied_corrections or [],
        source_candidate_hashes=packet_snapshot.source_candidate_hashes,
        source_evidence_anchors=evidence_anchors,
        property_count=len((corrected_packet or packet_snapshot).properties),
        material_role_count=len((corrected_packet or packet_snapshot).material_roles),
        device_stack_count=len((corrected_packet or packet_snapshot).device_stack),
        schema_error_codes=schema_errors or [],
        schema_warning_codes=[],
        adjudication_finding_codes=adjudication_errors or [],
        reviewer_id="reviewer-1",
        reviewer_notes="Checked.",
        metadata={
            "reviewed_extraction_candidate_only": True,
            "gold_records_created": False,
            "curated_dataset_written": False,
            "training_data_written": False,
        },
    )


def _write_candidates(path: Path, candidates: list[OledReviewedExtractionCandidate]) -> Path:
    path.write_text(
        "\n".join(json.dumps(candidate.model_dump(mode="json"), sort_keys=True) for candidate in candidates) + "\n",
        encoding="utf-8",
    )
    return path


def test_effective_packet_helper_prefers_corrected_snapshot() -> None:
    original = _packet("review:compiled-oled:effective", value=82, unit="%")
    corrected = _packet(
        "review:compiled-oled:effective",
        value=0.82,
        unit="fraction",
        reported_value_text="0.82",
        reported_decimal_places=2,
    )
    candidate = _reviewed_candidate(
        "reviewed-extraction:effective",
        status=OledReviewedExtractionStatus.CORRECTED,
        packet=original,
        corrected_packet=corrected,
    )

    effective = effective_reviewed_extraction_packet(candidate)

    assert effective.properties[0].value == 0.82
    assert effective.properties[0].unit == "fraction"


def test_accepted_candidate_converts_to_gold_candidate_with_provenance() -> None:
    candidate = _reviewed_candidate("reviewed-extraction:accepted")

    report = convert_reviewed_extractions_to_gold_candidates([candidate])

    assert report.converted_candidate_count == 1
    converted = report.gold_candidates[0]
    assert converted.gold_record is not None
    assert converted.source_reviewed_candidate_id == candidate.candidate_id
    assert converted.source_packet_id == candidate.source_packet_id
    assert converted.source_evidence_anchors == ["paper:p2:b0:table"]
    assert converted.metadata["candidate_only"] is True
    assert converted.metadata["curated_dataset_written"] is False
    assert converted.metadata["final_gold_dataset"] is False
    assert converted.gold_record.metadata["candidate_only"] is True
    observation = converted.gold_record.layered_record.interaction.properties[0]
    assert observation.reported_value_text == "82"
    assert observation.reported_decimal_places == 0


def test_corrected_candidate_uses_corrected_packet_values_and_preserves_corrections() -> None:
    corrected = _packet(
        "review:compiled-oled:corrected",
        value=0.82,
        unit="fraction",
        reported_value_text="0.82",
        reported_decimal_places=2,
    )
    correction = OledAppliedReviewCorrection(
        correction_type=OledReviewCorrectionType.PROPERTY_UNIT,
        field_path="properties[0].unit",
        original_value="%",
        proposed_value="fraction",
        application_status=OledCorrectionApplicationStatus.APPLIED,
    )
    candidate = _reviewed_candidate(
        "reviewed-extraction:corrected",
        status=OledReviewedExtractionStatus.CORRECTED,
        packet=_packet("review:compiled-oled:corrected", value=82, unit="%"),
        corrected_packet=corrected,
        applied_corrections=[correction],
    )

    gold_record = build_gold_record_candidate_from_reviewed_extraction(candidate)
    report = convert_reviewed_extractions_to_gold_candidates([candidate])

    assert gold_record.layered_record.interaction is not None
    assert gold_record.layered_record.interaction.properties[0].value == 0.82
    assert gold_record.layered_record.interaction.properties[0].unit == "fraction"
    assert gold_record.layered_record.interaction.properties[0].reported_value_text == "0.82"
    assert report.gold_candidates[0].gold_record is not None
    assert report.gold_candidates[0].gold_record.layered_record.interaction is not None
    assert report.gold_candidates[0].gold_record.metadata["applied_correction_statuses"] == ["applied"]


def test_rejected_and_needs_source_check_are_skipped_by_default() -> None:
    candidates = [
        _reviewed_candidate("reviewed-extraction:rejected", status=OledReviewedExtractionStatus.REJECTED),
        _reviewed_candidate(
            "reviewed-extraction:source-check",
            status=OledReviewedExtractionStatus.NEEDS_SOURCE_CHECK,
        ),
    ]

    report = convert_reviewed_extractions_to_gold_candidates(candidates)

    assert report.converted_candidate_count == 0
    assert report.status_counts == {"skipped": 2}
    assert "status_not_eligible" in report.finding_code_counts


def test_policy_blocks_schema_errors_and_missing_evidence_anchors() -> None:
    schema_error = _reviewed_candidate(
        "reviewed-extraction:schema-error",
        schema_errors=["unknown_property_label"],
    )
    no_anchors = _reviewed_candidate("reviewed-extraction:no-anchors", anchors=[])

    report = convert_reviewed_extractions_to_gold_candidates([schema_error, no_anchors])

    assert report.status_counts == {"rejected": 2}
    assert "source_schema_errors_present" in report.finding_code_counts
    assert "source_evidence_anchors_missing" in report.finding_code_counts
    assert all(candidate.gold_record is None for candidate in report.gold_candidates)


def test_policy_override_can_include_rejected_as_rejected_candidate_only() -> None:
    candidate = _reviewed_candidate("reviewed-extraction:rejected", status=OledReviewedExtractionStatus.REJECTED)

    report = convert_reviewed_extractions_to_gold_candidates(
        [candidate],
        policy=OledReviewedGoldConversionPolicy(include_rejected=True),
    )

    assert report.gold_candidates[0].status == OledReviewedGoldCandidateStatus.REJECTED
    assert report.gold_candidates[0].gold_record is None
    assert report.gold_candidates[0].metadata["candidate_only"] is True


def test_gold_validation_codes_are_attached_to_candidate_and_report() -> None:
    candidate = _reviewed_candidate(
        "reviewed-extraction:invalid-property",
        packet=_packet("review:compiled-oled:invalid-property", property_label="Not a known OLED property", property_id=None),
    )

    report = convert_reviewed_extractions_to_gold_candidates([candidate])

    converted = report.gold_candidates[0]
    assert converted.gold_record is not None
    assert converted.status == OledReviewedGoldCandidateStatus.INVALID
    assert "unknown_property_label" in converted.validation_error_codes
    assert "unknown_property_label" in report.validation_code_counts


def test_material_role_names_do_not_invent_smiles() -> None:
    gold_record = build_gold_record_candidate_from_reviewed_extraction(_reviewed_candidate("reviewed-extraction:smiles"))

    assert gold_record.layered_record.molecule is None or gold_record.layered_record.molecule.canonical_smiles is None
    assert gold_record.layered_record.interaction is not None
    assert gold_record.layered_record.interaction.host_smiles is None
    assert gold_record.layered_record.interaction.emitter_smiles is None
    assert gold_record.layered_record.interaction.metadata["host_name"] == "mCBP"


def test_load_reviewed_candidates_jsonl_handles_valid_empty_invalid_and_missing(tmp_path: Path) -> None:
    candidate = _reviewed_candidate("reviewed-extraction:load")
    path = tmp_path / "reviewed.jsonl"
    path.write_text("\n" + json.dumps(candidate.model_dump(mode="json"), sort_keys=True) + "\n\n", encoding="utf-8")

    loaded = load_oled_reviewed_extraction_candidates_jsonl(path)

    assert len(loaded) == 1
    assert loaded[0].candidate_id == "reviewed-extraction:load"

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps(candidate.model_dump(mode="json"), sort_keys=True) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_reviewed_candidate_jsonl:line-2"):
        load_oled_reviewed_extraction_candidates_jsonl(bad_path)

    with pytest.raises(ValueError, match="missing_reviewed_candidate_jsonl:"):
        load_oled_reviewed_extraction_candidates_jsonl(tmp_path / "missing.jsonl")


def test_writers_are_deterministic_and_redacted(tmp_path: Path) -> None:
    candidate = _reviewed_candidate("reviewed-extraction:writer").model_copy(
        update={
            "metadata": {
                "source_path": str(tmp_path / "paper.json"),
                "raw_text": "mCBP | D1 | 82",
            }
        }
    )
    report = convert_reviewed_extractions_to_gold_candidates([candidate])
    candidates_path = tmp_path / "gold_candidates.jsonl"
    report_path = tmp_path / "gold_report.json"

    write_oled_reviewed_gold_candidates_jsonl(report.gold_candidates, candidates_path)
    write_oled_reviewed_gold_conversion_report_json(report, report_path)
    candidates_text = candidates_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    candidate_payload = json.loads(candidates_text.splitlines()[0])
    report_payload = json.loads(report_text)

    assert candidates_text.splitlines()[0] == json.dumps(candidate_payload, sort_keys=True, separators=(",", ":"))
    assert report_text == json.dumps(report_payload, sort_keys=True, indent=2) + "\n"
    assert str(tmp_path) not in candidates_text
    assert str(tmp_path) not in report_text
    assert "mCBP | D1 | 82" not in candidates_text
    assert "mCBP | D1 | 82" not in report_text


def test_cli_smoke_writes_gold_candidates_and_report(tmp_path: Path) -> None:
    reviewed_path = _write_candidates(tmp_path / "reviewed.jsonl", [_reviewed_candidate("reviewed-extraction:cli")])
    candidates_path = tmp_path / "gold_candidates.jsonl"
    report_path = tmp_path / "gold_report.json"

    exit_code = main(
        [
            "--reviewed-candidates",
            str(reviewed_path),
            "--output-candidates",
            str(candidates_path),
            "--output-report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert candidates_path.exists()
    assert report_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["converted_candidate_count"] == 1


def test_public_reviewed_gold_candidate_api_is_exported_from_domain_package(tmp_path: Path) -> None:
    candidate = _reviewed_candidate("reviewed-extraction:package")
    reviewed_path = _write_candidates(tmp_path / "package-reviewed.jsonl", [candidate])

    loaded = package_load_oled_reviewed_extraction_candidates_jsonl(reviewed_path)
    effective = package_effective_reviewed_extraction_packet(loaded[0])
    gold_record = package_build_gold_record_candidate_from_reviewed_extraction(loaded[0])
    report = package_convert_reviewed_extractions_to_gold_candidates(loaded)
    candidates_path = tmp_path / "package-gold.jsonl"
    report_path = tmp_path / "package-report.json"
    package_write_oled_reviewed_gold_candidates_jsonl(report.gold_candidates, candidates_path)
    package_write_oled_reviewed_gold_conversion_report_json(report, report_path)

    assert isinstance(effective, OledMineruReviewPacket)
    assert isinstance(gold_record, OledGoldDatasetRecord)
    assert isinstance(report, OledReviewedGoldConversionReport)
    assert isinstance(report.gold_candidates[0], OledReviewedGoldCandidate)
    assert isinstance(OledReviewedGoldConversionFinding(code="x", message="y"), OledReviewedGoldConversionFinding)
    assert OledReviewedGoldCandidateStatus.CONVERTED.value == "converted"
    assert OledReviewedGoldConversionPolicy().candidate_only is True
    assert candidates_path.exists()
    assert report_path.exists()
