from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledAdjudicatedReviewPacket,
    OledAppliedReviewCorrection,
    OledCorrectionApplicationStatus,
    OledMineruReviewPacket,
    OledReviewedExtractionCandidate,
    OledReviewedExtractionFinding,
    OledReviewedExtractionStagingReport,
    OledReviewedExtractionStatus,
    OledReviewAdjudicationReport,
    OledReviewCorrection,
    OledReviewCorrectionType,
    OledReviewDecision,
    OledReviewDecisionEntry,
    OledReviewPacketMaterialRole,
    OledReviewPacketProperty,
    OledReviewPacketSourceRef,
    apply_oled_review_corrections_to_packet as package_apply_oled_review_corrections_to_packet,
    load_oled_review_adjudication_report_json as package_load_oled_review_adjudication_report_json,
    stage_oled_reviewed_extraction_candidates as package_stage_oled_reviewed_extraction_candidates,
    write_oled_reviewed_extraction_candidates_jsonl as package_write_oled_reviewed_extraction_candidates_jsonl,
    write_oled_reviewed_extraction_staging_report_json as package_write_oled_reviewed_extraction_staging_report_json,
)
from ai4s_agent.domains.oled_reviewed_extraction_candidates import (
    apply_oled_review_corrections_to_packet,
    load_oled_review_adjudication_report_json,
    main,
    stage_oled_reviewed_extraction_candidates,
    write_oled_reviewed_extraction_candidates_jsonl,
    write_oled_reviewed_extraction_staging_report_json,
)
from ai4s_agent.domains.oled_reviewed_gold_candidates import (
    convert_reviewed_extractions_to_gold_candidates,
)


def _packet(
    packet_id: str,
    *,
    decision: OledReviewDecision = OledReviewDecision.UNREVIEWED,
    notes: str | None = None,
) -> OledMineruReviewPacket:
    return OledMineruReviewPacket(
        packet_id=packet_id,
        paper_id="paper-stage",
        source_label="synthetic",
        compiled_record_id=packet_id.replace("review:", ""),
        compiled_status="partial",
        source_candidate_hashes=["hash-table"],
        source_evidence_anchors=["paper:p2:b0:table"],
        material_roles=[
            OledReviewPacketMaterialRole(
                role="host",
                material_name="mCBP",
                evidence_refs=[
                    OledReviewPacketSourceRef(
                        source_candidate_hash="hash-table",
                        source_evidence_anchor="paper:p2:b0:table",
                        source_candidate_type="table",
                        row_index=0,
                        column_name="Host",
                    )
                ],
            )
        ],
        properties=[
            OledReviewPacketProperty(
                layer="interaction",
                property_id="plqy",
                property_label="PLQY",
                value=82,
                unit="%",
                evidence_refs=[
                    OledReviewPacketSourceRef(
                        source_candidate_hash="hash-table",
                        source_evidence_anchor="paper:p2:b0:table",
                        source_candidate_type="table",
                        row_index=0,
                        column_name="PLQY (%)",
                    )
                ],
                confidence_score=0.7,
            )
        ],
        device_stack=["ITO", "HTL", "EML", "LiF", "Al"],
        review_decision=decision,
        reviewer_notes=notes,
        metadata={
            "review_packet_only": True,
            "gold_record_created": False,
            "curated_dataset_written": False,
        },
    )


def _adjudicated(
    status: str,
    packet_id: str,
    *,
    decision: OledReviewDecision,
    corrections: list[OledReviewCorrection] | None = None,
    notes: str | None = None,
    finding_codes: list[str] | None = None,
) -> OledAdjudicatedReviewPacket:
    packet = _packet(packet_id, decision=decision, notes=notes)
    return OledAdjudicatedReviewPacket(
        packet=packet,
        decision_entry=OledReviewDecisionEntry(
            packet_id=packet_id,
            review_decision=decision,
            reviewer_notes=notes,
            reviewer_id="reviewer-1",
            corrections=corrections or [],
        ),
        adjudication_status=status,  # type: ignore[arg-type]
        finding_codes=finding_codes or [],
        metadata={
            "adjudication_gate_only": True,
            "gold_record_created": False,
            "curated_dataset_written": False,
        },
    )


def _adjudication_report(adjudicated_packets: list[OledAdjudicatedReviewPacket]) -> OledReviewAdjudicationReport:
    counts: dict[str, int] = {}
    for packet in adjudicated_packets:
        counts[packet.adjudication_status] = counts.get(packet.adjudication_status, 0) + 1
    return OledReviewAdjudicationReport(
        review_manifest_id="round-stage",
        packet_count=len(adjudicated_packets),
        accepted_count=counts.get("accepted", 0),
        rejected_count=counts.get("rejected", 0),
        needs_correction_count=counts.get("needs_correction", 0),
        needs_source_check_count=counts.get("needs_source_check", 0),
        unreviewed_count=counts.get("unreviewed", 0),
        invalid_count=counts.get("invalid", 0),
        adjudicated_packets=adjudicated_packets,
        metadata={
            "adjudication_gate_only": True,
            "gold_records_created": False,
            "curated_dataset_written": False,
        },
    )


def _write_report(path: Path, report: OledReviewAdjudicationReport) -> Path:
    path.write_text(json.dumps(report.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    return path


def test_accepted_packet_stages_reviewed_extraction_candidate() -> None:
    report = stage_oled_reviewed_extraction_candidates(
        _adjudication_report(
            [
                _adjudicated(
                    "accepted",
                    "review:compiled-oled:accepted",
                    decision=OledReviewDecision.ACCEPT,
                    notes="Checked against table.",
                )
            ]
        )
    )

    assert report.is_valid is True
    assert report.staged_candidate_count == 1
    candidate = report.reviewed_candidates[0]
    assert candidate.status == OledReviewedExtractionStatus.ACCEPTED
    assert candidate.review_decision == OledReviewDecision.ACCEPT
    assert candidate.corrected_packet_snapshot is None
    assert candidate.property_count == 1
    assert candidate.material_role_count == 1
    assert candidate.device_stack_count == 5
    assert candidate.metadata["gold_records_created"] is False
    assert candidate.metadata["curated_dataset_written"] is False
    assert report.metadata["reviewed_extraction_candidate_only"] is True


def test_rejected_packets_are_excluded_by_default_and_warn() -> None:
    report = stage_oled_reviewed_extraction_candidates(
        _adjudication_report(
            [
                _adjudicated(
                    "rejected",
                    "review:compiled-oled:rejected",
                    decision=OledReviewDecision.REJECT,
                    notes="Wrong source row.",
                )
            ]
        )
    )

    assert report.staged_candidate_count == 0
    assert "rejected_packet_not_staged" in report.warning_codes


def test_rejected_packets_can_be_included() -> None:
    report = stage_oled_reviewed_extraction_candidates(
        _adjudication_report(
            [_adjudicated("rejected", "review:compiled-oled:rejected", decision=OledReviewDecision.REJECT)]
        ),
        include_rejected=True,
    )

    assert report.staged_candidate_count == 1
    assert report.reviewed_candidates[0].status == OledReviewedExtractionStatus.REJECTED


def test_needs_source_check_is_excluded_or_included_by_flag() -> None:
    adjudication_report = _adjudication_report(
        [
            _adjudicated(
                "needs_source_check",
                "review:compiled-oled:source-check",
                decision=OledReviewDecision.NEEDS_SOURCE_CHECK,
                notes="Needs SI check.",
            )
        ]
    )

    excluded = stage_oled_reviewed_extraction_candidates(adjudication_report)
    included = stage_oled_reviewed_extraction_candidates(adjudication_report, include_needs_source_check=True)

    assert excluded.staged_candidate_count == 0
    assert "needs_source_check_not_staged" in excluded.warning_codes
    assert included.staged_candidate_count == 1
    assert included.reviewed_candidates[0].status == OledReviewedExtractionStatus.NEEDS_SOURCE_CHECK


def test_apply_corrections_to_property_value_and_unit_without_mutating_original_packet() -> None:
    packet = _packet("review:compiled-oled:correction")

    corrected_packet, applied, findings = apply_oled_review_corrections_to_packet(
        packet,
        [
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.PROPERTY_VALUE,
                field_path="properties[0].value",
                original_value=82,
                proposed_value=0.82,
            ),
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.PROPERTY_UNIT,
                field_path="properties[0].unit",
                original_value="%",
                proposed_value="fraction",
            ),
        ],
    )

    assert [item.application_status for item in applied] == [
        OledCorrectionApplicationStatus.APPLIED,
        OledCorrectionApplicationStatus.APPLIED,
    ]
    assert corrected_packet.properties[0].value == 0.82
    assert corrected_packet.properties[0].unit == "fraction"
    assert packet.properties[0].value == 82
    assert packet.properties[0].unit == "%"
    assert findings == []


def test_numeric_correction_requires_replacement_reported_lexeme_when_packet_has_precision() -> None:
    packet = _packet("review:compiled-oled:reported-value-required")
    packet.properties[0] = packet.properties[0].model_copy(
        update={
            "value": 0.03,
            "unit": "fraction",
            "reported_value_text": "0.030",
            "reported_decimal_places": 3,
        }
    )

    corrected_packet, applied, findings = apply_oled_review_corrections_to_packet(
        packet,
        [
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.PROPERTY_VALUE,
                field_path="properties[0].value",
                original_value=0.03,
                proposed_value=0.04,
            )
        ],
    )

    assert corrected_packet.properties[0].value == 0.03
    assert corrected_packet.properties[0].reported_value_text == "0.030"
    assert corrected_packet.properties[0].reported_decimal_places == 3
    assert applied[0].application_status == OledCorrectionApplicationStatus.FAILED
    assert applied[0].finding_codes == ["numeric_correction_reported_value_fields_required"]
    assert [finding.code for finding in findings] == ["numeric_correction_reported_value_fields_required"]


def test_numeric_correction_rejects_inconsistent_replacement_reported_lexeme() -> None:
    packet = _packet("review:compiled-oled:reported-value-inconsistent")
    packet.properties[0] = packet.properties[0].model_copy(
        update={
            "value": 0.03,
            "unit": "fraction",
            "reported_value_text": "0.030",
            "reported_decimal_places": 3,
        }
    )

    corrected_packet, applied, findings = apply_oled_review_corrections_to_packet(
        packet,
        [
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.PROPERTY_VALUE,
                field_path="properties[0].value",
                original_value=0.03,
                proposed_value=0.04,
                proposed_reported_value_text="0.030",
                proposed_reported_decimal_places=3,
            )
        ],
    )

    assert corrected_packet.properties[0].value == 0.03
    assert corrected_packet.properties[0].reported_value_text == "0.030"
    assert applied[0].application_status == OledCorrectionApplicationStatus.FAILED
    assert applied[0].finding_codes == ["numeric_correction_reported_value_contract_invalid"]
    assert [finding.code for finding in findings] == ["numeric_correction_reported_value_contract_invalid"]


def test_trailing_zero_numeric_correction_survives_review_staging_and_gold_conversion() -> None:
    adjudicated = _adjudicated(
        "needs_correction",
        "review:compiled-oled:trailing-zero",
        decision=OledReviewDecision.NEEDS_CORRECTION,
        notes="Corrected against the source table; preserve its trailing zero.",
        corrections=[
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.PROPERTY_VALUE,
                field_path="properties[0].value",
                original_value=0.03,
                proposed_value=0.04,
                proposed_reported_value_text="0.040",
                proposed_reported_decimal_places=3,
                reason="The corrected source lexeme is 0.040.",
            )
        ],
    )
    adjudicated.packet.properties[0] = adjudicated.packet.properties[0].model_copy(
        update={
            "value": 0.03,
            "unit": "fraction",
            "reported_value_text": "0.030",
            "reported_decimal_places": 3,
        }
    )
    staging = stage_oled_reviewed_extraction_candidates(_adjudication_report([adjudicated]))

    assert staging.is_valid is True
    assert staging.staged_candidate_count == 1
    reviewed = staging.reviewed_candidates[0]
    assert reviewed.status == OledReviewedExtractionStatus.CORRECTED
    assert reviewed.corrected_packet_snapshot is not None
    corrected_property = reviewed.corrected_packet_snapshot.properties[0]
    assert corrected_property.value == 0.04
    assert corrected_property.reported_value_text == "0.040"
    assert corrected_property.reported_decimal_places == 3
    assert reviewed.applied_corrections[0].proposed_reported_value_text == "0.040"
    assert reviewed.applied_corrections[0].proposed_reported_decimal_places == 3
    assert adjudicated.packet.properties[0].reported_value_text == "0.030"

    conversion = convert_reviewed_extractions_to_gold_candidates([reviewed])

    assert conversion.converted_candidate_count == 1
    gold_record = conversion.gold_candidates[0].gold_record
    assert gold_record is not None
    assert gold_record.layered_record.interaction is not None
    observation = gold_record.layered_record.interaction.properties[0]
    assert observation.value == 0.04
    assert observation.reported_value_text == "0.040"
    assert observation.reported_decimal_places == 3


def test_original_value_mismatch_warns_but_still_applies_proposed_value() -> None:
    packet = _packet("review:compiled-oled:mismatch")

    corrected_packet, applied, findings = apply_oled_review_corrections_to_packet(
        packet,
        [
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.PROPERTY_VALUE,
                field_path="properties[0].value",
                original_value=81,
                proposed_value=83,
            )
        ],
    )

    assert corrected_packet.properties[0].value == 83
    assert applied[0].application_status == OledCorrectionApplicationStatus.APPLIED
    assert "correction_original_value_mismatch" in [finding.code for finding in findings]


def test_unsupported_correction_path_is_not_applied() -> None:
    corrected_packet, applied, findings = apply_oled_review_corrections_to_packet(
        _packet("review:compiled-oled:unsupported"),
        [
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.EVIDENCE_REF,
                field_path="source_evidence_anchors[0]",
                proposed_value="new-anchor",
            )
        ],
    )

    assert corrected_packet.source_evidence_anchors == ["paper:p2:b0:table"]
    assert applied[0].application_status == OledCorrectionApplicationStatus.FAILED
    assert "correction_field_path_unsupported" in [finding.code for finding in findings]


def test_device_stack_corrections_replace_whole_stack_and_single_index() -> None:
    packet = _packet("review:compiled-oled:device-stack")

    corrected_packet, applied, findings = apply_oled_review_corrections_to_packet(
        packet,
        [
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.DEVICE_STACK,
                field_path="device_stack",
                original_value=["ITO", "HTL", "EML", "LiF", "Al"],
                proposed_value=["ITO", "TAPC", "EML", "TmPyPB", "LiF", "Al"],
            ),
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.DEVICE_STACK,
                field_path="device_stack[2]",
                original_value="EML",
                proposed_value="EML:D1",
            ),
        ],
    )

    assert corrected_packet.device_stack == ["ITO", "TAPC", "EML:D1", "TmPyPB", "LiF", "Al"]
    assert all(item.application_status == OledCorrectionApplicationStatus.APPLIED for item in applied)
    assert findings == []


def test_needs_correction_stages_corrected_candidate_when_corrections_apply() -> None:
    report = stage_oled_reviewed_extraction_candidates(
        _adjudication_report(
            [
                _adjudicated(
                    "needs_correction",
                    "review:compiled-oled:needs-correction",
                    decision=OledReviewDecision.NEEDS_CORRECTION,
                    notes="Unit should be fraction.",
                    corrections=[
                        OledReviewCorrection(
                            correction_type=OledReviewCorrectionType.PROPERTY_UNIT,
                            field_path="properties[0].unit",
                            original_value="%",
                            proposed_value="fraction",
                        )
                    ],
                )
            ]
        )
    )

    assert report.staged_candidate_count == 1
    candidate = report.reviewed_candidates[0]
    assert candidate.status == OledReviewedExtractionStatus.CORRECTED
    assert candidate.corrected_packet_snapshot is not None
    assert candidate.corrected_packet_snapshot.properties[0].unit == "fraction"
    assert candidate.applied_corrections[0].application_status == OledCorrectionApplicationStatus.APPLIED


def test_load_adjudication_report_json_handles_valid_missing_and_invalid_json(tmp_path: Path) -> None:
    report = _adjudication_report([_adjudicated("accepted", "review:compiled-oled:load", decision=OledReviewDecision.ACCEPT)])
    path = _write_report(tmp_path / "adjudication.json", report)

    loaded = load_oled_review_adjudication_report_json(path)

    assert loaded.review_manifest_id == "round-stage"
    assert loaded.packet_count == 1

    with pytest.raises(ValueError, match="missing_adjudication_report:"):
        load_oled_review_adjudication_report_json(tmp_path / "missing.json")

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_adjudication_report_json:"):
        load_oled_review_adjudication_report_json(invalid_path)


def test_jsonl_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    staging_report = stage_oled_reviewed_extraction_candidates(
        _adjudication_report([_adjudicated("accepted", "review:compiled-oled:writer", decision=OledReviewDecision.ACCEPT)])
    )
    candidate = staging_report.reviewed_candidates[0].model_copy(
        update={
            "metadata": {
                **staging_report.reviewed_candidates[0].metadata,
                "source_path": str(tmp_path / "paper.json"),
                "raw_text": "mCBP | D1 | 82",
            }
        }
    )
    output_path = tmp_path / "reviewed_candidates.jsonl"

    write_oled_reviewed_extraction_candidates_jsonl([candidate], output_path)
    text = output_path.read_text(encoding="utf-8")
    payload = json.loads(text.splitlines()[0])

    assert text.splitlines()[0] == json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert str(tmp_path) not in text
    assert "mCBP | D1 | 82" not in text
    assert payload["metadata"]["gold_records_created"] is False


def test_cli_smoke_writes_candidates_and_report(tmp_path: Path) -> None:
    adjudication_path = _write_report(
        tmp_path / "adjudication.json",
        _adjudication_report([_adjudicated("accepted", "review:compiled-oled:cli", decision=OledReviewDecision.ACCEPT)]),
    )
    output_candidates = tmp_path / "reviewed_candidates.jsonl"
    output_report = tmp_path / "staging_report.json"

    exit_code = main(
        [
            "--adjudication-report",
            str(adjudication_path),
            "--output-candidates",
            str(output_candidates),
            "--output-report",
            str(output_report),
        ]
    )

    assert exit_code == 0
    assert output_candidates.exists()
    assert output_report.exists()
    assert json.loads(output_report.read_text(encoding="utf-8"))["staged_candidate_count"] == 1


def test_public_reviewed_extraction_api_is_exported_from_domain_package(tmp_path: Path) -> None:
    adjudication_report = _adjudication_report(
        [_adjudicated("accepted", "review:compiled-oled:package", decision=OledReviewDecision.ACCEPT)]
    )
    report = package_stage_oled_reviewed_extraction_candidates(adjudication_report)
    corrected, applied, findings = package_apply_oled_review_corrections_to_packet(
        _packet("review:compiled-oled:package-correction"),
        [
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.NOTES_ONLY,
                field_path="reviewer_notes",
                proposed_value="Checked.",
            )
        ],
    )
    adjudication_path = _write_report(tmp_path / "package-adjudication.json", adjudication_report)
    loaded = package_load_oled_review_adjudication_report_json(adjudication_path)
    candidates_path = tmp_path / "package-candidates.jsonl"
    report_path = tmp_path / "package-report.json"
    package_write_oled_reviewed_extraction_candidates_jsonl(report.reviewed_candidates, candidates_path)
    package_write_oled_reviewed_extraction_staging_report_json(report, report_path)

    assert isinstance(report, OledReviewedExtractionStagingReport)
    assert isinstance(report.reviewed_candidates[0], OledReviewedExtractionCandidate)
    assert isinstance(OledReviewedExtractionFinding(code="x", message="y"), OledReviewedExtractionFinding)
    assert isinstance(applied[0], OledAppliedReviewCorrection)
    assert isinstance(loaded, OledReviewAdjudicationReport)
    assert corrected.reviewer_notes == "Checked."
    assert findings == []
    assert OledReviewedExtractionStatus.ACCEPTED.value == "accepted"
    assert OledCorrectionApplicationStatus.APPLIED.value == "applied"
    assert candidates_path.exists()
    assert report_path.exists()
