from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledAdjudicatedReviewPacket,
    OledMineruReviewPacket,
    OledReviewAdjudicationFinding,
    OledReviewAdjudicationReport,
    OledReviewCorrection,
    OledReviewCorrectionType,
    OledReviewDecision,
    OledReviewDecisionEntry,
    OledReviewDecisionManifest,
    OledReviewPacketProperty,
    OledReviewPacketSourceRef,
    adjudicate_oled_mineru_review_packets as package_adjudicate_oled_mineru_review_packets,
    filter_adjudicated_oled_review_packets as package_filter_adjudicated_oled_review_packets,
    load_oled_mineru_review_packets_jsonl as package_load_oled_mineru_review_packets_jsonl,
    load_oled_review_decision_manifest as package_load_oled_review_decision_manifest,
    write_oled_review_adjudication_report_json as package_write_oled_review_adjudication_report_json,
)
from ai4s_agent.domains.oled_mineru_review_adjudication import (
    adjudicate_oled_mineru_review_packets,
    filter_adjudicated_oled_review_packets,
    load_oled_mineru_review_packets_jsonl,
    load_oled_review_decision_manifest,
    main,
    write_oled_review_adjudication_report_json,
)


def _packet(
    packet_id: str,
    decision: OledReviewDecision = OledReviewDecision.UNREVIEWED,
    *,
    schema_errors: list[str] | None = None,
    schema_warnings: list[str] | None = None,
    notes: str | None = None,
) -> OledMineruReviewPacket:
    return OledMineruReviewPacket(
        packet_id=packet_id,
        paper_id="paper-adjudication",
        source_label="synthetic",
        compiled_record_id=packet_id.replace("review:", ""),
        compiled_status="partial",
        source_candidate_hashes=["hash-table"],
        source_evidence_anchors=["paper:p2:b0:table"],
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
        schema_error_codes=schema_errors or [],
        schema_warning_codes=schema_warnings or [],
        review_decision=decision,
        reviewer_notes=notes,
        metadata={
            "review_packet_only": True,
            "gold_record_created": False,
            "curated_dataset_written": False,
        },
    )


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_packets_jsonl(path: Path, packets: list[OledMineruReviewPacket]) -> Path:
    path.write_text(
        "\n".join(json.dumps(packet.model_dump(mode="json"), sort_keys=True) for packet in packets) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_review_packets_jsonl_loads_valid_packets_and_ignores_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "packets.jsonl"
    packet = _packet("review:compiled-oled:abc123", OledReviewDecision.ACCEPT)
    path.write_text(
        "\n"
        + json.dumps(packet.model_dump(mode="json"), sort_keys=True)
        + "\n\n",
        encoding="utf-8",
    )

    packets = load_oled_mineru_review_packets_jsonl(path)

    assert len(packets) == 1
    assert packets[0].packet_id == "review:compiled-oled:abc123"
    assert packets[0].review_decision == OledReviewDecision.ACCEPT

    bad_path = tmp_path / "bad-packets.jsonl"
    bad_path.write_text(json.dumps(packet.model_dump(mode="json"), sort_keys=True) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_review_packet_jsonl:line-2"):
        load_oled_mineru_review_packets_jsonl(bad_path)

    path_leak = _packet("review:compiled-oled:path-leak").model_copy(update={"metadata": {"source_path": str(tmp_path)}})
    path_leak_path = _write_packets_jsonl(tmp_path / "path-leak.jsonl", [path_leak])
    with pytest.raises(ValueError, match="absolute_path_in_review_packet_metadata:review:compiled-oled:path-leak"):
        load_oled_mineru_review_packets_jsonl(path_leak_path)


def test_load_review_decision_manifest_handles_valid_missing_and_invalid_json(tmp_path: Path) -> None:
    manifest_path = _write_json(
        tmp_path / "decisions.json",
        {
            "review_manifest_id": "round-001",
            "packet_source_label": "smoke",
            "decisions": [
                {
                    "packet_id": "review:compiled-oled:abc123",
                    "review_decision": "accept",
                    "reviewer_notes": "Values match Table 2.",
                }
            ],
        },
    )

    manifest = load_oled_review_decision_manifest(manifest_path)

    assert manifest.review_manifest_id == "round-001"
    assert manifest.packet_source_label == "smoke"
    assert manifest.decisions[0].review_decision == OledReviewDecision.ACCEPT

    with pytest.raises(ValueError, match="missing_review_decision_manifest:"):
        load_oled_review_decision_manifest(tmp_path / "missing.json")

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_review_decision_manifest_json:"):
        load_oled_review_decision_manifest(invalid_path)


def test_adjudicate_embedded_accept_and_reject_decisions_counts_statuses() -> None:
    report = adjudicate_oled_mineru_review_packets(
        [
            _packet("review:compiled-oled:accepted", OledReviewDecision.ACCEPT),
            _packet("review:compiled-oled:rejected", OledReviewDecision.REJECT, notes="Wrong row."),
        ]
    )

    assert report.is_valid is True
    assert report.accepted_count == 1
    assert report.rejected_count == 1
    assert report.decision_counts == {"accept": 1, "reject": 1}
    assert [packet.adjudication_status for packet in report.adjudicated_packets] == ["accepted", "rejected"]


def test_overlay_decisions_from_manifest_without_mutating_original_packets() -> None:
    packets = [_packet("review:compiled-oled:abc123")]
    manifest = OledReviewDecisionManifest(
        review_manifest_id="round-overlay",
        decisions=[
            OledReviewDecisionEntry(
                packet_id="review:compiled-oled:abc123",
                review_decision=OledReviewDecision.ACCEPT,
                reviewer_notes="Checked.",
            )
        ],
    )

    report = adjudicate_oled_mineru_review_packets(packets, decision_manifest=manifest)

    assert report.review_manifest_id == "round-overlay"
    assert report.accepted_count == 1
    assert report.adjudicated_packets[0].decision_entry is not None
    assert report.adjudicated_packets[0].decision_entry.review_decision == OledReviewDecision.ACCEPT
    assert packets[0].review_decision == OledReviewDecision.UNREVIEWED


def test_duplicate_and_unknown_manifest_decisions_emit_errors() -> None:
    manifest = OledReviewDecisionManifest(
        review_manifest_id="round-bad",
        decisions=[
            OledReviewDecisionEntry(packet_id="review:compiled-oled:abc123", review_decision=OledReviewDecision.ACCEPT),
            OledReviewDecisionEntry(packet_id="review:compiled-oled:abc123", review_decision=OledReviewDecision.REJECT),
            OledReviewDecisionEntry(packet_id="review:compiled-oled:unknown", review_decision=OledReviewDecision.ACCEPT),
        ],
    )

    report = adjudicate_oled_mineru_review_packets(
        [_packet("review:compiled-oled:abc123")],
        decision_manifest=manifest,
    )

    assert report.is_valid is False
    assert "duplicate_decision_packet_id" in report.error_codes
    assert "unknown_decision_packet_id" in report.error_codes
    assert report.invalid_count == 1


def test_require_all_reviewed_flags_missing_and_unreviewed_packets() -> None:
    packets = [
        _packet("review:compiled-oled:missing"),
        _packet("review:compiled-oled:unreviewed"),
    ]
    manifest = OledReviewDecisionManifest(
        review_manifest_id="round-required",
        decisions=[
            OledReviewDecisionEntry(
                packet_id="review:compiled-oled:unreviewed",
                review_decision=OledReviewDecision.UNREVIEWED,
            )
        ],
    )

    report = adjudicate_oled_mineru_review_packets(
        packets,
        decision_manifest=manifest,
        require_all_reviewed=True,
    )

    assert report.invalid_count == 2
    assert "missing_review_decision" in report.error_codes
    assert "unreviewed_packet" in report.error_codes


def test_accept_with_schema_errors_is_blocked_by_default_and_can_be_allowed() -> None:
    packet = _packet(
        "review:compiled-oled:schema-error",
        OledReviewDecision.ACCEPT,
        schema_errors=["unknown_property_label"],
    )

    blocked = adjudicate_oled_mineru_review_packets([packet])
    allowed = adjudicate_oled_mineru_review_packets([packet], allow_accept_with_schema_errors=True)

    assert blocked.invalid_count == 1
    assert "accept_with_schema_errors" in blocked.error_codes
    assert allowed.accepted_count == 1
    assert allowed.is_valid is True


def test_needs_correction_validation_reports_warnings_and_errors() -> None:
    no_structured_corrections = OledReviewDecisionManifest(
        review_manifest_id="round-correction-warning",
        decisions=[
            OledReviewDecisionEntry(
                packet_id="review:compiled-oled:needs-correction",
                review_decision=OledReviewDecision.NEEDS_CORRECTION,
                reviewer_notes="Unit needs review.",
            )
        ],
    )
    warning_report = adjudicate_oled_mineru_review_packets(
        [_packet("review:compiled-oled:needs-correction")],
        decision_manifest=no_structured_corrections,
    )

    assert warning_report.needs_correction_count == 1
    assert "needs_correction_without_structured_corrections" in warning_report.warning_codes

    bad_correction = OledReviewDecisionManifest(
        review_manifest_id="round-correction-error",
        decisions=[
            OledReviewDecisionEntry(
                packet_id="review:compiled-oled:bad-correction",
                review_decision=OledReviewDecision.NEEDS_CORRECTION,
                corrections=[
                    OledReviewCorrection(
                        correction_type=OledReviewCorrectionType.PROPERTY_UNIT,
                        field_path="",
                        original_value=None,
                        proposed_value=None,
                    )
                ],
            )
        ],
    )
    error_report = adjudicate_oled_mineru_review_packets(
        [_packet("review:compiled-oled:bad-correction")],
        decision_manifest=bad_correction,
    )
    assert error_report.invalid_count == 1
    assert "correction_missing_field_path" in error_report.error_codes
    assert "correction_without_values" in error_report.warning_codes

    good_correction = OledReviewDecisionManifest(
        review_manifest_id="round-correction-good",
        decisions=[
            OledReviewDecisionEntry(
                packet_id="review:compiled-oled:good-correction",
                review_decision=OledReviewDecision.NEEDS_CORRECTION,
                corrections=[
                    OledReviewCorrection(
                        correction_type=OledReviewCorrectionType.PROPERTY_UNIT,
                        field_path="properties[0].unit",
                        original_value="%",
                        proposed_value="fraction",
                        reason="Table reports fraction.",
                    )
                ],
            )
        ],
    )
    good_report = adjudicate_oled_mineru_review_packets(
        [_packet("review:compiled-oled:good-correction")],
        decision_manifest=good_correction,
    )
    assert good_report.needs_correction_count == 1
    assert good_report.error_codes == []


def test_filter_helper_returns_requested_statuses() -> None:
    report = adjudicate_oled_mineru_review_packets(
        [
            _packet("review:compiled-oled:accepted", OledReviewDecision.ACCEPT),
            _packet("review:compiled-oled:needs", OledReviewDecision.NEEDS_CORRECTION, notes="Needs edit."),
        ]
    )

    accepted = filter_adjudicated_oled_review_packets(report, statuses=["accepted"])
    needs_correction = filter_adjudicated_oled_review_packets(report, statuses=["needs_correction"])

    assert [packet.packet.packet_id for packet in accepted] == ["review:compiled-oled:accepted"]
    assert [packet.packet.packet_id for packet in needs_correction] == ["review:compiled-oled:needs"]


def test_report_writer_is_deterministic_redacted_and_does_not_write_raw_text(tmp_path: Path) -> None:
    packet = _packet("review:compiled-oled:writer", OledReviewDecision.ACCEPT)
    packet = packet.model_copy(update={"metadata": {"source_path": str(tmp_path / "paper.json"), "raw_text": "mCBP | D1 | 82"}})
    report = adjudicate_oled_mineru_review_packets([packet])
    output_path = tmp_path / "adjudication.json"

    write_oled_review_adjudication_report_json(report, output_path)
    text = output_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert payload["accepted_count"] == 1
    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert str(tmp_path) not in text
    assert "mCBP | D1 | 82" not in text
    assert payload["metadata"]["gold_records_created"] is False


def test_cli_smoke_writes_valid_adjudication_report(tmp_path: Path) -> None:
    packets_path = _write_packets_jsonl(tmp_path / "packets.jsonl", [_packet("review:compiled-oled:cli")])
    decisions_path = _write_json(
        tmp_path / "decisions.json",
        {
            "review_manifest_id": "round-cli",
            "decisions": [
                {
                    "packet_id": "review:compiled-oled:cli",
                    "review_decision": "accept",
                    "reviewer_notes": "Checked.",
                }
            ],
        },
    )
    output_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--packets-jsonl",
            str(packets_path),
            "--decisions",
            str(decisions_path),
            "--output-report",
            str(output_path),
            "--require-all-reviewed",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted_count"] == 1


def test_public_adjudication_api_is_exported_from_domain_package(tmp_path: Path) -> None:
    packets_path = _write_packets_jsonl(tmp_path / "package-packets.jsonl", [_packet("review:compiled-oled:package")])
    decisions_path = _write_json(
        tmp_path / "package-decisions.json",
        {
            "review_manifest_id": "round-package",
            "decisions": [
                {
                    "packet_id": "review:compiled-oled:package",
                    "review_decision": "accept",
                }
            ],
        },
    )

    packets = package_load_oled_mineru_review_packets_jsonl(packets_path)
    manifest = package_load_oled_review_decision_manifest(decisions_path)
    report = package_adjudicate_oled_mineru_review_packets(packets, decision_manifest=manifest)
    accepted = package_filter_adjudicated_oled_review_packets(report, statuses=["accepted"])
    output_path = tmp_path / "package-report.json"
    package_write_oled_review_adjudication_report_json(report, output_path)

    assert isinstance(manifest, OledReviewDecisionManifest)
    assert isinstance(manifest.decisions[0], OledReviewDecisionEntry)
    assert isinstance(report, OledReviewAdjudicationReport)
    assert isinstance(report.adjudicated_packets[0], OledAdjudicatedReviewPacket)
    assert isinstance(OledReviewAdjudicationFinding(code="x", message="y"), OledReviewAdjudicationFinding)
    assert isinstance(
        OledReviewCorrection(
            correction_type=OledReviewCorrectionType.NOTES_ONLY,
            field_path="reviewer_notes",
            proposed_value="Checked.",
        ),
        OledReviewCorrection,
    )
    assert accepted[0].adjudication_status == "accepted"
    assert output_path.exists()
