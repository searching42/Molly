from __future__ import annotations

import json

import pytest

from ai4s_agent.domains.oled_layered_schema import (
    OledDeviceLayer,
    OledInteractionLayer,
    OledLayeredRecord,
    OledPropertyObservation,
)
from ai4s_agent.domains.oled_review_packets import (
    OledReviewItem,
    OledReviewPacket,
    OledReviewerDecision,
    OledReviewerDecisionTemplate,
    oled_review_packet_digest,
)
from ai4s_agent.domains.oled_schema_candidate_compiler import (
    OledCompiledLayeredRecordCandidate,
    OledSchemaCompilationGroupKey,
    OledSchemaCompilationStatus,
)
from ai4s_agent.domains.oled_reviewed_extraction_candidates import (
    stage_oled_reviewed_extraction_candidates,
)
from ai4s_agent.domains.oled_reviewed_gold_candidates import (
    convert_reviewed_extractions_to_gold_candidates,
)
from ai4s_agent.oled_review_adjudication_bridge import (
    OledReviewBridgeStatus,
    bind_oled_review_packet_source_payloads,
    build_legacy_adjudication_bundle,
    evaluate_oled_review_decisions,
    migrate_unchanged_oled_review_decisions,
)


def _packet() -> OledReviewPacket:
    return OledReviewPacket(
        run_id="review-run",
        generated_at="2026-07-10T00:00:00Z",
        review_items=[
            OledReviewItem(
                review_item_id="review:compiled",
                paper_id="paper-1",
                candidate_type="oled_compiled_record",
                priority="high",
                source_candidate_id="compiled-1",
                source_artifact="oled_compiled_records.json",
                evidence_location="paper-1:p2:b4",
            ),
            OledReviewItem(
                review_item_id="review:text",
                paper_id="paper-1",
                candidate_type="oled_text_evidence",
                priority="medium",
                source_candidate_id="text-1",
                source_artifact="oled_text_evidence_candidates.json",
                property_id="plqy",
                raw_value="82%",
                evidence_location="paper-1:p3:b2",
            ),
        ],
    )


def _pending_decisions(packet: OledReviewPacket | None = None) -> OledReviewerDecisionTemplate:
    source_packet = packet or _packet()
    return OledReviewerDecisionTemplate(
        run_id=source_packet.run_id,
        generated_at=source_packet.generated_at,
        source_packet_digest=oled_review_packet_digest(source_packet),
        decisions=[
            OledReviewerDecision(review_item_id="review:compiled"),
            OledReviewerDecision(review_item_id="review:text"),
        ],
    )


def _completed_decisions(packet: OledReviewPacket | None = None) -> OledReviewerDecisionTemplate:
    source_packet = packet or _packet()
    return OledReviewerDecisionTemplate(
        run_id=source_packet.run_id,
        generated_at=source_packet.generated_at,
        source_packet_digest=oled_review_packet_digest(source_packet),
        decisions=[
            OledReviewerDecision(
                review_item_id="review:compiled",
                review_status="reviewed",
                decision="accept",
                reviewer="benton",
                reviewed_at="2026-07-10T12:00:00+08:00",
            ),
            OledReviewerDecision(
                review_item_id="review:text",
                review_status="reviewed",
                decision="reject",
                reviewer="benton",
                reviewed_at="2026-07-10T12:01:00+08:00",
                comment="The statement is not tied to the claimed material.",
            ),
        ],
    )


def test_pending_template_is_ready_for_human_review() -> None:
    report = evaluate_oled_review_decisions(_packet(), _pending_decisions())

    assert report.status == OledReviewBridgeStatus.AWAITING_HUMAN_REVIEW
    assert report.item_count == 2
    assert report.pending_count == 2
    assert report.invalid_count == 0
    assert report.downstream_eligible_count == 1
    assert report.is_valid


def test_completed_template_is_ready_for_adjudication() -> None:
    report = evaluate_oled_review_decisions(
        _packet(),
        _completed_decisions(),
        require_all_reviewed=True,
    )

    assert report.status == OledReviewBridgeStatus.READY_FOR_ADJUDICATION
    assert report.reviewed_count == 2
    assert report.pending_count == 0
    assert report.outcome_counts == {"accepted": 1, "rejected": 1}
    assert report.downstream_ready_count == 1
    assert report.is_complete


def test_compiled_only_admission_packet_preserves_require_all_semantics() -> None:
    full_packet = _packet()
    admission_packet = full_packet.model_copy(
        update={
            "summary": {
                "packet_purpose": "compiled_only_admission",
                "full_qa_review_item_count": 2,
                "excluded_quality_review_item_count": 1,
            },
            "review_items": [full_packet.review_items[0]],
        }
    )
    admission_decisions = OledReviewerDecisionTemplate(
        run_id=admission_packet.run_id,
        generated_at=admission_packet.generated_at,
        source_packet_digest=oled_review_packet_digest(admission_packet),
        decisions=[
            OledReviewerDecision(
                review_item_id="review:compiled",
                review_status="reviewed",
                decision="accept",
                reviewer="benton",
                reviewed_at="2026-07-10T12:00:00+08:00",
            )
        ],
    )

    admission_report = evaluate_oled_review_decisions(
        admission_packet,
        admission_decisions,
        require_all_reviewed=True,
    )

    assert admission_report.status == OledReviewBridgeStatus.READY_FOR_ADJUDICATION
    assert admission_report.item_count == 1
    assert admission_report.reviewed_count == 1
    assert admission_report.downstream_ready_count == 1

    full_packet_decisions = admission_decisions.model_copy(
        update={"source_packet_digest": oled_review_packet_digest(full_packet)}
    )
    full_report = evaluate_oled_review_decisions(
        full_packet,
        full_packet_decisions,
        require_all_reviewed=True,
    )
    assert full_report.status == OledReviewBridgeStatus.BLOCKED
    assert "missing_decision_entry" in {finding.code for finding in full_report.findings}


def test_empty_compiled_admission_packet_has_no_eligible_items_status() -> None:
    packet = OledReviewPacket(
        run_id="empty-admission",
        generated_at="2026-07-10T00:00:00Z",
        summary={"packet_purpose": "compiled_only_admission"},
        review_items=[],
    )
    decisions = OledReviewerDecisionTemplate(
        run_id=packet.run_id,
        generated_at=packet.generated_at,
        source_packet_digest=oled_review_packet_digest(packet),
        decisions=[],
    )

    report = evaluate_oled_review_decisions(packet, decisions, require_all_reviewed=True)

    assert report.status == OledReviewBridgeStatus.NO_ELIGIBLE_ITEMS
    assert report.is_valid
    assert not report.is_complete
    assert "no_compiled_admission_items" in {finding.code for finding in report.findings}


def test_compiled_admission_packet_rejects_non_compiled_items() -> None:
    packet = _packet().model_copy(
        update={"summary": {"packet_purpose": "compiled_only_admission"}}
    )
    decisions = _pending_decisions(packet)

    report = evaluate_oled_review_decisions(packet, decisions)

    assert report.status == OledReviewBridgeStatus.BLOCKED
    assert "compiled_admission_contains_non_compiled_item" in {
        finding.code for finding in report.findings
    }


def test_decision_migration_compares_complete_source_payloads(tmp_path) -> None:
    source_compiled = tmp_path / "source_compiled.json"
    target_compiled = tmp_path / "target_compiled.json"
    source_text = tmp_path / "source_text.json"
    target_text = tmp_path / "target_text.json"
    source_compiled.write_text(
        json.dumps(
            {
                "compiled_records": [
                    {
                        "record_id": "compiled-1",
                        "layered_record": {"device": {"device_stack": ["ITO", "EML", "Al"]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    target_compiled.write_text(
        json.dumps(
            {
                "compiled_records": [
                    {
                        "record_id": "compiled-1",
                        "layered_record": {
                            "device": {
                                "device_stack": ["ITO", "EML", "Al"],
                                "fabrication_method": "vacuum_deposition",
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    text_payload = {
        "text_evidence_candidates": [
            {"candidate_id": "text-1", "property_id": "plqy", "raw_value": "82%"}
        ]
    }
    source_text.write_text(json.dumps(text_payload), encoding="utf-8")
    target_text.write_text(json.dumps(text_payload), encoding="utf-8")

    base_packet = _packet()
    source_packet = base_packet.model_copy(
        update={
            "review_items": [
                base_packet.review_items[0].model_copy(update={"source_artifact": str(source_compiled)}),
                base_packet.review_items[1].model_copy(update={"source_artifact": str(source_text)}),
            ]
        }
    )
    source_packet = bind_oled_review_packet_source_payloads(source_packet)
    target_packet = source_packet.model_copy(
        update={
            "run_id": "review-run-v2",
            "review_items": [
                source_packet.review_items[0].model_copy(
                    update={
                        "review_item_id": "review:v2:compiled",
                        "source_artifact": str(target_compiled),
                    }
                ),
                source_packet.review_items[1].model_copy(
                    update={
                        "review_item_id": "review:v2:text",
                        "source_artifact": str(target_text),
                    }
                ),
            ],
        }
    )
    target_packet = bind_oled_review_packet_source_payloads(target_packet)

    migrated, migration = migrate_unchanged_oled_review_decisions(
        source_packet,
        _completed_decisions(source_packet),
        target_packet,
    )

    assert migrated.run_id == "review-run-v2"
    assert migrated.decisions[0].review_item_id == "review:v2:compiled"
    assert migrated.decisions[0].review_status == "pending"
    assert migrated.decisions[0].decision == ""
    assert migrated.decisions[1].review_item_id == "review:v2:text"
    assert migrated.decisions[1].decision == "reject"
    assert migration.migrated_item_count == 1
    assert migration.migrated_reviewed_count == 1
    assert migration.reset_pending_count == 1
    assert migration.reset_items[0]["reason"] == "source_payload_changed"

    readiness = evaluate_oled_review_decisions(target_packet, migrated)
    assert readiness.status == OledReviewBridgeStatus.AWAITING_HUMAN_REVIEW
    assert readiness.reviewed_count == 1
    assert readiness.pending_count == 1
    assert readiness.invalid_count == 0


def test_packet_digest_mismatch_blocks_review() -> None:
    decisions = _completed_decisions().model_copy(update={"source_packet_digest": "sha256:" + "0" * 64})

    report = evaluate_oled_review_decisions(_packet(), decisions)

    assert report.status == OledReviewBridgeStatus.BLOCKED
    assert "source_packet_digest_mismatch" in {finding.code for finding in report.findings}


def test_pending_decision_must_not_retain_corrections_or_audit_fields() -> None:
    decisions = _pending_decisions()
    decisions.decisions[0] = decisions.decisions[0].model_copy(
        update={
            "corrected_value": "6.3",
            "reviewer": "stale-reviewer",
        }
    )

    report = evaluate_oled_review_decisions(_packet(), decisions)

    assert report.status == OledReviewBridgeStatus.BLOCKED
    assert report.invalid_count == 1
    assert "pending_decision_contains_review_data" in {finding.code for finding in report.findings}


def test_corrected_reported_value_fields_must_be_complete_and_tied_to_numeric_correction() -> None:
    decisions = _completed_decisions()
    decisions.decisions[0] = decisions.decisions[0].model_copy(
        update={
            "corrected_reported_value_text": "0.040",
            "comment": "Corrected source precision.",
        }
    )

    report = evaluate_oled_review_decisions(_packet(), decisions)

    assert report.status == OledReviewBridgeStatus.BLOCKED
    assert {finding.code for finding in report.findings} >= {
        "corrected_reported_value_fields_incomplete",
        "corrected_reported_value_requires_corrected_value",
    }


def test_completed_decision_requires_reviewer_and_timestamp() -> None:
    decisions = _completed_decisions()
    decisions.decisions[0] = decisions.decisions[0].model_copy(update={"reviewer": "", "reviewed_at": ""})

    report = evaluate_oled_review_decisions(_packet(), decisions, require_all_reviewed=True)

    assert report.status == OledReviewBridgeStatus.BLOCKED
    assert report.invalid_count == 1
    assert {finding.code for finding in report.findings} >= {"missing_reviewer", "missing_reviewed_at"}


def test_completed_compiled_decision_builds_legacy_adjudication_bundle(tmp_path) -> None:
    compiled_record = OledCompiledLayeredRecordCandidate(
        record_id="compiled-1",
        status=OledSchemaCompilationStatus.COMPILED,
        group_key=OledSchemaCompilationGroupKey(
            source_paper_id="paper-1",
            source_candidate_hashes=["source-hash"],
        ),
        layered_record=OledLayeredRecord(device=OledDeviceLayer(device_stack=["ITO", "EML", "Al"])),
        source_schema_candidate_ids=["schema-1"],
        source_candidate_hashes=["source-hash"],
        source_evidence_anchors=["paper-1:p2:b4"],
    )
    source_artifact = tmp_path / "source_payloads.json"
    source_artifact.write_text(
        json.dumps(
            {
                "compiled_records": [compiled_record.model_dump(mode="json")],
                "text_evidence_candidates": [{"candidate_id": "text-1"}],
            }
        ),
        encoding="utf-8",
    )
    base_packet = _packet()
    packet = base_packet.model_copy(
        update={
            "review_items": [
                item.model_copy(update={"source_artifact": str(source_artifact)})
                for item in base_packet.review_items
            ]
        }
    )
    packet = bind_oled_review_packet_source_payloads(packet)
    decisions = _completed_decisions(packet)

    bundle = build_legacy_adjudication_bundle(packet, decisions, [compiled_record])

    assert len(bundle.review_packets) == 1
    assert bundle.review_packets[0].compiled_record_id == "compiled-1"
    assert len(bundle.decision_manifest.decisions) == 1
    assert bundle.adjudication_report.packet_count == 1
    assert bundle.adjudication_report.accepted_count == 1

    altered_record = compiled_record.model_copy(
        update={"layered_record": OledLayeredRecord(device=OledDeviceLayer(device_stack=["ITO", "ETL", "Al"]))}
    )
    with pytest.raises(ValueError, match="compiled_record_payload_digest_mismatch"):
        build_legacy_adjudication_bundle(packet, decisions, [altered_record])

    source_artifact.write_text(
        json.dumps(
            {
                "compiled_records": [altered_record.model_dump(mode="json")],
                "text_evidence_candidates": [{"candidate_id": "text-1"}],
            }
        ),
        encoding="utf-8",
    )
    mutated_readiness = evaluate_oled_review_decisions(packet, decisions)
    assert mutated_readiness.status == OledReviewBridgeStatus.BLOCKED
    assert "source_payload_digest_mismatch" in {
        finding.code for finding in mutated_readiness.findings
    }


def test_bridge_numeric_correction_preserves_trailing_zero_through_gold_conversion(tmp_path) -> None:
    compiled_record = OledCompiledLayeredRecordCandidate(
        record_id="compiled-numeric",
        status=OledSchemaCompilationStatus.COMPILED,
        group_key=OledSchemaCompilationGroupKey(
            source_paper_id="paper-1",
            source_candidate_hashes=["source-hash"],
        ),
        layered_record=OledLayeredRecord(
            interaction=OledInteractionLayer(
                properties=[
                    OledPropertyObservation(
                        property_label="Photoluminescence quantum yield",
                        value=0.03,
                        unit="fraction",
                        reported_value_text="0.030",
                        reported_decimal_places=3,
                        metadata={"source_property_id": "plqy"},
                    )
                ]
            )
        ),
        source_schema_candidate_ids=["schema-numeric"],
        source_candidate_hashes=["source-hash"],
        source_evidence_anchors=["paper-1:p2:b4"],
    )
    source_artifact = tmp_path / "compiled_numeric.json"
    source_artifact.write_text(
        json.dumps({"compiled_records": [compiled_record.model_dump(mode="json")]}),
        encoding="utf-8",
    )
    packet = bind_oled_review_packet_source_payloads(
        OledReviewPacket(
            run_id="review-numeric",
            generated_at="2026-07-13T00:00:00Z",
            review_items=[
                OledReviewItem(
                    review_item_id="review:numeric",
                    paper_id="paper-1",
                    candidate_type="oled_compiled_record",
                    priority="high",
                    source_candidate_id="compiled-numeric",
                    source_artifact=str(source_artifact),
                    property_id="plqy",
                    property_label="Photoluminescence quantum yield",
                    raw_value="0.030",
                    numeric_value=0.03,
                    unit="fraction",
                    evidence_location="paper-1:p2:b4",
                )
            ],
        )
    )
    decisions = OledReviewerDecisionTemplate(
        run_id=packet.run_id,
        generated_at=packet.generated_at,
        source_packet_digest=oled_review_packet_digest(packet),
        decisions=[
            OledReviewerDecision(
                review_item_id="review:numeric",
                review_status="reviewed",
                decision="accept",
                corrected_value="0.040",
                corrected_reported_value_text="0.040",
                corrected_reported_decimal_places="3",
                reviewer="benton",
                reviewed_at="2026-07-13T12:00:00+08:00",
                comment="Corrected source value while retaining its reported precision.",
            )
        ],
    )

    bundle = build_legacy_adjudication_bundle(packet, decisions, [compiled_record])
    correction = bundle.decision_manifest.decisions[0].corrections[0]
    assert correction.proposed_value == 0.04
    assert correction.proposed_reported_value_text == "0.040"
    assert correction.proposed_reported_decimal_places == 3

    staging = stage_oled_reviewed_extraction_candidates(bundle.adjudication_report)
    conversion = convert_reviewed_extractions_to_gold_candidates(staging.reviewed_candidates)

    assert staging.reviewed_candidates[0].status.value == "corrected"
    assert conversion.converted_candidate_count == 1
    gold_record = conversion.gold_candidates[0].gold_record
    assert gold_record is not None
    assert gold_record.layered_record.interaction is not None
    observation = gold_record.layered_record.interaction.properties[0]
    assert observation.value == 0.04
    assert observation.reported_value_text == "0.040"
    assert observation.reported_decimal_places == 3
