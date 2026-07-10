from __future__ import annotations

from ai4s_agent.domains.oled_layered_schema import OledDeviceLayer, OledLayeredRecord
from ai4s_agent.domains.oled_review_packets import (
    OledReviewItem,
    OledReviewPacket,
    OledReviewerDecision,
    OledReviewerDecisionTemplate,
)
from ai4s_agent.domains.oled_schema_candidate_compiler import (
    OledCompiledLayeredRecordCandidate,
    OledSchemaCompilationGroupKey,
    OledSchemaCompilationStatus,
)
from ai4s_agent.oled_review_adjudication_bridge import (
    OledReviewBridgeStatus,
    build_legacy_adjudication_bundle,
    evaluate_oled_review_decisions,
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


def _pending_decisions() -> OledReviewerDecisionTemplate:
    return OledReviewerDecisionTemplate(
        run_id="review-run",
        generated_at="2026-07-10T00:00:00Z",
        decisions=[
            OledReviewerDecision(review_item_id="review:compiled"),
            OledReviewerDecision(review_item_id="review:text"),
        ],
    )


def _completed_decisions() -> OledReviewerDecisionTemplate:
    return OledReviewerDecisionTemplate(
        run_id="review-run",
        generated_at="2026-07-10T00:00:00Z",
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


def test_completed_decision_requires_reviewer_and_timestamp() -> None:
    decisions = _completed_decisions()
    decisions.decisions[0] = decisions.decisions[0].model_copy(update={"reviewer": "", "reviewed_at": ""})

    report = evaluate_oled_review_decisions(_packet(), decisions, require_all_reviewed=True)

    assert report.status == OledReviewBridgeStatus.BLOCKED
    assert report.invalid_count == 1
    assert {finding.code for finding in report.findings} >= {"missing_reviewer", "missing_reviewed_at"}


def test_completed_compiled_decision_builds_legacy_adjudication_bundle() -> None:
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

    bundle = build_legacy_adjudication_bundle(_packet(), _completed_decisions(), [compiled_record])

    assert len(bundle.review_packets) == 1
    assert bundle.review_packets[0].compiled_record_id == "compiled-1"
    assert len(bundle.decision_manifest.decisions) == 1
    assert bundle.adjudication_report.packet_count == 1
    assert bundle.adjudication_report.accepted_count == 1
