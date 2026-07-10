from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled_mineru_review_adjudication import (
    OledReviewAdjudicationReport,
    OledReviewCorrection,
    OledReviewCorrectionType,
    OledReviewDecisionEntry,
    OledReviewDecisionManifest,
    adjudicate_oled_mineru_review_packets,
    write_oled_review_adjudication_report_json,
)
from ai4s_agent.domains.oled_mineru_review_packets import (
    OledMineruReviewPacket,
    OledReviewDecision,
    build_oled_mineru_review_packets_from_compiled_records,
    write_oled_mineru_review_packets_jsonl,
)
from ai4s_agent.domains.oled_review_packets import (
    OledReviewItem,
    OledReviewPacket,
    OledReviewerDecision,
    OledReviewerDecisionTemplate,
)
from ai4s_agent.domains.oled_schema_candidate_compiler import OledCompiledLayeredRecordCandidate


class OledReviewBridgeStatus(str, Enum):
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"
    READY_FOR_ADJUDICATION = "ready_for_adjudication"
    BLOCKED = "blocked"


class OledReviewBridgeFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    review_item_id: str | None = None


class OledReviewBridgeItem(BaseModel):
    review_item_id: str
    candidate_type: str
    priority: str
    source_candidate_id: str
    outcome: Literal[
        "pending",
        "accepted",
        "accepted_with_corrections",
        "rejected",
        "needs_more_context",
        "invalid",
    ]
    downstream_eligible: bool = False
    reviewer: str = ""
    reviewed_at: str = ""
    has_corrections: bool = False
    finding_codes: list[str] = Field(default_factory=list)


class OledReviewBridgeReport(BaseModel):
    run_id: str
    status: OledReviewBridgeStatus
    packet_schema_version: str
    decision_schema_version: str
    item_count: int
    reviewed_count: int
    pending_count: int
    invalid_count: int
    downstream_eligible_count: int
    downstream_ready_count: int
    outcome_counts: dict[str, int] = Field(default_factory=dict)
    candidate_type_counts: dict[str, int] = Field(default_factory=dict)
    priority_counts: dict[str, int] = Field(default_factory=dict)
    items: list[OledReviewBridgeItem] = Field(default_factory=list)
    findings: list[OledReviewBridgeFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def is_complete(self) -> bool:
        return self.is_valid and self.pending_count == 0 and self.invalid_count == 0


class OledLegacyAdjudicationBundle(BaseModel):
    review_packets: list[OledMineruReviewPacket]
    decision_manifest: OledReviewDecisionManifest
    adjudication_report: OledReviewAdjudicationReport


def load_oled_review_packet_json(path: str | Path) -> OledReviewPacket:
    payload = _read_json(path, "review packet")
    try:
        return OledReviewPacket.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("invalid_oled_review_packet_json") from exc


def load_oled_reviewer_decisions_json(path: str | Path) -> OledReviewerDecisionTemplate:
    payload = _read_json(path, "reviewer decisions")
    try:
        return OledReviewerDecisionTemplate.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("invalid_oled_reviewer_decisions_json") from exc


def load_oled_compiled_records_json(path: str | Path) -> list[OledCompiledLayeredRecordCandidate]:
    payload = _read_json(path, "compiled records")
    records = payload.get("compiled_records")
    if not isinstance(records, list):
        raise ValueError("invalid_oled_compiled_records_json")
    try:
        return [OledCompiledLayeredRecordCandidate.model_validate(record) for record in records]
    except ValidationError as exc:
        raise ValueError("invalid_oled_compiled_records_json") from exc


def evaluate_oled_review_decisions(
    packet: OledReviewPacket,
    decisions: OledReviewerDecisionTemplate,
    *,
    require_all_reviewed: bool = False,
) -> OledReviewBridgeReport:
    findings: list[OledReviewBridgeFinding] = []
    if packet.run_id != decisions.run_id:
        findings.append(_finding("run_id_mismatch", "packet and decision run ids do not match", severity="error"))

    packet_items_by_id: dict[str, OledReviewItem] = {}
    for item in packet.review_items:
        if item.review_item_id in packet_items_by_id:
            findings.append(
                _finding(
                    "duplicate_review_item_id",
                    "review packet contains a duplicate review item id",
                    item.review_item_id,
                    severity="error",
                )
            )
        packet_items_by_id[item.review_item_id] = item

    decisions_by_id: dict[str, OledReviewerDecision] = {}
    for decision in decisions.decisions:
        if decision.review_item_id in decisions_by_id:
            findings.append(
                _finding(
                    "duplicate_decision_id",
                    "decision file contains a duplicate review item id",
                    decision.review_item_id,
                    severity="error",
                )
            )
        decisions_by_id[decision.review_item_id] = decision
        if decision.review_item_id not in packet_items_by_id:
            findings.append(
                _finding(
                    "unknown_decision_id",
                    "decision file references an item that is not in the packet",
                    decision.review_item_id,
                    severity="error",
                )
            )

    output_items: list[OledReviewBridgeItem] = []
    for item in packet.review_items:
        decision = decisions_by_id.get(item.review_item_id)
        item_findings: list[OledReviewBridgeFinding] = []
        if decision is None:
            item_findings.append(
                _finding(
                    "missing_decision_entry",
                    "review item has no corresponding decision entry",
                    item.review_item_id,
                    severity="error",
                )
            )
            outcome = "invalid"
            reviewer = ""
            reviewed_at = ""
            has_corrections = False
        else:
            item_findings.extend(_decision_findings(decision, require_all_reviewed=require_all_reviewed))
            outcome = _decision_outcome(decision, item_findings)
            reviewer = decision.reviewer.strip()
            reviewed_at = decision.reviewed_at.strip()
            has_corrections = _has_corrections(decision)
        findings.extend(item_findings)
        output_items.append(
            OledReviewBridgeItem(
                review_item_id=item.review_item_id,
                candidate_type=item.candidate_type,
                priority=item.priority,
                source_candidate_id=item.source_candidate_id,
                outcome=outcome,
                downstream_eligible=item.candidate_type == "oled_compiled_record",
                reviewer=reviewer,
                reviewed_at=reviewed_at,
                has_corrections=has_corrections,
                finding_codes=sorted({finding.code for finding in item_findings}),
            )
        )

    pending_count = sum(item.outcome == "pending" for item in output_items)
    invalid_count = sum(item.outcome == "invalid" for item in output_items)
    if any(finding.severity == "error" for finding in findings):
        status = OledReviewBridgeStatus.BLOCKED
    elif pending_count:
        status = OledReviewBridgeStatus.AWAITING_HUMAN_REVIEW
    else:
        status = OledReviewBridgeStatus.READY_FOR_ADJUDICATION

    return OledReviewBridgeReport(
        run_id=packet.run_id,
        status=status,
        packet_schema_version=packet.schema_version,
        decision_schema_version=decisions.schema_version,
        item_count=len(output_items),
        reviewed_count=sum(item.outcome not in {"pending", "invalid"} for item in output_items),
        pending_count=pending_count,
        invalid_count=invalid_count,
        downstream_eligible_count=sum(item.downstream_eligible for item in output_items),
        downstream_ready_count=sum(
            item.downstream_eligible and item.outcome in {"accepted", "accepted_with_corrections"}
            for item in output_items
        ),
        outcome_counts=dict(sorted(Counter(item.outcome for item in output_items).items())),
        candidate_type_counts=dict(sorted(Counter(item.candidate_type for item in output_items).items())),
        priority_counts=dict(sorted(Counter(item.priority for item in output_items).items())),
        items=output_items,
        findings=findings,
        metadata={
            "human_review_required": True,
            "review_decisions_mutate_training_data": False,
            "only_compiled_records_are_downstream_eligible": True,
            "raw_schema_and_text_items_are_quality_review_evidence_only": True,
            "require_all_reviewed": require_all_reviewed,
        },
    )


def build_legacy_adjudication_bundle(
    packet: OledReviewPacket,
    decisions: OledReviewerDecisionTemplate,
    compiled_records: list[OledCompiledLayeredRecordCandidate],
) -> OledLegacyAdjudicationBundle:
    readiness = evaluate_oled_review_decisions(packet, decisions, require_all_reviewed=True)
    if readiness.status != OledReviewBridgeStatus.READY_FOR_ADJUDICATION:
        raise ValueError("review_decisions_not_ready_for_adjudication")

    compiled_items = {
        item.source_candidate_id: item
        for item in packet.review_items
        if item.candidate_type == "oled_compiled_record"
    }
    decision_by_id = {decision.review_item_id: decision for decision in decisions.decisions}
    records_by_id = {record.record_id: record for record in compiled_records}
    missing_records = sorted(set(compiled_items) - set(records_by_id))
    if missing_records:
        raise ValueError("compiled_review_items_missing_source_records:" + ",".join(missing_records))

    legacy_packets: list[OledMineruReviewPacket] = []
    legacy_decisions: list[OledReviewDecisionEntry] = []
    for record_id, item in sorted(compiled_items.items()):
        record = records_by_id[record_id]
        built_packets = build_oled_mineru_review_packets_from_compiled_records(
            [record],
            paper_id=item.paper_id,
            source_label="oled_review_packet_bridge",
        )
        if len(built_packets) != 1:
            raise ValueError(f"compiled_record_packet_build_failed:{record_id}")
        legacy_packet = built_packets[0]
        legacy_packets.append(legacy_packet)
        legacy_decisions.append(_legacy_decision_entry(item, decision_by_id[item.review_item_id], legacy_packet))

    manifest = OledReviewDecisionManifest(
        review_manifest_id=f"oled-review:{packet.run_id}",
        packet_source_label="oled_review_packet.v1",
        decisions=legacy_decisions,
        metadata={
            "source_run_id": packet.run_id,
            "source_packet_schema_version": packet.schema_version,
            "source_decision_schema_version": decisions.schema_version,
            "compiled_records_only": True,
        },
    )
    adjudication = adjudicate_oled_mineru_review_packets(
        legacy_packets,
        decision_manifest=manifest,
        require_all_reviewed=True,
    )
    return OledLegacyAdjudicationBundle(
        review_packets=legacy_packets,
        decision_manifest=manifest,
        adjudication_report=adjudication,
    )


def write_oled_review_bridge_report_json(report: OledReviewBridgeReport, path: str | Path) -> None:
    write_json(Path(path), report.model_dump(mode="json"))


def write_oled_review_bridge_report_markdown(report: OledReviewBridgeReport, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(_bridge_markdown(report), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate OLED human review decisions and bridge compiled items.")
    parser.add_argument("--packet", required=True, help="Path to oled_review_packet.json.")
    parser.add_argument("--decisions", required=True, help="Path to the reviewer decision JSON.")
    parser.add_argument("--compiled-records", help="Path to oled_compiled_records.json for post-review bridging.")
    parser.add_argument("--output-report", required=True, help="Path to write bridge readiness JSON.")
    parser.add_argument("--output-markdown", help="Optional path to write bridge readiness Markdown.")
    parser.add_argument("--require-all-reviewed", action="store_true")
    parser.add_argument("--output-legacy-packets")
    parser.add_argument("--output-legacy-decisions")
    parser.add_argument("--output-adjudication-report")
    args = parser.parse_args(argv)
    try:
        packet = load_oled_review_packet_json(args.packet)
        decisions = load_oled_reviewer_decisions_json(args.decisions)
        readiness = evaluate_oled_review_decisions(
            packet,
            decisions,
            require_all_reviewed=args.require_all_reviewed,
        )
        write_oled_review_bridge_report_json(readiness, args.output_report)
        if args.output_markdown:
            write_oled_review_bridge_report_markdown(readiness, args.output_markdown)

        legacy_outputs_requested = any(
            [args.output_legacy_packets, args.output_legacy_decisions, args.output_adjudication_report]
        )
        if legacy_outputs_requested:
            if not args.compiled_records:
                raise ValueError("compiled_records_required_for_legacy_bridge")
            bundle = build_legacy_adjudication_bundle(
                packet,
                decisions,
                load_oled_compiled_records_json(args.compiled_records),
            )
            if args.output_legacy_packets:
                Path(args.output_legacy_packets).parent.mkdir(parents=True, exist_ok=True)
                write_oled_mineru_review_packets_jsonl(bundle.review_packets, args.output_legacy_packets)
            if args.output_legacy_decisions:
                write_json(Path(args.output_legacy_decisions), bundle.decision_manifest.model_dump(mode="json"))
            if args.output_adjudication_report:
                write_oled_review_adjudication_report_json(bundle.adjudication_report, args.output_adjudication_report)

        print(
            json.dumps(
                {
                    "run_id": readiness.run_id,
                    "status": readiness.status.value,
                    "item_count": readiness.item_count,
                    "reviewed_count": readiness.reviewed_count,
                    "pending_count": readiness.pending_count,
                    "invalid_count": readiness.invalid_count,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if readiness.status == OledReviewBridgeStatus.BLOCKED:
            return 1
        if args.require_all_reviewed and not readiness.is_complete:
            return 1
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _decision_findings(
    decision: OledReviewerDecision,
    *,
    require_all_reviewed: bool,
) -> list[OledReviewBridgeFinding]:
    findings: list[OledReviewBridgeFinding] = []
    item_id = decision.review_item_id
    if not decision.decision:
        if decision.review_status != "pending":
            findings.append(_finding("pending_decision_marked_reviewed", "blank decision must stay pending", item_id, severity="error"))
        if require_all_reviewed:
            findings.append(_finding("pending_decision", "review item is still pending", item_id, severity="error"))
        return findings

    if decision.review_status != "reviewed":
        findings.append(_finding("completed_decision_not_marked_reviewed", "completed decision must set review_status=reviewed", item_id, severity="error"))
    if not decision.reviewer.strip():
        findings.append(_finding("missing_reviewer", "completed decision requires reviewer", item_id, severity="error"))
    if not decision.reviewed_at.strip():
        findings.append(_finding("missing_reviewed_at", "completed decision requires reviewed_at", item_id, severity="error"))
    elif not _is_iso_timestamp(decision.reviewed_at):
        findings.append(_finding("invalid_reviewed_at", "reviewed_at must be an ISO-8601 timestamp", item_id, severity="error"))
    if decision.decision in {"reject", "needs_more_context"} and not decision.comment.strip():
        findings.append(_finding("decision_comment_required", "reject and needs_more_context require a comment", item_id, severity="error"))
    if _has_corrections(decision) and decision.decision != "accept":
        findings.append(_finding("corrections_require_accept", "structured corrections may only accompany accept", item_id, severity="error"))
    if _has_corrections(decision) and not decision.comment.strip():
        findings.append(_finding("correction_comment_required", "accepted corrections require a comment", item_id, severity="error"))
    return findings


def _decision_outcome(
    decision: OledReviewerDecision,
    findings: list[OledReviewBridgeFinding],
) -> Literal[
    "pending",
    "accepted",
    "accepted_with_corrections",
    "rejected",
    "needs_more_context",
    "invalid",
]:
    if any(finding.severity == "error" for finding in findings):
        return "invalid"
    if not decision.decision:
        return "pending"
    if decision.decision == "accept":
        return "accepted_with_corrections" if _has_corrections(decision) else "accepted"
    if decision.decision == "reject":
        return "rejected"
    return "needs_more_context"


def _legacy_decision_entry(
    item: OledReviewItem,
    decision: OledReviewerDecision,
    packet: OledMineruReviewPacket,
) -> OledReviewDecisionEntry:
    corrections = _legacy_corrections(item, decision, packet)
    if decision.decision == "accept":
        review_decision = OledReviewDecision.NEEDS_CORRECTION if corrections else OledReviewDecision.ACCEPT
    elif decision.decision == "reject":
        review_decision = OledReviewDecision.REJECT
    elif decision.decision == "needs_more_context":
        review_decision = OledReviewDecision.NEEDS_SOURCE_CHECK
    else:
        review_decision = OledReviewDecision.UNREVIEWED
    return OledReviewDecisionEntry(
        packet_id=packet.packet_id,
        review_decision=review_decision,
        reviewer_notes=decision.comment.strip() or None,
        reviewer_id=decision.reviewer.strip() or None,
        corrections=corrections,
        metadata={
            "source_review_item_id": item.review_item_id,
            "reviewed_at": decision.reviewed_at,
        },
    )


def _legacy_corrections(
    item: OledReviewItem,
    decision: OledReviewerDecision,
    packet: OledMineruReviewPacket,
) -> list[OledReviewCorrection]:
    if not _has_corrections(decision):
        return []
    if decision.corrected_condition.strip():
        raise ValueError(f"corrected_condition_requires_manual_materialization:{item.review_item_id}")

    corrections: list[OledReviewCorrection] = []
    property_values = {
        "property_id": decision.corrected_property_id.strip(),
        "value": decision.corrected_value.strip(),
        "unit": decision.corrected_unit.strip(),
    }
    if any(property_values.values()):
        property_index = _property_index(item, packet)
        if property_index is None:
            raise ValueError(f"property_correction_target_not_found:{item.review_item_id}")
        review_property = packet.properties[property_index]
        correction_types = {
            "property_id": OledReviewCorrectionType.PROPERTY_LABEL,
            "value": OledReviewCorrectionType.PROPERTY_VALUE,
            "unit": OledReviewCorrectionType.PROPERTY_UNIT,
        }
        for field_name, proposed in property_values.items():
            if not proposed:
                continue
            corrections.append(
                OledReviewCorrection(
                    correction_type=correction_types[field_name],
                    field_path=f"properties[{property_index}].{field_name}",
                    original_value=getattr(review_property, field_name),
                    proposed_value=_numeric_if_possible(proposed) if field_name == "value" else proposed,
                    reason=decision.comment.strip() or None,
                    source_evidence_anchors=packet.source_evidence_anchors,
                )
            )

    if decision.corrected_compound.strip():
        if len(packet.material_roles) != 1:
            raise ValueError(f"compound_correction_target_ambiguous:{item.review_item_id}")
        corrections.append(
            OledReviewCorrection(
                correction_type=OledReviewCorrectionType.MATERIAL_NAME,
                field_path="material_roles[0].material_name",
                original_value=packet.material_roles[0].material_name,
                proposed_value=decision.corrected_compound.strip(),
                reason=decision.comment.strip() or None,
                source_evidence_anchors=packet.source_evidence_anchors,
            )
        )
    return corrections


def _property_index(item: OledReviewItem, packet: OledMineruReviewPacket) -> int | None:
    matching = [
        index
        for index, prop in enumerate(packet.properties)
        if (item.property_id and prop.property_id == item.property_id)
        or (item.property_label and prop.property_label == item.property_label)
    ]
    if len(matching) == 1:
        return matching[0]
    if not matching and len(packet.properties) == 1:
        return 0
    return None


def _has_corrections(decision: OledReviewerDecision) -> bool:
    return any(
        str(value or "").strip()
        for value in (
            decision.corrected_property_id,
            decision.corrected_value,
            decision.corrected_unit,
            decision.corrected_compound,
            decision.corrected_condition,
        )
    )


def _is_iso_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _numeric_if_possible(value: str) -> float | int | str:
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"missing_{label.replace(' ', '_')}_json")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_{label.replace(' ', '_')}_json") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_{label.replace(' ', '_')}_json")
    return payload


def _finding(
    code: str,
    message: str,
    review_item_id: str | None = None,
    *,
    severity: Literal["error", "warning"] = "warning",
) -> OledReviewBridgeFinding:
    return OledReviewBridgeFinding(
        code=code,
        severity=severity,
        message=message,
        review_item_id=review_item_id,
    )


def _bridge_markdown(report: OledReviewBridgeReport) -> str:
    lines = [
        "# OLED Human Review Readiness",
        "",
        f"- Run: `{report.run_id}`",
        f"- Status: `{report.status.value}`",
        f"- Review items: {report.item_count}",
        f"- Reviewed: {report.reviewed_count}",
        f"- Pending: {report.pending_count}",
        f"- Invalid: {report.invalid_count}",
        f"- Downstream-eligible compiled items: {report.downstream_eligible_count}",
        "",
        "## Reviewer Rules",
        "",
        "1. Compare every item with the original PDF evidence before deciding.",
        "2. Set `review_status` to `reviewed` for every completed item.",
        "3. Use only `accept`, `reject`, or `needs_more_context` in `decision`.",
        "4. Fill `reviewer` and ISO-8601 `reviewed_at` for every completed item.",
        "5. Add a comment for rejects, context requests, and accepted corrections.",
        "6. Do not infer missing values or silently repair evidence.",
        "",
        "Only accepted compiled-record items are eligible for the legacy adjudication/gold-candidate chain. Text, schema, and raw items remain extraction-quality evidence.",
        "",
        "## Counts",
        "",
    ]
    for label, counts in (
        ("Outcomes", report.outcome_counts),
        ("Candidate types", report.candidate_type_counts),
        ("Priorities", report.priority_counts),
    ):
        lines.extend([f"### {label}", ""])
        lines.extend(f"- `{key}`: {value}" for key, value in sorted(counts.items()))
        lines.append("")
    if report.findings:
        lines.extend(["## Findings", ""])
        for finding in report.findings:
            item_text = f" (`{finding.review_item_id}`)" if finding.review_item_id else ""
            lines.append(f"- **{finding.severity}** `{finding.code}`{item_text}: {finding.message}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "OledLegacyAdjudicationBundle",
    "OledReviewBridgeFinding",
    "OledReviewBridgeItem",
    "OledReviewBridgeReport",
    "OledReviewBridgeStatus",
    "build_legacy_adjudication_bundle",
    "evaluate_oled_review_decisions",
    "load_oled_compiled_records_json",
    "load_oled_review_packet_json",
    "load_oled_reviewer_decisions_json",
    "write_oled_review_bridge_report_json",
    "write_oled_review_bridge_report_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
