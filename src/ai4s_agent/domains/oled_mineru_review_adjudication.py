from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path
from ai4s_agent.domains.oled_mineru_review_packets import (
    OledMineruReviewPacket,
    OledReviewDecision,
)


class OledReviewCorrectionType(str, Enum):
    PROPERTY_VALUE = "property_value"
    PROPERTY_UNIT = "property_unit"
    PROPERTY_LABEL = "property_label"
    MATERIAL_ROLE = "material_role"
    MATERIAL_NAME = "material_name"
    DEVICE_STACK = "device_stack"
    MEASUREMENT_CONDITION = "measurement_condition"
    EVIDENCE_REF = "evidence_ref"
    NOTES_ONLY = "notes_only"


class OledReviewCorrection(BaseModel):
    correction_type: OledReviewCorrectionType
    field_path: str
    original_value: float | int | str | list[str] | dict[str, Any] | None = None
    proposed_value: float | int | str | list[str] | dict[str, Any] | None = None
    reason: str | None = None
    source_evidence_anchors: list[str] = Field(default_factory=list)

    @field_validator("source_evidence_anchors")
    @classmethod
    def validate_sorted_unique_anchors(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledReviewDecisionEntry(BaseModel):
    packet_id: str
    review_decision: OledReviewDecision
    reviewer_notes: str | None = None
    reviewer_id: str | None = None
    corrections: list[OledReviewCorrection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("packet_id")
    @classmethod
    def validate_packet_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("packet_id is required")
        return clean


class OledReviewDecisionManifest(BaseModel):
    review_manifest_id: str
    packet_source_label: str | None = None
    decisions: list[OledReviewDecisionEntry]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("review_manifest_id")
    @classmethod
    def validate_review_manifest_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("review_manifest_id is required")
        return clean


class OledAdjudicatedReviewPacket(BaseModel):
    packet: OledMineruReviewPacket
    decision_entry: OledReviewDecisionEntry | None = None

    adjudication_status: Literal[
        "accepted",
        "rejected",
        "needs_correction",
        "needs_source_check",
        "unreviewed",
        "invalid",
    ]

    finding_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("finding_codes")
    @classmethod
    def validate_sorted_unique_codes(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledReviewAdjudicationFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    packet_id: str | None = None


class OledReviewAdjudicationReport(BaseModel):
    review_manifest_id: str
    packet_count: int

    accepted_count: int
    rejected_count: int
    needs_correction_count: int
    needs_source_check_count: int
    unreviewed_count: int
    invalid_count: int

    decision_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    adjudicated_packets: list[OledAdjudicatedReviewPacket] = Field(default_factory=list)
    findings: list[OledReviewAdjudicationFinding] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_mineru_review_packets_jsonl(path: str | Path) -> list[OledMineruReviewPacket]:
    packet_path = Path(path)
    _reject_forbidden_input(packet_path)
    packets: list[OledMineruReviewPacket] = []
    try:
        lines = packet_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"missing_review_packets_jsonl:{redact_oled_mineru_acceptance_path(packet_path)}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            packet = OledMineruReviewPacket.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid_review_packet_jsonl:line-{line_number}") from exc
        _reject_absolute_paths_in_packet_metadata(packet)
        packets.append(packet)
    return packets


def load_oled_review_decision_manifest(path: str | Path) -> OledReviewDecisionManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_review_decision_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid_review_decision_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}"
        ) from exc
    try:
        return OledReviewDecisionManifest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"invalid_review_decision_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}"
        ) from exc


def adjudicate_oled_mineru_review_packets(
    packets: Iterable[OledMineruReviewPacket],
    *,
    decision_manifest: OledReviewDecisionManifest | None = None,
    require_all_reviewed: bool = False,
    allow_accept_with_schema_errors: bool = False,
    allow_accept_with_schema_warnings: bool = True,
) -> OledReviewAdjudicationReport:
    packet_list = sorted(list(packets), key=lambda packet: packet.packet_id)
    findings: list[OledReviewAdjudicationFinding] = []
    decision_counts: Counter[str] = Counter()

    duplicate_decision_ids: set[str] = set()
    decisions_by_id: dict[str, OledReviewDecisionEntry] = {}
    if decision_manifest is not None:
        seen_decision_ids: set[str] = set()
        packet_ids = {packet.packet_id for packet in packet_list}
        for decision in decision_manifest.decisions:
            if decision.packet_id in seen_decision_ids:
                duplicate_decision_ids.add(decision.packet_id)
                findings.append(
                    OledReviewAdjudicationFinding(
                        code="duplicate_decision_packet_id",
                        severity="error",
                        message="decision manifest contains duplicate packet id",
                        packet_id=decision.packet_id,
                    )
                )
            else:
                decisions_by_id[decision.packet_id] = decision
            seen_decision_ids.add(decision.packet_id)
            if decision.packet_id not in packet_ids:
                findings.append(
                    OledReviewAdjudicationFinding(
                        code="unknown_decision_packet_id",
                        severity="error",
                        message="decision manifest references a packet id that is not in the packet set",
                        packet_id=decision.packet_id,
                    )
                )

    adjudicated_packets: list[OledAdjudicatedReviewPacket] = []
    for packet in packet_list:
        decision_entry = decisions_by_id.get(packet.packet_id)
        has_explicit_manifest_decision = decision_manifest is not None and packet.packet_id in decisions_by_id
        if decision_entry is None:
            if decision_manifest is not None and require_all_reviewed:
                packet_findings = [
                    OledReviewAdjudicationFinding(
                        code="missing_review_decision",
                        severity="error",
                        message="packet has no decision entry in a required review manifest",
                        packet_id=packet.packet_id,
                    )
                ]
                findings.extend(packet_findings)
                adjudicated_packets.append(_adjudicated_packet(packet, None, "invalid", packet_findings))
                decision_counts.update([OledReviewDecision.UNREVIEWED.value])
                continue
            decision = packet.review_decision
            reviewer_notes = packet.reviewer_notes
            corrections: list[OledReviewCorrection] = []
        else:
            decision = decision_entry.review_decision
            reviewer_notes = decision_entry.reviewer_notes
            corrections = decision_entry.corrections

        packet_findings = _validate_packet_decision(
            packet,
            decision=decision,
            reviewer_notes=reviewer_notes,
            corrections=corrections,
            require_all_reviewed=require_all_reviewed,
            allow_accept_with_schema_errors=allow_accept_with_schema_errors,
            allow_accept_with_schema_warnings=allow_accept_with_schema_warnings,
        )
        if packet.packet_id in duplicate_decision_ids:
            packet_findings.append(
                OledReviewAdjudicationFinding(
                    code="duplicate_decision_packet_id",
                    severity="error",
                    message="packet has duplicate decision entries",
                    packet_id=packet.packet_id,
                )
            )
        findings.extend(packet_findings)
        status = _status_from_decision(decision, packet_findings)
        if not has_explicit_manifest_decision and decision_manifest is not None and not require_all_reviewed:
            decision_entry = None
        adjudicated_packets.append(_adjudicated_packet(packet, decision_entry, status, packet_findings))
        decision_counts.update([decision.value])

    status_counts = Counter(packet.adjudication_status for packet in adjudicated_packets)
    finding_code_counts = Counter(finding.code for finding in findings)
    return OledReviewAdjudicationReport(
        review_manifest_id=(
            decision_manifest.review_manifest_id
            if decision_manifest is not None
            else "embedded-review-decisions"
        ),
        packet_count=len(packet_list),
        accepted_count=status_counts["accepted"],
        rejected_count=status_counts["rejected"],
        needs_correction_count=status_counts["needs_correction"],
        needs_source_check_count=status_counts["needs_source_check"],
        unreviewed_count=status_counts["unreviewed"],
        invalid_count=status_counts["invalid"],
        decision_counts=dict(sorted(decision_counts.items())),
        finding_code_counts=dict(sorted(finding_code_counts.items())),
        adjudicated_packets=adjudicated_packets,
        findings=findings,
        metadata={
            "adjudication_gate_only": True,
            "gold_records_created": False,
            "curated_dataset_written": False,
            "llm_called": False,
            "mineru_called": False,
            "pdfs_read": False,
            "images_read": False,
            "model_backends_run": False,
            "require_all_reviewed": require_all_reviewed,
            "allow_accept_with_schema_errors": allow_accept_with_schema_errors,
            "allow_accept_with_schema_warnings": allow_accept_with_schema_warnings,
        },
    )


def filter_adjudicated_oled_review_packets(
    report: OledReviewAdjudicationReport,
    *,
    statuses: Iterable[str],
) -> list[OledAdjudicatedReviewPacket]:
    allowed_statuses = {str(status).strip() for status in statuses if str(status).strip()}
    return [
        adjudicated_packet
        for adjudicated_packet in report.adjudicated_packets
        if adjudicated_packet.adjudication_status in allowed_statuses
    ]


def write_oled_review_adjudication_report_json(
    report: OledReviewAdjudicationReport,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adjudicate OLED MinerU review packet decisions.")
    parser.add_argument("--packets-jsonl", required=True, help="Path to review packet JSONL.")
    parser.add_argument("--decisions", help="Optional path to review decision manifest JSON.")
    parser.add_argument("--output-report", required=True, help="Path to write adjudication report JSON.")
    parser.add_argument("--require-all-reviewed", action="store_true", help="Require a reviewed decision for every packet.")
    parser.add_argument(
        "--allow-accept-with-schema-errors",
        action="store_true",
        help="Allow accepted packets that still carry schema error codes.",
    )
    parser.add_argument(
        "--disallow-accept-with-schema-warnings",
        action="store_true",
        help="Treat accepted packets with schema warnings as errors.",
    )
    args = parser.parse_args(argv)
    try:
        packets = load_oled_mineru_review_packets_jsonl(args.packets_jsonl)
        decision_manifest = load_oled_review_decision_manifest(args.decisions) if args.decisions else None
        report = adjudicate_oled_mineru_review_packets(
            packets,
            decision_manifest=decision_manifest,
            require_all_reviewed=args.require_all_reviewed,
            allow_accept_with_schema_errors=args.allow_accept_with_schema_errors,
            allow_accept_with_schema_warnings=not args.disallow_accept_with_schema_warnings,
        )
        write_oled_review_adjudication_report_json(report, args.output_report)
        summary = {
            "review_manifest_id": report.review_manifest_id,
            "packet_count": report.packet_count,
            "accepted_count": report.accepted_count,
            "invalid_count": report.invalid_count,
            "finding_code_counts": report.finding_code_counts,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _validate_packet_decision(
    packet: OledMineruReviewPacket,
    *,
    decision: OledReviewDecision,
    reviewer_notes: str | None,
    corrections: list[OledReviewCorrection],
    require_all_reviewed: bool,
    allow_accept_with_schema_errors: bool,
    allow_accept_with_schema_warnings: bool,
) -> list[OledReviewAdjudicationFinding]:
    findings: list[OledReviewAdjudicationFinding] = []
    has_notes = bool(str(reviewer_notes or "").strip())
    if require_all_reviewed and decision == OledReviewDecision.UNREVIEWED:
        findings.append(
            _finding(
                "unreviewed_packet",
                "packet remains unreviewed while require_all_reviewed is enabled",
                packet.packet_id,
                severity="error",
            )
        )
    if decision == OledReviewDecision.ACCEPT:
        if packet.schema_error_codes and not allow_accept_with_schema_errors:
            findings.append(
                _finding(
                    "accept_with_schema_errors",
                    "accepted packet still has schema error codes",
                    packet.packet_id,
                    severity="error",
                )
            )
        if packet.schema_warning_codes and not allow_accept_with_schema_warnings:
            findings.append(
                _finding(
                    "accept_with_schema_warnings",
                    "accepted packet still has schema warning codes",
                    packet.packet_id,
                    severity="error",
                )
            )
    if decision == OledReviewDecision.REJECT and not has_notes and not corrections:
        findings.append(_finding("reject_without_notes", "rejected packet has no notes or corrections", packet.packet_id))
    if decision == OledReviewDecision.NEEDS_CORRECTION and not corrections:
        findings.append(
            _finding(
                "needs_correction_without_structured_corrections",
                "packet marked needs_correction without structured corrections",
                packet.packet_id,
            )
        )
    if decision == OledReviewDecision.NEEDS_SOURCE_CHECK and not has_notes:
        findings.append(
            _finding("source_check_without_notes", "packet marked needs_source_check without reviewer notes", packet.packet_id)
        )
    for correction in corrections:
        if not str(correction.field_path or "").strip():
            findings.append(
                _finding(
                    "correction_missing_field_path",
                    "review correction has no field_path",
                    packet.packet_id,
                    severity="error",
                )
            )
        if correction.original_value is None and correction.proposed_value is None:
            findings.append(
                _finding(
                    "correction_without_values",
                    "review correction has neither original_value nor proposed_value",
                    packet.packet_id,
                )
            )
    return findings


def _status_from_decision(
    decision: OledReviewDecision,
    findings: list[OledReviewAdjudicationFinding],
) -> Literal["accepted", "rejected", "needs_correction", "needs_source_check", "unreviewed", "invalid"]:
    if any(finding.severity == "error" for finding in findings):
        return "invalid"
    return {
        OledReviewDecision.ACCEPT: "accepted",
        OledReviewDecision.REJECT: "rejected",
        OledReviewDecision.NEEDS_CORRECTION: "needs_correction",
        OledReviewDecision.NEEDS_SOURCE_CHECK: "needs_source_check",
        OledReviewDecision.UNREVIEWED: "unreviewed",
    }[decision]


def _adjudicated_packet(
    packet: OledMineruReviewPacket,
    decision_entry: OledReviewDecisionEntry | None,
    status: Literal["accepted", "rejected", "needs_correction", "needs_source_check", "unreviewed", "invalid"],
    findings: list[OledReviewAdjudicationFinding],
) -> OledAdjudicatedReviewPacket:
    return OledAdjudicatedReviewPacket(
        packet=packet,
        decision_entry=decision_entry,
        adjudication_status=status,
        finding_codes=[finding.code for finding in findings],
        metadata={
            "adjudication_gate_only": True,
            "eligible_for_later_conversion": status == "accepted",
            "gold_record_created": False,
            "curated_dataset_written": False,
        },
    )


def _finding(
    code: str,
    message: str,
    packet_id: str | None,
    *,
    severity: Literal["error", "warning"] = "warning",
) -> OledReviewAdjudicationFinding:
    return OledReviewAdjudicationFinding(
        code=code,
        severity=severity,
        message=message,
        packet_id=packet_id,
    )


def _reject_forbidden_input(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"forbidden_pdf_input:{redact_oled_mineru_acceptance_path(path)}")
    if suffix in _FORBIDDEN_IMAGE_SUFFIXES:
        raise ValueError(f"forbidden_image_input:{redact_oled_mineru_acceptance_path(path)}")


def _reject_absolute_paths_in_packet_metadata(packet: OledMineruReviewPacket) -> None:
    if _contains_absolute_path(packet.metadata):
        raise ValueError(f"absolute_path_in_review_packet_metadata:{packet.packet_id}")


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute()
    return False


def _sanitize_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _is_forbidden_payload_key(key):
                continue
            sanitized_value = _sanitize_for_output(raw_value)
            if sanitized_value in (None, {}, []):
                continue
            output[key] = sanitized_value
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            sanitized_item = _sanitize_for_output(item)
            if sanitized_item not in (None, {}, []):
                output.append(sanitized_item)
        return output
    if isinstance(value, tuple):
        output = []
        for item in value:
            sanitized_item = _sanitize_for_output(item)
            if sanitized_item not in (None, {}, []):
                output.append(sanitized_item)
        return output
    if isinstance(value, str):
        if Path(value).is_absolute():
            return redact_oled_mineru_acceptance_path(value)
        if len(value) > _MAX_REVIEW_STRING_LENGTH:
            return value[: _MAX_REVIEW_STRING_LENGTH - 3] + "..."
        return value
    return value


def _is_forbidden_payload_key(key: str) -> bool:
    normalized = key.lower()
    return any(
        token in normalized
        for token in (
            "raw_text",
            "full_text",
            "parsed_json",
            "table_body",
            "html_table",
            "markdown_table",
        )
    )


_MAX_REVIEW_STRING_LENGTH = 240

_FORBIDDEN_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".svg",
}


__all__ = [
    "OledReviewCorrectionType",
    "OledReviewCorrection",
    "OledReviewDecisionEntry",
    "OledReviewDecisionManifest",
    "OledAdjudicatedReviewPacket",
    "OledReviewAdjudicationFinding",
    "OledReviewAdjudicationReport",
    "load_oled_mineru_review_packets_jsonl",
    "load_oled_review_decision_manifest",
    "adjudicate_oled_mineru_review_packets",
    "filter_adjudicated_oled_review_packets",
    "write_oled_review_adjudication_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
