from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path
from ai4s_agent.domains.oled_mineru_review_adjudication import (
    OledAdjudicatedReviewPacket,
    OledReviewAdjudicationReport,
    OledReviewCorrection,
    OledReviewCorrectionType,
)
from ai4s_agent.domains.oled_mineru_review_packets import OledMineruReviewPacket, OledReviewDecision


class OledReviewedExtractionStatus(str, Enum):
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    NEEDS_SOURCE_CHECK = "needs_source_check"
    NEEDS_CORRECTION = "needs_correction"
    INVALID = "invalid"


class OledCorrectionApplicationStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    APPLIED = "applied"
    PARTIALLY_APPLIED = "partially_applied"
    NOT_APPLIED = "not_applied"
    FAILED = "failed"


class OledAppliedReviewCorrection(BaseModel):
    correction_type: OledReviewCorrectionType
    field_path: str
    original_value: float | int | str | list[str] | dict[str, Any] | None = None
    proposed_value: float | int | str | list[str] | dict[str, Any] | None = None
    application_status: OledCorrectionApplicationStatus
    reason: str | None = None
    finding_codes: list[str] = Field(default_factory=list)
    source_evidence_anchors: list[str] = Field(default_factory=list)

    @field_validator("finding_codes", "source_evidence_anchors")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledReviewedExtractionCandidate(BaseModel):
    candidate_id: str
    source_packet_id: str
    source_compiled_record_id: str
    paper_id: str
    source_label: str | None = None

    status: OledReviewedExtractionStatus
    review_decision: OledReviewDecision

    packet_snapshot: OledMineruReviewPacket
    corrected_packet_snapshot: OledMineruReviewPacket | None = None

    applied_corrections: list[OledAppliedReviewCorrection] = Field(default_factory=list)

    source_candidate_hashes: list[str] = Field(default_factory=list)
    source_evidence_anchors: list[str] = Field(default_factory=list)

    property_count: int = 0
    material_role_count: int = 0
    device_stack_count: int = 0

    schema_error_codes: list[str] = Field(default_factory=list)
    schema_warning_codes: list[str] = Field(default_factory=list)
    adjudication_finding_codes: list[str] = Field(default_factory=list)

    reviewer_id: str | None = None
    reviewer_notes: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_candidate_hashes",
        "source_evidence_anchors",
        "schema_error_codes",
        "schema_warning_codes",
        "adjudication_finding_codes",
    )
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledReviewedExtractionFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    source_packet_id: str | None = None
    candidate_id: str | None = None


class OledReviewedExtractionStagingReport(BaseModel):
    review_manifest_id: str
    source_packet_count: int
    staged_candidate_count: int

    status_counts: dict[str, int] = Field(default_factory=dict)
    correction_status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    reviewed_candidates: list[OledReviewedExtractionCandidate] = Field(default_factory=list)
    findings: list[OledReviewedExtractionFinding] = Field(default_factory=list)

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


def stage_oled_reviewed_extraction_candidates(
    adjudication_report: OledReviewAdjudicationReport,
    *,
    include_rejected: bool = False,
    include_needs_source_check: bool = False,
    apply_corrections: bool = True,
) -> OledReviewedExtractionStagingReport:
    reviewed_candidates: list[OledReviewedExtractionCandidate] = []
    findings: list[OledReviewedExtractionFinding] = []

    for adjudicated_packet in sorted(
        adjudication_report.adjudicated_packets,
        key=lambda item: item.packet.packet_id,
    ):
        packet_status = adjudicated_packet.adjudication_status
        if packet_status == "accepted":
            reviewed_candidates.append(
                _candidate_from_adjudicated_packet(
                    adjudicated_packet,
                    status=OledReviewedExtractionStatus.ACCEPTED,
                )
            )
            continue
        if packet_status == "needs_correction":
            candidate, correction_findings = _stage_needs_correction_packet(
                adjudicated_packet,
                apply_corrections=apply_corrections,
            )
            reviewed_candidates.append(candidate)
            findings.extend(correction_findings)
            continue
        if packet_status == "rejected":
            if include_rejected:
                reviewed_candidates.append(
                    _candidate_from_adjudicated_packet(
                        adjudicated_packet,
                        status=OledReviewedExtractionStatus.REJECTED,
                    )
                )
            else:
                findings.append(
                    _finding(
                        "rejected_packet_not_staged",
                        "rejected adjudicated packet was not staged",
                        adjudicated_packet.packet.packet_id,
                    )
                )
            continue
        if packet_status == "needs_source_check":
            if include_needs_source_check:
                reviewed_candidates.append(
                    _candidate_from_adjudicated_packet(
                        adjudicated_packet,
                        status=OledReviewedExtractionStatus.NEEDS_SOURCE_CHECK,
                    )
                )
            else:
                findings.append(
                    _finding(
                        "needs_source_check_not_staged",
                        "needs_source_check adjudicated packet was not staged",
                        adjudicated_packet.packet.packet_id,
                    )
                )
            continue
        if packet_status == "invalid":
            findings.append(
                _finding(
                    "invalid_packet_not_staged",
                    "invalid adjudicated packet was not staged",
                    adjudicated_packet.packet.packet_id,
                    severity="error",
                )
            )
            continue
        if packet_status == "unreviewed":
            findings.append(
                _finding(
                    "unreviewed_packet_not_staged",
                    "unreviewed adjudicated packet was not staged",
                    adjudicated_packet.packet.packet_id,
                )
            )

    status_counts = Counter(candidate.status.value for candidate in reviewed_candidates)
    correction_status_counts = Counter(
        correction.application_status.value
        for candidate in reviewed_candidates
        for correction in candidate.applied_corrections
    )
    finding_code_counts = Counter(finding.code for finding in findings)
    return OledReviewedExtractionStagingReport(
        review_manifest_id=adjudication_report.review_manifest_id,
        source_packet_count=adjudication_report.packet_count,
        staged_candidate_count=len(reviewed_candidates),
        status_counts=dict(sorted(status_counts.items())),
        correction_status_counts=dict(sorted(correction_status_counts.items())),
        finding_code_counts=dict(sorted(finding_code_counts.items())),
        reviewed_candidates=reviewed_candidates,
        findings=findings,
        metadata={
            "reviewed_extraction_candidate_only": True,
            "gold_records_created": False,
            "curated_dataset_written": False,
            "training_data_written": False,
            "llm_called": False,
            "mineru_called": False,
            "pdfs_read": False,
            "images_read": False,
            "model_backends_run": False,
            "apply_corrections": apply_corrections,
            "include_rejected": include_rejected,
            "include_needs_source_check": include_needs_source_check,
        },
    )


def apply_oled_review_corrections_to_packet(
    packet: OledMineruReviewPacket,
    corrections: Iterable[OledReviewCorrection],
) -> tuple[OledMineruReviewPacket, list[OledAppliedReviewCorrection], list[OledReviewedExtractionFinding]]:
    corrected_packet = packet.model_copy(deep=True)
    applied_corrections: list[OledAppliedReviewCorrection] = []
    findings: list[OledReviewedExtractionFinding] = []
    for correction in corrections:
        applied, correction_findings = _apply_single_correction(corrected_packet, correction, packet.packet_id)
        applied_corrections.append(applied)
        findings.extend(correction_findings)
    return corrected_packet, applied_corrections, findings


def load_oled_review_adjudication_report_json(path: str | Path) -> OledReviewAdjudicationReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_adjudication_report:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_adjudication_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    try:
        return OledReviewAdjudicationReport.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid_adjudication_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc


def write_oled_reviewed_extraction_candidates_jsonl(
    candidates: Iterable[OledReviewedExtractionCandidate],
    path: str | Path,
) -> None:
    lines = [
        json.dumps(
            _sanitize_for_output(candidate.model_dump(mode="json", exclude_none=True)),
            sort_keys=True,
            separators=(",", ":"),
        )
        for candidate in sorted(candidates, key=lambda item: item.candidate_id)
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_oled_reviewed_extraction_staging_report_json(
    report: OledReviewedExtractionStagingReport,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage OLED reviewed extraction candidates from adjudication reports.")
    parser.add_argument("--adjudication-report", required=True, help="Path to OLED review adjudication report JSON.")
    parser.add_argument("--output-candidates", help="Optional path to write reviewed candidates JSONL.")
    parser.add_argument("--output-report", help="Optional path to write staging report JSON.")
    parser.add_argument("--include-rejected", action="store_true", help="Include rejected packets in staging output.")
    parser.add_argument(
        "--include-needs-source-check",
        action="store_true",
        help="Include needs_source_check packets in staging output.",
    )
    parser.add_argument(
        "--no-apply-corrections",
        action="store_true",
        help="Stage needs_correction packets without applying correction proposals.",
    )
    args = parser.parse_args(argv)
    if not args.output_candidates and not args.output_report:
        print("output_required:candidates_or_report", file=sys.stderr)
        return 1
    try:
        adjudication_report = load_oled_review_adjudication_report_json(args.adjudication_report)
        staging_report = stage_oled_reviewed_extraction_candidates(
            adjudication_report,
            include_rejected=args.include_rejected,
            include_needs_source_check=args.include_needs_source_check,
            apply_corrections=not args.no_apply_corrections,
        )
        if args.output_candidates:
            write_oled_reviewed_extraction_candidates_jsonl(staging_report.reviewed_candidates, args.output_candidates)
        if args.output_report:
            write_oled_reviewed_extraction_staging_report_json(staging_report, args.output_report)
        summary = {
            "review_manifest_id": staging_report.review_manifest_id,
            "source_packet_count": staging_report.source_packet_count,
            "staged_candidate_count": staging_report.staged_candidate_count,
            "status_counts": staging_report.status_counts,
            "finding_code_counts": staging_report.finding_code_counts,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if staging_report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _stage_needs_correction_packet(
    adjudicated_packet: OledAdjudicatedReviewPacket,
    *,
    apply_corrections: bool,
) -> tuple[OledReviewedExtractionCandidate, list[OledReviewedExtractionFinding]]:
    corrections = adjudicated_packet.decision_entry.corrections if adjudicated_packet.decision_entry else []
    if not apply_corrections or not corrections:
        return (
            _candidate_from_adjudicated_packet(
                adjudicated_packet,
                status=OledReviewedExtractionStatus.NEEDS_CORRECTION,
            ),
            [],
        )
    corrected_packet, applied_corrections, findings = apply_oled_review_corrections_to_packet(
        adjudicated_packet.packet,
        corrections,
    )
    statuses = [correction.application_status for correction in applied_corrections]
    if statuses and all(status == OledCorrectionApplicationStatus.APPLIED for status in statuses):
        candidate_status = OledReviewedExtractionStatus.CORRECTED
    elif any(status == OledCorrectionApplicationStatus.APPLIED for status in statuses):
        candidate_status = OledReviewedExtractionStatus.NEEDS_CORRECTION
        _append_missing_status_finding(findings, "corrections_partially_applied", adjudicated_packet.packet.packet_id)
    else:
        candidate_status = OledReviewedExtractionStatus.NEEDS_CORRECTION
        _append_missing_status_finding(findings, "corrections_not_applied", adjudicated_packet.packet.packet_id)
    return (
        _candidate_from_adjudicated_packet(
            adjudicated_packet,
            status=candidate_status,
            corrected_packet=corrected_packet,
            applied_corrections=applied_corrections,
            extra_finding_codes=[finding.code for finding in findings],
        ),
        findings,
    )


def _candidate_from_adjudicated_packet(
    adjudicated_packet: OledAdjudicatedReviewPacket,
    *,
    status: OledReviewedExtractionStatus,
    corrected_packet: OledMineruReviewPacket | None = None,
    applied_corrections: list[OledAppliedReviewCorrection] | None = None,
    extra_finding_codes: list[str] | None = None,
) -> OledReviewedExtractionCandidate:
    packet = _sanitize_packet(adjudicated_packet.packet)
    corrected_snapshot = _sanitize_packet(corrected_packet) if corrected_packet is not None else None
    decision_entry = adjudicated_packet.decision_entry
    review_decision = decision_entry.review_decision if decision_entry is not None else packet.review_decision
    reviewer_notes = decision_entry.reviewer_notes if decision_entry is not None else packet.reviewer_notes
    reviewer_id = decision_entry.reviewer_id if decision_entry is not None else None
    finding_codes = sorted({*adjudicated_packet.finding_codes, *(extra_finding_codes or [])})
    effective_packet = corrected_snapshot or packet
    return OledReviewedExtractionCandidate(
        candidate_id=_candidate_id(packet.packet_id),
        source_packet_id=packet.packet_id,
        source_compiled_record_id=packet.compiled_record_id,
        paper_id=packet.paper_id,
        source_label=packet.source_label,
        status=status,
        review_decision=review_decision,
        packet_snapshot=packet,
        corrected_packet_snapshot=corrected_snapshot,
        applied_corrections=applied_corrections or [],
        source_candidate_hashes=packet.source_candidate_hashes,
        source_evidence_anchors=packet.source_evidence_anchors,
        property_count=len(effective_packet.properties),
        material_role_count=len(effective_packet.material_roles),
        device_stack_count=len(effective_packet.device_stack),
        schema_error_codes=packet.schema_error_codes,
        schema_warning_codes=packet.schema_warning_codes,
        adjudication_finding_codes=finding_codes,
        reviewer_id=reviewer_id,
        reviewer_notes=reviewer_notes,
        metadata={
            "reviewed_extraction_candidate_only": True,
            "gold_records_created": False,
            "curated_dataset_written": False,
            "training_data_written": False,
            "source_adjudication_status": adjudicated_packet.adjudication_status,
            "corrections_applied": bool(applied_corrections),
        },
    )


def _apply_single_correction(
    packet: OledMineruReviewPacket,
    correction: OledReviewCorrection,
    packet_id: str,
) -> tuple[OledAppliedReviewCorrection, list[OledReviewedExtractionFinding]]:
    findings: list[OledReviewedExtractionFinding] = []
    finding_codes: list[str] = []
    path = str(correction.field_path or "").strip()
    if not path:
        return (
            _applied_correction(correction, OledCorrectionApplicationStatus.FAILED, ["correction_missing_field_path"]),
            [
                _finding(
                    "correction_missing_field_path",
                    "review correction has no field_path",
                    packet_id,
                    severity="error",
                )
            ],
        )
    current_result = _get_supported_path_value(packet, path)
    if not current_result.supported:
        return (
            _applied_correction(correction, OledCorrectionApplicationStatus.FAILED, ["correction_field_path_unsupported"]),
            [
                _finding(
                    "correction_field_path_unsupported",
                    "correction field_path is not supported for deterministic application",
                    packet_id,
                    severity="error",
                )
            ],
        )
    if correction.proposed_value is None and correction.correction_type != OledReviewCorrectionType.NOTES_ONLY:
        return (
            _applied_correction(correction, OledCorrectionApplicationStatus.NOT_APPLIED, ["correction_missing_proposed_value"]),
            [
                _finding(
                    "correction_missing_proposed_value",
                    "correction has no proposed_value",
                    packet_id,
                    severity="error",
                )
            ],
        )
    if correction.original_value is not None and correction.original_value != current_result.value:
        findings.append(
            _finding(
                "correction_original_value_mismatch",
                "correction original_value does not match current packet value; proposed value was still applied",
                packet_id,
            )
        )
        finding_codes.append("correction_original_value_mismatch")
    try:
        _set_supported_path_value(packet, path, correction.proposed_value)
    except (TypeError, ValueError, IndexError) as exc:
        findings.append(
            _finding(
                "correction_application_failed",
                str(exc),
                packet_id,
                severity="error",
            )
        )
        finding_codes.append("correction_application_failed")
        return _applied_correction(correction, OledCorrectionApplicationStatus.FAILED, finding_codes), findings
    return _applied_correction(correction, OledCorrectionApplicationStatus.APPLIED, finding_codes), findings


class _PathValue(BaseModel):
    supported: bool
    value: Any = None


def _get_supported_path_value(packet: OledMineruReviewPacket, path: str) -> _PathValue:
    property_match = _INDEXED_FIELD_RE.fullmatch(path)
    if property_match:
        collection_name, raw_index, field_name = property_match.groups()
        index = int(raw_index)
        if collection_name == "properties" and field_name in {"value", "unit", "property_label", "property_id"}:
            if index >= len(packet.properties):
                return _PathValue(supported=False)
            return _PathValue(supported=True, value=getattr(packet.properties[index], field_name))
        if collection_name == "material_roles" and field_name in {"role", "material_name"}:
            if index >= len(packet.material_roles):
                return _PathValue(supported=False)
            return _PathValue(supported=True, value=getattr(packet.material_roles[index], field_name))
        return _PathValue(supported=False)
    stack_match = _DEVICE_STACK_INDEX_RE.fullmatch(path)
    if stack_match:
        index = int(stack_match.group(1))
        if index >= len(packet.device_stack):
            return _PathValue(supported=False)
        return _PathValue(supported=True, value=packet.device_stack[index])
    if path == "device_stack":
        return _PathValue(supported=True, value=list(packet.device_stack))
    if path == "reviewer_notes":
        return _PathValue(supported=True, value=packet.reviewer_notes)
    metadata_match = _METADATA_RE.fullmatch(path)
    if metadata_match:
        key = metadata_match.group(1)
        return _PathValue(supported=True, value=packet.metadata.get(key))
    return _PathValue(supported=False)


def _set_supported_path_value(packet: OledMineruReviewPacket, path: str, proposed_value: Any) -> None:
    property_match = _INDEXED_FIELD_RE.fullmatch(path)
    if property_match:
        collection_name, raw_index, field_name = property_match.groups()
        index = int(raw_index)
        if collection_name == "properties":
            value = _coerce_property_field(field_name, proposed_value)
            packet.properties[index] = packet.properties[index].model_copy(update={field_name: value})
            return
        if collection_name == "material_roles":
            value = str(proposed_value)
            packet.material_roles[index] = packet.material_roles[index].model_copy(update={field_name: value})
            return
    stack_match = _DEVICE_STACK_INDEX_RE.fullmatch(path)
    if stack_match:
        packet.device_stack[int(stack_match.group(1))] = str(proposed_value)
        return
    if path == "device_stack":
        if not isinstance(proposed_value, list):
            raise TypeError("device_stack correction requires list[str] proposed_value")
        packet.device_stack = [str(item) for item in proposed_value]
        return
    if path == "reviewer_notes":
        packet.reviewer_notes = None if proposed_value is None else str(proposed_value)
        return
    metadata_match = _METADATA_RE.fullmatch(path)
    if metadata_match:
        packet.metadata[metadata_match.group(1)] = proposed_value
        return
    raise ValueError("unsupported correction field_path")


def _coerce_property_field(field_name: str, value: Any) -> Any:
    if field_name == "value":
        if isinstance(value, bool):
            raise TypeError("property value cannot be bool")
        if value is None or isinstance(value, (float, int, str)):
            return value
        raise TypeError("property value must be float, int, str, or None")
    if field_name in {"unit", "property_label", "property_id"}:
        return None if value is None and field_name == "property_id" else str(value)
    raise ValueError("unsupported property field")


def _applied_correction(
    correction: OledReviewCorrection,
    status: OledCorrectionApplicationStatus,
    finding_codes: list[str] | None = None,
) -> OledAppliedReviewCorrection:
    return OledAppliedReviewCorrection(
        correction_type=correction.correction_type,
        field_path=correction.field_path,
        original_value=correction.original_value,
        proposed_value=correction.proposed_value,
        application_status=status,
        reason=correction.reason,
        finding_codes=finding_codes or [],
        source_evidence_anchors=correction.source_evidence_anchors,
    )


def _append_missing_status_finding(
    findings: list[OledReviewedExtractionFinding],
    code: str,
    packet_id: str,
) -> None:
    if code not in {finding.code for finding in findings}:
        findings.append(_finding(code, "not all review corrections were applied", packet_id))


def _sanitize_packet(packet: OledMineruReviewPacket | None) -> OledMineruReviewPacket | None:
    if packet is None:
        return None
    payload = _sanitize_for_output(packet.model_dump(mode="json", exclude_none=True))
    return OledMineruReviewPacket.model_validate(payload)


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


def _reject_forbidden_input(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"forbidden_pdf_input:{redact_oled_mineru_acceptance_path(path)}")
    if suffix in _FORBIDDEN_IMAGE_SUFFIXES:
        raise ValueError(f"forbidden_image_input:{redact_oled_mineru_acceptance_path(path)}")


def _finding(
    code: str,
    message: str,
    packet_id: str | None,
    *,
    severity: Literal["error", "warning"] = "warning",
    candidate_id: str | None = None,
) -> OledReviewedExtractionFinding:
    return OledReviewedExtractionFinding(
        code=code,
        severity=severity,
        message=message,
        source_packet_id=packet_id,
        candidate_id=candidate_id,
    )


def _candidate_id(packet_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", packet_id.strip())
    return f"reviewed-extraction:{safe_id}"


_INDEXED_FIELD_RE = re.compile(r"^(properties|material_roles)\[(\d+)\]\.([A-Za-z_][A-Za-z0-9_]*)$")
_DEVICE_STACK_INDEX_RE = re.compile(r"^device_stack\[(\d+)\]$")
_METADATA_RE = re.compile(r"^metadata\.([A-Za-z0-9_.:-]+)$")
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
    "OledReviewedExtractionStatus",
    "OledCorrectionApplicationStatus",
    "OledAppliedReviewCorrection",
    "OledReviewedExtractionCandidate",
    "OledReviewedExtractionFinding",
    "OledReviewedExtractionStagingReport",
    "stage_oled_reviewed_extraction_candidates",
    "apply_oled_review_corrections_to_packet",
    "load_oled_review_adjudication_report_json",
    "write_oled_reviewed_extraction_candidates_jsonl",
    "write_oled_reviewed_extraction_staging_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
