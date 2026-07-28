from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_gold_validation import (
    OledGoldDatasetRecord,
    validate_oled_gold_dataset,
)
from ai4s_agent.domains.oled_layered_schema import (
    OledConfidenceAssessment,
    OledDeviceLayer,
    OledEvidenceSource,
    OledEvidenceType,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path
from ai4s_agent.domains.oled_mineru_review_packets import (
    OledMineruReviewPacket,
    OledReviewPacketProperty,
    OledReviewPacketSourceRef,
)
from ai4s_agent.domains.oled_reviewed_extraction_candidates import (
    OledReviewedExtractionCandidate,
    OledReviewedExtractionStatus,
)


class OledReviewedGoldCandidateStatus(str, Enum):
    CONVERTED = "converted"
    CONVERTED_WITH_WARNINGS = "converted_with_warnings"
    REJECTED = "rejected"
    INVALID = "invalid"
    SKIPPED = "skipped"


class OledReviewedGoldConversionPolicy(BaseModel):
    include_corrected: bool = True
    include_accepted: bool = True
    include_rejected: bool = False
    include_needs_source_check: bool = False
    require_no_schema_errors: bool = True
    require_no_adjudication_errors: bool = True
    require_evidence_anchors: bool = True
    require_review_decision: bool = True
    candidate_only: bool = True


class OledReviewedGoldCandidate(BaseModel):
    candidate_id: str
    status: OledReviewedGoldCandidateStatus

    source_reviewed_candidate_id: str
    source_packet_id: str
    source_compiled_record_id: str
    paper_id: str
    source_label: str | None = None

    gold_record: OledGoldDatasetRecord | None = None

    source_candidate_hashes: list[str] = Field(default_factory=list)
    source_evidence_anchors: list[str] = Field(default_factory=list)

    validation_error_codes: list[str] = Field(default_factory=list)
    validation_warning_codes: list[str] = Field(default_factory=list)

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_candidate_hashes",
        "source_evidence_anchors",
        "validation_error_codes",
        "validation_warning_codes",
        "reason_codes",
    )
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledReviewedGoldConversionFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    source_reviewed_candidate_id: str | None = None
    gold_candidate_id: str | None = None


class OledReviewedGoldConversionReport(BaseModel):
    source_reviewed_candidate_count: int
    converted_candidate_count: int

    status_counts: dict[str, int] = Field(default_factory=dict)
    validation_code_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    gold_candidates: list[OledReviewedGoldCandidate] = Field(default_factory=list)
    findings: list[OledReviewedGoldConversionFinding] = Field(default_factory=list)

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


def convert_reviewed_extractions_to_gold_candidates(
    reviewed_candidates: Iterable[OledReviewedExtractionCandidate],
    *,
    policy: OledReviewedGoldConversionPolicy | None = None,
) -> OledReviewedGoldConversionReport:
    conversion_policy = policy or OledReviewedGoldConversionPolicy()
    source_candidates = sorted(list(reviewed_candidates), key=lambda candidate: candidate.candidate_id)
    gold_candidates: list[OledReviewedGoldCandidate] = []
    findings: list[OledReviewedGoldConversionFinding] = []
    validation_targets: list[tuple[OledReviewedGoldCandidate, OledGoldDatasetRecord]] = []

    for reviewed_candidate in source_candidates:
        gate_status, gate_reason, gate_findings = _policy_gate(reviewed_candidate, conversion_policy)
        findings.extend(gate_findings)
        if gate_status is not None:
            gold_candidates.append(
                _gold_candidate_shell(
                    reviewed_candidate,
                    status=gate_status,
                    reason_codes=gate_reason,
                )
            )
            continue

        try:
            gold_record = build_gold_record_candidate_from_reviewed_extraction(reviewed_candidate)
        except (TypeError, ValueError, ValidationError) as exc:
            candidate = _gold_candidate_shell(
                reviewed_candidate,
                status=OledReviewedGoldCandidateStatus.INVALID,
                reason_codes=["gold_record_candidate_build_failed"],
            )
            findings.append(
                OledReviewedGoldConversionFinding(
                    code="gold_record_candidate_build_failed",
                    severity="error",
                    message=str(exc).splitlines()[0],
                    source_reviewed_candidate_id=reviewed_candidate.candidate_id,
                    gold_candidate_id=candidate.candidate_id,
                )
            )
            gold_candidates.append(candidate)
            continue
        converted = _gold_candidate_shell(
            reviewed_candidate,
            status=OledReviewedGoldCandidateStatus.CONVERTED,
            gold_record=gold_record,
            reason_codes=["gold_record_candidate_built"],
        )
        gold_candidates.append(converted)
        validation_targets.append((converted, gold_record))

    validation_report = validate_oled_gold_dataset(record for _, record in validation_targets)
    validation_errors_by_record: dict[str, list[str]] = defaultdict(list)
    validation_warnings_by_record: dict[str, list[str]] = defaultdict(list)
    for finding in validation_report.findings:
        if finding.severity == "error":
            validation_errors_by_record[finding.record_id].append(finding.code)
            severity: Literal["error", "warning"] = "error"
        else:
            validation_warnings_by_record[finding.record_id].append(finding.code)
            severity = "warning"
        candidate = next(
            (candidate for candidate, record in validation_targets if record.record_id == finding.record_id),
            None,
        )
        findings.append(
            OledReviewedGoldConversionFinding(
                code=finding.code,
                severity=severity,
                message=finding.message,
                source_reviewed_candidate_id=(candidate.source_reviewed_candidate_id if candidate else None),
                gold_candidate_id=(candidate.candidate_id if candidate else None),
            )
        )

    refreshed_candidates: list[OledReviewedGoldCandidate] = []
    for candidate in gold_candidates:
        if candidate.gold_record is None:
            refreshed_candidates.append(candidate)
            continue
        validation_error_codes = sorted(set(validation_errors_by_record.get(candidate.gold_record.record_id, [])))
        validation_warning_codes = sorted(set(validation_warnings_by_record.get(candidate.gold_record.record_id, [])))
        if validation_error_codes:
            status = OledReviewedGoldCandidateStatus.INVALID
        elif validation_warning_codes:
            status = OledReviewedGoldCandidateStatus.CONVERTED_WITH_WARNINGS
        else:
            status = OledReviewedGoldCandidateStatus.CONVERTED
        refreshed_candidates.append(
            candidate.model_copy(
                update={
                    "status": status,
                    "validation_error_codes": validation_error_codes,
                    "validation_warning_codes": validation_warning_codes,
                    "reason_codes": sorted(
                        {
                            *candidate.reason_codes,
                            *("gold_validation_errors" for _ in validation_error_codes[:1]),
                            *("gold_validation_warnings" for _ in validation_warning_codes[:1]),
                        }
                    ),
                }
            )
        )
    status_counts = Counter(candidate.status.value for candidate in refreshed_candidates)
    validation_code_counts = Counter(
        code
        for candidate in refreshed_candidates
        for code in [*candidate.validation_error_codes, *candidate.validation_warning_codes]
    )
    finding_code_counts = Counter(finding.code for finding in findings)
    return OledReviewedGoldConversionReport(
        source_reviewed_candidate_count=len(source_candidates),
        converted_candidate_count=sum(1 for candidate in refreshed_candidates if candidate.gold_record is not None),
        status_counts=dict(sorted(status_counts.items())),
        validation_code_counts=dict(sorted(validation_code_counts.items())),
        finding_code_counts=dict(sorted(finding_code_counts.items())),
        gold_candidates=refreshed_candidates,
        findings=findings,
        metadata={
            "candidate_only": conversion_policy.candidate_only,
            "curated_dataset_written": False,
            "training_data_written": False,
            "final_gold_dataset": False,
            "gold_validation_ran": True,
            "dataset_views_run": False,
            "leakage_splits_run": False,
            "model_backends_run": False,
            "llm_called": False,
            "mineru_called": False,
            "pdfs_read": False,
            "images_read": False,
        },
    )


def effective_reviewed_extraction_packet(
    candidate: OledReviewedExtractionCandidate,
) -> OledMineruReviewPacket:
    return candidate.corrected_packet_snapshot or candidate.packet_snapshot


def build_gold_record_candidate_from_reviewed_extraction(
    candidate: OledReviewedExtractionCandidate,
) -> OledGoldDatasetRecord:
    packet = effective_reviewed_extraction_packet(candidate)
    layered_record = _layered_record_from_packet(packet, candidate)
    return OledGoldDatasetRecord(
        record_id=_gold_record_id(candidate),
        layered_record=layered_record,
        evidence_refs=list(candidate.source_evidence_anchors or packet.source_evidence_anchors),
        reviewer=candidate.reviewer_id,
        notes=candidate.reviewer_notes,
        metadata={
            "candidate_only": True,
            "curated_dataset_written": False,
            "training_data_written": False,
            "final_gold_dataset": False,
            "source_reviewed_candidate_id": candidate.candidate_id,
            "source_packet_id": candidate.source_packet_id,
            "source_compiled_record_id": candidate.source_compiled_record_id,
            "paper_id": candidate.paper_id,
            "source_label": candidate.source_label,
            "review_decision": candidate.review_decision.value,
            "reviewed_extraction_status": candidate.status.value,
            "applied_correction_statuses": [correction.application_status.value for correction in candidate.applied_corrections],
            "source_candidate_hashes": list(candidate.source_candidate_hashes),
            "source_evidence_anchors": list(candidate.source_evidence_anchors),
        },
    )


def load_oled_reviewed_extraction_candidates_jsonl(
    path: str | Path,
) -> list[OledReviewedExtractionCandidate]:
    reviewed_path = Path(path)
    _reject_forbidden_input(reviewed_path)
    if not reviewed_path.exists():
        raise ValueError(f"missing_reviewed_candidate_jsonl:{redact_oled_mineru_acceptance_path(reviewed_path)}")
    candidates: list[OledReviewedExtractionCandidate] = []
    for line_number, raw_line in enumerate(reviewed_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            candidate = OledReviewedExtractionCandidate.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid_reviewed_candidate_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(candidate.metadata):
            raise ValueError(f"absolute_path_in_reviewed_candidate_metadata:{candidate.candidate_id}")
        candidates.append(candidate)
    return candidates


def write_oled_reviewed_gold_candidates_jsonl(
    candidates: Iterable[OledReviewedGoldCandidate],
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


def write_oled_reviewed_gold_conversion_report_json(
    report: OledReviewedGoldConversionReport,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert OLED reviewed extraction candidates to gold candidates.")
    parser.add_argument("--reviewed-candidates", required=True, help="Path to reviewed extraction candidates JSONL.")
    parser.add_argument("--output-candidates", help="Optional path to write gold candidates JSONL.")
    parser.add_argument("--output-report", help="Optional path to write conversion report JSON.")
    args = parser.parse_args(argv)
    if not args.output_candidates and not args.output_report:
        print("output_required:candidates_or_report", file=sys.stderr)
        return 1
    try:
        reviewed_candidates = load_oled_reviewed_extraction_candidates_jsonl(args.reviewed_candidates)
        report = convert_reviewed_extractions_to_gold_candidates(reviewed_candidates)
        if args.output_candidates:
            write_oled_reviewed_gold_candidates_jsonl(report.gold_candidates, args.output_candidates)
        if args.output_report:
            write_oled_reviewed_gold_conversion_report_json(report, args.output_report)
        summary = {
            "source_reviewed_candidate_count": report.source_reviewed_candidate_count,
            "converted_candidate_count": report.converted_candidate_count,
            "status_counts": report.status_counts,
            "validation_code_counts": report.validation_code_counts,
            "finding_code_counts": report.finding_code_counts,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _policy_gate(
    candidate: OledReviewedExtractionCandidate,
    policy: OledReviewedGoldConversionPolicy,
) -> tuple[OledReviewedGoldCandidateStatus | None, list[str], list[OledReviewedGoldConversionFinding]]:
    findings: list[OledReviewedGoldConversionFinding] = []
    if candidate.status == OledReviewedExtractionStatus.ACCEPTED and not policy.include_accepted:
        return OledReviewedGoldCandidateStatus.SKIPPED, ["status_not_eligible"], [
            _finding("status_not_eligible", "accepted reviewed candidate excluded by policy", candidate)
        ]
    if candidate.status == OledReviewedExtractionStatus.CORRECTED and not policy.include_corrected:
        return OledReviewedGoldCandidateStatus.SKIPPED, ["status_not_eligible"], [
            _finding("status_not_eligible", "corrected reviewed candidate excluded by policy", candidate)
        ]
    if candidate.status == OledReviewedExtractionStatus.REJECTED:
        if policy.include_rejected:
            return OledReviewedGoldCandidateStatus.REJECTED, ["source_status_rejected"], []
        return OledReviewedGoldCandidateStatus.SKIPPED, ["status_not_eligible"], [
            _finding("status_not_eligible", "rejected reviewed candidate excluded by policy", candidate)
        ]
    if candidate.status == OledReviewedExtractionStatus.NEEDS_SOURCE_CHECK:
        if policy.include_needs_source_check:
            return OledReviewedGoldCandidateStatus.SKIPPED, ["source_status_needs_source_check"], []
        return OledReviewedGoldCandidateStatus.SKIPPED, ["status_not_eligible"], [
            _finding("status_not_eligible", "needs_source_check reviewed candidate excluded by policy", candidate)
        ]
    if candidate.status in {OledReviewedExtractionStatus.NEEDS_CORRECTION, OledReviewedExtractionStatus.INVALID}:
        return OledReviewedGoldCandidateStatus.SKIPPED, ["status_not_eligible"], [
            _finding("status_not_eligible", f"{candidate.status.value} reviewed candidate excluded by policy", candidate)
        ]
    if policy.require_no_schema_errors and candidate.schema_error_codes:
        return OledReviewedGoldCandidateStatus.REJECTED, ["source_schema_errors_present"], [
            _finding(
                "source_schema_errors_present",
                "reviewed candidate carries schema error codes",
                candidate,
                severity="error",
            )
        ]
    if policy.require_no_adjudication_errors and _has_error_like_adjudication_code(candidate):
        return OledReviewedGoldCandidateStatus.REJECTED, ["source_adjudication_errors_present"], [
            _finding(
                "source_adjudication_errors_present",
                "reviewed candidate carries adjudication error codes",
                candidate,
                severity="error",
            )
        ]
    if policy.require_evidence_anchors and not candidate.source_evidence_anchors:
        return OledReviewedGoldCandidateStatus.REJECTED, ["source_evidence_anchors_missing"], [
            _finding(
                "source_evidence_anchors_missing",
                "reviewed candidate has no source evidence anchors",
                candidate,
                severity="error",
            )
        ]
    if policy.require_review_decision and not candidate.review_decision:
        return OledReviewedGoldCandidateStatus.REJECTED, ["review_decision_missing"], [
            _finding("review_decision_missing", "reviewed candidate has no review decision", candidate, severity="error")
        ]
    return None, [], findings


def _gold_candidate_shell(
    candidate: OledReviewedExtractionCandidate,
    *,
    status: OledReviewedGoldCandidateStatus,
    gold_record: OledGoldDatasetRecord | None = None,
    reason_codes: list[str] | None = None,
) -> OledReviewedGoldCandidate:
    return OledReviewedGoldCandidate(
        candidate_id=_gold_candidate_id(candidate),
        status=status,
        source_reviewed_candidate_id=candidate.candidate_id,
        source_packet_id=candidate.source_packet_id,
        source_compiled_record_id=candidate.source_compiled_record_id,
        paper_id=candidate.paper_id,
        source_label=candidate.source_label,
        gold_record=gold_record,
        source_candidate_hashes=candidate.source_candidate_hashes,
        source_evidence_anchors=candidate.source_evidence_anchors,
        reason_codes=reason_codes or [],
        metadata={
            "candidate_only": True,
            "curated_dataset_written": False,
            "training_data_written": False,
            "final_gold_dataset": False,
            "source_reviewed_extraction_status": candidate.status.value,
            "review_decision": candidate.review_decision.value,
        },
    )


def _layered_record_from_packet(
    packet: OledMineruReviewPacket,
    candidate: OledReviewedExtractionCandidate,
) -> OledLayeredRecord:
    molecule: OledMolecularLayer | None = None
    interaction: OledInteractionLayer | None = None
    device: OledDeviceLayer | None = None
    measurement: OledMeasurementLayer | None = None

    if packet.material_roles:
        interaction = interaction or OledInteractionLayer()
        for material_role in packet.material_roles:
            role = str(material_role.role).strip()
            name = str(material_role.material_name).strip()
            if not role or not name:
                continue
            role_key = role.lower()
            interaction.metadata.setdefault("material_roles", {})[role_key] = name
            interaction.metadata.setdefault("material_role_candidates", []).append(
                {
                    "role": role_key,
                    "material_name": name,
                    "evidence_refs": [ref.model_dump(mode="json", exclude_none=True) for ref in material_role.evidence_refs],
                }
            )
            if role_key == "host":
                interaction.metadata["host_name"] = name
            elif role_key in {"emitter", "emitter_dopant", "fluorescent_emitter"}:
                interaction.metadata["emitter_name"] = name
            elif role_key in {"assistant_dopant", "dopant", "tadf_assistant"}:
                interaction.metadata["assistant_dopant_name"] = name
            else:
                interaction.metadata[f"{role_key}_name"] = name

    if packet.device_stack:
        device = device or OledDeviceLayer()
        device.device_stack = [str(item) for item in packet.device_stack]
        device.metadata.update(
            {
                "raw_stack_preserved": True,
                "normalization_policy": "no_device_material_normalization",
            }
        )

    for review_property in packet.properties:
        observation = _property_observation(review_property, candidate)
        layer = _layer_from_review_property(review_property.layer)
        if layer == OledCausalLayer.MOLECULE:
            molecule = molecule or OledMolecularLayer()
            molecule.properties.append(observation)
        elif layer == OledCausalLayer.INTERACTION:
            interaction = interaction or OledInteractionLayer()
            interaction.properties.append(observation)
        elif layer == OledCausalLayer.DEVICE:
            device = device or OledDeviceLayer()
            device.properties.append(observation)
        elif layer == OledCausalLayer.MEASUREMENT:
            measurement = measurement or OledMeasurementLayer()
            measurement.measurements.append(observation)

    common_metadata = {
        "source_reviewed_candidate_id": candidate.candidate_id,
        "source_packet_id": candidate.source_packet_id,
        "paper_id": candidate.paper_id,
        "source_label": candidate.source_label,
        "candidate_only": True,
    }
    if interaction is not None and molecule is None:
        molecule = OledMolecularLayer(metadata={"context_only": True})
    for layer_model in (molecule, interaction, device, measurement):
        if layer_model is not None:
            layer_model.metadata.update({key: value for key, value in common_metadata.items() if value is not None})
    return OledLayeredRecord(
        molecule=molecule,
        interaction=interaction,
        device=device,
        measurement=measurement,
    )


def _property_observation(
    review_property: OledReviewPacketProperty,
    candidate: OledReviewedExtractionCandidate,
) -> OledPropertyObservation:
    layer = _layer_from_review_property(review_property.layer)
    return OledPropertyObservation(
        property_label=review_property.property_label,
        value=review_property.value,
        unit=review_property.unit,
        reported_value_text=review_property.reported_value_text,
        reported_decimal_places=review_property.reported_decimal_places,
        condition=_condition_from_summary(review_property.condition_summary),
        evidence_sources=_evidence_sources(review_property, layer, candidate),
        confidence=OledConfidenceAssessment(
            score=review_property.confidence_score if review_property.confidence_score is not None else 0.5,
            rationale=["converted from reviewed extraction candidate"],
        ),
        metadata={
            "source_property_id": review_property.property_id,
            "source_reviewed_candidate_id": candidate.candidate_id,
            "source_packet_id": candidate.source_packet_id,
            "candidate_only": True,
        },
    )


def _condition_from_summary(summary: dict[str, Any]) -> OledMeasurementCondition | None:
    if not summary:
        return None
    updates: dict[str, Any] = {}
    for field_name in ("luminance_cd_m2", "current_density_ma_cm2", "voltage_v", "temperature_k", "condition_label", "atmosphere"):
        if field_name in summary:
            updates[field_name] = summary[field_name]
    if "metadata" in summary and isinstance(summary["metadata"], dict):
        updates["metadata"] = summary["metadata"]
    return OledMeasurementCondition(**updates) if updates else None


def _evidence_sources(
    review_property: OledReviewPacketProperty,
    layer: OledCausalLayer,
    candidate: OledReviewedExtractionCandidate,
) -> list[OledEvidenceSource]:
    refs = list(review_property.evidence_refs)
    if not refs and candidate.source_evidence_anchors:
        refs = [
            OledReviewPacketSourceRef(
                source_candidate_hash=(candidate.source_candidate_hashes[0] if candidate.source_candidate_hashes else "reviewed"),
                source_evidence_anchor=anchor,
                source_candidate_type="manual_review",
            )
            for anchor in candidate.source_evidence_anchors
        ]
    return [
        OledEvidenceSource(
            source_id=f"{ref.source_candidate_hash}:{ref.source_evidence_anchor}",
            source_type=_evidence_type(ref.source_candidate_type),
            layer=layer,
            locator=ref.source_evidence_anchor,
            metadata={
                "source_reviewed_candidate_id": candidate.candidate_id,
                "source_packet_id": candidate.source_packet_id,
                "row_index": ref.row_index,
                "column_name": ref.column_name,
                "field_name": ref.field_name,
                "candidate_only": True,
            },
        )
        for ref in refs
    ]


def _layer_from_review_property(value: str) -> OledCausalLayer:
    normalized = str(value or "").strip().lower()
    return {
        "molecule": OledCausalLayer.MOLECULE,
        "interaction": OledCausalLayer.INTERACTION,
        "device": OledCausalLayer.DEVICE,
        "measurement": OledCausalLayer.MEASUREMENT,
    }.get(normalized, OledCausalLayer.INTERACTION)


def _evidence_type(value: str | None) -> OledEvidenceType:
    normalized = str(value or "").strip().lower()
    if normalized == "table":
        return OledEvidenceType.TABLE
    if normalized == "figure":
        return OledEvidenceType.FIGURE
    if normalized == "text":
        return OledEvidenceType.TEXT
    if normalized == "manual_review":
        return OledEvidenceType.MANUAL_REVIEW
    return OledEvidenceType.TEXT


def _finding(
    code: str,
    message: str,
    candidate: OledReviewedExtractionCandidate,
    *,
    severity: Literal["error", "warning"] = "warning",
) -> OledReviewedGoldConversionFinding:
    return OledReviewedGoldConversionFinding(
        code=code,
        severity=severity,
        message=message,
        source_reviewed_candidate_id=candidate.candidate_id,
        gold_candidate_id=_gold_candidate_id(candidate),
    )


def _has_error_like_adjudication_code(candidate: OledReviewedExtractionCandidate) -> bool:
    return any(code.endswith("_error") or "error" in code for code in candidate.adjudication_finding_codes)


def _gold_candidate_id(candidate: OledReviewedExtractionCandidate) -> str:
    return f"gold-candidate:{_safe_identifier(candidate.candidate_id)}"


def _gold_record_id(candidate: OledReviewedExtractionCandidate) -> str:
    return f"gold-record-candidate:{_safe_identifier(candidate.candidate_id)}"


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", value.strip())


def _reject_forbidden_input(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"forbidden_pdf_input:{redact_oled_mineru_acceptance_path(path)}")
    if suffix in _FORBIDDEN_IMAGE_SUFFIXES:
        raise ValueError(f"forbidden_image_input:{redact_oled_mineru_acceptance_path(path)}")


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
    "OledReviewedGoldCandidateStatus",
    "OledReviewedGoldConversionPolicy",
    "OledReviewedGoldCandidate",
    "OledReviewedGoldConversionFinding",
    "OledReviewedGoldConversionReport",
    "effective_reviewed_extraction_packet",
    "build_gold_record_candidate_from_reviewed_extraction",
    "convert_reviewed_extractions_to_gold_candidates",
    "load_oled_reviewed_extraction_candidates_jsonl",
    "write_oled_reviewed_gold_candidates_jsonl",
    "write_oled_reviewed_gold_conversion_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
