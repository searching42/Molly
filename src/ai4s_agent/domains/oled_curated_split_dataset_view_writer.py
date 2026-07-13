from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai4s_agent.domains.oled_curated_dataset_split_preflight import (
    OledCuratedDatasetSplitPreflightReport,
    OledCuratedDatasetSplitPreflightStatus,
    OledDatasetViewRowSplitAssignment,
    OledDatasetViewRowSplitStatus,
    load_oled_curated_dataset_view_rows_from_manifest,
    load_oled_curated_dataset_view_writer_manifest_json,
)
from ai4s_agent.domains.oled_curated_dataset_view_writer import (
    OledCuratedDatasetViewRowArtifact,
    OledCuratedDatasetViewWriterManifest,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledCuratedSplitDatasetViewWriterPolicy(BaseModel):
    require_confirmation: bool = True
    require_split_preflight_valid: bool = True
    allow_split_preflight_warnings: bool = True
    require_all_rows_assigned: bool = True
    reject_cross_split_rows: bool = True
    reject_unassigned_rows: bool = True
    include_feature_payload: bool = False
    write_training_data: bool = False
    run_feature_materialization: bool = False
    run_model_backends: bool = False


class OledCuratedSplitDatasetViewWriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class OledCuratedSplitDatasetViewRowArtifact(BaseModel):
    row_id: str
    split_row_id: str

    split: str
    view_kind: str
    target_property_id: str

    record_id: str
    source_record_ids: list[str] = Field(default_factory=list)

    target_value: float | int | str | None = None
    target_unit: str | None = None
    target_reported_value_text: str | None = None
    target_reported_decimal_places: int | None = Field(default=None, ge=0)
    target_reported_unit: str | None = None
    target_layer: str

    condition_hash: str | None = None
    dedup_key_hash: str | None = None

    evidence_refs: list[str] = Field(default_factory=list)
    confidence_score: float | None = None

    feature_view: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)

    source_record_splits: dict[str, str] = Field(default_factory=dict)
    assignment_reason_codes: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_record_ids", "evidence_refs", "assignment_reason_codes")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedSplitDatasetViewFileResult(BaseModel):
    split: str
    view_kind: str
    target_property_id: str

    status: OledCuratedSplitDatasetViewWriteStatus
    row_count: int = 0

    output_jsonl_path: str | None = None
    output_sha256: str | None = None

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason_codes")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedSplitDatasetViewWriterFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    split: str | None = None
    view_kind: str | None = None
    target_property_id: str | None = None
    row_id: str | None = None
    output_jsonl_path: str | None = None


class OledCuratedSplitDatasetViewWriterManifest(BaseModel):
    manifest_id: str

    source_dataset_view_manifest_id: str | None = None
    source_split_preflight_status: str | None = None

    output_directory: str | None = None
    output_file_count: int = 0
    output_row_count: int = 0

    splits: list[str] = Field(default_factory=list)
    view_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)

    status_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    rows_by_split: dict[str, int] = Field(default_factory=dict)

    file_results: list[OledCuratedSplitDatasetViewFileResult] = Field(default_factory=list)

    policy: OledCuratedSplitDatasetViewWriterPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(result.status == OledCuratedSplitDatasetViewWriteStatus.REJECTED for result in self.file_results)


class OledCuratedSplitDatasetViewWriterReport(BaseModel):
    manifest: OledCuratedSplitDatasetViewWriterManifest
    split_row_artifacts: list[OledCuratedSplitDatasetViewRowArtifact] = Field(default_factory=list)
    findings: list[OledCuratedSplitDatasetViewWriterFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_curated_dataset_split_preflight_report_json(
    path: str | Path,
) -> OledCuratedDatasetSplitPreflightReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_split_preflight_report:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = OledCuratedDatasetSplitPreflightReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_split_preflight_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    if _contains_absolute_path(report.metadata):
        raise ValueError("absolute_path_in_split_preflight_report_metadata")
    return report


def build_oled_curated_split_dataset_view_row_artifacts(
    rows: Iterable[OledCuratedDatasetViewRowArtifact],
    *,
    split_preflight_report: OledCuratedDatasetSplitPreflightReport,
    policy: OledCuratedSplitDatasetViewWriterPolicy | None = None,
) -> tuple[list[OledCuratedSplitDatasetViewRowArtifact], list[OledCuratedSplitDatasetViewWriterFinding]]:
    writer_policy = policy or OledCuratedSplitDatasetViewWriterPolicy()
    assignment_by_row_id, duplicate_findings = _assignment_by_row_id(split_preflight_report.row_assignments)
    split_rows: list[OledCuratedSplitDatasetViewRowArtifact] = []
    findings: list[OledCuratedSplitDatasetViewWriterFinding] = list(duplicate_findings)

    for row in sorted(list(rows), key=lambda item: (item.view_kind, item.target_property_id, item.record_id, item.row_id)):
        assignment = assignment_by_row_id.get(row.row_id)
        if assignment is None:
            severity = "error" if writer_policy.require_all_rows_assigned or writer_policy.reject_unassigned_rows else "warning"
            findings.append(_row_finding("row_unassigned_rejected", severity, "dataset-view row has no split assignment", row))
            continue
        if assignment.status == OledDatasetViewRowSplitStatus.CROSS_SPLIT_SOURCE_RECORDS:
            severity = "error" if writer_policy.reject_cross_split_rows else "warning"
            findings.append(_row_finding("cross_split_row_rejected", severity, "cross-split dataset-view row is not materialized", row))
            continue
        if assignment.status == OledDatasetViewRowSplitStatus.UNASSIGNED or not assignment.split:
            severity = "error" if writer_policy.reject_unassigned_rows else "warning"
            findings.append(_row_finding("row_unassigned_rejected", severity, "unassigned dataset-view row is not materialized", row))
            continue
        split_rows.append(_split_row_from_row(row, assignment, writer_policy))

    return sorted(split_rows, key=lambda row: (row.split, row.view_kind, row.target_property_id, row.split_row_id)), _dedup_findings(findings)


def select_oled_curated_split_dataset_view_rows_for_write(
    rows: Iterable[OledCuratedDatasetViewRowArtifact],
    *,
    split_preflight_report: OledCuratedDatasetSplitPreflightReport,
    policy: OledCuratedSplitDatasetViewWriterPolicy | None = None,
    confirm_split_dataset_view_write: bool = False,
) -> OledCuratedSplitDatasetViewWriterReport:
    writer_policy = policy or OledCuratedSplitDatasetViewWriterPolicy()
    if writer_policy.require_confirmation and not confirm_split_dataset_view_write:
        raise ValueError("confirmation_required:split_dataset_view_write")

    preflight_findings = _preflight_gate_findings(split_preflight_report, writer_policy)
    if any(finding.severity == "error" for finding in preflight_findings):
        return OledCuratedSplitDatasetViewWriterReport(
            manifest=_manifest(
                policy=writer_policy,
                file_results=[],
                split_row_artifacts=[],
                source_split_preflight_status=_status_value(split_preflight_report.status),
            ),
            split_row_artifacts=[],
            findings=preflight_findings,
        )

    split_rows, row_findings = build_oled_curated_split_dataset_view_row_artifacts(
        rows,
        split_preflight_report=split_preflight_report,
        policy=writer_policy,
    )
    file_results = _file_results_for_split_rows(split_rows, writer_policy)
    return OledCuratedSplitDatasetViewWriterReport(
        manifest=_manifest(
            policy=writer_policy,
            file_results=file_results,
            split_row_artifacts=split_rows,
            source_split_preflight_status=_status_value(split_preflight_report.status),
        ),
        split_row_artifacts=split_rows,
        findings=_dedup_findings([*preflight_findings, *row_findings]),
    )


def write_oled_curated_split_dataset_view_rows_jsonl(
    rows: Iterable[OledCuratedSplitDatasetViewRowArtifact],
    path: str | Path,
) -> str:
    lines = [
        json.dumps(
            _sanitize_for_output(row.model_dump(mode="json", exclude_none=True)),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in sorted(rows, key=lambda item: item.split_row_id)
    ]
    payload = "\n".join(lines) + ("\n" if lines else "")
    encoded = payload.encode("utf-8")
    Path(path).write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_oled_curated_split_dataset_view_manifest_json(
    manifest: OledCuratedSplitDatasetViewWriterManifest,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def oled_split_dataset_view_output_filename(
    *,
    split: str,
    view_kind: str,
    target_property_id: str,
) -> str:
    return (
        "oled_split_view__"
        f"{_safe_filename_token(split)}__"
        f"{_safe_filename_token(view_kind)}__"
        f"{_safe_filename_token(target_property_id)}.jsonl"
    )


def run_oled_curated_split_dataset_view_writer_from_files(
    *,
    dataset_view_manifest_path: str | Path,
    split_preflight_report_path: str | Path,
    dataset_view_base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    policy: OledCuratedSplitDatasetViewWriterPolicy | None = None,
    confirm_split_dataset_view_write: bool = False,
    dry_run: bool = False,
) -> OledCuratedSplitDatasetViewWriterReport:
    writer_policy = policy or OledCuratedSplitDatasetViewWriterPolicy()
    if not dry_run and writer_policy.require_confirmation and not confirm_split_dataset_view_write:
        raise ValueError("confirmation_required:split_dataset_view_write")
    if not dry_run and output_dir is None:
        raise ValueError("output_dir_required:split_dataset_view_write")

    dataset_view_manifest = load_oled_curated_dataset_view_writer_manifest_json(dataset_view_manifest_path)
    base_dir = Path(dataset_view_base_dir) if dataset_view_base_dir is not None else Path(dataset_view_manifest_path).parent
    rows = load_oled_curated_dataset_view_rows_from_manifest(
        manifest=dataset_view_manifest,
        base_dir=base_dir,
    )
    split_preflight_report = load_oled_curated_dataset_split_preflight_report_json(split_preflight_report_path)
    selection_policy = writer_policy.model_copy(update={"require_confirmation": not dry_run and writer_policy.require_confirmation})
    report = select_oled_curated_split_dataset_view_rows_for_write(
        rows,
        split_preflight_report=split_preflight_report,
        policy=selection_policy,
        confirm_split_dataset_view_write=confirm_split_dataset_view_write or dry_run,
    )
    report = _attach_source_context(
        report,
        source_dataset_view_manifest_id=dataset_view_manifest.manifest_id,
        source_split_preflight_status=_status_value(split_preflight_report.status),
    )
    if dry_run:
        report = _mark_dry_run(report)
        if output_manifest_path is not None:
            write_oled_curated_split_dataset_view_manifest_json(report.manifest, output_manifest_path)
        return report

    if not report.is_valid:
        if output_manifest_path is not None:
            write_oled_curated_split_dataset_view_manifest_json(report.manifest, output_manifest_path)
        return report

    assert output_dir is not None
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report = _write_selected_split_files(report, output_root)
    if output_manifest_path is not None:
        write_oled_curated_split_dataset_view_manifest_json(report.manifest, output_manifest_path)
    return report


def load_oled_curated_split_dataset_view_rows_jsonl(
    path: str | Path,
) -> list[OledCuratedSplitDatasetViewRowArtifact]:
    rows_path = Path(path)
    _reject_forbidden_input(rows_path)
    if not rows_path.exists():
        raise ValueError(f"missing_split_dataset_view_rows_jsonl:{redact_oled_mineru_acceptance_path(rows_path)}")
    rows: list[OledCuratedSplitDatasetViewRowArtifact] = []
    for line_number, raw_line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            row = OledCuratedSplitDatasetViewRowArtifact.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid_split_dataset_view_rows_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(row.metadata):
            raise ValueError(f"absolute_path_in_split_dataset_view_row_metadata:{row.split_row_id}")
        rows.append(row)
    return sorted(rows, key=lambda row: row.split_row_id)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write split-assigned OLED dataset-view row artifacts under explicit gates.")
    parser.add_argument("--dataset-view-manifest", required=True, help="Path to dataset-view writer manifest JSON.")
    parser.add_argument("--split-preflight-report", required=True, help="Path to split preflight report JSON.")
    parser.add_argument("--dataset-view-base-dir", help="Base directory for dataset-view row JSONL paths.")
    parser.add_argument("--output-dir", help="Directory for split dataset-view row JSONL files.")
    parser.add_argument("--output-manifest", help="Optional path for split dataset-view writer manifest JSON.")
    parser.add_argument("--confirm-split-dataset-view-write", action="store_true", help="Confirm split dataset-view row writing.")
    parser.add_argument("--dry-run", action="store_true", help="Run selection without writing split row JSONL files.")
    parser.add_argument("--include-feature-payload", action="store_true", help="Include row feature payloads in JSONL.")
    args = parser.parse_args(argv)

    if not args.output_dir and not args.output_manifest:
        print("output_required:dir_or_manifest", file=sys.stderr)
        return 1
    if not args.dry_run and not args.confirm_split_dataset_view_write:
        print("confirmation_required:split_dataset_view_write", file=sys.stderr)
        return 1
    try:
        policy = OledCuratedSplitDatasetViewWriterPolicy(
            require_confirmation=not args.dry_run,
            include_feature_payload=args.include_feature_payload,
        )
        report = run_oled_curated_split_dataset_view_writer_from_files(
            dataset_view_manifest_path=args.dataset_view_manifest,
            split_preflight_report_path=args.split_preflight_report,
            dataset_view_base_dir=args.dataset_view_base_dir,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            policy=policy,
            confirm_split_dataset_view_write=args.confirm_split_dataset_view_write,
            dry_run=args.dry_run,
        )
        summary = {
            "output_file_count": report.manifest.output_file_count,
            "output_row_count": report.manifest.output_row_count,
            "rows_by_split": report.manifest.rows_by_split,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _assignment_by_row_id(
    assignments: list[OledDatasetViewRowSplitAssignment],
) -> tuple[dict[str, OledDatasetViewRowSplitAssignment], list[OledCuratedSplitDatasetViewWriterFinding]]:
    output: dict[str, OledDatasetViewRowSplitAssignment] = {}
    findings: list[OledCuratedSplitDatasetViewWriterFinding] = []
    for assignment in sorted(assignments, key=lambda item: (item.row_id, item.split or "")):
        if assignment.row_id in output:
            findings.append(
                OledCuratedSplitDatasetViewWriterFinding(
                    code="duplicate_split_assignment_row_id",
                    severity="error",
                    message=f"split preflight report contains duplicate assignment for row `{assignment.row_id}`",
                    split=assignment.split,
                    view_kind=assignment.view_kind,
                    target_property_id=assignment.target_property_id,
                    row_id=assignment.row_id,
                )
            )
            continue
        output[assignment.row_id] = assignment
    return output, findings


def _split_row_from_row(
    row: OledCuratedDatasetViewRowArtifact,
    assignment: OledDatasetViewRowSplitAssignment,
    policy: OledCuratedSplitDatasetViewWriterPolicy,
) -> OledCuratedSplitDatasetViewRowArtifact:
    assert assignment.split is not None
    features = _sanitize_for_output(row.features) if policy.include_feature_payload else {}
    metadata = _sanitize_for_output(row.metadata)
    metadata.update(
        {
            "split_dataset_view_row_artifact": True,
            "training_data_record": False,
            "ml_ready_training_data_record": False,
            "feature_payload_omitted": not policy.include_feature_payload,
            "condition_hash": row.condition_hash,
            "dedup_key_hash": row.dedup_key_hash,
        }
    )
    if policy.include_feature_payload:
        metadata.pop("feature_payload_omitted", None)
    return OledCuratedSplitDatasetViewRowArtifact(
        row_id=row.row_id,
        split_row_id=_split_row_id(row=row, assignment=assignment),
        split=assignment.split,
        view_kind=row.view_kind,
        target_property_id=row.target_property_id,
        record_id=row.record_id,
        source_record_ids=row.source_record_ids,
        target_value=row.target_value,
        target_unit=row.target_unit,
        target_reported_value_text=row.target_reported_value_text,
        target_reported_decimal_places=row.target_reported_decimal_places,
        target_reported_unit=row.target_reported_unit,
        target_layer=row.target_layer,
        condition_hash=row.condition_hash,
        dedup_key_hash=row.dedup_key_hash,
        evidence_refs=row.evidence_refs,
        confidence_score=row.confidence_score,
        feature_view=row.feature_view,
        features=features,
        source_record_splits=assignment.source_record_splits,
        assignment_reason_codes=assignment.reason_codes,
        metadata=metadata,
    )


def _split_row_id(
    *,
    row: OledCuratedDatasetViewRowArtifact,
    assignment: OledDatasetViewRowSplitAssignment,
) -> str:
    payload = {
        "row_id": row.row_id,
        "split": assignment.split,
        "view_kind": row.view_kind,
        "target_property_id": row.target_property_id,
        "record_id": row.record_id,
        "source_record_ids": row.source_record_ids,
        "condition_hash": row.condition_hash,
        "dedup_key_hash": row.dedup_key_hash,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-split-view-row:{digest[:20]}"


def _row_finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    row: OledCuratedDatasetViewRowArtifact,
) -> OledCuratedSplitDatasetViewWriterFinding:
    return OledCuratedSplitDatasetViewWriterFinding(
        code=code,
        severity=severity,
        message=message,
        view_kind=row.view_kind,
        target_property_id=row.target_property_id,
        row_id=row.row_id,
    )


def _preflight_gate_findings(
    split_preflight_report: OledCuratedDatasetSplitPreflightReport,
    policy: OledCuratedSplitDatasetViewWriterPolicy,
) -> list[OledCuratedSplitDatasetViewWriterFinding]:
    findings: list[OledCuratedSplitDatasetViewWriterFinding] = []
    if policy.require_split_preflight_valid and not split_preflight_report.is_valid:
        findings.append(
            OledCuratedSplitDatasetViewWriterFinding(
                code="split_preflight_failed",
                severity="error",
                message="split dataset-view writer blocked because split preflight report is invalid",
            )
        )
    if not policy.allow_split_preflight_warnings and split_preflight_report.warning_codes:
        findings.append(
            OledCuratedSplitDatasetViewWriterFinding(
                code="split_preflight_warnings_present",
                severity="error",
                message="split dataset-view writer blocked because split preflight warnings are disallowed",
            )
        )
    return findings


def _file_results_for_split_rows(
    split_rows: list[OledCuratedSplitDatasetViewRowArtifact],
    policy: OledCuratedSplitDatasetViewWriterPolicy,
) -> list[OledCuratedSplitDatasetViewFileResult]:
    grouped: dict[tuple[str, str, str], list[OledCuratedSplitDatasetViewRowArtifact]] = defaultdict(list)
    for row in split_rows:
        grouped[(row.split, row.view_kind, row.target_property_id)].append(row)
    results: list[OledCuratedSplitDatasetViewFileResult] = []
    for (split, view_kind, target_property_id), group in sorted(grouped.items()):
        reason_codes = ["selected_for_write", "row_assigned"]
        if not policy.include_feature_payload:
            reason_codes.append("feature_payload_omitted")
        results.append(
            OledCuratedSplitDatasetViewFileResult(
                split=split,
                view_kind=view_kind,
                target_property_id=target_property_id,
                status=OledCuratedSplitDatasetViewWriteStatus.WRITTEN,
                row_count=len(group),
                reason_codes=reason_codes,
                metadata={"split_dataset_view_rows_written": False},
            )
        )
    return results


def _manifest(
    *,
    policy: OledCuratedSplitDatasetViewWriterPolicy,
    file_results: list[OledCuratedSplitDatasetViewFileResult],
    split_row_artifacts: list[OledCuratedSplitDatasetViewRowArtifact],
    source_dataset_view_manifest_id: str | None = None,
    source_split_preflight_status: str | None = None,
    output_directory: str | None = None,
    split_dataset_view_rows_written: bool = False,
) -> OledCuratedSplitDatasetViewWriterManifest:
    return OledCuratedSplitDatasetViewWriterManifest(
        manifest_id=_manifest_id(policy, file_results),
        source_dataset_view_manifest_id=source_dataset_view_manifest_id,
        source_split_preflight_status=source_split_preflight_status,
        output_directory=output_directory,
        output_file_count=sum(1 for result in file_results if result.status == OledCuratedSplitDatasetViewWriteStatus.WRITTEN and result.row_count > 0),
        output_row_count=len(split_row_artifacts),
        splits=sorted({row.split for row in split_row_artifacts}),
        view_kinds=sorted({row.view_kind for row in split_row_artifacts}),
        target_property_ids=sorted({row.target_property_id for row in split_row_artifacts}),
        status_counts=dict(sorted(Counter(result.status.value for result in file_results).items())),
        reason_code_counts=dict(sorted(Counter(code for result in file_results for code in result.reason_codes).items())),
        rows_by_split=dict(sorted(Counter(row.split for row in split_row_artifacts).items())),
        file_results=sorted(file_results, key=lambda item: (item.split, item.view_kind, item.target_property_id)),
        policy=policy,
        metadata=_safety_metadata(split_dataset_view_rows_written=split_dataset_view_rows_written),
    )


def _manifest_id(
    policy: OledCuratedSplitDatasetViewWriterPolicy,
    file_results: list[OledCuratedSplitDatasetViewFileResult],
) -> str:
    payload = {
        "policy": policy.model_dump(mode="json"),
        "file_results": [
            {
                "split": result.split,
                "view_kind": result.view_kind,
                "target_property_id": result.target_property_id,
                "row_count": result.row_count,
                "status": result.status.value,
            }
            for result in sorted(file_results, key=lambda item: (item.split, item.view_kind, item.target_property_id))
        ],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-curated-split-dataset-view-writer:{digest[:16]}"


def _attach_source_context(
    report: OledCuratedSplitDatasetViewWriterReport,
    *,
    source_dataset_view_manifest_id: str | None,
    source_split_preflight_status: str | None,
) -> OledCuratedSplitDatasetViewWriterReport:
    manifest = report.manifest.model_copy(
        update={
            "source_dataset_view_manifest_id": source_dataset_view_manifest_id,
            "source_split_preflight_status": source_split_preflight_status,
        }
    )
    return report.model_copy(update={"manifest": manifest})


def _mark_dry_run(report: OledCuratedSplitDatasetViewWriterReport) -> OledCuratedSplitDatasetViewWriterReport:
    refreshed_results = [
        result.model_copy(
            update={
                "reason_codes": sorted({*result.reason_codes, "dry_run_no_rows_written"}),
                "metadata": {**result.metadata, "split_dataset_view_rows_written": False},
            }
        )
        for result in report.manifest.file_results
    ]
    manifest = _manifest(
        policy=report.manifest.policy,
        file_results=refreshed_results,
        split_row_artifacts=report.split_row_artifacts,
        source_dataset_view_manifest_id=report.manifest.source_dataset_view_manifest_id,
        source_split_preflight_status=report.manifest.source_split_preflight_status,
        split_dataset_view_rows_written=False,
    )
    return report.model_copy(update={"manifest": manifest})


def _write_selected_split_files(
    report: OledCuratedSplitDatasetViewWriterReport,
    output_dir: Path,
) -> OledCuratedSplitDatasetViewWriterReport:
    grouped_rows: dict[tuple[str, str, str], list[OledCuratedSplitDatasetViewRowArtifact]] = defaultdict(list)
    for row in report.split_row_artifacts:
        grouped_rows[(row.split, row.view_kind, row.target_property_id)].append(row)

    refreshed_results: list[OledCuratedSplitDatasetViewFileResult] = []
    for result in report.manifest.file_results:
        if result.status != OledCuratedSplitDatasetViewWriteStatus.WRITTEN or result.row_count <= 0:
            refreshed_results.append(result)
            continue
        filename = oled_split_dataset_view_output_filename(
            split=result.split,
            view_kind=result.view_kind,
            target_property_id=result.target_property_id,
        )
        output_path = output_dir / filename
        output_sha = write_oled_curated_split_dataset_view_rows_jsonl(
            grouped_rows[(result.split, result.view_kind, result.target_property_id)],
            output_path,
        )
        refreshed_results.append(
            result.model_copy(
                update={
                    "output_jsonl_path": filename,
                    "output_sha256": output_sha,
                    "metadata": {**result.metadata, "split_dataset_view_rows_written": True},
                }
            )
        )

    manifest = _manifest(
        policy=report.manifest.policy,
        file_results=refreshed_results,
        split_row_artifacts=report.split_row_artifacts,
        source_dataset_view_manifest_id=report.manifest.source_dataset_view_manifest_id,
        source_split_preflight_status=report.manifest.source_split_preflight_status,
        output_directory=redact_oled_mineru_acceptance_path(output_dir),
        split_dataset_view_rows_written=True,
    )
    return report.model_copy(update={"manifest": manifest})


def _status_value(status: OledCuratedDatasetSplitPreflightStatus | str) -> str:
    return status.value if isinstance(status, OledCuratedDatasetSplitPreflightStatus) else str(status)


def _safe_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_") or "unknown"


def _safety_metadata(*, split_dataset_view_rows_written: bool) -> dict[str, Any]:
    return {
        "split_dataset_view_writer": True,
        "split_dataset_view_rows_written": split_dataset_view_rows_written,
        "training_data_written": False,
        "ml_ready_training_data_written": False,
        "feature_materialization_outputs_written": False,
        "model_backends_run": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


def _dedup_findings(
    findings: list[OledCuratedSplitDatasetViewWriterFinding],
) -> list[OledCuratedSplitDatasetViewWriterFinding]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[OledCuratedSplitDatasetViewWriterFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.split or "",
            finding.view_kind or "",
            finding.target_property_id or "",
            finding.row_id or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return sorted(
        deduped,
        key=lambda item: (
            item.severity,
            item.code,
            item.split or "",
            item.view_kind or "",
            item.target_property_id or "",
            item.row_id or "",
        ),
    )


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
        if len(value) > _MAX_OUTPUT_STRING_LENGTH:
            return value[: _MAX_OUTPUT_STRING_LENGTH - 3] + "..."
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
            "layered_record",
            "gold_record",
        )
    )


_MAX_OUTPUT_STRING_LENGTH = 240

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
    "OledCuratedSplitDatasetViewWriterPolicy",
    "OledCuratedSplitDatasetViewWriteStatus",
    "OledCuratedSplitDatasetViewRowArtifact",
    "OledCuratedSplitDatasetViewFileResult",
    "OledCuratedSplitDatasetViewWriterFinding",
    "OledCuratedSplitDatasetViewWriterManifest",
    "OledCuratedSplitDatasetViewWriterReport",
    "load_oled_curated_dataset_split_preflight_report_json",
    "build_oled_curated_split_dataset_view_row_artifacts",
    "select_oled_curated_split_dataset_view_rows_for_write",
    "write_oled_curated_split_dataset_view_rows_jsonl",
    "write_oled_curated_split_dataset_view_manifest_json",
    "oled_split_dataset_view_output_filename",
    "run_oled_curated_split_dataset_view_writer_from_files",
    "load_oled_curated_split_dataset_view_rows_jsonl",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
