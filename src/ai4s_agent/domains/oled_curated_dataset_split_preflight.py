from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai4s_agent.domains.oled_curated_dataset_view_writer import (
    OledCuratedDatasetViewRowArtifact,
    OledCuratedDatasetViewWriteStatus,
    OledCuratedDatasetViewWriterManifest,
    load_oled_curated_dataset_view_rows_jsonl,
)
from ai4s_agent.domains.oled_curated_gold_view_preflight import (
    check_oled_curated_gold_manifest_integrity,
    load_oled_curated_gold_manifest_json,
    load_oled_curated_gold_records_jsonl,
)
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord, validate_oled_gold_dataset
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path
from ai4s_agent.domains.oled_split_leakage import (
    OledLeakageGroupKind,
    OledLeakageGuardSplitPlan,
    build_oled_leakage_guard_split,
    validate_oled_split_leakage,
)


class OledCuratedDatasetSplitPreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledDatasetViewRowSplitStatus(str, Enum):
    ASSIGNED = "assigned"
    UNASSIGNED = "unassigned"
    CROSS_SPLIT_SOURCE_RECORDS = "cross_split_source_records"


class OledCuratedDatasetSplitPreflightPolicy(BaseModel):
    split_names: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])
    leakage_group_kinds: list[str] = Field(default_factory=list)
    require_gold_validation_success: bool = True
    require_split_leakage_valid: bool = True
    require_all_rows_assigned: bool = True
    allow_empty_split: bool = False
    require_dataset_view_manifest_sha256: bool = True
    require_curated_gold_manifest_sha256: bool = True
    fail_on_cross_split_source_rows: bool = True

    @field_validator("split_names")
    @classmethod
    def validate_split_names(cls, value: list[str]) -> list[str]:
        clean = [str(item).strip() for item in value if str(item).strip()]
        if not clean:
            raise ValueError("split_names are required")
        return clean


class OledDatasetViewRowSplitAssignment(BaseModel):
    row_id: str
    view_kind: str
    target_property_id: str
    record_id: str
    source_record_ids: list[str] = Field(default_factory=list)

    split: str | None = None
    status: OledDatasetViewRowSplitStatus

    source_record_splits: dict[str, str] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)

    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_record_ids", "reason_codes", "evidence_refs")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledDatasetViewSplitSummary(BaseModel):
    view_kind: str
    target_property_id: str

    row_count: int
    assigned_row_count: int
    unassigned_row_count: int
    cross_split_row_count: int

    rows_by_split: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)


class OledCuratedDatasetSplitPreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    record_id: str | None = None
    row_id: str | None = None
    view_kind: str | None = None
    target_property_id: str | None = None
    split: str | None = None


class OledCuratedDatasetSplitPreflightReport(BaseModel):
    status: OledCuratedDatasetSplitPreflightStatus

    input_gold_record_count: int
    input_dataset_view_row_count: int

    split_names: list[str] = Field(default_factory=list)
    leakage_group_kinds: list[str] = Field(default_factory=list)

    split_plan: OledLeakageGuardSplitPlan | None = None
    split_leakage_error_codes: list[str] = Field(default_factory=list)
    split_leakage_warning_codes: list[str] = Field(default_factory=list)

    gold_validation_error_codes: list[str] = Field(default_factory=list)
    gold_validation_warning_codes: list[str] = Field(default_factory=list)

    row_assignments: list[OledDatasetViewRowSplitAssignment] = Field(default_factory=list)
    view_summaries: list[OledDatasetViewSplitSummary] = Field(default_factory=list)

    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)
    rows_by_split: dict[str, int] = Field(default_factory=dict)

    findings: list[OledCuratedDatasetSplitPreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledCuratedDatasetSplitPreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_curated_dataset_view_writer_manifest_json(
    path: str | Path,
) -> OledCuratedDatasetViewWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_dataset_view_writer_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledCuratedDatasetViewWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_dataset_view_writer_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.metadata):
        raise ValueError(f"absolute_path_in_dataset_view_writer_manifest_metadata:{manifest.manifest_id}")
    for result in manifest.file_results:
        if _contains_absolute_path(result.metadata):
            raise ValueError(
                "absolute_path_in_dataset_view_writer_file_result_metadata:"
                f"{result.view_kind}:{result.target_property_id}"
            )
    return manifest


def load_oled_curated_dataset_view_rows_from_manifest(
    *,
    manifest: OledCuratedDatasetViewWriterManifest,
    base_dir: str | Path,
) -> list[OledCuratedDatasetViewRowArtifact]:
    rows: list[OledCuratedDatasetViewRowArtifact] = []
    root = Path(base_dir)
    for result in sorted(manifest.file_results, key=lambda item: (item.view_kind, item.target_property_id)):
        if result.status != OledCuratedDatasetViewWriteStatus.WRITTEN or not result.output_jsonl_path:
            continue
        row_path = Path(result.output_jsonl_path)
        if row_path.suffix.lower() != ".jsonl":
            raise ValueError(f"forbidden_dataset_view_rows_input:{redact_oled_mineru_acceptance_path(row_path)}")
        resolved_path = row_path if row_path.is_absolute() else root / row_path
        if result.output_sha256:
            actual_sha = _sha256_file(resolved_path)
            if actual_sha != result.output_sha256:
                raise ValueError(f"dataset_view_rows_sha256_mismatch:{redact_oled_mineru_acceptance_path(row_path)}")
        rows.extend(load_oled_curated_dataset_view_rows_jsonl(resolved_path))
    return sorted(rows, key=lambda row: row.row_id)


def run_oled_curated_dataset_split_preflight(
    *,
    gold_records: Iterable[OledGoldDatasetRecord],
    dataset_view_rows: Iterable[OledCuratedDatasetViewRowArtifact],
    policy: OledCuratedDatasetSplitPreflightPolicy | None = None,
) -> OledCuratedDatasetSplitPreflightReport:
    preflight_policy = policy or OledCuratedDatasetSplitPreflightPolicy()
    records = sorted(list(gold_records), key=lambda item: item.record_id)
    rows = sorted(list(dataset_view_rows), key=lambda item: row_sort_key(item))
    group_kinds = _leakage_group_kinds(preflight_policy)
    findings: list[OledCuratedDatasetSplitPreflightFinding] = []

    gold_report = validate_oled_gold_dataset(records)
    gold_error_codes = gold_report.error_codes
    gold_warning_codes = gold_report.warning_codes
    if gold_error_codes and preflight_policy.require_gold_validation_success:
        findings.append(
            OledCuratedDatasetSplitPreflightFinding(
                code="gold_validation_errors_present",
                severity="error",
                message="gold validation errors block leakage-split preflight",
            )
        )

    split_plan: OledLeakageGuardSplitPlan | None = None
    split_error_codes: list[str] = []
    split_warning_codes: list[str] = []
    if not gold_error_codes or not preflight_policy.require_gold_validation_success:
        try:
            split_plan = build_oled_leakage_guard_split(
                records,
                group_kinds=group_kinds,
                split_names=tuple(preflight_policy.split_names),
            )
            leakage_report = validate_oled_split_leakage(split_plan.assignments)
            split_error_codes = leakage_report.error_codes
            split_warning_codes = [
                finding.code for finding in leakage_report.findings if finding.severity == "warning"
            ]
            if split_error_codes and preflight_policy.require_split_leakage_valid:
                findings.append(
                    OledCuratedDatasetSplitPreflightFinding(
                        code="split_leakage_errors_present",
                        severity="error",
                        message="split leakage validation reported errors",
                    )
                )
        except Exception as exc:
            findings.append(
                OledCuratedDatasetSplitPreflightFinding(
                    code="split_plan_build_failed",
                    severity="error",
                    message=str(exc).splitlines()[0],
                )
            )

    row_assignments: list[OledDatasetViewRowSplitAssignment] = []
    if split_plan is not None:
        row_assignments, row_findings = _assign_rows_to_splits(rows, split_plan, preflight_policy)
        findings.extend(row_findings)
        findings.extend(_empty_split_findings(row_assignments, preflight_policy))

    report = OledCuratedDatasetSplitPreflightReport(
        status=OledCuratedDatasetSplitPreflightStatus.PASSED,
        input_gold_record_count=len(records),
        input_dataset_view_row_count=len(rows),
        split_names=preflight_policy.split_names,
        leakage_group_kinds=[kind.value for kind in group_kinds],
        split_plan=split_plan,
        split_leakage_error_codes=split_error_codes,
        split_leakage_warning_codes=split_warning_codes,
        gold_validation_error_codes=gold_error_codes,
        gold_validation_warning_codes=gold_warning_codes,
        row_assignments=row_assignments,
        view_summaries=_view_summaries(row_assignments),
        findings=_dedup_findings(findings),
        metadata=_safety_metadata(),
    )
    return _refresh_report(report)


def run_oled_curated_dataset_split_preflight_from_files(
    *,
    curated_gold_jsonl_path: str | Path,
    dataset_view_manifest_path: str | Path,
    dataset_view_base_dir: str | Path | None = None,
    curated_gold_manifest_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledCuratedDatasetSplitPreflightPolicy | None = None,
) -> OledCuratedDatasetSplitPreflightReport:
    preflight_policy = policy or OledCuratedDatasetSplitPreflightPolicy()
    records = load_oled_curated_gold_records_jsonl(curated_gold_jsonl_path)
    dataset_view_manifest = load_oled_curated_dataset_view_writer_manifest_json(dataset_view_manifest_path)
    base_dir = Path(dataset_view_base_dir) if dataset_view_base_dir is not None else Path(dataset_view_manifest_path).parent
    extra_findings: list[OledCuratedDatasetSplitPreflightFinding] = []
    if preflight_policy.require_dataset_view_manifest_sha256:
        for result in dataset_view_manifest.file_results:
            if result.status == OledCuratedDatasetViewWriteStatus.WRITTEN and result.output_jsonl_path and not result.output_sha256:
                extra_findings.append(
                    OledCuratedDatasetSplitPreflightFinding(
                        code="dataset_view_manifest_missing_sha256",
                        severity="error",
                        message="dataset-view writer manifest has a written file without output_sha256",
                        view_kind=result.view_kind,
                        target_property_id=result.target_property_id,
                    )
                )
    rows = load_oled_curated_dataset_view_rows_from_manifest(
        manifest=dataset_view_manifest,
        base_dir=base_dir,
    )
    if curated_gold_manifest_path is not None:
        curated_manifest = load_oled_curated_gold_manifest_json(curated_gold_manifest_path)
        integrity_status, integrity_findings, _input_sha = check_oled_curated_gold_manifest_integrity(
            input_jsonl_path=curated_gold_jsonl_path,
            manifest=curated_manifest,
        )
        if preflight_policy.require_curated_gold_manifest_sha256 and integrity_status.value != "matched":
            extra_findings.extend(
                OledCuratedDatasetSplitPreflightFinding(
                    code=_curated_manifest_integrity_code(finding.code),
                    severity="error",
                    message=finding.message,
                )
                for finding in integrity_findings
            )
            if not integrity_findings:
                extra_findings.append(
                    OledCuratedDatasetSplitPreflightFinding(
                        code=_curated_manifest_integrity_code(integrity_status.value),
                        severity="error",
                        message=f"curated gold manifest integrity status is `{integrity_status.value}`",
                    )
                )
    report = run_oled_curated_dataset_split_preflight(
        gold_records=records,
        dataset_view_rows=rows,
        policy=preflight_policy,
    )
    if extra_findings:
        report = _refresh_report(report.model_copy(update={"findings": _dedup_findings([*report.findings, *extra_findings])}))
    if output_report_path is not None:
        write_oled_curated_dataset_split_preflight_report_json(report, output_report_path)
    return report


def write_oled_curated_dataset_split_preflight_report_json(
    report: OledCuratedDatasetSplitPreflightReport,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED curated dataset leakage-split preflight.")
    parser.add_argument("--curated-gold-jsonl", required=True, help="Path to curated gold-record JSONL.")
    parser.add_argument("--dataset-view-manifest", required=True, help="Path to dataset-view writer manifest JSON.")
    parser.add_argument("--dataset-view-base-dir", help="Base directory for dataset-view row JSONL paths.")
    parser.add_argument("--curated-gold-manifest", help="Optional curated gold writer manifest JSON.")
    parser.add_argument("--output-report", help="Optional path for split preflight report JSON.")
    parser.add_argument("--split-name", action="append", default=[], help="Split name; repeat or comma-separate.")
    parser.add_argument("--leakage-group-kind", action="append", default=[], help="Leakage group kind; repeat or comma-separate.")
    args = parser.parse_args(argv)

    try:
        policy = OledCuratedDatasetSplitPreflightPolicy(
            split_names=_split_cli_values(args.split_name) or ["train", "validation", "test"],
            leakage_group_kinds=_split_cli_values(args.leakage_group_kind),
        )
        report = run_oled_curated_dataset_split_preflight_from_files(
            curated_gold_jsonl_path=args.curated_gold_jsonl,
            dataset_view_manifest_path=args.dataset_view_manifest,
            dataset_view_base_dir=args.dataset_view_base_dir,
            curated_gold_manifest_path=args.curated_gold_manifest,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": report.status.value,
            "input_gold_record_count": report.input_gold_record_count,
            "input_dataset_view_row_count": report.input_dataset_view_row_count,
            "rows_by_split": report.rows_by_split,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def row_sort_key(row: OledCuratedDatasetViewRowArtifact) -> tuple[str, str, str, str]:
    return (row.view_kind, row.target_property_id, row.record_id, row.row_id)


def _assign_rows_to_splits(
    rows: list[OledCuratedDatasetViewRowArtifact],
    split_plan: OledLeakageGuardSplitPlan,
    policy: OledCuratedDatasetSplitPreflightPolicy,
) -> tuple[list[OledDatasetViewRowSplitAssignment], list[OledCuratedDatasetSplitPreflightFinding]]:
    assignments: list[OledDatasetViewRowSplitAssignment] = []
    findings: list[OledCuratedDatasetSplitPreflightFinding] = []
    for row in rows:
        effective_source_ids = sorted({source_id for source_id in (row.source_record_ids or [row.record_id]) if source_id})
        if not effective_source_ids:
            severity = "error" if policy.require_all_rows_assigned else "warning"
            findings.append(
                _row_finding(
                    code="row_missing_source_record_ids",
                    severity=severity,
                    message="dataset-view row has no record_id or source_record_ids",
                    row=row,
                )
            )
            assignments.append(
                _row_assignment(
                    row,
                    source_record_ids=[],
                    source_record_splits={},
                    status=OledDatasetViewRowSplitStatus.UNASSIGNED,
                    reason_codes=["row_missing_source_record_ids"],
                )
            )
            continue

        source_record_splits: dict[str, str] = {}
        unknown_source_ids: list[str] = []
        for source_id in effective_source_ids:
            try:
                source_record_splits[source_id] = split_plan.split_for_record(source_id)
            except KeyError:
                unknown_source_ids.append(source_id)
        if unknown_source_ids:
            severity = "error" if policy.require_all_rows_assigned else "warning"
            for source_id in unknown_source_ids:
                findings.append(
                    _row_finding(
                        code="unknown_row_source_record",
                        severity=severity,
                        message=f"dataset-view row source record `{source_id}` is not present in split plan",
                        row=row,
                        record_id=source_id,
                    )
                )
            assignments.append(
                _row_assignment(
                    row,
                    source_record_ids=effective_source_ids,
                    source_record_splits=source_record_splits,
                    status=OledDatasetViewRowSplitStatus.UNASSIGNED,
                    reason_codes=["unknown_row_source_record"],
                )
            )
            continue

        splits = sorted(set(source_record_splits.values()))
        if len(splits) == 1:
            assignments.append(
                _row_assignment(
                    row,
                    source_record_ids=effective_source_ids,
                    source_record_splits=source_record_splits,
                    status=OledDatasetViewRowSplitStatus.ASSIGNED,
                    split=splits[0],
                    reason_codes=["row_assigned"],
                )
            )
            continue

        severity = "error" if policy.fail_on_cross_split_source_rows else "warning"
        findings.append(
            _row_finding(
                code="row_source_records_cross_split",
                severity=severity,
                message=f"dataset-view row source records span multiple splits: {', '.join(splits)}",
                row=row,
            )
        )
        assignments.append(
            _row_assignment(
                row,
                source_record_ids=effective_source_ids,
                source_record_splits=source_record_splits,
                status=OledDatasetViewRowSplitStatus.CROSS_SPLIT_SOURCE_RECORDS,
                reason_codes=["row_source_records_cross_split"],
            )
        )
    return sorted(assignments, key=lambda item: (item.view_kind, item.target_property_id, item.record_id, item.row_id)), findings


def _row_assignment(
    row: OledCuratedDatasetViewRowArtifact,
    *,
    source_record_ids: list[str],
    source_record_splits: dict[str, str],
    status: OledDatasetViewRowSplitStatus,
    split: str | None = None,
    reason_codes: list[str],
) -> OledDatasetViewRowSplitAssignment:
    return OledDatasetViewRowSplitAssignment(
        row_id=row.row_id,
        view_kind=row.view_kind,
        target_property_id=row.target_property_id,
        record_id=row.record_id,
        source_record_ids=source_record_ids,
        split=split,
        status=status,
        source_record_splits=dict(sorted(source_record_splits.items())),
        reason_codes=reason_codes,
        evidence_refs=row.evidence_refs,
        metadata={
            "condition_hash": row.condition_hash,
            "dedup_key_hash": row.dedup_key_hash,
            "split_assignment_preflight_only": True,
        },
    )


def _row_finding(
    *,
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    row: OledCuratedDatasetViewRowArtifact,
    record_id: str | None = None,
) -> OledCuratedDatasetSplitPreflightFinding:
    return OledCuratedDatasetSplitPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        record_id=record_id or row.record_id,
        row_id=row.row_id,
        view_kind=row.view_kind,
        target_property_id=row.target_property_id,
    )


def _empty_split_findings(
    assignments: list[OledDatasetViewRowSplitAssignment],
    policy: OledCuratedDatasetSplitPreflightPolicy,
) -> list[OledCuratedDatasetSplitPreflightFinding]:
    if policy.allow_empty_split:
        return []
    assigned_counts = Counter(assignment.split for assignment in assignments if assignment.status == OledDatasetViewRowSplitStatus.ASSIGNED and assignment.split)
    findings: list[OledCuratedDatasetSplitPreflightFinding] = []
    for split_name in policy.split_names:
        if assigned_counts.get(split_name, 0) > 0:
            continue
        findings.append(
            OledCuratedDatasetSplitPreflightFinding(
                code="empty_split",
                severity="error",
                message=f"split `{split_name}` has zero assigned dataset-view rows",
                split=split_name,
            )
        )
    return findings


def _view_summaries(
    assignments: list[OledDatasetViewRowSplitAssignment],
) -> list[OledDatasetViewSplitSummary]:
    grouped: dict[tuple[str, str], list[OledDatasetViewRowSplitAssignment]] = defaultdict(list)
    for assignment in assignments:
        grouped[(assignment.view_kind, assignment.target_property_id)].append(assignment)

    summaries: list[OledDatasetViewSplitSummary] = []
    for (view_kind, target_property_id), group in sorted(grouped.items()):
        rows_by_split = Counter(
            assignment.split
            for assignment in group
            if assignment.status == OledDatasetViewRowSplitStatus.ASSIGNED and assignment.split
        )
        status_counts = Counter(assignment.status.value for assignment in group)
        reason_counts = Counter(code for assignment in group for code in assignment.reason_codes)
        summaries.append(
            OledDatasetViewSplitSummary(
                view_kind=view_kind,
                target_property_id=target_property_id,
                row_count=len(group),
                assigned_row_count=status_counts.get(OledDatasetViewRowSplitStatus.ASSIGNED.value, 0),
                unassigned_row_count=status_counts.get(OledDatasetViewRowSplitStatus.UNASSIGNED.value, 0),
                cross_split_row_count=status_counts.get(OledDatasetViewRowSplitStatus.CROSS_SPLIT_SOURCE_RECORDS.value, 0),
                rows_by_split=dict(sorted((split, count) for split, count in rows_by_split.items() if split)),
                status_counts=dict(sorted(status_counts.items())),
                reason_code_counts=dict(sorted(reason_counts.items())),
            )
        )
    return summaries


def _refresh_report(report: OledCuratedDatasetSplitPreflightReport) -> OledCuratedDatasetSplitPreflightReport:
    status_counts = Counter(assignment.status.value for assignment in report.row_assignments)
    finding_code_counts = Counter(finding.code for finding in report.findings)
    rows_by_split = Counter(
        assignment.split
        for assignment in report.row_assignments
        if assignment.status == OledDatasetViewRowSplitStatus.ASSIGNED and assignment.split
    )
    has_errors = any(finding.severity == "error" for finding in report.findings)
    has_warnings = any(finding.severity == "warning" for finding in report.findings)
    if has_errors:
        status = OledCuratedDatasetSplitPreflightStatus.FAILED
    elif has_warnings or report.gold_validation_warning_codes or report.split_leakage_warning_codes:
        status = OledCuratedDatasetSplitPreflightStatus.PASSED_WITH_WARNINGS
    else:
        status = OledCuratedDatasetSplitPreflightStatus.PASSED
    return report.model_copy(
        update={
            "status": status,
            "findings": _dedup_findings(report.findings),
            "status_counts": dict(sorted(status_counts.items())),
            "finding_code_counts": dict(sorted(finding_code_counts.items())),
            "rows_by_split": dict(sorted((split, count) for split, count in rows_by_split.items() if split)),
            "view_summaries": _view_summaries(report.row_assignments),
        }
    )


def _leakage_group_kinds(policy: OledCuratedDatasetSplitPreflightPolicy) -> list[OledLeakageGroupKind]:
    if not policy.leakage_group_kinds:
        return list(OledLeakageGroupKind)
    return [OledLeakageGroupKind(str(item)) for item in policy.leakage_group_kinds]


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _curated_manifest_integrity_code(code: str) -> str:
    if "mismatch" in code:
        return "curated_gold_manifest_sha256_mismatch"
    if "missing" in code:
        return "curated_gold_manifest_sha256_missing"
    return f"curated_gold_manifest_integrity_{code}"


def _sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"missing_dataset_view_rows_jsonl:{redact_oled_mineru_acceptance_path(file_path)}")
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _dedup_findings(
    findings: list[OledCuratedDatasetSplitPreflightFinding],
) -> list[OledCuratedDatasetSplitPreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    deduped: list[OledCuratedDatasetSplitPreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.record_id or "",
            finding.row_id or "",
            finding.view_kind or "",
            finding.target_property_id or "",
            finding.split or "",
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
            item.view_kind or "",
            item.target_property_id or "",
            item.row_id or "",
            item.split or "",
        ),
    )


def _safety_metadata() -> dict[str, Any]:
    return {
        "split_preflight_only": True,
        "split_rows_written": False,
        "training_data_written": False,
        "feature_materialization_outputs_written": False,
        "model_backends_run": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


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
    "OledCuratedDatasetSplitPreflightStatus",
    "OledDatasetViewRowSplitStatus",
    "OledCuratedDatasetSplitPreflightPolicy",
    "OledDatasetViewRowSplitAssignment",
    "OledDatasetViewSplitSummary",
    "OledCuratedDatasetSplitPreflightFinding",
    "OledCuratedDatasetSplitPreflightReport",
    "load_oled_curated_dataset_view_writer_manifest_json",
    "load_oled_curated_dataset_view_rows_from_manifest",
    "run_oled_curated_dataset_split_preflight",
    "run_oled_curated_dataset_split_preflight_from_files",
    "write_oled_curated_dataset_split_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
