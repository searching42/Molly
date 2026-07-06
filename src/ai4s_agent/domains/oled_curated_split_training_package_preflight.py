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

from pydantic import BaseModel, Field, ValidationError

from ai4s_agent.domains.oled_curated_split_feature_writer import (
    OledCuratedSplitFeatureRowArtifact,
    OledCuratedSplitFeatureWriteStatus,
    OledCuratedSplitFeatureWriterManifest,
    load_oled_curated_split_feature_rows_jsonl,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledCuratedSplitTrainingPackagePreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledTrainingFeatureColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    SEQUENCE = "sequence"
    DICT = "dict"
    MIXED = "mixed"
    MISSING_ONLY = "missing_only"


class OledCuratedSplitTrainingPackagePreflightPolicy(BaseModel):
    require_manifest_sha256: bool = True
    require_nonempty_splits: bool = True
    require_train_split: bool = True
    require_target_values: bool = True
    require_evidence_refs: bool = True
    require_consistent_feature_columns: bool = True
    fail_on_duplicate_feature_row_id: bool = True
    fail_on_cross_split_feature_row_id: bool = True
    fail_on_missing_target: bool = True
    fail_on_missing_evidence: bool = True
    fail_on_missing_required_features: bool = False
    allowed_splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])
    required_feature_columns: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)


class OledTrainingFeatureColumnSummary(BaseModel):
    column_name: str
    kind: OledTrainingFeatureColumnKind
    present_count: int = 0
    missing_count: int = 0
    splits_present: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)
    example_values: list[str] = Field(default_factory=list)


class OledTrainingSplitSummary(BaseModel):
    split: str
    row_count: int
    target_property_counts: dict[str, int] = Field(default_factory=dict)
    feature_view_counts: dict[str, int] = Field(default_factory=dict)
    missing_target_count: int = 0
    missing_evidence_count: int = 0
    feature_column_count: int = 0
    missing_feature_value_counts: dict[str, int] = Field(default_factory=dict)


class OledTrainingTargetSummary(BaseModel):
    target_property_id: str
    row_count: int
    splits_present: list[str] = Field(default_factory=list)
    feature_views_present: list[str] = Field(default_factory=list)
    numeric_target_count: int = 0
    nonnumeric_target_count: int = 0
    missing_target_count: int = 0
    target_units: list[str] = Field(default_factory=list)


class OledCuratedSplitTrainingPackagePreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    split: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    feature_row_id: str | None = None
    column_name: str | None = None


class OledCuratedSplitTrainingPackagePreflightReport(BaseModel):
    status: OledCuratedSplitTrainingPackagePreflightStatus

    input_feature_row_count: int

    splits: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    split_summaries: list[OledTrainingSplitSummary] = Field(default_factory=list)
    target_summaries: list[OledTrainingTargetSummary] = Field(default_factory=list)
    feature_column_summaries: list[OledTrainingFeatureColumnSummary] = Field(default_factory=list)

    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)
    rows_by_split: dict[str, int] = Field(default_factory=dict)

    findings: list[OledCuratedSplitTrainingPackagePreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledCuratedSplitTrainingPackagePreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_curated_split_feature_writer_manifest_json(
    path: str | Path,
) -> OledCuratedSplitFeatureWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_split_feature_writer_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledCuratedSplitFeatureWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_split_feature_writer_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.metadata):
        raise ValueError(f"absolute_path_in_split_feature_writer_manifest_metadata:{manifest.manifest_id}")
    for result in manifest.file_results:
        if _contains_absolute_path(result.metadata):
            raise ValueError(
                "absolute_path_in_split_feature_writer_file_result_metadata:"
                f"{result.split}:{result.target_property_id}:{result.feature_view}"
            )
    return manifest


def load_oled_curated_split_feature_rows_from_manifest(
    *,
    manifest: OledCuratedSplitFeatureWriterManifest,
    base_dir: str | Path,
) -> list[OledCuratedSplitFeatureRowArtifact]:
    rows: list[OledCuratedSplitFeatureRowArtifact] = []
    root = Path(base_dir)
    for result in sorted(manifest.file_results, key=lambda item: (item.split, item.target_property_id, item.feature_view)):
        if _status_value(result.status) != OledCuratedSplitFeatureWriteStatus.WRITTEN.value or not result.output_jsonl_path:
            continue
        row_path = Path(result.output_jsonl_path)
        if row_path.suffix.lower() != ".jsonl":
            raise ValueError(f"forbidden_split_feature_rows_input:{redact_oled_mineru_acceptance_path(row_path)}")
        resolved_path = row_path if row_path.is_absolute() else root / row_path
        if result.output_sha256:
            actual_sha = _sha256_file(resolved_path)
            if actual_sha != result.output_sha256:
                raise ValueError(f"split_feature_rows_sha256_mismatch:{redact_oled_mineru_acceptance_path(row_path)}")
        rows.extend(load_oled_curated_split_feature_rows_jsonl(resolved_path))
    return sorted(rows, key=lambda row: row.feature_row_id)


def run_oled_curated_split_training_package_preflight(
    *,
    feature_rows: Iterable[OledCuratedSplitFeatureRowArtifact],
    policy: OledCuratedSplitTrainingPackagePreflightPolicy | None = None,
) -> OledCuratedSplitTrainingPackagePreflightReport:
    preflight_policy = policy or OledCuratedSplitTrainingPackagePreflightPolicy()
    rows = sorted(list(feature_rows), key=lambda row: (row.split, row.target_property_id, row.feature_view, row.feature_row_id))
    processed_rows = [
        row
        for row in rows
        if row.target_property_id in _target_property_ids(preflight_policy)
        and (not preflight_policy.feature_views or row.feature_view in _feature_views(preflight_policy))
    ]
    findings = _validate_rows(processed_rows, preflight_policy)
    findings = _dedup_findings(findings)
    status = _status_from_findings(findings)
    return OledCuratedSplitTrainingPackagePreflightReport(
        status=status,
        input_feature_row_count=len(rows),
        splits=sorted({row.split for row in processed_rows}),
        target_property_ids=sorted({row.target_property_id for row in processed_rows}),
        feature_views=sorted({row.feature_view for row in processed_rows}),
        split_summaries=_split_summaries(processed_rows),
        target_summaries=_target_summaries(processed_rows),
        feature_column_summaries=_feature_column_summaries(processed_rows),
        status_counts=dict(sorted(Counter(finding.severity for finding in findings).items())),
        finding_code_counts=dict(sorted(Counter(finding.code for finding in findings).items())),
        rows_by_split=dict(sorted(Counter(row.split for row in processed_rows).items())),
        findings=findings,
        metadata=_safety_metadata(),
    )


def run_oled_curated_split_training_package_preflight_from_files(
    *,
    split_feature_manifest_path: str | Path,
    split_feature_base_dir: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledCuratedSplitTrainingPackagePreflightPolicy | None = None,
) -> OledCuratedSplitTrainingPackagePreflightReport:
    preflight_policy = policy or OledCuratedSplitTrainingPackagePreflightPolicy()
    manifest = load_oled_curated_split_feature_writer_manifest_json(split_feature_manifest_path)
    base_dir = Path(split_feature_base_dir) if split_feature_base_dir is not None else Path(split_feature_manifest_path).parent
    rows = load_oled_curated_split_feature_rows_from_manifest(manifest=manifest, base_dir=base_dir)
    report = run_oled_curated_split_training_package_preflight(feature_rows=rows, policy=preflight_policy)
    manifest_findings = _manifest_sha_findings(manifest, preflight_policy)
    if manifest_findings:
        findings = _dedup_findings([*report.findings, *manifest_findings])
        report = report.model_copy(
            update={
                "status": _status_from_findings(findings),
                "findings": findings,
                "status_counts": dict(sorted(Counter(finding.severity for finding in findings).items())),
                "finding_code_counts": dict(sorted(Counter(finding.code for finding in findings).items())),
            }
        )
    if output_report_path is not None:
        write_oled_curated_split_training_package_preflight_report_json(report, output_report_path)
    return report


def write_oled_curated_split_training_package_preflight_report_json(
    report: OledCuratedSplitTrainingPackagePreflightReport,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED split training-package preflight.")
    parser.add_argument("--split-feature-manifest", required=True, help="Path to split feature writer manifest JSON.")
    parser.add_argument("--split-feature-base-dir", help="Base directory for split feature row JSONL files.")
    parser.add_argument("--output-report", help="Optional path for training-package preflight report JSON.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--required-feature-column", action="append", default=[], help="Required feature column; repeat or comma-separate.")
    args = parser.parse_args(argv)

    try:
        policy = OledCuratedSplitTrainingPackagePreflightPolicy(
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            required_feature_columns=_split_cli_values(args.required_feature_column),
        )
        report = run_oled_curated_split_training_package_preflight_from_files(
            split_feature_manifest_path=args.split_feature_manifest,
            split_feature_base_dir=args.split_feature_base_dir,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": report.status.value,
            "input_feature_row_count": report.input_feature_row_count,
            "rows_by_split": report.rows_by_split,
            "target_property_ids": report.target_property_ids,
            "feature_views": report.feature_views,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _validate_rows(
    rows: list[OledCuratedSplitFeatureRowArtifact],
    policy: OledCuratedSplitTrainingPackagePreflightPolicy,
) -> list[OledCuratedSplitTrainingPackagePreflightFinding]:
    findings: list[OledCuratedSplitTrainingPackagePreflightFinding] = []
    findings.extend(_duplicate_findings(rows, policy))
    findings.extend(_split_findings(rows, policy))
    findings.extend(_target_findings(rows, policy))
    findings.extend(_feature_schema_findings(rows, policy))
    findings.extend(_cross_split_identity_findings(rows))
    return findings


def _duplicate_findings(
    rows: list[OledCuratedSplitFeatureRowArtifact],
    policy: OledCuratedSplitTrainingPackagePreflightPolicy,
) -> list[OledCuratedSplitTrainingPackagePreflightFinding]:
    findings: list[OledCuratedSplitTrainingPackagePreflightFinding] = []
    by_feature_row_id: dict[str, list[OledCuratedSplitFeatureRowArtifact]] = defaultdict(list)
    for row in rows:
        by_feature_row_id[row.feature_row_id].append(row)
    for feature_row_id, group in sorted(by_feature_row_id.items()):
        if len(group) <= 1:
            continue
        severity = "error" if policy.fail_on_duplicate_feature_row_id else "warning"
        findings.append(
            OledCuratedSplitTrainingPackagePreflightFinding(
                code="duplicate_feature_row_id",
                severity=severity,
                message="duplicate feature row id found in split feature rows",
                feature_row_id=feature_row_id,
            )
        )
        if len({row.split for row in group}) > 1:
            findings.append(
                OledCuratedSplitTrainingPackagePreflightFinding(
                    code="cross_split_feature_row_id",
                    severity="error" if policy.fail_on_cross_split_feature_row_id else "warning",
                    message="same feature row id appears in multiple splits",
                    feature_row_id=feature_row_id,
                )
            )
    return findings


def _split_findings(
    rows: list[OledCuratedSplitFeatureRowArtifact],
    policy: OledCuratedSplitTrainingPackagePreflightPolicy,
) -> list[OledCuratedSplitTrainingPackagePreflightFinding]:
    findings: list[OledCuratedSplitTrainingPackagePreflightFinding] = []
    allowed_splits = _allowed_splits(policy)
    present_splits = {row.split for row in rows}
    for row in rows:
        if row.split not in allowed_splits:
            findings.append(
                OledCuratedSplitTrainingPackagePreflightFinding(
                    code="unknown_split",
                    severity="error",
                    message="feature row uses a split outside the allowed split set",
                    split=row.split,
                    target_property_id=row.target_property_id,
                    feature_view=row.feature_view,
                    feature_row_id=row.feature_row_id,
                )
            )
    if policy.require_train_split and "train" not in present_splits:
        findings.append(
            OledCuratedSplitTrainingPackagePreflightFinding(
                code="missing_train_split",
                severity="error",
                message="training-package preflight requires a nonempty train split",
                split="train",
            )
        )
    if policy.require_nonempty_splits:
        for split in allowed_splits:
            if split not in present_splits:
                findings.append(
                    OledCuratedSplitTrainingPackagePreflightFinding(
                        code="empty_split",
                        severity="error",
                        message="allowed split has no feature rows",
                        split=split,
                    )
                )
    return findings


def _target_findings(
    rows: list[OledCuratedSplitFeatureRowArtifact],
    policy: OledCuratedSplitTrainingPackagePreflightPolicy,
) -> list[OledCuratedSplitTrainingPackagePreflightFinding]:
    findings: list[OledCuratedSplitTrainingPackagePreflightFinding] = []
    units_by_target: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if _is_missing_value(row.target_value):
            findings.append(
                OledCuratedSplitTrainingPackagePreflightFinding(
                    code="missing_target_value",
                    severity="error" if policy.fail_on_missing_target or policy.require_target_values else "warning",
                    message="feature row has no target value",
                    split=row.split,
                    target_property_id=row.target_property_id,
                    feature_view=row.feature_view,
                    feature_row_id=row.feature_row_id,
                )
            )
        elif not _is_numeric_target(row.target_value):
            findings.append(
                OledCuratedSplitTrainingPackagePreflightFinding(
                    code="nonnumeric_target_value",
                    severity="warning",
                    message="feature row target is not numeric",
                    split=row.split,
                    target_property_id=row.target_property_id,
                    feature_view=row.feature_view,
                    feature_row_id=row.feature_row_id,
                )
            )
        if not row.evidence_refs:
            findings.append(
                OledCuratedSplitTrainingPackagePreflightFinding(
                    code="missing_evidence_refs",
                    severity="error" if policy.fail_on_missing_evidence or policy.require_evidence_refs else "warning",
                    message="feature row has no evidence refs",
                    split=row.split,
                    target_property_id=row.target_property_id,
                    feature_view=row.feature_view,
                    feature_row_id=row.feature_row_id,
                )
            )
        clean_unit = _clean_unit(row.target_unit)
        if clean_unit:
            units_by_target[row.target_property_id].add(clean_unit)
    for target_property_id, units in sorted(units_by_target.items()):
        if len(units) > 1:
            findings.append(
                OledCuratedSplitTrainingPackagePreflightFinding(
                    code="target_unit_variation",
                    severity="warning",
                    message="target property appears with multiple target units",
                    target_property_id=target_property_id,
                )
            )
    return findings


def _feature_schema_findings(
    rows: list[OledCuratedSplitFeatureRowArtifact],
    policy: OledCuratedSplitTrainingPackagePreflightPolicy,
) -> list[OledCuratedSplitTrainingPackagePreflightFinding]:
    findings: list[OledCuratedSplitTrainingPackagePreflightFinding] = []
    grouped: dict[tuple[str, str], list[OledCuratedSplitFeatureRowArtifact]] = defaultdict(list)
    for row in rows:
        grouped[(row.target_property_id, row.feature_view)].append(row)
    for (target_property_id, feature_view), group in sorted(grouped.items()):
        column_sets = {
            tuple(sorted(column for column, value in row.features.items() if not _is_missing_value(value)))
            for row in group
        }
        if len(column_sets) > 1:
            findings.append(
                OledCuratedSplitTrainingPackagePreflightFinding(
                    code="inconsistent_feature_columns",
                    severity="error" if policy.require_consistent_feature_columns else "warning",
                    message="feature columns vary within target property and feature view",
                    target_property_id=target_property_id,
                    feature_view=feature_view,
                )
            )
    required_columns = _required_feature_columns(policy)
    for row in rows:
        for column in required_columns:
            if column not in row.features:
                findings.append(
                    OledCuratedSplitTrainingPackagePreflightFinding(
                        code="required_feature_column_missing",
                        severity="error" if policy.fail_on_missing_required_features else "warning",
                        message="required feature column is absent from a feature row",
                        split=row.split,
                        target_property_id=row.target_property_id,
                        feature_view=row.feature_view,
                        feature_row_id=row.feature_row_id,
                        column_name=column,
                    )
                )
            elif _is_missing_value(row.features.get(column)):
                findings.append(
                    OledCuratedSplitTrainingPackagePreflightFinding(
                        code="required_feature_value_missing",
                        severity="error" if policy.fail_on_missing_required_features else "warning",
                        message="required feature column has a missing value",
                        split=row.split,
                        target_property_id=row.target_property_id,
                        feature_view=row.feature_view,
                        feature_row_id=row.feature_row_id,
                        column_name=column,
                    )
                )
        for column, value in sorted(row.features.items()):
            if column not in required_columns and _is_missing_value(value):
                findings.append(
                    OledCuratedSplitTrainingPackagePreflightFinding(
                        code="missing_optional_feature_values",
                        severity="warning",
                        message="optional feature column has a missing value",
                        split=row.split,
                        target_property_id=row.target_property_id,
                        feature_view=row.feature_view,
                        feature_row_id=row.feature_row_id,
                        column_name=column,
                    )
                )
    return findings


def _cross_split_identity_findings(
    rows: list[OledCuratedSplitFeatureRowArtifact],
) -> list[OledCuratedSplitTrainingPackagePreflightFinding]:
    findings: list[OledCuratedSplitTrainingPackagePreflightFinding] = []
    for code, attr in (("row_id_cross_split", "row_id"), ("record_id_cross_split", "record_id")):
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            grouped[getattr(row, attr)].add(row.split)
        for identifier, splits in sorted(grouped.items()):
            if len(splits) > 1:
                findings.append(
                    OledCuratedSplitTrainingPackagePreflightFinding(
                        code=code,
                        severity="warning",
                        message=f"{attr} appears in multiple splits; leakage preflight remains authoritative",
                        split=",".join(sorted(splits)),
                    )
                )
    return findings


def _manifest_sha_findings(
    manifest: OledCuratedSplitFeatureWriterManifest,
    policy: OledCuratedSplitTrainingPackagePreflightPolicy,
) -> list[OledCuratedSplitTrainingPackagePreflightFinding]:
    if not policy.require_manifest_sha256:
        return []
    findings: list[OledCuratedSplitTrainingPackagePreflightFinding] = []
    for result in manifest.file_results:
        if _status_value(result.status) == OledCuratedSplitFeatureWriteStatus.WRITTEN.value and result.output_jsonl_path and not result.output_sha256:
            findings.append(
                OledCuratedSplitTrainingPackagePreflightFinding(
                    code="missing_split_feature_rows_sha256",
                    severity="error",
                    message="split feature writer manifest file result lacks output_sha256",
                    split=result.split,
                    target_property_id=result.target_property_id,
                    feature_view=result.feature_view,
                )
            )
    return findings


def _split_summaries(rows: list[OledCuratedSplitFeatureRowArtifact]) -> list[OledTrainingSplitSummary]:
    grouped: dict[str, list[OledCuratedSplitFeatureRowArtifact]] = defaultdict(list)
    for row in rows:
        grouped[row.split].append(row)
    output: list[OledTrainingSplitSummary] = []
    for split, group in sorted(grouped.items()):
        feature_columns = sorted({column for row in group for column in row.features})
        output.append(
            OledTrainingSplitSummary(
                split=split,
                row_count=len(group),
                target_property_counts=dict(sorted(Counter(row.target_property_id for row in group).items())),
                feature_view_counts=dict(sorted(Counter(row.feature_view for row in group).items())),
                missing_target_count=sum(1 for row in group if _is_missing_value(row.target_value)),
                missing_evidence_count=sum(1 for row in group if not row.evidence_refs),
                feature_column_count=len(feature_columns),
                missing_feature_value_counts=_missing_counts(group, feature_columns),
            )
        )
    return output


def _target_summaries(rows: list[OledCuratedSplitFeatureRowArtifact]) -> list[OledTrainingTargetSummary]:
    grouped: dict[str, list[OledCuratedSplitFeatureRowArtifact]] = defaultdict(list)
    for row in rows:
        grouped[row.target_property_id].append(row)
    output: list[OledTrainingTargetSummary] = []
    for target_property_id, group in sorted(grouped.items()):
        output.append(
            OledTrainingTargetSummary(
                target_property_id=target_property_id,
                row_count=len(group),
                splits_present=sorted({row.split for row in group}),
                feature_views_present=sorted({row.feature_view for row in group}),
                numeric_target_count=sum(1 for row in group if _is_numeric_target(row.target_value)),
                nonnumeric_target_count=sum(1 for row in group if row.target_value is not None and not _is_numeric_target(row.target_value)),
                missing_target_count=sum(1 for row in group if _is_missing_value(row.target_value)),
                target_units=sorted({_clean_unit(row.target_unit) for row in group if _clean_unit(row.target_unit)}),
            )
        )
    return output


def _feature_column_summaries(rows: list[OledCuratedSplitFeatureRowArtifact]) -> list[OledTrainingFeatureColumnSummary]:
    columns = sorted({column for row in rows for column in row.features})
    summaries: list[OledTrainingFeatureColumnSummary] = []
    for column in columns:
        values = [row.features.get(column) for row in rows if column in row.features and not _is_missing_value(row.features.get(column))]
        present_rows = [row for row in rows if column in row.features and not _is_missing_value(row.features.get(column))]
        summaries.append(
            OledTrainingFeatureColumnSummary(
                column_name=column,
                kind=_column_kind(values),
                present_count=len(present_rows),
                missing_count=len(rows) - len(present_rows),
                splits_present=sorted({row.split for row in present_rows}),
                target_property_ids=sorted({row.target_property_id for row in present_rows}),
                feature_views=sorted({row.feature_view for row in present_rows}),
                example_values=_example_values(values),
            )
        )
    return summaries


def _missing_counts(rows: list[OledCuratedSplitFeatureRowArtifact], columns: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for column in columns:
            if column not in row.features or _is_missing_value(row.features.get(column)):
                counts[column] += 1
    return dict(sorted(counts.items()))


def _column_kind(values: list[Any]) -> OledTrainingFeatureColumnKind:
    if not values:
        return OledTrainingFeatureColumnKind.MISSING_ONLY
    kinds = {_value_kind(value) for value in values}
    if len(kinds) == 1:
        return next(iter(kinds))
    return OledTrainingFeatureColumnKind.MIXED


def _value_kind(value: Any) -> OledTrainingFeatureColumnKind:
    if isinstance(value, bool):
        return OledTrainingFeatureColumnKind.BOOLEAN
    if isinstance(value, (int, float)):
        return OledTrainingFeatureColumnKind.NUMERIC
    if isinstance(value, str):
        return OledTrainingFeatureColumnKind.CATEGORICAL
    if isinstance(value, (list, tuple, set)):
        return OledTrainingFeatureColumnKind.SEQUENCE
    if isinstance(value, dict):
        return OledTrainingFeatureColumnKind.DICT
    return OledTrainingFeatureColumnKind.MIXED


def _example_values(values: list[Any]) -> list[str]:
    examples: set[str] = set()
    for value in values:
        text = json.dumps(_sanitize_for_output(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if len(text) > 80:
            text = text[:77] + "..."
        examples.add(text)
    return sorted(examples)[:3]


def _status_from_findings(
    findings: list[OledCuratedSplitTrainingPackagePreflightFinding],
) -> OledCuratedSplitTrainingPackagePreflightStatus:
    if any(finding.severity == "error" for finding in findings):
        return OledCuratedSplitTrainingPackagePreflightStatus.FAILED
    if any(finding.severity == "warning" for finding in findings):
        return OledCuratedSplitTrainingPackagePreflightStatus.PASSED_WITH_WARNINGS
    return OledCuratedSplitTrainingPackagePreflightStatus.PASSED


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _is_numeric_target(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _clean_unit(value: str | None) -> str | None:
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _target_property_ids(policy: OledCuratedSplitTrainingPackagePreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.target_property_ids if str(item).strip()}


def _feature_views(policy: OledCuratedSplitTrainingPackagePreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.feature_views if str(item).strip()}


def _allowed_splits(policy: OledCuratedSplitTrainingPackagePreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.allowed_splits if str(item).strip()}


def _required_feature_columns(policy: OledCuratedSplitTrainingPackagePreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.required_feature_columns if str(item).strip()}


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"missing_split_feature_rows_jsonl:{redact_oled_mineru_acceptance_path(file_path)}")
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _dedup_findings(
    findings: list[OledCuratedSplitTrainingPackagePreflightFinding],
) -> list[OledCuratedSplitTrainingPackagePreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    deduped: list[OledCuratedSplitTrainingPackagePreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.split or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.feature_row_id or "",
            finding.column_name or "",
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
            item.target_property_id or "",
            item.feature_view or "",
            item.feature_row_id or "",
            item.column_name or "",
        ),
    )


def _safety_metadata() -> dict[str, Any]:
    return {
        "training_package_preflight_only": True,
        "training_package_written": False,
        "ml_ready_training_data_written": False,
        "model_backends_run": False,
        "baseline_backend_run": False,
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
            "feature_row_artifacts",
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
    "OledCuratedSplitTrainingPackagePreflightStatus",
    "OledTrainingFeatureColumnKind",
    "OledCuratedSplitTrainingPackagePreflightPolicy",
    "OledTrainingFeatureColumnSummary",
    "OledTrainingSplitSummary",
    "OledTrainingTargetSummary",
    "OledCuratedSplitTrainingPackagePreflightFinding",
    "OledCuratedSplitTrainingPackagePreflightReport",
    "load_oled_curated_split_feature_writer_manifest_json",
    "load_oled_curated_split_feature_rows_from_manifest",
    "run_oled_curated_split_training_package_preflight",
    "run_oled_curated_split_training_package_preflight_from_files",
    "write_oled_curated_split_training_package_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
