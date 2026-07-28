from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from ai4s_agent.domains.oled_curated_split_training_package_writer import (
    OledCuratedSplitTrainingPackageWriteStatus,
    OledCuratedSplitTrainingPackageWriterManifest,
    OledCuratedTrainingPackageRow,
    OledCuratedTrainingPackageSchema,
    load_oled_curated_training_rows_jsonl,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledTrainingPackageBackendPreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledTrainingBackendReadinessStatus(str, Enum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class OledTrainingPackageBackendKind(str, Enum):
    TABULAR_RIDGE_SKLEARN = "tabular_ridge_sklearn"
    TABULAR_RANDOM_FOREST_SKLEARN = "tabular_random_forest_sklearn"


class OledTrainingPackageBackendPreflightPolicy(BaseModel):
    require_manifest_sha256: bool = True
    require_schema_sha256: bool = True
    require_training_rows_sha256: bool = True
    require_train_split: bool = True
    require_numeric_targets: bool = True
    require_nonempty_feature_matrix: bool = True
    require_validation_or_test_split: bool = True
    fail_on_missing_target: bool = True
    fail_on_empty_features: bool = True
    fail_on_inconsistent_feature_columns: bool = False
    min_train_rows: int = 1
    min_eval_rows: int = 1
    backend_kinds: list[str] = Field(default_factory=lambda: ["tabular_ridge_sklearn", "tabular_random_forest_sklearn"])
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])


class OledTrainingFeatureMatrixSummary(BaseModel):
    target_property_id: str
    feature_view: str

    row_count: int
    train_row_count: int
    validation_row_count: int = 0
    test_row_count: int = 0

    numeric_target_count: int = 0
    nonnumeric_target_count: int = 0
    missing_target_count: int = 0

    raw_feature_column_count: int = 0
    flattened_feature_column_count: int = 0
    missing_feature_row_count: int = 0

    flattened_feature_columns_preview: list[str] = Field(default_factory=list)
    rows_by_split: dict[str, int] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)


class OledTrainingBackendReadinessResult(BaseModel):
    backend_kind: str
    target_property_id: str
    feature_view: str

    status: OledTrainingBackendReadinessStatus

    dependency_available: bool | None = None
    dependency_name: str | None = None

    train_row_count: int = 0
    eval_row_count: int = 0
    flattened_feature_column_count: int = 0
    numeric_target_count: int = 0

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledTrainingPackageBackendPreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    backend_kind: str | None = None
    split: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    training_row_id: str | None = None
    column_name: str | None = None


class OledTrainingPackageBackendPreflightReport(BaseModel):
    status: OledTrainingPackageBackendPreflightStatus

    input_training_row_count: int

    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=list)
    backend_kinds: list[str] = Field(default_factory=list)

    matrix_summaries: list[OledTrainingFeatureMatrixSummary] = Field(default_factory=list)
    backend_results: list[OledTrainingBackendReadinessResult] = Field(default_factory=list)

    rows_by_split: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    findings: list[OledTrainingPackageBackendPreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledTrainingPackageBackendPreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_training_package_writer_manifest_json(
    path: str | Path,
) -> OledCuratedSplitTrainingPackageWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_training_package_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledCuratedSplitTrainingPackageWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_training_package_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.metadata):
        raise ValueError("absolute_path_in_training_package_manifest_metadata")
    return manifest


def load_oled_training_package_schema_json(
    path: str | Path,
) -> OledCuratedTrainingPackageSchema:
    schema_path = Path(path)
    _reject_forbidden_input(schema_path)
    if not schema_path.exists():
        raise ValueError(f"missing_training_package_schema:{redact_oled_mineru_acceptance_path(schema_path)}")
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        schema = OledCuratedTrainingPackageSchema.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_training_package_schema_json:{redact_oled_mineru_acceptance_path(schema_path)}") from exc
    if _contains_absolute_path(schema.metadata):
        raise ValueError("absolute_path_in_training_package_schema_metadata")
    return schema


def load_oled_training_rows_from_manifest(
    *,
    manifest: OledCuratedSplitTrainingPackageWriterManifest,
    base_dir: str | Path,
) -> list[OledCuratedTrainingPackageRow]:
    rows: list[OledCuratedTrainingPackageRow] = []
    for file_result in manifest.file_results:
        if file_result.artifact_kind != "training_rows":
            continue
        if _status_value(file_result.status) != OledCuratedSplitTrainingPackageWriteStatus.WRITTEN.value:
            continue
        if not file_result.output_path:
            continue
        row_path = _resolve_manifest_path(file_result.output_path, base_dir)
        _reject_forbidden_input(row_path)
        if file_result.output_sha256 is not None:
            actual_sha = _sha256_file(row_path)
            if actual_sha != file_result.output_sha256:
                raise ValueError(f"training_rows_sha256_mismatch:{redact_oled_mineru_acceptance_path(row_path)}")
        rows.extend(load_oled_curated_training_rows_jsonl(row_path))
    return sorted(rows, key=lambda row: row.training_row_id)


def load_oled_training_schema_from_manifest(
    *,
    manifest: OledCuratedSplitTrainingPackageWriterManifest,
    base_dir: str | Path,
) -> OledCuratedTrainingPackageSchema | None:
    for file_result in manifest.file_results:
        if file_result.artifact_kind != "schema":
            continue
        if _status_value(file_result.status) != OledCuratedSplitTrainingPackageWriteStatus.WRITTEN.value:
            continue
        if not file_result.output_path:
            continue
        schema_path = _resolve_manifest_path(file_result.output_path, base_dir)
        _reject_forbidden_input(schema_path)
        if file_result.output_sha256 is not None:
            actual_sha = _sha256_file(schema_path)
            if actual_sha != file_result.output_sha256:
                raise ValueError(f"training_schema_sha256_mismatch:{redact_oled_mineru_acceptance_path(schema_path)}")
        return load_oled_training_package_schema_json(schema_path)
    return None


def flatten_oled_training_features_for_preflight(
    features: dict[str, Any],
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in sorted(features.items()):
        _flatten_value(str(key), value, flattened)
    return dict(sorted(flattened.items()))


def run_oled_training_package_backend_preflight(
    *,
    training_rows: Iterable[OledCuratedTrainingPackageRow],
    schema: OledCuratedTrainingPackageSchema | None = None,
    policy: OledTrainingPackageBackendPreflightPolicy | None = None,
) -> OledTrainingPackageBackendPreflightReport:
    preflight_policy = policy or OledTrainingPackageBackendPreflightPolicy()
    input_rows = list(training_rows)
    findings: list[OledTrainingPackageBackendPreflightFinding] = []
    allowed_targets = _target_property_ids(preflight_policy)
    allowed_views = _feature_views(preflight_policy)
    allowed_splits = _splits(preflight_policy)

    selected_rows: list[OledCuratedTrainingPackageRow] = []
    for row in sorted(input_rows, key=lambda item: item.training_row_id):
        if row.target_property_id not in allowed_targets:
            continue
        if allowed_views and row.feature_view not in allowed_views:
            continue
        if row.split not in allowed_splits:
            findings.append(
                _finding(
                    "unknown_split",
                    "error",
                    "training row split is outside the allowed split set",
                    row=row,
                )
            )
            continue
        selected_rows.append(row)

    if schema is None and preflight_policy.require_schema_sha256:
        findings.append(
            OledTrainingPackageBackendPreflightFinding(
                code="missing_training_package_schema",
                severity="error",
                message="training package schema is required for backend preflight",
            )
        )

    findings.extend(_row_findings(selected_rows, preflight_policy))
    if schema is not None:
        findings.extend(_schema_findings(selected_rows, schema, preflight_policy))

    matrix_summaries = _matrix_summaries(selected_rows, preflight_policy)
    findings.extend(_matrix_findings(matrix_summaries, preflight_policy))
    backend_results = _backend_results(matrix_summaries, preflight_policy)
    findings.extend(_backend_findings(backend_results))
    findings = _dedup_findings(findings)
    status = _report_status(findings, backend_results)
    return OledTrainingPackageBackendPreflightReport(
        status=status,
        input_training_row_count=len(input_rows),
        target_property_ids=sorted({row.target_property_id for row in selected_rows}),
        feature_views=sorted({row.feature_view for row in selected_rows}),
        splits=sorted({row.split for row in selected_rows}),
        backend_kinds=_backend_kinds(preflight_policy),
        matrix_summaries=matrix_summaries,
        backend_results=backend_results,
        rows_by_split=dict(sorted(Counter(row.split for row in selected_rows).items())),
        status_counts=dict(sorted(Counter(_status_value(result.status) for result in backend_results).items())),
        finding_code_counts=dict(sorted(Counter(finding.code for finding in findings).items())),
        findings=findings,
        metadata=_safety_metadata(),
    )


def run_oled_training_package_backend_preflight_from_files(
    *,
    training_package_manifest_path: str | Path,
    training_package_base_dir: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledTrainingPackageBackendPreflightPolicy | None = None,
) -> OledTrainingPackageBackendPreflightReport:
    preflight_policy = policy or OledTrainingPackageBackendPreflightPolicy()
    manifest = load_oled_training_package_writer_manifest_json(training_package_manifest_path)
    base_dir = Path(training_package_base_dir) if training_package_base_dir is not None else Path(training_package_manifest_path).parent
    manifest_findings = _manifest_findings(manifest, preflight_policy)
    rows = load_oled_training_rows_from_manifest(manifest=manifest, base_dir=base_dir)
    schema = load_oled_training_schema_from_manifest(manifest=manifest, base_dir=base_dir)
    report = run_oled_training_package_backend_preflight(
        training_rows=rows,
        schema=schema,
        policy=preflight_policy,
    )
    if manifest_findings:
        report = _with_extra_findings(report, manifest_findings)
    if output_report_path is not None:
        write_oled_training_package_backend_preflight_report_json(report, output_report_path)
    return report


def write_oled_training_package_backend_preflight_report_json(
    report: OledTrainingPackageBackendPreflightReport,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED training package backend readiness preflight.")
    parser.add_argument("--training-package-manifest", required=True, help="Path to training package writer manifest JSON.")
    parser.add_argument("--training-package-base-dir", help="Base directory for training package artifacts.")
    parser.add_argument("--output-report", help="Optional path for backend preflight report JSON.")
    parser.add_argument("--backend-kind", action="append", default=[], help="Backend kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--split", action="append", default=[], help="Split name; repeat or comma-separate.")
    args = parser.parse_args(argv)

    try:
        policy = OledTrainingPackageBackendPreflightPolicy(
            backend_kinds=_split_cli_values(args.backend_kind) or ["tabular_ridge_sklearn", "tabular_random_forest_sklearn"],
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            splits=_split_cli_values(args.split) or ["train", "validation", "test"],
        )
        report = run_oled_training_package_backend_preflight_from_files(
            training_package_manifest_path=args.training_package_manifest,
            training_package_base_dir=args.training_package_base_dir,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": _status_value(report.status),
            "input_training_row_count": report.input_training_row_count,
            "matrix_count": len(report.matrix_summaries),
            "backend_result_count": len(report.backend_results),
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _matrix_summaries(
    rows: list[OledCuratedTrainingPackageRow],
    policy: OledTrainingPackageBackendPreflightPolicy,
) -> list[OledTrainingFeatureMatrixSummary]:
    grouped: dict[tuple[str, str], list[OledCuratedTrainingPackageRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.target_property_id, row.feature_view)].append(row)
    summaries: list[OledTrainingFeatureMatrixSummary] = []
    for (target_property_id, feature_view), group in sorted(grouped.items()):
        rows_by_split = dict(sorted(Counter(row.split for row in group).items()))
        raw_feature_columns = sorted({key for row in group for key in row.features})
        flattened_feature_columns = sorted({key for row in group for key in flatten_oled_training_features_for_preflight(row.features)})
        reason_codes: set[str] = {"matrix_built"}
        if rows_by_split.get("train", 0) < policy.min_train_rows:
            reason_codes.add("missing_train_rows")
        eval_count = rows_by_split.get("validation", 0) + rows_by_split.get("test", 0)
        if eval_count < policy.min_eval_rows:
            reason_codes.add("missing_eval_rows")
        if not flattened_feature_columns:
            reason_codes.add("empty_flattened_feature_matrix")
        summaries.append(
            OledTrainingFeatureMatrixSummary(
                target_property_id=target_property_id,
                feature_view=feature_view,
                row_count=len(group),
                train_row_count=rows_by_split.get("train", 0),
                validation_row_count=rows_by_split.get("validation", 0),
                test_row_count=rows_by_split.get("test", 0),
                numeric_target_count=sum(1 for row in group if _is_numeric_target(row.target_value)),
                nonnumeric_target_count=sum(1 for row in group if not _is_missing_value(row.target_value) and not _is_numeric_target(row.target_value)),
                missing_target_count=sum(1 for row in group if _is_missing_value(row.target_value)),
                raw_feature_column_count=len(raw_feature_columns),
                flattened_feature_column_count=len(flattened_feature_columns),
                missing_feature_row_count=sum(1 for row in group if not row.features or not flatten_oled_training_features_for_preflight(row.features)),
                flattened_feature_columns_preview=flattened_feature_columns[:20],
                rows_by_split=rows_by_split,
                reason_codes=sorted(reason_codes),
            )
        )
    return summaries


def _backend_results(
    summaries: list[OledTrainingFeatureMatrixSummary],
    policy: OledTrainingPackageBackendPreflightPolicy,
) -> list[OledTrainingBackendReadinessResult]:
    results: list[OledTrainingBackendReadinessResult] = []
    sklearn_available = importlib.util.find_spec("sklearn") is not None
    for summary in summaries:
        for backend_kind in _backend_kinds(policy):
            reason_codes: set[str] = set()
            status = OledTrainingBackendReadinessStatus.READY
            if not sklearn_available and backend_kind in {
                OledTrainingPackageBackendKind.TABULAR_RIDGE_SKLEARN.value,
                OledTrainingPackageBackendKind.TABULAR_RANDOM_FOREST_SKLEARN.value,
            }:
                status = OledTrainingBackendReadinessStatus.SKIPPED
                reason_codes.add("optional_dependency_unavailable:sklearn")
            if summary.train_row_count < policy.min_train_rows:
                status = OledTrainingBackendReadinessStatus.BLOCKED
                reason_codes.add("insufficient_train_rows")
            eval_row_count = summary.validation_row_count + summary.test_row_count
            if policy.require_validation_or_test_split and eval_row_count < policy.min_eval_rows:
                status = OledTrainingBackendReadinessStatus.BLOCKED
                reason_codes.add("insufficient_eval_rows")
            if summary.nonnumeric_target_count > 0 or summary.numeric_target_count == 0:
                status = OledTrainingBackendReadinessStatus.BLOCKED
                reason_codes.add("nonnumeric_targets_present")
            if policy.require_nonempty_feature_matrix and summary.flattened_feature_column_count == 0:
                status = OledTrainingBackendReadinessStatus.BLOCKED
                reason_codes.add("empty_flattened_feature_matrix")
            if status == OledTrainingBackendReadinessStatus.READY and summary.missing_feature_row_count:
                status = OledTrainingBackendReadinessStatus.READY_WITH_WARNINGS
                reason_codes.add("missing_feature_rows_present")
            if not reason_codes and status == OledTrainingBackendReadinessStatus.READY:
                reason_codes.add("backend_ready")
            results.append(
                OledTrainingBackendReadinessResult(
                    backend_kind=backend_kind,
                    target_property_id=summary.target_property_id,
                    feature_view=summary.feature_view,
                    status=status,
                    dependency_available=sklearn_available,
                    dependency_name="sklearn",
                    train_row_count=summary.train_row_count,
                    eval_row_count=eval_row_count,
                    flattened_feature_column_count=summary.flattened_feature_column_count,
                    numeric_target_count=summary.numeric_target_count,
                    reason_codes=sorted(reason_codes),
                    metadata={
                        "backend_preflight_only": True,
                        "backend_fit_called": False,
                        "backend_predict_called": False,
                    },
                )
            )
    return results


def _row_findings(
    rows: list[OledCuratedTrainingPackageRow],
    policy: OledTrainingPackageBackendPreflightPolicy,
) -> list[OledTrainingPackageBackendPreflightFinding]:
    findings: list[OledTrainingPackageBackendPreflightFinding] = []
    for row in rows:
        if _is_missing_value(row.target_value):
            findings.append(
                _finding(
                    "missing_target_value",
                    "error" if policy.fail_on_missing_target else "warning",
                    "training row target value is missing",
                    row=row,
                )
            )
        elif not _is_numeric_target(row.target_value):
            findings.append(
                _finding(
                    "nonnumeric_target_value",
                    "error" if policy.require_numeric_targets else "warning",
                    "training row target value is not numeric",
                    row=row,
                )
            )
        if not row.features:
            findings.append(
                _finding(
                    "empty_features",
                    "error" if policy.fail_on_empty_features else "warning",
                    "training row feature dictionary is empty",
                    row=row,
                )
            )
    return findings


def _matrix_findings(
    summaries: list[OledTrainingFeatureMatrixSummary],
    policy: OledTrainingPackageBackendPreflightPolicy,
) -> list[OledTrainingPackageBackendPreflightFinding]:
    findings: list[OledTrainingPackageBackendPreflightFinding] = []
    for summary in summaries:
        if policy.require_train_split and summary.train_row_count < policy.min_train_rows:
            findings.append(
                _summary_finding(
                    "missing_train_rows",
                    "error",
                    "feature matrix has insufficient train rows",
                    summary,
                )
            )
        eval_count = summary.validation_row_count + summary.test_row_count
        if policy.require_validation_or_test_split and eval_count < policy.min_eval_rows:
            findings.append(
                _summary_finding(
                    "missing_eval_rows",
                    "error",
                    "feature matrix has insufficient validation/test rows",
                    summary,
                )
            )
        if policy.require_nonempty_feature_matrix and summary.flattened_feature_column_count == 0:
            findings.append(
                _summary_finding(
                    "empty_flattened_feature_matrix",
                    "error",
                    "feature matrix has no flattened feature columns",
                    summary,
                )
            )
    return findings


def _backend_findings(
    results: list[OledTrainingBackendReadinessResult],
) -> list[OledTrainingPackageBackendPreflightFinding]:
    findings: list[OledTrainingPackageBackendPreflightFinding] = []
    for result in results:
        if any(code.startswith("optional_dependency_unavailable") for code in result.reason_codes):
            findings.append(
                OledTrainingPackageBackendPreflightFinding(
                    code="optional_dependency_unavailable:sklearn",
                    severity="warning",
                    message="optional sklearn dependency is unavailable; backend readiness is skipped",
                    backend_kind=result.backend_kind,
                    target_property_id=result.target_property_id,
                    feature_view=result.feature_view,
                )
            )
    return findings


def _schema_findings(
    rows: list[OledCuratedTrainingPackageRow],
    schema: OledCuratedTrainingPackageSchema,
    policy: OledTrainingPackageBackendPreflightPolicy,
) -> list[OledTrainingPackageBackendPreflightFinding]:
    row_columns = {key for row in rows for key in row.features}
    schema_columns = set(schema.feature_columns)
    missing_in_rows = sorted(schema_columns - row_columns)
    missing_in_schema = sorted(row_columns - schema_columns)
    findings: list[OledTrainingPackageBackendPreflightFinding] = []
    severity: Literal["error", "warning"] = "error" if policy.fail_on_inconsistent_feature_columns else "warning"
    for column in missing_in_rows:
        findings.append(
            OledTrainingPackageBackendPreflightFinding(
                code="schema_feature_column_mismatch",
                severity=severity,
                message="schema feature column is absent from training rows",
                column_name=column,
            )
        )
    for column in missing_in_schema:
        findings.append(
            OledTrainingPackageBackendPreflightFinding(
                code="schema_feature_column_mismatch",
                severity=severity,
                message="training row feature column is absent from schema",
                column_name=column,
            )
        )
    return findings


def _manifest_findings(
    manifest: OledCuratedSplitTrainingPackageWriterManifest,
    policy: OledTrainingPackageBackendPreflightPolicy,
) -> list[OledTrainingPackageBackendPreflightFinding]:
    findings: list[OledTrainingPackageBackendPreflightFinding] = []
    has_schema = False
    for result in manifest.file_results:
        if result.artifact_kind == "training_rows" and _status_value(result.status) == OledCuratedSplitTrainingPackageWriteStatus.WRITTEN.value:
            if policy.require_training_rows_sha256 and not result.output_sha256:
                findings.append(
                    OledTrainingPackageBackendPreflightFinding(
                        code="missing_training_rows_sha256",
                        severity="error",
                        message="training row file result is missing output_sha256",
                        split=result.split,
                        target_property_id=result.target_property_id,
                        feature_view=result.feature_view,
                    )
                )
        if result.artifact_kind == "schema" and _status_value(result.status) == OledCuratedSplitTrainingPackageWriteStatus.WRITTEN.value:
            has_schema = True
            if policy.require_schema_sha256 and not result.output_sha256:
                findings.append(
                    OledTrainingPackageBackendPreflightFinding(
                        code="missing_training_schema_sha256",
                        severity="error",
                        message="schema file result is missing output_sha256",
                    )
                )
    if policy.require_schema_sha256 and not has_schema:
        findings.append(
            OledTrainingPackageBackendPreflightFinding(
                code="missing_training_package_schema",
                severity="error",
                message="manifest does not reference a written training package schema",
            )
        )
    return findings


def _with_extra_findings(
    report: OledTrainingPackageBackendPreflightReport,
    extra_findings: list[OledTrainingPackageBackendPreflightFinding],
) -> OledTrainingPackageBackendPreflightReport:
    findings = _dedup_findings([*report.findings, *extra_findings])
    status = _report_status(findings, report.backend_results)
    return report.model_copy(
        update={
            "status": status,
            "findings": findings,
            "finding_code_counts": dict(sorted(Counter(finding.code for finding in findings).items())),
        }
    )


def _summary_finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    summary: OledTrainingFeatureMatrixSummary,
) -> OledTrainingPackageBackendPreflightFinding:
    return OledTrainingPackageBackendPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        target_property_id=summary.target_property_id,
        feature_view=summary.feature_view,
    )


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    row: OledCuratedTrainingPackageRow,
) -> OledTrainingPackageBackendPreflightFinding:
    return OledTrainingPackageBackendPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        split=row.split,
        target_property_id=row.target_property_id,
        feature_view=row.feature_view,
        training_row_id=row.training_row_id,
    )


def _report_status(
    findings: list[OledTrainingPackageBackendPreflightFinding],
    backend_results: list[OledTrainingBackendReadinessResult],
) -> OledTrainingPackageBackendPreflightStatus:
    if any(finding.severity == "error" for finding in findings):
        return OledTrainingPackageBackendPreflightStatus.FAILED
    if any(finding.severity == "warning" for finding in findings):
        return OledTrainingPackageBackendPreflightStatus.PASSED_WITH_WARNINGS
    if any(result.status in {OledTrainingBackendReadinessStatus.READY_WITH_WARNINGS, OledTrainingBackendReadinessStatus.SKIPPED} for result in backend_results):
        return OledTrainingPackageBackendPreflightStatus.PASSED_WITH_WARNINGS
    return OledTrainingPackageBackendPreflightStatus.PASSED


def _flatten_value(prefix: str, value: Any, output: dict[str, float]) -> None:
    if _is_missing_value(value):
        return
    if isinstance(value, bool):
        output[prefix] = 1.0 if value else 0.0
        return
    if isinstance(value, (int, float)):
        output[prefix] = float(value)
        return
    if isinstance(value, str):
        clean = " ".join(value.strip().lower().split())
        if clean:
            output[f"{prefix}={clean}"] = 1.0
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _flatten_value(f"{prefix}.{index}", item, output)
        return
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            _flatten_value(f"{prefix}.{key}", item, output)


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


def _target_property_ids(policy: OledTrainingPackageBackendPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.target_property_ids if str(item).strip()}


def _feature_views(policy: OledTrainingPackageBackendPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.feature_views if str(item).strip()}


def _splits(policy: OledTrainingPackageBackendPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.splits if str(item).strip()}


def _backend_kinds(policy: OledTrainingPackageBackendPreflightPolicy) -> list[str]:
    return sorted({str(item).strip() for item in policy.backend_kinds if str(item).strip()})


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _resolve_manifest_path(output_path: str, base_dir: str | Path) -> Path:
    candidate = Path(output_path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _dedup_findings(
    findings: list[OledTrainingPackageBackendPreflightFinding],
) -> list[OledTrainingPackageBackendPreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str, str, str]] = set()
    deduped: list[OledTrainingPackageBackendPreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.backend_kind or "",
            finding.split or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.training_row_id or "",
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
            item.backend_kind or "",
            item.split or "",
            item.target_property_id or "",
            item.feature_view or "",
            item.training_row_id or "",
            item.column_name or "",
        ),
    )


def _safety_metadata() -> dict[str, Any]:
    return {
        "backend_preflight_only": True,
        "baseline_backend_run": False,
        "model_backends_run": False,
        "models_trained": False,
        "predictions_written": False,
        "benchmark_results_written": False,
        "benchmark_validated": False,
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
            "training_rows",
            "feature_rows",
            "features",
            "gold_record",
            "layered_record",
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
    "OledTrainingPackageBackendPreflightStatus",
    "OledTrainingBackendReadinessStatus",
    "OledTrainingPackageBackendKind",
    "OledTrainingPackageBackendPreflightPolicy",
    "OledTrainingFeatureMatrixSummary",
    "OledTrainingBackendReadinessResult",
    "OledTrainingPackageBackendPreflightFinding",
    "OledTrainingPackageBackendPreflightReport",
    "load_oled_training_package_writer_manifest_json",
    "load_oled_training_package_schema_json",
    "load_oled_training_rows_from_manifest",
    "load_oled_training_schema_from_manifest",
    "flatten_oled_training_features_for_preflight",
    "run_oled_training_package_backend_preflight",
    "run_oled_training_package_backend_preflight_from_files",
    "write_oled_training_package_backend_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
