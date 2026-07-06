from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai4s_agent.domains.oled_curated_split_training_package_writer import (
    OledCuratedSplitTrainingPackageWriterManifest,
    OledCuratedTrainingPackageRow,
    OledCuratedTrainingPackageSchema,
)
from ai4s_agent.domains.oled_curated_training_package_backend_preflight import (
    OledTrainingPackageBackendPreflightReport,
    OledTrainingPackageBackendPreflightStatus,
    flatten_oled_training_features_for_preflight,
    load_oled_training_package_writer_manifest_json,
    load_oled_training_rows_from_manifest,
    load_oled_training_schema_from_manifest,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledCuratedTrainingBaselineKind(str, Enum):
    MEAN_BASELINE = "mean_baseline"
    TABULAR_RIDGE_SKLEARN = "tabular_ridge_sklearn"
    TABULAR_RANDOM_FOREST_SKLEARN = "tabular_random_forest_sklearn"


class OledCuratedTrainingBaselineRunnerPolicy(BaseModel):
    require_confirmation: bool = True
    require_backend_preflight_valid: bool = True
    allow_backend_preflight_warnings: bool = True
    require_train_split: bool = True
    require_eval_split: bool = True
    require_numeric_targets: bool = True
    require_nonempty_features: bool = True

    baseline_kinds: list[str] = Field(default_factory=lambda: ["mean_baseline"])
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])

    write_predictions: bool = True
    write_metrics: bool = True
    benchmark_validated: bool = False
    register_benchmark: bool = False


class OledCuratedTrainingBaselineRunStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class OledCuratedTrainingBaselinePrediction(BaseModel):
    prediction_id: str

    baseline_kind: str
    split: str
    target_property_id: str
    feature_view: str

    training_row_id: str
    record_id: str
    feature_row_id: str | None = None

    y_true: float | int | None = None
    y_pred: float | int | None = None
    residual: float | None = None
    absolute_error: float | None = None

    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_refs")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedTrainingBaselineMetrics(BaseModel):
    baseline_kind: str
    target_property_id: str
    feature_view: str
    split: str

    row_count: int = 0
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    bias: float | None = None
    target_mean: float | None = None
    prediction_mean: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class OledCuratedTrainingBaselineRunResult(BaseModel):
    baseline_kind: str
    target_property_id: str
    feature_view: str

    status: OledCuratedTrainingBaselineRunStatus

    train_row_count: int = 0
    validation_row_count: int = 0
    test_row_count: int = 0
    prediction_count: int = 0

    metric_splits: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    prediction_jsonl_path: str | None = None
    prediction_sha256: str | None = None
    metrics_json_path: str | None = None
    metrics_sha256: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metric_splits", "reason_codes")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedTrainingBaselineRunnerFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    split: str | None = None
    training_row_id: str | None = None


class OledCuratedTrainingBaselineRunnerManifest(BaseModel):
    manifest_id: str

    source_training_package_manifest_id: str | None = None
    source_backend_preflight_status: str | None = None

    output_directory: str | None = None
    output_file_count: int = 0
    output_prediction_count: int = 0

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    run_results: list[OledCuratedTrainingBaselineRunResult] = Field(default_factory=list)

    status_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)

    policy: OledCuratedTrainingBaselineRunnerPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(result.status == OledCuratedTrainingBaselineRunStatus.FAILED for result in self.run_results)


class OledCuratedTrainingBaselineRunnerReport(BaseModel):
    manifest: OledCuratedTrainingBaselineRunnerManifest
    predictions: list[OledCuratedTrainingBaselinePrediction] = Field(default_factory=list)
    metrics: list[OledCuratedTrainingBaselineMetrics] = Field(default_factory=list)
    findings: list[OledCuratedTrainingBaselineRunnerFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_training_package_backend_preflight_report_json(
    path: str | Path,
) -> OledTrainingPackageBackendPreflightReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_backend_preflight_report:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = OledTrainingPackageBackendPreflightReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_backend_preflight_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    if _contains_absolute_path(report.metadata):
        raise ValueError("absolute_path_in_backend_preflight_report_metadata")
    return report


def select_oled_training_baselines_for_run(
    *,
    training_rows: Iterable[OledCuratedTrainingPackageRow],
    schema: OledCuratedTrainingPackageSchema | None,
    backend_preflight_report: OledTrainingPackageBackendPreflightReport,
    policy: OledCuratedTrainingBaselineRunnerPolicy | None = None,
    confirm_baseline_run: bool = False,
) -> OledCuratedTrainingBaselineRunnerReport:
    runner_policy = policy or OledCuratedTrainingBaselineRunnerPolicy()
    if runner_policy.require_confirmation and not confirm_baseline_run:
        raise ValueError("confirmation_required:baseline_run")

    rows = _filter_rows(list(training_rows), runner_policy)
    findings = _preflight_gate_findings(backend_preflight_report, runner_policy)
    if any(finding.severity == "error" for finding in findings):
        return OledCuratedTrainingBaselineRunnerReport(
            manifest=_manifest(
                policy=runner_policy,
                run_results=[],
                predictions=[],
                source_backend_preflight_status=_status_value(backend_preflight_report.status),
                baseline_backend_run=False,
                predictions_written=False,
                metrics_written=False,
            ),
            predictions=[],
            metrics=[],
            findings=findings,
        )

    predictions: list[OledCuratedTrainingBaselinePrediction] = []
    metrics: list[OledCuratedTrainingBaselineMetrics] = []
    run_results: list[OledCuratedTrainingBaselineRunResult] = []
    targets = _selected_targets(rows, runner_policy)
    views = _selected_feature_views(rows, runner_policy)

    for baseline_kind in _baseline_kinds(runner_policy):
        if baseline_kind not in {kind.value for kind in OledCuratedTrainingBaselineKind}:
            finding = OledCuratedTrainingBaselineRunnerFinding(
                code="unsupported_baseline_kind",
                severity="warning",
                message="requested baseline kind is not supported",
                baseline_kind=baseline_kind,
            )
            findings.append(finding)
            run_results.append(_run_result(baseline_kind, "", "", OledCuratedTrainingBaselineRunStatus.SKIPPED, [], [], ["unsupported_baseline_kind"]))
            continue
        for target_property_id in targets:
            for feature_view in views:
                group = _rows_for_target_view(rows, target_property_id, feature_view)
                if not group:
                    continue
                validation_findings = _baseline_group_findings(group, baseline_kind, target_property_id, feature_view, runner_policy)
                if any(finding.severity == "error" for finding in validation_findings):
                    findings.extend(validation_findings)
                    run_results.append(_run_result(baseline_kind, target_property_id, feature_view, OledCuratedTrainingBaselineRunStatus.FAILED, [], [], sorted({finding.code for finding in validation_findings})))
                    continue
                if baseline_kind == OledCuratedTrainingBaselineKind.MEAN_BASELINE.value:
                    run_predictions, run_metrics = run_oled_mean_baseline_on_training_rows(
                        group,
                        target_property_id=target_property_id,
                        feature_view=feature_view,
                    )
                    reason_codes = ["baseline_completed", "selected_for_run"]
                    status = OledCuratedTrainingBaselineRunStatus.COMPLETED
                    run_findings: list[OledCuratedTrainingBaselineRunnerFinding] = []
                else:
                    run_predictions, run_metrics, run_findings = run_oled_sklearn_baseline_on_training_rows(
                        group,
                        baseline_kind=baseline_kind,
                        target_property_id=target_property_id,
                        feature_view=feature_view,
                    )
                    findings.extend(run_findings)
                    if run_predictions:
                        reason_codes = ["baseline_completed", "selected_for_run"]
                        status = OledCuratedTrainingBaselineRunStatus.COMPLETED
                    else:
                        reason_codes = sorted({finding.code for finding in run_findings}) or ["baseline_skipped"]
                        status = OledCuratedTrainingBaselineRunStatus.SKIPPED
                predictions.extend(run_predictions)
                metrics.extend(run_metrics)
                run_results.append(_run_result(baseline_kind, target_property_id, feature_view, status, run_predictions, run_metrics, reason_codes))

    findings = _dedup_findings(findings)
    predictions = sorted(predictions, key=lambda prediction: prediction.prediction_id)
    metrics = sorted(metrics, key=lambda metric: (metric.baseline_kind, metric.target_property_id, metric.feature_view, metric.split))
    return OledCuratedTrainingBaselineRunnerReport(
        manifest=_manifest(
            policy=runner_policy,
            run_results=run_results,
            predictions=predictions,
            source_backend_preflight_status=_status_value(backend_preflight_report.status),
            baseline_backend_run=bool(predictions),
            predictions_written=False,
            metrics_written=False,
        ),
        predictions=predictions,
        metrics=metrics,
        findings=findings,
    )


def run_oled_mean_baseline_on_training_rows(
    rows: Iterable[OledCuratedTrainingPackageRow],
    *,
    target_property_id: str,
    feature_view: str,
) -> tuple[list[OledCuratedTrainingBaselinePrediction], list[OledCuratedTrainingBaselineMetrics]]:
    group = _rows_for_target_view(list(rows), target_property_id, feature_view)
    train_values = [_numeric_target(row.target_value) for row in group if row.split == "train" and _numeric_target(row.target_value) is not None]
    if not train_values:
        raise ValueError("missing_numeric_train_targets")
    train_mean = sum(train_values) / len(train_values)
    predictions = [_prediction_from_row(row, baseline_kind=OledCuratedTrainingBaselineKind.MEAN_BASELINE.value, y_pred=train_mean) for row in group]
    return predictions, _metrics_for_predictions(predictions)


def run_oled_sklearn_baseline_on_training_rows(
    rows: Iterable[OledCuratedTrainingPackageRow],
    *,
    baseline_kind: OledCuratedTrainingBaselineKind | str,
    target_property_id: str,
    feature_view: str,
) -> tuple[list[OledCuratedTrainingBaselinePrediction], list[OledCuratedTrainingBaselineMetrics], list[OledCuratedTrainingBaselineRunnerFinding]]:
    clean_baseline = _status_value(baseline_kind)
    if importlib.util.find_spec("sklearn") is None:
        return [], [], [
            OledCuratedTrainingBaselineRunnerFinding(
                code="optional_dependency_unavailable:sklearn",
                severity="warning",
                message="sklearn is unavailable; sklearn baseline skipped",
                baseline_kind=clean_baseline,
                target_property_id=target_property_id,
                feature_view=feature_view,
            )
        ]
    group = _rows_for_target_view(list(rows), target_property_id, feature_view)
    train_rows = [row for row in group if row.split == "train" and _numeric_target(row.target_value) is not None]
    if not train_rows:
        return [], [], [
            OledCuratedTrainingBaselineRunnerFinding(
                code="missing_numeric_train_targets",
                severity="error",
                message="sklearn baseline requires numeric train targets",
                baseline_kind=clean_baseline,
                target_property_id=target_property_id,
                feature_view=feature_view,
            )
        ]
    feature_columns = _feature_columns(group)
    if not feature_columns:
        return [], [], [
            OledCuratedTrainingBaselineRunnerFinding(
                code="empty_features",
                severity="error",
                message="sklearn baseline requires nonempty flattened features",
                baseline_kind=clean_baseline,
                target_property_id=target_property_id,
                feature_view=feature_view,
            )
        ]

    if clean_baseline == OledCuratedTrainingBaselineKind.TABULAR_RIDGE_SKLEARN.value:
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)
    elif clean_baseline == OledCuratedTrainingBaselineKind.TABULAR_RANDOM_FOREST_SKLEARN.value:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=64, random_state=0)
    else:
        return [], [], [
            OledCuratedTrainingBaselineRunnerFinding(
                code="unsupported_baseline_kind",
                severity="warning",
                message="requested sklearn baseline kind is not supported",
                baseline_kind=clean_baseline,
                target_property_id=target_property_id,
                feature_view=feature_view,
            )
        ]
    model.fit(_feature_matrix(train_rows, feature_columns), [float(row.target_value) for row in train_rows])
    y_pred = [float(value) for value in model.predict(_feature_matrix(group, feature_columns))]
    predictions = [
        _prediction_from_row(row, baseline_kind=clean_baseline, y_pred=prediction)
        for row, prediction in zip(group, y_pred, strict=True)
    ]
    return predictions, _metrics_for_predictions(predictions), []


def write_oled_training_baseline_predictions_jsonl(
    predictions: Iterable[OledCuratedTrainingBaselinePrediction],
    path: str | Path,
) -> str:
    lines = [
        json.dumps(_sanitize_for_output(prediction.model_dump(mode="json", exclude_none=True)), sort_keys=True, separators=(",", ":"))
        for prediction in sorted(predictions, key=lambda item: item.prediction_id)
    ]
    payload = "\n".join(lines) + ("\n" if lines else "")
    encoded = payload.encode("utf-8")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_oled_training_baseline_metrics_json(
    metrics: Iterable[OledCuratedTrainingBaselineMetrics],
    path: str | Path,
) -> str:
    payload = json.dumps(
        [
            _sanitize_for_output(metric.model_dump(mode="json", exclude_none=True))
            for metric in sorted(metrics, key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view, item.split))
        ],
        sort_keys=True,
        indent=2,
    ) + "\n"
    encoded = payload.encode("utf-8")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_oled_training_baseline_manifest_json(
    manifest: OledCuratedTrainingBaselineRunnerManifest,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def oled_training_baseline_predictions_filename(
    *,
    baseline_kind: str,
    target_property_id: str,
    feature_view: str,
) -> str:
    return (
        "oled_baseline_predictions__"
        f"{_safe_filename_token(baseline_kind)}__"
        f"{_safe_filename_token(target_property_id)}__"
        f"{_safe_filename_token(feature_view)}.jsonl"
    )


def oled_training_baseline_metrics_filename(
    *,
    baseline_kind: str,
    target_property_id: str,
    feature_view: str,
) -> str:
    return (
        "oled_baseline_metrics__"
        f"{_safe_filename_token(baseline_kind)}__"
        f"{_safe_filename_token(target_property_id)}__"
        f"{_safe_filename_token(feature_view)}.json"
    )


def run_oled_curated_training_baseline_runner_from_files(
    *,
    training_package_manifest_path: str | Path,
    backend_preflight_report_path: str | Path,
    training_package_base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    policy: OledCuratedTrainingBaselineRunnerPolicy | None = None,
    confirm_baseline_run: bool = False,
    dry_run: bool = False,
) -> OledCuratedTrainingBaselineRunnerReport:
    runner_policy = policy or OledCuratedTrainingBaselineRunnerPolicy()
    if not dry_run and runner_policy.require_confirmation and not confirm_baseline_run:
        raise ValueError("confirmation_required:baseline_run")
    if not dry_run and output_dir is None:
        raise ValueError("output_dir_required:baseline_run")
    training_manifest = load_oled_training_package_writer_manifest_json(training_package_manifest_path)
    base_dir = Path(training_package_base_dir) if training_package_base_dir is not None else Path(training_package_manifest_path).parent
    training_rows = load_oled_training_rows_from_manifest(manifest=training_manifest, base_dir=base_dir)
    schema = load_oled_training_schema_from_manifest(manifest=training_manifest, base_dir=base_dir)
    backend_preflight_report = load_oled_training_package_backend_preflight_report_json(backend_preflight_report_path)
    selection_policy = runner_policy.model_copy(update={"require_confirmation": not dry_run and runner_policy.require_confirmation})
    report = select_oled_training_baselines_for_run(
        training_rows=training_rows,
        schema=schema,
        backend_preflight_report=backend_preflight_report,
        policy=selection_policy,
        confirm_baseline_run=confirm_baseline_run or dry_run,
    )
    report = _attach_source_context(
        report,
        source_training_package_manifest_id=training_manifest.manifest_id,
        source_backend_preflight_status=_status_value(backend_preflight_report.status),
    )
    if dry_run:
        report = _mark_dry_run(report)
        if output_manifest_path is not None:
            write_oled_training_baseline_manifest_json(report.manifest, output_manifest_path)
        return report
    if not report.is_valid:
        if output_manifest_path is not None:
            write_oled_training_baseline_manifest_json(report.manifest, output_manifest_path)
        return report

    assert output_dir is not None
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report = _write_baseline_run_files(report, output_root)
    if output_manifest_path is not None:
        write_oled_training_baseline_manifest_json(report.manifest, output_manifest_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run controlled OLED training package baseline artifacts.")
    parser.add_argument("--training-package-manifest", required=True, help="Path to training package manifest JSON.")
    parser.add_argument("--backend-preflight-report", required=True, help="Path to backend preflight report JSON.")
    parser.add_argument("--training-package-base-dir", help="Base directory for training package artifacts.")
    parser.add_argument("--output-dir", help="Directory for baseline prediction/metrics artifacts.")
    parser.add_argument("--output-manifest", help="Optional path for baseline run manifest JSON.")
    parser.add_argument("--confirm-baseline-run", action="store_true", help="Confirm baseline execution and artifact writing.")
    parser.add_argument("--dry-run", action="store_true", help="Run selection and baseline assembly without writing prediction/metrics files.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    args = parser.parse_args(argv)

    if not args.output_dir and not args.output_manifest:
        print("output_required:dir_or_manifest", file=sys.stderr)
        return 1
    if not args.dry_run and not args.confirm_baseline_run:
        print("confirmation_required:baseline_run", file=sys.stderr)
        return 1
    try:
        policy = OledCuratedTrainingBaselineRunnerPolicy(
            require_confirmation=not args.dry_run,
            baseline_kinds=_split_cli_values(args.baseline_kind) or ["mean_baseline"],
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
        )
        report = run_oled_curated_training_baseline_runner_from_files(
            training_package_manifest_path=args.training_package_manifest,
            backend_preflight_report_path=args.backend_preflight_report,
            training_package_base_dir=args.training_package_base_dir,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            policy=policy,
            confirm_baseline_run=args.confirm_baseline_run,
            dry_run=args.dry_run,
        )
        summary = {
            "output_file_count": report.manifest.output_file_count,
            "output_prediction_count": report.manifest.output_prediction_count,
            "status_counts": report.manifest.status_counts,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _prediction_from_row(
    row: OledCuratedTrainingPackageRow,
    *,
    baseline_kind: str,
    y_pred: float,
) -> OledCuratedTrainingBaselinePrediction:
    y_true = _numeric_target(row.target_value)
    residual = round(y_pred - y_true, 6) if y_true is not None else None
    absolute_error = round(abs(residual), 6) if residual is not None else None
    prediction = OledCuratedTrainingBaselinePrediction(
        prediction_id="",
        baseline_kind=baseline_kind,
        split=row.split,
        target_property_id=row.target_property_id,
        feature_view=row.feature_view,
        training_row_id=row.training_row_id,
        record_id=row.record_id,
        feature_row_id=row.feature_row_id,
        y_true=y_true,
        y_pred=round(float(y_pred), 6),
        residual=residual,
        absolute_error=absolute_error,
        evidence_refs=row.evidence_refs,
        metadata={
            "baseline_prediction": True,
            "benchmark_validated": False,
            "model_backend_run": True,
            "llm_called": False,
            "mineru_called": False,
        },
    )
    return prediction.model_copy(update={"prediction_id": _prediction_id(prediction)})


def _prediction_id(prediction: OledCuratedTrainingBaselinePrediction) -> str:
    payload = {
        "baseline_kind": prediction.baseline_kind,
        "target_property_id": prediction.target_property_id,
        "feature_view": prediction.feature_view,
        "training_row_id": prediction.training_row_id,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-baseline-prediction:{digest[:20]}"


def _metrics_for_predictions(
    predictions: list[OledCuratedTrainingBaselinePrediction],
) -> list[OledCuratedTrainingBaselineMetrics]:
    grouped: dict[tuple[str, str, str, str], list[OledCuratedTrainingBaselinePrediction]] = defaultdict(list)
    for prediction in predictions:
        if prediction.y_true is None or prediction.y_pred is None:
            continue
        grouped[(prediction.baseline_kind, prediction.target_property_id, prediction.feature_view, prediction.split)].append(prediction)
    metrics: list[OledCuratedTrainingBaselineMetrics] = []
    for (baseline_kind, target_property_id, feature_view, split), group in sorted(grouped.items()):
        y_true = [float(prediction.y_true) for prediction in group if prediction.y_true is not None]
        y_pred = [float(prediction.y_pred) for prediction in group if prediction.y_pred is not None]
        values = _regression_metrics(y_true, y_pred)
        metrics.append(
            OledCuratedTrainingBaselineMetrics(
                baseline_kind=baseline_kind,
                target_property_id=target_property_id,
                feature_view=feature_view,
                split=split,
                row_count=len(group),
                metadata={
                    "baseline_metrics": True,
                    "benchmark_validated": False,
                    "model_backend_run": True,
                },
                **values,
            )
        )
    return metrics


def _regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    n = len(y_true)
    errors = [predicted - actual for actual, predicted in zip(y_true, y_pred, strict=True)]
    absolute_errors = [abs(error) for error in errors]
    squared_errors = [error * error for error in errors]
    target_mean = sum(y_true) / n
    prediction_mean = sum(y_pred) / n
    sse = sum(squared_errors)
    sst = sum((value - target_mean) ** 2 for value in y_true)
    r2 = 0.0 if sst == 0 else 1.0 - (sse / sst)
    return {
        "bias": round(sum(errors) / n, 6),
        "mae": round(sum(absolute_errors) / n, 6),
        "prediction_mean": round(prediction_mean, 6),
        "r2": round(r2, 6),
        "rmse": round(math.sqrt(sse / n), 6),
        "target_mean": round(target_mean, 6),
    }


def _feature_columns(rows: list[OledCuratedTrainingPackageRow]) -> list[str]:
    return sorted({column for row in rows for column in flatten_oled_training_features_for_preflight(row.features)})


def _feature_matrix(rows: list[OledCuratedTrainingPackageRow], columns: list[str]) -> list[list[float]]:
    flattened_rows = [flatten_oled_training_features_for_preflight(row.features) for row in rows]
    return [[float(flattened.get(column, 0.0)) for column in columns] for flattened in flattened_rows]


def _baseline_group_findings(
    rows: list[OledCuratedTrainingPackageRow],
    baseline_kind: str,
    target_property_id: str,
    feature_view: str,
    policy: OledCuratedTrainingBaselineRunnerPolicy,
) -> list[OledCuratedTrainingBaselineRunnerFinding]:
    findings: list[OledCuratedTrainingBaselineRunnerFinding] = []
    train_numeric = [row for row in rows if row.split == "train" and _numeric_target(row.target_value) is not None]
    if policy.require_train_split and not train_numeric:
        findings.append(
            _finding(
                "missing_numeric_train_targets",
                "error",
                "baseline run requires numeric train targets",
                baseline_kind=baseline_kind,
                target_property_id=target_property_id,
                feature_view=feature_view,
            )
        )
    eval_rows = [row for row in rows if row.split in {"validation", "test"} and _numeric_target(row.target_value) is not None]
    if policy.require_eval_split and not eval_rows:
        findings.append(
            _finding(
                "missing_eval_rows",
                "error",
                "baseline run requires validation or test rows",
                baseline_kind=baseline_kind,
                target_property_id=target_property_id,
                feature_view=feature_view,
            )
        )
    if policy.require_nonempty_features and not _feature_columns(rows):
        findings.append(
            _finding(
                "empty_features",
                "error",
                "baseline run requires nonempty flattened features",
                baseline_kind=baseline_kind,
                target_property_id=target_property_id,
                feature_view=feature_view,
            )
        )
    return findings


def _preflight_gate_findings(
    backend_preflight_report: OledTrainingPackageBackendPreflightReport,
    policy: OledCuratedTrainingBaselineRunnerPolicy,
) -> list[OledCuratedTrainingBaselineRunnerFinding]:
    findings: list[OledCuratedTrainingBaselineRunnerFinding] = []
    if policy.require_backend_preflight_valid and (
        _status_value(backend_preflight_report.status) == OledTrainingPackageBackendPreflightStatus.FAILED.value
        or not backend_preflight_report.is_valid
    ):
        findings.append(
            OledCuratedTrainingBaselineRunnerFinding(
                code="backend_preflight_failed",
                severity="error",
                message="baseline runner blocked because backend preflight report is invalid",
            )
        )
    if not policy.allow_backend_preflight_warnings and backend_preflight_report.warning_codes:
        findings.append(
            OledCuratedTrainingBaselineRunnerFinding(
                code="backend_preflight_warnings_present",
                severity="error",
                message="baseline runner blocked because backend preflight warnings are disallowed",
            )
        )
    return findings


def _run_result(
    baseline_kind: str,
    target_property_id: str,
    feature_view: str,
    status: OledCuratedTrainingBaselineRunStatus,
    predictions: list[OledCuratedTrainingBaselinePrediction],
    metrics: list[OledCuratedTrainingBaselineMetrics],
    reason_codes: list[str],
) -> OledCuratedTrainingBaselineRunResult:
    group_predictions = [prediction for prediction in predictions if prediction.baseline_kind == baseline_kind and prediction.target_property_id == target_property_id and prediction.feature_view == feature_view]
    return OledCuratedTrainingBaselineRunResult(
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        status=status,
        train_row_count=sum(1 for prediction in group_predictions if prediction.split == "train"),
        validation_row_count=sum(1 for prediction in group_predictions if prediction.split == "validation"),
        test_row_count=sum(1 for prediction in group_predictions if prediction.split == "test"),
        prediction_count=len(group_predictions),
        metric_splits=sorted({metric.split for metric in metrics if metric.baseline_kind == baseline_kind and metric.target_property_id == target_property_id and metric.feature_view == feature_view}),
        reason_codes=reason_codes,
        metadata={
            "baseline_run_artifact": True,
            "benchmark_validated": False,
        },
    )


def _manifest(
    *,
    policy: OledCuratedTrainingBaselineRunnerPolicy,
    run_results: list[OledCuratedTrainingBaselineRunResult],
    predictions: list[OledCuratedTrainingBaselinePrediction],
    source_training_package_manifest_id: str | None = None,
    source_backend_preflight_status: str | None = None,
    output_directory: str | None = None,
    baseline_backend_run: bool,
    predictions_written: bool,
    metrics_written: bool,
) -> OledCuratedTrainingBaselineRunnerManifest:
    return OledCuratedTrainingBaselineRunnerManifest(
        manifest_id=_manifest_id(policy, run_results),
        source_training_package_manifest_id=source_training_package_manifest_id,
        source_backend_preflight_status=source_backend_preflight_status,
        output_directory=output_directory,
        output_file_count=sum(
            int(bool(result.prediction_sha256)) + int(bool(result.metrics_sha256))
            for result in run_results
            if result.status == OledCuratedTrainingBaselineRunStatus.COMPLETED
        ),
        output_prediction_count=sum(result.prediction_count for result in run_results if result.status == OledCuratedTrainingBaselineRunStatus.COMPLETED),
        baseline_kinds=sorted({result.baseline_kind for result in run_results if result.baseline_kind}),
        target_property_ids=sorted({result.target_property_id for result in run_results if result.target_property_id}),
        feature_views=sorted({result.feature_view for result in run_results if result.feature_view}),
        run_results=sorted(run_results, key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view)),
        status_counts=dict(sorted(Counter(_status_value(result.status) for result in run_results).items())),
        reason_code_counts=dict(sorted(Counter(code for result in run_results for code in result.reason_codes).items())),
        policy=policy,
        metadata=_safety_metadata(
            baseline_backend_run=baseline_backend_run,
            models_fitted=baseline_backend_run,
            predictions_written=predictions_written,
            metrics_written=metrics_written,
        ),
    )


def _manifest_id(
    policy: OledCuratedTrainingBaselineRunnerPolicy,
    run_results: list[OledCuratedTrainingBaselineRunResult],
) -> str:
    payload = {
        "policy": policy.model_dump(mode="json"),
        "run_results": [
            {
                "baseline_kind": result.baseline_kind,
                "target_property_id": result.target_property_id,
                "feature_view": result.feature_view,
                "status": _status_value(result.status),
                "prediction_count": result.prediction_count,
            }
            for result in sorted(run_results, key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view))
        ],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-baseline-runner:{digest[:16]}"


def _attach_source_context(
    report: OledCuratedTrainingBaselineRunnerReport,
    *,
    source_training_package_manifest_id: str | None,
    source_backend_preflight_status: str | None,
) -> OledCuratedTrainingBaselineRunnerReport:
    manifest = report.manifest.model_copy(
        update={
            "source_training_package_manifest_id": source_training_package_manifest_id,
            "source_backend_preflight_status": source_backend_preflight_status,
        }
    )
    return report.model_copy(update={"manifest": manifest})


def _mark_dry_run(report: OledCuratedTrainingBaselineRunnerReport) -> OledCuratedTrainingBaselineRunnerReport:
    refreshed_results = [
        result.model_copy(update={"reason_codes": sorted({*result.reason_codes, "dry_run_no_files_written"})})
        for result in report.manifest.run_results
    ]
    manifest = _manifest(
        policy=report.manifest.policy,
        run_results=refreshed_results,
        predictions=report.predictions,
        source_training_package_manifest_id=report.manifest.source_training_package_manifest_id,
        source_backend_preflight_status=report.manifest.source_backend_preflight_status,
        baseline_backend_run=bool(report.predictions),
        predictions_written=False,
        metrics_written=False,
    )
    return report.model_copy(update={"manifest": manifest})


def _write_baseline_run_files(
    report: OledCuratedTrainingBaselineRunnerReport,
    output_dir: Path,
) -> OledCuratedTrainingBaselineRunnerReport:
    refreshed_results: list[OledCuratedTrainingBaselineRunResult] = []
    for result in report.manifest.run_results:
        if result.status != OledCuratedTrainingBaselineRunStatus.COMPLETED:
            refreshed_results.append(result)
            continue
        run_predictions = [
            prediction
            for prediction in report.predictions
            if prediction.baseline_kind == result.baseline_kind
            and prediction.target_property_id == result.target_property_id
            and prediction.feature_view == result.feature_view
        ]
        run_metrics = [
            metric
            for metric in report.metrics
            if metric.baseline_kind == result.baseline_kind
            and metric.target_property_id == result.target_property_id
            and metric.feature_view == result.feature_view
        ]
        prediction_path = None
        prediction_sha = None
        if report.manifest.policy.write_predictions:
            prediction_filename = oled_training_baseline_predictions_filename(
                baseline_kind=result.baseline_kind,
                target_property_id=result.target_property_id,
                feature_view=result.feature_view,
            )
            prediction_sha = write_oled_training_baseline_predictions_jsonl(run_predictions, output_dir / prediction_filename)
            prediction_path = prediction_filename
        metrics_path = None
        metrics_sha = None
        if report.manifest.policy.write_metrics:
            metrics_filename = oled_training_baseline_metrics_filename(
                baseline_kind=result.baseline_kind,
                target_property_id=result.target_property_id,
                feature_view=result.feature_view,
            )
            metrics_sha = write_oled_training_baseline_metrics_json(run_metrics, output_dir / metrics_filename)
            metrics_path = metrics_filename
        refreshed_results.append(
            result.model_copy(
                update={
                    "prediction_jsonl_path": prediction_path,
                    "prediction_sha256": prediction_sha,
                    "metrics_json_path": metrics_path,
                    "metrics_sha256": metrics_sha,
                }
            )
        )
    manifest = _manifest(
        policy=report.manifest.policy,
        run_results=refreshed_results,
        predictions=report.predictions,
        source_training_package_manifest_id=report.manifest.source_training_package_manifest_id,
        source_backend_preflight_status=report.manifest.source_backend_preflight_status,
        output_directory=redact_oled_mineru_acceptance_path(output_dir),
        baseline_backend_run=bool(report.predictions),
        predictions_written=report.manifest.policy.write_predictions and bool(report.predictions),
        metrics_written=report.manifest.policy.write_metrics and bool(report.metrics),
    )
    return report.model_copy(update={"manifest": manifest})


def _filter_rows(
    rows: list[OledCuratedTrainingPackageRow],
    policy: OledCuratedTrainingBaselineRunnerPolicy,
) -> list[OledCuratedTrainingPackageRow]:
    targets = _target_property_ids(policy)
    views = _feature_views(policy)
    splits = _splits(policy)
    return sorted(
        [
            row
            for row in rows
            if row.target_property_id in targets
            and (not views or row.feature_view in views)
            and row.split in splits
        ],
        key=lambda item: item.training_row_id,
    )


def _selected_targets(rows: list[OledCuratedTrainingPackageRow], policy: OledCuratedTrainingBaselineRunnerPolicy) -> list[str]:
    available = {row.target_property_id for row in rows}
    return sorted(target for target in _target_property_ids(policy) if target in available)


def _selected_feature_views(rows: list[OledCuratedTrainingPackageRow], policy: OledCuratedTrainingBaselineRunnerPolicy) -> list[str]:
    available = {row.feature_view for row in rows}
    requested = _feature_views(policy)
    return sorted((requested or available) & available)


def _rows_for_target_view(
    rows: list[OledCuratedTrainingPackageRow],
    target_property_id: str,
    feature_view: str,
) -> list[OledCuratedTrainingPackageRow]:
    return sorted(
        [row for row in rows if row.target_property_id == target_property_id and row.feature_view == feature_view],
        key=lambda item: item.training_row_id,
    )


def _numeric_target(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    split: str | None = None,
    training_row_id: str | None = None,
) -> OledCuratedTrainingBaselineRunnerFinding:
    return OledCuratedTrainingBaselineRunnerFinding(
        code=code,
        severity=severity,
        message=message,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        split=split,
        training_row_id=training_row_id,
    )


def _baseline_kinds(policy: OledCuratedTrainingBaselineRunnerPolicy) -> list[str]:
    return sorted({str(item).strip() for item in policy.baseline_kinds if str(item).strip()})


def _target_property_ids(policy: OledCuratedTrainingBaselineRunnerPolicy) -> set[str]:
    return {str(item).strip() for item in policy.target_property_ids if str(item).strip()}


def _feature_views(policy: OledCuratedTrainingBaselineRunnerPolicy) -> set[str]:
    return {str(item).strip() for item in policy.feature_views if str(item).strip()}


def _splits(policy: OledCuratedTrainingBaselineRunnerPolicy) -> set[str]:
    return {str(item).strip() for item in policy.splits if str(item).strip()}


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _safe_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_") or "unknown"


def _dedup_findings(
    findings: list[OledCuratedTrainingBaselineRunnerFinding],
) -> list[OledCuratedTrainingBaselineRunnerFinding]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[OledCuratedTrainingBaselineRunnerFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.baseline_kind or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.training_row_id or "",
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
            item.baseline_kind or "",
            item.target_property_id or "",
            item.feature_view or "",
            item.training_row_id or "",
        ),
    )


def _safety_metadata(
    *,
    baseline_backend_run: bool,
    models_fitted: bool,
    predictions_written: bool,
    metrics_written: bool,
) -> dict[str, Any]:
    return {
        "baseline_runner": True,
        "baseline_backend_run": baseline_backend_run,
        "models_fitted": models_fitted,
        "predictions_written": predictions_written,
        "metrics_written": metrics_written,
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
            "feature_dict",
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
    "OledCuratedTrainingBaselineKind",
    "OledCuratedTrainingBaselineRunnerPolicy",
    "OledCuratedTrainingBaselineRunStatus",
    "OledCuratedTrainingBaselinePrediction",
    "OledCuratedTrainingBaselineMetrics",
    "OledCuratedTrainingBaselineRunResult",
    "OledCuratedTrainingBaselineRunnerFinding",
    "OledCuratedTrainingBaselineRunnerManifest",
    "OledCuratedTrainingBaselineRunnerReport",
    "load_oled_training_package_backend_preflight_report_json",
    "select_oled_training_baselines_for_run",
    "run_oled_mean_baseline_on_training_rows",
    "run_oled_sklearn_baseline_on_training_rows",
    "write_oled_training_baseline_predictions_jsonl",
    "write_oled_training_baseline_metrics_json",
    "write_oled_training_baseline_manifest_json",
    "oled_training_baseline_predictions_filename",
    "oled_training_baseline_metrics_filename",
    "run_oled_curated_training_baseline_runner_from_files",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
