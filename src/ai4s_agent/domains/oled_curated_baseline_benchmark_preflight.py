from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from ai4s_agent.domains.oled_curated_training_package_baseline_runner import (
    OledCuratedTrainingBaselineMetrics,
    OledCuratedTrainingBaselinePrediction,
    OledCuratedTrainingBaselineRunStatus,
    OledCuratedTrainingBaselineRunnerManifest,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledBaselineBenchmarkPreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledBaselineBenchmarkArtifactStatus(str, Enum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    FAILED = "failed"
    SKIPPED = "skipped"


class OledBaselineBenchmarkPreflightPolicy(BaseModel):
    require_manifest_sha256: bool = True
    require_prediction_sha256: bool = True
    require_metrics_sha256: bool = True
    require_completed_runs: bool = True
    allow_skipped_runs: bool = True
    fail_on_failed_runs: bool = True
    require_predictions: bool = True
    require_metrics: bool = True
    require_eval_split_metrics: bool = True
    require_train_split_predictions: bool = True
    require_no_benchmark_validated_claims: bool = True
    fail_on_metric_mismatch: bool = True
    metric_tolerance: float = 1e-6

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])


class OledBaselinePredictionCoverageSummary(BaseModel):
    baseline_kind: str
    target_property_id: str
    feature_view: str

    prediction_count: int = 0
    numeric_prediction_count: int = 0
    numeric_target_count: int = 0

    rows_by_split: dict[str, int] = Field(default_factory=dict)
    missing_prediction_count: int = 0
    missing_target_count: int = 0

    duplicate_prediction_id_count: int = 0
    evidence_ref_count: int = 0

    status: OledBaselineBenchmarkArtifactStatus
    reason_codes: list[str] = Field(default_factory=list)


class OledBaselineMetricsConsistencySummary(BaseModel):
    baseline_kind: str
    target_property_id: str
    feature_view: str
    split: str

    prediction_row_count: int = 0
    metric_row_count: int = 0

    recomputed_mae: float | None = None
    reported_mae: float | None = None
    recomputed_rmse: float | None = None
    reported_rmse: float | None = None
    recomputed_r2: float | None = None
    reported_r2: float | None = None
    recomputed_bias: float | None = None
    reported_bias: float | None = None
    recomputed_target_mean: float | None = None
    reported_target_mean: float | None = None
    recomputed_prediction_mean: float | None = None
    reported_prediction_mean: float | None = None

    status: OledBaselineBenchmarkArtifactStatus
    reason_codes: list[str] = Field(default_factory=list)


class OledBaselineBenchmarkPreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    split: str | None = None
    prediction_id: str | None = None
    metrics_path: str | None = None
    prediction_path: str | None = None


class OledBaselineBenchmarkPreflightReport(BaseModel):
    status: OledBaselineBenchmarkPreflightStatus

    input_prediction_count: int
    input_metric_count: int

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=list)

    coverage_summaries: list[OledBaselinePredictionCoverageSummary] = Field(default_factory=list)
    metrics_summaries: list[OledBaselineMetricsConsistencySummary] = Field(default_factory=list)

    rows_by_split: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    findings: list[OledBaselineBenchmarkPreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledBaselineBenchmarkPreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_training_baseline_runner_manifest_json(
    path: str | Path,
) -> OledCuratedTrainingBaselineRunnerManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_baseline_run_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledCuratedTrainingBaselineRunnerManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_baseline_run_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.metadata):
        raise ValueError("absolute_path_in_baseline_run_manifest_metadata")
    return manifest


def load_oled_training_baseline_predictions_jsonl(
    path: str | Path,
) -> list[OledCuratedTrainingBaselinePrediction]:
    prediction_path = Path(path)
    _reject_forbidden_input(prediction_path)
    if not prediction_path.exists():
        raise ValueError(f"missing_baseline_predictions_jsonl:{redact_oled_mineru_acceptance_path(prediction_path)}")
    predictions: list[OledCuratedTrainingBaselinePrediction] = []
    for line_number, raw_line in enumerate(prediction_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if _contains_feature_payload(payload):
                raise ValueError("raw_feature_payload_leaked")
            prediction = OledCuratedTrainingBaselinePrediction.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"invalid_baseline_predictions_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(prediction.metadata):
            raise ValueError(f"absolute_path_in_baseline_prediction_metadata:{prediction.prediction_id}")
        predictions.append(prediction)
    return sorted(predictions, key=lambda prediction: prediction.prediction_id)


def load_oled_training_baseline_metrics_json(
    path: str | Path,
) -> list[OledCuratedTrainingBaselineMetrics]:
    metrics_path = Path(path)
    _reject_forbidden_input(metrics_path)
    if not metrics_path.exists():
        raise ValueError(f"missing_baseline_metrics_json:{redact_oled_mineru_acceptance_path(metrics_path)}")
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("metrics payload must be a list")
        metrics = [OledCuratedTrainingBaselineMetrics.model_validate(item) for item in payload]
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid_baseline_metrics_json:{redact_oled_mineru_acceptance_path(metrics_path)}") from exc
    if any(_contains_absolute_path(metric.metadata) for metric in metrics):
        raise ValueError("absolute_path_in_baseline_metrics_metadata")
    return sorted(metrics, key=lambda metric: (metric.baseline_kind, metric.target_property_id, metric.feature_view, metric.split))


def load_oled_training_baseline_artifacts_from_manifest(
    *,
    manifest: OledCuratedTrainingBaselineRunnerManifest,
    base_dir: str | Path,
) -> tuple[list[OledCuratedTrainingBaselinePrediction], list[OledCuratedTrainingBaselineMetrics]]:
    predictions: list[OledCuratedTrainingBaselinePrediction] = []
    metrics: list[OledCuratedTrainingBaselineMetrics] = []
    for run_result in manifest.run_results:
        if _status_value(run_result.status) != OledCuratedTrainingBaselineRunStatus.COMPLETED.value:
            continue
        if run_result.prediction_jsonl_path:
            prediction_path = _resolve_manifest_path(run_result.prediction_jsonl_path, base_dir)
            if run_result.prediction_sha256 is not None and _sha256_file(prediction_path) != run_result.prediction_sha256:
                raise ValueError(f"baseline_predictions_sha256_mismatch:{redact_oled_mineru_acceptance_path(prediction_path)}")
            predictions.extend(load_oled_training_baseline_predictions_jsonl(prediction_path))
        if run_result.metrics_json_path:
            metrics_path = _resolve_manifest_path(run_result.metrics_json_path, base_dir)
            if run_result.metrics_sha256 is not None and _sha256_file(metrics_path) != run_result.metrics_sha256:
                raise ValueError(f"baseline_metrics_sha256_mismatch:{redact_oled_mineru_acceptance_path(metrics_path)}")
            metrics.extend(load_oled_training_baseline_metrics_json(metrics_path))
    return (
        sorted(predictions, key=lambda prediction: prediction.prediction_id),
        sorted(metrics, key=lambda metric: (metric.baseline_kind, metric.target_property_id, metric.feature_view, metric.split)),
    )


def run_oled_baseline_benchmark_preflight(
    *,
    manifest: OledCuratedTrainingBaselineRunnerManifest,
    predictions: Iterable[OledCuratedTrainingBaselinePrediction],
    metrics: Iterable[OledCuratedTrainingBaselineMetrics],
    policy: OledBaselineBenchmarkPreflightPolicy | None = None,
) -> OledBaselineBenchmarkPreflightReport:
    preflight_policy = policy or OledBaselineBenchmarkPreflightPolicy()
    input_predictions = list(predictions)
    input_metrics = list(metrics)
    selected_predictions = _filter_predictions(input_predictions, preflight_policy)
    selected_metrics = _filter_metrics(input_metrics, preflight_policy)
    findings: list[OledBaselineBenchmarkPreflightFinding] = []
    findings.extend(_manifest_findings(manifest, preflight_policy))
    findings.extend(_prediction_findings(selected_predictions, preflight_policy))
    coverage_summaries, coverage_findings = _coverage_summaries(selected_predictions, preflight_policy)
    findings.extend(coverage_findings)
    metrics_summaries, metric_findings = _metrics_summaries(selected_predictions, selected_metrics, preflight_policy)
    findings.extend(metric_findings)
    findings = _dedup_findings(findings)
    status = _report_status(findings)
    status_counts = Counter(_status_value(summary.status) for summary in coverage_summaries)
    status_counts.update(_status_value(summary.status) for summary in metrics_summaries)
    return OledBaselineBenchmarkPreflightReport(
        status=status,
        input_prediction_count=len(input_predictions),
        input_metric_count=len(input_metrics),
        baseline_kinds=sorted({prediction.baseline_kind for prediction in selected_predictions} | {metric.baseline_kind for metric in selected_metrics}),
        target_property_ids=sorted({prediction.target_property_id for prediction in selected_predictions} | {metric.target_property_id for metric in selected_metrics}),
        feature_views=sorted({prediction.feature_view for prediction in selected_predictions} | {metric.feature_view for metric in selected_metrics}),
        splits=sorted({prediction.split for prediction in selected_predictions} | {metric.split for metric in selected_metrics}),
        coverage_summaries=coverage_summaries,
        metrics_summaries=metrics_summaries,
        rows_by_split=dict(sorted(Counter(prediction.split for prediction in selected_predictions).items())),
        status_counts=dict(sorted(status_counts.items())),
        finding_code_counts=dict(sorted(Counter(finding.code for finding in findings).items())),
        findings=findings,
        metadata=_safety_metadata(),
    )


def run_oled_baseline_benchmark_preflight_from_files(
    *,
    baseline_run_manifest_path: str | Path,
    baseline_run_base_dir: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledBaselineBenchmarkPreflightPolicy | None = None,
) -> OledBaselineBenchmarkPreflightReport:
    manifest = load_oled_training_baseline_runner_manifest_json(baseline_run_manifest_path)
    base_dir = Path(baseline_run_base_dir) if baseline_run_base_dir is not None else Path(baseline_run_manifest_path).parent
    predictions, metrics = load_oled_training_baseline_artifacts_from_manifest(manifest=manifest, base_dir=base_dir)
    report = run_oled_baseline_benchmark_preflight(
        manifest=manifest,
        predictions=predictions,
        metrics=metrics,
        policy=policy,
    )
    if output_report_path is not None:
        write_oled_baseline_benchmark_preflight_report_json(report, output_report_path)
    return report


def write_oled_baseline_benchmark_preflight_report_json(
    report: OledBaselineBenchmarkPreflightReport,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED baseline benchmark-readiness preflight.")
    parser.add_argument("--baseline-run-manifest", required=True, help="Path to baseline run manifest JSON.")
    parser.add_argument("--baseline-run-base-dir", help="Base directory for baseline run artifacts.")
    parser.add_argument("--output-report", help="Optional path for benchmark preflight report JSON.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--split", action="append", default=[], help="Split name; repeat or comma-separate.")
    args = parser.parse_args(argv)
    try:
        policy = OledBaselineBenchmarkPreflightPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            splits=_split_cli_values(args.split) or ["train", "validation", "test"],
        )
        report = run_oled_baseline_benchmark_preflight_from_files(
            baseline_run_manifest_path=args.baseline_run_manifest,
            baseline_run_base_dir=args.baseline_run_base_dir,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": _status_value(report.status),
            "input_prediction_count": report.input_prediction_count,
            "input_metric_count": report.input_metric_count,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _manifest_findings(
    manifest: OledCuratedTrainingBaselineRunnerManifest,
    policy: OledBaselineBenchmarkPreflightPolicy,
) -> list[OledBaselineBenchmarkPreflightFinding]:
    findings: list[OledBaselineBenchmarkPreflightFinding] = []
    if policy.require_no_benchmark_validated_claims and bool(manifest.metadata.get("benchmark_validated")):
        findings.append(_finding("benchmark_validated_source_claim", "error", "baseline run manifest claims benchmark validation"))
    completed = [result for result in manifest.run_results if _status_value(result.status) == OledCuratedTrainingBaselineRunStatus.COMPLETED.value]
    if policy.require_completed_runs and not completed:
        findings.append(_finding("missing_completed_baseline_run", "error", "baseline run manifest has no completed runs"))
    for result in manifest.run_results:
        status = _status_value(result.status)
        if status == OledCuratedTrainingBaselineRunStatus.FAILED.value and policy.fail_on_failed_runs:
            findings.append(_finding("failed_baseline_run", "error", "baseline run manifest contains failed run", result=result))
        if status == OledCuratedTrainingBaselineRunStatus.SKIPPED.value and not policy.allow_skipped_runs:
            findings.append(_finding("skipped_baseline_run", "error", "baseline run manifest contains skipped run", result=result))
        if status == OledCuratedTrainingBaselineRunStatus.COMPLETED.value:
            if policy.require_predictions and not result.prediction_jsonl_path:
                findings.append(_finding("missing_prediction_artifact", "error", "completed run is missing prediction artifact", result=result))
            if policy.require_metrics and not result.metrics_json_path:
                findings.append(_finding("missing_metrics_artifact", "error", "completed run is missing metrics artifact", result=result))
            if policy.require_prediction_sha256 and result.prediction_jsonl_path and not result.prediction_sha256:
                findings.append(_finding("missing_prediction_artifact", "error", "completed run prediction artifact lacks sha256", result=result))
            if policy.require_metrics_sha256 and result.metrics_json_path and not result.metrics_sha256:
                findings.append(_finding("missing_metrics_artifact", "error", "completed run metrics artifact lacks sha256", result=result))
    return findings


def _prediction_findings(
    predictions: list[OledCuratedTrainingBaselinePrediction],
    policy: OledBaselineBenchmarkPreflightPolicy,
) -> list[OledBaselineBenchmarkPreflightFinding]:
    findings: list[OledBaselineBenchmarkPreflightFinding] = []
    counts = Counter(prediction.prediction_id for prediction in predictions)
    for prediction in predictions:
        if counts[prediction.prediction_id] > 1:
            findings.append(_prediction_finding("duplicate_prediction_id", "error", "prediction id is duplicated", prediction))
        if _numeric(prediction.y_pred) is None:
            findings.append(_prediction_finding("missing_prediction_value", "error", "prediction value is missing or nonnumeric", prediction))
        if _numeric(prediction.y_true) is None:
            findings.append(_prediction_finding("missing_target_value", "error", "target value is missing or nonnumeric", prediction))
        if not prediction.evidence_refs:
            findings.append(_prediction_finding("missing_evidence_refs", "error", "prediction lacks evidence refs", prediction))
        if policy.require_no_benchmark_validated_claims and bool(prediction.metadata.get("benchmark_validated")):
            findings.append(_prediction_finding("benchmark_validated_source_claim", "error", "prediction metadata claims benchmark validation", prediction))
        if _contains_feature_payload(prediction.model_dump(mode="json")):
            findings.append(_prediction_finding("raw_feature_payload_leaked", "error", "prediction payload includes feature dictionary", prediction))
    return findings


def _coverage_summaries(
    predictions: list[OledCuratedTrainingBaselinePrediction],
    policy: OledBaselineBenchmarkPreflightPolicy,
) -> tuple[list[OledBaselinePredictionCoverageSummary], list[OledBaselineBenchmarkPreflightFinding]]:
    grouped: dict[tuple[str, str, str], list[OledCuratedTrainingBaselinePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction.baseline_kind, prediction.target_property_id, prediction.feature_view)].append(prediction)
    summaries: list[OledBaselinePredictionCoverageSummary] = []
    findings: list[OledBaselineBenchmarkPreflightFinding] = []
    for (baseline_kind, target_property_id, feature_view), group in sorted(grouped.items()):
        rows_by_split = dict(sorted(Counter(prediction.split for prediction in group).items()))
        duplicate_count = sum(count - 1 for count in Counter(prediction.prediction_id for prediction in group).values() if count > 1)
        missing_prediction_count = sum(1 for prediction in group if _numeric(prediction.y_pred) is None)
        missing_target_count = sum(1 for prediction in group if _numeric(prediction.y_true) is None)
        evidence_ref_count = sum(len(prediction.evidence_refs) for prediction in group)
        reason_codes: set[str] = {"prediction_coverage_ready"}
        status = OledBaselineBenchmarkArtifactStatus.READY
        if policy.require_train_split_predictions and rows_by_split.get("train", 0) == 0:
            reason_codes.add("missing_train_predictions")
            findings.append(
                _summary_finding("missing_train_predictions", "error", "prediction artifact lacks train split predictions", baseline_kind, target_property_id, feature_view)
            )
            status = OledBaselineBenchmarkArtifactStatus.FAILED
        if duplicate_count:
            reason_codes.add("duplicate_prediction_id")
            status = OledBaselineBenchmarkArtifactStatus.FAILED
        if missing_prediction_count:
            reason_codes.add("missing_prediction_value")
            status = OledBaselineBenchmarkArtifactStatus.FAILED
        if missing_target_count:
            reason_codes.add("missing_target_value")
            status = OledBaselineBenchmarkArtifactStatus.FAILED
        summaries.append(
            OledBaselinePredictionCoverageSummary(
                baseline_kind=baseline_kind,
                target_property_id=target_property_id,
                feature_view=feature_view,
                prediction_count=len(group),
                numeric_prediction_count=sum(1 for prediction in group if _numeric(prediction.y_pred) is not None),
                numeric_target_count=sum(1 for prediction in group if _numeric(prediction.y_true) is not None),
                rows_by_split=rows_by_split,
                missing_prediction_count=missing_prediction_count,
                missing_target_count=missing_target_count,
                duplicate_prediction_id_count=duplicate_count,
                evidence_ref_count=evidence_ref_count,
                status=status,
                reason_codes=sorted(reason_codes),
            )
        )
    return summaries, findings


def _metrics_summaries(
    predictions: list[OledCuratedTrainingBaselinePrediction],
    metrics: list[OledCuratedTrainingBaselineMetrics],
    policy: OledBaselineBenchmarkPreflightPolicy,
) -> tuple[list[OledBaselineMetricsConsistencySummary], list[OledBaselineBenchmarkPreflightFinding]]:
    prediction_groups: dict[tuple[str, str, str, str], list[OledCuratedTrainingBaselinePrediction]] = defaultdict(list)
    for prediction in predictions:
        prediction_groups[(prediction.baseline_kind, prediction.target_property_id, prediction.feature_view, prediction.split)].append(prediction)
    metric_by_key = {(metric.baseline_kind, metric.target_property_id, metric.feature_view, metric.split): metric for metric in metrics}
    summaries: list[OledBaselineMetricsConsistencySummary] = []
    findings: list[OledBaselineBenchmarkPreflightFinding] = []
    for key, group in sorted(prediction_groups.items()):
        baseline_kind, target_property_id, feature_view, split = key
        numeric_predictions = [prediction for prediction in group if _numeric(prediction.y_true) is not None and _numeric(prediction.y_pred) is not None]
        if not numeric_predictions:
            continue
        recomputed = _regression_metrics(
            [float(prediction.y_true) for prediction in numeric_predictions if prediction.y_true is not None],
            [float(prediction.y_pred) for prediction in numeric_predictions if prediction.y_pred is not None],
        )
        reported = metric_by_key.get(key)
        reason_codes: set[str] = {"metrics_consistent"}
        status = OledBaselineBenchmarkArtifactStatus.READY
        if reported is None:
            reason_codes.add("metric_split_missing")
            findings.append(_summary_finding("metric_split_missing", "error", "metrics artifact lacks split present in predictions", baseline_kind, target_property_id, feature_view, split=split))
            if split in {"validation", "test"} and policy.require_eval_split_metrics:
                findings.append(_summary_finding("missing_eval_metrics", "error", "metrics artifact lacks evaluation split metrics", baseline_kind, target_property_id, feature_view, split=split))
                reason_codes.add("missing_eval_metrics")
            status = OledBaselineBenchmarkArtifactStatus.FAILED
            summaries.append(_metric_summary(baseline_kind, target_property_id, feature_view, split, numeric_predictions, None, recomputed, status, reason_codes))
            continue
        metric_findings = _metric_value_findings(recomputed, reported, policy)
        findings.extend(metric_findings)
        if metric_findings:
            reason_codes.update(finding.code for finding in metric_findings)
            status = OledBaselineBenchmarkArtifactStatus.FAILED
        summaries.append(_metric_summary(baseline_kind, target_property_id, feature_view, split, numeric_predictions, reported, recomputed, status, reason_codes))
    return summaries, findings


def _metric_summary(
    baseline_kind: str,
    target_property_id: str,
    feature_view: str,
    split: str,
    predictions: list[OledCuratedTrainingBaselinePrediction],
    reported: OledCuratedTrainingBaselineMetrics | None,
    recomputed: dict[str, float],
    status: OledBaselineBenchmarkArtifactStatus,
    reason_codes: set[str],
) -> OledBaselineMetricsConsistencySummary:
    return OledBaselineMetricsConsistencySummary(
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        split=split,
        prediction_row_count=len(predictions),
        metric_row_count=reported.row_count if reported is not None else 0,
        recomputed_mae=recomputed.get("mae"),
        reported_mae=reported.mae if reported is not None else None,
        recomputed_rmse=recomputed.get("rmse"),
        reported_rmse=reported.rmse if reported is not None else None,
        recomputed_r2=recomputed.get("r2"),
        reported_r2=reported.r2 if reported is not None else None,
        recomputed_bias=recomputed.get("bias"),
        reported_bias=reported.bias if reported is not None else None,
        recomputed_target_mean=recomputed.get("target_mean"),
        reported_target_mean=reported.target_mean if reported is not None else None,
        recomputed_prediction_mean=recomputed.get("prediction_mean"),
        reported_prediction_mean=reported.prediction_mean if reported is not None else None,
        status=status,
        reason_codes=sorted(reason_codes),
    )


def _metric_value_findings(
    recomputed: dict[str, float],
    reported: OledCuratedTrainingBaselineMetrics,
    policy: OledBaselineBenchmarkPreflightPolicy,
) -> list[OledBaselineBenchmarkPreflightFinding]:
    findings: list[OledBaselineBenchmarkPreflightFinding] = []
    if reported.row_count != int(recomputed["row_count"]):
        findings.append(
            _summary_finding("metric_row_count_mismatch", "error", "reported metric row count differs from prediction rows", reported.baseline_kind, reported.target_property_id, reported.feature_view, split=reported.split)
        )
    for field in ("mae", "rmse", "r2", "bias", "target_mean", "prediction_mean"):
        reported_value = getattr(reported, field)
        recomputed_value = recomputed[field]
        if reported_value is None or not math.isfinite(float(reported_value)):
            findings.append(
                _summary_finding("nonfinite_metric", "error", "reported metric is missing or nonfinite", reported.baseline_kind, reported.target_property_id, reported.feature_view, split=reported.split)
            )
            continue
        if abs(float(reported_value) - recomputed_value) > policy.metric_tolerance:
            findings.append(
                _summary_finding(
                    "metric_value_mismatch",
                    "error" if policy.fail_on_metric_mismatch else "warning",
                    f"reported metric {field} differs from recomputed value",
                    reported.baseline_kind,
                    reported.target_property_id,
                    reported.feature_view,
                    split=reported.split,
                )
            )
    if bool(reported.metadata.get("benchmark_validated")) and policy.require_no_benchmark_validated_claims:
        findings.append(
            _summary_finding("benchmark_validated_source_claim", "error", "metrics metadata claims benchmark validation", reported.baseline_kind, reported.target_property_id, reported.feature_view, split=reported.split)
        )
    return findings


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
        "row_count": float(n),
        "bias": round(sum(errors) / n, 6),
        "mae": round(sum(absolute_errors) / n, 6),
        "prediction_mean": round(prediction_mean, 6),
        "r2": round(r2, 6),
        "rmse": round(math.sqrt(sse / n), 6),
        "target_mean": round(target_mean, 6),
    }


def _filter_predictions(
    predictions: list[OledCuratedTrainingBaselinePrediction],
    policy: OledBaselineBenchmarkPreflightPolicy,
) -> list[OledCuratedTrainingBaselinePrediction]:
    baselines = _baseline_kinds(policy)
    targets = _target_property_ids(policy)
    views = _feature_views(policy)
    splits = _splits(policy)
    return sorted(
        [
            prediction
            for prediction in predictions
            if (not baselines or prediction.baseline_kind in baselines)
            and prediction.target_property_id in targets
            and (not views or prediction.feature_view in views)
            and prediction.split in splits
        ],
        key=lambda prediction: prediction.prediction_id,
    )


def _filter_metrics(
    metrics: list[OledCuratedTrainingBaselineMetrics],
    policy: OledBaselineBenchmarkPreflightPolicy,
) -> list[OledCuratedTrainingBaselineMetrics]:
    baselines = _baseline_kinds(policy)
    targets = _target_property_ids(policy)
    views = _feature_views(policy)
    splits = _splits(policy)
    return sorted(
        [
            metric
            for metric in metrics
            if (not baselines or metric.baseline_kind in baselines)
            and metric.target_property_id in targets
            and (not views or metric.feature_view in views)
            and metric.split in splits
        ],
        key=lambda metric: (metric.baseline_kind, metric.target_property_id, metric.feature_view, metric.split),
    )


def _numeric(value: Any) -> float | None:
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
    result: Any | None = None,
) -> OledBaselineBenchmarkPreflightFinding:
    return OledBaselineBenchmarkPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        baseline_kind=getattr(result, "baseline_kind", None),
        target_property_id=getattr(result, "target_property_id", None),
        feature_view=getattr(result, "feature_view", None),
    )


def _prediction_finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    prediction: OledCuratedTrainingBaselinePrediction,
) -> OledBaselineBenchmarkPreflightFinding:
    return OledBaselineBenchmarkPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        baseline_kind=prediction.baseline_kind,
        target_property_id=prediction.target_property_id,
        feature_view=prediction.feature_view,
        split=prediction.split,
        prediction_id=prediction.prediction_id,
    )


def _summary_finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    baseline_kind: str,
    target_property_id: str,
    feature_view: str,
    *,
    split: str | None = None,
) -> OledBaselineBenchmarkPreflightFinding:
    return OledBaselineBenchmarkPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        split=split,
    )


def _report_status(
    findings: list[OledBaselineBenchmarkPreflightFinding],
) -> OledBaselineBenchmarkPreflightStatus:
    if any(finding.severity == "error" for finding in findings):
        return OledBaselineBenchmarkPreflightStatus.FAILED
    if any(finding.severity == "warning" for finding in findings):
        return OledBaselineBenchmarkPreflightStatus.PASSED_WITH_WARNINGS
    return OledBaselineBenchmarkPreflightStatus.PASSED


def _baseline_kinds(policy: OledBaselineBenchmarkPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.baseline_kinds if str(item).strip()}


def _target_property_ids(policy: OledBaselineBenchmarkPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.target_property_ids if str(item).strip()}


def _feature_views(policy: OledBaselineBenchmarkPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.feature_views if str(item).strip()}


def _splits(policy: OledBaselineBenchmarkPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.splits if str(item).strip()}


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
    findings: list[OledBaselineBenchmarkPreflightFinding],
) -> list[OledBaselineBenchmarkPreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    deduped: list[OledBaselineBenchmarkPreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.baseline_kind or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.split or "",
            finding.prediction_id or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return sorted(
        deduped,
        key=lambda finding: (
            finding.severity,
            finding.code,
            finding.baseline_kind or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.split or "",
            finding.prediction_id or "",
        ),
    )


def _safety_metadata() -> dict[str, Any]:
    return {
        "benchmark_preflight_only": True,
        "benchmark_results_written": False,
        "benchmark_registered": False,
        "benchmark_validated": False,
        "baseline_backend_run": False,
        "models_fitted": False,
        "predictions_written": False,
        "metrics_written": False,
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


def _contains_feature_payload(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() == "features" or _contains_feature_payload(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_feature_payload(item) for item in value)
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
        normalized == token
        or normalized.endswith(f"_{token}")
        or normalized.startswith(f"{token}_payload")
        for token in (
            "raw_text",
            "full_text",
            "parsed_json",
            "table_body",
            "html_table",
            "markdown_table",
            "features",
            "prediction_payload",
            "metrics_payload",
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
    "OledBaselineBenchmarkPreflightStatus",
    "OledBaselineBenchmarkArtifactStatus",
    "OledBaselineBenchmarkPreflightPolicy",
    "OledBaselinePredictionCoverageSummary",
    "OledBaselineMetricsConsistencySummary",
    "OledBaselineBenchmarkPreflightFinding",
    "OledBaselineBenchmarkPreflightReport",
    "load_oled_training_baseline_runner_manifest_json",
    "load_oled_training_baseline_predictions_jsonl",
    "load_oled_training_baseline_metrics_json",
    "load_oled_training_baseline_artifacts_from_manifest",
    "run_oled_baseline_benchmark_preflight",
    "run_oled_baseline_benchmark_preflight_from_files",
    "write_oled_baseline_benchmark_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
