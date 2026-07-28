from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from ai4s_agent.domains.oled_curated_baseline_benchmark_preflight import (
    OledBaselineBenchmarkPreflightReport,
    OledBaselineBenchmarkPreflightStatus,
    load_oled_training_baseline_artifacts_from_manifest,
    load_oled_training_baseline_runner_manifest_json,
)
from ai4s_agent.domains.oled_curated_training_package_baseline_runner import (
    OledCuratedTrainingBaselineMetrics,
    OledCuratedTrainingBaselinePrediction,
    OledCuratedTrainingBaselineRunStatus,
    OledCuratedTrainingBaselineRunnerManifest,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledBaselineBenchmarkReportWriterPolicy(BaseModel):
    require_confirmation: bool = True
    require_preflight_valid: bool = True
    allow_preflight_warnings: bool = True
    require_completed_runs: bool = True
    require_metrics: bool = True
    require_predictions: bool = True
    require_no_benchmark_validated_claims: bool = True

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])

    write_json_report: bool = True
    write_markdown_report: bool = True
    benchmark_validated: bool = False
    register_benchmark: bool = False


class OledBaselineBenchmarkReportWriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class OledBaselineBenchmarkMetricCard(BaseModel):
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

    metric_status: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class OledBaselineBenchmarkRunCard(BaseModel):
    baseline_kind: str
    target_property_id: str
    feature_view: str

    run_status: str
    prediction_count: int = 0

    train_row_count: int = 0
    validation_row_count: int = 0
    test_row_count: int = 0

    metric_splits: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    prediction_artifact_sha256: str | None = None
    metrics_artifact_sha256: str | None = None

    metrics: list[OledBaselineBenchmarkMetricCard] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledBaselineBenchmarkCandidateReport(BaseModel):
    report_id: str

    source_baseline_run_manifest_id: str | None = None
    source_benchmark_preflight_status: str | None = None

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=list)

    input_prediction_count: int = 0
    input_metric_count: int = 0

    run_cards: list[OledBaselineBenchmarkRunCard] = Field(default_factory=list)

    coverage_status_counts: dict[str, int] = Field(default_factory=dict)
    metric_status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    caveats: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class OledBaselineBenchmarkReportFileResult(BaseModel):
    artifact_kind: Literal["benchmark_report_json", "benchmark_report_markdown", "manifest"]

    status: OledBaselineBenchmarkReportWriteStatus
    output_path: str | None = None
    output_sha256: str | None = None

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledBaselineBenchmarkReportWriterFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    split: str | None = None
    output_path: str | None = None


class OledBaselineBenchmarkReportWriterManifest(BaseModel):
    manifest_id: str

    source_baseline_run_manifest_id: str | None = None
    source_benchmark_preflight_status: str | None = None

    output_directory: str | None = None
    output_file_count: int = 0

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    file_results: list[OledBaselineBenchmarkReportFileResult] = Field(default_factory=list)

    policy: OledBaselineBenchmarkReportWriterPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(result.status == OledBaselineBenchmarkReportWriteStatus.REJECTED for result in self.file_results)


class OledBaselineBenchmarkReportWriterReport(BaseModel):
    manifest: OledBaselineBenchmarkReportWriterManifest
    benchmark_report: OledBaselineBenchmarkCandidateReport | None = None
    findings: list[OledBaselineBenchmarkReportWriterFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.error_codes and self.manifest.is_valid

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_baseline_benchmark_preflight_report_json(
    path: str | Path,
) -> OledBaselineBenchmarkPreflightReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_benchmark_preflight_report:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = OledBaselineBenchmarkPreflightReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_benchmark_preflight_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    if _contains_absolute_path(report.metadata):
        raise ValueError("absolute_path_in_benchmark_preflight_report_metadata")
    return report


def build_oled_baseline_benchmark_candidate_report(
    *,
    baseline_run_manifest: OledCuratedTrainingBaselineRunnerManifest,
    predictions: Iterable[OledCuratedTrainingBaselinePrediction],
    metrics: Iterable[OledCuratedTrainingBaselineMetrics],
    benchmark_preflight_report: OledBaselineBenchmarkPreflightReport,
    policy: OledBaselineBenchmarkReportWriterPolicy | None = None,
) -> tuple[OledBaselineBenchmarkCandidateReport | None, list[OledBaselineBenchmarkReportWriterFinding]]:
    writer_policy = policy or OledBaselineBenchmarkReportWriterPolicy()
    input_predictions = list(predictions)
    input_metrics = list(metrics)
    selected_predictions = _filter_predictions(input_predictions, writer_policy)
    selected_metrics = _filter_metrics(input_metrics, writer_policy)
    selected_results = _filter_run_results(baseline_run_manifest.run_results, writer_policy)

    findings = _gate_findings(
        manifest=baseline_run_manifest,
        predictions=selected_predictions,
        metrics=selected_metrics,
        benchmark_preflight_report=benchmark_preflight_report,
        policy=writer_policy,
    )
    if any(finding.severity == "error" for finding in findings):
        return None, _dedup_findings(findings)

    metric_cards_by_key = _metric_cards_by_key(selected_metrics, benchmark_preflight_report)
    run_cards: list[OledBaselineBenchmarkRunCard] = []
    prediction_counts = Counter((item.baseline_kind, item.target_property_id, item.feature_view) for item in selected_predictions)
    for result in selected_results:
        key = (result.baseline_kind, result.target_property_id, result.feature_view)
        metric_cards = metric_cards_by_key.get(key, [])
        run_cards.append(
            OledBaselineBenchmarkRunCard(
                baseline_kind=result.baseline_kind,
                target_property_id=result.target_property_id,
                feature_view=result.feature_view,
                run_status=_status_value(result.status),
                prediction_count=prediction_counts.get(key, result.prediction_count),
                train_row_count=result.train_row_count,
                validation_row_count=result.validation_row_count,
                test_row_count=result.test_row_count,
                metric_splits=sorted({metric.split for metric in metric_cards} or set(result.metric_splits)),
                reason_codes=sorted(set(result.reason_codes) | {"selected_for_report"}),
                prediction_artifact_sha256=result.prediction_sha256,
                metrics_artifact_sha256=result.metrics_sha256,
                metrics=metric_cards,
                metadata={"benchmark_candidate_run_card": True, "benchmark_validated": False},
            )
        )

    report = OledBaselineBenchmarkCandidateReport(
        report_id=_report_id(baseline_run_manifest.manifest_id, benchmark_preflight_report.status),
        source_baseline_run_manifest_id=baseline_run_manifest.manifest_id,
        source_benchmark_preflight_status=_status_value(benchmark_preflight_report.status),
        baseline_kinds=sorted({card.baseline_kind for card in run_cards}),
        target_property_ids=sorted({card.target_property_id for card in run_cards}),
        feature_views=sorted({card.feature_view for card in run_cards}),
        splits=sorted({metric.split for card in run_cards for metric in card.metrics} | {split for card in run_cards for split in card.metric_splits}),
        input_prediction_count=len(input_predictions),
        input_metric_count=len(input_metrics),
        run_cards=run_cards,
        coverage_status_counts=dict(sorted(benchmark_preflight_report.status_counts.items())),
        metric_status_counts=dict(sorted(Counter(_status_value(summary.status) for summary in benchmark_preflight_report.metrics_summaries).items())),
        finding_code_counts=dict(sorted(benchmark_preflight_report.finding_code_counts.items())),
        caveats=["baseline_candidate_report_only", "not_benchmark_validated", "not_scientific_performance_claim"],
        metadata=_report_metadata(candidate_written=False),
    )
    return report, _dedup_findings(findings)


def select_oled_baseline_benchmark_report_for_write(
    *,
    baseline_run_manifest: OledCuratedTrainingBaselineRunnerManifest,
    predictions: Iterable[OledCuratedTrainingBaselinePrediction],
    metrics: Iterable[OledCuratedTrainingBaselineMetrics],
    benchmark_preflight_report: OledBaselineBenchmarkPreflightReport,
    policy: OledBaselineBenchmarkReportWriterPolicy | None = None,
    confirm_benchmark_report_write: bool = False,
) -> OledBaselineBenchmarkReportWriterReport:
    writer_policy = policy or OledBaselineBenchmarkReportWriterPolicy()
    if writer_policy.require_confirmation and not confirm_benchmark_report_write:
        raise ValueError("confirmation_required:benchmark_report_write")
    benchmark_report, findings = build_oled_baseline_benchmark_candidate_report(
        baseline_run_manifest=baseline_run_manifest,
        predictions=predictions,
        metrics=metrics,
        benchmark_preflight_report=benchmark_preflight_report,
        policy=writer_policy,
    )
    manifest = _manifest(
        policy=writer_policy,
        benchmark_report=benchmark_report,
        findings=findings,
        baseline_run_manifest=baseline_run_manifest,
        benchmark_preflight_report=benchmark_preflight_report,
        output_directory=None,
        candidate_written=False,
    )
    return OledBaselineBenchmarkReportWriterReport(manifest=manifest, benchmark_report=benchmark_report, findings=findings)


def write_oled_baseline_benchmark_report_json(
    report: OledBaselineBenchmarkCandidateReport,
    path: str | Path,
) -> str:
    payload = json.dumps(_sanitize_for_output(report.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n"
    return _write_bytes(path, payload.encode("utf-8"))


def write_oled_baseline_benchmark_report_markdown(
    report: OledBaselineBenchmarkCandidateReport,
    path: str | Path,
) -> str:
    lines = [
        "# OLED Baseline Benchmark Candidate Report",
        "",
        f"- Report id: `{report.report_id}`",
        f"- Source baseline run manifest: `{report.source_baseline_run_manifest_id or ''}`",
        f"- Benchmark preflight status: `{report.source_benchmark_preflight_status or ''}`",
        f"- Benchmark validated: `{str(bool(report.metadata.get('benchmark_validated'))).lower()}`",
        f"- Benchmark registered: `{str(bool(report.metadata.get('benchmark_registered'))).lower()}`",
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- `{caveat}`" for caveat in sorted(report.caveats))
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Baseline | Target | Feature view | Status | Predictions | Metric splits |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for card in sorted(report.run_cards, key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view)):
        lines.append(
            "| "
            f"{_md(card.baseline_kind)} | {_md(card.target_property_id)} | {_md(card.feature_view)} | {_md(card.run_status)} | "
            f"{card.prediction_count} | {_md(', '.join(card.metric_splits))} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Baseline | Target | Feature view | Split | Rows | MAE | RMSE | R2 | Bias |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    metric_cards = [metric for card in report.run_cards for metric in card.metrics]
    for metric in sorted(metric_cards, key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view, item.split)):
        lines.append(
            "| "
            f"{_md(metric.baseline_kind)} | {_md(metric.target_property_id)} | {_md(metric.feature_view)} | {_md(metric.split)} | "
            f"{metric.row_count} | {_format_metric(metric.mae)} | {_format_metric(metric.rmse)} | {_format_metric(metric.r2)} | {_format_metric(metric.bias)} |"
        )
    lines.extend(
        [
            "",
            "This candidate report is not a benchmark registration record and does not validate scientific performance.",
            "",
        ]
    )
    return _write_bytes(path, ("\n".join(lines)).encode("utf-8"))


def write_oled_baseline_benchmark_report_manifest_json(
    manifest: OledBaselineBenchmarkReportWriterManifest,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def oled_baseline_benchmark_report_json_filename() -> str:
    return "oled_baseline_benchmark_candidate_report.json"


def oled_baseline_benchmark_report_markdown_filename() -> str:
    return "oled_baseline_benchmark_candidate_report.md"


def run_oled_baseline_benchmark_report_writer_from_files(
    *,
    baseline_run_manifest_path: str | Path,
    benchmark_preflight_report_path: str | Path,
    baseline_run_base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    policy: OledBaselineBenchmarkReportWriterPolicy | None = None,
    confirm_benchmark_report_write: bool = False,
    dry_run: bool = False,
) -> OledBaselineBenchmarkReportWriterReport:
    writer_policy = policy or OledBaselineBenchmarkReportWriterPolicy()
    if not output_dir and not output_manifest_path:
        raise ValueError("output_required:dir_or_manifest")
    if not dry_run and writer_policy.require_confirmation and not confirm_benchmark_report_write:
        raise ValueError("confirmation_required:benchmark_report_write")
    if not dry_run and output_dir is None:
        raise ValueError("output_dir_required:benchmark_report_write")
    baseline_manifest = load_oled_training_baseline_runner_manifest_json(baseline_run_manifest_path)
    base_dir = Path(baseline_run_base_dir) if baseline_run_base_dir is not None else Path(baseline_run_manifest_path).parent
    predictions, metrics = load_oled_training_baseline_artifacts_from_manifest(manifest=baseline_manifest, base_dir=base_dir)
    preflight_report = load_oled_baseline_benchmark_preflight_report_json(benchmark_preflight_report_path)
    selection_policy = writer_policy.model_copy(update={"require_confirmation": not dry_run and writer_policy.require_confirmation})
    report = select_oled_baseline_benchmark_report_for_write(
        baseline_run_manifest=baseline_manifest,
        predictions=predictions,
        metrics=metrics,
        benchmark_preflight_report=preflight_report,
        policy=selection_policy,
        confirm_benchmark_report_write=confirm_benchmark_report_write or dry_run,
    )
    if dry_run:
        report = _mark_dry_run(report)
        if output_manifest_path is not None:
            write_oled_baseline_benchmark_report_manifest_json(report.manifest, output_manifest_path)
        return report
    if not report.is_valid or report.benchmark_report is None:
        if output_manifest_path is not None:
            write_oled_baseline_benchmark_report_manifest_json(report.manifest, output_manifest_path)
        return report

    assert output_dir is not None
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report = _write_report_files(report, output_root)
    if output_manifest_path is not None:
        write_oled_baseline_benchmark_report_manifest_json(report.manifest, output_manifest_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write OLED baseline benchmark candidate report artifacts.")
    parser.add_argument("--baseline-run-manifest", required=True, help="Path to baseline run manifest JSON.")
    parser.add_argument("--benchmark-preflight-report", required=True, help="Path to benchmark preflight report JSON.")
    parser.add_argument("--baseline-run-base-dir", help="Base directory for baseline run artifacts.")
    parser.add_argument("--output-dir", help="Directory for benchmark report artifacts.")
    parser.add_argument("--output-manifest", help="Optional path for benchmark report writer manifest JSON.")
    parser.add_argument("--confirm-benchmark-report-write", action="store_true", help="Confirm benchmark candidate report writing.")
    parser.add_argument("--dry-run", action="store_true", help="Build report object without writing JSON/Markdown reports.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--json-only", action="store_true", help="Write only JSON report artifact.")
    parser.add_argument("--markdown-only", action="store_true", help="Write only Markdown report artifact.")
    args = parser.parse_args(argv)

    if not args.output_dir and not args.output_manifest:
        print("output_required:dir_or_manifest", file=sys.stderr)
        return 1
    if not args.dry_run and not args.confirm_benchmark_report_write:
        print("confirmation_required:benchmark_report_write", file=sys.stderr)
        return 1
    if args.json_only and args.markdown_only:
        print("output_format_conflict:json_only_and_markdown_only", file=sys.stderr)
        return 1
    try:
        policy = OledBaselineBenchmarkReportWriterPolicy(
            require_confirmation=not args.dry_run,
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            write_json_report=not args.markdown_only,
            write_markdown_report=not args.json_only,
        )
        report = run_oled_baseline_benchmark_report_writer_from_files(
            baseline_run_manifest_path=args.baseline_run_manifest,
            benchmark_preflight_report_path=args.benchmark_preflight_report,
            baseline_run_base_dir=args.baseline_run_base_dir,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            policy=policy,
            confirm_benchmark_report_write=args.confirm_benchmark_report_write,
            dry_run=args.dry_run,
        )
        summary = {
            "status": "valid" if report.is_valid else "invalid",
            "run_card_count": len(report.benchmark_report.run_cards) if report.benchmark_report is not None else 0,
            "output_file_count": report.manifest.output_file_count,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _gate_findings(
    *,
    manifest: OledCuratedTrainingBaselineRunnerManifest,
    predictions: list[OledCuratedTrainingBaselinePrediction],
    metrics: list[OledCuratedTrainingBaselineMetrics],
    benchmark_preflight_report: OledBaselineBenchmarkPreflightReport,
    policy: OledBaselineBenchmarkReportWriterPolicy,
) -> list[OledBaselineBenchmarkReportWriterFinding]:
    findings: list[OledBaselineBenchmarkReportWriterFinding] = []
    if policy.require_preflight_valid and benchmark_preflight_report.status == OledBaselineBenchmarkPreflightStatus.FAILED:
        findings.append(_finding("benchmark_preflight_failed", "error", "benchmark preflight failed"))
    if not policy.allow_preflight_warnings and benchmark_preflight_report.warning_codes:
        findings.append(_finding("benchmark_preflight_warnings_present", "error", "benchmark preflight has warnings"))
    if policy.require_completed_runs and not any(result.status == OledCuratedTrainingBaselineRunStatus.COMPLETED for result in manifest.run_results):
        findings.append(_finding("missing_completed_baseline_run", "error", "baseline manifest has no completed runs"))
    if policy.require_predictions and not predictions:
        findings.append(_finding("missing_prediction_artifact", "error", "no selected predictions available"))
    if policy.require_metrics and not metrics:
        findings.append(_finding("missing_metrics_artifact", "error", "no selected metrics available"))
    if policy.benchmark_validated or policy.register_benchmark:
        findings.append(_finding("benchmark_validated_source_claim", "error", "writer policy cannot benchmark-validate or register outputs"))
    if policy.require_no_benchmark_validated_claims:
        if bool(manifest.metadata.get("benchmark_validated")) or bool(benchmark_preflight_report.metadata.get("benchmark_validated")):
            findings.append(_finding("benchmark_validated_source_claim", "error", "source metadata claims benchmark validation"))
        for prediction in predictions:
            if bool(prediction.metadata.get("benchmark_validated")):
                findings.append(
                    _finding(
                        "benchmark_validated_source_claim",
                        "error",
                        "prediction metadata claims benchmark validation",
                        baseline_kind=prediction.baseline_kind,
                        target_property_id=prediction.target_property_id,
                        feature_view=prediction.feature_view,
                        split=prediction.split,
                    )
                )
        for metric in metrics:
            if bool(metric.metadata.get("benchmark_validated")):
                findings.append(
                    _finding(
                        "benchmark_validated_source_claim",
                        "error",
                        "metric metadata claims benchmark validation",
                        baseline_kind=metric.baseline_kind,
                        target_property_id=metric.target_property_id,
                        feature_view=metric.feature_view,
                        split=metric.split,
                    )
                )
    return findings


def _metric_cards_by_key(
    metrics: list[OledCuratedTrainingBaselineMetrics],
    preflight_report: OledBaselineBenchmarkPreflightReport,
) -> dict[tuple[str, str, str], list[OledBaselineBenchmarkMetricCard]]:
    summary_by_key = {
        (summary.baseline_kind, summary.target_property_id, summary.feature_view, summary.split): summary
        for summary in preflight_report.metrics_summaries
    }
    output: dict[tuple[str, str, str], list[OledBaselineBenchmarkMetricCard]] = {}
    for metric in sorted(metrics, key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view, item.split)):
        summary = summary_by_key.get((metric.baseline_kind, metric.target_property_id, metric.feature_view, metric.split))
        key = (metric.baseline_kind, metric.target_property_id, metric.feature_view)
        output.setdefault(key, []).append(
            OledBaselineBenchmarkMetricCard(
                baseline_kind=metric.baseline_kind,
                target_property_id=metric.target_property_id,
                feature_view=metric.feature_view,
                split=metric.split,
                row_count=metric.row_count,
                mae=metric.mae,
                rmse=metric.rmse,
                r2=metric.r2,
                bias=metric.bias,
                target_mean=metric.target_mean,
                prediction_mean=metric.prediction_mean,
                metric_status=_status_value(summary.status) if summary is not None else None,
                reason_codes=summary.reason_codes if summary is not None else [],
            )
        )
    return output


def _manifest(
    *,
    policy: OledBaselineBenchmarkReportWriterPolicy,
    benchmark_report: OledBaselineBenchmarkCandidateReport | None,
    findings: list[OledBaselineBenchmarkReportWriterFinding],
    baseline_run_manifest: OledCuratedTrainingBaselineRunnerManifest,
    benchmark_preflight_report: OledBaselineBenchmarkPreflightReport,
    output_directory: str | None,
    candidate_written: bool,
    file_results: list[OledBaselineBenchmarkReportFileResult] | None = None,
) -> OledBaselineBenchmarkReportWriterManifest:
    return OledBaselineBenchmarkReportWriterManifest(
        manifest_id=_report_id(baseline_run_manifest.manifest_id, benchmark_preflight_report.status).replace("report:", "manifest:"),
        source_baseline_run_manifest_id=baseline_run_manifest.manifest_id,
        source_benchmark_preflight_status=_status_value(benchmark_preflight_report.status),
        output_directory=output_directory,
        output_file_count=sum(1 for result in (file_results or []) if result.status == OledBaselineBenchmarkReportWriteStatus.WRITTEN),
        baseline_kinds=benchmark_report.baseline_kinds if benchmark_report is not None else [],
        target_property_ids=benchmark_report.target_property_ids if benchmark_report is not None else [],
        feature_views=benchmark_report.feature_views if benchmark_report is not None else [],
        file_results=file_results or _selection_file_results(benchmark_report, findings),
        policy=policy,
        metadata=_report_metadata(candidate_written=candidate_written),
    )


def _selection_file_results(
    benchmark_report: OledBaselineBenchmarkCandidateReport | None,
    findings: list[OledBaselineBenchmarkReportWriterFinding],
) -> list[OledBaselineBenchmarkReportFileResult]:
    if benchmark_report is None:
        return [
            OledBaselineBenchmarkReportFileResult(
                artifact_kind="benchmark_report_json",
                status=OledBaselineBenchmarkReportWriteStatus.REJECTED,
                reason_codes=sorted({finding.code for finding in findings} or {"report_rejected"}),
            )
        ]
    return [
        OledBaselineBenchmarkReportFileResult(
            artifact_kind="benchmark_report_json",
            status=OledBaselineBenchmarkReportWriteStatus.SKIPPED,
            reason_codes=["selected_for_report"],
        ),
        OledBaselineBenchmarkReportFileResult(
            artifact_kind="benchmark_report_markdown",
            status=OledBaselineBenchmarkReportWriteStatus.SKIPPED,
            reason_codes=["selected_for_report"],
        ),
    ]


def _write_report_files(
    writer_report: OledBaselineBenchmarkReportWriterReport,
    output_root: Path,
) -> OledBaselineBenchmarkReportWriterReport:
    assert writer_report.benchmark_report is not None
    benchmark_report = writer_report.benchmark_report
    file_results: list[OledBaselineBenchmarkReportFileResult] = []
    if writer_report.manifest.policy.write_json_report:
        path = output_root / oled_baseline_benchmark_report_json_filename()
        sha = write_oled_baseline_benchmark_report_json(benchmark_report, path)
        file_results.append(
            OledBaselineBenchmarkReportFileResult(
                artifact_kind="benchmark_report_json",
                status=OledBaselineBenchmarkReportWriteStatus.WRITTEN,
                output_path=path.name,
                output_sha256=sha,
                reason_codes=["report_json_written", "selected_for_report"],
            )
        )
    if writer_report.manifest.policy.write_markdown_report:
        path = output_root / oled_baseline_benchmark_report_markdown_filename()
        sha = write_oled_baseline_benchmark_report_markdown(benchmark_report, path)
        file_results.append(
            OledBaselineBenchmarkReportFileResult(
                artifact_kind="benchmark_report_markdown",
                status=OledBaselineBenchmarkReportWriteStatus.WRITTEN,
                output_path=path.name,
                output_sha256=sha,
                reason_codes=["report_markdown_written", "selected_for_report"],
            )
        )
    updated_report = benchmark_report.model_copy(update={"metadata": _report_metadata(candidate_written=True)})
    manifest = writer_report.manifest.model_copy(
        update={
            "output_directory": output_root.name,
            "output_file_count": len(file_results),
            "file_results": file_results,
            "metadata": _report_metadata(candidate_written=True),
        }
    )
    return writer_report.model_copy(update={"manifest": manifest, "benchmark_report": updated_report})


def _mark_dry_run(
    writer_report: OledBaselineBenchmarkReportWriterReport,
) -> OledBaselineBenchmarkReportWriterReport:
    manifest = writer_report.manifest.model_copy(
        update={
            "metadata": {
                **writer_report.manifest.metadata,
                "benchmark_candidate_report_written": False,
                "dry_run_no_files_written": True,
            },
            "file_results": [
                result.model_copy(update={"reason_codes": sorted(set(result.reason_codes) | {"dry_run_no_files_written"})})
                for result in writer_report.manifest.file_results
            ],
        }
    )
    return writer_report.model_copy(update={"manifest": manifest})


def _filter_predictions(
    predictions: list[OledCuratedTrainingBaselinePrediction],
    policy: OledBaselineBenchmarkReportWriterPolicy,
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
        key=lambda item: item.prediction_id,
    )


def _filter_metrics(
    metrics: list[OledCuratedTrainingBaselineMetrics],
    policy: OledBaselineBenchmarkReportWriterPolicy,
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
        key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view, item.split),
    )


def _filter_run_results(
    run_results: list[Any],
    policy: OledBaselineBenchmarkReportWriterPolicy,
) -> list[Any]:
    baselines = _baseline_kinds(policy)
    targets = _target_property_ids(policy)
    views = _feature_views(policy)
    return sorted(
        [
            result
            for result in run_results
            if (not baselines or result.baseline_kind in baselines)
            and result.target_property_id in targets
            and (not views or result.feature_view in views)
        ],
        key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view),
    )


def _baseline_kinds(policy: OledBaselineBenchmarkReportWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.baseline_kinds if str(item).strip()}


def _target_property_ids(policy: OledBaselineBenchmarkReportWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.target_property_ids if str(item).strip()}


def _feature_views(policy: OledBaselineBenchmarkReportWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.feature_views if str(item).strip()}


def _splits(policy: OledBaselineBenchmarkReportWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.splits if str(item).strip()}


def _report_id(manifest_id: str | None, preflight_status: Enum | str) -> str:
    return "report:oled-baseline-benchmark:" + _safe_id_token(f"{manifest_id or 'unknown'}:{_status_value(preflight_status)}")


def _safe_id_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", value).strip("-").lower() or "unknown"


def _write_bytes(path: str | Path, payload: bytes) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _format_metric(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    split: str | None = None,
) -> OledBaselineBenchmarkReportWriterFinding:
    return OledBaselineBenchmarkReportWriterFinding(
        code=code,
        severity=severity,
        message=message,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        split=split,
    )


def _dedup_findings(
    findings: list[OledBaselineBenchmarkReportWriterFinding],
) -> list[OledBaselineBenchmarkReportWriterFinding]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    output: list[OledBaselineBenchmarkReportWriterFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.baseline_kind or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.split or "",
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return sorted(output, key=lambda item: (item.severity, item.code, item.baseline_kind or "", item.target_property_id or "", item.feature_view or "", item.split or ""))


def _report_metadata(*, candidate_written: bool) -> dict[str, Any]:
    return {
        "benchmark_report_writer": True,
        "benchmark_candidate_report": True,
        "benchmark_candidate_report_written": candidate_written,
        "benchmark_results_written": False,
        "benchmark_registered": False,
        "benchmark_validated": False,
        "scientific_claim_validated": False,
        "baseline_backend_rerun": False,
        "models_fitted": False,
        "predictions_written": False,
        "metrics_written": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


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
            "prediction_rows",
            "metrics_payload",
            "gold_record",
            "layered_record",
        )
    )


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute()
    return False


def _reject_forbidden_input(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"forbidden_pdf_input:{redact_oled_mineru_acceptance_path(path)}")
    if suffix in _FORBIDDEN_IMAGE_SUFFIXES:
        raise ValueError(f"forbidden_image_input:{redact_oled_mineru_acceptance_path(path)}")


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


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
    "OledBaselineBenchmarkReportWriterPolicy",
    "OledBaselineBenchmarkReportWriteStatus",
    "OledBaselineBenchmarkMetricCard",
    "OledBaselineBenchmarkRunCard",
    "OledBaselineBenchmarkCandidateReport",
    "OledBaselineBenchmarkReportFileResult",
    "OledBaselineBenchmarkReportWriterFinding",
    "OledBaselineBenchmarkReportWriterManifest",
    "OledBaselineBenchmarkReportWriterReport",
    "load_oled_baseline_benchmark_preflight_report_json",
    "build_oled_baseline_benchmark_candidate_report",
    "select_oled_baseline_benchmark_report_for_write",
    "write_oled_baseline_benchmark_report_json",
    "write_oled_baseline_benchmark_report_markdown",
    "write_oled_baseline_benchmark_report_manifest_json",
    "oled_baseline_benchmark_report_json_filename",
    "oled_baseline_benchmark_report_markdown_filename",
    "run_oled_baseline_benchmark_report_writer_from_files",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
