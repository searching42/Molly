from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledBaselineBenchmarkArtifactStatus,
    OledBaselineBenchmarkCandidateReport,
    OledBaselineBenchmarkMetricCard,
    OledBaselineBenchmarkPreflightFinding,
    OledBaselineBenchmarkPreflightPolicy,
    OledBaselineBenchmarkPreflightStatus,
    OledBaselineBenchmarkReportFileResult,
    OledBaselineBenchmarkReportWriteStatus,
    OledBaselineBenchmarkReportWriterFinding,
    OledBaselineBenchmarkReportWriterManifest,
    OledBaselineBenchmarkReportWriterPolicy,
    OledBaselineBenchmarkReportWriterReport,
    OledBaselineBenchmarkRunCard,
    OledBaselineMetricsConsistencySummary,
    OledBaselinePredictionCoverageSummary,
    OledCuratedTrainingBaselineKind,
    OledCuratedTrainingBaselineMetrics,
    OledCuratedTrainingBaselinePrediction,
    OledCuratedTrainingBaselineRunResult,
    OledCuratedTrainingBaselineRunStatus,
    OledCuratedTrainingBaselineRunnerManifest,
    OledCuratedTrainingBaselineRunnerPolicy,
    build_oled_baseline_benchmark_candidate_report as package_build_report,
    load_oled_baseline_benchmark_preflight_report_json as package_load_preflight,
    oled_baseline_benchmark_report_json_filename as package_json_filename,
    oled_baseline_benchmark_report_markdown_filename as package_markdown_filename,
    run_oled_baseline_benchmark_preflight,
    run_oled_baseline_benchmark_report_writer_from_files as package_run_from_files,
    select_oled_baseline_benchmark_report_for_write as package_select_report,
    write_oled_baseline_benchmark_report_json as package_write_json,
    write_oled_baseline_benchmark_report_manifest_json as package_write_manifest,
    write_oled_baseline_benchmark_report_markdown as package_write_markdown,
    write_oled_baseline_benchmark_preflight_report_json,
    write_oled_training_baseline_manifest_json,
    write_oled_training_baseline_metrics_json,
    write_oled_training_baseline_predictions_jsonl,
)
from ai4s_agent.domains.oled_curated_baseline_benchmark_report_writer import (
    build_oled_baseline_benchmark_candidate_report,
    load_oled_baseline_benchmark_preflight_report_json,
    main,
    oled_baseline_benchmark_report_json_filename,
    oled_baseline_benchmark_report_markdown_filename,
    run_oled_baseline_benchmark_report_writer_from_files,
    select_oled_baseline_benchmark_report_for_write,
    write_oled_baseline_benchmark_report_json,
    write_oled_baseline_benchmark_report_manifest_json,
    write_oled_baseline_benchmark_report_markdown,
)


def _prediction(
    suffix: str,
    *,
    split: str,
    y_true: float | int | None,
    y_pred: float | int | None = 22.0,
    metadata: dict | None = None,
) -> OledCuratedTrainingBaselinePrediction:
    residual = (float(y_pred) - float(y_true)) if y_true is not None and y_pred is not None else None
    return OledCuratedTrainingBaselinePrediction(
        prediction_id=f"prediction-{suffix}",
        baseline_kind=OledCuratedTrainingBaselineKind.MEAN_BASELINE.value,
        split=split,
        target_property_id="eqe_percent",
        feature_view="full_context",
        training_row_id=f"training-row-{suffix}",
        record_id=f"record-{suffix}",
        feature_row_id=f"feature-row-{suffix}",
        y_true=y_true,
        y_pred=y_pred,
        residual=round(residual, 6) if residual is not None else None,
        absolute_error=round(abs(residual), 6) if residual is not None else None,
        evidence_refs=[f"paper:test:{suffix}"],
        metadata=metadata
        if metadata is not None
        else {
            "baseline_prediction": True,
            "benchmark_validated": False,
            "model_backend_run": True,
        },
    )


def _predictions(*, metadata: dict | None = None) -> list[OledCuratedTrainingBaselinePrediction]:
    return [
        _prediction("train-a", split="train", y_true=20.0, metadata=metadata),
        _prediction("train-b", split="train", y_true=24.0, metadata=metadata),
        _prediction("validation", split="validation", y_true=26.0, metadata=metadata),
        _prediction("test", split="test", y_true=18.0, metadata=metadata),
    ]


def _metric(
    split: str,
    *,
    row_count: int,
    mae: float,
    rmse: float,
    r2: float,
    bias: float,
    target_mean: float,
    prediction_mean: float,
    metadata: dict | None = None,
) -> OledCuratedTrainingBaselineMetrics:
    return OledCuratedTrainingBaselineMetrics(
        baseline_kind=OledCuratedTrainingBaselineKind.MEAN_BASELINE.value,
        target_property_id="eqe_percent",
        feature_view="full_context",
        split=split,
        row_count=row_count,
        mae=mae,
        rmse=rmse,
        r2=r2,
        bias=bias,
        target_mean=target_mean,
        prediction_mean=prediction_mean,
        metadata=metadata if metadata is not None else {"baseline_metrics": True, "benchmark_validated": False},
    )


def _metrics(*, metadata: dict | None = None) -> list[OledCuratedTrainingBaselineMetrics]:
    return [
        _metric("train", row_count=2, mae=2.0, rmse=2.0, r2=0.0, bias=0.0, target_mean=22.0, prediction_mean=22.0, metadata=metadata),
        _metric("validation", row_count=1, mae=4.0, rmse=4.0, r2=0.0, bias=-4.0, target_mean=26.0, prediction_mean=22.0, metadata=metadata),
        _metric("test", row_count=1, mae=4.0, rmse=4.0, r2=0.0, bias=4.0, target_mean=18.0, prediction_mean=22.0, metadata=metadata),
    ]


def _manifest(
    *,
    prediction_sha: str | None = "prediction-sha",
    metrics_sha: str | None = "metrics-sha",
    metadata: dict | None = None,
) -> OledCuratedTrainingBaselineRunnerManifest:
    return OledCuratedTrainingBaselineRunnerManifest(
        manifest_id="oled-baseline-runner:test",
        source_training_package_manifest_id="training-package:test",
        source_backend_preflight_status="passed",
        output_directory="baseline_run",
        output_file_count=2,
        output_prediction_count=4,
        baseline_kinds=[OledCuratedTrainingBaselineKind.MEAN_BASELINE.value],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_results=[
            OledCuratedTrainingBaselineRunResult(
                baseline_kind=OledCuratedTrainingBaselineKind.MEAN_BASELINE.value,
                target_property_id="eqe_percent",
                feature_view="full_context",
                status=OledCuratedTrainingBaselineRunStatus.COMPLETED,
                train_row_count=2,
                validation_row_count=1,
                test_row_count=1,
                prediction_count=4,
                metric_splits=["train", "validation", "test"],
                reason_codes=["baseline_completed", "selected_for_run"],
                prediction_jsonl_path="predictions.jsonl",
                prediction_sha256=prediction_sha,
                metrics_json_path="metrics.json",
                metrics_sha256=metrics_sha,
            )
        ],
        status_counts={"completed": 1},
        reason_code_counts={"baseline_completed": 1, "selected_for_run": 1},
        policy=OledCuratedTrainingBaselineRunnerPolicy(),
        metadata=metadata
        if metadata is not None
        else {
            "baseline_runner": True,
            "benchmark_validated": False,
            "benchmark_results_written": False,
        },
    )


def _preflight_report(*, status: OledBaselineBenchmarkPreflightStatus = OledBaselineBenchmarkPreflightStatus.PASSED, warning: bool = False, metadata: dict | None = None):
    report = run_oled_baseline_benchmark_preflight(
        manifest=_manifest(),
        predictions=_predictions(),
        metrics=_metrics(),
        policy=OledBaselineBenchmarkPreflightPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    if warning:
        report.findings.append(
            OledBaselineBenchmarkPreflightFinding(
                code="synthetic_warning",
                severity="warning",
                message="synthetic warning",
            )
        )
        report = report.model_copy(update={"status": OledBaselineBenchmarkPreflightStatus.PASSED_WITH_WARNINGS})
    if status != report.status:
        report = report.model_copy(update={"status": status})
    if metadata is not None:
        report = report.model_copy(update={"metadata": metadata})
    return report


def _write_artifacts(tmp_path: Path):
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_sha = write_oled_training_baseline_predictions_jsonl(_predictions(), prediction_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_sha = write_oled_training_baseline_metrics_json(_metrics(), metrics_path)
    manifest = _manifest(prediction_sha=prediction_sha, metrics_sha=metrics_sha)
    manifest_path = tmp_path / "baseline_manifest.json"
    write_oled_training_baseline_manifest_json(manifest, manifest_path)
    preflight = run_oled_baseline_benchmark_preflight(
        manifest=manifest,
        predictions=_predictions(),
        metrics=_metrics(),
        policy=OledBaselineBenchmarkPreflightPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    preflight_path = tmp_path / "benchmark_preflight.json"
    write_oled_baseline_benchmark_preflight_report_json(preflight, preflight_path)
    return manifest_path, preflight_path, manifest, preflight


def test_confirmation_gate() -> None:
    with pytest.raises(ValueError, match="confirmation_required:benchmark_report_write"):
        select_oled_baseline_benchmark_report_for_write(
            baseline_run_manifest=_manifest(),
            predictions=_predictions(),
            metrics=_metrics(),
            benchmark_preflight_report=_preflight_report(),
        )


def test_build_report_success() -> None:
    report, findings = build_oled_baseline_benchmark_candidate_report(
        baseline_run_manifest=_manifest(),
        predictions=_predictions(),
        metrics=_metrics(),
        benchmark_preflight_report=_preflight_report(),
        policy=OledBaselineBenchmarkReportWriterPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )

    assert findings == []
    assert report is not None
    assert report.source_baseline_run_manifest_id == "oled-baseline-runner:test"
    assert report.source_benchmark_preflight_status == "passed"
    assert report.metadata["benchmark_validated"] is False
    assert report.metadata["benchmark_registered"] is False
    assert "not_benchmark_validated" in report.caveats
    assert len(report.run_cards) == 1
    run_card = report.run_cards[0]
    assert run_card.prediction_count == 4
    assert run_card.metrics[0].target_property_id == "eqe_percent"
    assert {metric.split for metric in run_card.metrics} == {"train", "validation", "test"}


def test_invalid_preflight_blocks_writer() -> None:
    writer_report = select_oled_baseline_benchmark_report_for_write(
        baseline_run_manifest=_manifest(),
        predictions=_predictions(),
        metrics=_metrics(),
        benchmark_preflight_report=_preflight_report(status=OledBaselineBenchmarkPreflightStatus.FAILED),
        confirm_benchmark_report_write=True,
    )

    assert writer_report.benchmark_report is None
    assert not writer_report.is_valid
    assert "benchmark_preflight_failed" in writer_report.error_codes


def test_preflight_warnings_disallowed() -> None:
    writer_report = select_oled_baseline_benchmark_report_for_write(
        baseline_run_manifest=_manifest(),
        predictions=_predictions(),
        metrics=_metrics(),
        benchmark_preflight_report=_preflight_report(warning=True),
        policy=OledBaselineBenchmarkReportWriterPolicy(allow_preflight_warnings=False),
        confirm_benchmark_report_write=True,
    )

    assert writer_report.benchmark_report is None
    assert "benchmark_preflight_warnings_present" in writer_report.error_codes


def test_benchmark_validated_source_claim_rejected() -> None:
    cases = [
        (_manifest(metadata={"benchmark_validated": True}), _predictions(), _metrics(), _preflight_report()),
        (_manifest(), _predictions(metadata={"benchmark_validated": True}), _metrics(), _preflight_report()),
        (_manifest(), _predictions(), _metrics(metadata={"benchmark_validated": True}), _preflight_report()),
        (_manifest(), _predictions(), _metrics(), _preflight_report(metadata={"benchmark_validated": True})),
    ]
    for manifest, predictions, metrics, preflight in cases:
        report, findings = build_oled_baseline_benchmark_candidate_report(
            baseline_run_manifest=manifest,
            predictions=predictions,
            metrics=metrics,
            benchmark_preflight_report=preflight,
        )
        assert report is None
        assert "benchmark_validated_source_claim" in [finding.code for finding in findings]


def test_json_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report, _ = build_oled_baseline_benchmark_candidate_report(
        baseline_run_manifest=_manifest(),
        predictions=_predictions(),
        metrics=_metrics(),
        benchmark_preflight_report=_preflight_report(),
    )
    assert report is not None
    path = tmp_path / oled_baseline_benchmark_report_json_filename()
    first_sha = write_oled_baseline_benchmark_report_json(report, path)
    first_text = path.read_text(encoding="utf-8")
    second_sha = write_oled_baseline_benchmark_report_json(report, path)

    assert first_sha == second_sha
    assert first_text == path.read_text(encoding="utf-8")
    assert str(tmp_path) not in first_text
    assert "Raw Paper Text" not in first_text
    assert "features" not in first_text
    assert "prediction-train" not in first_text
    assert "training-row-" not in first_text
    assert '"benchmark_validated": false' in first_text


def test_markdown_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report, _ = build_oled_baseline_benchmark_candidate_report(
        baseline_run_manifest=_manifest(),
        predictions=_predictions(),
        metrics=_metrics(),
        benchmark_preflight_report=_preflight_report(),
    )
    assert report is not None
    path = tmp_path / oled_baseline_benchmark_report_markdown_filename()
    first_sha = write_oled_baseline_benchmark_report_markdown(report, path)
    first_text = path.read_text(encoding="utf-8")
    second_sha = write_oled_baseline_benchmark_report_markdown(report, path)

    assert first_sha == second_sha
    assert "OLED Baseline Benchmark Candidate Report" in first_text
    assert "not_benchmark_validated" in first_text
    assert "| Baseline | Target | Feature view | Split | Rows | MAE | RMSE | R2 | Bias |" in first_text
    assert "prediction-" not in first_text
    assert "Raw Paper Text" not in first_text
    assert "features" not in first_text


def test_manifest_writer_is_deterministic(tmp_path: Path) -> None:
    manifest = OledBaselineBenchmarkReportWriterManifest(
        manifest_id="oled-baseline-benchmark-report:test",
        output_directory="benchmark_report",
        output_file_count=2,
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledBaselineBenchmarkReportFileResult(
                artifact_kind="benchmark_report_json",
                status=OledBaselineBenchmarkReportWriteStatus.WRITTEN,
                output_path="oled_baseline_benchmark_candidate_report.json",
                output_sha256="abc123",
                reason_codes=["report_json_written"],
            )
        ],
        policy=OledBaselineBenchmarkReportWriterPolicy(),
        metadata={"benchmark_report_writer": True, "benchmark_registered": False},
    )
    path = tmp_path / "manifest.json"
    write_oled_baseline_benchmark_report_manifest_json(manifest, path)
    first_text = path.read_text(encoding="utf-8")
    write_oled_baseline_benchmark_report_manifest_json(manifest, path)

    assert first_text == path.read_text(encoding="utf-8")
    assert "abc123" in first_text
    assert '"benchmark_registered": false' in first_text


def test_combined_runner_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _ = _write_artifacts(tmp_path)
    output_dir = tmp_path / "reports"
    output_manifest = tmp_path / "report_manifest.json"

    report = run_oled_baseline_benchmark_report_writer_from_files(
        baseline_run_manifest_path=manifest_path,
        benchmark_preflight_report_path=preflight_path,
        baseline_run_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert not output_dir.exists()
    assert report.manifest.metadata["benchmark_candidate_report_written"] is False


def test_combined_runner_write_mode(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _ = _write_artifacts(tmp_path)
    output_dir = tmp_path / "reports"
    output_manifest = tmp_path / "report_manifest.json"

    report = run_oled_baseline_benchmark_report_writer_from_files(
        baseline_run_manifest_path=manifest_path,
        benchmark_preflight_report_path=preflight_path,
        baseline_run_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        confirm_benchmark_report_write=True,
    )

    assert report.is_valid
    assert (output_dir / oled_baseline_benchmark_report_json_filename()).exists()
    assert (output_dir / oled_baseline_benchmark_report_markdown_filename()).exists()
    assert output_manifest.exists()
    assert all(result.output_sha256 for result in report.manifest.file_results if result.status == OledBaselineBenchmarkReportWriteStatus.WRITTEN)


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, preflight_path, _, _ = _write_artifacts(tmp_path)
    output_dir = tmp_path / "cli-report"
    output_manifest = tmp_path / "cli-report-manifest.json"

    exit_code = main(
        [
            "--baseline-run-manifest",
            str(manifest_path),
            "--benchmark-preflight-report",
            str(preflight_path),
            "--baseline-run-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-benchmark-report-write",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_manifest.exists()
    assert (output_dir / oled_baseline_benchmark_report_json_filename()).exists()
    assert "prediction-" not in captured.out
    assert "metrics" not in captured.out.lower()


def test_package_exports() -> None:
    assert OledBaselineBenchmarkReportWriterPolicy
    assert OledBaselineBenchmarkReportWriteStatus
    assert OledBaselineBenchmarkMetricCard
    assert OledBaselineBenchmarkRunCard
    assert OledBaselineBenchmarkCandidateReport
    assert OledBaselineBenchmarkReportFileResult
    assert OledBaselineBenchmarkReportWriterFinding
    assert OledBaselineBenchmarkReportWriterManifest
    assert OledBaselineBenchmarkReportWriterReport
    assert package_load_preflight
    assert package_build_report
    assert package_select_report
    assert package_write_json
    assert package_write_markdown
    assert package_write_manifest
    assert package_json_filename() == "oled_baseline_benchmark_candidate_report.json"
    assert package_markdown_filename() == "oled_baseline_benchmark_candidate_report.md"
    assert package_run_from_files
