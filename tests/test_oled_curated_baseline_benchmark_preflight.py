from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledBaselineBenchmarkArtifactStatus,
    OledBaselineBenchmarkPreflightFinding,
    OledBaselineBenchmarkPreflightPolicy,
    OledBaselineBenchmarkPreflightReport,
    OledBaselineBenchmarkPreflightStatus,
    OledBaselineMetricsConsistencySummary,
    OledBaselinePredictionCoverageSummary,
    OledCuratedTrainingBaselineKind,
    OledCuratedTrainingBaselineMetrics,
    OledCuratedTrainingBaselinePrediction,
    OledCuratedTrainingBaselineRunResult,
    OledCuratedTrainingBaselineRunStatus,
    OledCuratedTrainingBaselineRunnerManifest,
    OledCuratedTrainingBaselineRunnerPolicy,
    load_oled_training_baseline_artifacts_from_manifest as package_load_artifacts,
    load_oled_training_baseline_metrics_json as package_load_metrics,
    load_oled_training_baseline_predictions_jsonl as package_load_predictions,
    load_oled_training_baseline_runner_manifest_json as package_load_manifest,
    run_oled_baseline_benchmark_preflight as package_run_preflight,
    run_oled_baseline_benchmark_preflight_from_files as package_run_from_files,
    write_oled_baseline_benchmark_preflight_report_json as package_write_report,
    write_oled_training_baseline_manifest_json,
    write_oled_training_baseline_metrics_json,
    write_oled_training_baseline_predictions_jsonl,
)
from ai4s_agent.domains.oled_curated_baseline_benchmark_preflight import (
    load_oled_training_baseline_artifacts_from_manifest,
    load_oled_training_baseline_metrics_json,
    load_oled_training_baseline_predictions_jsonl,
    load_oled_training_baseline_runner_manifest_json,
    main,
    run_oled_baseline_benchmark_preflight,
    run_oled_baseline_benchmark_preflight_from_files,
    write_oled_baseline_benchmark_preflight_report_json,
)


def _prediction(
    suffix: str,
    *,
    split: str,
    y_true: float | int | None,
    y_pred: float | int | None = 22.0,
    prediction_id: str | None = None,
    evidence_refs: list[str] | None = None,
    metadata: dict | None = None,
) -> OledCuratedTrainingBaselinePrediction:
    residual = (float(y_pred) - float(y_true)) if y_true is not None and y_pred is not None else None
    return OledCuratedTrainingBaselinePrediction(
        prediction_id=prediction_id or f"prediction-{suffix}",
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
        evidence_refs=evidence_refs if evidence_refs is not None else [f"paper:test:{suffix}"],
        metadata=metadata
        if metadata is not None
        else {
            "baseline_prediction": True,
            "benchmark_validated": False,
            "model_backend_run": True,
        },
    )


def _predictions() -> list[OledCuratedTrainingBaselinePrediction]:
    return [
        _prediction("train-a", split="train", y_true=20.0),
        _prediction("train-b", split="train", y_true=24.0),
        _prediction("validation", split="validation", y_true=26.0),
        _prediction("test", split="test", y_true=18.0),
    ]


def _metric(split: str, *, row_count: int, mae: float, rmse: float, r2: float, bias: float, target_mean: float, prediction_mean: float) -> OledCuratedTrainingBaselineMetrics:
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
        metadata={"baseline_metrics": True, "benchmark_validated": False},
    )


def _metrics() -> list[OledCuratedTrainingBaselineMetrics]:
    return [
        _metric("train", row_count=2, mae=2.0, rmse=2.0, r2=0.0, bias=0.0, target_mean=22.0, prediction_mean=22.0),
        _metric("validation", row_count=1, mae=4.0, rmse=4.0, r2=0.0, bias=-4.0, target_mean=26.0, prediction_mean=22.0),
        _metric("test", row_count=1, mae=4.0, rmse=4.0, r2=0.0, bias=4.0, target_mean=18.0, prediction_mean=22.0),
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
            "baseline_backend_run": True,
            "models_fitted": True,
            "predictions_written": True,
            "metrics_written": True,
            "benchmark_validated": False,
        },
    )


def _write_artifacts(tmp_path: Path):
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_sha = write_oled_training_baseline_predictions_jsonl(_predictions(), prediction_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_sha = write_oled_training_baseline_metrics_json(_metrics(), metrics_path)
    manifest = _manifest(prediction_sha=prediction_sha, metrics_sha=metrics_sha)
    manifest_path = tmp_path / "baseline_manifest.json"
    write_oled_training_baseline_manifest_json(manifest, manifest_path)
    return manifest_path, manifest


def test_main_preflight_success_recomputes_metrics() -> None:
    report = run_oled_baseline_benchmark_preflight(
        manifest=_manifest(),
        predictions=_predictions(),
        metrics=_metrics(),
        policy=OledBaselineBenchmarkPreflightPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )

    assert report.is_valid
    assert report.status == OledBaselineBenchmarkPreflightStatus.PASSED
    assert report.input_prediction_count == 4
    assert report.input_metric_count == 3
    assert report.coverage_summaries[0].status == OledBaselineBenchmarkArtifactStatus.READY
    validation_summary = next(summary for summary in report.metrics_summaries if summary.split == "validation")
    assert validation_summary.recomputed_mae == validation_summary.reported_mae == 4.0
    assert report.metadata["benchmark_preflight_only"] is True
    assert report.metadata["benchmark_registered"] is False


def test_duplicate_prediction_id_fails() -> None:
    predictions = _predictions()
    predictions[1] = predictions[1].model_copy(update={"prediction_id": predictions[0].prediction_id})

    report = run_oled_baseline_benchmark_preflight(manifest=_manifest(), predictions=predictions, metrics=_metrics())

    assert not report.is_valid
    assert "duplicate_prediction_id" in report.error_codes


def test_missing_prediction_and_target_values_fail() -> None:
    predictions = [
        _prediction("missing-pred", split="train", y_true=20.0, y_pred=None),
        _prediction("missing-target", split="validation", y_true=None, y_pred=22.0),
    ]

    report = run_oled_baseline_benchmark_preflight(manifest=_manifest(), predictions=predictions, metrics=[])

    assert "missing_prediction_value" in report.error_codes
    assert "missing_target_value" in report.error_codes


def test_metric_recomputation_mismatch_fails() -> None:
    bad_metrics = _metrics()
    bad_metrics[1] = bad_metrics[1].model_copy(update={"mae": 99.0})

    report = run_oled_baseline_benchmark_preflight(manifest=_manifest(), predictions=_predictions(), metrics=bad_metrics)

    assert not report.is_valid
    assert "metric_value_mismatch" in report.error_codes


def test_metric_row_count_mismatch_fails() -> None:
    bad_metrics = _metrics()
    bad_metrics[0] = bad_metrics[0].model_copy(update={"row_count": 99})

    report = run_oled_baseline_benchmark_preflight(manifest=_manifest(), predictions=_predictions(), metrics=bad_metrics)

    assert not report.is_valid
    assert "metric_row_count_mismatch" in report.error_codes


def test_missing_metric_split_reports_missing_eval_metrics() -> None:
    metrics = [metric for metric in _metrics() if metric.split != "validation"]

    report = run_oled_baseline_benchmark_preflight(manifest=_manifest(), predictions=_predictions(), metrics=metrics)

    assert not report.is_valid
    assert "metric_split_missing" in report.error_codes
    assert "missing_eval_metrics" in report.error_codes


def test_benchmark_validated_claim_rejected() -> None:
    predictions = [_prediction("claim", split="train", y_true=20.0, metadata={"benchmark_validated": True})]
    manifest = _manifest(metadata={"benchmark_validated": True})

    report = run_oled_baseline_benchmark_preflight(manifest=manifest, predictions=predictions, metrics=[])

    assert not report.is_valid
    assert "benchmark_validated_source_claim" in report.error_codes


def test_manifest_and_artifact_loader_verify_sha(tmp_path: Path) -> None:
    manifest_path, manifest = _write_artifacts(tmp_path)

    loaded_manifest = load_oled_training_baseline_runner_manifest_json(manifest_path)
    predictions, metrics = load_oled_training_baseline_artifacts_from_manifest(manifest=loaded_manifest, base_dir=tmp_path)

    assert loaded_manifest.manifest_id == manifest.manifest_id
    assert len(predictions) == 4
    assert len(metrics) == 3

    bad_prediction_manifest = _manifest(prediction_sha="wrong-sha", metrics_sha=None)
    with pytest.raises(ValueError, match="baseline_predictions_sha256_mismatch:"):
        load_oled_training_baseline_artifacts_from_manifest(manifest=bad_prediction_manifest, base_dir=tmp_path)

    bad_metrics_manifest = _manifest(prediction_sha=None, metrics_sha="wrong-sha")
    with pytest.raises(ValueError, match="baseline_metrics_sha256_mismatch:"):
        load_oled_training_baseline_artifacts_from_manifest(manifest=bad_metrics_manifest, base_dir=tmp_path)


def test_prediction_and_metrics_loaders_reject_invalid_input(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    write_oled_training_baseline_predictions_jsonl(_predictions(), prediction_path)
    bad_prediction_path = tmp_path / "bad_predictions.jsonl"
    bad_prediction_path.write_text("{bad-json}\n", encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    write_oled_training_baseline_metrics_json(_metrics(), metrics_path)
    bad_metrics_path = tmp_path / "bad_metrics.json"
    bad_metrics_path.write_text("{bad-json}\n", encoding="utf-8")

    assert len(load_oled_training_baseline_predictions_jsonl(prediction_path)) == 4
    with pytest.raises(ValueError, match="invalid_baseline_predictions_jsonl:line-1"):
        load_oled_training_baseline_predictions_jsonl(bad_prediction_path)
    assert len(load_oled_training_baseline_metrics_json(metrics_path)) == 3
    with pytest.raises(ValueError, match="invalid_baseline_metrics_json:"):
        load_oled_training_baseline_metrics_json(bad_metrics_path)


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    manifest_path, _ = _write_artifacts(tmp_path)
    output_report = tmp_path / "benchmark_preflight.json"

    report = run_oled_baseline_benchmark_preflight_from_files(
        baseline_run_manifest_path=manifest_path,
        baseline_run_base_dir=tmp_path,
        output_report_path=output_report,
    )

    assert report.is_valid
    assert output_report.exists()
    assert not list(tmp_path.glob("*benchmark_result*"))
    assert not list(tmp_path.glob("*benchmark_validated*"))


def test_report_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_baseline_benchmark_preflight(manifest=_manifest(), predictions=_predictions(), metrics=_metrics())
    output_path = tmp_path / "report.json"

    write_oled_baseline_benchmark_preflight_report_json(report, output_path)
    first_text = output_path.read_text(encoding="utf-8")
    write_oled_baseline_benchmark_preflight_report_json(report, output_path)
    payload = json.loads(first_text)

    assert output_path.read_text(encoding="utf-8") == first_text
    assert first_text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert str(tmp_path) not in first_text
    assert "raw paper text" not in first_text
    assert "features" not in first_text
    assert "prediction-" not in first_text


def test_cli_smoke_writes_compact_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, _ = _write_artifacts(tmp_path)
    output_report = tmp_path / "benchmark_preflight.json"

    exit_code = main(
        [
            "--baseline-run-manifest",
            str(manifest_path),
            "--baseline-run-base-dir",
            str(tmp_path),
            "--output-report",
            str(output_report),
            "--target-property-id",
            "eqe_percent",
            "--feature-view",
            "full_context",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output_report.exists()
    assert "predictions" not in stdout
    assert "metrics" not in stdout
    assert json.loads(stdout)["input_prediction_count"] == 4


def test_package_exports_for_benchmark_preflight(tmp_path: Path) -> None:
    manifest_path, manifest = _write_artifacts(tmp_path)
    loaded_manifest = package_load_manifest(manifest_path)
    predictions = package_load_predictions(tmp_path / "predictions.jsonl")
    metrics = package_load_metrics(tmp_path / "metrics.json")
    loaded_predictions, loaded_metrics = package_load_artifacts(manifest=loaded_manifest, base_dir=tmp_path)
    report = package_run_preflight(manifest=manifest, predictions=predictions, metrics=metrics)
    runner_report = package_run_from_files(baseline_run_manifest_path=manifest_path, baseline_run_base_dir=tmp_path)
    output_path = tmp_path / "package-report.json"
    package_write_report(report, output_path)

    assert isinstance(report, OledBaselineBenchmarkPreflightReport)
    assert isinstance(report.coverage_summaries[0], OledBaselinePredictionCoverageSummary)
    assert isinstance(report.metrics_summaries[0], OledBaselineMetricsConsistencySummary)
    assert isinstance(OledBaselineBenchmarkPreflightFinding(code="x", message="y"), OledBaselineBenchmarkPreflightFinding)
    assert isinstance(OledBaselineBenchmarkPreflightPolicy(), OledBaselineBenchmarkPreflightPolicy)
    assert OledBaselineBenchmarkPreflightStatus.PASSED.value == "passed"
    assert OledBaselineBenchmarkArtifactStatus.READY.value == "ready"
    assert len(loaded_predictions) == len(predictions)
    assert len(loaded_metrics) == len(metrics)
    assert runner_report.is_valid
    assert output_path.exists()
