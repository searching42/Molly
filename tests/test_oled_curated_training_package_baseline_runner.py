from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledCuratedSplitTrainingPackageWriteStatus,
    OledCuratedSplitTrainingPackageWriterManifest,
    OledCuratedSplitTrainingPackageWriterPolicy,
    OledCuratedTrainingBaselineKind,
    OledCuratedTrainingBaselineMetrics,
    OledCuratedTrainingBaselinePrediction,
    OledCuratedTrainingBaselineRunResult,
    OledCuratedTrainingBaselineRunStatus,
    OledCuratedTrainingBaselineRunnerFinding,
    OledCuratedTrainingBaselineRunnerManifest,
    OledCuratedTrainingBaselineRunnerPolicy,
    OledCuratedTrainingBaselineRunnerReport,
    OledCuratedTrainingPackageFileResult,
    OledCuratedTrainingPackageRow,
    OledCuratedTrainingPackageSchema,
    OledTrainingPackageBackendPreflightPolicy,
    load_oled_training_package_backend_preflight_report_json as package_load_backend_preflight,
    oled_training_baseline_metrics_filename as package_metrics_filename,
    oled_training_baseline_predictions_filename as package_predictions_filename,
    run_oled_curated_training_baseline_runner_from_files as package_run_from_files,
    run_oled_mean_baseline_on_training_rows as package_run_mean,
    run_oled_sklearn_baseline_on_training_rows as package_run_sklearn,
    run_oled_training_package_backend_preflight,
    select_oled_training_baselines_for_run as package_select,
    write_oled_curated_training_package_manifest_json,
    write_oled_curated_training_package_schema_json,
    write_oled_curated_training_rows_jsonl,
    write_oled_training_baseline_manifest_json as package_write_manifest,
    write_oled_training_baseline_metrics_json as package_write_metrics,
    write_oled_training_baseline_predictions_jsonl as package_write_predictions,
    write_oled_training_package_backend_preflight_report_json,
)
from ai4s_agent.domains.oled_curated_training_package_baseline_runner import (
    load_oled_training_package_backend_preflight_report_json,
    main,
    oled_training_baseline_metrics_filename,
    oled_training_baseline_predictions_filename,
    run_oled_curated_training_baseline_runner_from_files,
    run_oled_mean_baseline_on_training_rows,
    run_oled_sklearn_baseline_on_training_rows,
    select_oled_training_baselines_for_run,
    write_oled_training_baseline_manifest_json,
    write_oled_training_baseline_metrics_json,
    write_oled_training_baseline_predictions_jsonl,
)


def _training_row(
    suffix: str,
    *,
    split: str = "train",
    target_value: float | int | str = 21.0,
    target_property_id: str = "eqe_percent",
    feature_view: str = "full_context",
    features: dict | None = None,
) -> OledCuratedTrainingPackageRow:
    return OledCuratedTrainingPackageRow(
        training_row_id=f"training-row-{suffix}",
        split=split,
        feature_row_id=f"feature-row-{suffix}",
        split_row_id=f"split-row-{suffix}",
        row_id=f"row-{suffix}",
        record_id=f"record-{suffix}",
        source_record_ids=[f"record-{suffix}"],
        target_property_id=target_property_id,
        target_value=target_value,
        target_unit="%",
        feature_view=feature_view,
        features=features
        if features is not None
        else {
            "numeric_feature": 1.5,
            "boolean_feature": True,
            "categorical_feature": "host-a",
            "nested": {"depth": 2},
        },
        condition_hash=f"condition-{suffix}",
        confidence_score=0.9,
        evidence_refs=[f"paper:test:{suffix}"],
        metadata={
            "ml_ready_training_row": True,
            "benchmark_validated": False,
            "model_backend_run": False,
        },
    )


def _valid_rows() -> list[OledCuratedTrainingPackageRow]:
    return [
        _training_row("train-a", split="train", target_value=20.0),
        _training_row("train-b", split="train", target_value=24.0),
        _training_row("validation", split="validation", target_value=26.0),
        _training_row("test", split="test", target_value=18.0),
    ]


def _schema() -> OledCuratedTrainingPackageSchema:
    return OledCuratedTrainingPackageSchema(
        schema_id="oled-training-schema:test",
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        splits=["train", "validation", "test"],
        target_columns=["target_property_id", "target_value", "target_unit"],
        feature_columns=["boolean_feature", "categorical_feature", "nested", "numeric_feature"],
        metadata_columns=[
            "training_row_id",
            "split",
            "record_id",
            "feature_row_id",
            "split_row_id",
            "condition_hash",
            "confidence_score",
            "evidence_refs",
        ],
        feature_column_kinds={"numeric_feature": "numeric"},
        required_columns=[],
        metadata={"training_package_schema": True, "benchmark_validated": False},
    )


def _backend_preflight(rows: list[OledCuratedTrainingPackageRow] | None = None):
    return run_oled_training_package_backend_preflight(
        training_rows=rows or _valid_rows(),
        schema=_schema(),
        policy=OledTrainingPackageBackendPreflightPolicy(
            target_property_ids=["eqe_percent"],
            feature_views=["full_context"],
        ),
    )


def _package_manifest(
    *,
    rows_sha: str | None = "rows-sha",
    schema_sha: str | None = "schema-sha",
) -> OledCuratedSplitTrainingPackageWriterManifest:
    return OledCuratedSplitTrainingPackageWriterManifest(
        manifest_id="oled-training-package-writer:test",
        output_file_count=2,
        output_row_count=4,
        splits=["train", "validation", "test"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        rows_by_split={"train": 2, "validation": 1, "test": 1},
        rows_by_target={"eqe_percent": 4},
        rows_by_feature_view={"full_context": 4},
        file_results=[
            OledCuratedTrainingPackageFileResult(
                split="train",
                target_property_id="eqe_percent",
                feature_view="full_context",
                artifact_kind="training_rows",
                status=OledCuratedSplitTrainingPackageWriteStatus.WRITTEN,
                row_count=4,
                output_path="training_rows.jsonl",
                output_sha256=rows_sha,
                reason_codes=["selected_for_write"],
            ),
            OledCuratedTrainingPackageFileResult(
                artifact_kind="schema",
                status=OledCuratedSplitTrainingPackageWriteStatus.WRITTEN,
                output_path="oled_training_schema.json",
                output_sha256=schema_sha,
                reason_codes=["selected_for_write"],
            ),
        ],
        policy=OledCuratedSplitTrainingPackageWriterPolicy(),
        metadata={"training_package_writer": True, "training_package_written": True},
    )


def _write_package_files(tmp_path: Path, rows: list[OledCuratedTrainingPackageRow] | None = None):
    package_rows = rows or _valid_rows()
    rows_path = tmp_path / "training_rows.jsonl"
    rows_sha = write_oled_curated_training_rows_jsonl(package_rows, rows_path)
    schema_path = tmp_path / "oled_training_schema.json"
    schema_sha = write_oled_curated_training_package_schema_json(_schema(), schema_path)
    manifest = _package_manifest(rows_sha=rows_sha, schema_sha=schema_sha)
    manifest_path = tmp_path / "training_manifest.json"
    write_oled_curated_training_package_manifest_json(manifest, manifest_path)
    preflight_path = tmp_path / "backend_preflight.json"
    write_oled_training_package_backend_preflight_report_json(_backend_preflight(package_rows), preflight_path)
    return manifest_path, preflight_path


def test_confirmation_gate_requires_explicit_baseline_run() -> None:
    with pytest.raises(ValueError, match="confirmation_required:baseline_run"):
        select_oled_training_baselines_for_run(
            training_rows=_valid_rows(),
            schema=_schema(),
            backend_preflight_report=_backend_preflight(),
        )


def test_mean_baseline_success_predictions_metrics_and_deterministic_ids() -> None:
    first_predictions, first_metrics = run_oled_mean_baseline_on_training_rows(
        _valid_rows(),
        target_property_id="eqe_percent",
        feature_view="full_context",
    )
    second_predictions, _ = run_oled_mean_baseline_on_training_rows(
        _valid_rows(),
        target_property_id="eqe_percent",
        feature_view="full_context",
    )

    assert len(first_predictions) == 4
    assert [prediction.prediction_id for prediction in first_predictions] == [prediction.prediction_id for prediction in second_predictions]
    assert {prediction.split for prediction in first_predictions} == {"train", "validation", "test"}
    assert all(prediction.y_pred == 22.0 for prediction in first_predictions)
    assert all(prediction.metadata["baseline_prediction"] is True for prediction in first_predictions)
    assert all(prediction.metadata["benchmark_validated"] is False for prediction in first_predictions)
    assert {metric.split for metric in first_metrics} == {"train", "validation", "test"}
    assert next(metric for metric in first_metrics if metric.split == "validation").mae == 4.0


def test_missing_train_targets_block_runner() -> None:
    rows = [
        _training_row("train", split="train", target_value="high"),
        _training_row("validation", split="validation", target_value=21.0),
    ]

    report = select_oled_training_baselines_for_run(
        training_rows=rows,
        schema=_schema(),
        backend_preflight_report=_backend_preflight(_valid_rows()),
        confirm_baseline_run=True,
    )

    assert not report.is_valid
    assert "missing_numeric_train_targets" in report.error_codes
    assert report.predictions == []


def test_backend_preflight_invalid_blocks_runner() -> None:
    report = select_oled_training_baselines_for_run(
        training_rows=_valid_rows(),
        schema=_schema(),
        backend_preflight_report=_backend_preflight().model_copy(update={"status": "failed"}),
        confirm_baseline_run=True,
    )

    assert not report.is_valid
    assert "backend_preflight_failed" in report.error_codes


def test_sklearn_unavailable_skips_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "sklearn" else importlib.util.find_spec(name))

    predictions, metrics, findings = run_oled_sklearn_baseline_on_training_rows(
        _valid_rows(),
        baseline_kind=OledCuratedTrainingBaselineKind.TABULAR_RIDGE_SKLEARN,
        target_property_id="eqe_percent",
        feature_view="full_context",
    )

    assert predictions == []
    assert metrics == []
    assert [finding.code for finding in findings] == ["optional_dependency_unavailable:sklearn"]


def test_selection_uses_flattening_path_for_sklearn_when_available_or_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _valid_rows()
    original_find_spec = importlib.util.find_spec
    if original_find_spec("sklearn") is None:
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "sklearn" else original_find_spec(name))
        _, _, findings = run_oled_sklearn_baseline_on_training_rows(
            rows,
            baseline_kind=OledCuratedTrainingBaselineKind.TABULAR_RANDOM_FOREST_SKLEARN,
            target_property_id="eqe_percent",
            feature_view="full_context",
        )
        assert findings[0].code == "optional_dependency_unavailable:sklearn"
    else:
        predictions, metrics, findings = run_oled_sklearn_baseline_on_training_rows(
            rows,
            baseline_kind=OledCuratedTrainingBaselineKind.TABULAR_RIDGE_SKLEARN,
            target_property_id="eqe_percent",
            feature_view="full_context",
        )
        assert not findings
        assert predictions
        assert metrics


def test_prediction_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    predictions, _ = run_oled_mean_baseline_on_training_rows(_valid_rows(), target_property_id="eqe_percent", feature_view="full_context")
    output_path = tmp_path / "predictions.jsonl"

    first_sha = write_oled_training_baseline_predictions_jsonl(predictions, output_path)
    first_text = output_path.read_text(encoding="utf-8")
    second_sha = write_oled_training_baseline_predictions_jsonl(predictions, output_path)

    assert first_sha == second_sha
    assert first_text == output_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in first_text
    assert "features" not in first_text
    assert "raw paper text" not in first_text


def test_metrics_writer_is_deterministic(tmp_path: Path) -> None:
    _, metrics = run_oled_mean_baseline_on_training_rows(_valid_rows(), target_property_id="eqe_percent", feature_view="full_context")
    output_path = tmp_path / "metrics.json"

    first_sha = write_oled_training_baseline_metrics_json(metrics, output_path)
    first_text = output_path.read_text(encoding="utf-8")
    second_sha = write_oled_training_baseline_metrics_json(metrics, output_path)

    assert first_sha == second_sha
    assert first_text == output_path.read_text(encoding="utf-8")
    assert json.loads(first_text)[0]["baseline_kind"] == "mean_baseline"


def test_manifest_writer_is_deterministic_and_has_safety_metadata(tmp_path: Path) -> None:
    manifest = OledCuratedTrainingBaselineRunnerManifest(
        manifest_id="oled-baseline-runner:test",
        output_file_count=2,
        output_prediction_count=4,
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_results=[
            OledCuratedTrainingBaselineRunResult(
                baseline_kind="mean_baseline",
                target_property_id="eqe_percent",
                feature_view="full_context",
                status=OledCuratedTrainingBaselineRunStatus.COMPLETED,
                prediction_count=4,
                prediction_jsonl_path="predictions.jsonl",
                prediction_sha256="abc",
                metrics_json_path="metrics.json",
                metrics_sha256="def",
                reason_codes=["baseline_completed"],
            )
        ],
        status_counts={"completed": 1},
        reason_code_counts={"baseline_completed": 1},
        policy=OledCuratedTrainingBaselineRunnerPolicy(),
        metadata={
            "baseline_runner": True,
            "baseline_backend_run": True,
            "models_fitted": True,
            "predictions_written": True,
            "metrics_written": True,
            "benchmark_validated": False,
        },
    )
    output_path = tmp_path / "manifest.json"

    write_oled_training_baseline_manifest_json(manifest, output_path)
    text = output_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert payload["metadata"]["benchmark_validated"] is False
    assert payload["run_results"][0]["prediction_sha256"] == "abc"


def test_combined_runner_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    manifest_path, preflight_path = _write_package_files(tmp_path)
    output_manifest = tmp_path / "baseline_manifest.json"

    report = run_oled_curated_training_baseline_runner_from_files(
        training_package_manifest_path=manifest_path,
        backend_preflight_report_path=preflight_path,
        training_package_base_dir=tmp_path,
        output_manifest_path=output_manifest,
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert report.predictions
    assert "dry_run_no_files_written" in report.manifest.run_results[0].reason_codes
    assert not list(tmp_path.glob("oled_baseline_predictions__*.jsonl"))
    assert not list(tmp_path.glob("oled_baseline_metrics__*.json"))


def test_combined_runner_write_mode_writes_artifacts(tmp_path: Path) -> None:
    manifest_path, preflight_path = _write_package_files(tmp_path)
    output_dir = tmp_path / "baseline_run"
    output_manifest = tmp_path / "baseline_manifest.json"

    report = run_oled_curated_training_baseline_runner_from_files(
        training_package_manifest_path=manifest_path,
        backend_preflight_report_path=preflight_path,
        training_package_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        confirm_baseline_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert list(output_dir.glob("oled_baseline_predictions__*.jsonl"))
    assert list(output_dir.glob("oled_baseline_metrics__*.json"))
    assert all(result.prediction_sha256 for result in report.manifest.run_results if result.status == OledCuratedTrainingBaselineRunStatus.COMPLETED)


def test_cli_smoke_writes_compact_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, preflight_path = _write_package_files(tmp_path)
    output_dir = tmp_path / "baseline_run"
    output_manifest = tmp_path / "baseline_manifest.json"

    exit_code = main(
        [
            "--training-package-manifest",
            str(manifest_path),
            "--backend-preflight-report",
            str(preflight_path),
            "--training-package-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-baseline-run",
            "--baseline-kind",
            "mean_baseline",
            "--target-property-id",
            "eqe_percent",
            "--feature-view",
            "full_context",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output_manifest.exists()
    assert "predictions" not in stdout
    assert "features" not in stdout
    assert json.loads(stdout)["output_prediction_count"] == 4


def test_package_exports_for_baseline_runner(tmp_path: Path) -> None:
    manifest_path, preflight_path = _write_package_files(tmp_path)
    preflight = package_load_backend_preflight(preflight_path)
    predictions, metrics = package_run_mean(_valid_rows(), target_property_id="eqe_percent", feature_view="full_context")
    _, _, sklearn_findings = package_run_sklearn(
        _valid_rows(),
        baseline_kind=OledCuratedTrainingBaselineKind.TABULAR_RIDGE_SKLEARN,
        target_property_id="eqe_percent",
        feature_view="full_context",
    )
    selection_report = package_select(
        training_rows=_valid_rows(),
        schema=_schema(),
        backend_preflight_report=preflight,
        confirm_baseline_run=True,
    )
    output_dir = tmp_path / "package-export-output"
    runner_report = package_run_from_files(
        training_package_manifest_path=manifest_path,
        backend_preflight_report_path=preflight_path,
        training_package_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=tmp_path / "package-export-manifest.json",
        confirm_baseline_run=True,
    )
    pred_sha = package_write_predictions(predictions, tmp_path / package_predictions_filename(baseline_kind="mean_baseline", target_property_id="eqe_percent", feature_view="full_context"))
    metric_sha = package_write_metrics(metrics, tmp_path / package_metrics_filename(baseline_kind="mean_baseline", target_property_id="eqe_percent", feature_view="full_context"))
    package_write_manifest(runner_report.manifest, tmp_path / "package-manifest.json")

    assert isinstance(preflight, type(_backend_preflight()))
    assert isinstance(selection_report, OledCuratedTrainingBaselineRunnerReport)
    assert isinstance(selection_report.predictions[0], OledCuratedTrainingBaselinePrediction)
    assert isinstance(selection_report.metrics[0], OledCuratedTrainingBaselineMetrics)
    assert isinstance(selection_report.manifest.run_results[0], OledCuratedTrainingBaselineRunResult)
    assert isinstance(selection_report.manifest, OledCuratedTrainingBaselineRunnerManifest)
    assert isinstance(OledCuratedTrainingBaselineRunnerFinding(code="x", message="y"), OledCuratedTrainingBaselineRunnerFinding)
    assert isinstance(OledCuratedTrainingBaselineRunnerPolicy(), OledCuratedTrainingBaselineRunnerPolicy)
    assert OledCuratedTrainingBaselineKind.MEAN_BASELINE.value == "mean_baseline"
    assert OledCuratedTrainingBaselineRunStatus.COMPLETED.value == "completed"
    assert pred_sha
    assert metric_sha
    assert runner_report.is_valid
    assert isinstance(sklearn_findings, list)
