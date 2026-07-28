from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledCuratedSplitTrainingPackageWriteStatus,
    OledCuratedSplitTrainingPackageWriterManifest,
    OledCuratedSplitTrainingPackageWriterPolicy,
    OledCuratedTrainingPackageFileResult,
    OledCuratedTrainingPackageRow,
    OledCuratedTrainingPackageSchema,
    OledTrainingBackendReadinessResult,
    OledTrainingBackendReadinessStatus,
    OledTrainingFeatureMatrixSummary,
    OledTrainingPackageBackendKind,
    OledTrainingPackageBackendPreflightFinding,
    OledTrainingPackageBackendPreflightPolicy,
    OledTrainingPackageBackendPreflightReport,
    OledTrainingPackageBackendPreflightStatus,
    flatten_oled_training_features_for_preflight as package_flatten_features,
    load_oled_training_package_schema_json as package_load_schema,
    load_oled_training_package_writer_manifest_json as package_load_manifest,
    load_oled_training_rows_from_manifest as package_load_rows_from_manifest,
    load_oled_training_schema_from_manifest as package_load_schema_from_manifest,
    run_oled_training_package_backend_preflight as package_run_preflight,
    run_oled_training_package_backend_preflight_from_files as package_run_from_files,
    write_oled_curated_training_package_manifest_json,
    write_oled_curated_training_package_schema_json,
    write_oled_curated_training_rows_jsonl,
    write_oled_training_package_backend_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_training_package_backend_preflight import (
    flatten_oled_training_features_for_preflight,
    load_oled_training_package_schema_json,
    load_oled_training_package_writer_manifest_json,
    load_oled_training_rows_from_manifest,
    load_oled_training_schema_from_manifest,
    main,
    run_oled_training_package_backend_preflight,
    run_oled_training_package_backend_preflight_from_files,
    write_oled_training_package_backend_preflight_report_json,
)


def _training_row(
    suffix: str,
    *,
    split: str = "train",
    target_value: float | int | str | None = 21.0,
    target_property_id: str = "eqe_percent",
    feature_view: str = "full_context",
    features: dict | None = None,
    evidence_refs: list[str] | None = None,
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
        target_value=target_value if target_value is not None else "",
        target_unit="%",
        feature_view=feature_view,
        features=features
        if features is not None
        else {
            "numeric_feature": 1.5,
            "boolean_feature": True,
            "categorical_feature": "host-a",
        },
        condition_hash=f"condition-{suffix}",
        confidence_score=0.9,
        evidence_refs=evidence_refs if evidence_refs is not None else [f"paper:test:{suffix}"],
        metadata={
            "ml_ready_training_row": True,
            "benchmark_validated": False,
            "model_backend_run": False,
        },
    )


def _valid_rows() -> list[OledCuratedTrainingPackageRow]:
    return [
        _training_row("train", split="train", target_value=21.0),
        _training_row("validation", split="validation", target_value=22.0),
        _training_row("test", split="test", target_value=23.0),
    ]


def _schema(
    *,
    feature_columns: list[str] | None = None,
    target_property_ids: list[str] | None = None,
    feature_views: list[str] | None = None,
    splits: list[str] | None = None,
) -> OledCuratedTrainingPackageSchema:
    columns = feature_columns or ["boolean_feature", "categorical_feature", "numeric_feature"]
    return OledCuratedTrainingPackageSchema(
        schema_id="oled-training-schema:test",
        target_property_ids=target_property_ids or ["eqe_percent"],
        feature_views=feature_views or ["full_context"],
        splits=splits or ["train", "validation", "test"],
        target_columns=["target_property_id", "target_value", "target_unit"],
        feature_columns=columns,
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
        feature_column_kinds={column: "numeric" for column in columns},
        required_columns=[],
        metadata={"training_package_schema": True, "benchmark_validated": False},
    )


def _manifest(
    *,
    rows_path: str = "training_rows.jsonl",
    rows_sha: str | None = "row-sha",
    schema_path: str = "oled_training_schema.json",
    schema_sha: str | None = "schema-sha",
) -> OledCuratedSplitTrainingPackageWriterManifest:
    return OledCuratedSplitTrainingPackageWriterManifest(
        manifest_id="oled-training-package-writer:test",
        source_split_feature_manifest_id="split-feature:test",
        source_training_preflight_status="passed",
        output_file_count=2,
        output_row_count=3,
        splits=["train", "validation", "test"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        rows_by_split={"train": 1, "validation": 1, "test": 1},
        rows_by_target={"eqe_percent": 3},
        rows_by_feature_view={"full_context": 3},
        file_results=[
            OledCuratedTrainingPackageFileResult(
                split="train",
                target_property_id="eqe_percent",
                feature_view="full_context",
                artifact_kind="training_rows",
                status=OledCuratedSplitTrainingPackageWriteStatus.WRITTEN,
                row_count=3,
                output_path=rows_path,
                output_sha256=rows_sha,
                reason_codes=["selected_for_write"],
            ),
            OledCuratedTrainingPackageFileResult(
                artifact_kind="schema",
                status=OledCuratedSplitTrainingPackageWriteStatus.WRITTEN,
                output_path=schema_path,
                output_sha256=schema_sha,
                reason_codes=["selected_for_write"],
            ),
        ],
        policy=OledCuratedSplitTrainingPackageWriterPolicy(),
        metadata={
            "training_package_writer": True,
            "training_package_written": True,
            "ml_ready_training_data_written": True,
            "benchmark_validated": False,
        },
    )


def _write_manifest(path: Path, manifest: OledCuratedSplitTrainingPackageWriterManifest) -> Path:
    write_oled_curated_training_package_manifest_json(manifest, path)
    return path


def _write_package_files(tmp_path: Path, *, rows: list[OledCuratedTrainingPackageRow] | None = None):
    package_rows = rows or _valid_rows()
    rows_path = tmp_path / "training_rows.jsonl"
    rows_sha = write_oled_curated_training_rows_jsonl(package_rows, rows_path)
    schema_path = tmp_path / "oled_training_schema.json"
    schema_sha = write_oled_curated_training_package_schema_json(_schema(), schema_path)
    manifest_path = _write_manifest(tmp_path / "training_manifest.json", _manifest(rows_sha=rows_sha, schema_sha=schema_sha))
    return manifest_path, rows_path, schema_path


def test_main_preflight_success_produces_backend_readiness() -> None:
    report = run_oled_training_package_backend_preflight(
        training_rows=_valid_rows(),
        schema=_schema(),
        policy=OledTrainingPackageBackendPreflightPolicy(
            target_property_ids=["eqe_percent"],
            feature_views=["full_context"],
        ),
    )

    assert report.is_valid
    assert report.status in {
        OledTrainingPackageBackendPreflightStatus.PASSED,
        OledTrainingPackageBackendPreflightStatus.PASSED_WITH_WARNINGS,
    }
    assert report.input_training_row_count == 3
    assert report.matrix_summaries[0].train_row_count == 1
    assert report.matrix_summaries[0].flattened_feature_column_count > 0
    assert {result.backend_kind for result in report.backend_results} == {
        OledTrainingPackageBackendKind.TABULAR_RIDGE_SKLEARN.value,
        OledTrainingPackageBackendKind.TABULAR_RANDOM_FOREST_SKLEARN.value,
    }
    assert all(result.status != OledTrainingBackendReadinessStatus.BLOCKED for result in report.backend_results)
    assert report.metadata["backend_preflight_only"] is True
    assert report.metadata["model_backends_run"] is False


def test_missing_train_split_fails() -> None:
    rows = [
        _training_row("validation", split="validation", target_value=22.0),
        _training_row("test", split="test", target_value=23.0),
    ]

    report = run_oled_training_package_backend_preflight(training_rows=rows, schema=_schema())

    assert not report.is_valid
    assert report.status == OledTrainingPackageBackendPreflightStatus.FAILED
    assert "missing_train_rows" in report.error_codes


def test_missing_eval_split_fails_when_required() -> None:
    rows = [_training_row("train", split="train", target_value=21.0)]

    report = run_oled_training_package_backend_preflight(training_rows=rows, schema=_schema())

    assert not report.is_valid
    assert "missing_eval_rows" in report.error_codes


def test_nonnumeric_target_blocks_backend_readiness() -> None:
    rows = [
        _training_row("train", split="train", target_value="high"),
        _training_row("validation", split="validation", target_value="medium"),
    ]

    report = run_oled_training_package_backend_preflight(training_rows=rows, schema=_schema())

    assert not report.is_valid
    assert "nonnumeric_target_value" in report.error_codes
    assert all(result.status == OledTrainingBackendReadinessStatus.BLOCKED for result in report.backend_results)
    assert "nonnumeric_targets_present" in report.backend_results[0].reason_codes


def test_empty_features_report_matrix_errors() -> None:
    rows = [
        _training_row("train", split="train", features={}),
        _training_row("validation", split="validation", features={}),
    ]

    report = run_oled_training_package_backend_preflight(training_rows=rows, schema=_schema())

    assert not report.is_valid
    assert "empty_features" in report.error_codes
    assert "empty_flattened_feature_matrix" in report.error_codes


def test_feature_flattening_is_deterministic() -> None:
    features = {
        "none": None,
        "empty": "",
        "empty_list": [],
        "numeric": 2,
        "flag": True,
        "label": "Host A",
        "sequence": [1, "Blue"],
        "nested": {"z": False, "a": 3.5},
    }

    flattened = flatten_oled_training_features_for_preflight(features)

    assert list(flattened) == sorted(flattened)
    assert flattened == {
        "flag": 1.0,
        "label=host a": 1.0,
        "nested.a": 3.5,
        "nested.z": 0.0,
        "numeric": 2.0,
        "sequence.0": 1.0,
        "sequence.1=blue": 1.0,
    }


def test_schema_mismatch_reports_finding() -> None:
    rows = [_training_row("train", split="train"), _training_row("validation", split="validation")]
    schema = _schema(feature_columns=["numeric_feature", "schema_only_column"])

    report = run_oled_training_package_backend_preflight(training_rows=rows, schema=schema)

    assert "schema_feature_column_mismatch" in report.warning_codes


def test_manifest_loader_accepts_valid_and_rejects_bad_input(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.json", _manifest())
    bad_path = tmp_path / "bad_manifest.json"
    bad_path.write_text("{bad-json}\n", encoding="utf-8")

    loaded = load_oled_training_package_writer_manifest_json(manifest_path)

    assert loaded.manifest_id == "oled-training-package-writer:test"
    with pytest.raises(ValueError, match="missing_training_package_manifest:"):
        load_oled_training_package_writer_manifest_json(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="invalid_training_package_manifest_json:"):
        load_oled_training_package_writer_manifest_json(bad_path)


def test_row_and_schema_loading_from_manifest_verifies_sha(tmp_path: Path) -> None:
    manifest_path, rows_path, schema_path = _write_package_files(tmp_path)
    manifest = load_oled_training_package_writer_manifest_json(manifest_path)

    loaded_rows = load_oled_training_rows_from_manifest(manifest=manifest, base_dir=tmp_path)
    loaded_schema = load_oled_training_schema_from_manifest(manifest=manifest, base_dir=tmp_path)

    assert {row.training_row_id for row in loaded_rows} == {row.training_row_id for row in _valid_rows()}
    assert loaded_schema is not None
    assert loaded_schema.schema_id == _schema().schema_id

    bad_rows_manifest = _manifest(rows_path=rows_path.name, rows_sha="wrong-sha")
    with pytest.raises(ValueError, match="training_rows_sha256_mismatch:"):
        load_oled_training_rows_from_manifest(manifest=bad_rows_manifest, base_dir=tmp_path)

    bad_schema_manifest = _manifest(rows_path=rows_path.name, rows_sha=None, schema_path=schema_path.name, schema_sha="wrong-sha")
    with pytest.raises(ValueError, match="training_schema_sha256_mismatch:"):
        load_oled_training_schema_from_manifest(manifest=bad_schema_manifest, base_dir=tmp_path)


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_package_files(tmp_path)
    output_report = tmp_path / "backend_preflight.json"

    report = run_oled_training_package_backend_preflight_from_files(
        training_package_manifest_path=manifest_path,
        training_package_base_dir=tmp_path,
        output_report_path=output_report,
        policy=OledTrainingPackageBackendPreflightPolicy(
            target_property_ids=["eqe_percent"],
            feature_views=["full_context"],
        ),
    )

    assert report.is_valid
    assert output_report.exists()
    assert not list(tmp_path.glob("*prediction*"))
    assert not list(tmp_path.glob("*model*"))


def test_report_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_training_package_backend_preflight(training_rows=_valid_rows(), schema=_schema())
    output_path = tmp_path / "report.json"

    write_oled_training_package_backend_preflight_report_json(report, output_path)
    first_text = output_path.read_text(encoding="utf-8")
    write_oled_training_package_backend_preflight_report_json(report, output_path)
    payload = json.loads(first_text)

    assert output_path.read_text(encoding="utf-8") == first_text
    assert first_text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert str(tmp_path) not in first_text
    assert "raw paper text" not in first_text
    assert '"features"' not in first_text


def test_cli_smoke_writes_compact_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, _, _ = _write_package_files(tmp_path)
    output_report = tmp_path / "backend_preflight.json"

    exit_code = main(
        [
            "--training-package-manifest",
            str(manifest_path),
            "--training-package-base-dir",
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
    assert "training_rows" not in stdout
    assert "features" not in stdout
    assert json.loads(stdout)["input_training_row_count"] == 3


def test_package_exports_for_backend_preflight(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_package_files(tmp_path)
    manifest = package_load_manifest(manifest_path)
    rows = package_load_rows_from_manifest(manifest=manifest, base_dir=tmp_path)
    schema = package_load_schema_from_manifest(manifest=manifest, base_dir=tmp_path)
    assert schema is not None
    loaded_schema = package_load_schema(tmp_path / "oled_training_schema.json")
    flattened = package_flatten_features({"x": 1, "name": "Host A"})
    report = package_run_preflight(training_rows=rows, schema=schema)
    runner_report = package_run_from_files(training_package_manifest_path=manifest_path, training_package_base_dir=tmp_path)
    report_path = tmp_path / "package-report.json"
    package_write_report(report, report_path)

    assert isinstance(report, OledTrainingPackageBackendPreflightReport)
    assert isinstance(report.matrix_summaries[0], OledTrainingFeatureMatrixSummary)
    assert isinstance(report.backend_results[0], OledTrainingBackendReadinessResult)
    assert isinstance(
        OledTrainingPackageBackendPreflightFinding(code="x", message="y"),
        OledTrainingPackageBackendPreflightFinding,
    )
    assert isinstance(OledTrainingPackageBackendPreflightPolicy(), OledTrainingPackageBackendPreflightPolicy)
    assert OledTrainingPackageBackendPreflightStatus.PASSED.value == "passed"
    assert OledTrainingBackendReadinessStatus.READY.value == "ready"
    assert OledTrainingPackageBackendKind.TABULAR_RIDGE_SKLEARN.value == "tabular_ridge_sklearn"
    assert loaded_schema.schema_id == schema.schema_id
    assert flattened["name=host a"] == 1.0
    assert runner_report.input_training_row_count == len(rows)
    assert report_path.exists()
