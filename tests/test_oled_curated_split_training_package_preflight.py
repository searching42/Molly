from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledCuratedSplitFeatureFileResult,
    OledCuratedSplitFeatureRowArtifact,
    OledCuratedSplitFeatureWriteStatus,
    OledCuratedSplitFeatureWriterManifest,
    OledCuratedSplitFeatureWriterPolicy,
    OledCuratedSplitTrainingPackagePreflightFinding,
    OledCuratedSplitTrainingPackagePreflightPolicy,
    OledCuratedSplitTrainingPackagePreflightReport,
    OledCuratedSplitTrainingPackagePreflightStatus,
    OledTrainingFeatureColumnKind,
    OledTrainingFeatureColumnSummary,
    OledTrainingSplitSummary,
    OledTrainingTargetSummary,
    load_oled_curated_split_feature_rows_from_manifest as package_load_rows_from_manifest,
    load_oled_curated_split_feature_writer_manifest_json as package_load_manifest,
    run_oled_curated_split_training_package_preflight as package_run_preflight,
    run_oled_curated_split_training_package_preflight_from_files as package_run_from_files,
    write_oled_curated_split_feature_manifest_json,
    write_oled_curated_split_feature_rows_jsonl,
    write_oled_curated_split_training_package_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_split_training_package_preflight import (
    load_oled_curated_split_feature_rows_from_manifest,
    load_oled_curated_split_feature_writer_manifest_json,
    main,
    run_oled_curated_split_training_package_preflight,
    run_oled_curated_split_training_package_preflight_from_files,
    write_oled_curated_split_training_package_preflight_report_json,
)


def _feature_row(
    row_id: str,
    *,
    split: str = "train",
    target_value: float | int | str | None = 21.0,
    target_unit: str | None = "%",
    target_property_id: str = "eqe_percent",
    feature_view: str = "full_context",
    features: dict | None = None,
    evidence_refs: list[str] | None = None,
    feature_row_id: str | None = None,
    record_id: str | None = None,
) -> OledCuratedSplitFeatureRowArtifact:
    clean_record_id = record_id or f"record-{row_id}"
    return OledCuratedSplitFeatureRowArtifact(
        feature_row_id=feature_row_id or f"feature-row-{row_id}",
        split=split,
        split_row_id=f"split-row-{row_id}",
        row_id=f"row-{row_id}",
        record_id=clean_record_id,
        source_record_ids=[clean_record_id],
        view_kind="raw_all_measurements",
        target_property_id=target_property_id,
        feature_view=feature_view,
        target_value=target_value,
        target_unit=target_unit,
        condition_hash=f"condition-{row_id}",
        confidence_score=0.9,
        evidence_refs=evidence_refs if evidence_refs is not None else [f"paper:test:{row_id}"],
        features=features
        if features is not None
        else {
            "numeric_feature": 1.0,
            "categorical_feature": "host-a",
            "boolean_feature": True,
            "sequence_feature": ["ITO", "EML", "Al"],
            "dict_feature": {"htl": "TAPC"},
        },
        missing_feature_columns=[],
        present_feature_columns=[],
        alignment_status="matched",
        alignment_reason_codes=["matched"],
        metadata={
            "split_feature_row_artifact": True,
            "ml_ready_training_data_record": False,
            "training_package_written": False,
        },
    )


def _valid_rows() -> list[OledCuratedSplitFeatureRowArtifact]:
    return [
        _feature_row("train", split="train", target_value=21.0),
        _feature_row("validation", split="validation", target_value=22.0),
        _feature_row("test", split="test", target_value=23.0),
    ]


def _manifest(rows_path: str, sha: str | None) -> OledCuratedSplitFeatureWriterManifest:
    return OledCuratedSplitFeatureWriterManifest(
        manifest_id="oled-curated-split-feature-writer:test",
        output_file_count=1,
        output_row_count=3,
        splits=["train", "validation", "test"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledCuratedSplitFeatureFileResult(
                split="train",
                target_property_id="eqe_percent",
                feature_view="full_context",
                status=OledCuratedSplitFeatureWriteStatus.WRITTEN,
                row_count=3,
                output_jsonl_path=rows_path,
                output_sha256=sha,
                reason_codes=["selected_for_write"],
            )
        ],
        policy=OledCuratedSplitFeatureWriterPolicy(),
        metadata={"split_feature_writer": True, "split_feature_rows_written": True},
    )


def _write_manifest(path: Path, manifest: OledCuratedSplitFeatureWriterManifest) -> Path:
    write_oled_curated_split_feature_manifest_json(manifest, path)
    return path


def test_main_preflight_success_summarizes_splits_targets_and_columns() -> None:
    report = run_oled_curated_split_training_package_preflight(feature_rows=_valid_rows())

    assert report.status == OledCuratedSplitTrainingPackagePreflightStatus.PASSED
    assert report.is_valid
    assert report.input_feature_row_count == 3
    assert report.rows_by_split == {"test": 1, "train": 1, "validation": 1}
    assert {summary.split for summary in report.split_summaries} == {"train", "validation", "test"}
    assert report.target_summaries[0].numeric_target_count == 3
    column_kinds = {summary.column_name: summary.kind for summary in report.feature_column_summaries}
    assert column_kinds["numeric_feature"] == OledTrainingFeatureColumnKind.NUMERIC
    assert column_kinds["categorical_feature"] == OledTrainingFeatureColumnKind.CATEGORICAL
    assert column_kinds["boolean_feature"] == OledTrainingFeatureColumnKind.BOOLEAN
    assert column_kinds["sequence_feature"] == OledTrainingFeatureColumnKind.SEQUENCE
    assert column_kinds["dict_feature"] == OledTrainingFeatureColumnKind.DICT
    assert report.metadata["training_package_preflight_only"] is True
    assert report.metadata["training_package_written"] is False


def test_duplicate_feature_row_id_fails() -> None:
    rows = _valid_rows()
    rows[1] = rows[1].model_copy(update={"feature_row_id": rows[0].feature_row_id})

    report = run_oled_curated_split_training_package_preflight(feature_rows=rows)

    assert report.status == OledCuratedSplitTrainingPackagePreflightStatus.FAILED
    assert "duplicate_feature_row_id" in report.error_codes


def test_missing_target_value_fails() -> None:
    rows = _valid_rows()
    rows[0] = rows[0].model_copy(update={"target_value": None})

    report = run_oled_curated_split_training_package_preflight(feature_rows=rows)

    assert report.status == OledCuratedSplitTrainingPackagePreflightStatus.FAILED
    assert "missing_target_value" in report.error_codes


def test_missing_evidence_refs_fail() -> None:
    rows = _valid_rows()
    rows[0] = rows[0].model_copy(update={"evidence_refs": []})

    report = run_oled_curated_split_training_package_preflight(feature_rows=rows)

    assert report.status == OledCuratedSplitTrainingPackagePreflightStatus.FAILED
    assert "missing_evidence_refs" in report.error_codes


def test_inconsistent_feature_columns_are_reported() -> None:
    rows = _valid_rows()
    rows[0] = rows[0].model_copy(update={"features": {"numeric_feature": 1.0}})

    report = run_oled_curated_split_training_package_preflight(feature_rows=rows)

    assert report.status == OledCuratedSplitTrainingPackagePreflightStatus.FAILED
    assert "inconsistent_feature_columns" in report.error_codes


def test_required_feature_columns_report_absent_and_missing_values() -> None:
    rows = _valid_rows()
    rows[0] = rows[0].model_copy(update={"features": {"numeric_feature": 1.0}})
    rows[1] = rows[1].model_copy(update={"features": {**rows[1].features, "required_column": None}})

    report = run_oled_curated_split_training_package_preflight(
        feature_rows=rows,
        policy=OledCuratedSplitTrainingPackagePreflightPolicy(
            required_feature_columns=["required_column"],
            fail_on_missing_required_features=True,
        ),
    )

    assert "required_feature_column_missing" in report.error_codes
    assert "required_feature_value_missing" in report.error_codes


def test_missing_optional_feature_values_warn_by_default() -> None:
    rows = _valid_rows()
    rows[0] = rows[0].model_copy(update={"features": {**rows[0].features, "optional_sparse": None}})

    report = run_oled_curated_split_training_package_preflight(feature_rows=rows)

    assert report.status == OledCuratedSplitTrainingPackagePreflightStatus.PASSED_WITH_WARNINGS
    assert "missing_optional_feature_values" in report.warning_codes


def test_split_checks_cover_missing_train_unknown_and_empty_splits() -> None:
    missing_train_report = run_oled_curated_split_training_package_preflight(
        feature_rows=[_feature_row("validation", split="validation"), _feature_row("test", split="test")]
    )
    unknown_split_report = run_oled_curated_split_training_package_preflight(
        feature_rows=[*_valid_rows(), _feature_row("holdout", split="holdout")]
    )
    empty_allowed_report = run_oled_curated_split_training_package_preflight(
        feature_rows=[_feature_row("train", split="train")],
        policy=OledCuratedSplitTrainingPackagePreflightPolicy(require_nonempty_splits=False),
    )

    assert "missing_train_split" in missing_train_report.error_codes
    assert "empty_split" in missing_train_report.error_codes
    assert "unknown_split" in unknown_split_report.error_codes
    assert "empty_split" not in empty_allowed_report.error_codes


def test_manifest_loader_and_rows_loader_verify_sha(tmp_path: Path) -> None:
    rows_path = tmp_path / "split_features.jsonl"
    rows_sha = write_oled_curated_split_feature_rows_jsonl(_valid_rows(), rows_path)
    manifest_path = _write_manifest(tmp_path / "split_feature_manifest.json", _manifest("split_features.jsonl", rows_sha))

    manifest = load_oled_curated_split_feature_writer_manifest_json(manifest_path)
    loaded_rows = load_oled_curated_split_feature_rows_from_manifest(manifest=manifest, base_dir=tmp_path)

    assert loaded_rows[0].feature_row_id == "feature-row-test"
    with pytest.raises(ValueError, match="split_feature_rows_sha256_mismatch:"):
        load_oled_curated_split_feature_rows_from_manifest(
            manifest=_manifest("split_features.jsonl", "not-the-sha"),
            base_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="missing_split_feature_writer_manifest:"):
        load_oled_curated_split_feature_writer_manifest_json(tmp_path / "missing.json")
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text("{bad-json}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_split_feature_writer_manifest_json:"):
        load_oled_curated_split_feature_writer_manifest_json(bad_manifest)


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    rows_path = tmp_path / "split_features.jsonl"
    rows_sha = write_oled_curated_split_feature_rows_jsonl(_valid_rows(), rows_path)
    manifest_path = _write_manifest(tmp_path / "split_feature_manifest.json", _manifest("split_features.jsonl", rows_sha))
    output_report = tmp_path / "training_preflight.json"

    report = run_oled_curated_split_training_package_preflight_from_files(
        split_feature_manifest_path=manifest_path,
        split_feature_base_dir=tmp_path,
        output_report_path=output_report,
    )

    assert report.is_valid
    assert output_report.exists()
    assert not list(tmp_path.glob("*training_package*.jsonl"))
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.parquet"))


def test_report_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_curated_split_training_package_preflight(feature_rows=_valid_rows())
    output_path = tmp_path / "training_preflight.json"

    write_oled_curated_split_training_package_preflight_report_json(report, output_path)
    text = output_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert str(tmp_path) not in text
    assert "raw paper text" not in text
    assert '"features"' not in text
    assert "feature_row_artifacts" not in text


def test_cli_smoke_writes_compact_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows_path = tmp_path / "split_features.jsonl"
    rows_sha = write_oled_curated_split_feature_rows_jsonl(_valid_rows(), rows_path)
    manifest_path = _write_manifest(tmp_path / "split_feature_manifest.json", _manifest("split_features.jsonl", rows_sha))
    output_report = tmp_path / "training_preflight.json"

    exit_code = main(
        [
            "--split-feature-manifest",
            str(manifest_path),
            "--split-feature-base-dir",
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
    assert "feature_column_summaries" not in stdout
    assert json.loads(stdout)["input_feature_row_count"] == 3


def test_package_exports_for_training_package_preflight(tmp_path: Path) -> None:
    rows_path = tmp_path / "package-split-features.jsonl"
    rows_sha = write_oled_curated_split_feature_rows_jsonl(_valid_rows(), rows_path)
    manifest_path = _write_manifest(tmp_path / "package-feature-manifest.json", _manifest("package-split-features.jsonl", rows_sha))
    manifest = package_load_manifest(manifest_path)
    rows = package_load_rows_from_manifest(manifest=manifest, base_dir=tmp_path)
    report = package_run_preflight(feature_rows=rows)
    output_report = tmp_path / "package-training-preflight.json"
    package_write_report(report, output_report)
    file_report = package_run_from_files(
        split_feature_manifest_path=manifest_path,
        split_feature_base_dir=tmp_path,
        output_report_path=tmp_path / "package-file-training-preflight.json",
    )

    assert isinstance(report, OledCuratedSplitTrainingPackagePreflightReport)
    assert isinstance(report.split_summaries[0], OledTrainingSplitSummary)
    assert isinstance(report.target_summaries[0], OledTrainingTargetSummary)
    assert isinstance(report.feature_column_summaries[0], OledTrainingFeatureColumnSummary)
    assert isinstance(
        OledCuratedSplitTrainingPackagePreflightFinding(code="x", message="y"),
        OledCuratedSplitTrainingPackagePreflightFinding,
    )
    assert OledCuratedSplitTrainingPackagePreflightStatus.PASSED.value == "passed"
    assert OledTrainingFeatureColumnKind.NUMERIC.value == "numeric"
    assert isinstance(OledCuratedSplitTrainingPackagePreflightPolicy(), OledCuratedSplitTrainingPackagePreflightPolicy)
    assert output_report.exists()
    assert file_report.is_valid
