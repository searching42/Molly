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
    OledCuratedSplitTrainingPackagePreflightPolicy,
    OledCuratedSplitTrainingPackageWriteStatus,
    OledCuratedSplitTrainingPackageWriterFinding,
    OledCuratedSplitTrainingPackageWriterManifest,
    OledCuratedSplitTrainingPackageWriterPolicy,
    OledCuratedSplitTrainingPackageWriterReport,
    OledCuratedTrainingPackageFileResult,
    OledCuratedTrainingPackageRow,
    OledCuratedTrainingPackageSchema,
    build_oled_curated_training_package_rows as package_build_rows,
    build_oled_curated_training_package_schema as package_build_schema,
    load_oled_curated_training_rows_jsonl as package_load_rows,
    load_oled_curated_split_training_package_preflight_report_json as package_load_preflight,
    oled_training_rows_output_filename as package_rows_filename,
    oled_training_schema_output_filename as package_schema_filename,
    run_oled_curated_split_training_package_preflight,
    run_oled_curated_split_training_package_writer_from_files as package_run_from_files,
    select_oled_curated_split_training_package_for_write as package_select_for_write,
    write_oled_curated_split_feature_manifest_json,
    write_oled_curated_split_feature_rows_jsonl,
    write_oled_curated_split_training_package_preflight_report_json,
    write_oled_curated_training_package_manifest_json as package_write_manifest,
    write_oled_curated_training_package_schema_json as package_write_schema,
    write_oled_curated_training_rows_jsonl as package_write_rows,
)
from ai4s_agent.domains.oled_curated_split_training_package_writer import (
    build_oled_curated_training_package_rows,
    build_oled_curated_training_package_schema,
    load_oled_curated_split_training_package_preflight_report_json,
    load_oled_curated_training_rows_jsonl,
    main,
    oled_training_rows_output_filename,
    oled_training_schema_output_filename,
    run_oled_curated_split_training_package_writer_from_files,
    select_oled_curated_split_training_package_for_write,
    write_oled_curated_training_package_manifest_json,
    write_oled_curated_training_package_schema_json,
    write_oled_curated_training_rows_jsonl,
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
    metadata: dict | None = None,
) -> OledCuratedSplitFeatureRowArtifact:
    return OledCuratedSplitFeatureRowArtifact(
        feature_row_id=f"feature-row-{row_id}",
        split=split,
        split_row_id=f"split-row-{row_id}",
        row_id=f"row-{row_id}",
        record_id=f"record-{row_id}",
        source_record_ids=[f"record-{row_id}"],
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
        },
        missing_feature_columns=[],
        present_feature_columns=[],
        alignment_status="matched",
        alignment_reason_codes=["matched"],
        metadata=metadata
        if metadata is not None
        else {
            "split_feature_row_artifact": True,
            "ml_ready_training_data_record": False,
            "training_package_written": False,
            "benchmark_validated": False,
        },
    )


def _valid_feature_rows() -> list[OledCuratedSplitFeatureRowArtifact]:
    return [
        _feature_row("train", split="train", target_value=21.0),
        _feature_row("validation", split="validation", target_value=22.0),
        _feature_row("test", split="test", target_value=23.0),
    ]


def _preflight(rows: list[OledCuratedSplitFeatureRowArtifact] | None = None):
    return run_oled_curated_split_training_package_preflight(
        feature_rows=rows or _valid_feature_rows(),
        policy=OledCuratedSplitTrainingPackagePreflightPolicy(
            target_property_ids=["eqe_percent"],
            feature_views=["full_context"],
        ),
    )


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


def _write_feature_manifest(path: Path, manifest: OledCuratedSplitFeatureWriterManifest) -> Path:
    write_oled_curated_split_feature_manifest_json(manifest, path)
    return path


def test_confirmation_gate_requires_explicit_training_package_write() -> None:
    with pytest.raises(ValueError, match="confirmation_required:training_package_write"):
        select_oled_curated_split_training_package_for_write(
            feature_rows=_valid_feature_rows(),
            preflight_report=_preflight(),
        )


def test_build_training_rows_preserves_values_and_deterministic_id() -> None:
    feature_rows = _valid_feature_rows()
    first, findings = build_oled_curated_training_package_rows(
        feature_rows,
        preflight_report=_preflight(feature_rows),
    )
    second, _ = build_oled_curated_training_package_rows(
        feature_rows,
        preflight_report=_preflight(feature_rows),
    )

    assert not findings
    assert len(first) == 3
    assert first[0].training_row_id == second[0].training_row_id
    assert first[0].split in {"train", "validation", "test"}
    assert first[0].target_property_id == "eqe_percent"
    assert first[0].feature_view == "full_context"
    assert first[0].features["numeric_feature"] == 1.0
    assert first[0].metadata["ml_ready_training_row"] is True
    assert first[0].metadata["benchmark_validated"] is False
    assert first[0].metadata["model_backend_run"] is False


def test_invalid_preflight_blocks_writer() -> None:
    report = select_oled_curated_split_training_package_for_write(
        feature_rows=_valid_feature_rows(),
        preflight_report=_preflight().model_copy(update={"status": "failed"}),
        confirm_training_package_write=True,
    )

    assert not report.is_valid
    assert report.training_rows == []
    assert "training_package_preflight_failed" in report.error_codes


def test_missing_target_rejected() -> None:
    bad_row = _feature_row("bad-target", target_value=None)

    rows, findings = build_oled_curated_training_package_rows(
        [bad_row],
        preflight_report=_preflight(),
    )

    assert rows == []
    assert [finding.code for finding in findings] == ["missing_target_rejected"]


def test_missing_evidence_rejected() -> None:
    bad_row = _feature_row("bad-evidence", evidence_refs=[])

    rows, findings = build_oled_curated_training_package_rows(
        [bad_row],
        preflight_report=_preflight(),
    )

    assert rows == []
    assert [finding.code for finding in findings] == ["missing_evidence_rejected"]


def test_empty_features_rejected() -> None:
    bad_row = _feature_row("empty-features", features={})

    rows, findings = build_oled_curated_training_package_rows(
        [bad_row],
        preflight_report=_preflight(),
    )

    assert rows == []
    assert [finding.code for finding in findings] == ["empty_features_rejected"]


def test_source_claims_benchmark_validated_rejected() -> None:
    bad_row = _feature_row("benchmark", metadata={"benchmark_validated": True})

    rows, findings = build_oled_curated_training_package_rows(
        [bad_row],
        preflight_report=_preflight(),
    )

    assert rows == []
    assert [finding.code for finding in findings] == ["source_claims_benchmark_validated"]


def test_schema_builder_derives_columns_and_kinds() -> None:
    training_rows, _ = build_oled_curated_training_package_rows(
        _valid_feature_rows(),
        preflight_report=_preflight(),
    )

    schema = build_oled_curated_training_package_schema(training_rows, preflight_report=_preflight())

    assert schema.target_columns == ["target_property_id", "target_value", "target_unit"]
    assert "training_row_id" in schema.metadata_columns
    assert "evidence_refs" in schema.metadata_columns
    assert "numeric_feature" in schema.feature_columns
    assert schema.feature_column_kinds["numeric_feature"] == "numeric"
    assert schema.feature_column_kinds["categorical_feature"] == "categorical"
    assert schema.required_columns == []
    assert schema.metadata["benchmark_validated"] is False


def test_jsonl_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    training_rows, _ = build_oled_curated_training_package_rows(
        _valid_feature_rows(),
        preflight_report=_preflight(),
    )
    output_path = tmp_path / "training_rows.jsonl"

    first_sha = write_oled_curated_training_rows_jsonl(training_rows, output_path)
    first_text = output_path.read_text(encoding="utf-8")
    second_sha = write_oled_curated_training_rows_jsonl(training_rows, output_path)

    assert first_sha == second_sha
    assert first_text == output_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in first_text
    assert "raw paper text" not in first_text
    assert "gold_record" not in first_text


def test_schema_writer_is_deterministic(tmp_path: Path) -> None:
    training_rows, _ = build_oled_curated_training_package_rows(_valid_feature_rows(), preflight_report=_preflight())
    schema = build_oled_curated_training_package_schema(training_rows, preflight_report=_preflight())
    output_path = tmp_path / "schema.json"

    first_sha = write_oled_curated_training_package_schema_json(schema, output_path)
    text = output_path.read_text(encoding="utf-8")
    second_sha = write_oled_curated_training_package_schema_json(schema, output_path)

    assert first_sha == second_sha
    assert text == output_path.read_text(encoding="utf-8")
    assert json.loads(text)["metadata"]["benchmark_validated"] is False


def test_manifest_writer_is_deterministic_and_has_safety_metadata(tmp_path: Path) -> None:
    manifest = OledCuratedSplitTrainingPackageWriterManifest(
        manifest_id="oled-training-package-writer:test",
        output_file_count=2,
        output_row_count=3,
        splits=["train"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        rows_by_split={"train": 3},
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
                output_path="oled_training_rows__train__eqe_percent__full_context.jsonl",
                output_sha256="abc123",
                reason_codes=["selected_for_write"],
            ),
            OledCuratedTrainingPackageFileResult(
                artifact_kind="schema",
                status=OledCuratedSplitTrainingPackageWriteStatus.WRITTEN,
                output_path="oled_training_schema.json",
                output_sha256="def456",
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
    output_path = tmp_path / "manifest.json"

    write_oled_curated_training_package_manifest_json(manifest, output_path)
    text = output_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert payload["metadata"]["training_package_writer"] is True
    assert payload["metadata"]["benchmark_validated"] is False
    assert payload["file_results"][0]["output_sha256"] == "abc123"


def test_combined_runner_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    feature_path = tmp_path / "split_features.jsonl"
    feature_sha = write_oled_curated_split_feature_rows_jsonl(_valid_feature_rows(), feature_path)
    feature_manifest_path = _write_feature_manifest(tmp_path / "split_feature_manifest.json", _manifest("split_features.jsonl", feature_sha))
    preflight_path = tmp_path / "training_preflight.json"
    write_oled_curated_split_training_package_preflight_report_json(_preflight(), preflight_path)
    output_manifest = tmp_path / "training_manifest.json"

    report = run_oled_curated_split_training_package_writer_from_files(
        split_feature_manifest_path=feature_manifest_path,
        training_preflight_report_path=preflight_path,
        split_feature_base_dir=tmp_path,
        output_manifest_path=output_manifest,
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert report.training_rows
    assert "dry_run_no_files_written" in report.manifest.file_results[0].reason_codes
    assert not list(tmp_path.glob("oled_training_rows__*.jsonl"))
    assert not (tmp_path / "oled_training_schema.json").exists()


def test_combined_runner_write_mode_writes_rows_schema_and_manifest(tmp_path: Path) -> None:
    feature_path = tmp_path / "split_features.jsonl"
    feature_sha = write_oled_curated_split_feature_rows_jsonl(_valid_feature_rows(), feature_path)
    feature_manifest_path = _write_feature_manifest(tmp_path / "split_feature_manifest.json", _manifest("split_features.jsonl", feature_sha))
    preflight_path = tmp_path / "training_preflight.json"
    write_oled_curated_split_training_package_preflight_report_json(_preflight(), preflight_path)
    output_dir = tmp_path / "training_package"
    output_manifest = tmp_path / "training_manifest.json"

    report = run_oled_curated_split_training_package_writer_from_files(
        split_feature_manifest_path=feature_manifest_path,
        training_preflight_report_path=preflight_path,
        split_feature_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        confirm_training_package_write=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert list(output_dir.glob("oled_training_rows__*.jsonl"))
    assert (output_dir / "oled_training_schema.json").exists()
    assert all(result.output_sha256 for result in report.manifest.file_results)


def test_training_rows_loader_accepts_valid_jsonl_and_rejects_bad_input(tmp_path: Path) -> None:
    training_rows, _ = build_oled_curated_training_package_rows(_valid_feature_rows(), preflight_report=_preflight())
    output_path = tmp_path / "training_rows.jsonl"
    write_oled_curated_training_rows_jsonl(training_rows, output_path)
    bad_path = tmp_path / "bad_training_rows.jsonl"
    bad_path.write_text("{bad-json}\n", encoding="utf-8")

    loaded = load_oled_curated_training_rows_jsonl(output_path)

    assert loaded[0].training_row_id == training_rows[0].training_row_id
    with pytest.raises(ValueError, match="invalid_training_rows_jsonl:line-1"):
        load_oled_curated_training_rows_jsonl(bad_path)
    with pytest.raises(ValueError, match="missing_training_rows_jsonl:"):
        load_oled_curated_training_rows_jsonl(tmp_path / "missing.jsonl")


def test_cli_smoke_writes_compact_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    feature_path = tmp_path / "split_features.jsonl"
    feature_sha = write_oled_curated_split_feature_rows_jsonl(_valid_feature_rows(), feature_path)
    feature_manifest_path = _write_feature_manifest(tmp_path / "split_feature_manifest.json", _manifest("split_features.jsonl", feature_sha))
    preflight_path = tmp_path / "training_preflight.json"
    write_oled_curated_split_training_package_preflight_report_json(_preflight(), preflight_path)
    output_dir = tmp_path / "training_package"
    output_manifest = tmp_path / "training_manifest.json"

    exit_code = main(
        [
            "--split-feature-manifest",
            str(feature_manifest_path),
            "--training-preflight-report",
            str(preflight_path),
            "--split-feature-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-training-package-write",
            "--target-property-id",
            "eqe_percent",
            "--feature-view",
            "full_context",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output_manifest.exists()
    assert "training_rows" not in stdout
    assert json.loads(stdout)["output_row_count"] == 3


def test_package_exports_for_training_package_writer(tmp_path: Path) -> None:
    training_rows, findings = package_build_rows(_valid_feature_rows(), preflight_report=_preflight())
    schema = package_build_schema(training_rows, preflight_report=_preflight())
    rows_path = tmp_path / "package-training-rows.jsonl"
    rows_sha = package_write_rows(training_rows, rows_path)
    loaded_rows = package_load_rows(rows_path)
    schema_path = tmp_path / "package-schema.json"
    schema_sha = package_write_schema(schema, schema_path)
    manifest = OledCuratedSplitTrainingPackageWriterManifest(
        manifest_id="oled-training-package-writer:test",
        output_file_count=2,
        output_row_count=len(training_rows),
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
                output_path=package_rows_filename(split="train", target_property_id="eqe_percent", feature_view="full_context"),
                output_sha256=rows_sha,
                reason_codes=["selected_for_write"],
            ),
            OledCuratedTrainingPackageFileResult(
                artifact_kind="schema",
                status=OledCuratedSplitTrainingPackageWriteStatus.WRITTEN,
                output_path=package_schema_filename(),
                output_sha256=schema_sha,
                reason_codes=["selected_for_write"],
            ),
        ],
        policy=OledCuratedSplitTrainingPackageWriterPolicy(),
        metadata={"training_package_writer": True},
    )
    package_write_manifest(manifest, tmp_path / "package-manifest.json")
    preflight_path = tmp_path / "package-preflight.json"
    write_oled_curated_split_training_package_preflight_report_json(_preflight(), preflight_path)
    loaded_preflight = package_load_preflight(preflight_path)
    feature_path = tmp_path / "package-split-features.jsonl"
    feature_sha = write_oled_curated_split_feature_rows_jsonl(_valid_feature_rows(), feature_path)
    feature_manifest_path = _write_feature_manifest(tmp_path / "package-feature-manifest.json", _manifest("package-split-features.jsonl", feature_sha))
    runner_report = package_run_from_files(
        split_feature_manifest_path=feature_manifest_path,
        training_preflight_report_path=preflight_path,
        split_feature_base_dir=tmp_path,
        output_manifest_path=tmp_path / "package-runner-manifest.json",
        dry_run=True,
    )
    selection_report = package_select_for_write(
        feature_rows=_valid_feature_rows(),
        preflight_report=_preflight(),
        confirm_training_package_write=True,
    )

    assert not findings
    assert isinstance(training_rows[0], OledCuratedTrainingPackageRow)
    assert isinstance(schema, OledCuratedTrainingPackageSchema)
    assert isinstance(selection_report, OledCuratedSplitTrainingPackageWriterReport)
    assert isinstance(manifest.file_results[0], OledCuratedTrainingPackageFileResult)
    assert isinstance(
        OledCuratedSplitTrainingPackageWriterFinding(code="x", message="y"),
        OledCuratedSplitTrainingPackageWriterFinding,
    )
    assert OledCuratedSplitTrainingPackageWriteStatus.WRITTEN.value == "written"
    assert isinstance(OledCuratedSplitTrainingPackageWriterPolicy(), OledCuratedSplitTrainingPackageWriterPolicy)
    assert loaded_rows[0].training_row_id == training_rows[0].training_row_id
    assert loaded_preflight.is_valid
    assert runner_report.is_valid
