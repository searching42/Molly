from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledConfidenceAssessment,
    OledCausalLayer,
    OledConfounderFlags,
    OledCuratedDatasetSplitPreflightPolicy,
    OledCuratedSplitDatasetViewFileResult,
    OledCuratedSplitDatasetViewRowArtifact,
    OledCuratedSplitDatasetViewWriteStatus,
    OledCuratedSplitDatasetViewWriterManifest,
    OledCuratedSplitDatasetViewWriterPolicy,
    OledCuratedSplitFeatureFileResult,
    OledCuratedSplitFeaturePreflightPolicy,
    OledCuratedSplitFeaturePreflightStatus,
    OledCuratedSplitFeatureRowArtifact,
    OledCuratedSplitFeatureWriteStatus,
    OledCuratedSplitFeatureWriterFinding,
    OledCuratedSplitFeatureWriterManifest,
    OledCuratedSplitFeatureWriterPolicy,
    OledCuratedSplitFeatureWriterReport,
    OledDeviceLayer,
    OledEvidenceSource,
    OledEvidenceType,
    OledGoldDatasetRecord,
    OledInteractionLayer,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
    OledSplitFeatureRowAlignmentStatus,
    build_oled_curated_dataset_view_row_artifacts,
    build_oled_curated_split_dataset_view_row_artifacts,
    build_oled_curated_split_feature_row_artifacts as package_build_feature_rows,
    load_oled_curated_split_feature_preflight_report_json as package_load_preflight_report,
    load_oled_curated_split_feature_rows_jsonl as package_load_feature_rows,
    oled_split_feature_output_filename as package_output_filename,
    run_oled_curated_dataset_split_preflight,
    run_oled_curated_split_feature_preflight,
    run_oled_curated_split_feature_writer_from_files as package_run_from_files,
    select_oled_curated_split_feature_rows_for_write as package_select_for_write,
    write_oled_curated_split_dataset_view_manifest_json,
    write_oled_curated_split_dataset_view_rows_jsonl,
    write_oled_curated_split_feature_manifest_json as package_write_manifest,
    write_oled_curated_split_feature_preflight_report_json,
    write_oled_curated_split_feature_rows_jsonl as package_write_feature_rows,
)
from ai4s_agent.domains.oled_curated_split_feature_writer import (
    build_oled_curated_split_feature_row_artifacts,
    load_oled_curated_split_feature_preflight_report_json,
    load_oled_curated_split_feature_rows_jsonl,
    main,
    oled_split_feature_output_filename,
    run_oled_curated_split_feature_writer_from_files,
    select_oled_curated_split_feature_rows_for_write,
    write_oled_curated_split_feature_manifest_json,
    write_oled_curated_split_feature_rows_jsonl,
)
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind, build_oled_dataset_view


def _gold_record(
    record_id: str,
    *,
    value: float = 21.0,
    complete_features: bool = True,
    measurement_count: int = 1,
) -> OledGoldDatasetRecord:
    evidence_ref = f"paper-{record_id}:table-1:row-1"
    measurements: list[OledPropertyObservation] = []
    for index in range(measurement_count):
        measurements.append(
            OledPropertyObservation(
                property_label="EQE (%)",
                value=value + index,
                unit="%",
                condition=OledMeasurementCondition(
                    luminance_cd_m2=100 + index,
                    current_density_ma_cm2=5.0 + index,
                    voltage_v=4.2 + index if complete_features else None,
                    temperature_k=298.15,
                    atmosphere="N2" if complete_features else None,
                    condition_label=f"condition-{index}" if complete_features else None,
                ),
                evidence_sources=[
                    OledEvidenceSource(
                        source_id=f"{evidence_ref}:measurement-{index}",
                        source_type=OledEvidenceType.TABLE,
                        layer=OledCausalLayer.MEASUREMENT,
                    )
                ],
                confidence=OledConfidenceAssessment(score=0.91),
            )
        )
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles=f"C1=CC={record_id}",
                inchikey=f"FEATURE-WRITER-INCHIKEY-{record_id.upper()}",
            ),
            interaction=OledInteractionLayer(
                emitter_smiles=f"emitter-{record_id}",
                host_smiles=f"host-{record_id}" if complete_features else None,
                doping_ratio=0.08,
                film_type="doped" if complete_features else None,
                matrix_type="host_guest" if complete_features else None,
                aggregation_state="film" if complete_features else None,
            ),
            device=OledDeviceLayer(
                device_stack=["ITO", f"HTL-{record_id}", f"EML-{record_id}", "ETL", "Al"],
                etl_material="TPBi" if complete_features else None,
                htl_material="TAPC" if complete_features else None,
                fabrication_method="thermal_evaporation" if complete_features else None,
                outcoupling_structure="none" if complete_features else None,
                layer_thickness_nm={"HTL": 35.0, "EML": 20.0} if complete_features else {},
            ),
            measurement=OledMeasurementLayer(measurements=measurements),
            confounder_flags=OledConfounderFlags(is_device_optimized=True),
        ),
        evidence_refs=[evidence_ref],
        reviewer="reviewer-1",
        metadata={"curated_dataset_written": True, "training_data_written": False},
    )


def _records(count: int = 3) -> list[OledGoldDatasetRecord]:
    return [_gold_record(f"gold-feature-writer-{index}", value=21.0 + index) for index in range(count)]


def _dataset_rows(records: list[OledGoldDatasetRecord]):
    view_report = build_oled_dataset_view(
        records,
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
        target_property_id="eqe_percent",
    )
    return build_oled_curated_dataset_view_row_artifacts(view_report)


def _split_rows(records: list[OledGoldDatasetRecord]) -> list[OledCuratedSplitDatasetViewRowArtifact]:
    rows = _dataset_rows(records)
    split_preflight = run_oled_curated_dataset_split_preflight(
        gold_records=records,
        dataset_view_rows=rows,
        policy=OledCuratedDatasetSplitPreflightPolicy(allow_empty_split=True),
    )
    split_rows, findings = build_oled_curated_split_dataset_view_row_artifacts(
        rows,
        split_preflight_report=split_preflight,
    )
    assert not findings
    return split_rows


def _feature_preflight(records: list[OledGoldDatasetRecord], split_rows: list[OledCuratedSplitDatasetViewRowArtifact]):
    return run_oled_curated_split_feature_preflight(
        gold_records=records,
        split_rows=split_rows,
        policy=OledCuratedSplitFeaturePreflightPolicy(
            feature_views=["full_context"],
            target_property_ids=["eqe_percent"],
        ),
    )


def _write_gold_records(path: Path, records: list[OledGoldDatasetRecord]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record.model_dump(mode="json"), sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _split_manifest(rows_path: str, sha: str | None) -> OledCuratedSplitDatasetViewWriterManifest:
    return OledCuratedSplitDatasetViewWriterManifest(
        manifest_id="oled-split-dataset-view-writer:test",
        output_file_count=1,
        output_row_count=1,
        splits=["train"],
        view_kinds=["raw_all_measurements"],
        target_property_ids=["eqe_percent"],
        file_results=[
            OledCuratedSplitDatasetViewFileResult(
                split="train",
                view_kind="raw_all_measurements",
                target_property_id="eqe_percent",
                status=OledCuratedSplitDatasetViewWriteStatus.WRITTEN,
                row_count=1,
                output_jsonl_path=rows_path,
                output_sha256=sha,
                reason_codes=["selected_for_write"],
            )
        ],
        policy=OledCuratedSplitDatasetViewWriterPolicy(),
        metadata={"split_dataset_view_writer": True, "split_dataset_view_rows_written": True},
    )


def _write_split_manifest(path: Path, manifest: OledCuratedSplitDatasetViewWriterManifest) -> Path:
    write_oled_curated_split_dataset_view_manifest_json(manifest, path)
    return path


def test_confirmation_gate_requires_explicit_split_feature_write() -> None:
    records = _records(1)
    split_rows = _split_rows(records)

    with pytest.raises(ValueError, match="confirmation_required:split_feature_write"):
        select_oled_curated_split_feature_rows_for_write(
            gold_records=records,
            split_rows=split_rows,
            feature_preflight_report=_feature_preflight(records, split_rows),
        )


def test_build_feature_row_artifacts_preserves_matched_features_and_deterministic_id() -> None:
    records = _records(2)
    split_rows = _split_rows(records)
    preflight = _feature_preflight(records, split_rows)

    first, findings = build_oled_curated_split_feature_row_artifacts(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )
    second, _ = build_oled_curated_split_feature_row_artifacts(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )

    assert not findings
    assert first
    assert first[0].feature_row_id == second[0].feature_row_id
    assert first[0].split
    assert first[0].feature_view == "full_context"
    assert first[0].target_property_id == "eqe_percent"
    assert first[0].features
    assert first[0].alignment_status == OledSplitFeatureRowAlignmentStatus.MATCHED.value
    assert first[0].metadata["split_feature_row_artifact"] is True
    assert first[0].metadata["ml_ready_training_data_record"] is False


def test_missing_feature_row_is_rejected() -> None:
    records = _records(1)
    split_row = _split_rows(records)[0].model_copy(update={"record_id": "missing-record"})
    preflight = _feature_preflight(records, [split_row])

    feature_rows, findings = build_oled_curated_split_feature_row_artifacts(
        gold_records=records,
        split_rows=[split_row],
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )

    assert feature_rows == []
    assert [finding.code for finding in findings] == ["missing_feature_row_rejected"]


def test_ambiguous_feature_row_is_rejected() -> None:
    records = [_gold_record("gold-feature-writer-ambiguous", measurement_count=2)]
    split_row = _split_rows(records)[0].model_copy(update={"condition_hash": None})
    preflight = _feature_preflight(records, [split_row])

    feature_rows, findings = build_oled_curated_split_feature_row_artifacts(
        gold_records=records,
        split_rows=[split_row],
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )

    assert feature_rows == []
    assert [finding.code for finding in findings] == ["ambiguous_feature_row_rejected"]


def test_target_mismatch_is_rejected() -> None:
    records = _records(1)
    split_row = _split_rows(records)[0].model_copy(update={"target_value": 999.0})
    preflight = _feature_preflight(records, [split_row])

    feature_rows, findings = build_oled_curated_split_feature_row_artifacts(
        gold_records=records,
        split_rows=[split_row],
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )

    assert feature_rows == []
    assert [finding.code for finding in findings] == ["target_mismatch_rejected"]


def test_missing_feature_values_allowed_by_default() -> None:
    records = [_gold_record("gold-feature-writer-sparse", complete_features=False)]
    split_rows = _split_rows(records)
    preflight = _feature_preflight(records, split_rows)

    report = select_oled_curated_split_feature_rows_for_write(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
        confirm_split_feature_write=True,
    )

    assert report.is_valid
    assert report.feature_row_artifacts
    assert report.feature_row_artifacts[0].missing_feature_columns
    assert "missing_feature_values_allowed" in report.feature_row_artifacts[0].alignment_reason_codes


def test_missing_feature_values_rejected_by_policy() -> None:
    records = [_gold_record("gold-feature-writer-sparse", complete_features=False)]
    split_rows = _split_rows(records)
    preflight = _feature_preflight(records, split_rows)

    feature_rows, findings = build_oled_curated_split_feature_row_artifacts(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(
            feature_views=["full_context"],
            target_property_ids=["eqe_percent"],
            allow_missing_feature_values=False,
        ),
    )

    assert feature_rows == []
    assert [finding.code for finding in findings] == ["missing_feature_values_rejected"]


def test_invalid_preflight_blocks_writer() -> None:
    records = _records(1)
    split_rows = _split_rows(records)
    preflight = _feature_preflight(records, split_rows).model_copy(update={"status": "failed"})

    report = select_oled_curated_split_feature_rows_for_write(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
        confirm_split_feature_write=True,
    )

    assert not report.is_valid
    assert report.feature_row_artifacts == []
    assert "feature_preflight_failed" in report.error_codes


def test_jsonl_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    records = _records(1)
    split_rows = _split_rows(records)
    feature_rows, _ = build_oled_curated_split_feature_row_artifacts(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=_feature_preflight(records, split_rows),
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )
    output_path = tmp_path / "feature_rows.jsonl"

    first_hash = write_oled_curated_split_feature_rows_jsonl(feature_rows, output_path)
    first_text = output_path.read_text(encoding="utf-8")
    second_hash = write_oled_curated_split_feature_rows_jsonl(feature_rows, output_path)

    assert first_hash == second_hash
    assert first_text == output_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in first_text
    assert "raw paper text" not in first_text
    assert "layered_record" not in first_text


def test_manifest_writer_is_deterministic_and_has_safety_metadata(tmp_path: Path) -> None:
    manifest = OledCuratedSplitFeatureWriterManifest(
        manifest_id="oled-curated-split-feature-writer:test",
        output_file_count=1,
        output_row_count=2,
        splits=["train"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledCuratedSplitFeatureFileResult(
                split="train",
                target_property_id="eqe_percent",
                feature_view="full_context",
                status=OledCuratedSplitFeatureWriteStatus.WRITTEN,
                row_count=2,
                output_jsonl_path="oled_split_features__train__eqe_percent__full_context.jsonl",
                output_sha256="abc123",
                reason_codes=["selected_for_write"],
            )
        ],
        policy=OledCuratedSplitFeatureWriterPolicy(),
        metadata={
            "split_feature_writer": True,
            "split_feature_rows_written": True,
            "training_data_written": False,
            "ml_ready_training_data_written": False,
            "model_backends_run": False,
        },
    )
    output_path = tmp_path / "feature_manifest.json"

    write_oled_curated_split_feature_manifest_json(manifest, output_path)
    text = output_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert payload["metadata"]["split_feature_writer"] is True
    assert payload["metadata"]["ml_ready_training_data_written"] is False
    assert str(tmp_path) not in text


def test_combined_runner_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    records = _records(2)
    split_rows = _split_rows(records)
    curated_path = _write_gold_records(tmp_path / "curated_gold.jsonl", records)
    split_rows_path = tmp_path / "split_rows.jsonl"
    split_rows_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, split_rows_path)
    split_manifest_path = _write_split_manifest(tmp_path / "split_manifest.json", _split_manifest("split_rows.jsonl", split_rows_sha))
    preflight_path = tmp_path / "feature_preflight.json"
    write_oled_curated_split_feature_preflight_report_json(_feature_preflight(records, split_rows), preflight_path)
    output_manifest = tmp_path / "feature_writer_manifest.json"

    report = run_oled_curated_split_feature_writer_from_files(
        curated_gold_jsonl_path=curated_path,
        split_dataset_view_manifest_path=split_manifest_path,
        feature_preflight_report_path=preflight_path,
        split_dataset_view_base_dir=tmp_path,
        output_manifest_path=output_manifest,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert report.feature_row_artifacts
    assert not list(tmp_path.glob("oled_split_features__*.jsonl"))
    assert "dry_run_no_rows_written" in report.manifest.reason_code_counts


def test_combined_runner_write_mode_writes_split_feature_files(tmp_path: Path) -> None:
    records = _records(2)
    split_rows = _split_rows(records)
    curated_path = _write_gold_records(tmp_path / "curated_gold.jsonl", records)
    split_rows_path = tmp_path / "split_rows.jsonl"
    split_rows_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, split_rows_path)
    split_manifest_path = _write_split_manifest(tmp_path / "split_manifest.json", _split_manifest("split_rows.jsonl", split_rows_sha))
    preflight_path = tmp_path / "feature_preflight.json"
    write_oled_curated_split_feature_preflight_report_json(_feature_preflight(records, split_rows), preflight_path)
    output_dir = tmp_path / "split_features"
    output_manifest = tmp_path / "feature_writer_manifest.json"

    report = run_oled_curated_split_feature_writer_from_files(
        curated_gold_jsonl_path=curated_path,
        split_dataset_view_manifest_path=split_manifest_path,
        feature_preflight_report_path=preflight_path,
        split_dataset_view_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
        confirm_split_feature_write=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert report.manifest.output_file_count >= 1
    assert all(result.output_sha256 for result in report.manifest.file_results)
    assert list(output_dir.glob("oled_split_features__*.jsonl"))


def test_feature_rows_loader_accepts_valid_jsonl_and_rejects_bad_input(tmp_path: Path) -> None:
    records = _records(1)
    split_rows = _split_rows(records)
    feature_rows, _ = build_oled_curated_split_feature_row_artifacts(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=_feature_preflight(records, split_rows),
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )
    path = tmp_path / "feature_rows.jsonl"
    write_oled_curated_split_feature_rows_jsonl(feature_rows, path)
    bad_path = tmp_path / "bad_feature_rows.jsonl"
    bad_path.write_text("{bad-json}\n", encoding="utf-8")

    loaded = load_oled_curated_split_feature_rows_jsonl(path)

    assert loaded[0].feature_row_id == feature_rows[0].feature_row_id
    with pytest.raises(ValueError, match="invalid_split_feature_rows_jsonl:line-1"):
        load_oled_curated_split_feature_rows_jsonl(bad_path)
    with pytest.raises(ValueError, match="missing_split_feature_rows_jsonl:"):
        load_oled_curated_split_feature_rows_jsonl(tmp_path / "missing.jsonl")


def test_cli_smoke_writes_compact_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records = _records(2)
    split_rows = _split_rows(records)
    curated_path = _write_gold_records(tmp_path / "curated_gold.jsonl", records)
    split_rows_path = tmp_path / "split_rows.jsonl"
    split_rows_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, split_rows_path)
    split_manifest_path = _write_split_manifest(tmp_path / "split_manifest.json", _split_manifest("split_rows.jsonl", split_rows_sha))
    preflight_path = tmp_path / "feature_preflight.json"
    write_oled_curated_split_feature_preflight_report_json(_feature_preflight(records, split_rows), preflight_path)
    output_dir = tmp_path / "split_features"
    output_manifest = tmp_path / "feature_manifest.json"

    exit_code = main(
        [
            "--curated-gold-jsonl",
            str(curated_path),
            "--split-dataset-view-manifest",
            str(split_manifest_path),
            "--feature-preflight-report",
            str(preflight_path),
            "--split-dataset-view-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-split-feature-write",
            "--feature-view",
            "full_context",
            "--target-property-id",
            "eqe_percent",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output_manifest.exists()
    assert list(output_dir.glob("oled_split_features__*.jsonl"))
    assert "feature_row_artifacts" not in stdout
    assert json.loads(stdout)["output_row_count"] == len(split_rows)


def test_package_exports_for_split_feature_writer(tmp_path: Path) -> None:
    records = _records(1)
    split_rows = _split_rows(records)
    preflight = _feature_preflight(records, split_rows)
    feature_rows, findings = package_build_feature_rows(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )
    rows_path = tmp_path / "package-feature-rows.jsonl"
    rows_sha = package_write_feature_rows(feature_rows, rows_path)
    loaded_rows = package_load_feature_rows(rows_path)
    manifest = OledCuratedSplitFeatureWriterManifest(
        manifest_id="oled-curated-split-feature-writer:test",
        output_file_count=1,
        output_row_count=len(feature_rows),
        splits=sorted({row.split for row in feature_rows}),
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledCuratedSplitFeatureFileResult(
                split=feature_rows[0].split,
                target_property_id="eqe_percent",
                feature_view="full_context",
                status=OledCuratedSplitFeatureWriteStatus.WRITTEN,
                row_count=len(feature_rows),
                output_jsonl_path=package_output_filename(
                    split=feature_rows[0].split,
                    target_property_id="eqe_percent",
                    feature_view="full_context",
                ),
                output_sha256=rows_sha,
                reason_codes=["selected_for_write"],
            )
        ],
        policy=OledCuratedSplitFeatureWriterPolicy(),
        metadata={"split_feature_writer": True},
    )
    package_write_manifest(manifest, tmp_path / "package-feature-manifest.json")
    selection_report = package_select_for_write(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=preflight,
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
        confirm_split_feature_write=True,
    )
    preflight_path = tmp_path / "package-feature-preflight.json"
    write_oled_curated_split_feature_preflight_report_json(preflight, preflight_path)
    loaded_preflight = package_load_preflight_report(preflight_path)
    curated_path = _write_gold_records(tmp_path / "package-curated.jsonl", records)
    split_rows_path = tmp_path / "package-split-rows.jsonl"
    split_rows_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, split_rows_path)
    split_manifest_path = _write_split_manifest(tmp_path / "package-split-manifest.json", _split_manifest("package-split-rows.jsonl", split_rows_sha))
    runner_report = package_run_from_files(
        curated_gold_jsonl_path=curated_path,
        split_dataset_view_manifest_path=split_manifest_path,
        feature_preflight_report_path=preflight_path,
        split_dataset_view_base_dir=tmp_path,
        output_manifest_path=tmp_path / "package-runner-manifest.json",
        policy=OledCuratedSplitFeatureWriterPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
        dry_run=True,
    )

    assert not findings
    assert isinstance(feature_rows[0], OledCuratedSplitFeatureRowArtifact)
    assert isinstance(OledCuratedSplitFeatureWriterPolicy(), OledCuratedSplitFeatureWriterPolicy)
    assert isinstance(
        OledCuratedSplitFeatureWriterFinding(code="x", message="y"),
        OledCuratedSplitFeatureWriterFinding,
    )
    assert isinstance(selection_report, OledCuratedSplitFeatureWriterReport)
    assert isinstance(manifest.file_results[0], OledCuratedSplitFeatureFileResult)
    assert OledCuratedSplitFeatureWriteStatus.WRITTEN.value == "written"
    assert loaded_rows[0].feature_row_id == feature_rows[0].feature_row_id
    assert loaded_preflight.status == OledCuratedSplitFeaturePreflightStatus.PASSED
    assert runner_report.is_valid
