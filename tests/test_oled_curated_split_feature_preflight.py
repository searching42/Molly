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
    OledCuratedSplitDatasetViewWriteStatus,
    OledCuratedSplitDatasetViewWriterManifest,
    OledCuratedSplitDatasetViewWriterPolicy,
    OledCuratedSplitFeaturePreflightFinding,
    OledCuratedSplitFeaturePreflightPolicy,
    OledCuratedSplitFeaturePreflightReport,
    OledCuratedSplitFeaturePreflightStatus,
    OledCuratedSplitDatasetViewRowArtifact,
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
    OledSplitFeaturePreflightSummary,
    OledSplitFeatureRowAlignment,
    OledSplitFeatureRowAlignmentStatus,
    build_oled_curated_dataset_view_row_artifacts,
    build_oled_curated_split_dataset_view_row_artifacts,
    load_oled_curated_split_dataset_view_rows_from_manifest as package_load_split_rows_from_manifest,
    load_oled_curated_split_dataset_view_writer_manifest_json as package_load_split_manifest,
    run_oled_curated_dataset_split_preflight,
    run_oled_curated_split_feature_preflight as package_run_preflight,
    run_oled_curated_split_feature_preflight_from_files as package_run_from_files,
    write_oled_curated_split_dataset_view_rows_jsonl,
    write_oled_curated_split_feature_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_split_feature_preflight import (
    load_oled_curated_split_dataset_view_rows_from_manifest,
    load_oled_curated_split_dataset_view_writer_manifest_json,
    main,
    run_oled_curated_split_feature_preflight,
    run_oled_curated_split_feature_preflight_from_files,
    write_oled_curated_split_feature_preflight_report_json,
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
                inchikey=f"FEATURE-INCHIKEY-{record_id.upper()}",
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
    return [_gold_record(f"gold-feature-{index}", value=21.0 + index) for index in range(count)]


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
    path.write_text(json.dumps(manifest.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    return path


def test_main_preflight_success_aligns_full_context_rows() -> None:
    records = _records(3)
    split_rows = _split_rows(records)

    report = run_oled_curated_split_feature_preflight(
        gold_records=records,
        split_rows=split_rows,
        policy=OledCuratedSplitFeaturePreflightPolicy(
            feature_views=["full_context"],
            target_property_ids=["eqe_percent"],
        ),
    )

    assert report.status == OledCuratedSplitFeaturePreflightStatus.PASSED
    assert report.is_valid
    assert report.row_alignments
    assert {alignment.status for alignment in report.row_alignments} == {OledSplitFeatureRowAlignmentStatus.MATCHED}
    assert report.metadata["feature_preflight_only"] is True
    assert report.metadata["feature_tables_written"] is False
    assert report.metadata["training_data_written"] is False


def test_gold_validation_failure_reports_errors() -> None:
    invalid_record = _gold_record("gold-invalid").model_copy(update={"evidence_refs": []})

    report = run_oled_curated_split_feature_preflight(
        gold_records=[invalid_record],
        split_rows=[],
    )

    assert report.status == OledCuratedSplitFeaturePreflightStatus.FAILED
    assert "gold_missing_evidence_refs" in report.gold_validation_error_codes
    assert "gold_validation_errors_present" in report.error_codes


def test_missing_feature_row_is_reported_for_unknown_record() -> None:
    records = _records(1)
    split_row = _split_rows(records)[0].model_copy(update={"record_id": "missing-record"})

    report = run_oled_curated_split_feature_preflight(
        gold_records=records,
        split_rows=[split_row],
        policy=OledCuratedSplitFeaturePreflightPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )

    assert report.status == OledCuratedSplitFeaturePreflightStatus.FAILED
    assert report.row_alignments[0].status == OledSplitFeatureRowAlignmentStatus.MISSING_FEATURE_ROW
    assert "missing_feature_row" in report.error_codes


def test_ambiguous_feature_row_when_condition_hash_cannot_disambiguate() -> None:
    records = [_gold_record("gold-ambiguous", measurement_count=2)]
    split_row = _split_rows(records)[0].model_copy(update={"condition_hash": None})

    report = run_oled_curated_split_feature_preflight(
        gold_records=records,
        split_rows=[split_row],
        policy=OledCuratedSplitFeaturePreflightPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )

    assert report.status == OledCuratedSplitFeaturePreflightStatus.FAILED
    assert report.row_alignments[0].status == OledSplitFeatureRowAlignmentStatus.AMBIGUOUS_FEATURE_ROW
    assert "ambiguous_feature_row" in report.error_codes


def test_target_mismatch_is_detected() -> None:
    records = _records(1)
    split_row = _split_rows(records)[0].model_copy(update={"target_value": 999.0})

    report = run_oled_curated_split_feature_preflight(
        gold_records=records,
        split_rows=[split_row],
        policy=OledCuratedSplitFeaturePreflightPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )

    assert report.status == OledCuratedSplitFeaturePreflightStatus.FAILED
    assert report.row_alignments[0].status == OledSplitFeatureRowAlignmentStatus.TARGET_MISMATCH
    assert "target_mismatch" in report.error_codes


def test_missing_feature_values_warn_by_default_and_can_error() -> None:
    records = [_gold_record("gold-sparse", complete_features=False)]
    split_rows = _split_rows(records)

    warning_report = run_oled_curated_split_feature_preflight(
        gold_records=records,
        split_rows=split_rows,
        policy=OledCuratedSplitFeaturePreflightPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )
    error_report = run_oled_curated_split_feature_preflight(
        gold_records=records,
        split_rows=split_rows,
        policy=OledCuratedSplitFeaturePreflightPolicy(
            feature_views=["full_context"],
            target_property_ids=["eqe_percent"],
            fail_on_missing_features=True,
        ),
    )

    assert "missing_feature_values" in warning_report.warning_codes
    assert warning_report.status == OledCuratedSplitFeaturePreflightStatus.PASSED_WITH_WARNINGS
    assert "missing_feature_values" in error_report.error_codes
    assert error_report.status == OledCuratedSplitFeaturePreflightStatus.FAILED


def test_manifest_loader_and_rows_loader_verify_sha(tmp_path: Path) -> None:
    records = _records(1)
    split_rows = _split_rows(records)
    rows_path = tmp_path / "split_rows.jsonl"
    rows_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, rows_path)
    manifest_path = _write_split_manifest(tmp_path / "split_manifest.json", _split_manifest("split_rows.jsonl", rows_sha))

    manifest = load_oled_curated_split_dataset_view_writer_manifest_json(manifest_path)
    loaded_rows = load_oled_curated_split_dataset_view_rows_from_manifest(manifest=manifest, base_dir=tmp_path)

    assert loaded_rows[0].split_row_id == split_rows[0].split_row_id
    with pytest.raises(ValueError, match="split_dataset_view_rows_sha256_mismatch:"):
        load_oled_curated_split_dataset_view_rows_from_manifest(
            manifest=_split_manifest("split_rows.jsonl", "not-the-sha"),
            base_dir=tmp_path,
        )


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    records = _records(3)
    split_rows = _split_rows(records)
    curated_path = _write_gold_records(tmp_path / "curated_gold.jsonl", records)
    rows_path = tmp_path / "split_rows.jsonl"
    rows_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, rows_path)
    manifest_path = _write_split_manifest(tmp_path / "split_manifest.json", _split_manifest("split_rows.jsonl", rows_sha))
    output_report = tmp_path / "feature_preflight.json"

    report = run_oled_curated_split_feature_preflight_from_files(
        curated_gold_jsonl_path=curated_path,
        split_dataset_view_manifest_path=manifest_path,
        split_dataset_view_base_dir=tmp_path,
        output_report_path=output_report,
        policy=OledCuratedSplitFeaturePreflightPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )

    assert report.is_valid
    assert output_report.exists()
    assert not list(tmp_path.glob("*feature_table*.jsonl"))


def test_report_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_curated_split_feature_preflight(
        gold_records=_records(1),
        split_rows=_split_rows(_records(1)),
        policy=OledCuratedSplitFeaturePreflightPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )
    output_path = tmp_path / "report.json"

    write_oled_curated_split_feature_preflight_report_json(report, output_path)
    text = output_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert str(tmp_path) not in text
    assert "raw paper text" not in text
    assert '"gold_record"' not in text
    assert "layered_record" not in text


def test_cli_smoke_writes_compact_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records = _records(3)
    split_rows = _split_rows(records)
    curated_path = _write_gold_records(tmp_path / "curated_gold.jsonl", records)
    rows_path = tmp_path / "split_rows.jsonl"
    rows_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, rows_path)
    manifest_path = _write_split_manifest(tmp_path / "split_manifest.json", _split_manifest("split_rows.jsonl", rows_sha))
    output_report = tmp_path / "feature_preflight.json"

    exit_code = main(
        [
            "--curated-gold-jsonl",
            str(curated_path),
            "--split-dataset-view-manifest",
            str(manifest_path),
            "--split-dataset-view-base-dir",
            str(tmp_path),
            "--output-report",
            str(output_report),
            "--feature-view",
            "full_context",
            "--target-property-id",
            "eqe_percent",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output_report.exists()
    assert "row_alignments" not in stdout
    assert json.loads(stdout)["input_split_row_count"] == len(split_rows)


def test_package_exports_for_split_feature_preflight(tmp_path: Path) -> None:
    records = _records(1)
    split_rows = _split_rows(records)
    report = package_run_preflight(
        gold_records=records,
        split_rows=split_rows,
        policy=OledCuratedSplitFeaturePreflightPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )
    report_path = tmp_path / "package-report.json"
    package_write_report(report, report_path)
    rows_path = tmp_path / "package-split-rows.jsonl"
    rows_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, rows_path)
    manifest_path = _write_split_manifest(tmp_path / "package-split-manifest.json", _split_manifest("package-split-rows.jsonl", rows_sha))
    loaded_manifest = package_load_split_manifest(manifest_path)
    loaded_rows = package_load_split_rows_from_manifest(manifest=loaded_manifest, base_dir=tmp_path)
    runner_report = package_run_from_files(
        curated_gold_jsonl_path=_write_gold_records(tmp_path / "package-curated.jsonl", records),
        split_dataset_view_manifest_path=manifest_path,
        split_dataset_view_base_dir=tmp_path,
        output_report_path=tmp_path / "package-runner-report.json",
        policy=OledCuratedSplitFeaturePreflightPolicy(feature_views=["full_context"], target_property_ids=["eqe_percent"]),
    )

    assert isinstance(report, OledCuratedSplitFeaturePreflightReport)
    assert isinstance(report.row_alignments[0], OledSplitFeatureRowAlignment)
    assert isinstance(report.summaries[0], OledSplitFeaturePreflightSummary)
    assert isinstance(OledCuratedSplitFeaturePreflightFinding(code="x", message="y"), OledCuratedSplitFeaturePreflightFinding)
    assert OledCuratedSplitFeaturePreflightStatus.PASSED.value == "passed"
    assert OledSplitFeatureRowAlignmentStatus.MATCHED.value == "matched"
    assert isinstance(OledCuratedSplitFeaturePreflightPolicy(), OledCuratedSplitFeaturePreflightPolicy)
    assert loaded_rows[0].split_row_id == split_rows[0].split_row_id
    assert runner_report.is_valid
    assert report_path.exists()
