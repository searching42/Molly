from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledConfidenceAssessment,
    OledCausalLayer,
    OledConfounderFlags,
    OledCuratedDatasetSplitPreflightFinding,
    OledCuratedDatasetSplitPreflightPolicy,
    OledCuratedDatasetSplitPreflightReport,
    OledCuratedDatasetSplitPreflightStatus,
    OledCuratedDatasetViewFileResult,
    OledCuratedDatasetViewRowArtifact,
    OledCuratedDatasetViewWriteStatus,
    OledCuratedDatasetViewWriterManifest,
    OledCuratedDatasetViewWriterPolicy,
    OledDatasetViewRowSplitAssignment,
    OledDatasetViewRowSplitStatus,
    OledDatasetViewSplitSummary,
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
    build_oled_curated_dataset_view_row_artifacts,
    load_oled_curated_dataset_view_rows_from_manifest as package_load_rows_from_manifest,
    load_oled_curated_dataset_view_writer_manifest_json as package_load_manifest,
    run_oled_curated_dataset_split_preflight as package_run_preflight,
    run_oled_curated_dataset_split_preflight_from_files as package_run_preflight_from_files,
    write_oled_curated_dataset_split_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_dataset_split_preflight import (
    load_oled_curated_dataset_view_rows_from_manifest,
    load_oled_curated_dataset_view_writer_manifest_json,
    main,
    run_oled_curated_dataset_split_preflight,
    run_oled_curated_dataset_split_preflight_from_files,
    write_oled_curated_dataset_split_preflight_report_json,
)
from ai4s_agent.domains.oled_curated_dataset_view_writer import (
    write_oled_curated_dataset_view_manifest_json,
    write_oled_curated_dataset_view_rows_jsonl,
)
from ai4s_agent.domains.oled_curated_gold_view_preflight import (
    write_oled_curated_gold_view_preflight_report_json,
)
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind, build_oled_dataset_view


def _gold_record(record_id: str, *, value: float = 18.0) -> OledGoldDatasetRecord:
    evidence_ref = f"paper-{record_id}:table-1:row-1"
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles=f"C1=CC={record_id}",
                inchikey=f"INCHIKEY-{record_id.upper()}",
            ),
            interaction=OledInteractionLayer(
                emitter_smiles=f"emitter-{record_id}",
                host_smiles=f"host-{record_id}",
                doping_ratio=0.08,
            ),
            device=OledDeviceLayer(
                device_stack=["ITO", f"HTL-{record_id}", f"EML-{record_id}", "ETL", "Al"],
                etl_material="TPBi",
                htl_material="TAPC",
            ),
            measurement=OledMeasurementLayer(
                measurements=[
                    OledPropertyObservation(
                        property_label="EQE (%)",
                        value=value,
                        unit="%",
                        condition=OledMeasurementCondition(
                            luminance_cd_m2=100,
                            current_density_ma_cm2=4.2,
                            temperature_k=298.15,
                        ),
                        evidence_sources=[
                            OledEvidenceSource(
                                source_id=evidence_ref,
                                source_type=OledEvidenceType.TABLE,
                                layer=OledCausalLayer.MEASUREMENT,
                            )
                        ],
                        confidence=OledConfidenceAssessment(score=0.91),
                    )
                ]
            ),
            confounder_flags=OledConfounderFlags(is_device_optimized=True),
        ),
        evidence_refs=[evidence_ref],
        reviewer="reviewer-1",
        metadata={"curated_dataset_written": True, "training_data_written": False},
    )


def _records(count: int = 3) -> list[OledGoldDatasetRecord]:
    return [_gold_record(f"gold-{index}", value=18.0 + index) for index in range(count)]


def _rows(records: list[OledGoldDatasetRecord]) -> list[OledCuratedDatasetViewRowArtifact]:
    view_report = build_oled_dataset_view(
        records,
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
        target_property_id="eqe_percent",
    )
    return build_oled_curated_dataset_view_row_artifacts(view_report)


def _write_records(path: Path, records: list[OledGoldDatasetRecord]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record.model_dump(mode="json"), sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _view_manifest(rows_path: str, sha: str | None) -> OledCuratedDatasetViewWriterManifest:
    return OledCuratedDatasetViewWriterManifest(
        manifest_id="oled-dataset-view-writer:test",
        output_file_count=1,
        output_row_count=1,
        file_results=[
            OledCuratedDatasetViewFileResult(
                view_kind="raw_all_measurements",
                target_property_id="eqe_percent",
                status=OledCuratedDatasetViewWriteStatus.WRITTEN,
                row_count=1,
                output_jsonl_path=rows_path,
                output_sha256=sha,
                reason_codes=["selected_for_write"],
            )
        ],
        policy=OledCuratedDatasetViewWriterPolicy(
            view_kinds=["raw_all_measurements"],
            target_property_ids=["eqe_percent"],
        ),
        metadata={"dataset_view_writer": True, "dataset_view_rows_written": True},
    )


def _write_view_manifest(path: Path, manifest: OledCuratedDatasetViewWriterManifest) -> Path:
    write_oled_curated_dataset_view_manifest_json(manifest, path)
    return path


def test_main_split_preflight_success_assigns_rows() -> None:
    records = _records(3)
    rows = _rows(records)

    report = run_oled_curated_dataset_split_preflight(
        gold_records=records,
        dataset_view_rows=rows,
    )

    assert report.is_valid
    assert report.status == OledCuratedDatasetSplitPreflightStatus.PASSED
    assert report.split_plan is not None
    assert len(report.row_assignments) == len(rows)
    assert {assignment.status for assignment in report.row_assignments} == {OledDatasetViewRowSplitStatus.ASSIGNED}
    assert sum(report.rows_by_split.values()) == len(rows)
    assert report.metadata["split_preflight_only"] is True
    assert report.metadata["split_rows_written"] is False
    assert report.metadata["training_data_written"] is False


def test_gold_validation_failure_reports_errors() -> None:
    invalid_record = _gold_record("gold-invalid").model_copy(update={"evidence_refs": []})

    report = run_oled_curated_dataset_split_preflight(
        gold_records=[invalid_record],
        dataset_view_rows=[],
        policy=OledCuratedDatasetSplitPreflightPolicy(allow_empty_split=True),
    )

    assert report.status == OledCuratedDatasetSplitPreflightStatus.FAILED
    assert "gold_missing_evidence_refs" in report.gold_validation_error_codes
    assert "gold_validation_errors_present" in report.error_codes


def test_cross_split_source_row_is_detected() -> None:
    records = _records(2)
    base_row = _rows(records)[0]
    cross_row = base_row.model_copy(
        update={
            "row_id": "row-cross-split",
            "record_id": records[0].record_id,
            "source_record_ids": [records[0].record_id, records[1].record_id],
        }
    )

    report = run_oled_curated_dataset_split_preflight(
        gold_records=records,
        dataset_view_rows=[cross_row],
        policy=OledCuratedDatasetSplitPreflightPolicy(split_names=["train", "test"]),
    )

    assert report.status == OledCuratedDatasetSplitPreflightStatus.FAILED
    assert report.row_assignments[0].status == OledDatasetViewRowSplitStatus.CROSS_SPLIT_SOURCE_RECORDS
    assert "row_source_records_cross_split" in report.error_codes


def test_unknown_row_source_record_is_unassigned() -> None:
    records = _records(3)
    unknown_row = _rows(records)[0].model_copy(
        update={"row_id": "row-unknown", "record_id": "missing-record", "source_record_ids": ["missing-record"]}
    )

    report = run_oled_curated_dataset_split_preflight(
        gold_records=records,
        dataset_view_rows=[unknown_row],
    )

    assert report.status == OledCuratedDatasetSplitPreflightStatus.FAILED
    assert report.row_assignments[0].status == OledDatasetViewRowSplitStatus.UNASSIGNED
    assert "unknown_row_source_record" in report.error_codes


def test_empty_split_policy_can_fail_or_allow_empty_splits() -> None:
    records = _records(2)
    rows = _rows(records)

    strict_report = run_oled_curated_dataset_split_preflight(
        gold_records=records,
        dataset_view_rows=rows,
    )
    allowed_report = run_oled_curated_dataset_split_preflight(
        gold_records=records,
        dataset_view_rows=rows,
        policy=OledCuratedDatasetSplitPreflightPolicy(allow_empty_split=True),
    )

    assert strict_report.status == OledCuratedDatasetSplitPreflightStatus.FAILED
    assert "empty_split" in strict_report.error_codes
    assert "empty_split" not in allowed_report.error_codes


def test_dataset_view_writer_manifest_loader_handles_valid_missing_and_invalid(tmp_path: Path) -> None:
    manifest_path = _write_view_manifest(tmp_path / "manifest.json", _view_manifest("rows.jsonl", "abc"))

    loaded = load_oled_curated_dataset_view_writer_manifest_json(manifest_path)

    assert loaded.manifest_id == "oled-dataset-view-writer:test"
    with pytest.raises(ValueError, match="missing_dataset_view_writer_manifest:"):
        load_oled_curated_dataset_view_writer_manifest_json(tmp_path / "missing.json")
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_dataset_view_writer_manifest_json:"):
        load_oled_curated_dataset_view_writer_manifest_json(bad_path)


def test_load_rows_from_manifest_verifies_sha(tmp_path: Path) -> None:
    rows = _rows(_records(1))
    rows_path = tmp_path / "rows.jsonl"
    sha = write_oled_curated_dataset_view_rows_jsonl(rows, rows_path)

    loaded = load_oled_curated_dataset_view_rows_from_manifest(
        manifest=_view_manifest("rows.jsonl", sha),
        base_dir=tmp_path,
    )

    assert loaded[0].row_id == rows[0].row_id
    with pytest.raises(ValueError, match="dataset_view_rows_sha256_mismatch:"):
        load_oled_curated_dataset_view_rows_from_manifest(
            manifest=_view_manifest("rows.jsonl", "not-the-sha"),
            base_dir=tmp_path,
        )


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    records = _records(3)
    rows = _rows(records)
    curated_path = _write_records(tmp_path / "curated_gold.jsonl", records)
    rows_path = tmp_path / "rows.jsonl"
    rows_sha = write_oled_curated_dataset_view_rows_jsonl(rows, rows_path)
    manifest_path = _write_view_manifest(tmp_path / "dataset_view_manifest.json", _view_manifest("rows.jsonl", rows_sha))
    output_report = tmp_path / "split_preflight_report.json"

    report = run_oled_curated_dataset_split_preflight_from_files(
        curated_gold_jsonl_path=curated_path,
        dataset_view_manifest_path=manifest_path,
        output_report_path=output_report,
    )

    assert report.is_valid
    assert output_report.exists()
    assert not (tmp_path / "train.jsonl").exists()
    assert json.loads(output_report.read_text(encoding="utf-8"))["metadata"]["split_rows_written"] is False


def test_report_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_curated_dataset_split_preflight(
        gold_records=_records(3),
        dataset_view_rows=_rows(_records(3)),
    )
    output_path = tmp_path / "report.json"

    write_oled_curated_dataset_split_preflight_report_json(report, output_path)
    text = output_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert str(tmp_path) not in text
    assert "raw paper text" not in text
    assert "layered_record" not in text


def test_cli_smoke_writes_compact_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records = _records(3)
    rows = _rows(records)
    curated_path = _write_records(tmp_path / "curated_gold.jsonl", records)
    rows_path = tmp_path / "rows.jsonl"
    rows_sha = write_oled_curated_dataset_view_rows_jsonl(rows, rows_path)
    manifest_path = _write_view_manifest(tmp_path / "dataset_view_manifest.json", _view_manifest("rows.jsonl", rows_sha))
    output_report = tmp_path / "split_report.json"

    exit_code = main(
        [
            "--curated-gold-jsonl",
            str(curated_path),
            "--dataset-view-manifest",
            str(manifest_path),
            "--dataset-view-base-dir",
            str(tmp_path),
            "--output-report",
            str(output_report),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output_report.exists()
    assert "row_assignments" not in stdout
    assert json.loads(stdout)["input_dataset_view_row_count"] == len(rows)


def test_public_split_preflight_api_is_exported_from_domain_package(tmp_path: Path) -> None:
    records = _records(3)
    rows = _rows(records)
    report = package_run_preflight(gold_records=records, dataset_view_rows=rows)
    output_path = tmp_path / "package-report.json"
    package_write_report(report, output_path)
    rows_path = tmp_path / "package-rows.jsonl"
    rows_sha = write_oled_curated_dataset_view_rows_jsonl(rows, rows_path)
    manifest_path = _write_view_manifest(tmp_path / "package-manifest.json", _view_manifest("package-rows.jsonl", rows_sha))

    loaded_manifest = package_load_manifest(manifest_path)
    loaded_rows = package_load_rows_from_manifest(manifest=loaded_manifest, base_dir=tmp_path)
    file_report = package_run_preflight_from_files(
        curated_gold_jsonl_path=_write_records(tmp_path / "package-curated.jsonl", records),
        dataset_view_manifest_path=manifest_path,
    )

    assert isinstance(report, OledCuratedDatasetSplitPreflightReport)
    assert isinstance(report.findings[0] if report.findings else OledCuratedDatasetSplitPreflightFinding(code="x", message="y"), OledCuratedDatasetSplitPreflightFinding)
    assert isinstance(report.row_assignments[0], OledDatasetViewRowSplitAssignment)
    assert isinstance(report.view_summaries[0], OledDatasetViewSplitSummary)
    assert OledDatasetViewRowSplitStatus.ASSIGNED.value == "assigned"
    assert OledCuratedDatasetSplitPreflightStatus.PASSED.value == "passed"
    assert isinstance(OledCuratedDatasetSplitPreflightPolicy(), OledCuratedDatasetSplitPreflightPolicy)
    assert loaded_rows[0].row_id == rows[0].row_id
    assert file_report.is_valid
    assert output_path.exists()
