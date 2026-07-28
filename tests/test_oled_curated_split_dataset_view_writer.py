from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledConfidenceAssessment,
    OledCausalLayer,
    OledConfounderFlags,
    OledCuratedDatasetSplitPreflightPolicy,
    OledCuratedDatasetViewFileResult,
    OledCuratedDatasetViewRowArtifact,
    OledCuratedDatasetViewWriteStatus,
    OledCuratedDatasetViewWriterManifest,
    OledCuratedDatasetViewWriterPolicy,
    OledCuratedSplitDatasetViewFileResult,
    OledCuratedSplitDatasetViewRowArtifact,
    OledCuratedSplitDatasetViewWriteStatus,
    OledCuratedSplitDatasetViewWriterFinding,
    OledCuratedSplitDatasetViewWriterManifest,
    OledCuratedSplitDatasetViewWriterPolicy,
    OledCuratedSplitDatasetViewWriterReport,
    OledDatasetViewKind,
    OledDatasetViewRowSplitStatus,
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
    build_oled_curated_split_dataset_view_row_artifacts as package_build_split_rows,
    load_oled_curated_dataset_split_preflight_report_json as package_load_preflight,
    load_oled_curated_split_dataset_view_rows_jsonl as package_load_split_rows,
    oled_split_dataset_view_output_filename as package_output_filename,
    run_oled_curated_dataset_split_preflight,
    run_oled_curated_split_dataset_view_writer_from_files as package_run_from_files,
    select_oled_curated_split_dataset_view_rows_for_write as package_select_for_write,
    write_oled_curated_dataset_split_preflight_report_json,
    write_oled_curated_split_dataset_view_manifest_json as package_write_manifest,
    write_oled_curated_split_dataset_view_rows_jsonl as package_write_split_rows,
)
from ai4s_agent.domains.oled_curated_dataset_view_writer import (
    write_oled_curated_dataset_view_manifest_json,
    write_oled_curated_dataset_view_rows_jsonl,
)
from ai4s_agent.domains.oled_curated_split_dataset_view_writer import (
    build_oled_curated_split_dataset_view_row_artifacts,
    load_oled_curated_dataset_split_preflight_report_json,
    load_oled_curated_split_dataset_view_rows_jsonl,
    main,
    oled_split_dataset_view_output_filename,
    run_oled_curated_split_dataset_view_writer_from_files,
    select_oled_curated_split_dataset_view_rows_for_write,
    write_oled_curated_split_dataset_view_manifest_json,
    write_oled_curated_split_dataset_view_rows_jsonl,
)
from ai4s_agent.domains.oled_dataset_views import build_oled_dataset_view


def _gold_record(record_id: str, *, value: float = 20.0) -> OledGoldDatasetRecord:
    evidence_ref = f"paper-{record_id}:table-1:row-1"
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles=f"C1=CC={record_id}",
                inchikey=f"SPLIT-INCHIKEY-{record_id.upper()}",
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
                            current_density_ma_cm2=5.0,
                            temperature_k=298.15,
                        ),
                        evidence_sources=[
                            OledEvidenceSource(
                                source_id=evidence_ref,
                                source_type=OledEvidenceType.TABLE,
                                layer=OledCausalLayer.MEASUREMENT,
                            )
                        ],
                        confidence=OledConfidenceAssessment(score=0.92),
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
    return [_gold_record(f"gold-split-{index}", value=20.0 + index) for index in range(count)]


def _rows(records: list[OledGoldDatasetRecord], *, include_features: bool = False) -> list[OledCuratedDatasetViewRowArtifact]:
    view_report = build_oled_dataset_view(
        records,
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
        target_property_id="eqe_percent",
    )
    return build_oled_curated_dataset_view_row_artifacts(view_report, include_feature_payload=include_features)


def _preflight(records: list[OledGoldDatasetRecord], rows: list[OledCuratedDatasetViewRowArtifact]):
    return run_oled_curated_dataset_split_preflight(
        gold_records=records,
        dataset_view_rows=rows,
        policy=OledCuratedDatasetSplitPreflightPolicy(allow_empty_split=True),
    )


def _write_preflight(path: Path, report) -> Path:
    write_oled_curated_dataset_split_preflight_report_json(report, path)
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


def test_confirmation_gate_requires_explicit_split_dataset_view_write() -> None:
    records = _records(3)
    rows = _rows(records)

    with pytest.raises(ValueError, match="confirmation_required:split_dataset_view_write"):
        select_oled_curated_split_dataset_view_rows_for_write(
            rows,
            split_preflight_report=_preflight(records, rows),
        )


def test_build_split_row_artifacts_preserves_assignment_and_deterministic_id() -> None:
    records = _records(3)
    rows = _rows(records)
    preflight = _preflight(records, rows)

    first, findings = build_oled_curated_split_dataset_view_row_artifacts(
        rows,
        split_preflight_report=preflight,
    )
    second, _ = build_oled_curated_split_dataset_view_row_artifacts(
        rows,
        split_preflight_report=preflight,
    )

    assert not findings
    assert len(first) == len(rows)
    assert first[0].split_row_id == second[0].split_row_id
    assert {row.row_id for row in first} == {row.row_id for row in rows}
    assert first[0].split
    assert first[0].source_record_splits
    assert first[0].assignment_reason_codes == ["row_assigned"]


def test_unassigned_row_is_rejected_by_default() -> None:
    records = _records(3)
    rows = _rows(records)
    bad_row = rows[0].model_copy(update={"row_id": "missing-assignment-row"})

    split_rows, findings = build_oled_curated_split_dataset_view_row_artifacts(
        [bad_row],
        split_preflight_report=_preflight(records, rows),
    )

    assert split_rows == []
    assert [finding.code for finding in findings] == ["row_unassigned_rejected"]


def test_cross_split_row_is_rejected_by_default() -> None:
    records = _records(2)
    rows = _rows(records)
    cross_row = rows[0].model_copy(
        update={
            "row_id": "cross-row",
            "record_id": records[0].record_id,
            "source_record_ids": [records[0].record_id, records[1].record_id],
        }
    )
    preflight = run_oled_curated_dataset_split_preflight(
        gold_records=records,
        dataset_view_rows=[cross_row],
        policy=OledCuratedDatasetSplitPreflightPolicy(split_names=["train", "test"], allow_empty_split=True),
    )

    split_rows, findings = build_oled_curated_split_dataset_view_row_artifacts(
        [cross_row],
        split_preflight_report=preflight,
    )

    assert split_rows == []
    assert [finding.code for finding in findings] == ["cross_split_row_rejected"]


def test_invalid_preflight_blocks_writer() -> None:
    records = _records(3)
    rows = _rows(records)
    preflight = _preflight(records, rows).model_copy(update={"status": "failed"})

    report = select_oled_curated_split_dataset_view_rows_for_write(
        rows,
        split_preflight_report=preflight,
        confirm_split_dataset_view_write=True,
    )

    assert not report.is_valid
    assert report.split_row_artifacts == []
    assert "split_preflight_failed" in report.error_codes


def test_feature_payload_omitted_by_default() -> None:
    records = _records(3)
    rows = _rows(records, include_features=True)

    split_rows, _ = build_oled_curated_split_dataset_view_row_artifacts(
        rows,
        split_preflight_report=_preflight(records, rows),
    )

    assert rows[0].features
    assert split_rows[0].features == {}
    assert split_rows[0].metadata["feature_payload_omitted"] is True


def test_include_feature_payload_policy_preserves_features() -> None:
    records = _records(3)
    rows = _rows(records, include_features=True)

    split_rows, _ = build_oled_curated_split_dataset_view_row_artifacts(
        rows,
        split_preflight_report=_preflight(records, rows),
        policy=OledCuratedSplitDatasetViewWriterPolicy(include_feature_payload=True),
    )

    assert split_rows[0].features
    assert split_rows[0].metadata.get("feature_payload_omitted") is not True


def test_split_rows_jsonl_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    records = _records(3)
    rows = _rows(records)
    split_rows, _ = build_oled_curated_split_dataset_view_row_artifacts(
        rows,
        split_preflight_report=_preflight(records, rows),
    )
    output_path = tmp_path / "split_rows.jsonl"

    first_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, output_path)
    first_text = output_path.read_text(encoding="utf-8")
    second_sha = write_oled_curated_split_dataset_view_rows_jsonl(split_rows, output_path)

    assert first_sha == second_sha
    assert first_text.splitlines()[0] == json.dumps(json.loads(first_text.splitlines()[0]), sort_keys=True, separators=(",", ":"))
    assert str(tmp_path) not in first_text
    assert "raw paper text" not in first_text
    assert "gold_record" not in first_text


def test_manifest_writer_is_deterministic_and_has_safety_metadata(tmp_path: Path) -> None:
    records = _records(3)
    rows = _rows(records)
    report = select_oled_curated_split_dataset_view_rows_for_write(
        rows,
        split_preflight_report=_preflight(records, rows),
        confirm_split_dataset_view_write=True,
    )
    manifest = report.manifest.model_copy(
        update={
            "file_results": [
                result.model_copy(update={"output_jsonl_path": "rows.jsonl", "output_sha256": "abc"})
                for result in report.manifest.file_results
            ]
        }
    )
    manifest_path = tmp_path / "manifest.json"

    write_oled_curated_split_dataset_view_manifest_json(manifest, manifest_path)
    text = manifest_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert payload["rows_by_split"]
    assert payload["metadata"]["training_data_written"] is False
    assert payload["metadata"]["ml_ready_training_data_written"] is False


def test_combined_runner_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    records = _records(3)
    rows = _rows(records)
    rows_path = tmp_path / "rows.jsonl"
    rows_sha = write_oled_curated_dataset_view_rows_jsonl(rows, rows_path)
    view_manifest_path = _write_view_manifest(tmp_path / "dataset_view_manifest.json", _view_manifest("rows.jsonl", rows_sha))
    split_preflight_path = _write_preflight(tmp_path / "split_preflight.json", _preflight(records, rows))
    output_dir = tmp_path / "split_views"
    output_manifest = tmp_path / "split_manifest.json"

    report = run_oled_curated_split_dataset_view_writer_from_files(
        dataset_view_manifest_path=view_manifest_path,
        split_preflight_report_path=split_preflight_path,
        dataset_view_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert not output_dir.exists()
    assert "dry_run_no_rows_written" in report.manifest.reason_code_counts


def test_combined_runner_write_mode_writes_split_rows_and_manifest(tmp_path: Path) -> None:
    records = _records(3)
    rows = _rows(records)
    rows_path = tmp_path / "rows.jsonl"
    rows_sha = write_oled_curated_dataset_view_rows_jsonl(rows, rows_path)
    view_manifest_path = _write_view_manifest(tmp_path / "dataset_view_manifest.json", _view_manifest("rows.jsonl", rows_sha))
    split_preflight_path = _write_preflight(tmp_path / "split_preflight.json", _preflight(records, rows))
    output_dir = tmp_path / "split_views"
    output_manifest = tmp_path / "split_manifest.json"

    report = run_oled_curated_split_dataset_view_writer_from_files(
        dataset_view_manifest_path=view_manifest_path,
        split_preflight_report_path=split_preflight_path,
        dataset_view_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        confirm_split_dataset_view_write=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert list(output_dir.glob("oled_split_view__*.jsonl"))
    assert all(result.output_sha256 for result in report.manifest.file_results)


def test_load_split_rows_jsonl_handles_valid_invalid_and_missing(tmp_path: Path) -> None:
    records = _records(3)
    rows = _rows(records)
    split_rows, _ = build_oled_curated_split_dataset_view_row_artifacts(
        rows,
        split_preflight_report=_preflight(records, rows),
    )
    split_rows_path = tmp_path / "split_rows.jsonl"
    write_oled_curated_split_dataset_view_rows_jsonl(split_rows, split_rows_path)

    loaded = load_oled_curated_split_dataset_view_rows_jsonl(split_rows_path)

    assert {row.split_row_id for row in loaded} == {row.split_row_id for row in split_rows}
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps(split_rows[0].model_dump(mode="json")) + "\n{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_split_dataset_view_rows_jsonl:line-2"):
        load_oled_curated_split_dataset_view_rows_jsonl(bad_path)
    with pytest.raises(ValueError, match="missing_split_dataset_view_rows_jsonl:"):
        load_oled_curated_split_dataset_view_rows_jsonl(tmp_path / "missing.jsonl")


def test_load_split_preflight_report_json_handles_valid_invalid_and_missing(tmp_path: Path) -> None:
    records = _records(3)
    rows = _rows(records)
    preflight_path = _write_preflight(tmp_path / "split_preflight.json", _preflight(records, rows))

    loaded = load_oled_curated_dataset_split_preflight_report_json(preflight_path)

    assert loaded.input_dataset_view_row_count == len(rows)
    with pytest.raises(ValueError, match="missing_split_preflight_report:"):
        load_oled_curated_dataset_split_preflight_report_json(tmp_path / "missing.json")
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_split_preflight_report_json:"):
        load_oled_curated_dataset_split_preflight_report_json(bad_path)


def test_cli_smoke_writes_outputs_and_compact_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records = _records(3)
    rows = _rows(records)
    rows_path = tmp_path / "rows.jsonl"
    rows_sha = write_oled_curated_dataset_view_rows_jsonl(rows, rows_path)
    view_manifest_path = _write_view_manifest(tmp_path / "dataset_view_manifest.json", _view_manifest("rows.jsonl", rows_sha))
    split_preflight_path = _write_preflight(tmp_path / "split_preflight.json", _preflight(records, rows))
    output_dir = tmp_path / "split_views"
    output_manifest = tmp_path / "split_manifest.json"

    exit_code = main(
        [
            "--dataset-view-manifest",
            str(view_manifest_path),
            "--split-preflight-report",
            str(split_preflight_path),
            "--dataset-view-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-split-dataset-view-write",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output_dir.exists()
    assert output_manifest.exists()
    assert "split_row_artifacts" not in stdout
    assert json.loads(stdout)["output_row_count"] == len(rows)


def test_package_exports_for_split_dataset_view_writer(tmp_path: Path) -> None:
    records = _records(3)
    rows = _rows(records)
    preflight = _preflight(records, rows)
    split_rows, findings = package_build_split_rows(rows, split_preflight_report=preflight)
    rows_path = tmp_path / package_output_filename(
        split=split_rows[0].split,
        view_kind=split_rows[0].view_kind,
        target_property_id=split_rows[0].target_property_id,
    )
    sha = package_write_split_rows(split_rows, rows_path)
    loaded_rows = package_load_split_rows(rows_path)
    report = package_select_for_write(rows, split_preflight_report=preflight, confirm_split_dataset_view_write=True)
    manifest_path = tmp_path / "package-manifest.json"
    package_write_manifest(report.manifest, manifest_path)
    preflight_path = _write_preflight(tmp_path / "package-preflight.json", preflight)
    loaded_preflight = package_load_preflight(preflight_path)
    dataset_rows_path = tmp_path / "package-dataset-rows.jsonl"
    dataset_rows_sha = write_oled_curated_dataset_view_rows_jsonl(rows, dataset_rows_path)
    dataset_manifest_path = _write_view_manifest(
        tmp_path / "package-dataset-manifest.json",
        _view_manifest("package-dataset-rows.jsonl", dataset_rows_sha),
    )
    runner_report = package_run_from_files(
        dataset_view_manifest_path=dataset_manifest_path,
        split_preflight_report_path=preflight_path,
        dataset_view_base_dir=tmp_path,
        output_manifest_path=tmp_path / "package-runner-manifest.json",
        dry_run=True,
    )

    assert not findings
    assert sha
    assert isinstance(split_rows[0], OledCuratedSplitDatasetViewRowArtifact)
    assert isinstance(report.manifest.file_results[0], OledCuratedSplitDatasetViewFileResult)
    assert isinstance(OledCuratedSplitDatasetViewWriterFinding(code="x", message="y"), OledCuratedSplitDatasetViewWriterFinding)
    assert isinstance(report.manifest, OledCuratedSplitDatasetViewWriterManifest)
    assert isinstance(report, OledCuratedSplitDatasetViewWriterReport)
    assert OledCuratedSplitDatasetViewWriteStatus.WRITTEN.value == "written"
    assert {row.split_row_id for row in loaded_rows} == {row.split_row_id for row in split_rows}
    assert loaded_preflight.input_dataset_view_row_count == len(rows)
    assert runner_report.is_valid
    assert manifest_path.exists()
