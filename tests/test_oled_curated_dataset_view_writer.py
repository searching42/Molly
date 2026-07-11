from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledConfidenceAssessment,
    OledCausalLayer,
    OledConfounderFlags,
    OledCuratedDatasetViewFileResult,
    OledCuratedDatasetViewRowArtifact,
    OledCuratedDatasetViewWriteStatus,
    OledCuratedDatasetViewWriterFinding,
    OledCuratedDatasetViewWriterManifest,
    OledCuratedDatasetViewWriterPolicy,
    OledCuratedDatasetViewWriterReport,
    OledCuratedGoldManifest,
    OledCuratedGoldViewPreflightReport,
    OledCuratedGoldViewPreflightStatus,
    OledCuratedGoldWriterPolicy,
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
    build_oled_curated_dataset_view_row_artifacts as package_build_oled_curated_dataset_view_row_artifacts,
    load_oled_curated_dataset_view_rows_jsonl as package_load_oled_curated_dataset_view_rows_jsonl,
    oled_dataset_view_output_filename as package_oled_dataset_view_output_filename,
    run_oled_curated_dataset_view_writer_from_files as package_run_oled_curated_dataset_view_writer_from_files,
    select_oled_curated_dataset_views_for_write as package_select_oled_curated_dataset_views_for_write,
    write_oled_curated_dataset_view_manifest_json as package_write_oled_curated_dataset_view_manifest_json,
    write_oled_curated_dataset_view_rows_jsonl as package_write_oled_curated_dataset_view_rows_jsonl,
)
from ai4s_agent.domains.oled_curated_dataset_view_writer import (
    build_oled_curated_dataset_view_row_artifacts,
    load_oled_curated_dataset_view_rows_jsonl,
    main,
    oled_dataset_view_output_filename,
    run_oled_curated_dataset_view_writer_from_files,
    select_oled_curated_dataset_views_for_write,
    write_oled_curated_dataset_view_manifest_json,
    write_oled_curated_dataset_view_rows_jsonl,
)
from ai4s_agent.domains.oled_curated_gold_view_preflight import (
    sha256_file,
)
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind, build_oled_dataset_view


def _device_gold_record(
    record_id: str = "gold-view-writer",
    *,
    property_label: str = "EQE (%)",
    value: float = 19.5,
    reported_value_text: str | None = None,
    reported_decimal_places: int | None = None,
    raw_metadata: bool = False,
) -> OledGoldDatasetRecord:
    evidence_ref = f"paper:{record_id}:table-1:row-1"
    return OledGoldDatasetRecord(
        record_id=record_id,
        layered_record=OledLayeredRecord(
            molecule=OledMolecularLayer(
                canonical_smiles="N1C=CC=C1",
                inchikey="DEVICE-INCHIKEY",
                properties=[
                    OledPropertyObservation(
                        property_label="ΔE ST",
                        value=0.12,
                        unit="eV",
                        evidence_sources=[
                            OledEvidenceSource(
                                source_id=f"{evidence_ref}:delta",
                                source_type=OledEvidenceType.TABLE,
                                layer=OledCausalLayer.MOLECULE,
                            )
                        ],
                        confidence=OledConfidenceAssessment(score=0.87),
                    )
                ],
            ),
            interaction=OledInteractionLayer(
                emitter_smiles="N1C=CC=C1",
                host_smiles="c1ccccc1",
                doping_ratio=0.08,
                film_type="doped",
                properties=[
                    OledPropertyObservation(
                        property_label="PLQY",
                        value=82,
                        unit="%",
                        evidence_sources=[
                            OledEvidenceSource(
                                source_id=f"{evidence_ref}:plqy",
                                source_type=OledEvidenceType.TABLE,
                                layer=OledCausalLayer.INTERACTION,
                            )
                        ],
                        confidence=OledConfidenceAssessment(score=0.9),
                    )
                ],
            ),
            device=OledDeviceLayer(
                device_stack=["ITO", "HTL", "EML", "ETL", "Al"],
                etl_material="TPBi",
                htl_material="TAPC",
            ),
            measurement=OledMeasurementLayer(
                measurements=[
                    OledPropertyObservation(
                        property_label=property_label,
                        value=value,
                        unit="%",
                        reported_value_text=reported_value_text,
                        reported_decimal_places=reported_decimal_places,
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
                        confidence=OledConfidenceAssessment(score=0.92),
                    )
                ]
            ),
            confounder_flags=OledConfounderFlags(is_device_optimized=True),
        ),
        evidence_refs=[evidence_ref],
        reviewer="reviewer-1",
        metadata={
            "curated_dataset_written": True,
            "training_data_written": False,
            **({"raw_text": "full raw paper text should never be written"} if raw_metadata else {}),
        },
    )


def _manifest(output_sha256: str | None, *, output_jsonl_path: str | None = "curated_gold_records.jsonl") -> OledCuratedGoldManifest:
    return OledCuratedGoldManifest(
        manifest_id="oled-curated-gold-writer:test",
        input_candidate_count=1,
        output_record_count=1,
        output_jsonl_path=output_jsonl_path,
        output_sha256=output_sha256,
        policy=OledCuratedGoldWriterPolicy(),
        metadata={"curated_gold_writer": True, "training_data_written": False},
    )


def _write_records(path: Path, records: list[OledGoldDatasetRecord]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record.model_dump(mode="json"), sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _write_manifest(path: Path, manifest: OledCuratedGoldManifest) -> Path:
    path.write_text(json.dumps(manifest.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    return path


def test_confirmation_gate_requires_explicit_dataset_view_write() -> None:
    with pytest.raises(ValueError, match="confirmation_required:dataset_view_write"):
        select_oled_curated_dataset_views_for_write([_device_gold_record()])


def test_build_row_artifacts_omit_features_by_default_and_are_deterministic() -> None:
    view_report = build_oled_dataset_view(
        [
            _device_gold_record(
                "gold-artifact",
                reported_value_text="19.50",
                reported_decimal_places=2,
            )
        ],
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
        target_property_id="eqe_percent",
    )

    first = build_oled_curated_dataset_view_row_artifacts(view_report)
    second = build_oled_curated_dataset_view_row_artifacts(view_report)

    assert len(first) == 1
    assert first[0].row_id == second[0].row_id
    assert first[0].view_kind == "raw_all_measurements"
    assert first[0].target_property_id == "eqe_percent"
    assert first[0].target_reported_value_text == "19.50"
    assert first[0].target_reported_decimal_places == 2
    assert first[0].target_reported_unit == "%"
    assert first[0].record_id == "gold-artifact"
    assert first[0].evidence_refs
    assert first[0].features == {}
    assert first[0].metadata["feature_payload_omitted"] is True


def test_include_feature_payload_policy_preserves_features() -> None:
    view_report = build_oled_dataset_view(
        [_device_gold_record("gold-feature")],
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
        target_property_id="eqe_percent",
    )

    artifacts = build_oled_curated_dataset_view_row_artifacts(view_report, include_feature_payload=True)

    assert artifacts[0].features
    assert artifacts[0].metadata.get("feature_payload_omitted") is not True


def test_select_valid_raw_all_measurements_view() -> None:
    report = select_oled_curated_dataset_views_for_write(
        [_device_gold_record("gold-select")],
        policy=OledCuratedDatasetViewWriterPolicy(
            view_kinds=["raw_all_measurements"],
            target_property_ids=["eqe_percent"],
        ),
        confirm_dataset_view_write=True,
    )

    assert report.is_valid
    assert report.row_artifacts
    assert report.manifest.output_row_count == 1
    assert report.manifest.file_results[0].status == OledCuratedDatasetViewWriteStatus.WRITTEN
    assert "selected_for_write" in report.manifest.file_results[0].reason_codes
    assert report.manifest.metadata["training_data_written"] is False
    assert report.manifest.metadata["leakage_splits_run"] is False


def test_empty_view_is_skipped_by_default() -> None:
    report = select_oled_curated_dataset_views_for_write(
        [_device_gold_record("gold-empty")],
        policy=OledCuratedDatasetViewWriterPolicy(
            view_kinds=["raw_all_measurements"],
            target_property_ids=["lifetime_hours"],
        ),
        confirm_dataset_view_write=True,
    )

    assert report.is_valid
    assert report.row_artifacts == []
    assert report.manifest.file_results[0].status == OledCuratedDatasetViewWriteStatus.SKIPPED
    assert "empty_view_skipped" in report.manifest.file_results[0].reason_codes


def test_invalid_preflight_blocks_writer() -> None:
    preflight = OledCuratedGoldViewPreflightReport(
        status=OledCuratedGoldViewPreflightStatus.FAILED,
        input_record_count=1,
        manifest_integrity_status="not_provided",
        findings=[],
    )

    report = select_oled_curated_dataset_views_for_write(
        [_device_gold_record("gold-preflight-blocked")],
        preflight_report=preflight,
        policy=OledCuratedDatasetViewWriterPolicy(
            view_kinds=["raw_all_measurements"],
            target_property_ids=["eqe_percent"],
        ),
        confirm_dataset_view_write=True,
    )

    assert not report.is_valid
    assert report.row_artifacts == []
    assert "preflight_failed" in report.error_codes


def test_rows_jsonl_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    view_report = build_oled_dataset_view(
        [_device_gold_record("gold-jsonl", raw_metadata=True)],
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
        target_property_id="eqe_percent",
    )
    rows = build_oled_curated_dataset_view_row_artifacts(view_report)
    output_path = tmp_path / "rows.jsonl"

    first_hash = write_oled_curated_dataset_view_rows_jsonl(rows, output_path)
    first_text = output_path.read_text(encoding="utf-8")
    second_hash = write_oled_curated_dataset_view_rows_jsonl(rows, output_path)

    assert first_hash == second_hash == hashlib.sha256(first_text.encode("utf-8")).hexdigest()
    assert first_text.splitlines()[0] == json.dumps(json.loads(first_text.splitlines()[0]), sort_keys=True, separators=(",", ":"))
    assert str(tmp_path) not in first_text
    assert "full raw paper text" not in first_text
    assert "layered_record" not in first_text


def test_manifest_writer_is_deterministic(tmp_path: Path) -> None:
    report = select_oled_curated_dataset_views_for_write(
        [_device_gold_record("gold-manifest")],
        policy=OledCuratedDatasetViewWriterPolicy(
            view_kinds=["raw_all_measurements"],
            target_property_ids=["eqe_percent"],
        ),
        confirm_dataset_view_write=True,
    )
    manifest = report.manifest.model_copy(update={"output_directory": "views"})
    manifest_path = tmp_path / "manifest.json"

    write_oled_curated_dataset_view_manifest_json(manifest, manifest_path)
    text = manifest_path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert payload["policy"]["include_feature_payload"] is False
    assert payload["metadata"]["training_data_written"] is False


def test_combined_runner_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    curated_path = _write_records(tmp_path / "curated_gold.jsonl", [_device_gold_record("gold-dry-run")])
    manifest_path = _write_manifest(tmp_path / "curated_manifest.json", _manifest(sha256_file(curated_path)))
    output_dir = tmp_path / "views"
    output_manifest = tmp_path / "dataset_view_manifest.json"

    report = run_oled_curated_dataset_view_writer_from_files(
        curated_gold_jsonl_path=curated_path,
        curated_gold_manifest_path=manifest_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        policy=OledCuratedDatasetViewWriterPolicy(
            view_kinds=["raw_all_measurements"],
            target_property_ids=["eqe_percent"],
        ),
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert not output_dir.exists()
    assert "dry_run_no_rows_written" in report.manifest.reason_code_counts


def test_combined_runner_write_mode_writes_rows_and_manifest(tmp_path: Path) -> None:
    curated_path = _write_records(tmp_path / "curated_gold.jsonl", [_device_gold_record("gold-write")])
    manifest_path = _write_manifest(tmp_path / "curated_manifest.json", _manifest(sha256_file(curated_path)))
    output_dir = tmp_path / "views"
    output_manifest = tmp_path / "dataset_view_manifest.json"

    report = run_oled_curated_dataset_view_writer_from_files(
        curated_gold_jsonl_path=curated_path,
        curated_gold_manifest_path=manifest_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        policy=OledCuratedDatasetViewWriterPolicy(
            view_kinds=["raw_all_measurements"],
            target_property_ids=["eqe_percent"],
        ),
        confirm_dataset_view_write=True,
    )

    output_file = output_dir / oled_dataset_view_output_filename(view_kind="raw_all_measurements", target_property_id="eqe_percent")
    assert report.is_valid
    assert output_file.exists()
    assert output_manifest.exists()
    assert report.manifest.file_results[0].output_sha256
    assert json.loads(output_manifest.read_text(encoding="utf-8"))["output_row_count"] == 1


def test_load_dataset_view_rows_jsonl_handles_valid_invalid_and_missing(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    view_report = build_oled_dataset_view(
        [_device_gold_record("gold-load-rows")],
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
        target_property_id="eqe_percent",
    )
    rows = build_oled_curated_dataset_view_row_artifacts(view_report)
    write_oled_curated_dataset_view_rows_jsonl(rows, rows_path)

    loaded = load_oled_curated_dataset_view_rows_jsonl(rows_path)

    assert len(loaded) == 1
    assert loaded[0].row_id == rows[0].row_id
    bad_path = tmp_path / "bad-rows.jsonl"
    bad_path.write_text(json.dumps(rows[0].model_dump(mode="json")) + "\n{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_dataset_view_rows_jsonl:line-2"):
        load_oled_curated_dataset_view_rows_jsonl(bad_path)
    with pytest.raises(ValueError, match="missing_dataset_view_rows_jsonl:"):
        load_oled_curated_dataset_view_rows_jsonl(tmp_path / "missing.jsonl")


def test_cli_smoke_writes_outputs_and_prints_compact_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    curated_path = _write_records(tmp_path / "curated_gold.jsonl", [_device_gold_record("gold-cli")])
    manifest_path = _write_manifest(tmp_path / "curated_manifest.json", _manifest(sha256_file(curated_path)))
    output_dir = tmp_path / "views"
    output_manifest = tmp_path / "dataset_view_manifest.json"

    exit_code = main(
        [
            "--curated-gold-jsonl",
            str(curated_path),
            "--curated-gold-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-dataset-view-write",
            "--view-kind",
            "raw_all_measurements",
            "--target-property-id",
            "eqe_percent",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert output_dir.exists()
    assert output_manifest.exists()
    assert "row_artifacts" not in stdout
    assert json.loads(stdout)["output_row_count"] == 1


def test_public_curated_dataset_view_writer_api_is_exported_from_domain_package(tmp_path: Path) -> None:
    view_report = build_oled_dataset_view(
        [_device_gold_record("gold-package")],
        view_kind=OledDatasetViewKind.RAW_ALL_MEASUREMENTS,
        target_property_id="eqe_percent",
    )
    rows = package_build_oled_curated_dataset_view_row_artifacts(view_report)
    rows_path = tmp_path / package_oled_dataset_view_output_filename(view_kind="raw_all_measurements", target_property_id="eqe_percent")
    sha = package_write_oled_curated_dataset_view_rows_jsonl(rows, rows_path)
    loaded_rows = package_load_oled_curated_dataset_view_rows_jsonl(rows_path)
    report = package_select_oled_curated_dataset_views_for_write(
        [_device_gold_record("gold-package-select")],
        policy=OledCuratedDatasetViewWriterPolicy(
            view_kinds=["raw_all_measurements"],
            target_property_ids=["eqe_percent"],
        ),
        confirm_dataset_view_write=True,
    )
    manifest_path = tmp_path / "package-manifest.json"
    package_write_oled_curated_dataset_view_manifest_json(report.manifest, manifest_path)
    curated_path = _write_records(tmp_path / "package-curated.jsonl", [_device_gold_record("gold-package-runner")])
    curated_manifest_path = _write_manifest(tmp_path / "package-curated-manifest.json", _manifest(sha256_file(curated_path)))
    runner_report = package_run_oled_curated_dataset_view_writer_from_files(
        curated_gold_jsonl_path=curated_path,
        curated_gold_manifest_path=curated_manifest_path,
        output_manifest_path=tmp_path / "package-runner-manifest.json",
        policy=OledCuratedDatasetViewWriterPolicy(
            view_kinds=["raw_all_measurements"],
            target_property_ids=["eqe_percent"],
        ),
        dry_run=True,
    )

    assert sha
    assert isinstance(rows[0], OledCuratedDatasetViewRowArtifact)
    assert isinstance(report.manifest.file_results[0], OledCuratedDatasetViewFileResult)
    assert isinstance(OledCuratedDatasetViewWriterFinding(code="x", message="y"), OledCuratedDatasetViewWriterFinding)
    assert isinstance(report.manifest, OledCuratedDatasetViewWriterManifest)
    assert isinstance(report, OledCuratedDatasetViewWriterReport)
    assert OledCuratedDatasetViewWriteStatus.WRITTEN.value == "written"
    assert loaded_rows[0].row_id == rows[0].row_id
    assert runner_report.is_valid
    assert manifest_path.exists()
