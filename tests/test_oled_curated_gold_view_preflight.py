from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledConfidenceAssessment,
    OledCausalLayer,
    OledConfounderFlags,
    OledCuratedGoldDatasetViewPreflightResult,
    OledCuratedGoldManifest,
    OledCuratedGoldManifestIntegrityStatus,
    OledCuratedGoldViewPreflightFinding,
    OledCuratedGoldViewPreflightPolicy,
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
    check_oled_curated_gold_manifest_integrity as package_check_oled_curated_gold_manifest_integrity,
    load_oled_curated_gold_manifest_json as package_load_oled_curated_gold_manifest_json,
    load_oled_curated_gold_records_jsonl as package_load_oled_curated_gold_records_jsonl,
    run_oled_curated_gold_view_preflight as package_run_oled_curated_gold_view_preflight,
    run_oled_curated_gold_view_preflight_from_files as package_run_oled_curated_gold_view_preflight_from_files,
    sha256_file as package_sha256_file,
    write_oled_curated_gold_view_preflight_report_json as package_write_oled_curated_gold_view_preflight_report_json,
)
from ai4s_agent.domains.oled_curated_gold_view_preflight import (
    check_oled_curated_gold_manifest_integrity,
    load_oled_curated_gold_manifest_json,
    load_oled_curated_gold_records_jsonl,
    main,
    run_oled_curated_gold_view_preflight,
    run_oled_curated_gold_view_preflight_from_files,
    sha256_file,
    write_oled_curated_gold_view_preflight_report_json,
)


def _device_gold_record(
    record_id: str = "gold-view-valid",
    *,
    property_label: str = "EQE (%)",
    value: float = 19.5,
    unit: str = "%",
    evidence_refs: list[str] | None = None,
    raw_metadata: bool = False,
) -> OledGoldDatasetRecord:
    evidence_ref = (evidence_refs or [f"paper:{record_id}:table-1:row-1"])[0]
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
                        unit=unit,
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
            "candidate_only": False,
            "curated_dataset_written": True,
            "training_data_written": False,
            **({"raw_text": "full paper text should not appear"} if raw_metadata else {}),
        },
    )


def _manifest(
    output_sha256: str | None,
    *,
    output_jsonl_path: str | None = "curated_gold_records.jsonl",
) -> OledCuratedGoldManifest:
    return OledCuratedGoldManifest(
        manifest_id="oled-curated-gold-writer:test",
        input_candidate_count=1,
        output_record_count=1,
        output_jsonl_path=output_jsonl_path,
        output_sha256=output_sha256,
        policy=OledCuratedGoldWriterPolicy(),
        metadata={
            "curated_gold_writer": True,
            "training_data_written": False,
            "dataset_views_run": False,
        },
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


def test_load_curated_gold_records_jsonl_handles_valid_empty_invalid_and_missing(tmp_path: Path) -> None:
    path = _write_records(tmp_path / "curated_gold.jsonl", [_device_gold_record("gold-load")])
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    loaded = load_oled_curated_gold_records_jsonl(path)

    assert len(loaded) == 1
    assert loaded[0].record_id == "gold-load"

    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps(_device_gold_record().model_dump(mode="json")) + "\n{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_curated_gold_jsonl:line-2"):
        load_oled_curated_gold_records_jsonl(bad_path)
    with pytest.raises(ValueError, match="missing_curated_gold_jsonl:"):
        load_oled_curated_gold_records_jsonl(tmp_path / "missing.jsonl")


def test_load_curated_gold_manifest_json_handles_valid_missing_and_invalid(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.json", _manifest("abc123"))

    manifest = load_oled_curated_gold_manifest_json(manifest_path)

    assert manifest.manifest_id == "oled-curated-gold-writer:test"
    with pytest.raises(ValueError, match="missing_curated_gold_manifest:"):
        load_oled_curated_gold_manifest_json(tmp_path / "missing-manifest.json")
    bad_path = tmp_path / "bad-manifest.json"
    bad_path.write_text("{bad json}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_curated_gold_manifest_json:"):
        load_oled_curated_gold_manifest_json(bad_path)


def test_sha256_integrity_statuses(tmp_path: Path) -> None:
    jsonl_path = _write_records(tmp_path / "curated_gold.jsonl", [_device_gold_record("gold-sha")])
    digest = sha256_file(jsonl_path)

    matched_status, matched_findings, matched_digest = check_oled_curated_gold_manifest_integrity(
        input_jsonl_path=jsonl_path,
        manifest=_manifest(digest),
    )
    mismatched_status, mismatched_findings, _ = check_oled_curated_gold_manifest_integrity(
        input_jsonl_path=jsonl_path,
        manifest=_manifest("not-the-digest"),
    )
    missing_manifest_status, missing_manifest_findings, missing_manifest_digest = check_oled_curated_gold_manifest_integrity(
        input_jsonl_path=jsonl_path,
        manifest=None,
    )
    missing_sha_status, missing_sha_findings, _ = check_oled_curated_gold_manifest_integrity(
        input_jsonl_path=jsonl_path,
        manifest=_manifest(None),
    )

    assert matched_status == OledCuratedGoldManifestIntegrityStatus.MATCHED
    assert matched_findings == []
    assert matched_digest == digest
    assert mismatched_status == OledCuratedGoldManifestIntegrityStatus.MISMATCHED
    assert mismatched_findings[0].code == "manifest_sha256_mismatch"
    assert missing_manifest_status == OledCuratedGoldManifestIntegrityStatus.NOT_PROVIDED
    assert missing_manifest_findings == []
    assert missing_manifest_digest is None
    assert missing_sha_status == OledCuratedGoldManifestIntegrityStatus.MISSING_SHA256
    assert missing_sha_findings[0].code == "manifest_missing_sha256"


def test_gold_validation_errors_are_reported() -> None:
    invalid_record = _device_gold_record("gold-invalid", property_label="Not a known OLED property")

    report = run_oled_curated_gold_view_preflight([invalid_record])

    assert report.status == OledCuratedGoldViewPreflightStatus.FAILED
    assert "unknown_property_label" in report.gold_validation_error_codes
    assert "gold_validation_errors_present" in report.finding_code_counts


def test_dataset_view_preflight_builds_view_results() -> None:
    report = run_oled_curated_gold_view_preflight(
        [_device_gold_record("gold-view")],
        policy=OledCuratedGoldViewPreflightPolicy(
            target_property_ids=["eqe_percent"],
            view_kinds=["raw_all_measurements"],
        ),
    )

    assert report.is_valid
    assert report.view_results
    result = report.view_results[0]
    assert result.view_kind == "raw_all_measurements"
    assert result.target_property_id == "eqe_percent"
    assert result.row_count == 1
    assert result.status == OledCuratedGoldViewPreflightStatus.PASSED
    assert result.evidence_anchor_count >= 1
    assert result.layer_counts == {"measurement": 1}


def test_empty_view_behavior_is_controlled_by_policy() -> None:
    allowed = run_oled_curated_gold_view_preflight(
        [_device_gold_record("gold-empty-allowed")],
        policy=OledCuratedGoldViewPreflightPolicy(
            target_property_ids=["lifetime_hours"],
            view_kinds=["raw_all_measurements"],
            include_empty_views=True,
        ),
    )
    disallowed = run_oled_curated_gold_view_preflight(
        [_device_gold_record("gold-empty-disallowed")],
        policy=OledCuratedGoldViewPreflightPolicy(
            target_property_ids=["lifetime_hours"],
            view_kinds=["raw_all_measurements"],
            include_empty_views=False,
        ),
    )

    assert allowed.is_valid
    assert allowed.view_results[0].status == OledCuratedGoldViewPreflightStatus.PASSED_WITH_WARNINGS
    assert "empty_view" in allowed.view_results[0].reason_codes
    assert not disallowed.is_valid
    assert disallowed.view_results[0].status == OledCuratedGoldViewPreflightStatus.FAILED


def test_combined_runner_from_files_loads_manifest_and_writes_report(tmp_path: Path) -> None:
    jsonl_path = _write_records(tmp_path / "curated_gold.jsonl", [_device_gold_record("gold-files")])
    manifest_path = _write_manifest(tmp_path / "manifest.json", _manifest(sha256_file(jsonl_path)))
    report_path = tmp_path / "view_preflight_report.json"

    report = run_oled_curated_gold_view_preflight_from_files(
        curated_gold_jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        output_report_path=report_path,
        policy=OledCuratedGoldViewPreflightPolicy(
            target_property_ids=["eqe_percent"],
            view_kinds=["raw_all_measurements"],
        ),
    )

    assert report.is_valid
    assert report.manifest_integrity_status == OledCuratedGoldManifestIntegrityStatus.MATCHED
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["input_record_count"] == 1
    assert not (tmp_path / "dataset_view_rows.jsonl").exists()


def test_report_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_curated_gold_view_preflight(
        [_device_gold_record("gold-redacted", raw_metadata=True)],
        input_sha256="abc123",
        policy=OledCuratedGoldViewPreflightPolicy(
            target_property_ids=["eqe_percent"],
            view_kinds=["raw_all_measurements"],
        ),
    )
    path = tmp_path / "report.json"

    write_oled_curated_gold_view_preflight_report_json(report, path)
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2) + "\n"
    assert str(tmp_path) not in text
    assert "full paper text should not appear" not in text


def test_cli_smoke_outputs_compact_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jsonl_path = _write_records(tmp_path / "curated_gold.jsonl", [_device_gold_record("gold-cli")])
    manifest_path = _write_manifest(tmp_path / "manifest.json", _manifest(sha256_file(jsonl_path)))
    report_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--curated-gold-jsonl",
            str(jsonl_path),
            "--manifest",
            str(manifest_path),
            "--output-report",
            str(report_path),
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert report_path.exists()
    assert "layered_record" not in stdout
    assert json.loads(stdout)["input_record_count"] == 1


def test_public_curated_gold_view_preflight_api_is_exported_from_domain_package(tmp_path: Path) -> None:
    jsonl_path = _write_records(tmp_path / "package-curated.jsonl", [_device_gold_record("gold-package")])
    manifest_path = _write_manifest(tmp_path / "package-manifest.json", _manifest(package_sha256_file(jsonl_path)))
    report_path = tmp_path / "package-report.json"

    loaded = package_load_oled_curated_gold_records_jsonl(jsonl_path)
    manifest = package_load_oled_curated_gold_manifest_json(manifest_path)
    integrity_status, integrity_findings, input_sha = package_check_oled_curated_gold_manifest_integrity(
        input_jsonl_path=jsonl_path,
        manifest=manifest,
    )
    report = package_run_oled_curated_gold_view_preflight(
        loaded,
        manifest=manifest,
        input_sha256=input_sha,
        manifest_integrity_status=integrity_status,
        policy=OledCuratedGoldViewPreflightPolicy(
            target_property_ids=["eqe_percent"],
            view_kinds=["raw_all_measurements"],
        ),
    )
    file_report = package_run_oled_curated_gold_view_preflight_from_files(
        curated_gold_jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        output_report_path=report_path,
        policy=OledCuratedGoldViewPreflightPolicy(
            target_property_ids=["eqe_percent"],
            view_kinds=["raw_all_measurements"],
        ),
    )
    package_write_oled_curated_gold_view_preflight_report_json(file_report, report_path)

    assert integrity_status == OledCuratedGoldManifestIntegrityStatus.MATCHED
    assert integrity_findings == []
    assert isinstance(report, OledCuratedGoldViewPreflightReport)
    assert isinstance(report.view_results[0], OledCuratedGoldDatasetViewPreflightResult)
    assert isinstance(OledCuratedGoldViewPreflightFinding(code="x", message="y"), OledCuratedGoldViewPreflightFinding)
    assert OledCuratedGoldViewPreflightPolicy().target_property_ids == ["eqe_percent", "plqy", "delta_e_st_ev"]
    assert OledCuratedGoldViewPreflightStatus.PASSED.value == "passed"
    assert report_path.exists()
