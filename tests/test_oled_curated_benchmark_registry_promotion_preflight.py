from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.domains import (
    OledBenchmarkRegistryEntry,
    OledBenchmarkRegistryEntryStatus,
    OledBenchmarkRegistryFileResult,
    OledBenchmarkRegistryIndexRecord,
    OledBenchmarkRegistryPromotionArtifactStatus,
    OledBenchmarkRegistryPromotionEntrySummary,
    OledBenchmarkRegistryPromotionPreflightFinding,
    OledBenchmarkRegistryPromotionPreflightPolicy,
    OledBenchmarkRegistryPromotionPreflightReport,
    OledBenchmarkRegistryPromotionPreflightStatus,
    OledBenchmarkRegistryPromotionArtifactSummary,
    OledBenchmarkRegistryWriteStatus,
    OledBenchmarkRegistryWriterManifest,
    OledBenchmarkRegistryWriterPolicy,
    load_oled_benchmark_registry_artifacts_from_manifest as package_load_artifacts,
    load_oled_benchmark_registry_writer_manifest_json as package_load_manifest,
    run_oled_benchmark_registry_promotion_preflight as package_run_preflight,
    run_oled_benchmark_registry_promotion_preflight_from_files as package_run_from_files,
    write_oled_benchmark_registry_entry_json,
    write_oled_benchmark_registry_index_jsonl,
    write_oled_benchmark_registry_manifest_json,
    write_oled_benchmark_registry_promotion_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_benchmark_registry_promotion_preflight import (
    load_oled_benchmark_registry_artifacts_from_manifest,
    load_oled_benchmark_registry_writer_manifest_json,
    main,
    run_oled_benchmark_registry_promotion_preflight,
    run_oled_benchmark_registry_promotion_preflight_from_files,
    write_oled_benchmark_registry_promotion_preflight_report_json,
)


def _entry(
    *,
    registry_status: str | OledBenchmarkRegistryEntryStatus = OledBenchmarkRegistryEntryStatus.CANDIDATE,
    metadata: dict | None = None,
    source_candidate_report_id: str | None = "report:oled-baseline-benchmark:test",
    source_benchmark_report_manifest_id: str | None = "manifest:oled-baseline-benchmark:test",
    source_benchmark_registry_preflight_status: str | None = "passed",
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
) -> OledBenchmarkRegistryEntry:
    return OledBenchmarkRegistryEntry(
        registry_entry_id="entry:oled-benchmark-registry:test",
        registry_status=registry_status,
        source_benchmark_report_manifest_id=source_benchmark_report_manifest_id,
        source_benchmark_registry_preflight_status=source_benchmark_registry_preflight_status,
        source_candidate_report_id=source_candidate_report_id,
        source_baseline_run_manifest_id="oled-baseline-runner:test",
        source_benchmark_preflight_status="passed",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=run_card_count,
        metric_card_count=metric_card_count,
        report_json_path="oled_baseline_benchmark_candidate_report.json",
        report_json_sha256="report-json-sha",
        report_markdown_path="oled_baseline_benchmark_candidate_report.md",
        report_markdown_sha256="report-md-sha",
        caveats=caveats
        if caveats is not None
        else ["baseline_candidate_report_only", "not_benchmark_validated", "not_scientific_performance_claim"],
        reason_codes=["selected_for_registry"],
        metadata=metadata
        if metadata is not None
        else {
            "benchmark_registry_entry": True,
            "benchmark_registry_writer": True,
            "registry_status": "candidate",
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _index_record(
    *,
    registry_entry_id: str = "entry:oled-benchmark-registry:test",
    registry_status: str = "candidate",
    metadata: dict | None = None,
    output_path: str | None = "oled_benchmark_registry_entry.json",
    output_sha256: str | None = "entry-sha",
) -> OledBenchmarkRegistryIndexRecord:
    return OledBenchmarkRegistryIndexRecord(
        registry_entry_id=registry_entry_id,
        registry_status=registry_status,
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        source_benchmark_registry_preflight_status="passed",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=1,
        metric_card_count=3,
        output_registry_entry_json_path=output_path,
        output_registry_entry_json_sha256=output_sha256,
        benchmark_validated=False,
        scientific_claim_validated=False,
        metadata=metadata
        if metadata is not None
        else {
            "benchmark_registry_index_record": True,
            "registry_status": "candidate",
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _manifest(
    *,
    entry_sha: str | None = "entry-sha",
    index_sha: str | None = "index-sha",
    metadata: dict | None = None,
) -> OledBenchmarkRegistryWriterManifest:
    return OledBenchmarkRegistryWriterManifest(
        manifest_id="manifest:oled-benchmark-registry:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        source_benchmark_registry_preflight_status="passed",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        output_directory="benchmark_registry",
        output_file_count=2,
        registry_entry_ids=["entry:oled-benchmark-registry:test"],
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledBenchmarkRegistryFileResult(
                artifact_kind="registry_entry_json",
                status=OledBenchmarkRegistryWriteStatus.WRITTEN,
                output_path="oled_benchmark_registry_entry.json",
                output_sha256=entry_sha,
                reason_codes=["registry_entry_json_written"],
            ),
            OledBenchmarkRegistryFileResult(
                artifact_kind="registry_index_jsonl",
                status=OledBenchmarkRegistryWriteStatus.WRITTEN,
                output_path="oled_benchmark_registry_index.jsonl",
                output_sha256=index_sha,
                reason_codes=["registry_index_jsonl_written"],
            ),
        ],
        policy=OledBenchmarkRegistryWriterPolicy(),
        metadata=metadata
        if metadata is not None
        else {
            "benchmark_registry_writer": True,
            "benchmark_registry_entry_written": True,
            "benchmark_registry_index_written": True,
            "registry_status": "candidate",
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _write_registry_package(tmp_path: Path):
    entry = _entry()
    entry_path = tmp_path / "oled_benchmark_registry_entry.json"
    entry_sha = write_oled_benchmark_registry_entry_json(entry, entry_path)
    index = [_index_record(output_sha256=entry_sha)]
    index_path = tmp_path / "oled_benchmark_registry_index.jsonl"
    index_sha = write_oled_benchmark_registry_index_jsonl(index, index_path)
    manifest = _manifest(entry_sha=entry_sha, index_sha=index_sha)
    manifest_path = tmp_path / "benchmark_registry_manifest.json"
    write_oled_benchmark_registry_manifest_json(manifest, manifest_path)
    return manifest_path, manifest, entry, index


def test_main_preflight_success() -> None:
    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record()],
    )

    assert report.status == OledBenchmarkRegistryPromotionPreflightStatus.PASSED
    assert report.is_valid
    assert report.input_registry_entry_count == 1
    assert report.input_registry_index_record_count == 1
    assert report.entry_summaries[0].artifact_status == OledBenchmarkRegistryPromotionArtifactStatus.READY
    assert report.metadata["benchmark_registry_promotion_preflight_only"] is True
    assert report.metadata["benchmark_promotion_written"] is False


def test_missing_entry_and_index_fail() -> None:
    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=_manifest(),
        registry_entry=None,
        registry_index_records=[],
    )

    assert not report.is_valid
    assert "missing_benchmark_registry_entry_json" in report.error_codes
    assert "missing_benchmark_registry_index_jsonl" in report.error_codes


def test_non_candidate_status_rejected() -> None:
    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(registry_status=OledBenchmarkRegistryEntryStatus.REJECTED),
        registry_index_records=[_index_record(registry_status="rejected")],
    )

    assert "registry_status_not_candidate" in report.error_codes
    assert "index_status_not_candidate" in report.error_codes


def test_validation_claims_rejected() -> None:
    manifest = _manifest(metadata={"benchmark_validated": True, "scientific_claim_validated": True})
    entry = _entry(metadata={"benchmark_validated": True, "scientific_claim_validated": True})
    index = [_index_record(metadata={"benchmark_validated": True, "scientific_claim_validated": True})]

    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=manifest,
        registry_entry=entry,
        registry_index_records=index,
    )

    assert "benchmark_validated_source_claim" in report.error_codes
    assert "scientific_claim_validated_source_claim" in report.error_codes


def test_missing_source_ids() -> None:
    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(
            source_candidate_report_id=None,
            source_benchmark_report_manifest_id=None,
            source_benchmark_registry_preflight_status=None,
        ),
        registry_index_records=[_index_record()],
    )

    assert "missing_source_candidate_report_id" in report.error_codes
    assert "missing_source_benchmark_report_manifest_id" in report.error_codes
    assert "missing_source_benchmark_registry_preflight_status" in report.error_codes


def test_missing_caveats_and_cards() -> None:
    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(caveats=["baseline_candidate_report_only"], run_card_count=0, metric_card_count=0),
        registry_index_records=[_index_record()],
    )

    assert "missing_required_caveat" in report.error_codes
    assert "missing_run_cards" in report.error_codes
    assert "missing_metric_cards" in report.error_codes


def test_index_mismatch_and_multiple_index_records() -> None:
    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record(registry_entry_id="different-entry"), _index_record(registry_entry_id="another-entry")],
    )

    assert "registry_entry_not_in_index" in report.error_codes
    assert "multiple_index_records" in report.error_codes

    allowed = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record(), _index_record(registry_entry_id="another-entry")],
        policy=OledBenchmarkRegistryPromotionPreflightPolicy(require_single_entry_index=False),
    )
    assert "multiple_index_records" not in allowed.error_codes


def test_manifest_loader_and_artifact_loader(tmp_path: Path) -> None:
    manifest_path, manifest, entry, index = _write_registry_package(tmp_path)

    loaded_manifest = load_oled_benchmark_registry_writer_manifest_json(manifest_path)
    loaded_entry, loaded_index = load_oled_benchmark_registry_artifacts_from_manifest(
        manifest=loaded_manifest,
        base_dir=tmp_path,
    )

    assert loaded_manifest.manifest_id == manifest.manifest_id
    assert loaded_entry == entry
    assert loaded_index == index

    bad_manifest = manifest.model_copy(
        update={
            "file_results": [
                manifest.file_results[0].model_copy(update={"output_sha256": "bad-sha"}),
                manifest.file_results[1],
            ]
        }
    )
    try:
        load_oled_benchmark_registry_artifacts_from_manifest(manifest=bad_manifest, base_dir=tmp_path)
    except ValueError as exc:
        assert "benchmark_registry_entry_sha256_mismatch:" in str(exc)
    else:
        raise AssertionError("expected entry SHA mismatch")

    bad_manifest = manifest.model_copy(
        update={
            "file_results": [
                manifest.file_results[0],
                manifest.file_results[1].model_copy(update={"output_sha256": "bad-sha"}),
            ]
        }
    )
    try:
        load_oled_benchmark_registry_artifacts_from_manifest(manifest=bad_manifest, base_dir=tmp_path)
    except ValueError as exc:
        assert "benchmark_registry_index_sha256_mismatch:" in str(exc)
    else:
        raise AssertionError("expected index SHA mismatch")


def test_combined_runner_from_files(tmp_path: Path) -> None:
    manifest_path, _, _, _ = _write_registry_package(tmp_path)
    output_report = tmp_path / "promotion_preflight.json"

    report = run_oled_benchmark_registry_promotion_preflight_from_files(
        registry_writer_manifest_path=manifest_path,
        registry_base_dir=tmp_path,
        output_report_path=output_report,
    )

    assert report.is_valid
    assert output_report.exists()
    assert not (tmp_path / "promoted_registry.json").exists()


def test_report_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(metadata={"note": str(tmp_path), "benchmark_validated": False}),
        registry_index_records=[_index_record()],
    )
    path = tmp_path / "report.json"

    write_oled_benchmark_registry_promotion_preflight_report_json(report, path)
    first = path.read_text(encoding="utf-8")
    write_oled_benchmark_registry_promotion_preflight_report_json(report, path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert str(tmp_path) not in first
    assert "raw_text" not in first
    assert "features" not in first
    assert "prediction_id" not in first


def test_cli_smoke(tmp_path: Path, capsys) -> None:
    manifest_path, _, _, _ = _write_registry_package(tmp_path)
    output_report = tmp_path / "cli_report.json"

    exit_code = main(
        [
            "--registry-writer-manifest",
            str(manifest_path),
            "--registry-base-dir",
            str(tmp_path),
            "--output-report",
            str(output_report),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_report.exists()
    assert "prediction_id" not in captured.out
    assert "features" not in captured.out


def test_package_exports() -> None:
    assert OledBenchmarkRegistryPromotionPreflightStatus
    assert OledBenchmarkRegistryPromotionArtifactStatus
    assert OledBenchmarkRegistryPromotionPreflightPolicy
    assert OledBenchmarkRegistryPromotionArtifactSummary
    assert OledBenchmarkRegistryPromotionEntrySummary
    assert OledBenchmarkRegistryPromotionPreflightFinding
    assert OledBenchmarkRegistryPromotionPreflightReport
    assert package_load_manifest
    assert package_load_artifacts
    assert package_run_preflight
    assert package_run_from_files
    assert package_write_report
