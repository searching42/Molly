from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledBenchmarkPromotedRegistryEntry,
    OledBenchmarkPromotedRegistryEntryStatus,
    OledBenchmarkPromotedRegistryIndexRecord,
    OledBenchmarkRegistryPromotionFileResult,
    OledBenchmarkRegistryPromotionWriteStatus,
    OledBenchmarkRegistryPromotionWriterManifest,
    OledBenchmarkRegistryPromotionWriterPolicy,
    OledPromotedRegistryPublicationArtifactStatus,
    OledPromotedRegistryPublicationArtifactSummary,
    OledPromotedRegistryPublicationEntrySummary,
    OledPromotedRegistryPublicationPreflightFinding,
    OledPromotedRegistryPublicationPreflightPolicy,
    OledPromotedRegistryPublicationPreflightReport,
    OledPromotedRegistryPublicationPreflightStatus,
    load_oled_benchmark_registry_promotion_writer_manifest_json as package_load_manifest,
    load_oled_promoted_registry_artifacts_from_manifest as package_load_artifacts,
    run_oled_promoted_registry_publication_preflight as package_run_preflight,
    run_oled_promoted_registry_publication_preflight_from_files as package_run_from_files,
    write_oled_promoted_registry_publication_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_benchmark_registry_promotion_writer import (
    write_oled_benchmark_promoted_registry_entry_json,
    write_oled_benchmark_promoted_registry_index_jsonl,
    write_oled_benchmark_registry_promotion_manifest_json,
)
from ai4s_agent.domains.oled_curated_promoted_registry_publication_preflight import (
    load_oled_benchmark_registry_promotion_writer_manifest_json,
    load_oled_promoted_registry_artifacts_from_manifest,
    main,
    run_oled_promoted_registry_publication_preflight,
    run_oled_promoted_registry_publication_preflight_from_files,
    write_oled_promoted_registry_publication_preflight_report_json,
)


def _entry(
    *,
    promotion_status: str | OledBenchmarkPromotedRegistryEntryStatus = OledBenchmarkPromotedRegistryEntryStatus.PROMOTED_CANDIDATE,
    metadata: dict | None = None,
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
    source_registry_writer_manifest_id: str | None = "manifest:oled-benchmark-registry:test",
    source_registry_entry_id: str | None = "entry:oled-benchmark-registry:test",
    source_registry_promotion_preflight_status: str | None = "passed",
    source_candidate_report_id: str | None = "report:oled-baseline-benchmark:test",
    source_benchmark_report_manifest_id: str | None = "manifest:oled-baseline-benchmark:test",
) -> OledBenchmarkPromotedRegistryEntry:
    return OledBenchmarkPromotedRegistryEntry(
        promoted_entry_id="entry:oled-benchmark-promoted-registry:test",
        promotion_status=promotion_status,
        source_registry_writer_manifest_id=source_registry_writer_manifest_id,
        source_registry_entry_id=source_registry_entry_id,
        source_registry_promotion_preflight_status=source_registry_promotion_preflight_status,
        source_candidate_report_id=source_candidate_report_id,
        source_benchmark_report_manifest_id=source_benchmark_report_manifest_id,
        source_benchmark_registry_preflight_status="passed",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=run_card_count,
        metric_card_count=metric_card_count,
        source_registry_entry_json_path="oled_benchmark_registry_entry.json",
        source_registry_entry_json_sha256="source-entry-sha",
        source_registry_index_jsonl_path="oled_benchmark_registry_index.jsonl",
        source_registry_index_jsonl_sha256="source-index-sha",
        caveats=caveats
        if caveats is not None
        else ["baseline_candidate_report_only", "not_benchmark_validated", "not_scientific_performance_claim"],
        promotion_reason_codes=["selected_for_promotion"],
        metadata=metadata
        if metadata is not None
        else {
            "benchmark_registry_promotion_writer": True,
            "promotion_status": "promoted_candidate",
            "benchmark_validated": False,
            "scientific_claim_validated": False,
            "benchmark_published": False,
            "globally_registered": False,
        },
    )


def _index_record(
    *,
    promoted_entry_id: str = "entry:oled-benchmark-promoted-registry:test",
    promotion_status: str = "promoted_candidate",
    metadata: dict | None = None,
    output_sha256: str | None = "promoted-entry-sha",
) -> OledBenchmarkPromotedRegistryIndexRecord:
    return OledBenchmarkPromotedRegistryIndexRecord(
        promoted_entry_id=promoted_entry_id,
        promotion_status=promotion_status,
        source_registry_entry_id="entry:oled-benchmark-registry:test",
        source_registry_writer_manifest_id="manifest:oled-benchmark-registry:test",
        source_registry_promotion_preflight_status="passed",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=1,
        metric_card_count=3,
        output_promoted_entry_json_path="oled_benchmark_promoted_registry_entry.json",
        output_promoted_entry_json_sha256=output_sha256,
        benchmark_validated=False,
        scientific_claim_validated=False,
        metadata=metadata
        if metadata is not None
        else {
            "benchmark_promoted_registry_index_record": True,
            "promotion_status": "promoted_candidate",
            "benchmark_validated": False,
            "scientific_claim_validated": False,
            "benchmark_published": False,
            "globally_registered": False,
        },
    )


def _manifest(
    *,
    entry_sha: str | None = "promoted-entry-sha",
    index_sha: str | None = "promoted-index-sha",
    metadata: dict | None = None,
    source_registry_writer_manifest_id: str | None = "manifest:oled-benchmark-registry:test",
    source_registry_entry_id: str | None = "entry:oled-benchmark-registry:test",
    source_registry_promotion_preflight_status: str | None = "passed",
) -> OledBenchmarkRegistryPromotionWriterManifest:
    return OledBenchmarkRegistryPromotionWriterManifest(
        manifest_id="manifest:oled-benchmark-promoted-registry:test",
        source_registry_writer_manifest_id=source_registry_writer_manifest_id,
        source_registry_entry_id=source_registry_entry_id,
        source_registry_promotion_preflight_status=source_registry_promotion_preflight_status,
        output_directory="promoted_registry",
        output_file_count=2,
        promoted_entry_ids=["entry:oled-benchmark-promoted-registry:test"],
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledBenchmarkRegistryPromotionFileResult(
                artifact_kind="promoted_registry_entry_json",
                status=OledBenchmarkRegistryPromotionWriteStatus.WRITTEN,
                output_path="oled_benchmark_promoted_registry_entry.json",
                output_sha256=entry_sha,
                reason_codes=["promoted_registry_entry_json_written"],
            ),
            OledBenchmarkRegistryPromotionFileResult(
                artifact_kind="promoted_registry_index_jsonl",
                status=OledBenchmarkRegistryPromotionWriteStatus.WRITTEN,
                output_path="oled_benchmark_promoted_registry_index.jsonl",
                output_sha256=index_sha,
                reason_codes=["promoted_registry_index_jsonl_written"],
            ),
        ],
        policy=OledBenchmarkRegistryPromotionWriterPolicy(),
        metadata=metadata
        if metadata is not None
        else {
            "benchmark_registry_promotion_writer": True,
            "promotion_status": "promoted_candidate",
            "benchmark_validated": False,
            "scientific_claim_validated": False,
            "benchmark_published": False,
            "globally_registered": False,
        },
    )


def _write_promoted_package(tmp_path: Path) -> tuple[Path, OledBenchmarkRegistryPromotionWriterManifest, OledBenchmarkPromotedRegistryEntry, list[OledBenchmarkPromotedRegistryIndexRecord]]:
    entry = _entry()
    entry_path = tmp_path / "oled_benchmark_promoted_registry_entry.json"
    entry_sha = write_oled_benchmark_promoted_registry_entry_json(entry, entry_path)
    index = [_index_record(output_sha256=entry_sha)]
    index_path = tmp_path / "oled_benchmark_promoted_registry_index.jsonl"
    index_sha = write_oled_benchmark_promoted_registry_index_jsonl(index, index_path)
    manifest = _manifest(entry_sha=entry_sha, index_sha=index_sha)
    manifest_path = tmp_path / "promotion_writer_manifest.json"
    write_oled_benchmark_registry_promotion_manifest_json(manifest, manifest_path)
    return manifest_path, manifest, entry, index


def test_main_preflight_success() -> None:
    report = run_oled_promoted_registry_publication_preflight(
        promotion_writer_manifest=_manifest(),
        promoted_entry=_entry(),
        promoted_index_records=[_index_record()],
    )

    assert report.status == OledPromotedRegistryPublicationPreflightStatus.PASSED
    assert report.is_valid
    assert report.source_promoted_entry_id == "entry:oled-benchmark-promoted-registry:test"
    assert report.entry_summaries[0].artifact_status == OledPromotedRegistryPublicationArtifactStatus.READY
    assert report.metadata["promoted_registry_publication_preflight_only"] is True
    assert report.metadata["final_registry_written"] is False


def test_missing_promoted_entry_and_index() -> None:
    report = run_oled_promoted_registry_publication_preflight(
        promotion_writer_manifest=_manifest(),
        promoted_entry=None,
        promoted_index_records=[],
    )

    assert report.status == OledPromotedRegistryPublicationPreflightStatus.FAILED
    assert "missing_promoted_registry_entry_json" in report.error_codes
    assert "missing_promoted_registry_index_jsonl" in report.error_codes


def test_non_promoted_candidate_status_rejected() -> None:
    report = run_oled_promoted_registry_publication_preflight(
        promotion_writer_manifest=_manifest(),
        promoted_entry=_entry(promotion_status=OledBenchmarkPromotedRegistryEntryStatus.REJECTED),
        promoted_index_records=[_index_record(promotion_status="rejected")],
    )

    assert "promotion_status_not_promoted_candidate" in report.error_codes
    assert "index_status_not_promoted_candidate" in report.error_codes


def test_validation_and_publication_claims_rejected() -> None:
    metadata = {
        "benchmark_validated": True,
        "scientific_claim_validated": True,
        "benchmark_published": True,
        "globally_registered": True,
    }
    report = run_oled_promoted_registry_publication_preflight(
        promotion_writer_manifest=_manifest(metadata=metadata),
        promoted_entry=_entry(metadata=metadata),
        promoted_index_records=[_index_record(metadata=metadata)],
    )

    assert "benchmark_validated_source_claim" in report.error_codes
    assert "scientific_claim_validated_source_claim" in report.error_codes
    assert "publication_source_claim" in report.error_codes


def test_missing_source_ids() -> None:
    report = run_oled_promoted_registry_publication_preflight(
        promotion_writer_manifest=_manifest(
            source_registry_writer_manifest_id=None,
            source_registry_entry_id=None,
            source_registry_promotion_preflight_status=None,
        ),
        promoted_entry=_entry(
            source_registry_writer_manifest_id=None,
            source_registry_entry_id=None,
            source_registry_promotion_preflight_status=None,
            source_candidate_report_id=None,
            source_benchmark_report_manifest_id=None,
        ),
        promoted_index_records=[_index_record()],
    )

    codes = set(report.error_codes)
    assert "missing_source_registry_writer_manifest_id" in codes
    assert "missing_source_registry_entry_id" in codes
    assert "missing_source_registry_promotion_preflight_status" in codes
    assert "missing_source_candidate_report_id" in codes
    assert "missing_source_benchmark_report_manifest_id" in codes


def test_missing_caveats_run_cards_and_metric_cards() -> None:
    report = run_oled_promoted_registry_publication_preflight(
        promotion_writer_manifest=_manifest(),
        promoted_entry=_entry(caveats=["baseline_candidate_report_only"], run_card_count=0, metric_card_count=0),
        promoted_index_records=[_index_record()],
    )

    assert "missing_required_caveat" in report.error_codes
    assert "missing_run_cards" in report.error_codes
    assert "missing_metric_cards" in report.error_codes


def test_promoted_index_mismatch_and_multiple_records() -> None:
    entry = _entry()
    report = run_oled_promoted_registry_publication_preflight(
        promotion_writer_manifest=_manifest(),
        promoted_entry=entry,
        promoted_index_records=[
            _index_record(promoted_entry_id="entry:other"),
            _index_record(promoted_entry_id="entry:another"),
        ],
    )

    assert "promoted_entry_not_in_index" in report.error_codes
    assert "multiple_promoted_index_records" in report.error_codes

    allowed = run_oled_promoted_registry_publication_preflight(
        promotion_writer_manifest=_manifest(),
        promoted_entry=entry,
        promoted_index_records=[
            _index_record(promoted_entry_id=entry.promoted_entry_id),
            _index_record(promoted_entry_id="entry:another"),
        ],
        policy=OledPromotedRegistryPublicationPreflightPolicy(require_single_promoted_index_record=False),
    )
    assert "multiple_promoted_index_records" not in allowed.error_codes


def test_manifest_loader_and_artifact_loader_verify_sha(tmp_path: Path) -> None:
    manifest_path, manifest, entry, index = _write_promoted_package(tmp_path)

    loaded_manifest = load_oled_benchmark_registry_promotion_writer_manifest_json(manifest_path)
    loaded_entry, loaded_index = load_oled_promoted_registry_artifacts_from_manifest(
        manifest=loaded_manifest,
        base_dir=tmp_path,
    )

    assert loaded_manifest == manifest
    assert loaded_entry == entry
    assert loaded_index == index

    bad_manifest = manifest.model_copy(
        update={
            "file_results": [
                manifest.file_results[0].model_copy(update={"output_sha256": "bad"}),
                manifest.file_results[1],
            ]
        }
    )
    with pytest.raises(ValueError, match="promoted_registry_entry_sha256_mismatch:"):
        load_oled_promoted_registry_artifacts_from_manifest(manifest=bad_manifest, base_dir=tmp_path)

    bad_index_manifest = manifest.model_copy(
        update={
            "file_results": [
                manifest.file_results[0],
                manifest.file_results[1].model_copy(update={"output_sha256": "bad"}),
            ]
        }
    )
    with pytest.raises(ValueError, match="promoted_registry_index_sha256_mismatch:"):
        load_oled_promoted_registry_artifacts_from_manifest(manifest=bad_index_manifest, base_dir=tmp_path)


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    manifest_path, _, _, _ = _write_promoted_package(tmp_path)
    output_report = tmp_path / "publication_preflight.json"

    report = run_oled_promoted_registry_publication_preflight_from_files(
        promotion_writer_manifest_path=manifest_path,
        promoted_registry_base_dir=tmp_path,
        output_report_path=output_report,
    )

    assert report.is_valid
    assert output_report.exists()
    assert not (tmp_path / "final_registry.json").exists()
    assert not (tmp_path / "global_registry.jsonl").exists()


def test_report_writer_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_promoted_registry_publication_preflight(
        promotion_writer_manifest=_manifest(),
        promoted_entry=_entry(metadata={"raw_text": "paper text", "benchmark_validated": False}),
        promoted_index_records=[_index_record(metadata={"features": {"leak": 1}})],
    )
    path = tmp_path / "report.json"

    write_oled_promoted_registry_publication_preflight_report_json(report, path)
    first = path.read_text(encoding="utf-8")
    write_oled_promoted_registry_publication_preflight_report_json(report, path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert str(tmp_path) not in first
    assert "paper text" not in first
    assert "features" not in first
    assert "raw_text" not in first


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, _, _, _ = _write_promoted_package(tmp_path)
    output_report = tmp_path / "cli-publication-preflight.json"

    exit_code = main(
        [
            "--promotion-writer-manifest",
            str(manifest_path),
            "--promoted-registry-base-dir",
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
    assert OledPromotedRegistryPublicationPreflightStatus
    assert OledPromotedRegistryPublicationArtifactStatus
    assert OledPromotedRegistryPublicationPreflightPolicy
    assert OledPromotedRegistryPublicationArtifactSummary
    assert OledPromotedRegistryPublicationEntrySummary
    assert OledPromotedRegistryPublicationPreflightFinding
    assert OledPromotedRegistryPublicationPreflightReport
    assert package_load_manifest
    assert package_load_artifacts
    assert package_run_preflight
    assert package_run_from_files
    assert package_write_report
