from __future__ import annotations

from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledPromotedRegistryPublicationFileResult,
    OledPromotedRegistryPublicationWriteStatus,
    OledPromotedRegistryPublicationWriterManifest,
    OledPromotedRegistryPublicationWriterPolicy,
    OledPublicationCandidateFinalRegistryArtifactStatus,
    OledPublicationCandidateFinalRegistryArtifactSummary,
    OledPublicationCandidateFinalRegistryEntrySummary,
    OledPublicationCandidateFinalRegistryPreflightFinding,
    OledPublicationCandidateFinalRegistryPreflightPolicy,
    OledPublicationCandidateFinalRegistryPreflightReport,
    OledPublicationCandidateFinalRegistryPreflightStatus,
    OledPublicationCandidateRegistryEntry,
    OledPublicationCandidateRegistryEntryStatus,
    OledPublicationCandidateRegistryIndexRecord,
    load_oled_promoted_registry_publication_writer_manifest_json as package_load_manifest,
    load_oled_publication_candidate_registry_artifacts_from_manifest as package_load_artifacts,
    run_oled_publication_candidate_final_registry_preflight as package_run_preflight,
    run_oled_publication_candidate_final_registry_preflight_from_files as package_run_from_files,
    write_oled_publication_candidate_final_registry_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_promoted_registry_publication_writer import (
    oled_publication_candidate_registry_entry_filename,
    oled_publication_candidate_registry_index_filename,
    write_oled_promoted_registry_publication_manifest_json,
    write_oled_publication_candidate_registry_entry_json,
    write_oled_publication_candidate_registry_index_jsonl,
)
from ai4s_agent.domains.oled_curated_publication_candidate_final_registry_preflight import (
    load_oled_promoted_registry_publication_writer_manifest_json,
    load_oled_publication_candidate_registry_artifacts_from_manifest,
    main,
    run_oled_publication_candidate_final_registry_preflight,
    run_oled_publication_candidate_final_registry_preflight_from_files,
    write_oled_publication_candidate_final_registry_preflight_report_json,
)


def _entry(
    *,
    publication_status: str | OledPublicationCandidateRegistryEntryStatus = OledPublicationCandidateRegistryEntryStatus.PUBLICATION_CANDIDATE,
    metadata: dict | None = None,
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
    source_promotion_writer_manifest_id: str | None = "manifest:oled-publication-candidate:test",
    source_promoted_entry_id: str | None = "entry:oled-benchmark-promoted-registry:test",
    source_publication_preflight_status: str | None = "passed",
    source_registry_entry_id: str | None = "entry:oled-benchmark-registry:test",
    source_registry_writer_manifest_id: str | None = "manifest:oled-benchmark-registry:test",
    source_candidate_report_id: str | None = "report:oled-baseline-benchmark:test",
    source_benchmark_report_manifest_id: str | None = "manifest:oled-baseline-benchmark:test",
) -> OledPublicationCandidateRegistryEntry:
    return OledPublicationCandidateRegistryEntry(
        publication_entry_id="entry:oled-publication-candidate-registry:test",
        publication_status=publication_status,
        source_promotion_writer_manifest_id=source_promotion_writer_manifest_id,
        source_promoted_entry_id=source_promoted_entry_id,
        source_publication_preflight_status=source_publication_preflight_status,
        source_registry_entry_id=source_registry_entry_id,
        source_registry_writer_manifest_id=source_registry_writer_manifest_id,
        source_candidate_report_id=source_candidate_report_id,
        source_benchmark_report_manifest_id=source_benchmark_report_manifest_id,
        source_registry_promotion_preflight_status="passed",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=run_card_count,
        metric_card_count=metric_card_count,
        source_promoted_entry_json_path="oled_benchmark_promoted_registry_entry.json",
        source_promoted_entry_json_sha256="source-promoted-entry-sha",
        source_promoted_index_jsonl_path="oled_benchmark_promoted_registry_index.jsonl",
        source_promoted_index_jsonl_sha256="source-promoted-index-sha",
        caveats=caveats
        if caveats is not None
        else ["baseline_candidate_report_only", "not_benchmark_validated", "not_scientific_performance_claim"],
        publication_reason_codes=["selected_for_publication_candidate"],
        metadata=metadata
        if metadata is not None
        else {
            "promoted_registry_publication_writer": True,
            "publication_candidate_entry": True,
            "publication_status": "publication_candidate",
            "final_registry_written": False,
            "global_registry_mutated": False,
            "benchmark_published": False,
            "benchmark_registered": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _index_record(
    *,
    publication_entry_id: str = "entry:oled-publication-candidate-registry:test",
    publication_status: str = "publication_candidate",
    metadata: dict | None = None,
    output_sha256: str | None = "publication-entry-sha",
) -> OledPublicationCandidateRegistryIndexRecord:
    return OledPublicationCandidateRegistryIndexRecord(
        publication_entry_id=publication_entry_id,
        publication_status=publication_status,
        source_promoted_entry_id="entry:oled-benchmark-promoted-registry:test",
        source_promotion_writer_manifest_id="manifest:oled-publication-candidate:test",
        source_publication_preflight_status="passed",
        source_registry_entry_id="entry:oled-benchmark-registry:test",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=1,
        metric_card_count=3,
        output_publication_entry_json_path="oled_publication_candidate_registry_entry.json",
        output_publication_entry_json_sha256=output_sha256,
        benchmark_registered=False,
        benchmark_validated=False,
        scientific_claim_validated=False,
        metadata=metadata
        if metadata is not None
        else {
            "publication_candidate_registry_index_record": True,
            "publication_status": "publication_candidate",
            "benchmark_registered": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
            "final_registry_written": False,
        },
    )


def _manifest(
    *,
    entry_sha: str | None = "publication-entry-sha",
    index_sha: str | None = "publication-index-sha",
    metadata: dict | None = None,
    source_promotion_writer_manifest_id: str | None = "manifest:oled-benchmark-promoted-registry:test",
    source_promoted_entry_id: str | None = "entry:oled-benchmark-promoted-registry:test",
    source_publication_preflight_status: str | None = "passed",
) -> OledPromotedRegistryPublicationWriterManifest:
    return OledPromotedRegistryPublicationWriterManifest(
        manifest_id="manifest:oled-publication-candidate:test",
        source_promotion_writer_manifest_id=source_promotion_writer_manifest_id,
        source_promoted_entry_id=source_promoted_entry_id,
        source_publication_preflight_status=source_publication_preflight_status,
        output_directory="publication_candidate_registry",
        output_file_count=2,
        publication_entry_ids=["entry:oled-publication-candidate-registry:test"],
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledPromotedRegistryPublicationFileResult(
                artifact_kind="publication_candidate_entry_json",
                status=OledPromotedRegistryPublicationWriteStatus.WRITTEN,
                output_path=oled_publication_candidate_registry_entry_filename(),
                output_sha256=entry_sha,
                reason_codes=["publication_candidate_entry_json_written"],
            ),
            OledPromotedRegistryPublicationFileResult(
                artifact_kind="publication_candidate_index_jsonl",
                status=OledPromotedRegistryPublicationWriteStatus.WRITTEN,
                output_path=oled_publication_candidate_registry_index_filename(),
                output_sha256=index_sha,
                reason_codes=["publication_candidate_index_jsonl_written"],
            ),
        ],
        policy=OledPromotedRegistryPublicationWriterPolicy(),
        metadata=metadata
        if metadata is not None
        else {
            "promoted_registry_publication_writer": True,
            "publication_candidate_written": True,
            "publication_candidate_entry_written": True,
            "publication_candidate_index_written": True,
            "publication_status": "publication_candidate",
            "final_registry_written": False,
            "global_registry_mutated": False,
            "benchmark_published": False,
            "benchmark_registered": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _write_publication_package(
    tmp_path: Path,
) -> tuple[Path, OledPromotedRegistryPublicationWriterManifest, OledPublicationCandidateRegistryEntry, list[OledPublicationCandidateRegistryIndexRecord]]:
    entry = _entry()
    entry_path = tmp_path / oled_publication_candidate_registry_entry_filename()
    entry_sha = write_oled_publication_candidate_registry_entry_json(entry, entry_path)
    index = [_index_record(output_sha256=entry_sha)]
    index_path = tmp_path / oled_publication_candidate_registry_index_filename()
    index_sha = write_oled_publication_candidate_registry_index_jsonl(index, index_path)
    manifest = _manifest(entry_sha=entry_sha, index_sha=index_sha)
    manifest_path = tmp_path / "publication_manifest.json"
    write_oled_promoted_registry_publication_manifest_json(manifest, manifest_path)
    return manifest_path, manifest, entry, index


def test_main_preflight_success() -> None:
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_manifest(),
        publication_entry=_entry(),
        publication_index_records=[_index_record()],
    )

    assert report.status == OledPublicationCandidateFinalRegistryPreflightStatus.PASSED
    assert report.is_valid
    assert report.source_publication_entry_id == "entry:oled-publication-candidate-registry:test"
    assert report.entry_summaries[0].artifact_status == OledPublicationCandidateFinalRegistryArtifactStatus.READY
    assert report.metadata["publication_candidate_final_registry_preflight_only"] is True
    assert report.metadata["final_registry_written"] is False


def test_missing_publication_entry_and_index() -> None:
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_manifest(),
        publication_entry=None,
        publication_index_records=[],
    )

    assert report.status == OledPublicationCandidateFinalRegistryPreflightStatus.FAILED
    assert "missing_publication_candidate_registry_entry_json" in report.error_codes
    assert "missing_publication_candidate_registry_index_jsonl" in report.error_codes


def test_non_publication_candidate_status_rejected() -> None:
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_manifest(),
        publication_entry=_entry(publication_status=OledPublicationCandidateRegistryEntryStatus.REJECTED),
        publication_index_records=[_index_record(publication_status="rejected")],
    )

    assert report.status == OledPublicationCandidateFinalRegistryPreflightStatus.FAILED
    assert "publication_status_not_publication_candidate" in report.error_codes
    assert "index_status_not_publication_candidate" in report.error_codes


def test_validation_and_final_registry_claims_rejected() -> None:
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_manifest(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "final_registry_written": True}
        ),
        publication_entry=_entry(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "global_registry_mutated": True}
        ),
        publication_index_records=[
            _index_record(metadata={"benchmark_validated": True, "scientific_claim_validated": True, "benchmark_registered": True})
        ],
    )

    assert report.status == OledPublicationCandidateFinalRegistryPreflightStatus.FAILED
    assert "benchmark_validated_source_claim" in report.error_codes
    assert "scientific_claim_validated_source_claim" in report.error_codes
    assert "final_registry_source_claim" in report.error_codes


def test_missing_source_ids() -> None:
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_manifest(
            source_promotion_writer_manifest_id=None,
            source_promoted_entry_id=None,
            source_publication_preflight_status=None,
        ),
        publication_entry=_entry(
            source_promotion_writer_manifest_id=None,
            source_promoted_entry_id=None,
            source_publication_preflight_status=None,
            source_registry_entry_id=None,
            source_registry_writer_manifest_id=None,
            source_candidate_report_id=None,
            source_benchmark_report_manifest_id=None,
        ),
        publication_index_records=[_index_record()],
    )

    assert report.status == OledPublicationCandidateFinalRegistryPreflightStatus.FAILED
    for code in (
        "missing_source_promotion_writer_manifest_id",
        "missing_source_promoted_entry_id",
        "missing_source_publication_preflight_status",
        "missing_source_registry_entry_id",
        "missing_source_registry_writer_manifest_id",
        "missing_source_candidate_report_id",
        "missing_source_benchmark_report_manifest_id",
    ):
        assert code in report.error_codes


def test_missing_caveats_cards() -> None:
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_manifest(),
        publication_entry=_entry(caveats=["baseline_candidate_report_only"], run_card_count=0, metric_card_count=0),
        publication_index_records=[_index_record()],
    )

    assert report.status == OledPublicationCandidateFinalRegistryPreflightStatus.FAILED
    assert "missing_required_caveat" in report.error_codes
    assert "missing_run_cards" in report.error_codes
    assert "missing_metric_cards" in report.error_codes


def test_publication_index_mismatch_and_multiple_records() -> None:
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_manifest(),
        publication_entry=_entry(),
        publication_index_records=[
            _index_record(publication_entry_id="entry:other"),
            _index_record(publication_entry_id="entry:another"),
        ],
    )

    assert report.status == OledPublicationCandidateFinalRegistryPreflightStatus.FAILED
    assert "publication_entry_not_in_index" in report.error_codes
    assert "multiple_publication_index_records" in report.error_codes

    allowed = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_manifest(),
        publication_entry=_entry(),
        publication_index_records=[_index_record(), _index_record()],
        policy=OledPublicationCandidateFinalRegistryPreflightPolicy(require_single_publication_index_record=False),
    )
    assert "multiple_publication_index_records" not in allowed.error_codes


def test_manifest_loader_and_artifact_loader(tmp_path: Path) -> None:
    manifest_path, _, _, _ = _write_publication_package(tmp_path)

    manifest = load_oled_promoted_registry_publication_writer_manifest_json(manifest_path)
    entry, index = load_oled_publication_candidate_registry_artifacts_from_manifest(manifest=manifest, base_dir=tmp_path)

    assert manifest.manifest_id == "manifest:oled-publication-candidate:test"
    assert entry is not None
    assert entry.publication_entry_id == "entry:oled-publication-candidate-registry:test"
    assert len(index) == 1

    bad_manifest = manifest.model_copy(
        update={
            "file_results": [
                manifest.file_results[0].model_copy(update={"output_sha256": "bad-sha"}),
                manifest.file_results[1],
            ]
        }
    )
    with pytest.raises(ValueError, match="publication_candidate_entry_sha256_mismatch:"):
        load_oled_publication_candidate_registry_artifacts_from_manifest(manifest=bad_manifest, base_dir=tmp_path)


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    manifest_path, _, _, _ = _write_publication_package(tmp_path)
    output_report = tmp_path / "final_registry_preflight.json"

    report = run_oled_publication_candidate_final_registry_preflight_from_files(
        publication_writer_manifest_path=manifest_path,
        publication_candidate_base_dir=tmp_path,
        output_report_path=output_report,
    )

    assert report.is_valid
    assert output_report.exists()
    assert not (tmp_path / "final_registry.json").exists()


def test_report_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_manifest(),
        publication_entry=_entry(),
        publication_index_records=[_index_record()],
    )
    path = tmp_path / "report.json"

    write_oled_publication_candidate_final_registry_preflight_report_json(report, path)
    first = path.read_text(encoding="utf-8")
    write_oled_publication_candidate_final_registry_preflight_report_json(report, path)

    assert first == path.read_text(encoding="utf-8")
    assert str(tmp_path) not in first
    assert "raw_text" not in first
    assert "features" not in first
    assert "prediction_id" not in first


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, _, _, _ = _write_publication_package(tmp_path)
    output_report = tmp_path / "cli-final-registry-preflight.json"

    exit_code = main(
        [
            "--publication-writer-manifest",
            str(manifest_path),
            "--publication-candidate-base-dir",
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
    assert OledPublicationCandidateFinalRegistryPreflightStatus
    assert OledPublicationCandidateFinalRegistryArtifactStatus
    assert OledPublicationCandidateFinalRegistryPreflightPolicy
    assert OledPublicationCandidateFinalRegistryArtifactSummary
    assert OledPublicationCandidateFinalRegistryEntrySummary
    assert OledPublicationCandidateFinalRegistryPreflightFinding
    assert OledPublicationCandidateFinalRegistryPreflightReport
    assert package_load_manifest
    assert package_load_artifacts
    assert package_run_preflight
    assert package_run_from_files
    assert package_write_report
