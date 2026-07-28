from __future__ import annotations

from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledFinalRegistryCandidateEntry,
    OledFinalRegistryCandidateEntryStatus,
    OledFinalRegistryCandidateIndexRecord,
    OledFinalRegistryExistingRecordSummary,
    OledFinalRegistryGlobalAppendArtifactStatus,
    OledFinalRegistryGlobalAppendArtifactSummary,
    OledFinalRegistryGlobalAppendEntrySummary,
    OledFinalRegistryGlobalAppendPreflightFinding,
    OledFinalRegistryGlobalAppendPreflightPolicy,
    OledFinalRegistryGlobalAppendPreflightReport,
    OledFinalRegistryGlobalAppendPreflightStatus,
    OledPublicationCandidateFinalRegistryFileResult,
    OledPublicationCandidateFinalRegistryWriteStatus,
    OledPublicationCandidateFinalRegistryWriterManifest,
    OledPublicationCandidateFinalRegistryWriterPolicy,
    load_oled_existing_final_registry_snapshot_jsonl as package_load_existing_snapshot,
    load_oled_final_registry_candidate_artifacts_from_manifest as package_load_artifacts,
    load_oled_publication_candidate_final_registry_writer_manifest_json as package_load_manifest,
    run_oled_final_registry_global_append_preflight as package_run_preflight,
    run_oled_final_registry_global_append_preflight_from_files as package_run_from_files,
    write_oled_final_registry_global_append_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_publication_candidate_final_registry_writer import (
    oled_final_registry_candidate_entry_filename,
    oled_final_registry_candidate_index_filename,
    write_oled_final_registry_candidate_entry_json,
    write_oled_final_registry_candidate_index_jsonl,
    write_oled_publication_candidate_final_registry_manifest_json,
)
from ai4s_agent.domains.oled_curated_final_registry_global_append_preflight import (
    load_oled_existing_final_registry_snapshot_jsonl,
    load_oled_final_registry_candidate_artifacts_from_manifest,
    load_oled_publication_candidate_final_registry_writer_manifest_json,
    main,
    run_oled_final_registry_global_append_preflight,
    run_oled_final_registry_global_append_preflight_from_files,
    write_oled_final_registry_global_append_preflight_report_json,
)


def _entry(
    *,
    final_registry_status: str | OledFinalRegistryCandidateEntryStatus = OledFinalRegistryCandidateEntryStatus.FINAL_REGISTRY_CANDIDATE,
    metadata: dict | None = None,
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
    source_publication_writer_manifest_id: str | None = "manifest:oled-publication-candidate-final-registry:test",
    source_publication_entry_id: str | None = "entry:oled-publication-candidate-registry:test",
    source_final_registry_preflight_status: str | None = "passed",
    source_promoted_entry_id: str | None = "entry:oled-benchmark-promoted-registry:test",
    source_promotion_writer_manifest_id: str | None = "manifest:oled-benchmark-promoted-registry:test",
    source_registry_entry_id: str | None = "entry:oled-benchmark-registry:test",
    source_registry_writer_manifest_id: str | None = "manifest:oled-benchmark-registry:test",
    source_candidate_report_id: str | None = "report:oled-baseline-benchmark:test",
    source_benchmark_report_manifest_id: str | None = "manifest:oled-baseline-benchmark:test",
) -> OledFinalRegistryCandidateEntry:
    return OledFinalRegistryCandidateEntry(
        final_registry_entry_id="entry:oled-final-registry-candidate:test",
        final_registry_status=final_registry_status,
        source_publication_writer_manifest_id=source_publication_writer_manifest_id,
        source_publication_entry_id=source_publication_entry_id,
        source_final_registry_preflight_status=source_final_registry_preflight_status,
        source_promoted_entry_id=source_promoted_entry_id,
        source_promotion_writer_manifest_id=source_promotion_writer_manifest_id,
        source_registry_entry_id=source_registry_entry_id,
        source_registry_writer_manifest_id=source_registry_writer_manifest_id,
        source_candidate_report_id=source_candidate_report_id,
        source_benchmark_report_manifest_id=source_benchmark_report_manifest_id,
        source_publication_preflight_status="passed",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=run_card_count,
        metric_card_count=metric_card_count,
        source_publication_entry_json_path="oled_publication_candidate_registry_entry.json",
        source_publication_entry_json_sha256="source-publication-entry-sha",
        source_publication_index_jsonl_path="oled_publication_candidate_registry_index.jsonl",
        source_publication_index_jsonl_sha256="source-publication-index-sha",
        caveats=caveats
        if caveats is not None
        else ["baseline_candidate_report_only", "not_benchmark_validated", "not_scientific_performance_claim"],
        final_registry_reason_codes=["selected_for_final_registry_candidate"],
        metadata=metadata
        if metadata is not None
        else {
            "publication_candidate_final_registry_writer": True,
            "final_registry_candidate_entry": True,
            "final_registry_status": "final_registry_candidate",
            "global_registry_mutated": False,
            "benchmark_published": False,
            "benchmark_registered": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _index_record(
    *,
    final_registry_entry_id: str = "entry:oled-final-registry-candidate:test",
    final_registry_status: str = "final_registry_candidate",
    metadata: dict | None = None,
    output_sha256: str | None = "final-registry-entry-sha",
) -> OledFinalRegistryCandidateIndexRecord:
    return OledFinalRegistryCandidateIndexRecord(
        final_registry_entry_id=final_registry_entry_id,
        final_registry_status=final_registry_status,
        source_publication_entry_id="entry:oled-publication-candidate-registry:test",
        source_publication_writer_manifest_id="manifest:oled-publication-candidate-final-registry:test",
        source_final_registry_preflight_status="passed",
        source_promoted_entry_id="entry:oled-benchmark-promoted-registry:test",
        source_registry_entry_id="entry:oled-benchmark-registry:test",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=1,
        metric_card_count=3,
        output_final_registry_entry_json_path=oled_final_registry_candidate_entry_filename(),
        output_final_registry_entry_json_sha256=output_sha256,
        benchmark_published=False,
        benchmark_registered=False,
        benchmark_validated=False,
        scientific_claim_validated=False,
        metadata=metadata
        if metadata is not None
        else {
            "final_registry_candidate_index_record": True,
            "final_registry_status": "final_registry_candidate",
            "global_registry_mutated": False,
            "benchmark_published": False,
            "benchmark_registered": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _manifest(
    *,
    entry_sha: str | None = "final-registry-entry-sha",
    index_sha: str | None = "final-registry-index-sha",
    metadata: dict | None = None,
    source_publication_writer_manifest_id: str | None = "manifest:oled-publication-candidate-final-registry:test",
    source_publication_entry_id: str | None = "entry:oled-publication-candidate-registry:test",
    source_final_registry_preflight_status: str | None = "passed",
) -> OledPublicationCandidateFinalRegistryWriterManifest:
    return OledPublicationCandidateFinalRegistryWriterManifest(
        manifest_id="manifest:oled-final-registry-candidate:test",
        source_publication_writer_manifest_id=source_publication_writer_manifest_id,
        source_publication_entry_id=source_publication_entry_id,
        source_final_registry_preflight_status=source_final_registry_preflight_status,
        output_directory="final_registry_candidate",
        output_file_count=2,
        final_registry_entry_ids=["entry:oled-final-registry-candidate:test"],
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledPublicationCandidateFinalRegistryFileResult(
                artifact_kind="final_registry_candidate_entry_json",
                status=OledPublicationCandidateFinalRegistryWriteStatus.WRITTEN,
                output_path=oled_final_registry_candidate_entry_filename(),
                output_sha256=entry_sha,
                reason_codes=["final_registry_candidate_entry_json_written"],
            ),
            OledPublicationCandidateFinalRegistryFileResult(
                artifact_kind="final_registry_candidate_index_jsonl",
                status=OledPublicationCandidateFinalRegistryWriteStatus.WRITTEN,
                output_path=oled_final_registry_candidate_index_filename(),
                output_sha256=index_sha,
                reason_codes=["final_registry_candidate_index_jsonl_written"],
            ),
        ],
        policy=OledPublicationCandidateFinalRegistryWriterPolicy(),
        metadata=metadata
        if metadata is not None
        else {
            "publication_candidate_final_registry_writer": True,
            "final_registry_candidate_written": True,
            "final_registry_candidate_entry_written": True,
            "final_registry_candidate_index_written": True,
            "final_registry_status": "final_registry_candidate",
            "global_registry_mutated": False,
            "benchmark_published": False,
            "benchmark_registered": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _existing(
    *,
    registry_entry_id: str | None = "entry:existing-final-registry:other",
    source_publication_entry_id: str | None = "entry:other-publication",
    source_publication_writer_manifest_id: str | None = "manifest:other-publication-writer",
    source_candidate_report_id: str | None = "report:other",
    source_benchmark_report_manifest_id: str | None = "manifest:other-benchmark",
    metadata: dict | None = None,
) -> OledFinalRegistryExistingRecordSummary:
    return OledFinalRegistryExistingRecordSummary(
        registry_entry_id=registry_entry_id,
        registry_status="global_candidate",
        source_publication_entry_id=source_publication_entry_id,
        source_publication_writer_manifest_id=source_publication_writer_manifest_id,
        source_candidate_report_id=source_candidate_report_id,
        source_benchmark_report_manifest_id=source_benchmark_report_manifest_id,
        metadata=metadata if metadata is not None else {"existing_registry_record": True},
    )


def _write_final_registry_package(
    tmp_path: Path,
) -> tuple[Path, OledPublicationCandidateFinalRegistryWriterManifest, OledFinalRegistryCandidateEntry, list[OledFinalRegistryCandidateIndexRecord]]:
    entry = _entry()
    entry_path = tmp_path / oled_final_registry_candidate_entry_filename()
    entry_sha = write_oled_final_registry_candidate_entry_json(entry, entry_path)
    index = [_index_record(output_sha256=entry_sha)]
    index_path = tmp_path / oled_final_registry_candidate_index_filename()
    index_sha = write_oled_final_registry_candidate_index_jsonl(index, index_path)
    manifest = _manifest(entry_sha=entry_sha, index_sha=index_sha)
    manifest_path = tmp_path / "final_registry_manifest.json"
    write_oled_publication_candidate_final_registry_manifest_json(manifest, manifest_path)
    return manifest_path, manifest, entry, index


def test_main_preflight_success_without_existing_registry_snapshot() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
    )

    assert report.status == OledFinalRegistryGlobalAppendPreflightStatus.PASSED
    assert report.is_valid
    assert report.source_final_registry_entry_id == "entry:oled-final-registry-candidate:test"
    assert report.entry_summaries[0].artifact_status == OledFinalRegistryGlobalAppendArtifactStatus.READY
    assert report.metadata["final_registry_global_append_preflight_only"] is True
    assert report.metadata["global_registry_mutated"] is False


def test_main_preflight_success_with_non_conflicting_existing_snapshot() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        existing_registry_records=[_existing()],
    )

    assert report.status == OledFinalRegistryGlobalAppendPreflightStatus.PASSED
    assert report.input_existing_registry_record_count == 1
    assert report.entry_summaries[0].existing_registry_duplicate_entry_count == 0
    assert report.entry_summaries[0].existing_registry_duplicate_source_chain_count == 0


def test_missing_final_registry_candidate_entry_and_index() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=None,
        final_registry_index_records=[],
    )

    assert report.status == OledFinalRegistryGlobalAppendPreflightStatus.FAILED
    assert "missing_final_registry_candidate_entry_json" in report.error_codes
    assert "missing_final_registry_candidate_index_jsonl" in report.error_codes


def test_non_final_registry_candidate_status_rejected() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(final_registry_status=OledFinalRegistryCandidateEntryStatus.REJECTED),
        final_registry_index_records=[_index_record(final_registry_status="rejected")],
    )

    assert "final_registry_status_not_candidate" in report.error_codes
    assert "index_status_not_final_registry_candidate" in report.error_codes


def test_validation_and_global_registry_claims_rejected() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "global_registry_mutated": True}
        ),
        final_registry_entry=_entry(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "benchmark_published": True}
        ),
        final_registry_index_records=[
            _index_record(metadata={"benchmark_validated": True, "scientific_claim_validated": True, "globally_registered": True})
        ],
    )

    assert report.status == OledFinalRegistryGlobalAppendPreflightStatus.FAILED
    assert "benchmark_validated_source_claim" in report.error_codes
    assert "scientific_claim_validated_source_claim" in report.error_codes
    assert "global_registry_source_claim" in report.error_codes


def test_missing_source_ids() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(source_publication_writer_manifest_id=None, source_publication_entry_id=None, source_final_registry_preflight_status=None),
        final_registry_entry=_entry(
            source_publication_writer_manifest_id=None,
            source_publication_entry_id=None,
            source_final_registry_preflight_status=None,
            source_promoted_entry_id=None,
            source_promotion_writer_manifest_id=None,
            source_registry_entry_id=None,
            source_registry_writer_manifest_id=None,
            source_candidate_report_id=None,
            source_benchmark_report_manifest_id=None,
        ),
        final_registry_index_records=[_index_record()],
    )

    assert "missing_source_publication_writer_manifest_id" in report.error_codes
    assert "missing_source_publication_entry_id" in report.error_codes
    assert "missing_source_final_registry_preflight_status" in report.error_codes
    assert "missing_source_promoted_entry_id" in report.error_codes
    assert "missing_source_promotion_writer_manifest_id" in report.error_codes
    assert "missing_source_registry_entry_id" in report.error_codes
    assert "missing_source_registry_writer_manifest_id" in report.error_codes
    assert "missing_source_candidate_report_id" in report.error_codes
    assert "missing_source_benchmark_report_manifest_id" in report.error_codes


def test_missing_caveats_run_cards_and_metric_cards() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(caveats=["baseline_candidate_report_only"], run_card_count=0, metric_card_count=0),
        final_registry_index_records=[_index_record()],
    )

    assert "missing_required_caveat" in report.error_codes
    assert "missing_run_cards" in report.error_codes
    assert "missing_metric_cards" in report.error_codes


def test_final_registry_index_mismatch_and_multiple_records() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[
            _index_record(final_registry_entry_id="entry:other"),
            _index_record(final_registry_entry_id="entry:another"),
        ],
    )

    assert "final_registry_entry_not_in_index" in report.error_codes
    assert "multiple_final_registry_index_records" in report.error_codes


def test_existing_registry_duplicate_entry_id_rejected() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        existing_registry_records=[_existing(registry_entry_id="entry:oled-final-registry-candidate:test")],
    )

    assert "existing_registry_duplicate_entry_id" in report.error_codes
    assert report.entry_summaries[0].existing_registry_duplicate_entry_count == 1


def test_existing_registry_duplicate_source_chain_rejected() -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        existing_registry_records=[
            _existing(
                source_publication_entry_id="entry:oled-publication-candidate-registry:test",
                source_candidate_report_id="report:oled-baseline-benchmark:test",
                source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
            )
        ],
    )

    assert "existing_registry_duplicate_source_chain" in report.error_codes
    assert report.entry_summaries[0].existing_registry_duplicate_source_chain_count == 1


def test_manifest_loader_and_artifact_loader_verify_sha(tmp_path: Path) -> None:
    manifest_path, manifest, entry, index = _write_final_registry_package(tmp_path)

    loaded_manifest = load_oled_publication_candidate_final_registry_writer_manifest_json(manifest_path)
    loaded_entry, loaded_index = load_oled_final_registry_candidate_artifacts_from_manifest(manifest=loaded_manifest, base_dir=tmp_path)

    assert loaded_manifest == manifest
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
    with pytest.raises(ValueError, match="final_registry_candidate_entry_sha256_mismatch:"):
        load_oled_final_registry_candidate_artifacts_from_manifest(manifest=bad_manifest, base_dir=tmp_path)


def test_existing_snapshot_loader_valid_and_invalid_jsonl(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "existing.jsonl"
    snapshot_path.write_text(
        '{"registry_entry_id":"existing:1","registry_status":"global","source_publication_entry_id":"entry:other"}\n'
        '{"final_registry_entry_id":"existing:2","final_registry_status":"global","source_candidate_report_id":"report:other"}\n',
        encoding="utf-8",
    )

    records = load_oled_existing_final_registry_snapshot_jsonl(snapshot_path)
    assert [record.registry_entry_id for record in records] == ["existing:1", "existing:2"]

    bad_path = tmp_path / "bad-existing.jsonl"
    bad_path.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_existing_final_registry_snapshot_jsonl:line-1"):
        load_oled_existing_final_registry_snapshot_jsonl(bad_path)

    raw_path = tmp_path / "raw-existing.jsonl"
    raw_path.write_text('{"registry_entry_id":"x","features":{}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_existing_final_registry_snapshot_jsonl:line-1"):
        load_oled_existing_final_registry_snapshot_jsonl(raw_path)


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    manifest_path, _, _, _ = _write_final_registry_package(tmp_path)
    existing_path = tmp_path / "existing.jsonl"
    existing_path.write_text('{"registry_entry_id":"existing:1","source_publication_entry_id":"entry:other"}\n', encoding="utf-8")
    output_report = tmp_path / "global_append_preflight.json"

    report = run_oled_final_registry_global_append_preflight_from_files(
        final_registry_writer_manifest_path=manifest_path,
        final_registry_candidate_base_dir=tmp_path,
        existing_registry_snapshot_path=existing_path,
        output_report_path=output_report,
    )

    assert report.is_valid
    assert output_report.exists()
    assert not (tmp_path / "global_registry.jsonl").exists()


def test_report_writer_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
    )
    output_path = tmp_path / "report.json"

    write_oled_final_registry_global_append_preflight_report_json(report, output_path)
    first = output_path.read_text(encoding="utf-8")
    write_oled_final_registry_global_append_preflight_report_json(report, output_path)

    assert first == output_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in first
    assert "raw_text" not in first
    assert "features" not in first
    assert "prediction_id" not in first


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, _, _, _ = _write_final_registry_package(tmp_path)
    output_report = tmp_path / "cli-global-append-preflight.json"

    exit_code = main(
        [
            "--final-registry-writer-manifest",
            str(manifest_path),
            "--final-registry-candidate-base-dir",
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
    assert OledFinalRegistryGlobalAppendPreflightStatus
    assert OledFinalRegistryGlobalAppendArtifactStatus
    assert OledFinalRegistryGlobalAppendPreflightPolicy
    assert OledFinalRegistryGlobalAppendArtifactSummary
    assert OledFinalRegistryGlobalAppendEntrySummary
    assert OledFinalRegistryExistingRecordSummary
    assert OledFinalRegistryGlobalAppendPreflightFinding
    assert OledFinalRegistryGlobalAppendPreflightReport
    assert package_load_manifest
    assert package_load_artifacts
    assert package_load_existing_snapshot
    assert package_run_preflight
    assert package_run_from_files
    assert package_write_report
