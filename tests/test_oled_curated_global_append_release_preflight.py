from __future__ import annotations

from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledFinalRegistryExistingRecordSummary,
    OledFinalRegistryGlobalAppendWriteStatus,
    OledFinalRegistryGlobalAppendWriterManifest,
    OledFinalRegistryGlobalAppendWriterPolicy,
    OledGlobalAppendCandidateEntry,
    OledGlobalAppendCandidateEntryStatus,
    OledGlobalAppendCandidateIndexRecord,
    OledGlobalAppendFileResult,
    OledGlobalAppendReleaseArtifactStatus,
    OledGlobalAppendReleaseArtifactSummary,
    OledGlobalAppendReleaseEntrySummary,
    OledGlobalAppendReleasePreflightFinding,
    OledGlobalAppendReleasePreflightPolicy,
    OledGlobalAppendReleasePreflightReport,
    OledGlobalAppendReleasePreflightStatus,
    load_oled_final_registry_global_append_writer_manifest_json as package_load_manifest,
    load_oled_global_append_artifacts_from_manifest as package_load_artifacts,
    run_oled_global_append_release_preflight as package_run_preflight,
    run_oled_global_append_release_preflight_from_files as package_run_from_files,
    write_oled_global_append_release_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_final_registry_global_append_writer import (
    oled_global_append_candidate_delta_filename,
    oled_global_append_candidate_entry_filename,
    oled_global_registry_snapshot_filename,
    write_oled_final_registry_global_append_manifest_json,
    write_oled_global_append_candidate_delta_jsonl,
    write_oled_global_append_candidate_entry_json,
    write_oled_global_registry_snapshot_jsonl,
)
from ai4s_agent.domains.oled_curated_global_append_release_preflight import (
    load_oled_final_registry_global_append_writer_manifest_json,
    load_oled_global_append_artifacts_from_manifest,
    main,
    run_oled_global_append_release_preflight,
    run_oled_global_append_release_preflight_from_files,
    write_oled_global_append_release_preflight_report_json,
)


def _entry(
    *,
    global_append_status: str | OledGlobalAppendCandidateEntryStatus = OledGlobalAppendCandidateEntryStatus.GLOBAL_APPEND_CANDIDATE,
    metadata: dict | None = None,
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
    source_final_registry_writer_manifest_id: str | None = "manifest:oled-final-registry-candidate:test",
    source_final_registry_entry_id: str | None = "entry:oled-final-registry-candidate:test",
    source_global_append_preflight_status: str | None = "passed",
    source_publication_entry_id: str | None = "entry:oled-publication-candidate-registry:test",
    source_publication_writer_manifest_id: str | None = "manifest:oled-publication-candidate-final-registry:test",
    source_promoted_entry_id: str | None = "entry:oled-benchmark-promoted-registry:test",
    source_promotion_writer_manifest_id: str | None = "manifest:oled-benchmark-promoted-registry:test",
    source_registry_entry_id: str | None = "entry:oled-benchmark-registry:test",
    source_registry_writer_manifest_id: str | None = "manifest:oled-benchmark-registry:test",
    source_candidate_report_id: str | None = "report:oled-baseline-benchmark:test",
    source_benchmark_report_manifest_id: str | None = "manifest:oled-baseline-benchmark:test",
) -> OledGlobalAppendCandidateEntry:
    return OledGlobalAppendCandidateEntry(
        global_append_entry_id="entry:oled-global-append-candidate:test",
        global_append_status=global_append_status,
        source_final_registry_writer_manifest_id=source_final_registry_writer_manifest_id,
        source_final_registry_entry_id=source_final_registry_entry_id,
        source_global_append_preflight_status=source_global_append_preflight_status,
        source_publication_entry_id=source_publication_entry_id,
        source_publication_writer_manifest_id=source_publication_writer_manifest_id,
        source_promoted_entry_id=source_promoted_entry_id,
        source_promotion_writer_manifest_id=source_promotion_writer_manifest_id,
        source_registry_entry_id=source_registry_entry_id,
        source_registry_writer_manifest_id=source_registry_writer_manifest_id,
        source_candidate_report_id=source_candidate_report_id,
        source_benchmark_report_manifest_id=source_benchmark_report_manifest_id,
        source_final_registry_preflight_status="passed",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=run_card_count,
        metric_card_count=metric_card_count,
        source_final_registry_entry_json_path="oled_final_registry_candidate_entry.json",
        source_final_registry_entry_json_sha256="source-final-entry-sha",
        source_final_registry_index_jsonl_path="oled_final_registry_candidate_index.jsonl",
        source_final_registry_index_jsonl_sha256="source-final-index-sha",
        caveats=caveats
        if caveats is not None
        else ["baseline_candidate_report_only", "not_benchmark_validated", "not_scientific_performance_claim"],
        append_reason_codes=["selected_for_global_append_candidate"],
        metadata=metadata
        if metadata is not None
        else {
            "final_registry_global_append_writer": True,
            "global_append_candidate_entry": True,
            "global_append_status": "global_append_candidate",
            "global_registry_mutated": False,
            "external_publication_written": False,
            "benchmark_published": False,
            "benchmark_registered": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _delta(
    *,
    global_append_entry_id: str = "entry:oled-global-append-candidate:test",
    global_append_status: str = "global_append_candidate",
    metadata: dict | None = None,
    output_sha256: str | None = "global-append-entry-sha",
) -> OledGlobalAppendCandidateIndexRecord:
    return OledGlobalAppendCandidateIndexRecord(
        global_append_entry_id=global_append_entry_id,
        global_append_status=global_append_status,
        source_final_registry_entry_id="entry:oled-final-registry-candidate:test",
        source_final_registry_writer_manifest_id="manifest:oled-final-registry-candidate:test",
        source_global_append_preflight_status="passed",
        source_publication_entry_id="entry:oled-publication-candidate-registry:test",
        source_promoted_entry_id="entry:oled-benchmark-promoted-registry:test",
        source_registry_entry_id="entry:oled-benchmark-registry:test",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=1,
        metric_card_count=3,
        output_global_append_entry_json_path=oled_global_append_candidate_entry_filename(),
        output_global_append_entry_json_sha256=output_sha256,
        benchmark_published=False,
        benchmark_registered=False,
        benchmark_validated=False,
        scientific_claim_validated=False,
        metadata=metadata
        if metadata is not None
        else {
            "global_append_candidate_index_record": True,
            "global_append_status": "global_append_candidate",
            "global_registry_mutated": False,
            "external_publication_written": False,
            "benchmark_published": False,
            "benchmark_registered": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _snapshot_append_summary() -> OledFinalRegistryExistingRecordSummary:
    return OledFinalRegistryExistingRecordSummary(
        registry_entry_id=None,
        registry_status=None,
        source_publication_entry_id="entry:oled-publication-candidate-registry:test",
        source_publication_writer_manifest_id=None,
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        metadata={"global_append_entry_id": "entry:oled-global-append-candidate:test"},
    )


def _prior(
    idx: int = 1,
    *,
    metadata: dict | None = None,
) -> OledFinalRegistryExistingRecordSummary:
    return OledFinalRegistryExistingRecordSummary(
        registry_entry_id=f"entry:existing-final-registry:{idx}",
        registry_status="global_candidate",
        source_publication_entry_id=f"entry:prior-publication:{idx}",
        source_publication_writer_manifest_id=f"manifest:prior-publication:{idx}",
        source_candidate_report_id=f"report:prior:{idx}",
        source_benchmark_report_manifest_id=f"manifest:prior-benchmark:{idx}",
        metadata=metadata if metadata is not None else {"existing_registry_record": True, "ordinal": idx},
    )


def _manifest(
    *,
    entry_sha: str | None = "global-append-entry-sha",
    delta_sha: str | None = "global-append-delta-sha",
    snapshot_sha: str | None = "global-registry-snapshot-sha",
    metadata: dict | None = None,
    source_final_registry_writer_manifest_id: str | None = "manifest:oled-final-registry-candidate:test",
    source_final_registry_entry_id: str | None = "entry:oled-final-registry-candidate:test",
    source_global_append_preflight_status: str | None = "passed",
) -> OledFinalRegistryGlobalAppendWriterManifest:
    return OledFinalRegistryGlobalAppendWriterManifest(
        manifest_id="manifest:oled-global-append-candidate:test",
        source_final_registry_writer_manifest_id=source_final_registry_writer_manifest_id,
        source_final_registry_entry_id=source_final_registry_entry_id,
        source_global_append_preflight_status=source_global_append_preflight_status,
        existing_registry_snapshot_path="existing_snapshot.jsonl",
        existing_registry_record_count=1,
        output_directory="global_append_candidate",
        output_file_count=3,
        global_append_entry_ids=["entry:oled-global-append-candidate:test"],
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledGlobalAppendFileResult(
                artifact_kind="global_append_entry_json",
                status=OledFinalRegistryGlobalAppendWriteStatus.WRITTEN,
                output_path=oled_global_append_candidate_entry_filename(),
                output_sha256=entry_sha,
                reason_codes=["global_append_candidate_entry_json_written"],
            ),
            OledGlobalAppendFileResult(
                artifact_kind="global_append_delta_jsonl",
                status=OledFinalRegistryGlobalAppendWriteStatus.WRITTEN,
                output_path=oled_global_append_candidate_delta_filename(),
                output_sha256=delta_sha,
                reason_codes=["global_append_candidate_delta_jsonl_written"],
            ),
            OledGlobalAppendFileResult(
                artifact_kind="global_registry_snapshot_jsonl",
                status=OledFinalRegistryGlobalAppendWriteStatus.WRITTEN,
                output_path=oled_global_registry_snapshot_filename(),
                output_sha256=snapshot_sha,
                reason_codes=["global_registry_snapshot_jsonl_written"],
            ),
        ],
        policy=OledFinalRegistryGlobalAppendWriterPolicy(),
        metadata=metadata
        if metadata is not None
        else {
            "final_registry_global_append_writer": True,
            "global_append_candidate_written": True,
            "global_append_entry_written": True,
            "global_append_delta_written": True,
            "global_registry_snapshot_written": True,
            "global_registry_mutated": False,
            "external_publication_written": False,
            "benchmark_published": False,
            "benchmark_registered": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _write_global_append_package(
    tmp_path: Path,
    *,
    prior_records: list[OledFinalRegistryExistingRecordSummary] | None = None,
) -> tuple[Path, OledFinalRegistryGlobalAppendWriterManifest, OledGlobalAppendCandidateEntry, list[OledGlobalAppendCandidateIndexRecord]]:
    prior_records = prior_records or [_prior()]
    entry = _entry()
    entry_path = tmp_path / oled_global_append_candidate_entry_filename()
    entry_sha = write_oled_global_append_candidate_entry_json(entry, entry_path)
    delta = [_delta(output_sha256=entry_sha)]
    delta_path = tmp_path / oled_global_append_candidate_delta_filename()
    delta_sha = write_oled_global_append_candidate_delta_jsonl(delta, delta_path)
    snapshot_path = tmp_path / oled_global_registry_snapshot_filename()
    snapshot_sha = write_oled_global_registry_snapshot_jsonl(existing_records=prior_records, append_records=delta, path=snapshot_path)
    manifest = _manifest(entry_sha=entry_sha, delta_sha=delta_sha, snapshot_sha=snapshot_sha)
    manifest_path = tmp_path / "global_append_manifest.json"
    write_oled_final_registry_global_append_manifest_json(manifest, manifest_path)
    return manifest_path, manifest, entry, delta


def test_main_preflight_success_without_prior_snapshot() -> None:
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(),
        global_append_entry=_entry(),
        global_append_delta_records=[_delta()],
        global_registry_snapshot_records=[_snapshot_append_summary()],
    )

    assert report.status == OledGlobalAppendReleasePreflightStatus.PASSED
    assert report.is_valid
    assert report.source_global_append_entry_id == "entry:oled-global-append-candidate:test"
    assert report.entry_summaries[0].artifact_status == OledGlobalAppendReleaseArtifactStatus.READY
    assert report.metadata["global_append_release_preflight_only"] is True
    assert report.metadata["global_registry_mutated"] is False


def test_main_preflight_success_with_prior_snapshot_preserved() -> None:
    prior = [_prior(1), _prior(2)]
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(),
        global_append_entry=_entry(),
        global_append_delta_records=[_delta()],
        global_registry_snapshot_records=[*prior, _snapshot_append_summary()],
        prior_registry_snapshot_records=prior,
    )

    assert report.status == OledGlobalAppendReleasePreflightStatus.PASSED
    summary = report.entry_summaries[0]
    assert summary.prior_snapshot_record_count == 2
    assert summary.preserved_prior_snapshot_record_count == 2


def test_missing_entry_delta_and_snapshot() -> None:
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(),
        global_append_entry=None,
        global_append_delta_records=[],
        global_registry_snapshot_records=[],
    )

    assert report.status == OledGlobalAppendReleasePreflightStatus.FAILED
    assert "missing_global_append_candidate_entry_json" in report.error_codes
    assert "missing_global_append_candidate_delta_jsonl" in report.error_codes
    assert "missing_global_registry_snapshot_jsonl" in report.error_codes


def test_non_global_append_candidate_status_rejected() -> None:
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(),
        global_append_entry=_entry(global_append_status=OledGlobalAppendCandidateEntryStatus.REJECTED),
        global_append_delta_records=[_delta(global_append_status="rejected")],
        global_registry_snapshot_records=[_snapshot_append_summary()],
    )

    assert "global_append_status_not_candidate" in report.error_codes
    assert "delta_status_not_global_append_candidate" in report.error_codes


def test_validation_external_publication_and_global_mutation_claims_rejected() -> None:
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "global_registry_mutated": True}
        ),
        global_append_entry=_entry(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "benchmark_published": True}
        ),
        global_append_delta_records=[
            _delta(metadata={"benchmark_validated": True, "scientific_claim_validated": True, "globally_registered": True})
        ],
        global_registry_snapshot_records=[
            _snapshot_append_summary().model_copy(
                update={"metadata": {"benchmark_validated": True, "scientific_claim_validated": True, "benchmark_published": True}}
            )
        ],
    )

    assert "benchmark_validated_source_claim" in report.error_codes
    assert "scientific_claim_validated_source_claim" in report.error_codes
    assert "external_publication_source_claim" in report.error_codes


def test_missing_source_ids() -> None:
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(
            source_final_registry_writer_manifest_id=None,
            source_final_registry_entry_id=None,
            source_global_append_preflight_status=None,
        ),
        global_append_entry=_entry(
            source_final_registry_writer_manifest_id=None,
            source_final_registry_entry_id=None,
            source_global_append_preflight_status=None,
            source_publication_entry_id=None,
            source_publication_writer_manifest_id=None,
            source_promoted_entry_id=None,
            source_promotion_writer_manifest_id=None,
            source_registry_entry_id=None,
            source_registry_writer_manifest_id=None,
            source_candidate_report_id=None,
            source_benchmark_report_manifest_id=None,
        ),
        global_append_delta_records=[_delta()],
        global_registry_snapshot_records=[_snapshot_append_summary()],
    )

    assert "missing_source_final_registry_writer_manifest_id" in report.error_codes
    assert "missing_source_final_registry_entry_id" in report.error_codes
    assert "missing_source_global_append_preflight_status" in report.error_codes
    assert "missing_source_publication_entry_id" in report.error_codes
    assert "missing_source_publication_writer_manifest_id" in report.error_codes
    assert "missing_source_promoted_entry_id" in report.error_codes
    assert "missing_source_promotion_writer_manifest_id" in report.error_codes
    assert "missing_source_registry_entry_id" in report.error_codes
    assert "missing_source_registry_writer_manifest_id" in report.error_codes
    assert "missing_source_candidate_report_id" in report.error_codes
    assert "missing_source_benchmark_report_manifest_id" in report.error_codes


def test_missing_caveats_run_cards_and_metric_cards() -> None:
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(),
        global_append_entry=_entry(caveats=["baseline_candidate_report_only"], run_card_count=0, metric_card_count=0),
        global_append_delta_records=[_delta()],
        global_registry_snapshot_records=[_snapshot_append_summary()],
    )

    assert "missing_required_caveat" in report.error_codes
    assert "missing_run_cards" in report.error_codes
    assert "missing_metric_cards" in report.error_codes


def test_delta_mismatch_and_multiple_delta_records() -> None:
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(),
        global_append_entry=_entry(),
        global_append_delta_records=[_delta(global_append_entry_id="entry:other"), _delta(global_append_entry_id="entry:second")],
        global_registry_snapshot_records=[_snapshot_append_summary()],
    )

    assert "global_append_entry_not_in_delta" in report.error_codes
    assert "multiple_global_append_delta_records" in report.error_codes


def test_new_snapshot_missing_append_record() -> None:
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(),
        global_append_entry=_entry(),
        global_append_delta_records=[_delta()],
        global_registry_snapshot_records=[_prior()],
    )

    assert "global_append_entry_not_in_snapshot" in report.error_codes
    assert "delta_record_not_in_snapshot" in report.error_codes


def test_prior_snapshot_not_preserved() -> None:
    prior = [_prior(1), _prior(2)]
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(),
        global_append_entry=_entry(),
        global_append_delta_records=[_delta()],
        global_registry_snapshot_records=[_prior(2), _prior(1), _snapshot_append_summary()],
        prior_registry_snapshot_records=prior,
    )

    assert "prior_snapshot_not_preserved" in report.error_codes


def test_manifest_loader_and_artifact_loader_verify_sha(tmp_path: Path) -> None:
    manifest_path, manifest, entry, delta = _write_global_append_package(tmp_path)

    loaded_manifest = load_oled_final_registry_global_append_writer_manifest_json(manifest_path)
    loaded_entry, loaded_delta, loaded_snapshot = load_oled_global_append_artifacts_from_manifest(
        manifest=loaded_manifest,
        base_dir=tmp_path,
    )

    assert loaded_manifest == manifest
    assert loaded_entry == entry
    assert loaded_delta == delta
    assert len(loaded_snapshot) == 2

    bad_manifest = manifest.model_copy(
        update={
            "file_results": [
                result.model_copy(update={"output_sha256": "bad-sha"}) if result.artifact_kind == "global_append_delta_jsonl" else result
                for result in manifest.file_results
            ]
        }
    )
    with pytest.raises(ValueError, match="global_append_delta_sha256_mismatch:"):
        load_oled_global_append_artifacts_from_manifest(manifest=bad_manifest, base_dir=tmp_path)


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    prior = [_prior()]
    manifest_path, _, _, _ = _write_global_append_package(tmp_path, prior_records=prior)
    prior_path = tmp_path / "prior_snapshot.jsonl"
    write_oled_global_registry_snapshot_jsonl(existing_records=prior, append_records=[], path=prior_path)
    output_report = tmp_path / "release_preflight_report.json"

    report = run_oled_global_append_release_preflight_from_files(
        global_append_writer_manifest_path=manifest_path,
        global_append_base_dir=tmp_path,
        prior_registry_snapshot_path=prior_path,
        output_report_path=output_report,
    )

    assert report.is_valid
    assert output_report.exists()
    assert not (tmp_path / "release.jsonl").exists()


def test_report_writer_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_manifest(),
        global_append_entry=_entry(),
        global_append_delta_records=[_delta()],
        global_registry_snapshot_records=[_snapshot_append_summary()],
    )
    path = tmp_path / "report.json"

    write_oled_global_append_release_preflight_report_json(report, path)
    first = path.read_text(encoding="utf-8")
    write_oled_global_append_release_preflight_report_json(report, path)

    assert first == path.read_text(encoding="utf-8")
    assert "raw_text" not in first
    assert "features" not in first
    assert str(tmp_path) not in first


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prior = [_prior()]
    manifest_path, _, _, _ = _write_global_append_package(tmp_path, prior_records=prior)
    prior_path = tmp_path / "prior_snapshot.jsonl"
    write_oled_global_registry_snapshot_jsonl(existing_records=prior, append_records=[], path=prior_path)
    output_report = tmp_path / "cli-release-report.json"

    exit_code = main(
        [
            "--global-append-writer-manifest",
            str(manifest_path),
            "--global-append-base-dir",
            str(tmp_path),
            "--prior-registry-snapshot",
            str(prior_path),
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
    assert OledGlobalAppendReleasePreflightStatus
    assert OledGlobalAppendReleaseArtifactStatus
    assert OledGlobalAppendReleasePreflightPolicy
    assert OledGlobalAppendReleaseArtifactSummary
    assert OledGlobalAppendReleaseEntrySummary
    assert OledGlobalAppendReleasePreflightFinding
    assert OledGlobalAppendReleasePreflightReport
    assert package_load_manifest
    assert package_load_artifacts
    assert package_run_preflight
    assert package_run_from_files
    assert package_write_report
