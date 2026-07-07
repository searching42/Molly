from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains.oled_curated_final_registry_global_append_preflight import (
    OledFinalRegistryExistingRecordSummary,
)
from ai4s_agent.domains.oled_curated_global_append_release_writer import (
    OledGlobalAppendReleaseFileResult,
    OledGlobalAppendReleaseWriteStatus,
    OledGlobalAppendReleaseWriterManifest,
    OledGlobalAppendReleaseWriterPolicy,
    OledReleaseCandidateDeltaRecord,
    OledReleaseCandidateEntry,
    OledReleaseCandidateEntryStatus,
    oled_release_candidate_delta_filename,
    oled_release_candidate_entry_filename,
    oled_release_candidate_snapshot_filename,
    write_oled_global_append_release_manifest_json,
    write_oled_release_candidate_delta_jsonl,
    write_oled_release_candidate_entry_json,
    write_oled_release_candidate_snapshot_jsonl,
)
from ai4s_agent.domains.oled_curated_release_candidate_external_publication_preflight import (
    OledReleaseCandidateExternalPublicationPreflightPolicy,
    OledReleaseCandidateExternalPublicationPreflightStatus,
    OledReleaseCandidateSnapshotRecordSummary,
    load_oled_global_append_release_writer_manifest_json,
    load_oled_release_candidate_artifacts_from_manifest,
    load_oled_release_candidate_snapshot_jsonl,
    main,
    run_oled_release_candidate_external_publication_preflight,
    run_oled_release_candidate_external_publication_preflight_from_files,
    write_oled_release_candidate_external_publication_preflight_report_json,
)


def _entry(
    *,
    release_status: str | OledReleaseCandidateEntryStatus = OledReleaseCandidateEntryStatus.RELEASE_CANDIDATE,
    release_entry_id: str = "entry:oled-release-candidate:test",
    metadata: dict | None = None,
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
    source_global_append_writer_manifest_id: str | None = "manifest:oled-global-append:test",
    source_global_append_entry_id: str | None = "entry:oled-global-append:test",
    source_release_preflight_status: str | None = "passed",
    source_final_registry_entry_id: str | None = "entry:oled-final-registry:test",
    source_final_registry_writer_manifest_id: str | None = "manifest:oled-final-registry:test",
    source_publication_entry_id: str | None = "entry:oled-publication:test",
    source_publication_writer_manifest_id: str | None = "manifest:oled-publication:test",
    source_promoted_entry_id: str | None = "entry:oled-promoted:test",
    source_promotion_writer_manifest_id: str | None = "manifest:oled-promotion:test",
    source_registry_entry_id: str | None = "entry:oled-registry:test",
    source_registry_writer_manifest_id: str | None = "manifest:oled-registry:test",
    source_candidate_report_id: str | None = "report:oled-candidate:test",
    source_benchmark_report_manifest_id: str | None = "manifest:oled-benchmark-report:test",
) -> OledReleaseCandidateEntry:
    return OledReleaseCandidateEntry(
        release_entry_id=release_entry_id,
        release_status=release_status,
        source_global_append_writer_manifest_id=source_global_append_writer_manifest_id,
        source_global_append_entry_id=source_global_append_entry_id,
        source_release_preflight_status=source_release_preflight_status,
        source_final_registry_entry_id=source_final_registry_entry_id,
        source_final_registry_writer_manifest_id=source_final_registry_writer_manifest_id,
        source_publication_entry_id=source_publication_entry_id,
        source_publication_writer_manifest_id=source_publication_writer_manifest_id,
        source_promoted_entry_id=source_promoted_entry_id,
        source_promotion_writer_manifest_id=source_promotion_writer_manifest_id,
        source_registry_entry_id=source_registry_entry_id,
        source_registry_writer_manifest_id=source_registry_writer_manifest_id,
        source_candidate_report_id=source_candidate_report_id,
        source_benchmark_report_manifest_id=source_benchmark_report_manifest_id,
        source_global_append_preflight_status="passed",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=run_card_count,
        metric_card_count=metric_card_count,
        source_global_append_entry_json_path="oled_global_append_candidate_entry.json",
        source_global_append_entry_json_sha256="source-entry-sha",
        source_global_append_delta_jsonl_path="oled_global_append_candidate_delta.jsonl",
        source_global_append_delta_jsonl_sha256="source-delta-sha",
        source_global_registry_snapshot_jsonl_path="oled_global_registry_snapshot.jsonl",
        source_global_registry_snapshot_jsonl_sha256="source-snapshot-sha",
        caveats=caveats
        if caveats is not None
        else ["baseline_candidate_report_only", "not_benchmark_validated", "not_scientific_performance_claim"],
        release_reason_codes=["selected_for_release_candidate"],
        metadata=metadata
        if metadata is not None
        else {
            "release_candidate_entry": True,
            "release_status": "release_candidate",
            "external_publication_written": False,
            "github_release_created": False,
            "git_tag_created": False,
            "artifact_uploaded": False,
            "global_registry_mutated": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _delta(
    *,
    release_entry_id: str = "entry:oled-release-candidate:test",
    release_status: str = "release_candidate",
    metadata: dict | None = None,
) -> OledReleaseCandidateDeltaRecord:
    return OledReleaseCandidateDeltaRecord(
        release_entry_id=release_entry_id,
        release_status=release_status,
        source_global_append_entry_id="entry:oled-global-append:test",
        source_global_append_writer_manifest_id="manifest:oled-global-append:test",
        source_release_preflight_status="passed",
        source_final_registry_entry_id="entry:oled-final-registry:test",
        source_publication_entry_id="entry:oled-publication:test",
        source_promoted_entry_id="entry:oled-promoted:test",
        source_registry_entry_id="entry:oled-registry:test",
        source_candidate_report_id="report:oled-candidate:test",
        source_benchmark_report_manifest_id="manifest:oled-benchmark-report:test",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=1,
        metric_card_count=3,
        output_release_entry_json_path=oled_release_candidate_entry_filename(),
        output_release_entry_json_sha256="release-entry-sha",
        benchmark_published=False,
        benchmark_registered=False,
        benchmark_validated=False,
        scientific_claim_validated=False,
        metadata=metadata
        if metadata is not None
        else {
            "release_candidate_delta_record": True,
            "release_status": "release_candidate",
            "external_publication_written": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _release_snapshot_summary(
    release_entry_id: str = "entry:oled-release-candidate:test",
) -> OledReleaseCandidateSnapshotRecordSummary:
    return OledReleaseCandidateSnapshotRecordSummary(
        release_entry_id=release_entry_id,
        release_status="release_candidate",
        source_global_append_entry_id="entry:oled-global-append:test",
        source_global_append_writer_manifest_id="manifest:oled-global-append:test",
        source_final_registry_entry_id="entry:oled-final-registry:test",
        source_publication_entry_id="entry:oled-publication:test",
        source_candidate_report_id="report:oled-candidate:test",
        source_benchmark_report_manifest_id="manifest:oled-benchmark-report:test",
        metadata={"release_entry_id": release_entry_id, "release_status": "release_candidate"},
    )


def _prior_summary(idx: int = 1) -> OledReleaseCandidateSnapshotRecordSummary:
    return OledReleaseCandidateSnapshotRecordSummary(
        registry_entry_id=f"entry:prior:{idx}",
        release_status="global_append_candidate",
        source_publication_entry_id=f"entry:prior-publication:{idx}",
        source_candidate_report_id=f"report:prior:{idx}",
        source_benchmark_report_manifest_id=f"manifest:prior-benchmark:{idx}",
        metadata={"prior_snapshot_record": True, "ordinal": idx},
    )


def _prior_writer_record(idx: int = 1) -> OledFinalRegistryExistingRecordSummary:
    return OledFinalRegistryExistingRecordSummary(
        registry_entry_id=f"entry:prior:{idx}",
        registry_status="global_append_candidate",
        source_publication_entry_id=f"entry:prior-publication:{idx}",
        source_publication_writer_manifest_id=f"manifest:prior-publication:{idx}",
        source_candidate_report_id=f"report:prior:{idx}",
        source_benchmark_report_manifest_id=f"manifest:prior-benchmark:{idx}",
        metadata={"prior_snapshot_record": True, "ordinal": idx},
    )


def _manifest(
    *,
    entry_sha: str | None = "release-entry-sha",
    delta_sha: str | None = "release-delta-sha",
    snapshot_sha: str | None = "release-snapshot-sha",
    metadata: dict | None = None,
    source_global_append_writer_manifest_id: str | None = "manifest:oled-global-append:test",
    source_global_append_entry_id: str | None = "entry:oled-global-append:test",
    source_release_preflight_status: str | None = "passed",
) -> OledGlobalAppendReleaseWriterManifest:
    return OledGlobalAppendReleaseWriterManifest(
        manifest_id="manifest:oled-release-candidate:test",
        source_global_append_writer_manifest_id=source_global_append_writer_manifest_id,
        source_global_append_entry_id=source_global_append_entry_id,
        source_release_preflight_status=source_release_preflight_status,
        prior_registry_snapshot_path="prior_snapshot.jsonl",
        prior_registry_record_count=1,
        release_snapshot_record_count=2,
        output_directory="release_candidate",
        output_file_count=3,
        release_entry_ids=["entry:oled-release-candidate:test"],
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledGlobalAppendReleaseFileResult(
                artifact_kind="release_candidate_entry_json",
                status=OledGlobalAppendReleaseWriteStatus.WRITTEN,
                output_path=oled_release_candidate_entry_filename(),
                output_sha256=entry_sha,
                reason_codes=["release_candidate_entry_json_written"],
            ),
            OledGlobalAppendReleaseFileResult(
                artifact_kind="release_candidate_delta_jsonl",
                status=OledGlobalAppendReleaseWriteStatus.WRITTEN,
                output_path=oled_release_candidate_delta_filename(),
                output_sha256=delta_sha,
                reason_codes=["release_candidate_delta_jsonl_written"],
            ),
            OledGlobalAppendReleaseFileResult(
                artifact_kind="release_candidate_snapshot_jsonl",
                status=OledGlobalAppendReleaseWriteStatus.WRITTEN,
                output_path=oled_release_candidate_snapshot_filename(),
                output_sha256=snapshot_sha,
                reason_codes=["release_candidate_snapshot_jsonl_written"],
            ),
        ],
        policy=OledGlobalAppendReleaseWriterPolicy(),
        metadata=metadata
        if metadata is not None
        else {
            "global_append_release_writer": True,
            "release_candidate_written": True,
            "external_publication_written": False,
            "github_release_created": False,
            "git_tag_created": False,
            "artifact_uploaded": False,
            "benchmark_validated": False,
            "scientific_claim_validated": False,
        },
    )


def _run(
    *,
    entry: OledReleaseCandidateEntry | None | object = ...,
    deltas: list[OledReleaseCandidateDeltaRecord] | None = None,
    snapshot: list[OledReleaseCandidateSnapshotRecordSummary] | None = None,
    prior: list[OledReleaseCandidateSnapshotRecordSummary] | None = None,
    manifest: OledGlobalAppendReleaseWriterManifest | None = None,
):
    actual_entry = _entry() if entry is ... else entry
    return run_oled_release_candidate_external_publication_preflight(
        release_writer_manifest=manifest or _manifest(),
        release_entry=actual_entry,
        release_delta_records=deltas if deltas is not None else [_delta()],
        release_snapshot_records=snapshot if snapshot is not None else [_release_snapshot_summary()],
        prior_registry_snapshot_records=prior,
    )


def _write_package(tmp_path: Path):
    entry = _entry()
    delta_records = [_delta()]
    entry_sha = write_oled_release_candidate_entry_json(entry, tmp_path / oled_release_candidate_entry_filename())
    delta_sha = write_oled_release_candidate_delta_jsonl(delta_records, tmp_path / oled_release_candidate_delta_filename())
    snapshot_sha = write_oled_release_candidate_snapshot_jsonl(
        [_prior_writer_record()],
        delta_records,
        tmp_path / oled_release_candidate_snapshot_filename(),
    )
    manifest = _manifest(entry_sha=entry_sha, delta_sha=delta_sha, snapshot_sha=snapshot_sha)
    manifest_path = tmp_path / "release_manifest.json"
    write_oled_global_append_release_manifest_json(manifest, manifest_path)
    return manifest_path, manifest


def test_main_preflight_success_without_prior_snapshot():
    report = _run()

    assert report.status == OledReleaseCandidateExternalPublicationPreflightStatus.PASSED
    assert report.is_valid
    assert report.entry_summaries[0].matched_release_delta_record_count == 1
    assert report.entry_summaries[0].matched_release_snapshot_record_count == 1
    assert report.metadata["release_candidate_external_publication_preflight_only"] is True
    assert report.metadata["github_release_created"] is False
    assert report.metadata["artifact_uploaded"] is False


def test_main_preflight_success_with_prior_snapshot_preserved():
    prior = [_prior_summary()]
    report = _run(snapshot=[_prior_summary(), _release_snapshot_summary()], prior=prior)

    assert report.is_valid
    assert report.entry_summaries[0].preserved_prior_snapshot_record_count == 1


def test_missing_release_entry_delta_snapshot():
    report = _run(entry=None, deltas=[], snapshot=[])

    assert report.status == OledReleaseCandidateExternalPublicationPreflightStatus.FAILED
    assert "missing_release_candidate_entry_json" in report.error_codes
    assert "missing_release_candidate_delta_jsonl" in report.error_codes
    assert "missing_release_candidate_snapshot_jsonl" in report.error_codes


def test_non_release_candidate_status_rejected():
    report = _run(entry=_entry(release_status="rejected"), deltas=[_delta(release_status="rejected")])

    assert "release_status_not_candidate" in report.error_codes
    assert "delta_status_not_release_candidate" in report.error_codes


@pytest.mark.parametrize(
    ("metadata", "code"),
    [
        ({"benchmark_validated": True}, "benchmark_validated_source_claim"),
        ({"scientific_claim_validated": True}, "scientific_claim_validated_source_claim"),
        ({"external_publication_written": True}, "external_publication_source_claim"),
        ({"github_release_created": True}, "github_release_source_claim"),
        ({"git_tag_created": True}, "git_tag_source_claim"),
        ({"artifact_uploaded": True}, "artifact_upload_source_claim"),
        ({"global_registry_mutated": True}, "external_publication_source_claim"),
    ],
)
def test_validation_and_publication_claims_rejected(metadata, code):
    report = _run(entry=_entry(metadata=metadata))

    assert code in report.error_codes


def test_missing_source_ids():
    report = _run(
        entry=_entry(
            source_global_append_writer_manifest_id=None,
            source_global_append_entry_id=None,
            source_release_preflight_status=None,
            source_final_registry_entry_id=None,
            source_final_registry_writer_manifest_id=None,
            source_publication_entry_id=None,
            source_publication_writer_manifest_id=None,
            source_promoted_entry_id=None,
            source_promotion_writer_manifest_id=None,
            source_registry_entry_id=None,
            source_registry_writer_manifest_id=None,
            source_candidate_report_id=None,
            source_benchmark_report_manifest_id=None,
        ),
        manifest=_manifest(
            source_global_append_writer_manifest_id=None,
            source_global_append_entry_id=None,
            source_release_preflight_status=None,
        ),
    )

    assert "missing_source_global_append_writer_manifest_id" in report.error_codes
    assert "missing_source_global_append_entry_id" in report.error_codes
    assert "missing_source_release_preflight_status" in report.error_codes
    assert "missing_source_final_registry_entry_id" in report.error_codes
    assert "missing_source_benchmark_report_manifest_id" in report.error_codes


def test_missing_caveats_run_cards_metric_cards():
    report = _run(entry=_entry(caveats=["baseline_candidate_report_only"], run_card_count=0, metric_card_count=0))

    assert "missing_required_caveat" in report.error_codes
    assert "missing_run_cards" in report.error_codes
    assert "missing_metric_cards" in report.error_codes


def test_delta_mismatch_and_multiple_records():
    report = _run(
        deltas=[
            _delta(release_entry_id="entry:other:1"),
            _delta(release_entry_id="entry:other:2"),
        ],
        snapshot=[
            OledReleaseCandidateSnapshotRecordSummary(
                release_entry_id="entry:unrelated",
                release_status="release_candidate",
                source_global_append_entry_id="entry:unrelated-global-append",
                source_candidate_report_id="report:unrelated",
                source_benchmark_report_manifest_id="manifest:unrelated",
            )
        ],
    )

    assert "multiple_release_delta_records" in report.error_codes
    assert "release_entry_not_in_delta" in report.error_codes
    assert "release_delta_record_not_in_snapshot" in report.error_codes


def test_release_snapshot_missing_release_record():
    report = _run(snapshot=[_prior_summary()])

    assert "release_entry_not_in_snapshot" in report.error_codes
    assert "release_delta_record_not_in_snapshot" in report.error_codes


def test_prior_snapshot_not_preserved():
    report = _run(snapshot=[_release_snapshot_summary(), _prior_summary()], prior=[_prior_summary()])

    assert "prior_snapshot_not_preserved" in report.error_codes


def test_manifest_loader_and_artifact_loader_with_sha_verification(tmp_path: Path):
    manifest_path, manifest = _write_package(tmp_path)

    loaded_manifest = load_oled_global_append_release_writer_manifest_json(manifest_path)
    entry, deltas, snapshot = load_oled_release_candidate_artifacts_from_manifest(
        manifest=loaded_manifest,
        base_dir=tmp_path,
    )

    assert entry is not None
    assert entry.release_entry_id == "entry:oled-release-candidate:test"
    assert len(deltas) == 1
    assert len(snapshot) == 2

    bad_manifest = manifest.model_copy(
        update={
            "file_results": [
                result.model_copy(update={"output_sha256": "bad-sha"})
                if result.artifact_kind == "release_candidate_entry_json"
                else result
                for result in manifest.file_results
            ]
        }
    )
    with pytest.raises(ValueError, match="release_candidate_entry_sha256_mismatch:"):
        load_oled_release_candidate_artifacts_from_manifest(manifest=bad_manifest, base_dir=tmp_path)


def test_release_snapshot_loader_valid_and_invalid_jsonl(tmp_path: Path):
    manifest_path, _manifest_obj = _write_package(tmp_path)
    loaded_manifest = load_oled_global_append_release_writer_manifest_json(manifest_path)
    _entry_obj, _deltas, snapshot = load_oled_release_candidate_artifacts_from_manifest(
        manifest=loaded_manifest,
        base_dir=tmp_path,
    )

    assert [record.release_entry_id for record in snapshot][-1] == "entry:oled-release-candidate:test"

    invalid_path = tmp_path / "invalid_snapshot.jsonl"
    invalid_path.write_text('{"release_entry_id": "ok"}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_release_candidate_snapshot_jsonl:line-2"):
        load_oled_release_candidate_snapshot_jsonl(invalid_path)


def test_combined_runner_from_files_writes_report_only(tmp_path: Path):
    manifest_path, _manifest_obj = _write_package(tmp_path)
    report_path = tmp_path / "external_publication_preflight_report.json"

    report = run_oled_release_candidate_external_publication_preflight_from_files(
        release_writer_manifest_path=manifest_path,
        release_candidate_base_dir=tmp_path,
        output_report_path=report_path,
    )

    assert report.is_valid
    assert report_path.exists()
    assert not (tmp_path / "github_release.json").exists()


def test_report_writer_deterministic_and_redacted(tmp_path: Path):
    report = _run()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_oled_release_candidate_external_publication_preflight_report_json(report, first)
    write_oled_release_candidate_external_publication_preflight_report_json(report, second)

    first_text = first.read_text(encoding="utf-8")
    assert first_text == second.read_text(encoding="utf-8")
    assert str(tmp_path) not in first_text
    assert "raw_text" not in first_text
    assert '"features"' not in first_text
    assert "prediction_id" not in first_text


def test_cli_smoke(tmp_path: Path, capsys):
    manifest_path, _manifest_obj = _write_package(tmp_path)
    report_path = tmp_path / "cli_report.json"

    exit_code = main(
        [
            "--release-writer-manifest",
            str(manifest_path),
            "--release-candidate-base-dir",
            str(tmp_path),
            "--output-report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert report_path.exists()
    assert "entry:oled-release-candidate:test" not in captured.out
    assert json.loads(captured.out)["status"] == "passed"


def test_package_exports():
    from ai4s_agent.domains import (  # noqa: PLC0415
        OledReleaseCandidateExternalPublicationArtifactStatus,
        OledReleaseCandidateExternalPublicationArtifactSummary,
        OledReleaseCandidateExternalPublicationEntrySummary,
        OledReleaseCandidateExternalPublicationPreflightFinding,
        OledReleaseCandidateExternalPublicationPreflightPolicy,
        OledReleaseCandidateExternalPublicationPreflightReport,
        OledReleaseCandidateExternalPublicationPreflightStatus,
        OledReleaseCandidateSnapshotRecordSummary,
        load_oled_global_append_release_writer_manifest_json as exported_load_manifest,
        load_oled_release_candidate_artifacts_from_manifest as exported_load_artifacts,
        load_oled_release_candidate_snapshot_jsonl as exported_load_snapshot,
        run_oled_release_candidate_external_publication_preflight as exported_run,
        run_oled_release_candidate_external_publication_preflight_from_files as exported_run_files,
        write_oled_release_candidate_external_publication_preflight_report_json as exported_write,
    )

    assert OledReleaseCandidateExternalPublicationPreflightStatus.PASSED.value == "passed"
    assert OledReleaseCandidateExternalPublicationArtifactStatus.READY.value == "ready"
    assert OledReleaseCandidateExternalPublicationPreflightPolicy().require_release_entry_json is True
    assert OledReleaseCandidateExternalPublicationArtifactSummary is not None
    assert OledReleaseCandidateSnapshotRecordSummary is not None
    assert OledReleaseCandidateExternalPublicationEntrySummary is not None
    assert OledReleaseCandidateExternalPublicationPreflightFinding is not None
    assert OledReleaseCandidateExternalPublicationPreflightReport is not None
    assert exported_load_manifest is load_oled_global_append_release_writer_manifest_json
    assert exported_load_snapshot is load_oled_release_candidate_snapshot_jsonl
    assert exported_load_artifacts is load_oled_release_candidate_artifacts_from_manifest
    assert exported_run is run_oled_release_candidate_external_publication_preflight
    assert exported_run_files is run_oled_release_candidate_external_publication_preflight_from_files
    assert exported_write is write_oled_release_candidate_external_publication_preflight_report_json
