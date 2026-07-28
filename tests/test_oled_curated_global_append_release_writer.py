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
    OledGlobalAppendReleaseFileResult,
    OledGlobalAppendReleasePreflightPolicy,
    OledGlobalAppendReleasePreflightStatus,
    OledGlobalAppendReleaseWriteStatus,
    OledGlobalAppendReleaseWriterFinding,
    OledGlobalAppendReleaseWriterManifest,
    OledGlobalAppendReleaseWriterPolicy,
    OledGlobalAppendReleaseWriterReport,
    OledReleaseCandidateDeltaRecord,
    OledReleaseCandidateEntry,
    OledReleaseCandidateEntryStatus,
    build_oled_release_candidate_delta_records as package_build_delta,
    build_oled_release_candidate_entry as package_build_entry,
    load_oled_global_append_release_preflight_report_json as package_load_preflight,
    load_oled_release_candidate_delta_jsonl as package_load_delta,
    load_oled_release_candidate_entry_json as package_load_entry,
    oled_release_candidate_delta_filename as package_delta_filename,
    oled_release_candidate_entry_filename as package_entry_filename,
    oled_release_candidate_snapshot_filename as package_snapshot_filename,
    run_oled_global_append_release_preflight,
    run_oled_global_append_release_writer_from_files as package_run_from_files,
    select_oled_global_append_release_for_write as package_select,
    write_oled_global_append_release_manifest_json as package_write_manifest,
    write_oled_release_candidate_delta_jsonl as package_write_delta,
    write_oled_release_candidate_entry_json as package_write_entry,
    write_oled_release_candidate_snapshot_jsonl as package_write_snapshot,
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
    write_oled_global_append_release_preflight_report_json,
)
from ai4s_agent.domains.oled_curated_global_append_release_writer import (
    build_oled_release_candidate_delta_records,
    build_oled_release_candidate_entry,
    load_oled_global_append_release_preflight_report_json,
    load_oled_release_candidate_delta_jsonl,
    load_oled_release_candidate_entry_json,
    main,
    oled_release_candidate_delta_filename,
    oled_release_candidate_entry_filename,
    oled_release_candidate_snapshot_filename,
    run_oled_global_append_release_writer_from_files,
    select_oled_global_append_release_for_write,
    write_oled_global_append_release_manifest_json,
    write_oled_release_candidate_delta_jsonl,
    write_oled_release_candidate_entry_json,
    write_oled_release_candidate_snapshot_jsonl,
)


def _source_entry(
    *,
    global_append_status: str | OledGlobalAppendCandidateEntryStatus = OledGlobalAppendCandidateEntryStatus.GLOBAL_APPEND_CANDIDATE,
    metadata: dict | None = None,
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
) -> OledGlobalAppendCandidateEntry:
    return OledGlobalAppendCandidateEntry(
        global_append_entry_id="entry:oled-global-append-candidate:test",
        global_append_status=global_append_status,
        source_final_registry_writer_manifest_id="manifest:oled-final-registry-candidate:test",
        source_final_registry_entry_id="entry:oled-final-registry-candidate:test",
        source_global_append_preflight_status="passed",
        source_publication_entry_id="entry:oled-publication-candidate:test",
        source_publication_writer_manifest_id="manifest:oled-publication-candidate:test",
        source_promoted_entry_id="entry:oled-promoted-registry:test",
        source_promotion_writer_manifest_id="manifest:oled-promotion:test",
        source_registry_entry_id="entry:oled-registry:test",
        source_registry_writer_manifest_id="manifest:oled-registry:test",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
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


def _source_delta(
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
        source_publication_entry_id="entry:oled-publication-candidate:test",
        source_promoted_entry_id="entry:oled-promoted-registry:test",
        source_registry_entry_id="entry:oled-registry:test",
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


def _snapshot_record() -> OledFinalRegistryExistingRecordSummary:
    return OledFinalRegistryExistingRecordSummary(
        registry_entry_id="entry:existing-final-registry:prior",
        registry_status="candidate",
        source_publication_entry_id="entry:prior-publication",
        source_publication_writer_manifest_id="manifest:prior-publication",
        source_candidate_report_id="report:prior",
        source_benchmark_report_manifest_id="manifest:prior",
        metadata={
            "existing_registry_record": True,
            "global_append_entry_id": "entry:oled-global-append-candidate:test",
        },
    )


def _source_manifest(
    *,
    entry_sha: str | None = "global-append-entry-sha",
    delta_sha: str | None = "global-append-delta-sha",
    snapshot_sha: str | None = "global-registry-snapshot-sha",
    metadata: dict | None = None,
) -> OledFinalRegistryGlobalAppendWriterManifest:
    return OledFinalRegistryGlobalAppendWriterManifest(
        manifest_id="manifest:oled-global-append-candidate:test",
        source_final_registry_writer_manifest_id="manifest:oled-final-registry-candidate:test",
        source_final_registry_entry_id="entry:oled-final-registry-candidate:test",
        source_global_append_preflight_status="passed",
        existing_registry_snapshot_path=None,
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


def _preflight(
    *,
    status: OledGlobalAppendReleasePreflightStatus = OledGlobalAppendReleasePreflightStatus.PASSED,
    warning: bool = False,
    metadata: dict | None = None,
):
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(),
        global_append_delta_records=[_source_delta()],
        global_registry_snapshot_records=[_snapshot_record()],
        prior_registry_snapshot_records=[_snapshot_record()],
        policy=OledGlobalAppendReleasePreflightPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    if warning:
        report = report.model_copy(
            update={
                "status": OledGlobalAppendReleasePreflightStatus.PASSED_WITH_WARNINGS,
                "findings": [
                    *report.findings,
                    {"code": "synthetic_warning", "severity": "warning", "message": "synthetic warning"},
                ],
            }
        )
    if status != report.status:
        report = report.model_copy(update={"status": status})
    if metadata is not None:
        report = report.model_copy(update={"metadata": metadata})
    return report


def _write_source_package(tmp_path: Path) -> tuple[Path, Path]:
    source_entry = _source_entry()
    entry_path = tmp_path / oled_global_append_candidate_entry_filename()
    entry_sha = write_oled_global_append_candidate_entry_json(source_entry, entry_path)
    source_delta = [_source_delta(output_sha256=entry_sha)]
    delta_path = tmp_path / oled_global_append_candidate_delta_filename()
    delta_sha = write_oled_global_append_candidate_delta_jsonl(source_delta, delta_path)
    snapshot_path = tmp_path / oled_global_registry_snapshot_filename()
    snapshot_sha = write_oled_global_registry_snapshot_jsonl(existing_records=[_snapshot_record()], append_records=source_delta, path=snapshot_path)
    manifest = _source_manifest(entry_sha=entry_sha, delta_sha=delta_sha, snapshot_sha=snapshot_sha)
    manifest_path = tmp_path / "global_append_manifest.json"
    write_oled_final_registry_global_append_manifest_json(manifest, manifest_path)
    preflight = run_oled_global_append_release_preflight(
        global_append_writer_manifest=manifest,
        global_append_entry=source_entry,
        global_append_delta_records=source_delta,
        global_registry_snapshot_records=[_snapshot_record()],
        prior_registry_snapshot_records=[_snapshot_record()],
    )
    preflight_path = tmp_path / "release_preflight.json"
    write_oled_global_append_release_preflight_report_json(preflight, preflight_path)
    return manifest_path, preflight_path


def test_confirmation_gate() -> None:
    with pytest.raises(ValueError, match="confirmation_required:global_append_release_write"):
        select_oled_global_append_release_for_write(
            global_append_writer_manifest=_source_manifest(),
            global_append_entry=_source_entry(),
            global_append_delta_records=[_source_delta()],
            release_preflight_report=_preflight(),
        )


def test_build_release_candidate_entry_success() -> None:
    entry, findings = build_oled_release_candidate_entry(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(),
        global_append_delta_records=[_source_delta()],
        release_preflight_report=_preflight(),
        existing_registry_records=[_snapshot_record()],
        policy=OledGlobalAppendReleaseWriterPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    second_entry, _ = build_oled_release_candidate_entry(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(),
        global_append_delta_records=[_source_delta()],
        release_preflight_report=_preflight(),
    )

    assert findings == []
    assert entry is not None
    assert second_entry is not None
    assert entry.release_entry_id == second_entry.release_entry_id
    assert entry.release_status == OledReleaseCandidateEntryStatus.RELEASE_CANDIDATE
    assert entry.source_global_append_entry_id == "entry:oled-global-append-candidate:test"
    assert entry.source_release_preflight_status == "passed"
    assert entry.source_final_registry_entry_id == "entry:oled-final-registry-candidate:test"
    assert entry.metadata["benchmark_validated"] is False
    assert entry.metadata["scientific_claim_validated"] is False
    assert entry.metadata["global_registry_mutated"] is False


def test_invalid_release_preflight_blocks_writer() -> None:
    report = select_oled_global_append_release_for_write(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(),
        global_append_delta_records=[_source_delta()],
        release_preflight_report=_preflight(status=OledGlobalAppendReleasePreflightStatus.FAILED),
        confirm_global_append_release_write=True,
    )

    assert not report.is_valid
    assert report.release_entry is None
    assert "release_preflight_failed" in report.error_codes


def test_release_preflight_warnings_disallowed() -> None:
    report = select_oled_global_append_release_for_write(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(),
        global_append_delta_records=[_source_delta()],
        release_preflight_report=_preflight(warning=True),
        policy=OledGlobalAppendReleaseWriterPolicy(allow_release_preflight_warnings=False),
        confirm_global_append_release_write=True,
    )

    assert not report.is_valid
    assert "release_preflight_warnings_present" in report.error_codes


def test_validation_external_publication_and_global_claims_rejected() -> None:
    entry, findings = build_oled_release_candidate_entry(
        global_append_writer_manifest=_source_manifest(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "global_registry_mutated": True}
        ),
        global_append_entry=_source_entry(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "benchmark_published": True}
        ),
        global_append_delta_records=[
            _source_delta(metadata={"benchmark_validated": True, "scientific_claim_validated": True, "global_registry_mutated": True})
        ],
        release_preflight_report=_preflight(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "externally_published": True}
        ),
        existing_registry_records=[_snapshot_record()],
        policy=OledGlobalAppendReleaseWriterPolicy(benchmark_validated=True, scientific_claim_validated=True, externally_published=True),
    )

    assert entry is None
    codes = {finding.code for finding in findings}
    assert "benchmark_validated_source_claim" in codes
    assert "scientific_claim_validated_source_claim" in codes
    assert "external_publication_source_claim" in codes


def test_missing_caveats_cards_and_source_status() -> None:
    entry, findings = build_oled_release_candidate_entry(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(
            global_append_status=OledGlobalAppendCandidateEntryStatus.REJECTED,
            caveats=["baseline_candidate_report_only"],
            run_card_count=0,
            metric_card_count=0,
        ),
        global_append_delta_records=[_source_delta()],
        release_preflight_report=_preflight(),
    )

    assert entry is None
    codes = {finding.code for finding in findings}
    assert "global_append_status_not_candidate" in codes
    assert "missing_required_caveat" in codes
    assert "missing_run_cards" in codes
    assert "missing_metric_cards" in codes


def test_entry_json_writer_deterministic_and_redacted(tmp_path: Path) -> None:
    entry, _ = build_oled_release_candidate_entry(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(),
        global_append_delta_records=[_source_delta()],
        release_preflight_report=_preflight(),
    )
    assert entry is not None
    path = tmp_path / "release_entry.json"

    sha1 = write_oled_release_candidate_entry_json(entry, path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_release_candidate_entry_json(entry, path)
    loaded = load_oled_release_candidate_entry_json(path)

    assert sha1 == sha2
    assert first == path.read_text(encoding="utf-8")
    assert loaded == entry
    assert "prediction_id" not in first
    assert "raw_text" not in first
    assert "features" not in first


def test_delta_jsonl_writer_deterministic_and_redacted(tmp_path: Path) -> None:
    entry, _ = build_oled_release_candidate_entry(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(),
        global_append_delta_records=[_source_delta()],
        release_preflight_report=_preflight(),
    )
    assert entry is not None
    records = build_oled_release_candidate_delta_records(entry)
    path = tmp_path / "release_delta.jsonl"

    sha1 = write_oled_release_candidate_delta_jsonl(records, path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_release_candidate_delta_jsonl(records, path)
    loaded = load_oled_release_candidate_delta_jsonl(path)

    assert sha1 == sha2
    assert first.count("\n") == 1
    assert loaded == records
    assert "raw_text" not in first
    assert "features" not in first


def test_snapshot_writer_preserves_snapshot_records_before_release_records(tmp_path: Path) -> None:
    entry, _ = build_oled_release_candidate_entry(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(),
        global_append_delta_records=[_source_delta()],
        release_preflight_report=_preflight(),
    )
    assert entry is not None
    records = build_oled_release_candidate_delta_records(entry)
    path = tmp_path / "snapshot.jsonl"

    sha1 = write_oled_release_candidate_snapshot_jsonl([_snapshot_record()], records, path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_release_candidate_snapshot_jsonl([_snapshot_record()], records, path)
    lines = first.splitlines()

    assert sha1 == sha2
    assert len(lines) == 2
    assert "entry:existing-final-registry:prior" in lines[0]
    assert entry.release_entry_id in lines[1]
    assert "features" not in first


def test_manifest_writer_deterministic_and_includes_sha_safety_metadata(tmp_path: Path) -> None:
    report = select_oled_global_append_release_for_write(
        global_append_writer_manifest=_source_manifest(),
        global_append_entry=_source_entry(),
        global_append_delta_records=[_source_delta()],
        release_preflight_report=_preflight(),
        confirm_global_append_release_write=True,
    )
    manifest = report.manifest.model_copy(
        update={
            "file_results": [
                OledGlobalAppendReleaseFileResult(
                    artifact_kind="release_candidate_entry_json",
                    status=OledGlobalAppendReleaseWriteStatus.WRITTEN,
                    output_path="entry.json",
                    output_sha256="entry-sha",
                    reason_codes=["release_candidate_entry_json_written"],
                )
            ]
        }
    )
    path = tmp_path / "manifest.json"

    write_oled_global_append_release_manifest_json(manifest, path)
    first = path.read_text(encoding="utf-8")
    write_oled_global_append_release_manifest_json(manifest, path)

    assert first == path.read_text(encoding="utf-8")
    assert "entry-sha" in first
    assert '"benchmark_validated": false' in first
    assert '"global_registry_mutated": false' in first


def test_loaders_reject_invalid_json_and_raw_payload_leaks(tmp_path: Path) -> None:
    bad_entry = tmp_path / "bad_entry.json"
    bad_entry.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_release_candidate_entry_json:"):
        load_oled_release_candidate_entry_json(bad_entry)

    raw_entry = tmp_path / "raw_entry.json"
    raw_entry.write_text('{"release_entry_id":"x","features":{}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_release_candidate_entry_json:"):
        load_oled_release_candidate_entry_json(raw_entry)

    bad_delta = tmp_path / "bad_delta.jsonl"
    bad_delta.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_release_candidate_delta_jsonl:line-1"):
        load_oled_release_candidate_delta_jsonl(bad_delta)


def test_combined_runner_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    manifest_path, preflight_path = _write_source_package(tmp_path)
    output_manifest = tmp_path / "release_manifest.json"

    report = run_oled_global_append_release_writer_from_files(
        global_append_writer_manifest_path=manifest_path,
        release_preflight_report_path=preflight_path,
        global_append_base_dir=tmp_path,
        output_manifest_path=output_manifest,
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert not (tmp_path / oled_release_candidate_entry_filename()).exists()
    assert not (tmp_path / oled_release_candidate_delta_filename()).exists()
    assert not (tmp_path / oled_release_candidate_snapshot_filename()).exists()
    assert report.manifest.metadata["release_entry_written"] is False
    assert "dry_run_no_files_written" in report.manifest.file_results[0].reason_codes


def test_combined_runner_write_mode_writes_entry_delta_snapshot_manifest(tmp_path: Path) -> None:
    manifest_path, preflight_path = _write_source_package(tmp_path)
    prior_path = tmp_path / "prior_snapshot.jsonl"
    write_oled_release_candidate_snapshot_jsonl([_snapshot_record()], [], prior_path)
    output_dir = tmp_path / "release"
    output_manifest = tmp_path / "release_manifest.json"

    report = run_oled_global_append_release_writer_from_files(
        global_append_writer_manifest_path=manifest_path,
        release_preflight_report_path=preflight_path,
        global_append_base_dir=tmp_path,
        prior_registry_snapshot_path=prior_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        confirm_global_append_release_write=True,
    )

    assert report.is_valid
    assert (output_dir / oled_release_candidate_entry_filename()).exists()
    assert (output_dir / oled_release_candidate_delta_filename()).exists()
    assert (output_dir / oled_release_candidate_snapshot_filename()).exists()
    assert output_manifest.exists()
    assert all(result.output_sha256 for result in report.manifest.file_results if result.status == OledGlobalAppendReleaseWriteStatus.WRITTEN)


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, preflight_path = _write_source_package(tmp_path)
    output_dir = tmp_path / "cli-release"
    output_manifest = tmp_path / "cli-release-manifest.json"

    exit_code = main(
        [
            "--global-append-writer-manifest",
            str(manifest_path),
            "--release-preflight-report",
            str(preflight_path),
            "--global-append-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-global-append-release-write",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_manifest.exists()
    assert (output_dir / oled_release_candidate_entry_filename()).exists()
    assert "prediction_id" not in captured.out
    assert "features" not in captured.out


def test_package_exports() -> None:
    assert OledGlobalAppendReleaseWriterPolicy
    assert OledGlobalAppendReleaseWriteStatus
    assert OledReleaseCandidateEntryStatus
    assert OledReleaseCandidateEntry
    assert OledReleaseCandidateDeltaRecord
    assert OledGlobalAppendReleaseFileResult
    assert OledGlobalAppendReleaseWriterFinding
    assert OledGlobalAppendReleaseWriterManifest
    assert OledGlobalAppendReleaseWriterReport
    assert package_load_preflight
    assert package_build_entry
    assert package_build_delta
    assert package_select
    assert package_write_entry
    assert package_write_delta
    assert package_write_snapshot
    assert package_write_manifest
    assert package_load_entry
    assert package_load_delta
    assert package_entry_filename
    assert package_delta_filename
    assert package_snapshot_filename
    assert package_run_from_files
