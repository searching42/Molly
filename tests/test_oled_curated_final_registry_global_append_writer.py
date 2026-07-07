from __future__ import annotations

from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledFinalRegistryCandidateEntry,
    OledFinalRegistryCandidateEntryStatus,
    OledFinalRegistryCandidateIndexRecord,
    OledFinalRegistryExistingRecordSummary,
    OledFinalRegistryGlobalAppendPreflightPolicy,
    OledFinalRegistryGlobalAppendPreflightStatus,
    OledFinalRegistryGlobalAppendWriteStatus,
    OledFinalRegistryGlobalAppendWriterFinding,
    OledFinalRegistryGlobalAppendWriterManifest,
    OledFinalRegistryGlobalAppendWriterPolicy,
    OledFinalRegistryGlobalAppendWriterReport,
    OledGlobalAppendCandidateEntry,
    OledGlobalAppendCandidateEntryStatus,
    OledGlobalAppendCandidateIndexRecord,
    OledGlobalAppendFileResult,
    OledPublicationCandidateFinalRegistryFileResult,
    OledPublicationCandidateFinalRegistryWriteStatus,
    OledPublicationCandidateFinalRegistryWriterManifest,
    OledPublicationCandidateFinalRegistryWriterPolicy,
    build_oled_global_append_candidate_entry as package_build_entry,
    build_oled_global_append_candidate_index_records as package_build_index,
    load_oled_final_registry_global_append_preflight_report_json as package_load_preflight,
    load_oled_global_append_candidate_delta_jsonl as package_load_delta,
    load_oled_global_append_candidate_entry_json as package_load_entry,
    oled_global_append_candidate_delta_filename as package_delta_filename,
    oled_global_append_candidate_entry_filename as package_entry_filename,
    oled_global_registry_snapshot_filename as package_snapshot_filename,
    run_oled_final_registry_global_append_preflight,
    run_oled_final_registry_global_append_writer_from_files as package_run_from_files,
    select_oled_final_registry_global_append_for_write as package_select,
    write_oled_final_registry_global_append_manifest_json as package_write_manifest,
    write_oled_global_append_candidate_delta_jsonl as package_write_delta,
    write_oled_global_append_candidate_entry_json as package_write_entry,
    write_oled_global_registry_snapshot_jsonl as package_write_snapshot,
)
from ai4s_agent.domains.oled_curated_final_registry_global_append_preflight import (
    write_oled_final_registry_global_append_preflight_report_json,
)
from ai4s_agent.domains.oled_curated_final_registry_global_append_writer import (
    build_oled_global_append_candidate_entry,
    build_oled_global_append_candidate_index_records,
    load_oled_final_registry_global_append_preflight_report_json,
    load_oled_global_append_candidate_delta_jsonl,
    load_oled_global_append_candidate_entry_json,
    main,
    oled_global_append_candidate_delta_filename,
    oled_global_append_candidate_entry_filename,
    oled_global_registry_snapshot_filename,
    run_oled_final_registry_global_append_writer_from_files,
    select_oled_final_registry_global_append_for_write,
    write_oled_final_registry_global_append_manifest_json,
    write_oled_global_append_candidate_delta_jsonl,
    write_oled_global_append_candidate_entry_json,
    write_oled_global_registry_snapshot_jsonl,
)
from ai4s_agent.domains.oled_curated_publication_candidate_final_registry_writer import (
    oled_final_registry_candidate_entry_filename,
    oled_final_registry_candidate_index_filename,
    write_oled_final_registry_candidate_entry_json,
    write_oled_final_registry_candidate_index_jsonl,
    write_oled_publication_candidate_final_registry_manifest_json,
)


def _entry(
    *,
    final_registry_status: str | OledFinalRegistryCandidateEntryStatus = OledFinalRegistryCandidateEntryStatus.FINAL_REGISTRY_CANDIDATE,
    metadata: dict | None = None,
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
) -> OledFinalRegistryCandidateEntry:
    return OledFinalRegistryCandidateEntry(
        final_registry_entry_id="entry:oled-final-registry-candidate:test",
        final_registry_status=final_registry_status,
        source_publication_writer_manifest_id="manifest:oled-publication-candidate-final-registry:test",
        source_publication_entry_id="entry:oled-publication-candidate-registry:test",
        source_final_registry_preflight_status="passed",
        source_promoted_entry_id="entry:oled-benchmark-promoted-registry:test",
        source_promotion_writer_manifest_id="manifest:oled-benchmark-promoted-registry:test",
        source_registry_entry_id="entry:oled-benchmark-registry:test",
        source_registry_writer_manifest_id="manifest:oled-benchmark-registry:test",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
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
) -> OledPublicationCandidateFinalRegistryWriterManifest:
    return OledPublicationCandidateFinalRegistryWriterManifest(
        manifest_id="manifest:oled-final-registry-candidate:test",
        source_publication_writer_manifest_id="manifest:oled-publication-candidate-final-registry:test",
        source_publication_entry_id="entry:oled-publication-candidate-registry:test",
        source_final_registry_preflight_status="passed",
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


def _preflight(
    *,
    status: OledFinalRegistryGlobalAppendPreflightStatus = OledFinalRegistryGlobalAppendPreflightStatus.PASSED,
    warning: bool = False,
    metadata: dict | None = None,
):
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        existing_registry_records=[_existing()],
        policy=OledFinalRegistryGlobalAppendPreflightPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    if warning:
        report = report.model_copy(
            update={
                "status": OledFinalRegistryGlobalAppendPreflightStatus.PASSED_WITH_WARNINGS,
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


def _write_final_registry_package(
    tmp_path: Path,
) -> tuple[Path, Path, OledPublicationCandidateFinalRegistryWriterManifest, OledFinalRegistryCandidateEntry, list[OledFinalRegistryCandidateIndexRecord]]:
    entry = _entry()
    entry_path = tmp_path / oled_final_registry_candidate_entry_filename()
    entry_sha = write_oled_final_registry_candidate_entry_json(entry, entry_path)
    index = [_index_record(output_sha256=entry_sha)]
    index_path = tmp_path / oled_final_registry_candidate_index_filename()
    index_sha = write_oled_final_registry_candidate_index_jsonl(index, index_path)
    manifest = _manifest(entry_sha=entry_sha, index_sha=index_sha)
    manifest_path = tmp_path / "final_registry_manifest.json"
    write_oled_publication_candidate_final_registry_manifest_json(manifest, manifest_path)
    preflight = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=manifest,
        final_registry_entry=entry,
        final_registry_index_records=index,
        existing_registry_records=[_existing()],
    )
    preflight_path = tmp_path / "global_append_preflight.json"
    write_oled_final_registry_global_append_preflight_report_json(preflight, preflight_path)
    return manifest_path, preflight_path, manifest, entry, index


def test_confirmation_gate() -> None:
    with pytest.raises(ValueError, match="confirmation_required:final_registry_global_append_write"):
        select_oled_final_registry_global_append_for_write(
            final_registry_writer_manifest=_manifest(),
            final_registry_entry=_entry(),
            final_registry_index_records=[_index_record()],
            global_append_preflight_report=_preflight(),
        )


def test_build_global_append_candidate_entry_success() -> None:
    entry, findings = build_oled_global_append_candidate_entry(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        global_append_preflight_report=_preflight(),
        existing_registry_records=[_existing()],
        policy=OledFinalRegistryGlobalAppendWriterPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    second_entry, _ = build_oled_global_append_candidate_entry(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        global_append_preflight_report=_preflight(),
        existing_registry_records=[_existing()],
    )

    assert findings == []
    assert entry is not None
    assert second_entry is not None
    assert entry.global_append_entry_id == second_entry.global_append_entry_id
    assert entry.global_append_status == OledGlobalAppendCandidateEntryStatus.GLOBAL_APPEND_CANDIDATE
    assert entry.source_final_registry_entry_id == "entry:oled-final-registry-candidate:test"
    assert entry.source_global_append_preflight_status == "passed"
    assert entry.source_publication_entry_id == "entry:oled-publication-candidate-registry:test"
    assert entry.metadata["benchmark_validated"] is False
    assert entry.metadata["scientific_claim_validated"] is False
    assert entry.metadata["global_registry_mutated"] is False


def test_invalid_global_append_preflight_blocks_writer() -> None:
    report = select_oled_final_registry_global_append_for_write(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        global_append_preflight_report=_preflight(status=OledFinalRegistryGlobalAppendPreflightStatus.FAILED),
        confirm_final_registry_global_append_write=True,
    )

    assert not report.is_valid
    assert report.global_append_entry is None
    assert "global_append_preflight_failed" in report.error_codes


def test_global_append_preflight_warnings_disallowed() -> None:
    report = select_oled_final_registry_global_append_for_write(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        global_append_preflight_report=_preflight(warning=True),
        policy=OledFinalRegistryGlobalAppendWriterPolicy(allow_global_append_preflight_warnings=False),
        confirm_final_registry_global_append_write=True,
    )

    assert not report.is_valid
    assert "global_append_preflight_warnings_present" in report.error_codes


def test_validation_external_publication_and_global_claims_rejected() -> None:
    policy = OledFinalRegistryGlobalAppendWriterPolicy(
        benchmark_validated=True,
        scientific_claim_validated=True,
        externally_published=True,
        global_append_status="global_append_candidate",
    )
    entry, findings = build_oled_global_append_candidate_entry(
        final_registry_writer_manifest=_manifest(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "global_registry_mutated": True}
        ),
        final_registry_entry=_entry(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "benchmark_published": True}
        ),
        final_registry_index_records=[
            _index_record(metadata={"benchmark_validated": True, "scientific_claim_validated": True, "global_registry_mutated": True})
        ],
        global_append_preflight_report=_preflight(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "benchmark_published": True}
        ),
        existing_registry_records=[_existing(metadata={"benchmark_validated": True, "scientific_claim_validated": True, "globally_registered": True})],
        policy=policy,
    )

    assert entry is None
    codes = {finding.code for finding in findings}
    assert "benchmark_validated_source_claim" in codes
    assert "scientific_claim_validated_source_claim" in codes
    assert "external_publication_source_claim" in codes


def test_missing_caveats_cards_and_final_registry_candidate_status() -> None:
    entry, findings = build_oled_global_append_candidate_entry(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(
            final_registry_status=OledFinalRegistryCandidateEntryStatus.REJECTED,
            caveats=["baseline_candidate_report_only"],
            run_card_count=0,
            metric_card_count=0,
        ),
        final_registry_index_records=[_index_record()],
        global_append_preflight_report=_preflight(),
    )

    assert entry is None
    codes = {finding.code for finding in findings}
    assert "final_registry_status_not_candidate" in codes
    assert "missing_required_caveat" in codes
    assert "missing_run_cards" in codes
    assert "missing_metric_cards" in codes


def test_entry_json_writer_deterministic_and_redacted(tmp_path: Path) -> None:
    entry, _ = build_oled_global_append_candidate_entry(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        global_append_preflight_report=_preflight(),
    )
    assert entry is not None
    path = tmp_path / "global_append_entry.json"

    sha1 = write_oled_global_append_candidate_entry_json(entry, path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_global_append_candidate_entry_json(entry, path)
    loaded = load_oled_global_append_candidate_entry_json(path)

    assert sha1 == sha2
    assert first == path.read_text(encoding="utf-8")
    assert loaded == entry
    assert "prediction_id" not in first
    assert "raw_text" not in first
    assert "features" not in first


def test_delta_jsonl_writer_deterministic_and_redacted(tmp_path: Path) -> None:
    entry, _ = build_oled_global_append_candidate_entry(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        global_append_preflight_report=_preflight(),
    )
    assert entry is not None
    records = build_oled_global_append_candidate_index_records(entry)
    path = tmp_path / "global_append_delta.jsonl"

    sha1 = write_oled_global_append_candidate_delta_jsonl(records, path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_global_append_candidate_delta_jsonl(records, path)
    loaded = load_oled_global_append_candidate_delta_jsonl(path)

    assert sha1 == sha2
    assert first.count("\n") == 1
    assert loaded == records
    assert "raw_text" not in first
    assert "features" not in first


def test_snapshot_writer_preserves_existing_records_before_append_records(tmp_path: Path) -> None:
    entry, _ = build_oled_global_append_candidate_entry(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        global_append_preflight_report=_preflight(),
    )
    assert entry is not None
    records = build_oled_global_append_candidate_index_records(entry)
    path = tmp_path / "snapshot.jsonl"

    sha1 = write_oled_global_registry_snapshot_jsonl(existing_records=[_existing()], append_records=records, path=path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_global_registry_snapshot_jsonl(existing_records=[_existing()], append_records=records, path=path)
    lines = first.splitlines()

    assert sha1 == sha2
    assert len(lines) == 2
    assert "entry:existing-final-registry:other" in lines[0]
    assert entry.global_append_entry_id in lines[1]
    assert "features" not in first


def test_manifest_writer_deterministic_and_includes_sha_safety_metadata(tmp_path: Path) -> None:
    report = select_oled_final_registry_global_append_for_write(
        final_registry_writer_manifest=_manifest(),
        final_registry_entry=_entry(),
        final_registry_index_records=[_index_record()],
        global_append_preflight_report=_preflight(),
        confirm_final_registry_global_append_write=True,
    )
    manifest = report.manifest.model_copy(
        update={
            "file_results": [
                OledGlobalAppendFileResult(
                    artifact_kind="global_append_entry_json",
                    status=OledFinalRegistryGlobalAppendWriteStatus.WRITTEN,
                    output_path="entry.json",
                    output_sha256="entry-sha",
                    reason_codes=["global_append_candidate_entry_json_written"],
                )
            ]
        }
    )
    path = tmp_path / "manifest.json"

    write_oled_final_registry_global_append_manifest_json(manifest, path)
    first = path.read_text(encoding="utf-8")
    write_oled_final_registry_global_append_manifest_json(manifest, path)

    assert first == path.read_text(encoding="utf-8")
    assert "entry-sha" in first
    assert '"benchmark_validated": false' in first
    assert '"global_registry_mutated": false' in first


def test_loaders_reject_invalid_json_and_raw_payload_leaks(tmp_path: Path) -> None:
    bad_entry = tmp_path / "bad_entry.json"
    bad_entry.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_global_append_candidate_entry_json:"):
        load_oled_global_append_candidate_entry_json(bad_entry)

    raw_entry = tmp_path / "raw_entry.json"
    raw_entry.write_text('{"global_append_entry_id":"x","features":{}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_global_append_candidate_entry_json:"):
        load_oled_global_append_candidate_entry_json(raw_entry)

    bad_delta = tmp_path / "bad_delta.jsonl"
    bad_delta.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_global_append_candidate_delta_jsonl:line-1"):
        load_oled_global_append_candidate_delta_jsonl(bad_delta)


def test_combined_runner_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _, _ = _write_final_registry_package(tmp_path)
    output_manifest = tmp_path / "global_append_manifest.json"

    report = run_oled_final_registry_global_append_writer_from_files(
        final_registry_writer_manifest_path=manifest_path,
        global_append_preflight_report_path=preflight_path,
        final_registry_candidate_base_dir=tmp_path,
        output_manifest_path=output_manifest,
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert not (tmp_path / oled_global_append_candidate_entry_filename()).exists()
    assert not (tmp_path / oled_global_append_candidate_delta_filename()).exists()
    assert not (tmp_path / oled_global_registry_snapshot_filename()).exists()
    assert report.manifest.metadata["global_append_entry_written"] is False
    assert "dry_run_no_files_written" in report.manifest.file_results[0].reason_codes


def test_combined_runner_write_mode_writes_entry_delta_snapshot_manifest(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _, _ = _write_final_registry_package(tmp_path)
    existing_path = tmp_path / "existing_snapshot.jsonl"
    write_oled_global_registry_snapshot_jsonl(existing_records=[_existing()], append_records=[], path=existing_path)
    output_dir = tmp_path / "global-append"
    output_manifest = tmp_path / "global_append_manifest.json"

    report = run_oled_final_registry_global_append_writer_from_files(
        final_registry_writer_manifest_path=manifest_path,
        global_append_preflight_report_path=preflight_path,
        final_registry_candidate_base_dir=tmp_path,
        existing_registry_snapshot_path=existing_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        confirm_final_registry_global_append_write=True,
    )

    assert report.is_valid
    assert (output_dir / oled_global_append_candidate_entry_filename()).exists()
    assert (output_dir / oled_global_append_candidate_delta_filename()).exists()
    assert (output_dir / oled_global_registry_snapshot_filename()).exists()
    assert output_manifest.exists()
    assert all(
        result.output_sha256
        for result in report.manifest.file_results
        if result.status == OledFinalRegistryGlobalAppendWriteStatus.WRITTEN
    )


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, preflight_path, _, _, _ = _write_final_registry_package(tmp_path)
    output_dir = tmp_path / "cli-global-append"
    output_manifest = tmp_path / "cli-global-append-manifest.json"

    exit_code = main(
        [
            "--final-registry-writer-manifest",
            str(manifest_path),
            "--global-append-preflight-report",
            str(preflight_path),
            "--final-registry-candidate-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-final-registry-global-append-write",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_manifest.exists()
    assert (output_dir / oled_global_append_candidate_entry_filename()).exists()
    assert "prediction_id" not in captured.out
    assert "features" not in captured.out


def test_package_exports() -> None:
    assert OledFinalRegistryGlobalAppendWriterPolicy
    assert OledFinalRegistryGlobalAppendWriteStatus
    assert OledGlobalAppendCandidateEntryStatus
    assert OledGlobalAppendCandidateEntry
    assert OledGlobalAppendCandidateIndexRecord
    assert OledGlobalAppendFileResult
    assert OledFinalRegistryGlobalAppendWriterFinding
    assert OledFinalRegistryGlobalAppendWriterManifest
    assert OledFinalRegistryGlobalAppendWriterReport
    assert package_load_preflight
    assert package_build_entry
    assert package_build_index
    assert package_select
    assert package_write_entry
    assert package_write_delta
    assert package_write_snapshot
    assert package_write_manifest
    assert package_load_entry
    assert package_load_delta
    assert package_entry_filename() == "oled_global_append_candidate_entry.json"
    assert package_delta_filename() == "oled_global_append_candidate_delta.jsonl"
    assert package_snapshot_filename() == "oled_global_registry_snapshot.jsonl"
    assert package_run_from_files
