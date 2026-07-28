from __future__ import annotations

from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledFinalRegistryCandidateEntry,
    OledFinalRegistryCandidateEntryStatus,
    OledFinalRegistryCandidateIndexRecord,
    OledPromotedRegistryPublicationFileResult,
    OledPromotedRegistryPublicationWriteStatus,
    OledPromotedRegistryPublicationWriterManifest,
    OledPromotedRegistryPublicationWriterPolicy,
    OledPublicationCandidateFinalRegistryFileResult,
    OledPublicationCandidateFinalRegistryPreflightPolicy,
    OledPublicationCandidateFinalRegistryPreflightStatus,
    OledPublicationCandidateFinalRegistryWriteStatus,
    OledPublicationCandidateFinalRegistryWriterFinding,
    OledPublicationCandidateFinalRegistryWriterManifest,
    OledPublicationCandidateFinalRegistryWriterPolicy,
    OledPublicationCandidateFinalRegistryWriterReport,
    OledPublicationCandidateRegistryEntry,
    OledPublicationCandidateRegistryEntryStatus,
    OledPublicationCandidateRegistryIndexRecord,
    build_oled_final_registry_candidate_entry as package_build_entry,
    build_oled_final_registry_candidate_index_records as package_build_index,
    load_oled_final_registry_candidate_entry_json as package_load_entry,
    load_oled_final_registry_candidate_index_jsonl as package_load_index,
    load_oled_publication_candidate_final_registry_preflight_report_json as package_load_preflight,
    oled_final_registry_candidate_entry_filename as package_entry_filename,
    oled_final_registry_candidate_index_filename as package_index_filename,
    run_oled_publication_candidate_final_registry_preflight,
    run_oled_publication_candidate_final_registry_writer_from_files as package_run_from_files,
    select_oled_publication_candidate_final_registry_for_write as package_select,
    write_oled_final_registry_candidate_entry_json as package_write_entry,
    write_oled_final_registry_candidate_index_jsonl as package_write_index,
    write_oled_publication_candidate_final_registry_manifest_json as package_write_manifest,
)
from ai4s_agent.domains.oled_curated_promoted_registry_publication_writer import (
    oled_publication_candidate_registry_entry_filename,
    oled_publication_candidate_registry_index_filename,
    write_oled_promoted_registry_publication_manifest_json,
    write_oled_publication_candidate_registry_entry_json,
    write_oled_publication_candidate_registry_index_jsonl,
)
from ai4s_agent.domains.oled_curated_publication_candidate_final_registry_preflight import (
    write_oled_publication_candidate_final_registry_preflight_report_json,
)
from ai4s_agent.domains.oled_curated_publication_candidate_final_registry_writer import (
    build_oled_final_registry_candidate_entry,
    build_oled_final_registry_candidate_index_records,
    load_oled_final_registry_candidate_entry_json,
    load_oled_final_registry_candidate_index_jsonl,
    load_oled_publication_candidate_final_registry_preflight_report_json,
    main,
    oled_final_registry_candidate_entry_filename,
    oled_final_registry_candidate_index_filename,
    run_oled_publication_candidate_final_registry_writer_from_files,
    select_oled_publication_candidate_final_registry_for_write,
    write_oled_final_registry_candidate_entry_json,
    write_oled_final_registry_candidate_index_jsonl,
    write_oled_publication_candidate_final_registry_manifest_json,
)


def _publication_entry(
    *,
    publication_status: str | OledPublicationCandidateRegistryEntryStatus = OledPublicationCandidateRegistryEntryStatus.PUBLICATION_CANDIDATE,
    metadata: dict | None = None,
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
    source_promotion_writer_manifest_id: str | None = "manifest:oled-benchmark-promoted-registry:test",
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


def _publication_index_record(
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
        source_promotion_writer_manifest_id="manifest:oled-benchmark-promoted-registry:test",
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


def _publication_manifest(
    *,
    entry_sha: str | None = "publication-entry-sha",
    index_sha: str | None = "publication-index-sha",
    metadata: dict | None = None,
) -> OledPromotedRegistryPublicationWriterManifest:
    return OledPromotedRegistryPublicationWriterManifest(
        manifest_id="manifest:oled-publication-candidate:test",
        source_promotion_writer_manifest_id="manifest:oled-benchmark-promoted-registry:test",
        source_promoted_entry_id="entry:oled-benchmark-promoted-registry:test",
        source_publication_preflight_status="passed",
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


def _final_registry_preflight(
    *,
    status: OledPublicationCandidateFinalRegistryPreflightStatus = OledPublicationCandidateFinalRegistryPreflightStatus.PASSED,
    warning: bool = False,
    metadata: dict | None = None,
):
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=_publication_manifest(),
        publication_entry=_publication_entry(),
        publication_index_records=[_publication_index_record()],
        policy=OledPublicationCandidateFinalRegistryPreflightPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    if warning:
        report = report.model_copy(
            update={
                "status": OledPublicationCandidateFinalRegistryPreflightStatus.PASSED_WITH_WARNINGS,
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


def _write_publication_package(tmp_path: Path):
    entry = _publication_entry()
    entry_path = tmp_path / oled_publication_candidate_registry_entry_filename()
    entry_sha = write_oled_publication_candidate_registry_entry_json(entry, entry_path)
    index = [_publication_index_record(output_sha256=entry_sha)]
    index_path = tmp_path / oled_publication_candidate_registry_index_filename()
    index_sha = write_oled_publication_candidate_registry_index_jsonl(index, index_path)
    manifest = _publication_manifest(entry_sha=entry_sha, index_sha=index_sha)
    manifest_path = tmp_path / "publication_writer_manifest.json"
    write_oled_promoted_registry_publication_manifest_json(manifest, manifest_path)
    preflight = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=manifest,
        publication_entry=entry,
        publication_index_records=index,
    )
    preflight_path = tmp_path / "final_registry_preflight.json"
    write_oled_publication_candidate_final_registry_preflight_report_json(preflight, preflight_path)
    return manifest_path, preflight_path, manifest, entry, index, preflight


def test_confirmation_gate() -> None:
    with pytest.raises(ValueError, match="confirmation_required:publication_candidate_final_registry_write"):
        select_oled_publication_candidate_final_registry_for_write(
            publication_writer_manifest=_publication_manifest(),
            publication_entry=_publication_entry(),
            publication_index_records=[_publication_index_record()],
            final_registry_preflight_report=_final_registry_preflight(),
        )


def test_build_final_registry_candidate_entry_success() -> None:
    entry, findings = build_oled_final_registry_candidate_entry(
        publication_writer_manifest=_publication_manifest(),
        publication_entry=_publication_entry(),
        publication_index_records=[_publication_index_record()],
        final_registry_preflight_report=_final_registry_preflight(),
        policy=OledPublicationCandidateFinalRegistryWriterPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    second_entry, _ = build_oled_final_registry_candidate_entry(
        publication_writer_manifest=_publication_manifest(),
        publication_entry=_publication_entry(),
        publication_index_records=[_publication_index_record()],
        final_registry_preflight_report=_final_registry_preflight(),
    )

    assert findings == []
    assert entry is not None
    assert second_entry is not None
    assert entry.final_registry_entry_id == second_entry.final_registry_entry_id
    assert entry.final_registry_status == OledFinalRegistryCandidateEntryStatus.FINAL_REGISTRY_CANDIDATE
    assert entry.source_publication_entry_id == "entry:oled-publication-candidate-registry:test"
    assert entry.source_final_registry_preflight_status == "passed"
    assert entry.source_registry_entry_id == "entry:oled-benchmark-registry:test"
    assert entry.metadata["benchmark_validated"] is False
    assert entry.metadata["scientific_claim_validated"] is False
    assert entry.metadata["global_registry_mutated"] is False


def test_invalid_final_registry_preflight_blocks_writer() -> None:
    report = select_oled_publication_candidate_final_registry_for_write(
        publication_writer_manifest=_publication_manifest(),
        publication_entry=_publication_entry(),
        publication_index_records=[_publication_index_record()],
        final_registry_preflight_report=_final_registry_preflight(status=OledPublicationCandidateFinalRegistryPreflightStatus.FAILED),
        confirm_publication_candidate_final_registry_write=True,
    )

    assert not report.is_valid
    assert report.final_registry_entry is None
    assert "final_registry_preflight_failed" in report.error_codes


def test_final_registry_preflight_warnings_disallowed() -> None:
    report = select_oled_publication_candidate_final_registry_for_write(
        publication_writer_manifest=_publication_manifest(),
        publication_entry=_publication_entry(),
        publication_index_records=[_publication_index_record()],
        final_registry_preflight_report=_final_registry_preflight(warning=True),
        policy=OledPublicationCandidateFinalRegistryWriterPolicy(allow_final_registry_preflight_warnings=False),
        confirm_publication_candidate_final_registry_write=True,
    )

    assert not report.is_valid
    assert "final_registry_preflight_warnings_present" in report.error_codes


def test_validation_and_final_registry_claims_rejected() -> None:
    policy = OledPublicationCandidateFinalRegistryWriterPolicy(
        benchmark_validated=True,
        scientific_claim_validated=True,
        globally_registered=True,
    )
    entry, findings = build_oled_final_registry_candidate_entry(
        publication_writer_manifest=_publication_manifest(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "global_registry_mutated": True}
        ),
        publication_entry=_publication_entry(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "final_registry_written": True}
        ),
        publication_index_records=[
            _publication_index_record(metadata={"benchmark_validated": True, "scientific_claim_validated": True, "global_registry_mutated": True})
        ],
        final_registry_preflight_report=_final_registry_preflight(
            metadata={"benchmark_validated": True, "scientific_claim_validated": True, "final_registry_written": True}
        ),
        policy=policy,
    )

    assert entry is None
    codes = {finding.code for finding in findings}
    assert "benchmark_validated_source_claim" in codes
    assert "scientific_claim_validated_source_claim" in codes
    assert "final_registry_source_claim" in codes


def test_missing_caveats_cards_and_publication_candidate_status() -> None:
    entry, findings = build_oled_final_registry_candidate_entry(
        publication_writer_manifest=_publication_manifest(),
        publication_entry=_publication_entry(
            publication_status=OledPublicationCandidateRegistryEntryStatus.REJECTED,
            caveats=["baseline_candidate_report_only"],
            run_card_count=0,
            metric_card_count=0,
        ),
        publication_index_records=[_publication_index_record()],
        final_registry_preflight_report=_final_registry_preflight(),
    )

    assert entry is None
    codes = {finding.code for finding in findings}
    assert "publication_status_not_publication_candidate" in codes
    assert "missing_required_caveat" in codes
    assert "missing_run_cards" in codes
    assert "missing_metric_cards" in codes


def test_final_registry_entry_json_writer_and_loader(tmp_path: Path) -> None:
    entry, _ = build_oled_final_registry_candidate_entry(
        publication_writer_manifest=_publication_manifest(),
        publication_entry=_publication_entry(),
        publication_index_records=[_publication_index_record()],
        final_registry_preflight_report=_final_registry_preflight(),
    )
    assert entry is not None
    path = tmp_path / "final_registry_entry.json"

    sha1 = write_oled_final_registry_candidate_entry_json(entry, path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_final_registry_candidate_entry_json(entry, path)
    second = path.read_text(encoding="utf-8")
    loaded = load_oled_final_registry_candidate_entry_json(path)

    assert sha1 == sha2
    assert first == second
    assert loaded == entry
    assert "prediction_id" not in first
    assert "raw_text" not in first
    assert "features" not in first


def test_final_registry_index_jsonl_writer_and_loader(tmp_path: Path) -> None:
    entry, _ = build_oled_final_registry_candidate_entry(
        publication_writer_manifest=_publication_manifest(),
        publication_entry=_publication_entry(),
        publication_index_records=[_publication_index_record()],
        final_registry_preflight_report=_final_registry_preflight(),
    )
    assert entry is not None
    records = build_oled_final_registry_candidate_index_records(entry)
    path = tmp_path / "final_registry_index.jsonl"

    sha1 = write_oled_final_registry_candidate_index_jsonl(records, path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_final_registry_candidate_index_jsonl(records, path)
    loaded = load_oled_final_registry_candidate_index_jsonl(path)

    assert sha1 == sha2
    assert first.count("\n") == 1
    assert loaded == records
    assert "raw_text" not in first
    assert "features" not in first


def test_manifest_writer_is_deterministic(tmp_path: Path) -> None:
    report = select_oled_publication_candidate_final_registry_for_write(
        publication_writer_manifest=_publication_manifest(),
        publication_entry=_publication_entry(),
        publication_index_records=[_publication_index_record()],
        final_registry_preflight_report=_final_registry_preflight(),
        confirm_publication_candidate_final_registry_write=True,
    )
    manifest = report.manifest.model_copy(
        update={
            "file_results": [
                OledPublicationCandidateFinalRegistryFileResult(
                    artifact_kind="final_registry_candidate_entry_json",
                    status=OledPublicationCandidateFinalRegistryWriteStatus.WRITTEN,
                    output_path="entry.json",
                    output_sha256="entry-sha",
                    reason_codes=["final_registry_candidate_entry_json_written"],
                )
            ]
        }
    )
    path = tmp_path / "manifest.json"

    write_oled_publication_candidate_final_registry_manifest_json(manifest, path)
    first = path.read_text(encoding="utf-8")
    write_oled_publication_candidate_final_registry_manifest_json(manifest, path)

    assert first == path.read_text(encoding="utf-8")
    assert "entry-sha" in first
    assert '"benchmark_validated": false' in first
    assert '"global_registry_mutated": false' in first


def test_loaders_reject_invalid_json_and_raw_payload_leaks(tmp_path: Path) -> None:
    bad_entry = tmp_path / "bad_entry.json"
    bad_entry.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_final_registry_candidate_entry_json:"):
        load_oled_final_registry_candidate_entry_json(bad_entry)

    raw_entry = tmp_path / "raw_entry.json"
    raw_entry.write_text('{"final_registry_entry_id":"x","features":{}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_final_registry_candidate_entry_json:"):
        load_oled_final_registry_candidate_entry_json(raw_entry)

    bad_index = tmp_path / "bad_index.jsonl"
    bad_index.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_final_registry_candidate_index_jsonl:line-1"):
        load_oled_final_registry_candidate_index_jsonl(bad_index)


def test_combined_runner_dry_run(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _, _, _ = _write_publication_package(tmp_path)
    output_manifest = tmp_path / "final_registry_manifest.json"

    report = run_oled_publication_candidate_final_registry_writer_from_files(
        publication_writer_manifest_path=manifest_path,
        final_registry_preflight_report_path=preflight_path,
        publication_candidate_base_dir=tmp_path,
        output_manifest_path=output_manifest,
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert not (tmp_path / oled_final_registry_candidate_entry_filename()).exists()
    assert report.manifest.metadata["final_registry_candidate_entry_written"] is False
    assert "dry_run_no_files_written" in report.manifest.file_results[0].reason_codes


def test_combined_runner_write_mode(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _, _, _ = _write_publication_package(tmp_path)
    output_dir = tmp_path / "final-registry"
    output_manifest = tmp_path / "final_registry_manifest.json"

    report = run_oled_publication_candidate_final_registry_writer_from_files(
        publication_writer_manifest_path=manifest_path,
        final_registry_preflight_report_path=preflight_path,
        publication_candidate_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        confirm_publication_candidate_final_registry_write=True,
    )

    assert report.is_valid
    assert (output_dir / oled_final_registry_candidate_entry_filename()).exists()
    assert (output_dir / oled_final_registry_candidate_index_filename()).exists()
    assert output_manifest.exists()
    assert all(
        result.output_sha256
        for result in report.manifest.file_results
        if result.status == OledPublicationCandidateFinalRegistryWriteStatus.WRITTEN
    )


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, preflight_path, _, _, _, _ = _write_publication_package(tmp_path)
    output_dir = tmp_path / "cli-final-registry"
    output_manifest = tmp_path / "cli-final-registry-manifest.json"

    exit_code = main(
        [
            "--publication-writer-manifest",
            str(manifest_path),
            "--final-registry-preflight-report",
            str(preflight_path),
            "--publication-candidate-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-publication-candidate-final-registry-write",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_manifest.exists()
    assert (output_dir / oled_final_registry_candidate_entry_filename()).exists()
    assert "prediction_id" not in captured.out
    assert "features" not in captured.out


def test_package_exports() -> None:
    assert OledPublicationCandidateFinalRegistryWriterPolicy
    assert OledPublicationCandidateFinalRegistryWriteStatus
    assert OledFinalRegistryCandidateEntryStatus
    assert OledFinalRegistryCandidateEntry
    assert OledFinalRegistryCandidateIndexRecord
    assert OledPublicationCandidateFinalRegistryFileResult
    assert OledPublicationCandidateFinalRegistryWriterFinding
    assert OledPublicationCandidateFinalRegistryWriterManifest
    assert OledPublicationCandidateFinalRegistryWriterReport
    assert package_load_preflight
    assert package_build_entry
    assert package_build_index
    assert package_select
    assert package_write_entry
    assert package_write_index
    assert package_write_manifest
    assert package_load_entry
    assert package_load_index
    assert package_entry_filename() == "oled_final_registry_candidate_entry.json"
    assert package_index_filename() == "oled_final_registry_candidate_index.jsonl"
    assert package_run_from_files
