from __future__ import annotations

from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledBenchmarkPromotedRegistryEntry,
    OledBenchmarkPromotedRegistryEntryStatus,
    OledBenchmarkPromotedRegistryIndexRecord,
    OledBenchmarkRegistryEntry,
    OledBenchmarkRegistryEntryStatus,
    OledBenchmarkRegistryFileResult,
    OledBenchmarkRegistryIndexRecord,
    OledBenchmarkRegistryPromotionFileResult,
    OledBenchmarkRegistryPromotionPreflightPolicy,
    OledBenchmarkRegistryPromotionPreflightStatus,
    OledBenchmarkRegistryPromotionWriteStatus,
    OledBenchmarkRegistryPromotionWriterFinding,
    OledBenchmarkRegistryPromotionWriterManifest,
    OledBenchmarkRegistryPromotionWriterPolicy,
    OledBenchmarkRegistryPromotionWriterReport,
    OledBenchmarkRegistryWriteStatus,
    OledBenchmarkRegistryWriterManifest,
    OledBenchmarkRegistryWriterPolicy,
    build_oled_benchmark_promoted_registry_entry as package_build_entry,
    build_oled_benchmark_promoted_registry_index_records as package_build_index,
    load_oled_benchmark_promoted_registry_entry_json as package_load_entry,
    load_oled_benchmark_promoted_registry_index_jsonl as package_load_index,
    load_oled_benchmark_registry_promotion_preflight_report_json as package_load_preflight,
    oled_benchmark_promoted_registry_entry_filename as package_entry_filename,
    oled_benchmark_promoted_registry_index_filename as package_index_filename,
    run_oled_benchmark_registry_promotion_preflight,
    run_oled_benchmark_registry_promotion_writer_from_files as package_run_from_files,
    select_oled_benchmark_registry_promotion_for_write as package_select,
    write_oled_benchmark_promoted_registry_entry_json as package_write_entry,
    write_oled_benchmark_promoted_registry_index_jsonl as package_write_index,
    write_oled_benchmark_registry_entry_json,
    write_oled_benchmark_registry_index_jsonl,
    write_oled_benchmark_registry_manifest_json,
    write_oled_benchmark_registry_promotion_manifest_json as package_write_manifest,
    write_oled_benchmark_registry_promotion_preflight_report_json,
)
from ai4s_agent.domains.oled_curated_benchmark_registry_promotion_writer import (
    build_oled_benchmark_promoted_registry_entry,
    build_oled_benchmark_promoted_registry_index_records,
    load_oled_benchmark_promoted_registry_entry_json,
    load_oled_benchmark_promoted_registry_index_jsonl,
    load_oled_benchmark_registry_promotion_preflight_report_json,
    main,
    oled_benchmark_promoted_registry_entry_filename,
    oled_benchmark_promoted_registry_index_filename,
    run_oled_benchmark_registry_promotion_writer_from_files,
    select_oled_benchmark_registry_promotion_for_write,
    write_oled_benchmark_promoted_registry_entry_json,
    write_oled_benchmark_promoted_registry_index_jsonl,
    write_oled_benchmark_registry_promotion_manifest_json,
)


def _entry(
    *,
    registry_status: str | OledBenchmarkRegistryEntryStatus = OledBenchmarkRegistryEntryStatus.CANDIDATE,
    metadata: dict | None = None,
    caveats: list[str] | None = None,
    run_card_count: int = 1,
    metric_card_count: int = 3,
) -> OledBenchmarkRegistryEntry:
    return OledBenchmarkRegistryEntry(
        registry_entry_id="entry:oled-benchmark-registry:test",
        registry_status=registry_status,
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        source_benchmark_registry_preflight_status="passed",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
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
    metadata: dict | None = None,
    output_sha256: str | None = "entry-sha",
) -> OledBenchmarkRegistryIndexRecord:
    return OledBenchmarkRegistryIndexRecord(
        registry_entry_id=registry_entry_id,
        registry_status="candidate",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        source_benchmark_registry_preflight_status="passed",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        run_card_count=1,
        metric_card_count=3,
        output_registry_entry_json_path="oled_benchmark_registry_entry.json",
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


def _preflight(*, status: OledBenchmarkRegistryPromotionPreflightStatus = OledBenchmarkRegistryPromotionPreflightStatus.PASSED, warning: bool = False, metadata: dict | None = None):
    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record()],
        policy=OledBenchmarkRegistryPromotionPreflightPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    if warning:
        report = report.model_copy(
            update={
                "status": OledBenchmarkRegistryPromotionPreflightStatus.PASSED_WITH_WARNINGS,
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


def _write_source_registry_package(tmp_path: Path):
    entry = _entry()
    entry_path = tmp_path / "oled_benchmark_registry_entry.json"
    entry_sha = write_oled_benchmark_registry_entry_json(entry, entry_path)
    index = [_index_record(output_sha256=entry_sha)]
    index_path = tmp_path / "oled_benchmark_registry_index.jsonl"
    index_sha = write_oled_benchmark_registry_index_jsonl(index, index_path)
    manifest = _manifest(entry_sha=entry_sha, index_sha=index_sha)
    manifest_path = tmp_path / "benchmark_registry_manifest.json"
    write_oled_benchmark_registry_manifest_json(manifest, manifest_path)
    preflight = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=manifest,
        registry_entry=entry,
        registry_index_records=index,
    )
    preflight_path = tmp_path / "promotion_preflight.json"
    write_oled_benchmark_registry_promotion_preflight_report_json(preflight, preflight_path)
    return manifest_path, preflight_path, manifest, entry, index, preflight


def test_confirmation_gate() -> None:
    with pytest.raises(ValueError, match="confirmation_required:benchmark_registry_promotion_write"):
        select_oled_benchmark_registry_promotion_for_write(
            registry_writer_manifest=_manifest(),
            registry_entry=_entry(),
            registry_index_records=[_index_record()],
            promotion_preflight_report=_preflight(),
        )


def test_build_promoted_entry_success() -> None:
    promoted, findings = build_oled_benchmark_promoted_registry_entry(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record()],
        promotion_preflight_report=_preflight(),
        policy=OledBenchmarkRegistryPromotionWriterPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )

    assert findings == []
    assert promoted is not None
    assert promoted.promotion_status == OledBenchmarkPromotedRegistryEntryStatus.PROMOTED_CANDIDATE
    assert promoted.source_registry_entry_id == "entry:oled-benchmark-registry:test"
    assert promoted.source_registry_promotion_preflight_status == "passed"
    assert promoted.metadata["benchmark_validated"] is False
    assert promoted.metadata["scientific_claim_validated"] is False


def test_invalid_promotion_preflight_blocks_writer() -> None:
    report = select_oled_benchmark_registry_promotion_for_write(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record()],
        promotion_preflight_report=_preflight(status=OledBenchmarkRegistryPromotionPreflightStatus.FAILED),
        confirm_benchmark_registry_promotion_write=True,
    )

    assert not report.is_valid
    assert report.promoted_entry is None
    assert "promotion_preflight_failed" in report.error_codes


def test_promotion_preflight_warnings_disallowed() -> None:
    report = select_oled_benchmark_registry_promotion_for_write(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record()],
        promotion_preflight_report=_preflight(warning=True),
        policy=OledBenchmarkRegistryPromotionWriterPolicy(allow_promotion_preflight_warnings=False),
        confirm_benchmark_registry_promotion_write=True,
    )

    assert not report.is_valid
    assert "promotion_preflight_warnings_present" in report.error_codes


def test_validation_claims_rejected() -> None:
    policy = OledBenchmarkRegistryPromotionWriterPolicy(benchmark_validated=True, scientific_claim_validated=True)
    promoted, findings = build_oled_benchmark_promoted_registry_entry(
        registry_writer_manifest=_manifest(metadata={"benchmark_validated": True, "scientific_claim_validated": True}),
        registry_entry=_entry(metadata={"benchmark_validated": True, "scientific_claim_validated": True}),
        registry_index_records=[_index_record(metadata={"benchmark_validated": True, "scientific_claim_validated": True})],
        promotion_preflight_report=_preflight(metadata={"benchmark_validated": True, "scientific_claim_validated": True}),
        policy=policy,
    )

    assert promoted is None
    codes = {finding.code for finding in findings}
    assert "benchmark_validated_source_claim" in codes
    assert "scientific_claim_validated_source_claim" in codes


def test_missing_caveats_cards_and_candidate_status() -> None:
    promoted, findings = build_oled_benchmark_promoted_registry_entry(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(
            registry_status=OledBenchmarkRegistryEntryStatus.REJECTED,
            caveats=["baseline_candidate_report_only"],
            run_card_count=0,
            metric_card_count=0,
        ),
        registry_index_records=[_index_record()],
        promotion_preflight_report=_preflight(),
    )

    assert promoted is None
    codes = {finding.code for finding in findings}
    assert "registry_status_not_candidate" in codes
    assert "missing_required_caveat" in codes
    assert "missing_run_cards" in codes
    assert "missing_metric_cards" in codes


def test_promoted_entry_json_writer_and_loader(tmp_path: Path) -> None:
    promoted, _ = build_oled_benchmark_promoted_registry_entry(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record()],
        promotion_preflight_report=_preflight(),
    )
    assert promoted is not None
    path = tmp_path / "promoted_entry.json"

    sha1 = write_oled_benchmark_promoted_registry_entry_json(promoted, path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_benchmark_promoted_registry_entry_json(promoted, path)
    second = path.read_text(encoding="utf-8")
    loaded = load_oled_benchmark_promoted_registry_entry_json(path)

    assert sha1 == sha2
    assert first == second
    assert loaded == promoted
    assert "prediction_id" not in first
    assert "raw_text" not in first
    assert "features" not in first


def test_promoted_index_jsonl_writer_and_loader(tmp_path: Path) -> None:
    promoted, _ = build_oled_benchmark_promoted_registry_entry(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record()],
        promotion_preflight_report=_preflight(),
    )
    assert promoted is not None
    records = build_oled_benchmark_promoted_registry_index_records(promoted)
    path = tmp_path / "promoted_index.jsonl"

    sha1 = write_oled_benchmark_promoted_registry_index_jsonl(records, path)
    first = path.read_text(encoding="utf-8")
    sha2 = write_oled_benchmark_promoted_registry_index_jsonl(records, path)
    loaded = load_oled_benchmark_promoted_registry_index_jsonl(path)

    assert sha1 == sha2
    assert first.count("\n") == 1
    assert loaded == records
    assert "raw_text" not in first
    assert "features" not in first


def test_manifest_writer_is_deterministic(tmp_path: Path) -> None:
    report = select_oled_benchmark_registry_promotion_for_write(
        registry_writer_manifest=_manifest(),
        registry_entry=_entry(),
        registry_index_records=[_index_record()],
        promotion_preflight_report=_preflight(),
        confirm_benchmark_registry_promotion_write=True,
    )
    manifest = report.manifest.model_copy(
        update={
            "file_results": [
                OledBenchmarkRegistryPromotionFileResult(
                    artifact_kind="promoted_registry_entry_json",
                    status=OledBenchmarkRegistryPromotionWriteStatus.WRITTEN,
                    output_path="entry.json",
                    output_sha256="entry-sha",
                    reason_codes=["promoted_registry_entry_json_written"],
                )
            ]
        }
    )
    path = tmp_path / "manifest.json"

    write_oled_benchmark_registry_promotion_manifest_json(manifest, path)
    first = path.read_text(encoding="utf-8")
    write_oled_benchmark_registry_promotion_manifest_json(manifest, path)

    assert first == path.read_text(encoding="utf-8")
    assert "entry-sha" in first
    assert "benchmark_validated" in first
    assert "true" not in first.lower().split('"benchmark_validated": ')[1].split(",", 1)[0]


def test_loaders_reject_invalid_json(tmp_path: Path) -> None:
    bad_entry = tmp_path / "bad_entry.json"
    bad_entry.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_benchmark_promoted_registry_entry_json:"):
        load_oled_benchmark_promoted_registry_entry_json(bad_entry)

    bad_index = tmp_path / "bad_index.jsonl"
    bad_index.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_benchmark_promoted_registry_index_jsonl:line-1"):
        load_oled_benchmark_promoted_registry_index_jsonl(bad_index)


def test_combined_runner_dry_run(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _, _, _ = _write_source_registry_package(tmp_path)
    output_manifest = tmp_path / "promotion_manifest.json"

    report = run_oled_benchmark_registry_promotion_writer_from_files(
        registry_writer_manifest_path=manifest_path,
        registry_promotion_preflight_report_path=preflight_path,
        registry_base_dir=tmp_path,
        output_manifest_path=output_manifest,
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert not (tmp_path / "promoted").exists()
    assert report.manifest.metadata["promoted_registry_entry_written"] is False


def test_combined_runner_write_mode(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _, _, _ = _write_source_registry_package(tmp_path)
    output_dir = tmp_path / "promoted"
    output_manifest = tmp_path / "promotion_manifest.json"

    report = run_oled_benchmark_registry_promotion_writer_from_files(
        registry_writer_manifest_path=manifest_path,
        registry_promotion_preflight_report_path=preflight_path,
        registry_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        confirm_benchmark_registry_promotion_write=True,
    )

    assert report.is_valid
    assert (output_dir / oled_benchmark_promoted_registry_entry_filename()).exists()
    assert (output_dir / oled_benchmark_promoted_registry_index_filename()).exists()
    assert output_manifest.exists()
    assert all(result.output_sha256 for result in report.manifest.file_results if result.status == OledBenchmarkRegistryPromotionWriteStatus.WRITTEN)


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, preflight_path, _, _, _, _ = _write_source_registry_package(tmp_path)
    output_dir = tmp_path / "cli-promoted"
    output_manifest = tmp_path / "cli-promotion-manifest.json"

    exit_code = main(
        [
            "--registry-writer-manifest",
            str(manifest_path),
            "--registry-promotion-preflight-report",
            str(preflight_path),
            "--registry-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-benchmark-registry-promotion-write",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_manifest.exists()
    assert (output_dir / oled_benchmark_promoted_registry_entry_filename()).exists()
    assert "prediction_id" not in captured.out
    assert "features" not in captured.out


def test_package_exports() -> None:
    assert OledBenchmarkRegistryPromotionWriterPolicy
    assert OledBenchmarkRegistryPromotionWriteStatus
    assert OledBenchmarkPromotedRegistryEntryStatus
    assert OledBenchmarkPromotedRegistryEntry
    assert OledBenchmarkPromotedRegistryIndexRecord
    assert OledBenchmarkRegistryPromotionFileResult
    assert OledBenchmarkRegistryPromotionWriterFinding
    assert OledBenchmarkRegistryPromotionWriterManifest
    assert OledBenchmarkRegistryPromotionWriterReport
    assert package_load_preflight
    assert package_build_entry
    assert package_build_index
    assert package_select
    assert package_write_entry
    assert package_write_index
    assert package_write_manifest
    assert package_load_entry
    assert package_load_index
    assert package_entry_filename() == "oled_benchmark_promoted_registry_entry.json"
    assert package_index_filename() == "oled_benchmark_promoted_registry_index.jsonl"
    assert package_run_from_files
