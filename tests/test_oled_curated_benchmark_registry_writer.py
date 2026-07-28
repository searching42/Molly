from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.domains import (
    OledBaselineBenchmarkCandidateReport,
    OledBaselineBenchmarkMetricCard,
    OledBaselineBenchmarkReportFileResult,
    OledBaselineBenchmarkReportWriteStatus,
    OledBaselineBenchmarkReportWriterManifest,
    OledBaselineBenchmarkReportWriterPolicy,
    OledBaselineBenchmarkRunCard,
    OledBenchmarkRegistryEntry,
    OledBenchmarkRegistryEntryStatus,
    OledBenchmarkRegistryFileResult,
    OledBenchmarkRegistryIndexRecord,
    OledBenchmarkRegistryPreflightPolicy,
    OledBenchmarkRegistryPreflightStatus,
    OledBenchmarkRegistryWriteStatus,
    OledBenchmarkRegistryWriterFinding,
    OledBenchmarkRegistryWriterManifest,
    OledBenchmarkRegistryWriterPolicy,
    OledBenchmarkRegistryWriterReport,
    build_oled_benchmark_registry_entry as package_build_entry,
    build_oled_benchmark_registry_index_records as package_build_index,
    load_oled_benchmark_registry_entry_json as package_load_entry,
    load_oled_benchmark_registry_index_jsonl as package_load_index,
    load_oled_benchmark_registry_preflight_report_json as package_load_preflight,
    load_oled_baseline_benchmark_report_artifacts_from_manifest,
    oled_benchmark_registry_entry_filename as package_entry_filename,
    oled_benchmark_registry_index_filename as package_index_filename,
    run_oled_benchmark_registry_preflight,
    run_oled_benchmark_registry_writer_from_files as package_run_from_files,
    select_oled_benchmark_registry_entry_for_write as package_select_entry,
    write_oled_baseline_benchmark_report_json,
    write_oled_baseline_benchmark_report_manifest_json,
    write_oled_baseline_benchmark_report_markdown,
    write_oled_benchmark_registry_entry_json as package_write_entry,
    write_oled_benchmark_registry_index_jsonl as package_write_index,
    write_oled_benchmark_registry_manifest_json as package_write_manifest,
    write_oled_benchmark_registry_preflight_report_json,
)
from ai4s_agent.domains.oled_curated_benchmark_registry_writer import (
    build_oled_benchmark_registry_entry,
    build_oled_benchmark_registry_index_records,
    load_oled_benchmark_registry_entry_json,
    load_oled_benchmark_registry_index_jsonl,
    load_oled_benchmark_registry_preflight_report_json,
    main,
    oled_benchmark_registry_entry_filename,
    oled_benchmark_registry_index_filename,
    run_oled_benchmark_registry_writer_from_files,
    select_oled_benchmark_registry_entry_for_write,
    write_oled_benchmark_registry_entry_json,
    write_oled_benchmark_registry_index_jsonl,
    write_oled_benchmark_registry_manifest_json,
)


def _metric_card(split: str) -> OledBaselineBenchmarkMetricCard:
    return OledBaselineBenchmarkMetricCard(
        baseline_kind="mean_baseline",
        target_property_id="eqe_percent",
        feature_view="full_context",
        split=split,
        row_count=2 if split == "train" else 1,
        mae=2.0 if split == "train" else 4.0,
        rmse=2.0 if split == "train" else 4.0,
        r2=0.0,
        bias=0.0,
        target_mean=22.0,
        prediction_mean=22.0,
        metric_status="ready",
        reason_codes=["metrics_consistent"],
    )


def _run_card(*, metrics: list[OledBaselineBenchmarkMetricCard] | None = None) -> OledBaselineBenchmarkRunCard:
    metric_cards = metrics if metrics is not None else [_metric_card("train"), _metric_card("validation"), _metric_card("test")]
    return OledBaselineBenchmarkRunCard(
        baseline_kind="mean_baseline",
        target_property_id="eqe_percent",
        feature_view="full_context",
        run_status="completed",
        prediction_count=4,
        train_row_count=2,
        validation_row_count=1,
        test_row_count=1,
        metric_splits=[metric.split for metric in metric_cards],
        reason_codes=["selected_for_report"],
        prediction_artifact_sha256="prediction-sha",
        metrics_artifact_sha256="metrics-sha",
        metrics=metric_cards,
        metadata={"benchmark_candidate_run_card": True, "benchmark_validated": False},
    )


def _candidate_report(
    *,
    caveats: list[str] | None = None,
    run_cards: list[OledBaselineBenchmarkRunCard] | None = None,
    metadata: dict | None = None,
) -> OledBaselineBenchmarkCandidateReport:
    cards = run_cards if run_cards is not None else [_run_card()]
    return OledBaselineBenchmarkCandidateReport(
        report_id="report:oled-baseline-benchmark:test",
        source_baseline_run_manifest_id="oled-baseline-runner:test",
        source_benchmark_preflight_status="passed",
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        splits=["train", "validation", "test"],
        input_prediction_count=4,
        input_metric_count=3,
        run_cards=cards,
        coverage_status_counts={"ready": 1},
        metric_status_counts={"ready": 3},
        finding_code_counts={},
        caveats=caveats
        if caveats is not None
        else ["baseline_candidate_report_only", "not_benchmark_validated", "not_scientific_performance_claim"],
        metadata=metadata
        if metadata is not None
        else {
            "benchmark_candidate_report": True,
            "benchmark_validated": False,
            "benchmark_registered": False,
            "scientific_claim_validated": False,
        },
    )


def _report_manifest(*, json_sha: str | None = "json-sha", markdown_sha: str | None = "markdown-sha", metadata: dict | None = None) -> OledBaselineBenchmarkReportWriterManifest:
    return OledBaselineBenchmarkReportWriterManifest(
        manifest_id="manifest:oled-baseline-benchmark:test",
        source_baseline_run_manifest_id="oled-baseline-runner:test",
        source_benchmark_preflight_status="passed",
        output_directory="benchmark_report",
        output_file_count=2,
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledBaselineBenchmarkReportFileResult(
                artifact_kind="benchmark_report_json",
                status=OledBaselineBenchmarkReportWriteStatus.WRITTEN,
                output_path="candidate_report.json",
                output_sha256=json_sha,
                reason_codes=["report_json_written"],
            ),
            OledBaselineBenchmarkReportFileResult(
                artifact_kind="benchmark_report_markdown",
                status=OledBaselineBenchmarkReportWriteStatus.WRITTEN,
                output_path="candidate_report.md",
                output_sha256=markdown_sha,
                reason_codes=["report_markdown_written"],
            ),
        ],
        policy=OledBaselineBenchmarkReportWriterPolicy(),
        metadata=metadata
        if metadata is not None
        else {
            "benchmark_report_writer": True,
            "benchmark_validated": False,
            "benchmark_registered": False,
            "scientific_claim_validated": False,
        },
    )


def _registry_preflight(
    *,
    status: OledBenchmarkRegistryPreflightStatus = OledBenchmarkRegistryPreflightStatus.PASSED,
    warning: bool = False,
    metadata: dict | None = None,
):
    report = run_oled_benchmark_registry_preflight(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(),
        markdown_report=_markdown(),
        policy=OledBenchmarkRegistryPreflightPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )
    if warning:
        report = report.model_copy(
            update={
                "status": OledBenchmarkRegistryPreflightStatus.PASSED_WITH_WARNINGS,
                "findings": [
                    *report.findings,
                    {
                        "code": "synthetic_warning",
                        "severity": "warning",
                        "message": "synthetic warning",
                    },
                ],
            }
        )
    if status != report.status:
        report = report.model_copy(update={"status": status})
    if metadata is not None:
        report = report.model_copy(update={"metadata": metadata})
    return report


def _markdown(report: OledBaselineBenchmarkCandidateReport | None = None) -> str:
    selected = report or _candidate_report()
    return (
        "# OLED Baseline Benchmark Candidate Report\n\n"
        f"- Report id: `{selected.report_id}`\n"
        "- `baseline_candidate_report_only`\n"
        "- `not_benchmark_validated`\n"
        "- `not_scientific_performance_claim`\n\n"
        "This candidate report is not a benchmark registration record and does not validate scientific performance.\n"
    )


def _write_report_package(tmp_path: Path):
    candidate_report = _candidate_report()
    json_path = tmp_path / "candidate_report.json"
    json_sha = write_oled_baseline_benchmark_report_json(candidate_report, json_path)
    markdown_path = tmp_path / "candidate_report.md"
    markdown_sha = write_oled_baseline_benchmark_report_markdown(candidate_report, markdown_path)
    report_manifest = _report_manifest(json_sha=json_sha, markdown_sha=markdown_sha)
    manifest_path = tmp_path / "benchmark_report_manifest.json"
    write_oled_baseline_benchmark_report_manifest_json(report_manifest, manifest_path)
    loaded_report, markdown = load_oled_baseline_benchmark_report_artifacts_from_manifest(
        manifest=report_manifest,
        base_dir=tmp_path,
    )
    registry_preflight = run_oled_benchmark_registry_preflight(
        report_manifest=report_manifest,
        candidate_report=loaded_report,
        markdown_report=markdown,
    )
    preflight_path = tmp_path / "registry_preflight.json"
    write_oled_benchmark_registry_preflight_report_json(registry_preflight, preflight_path)
    return manifest_path, preflight_path, report_manifest, candidate_report, registry_preflight


def test_confirmation_gate() -> None:
    with pytest.raises(ValueError, match="confirmation_required:benchmark_registry_write"):
        select_oled_benchmark_registry_entry_for_write(
            report_manifest=_report_manifest(),
            candidate_report=_candidate_report(),
            registry_preflight_report=_registry_preflight(),
        )


def test_build_registry_entry_success() -> None:
    entry, findings = build_oled_benchmark_registry_entry(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(),
        registry_preflight_report=_registry_preflight(),
        policy=OledBenchmarkRegistryWriterPolicy(target_property_ids=["eqe_percent"], feature_views=["full_context"]),
    )

    assert findings == []
    assert entry is not None
    assert entry.registry_status == OledBenchmarkRegistryEntryStatus.CANDIDATE
    assert entry.source_benchmark_report_manifest_id == "manifest:oled-baseline-benchmark:test"
    assert entry.source_candidate_report_id == "report:oled-baseline-benchmark:test"
    assert entry.run_card_count == 1
    assert entry.metric_card_count == 3
    assert entry.metadata["benchmark_validated"] is False
    assert entry.metadata["scientific_claim_validated"] is False
    assert "selected_for_registry" in entry.reason_codes


def test_invalid_registry_preflight_blocks_writer() -> None:
    report = select_oled_benchmark_registry_entry_for_write(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(),
        registry_preflight_report=_registry_preflight(status=OledBenchmarkRegistryPreflightStatus.FAILED),
        confirm_benchmark_registry_write=True,
    )

    assert report.registry_entry is None
    assert "registry_preflight_failed" in report.error_codes


def test_registry_preflight_warnings_disallowed() -> None:
    report = select_oled_benchmark_registry_entry_for_write(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(),
        registry_preflight_report=_registry_preflight(warning=True),
        policy=OledBenchmarkRegistryWriterPolicy(allow_registry_preflight_warnings=False),
        confirm_benchmark_registry_write=True,
    )

    assert report.registry_entry is None
    assert "registry_preflight_warnings_present" in report.error_codes


def test_validation_claims_rejected() -> None:
    cases = [
        (_report_manifest(metadata={"benchmark_validated": True}), _candidate_report(), _registry_preflight()),
        (_report_manifest(metadata={"scientific_claim_validated": True}), _candidate_report(), _registry_preflight()),
        (_report_manifest(), _candidate_report(metadata={"benchmark_validated": True}), _registry_preflight()),
        (_report_manifest(), _candidate_report(metadata={"scientific_claim_validated": True}), _registry_preflight()),
        (_report_manifest(), _candidate_report(), _registry_preflight(metadata={"benchmark_validated": True})),
        (_report_manifest(), _candidate_report(), _registry_preflight(metadata={"scientific_claim_validated": True})),
    ]
    for manifest, candidate_report, preflight in cases:
        entry, findings = build_oled_benchmark_registry_entry(
            report_manifest=manifest,
            candidate_report=candidate_report,
            registry_preflight_report=preflight,
        )
        assert entry is None
        assert set(finding.code for finding in findings) & {
            "benchmark_validated_source_claim",
            "scientific_claim_validated_source_claim",
        }

    entry, policy_findings = build_oled_benchmark_registry_entry(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(),
        registry_preflight_report=_registry_preflight(),
        policy=OledBenchmarkRegistryWriterPolicy(benchmark_validated=True),
    )
    assert entry is None
    assert "benchmark_validated_source_claim" in [finding.code for finding in policy_findings]


def test_missing_caveats_and_cards_rejected() -> None:
    missing_caveat, missing_caveat_findings = build_oled_benchmark_registry_entry(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(caveats=["baseline_candidate_report_only"]),
        registry_preflight_report=_registry_preflight(),
    )
    no_runs, no_run_findings = build_oled_benchmark_registry_entry(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(run_cards=[]),
        registry_preflight_report=_registry_preflight(),
    )
    no_metrics, no_metric_findings = build_oled_benchmark_registry_entry(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(run_cards=[_run_card(metrics=[])]),
        registry_preflight_report=_registry_preflight(),
    )

    assert missing_caveat is None
    assert "missing_required_caveat" in [finding.code for finding in missing_caveat_findings]
    assert no_runs is None
    assert "missing_run_cards" in [finding.code for finding in no_run_findings]
    assert no_metrics is None
    assert "missing_metric_cards" in [finding.code for finding in no_metric_findings]


def test_registry_entry_json_writer_and_loader(tmp_path: Path) -> None:
    entry, _ = build_oled_benchmark_registry_entry(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(),
        registry_preflight_report=_registry_preflight(),
    )
    assert entry is not None
    path = tmp_path / oled_benchmark_registry_entry_filename()
    first_sha = write_oled_benchmark_registry_entry_json(entry, path)
    first_text = path.read_text(encoding="utf-8")
    second_sha = write_oled_benchmark_registry_entry_json(entry, path)

    assert first_sha == second_sha
    assert load_oled_benchmark_registry_entry_json(path) == entry
    assert "Raw Paper Text" not in first_text
    assert "features" not in first_text
    assert "prediction_id" not in first_text
    assert '"benchmark_validated": false' in first_text


def test_registry_index_jsonl_writer_and_loader(tmp_path: Path) -> None:
    entry, _ = build_oled_benchmark_registry_entry(
        report_manifest=_report_manifest(),
        candidate_report=_candidate_report(),
        registry_preflight_report=_registry_preflight(),
    )
    assert entry is not None
    records = build_oled_benchmark_registry_index_records(entry)
    path = tmp_path / oled_benchmark_registry_index_filename()
    first_sha = write_oled_benchmark_registry_index_jsonl(records, path)
    first_text = path.read_text(encoding="utf-8")
    second_sha = write_oled_benchmark_registry_index_jsonl(records, path)

    assert first_sha == second_sha
    assert first_text.count("\n") == 1
    assert load_oled_benchmark_registry_index_jsonl(path) == records
    assert "Raw Paper Text" not in first_text
    assert "features" not in first_text


def test_manifest_writer_is_deterministic(tmp_path: Path) -> None:
    manifest = OledBenchmarkRegistryWriterManifest(
        manifest_id="manifest:oled-benchmark-registry:test",
        source_benchmark_report_manifest_id="manifest:oled-baseline-benchmark:test",
        source_benchmark_registry_preflight_status="passed",
        source_candidate_report_id="report:oled-baseline-benchmark:test",
        output_directory="benchmark_registry",
        output_file_count=2,
        registry_entry_ids=["registry-entry:test"],
        baseline_kinds=["mean_baseline"],
        target_property_ids=["eqe_percent"],
        feature_views=["full_context"],
        file_results=[
            OledBenchmarkRegistryFileResult(
                artifact_kind="registry_entry_json",
                status=OledBenchmarkRegistryWriteStatus.WRITTEN,
                output_path="oled_benchmark_registry_entry.json",
                output_sha256="abc123",
                reason_codes=["registry_entry_json_written"],
            )
        ],
        policy=OledBenchmarkRegistryWriterPolicy(),
        metadata={"benchmark_registry_writer": True, "benchmark_validated": False},
    )
    path = tmp_path / "manifest.json"
    write_oled_benchmark_registry_manifest_json(manifest, path)
    first_text = path.read_text(encoding="utf-8")
    write_oled_benchmark_registry_manifest_json(manifest, path)

    assert first_text == path.read_text(encoding="utf-8")
    assert "abc123" in first_text
    assert '"benchmark_validated": false' in first_text


def test_loaders_reject_invalid_json(tmp_path: Path) -> None:
    invalid_entry = tmp_path / "entry.json"
    invalid_entry.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_benchmark_registry_entry_json:"):
        load_oled_benchmark_registry_entry_json(invalid_entry)

    invalid_index = tmp_path / "index.jsonl"
    invalid_index.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_benchmark_registry_index_jsonl:line-1"):
        load_oled_benchmark_registry_index_jsonl(invalid_index)


def test_combined_runner_dry_run(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _, _ = _write_report_package(tmp_path)
    output_manifest = tmp_path / "registry_manifest.json"

    report = run_oled_benchmark_registry_writer_from_files(
        benchmark_report_manifest_path=manifest_path,
        benchmark_registry_preflight_report_path=preflight_path,
        benchmark_report_base_dir=tmp_path,
        output_dir=tmp_path / "registry",
        output_manifest_path=output_manifest,
        dry_run=True,
    )

    assert report.is_valid
    assert output_manifest.exists()
    assert not (tmp_path / "registry").exists()
    assert report.manifest.metadata["benchmark_registry_entry_written"] is False


def test_combined_runner_write_mode(tmp_path: Path) -> None:
    manifest_path, preflight_path, _, _, _ = _write_report_package(tmp_path)
    output_dir = tmp_path / "registry"
    output_manifest = tmp_path / "registry_manifest.json"

    report = run_oled_benchmark_registry_writer_from_files(
        benchmark_report_manifest_path=manifest_path,
        benchmark_registry_preflight_report_path=preflight_path,
        benchmark_report_base_dir=tmp_path,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        confirm_benchmark_registry_write=True,
    )

    assert report.is_valid
    assert (output_dir / oled_benchmark_registry_entry_filename()).exists()
    assert (output_dir / oled_benchmark_registry_index_filename()).exists()
    assert output_manifest.exists()
    assert all(result.output_sha256 for result in report.manifest.file_results if result.status == OledBenchmarkRegistryWriteStatus.WRITTEN)


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, preflight_path, _, _, _ = _write_report_package(tmp_path)
    output_dir = tmp_path / "cli-registry"
    output_manifest = tmp_path / "cli-registry-manifest.json"

    exit_code = main(
        [
            "--benchmark-report-manifest",
            str(manifest_path),
            "--benchmark-registry-preflight-report",
            str(preflight_path),
            "--benchmark-report-base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-manifest",
            str(output_manifest),
            "--confirm-benchmark-registry-write",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_manifest.exists()
    assert (output_dir / oled_benchmark_registry_entry_filename()).exists()
    assert "prediction_id" not in captured.out
    assert "candidate_report" not in captured.out


def test_package_exports() -> None:
    assert OledBenchmarkRegistryWriterPolicy
    assert OledBenchmarkRegistryWriteStatus
    assert OledBenchmarkRegistryEntryStatus
    assert OledBenchmarkRegistryEntry
    assert OledBenchmarkRegistryIndexRecord
    assert OledBenchmarkRegistryFileResult
    assert OledBenchmarkRegistryWriterFinding
    assert OledBenchmarkRegistryWriterManifest
    assert OledBenchmarkRegistryWriterReport
    assert package_load_preflight
    assert package_build_entry
    assert package_build_index
    assert package_select_entry
    assert package_write_entry
    assert package_write_index
    assert package_write_manifest
    assert package_load_entry
    assert package_load_index
    assert package_entry_filename() == "oled_benchmark_registry_entry.json"
    assert package_index_filename() == "oled_benchmark_registry_index.jsonl"
    assert package_run_from_files
