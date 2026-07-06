from __future__ import annotations

import hashlib
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
    OledBenchmarkRegistryArtifactStatus,
    OledBenchmarkRegistryPreflightFinding,
    OledBenchmarkRegistryPreflightPolicy,
    OledBenchmarkRegistryPreflightReport,
    OledBenchmarkRegistryPreflightStatus,
    OledBenchmarkRegistryRunSummary,
    OledBenchmarkReportArtifactSummary,
    load_oled_baseline_benchmark_candidate_report_json as package_load_json,
    load_oled_baseline_benchmark_candidate_report_markdown as package_load_markdown,
    load_oled_baseline_benchmark_report_artifacts_from_manifest as package_load_artifacts,
    load_oled_baseline_benchmark_report_writer_manifest_json as package_load_manifest,
    run_oled_benchmark_registry_preflight as package_run_preflight,
    run_oled_benchmark_registry_preflight_from_files as package_run_from_files,
    write_oled_baseline_benchmark_report_json,
    write_oled_baseline_benchmark_report_manifest_json,
    write_oled_baseline_benchmark_report_markdown,
    write_oled_benchmark_registry_preflight_report_json as package_write_report,
)
from ai4s_agent.domains.oled_curated_benchmark_registry_preflight import (
    load_oled_baseline_benchmark_candidate_report_json,
    load_oled_baseline_benchmark_candidate_report_markdown,
    load_oled_baseline_benchmark_report_artifacts_from_manifest,
    load_oled_baseline_benchmark_report_writer_manifest_json,
    main,
    run_oled_benchmark_registry_preflight,
    run_oled_benchmark_registry_preflight_from_files,
    write_oled_benchmark_registry_preflight_report_json,
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
    source_baseline_id: str = "oled-baseline-runner:test",
    source_preflight_status: str = "passed",
    metadata: dict | None = None,
) -> OledBaselineBenchmarkCandidateReport:
    cards = run_cards if run_cards is not None else [_run_card()]
    return OledBaselineBenchmarkCandidateReport(
        report_id="report:oled-baseline-benchmark:test",
        source_baseline_run_manifest_id=source_baseline_id,
        source_benchmark_preflight_status=source_preflight_status,
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


def _markdown(report: OledBaselineBenchmarkCandidateReport | None = None) -> str:
    selected = report or _candidate_report()
    return (
        "# OLED Baseline Benchmark Candidate Report\n\n"
        f"- Report id: `{selected.report_id}`\n"
        "- `baseline_candidate_report_only`\n"
        "- `not_benchmark_validated`\n"
        "- `not_scientific_performance_claim`\n\n"
        "| Baseline | Target | Feature view | Split | Rows | MAE |\n"
        "| --- | --- | --- | --- | ---: | ---: |\n"
        "| mean_baseline | eqe_percent | full_context | validation | 1 | 4.0 |\n\n"
        "This candidate report is not a benchmark registration record and does not validate scientific performance.\n"
    )


def _manifest(*, json_sha: str | None = "json-sha", markdown_sha: str | None = "markdown-sha", metadata: dict | None = None) -> OledBaselineBenchmarkReportWriterManifest:
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


def _write_report_artifacts(tmp_path: Path):
    candidate_report = _candidate_report()
    json_path = tmp_path / "candidate_report.json"
    json_sha = write_oled_baseline_benchmark_report_json(candidate_report, json_path)
    markdown_path = tmp_path / "candidate_report.md"
    markdown_sha = write_oled_baseline_benchmark_report_markdown(candidate_report, markdown_path)
    manifest = _manifest(json_sha=json_sha, markdown_sha=markdown_sha)
    manifest_path = tmp_path / "benchmark_report_manifest.json"
    write_oled_baseline_benchmark_report_manifest_json(manifest, manifest_path)
    return manifest_path, manifest, candidate_report


def test_main_preflight_success() -> None:
    report = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(),
        markdown_report=_markdown(),
    )

    assert report.is_valid
    assert report.status == OledBenchmarkRegistryPreflightStatus.PASSED
    assert report.source_report_manifest_id == "manifest:oled-baseline-benchmark:test"
    assert report.source_baseline_run_manifest_id == "oled-baseline-runner:test"
    assert report.report_id == "report:oled-baseline-benchmark:test"
    assert report.input_run_card_count == 1
    assert report.input_metric_card_count == 3
    assert report.artifact_summaries[0].status == OledBenchmarkRegistryArtifactStatus.READY
    assert report.run_summaries[0].artifact_status == OledBenchmarkRegistryArtifactStatus.READY
    assert report.metadata["benchmark_registry_preflight_only"] is True
    assert report.metadata["benchmark_registered"] is False


def test_missing_candidate_report_fails() -> None:
    report = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=None,
        markdown_report=_markdown(),
    )

    assert not report.is_valid
    assert "missing_benchmark_candidate_report_json" in report.error_codes


def test_missing_markdown_can_be_allowed() -> None:
    failed = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(),
        markdown_report=None,
    )
    allowed = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(),
        markdown_report=None,
        policy=OledBenchmarkRegistryPreflightPolicy(require_markdown_report=False),
    )

    assert "missing_benchmark_candidate_report_markdown" in failed.error_codes
    assert allowed.is_valid


def test_benchmark_validated_and_registered_claims_rejected() -> None:
    cases = [
        (_manifest(metadata={"benchmark_validated": True}), _candidate_report(), _markdown()),
        (_manifest(metadata={"benchmark_registered": True}), _candidate_report(), _markdown()),
        (_manifest(), _candidate_report(metadata={"benchmark_validated": True}), _markdown()),
        (_manifest(), _candidate_report(metadata={"benchmark_registered": True}), _markdown()),
        (_manifest(), _candidate_report(metadata={"scientific_claim_validated": True}), _markdown()),
    ]

    for manifest, candidate_report, markdown in cases:
        report = run_oled_benchmark_registry_preflight(
            report_manifest=manifest,
            candidate_report=candidate_report,
            markdown_report=markdown,
        )
        assert not report.is_valid
        assert set(report.error_codes) & {
            "benchmark_validated_source_claim",
            "benchmark_registered_source_claim",
            "scientific_claim_validated_source_claim",
        }


def test_missing_caveats_fail() -> None:
    report = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(caveats=["baseline_candidate_report_only"]),
        markdown_report=_markdown(),
    )

    assert not report.is_valid
    assert "missing_required_caveat" in report.error_codes


def test_source_id_mismatch_fails() -> None:
    report = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(source_baseline_id="different-source"),
        markdown_report=_markdown(),
    )

    assert not report.is_valid
    assert "source_id_mismatch" in report.error_codes


def test_missing_run_or_metric_cards_fail() -> None:
    no_runs = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(run_cards=[]),
        markdown_report=_markdown(),
    )
    no_metrics = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(run_cards=[_run_card(metrics=[])]),
        markdown_report=_markdown(),
    )

    assert "missing_run_cards" in no_runs.error_codes
    assert "missing_metric_cards" in no_metrics.error_codes


def test_markdown_mismatch_and_leakage_fail() -> None:
    missing_id = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(),
        markdown_report="This candidate report is not a benchmark registration record and does not validate scientific performance.",
    )
    leaked = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(),
        markdown_report=_markdown() + "\nprediction_id: prediction-train-a\ntraining_row_id: training-row-a\n",
    )

    assert "markdown_report_id_mismatch" in missing_id.error_codes
    assert "markdown_raw_payload_leaked" in leaked.error_codes


def test_manifest_and_artifact_loader_verify_sha(tmp_path: Path) -> None:
    manifest_path, manifest, candidate_report = _write_report_artifacts(tmp_path)

    loaded_manifest = load_oled_baseline_benchmark_report_writer_manifest_json(manifest_path)
    loaded_report, loaded_markdown = load_oled_baseline_benchmark_report_artifacts_from_manifest(
        manifest=loaded_manifest,
        base_dir=tmp_path,
    )

    assert loaded_report == candidate_report
    assert loaded_markdown is not None and "not a benchmark registration" in loaded_markdown

    bad_json_manifest = manifest.model_copy(
        update={
            "file_results": [
                manifest.file_results[0].model_copy(update={"output_sha256": "wrong-json-sha"}),
                manifest.file_results[1],
            ]
        }
    )
    with pytest.raises(ValueError, match="benchmark_report_json_sha256_mismatch:"):
        load_oled_baseline_benchmark_report_artifacts_from_manifest(manifest=bad_json_manifest, base_dir=tmp_path)

    bad_markdown_manifest = manifest.model_copy(
        update={
            "file_results": [
                manifest.file_results[0],
                manifest.file_results[1].model_copy(update={"output_sha256": "wrong-markdown-sha"}),
            ]
        }
    )
    with pytest.raises(ValueError, match="benchmark_report_markdown_sha256_mismatch:"):
        load_oled_baseline_benchmark_report_artifacts_from_manifest(manifest=bad_markdown_manifest, base_dir=tmp_path)


def test_json_and_markdown_loaders_reject_invalid_input(tmp_path: Path) -> None:
    valid_json = tmp_path / "report.json"
    write_oled_baseline_benchmark_report_json(_candidate_report(), valid_json)
    assert load_oled_baseline_benchmark_candidate_report_json(valid_json).report_id == "report:oled-baseline-benchmark:test"

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_benchmark_candidate_report_json:"):
        load_oled_baseline_benchmark_candidate_report_json(invalid_json)

    leaked_markdown = tmp_path / "leaked.md"
    leaked_markdown.write_text("features: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_candidate_report_markdown_leakage:"):
        load_oled_baseline_benchmark_candidate_report_markdown(leaked_markdown)


def test_combined_runner_from_files_writes_report_only(tmp_path: Path) -> None:
    manifest_path, _, _ = _write_report_artifacts(tmp_path)
    output_report = tmp_path / "registry_preflight.json"

    report = run_oled_benchmark_registry_preflight_from_files(
        benchmark_report_manifest_path=manifest_path,
        benchmark_report_base_dir=tmp_path,
        output_report_path=output_report,
    )

    assert report.is_valid
    assert output_report.exists()
    assert not (tmp_path / "benchmark_registry.json").exists()


def test_report_writer_is_deterministic_and_redacted(tmp_path: Path) -> None:
    report = run_oled_benchmark_registry_preflight(
        report_manifest=_manifest(),
        candidate_report=_candidate_report(),
        markdown_report=_markdown(),
    )
    output_path = tmp_path / "registry_preflight.json"
    write_oled_benchmark_registry_preflight_report_json(report, output_path)
    first_text = output_path.read_text(encoding="utf-8")
    write_oled_benchmark_registry_preflight_report_json(report, output_path)

    assert first_text == output_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in first_text
    assert "Raw Paper Text" not in first_text
    assert "features" not in first_text
    assert "prediction_id" not in first_text
    assert '"benchmark_registered": false' in first_text


def test_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, _, _ = _write_report_artifacts(tmp_path)
    output_report = tmp_path / "cli_registry_preflight.json"

    exit_code = main(
        [
            "--benchmark-report-manifest",
            str(manifest_path),
            "--benchmark-report-base-dir",
            str(tmp_path),
            "--output-report",
            str(output_report),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_report.exists()
    assert "prediction_id" not in captured.out
    assert "metric_cards" not in captured.out


def test_package_exports() -> None:
    assert OledBenchmarkRegistryPreflightStatus
    assert OledBenchmarkRegistryArtifactStatus
    assert OledBenchmarkRegistryPreflightPolicy
    assert OledBenchmarkReportArtifactSummary
    assert OledBenchmarkRegistryRunSummary
    assert OledBenchmarkRegistryPreflightFinding
    assert OledBenchmarkRegistryPreflightReport
    assert package_load_manifest
    assert package_load_json
    assert package_load_markdown
    assert package_load_artifacts
    assert package_run_preflight
    assert package_run_from_files
    assert package_write_report
