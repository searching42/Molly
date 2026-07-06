from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from ai4s_agent.domains.oled_curated_baseline_benchmark_report_writer import (
    OledBaselineBenchmarkCandidateReport,
    OledBaselineBenchmarkReportFileResult,
    OledBaselineBenchmarkReportWriteStatus,
    OledBaselineBenchmarkReportWriterManifest,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledBenchmarkRegistryPreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledBenchmarkRegistryArtifactStatus(str, Enum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    FAILED = "failed"
    SKIPPED = "skipped"


class OledBenchmarkRegistryPreflightPolicy(BaseModel):
    require_report_manifest_sha256: bool = True
    require_report_json_sha256: bool = True
    require_report_markdown_sha256: bool = True

    require_json_report: bool = True
    require_markdown_report: bool = True

    require_source_baseline_run_manifest_id: bool = True
    require_source_benchmark_preflight_status: bool = True
    require_valid_benchmark_preflight_status: bool = True

    require_run_cards: bool = True
    require_metric_cards: bool = True
    require_caveats: bool = True

    require_no_benchmark_validated_claims: bool = True
    require_not_registered: bool = True
    require_no_scientific_claim: bool = True

    fail_on_markdown_mismatch: bool = True
    fail_on_missing_metric_cards: bool = True

    required_caveats: list[str] = Field(
        default_factory=lambda: [
            "baseline_candidate_report_only",
            "not_benchmark_validated",
            "not_scientific_performance_claim",
        ]
    )

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)


class OledBenchmarkReportArtifactSummary(BaseModel):
    artifact_kind: str

    status: OledBenchmarkRegistryArtifactStatus
    output_path: str | None = None
    output_sha256: str | None = None

    loaded: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class OledBenchmarkRegistryRunSummary(BaseModel):
    baseline_kind: str
    target_property_id: str
    feature_view: str

    run_status: str
    prediction_count: int = 0

    metric_card_count: int = 0
    metric_splits: list[str] = Field(default_factory=list)

    train_row_count: int = 0
    validation_row_count: int = 0
    test_row_count: int = 0

    artifact_status: OledBenchmarkRegistryArtifactStatus
    reason_codes: list[str] = Field(default_factory=list)


class OledBenchmarkRegistryPreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    artifact_kind: str | None = None
    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    output_path: str | None = None


class OledBenchmarkRegistryPreflightReport(BaseModel):
    status: OledBenchmarkRegistryPreflightStatus

    source_report_manifest_id: str | None = None
    source_baseline_run_manifest_id: str | None = None
    source_benchmark_preflight_status: str | None = None

    report_id: str | None = None

    input_run_card_count: int = 0
    input_metric_card_count: int = 0

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    artifact_summaries: list[OledBenchmarkReportArtifactSummary] = Field(default_factory=list)
    run_summaries: list[OledBenchmarkRegistryRunSummary] = Field(default_factory=list)

    caveats: list[str] = Field(default_factory=list)

    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    findings: list[OledBenchmarkRegistryPreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledBenchmarkRegistryPreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_baseline_benchmark_report_writer_manifest_json(
    path: str | Path,
) -> OledBaselineBenchmarkReportWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_benchmark_report_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledBaselineBenchmarkReportWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_benchmark_report_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.metadata):
        raise ValueError("absolute_path_in_benchmark_report_manifest_metadata")
    return manifest


def load_oled_baseline_benchmark_candidate_report_json(
    path: str | Path,
) -> OledBaselineBenchmarkCandidateReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_benchmark_candidate_report_json:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if _contains_forbidden_payload_key(payload):
            raise ValueError("forbidden benchmark candidate report payload")
        report = OledBaselineBenchmarkCandidateReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid_benchmark_candidate_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    if _contains_absolute_path(report.metadata):
        raise ValueError("absolute_path_in_benchmark_candidate_report_metadata")
    return report


def load_oled_baseline_benchmark_candidate_report_markdown(
    path: str | Path,
) -> str:
    markdown_path = Path(path)
    _reject_forbidden_input(markdown_path)
    if not markdown_path.exists():
        raise ValueError(f"missing_benchmark_candidate_report_markdown:{redact_oled_mineru_acceptance_path(markdown_path)}")
    text = markdown_path.read_text(encoding="utf-8")
    if _markdown_contains_forbidden_payload(text):
        raise ValueError(f"benchmark_candidate_report_markdown_leakage:{redact_oled_mineru_acceptance_path(markdown_path)}")
    return text


def load_oled_baseline_benchmark_report_artifacts_from_manifest(
    *,
    manifest: OledBaselineBenchmarkReportWriterManifest,
    base_dir: str | Path,
) -> tuple[OledBaselineBenchmarkCandidateReport | None, str | None]:
    candidate_report: OledBaselineBenchmarkCandidateReport | None = None
    markdown_report: str | None = None
    for file_result in manifest.file_results:
        if _status_value(file_result.status) != OledBaselineBenchmarkReportWriteStatus.WRITTEN.value:
            continue
        if not file_result.output_path:
            continue
        path = _resolve_manifest_path(file_result.output_path, base_dir)
        if file_result.artifact_kind == "benchmark_report_json":
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"benchmark_report_json_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            candidate_report = load_oled_baseline_benchmark_candidate_report_json(path)
        elif file_result.artifact_kind == "benchmark_report_markdown":
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"benchmark_report_markdown_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            markdown_report = load_oled_baseline_benchmark_candidate_report_markdown(path)
    return candidate_report, markdown_report


def run_oled_benchmark_registry_preflight(
    *,
    report_manifest: OledBaselineBenchmarkReportWriterManifest,
    candidate_report: OledBaselineBenchmarkCandidateReport | None,
    markdown_report: str | None = None,
    policy: OledBenchmarkRegistryPreflightPolicy | None = None,
) -> OledBenchmarkRegistryPreflightReport:
    preflight_policy = policy or OledBenchmarkRegistryPreflightPolicy()
    findings: list[OledBenchmarkRegistryPreflightFinding] = []
    findings.extend(_manifest_findings(report_manifest, preflight_policy))
    artifact_summaries = _artifact_summaries(report_manifest, candidate_report, markdown_report, preflight_policy)
    findings.extend(_artifact_findings(artifact_summaries))
    if candidate_report is not None:
        findings.extend(_candidate_report_findings(report_manifest, candidate_report, preflight_policy))
    elif preflight_policy.require_json_report:
        findings.append(_finding("missing_benchmark_candidate_report_json", "error", "benchmark candidate JSON report is required", artifact_kind="benchmark_report_json"))
    if markdown_report is not None:
        findings.extend(_markdown_findings(candidate_report, markdown_report, preflight_policy))
    elif preflight_policy.require_markdown_report:
        findings.append(
            _finding(
                "missing_benchmark_candidate_report_markdown",
                "error",
                "benchmark candidate Markdown report is required",
                artifact_kind="benchmark_report_markdown",
            )
        )

    run_summaries = _run_summaries(candidate_report, preflight_policy)
    findings.extend(_run_summary_findings(run_summaries, preflight_policy))
    findings = _dedup_findings(findings)
    status = _report_status(findings)
    status_counts = Counter(_status_value(summary.status) for summary in artifact_summaries)
    status_counts.update(_status_value(summary.artifact_status) for summary in run_summaries)
    return OledBenchmarkRegistryPreflightReport(
        status=status,
        source_report_manifest_id=report_manifest.manifest_id,
        source_baseline_run_manifest_id=report_manifest.source_baseline_run_manifest_id,
        source_benchmark_preflight_status=report_manifest.source_benchmark_preflight_status,
        report_id=candidate_report.report_id if candidate_report is not None else None,
        input_run_card_count=len(candidate_report.run_cards) if candidate_report is not None else 0,
        input_metric_card_count=sum(len(card.metrics) for card in candidate_report.run_cards) if candidate_report is not None else 0,
        baseline_kinds=sorted({summary.baseline_kind for summary in run_summaries}),
        target_property_ids=sorted({summary.target_property_id for summary in run_summaries}),
        feature_views=sorted({summary.feature_view for summary in run_summaries}),
        artifact_summaries=artifact_summaries,
        run_summaries=run_summaries,
        caveats=sorted(candidate_report.caveats) if candidate_report is not None else [],
        status_counts=dict(sorted(status_counts.items())),
        finding_code_counts=dict(sorted(Counter(finding.code for finding in findings).items())),
        findings=findings,
        metadata=_safety_metadata(),
    )


def run_oled_benchmark_registry_preflight_from_files(
    *,
    benchmark_report_manifest_path: str | Path,
    benchmark_report_base_dir: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledBenchmarkRegistryPreflightPolicy | None = None,
) -> OledBenchmarkRegistryPreflightReport:
    manifest = load_oled_baseline_benchmark_report_writer_manifest_json(benchmark_report_manifest_path)
    base_dir = Path(benchmark_report_base_dir) if benchmark_report_base_dir is not None else Path(benchmark_report_manifest_path).parent
    candidate_report, markdown_report = load_oled_baseline_benchmark_report_artifacts_from_manifest(manifest=manifest, base_dir=base_dir)
    report = run_oled_benchmark_registry_preflight(
        report_manifest=manifest,
        candidate_report=candidate_report,
        markdown_report=markdown_report,
        policy=policy,
    )
    if output_report_path is not None:
        write_oled_benchmark_registry_preflight_report_json(report, output_report_path)
    return report


def write_oled_benchmark_registry_preflight_report_json(
    report: OledBenchmarkRegistryPreflightReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    output_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED benchmark registry-readiness preflight.")
    parser.add_argument("--benchmark-report-manifest", required=True, help="Path to benchmark report writer manifest JSON.")
    parser.add_argument("--benchmark-report-base-dir", help="Base directory for benchmark report artifacts.")
    parser.add_argument("--output-report", help="Optional path for registry-readiness preflight report JSON.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--allow-missing-markdown", action="store_true", help="Do not require a Markdown candidate report.")
    args = parser.parse_args(argv)
    try:
        policy = OledBenchmarkRegistryPreflightPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            require_markdown_report=not args.allow_missing_markdown,
        )
        report = run_oled_benchmark_registry_preflight_from_files(
            benchmark_report_manifest_path=args.benchmark_report_manifest,
            benchmark_report_base_dir=args.benchmark_report_base_dir,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": _status_value(report.status),
            "run_summary_count": len(report.run_summaries),
            "artifact_summary_count": len(report.artifact_summaries),
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _manifest_findings(
    manifest: OledBaselineBenchmarkReportWriterManifest,
    policy: OledBenchmarkRegistryPreflightPolicy,
) -> list[OledBenchmarkRegistryPreflightFinding]:
    findings: list[OledBenchmarkRegistryPreflightFinding] = []
    if policy.require_source_baseline_run_manifest_id and not manifest.source_baseline_run_manifest_id:
        findings.append(_finding("missing_source_baseline_run_manifest_id", "error", "source baseline run manifest id is required"))
    if policy.require_source_benchmark_preflight_status and not manifest.source_benchmark_preflight_status:
        findings.append(_finding("missing_source_benchmark_preflight_status", "error", "source benchmark preflight status is required"))
    if policy.require_valid_benchmark_preflight_status and manifest.source_benchmark_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_benchmark_preflight_status", "error", "source benchmark preflight status is not registry-ready"))
    if policy.require_no_benchmark_validated_claims and bool(manifest.metadata.get("benchmark_validated")):
        findings.append(_finding("benchmark_validated_source_claim", "error", "manifest metadata claims benchmark validation"))
    if policy.require_not_registered and bool(manifest.metadata.get("benchmark_registered")):
        findings.append(_finding("benchmark_registered_source_claim", "error", "manifest metadata claims benchmark registration"))
    if policy.require_no_scientific_claim and bool(manifest.metadata.get("scientific_claim_validated")):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "manifest metadata claims scientific validation"))
    if bool(manifest.policy.benchmark_validated):
        findings.append(_finding("benchmark_validated_source_claim", "error", "manifest policy claims benchmark validation"))
    if bool(manifest.policy.register_benchmark):
        findings.append(_finding("benchmark_registered_source_claim", "error", "manifest policy claims benchmark registration"))
    return findings


def _artifact_summaries(
    manifest: OledBaselineBenchmarkReportWriterManifest,
    candidate_report: OledBaselineBenchmarkCandidateReport | None,
    markdown_report: str | None,
    policy: OledBenchmarkRegistryPreflightPolicy,
) -> list[OledBenchmarkReportArtifactSummary]:
    summaries: list[OledBenchmarkReportArtifactSummary] = []
    for artifact_kind, required, loaded in (
        ("benchmark_report_json", policy.require_json_report, candidate_report is not None),
        ("benchmark_report_markdown", policy.require_markdown_report, markdown_report is not None),
    ):
        file_result = _file_result_for_kind(manifest.file_results, artifact_kind)
        reasons: set[str] = set()
        status = OledBenchmarkRegistryArtifactStatus.READY
        if loaded:
            reasons.add("artifact_loaded")
        elif required:
            reasons.add(f"missing_{artifact_kind}")
            status = OledBenchmarkRegistryArtifactStatus.FAILED
        else:
            reasons.add("artifact_optional")
            status = OledBenchmarkRegistryArtifactStatus.SKIPPED
        if file_result and not file_result.output_sha256:
            required_sha = policy.require_report_json_sha256 if artifact_kind == "benchmark_report_json" else policy.require_report_markdown_sha256
            if required_sha:
                reasons.add(f"missing_{artifact_kind}_sha256")
                status = OledBenchmarkRegistryArtifactStatus.FAILED
        summaries.append(
            OledBenchmarkReportArtifactSummary(
                artifact_kind=artifact_kind,
                status=status,
                output_path=file_result.output_path if file_result is not None else None,
                output_sha256=file_result.output_sha256 if file_result is not None else None,
                loaded=loaded,
                reason_codes=sorted(reasons),
            )
        )
    return summaries


def _artifact_findings(
    summaries: list[OledBenchmarkReportArtifactSummary],
) -> list[OledBenchmarkRegistryPreflightFinding]:
    findings: list[OledBenchmarkRegistryPreflightFinding] = []
    for summary in summaries:
        if summary.status != OledBenchmarkRegistryArtifactStatus.FAILED:
            continue
        for reason in summary.reason_codes:
            code = _artifact_reason_to_code(reason)
            findings.append(
                _finding(
                    code,
                    "error",
                    "benchmark report artifact is not registry-ready",
                    artifact_kind=summary.artifact_kind,
                    output_path=summary.output_path,
                )
            )
    return findings


def _candidate_report_findings(
    manifest: OledBaselineBenchmarkReportWriterManifest,
    report: OledBaselineBenchmarkCandidateReport,
    policy: OledBenchmarkRegistryPreflightPolicy,
) -> list[OledBenchmarkRegistryPreflightFinding]:
    findings: list[OledBenchmarkRegistryPreflightFinding] = []
    if policy.require_source_baseline_run_manifest_id and not report.source_baseline_run_manifest_id:
        findings.append(_finding("missing_source_baseline_run_manifest_id", "error", "candidate report lacks source baseline run manifest id"))
    if policy.require_source_benchmark_preflight_status and not report.source_benchmark_preflight_status:
        findings.append(_finding("missing_source_benchmark_preflight_status", "error", "candidate report lacks source benchmark preflight status"))
    if policy.require_valid_benchmark_preflight_status and report.source_benchmark_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_benchmark_preflight_status", "error", "candidate report preflight status is not registry-ready"))
    if manifest.source_baseline_run_manifest_id and report.source_baseline_run_manifest_id and manifest.source_baseline_run_manifest_id != report.source_baseline_run_manifest_id:
        findings.append(_finding("source_id_mismatch", "error", "manifest and report source baseline ids differ"))
    if manifest.source_benchmark_preflight_status and report.source_benchmark_preflight_status and manifest.source_benchmark_preflight_status != report.source_benchmark_preflight_status:
        findings.append(_finding("source_id_mismatch", "error", "manifest and report source preflight statuses differ"))
    if policy.require_caveats:
        caveats = set(report.caveats)
        for caveat in policy.required_caveats:
            if caveat not in caveats:
                findings.append(_finding("missing_required_caveat", "error", "candidate report lacks required caveat"))
    if policy.require_no_benchmark_validated_claims and bool(report.metadata.get("benchmark_validated")):
        findings.append(_finding("benchmark_validated_source_claim", "error", "candidate report metadata claims benchmark validation"))
    if policy.require_not_registered and bool(report.metadata.get("benchmark_registered")):
        findings.append(_finding("benchmark_registered_source_claim", "error", "candidate report metadata claims benchmark registration"))
    if policy.require_no_scientific_claim and bool(report.metadata.get("scientific_claim_validated")):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "candidate report metadata claims scientific validation"))
    if _contains_forbidden_payload_key(report.model_dump(mode="json")):
        findings.append(_finding("raw_prediction_payload_leaked", "error", "candidate report contains raw prediction payload fields"))
    return findings


def _run_summaries(
    report: OledBaselineBenchmarkCandidateReport | None,
    policy: OledBenchmarkRegistryPreflightPolicy,
) -> list[OledBenchmarkRegistryRunSummary]:
    if report is None:
        return []
    baselines = _baseline_kinds(policy)
    targets = _target_property_ids(policy)
    views = _feature_views(policy)
    summaries: list[OledBenchmarkRegistryRunSummary] = []
    for card in sorted(report.run_cards, key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view)):
        if baselines and card.baseline_kind not in baselines:
            continue
        if targets and card.target_property_id not in targets:
            continue
        if views and card.feature_view not in views:
            continue
        reasons: set[str] = {"run_card_ready"}
        status = OledBenchmarkRegistryArtifactStatus.READY
        if not card.metrics:
            reasons.add("missing_metric_cards")
            status = OledBenchmarkRegistryArtifactStatus.FAILED
        summaries.append(
            OledBenchmarkRegistryRunSummary(
                baseline_kind=card.baseline_kind,
                target_property_id=card.target_property_id,
                feature_view=card.feature_view,
                run_status=card.run_status,
                prediction_count=card.prediction_count,
                metric_card_count=len(card.metrics),
                metric_splits=sorted({metric.split for metric in card.metrics}),
                train_row_count=card.train_row_count,
                validation_row_count=card.validation_row_count,
                test_row_count=card.test_row_count,
                artifact_status=status,
                reason_codes=sorted(reasons),
            )
        )
    return summaries


def _run_summary_findings(
    summaries: list[OledBenchmarkRegistryRunSummary],
    policy: OledBenchmarkRegistryPreflightPolicy,
) -> list[OledBenchmarkRegistryPreflightFinding]:
    findings: list[OledBenchmarkRegistryPreflightFinding] = []
    if policy.require_run_cards and not summaries:
        findings.append(_finding("missing_run_cards", "error", "candidate report has no selected run cards"))
    for summary in summaries:
        if policy.require_metric_cards and policy.fail_on_missing_metric_cards and summary.metric_card_count == 0:
            findings.append(
                _finding(
                    "missing_metric_cards",
                    "error",
                    "run card has no metric cards",
                    baseline_kind=summary.baseline_kind,
                    target_property_id=summary.target_property_id,
                    feature_view=summary.feature_view,
                )
            )
    return findings


def _markdown_findings(
    report: OledBaselineBenchmarkCandidateReport | None,
    markdown: str,
    policy: OledBenchmarkRegistryPreflightPolicy,
) -> list[OledBenchmarkRegistryPreflightFinding]:
    findings: list[OledBenchmarkRegistryPreflightFinding] = []
    lower = markdown.lower()
    severity: Literal["error", "warning"] = "error" if policy.fail_on_markdown_mismatch else "warning"
    if report is not None and report.report_id not in markdown:
        findings.append(_finding("markdown_report_id_mismatch", severity, "Markdown report does not contain candidate report id", artifact_kind="benchmark_report_markdown"))
    if policy.require_caveats:
        for caveat in policy.required_caveats:
            if caveat not in markdown:
                findings.append(_finding("missing_required_caveat", severity, "Markdown report lacks required caveat", artifact_kind="benchmark_report_markdown"))
    if "not a benchmark registration" not in lower and "not benchmark registration" not in lower:
        findings.append(_finding("markdown_safety_statement_missing", severity, "Markdown report lacks registry safety statement", artifact_kind="benchmark_report_markdown"))
    if _markdown_contains_forbidden_payload(markdown):
        findings.append(_finding("markdown_raw_payload_leaked", "error", "Markdown report contains raw payload markers", artifact_kind="benchmark_report_markdown"))
    return findings


def _artifact_reason_to_code(reason: str) -> str:
    if reason == "missing_benchmark_report_json":
        return "missing_benchmark_candidate_report_json"
    if reason == "missing_benchmark_report_markdown":
        return "missing_benchmark_candidate_report_markdown"
    return reason


def _file_result_for_kind(
    file_results: Iterable[OledBaselineBenchmarkReportFileResult],
    artifact_kind: str,
) -> OledBaselineBenchmarkReportFileResult | None:
    for file_result in file_results:
        if file_result.artifact_kind == artifact_kind:
            return file_result
    return None


def _resolve_manifest_path(output_path: str, base_dir: str | Path) -> Path:
    candidate = Path(output_path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _baseline_kinds(policy: OledBenchmarkRegistryPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.baseline_kinds if str(item).strip()}


def _target_property_ids(policy: OledBenchmarkRegistryPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.target_property_ids if str(item).strip()}


def _feature_views(policy: OledBenchmarkRegistryPreflightPolicy) -> set[str]:
    return {str(item).strip() for item in policy.feature_views if str(item).strip()}


def _report_status(
    findings: list[OledBenchmarkRegistryPreflightFinding],
) -> OledBenchmarkRegistryPreflightStatus:
    if any(finding.severity == "error" for finding in findings):
        return OledBenchmarkRegistryPreflightStatus.FAILED
    if any(finding.severity == "warning" for finding in findings):
        return OledBenchmarkRegistryPreflightStatus.PASSED_WITH_WARNINGS
    return OledBenchmarkRegistryPreflightStatus.PASSED


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    artifact_kind: str | None = None,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    output_path: str | None = None,
) -> OledBenchmarkRegistryPreflightFinding:
    return OledBenchmarkRegistryPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        artifact_kind=artifact_kind,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        output_path=output_path,
    )


def _dedup_findings(
    findings: list[OledBenchmarkRegistryPreflightFinding],
) -> list[OledBenchmarkRegistryPreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    deduped: list[OledBenchmarkRegistryPreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.artifact_kind or "",
            finding.baseline_kind or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.output_path or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return sorted(
        deduped,
        key=lambda item: (
            item.severity,
            item.code,
            item.artifact_kind or "",
            item.baseline_kind or "",
            item.target_property_id or "",
            item.feature_view or "",
        ),
    )


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute()
    return False


def _contains_forbidden_payload_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_JSON_KEYS:
                return True
            if _contains_forbidden_payload_key(item):
                return True
    if isinstance(value, list):
        return any(_contains_forbidden_payload_key(item) for item in value)
    return False


def _markdown_contains_forbidden_payload(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _FORBIDDEN_MARKDOWN_MARKERS)


def _reject_forbidden_input(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"forbidden_pdf_input:{redact_oled_mineru_acceptance_path(path)}")
    if suffix in _FORBIDDEN_IMAGE_SUFFIXES:
        raise ValueError(f"forbidden_image_input:{redact_oled_mineru_acceptance_path(path)}")


def _sanitize_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_JSON_KEYS:
                continue
            sanitized_value = _sanitize_for_output(raw_value)
            if sanitized_value in (None, {}, []):
                continue
            output[key] = sanitized_value
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            sanitized_item = _sanitize_for_output(item)
            if sanitized_item not in (None, {}, []):
                output.append(sanitized_item)
        return output
    if isinstance(value, str):
        if Path(value).is_absolute():
            return redact_oled_mineru_acceptance_path(value)
        if len(value) > _MAX_OUTPUT_STRING_LENGTH:
            return value[: _MAX_OUTPUT_STRING_LENGTH - 3] + "..."
        return value
    return value


def _safety_metadata() -> dict[str, Any]:
    return {
        "benchmark_registry_preflight_only": True,
        "benchmark_registry_written": False,
        "benchmark_registered": False,
        "benchmark_validated": False,
        "scientific_claim_validated": False,
        "baseline_backend_run": False,
        "models_fitted": False,
        "predictions_written": False,
        "metrics_written": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


_MAX_OUTPUT_STRING_LENGTH = 240

_FORBIDDEN_JSON_KEYS = {
    "raw_text",
    "full_text",
    "features",
    "prediction_id",
    "training_row_id",
}

_FORBIDDEN_MARKDOWN_MARKERS = {
    "raw_text",
    "full_text",
    "features",
    "prediction_id",
    "training_row_id",
}

_FORBIDDEN_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".svg",
}


__all__ = [
    "OledBenchmarkRegistryPreflightStatus",
    "OledBenchmarkRegistryArtifactStatus",
    "OledBenchmarkRegistryPreflightPolicy",
    "OledBenchmarkReportArtifactSummary",
    "OledBenchmarkRegistryRunSummary",
    "OledBenchmarkRegistryPreflightFinding",
    "OledBenchmarkRegistryPreflightReport",
    "load_oled_baseline_benchmark_report_writer_manifest_json",
    "load_oled_baseline_benchmark_candidate_report_json",
    "load_oled_baseline_benchmark_candidate_report_markdown",
    "load_oled_baseline_benchmark_report_artifacts_from_manifest",
    "run_oled_benchmark_registry_preflight",
    "run_oled_benchmark_registry_preflight_from_files",
    "write_oled_benchmark_registry_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
