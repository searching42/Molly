from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from ai4s_agent.domains.oled_curated_baseline_benchmark_report_writer import (
    OledBaselineBenchmarkCandidateReport,
    OledBaselineBenchmarkReportWriterManifest,
)
from ai4s_agent.domains.oled_curated_benchmark_registry_preflight import (
    OledBenchmarkRegistryPreflightReport,
    OledBenchmarkRegistryPreflightStatus,
    load_oled_baseline_benchmark_report_artifacts_from_manifest,
    load_oled_baseline_benchmark_report_writer_manifest_json,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledBenchmarkRegistryWriterPolicy(BaseModel):
    require_confirmation: bool = True
    require_registry_preflight_valid: bool = True
    allow_registry_preflight_warnings: bool = True

    require_candidate_report: bool = True
    require_report_manifest: bool = True
    require_report_json_sha256: bool = True
    require_report_markdown_sha256: bool = True

    require_source_baseline_run_manifest_id: bool = True
    require_source_benchmark_preflight_status: bool = True
    require_caveats: bool = True
    require_run_cards: bool = True
    require_metric_cards: bool = True

    require_no_benchmark_validated_claims: bool = True
    require_not_scientific_claim: bool = True

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)

    write_registry_entry_json: bool = True
    write_registry_index_jsonl: bool = True

    registry_status: Literal["candidate"] = "candidate"
    benchmark_validated: bool = False
    scientific_claim_validated: bool = False


class OledBenchmarkRegistryWriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class OledBenchmarkRegistryEntryStatus(str, Enum):
    CANDIDATE = "candidate"
    REJECTED = "rejected"


class OledBenchmarkRegistryEntry(BaseModel):
    registry_entry_id: str

    registry_status: OledBenchmarkRegistryEntryStatus = OledBenchmarkRegistryEntryStatus.CANDIDATE

    source_benchmark_report_manifest_id: str | None = None
    source_benchmark_registry_preflight_status: str | None = None

    source_candidate_report_id: str | None = None
    source_baseline_run_manifest_id: str | None = None
    source_benchmark_preflight_status: str | None = None

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    run_card_count: int = 0
    metric_card_count: int = 0

    report_json_path: str | None = None
    report_json_sha256: str | None = None
    report_markdown_path: str | None = None
    report_markdown_sha256: str | None = None

    caveats: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class OledBenchmarkRegistryIndexRecord(BaseModel):
    registry_entry_id: str
    registry_status: str

    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None
    source_benchmark_registry_preflight_status: str | None = None

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    run_card_count: int = 0
    metric_card_count: int = 0

    output_registry_entry_json_path: str | None = None
    output_registry_entry_json_sha256: str | None = None

    benchmark_validated: bool = False
    scientific_claim_validated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledBenchmarkRegistryFileResult(BaseModel):
    artifact_kind: Literal["registry_entry_json", "registry_index_jsonl", "manifest"]

    status: OledBenchmarkRegistryWriteStatus
    output_path: str | None = None
    output_sha256: str | None = None

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledBenchmarkRegistryWriterFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    output_path: str | None = None


class OledBenchmarkRegistryWriterManifest(BaseModel):
    manifest_id: str

    source_benchmark_report_manifest_id: str | None = None
    source_benchmark_registry_preflight_status: str | None = None
    source_candidate_report_id: str | None = None

    output_directory: str | None = None
    output_file_count: int = 0

    registry_entry_ids: list[str] = Field(default_factory=list)

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    file_results: list[OledBenchmarkRegistryFileResult] = Field(default_factory=list)

    policy: OledBenchmarkRegistryWriterPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(result.status == OledBenchmarkRegistryWriteStatus.REJECTED for result in self.file_results)


class OledBenchmarkRegistryWriterReport(BaseModel):
    manifest: OledBenchmarkRegistryWriterManifest
    registry_entry: OledBenchmarkRegistryEntry | None = None
    registry_index_records: list[OledBenchmarkRegistryIndexRecord] = Field(default_factory=list)
    findings: list[OledBenchmarkRegistryWriterFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.error_codes and self.manifest.is_valid

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_benchmark_registry_preflight_report_json(
    path: str | Path,
) -> OledBenchmarkRegistryPreflightReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_benchmark_registry_preflight_report:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = OledBenchmarkRegistryPreflightReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_benchmark_registry_preflight_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    if _contains_absolute_path(report.metadata):
        raise ValueError("absolute_path_in_benchmark_registry_preflight_report_metadata")
    return report


def build_oled_benchmark_registry_entry(
    *,
    report_manifest: OledBaselineBenchmarkReportWriterManifest,
    candidate_report: OledBaselineBenchmarkCandidateReport,
    registry_preflight_report: OledBenchmarkRegistryPreflightReport,
    policy: OledBenchmarkRegistryWriterPolicy | None = None,
) -> tuple[OledBenchmarkRegistryEntry | None, list[OledBenchmarkRegistryWriterFinding]]:
    writer_policy = policy or OledBenchmarkRegistryWriterPolicy()
    findings = _gate_findings(
        report_manifest=report_manifest,
        candidate_report=candidate_report,
        registry_preflight_report=registry_preflight_report,
        policy=writer_policy,
    )
    selected_run_cards = _filter_run_cards(candidate_report, writer_policy)
    metric_card_count = sum(len(card.metrics) for card in selected_run_cards)
    if writer_policy.require_run_cards and not selected_run_cards:
        findings.append(_finding("missing_run_cards", "error", "candidate report has no selected run cards"))
    if writer_policy.require_metric_cards and selected_run_cards and metric_card_count == 0:
        findings.append(_finding("missing_metric_cards", "error", "selected run cards have no metric cards"))
    for card in selected_run_cards:
        if writer_policy.require_metric_cards and not card.metrics:
            findings.append(
                _finding(
                    "missing_metric_cards",
                    "error",
                    "selected run card has no metric cards",
                    baseline_kind=card.baseline_kind,
                    target_property_id=card.target_property_id,
                    feature_view=card.feature_view,
                )
            )
    findings = _dedup_findings(findings)
    if any(finding.severity == "error" for finding in findings):
        return None, findings

    report_json_result = _file_result_for_kind(report_manifest.file_results, "benchmark_report_json")
    markdown_result = _file_result_for_kind(report_manifest.file_results, "benchmark_report_markdown")
    entry = OledBenchmarkRegistryEntry(
        registry_entry_id=_registry_entry_id(report_manifest.manifest_id, candidate_report.report_id, registry_preflight_report.status),
        registry_status=OledBenchmarkRegistryEntryStatus.CANDIDATE,
        source_benchmark_report_manifest_id=report_manifest.manifest_id,
        source_benchmark_registry_preflight_status=_status_value(registry_preflight_report.status),
        source_candidate_report_id=candidate_report.report_id,
        source_baseline_run_manifest_id=candidate_report.source_baseline_run_manifest_id,
        source_benchmark_preflight_status=candidate_report.source_benchmark_preflight_status,
        baseline_kinds=sorted({card.baseline_kind for card in selected_run_cards}),
        target_property_ids=sorted({card.target_property_id for card in selected_run_cards}),
        feature_views=sorted({card.feature_view for card in selected_run_cards}),
        run_card_count=len(selected_run_cards),
        metric_card_count=metric_card_count,
        report_json_path=report_json_result.output_path if report_json_result is not None else None,
        report_json_sha256=report_json_result.output_sha256 if report_json_result is not None else None,
        report_markdown_path=markdown_result.output_path if markdown_result is not None else None,
        report_markdown_sha256=markdown_result.output_sha256 if markdown_result is not None else None,
        caveats=sorted(candidate_report.caveats),
        reason_codes=["selected_for_registry"],
        metadata=_safety_metadata(entry_written=False, index_written=False),
    )
    return entry, findings


def build_oled_benchmark_registry_index_records(
    entry: OledBenchmarkRegistryEntry,
) -> list[OledBenchmarkRegistryIndexRecord]:
    return [
        OledBenchmarkRegistryIndexRecord(
            registry_entry_id=entry.registry_entry_id,
            registry_status=_status_value(entry.registry_status),
            source_candidate_report_id=entry.source_candidate_report_id,
            source_benchmark_report_manifest_id=entry.source_benchmark_report_manifest_id,
            source_benchmark_registry_preflight_status=entry.source_benchmark_registry_preflight_status,
            baseline_kinds=list(entry.baseline_kinds),
            target_property_ids=list(entry.target_property_ids),
            feature_views=list(entry.feature_views),
            run_card_count=entry.run_card_count,
            metric_card_count=entry.metric_card_count,
            benchmark_validated=False,
            scientific_claim_validated=False,
            metadata={
                "benchmark_registry_index_record": True,
                "registry_status": "candidate",
                "benchmark_validated": False,
                "scientific_claim_validated": False,
            },
        )
    ]


def select_oled_benchmark_registry_entry_for_write(
    *,
    report_manifest: OledBaselineBenchmarkReportWriterManifest,
    candidate_report: OledBaselineBenchmarkCandidateReport,
    registry_preflight_report: OledBenchmarkRegistryPreflightReport,
    policy: OledBenchmarkRegistryWriterPolicy | None = None,
    confirm_benchmark_registry_write: bool = False,
) -> OledBenchmarkRegistryWriterReport:
    writer_policy = policy or OledBenchmarkRegistryWriterPolicy()
    if writer_policy.require_confirmation and not confirm_benchmark_registry_write:
        raise ValueError("confirmation_required:benchmark_registry_write")
    entry, findings = build_oled_benchmark_registry_entry(
        report_manifest=report_manifest,
        candidate_report=candidate_report,
        registry_preflight_report=registry_preflight_report,
        policy=writer_policy,
    )
    index_records = build_oled_benchmark_registry_index_records(entry) if entry is not None else []
    manifest = _manifest(
        policy=writer_policy,
        entry=entry,
        findings=findings,
        report_manifest=report_manifest,
        registry_preflight_report=registry_preflight_report,
        output_directory=None,
        entry_written=False,
        index_written=False,
    )
    return OledBenchmarkRegistryWriterReport(
        manifest=manifest,
        registry_entry=entry,
        registry_index_records=index_records,
        findings=findings,
    )


def write_oled_benchmark_registry_entry_json(
    entry: OledBenchmarkRegistryEntry,
    path: str | Path,
) -> str:
    payload = json.dumps(_sanitize_for_output(entry.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n"
    return _write_bytes(path, payload.encode("utf-8"))


def write_oled_benchmark_registry_index_jsonl(
    records: Iterable[OledBenchmarkRegistryIndexRecord],
    path: str | Path,
) -> str:
    ordered = sorted(records, key=lambda item: item.registry_entry_id)
    lines = [
        json.dumps(_sanitize_for_output(record.model_dump(mode="json", exclude_none=True)), sort_keys=True, separators=(",", ":"))
        for record in ordered
    ]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    return _write_bytes(path, payload)


def write_oled_benchmark_registry_manifest_json(
    manifest: OledBenchmarkRegistryWriterManifest,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_oled_benchmark_registry_entry_json(
    path: str | Path,
) -> OledBenchmarkRegistryEntry:
    entry_path = Path(path)
    _reject_forbidden_input(entry_path)
    if not entry_path.exists():
        raise ValueError(f"missing_benchmark_registry_entry_json:{redact_oled_mineru_acceptance_path(entry_path)}")
    try:
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
        if _contains_forbidden_payload_key(payload):
            raise ValueError("forbidden benchmark registry entry payload")
        entry = OledBenchmarkRegistryEntry.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid_benchmark_registry_entry_json:{redact_oled_mineru_acceptance_path(entry_path)}") from exc
    if _contains_absolute_path(entry.model_dump(mode="json")):
        raise ValueError("absolute_path_in_benchmark_registry_entry")
    return entry


def load_oled_benchmark_registry_index_jsonl(
    path: str | Path,
) -> list[OledBenchmarkRegistryIndexRecord]:
    index_path = Path(path)
    _reject_forbidden_input(index_path)
    if not index_path.exists():
        raise ValueError(f"missing_benchmark_registry_index_jsonl:{redact_oled_mineru_acceptance_path(index_path)}")
    records: list[OledBenchmarkRegistryIndexRecord] = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if _contains_forbidden_payload_key(payload):
                raise ValueError("forbidden benchmark registry index payload")
            record = OledBenchmarkRegistryIndexRecord.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"invalid_benchmark_registry_index_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(record.model_dump(mode="json")):
            raise ValueError("absolute_path_in_benchmark_registry_index")
        records.append(record)
    return records


def oled_benchmark_registry_entry_filename() -> str:
    return "oled_benchmark_registry_entry.json"


def oled_benchmark_registry_index_filename() -> str:
    return "oled_benchmark_registry_index.jsonl"


def run_oled_benchmark_registry_writer_from_files(
    *,
    benchmark_report_manifest_path: str | Path,
    benchmark_registry_preflight_report_path: str | Path,
    benchmark_report_base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    policy: OledBenchmarkRegistryWriterPolicy | None = None,
    confirm_benchmark_registry_write: bool = False,
    dry_run: bool = False,
) -> OledBenchmarkRegistryWriterReport:
    writer_policy = policy or OledBenchmarkRegistryWriterPolicy()
    if not output_dir and not output_manifest_path:
        raise ValueError("output_required:dir_or_manifest")
    if not dry_run and writer_policy.require_confirmation and not confirm_benchmark_registry_write:
        raise ValueError("confirmation_required:benchmark_registry_write")

    report_manifest = load_oled_baseline_benchmark_report_writer_manifest_json(benchmark_report_manifest_path)
    base_dir = Path(benchmark_report_base_dir) if benchmark_report_base_dir is not None else Path(benchmark_report_manifest_path).parent
    candidate_report, _markdown_report = load_oled_baseline_benchmark_report_artifacts_from_manifest(manifest=report_manifest, base_dir=base_dir)
    if candidate_report is None:
        raise ValueError("missing_benchmark_candidate_report_json:from_manifest")
    registry_preflight_report = load_oled_benchmark_registry_preflight_report_json(benchmark_registry_preflight_report_path)

    writer_report = select_oled_benchmark_registry_entry_for_write(
        report_manifest=report_manifest,
        candidate_report=candidate_report,
        registry_preflight_report=registry_preflight_report,
        policy=writer_policy,
        confirm_benchmark_registry_write=confirm_benchmark_registry_write or dry_run,
    )
    if dry_run:
        writer_report = _mark_dry_run(writer_report)
    elif writer_report.registry_entry is not None and writer_report.is_valid:
        if output_dir is None:
            raise ValueError("output_dir_required:benchmark_registry_write")
        writer_report = _write_registry_files(writer_report, Path(output_dir))

    if output_manifest_path is not None:
        write_oled_benchmark_registry_manifest_json(writer_report.manifest, output_manifest_path)
    return writer_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write OLED benchmark candidate registry artifacts under an explicit gate.")
    parser.add_argument("--benchmark-report-manifest", required=True, help="Path to benchmark report writer manifest JSON.")
    parser.add_argument("--benchmark-registry-preflight-report", required=True, help="Path to registry-readiness preflight report JSON.")
    parser.add_argument("--benchmark-report-base-dir", help="Base directory for benchmark report artifacts.")
    parser.add_argument("--output-dir", help="Output directory for registry artifacts.")
    parser.add_argument("--output-manifest", help="Optional output manifest JSON path.")
    parser.add_argument("--confirm-benchmark-registry-write", action="store_true", help="Confirm benchmark registry artifact writing.")
    parser.add_argument("--dry-run", action="store_true", help="Build registry entry in memory and write only manifest if requested.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--entry-only", action="store_true", help="Write only registry entry JSON.")
    parser.add_argument("--index-only", action="store_true", help="Write only registry index JSONL.")
    args = parser.parse_args(argv)
    try:
        if not args.output_dir and not args.output_manifest:
            raise ValueError("output_required:dir_or_manifest")
        if args.entry_only and args.index_only:
            raise ValueError("conflicting_output_modes:entry_only,index_only")
        if not args.dry_run and not args.confirm_benchmark_registry_write:
            raise ValueError("confirmation_required:benchmark_registry_write")
        policy = OledBenchmarkRegistryWriterPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            write_registry_entry_json=not args.index_only,
            write_registry_index_jsonl=not args.entry_only,
        )
        report = run_oled_benchmark_registry_writer_from_files(
            benchmark_report_manifest_path=args.benchmark_report_manifest,
            benchmark_registry_preflight_report_path=args.benchmark_registry_preflight_report,
            benchmark_report_base_dir=args.benchmark_report_base_dir,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            policy=policy,
            confirm_benchmark_registry_write=args.confirm_benchmark_registry_write,
            dry_run=args.dry_run,
        )
        summary = {
            "status": "valid" if report.is_valid else "invalid",
            "registry_entry_selected": report.registry_entry is not None,
            "index_record_count": len(report.registry_index_records),
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _gate_findings(
    *,
    report_manifest: OledBaselineBenchmarkReportWriterManifest,
    candidate_report: OledBaselineBenchmarkCandidateReport,
    registry_preflight_report: OledBenchmarkRegistryPreflightReport,
    policy: OledBenchmarkRegistryWriterPolicy,
) -> list[OledBenchmarkRegistryWriterFinding]:
    findings: list[OledBenchmarkRegistryWriterFinding] = []
    if policy.require_registry_preflight_valid and registry_preflight_report.status == OledBenchmarkRegistryPreflightStatus.FAILED:
        findings.append(_finding("registry_preflight_failed", "error", "registry preflight failed"))
    if not policy.allow_registry_preflight_warnings and _warning_codes(registry_preflight_report):
        findings.append(_finding("registry_preflight_warnings_present", "error", "registry preflight has warnings"))
    if policy.require_report_manifest and not report_manifest.manifest_id:
        findings.append(_finding("missing_report_manifest", "error", "benchmark report writer manifest is required"))
    if policy.require_candidate_report and not candidate_report.report_id:
        findings.append(_finding("missing_candidate_report", "error", "candidate report is required"))
    if policy.require_source_baseline_run_manifest_id and not candidate_report.source_baseline_run_manifest_id:
        findings.append(_finding("missing_source_baseline_run_manifest_id", "error", "candidate report lacks source baseline run manifest id"))
    if policy.require_source_benchmark_preflight_status and not candidate_report.source_benchmark_preflight_status:
        findings.append(_finding("missing_source_benchmark_preflight_status", "error", "candidate report lacks source benchmark preflight status"))
    findings.extend(_report_artifact_findings(report_manifest, policy))
    if policy.require_caveats:
        caveats = set(candidate_report.caveats)
        for caveat in _REQUIRED_CAVEATS:
            if caveat not in caveats:
                findings.append(_finding("missing_required_caveat", "error", "candidate report lacks required caveat"))
    if policy.registry_status != "candidate" or bool(policy.benchmark_validated):
        findings.append(_finding("benchmark_validated_source_claim", "error", "policy cannot benchmark-validate registry outputs"))
    if bool(policy.scientific_claim_validated):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "policy cannot validate scientific claims"))
    if policy.require_no_benchmark_validated_claims and _has_truthy_metadata_key("benchmark_validated", report_manifest.metadata, candidate_report.metadata, registry_preflight_report.metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", "source metadata claims benchmark validation"))
    if policy.require_not_scientific_claim and _has_truthy_metadata_key("scientific_claim_validated", report_manifest.metadata, candidate_report.metadata, registry_preflight_report.metadata):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "source metadata claims scientific validation"))
    return findings


def _report_artifact_findings(
    report_manifest: OledBaselineBenchmarkReportWriterManifest,
    policy: OledBenchmarkRegistryWriterPolicy,
) -> list[OledBenchmarkRegistryWriterFinding]:
    findings: list[OledBenchmarkRegistryWriterFinding] = []
    json_result = _file_result_for_kind(report_manifest.file_results, "benchmark_report_json")
    markdown_result = _file_result_for_kind(report_manifest.file_results, "benchmark_report_markdown")
    if policy.require_report_json_sha256 and (json_result is None or not json_result.output_sha256):
        findings.append(_finding("missing_report_json_sha256", "error", "candidate report JSON SHA256 is required"))
    if policy.require_report_markdown_sha256 and (markdown_result is None or not markdown_result.output_sha256):
        findings.append(_finding("missing_report_markdown_sha256", "error", "candidate report Markdown SHA256 is required"))
    return findings


def _filter_run_cards(
    candidate_report: OledBaselineBenchmarkCandidateReport,
    policy: OledBenchmarkRegistryWriterPolicy,
) -> list[Any]:
    baselines = _baseline_kinds(policy)
    targets = _target_property_ids(policy)
    views = _feature_views(policy)
    return sorted(
        [
            card
            for card in candidate_report.run_cards
            if (not baselines or card.baseline_kind in baselines)
            and (not targets or card.target_property_id in targets)
            and (not views or card.feature_view in views)
        ],
        key=lambda item: (item.baseline_kind, item.target_property_id, item.feature_view),
    )


def _manifest(
    *,
    policy: OledBenchmarkRegistryWriterPolicy,
    entry: OledBenchmarkRegistryEntry | None,
    findings: list[OledBenchmarkRegistryWriterFinding],
    report_manifest: OledBaselineBenchmarkReportWriterManifest,
    registry_preflight_report: OledBenchmarkRegistryPreflightReport,
    output_directory: str | None,
    entry_written: bool,
    index_written: bool,
    file_results: list[OledBenchmarkRegistryFileResult] | None = None,
) -> OledBenchmarkRegistryWriterManifest:
    return OledBenchmarkRegistryWriterManifest(
        manifest_id=_registry_entry_id(report_manifest.manifest_id, entry.source_candidate_report_id if entry is not None else None, registry_preflight_report.status).replace(
            "entry:", "manifest:"
        ),
        source_benchmark_report_manifest_id=report_manifest.manifest_id,
        source_benchmark_registry_preflight_status=_status_value(registry_preflight_report.status),
        source_candidate_report_id=entry.source_candidate_report_id if entry is not None else None,
        output_directory=output_directory,
        output_file_count=sum(1 for result in (file_results or []) if result.status == OledBenchmarkRegistryWriteStatus.WRITTEN),
        registry_entry_ids=[entry.registry_entry_id] if entry is not None else [],
        baseline_kinds=entry.baseline_kinds if entry is not None else [],
        target_property_ids=entry.target_property_ids if entry is not None else [],
        feature_views=entry.feature_views if entry is not None else [],
        file_results=file_results or _selection_file_results(entry, findings),
        policy=policy,
        metadata=_safety_metadata(entry_written=entry_written, index_written=index_written),
    )


def _selection_file_results(
    entry: OledBenchmarkRegistryEntry | None,
    findings: list[OledBenchmarkRegistryWriterFinding],
) -> list[OledBenchmarkRegistryFileResult]:
    if entry is None:
        return [
            OledBenchmarkRegistryFileResult(
                artifact_kind="registry_entry_json",
                status=OledBenchmarkRegistryWriteStatus.REJECTED,
                reason_codes=sorted({finding.code for finding in findings} or {"registry_entry_rejected"}),
            )
        ]
    return [
        OledBenchmarkRegistryFileResult(
            artifact_kind="registry_entry_json",
            status=OledBenchmarkRegistryWriteStatus.SKIPPED,
            reason_codes=["selected_for_registry"],
        ),
        OledBenchmarkRegistryFileResult(
            artifact_kind="registry_index_jsonl",
            status=OledBenchmarkRegistryWriteStatus.SKIPPED,
            reason_codes=["selected_for_registry"],
        ),
    ]


def _write_registry_files(
    writer_report: OledBenchmarkRegistryWriterReport,
    output_root: Path,
) -> OledBenchmarkRegistryWriterReport:
    assert writer_report.registry_entry is not None
    entry = writer_report.registry_entry
    file_results: list[OledBenchmarkRegistryFileResult] = []
    index_records = list(writer_report.registry_index_records)
    entry_sha: str | None = None
    if writer_report.manifest.policy.write_registry_entry_json:
        entry_path = output_root / oled_benchmark_registry_entry_filename()
        entry_sha = write_oled_benchmark_registry_entry_json(
            entry.model_copy(update={"metadata": _safety_metadata(entry_written=True, index_written=False)}),
            entry_path,
        )
        file_results.append(
            OledBenchmarkRegistryFileResult(
                artifact_kind="registry_entry_json",
                status=OledBenchmarkRegistryWriteStatus.WRITTEN,
                output_path=entry_path.name,
                output_sha256=entry_sha,
                reason_codes=["registry_entry_json_written", "selected_for_registry"],
            )
        )
        index_records = [
            record.model_copy(
                update={
                    "output_registry_entry_json_path": entry_path.name,
                    "output_registry_entry_json_sha256": entry_sha,
                }
            )
            for record in index_records
        ]
    if writer_report.manifest.policy.write_registry_index_jsonl:
        index_path = output_root / oled_benchmark_registry_index_filename()
        index_sha = write_oled_benchmark_registry_index_jsonl(index_records, index_path)
        file_results.append(
            OledBenchmarkRegistryFileResult(
                artifact_kind="registry_index_jsonl",
                status=OledBenchmarkRegistryWriteStatus.WRITTEN,
                output_path=index_path.name,
                output_sha256=index_sha,
                reason_codes=["registry_index_jsonl_written", "selected_for_registry"],
            )
        )
    updated_entry = entry.model_copy(update={"metadata": _safety_metadata(entry_written=True, index_written=writer_report.manifest.policy.write_registry_index_jsonl)})
    manifest = writer_report.manifest.model_copy(
        update={
            "output_directory": output_root.name,
            "output_file_count": len(file_results),
            "file_results": file_results,
            "metadata": _safety_metadata(
                entry_written=writer_report.manifest.policy.write_registry_entry_json,
                index_written=writer_report.manifest.policy.write_registry_index_jsonl,
            ),
        }
    )
    return writer_report.model_copy(update={"manifest": manifest, "registry_entry": updated_entry, "registry_index_records": index_records})


def _mark_dry_run(
    writer_report: OledBenchmarkRegistryWriterReport,
) -> OledBenchmarkRegistryWriterReport:
    manifest = writer_report.manifest.model_copy(
        update={
            "metadata": {
                **writer_report.manifest.metadata,
                "dry_run_no_files_written": True,
                "benchmark_registry_entry_written": False,
                "benchmark_registry_index_written": False,
            },
            "file_results": [
                result.model_copy(update={"reason_codes": sorted(set(result.reason_codes) | {"dry_run_no_files_written"})})
                for result in writer_report.manifest.file_results
            ],
        }
    )
    return writer_report.model_copy(update={"manifest": manifest})


def _file_result_for_kind(file_results: Iterable[Any], artifact_kind: str) -> Any | None:
    for file_result in file_results:
        if file_result.artifact_kind == artifact_kind:
            return file_result
    return None


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _registry_entry_id(manifest_id: str | None, report_id: str | None, status: Enum | str) -> str:
    return "entry:oled-benchmark-registry:" + _safe_id_token(f"{manifest_id or 'unknown'}:{report_id or 'unknown'}:{_status_value(status)}")


def _safe_id_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", value).strip("-").lower() or "unknown"


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    output_path: str | None = None,
) -> OledBenchmarkRegistryWriterFinding:
    return OledBenchmarkRegistryWriterFinding(
        code=code,
        severity=severity,
        message=message,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        output_path=output_path,
    )


def _dedup_findings(
    findings: list[OledBenchmarkRegistryWriterFinding],
) -> list[OledBenchmarkRegistryWriterFinding]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    output: list[OledBenchmarkRegistryWriterFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.baseline_kind or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.output_path or "",
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return sorted(output, key=lambda item: (item.severity, item.code, item.baseline_kind or "", item.target_property_id or "", item.feature_view or ""))


def _baseline_kinds(policy: OledBenchmarkRegistryWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.baseline_kinds if str(item).strip()}


def _target_property_ids(policy: OledBenchmarkRegistryWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.target_property_ids if str(item).strip()}


def _feature_views(policy: OledBenchmarkRegistryWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.feature_views if str(item).strip()}


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _has_truthy_metadata_key(key: str, *metadata_items: dict[str, Any]) -> bool:
    return any(bool(metadata.get(key)) for metadata in metadata_items)


def _warning_codes(report: OledBenchmarkRegistryPreflightReport) -> list[str]:
    codes: list[str] = []
    for finding in report.findings:
        if isinstance(finding, dict):
            if finding.get("severity") == "warning" and finding.get("code"):
                codes.append(str(finding["code"]))
        elif finding.severity == "warning":
            codes.append(finding.code)
    return codes


def _write_bytes(path: str | Path, payload: bytes) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _reject_forbidden_input(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"forbidden_pdf_input:{redact_oled_mineru_acceptance_path(path)}")
    if suffix in _FORBIDDEN_IMAGE_SUFFIXES:
        raise ValueError(f"forbidden_image_input:{redact_oled_mineru_acceptance_path(path)}")


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
            if str(key).lower() in _FORBIDDEN_JSON_KEYS:
                return True
            if _contains_forbidden_payload_key(item):
                return True
    if isinstance(value, list):
        return any(_contains_forbidden_payload_key(item) for item in value)
    return False


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


def _safety_metadata(*, entry_written: bool, index_written: bool) -> dict[str, Any]:
    return {
        "benchmark_registry_writer": True,
        "benchmark_registry_entry_written": entry_written,
        "benchmark_registry_index_written": index_written,
        "registry_status": "candidate",
        "benchmark_validated": False,
        "scientific_claim_validated": False,
        "baseline_backend_rerun": False,
        "models_fitted": False,
        "predictions_written": False,
        "metrics_written": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


_REQUIRED_CAVEATS = {
    "baseline_candidate_report_only",
    "not_benchmark_validated",
    "not_scientific_performance_claim",
}

_MAX_OUTPUT_STRING_LENGTH = 240

_FORBIDDEN_JSON_KEYS = {
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
    "OledBenchmarkRegistryWriterPolicy",
    "OledBenchmarkRegistryWriteStatus",
    "OledBenchmarkRegistryEntryStatus",
    "OledBenchmarkRegistryEntry",
    "OledBenchmarkRegistryIndexRecord",
    "OledBenchmarkRegistryFileResult",
    "OledBenchmarkRegistryWriterFinding",
    "OledBenchmarkRegistryWriterManifest",
    "OledBenchmarkRegistryWriterReport",
    "load_oled_benchmark_registry_preflight_report_json",
    "build_oled_benchmark_registry_entry",
    "build_oled_benchmark_registry_index_records",
    "select_oled_benchmark_registry_entry_for_write",
    "write_oled_benchmark_registry_entry_json",
    "write_oled_benchmark_registry_index_jsonl",
    "write_oled_benchmark_registry_manifest_json",
    "load_oled_benchmark_registry_entry_json",
    "load_oled_benchmark_registry_index_jsonl",
    "oled_benchmark_registry_entry_filename",
    "oled_benchmark_registry_index_filename",
    "run_oled_benchmark_registry_writer_from_files",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
