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

from ai4s_agent.domains.oled_curated_final_registry_global_append_preflight import (
    OledFinalRegistryExistingRecordSummary,
    load_oled_existing_final_registry_snapshot_jsonl,
)
from ai4s_agent.domains.oled_curated_final_registry_global_append_writer import (
    OledFinalRegistryGlobalAppendWriteStatus,
    OledFinalRegistryGlobalAppendWriterManifest,
    OledGlobalAppendCandidateEntry,
    OledGlobalAppendCandidateEntryStatus,
    OledGlobalAppendCandidateIndexRecord,
    load_oled_global_append_candidate_delta_jsonl,
    load_oled_global_append_candidate_entry_json,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledGlobalAppendReleasePreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledGlobalAppendReleaseArtifactStatus(str, Enum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    FAILED = "failed"
    SKIPPED = "skipped"


class OledGlobalAppendReleasePreflightPolicy(BaseModel):
    require_global_append_writer_manifest_sha256: bool = True
    require_global_append_entry_sha256: bool = True
    require_global_append_delta_sha256: bool = True
    require_global_registry_snapshot_sha256: bool = True

    require_global_append_entry_json: bool = True
    require_global_append_delta_jsonl: bool = True
    require_global_registry_snapshot_jsonl: bool = True

    require_global_append_candidate_status: bool = True
    require_entry_in_delta: bool = True
    require_entry_in_snapshot: bool = True
    require_delta_records_in_snapshot: bool = True
    require_prior_snapshot_preserved: bool = True
    require_single_global_append_delta_record: bool = True

    require_source_final_registry_writer_manifest_id: bool = True
    require_source_final_registry_entry_id: bool = True
    require_source_global_append_preflight_status: bool = True
    require_source_publication_entry_id: bool = True
    require_source_publication_writer_manifest_id: bool = True
    require_source_promoted_entry_id: bool = True
    require_source_promotion_writer_manifest_id: bool = True
    require_source_registry_entry_id: bool = True
    require_source_registry_writer_manifest_id: bool = True
    require_source_candidate_report_id: bool = True
    require_source_benchmark_report_manifest_id: bool = True

    require_valid_global_append_preflight_status: bool = True
    require_caveats: bool = True
    require_run_cards: bool = True
    require_metric_cards: bool = True

    require_no_benchmark_validated_claims: bool = True
    require_no_scientific_claims: bool = True
    require_no_external_publication_claims: bool = True

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


class OledGlobalAppendReleaseArtifactSummary(BaseModel):
    artifact_kind: str
    status: OledGlobalAppendReleaseArtifactStatus
    output_path: str | None = None
    output_sha256: str | None = None
    loaded: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class OledGlobalAppendReleaseEntrySummary(BaseModel):
    global_append_entry_id: str
    global_append_status: str

    source_final_registry_entry_id: str | None = None
    source_final_registry_writer_manifest_id: str | None = None
    source_global_append_preflight_status: str | None = None

    source_publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    run_card_count: int = 0
    metric_card_count: int = 0

    delta_record_count: int = 0
    matched_delta_record_count: int = 0

    snapshot_record_count: int = 0
    matched_snapshot_record_count: int = 0
    prior_snapshot_record_count: int = 0
    preserved_prior_snapshot_record_count: int = 0

    artifact_status: OledGlobalAppendReleaseArtifactStatus
    reason_codes: list[str] = Field(default_factory=list)


class OledGlobalAppendReleasePreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    artifact_kind: str | None = None
    global_append_entry_id: str | None = None
    source_final_registry_entry_id: str | None = None
    source_publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    output_path: str | None = None


class OledGlobalAppendReleasePreflightReport(BaseModel):
    status: OledGlobalAppendReleasePreflightStatus

    source_global_append_writer_manifest_id: str | None = None
    source_global_append_entry_id: str | None = None
    source_final_registry_entry_id: str | None = None
    source_publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None
    source_global_append_preflight_status: str | None = None

    input_global_append_entry_count: int = 0
    input_global_append_delta_record_count: int = 0
    input_global_registry_snapshot_record_count: int = 0
    input_prior_registry_snapshot_record_count: int = 0

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    artifact_summaries: list[OledGlobalAppendReleaseArtifactSummary] = Field(default_factory=list)
    entry_summaries: list[OledGlobalAppendReleaseEntrySummary] = Field(default_factory=list)

    caveats: list[str] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    findings: list[OledGlobalAppendReleasePreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledGlobalAppendReleasePreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_final_registry_global_append_writer_manifest_json(
    path: str | Path,
) -> OledFinalRegistryGlobalAppendWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_final_registry_global_append_writer_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledFinalRegistryGlobalAppendWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_final_registry_global_append_writer_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.model_dump(mode="json")):
        raise ValueError("absolute_path_in_final_registry_global_append_writer_manifest")
    return manifest


def load_oled_global_append_artifacts_from_manifest(
    *,
    manifest: OledFinalRegistryGlobalAppendWriterManifest,
    base_dir: str | Path,
) -> tuple[
    OledGlobalAppendCandidateEntry | None,
    list[OledGlobalAppendCandidateIndexRecord],
    list[OledFinalRegistryExistingRecordSummary],
]:
    entry: OledGlobalAppendCandidateEntry | None = None
    delta_records: list[OledGlobalAppendCandidateIndexRecord] = []
    snapshot_records: list[OledFinalRegistryExistingRecordSummary] = []
    for file_result in manifest.file_results:
        if _status_value(file_result.status) != OledFinalRegistryGlobalAppendWriteStatus.WRITTEN.value:
            continue
        if not file_result.output_path:
            continue
        path = _resolve_manifest_path(file_result.output_path, base_dir)
        if file_result.artifact_kind == "global_append_entry_json":
            if not path.exists():
                raise ValueError(f"missing_global_append_candidate_entry_json:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"global_append_entry_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            entry = load_oled_global_append_candidate_entry_json(path)
        elif file_result.artifact_kind == "global_append_delta_jsonl":
            if not path.exists():
                raise ValueError(f"missing_global_append_candidate_delta_jsonl:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"global_append_delta_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            delta_records = load_oled_global_append_candidate_delta_jsonl(path)
        elif file_result.artifact_kind == "global_registry_snapshot_jsonl":
            if not path.exists():
                raise ValueError(f"missing_global_registry_snapshot_jsonl:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"global_registry_snapshot_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            snapshot_records = load_oled_existing_final_registry_snapshot_jsonl(path)
    return entry, delta_records, snapshot_records


def run_oled_global_append_release_preflight(
    *,
    global_append_writer_manifest: OledFinalRegistryGlobalAppendWriterManifest,
    global_append_entry: OledGlobalAppendCandidateEntry | None,
    global_append_delta_records: Iterable[OledGlobalAppendCandidateIndexRecord],
    global_registry_snapshot_records: Iterable[OledFinalRegistryExistingRecordSummary],
    prior_registry_snapshot_records: Iterable[OledFinalRegistryExistingRecordSummary] | None = None,
    policy: OledGlobalAppendReleasePreflightPolicy | None = None,
) -> OledGlobalAppendReleasePreflightReport:
    preflight_policy = policy or OledGlobalAppendReleasePreflightPolicy()
    delta_records = list(global_append_delta_records)
    snapshot_records = list(global_registry_snapshot_records)
    prior_records = list(prior_registry_snapshot_records or [])
    artifact_summaries = _artifact_summaries(global_append_writer_manifest, global_append_entry, delta_records, snapshot_records, preflight_policy)
    entry_summaries = _entry_summaries(global_append_entry, delta_records, snapshot_records, prior_records, preflight_policy)
    findings: list[OledGlobalAppendReleasePreflightFinding] = []
    findings.extend(_manifest_findings(global_append_writer_manifest, preflight_policy))
    findings.extend(_artifact_findings(artifact_summaries))
    findings.extend(_entry_findings(global_append_entry, delta_records, snapshot_records, preflight_policy))
    findings.extend(_delta_findings(global_append_entry, delta_records, snapshot_records, preflight_policy))
    findings.extend(_snapshot_findings(snapshot_records, prior_records, preflight_policy))
    findings = _dedup_findings(findings)
    status = _report_status(findings)
    status_counts = Counter(_status_value(summary.status) for summary in artifact_summaries)
    status_counts.update(_status_value(summary.artifact_status) for summary in entry_summaries)

    return OledGlobalAppendReleasePreflightReport(
        status=status,
        source_global_append_writer_manifest_id=global_append_writer_manifest.manifest_id,
        source_global_append_entry_id=global_append_entry.global_append_entry_id if global_append_entry is not None else None,
        source_final_registry_entry_id=(
            global_append_entry.source_final_registry_entry_id
            if global_append_entry is not None
            else global_append_writer_manifest.source_final_registry_entry_id
        ),
        source_publication_entry_id=global_append_entry.source_publication_entry_id if global_append_entry is not None else None,
        source_promoted_entry_id=global_append_entry.source_promoted_entry_id if global_append_entry is not None else None,
        source_registry_entry_id=global_append_entry.source_registry_entry_id if global_append_entry is not None else None,
        source_candidate_report_id=global_append_entry.source_candidate_report_id if global_append_entry is not None else None,
        source_benchmark_report_manifest_id=global_append_entry.source_benchmark_report_manifest_id if global_append_entry is not None else None,
        source_global_append_preflight_status=(
            global_append_entry.source_global_append_preflight_status
            if global_append_entry is not None
            else global_append_writer_manifest.source_global_append_preflight_status
        ),
        input_global_append_entry_count=1 if global_append_entry is not None else 0,
        input_global_append_delta_record_count=len(delta_records),
        input_global_registry_snapshot_record_count=len(snapshot_records),
        input_prior_registry_snapshot_record_count=len(prior_records),
        baseline_kinds=_selected_values(
            global_append_entry.baseline_kinds if global_append_entry is not None else global_append_writer_manifest.baseline_kinds,
            preflight_policy.baseline_kinds,
        ),
        target_property_ids=_selected_values(
            global_append_entry.target_property_ids if global_append_entry is not None else global_append_writer_manifest.target_property_ids,
            preflight_policy.target_property_ids,
        ),
        feature_views=_selected_values(
            global_append_entry.feature_views if global_append_entry is not None else global_append_writer_manifest.feature_views,
            preflight_policy.feature_views,
        ),
        artifact_summaries=artifact_summaries,
        entry_summaries=entry_summaries,
        caveats=sorted(global_append_entry.caveats) if global_append_entry is not None else [],
        status_counts=dict(sorted(status_counts.items())),
        finding_code_counts=dict(sorted(Counter(finding.code for finding in findings).items())),
        findings=findings,
        metadata=_safety_metadata(),
    )


def run_oled_global_append_release_preflight_from_files(
    *,
    global_append_writer_manifest_path: str | Path,
    global_append_base_dir: str | Path | None = None,
    prior_registry_snapshot_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledGlobalAppendReleasePreflightPolicy | None = None,
) -> OledGlobalAppendReleasePreflightReport:
    manifest = load_oled_final_registry_global_append_writer_manifest_json(global_append_writer_manifest_path)
    base_dir = Path(global_append_base_dir) if global_append_base_dir is not None else Path(global_append_writer_manifest_path).parent
    entry, delta_records, snapshot_records = load_oled_global_append_artifacts_from_manifest(manifest=manifest, base_dir=base_dir)
    prior_records = load_oled_existing_final_registry_snapshot_jsonl(prior_registry_snapshot_path) if prior_registry_snapshot_path is not None else None
    report = run_oled_global_append_release_preflight(
        global_append_writer_manifest=manifest,
        global_append_entry=entry,
        global_append_delta_records=delta_records,
        global_registry_snapshot_records=snapshot_records,
        prior_registry_snapshot_records=prior_records,
        policy=policy,
    )
    if output_report_path is not None:
        write_oled_global_append_release_preflight_report_json(report, output_report_path)
    return report


def write_oled_global_append_release_preflight_report_json(
    report: OledGlobalAppendReleasePreflightReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(report.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED global-append release-readiness preflight.")
    parser.add_argument("--global-append-writer-manifest", required=True, help="Path to global-append writer manifest JSON.")
    parser.add_argument("--global-append-base-dir", help="Base directory for global-append candidate artifacts.")
    parser.add_argument("--prior-registry-snapshot", help="Optional prior registry snapshot JSONL for preservation checks.")
    parser.add_argument("--output-report", help="Optional release-readiness preflight report JSON path.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--allow-multiple-global-append-delta-records", action="store_true", help="Allow multiple delta records.")
    parser.add_argument(
        "--allow-missing-prior-snapshot-preservation-check",
        action="store_true",
        help="Skip prior snapshot prefix preservation checks.",
    )
    args = parser.parse_args(argv)
    try:
        policy = OledGlobalAppendReleasePreflightPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            require_single_global_append_delta_record=not args.allow_multiple_global_append_delta_records,
            require_prior_snapshot_preserved=not args.allow_missing_prior_snapshot_preservation_check,
        )
        report = run_oled_global_append_release_preflight_from_files(
            global_append_writer_manifest_path=args.global_append_writer_manifest,
            global_append_base_dir=args.global_append_base_dir,
            prior_registry_snapshot_path=args.prior_registry_snapshot,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": _status_value(report.status),
            "entry_summary_count": len(report.entry_summaries),
            "artifact_summary_count": len(report.artifact_summaries),
            "delta_record_count": report.input_global_append_delta_record_count,
            "snapshot_record_count": report.input_global_registry_snapshot_record_count,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _manifest_findings(
    manifest: OledFinalRegistryGlobalAppendWriterManifest,
    policy: OledGlobalAppendReleasePreflightPolicy,
) -> list[OledGlobalAppendReleasePreflightFinding]:
    findings: list[OledGlobalAppendReleasePreflightFinding] = []
    if policy.require_source_final_registry_writer_manifest_id and not manifest.source_final_registry_writer_manifest_id:
        findings.append(_finding("missing_source_final_registry_writer_manifest_id", "error", "global-append writer manifest lacks source final-registry writer manifest id"))
    if policy.require_source_final_registry_entry_id and not manifest.source_final_registry_entry_id:
        findings.append(_finding("missing_source_final_registry_entry_id", "error", "global-append writer manifest lacks source final-registry entry id"))
    if policy.require_source_global_append_preflight_status and not manifest.source_global_append_preflight_status:
        findings.append(_finding("missing_source_global_append_preflight_status", "error", "global-append writer manifest lacks source global-append preflight status"))
    if policy.require_valid_global_append_preflight_status and manifest.source_global_append_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_global_append_preflight_status", "error", "source global-append preflight status is not valid"))
    if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", manifest.metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", "global-append writer manifest claims benchmark validation"))
    if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", manifest.metadata):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "global-append writer manifest claims scientific validation"))
    if policy.require_no_external_publication_claims and _metadata_claims_external_publication(manifest.metadata):
        findings.append(_finding("external_publication_source_claim", "error", "global-append writer manifest claims external publication or global mutation"))
    return findings


def _artifact_summaries(
    manifest: OledFinalRegistryGlobalAppendWriterManifest,
    entry: OledGlobalAppendCandidateEntry | None,
    delta_records: list[OledGlobalAppendCandidateIndexRecord],
    snapshot_records: list[OledFinalRegistryExistingRecordSummary],
    policy: OledGlobalAppendReleasePreflightPolicy,
) -> list[OledGlobalAppendReleaseArtifactSummary]:
    summaries: list[OledGlobalAppendReleaseArtifactSummary] = []
    for artifact_kind, required, loaded, require_sha in (
        ("global_append_entry_json", policy.require_global_append_entry_json, entry is not None, policy.require_global_append_entry_sha256),
        ("global_append_delta_jsonl", policy.require_global_append_delta_jsonl, bool(delta_records), policy.require_global_append_delta_sha256),
        ("global_registry_snapshot_jsonl", policy.require_global_registry_snapshot_jsonl, bool(snapshot_records), policy.require_global_registry_snapshot_sha256),
    ):
        file_result = _file_result_for_kind(manifest, artifact_kind)
        reasons: set[str] = set()
        status = OledGlobalAppendReleaseArtifactStatus.READY
        if loaded:
            reasons.add("artifact_loaded")
        elif required:
            reasons.add(_missing_artifact_reason(artifact_kind))
            status = OledGlobalAppendReleaseArtifactStatus.FAILED
        else:
            reasons.add("artifact_optional")
            status = OledGlobalAppendReleaseArtifactStatus.SKIPPED
        if require_sha and file_result is not None and not file_result.output_sha256:
            reasons.add(f"missing_{artifact_kind}_sha256")
            status = OledGlobalAppendReleaseArtifactStatus.FAILED
        summaries.append(
            OledGlobalAppendReleaseArtifactSummary(
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
    summaries: list[OledGlobalAppendReleaseArtifactSummary],
) -> list[OledGlobalAppendReleasePreflightFinding]:
    findings: list[OledGlobalAppendReleasePreflightFinding] = []
    for summary in summaries:
        if summary.status != OledGlobalAppendReleaseArtifactStatus.FAILED:
            continue
        for reason in summary.reason_codes:
            findings.append(
                _finding(
                    reason,
                    "error",
                    "global-append release artifact is not ready",
                    artifact_kind=summary.artifact_kind,
                    output_path=summary.output_path,
                )
            )
    return findings


def _entry_summaries(
    entry: OledGlobalAppendCandidateEntry | None,
    delta_records: list[OledGlobalAppendCandidateIndexRecord],
    snapshot_records: list[OledFinalRegistryExistingRecordSummary],
    prior_records: list[OledFinalRegistryExistingRecordSummary],
    policy: OledGlobalAppendReleasePreflightPolicy,
) -> list[OledGlobalAppendReleaseEntrySummary]:
    if entry is None:
        return []
    matched_delta_count = sum(1 for record in delta_records if record.global_append_entry_id == entry.global_append_entry_id)
    matched_snapshot_count = sum(1 for record in snapshot_records if _snapshot_record_matches_entry(record, entry))
    preserved_prior_count = _preserved_prior_count(prior_records, snapshot_records)
    reasons: set[str] = {"entry_loaded"}
    status = OledGlobalAppendReleaseArtifactStatus.READY
    for code, failed in (
        ("global_append_status_not_candidate", policy.require_global_append_candidate_status and _status_value(entry.global_append_status) != "global_append_candidate"),
        ("global_append_entry_not_in_delta", policy.require_entry_in_delta and matched_delta_count == 0),
        ("global_append_entry_not_in_snapshot", policy.require_entry_in_snapshot and matched_snapshot_count == 0),
        ("missing_run_cards", policy.require_run_cards and entry.run_card_count <= 0),
        ("missing_metric_cards", policy.require_metric_cards and entry.metric_card_count <= 0),
        ("prior_snapshot_not_preserved", bool(prior_records) and policy.require_prior_snapshot_preserved and preserved_prior_count != len(prior_records)),
    ):
        if failed:
            reasons.add(code)
            status = OledGlobalAppendReleaseArtifactStatus.FAILED
    return [
        OledGlobalAppendReleaseEntrySummary(
            global_append_entry_id=entry.global_append_entry_id,
            global_append_status=_status_value(entry.global_append_status),
            source_final_registry_entry_id=entry.source_final_registry_entry_id,
            source_final_registry_writer_manifest_id=entry.source_final_registry_writer_manifest_id,
            source_global_append_preflight_status=entry.source_global_append_preflight_status,
            source_publication_entry_id=entry.source_publication_entry_id,
            source_promoted_entry_id=entry.source_promoted_entry_id,
            source_registry_entry_id=entry.source_registry_entry_id,
            source_candidate_report_id=entry.source_candidate_report_id,
            source_benchmark_report_manifest_id=entry.source_benchmark_report_manifest_id,
            baseline_kinds=_selected_values(entry.baseline_kinds, policy.baseline_kinds),
            target_property_ids=_selected_values(entry.target_property_ids, policy.target_property_ids),
            feature_views=_selected_values(entry.feature_views, policy.feature_views),
            run_card_count=entry.run_card_count,
            metric_card_count=entry.metric_card_count,
            delta_record_count=len(delta_records),
            matched_delta_record_count=matched_delta_count,
            snapshot_record_count=len(snapshot_records),
            matched_snapshot_record_count=matched_snapshot_count,
            prior_snapshot_record_count=len(prior_records),
            preserved_prior_snapshot_record_count=preserved_prior_count,
            artifact_status=status,
            reason_codes=sorted(reasons),
        )
    ]


def _entry_findings(
    entry: OledGlobalAppendCandidateEntry | None,
    delta_records: list[OledGlobalAppendCandidateIndexRecord],
    snapshot_records: list[OledFinalRegistryExistingRecordSummary],
    policy: OledGlobalAppendReleasePreflightPolicy,
) -> list[OledGlobalAppendReleasePreflightFinding]:
    findings: list[OledGlobalAppendReleasePreflightFinding] = []
    if entry is None:
        if policy.require_global_append_entry_json:
            findings.append(_finding("missing_global_append_candidate_entry_json", "error", "global-append candidate entry JSON is required", artifact_kind="global_append_entry_json"))
        return findings
    if policy.require_global_append_candidate_status and _status_value(entry.global_append_status) != "global_append_candidate":
        findings.append(_finding("global_append_status_not_candidate", "error", "global-append entry status is not global_append_candidate", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_final_registry_writer_manifest_id and not entry.source_final_registry_writer_manifest_id:
        findings.append(_finding("missing_source_final_registry_writer_manifest_id", "error", "global-append entry lacks source final-registry writer manifest id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_final_registry_entry_id and not entry.source_final_registry_entry_id:
        findings.append(_finding("missing_source_final_registry_entry_id", "error", "global-append entry lacks source final-registry entry id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_global_append_preflight_status and not entry.source_global_append_preflight_status:
        findings.append(_finding("missing_source_global_append_preflight_status", "error", "global-append entry lacks source global-append preflight status", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_publication_entry_id and not entry.source_publication_entry_id:
        findings.append(_finding("missing_source_publication_entry_id", "error", "global-append entry lacks source publication entry id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_publication_writer_manifest_id and not entry.source_publication_writer_manifest_id:
        findings.append(_finding("missing_source_publication_writer_manifest_id", "error", "global-append entry lacks source publication writer manifest id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_promoted_entry_id and not entry.source_promoted_entry_id:
        findings.append(_finding("missing_source_promoted_entry_id", "error", "global-append entry lacks source promoted entry id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_promotion_writer_manifest_id and not entry.source_promotion_writer_manifest_id:
        findings.append(_finding("missing_source_promotion_writer_manifest_id", "error", "global-append entry lacks source promotion writer manifest id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_registry_entry_id and not entry.source_registry_entry_id:
        findings.append(_finding("missing_source_registry_entry_id", "error", "global-append entry lacks source registry entry id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_registry_writer_manifest_id and not entry.source_registry_writer_manifest_id:
        findings.append(_finding("missing_source_registry_writer_manifest_id", "error", "global-append entry lacks source registry writer manifest id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_candidate_report_id and not entry.source_candidate_report_id:
        findings.append(_finding("missing_source_candidate_report_id", "error", "global-append entry lacks source candidate report id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_source_benchmark_report_manifest_id and not entry.source_benchmark_report_manifest_id:
        findings.append(_finding("missing_source_benchmark_report_manifest_id", "error", "global-append entry lacks source benchmark report manifest id", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_valid_global_append_preflight_status and entry.source_global_append_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_global_append_preflight_status", "error", "global-append entry source preflight status is not valid", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_caveats:
        caveats = set(entry.caveats)
        for caveat in policy.required_caveats:
            if caveat not in caveats:
                findings.append(_finding("missing_required_caveat", "error", "global-append entry lacks required caveat", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_run_cards and entry.run_card_count <= 0:
        findings.append(_finding("missing_run_cards", "error", "global-append entry has no run cards", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_metric_cards and entry.metric_card_count <= 0:
        findings.append(_finding("missing_metric_cards", "error", "global-append entry has no metric cards", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_entry_in_delta and not any(record.global_append_entry_id == entry.global_append_entry_id for record in delta_records):
        findings.append(_finding("global_append_entry_not_in_delta", "error", "global-append entry is not referenced by delta", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_entry_in_snapshot and not any(_snapshot_record_matches_entry(record, entry) for record in snapshot_records):
        findings.append(_finding("global_append_entry_not_in_snapshot", "error", "global-append entry is not represented in snapshot", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", entry.metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", "global-append entry claims benchmark validation", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", entry.metadata):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "global-append entry claims scientific validation", global_append_entry_id=entry.global_append_entry_id))
    if policy.require_no_external_publication_claims and _metadata_claims_external_publication(entry.metadata):
        findings.append(_finding("external_publication_source_claim", "error", "global-append entry claims external publication or global mutation", global_append_entry_id=entry.global_append_entry_id))
    if _contains_forbidden_payload_key(entry.model_dump(mode="json"), raw_only=True):
        findings.append(_finding("raw_prediction_payload_leaked", "error", "global-append entry contains raw prediction payload", global_append_entry_id=entry.global_append_entry_id))
    if _contains_feature_payload_key(entry.model_dump(mode="json")):
        findings.append(_finding("raw_feature_payload_leaked", "error", "global-append entry contains feature payload", global_append_entry_id=entry.global_append_entry_id))
    if _contains_absolute_path(entry.model_dump(mode="json")):
        findings.append(_finding("absolute_path_leakage", "error", "global-append entry contains absolute path", global_append_entry_id=entry.global_append_entry_id))
    return findings


def _delta_findings(
    entry: OledGlobalAppendCandidateEntry | None,
    delta_records: list[OledGlobalAppendCandidateIndexRecord],
    snapshot_records: list[OledFinalRegistryExistingRecordSummary],
    policy: OledGlobalAppendReleasePreflightPolicy,
) -> list[OledGlobalAppendReleasePreflightFinding]:
    findings: list[OledGlobalAppendReleasePreflightFinding] = []
    if not delta_records:
        if policy.require_global_append_delta_jsonl:
            findings.append(_finding("missing_global_append_candidate_delta_jsonl", "error", "global-append delta JSONL is required", artifact_kind="global_append_delta_jsonl"))
        return findings
    if policy.require_single_global_append_delta_record and len(delta_records) > 1:
        findings.append(_finding("multiple_global_append_delta_records", "error", "global-append delta has multiple records", artifact_kind="global_append_delta_jsonl"))
    for record in delta_records:
        if policy.require_global_append_candidate_status and record.global_append_status != "global_append_candidate":
            findings.append(_finding("delta_status_not_global_append_candidate", "error", "global-append delta status is not global_append_candidate", global_append_entry_id=record.global_append_entry_id))
        if policy.require_delta_records_in_snapshot and not any(_snapshot_record_matches_delta(snapshot, record) for snapshot in snapshot_records):
            findings.append(_finding("delta_record_not_in_snapshot", "error", "global-append delta record is not represented in snapshot", global_append_entry_id=record.global_append_entry_id))
        if policy.require_no_benchmark_validated_claims and (record.benchmark_validated or _truthy_metadata_key("benchmark_validated", record.metadata)):
            findings.append(_finding("benchmark_validated_source_claim", "error", "global-append delta claims benchmark validation", global_append_entry_id=record.global_append_entry_id))
        if policy.require_no_scientific_claims and (record.scientific_claim_validated or _truthy_metadata_key("scientific_claim_validated", record.metadata)):
            findings.append(_finding("scientific_claim_validated_source_claim", "error", "global-append delta claims scientific validation", global_append_entry_id=record.global_append_entry_id))
        if policy.require_no_external_publication_claims and (
            record.benchmark_published or record.benchmark_registered or _metadata_claims_external_publication(record.metadata)
        ):
            findings.append(_finding("external_publication_source_claim", "error", "global-append delta claims external publication or global mutation", global_append_entry_id=record.global_append_entry_id))
        if _contains_forbidden_payload_key(record.model_dump(mode="json"), raw_only=True):
            findings.append(_finding("raw_prediction_payload_leaked", "error", "global-append delta contains raw prediction payload", global_append_entry_id=record.global_append_entry_id))
        if _contains_feature_payload_key(record.model_dump(mode="json")):
            findings.append(_finding("raw_feature_payload_leaked", "error", "global-append delta contains feature payload", global_append_entry_id=record.global_append_entry_id))
        if _contains_absolute_path(record.model_dump(mode="json")):
            findings.append(_finding("absolute_path_leakage", "error", "global-append delta contains absolute path", global_append_entry_id=record.global_append_entry_id))
    if entry is not None and policy.require_entry_in_delta and not any(record.global_append_entry_id == entry.global_append_entry_id for record in delta_records):
        findings.append(_finding("global_append_entry_not_in_delta", "error", "global-append entry is not present in delta", global_append_entry_id=entry.global_append_entry_id))
    return findings


def _snapshot_findings(
    snapshot_records: list[OledFinalRegistryExistingRecordSummary],
    prior_records: list[OledFinalRegistryExistingRecordSummary],
    policy: OledGlobalAppendReleasePreflightPolicy,
) -> list[OledGlobalAppendReleasePreflightFinding]:
    findings: list[OledGlobalAppendReleasePreflightFinding] = []
    if not snapshot_records:
        if policy.require_global_registry_snapshot_jsonl:
            findings.append(_finding("missing_global_registry_snapshot_jsonl", "error", "global registry snapshot JSONL is required", artifact_kind="global_registry_snapshot_jsonl"))
        return findings
    if prior_records and policy.require_prior_snapshot_preserved and _preserved_prior_count(prior_records, snapshot_records) != len(prior_records):
        findings.append(_finding("prior_snapshot_not_preserved", "error", "prior registry snapshot is not preserved as prefix", artifact_kind="global_registry_snapshot_jsonl"))
    for record in snapshot_records:
        payload = record.model_dump(mode="json", exclude_none=True)
        if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", record.metadata):
            findings.append(_finding("benchmark_validated_source_claim", "error", "snapshot record claims benchmark validation"))
        if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", record.metadata):
            findings.append(_finding("scientific_claim_validated_source_claim", "error", "snapshot record claims scientific validation"))
        if policy.require_no_external_publication_claims and _metadata_claims_external_publication(record.metadata):
            findings.append(_finding("external_publication_source_claim", "error", "snapshot record claims external publication or global mutation"))
        if _contains_forbidden_payload_key(payload, raw_only=True):
            findings.append(_finding("raw_prediction_payload_leaked", "error", "snapshot record contains raw prediction payload"))
        if _contains_feature_payload_key(payload):
            findings.append(_finding("raw_feature_payload_leaked", "error", "snapshot record contains feature payload"))
        if _contains_absolute_path(payload):
            findings.append(_finding("absolute_path_leakage", "error", "snapshot record contains absolute path"))
    return findings


def _missing_artifact_reason(artifact_kind: str) -> str:
    return {
        "global_append_entry_json": "missing_global_append_candidate_entry_json",
        "global_append_delta_jsonl": "missing_global_append_candidate_delta_jsonl",
        "global_registry_snapshot_jsonl": "missing_global_registry_snapshot_jsonl",
    }.get(artifact_kind, f"missing_{artifact_kind}")


def _file_result_for_kind(
    manifest: OledFinalRegistryGlobalAppendWriterManifest,
    artifact_kind: str,
) -> Any | None:
    for file_result in manifest.file_results:
        if file_result.artifact_kind == artifact_kind:
            return file_result
    return None


def _snapshot_record_matches_entry(record: OledFinalRegistryExistingRecordSummary, entry: OledGlobalAppendCandidateEntry) -> bool:
    return (
        record.metadata.get("global_append_entry_id") == entry.global_append_entry_id
        or (
            record.source_publication_entry_id == entry.source_publication_entry_id
            and record.source_candidate_report_id == entry.source_candidate_report_id
            and record.source_benchmark_report_manifest_id == entry.source_benchmark_report_manifest_id
        )
    )


def _snapshot_record_matches_delta(record: OledFinalRegistryExistingRecordSummary, delta: OledGlobalAppendCandidateIndexRecord) -> bool:
    return (
        record.metadata.get("global_append_entry_id") == delta.global_append_entry_id
        or (
            record.source_publication_entry_id == delta.source_publication_entry_id
            and record.source_candidate_report_id == delta.source_candidate_report_id
            and record.source_benchmark_report_manifest_id == delta.source_benchmark_report_manifest_id
        )
    )


def _preserved_prior_count(
    prior_records: list[OledFinalRegistryExistingRecordSummary],
    snapshot_records: list[OledFinalRegistryExistingRecordSummary],
) -> int:
    count = 0
    for prior, snapshot in zip(prior_records, snapshot_records, strict=False):
        if _summary_key(prior) != _summary_key(snapshot):
            break
        count += 1
    return count


def _summary_key(record: OledFinalRegistryExistingRecordSummary) -> tuple[str, str, str, str, str]:
    return (
        record.registry_entry_id or "",
        record.registry_status or "",
        record.source_publication_entry_id or "",
        record.source_candidate_report_id or "",
        record.source_benchmark_report_manifest_id or "",
    )


def _selected_values(values: Iterable[str], selected: Iterable[str]) -> list[str]:
    selected_set = {str(item).strip() for item in selected if str(item).strip()}
    return sorted({str(item) for item in values if not selected_set or str(item) in selected_set})


def _report_status(
    findings: list[OledGlobalAppendReleasePreflightFinding],
) -> OledGlobalAppendReleasePreflightStatus:
    if any(finding.severity == "error" for finding in findings):
        return OledGlobalAppendReleasePreflightStatus.FAILED
    if any(finding.severity == "warning" for finding in findings):
        return OledGlobalAppendReleasePreflightStatus.PASSED_WITH_WARNINGS
    return OledGlobalAppendReleasePreflightStatus.PASSED


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _truthy_metadata_key(key: str, metadata: dict[str, Any]) -> bool:
    return bool(metadata.get(key))


def _metadata_claims_external_publication(metadata: dict[str, Any]) -> bool:
    return any(
        bool(metadata.get(key))
        for key in (
            "benchmark_published",
            "benchmark_registered",
            "globally_registered",
            "global_registry_mutated",
            "external_publication_written",
            "externally_published",
            "published",
        )
    )


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    artifact_kind: str | None = None,
    global_append_entry_id: str | None = None,
    source_final_registry_entry_id: str | None = None,
    source_publication_entry_id: str | None = None,
    source_promoted_entry_id: str | None = None,
    source_registry_entry_id: str | None = None,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    output_path: str | None = None,
) -> OledGlobalAppendReleasePreflightFinding:
    return OledGlobalAppendReleasePreflightFinding(
        code=code,
        severity=severity,
        message=message,
        artifact_kind=artifact_kind,
        global_append_entry_id=global_append_entry_id,
        source_final_registry_entry_id=source_final_registry_entry_id,
        source_publication_entry_id=source_publication_entry_id,
        source_promoted_entry_id=source_promoted_entry_id,
        source_registry_entry_id=source_registry_entry_id,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        output_path=output_path,
    )


def _dedup_findings(
    findings: list[OledGlobalAppendReleasePreflightFinding],
) -> list[OledGlobalAppendReleasePreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str, str, str, str, str]] = set()
    output: list[OledGlobalAppendReleasePreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.artifact_kind or "",
            finding.global_append_entry_id or "",
            finding.source_final_registry_entry_id or "",
            finding.source_publication_entry_id or "",
            finding.source_promoted_entry_id or "",
            finding.source_registry_entry_id or "",
            finding.target_property_id or "",
            finding.feature_view or "",
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return sorted(
        output,
        key=lambda item: (
            item.severity,
            item.code,
            item.global_append_entry_id or "",
            item.source_final_registry_entry_id or "",
            item.source_publication_entry_id or "",
            item.source_promoted_entry_id or "",
            item.source_registry_entry_id or "",
        ),
    )


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _resolve_manifest_path(output_path: str, base_dir: str | Path) -> Path:
    path = Path(output_path)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _contains_forbidden_payload_key(value: Any, *, raw_only: bool = False) -> bool:
    forbidden = _RAW_FORBIDDEN_JSON_KEYS if raw_only else _FORBIDDEN_JSON_KEYS
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in forbidden:
                return True
            if _contains_forbidden_payload_key(item, raw_only=raw_only):
                return True
    if isinstance(value, list):
        return any(_contains_forbidden_payload_key(item, raw_only=raw_only) for item in value)
    return False


def _contains_feature_payload_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == "features":
                return True
            if _contains_feature_payload_key(item):
                return True
    if isinstance(value, list):
        return any(_contains_feature_payload_key(item) for item in value)
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


def _safety_metadata() -> dict[str, Any]:
    return {
        "global_append_release_preflight_only": True,
        "global_registry_mutated": False,
        "external_publication_written": False,
        "benchmark_published": False,
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


_FORBIDDEN_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
_RAW_FORBIDDEN_JSON_KEYS = {"raw_text", "full_text", "prediction_id", "training_row_id"}
_FORBIDDEN_JSON_KEYS = {*_RAW_FORBIDDEN_JSON_KEYS, "features", "feature_dict"}
_MAX_OUTPUT_STRING_LENGTH = 512


__all__ = [
    "OledGlobalAppendReleasePreflightStatus",
    "OledGlobalAppendReleaseArtifactStatus",
    "OledGlobalAppendReleasePreflightPolicy",
    "OledGlobalAppendReleaseArtifactSummary",
    "OledGlobalAppendReleaseEntrySummary",
    "OledGlobalAppendReleasePreflightFinding",
    "OledGlobalAppendReleasePreflightReport",
    "load_oled_final_registry_global_append_writer_manifest_json",
    "load_oled_global_append_artifacts_from_manifest",
    "run_oled_global_append_release_preflight",
    "run_oled_global_append_release_preflight_from_files",
    "write_oled_global_append_release_preflight_report_json",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
