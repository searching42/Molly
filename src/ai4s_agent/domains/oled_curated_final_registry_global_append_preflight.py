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

from ai4s_agent.domains.oled_curated_publication_candidate_final_registry_writer import (
    OledFinalRegistryCandidateEntry,
    OledFinalRegistryCandidateEntryStatus,
    OledFinalRegistryCandidateIndexRecord,
    OledPublicationCandidateFinalRegistryWriteStatus,
    OledPublicationCandidateFinalRegistryWriterManifest,
    load_oled_final_registry_candidate_entry_json,
    load_oled_final_registry_candidate_index_jsonl,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledFinalRegistryGlobalAppendPreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledFinalRegistryGlobalAppendArtifactStatus(str, Enum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    FAILED = "failed"
    SKIPPED = "skipped"


class OledFinalRegistryGlobalAppendPreflightPolicy(BaseModel):
    require_final_registry_writer_manifest_sha256: bool = True
    require_final_registry_entry_sha256: bool = True
    require_final_registry_index_sha256: bool = True

    require_final_registry_entry_json: bool = True
    require_final_registry_index_jsonl: bool = True

    require_final_registry_candidate_status: bool = True
    require_entry_in_index: bool = True
    require_single_final_registry_index_record: bool = True

    require_source_publication_writer_manifest_id: bool = True
    require_source_publication_entry_id: bool = True
    require_source_final_registry_preflight_status: bool = True
    require_source_promoted_entry_id: bool = True
    require_source_promotion_writer_manifest_id: bool = True
    require_source_registry_entry_id: bool = True
    require_source_registry_writer_manifest_id: bool = True
    require_source_candidate_report_id: bool = True
    require_source_benchmark_report_manifest_id: bool = True

    require_valid_final_registry_preflight_status: bool = True
    require_caveats: bool = True
    require_run_cards: bool = True
    require_metric_cards: bool = True

    require_no_benchmark_validated_claims: bool = True
    require_no_scientific_claims: bool = True
    require_no_global_registry_claims: bool = True

    fail_on_existing_registry_duplicate_entry_id: bool = True
    fail_on_existing_registry_duplicate_source_chain: bool = True

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


class OledFinalRegistryGlobalAppendArtifactSummary(BaseModel):
    artifact_kind: str
    status: OledFinalRegistryGlobalAppendArtifactStatus
    output_path: str | None = None
    output_sha256: str | None = None
    loaded: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class OledFinalRegistryGlobalAppendEntrySummary(BaseModel):
    final_registry_entry_id: str
    final_registry_status: str

    source_publication_entry_id: str | None = None
    source_publication_writer_manifest_id: str | None = None
    source_final_registry_preflight_status: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    run_card_count: int = 0
    metric_card_count: int = 0

    final_registry_index_record_count: int = 0
    matched_final_registry_index_record_count: int = 0

    existing_registry_duplicate_entry_count: int = 0
    existing_registry_duplicate_source_chain_count: int = 0

    artifact_status: OledFinalRegistryGlobalAppendArtifactStatus
    reason_codes: list[str] = Field(default_factory=list)


class OledFinalRegistryExistingRecordSummary(BaseModel):
    registry_entry_id: str | None = None
    registry_status: str | None = None
    source_publication_entry_id: str | None = None
    source_publication_writer_manifest_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledFinalRegistryGlobalAppendPreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    artifact_kind: str | None = None
    final_registry_entry_id: str | None = None
    source_publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    output_path: str | None = None


class OledFinalRegistryGlobalAppendPreflightReport(BaseModel):
    status: OledFinalRegistryGlobalAppendPreflightStatus

    source_final_registry_writer_manifest_id: str | None = None
    source_final_registry_entry_id: str | None = None
    source_publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None
    source_final_registry_preflight_status: str | None = None

    input_final_registry_entry_count: int = 0
    input_final_registry_index_record_count: int = 0
    input_existing_registry_record_count: int = 0

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    artifact_summaries: list[OledFinalRegistryGlobalAppendArtifactSummary] = Field(default_factory=list)
    entry_summaries: list[OledFinalRegistryGlobalAppendEntrySummary] = Field(default_factory=list)
    existing_registry_summaries: list[OledFinalRegistryExistingRecordSummary] = Field(default_factory=list)

    caveats: list[str] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    findings: list[OledFinalRegistryGlobalAppendPreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledFinalRegistryGlobalAppendPreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_publication_candidate_final_registry_writer_manifest_json(
    path: str | Path,
) -> OledPublicationCandidateFinalRegistryWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_publication_candidate_final_registry_writer_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledPublicationCandidateFinalRegistryWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_publication_candidate_final_registry_writer_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.model_dump(mode="json")):
        raise ValueError("absolute_path_in_publication_candidate_final_registry_writer_manifest")
    return manifest


def load_oled_final_registry_candidate_artifacts_from_manifest(
    *,
    manifest: OledPublicationCandidateFinalRegistryWriterManifest,
    base_dir: str | Path,
) -> tuple[OledFinalRegistryCandidateEntry | None, list[OledFinalRegistryCandidateIndexRecord]]:
    final_registry_entry: OledFinalRegistryCandidateEntry | None = None
    index_records: list[OledFinalRegistryCandidateIndexRecord] = []
    for file_result in manifest.file_results:
        if _status_value(file_result.status) != OledPublicationCandidateFinalRegistryWriteStatus.WRITTEN.value:
            continue
        if not file_result.output_path:
            continue
        path = _resolve_manifest_path(file_result.output_path, base_dir)
        if file_result.artifact_kind == "final_registry_candidate_entry_json":
            if not path.exists():
                raise ValueError(f"missing_final_registry_candidate_entry_json:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"final_registry_candidate_entry_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            final_registry_entry = load_oled_final_registry_candidate_entry_json(path)
        elif file_result.artifact_kind == "final_registry_candidate_index_jsonl":
            if not path.exists():
                raise ValueError(f"missing_final_registry_candidate_index_jsonl:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"final_registry_candidate_index_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            index_records = load_oled_final_registry_candidate_index_jsonl(path)
    return final_registry_entry, index_records


def load_oled_existing_final_registry_snapshot_jsonl(
    path: str | Path,
) -> list[OledFinalRegistryExistingRecordSummary]:
    snapshot_path = Path(path)
    _reject_forbidden_input(snapshot_path)
    if not snapshot_path.exists():
        raise ValueError(f"missing_existing_final_registry_snapshot_jsonl:{redact_oled_mineru_acceptance_path(snapshot_path)}")
    records: list[OledFinalRegistryExistingRecordSummary] = []
    for line_number, line in enumerate(snapshot_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if _contains_forbidden_payload_key(payload) or _contains_absolute_path(payload):
                raise ValueError("forbidden existing final registry snapshot payload")
            record = OledFinalRegistryExistingRecordSummary.model_validate(_existing_record_payload(payload))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"invalid_existing_final_registry_snapshot_jsonl:line-{line_number}") from exc
        records.append(record)
    return records


def run_oled_final_registry_global_append_preflight(
    *,
    final_registry_writer_manifest: OledPublicationCandidateFinalRegistryWriterManifest,
    final_registry_entry: OledFinalRegistryCandidateEntry | None,
    final_registry_index_records: Iterable[OledFinalRegistryCandidateIndexRecord],
    existing_registry_records: Iterable[OledFinalRegistryExistingRecordSummary] | None = None,
    policy: OledFinalRegistryGlobalAppendPreflightPolicy | None = None,
) -> OledFinalRegistryGlobalAppendPreflightReport:
    preflight_policy = policy or OledFinalRegistryGlobalAppendPreflightPolicy()
    index_records = list(final_registry_index_records)
    existing_records = list(existing_registry_records or [])
    artifact_summaries = _artifact_summaries(final_registry_writer_manifest, final_registry_entry, index_records, preflight_policy)
    entry_summaries = _entry_summaries(final_registry_entry, index_records, existing_records, preflight_policy)
    findings: list[OledFinalRegistryGlobalAppendPreflightFinding] = []
    findings.extend(_manifest_findings(final_registry_writer_manifest, preflight_policy))
    findings.extend(_artifact_findings(artifact_summaries))
    findings.extend(_entry_findings(final_registry_entry, index_records, preflight_policy))
    findings.extend(_index_findings(final_registry_entry, index_records, preflight_policy))
    findings.extend(_existing_registry_findings(final_registry_entry, existing_records, preflight_policy))
    findings = _dedup_findings(findings)
    status = _report_status(findings)
    status_counts = Counter(_status_value(summary.status) for summary in artifact_summaries)
    status_counts.update(_status_value(summary.artifact_status) for summary in entry_summaries)

    return OledFinalRegistryGlobalAppendPreflightReport(
        status=status,
        source_final_registry_writer_manifest_id=final_registry_writer_manifest.manifest_id,
        source_final_registry_entry_id=final_registry_entry.final_registry_entry_id if final_registry_entry is not None else None,
        source_publication_entry_id=(
            final_registry_entry.source_publication_entry_id
            if final_registry_entry is not None
            else final_registry_writer_manifest.source_publication_entry_id
        ),
        source_promoted_entry_id=final_registry_entry.source_promoted_entry_id if final_registry_entry is not None else None,
        source_registry_entry_id=final_registry_entry.source_registry_entry_id if final_registry_entry is not None else None,
        source_candidate_report_id=final_registry_entry.source_candidate_report_id if final_registry_entry is not None else None,
        source_benchmark_report_manifest_id=final_registry_entry.source_benchmark_report_manifest_id if final_registry_entry is not None else None,
        source_final_registry_preflight_status=(
            final_registry_entry.source_final_registry_preflight_status
            if final_registry_entry is not None
            else final_registry_writer_manifest.source_final_registry_preflight_status
        ),
        input_final_registry_entry_count=1 if final_registry_entry is not None else 0,
        input_final_registry_index_record_count=len(index_records),
        input_existing_registry_record_count=len(existing_records),
        baseline_kinds=_selected_values(
            final_registry_entry.baseline_kinds if final_registry_entry is not None else final_registry_writer_manifest.baseline_kinds,
            preflight_policy.baseline_kinds,
        ),
        target_property_ids=_selected_values(
            final_registry_entry.target_property_ids if final_registry_entry is not None else final_registry_writer_manifest.target_property_ids,
            preflight_policy.target_property_ids,
        ),
        feature_views=_selected_values(
            final_registry_entry.feature_views if final_registry_entry is not None else final_registry_writer_manifest.feature_views,
            preflight_policy.feature_views,
        ),
        artifact_summaries=artifact_summaries,
        entry_summaries=entry_summaries,
        existing_registry_summaries=existing_records,
        caveats=sorted(final_registry_entry.caveats) if final_registry_entry is not None else [],
        status_counts=dict(sorted(status_counts.items())),
        finding_code_counts=dict(sorted(Counter(finding.code for finding in findings).items())),
        findings=findings,
        metadata=_safety_metadata(),
    )


def run_oled_final_registry_global_append_preflight_from_files(
    *,
    final_registry_writer_manifest_path: str | Path,
    final_registry_candidate_base_dir: str | Path | None = None,
    existing_registry_snapshot_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledFinalRegistryGlobalAppendPreflightPolicy | None = None,
) -> OledFinalRegistryGlobalAppendPreflightReport:
    manifest = load_oled_publication_candidate_final_registry_writer_manifest_json(final_registry_writer_manifest_path)
    base_dir = Path(final_registry_candidate_base_dir) if final_registry_candidate_base_dir is not None else Path(final_registry_writer_manifest_path).parent
    final_registry_entry, index_records = load_oled_final_registry_candidate_artifacts_from_manifest(manifest=manifest, base_dir=base_dir)
    existing_records = load_oled_existing_final_registry_snapshot_jsonl(existing_registry_snapshot_path) if existing_registry_snapshot_path is not None else None
    report = run_oled_final_registry_global_append_preflight(
        final_registry_writer_manifest=manifest,
        final_registry_entry=final_registry_entry,
        final_registry_index_records=index_records,
        existing_registry_records=existing_records,
        policy=policy,
    )
    if output_report_path is not None:
        write_oled_final_registry_global_append_preflight_report_json(report, output_report_path)
    return report


def write_oled_final_registry_global_append_preflight_report_json(
    report: OledFinalRegistryGlobalAppendPreflightReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(report.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED final-registry global-append-readiness preflight.")
    parser.add_argument("--final-registry-writer-manifest", required=True, help="Path to final-registry candidate writer manifest JSON.")
    parser.add_argument("--final-registry-candidate-base-dir", help="Base directory for final-registry candidate artifacts.")
    parser.add_argument("--existing-registry-snapshot", help="Optional existing registry snapshot JSONL for duplicate checks.")
    parser.add_argument("--output-report", help="Optional global-append-readiness preflight report JSON path.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--allow-multiple-final-registry-index-records", action="store_true", help="Allow more than one final-registry index record.")
    parser.add_argument("--allow-existing-duplicate-entry-id", action="store_true", help="Allow duplicate existing registry entry ids.")
    parser.add_argument("--allow-existing-duplicate-source-chain", action="store_true", help="Allow duplicate existing source chains.")
    args = parser.parse_args(argv)
    try:
        policy = OledFinalRegistryGlobalAppendPreflightPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            require_single_final_registry_index_record=not args.allow_multiple_final_registry_index_records,
            fail_on_existing_registry_duplicate_entry_id=not args.allow_existing_duplicate_entry_id,
            fail_on_existing_registry_duplicate_source_chain=not args.allow_existing_duplicate_source_chain,
        )
        report = run_oled_final_registry_global_append_preflight_from_files(
            final_registry_writer_manifest_path=args.final_registry_writer_manifest,
            final_registry_candidate_base_dir=args.final_registry_candidate_base_dir,
            existing_registry_snapshot_path=args.existing_registry_snapshot,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": _status_value(report.status),
            "entry_summary_count": len(report.entry_summaries),
            "artifact_summary_count": len(report.artifact_summaries),
            "existing_registry_record_count": report.input_existing_registry_record_count,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _manifest_findings(
    manifest: OledPublicationCandidateFinalRegistryWriterManifest,
    policy: OledFinalRegistryGlobalAppendPreflightPolicy,
) -> list[OledFinalRegistryGlobalAppendPreflightFinding]:
    findings: list[OledFinalRegistryGlobalAppendPreflightFinding] = []
    if policy.require_source_publication_writer_manifest_id and not manifest.source_publication_writer_manifest_id:
        findings.append(_finding("missing_source_publication_writer_manifest_id", "error", "final-registry writer manifest lacks source publication writer manifest id"))
    if policy.require_source_publication_entry_id and not manifest.source_publication_entry_id:
        findings.append(_finding("missing_source_publication_entry_id", "error", "final-registry writer manifest lacks source publication entry id"))
    if policy.require_source_final_registry_preflight_status and not manifest.source_final_registry_preflight_status:
        findings.append(_finding("missing_source_final_registry_preflight_status", "error", "final-registry writer manifest lacks source final-registry preflight status"))
    if policy.require_valid_final_registry_preflight_status and manifest.source_final_registry_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_final_registry_preflight_status", "error", "source final-registry preflight status is not valid"))
    if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", manifest.metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", "final-registry writer manifest claims benchmark validation"))
    if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", manifest.metadata):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "final-registry writer manifest claims scientific validation"))
    if policy.require_no_global_registry_claims and _metadata_claims_global_registry(manifest.metadata):
        findings.append(_finding("global_registry_source_claim", "error", "final-registry writer manifest claims global registry mutation or publication"))
    return findings


def _artifact_summaries(
    manifest: OledPublicationCandidateFinalRegistryWriterManifest,
    entry: OledFinalRegistryCandidateEntry | None,
    index_records: list[OledFinalRegistryCandidateIndexRecord],
    policy: OledFinalRegistryGlobalAppendPreflightPolicy,
) -> list[OledFinalRegistryGlobalAppendArtifactSummary]:
    summaries: list[OledFinalRegistryGlobalAppendArtifactSummary] = []
    for artifact_kind, required, loaded, require_sha in (
        ("final_registry_candidate_entry_json", policy.require_final_registry_entry_json, entry is not None, policy.require_final_registry_entry_sha256),
        ("final_registry_candidate_index_jsonl", policy.require_final_registry_index_jsonl, bool(index_records), policy.require_final_registry_index_sha256),
    ):
        file_result = _file_result_for_kind(manifest, artifact_kind)
        reasons: set[str] = set()
        status = OledFinalRegistryGlobalAppendArtifactStatus.READY
        if loaded:
            reasons.add("artifact_loaded")
        elif required:
            reasons.add(_missing_artifact_reason(artifact_kind))
            status = OledFinalRegistryGlobalAppendArtifactStatus.FAILED
        else:
            reasons.add("artifact_optional")
            status = OledFinalRegistryGlobalAppendArtifactStatus.SKIPPED
        if require_sha and file_result is not None and not file_result.output_sha256:
            reasons.add(f"missing_{artifact_kind}_sha256")
            status = OledFinalRegistryGlobalAppendArtifactStatus.FAILED
        summaries.append(
            OledFinalRegistryGlobalAppendArtifactSummary(
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
    summaries: list[OledFinalRegistryGlobalAppendArtifactSummary],
) -> list[OledFinalRegistryGlobalAppendPreflightFinding]:
    findings: list[OledFinalRegistryGlobalAppendPreflightFinding] = []
    for summary in summaries:
        if summary.status != OledFinalRegistryGlobalAppendArtifactStatus.FAILED:
            continue
        for reason in summary.reason_codes:
            findings.append(
                _finding(
                    reason,
                    "error",
                    "final-registry global append artifact is not ready",
                    artifact_kind=summary.artifact_kind,
                    output_path=summary.output_path,
                )
            )
    return findings


def _entry_summaries(
    entry: OledFinalRegistryCandidateEntry | None,
    index_records: list[OledFinalRegistryCandidateIndexRecord],
    existing_records: list[OledFinalRegistryExistingRecordSummary],
    policy: OledFinalRegistryGlobalAppendPreflightPolicy,
) -> list[OledFinalRegistryGlobalAppendEntrySummary]:
    if entry is None:
        return []
    duplicate_entry_count, duplicate_source_chain_count = _existing_duplicate_counts(entry, existing_records)
    reasons: set[str] = {"entry_loaded"}
    status = OledFinalRegistryGlobalAppendArtifactStatus.READY
    matched_count = sum(1 for record in index_records if record.final_registry_entry_id == entry.final_registry_entry_id)
    if policy.require_final_registry_candidate_status and _status_value(entry.final_registry_status) != "final_registry_candidate":
        reasons.add("final_registry_status_not_candidate")
        status = OledFinalRegistryGlobalAppendArtifactStatus.FAILED
    if policy.require_entry_in_index and matched_count == 0:
        reasons.add("final_registry_entry_not_in_index")
        status = OledFinalRegistryGlobalAppendArtifactStatus.FAILED
    if policy.require_run_cards and entry.run_card_count <= 0:
        reasons.add("missing_run_cards")
        status = OledFinalRegistryGlobalAppendArtifactStatus.FAILED
    if policy.require_metric_cards and entry.metric_card_count <= 0:
        reasons.add("missing_metric_cards")
        status = OledFinalRegistryGlobalAppendArtifactStatus.FAILED
    if duplicate_entry_count and policy.fail_on_existing_registry_duplicate_entry_id:
        reasons.add("existing_registry_duplicate_entry_id")
        status = OledFinalRegistryGlobalAppendArtifactStatus.FAILED
    if duplicate_source_chain_count and policy.fail_on_existing_registry_duplicate_source_chain:
        reasons.add("existing_registry_duplicate_source_chain")
        status = OledFinalRegistryGlobalAppendArtifactStatus.FAILED
    return [
        OledFinalRegistryGlobalAppendEntrySummary(
            final_registry_entry_id=entry.final_registry_entry_id,
            final_registry_status=_status_value(entry.final_registry_status),
            source_publication_entry_id=entry.source_publication_entry_id,
            source_publication_writer_manifest_id=entry.source_publication_writer_manifest_id,
            source_final_registry_preflight_status=entry.source_final_registry_preflight_status,
            source_promoted_entry_id=entry.source_promoted_entry_id,
            source_registry_entry_id=entry.source_registry_entry_id,
            source_candidate_report_id=entry.source_candidate_report_id,
            source_benchmark_report_manifest_id=entry.source_benchmark_report_manifest_id,
            baseline_kinds=_selected_values(entry.baseline_kinds, policy.baseline_kinds),
            target_property_ids=_selected_values(entry.target_property_ids, policy.target_property_ids),
            feature_views=_selected_values(entry.feature_views, policy.feature_views),
            run_card_count=entry.run_card_count,
            metric_card_count=entry.metric_card_count,
            final_registry_index_record_count=len(index_records),
            matched_final_registry_index_record_count=matched_count,
            existing_registry_duplicate_entry_count=duplicate_entry_count,
            existing_registry_duplicate_source_chain_count=duplicate_source_chain_count,
            artifact_status=status,
            reason_codes=sorted(reasons),
        )
    ]


def _entry_findings(
    entry: OledFinalRegistryCandidateEntry | None,
    index_records: list[OledFinalRegistryCandidateIndexRecord],
    policy: OledFinalRegistryGlobalAppendPreflightPolicy,
) -> list[OledFinalRegistryGlobalAppendPreflightFinding]:
    findings: list[OledFinalRegistryGlobalAppendPreflightFinding] = []
    if entry is None:
        if policy.require_final_registry_entry_json:
            findings.append(_finding("missing_final_registry_candidate_entry_json", "error", "final-registry candidate entry JSON is required", artifact_kind="final_registry_candidate_entry_json"))
        return findings
    if policy.require_final_registry_candidate_status and _status_value(entry.final_registry_status) != "final_registry_candidate":
        findings.append(_finding("final_registry_status_not_candidate", "error", "final-registry candidate entry status is not final_registry_candidate", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_source_publication_writer_manifest_id and not entry.source_publication_writer_manifest_id:
        findings.append(_finding("missing_source_publication_writer_manifest_id", "error", "final-registry entry lacks source publication writer manifest id", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_source_publication_entry_id and not entry.source_publication_entry_id:
        findings.append(_finding("missing_source_publication_entry_id", "error", "final-registry entry lacks source publication entry id", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_source_final_registry_preflight_status and not entry.source_final_registry_preflight_status:
        findings.append(_finding("missing_source_final_registry_preflight_status", "error", "final-registry entry lacks source final-registry preflight status", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_source_promoted_entry_id and not entry.source_promoted_entry_id:
        findings.append(_finding("missing_source_promoted_entry_id", "error", "final-registry entry lacks source promoted entry id", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_source_promotion_writer_manifest_id and not entry.source_promotion_writer_manifest_id:
        findings.append(_finding("missing_source_promotion_writer_manifest_id", "error", "final-registry entry lacks source promotion writer manifest id", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_source_registry_entry_id and not entry.source_registry_entry_id:
        findings.append(_finding("missing_source_registry_entry_id", "error", "final-registry entry lacks source registry entry id", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_source_registry_writer_manifest_id and not entry.source_registry_writer_manifest_id:
        findings.append(_finding("missing_source_registry_writer_manifest_id", "error", "final-registry entry lacks source registry writer manifest id", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_source_candidate_report_id and not entry.source_candidate_report_id:
        findings.append(_finding("missing_source_candidate_report_id", "error", "final-registry entry lacks source candidate report id", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_source_benchmark_report_manifest_id and not entry.source_benchmark_report_manifest_id:
        findings.append(_finding("missing_source_benchmark_report_manifest_id", "error", "final-registry entry lacks source benchmark report manifest id", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_valid_final_registry_preflight_status and entry.source_final_registry_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_final_registry_preflight_status", "error", "final-registry entry source final-registry preflight status is not valid", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_caveats:
        caveats = set(entry.caveats)
        for caveat in policy.required_caveats:
            if caveat not in caveats:
                findings.append(_finding("missing_required_caveat", "error", "final-registry entry lacks required caveat", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_run_cards and entry.run_card_count <= 0:
        findings.append(_finding("missing_run_cards", "error", "final-registry entry has no run cards", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_metric_cards and entry.metric_card_count <= 0:
        findings.append(_finding("missing_metric_cards", "error", "final-registry entry has no metric cards", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", entry.metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", "final-registry entry claims benchmark validation", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", entry.metadata):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "final-registry entry claims scientific validation", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_no_global_registry_claims and _metadata_claims_global_registry(entry.metadata):
        findings.append(_finding("global_registry_source_claim", "error", "final-registry entry claims global registry mutation or publication", final_registry_entry_id=entry.final_registry_entry_id))
    if _contains_forbidden_payload_key(entry.model_dump(mode="json"), raw_only=True):
        findings.append(_finding("raw_prediction_payload_leaked", "error", "final-registry entry contains raw prediction payload", final_registry_entry_id=entry.final_registry_entry_id))
    if _contains_feature_payload_key(entry.model_dump(mode="json")):
        findings.append(_finding("raw_feature_payload_leaked", "error", "final-registry entry contains feature payload", final_registry_entry_id=entry.final_registry_entry_id))
    if _contains_absolute_path(entry.model_dump(mode="json")):
        findings.append(_finding("absolute_path_leakage", "error", "final-registry entry contains absolute path", final_registry_entry_id=entry.final_registry_entry_id))
    if policy.require_entry_in_index and not any(record.final_registry_entry_id == entry.final_registry_entry_id for record in index_records):
        findings.append(_finding("final_registry_entry_not_in_index", "error", "final-registry entry is not referenced by index", final_registry_entry_id=entry.final_registry_entry_id))
    return findings


def _index_findings(
    entry: OledFinalRegistryCandidateEntry | None,
    index_records: list[OledFinalRegistryCandidateIndexRecord],
    policy: OledFinalRegistryGlobalAppendPreflightPolicy,
) -> list[OledFinalRegistryGlobalAppendPreflightFinding]:
    findings: list[OledFinalRegistryGlobalAppendPreflightFinding] = []
    if not index_records:
        if policy.require_final_registry_index_jsonl:
            findings.append(_finding("missing_final_registry_candidate_index_jsonl", "error", "final-registry candidate index JSONL is required", artifact_kind="final_registry_candidate_index_jsonl"))
        return findings
    if policy.require_single_final_registry_index_record and len(index_records) > 1:
        findings.append(_finding("multiple_final_registry_index_records", "error", "final-registry candidate index has multiple records", artifact_kind="final_registry_candidate_index_jsonl"))
    for record in index_records:
        if policy.require_final_registry_candidate_status and record.final_registry_status != "final_registry_candidate":
            findings.append(_finding("index_status_not_final_registry_candidate", "error", "final-registry index status is not final_registry_candidate", final_registry_entry_id=record.final_registry_entry_id))
        if policy.require_no_benchmark_validated_claims and (record.benchmark_validated or _truthy_metadata_key("benchmark_validated", record.metadata)):
            findings.append(_finding("benchmark_validated_source_claim", "error", "final-registry index claims benchmark validation", final_registry_entry_id=record.final_registry_entry_id))
        if policy.require_no_scientific_claims and (record.scientific_claim_validated or _truthy_metadata_key("scientific_claim_validated", record.metadata)):
            findings.append(_finding("scientific_claim_validated_source_claim", "error", "final-registry index claims scientific validation", final_registry_entry_id=record.final_registry_entry_id))
        if policy.require_no_global_registry_claims and (
            record.benchmark_published or record.benchmark_registered or _metadata_claims_global_registry(record.metadata)
        ):
            findings.append(_finding("global_registry_source_claim", "error", "final-registry index claims global registry mutation or publication", final_registry_entry_id=record.final_registry_entry_id))
        if _contains_forbidden_payload_key(record.model_dump(mode="json"), raw_only=True):
            findings.append(_finding("raw_prediction_payload_leaked", "error", "final-registry index contains raw prediction payload", final_registry_entry_id=record.final_registry_entry_id))
        if _contains_feature_payload_key(record.model_dump(mode="json")):
            findings.append(_finding("raw_feature_payload_leaked", "error", "final-registry index contains feature payload", final_registry_entry_id=record.final_registry_entry_id))
        if _contains_absolute_path(record.model_dump(mode="json")):
            findings.append(_finding("absolute_path_leakage", "error", "final-registry index contains absolute path", final_registry_entry_id=record.final_registry_entry_id))
    if entry is not None and policy.require_entry_in_index and not any(record.final_registry_entry_id == entry.final_registry_entry_id for record in index_records):
        findings.append(_finding("final_registry_entry_not_in_index", "error", "final-registry entry id is absent from index", final_registry_entry_id=entry.final_registry_entry_id))
    return findings


def _existing_registry_findings(
    entry: OledFinalRegistryCandidateEntry | None,
    existing_records: list[OledFinalRegistryExistingRecordSummary],
    policy: OledFinalRegistryGlobalAppendPreflightPolicy,
) -> list[OledFinalRegistryGlobalAppendPreflightFinding]:
    if entry is None:
        return []
    duplicate_entry_count, duplicate_source_chain_count = _existing_duplicate_counts(entry, existing_records)
    findings: list[OledFinalRegistryGlobalAppendPreflightFinding] = []
    if duplicate_entry_count and policy.fail_on_existing_registry_duplicate_entry_id:
        findings.append(
            _finding(
                "existing_registry_duplicate_entry_id",
                "error",
                "existing registry snapshot already contains this final-registry entry id",
                final_registry_entry_id=entry.final_registry_entry_id,
            )
        )
    if duplicate_source_chain_count and policy.fail_on_existing_registry_duplicate_source_chain:
        findings.append(
            _finding(
                "existing_registry_duplicate_source_chain",
                "error",
                "existing registry snapshot already contains this source chain",
                final_registry_entry_id=entry.final_registry_entry_id,
                source_publication_entry_id=entry.source_publication_entry_id,
            )
        )
    return findings


def _existing_duplicate_counts(
    entry: OledFinalRegistryCandidateEntry,
    existing_records: list[OledFinalRegistryExistingRecordSummary],
) -> tuple[int, int]:
    duplicate_entry_count = sum(1 for record in existing_records if record.registry_entry_id == entry.final_registry_entry_id)
    duplicate_source_chain_count = 0
    for record in existing_records:
        same_publication_entry = bool(entry.source_publication_entry_id and record.source_publication_entry_id == entry.source_publication_entry_id)
        same_report_pair = bool(
            entry.source_candidate_report_id
            and entry.source_benchmark_report_manifest_id
            and record.source_candidate_report_id == entry.source_candidate_report_id
            and record.source_benchmark_report_manifest_id == entry.source_benchmark_report_manifest_id
        )
        if same_publication_entry or same_report_pair:
            duplicate_source_chain_count += 1
    return duplicate_entry_count, duplicate_source_chain_count


def _file_result_for_kind(manifest: OledPublicationCandidateFinalRegistryWriterManifest, artifact_kind: str) -> Any | None:
    for file_result in manifest.file_results:
        if file_result.artifact_kind == artifact_kind:
            return file_result
    return None


def _missing_artifact_reason(artifact_kind: str) -> str:
    if artifact_kind == "final_registry_candidate_entry_json":
        return "missing_final_registry_candidate_entry_json"
    if artifact_kind == "final_registry_candidate_index_jsonl":
        return "missing_final_registry_candidate_index_jsonl"
    return f"missing_{artifact_kind}"


def _selected_values(values: Iterable[str], selected: Iterable[str]) -> list[str]:
    selected_set = {str(item).strip() for item in selected if str(item).strip()}
    return sorted({str(item) for item in values if not selected_set or str(item) in selected_set})


def _report_status(
    findings: list[OledFinalRegistryGlobalAppendPreflightFinding],
) -> OledFinalRegistryGlobalAppendPreflightStatus:
    if any(finding.severity == "error" for finding in findings):
        return OledFinalRegistryGlobalAppendPreflightStatus.FAILED
    if any(finding.severity == "warning" for finding in findings):
        return OledFinalRegistryGlobalAppendPreflightStatus.PASSED_WITH_WARNINGS
    return OledFinalRegistryGlobalAppendPreflightStatus.PASSED


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    artifact_kind: str | None = None,
    final_registry_entry_id: str | None = None,
    source_publication_entry_id: str | None = None,
    source_promoted_entry_id: str | None = None,
    source_registry_entry_id: str | None = None,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    output_path: str | None = None,
) -> OledFinalRegistryGlobalAppendPreflightFinding:
    return OledFinalRegistryGlobalAppendPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        artifact_kind=artifact_kind,
        final_registry_entry_id=final_registry_entry_id,
        source_publication_entry_id=source_publication_entry_id,
        source_promoted_entry_id=source_promoted_entry_id,
        source_registry_entry_id=source_registry_entry_id,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        output_path=output_path,
    )


def _dedup_findings(
    findings: list[OledFinalRegistryGlobalAppendPreflightFinding],
) -> list[OledFinalRegistryGlobalAppendPreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str, str, str, str, str]] = set()
    output: list[OledFinalRegistryGlobalAppendPreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.artifact_kind or "",
            finding.final_registry_entry_id or "",
            finding.source_publication_entry_id or "",
            finding.source_promoted_entry_id or "",
            finding.source_registry_entry_id or "",
            finding.baseline_kind or "",
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
            item.final_registry_entry_id or "",
            item.source_publication_entry_id or "",
            item.source_promoted_entry_id or "",
            item.source_registry_entry_id or "",
        ),
    )


def _existing_record_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "registry_entry_id": payload.get("registry_entry_id") or payload.get("final_registry_entry_id"),
        "registry_status": payload.get("registry_status") or payload.get("final_registry_status"),
        "source_publication_entry_id": payload.get("source_publication_entry_id"),
        "source_publication_writer_manifest_id": payload.get("source_publication_writer_manifest_id"),
        "source_candidate_report_id": payload.get("source_candidate_report_id"),
        "source_benchmark_report_manifest_id": payload.get("source_benchmark_report_manifest_id"),
        "metadata": payload.get("metadata") or {},
    }


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


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _truthy_metadata_key(key: str, metadata: dict[str, Any]) -> bool:
    return bool(metadata.get(key))


def _metadata_claims_global_registry(metadata: dict[str, Any]) -> bool:
    return any(
        bool(metadata.get(key))
        for key in (
            "benchmark_published",
            "benchmark_registered",
            "globally_registered",
            "global_registry_mutated",
            "final_registry_written",
            "published",
        )
    )


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute()
    return False


def _contains_forbidden_payload_key(value: Any, *, raw_only: bool = False) -> bool:
    forbidden_keys = _RAW_PAYLOAD_JSON_KEYS if raw_only else _FORBIDDEN_JSON_KEYS
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in forbidden_keys:
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
        "final_registry_global_append_preflight_only": True,
        "global_registry_mutated": False,
        "final_registry_written": False,
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


_MAX_OUTPUT_STRING_LENGTH = 240

_RAW_PAYLOAD_JSON_KEYS = {
    "raw_text",
    "full_text",
    "prediction_id",
    "training_row_id",
}

_FORBIDDEN_JSON_KEYS = {
    *_RAW_PAYLOAD_JSON_KEYS,
    "features",
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
    "OledFinalRegistryGlobalAppendPreflightStatus",
    "OledFinalRegistryGlobalAppendArtifactStatus",
    "OledFinalRegistryGlobalAppendPreflightPolicy",
    "OledFinalRegistryGlobalAppendArtifactSummary",
    "OledFinalRegistryGlobalAppendEntrySummary",
    "OledFinalRegistryExistingRecordSummary",
    "OledFinalRegistryGlobalAppendPreflightFinding",
    "OledFinalRegistryGlobalAppendPreflightReport",
    "load_oled_publication_candidate_final_registry_writer_manifest_json",
    "load_oled_final_registry_candidate_artifacts_from_manifest",
    "load_oled_existing_final_registry_snapshot_jsonl",
    "run_oled_final_registry_global_append_preflight",
    "run_oled_final_registry_global_append_preflight_from_files",
    "write_oled_final_registry_global_append_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
