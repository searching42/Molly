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

from ai4s_agent.domains.oled_curated_global_append_release_writer import (
    OledGlobalAppendReleaseWriteStatus,
    OledGlobalAppendReleaseWriterManifest,
    OledReleaseCandidateDeltaRecord,
    OledReleaseCandidateEntry,
    OledReleaseCandidateEntryStatus,
    load_oled_release_candidate_delta_jsonl,
    load_oled_release_candidate_entry_json,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledReleaseCandidateExternalPublicationPreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledReleaseCandidateExternalPublicationArtifactStatus(str, Enum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    FAILED = "failed"
    SKIPPED = "skipped"


class OledReleaseCandidateExternalPublicationPreflightPolicy(BaseModel):
    require_release_writer_manifest_sha256: bool = True
    require_release_entry_sha256: bool = True
    require_release_delta_sha256: bool = True
    require_release_snapshot_sha256: bool = True

    require_release_entry_json: bool = True
    require_release_delta_jsonl: bool = True
    require_release_snapshot_jsonl: bool = True

    require_release_candidate_status: bool = True
    require_entry_in_delta: bool = True
    require_entry_in_snapshot: bool = True
    require_delta_records_in_snapshot: bool = True
    require_prior_snapshot_preserved: bool = True
    require_single_release_delta_record: bool = True

    require_source_global_append_writer_manifest_id: bool = True
    require_source_global_append_entry_id: bool = True
    require_source_release_preflight_status: bool = True

    require_source_final_registry_entry_id: bool = True
    require_source_final_registry_writer_manifest_id: bool = True
    require_source_publication_entry_id: bool = True
    require_source_publication_writer_manifest_id: bool = True
    require_source_promoted_entry_id: bool = True
    require_source_promotion_writer_manifest_id: bool = True
    require_source_registry_entry_id: bool = True
    require_source_registry_writer_manifest_id: bool = True
    require_source_candidate_report_id: bool = True
    require_source_benchmark_report_manifest_id: bool = True

    require_valid_release_preflight_status: bool = True
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


class OledReleaseCandidateExternalPublicationArtifactSummary(BaseModel):
    artifact_kind: str
    status: OledReleaseCandidateExternalPublicationArtifactStatus
    output_path: str | None = None
    output_sha256: str | None = None
    loaded: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class OledReleaseCandidateSnapshotRecordSummary(BaseModel):
    registry_entry_id: str | None = None
    release_entry_id: str | None = None
    release_status: str | None = None

    source_global_append_entry_id: str | None = None
    source_global_append_writer_manifest_id: str | None = None
    source_final_registry_entry_id: str | None = None
    source_publication_entry_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class OledReleaseCandidateExternalPublicationEntrySummary(BaseModel):
    release_entry_id: str
    release_status: str

    source_global_append_entry_id: str | None = None
    source_global_append_writer_manifest_id: str | None = None
    source_release_preflight_status: str | None = None

    source_final_registry_entry_id: str | None = None
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

    release_delta_record_count: int = 0
    matched_release_delta_record_count: int = 0

    release_snapshot_record_count: int = 0
    matched_release_snapshot_record_count: int = 0
    prior_snapshot_record_count: int = 0
    preserved_prior_snapshot_record_count: int = 0

    artifact_status: OledReleaseCandidateExternalPublicationArtifactStatus
    reason_codes: list[str] = Field(default_factory=list)


class OledReleaseCandidateExternalPublicationPreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    artifact_kind: str | None = None
    release_entry_id: str | None = None
    source_global_append_entry_id: str | None = None
    source_final_registry_entry_id: str | None = None
    source_publication_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    output_path: str | None = None


class OledReleaseCandidateExternalPublicationPreflightReport(BaseModel):
    status: OledReleaseCandidateExternalPublicationPreflightStatus

    source_release_writer_manifest_id: str | None = None
    source_release_entry_id: str | None = None
    source_global_append_entry_id: str | None = None
    source_final_registry_entry_id: str | None = None
    source_publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None
    source_release_preflight_status: str | None = None

    input_release_entry_count: int = 0
    input_release_delta_record_count: int = 0
    input_release_snapshot_record_count: int = 0
    input_prior_registry_snapshot_record_count: int = 0

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    artifact_summaries: list[OledReleaseCandidateExternalPublicationArtifactSummary] = Field(default_factory=list)
    entry_summaries: list[OledReleaseCandidateExternalPublicationEntrySummary] = Field(default_factory=list)

    caveats: list[str] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledReleaseCandidateExternalPublicationPreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_global_append_release_writer_manifest_json(
    path: str | Path,
) -> OledGlobalAppendReleaseWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_global_append_release_writer_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledGlobalAppendReleaseWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"invalid_global_append_release_writer_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}"
        ) from exc
    if _contains_absolute_path(manifest.model_dump(mode="json")):
        raise ValueError("absolute_path_in_global_append_release_writer_manifest")
    if _contains_forbidden_payload_key(manifest.model_dump(mode="json")):
        raise ValueError("raw_payload_in_global_append_release_writer_manifest")
    return manifest


def load_oled_release_candidate_snapshot_jsonl(
    path: str | Path,
) -> list[OledReleaseCandidateSnapshotRecordSummary]:
    snapshot_path = Path(path)
    _reject_forbidden_input(snapshot_path)
    if not snapshot_path.exists():
        raise ValueError(f"missing_release_candidate_snapshot_jsonl:{redact_oled_mineru_acceptance_path(snapshot_path)}")
    records: list[OledReleaseCandidateSnapshotRecordSummary] = []
    for line_number, line in enumerate(snapshot_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("release candidate snapshot line is not an object")
            if _contains_forbidden_payload_key(payload):
                raise ValueError("release candidate snapshot payload leaked raw data")
            if _contains_absolute_path(payload):
                raise ValueError("release candidate snapshot payload leaked absolute path")
            records.append(_snapshot_record_from_payload(payload))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"invalid_release_candidate_snapshot_jsonl:line-{line_number}") from exc
    return records


def load_oled_release_candidate_artifacts_from_manifest(
    *,
    manifest: OledGlobalAppendReleaseWriterManifest,
    base_dir: str | Path,
) -> tuple[
    OledReleaseCandidateEntry | None,
    list[OledReleaseCandidateDeltaRecord],
    list[OledReleaseCandidateSnapshotRecordSummary],
]:
    entry: OledReleaseCandidateEntry | None = None
    delta_records: list[OledReleaseCandidateDeltaRecord] = []
    snapshot_records: list[OledReleaseCandidateSnapshotRecordSummary] = []
    for file_result in manifest.file_results:
        if _status_value(file_result.status) != OledGlobalAppendReleaseWriteStatus.WRITTEN.value:
            continue
        if not file_result.output_path:
            continue
        path = _resolve_manifest_path(file_result.output_path, base_dir)
        if file_result.artifact_kind == "release_candidate_entry_json":
            if not path.exists():
                raise ValueError(f"missing_release_candidate_entry_json:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"release_candidate_entry_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            entry = load_oled_release_candidate_entry_json(path)
        elif file_result.artifact_kind == "release_candidate_delta_jsonl":
            if not path.exists():
                raise ValueError(f"missing_release_candidate_delta_jsonl:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"release_candidate_delta_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            delta_records = load_oled_release_candidate_delta_jsonl(path)
        elif file_result.artifact_kind == "release_candidate_snapshot_jsonl":
            if not path.exists():
                raise ValueError(f"missing_release_candidate_snapshot_jsonl:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"release_candidate_snapshot_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            snapshot_records = load_oled_release_candidate_snapshot_jsonl(path)
    return entry, delta_records, snapshot_records


def run_oled_release_candidate_external_publication_preflight(
    *,
    release_writer_manifest: OledGlobalAppendReleaseWriterManifest,
    release_entry: OledReleaseCandidateEntry | None,
    release_delta_records: Iterable[OledReleaseCandidateDeltaRecord],
    release_snapshot_records: Iterable[OledReleaseCandidateSnapshotRecordSummary],
    prior_registry_snapshot_records: Iterable[OledReleaseCandidateSnapshotRecordSummary] | None = None,
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy | None = None,
) -> OledReleaseCandidateExternalPublicationPreflightReport:
    preflight_policy = policy or OledReleaseCandidateExternalPublicationPreflightPolicy()
    delta_records = list(release_delta_records)
    snapshot_records = list(release_snapshot_records)
    prior_records = list(prior_registry_snapshot_records or [])

    artifact_summaries = _artifact_summaries(
        release_writer_manifest,
        release_entry,
        delta_records,
        snapshot_records,
        preflight_policy,
    )
    entry_summaries = _entry_summaries(release_entry, delta_records, snapshot_records, prior_records, preflight_policy)

    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding] = []
    findings.extend(_manifest_findings(release_writer_manifest, preflight_policy))
    findings.extend(_artifact_findings(artifact_summaries))
    findings.extend(_entry_findings(release_entry, delta_records, snapshot_records, preflight_policy))
    findings.extend(_delta_findings(release_entry, delta_records, snapshot_records, preflight_policy))
    findings.extend(_snapshot_findings(snapshot_records, prior_records, preflight_policy))
    findings = _dedup_findings(findings)

    status = _report_status(findings)
    status_counts = Counter(_status_value(summary.status) for summary in artifact_summaries)
    status_counts.update(_status_value(summary.artifact_status) for summary in entry_summaries)

    return OledReleaseCandidateExternalPublicationPreflightReport(
        status=status,
        source_release_writer_manifest_id=release_writer_manifest.manifest_id,
        source_release_entry_id=release_entry.release_entry_id if release_entry is not None else None,
        source_global_append_entry_id=(
            release_entry.source_global_append_entry_id
            if release_entry is not None
            else release_writer_manifest.source_global_append_entry_id
        ),
        source_final_registry_entry_id=release_entry.source_final_registry_entry_id if release_entry is not None else None,
        source_publication_entry_id=release_entry.source_publication_entry_id if release_entry is not None else None,
        source_promoted_entry_id=release_entry.source_promoted_entry_id if release_entry is not None else None,
        source_registry_entry_id=release_entry.source_registry_entry_id if release_entry is not None else None,
        source_candidate_report_id=release_entry.source_candidate_report_id if release_entry is not None else None,
        source_benchmark_report_manifest_id=release_entry.source_benchmark_report_manifest_id if release_entry is not None else None,
        source_release_preflight_status=(
            release_entry.source_release_preflight_status
            if release_entry is not None
            else release_writer_manifest.source_release_preflight_status
        ),
        input_release_entry_count=1 if release_entry is not None else 0,
        input_release_delta_record_count=len(delta_records),
        input_release_snapshot_record_count=len(snapshot_records),
        input_prior_registry_snapshot_record_count=len(prior_records),
        baseline_kinds=_selected_values(
            release_entry.baseline_kinds if release_entry is not None else release_writer_manifest.baseline_kinds,
            preflight_policy.baseline_kinds,
        ),
        target_property_ids=_selected_values(
            release_entry.target_property_ids if release_entry is not None else release_writer_manifest.target_property_ids,
            preflight_policy.target_property_ids,
        ),
        feature_views=_selected_values(
            release_entry.feature_views if release_entry is not None else release_writer_manifest.feature_views,
            preflight_policy.feature_views,
        ),
        artifact_summaries=artifact_summaries,
        entry_summaries=entry_summaries,
        caveats=sorted(release_entry.caveats) if release_entry is not None else [],
        status_counts=dict(sorted(status_counts.items())),
        finding_code_counts=dict(sorted(Counter(finding.code for finding in findings).items())),
        findings=findings,
        metadata=_safety_metadata(),
    )


def run_oled_release_candidate_external_publication_preflight_from_files(
    *,
    release_writer_manifest_path: str | Path,
    release_candidate_base_dir: str | Path | None = None,
    prior_registry_snapshot_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy | None = None,
) -> OledReleaseCandidateExternalPublicationPreflightReport:
    manifest = load_oled_global_append_release_writer_manifest_json(release_writer_manifest_path)
    base_dir = Path(release_candidate_base_dir) if release_candidate_base_dir is not None else Path(release_writer_manifest_path).parent
    entry, delta_records, snapshot_records = load_oled_release_candidate_artifacts_from_manifest(manifest=manifest, base_dir=base_dir)
    prior_records = (
        load_oled_release_candidate_snapshot_jsonl(prior_registry_snapshot_path)
        if prior_registry_snapshot_path is not None
        else None
    )
    report = run_oled_release_candidate_external_publication_preflight(
        release_writer_manifest=manifest,
        release_entry=entry,
        release_delta_records=delta_records,
        release_snapshot_records=snapshot_records,
        prior_registry_snapshot_records=prior_records,
        policy=policy,
    )
    if output_report_path is not None:
        write_oled_release_candidate_external_publication_preflight_report_json(report, output_report_path)
    return report


def write_oled_release_candidate_external_publication_preflight_report_json(
    report: OledReleaseCandidateExternalPublicationPreflightReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(report.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run read-only OLED release-candidate external-publication-readiness preflight."
    )
    parser.add_argument("--release-writer-manifest", required=True, help="Path to release writer manifest JSON.")
    parser.add_argument("--release-candidate-base-dir", help="Base directory for release candidate artifacts.")
    parser.add_argument("--prior-registry-snapshot", help="Optional prior registry snapshot JSONL.")
    parser.add_argument("--output-report", help="Optional external-publication-readiness preflight report JSON path.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--allow-multiple-release-delta-records", action="store_true", help="Allow multiple release delta records.")
    parser.add_argument(
        "--allow-missing-prior-snapshot-preservation-check",
        action="store_true",
        help="Skip prior snapshot prefix preservation checks.",
    )
    args = parser.parse_args(argv)
    try:
        policy = OledReleaseCandidateExternalPublicationPreflightPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            require_single_release_delta_record=not args.allow_multiple_release_delta_records,
            require_prior_snapshot_preserved=not args.allow_missing_prior_snapshot_preservation_check,
        )
        report = run_oled_release_candidate_external_publication_preflight_from_files(
            release_writer_manifest_path=args.release_writer_manifest,
            release_candidate_base_dir=args.release_candidate_base_dir,
            prior_registry_snapshot_path=args.prior_registry_snapshot,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": _status_value(report.status),
            "entry_summary_count": len(report.entry_summaries),
            "artifact_summary_count": len(report.artifact_summaries),
            "delta_record_count": report.input_release_delta_record_count,
            "snapshot_record_count": report.input_release_snapshot_record_count,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _manifest_findings(
    manifest: OledGlobalAppendReleaseWriterManifest,
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy,
) -> list[OledReleaseCandidateExternalPublicationPreflightFinding]:
    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding] = []
    if policy.require_source_global_append_writer_manifest_id and not manifest.source_global_append_writer_manifest_id:
        findings.append(
            _finding(
                "missing_source_global_append_writer_manifest_id",
                "error",
                "release writer manifest lacks source global-append writer manifest id",
            )
        )
    if policy.require_source_global_append_entry_id and not manifest.source_global_append_entry_id:
        findings.append(_finding("missing_source_global_append_entry_id", "error", "release writer manifest lacks source global-append entry id"))
    if policy.require_source_release_preflight_status and not manifest.source_release_preflight_status:
        findings.append(_finding("missing_source_release_preflight_status", "error", "release writer manifest lacks source release preflight status"))
    if policy.require_valid_release_preflight_status and manifest.source_release_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_release_preflight_status", "error", "source release preflight status is not valid"))
    findings.extend(_source_claim_findings(manifest.metadata, policy, "release writer manifest"))
    return findings


def _artifact_summaries(
    manifest: OledGlobalAppendReleaseWriterManifest,
    entry: OledReleaseCandidateEntry | None,
    delta_records: list[OledReleaseCandidateDeltaRecord],
    snapshot_records: list[OledReleaseCandidateSnapshotRecordSummary],
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy,
) -> list[OledReleaseCandidateExternalPublicationArtifactSummary]:
    summaries: list[OledReleaseCandidateExternalPublicationArtifactSummary] = []
    for artifact_kind, required, loaded, require_sha in (
        (
            "release_candidate_entry_json",
            policy.require_release_entry_json,
            entry is not None,
            policy.require_release_entry_sha256,
        ),
        (
            "release_candidate_delta_jsonl",
            policy.require_release_delta_jsonl,
            bool(delta_records),
            policy.require_release_delta_sha256,
        ),
        (
            "release_candidate_snapshot_jsonl",
            policy.require_release_snapshot_jsonl,
            bool(snapshot_records),
            policy.require_release_snapshot_sha256,
        ),
    ):
        file_result = _file_result_for_kind(manifest, artifact_kind)
        reasons: set[str] = set()
        status = OledReleaseCandidateExternalPublicationArtifactStatus.READY
        if loaded:
            reasons.add("artifact_loaded")
        elif required:
            reasons.add(_missing_artifact_reason(artifact_kind))
            status = OledReleaseCandidateExternalPublicationArtifactStatus.FAILED
        else:
            reasons.add("artifact_optional")
            status = OledReleaseCandidateExternalPublicationArtifactStatus.SKIPPED
        if require_sha and file_result is not None and not file_result.output_sha256:
            reasons.add(f"missing_{artifact_kind}_sha256")
            status = OledReleaseCandidateExternalPublicationArtifactStatus.FAILED
        summaries.append(
            OledReleaseCandidateExternalPublicationArtifactSummary(
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
    summaries: list[OledReleaseCandidateExternalPublicationArtifactSummary],
) -> list[OledReleaseCandidateExternalPublicationPreflightFinding]:
    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding] = []
    for summary in summaries:
        if summary.status != OledReleaseCandidateExternalPublicationArtifactStatus.FAILED:
            continue
        for reason in summary.reason_codes:
            findings.append(
                _finding(
                    reason,
                    "error",
                    "release candidate external-publication artifact is not ready",
                    artifact_kind=summary.artifact_kind,
                    output_path=summary.output_path,
                )
            )
    return findings


def _entry_summaries(
    entry: OledReleaseCandidateEntry | None,
    delta_records: list[OledReleaseCandidateDeltaRecord],
    snapshot_records: list[OledReleaseCandidateSnapshotRecordSummary],
    prior_records: list[OledReleaseCandidateSnapshotRecordSummary],
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy,
) -> list[OledReleaseCandidateExternalPublicationEntrySummary]:
    if entry is None:
        return []
    matched_delta_count = sum(1 for record in delta_records if record.release_entry_id == entry.release_entry_id)
    matched_snapshot_count = sum(1 for record in snapshot_records if _snapshot_record_matches_entry(record, entry))
    preserved_prior_count = _preserved_prior_count(prior_records, snapshot_records)
    reasons: set[str] = {"entry_loaded"}
    status = OledReleaseCandidateExternalPublicationArtifactStatus.READY
    for code, failed in (
        (
            "release_status_not_candidate",
            policy.require_release_candidate_status
            and _status_value(entry.release_status) != OledReleaseCandidateEntryStatus.RELEASE_CANDIDATE.value,
        ),
        ("release_entry_not_in_delta", policy.require_entry_in_delta and matched_delta_count == 0),
        ("release_entry_not_in_snapshot", policy.require_entry_in_snapshot and matched_snapshot_count == 0),
        ("missing_run_cards", policy.require_run_cards and entry.run_card_count <= 0),
        ("missing_metric_cards", policy.require_metric_cards and entry.metric_card_count <= 0),
        (
            "prior_snapshot_not_preserved",
            bool(prior_records) and policy.require_prior_snapshot_preserved and preserved_prior_count != len(prior_records),
        ),
    ):
        if failed:
            reasons.add(code)
            status = OledReleaseCandidateExternalPublicationArtifactStatus.FAILED
    return [
        OledReleaseCandidateExternalPublicationEntrySummary(
            release_entry_id=entry.release_entry_id,
            release_status=_status_value(entry.release_status),
            source_global_append_entry_id=entry.source_global_append_entry_id,
            source_global_append_writer_manifest_id=entry.source_global_append_writer_manifest_id,
            source_release_preflight_status=entry.source_release_preflight_status,
            source_final_registry_entry_id=entry.source_final_registry_entry_id,
            source_publication_entry_id=entry.source_publication_entry_id,
            source_promoted_entry_id=entry.source_promoted_entry_id,
            source_registry_entry_id=entry.source_registry_entry_id,
            source_candidate_report_id=entry.source_candidate_report_id,
            source_benchmark_report_manifest_id=entry.source_benchmark_report_manifest_id,
            baseline_kinds=list(entry.baseline_kinds),
            target_property_ids=list(entry.target_property_ids),
            feature_views=list(entry.feature_views),
            run_card_count=entry.run_card_count,
            metric_card_count=entry.metric_card_count,
            release_delta_record_count=len(delta_records),
            matched_release_delta_record_count=matched_delta_count,
            release_snapshot_record_count=len(snapshot_records),
            matched_release_snapshot_record_count=matched_snapshot_count,
            prior_snapshot_record_count=len(prior_records),
            preserved_prior_snapshot_record_count=preserved_prior_count,
            artifact_status=status,
            reason_codes=sorted(reasons),
        )
    ]


def _entry_findings(
    entry: OledReleaseCandidateEntry | None,
    delta_records: list[OledReleaseCandidateDeltaRecord],
    snapshot_records: list[OledReleaseCandidateSnapshotRecordSummary],
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy,
) -> list[OledReleaseCandidateExternalPublicationPreflightFinding]:
    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding] = []
    if entry is None:
        if policy.require_release_entry_json:
            findings.append(_finding("missing_release_candidate_entry_json", "error", "release candidate entry is required"))
        return findings
    if policy.require_release_candidate_status and _status_value(entry.release_status) != OledReleaseCandidateEntryStatus.RELEASE_CANDIDATE.value:
        findings.append(
            _finding(
                "release_status_not_candidate",
                "error",
                "release entry status is not release_candidate",
                release_entry_id=entry.release_entry_id,
            )
        )
    _required_entry_source_findings(entry, policy, findings)
    if policy.require_caveats:
        caveats = set(entry.caveats)
        for caveat in policy.required_caveats:
            if caveat not in caveats:
                findings.append(
                    _finding(
                        "missing_required_caveat",
                        "error",
                        "release entry lacks required caveat",
                        release_entry_id=entry.release_entry_id,
                    )
                )
    if policy.require_run_cards and entry.run_card_count <= 0:
        findings.append(_finding("missing_run_cards", "error", "release entry has no run cards", release_entry_id=entry.release_entry_id))
    if policy.require_metric_cards and entry.metric_card_count <= 0:
        findings.append(
            _finding("missing_metric_cards", "error", "release entry has no metric cards", release_entry_id=entry.release_entry_id)
        )
    if policy.require_entry_in_delta and not any(record.release_entry_id == entry.release_entry_id for record in delta_records):
        findings.append(
            _finding("release_entry_not_in_delta", "error", "release entry is not represented in delta records", release_entry_id=entry.release_entry_id)
        )
    if policy.require_entry_in_snapshot and not any(_snapshot_record_matches_entry(record, entry) for record in snapshot_records):
        findings.append(
            _finding(
                "release_entry_not_in_snapshot",
                "error",
                "release entry is not represented in release snapshot",
                release_entry_id=entry.release_entry_id,
            )
        )
    findings.extend(_source_claim_findings(entry.metadata, policy, "release entry", release_entry_id=entry.release_entry_id))
    findings.extend(_payload_findings(entry.model_dump(mode="json"), release_entry_id=entry.release_entry_id))
    return findings


def _delta_findings(
    entry: OledReleaseCandidateEntry | None,
    delta_records: list[OledReleaseCandidateDeltaRecord],
    snapshot_records: list[OledReleaseCandidateSnapshotRecordSummary],
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy,
) -> list[OledReleaseCandidateExternalPublicationPreflightFinding]:
    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding] = []
    if not delta_records:
        if policy.require_release_delta_jsonl:
            findings.append(_finding("missing_release_candidate_delta_jsonl", "error", "release delta JSONL is required"))
        return findings
    if policy.require_single_release_delta_record and len(delta_records) != 1:
        findings.append(_finding("multiple_release_delta_records", "error", "release delta contains multiple records"))
    for record in delta_records:
        if _status_value(record.release_status) != OledReleaseCandidateEntryStatus.RELEASE_CANDIDATE.value:
            findings.append(
                _finding(
                    "delta_status_not_release_candidate",
                    "error",
                    "release delta record status is not release_candidate",
                    release_entry_id=record.release_entry_id,
                )
            )
        if policy.require_delta_records_in_snapshot and not any(_snapshot_record_matches_delta(snapshot, record) for snapshot in snapshot_records):
            findings.append(
                _finding(
                    "release_delta_record_not_in_snapshot",
                    "error",
                    "release delta record is not represented in release snapshot",
                    release_entry_id=record.release_entry_id,
                )
            )
        if record.benchmark_validated:
            findings.append(
                _finding("benchmark_validated_source_claim", "error", "release delta claims benchmark validation", release_entry_id=record.release_entry_id)
            )
        if record.scientific_claim_validated:
            findings.append(
                _finding(
                    "scientific_claim_validated_source_claim",
                    "error",
                    "release delta claims scientific validation",
                    release_entry_id=record.release_entry_id,
                )
            )
        if record.benchmark_published or record.benchmark_registered:
            findings.append(
                _finding(
                    "external_publication_source_claim",
                    "error",
                    "release delta claims publication or registration",
                    release_entry_id=record.release_entry_id,
                )
            )
        findings.extend(_source_claim_findings(record.metadata, policy, "release delta", release_entry_id=record.release_entry_id))
        findings.extend(_payload_findings(record.model_dump(mode="json"), release_entry_id=record.release_entry_id))
    if entry is not None and policy.require_entry_in_delta and not any(record.release_entry_id == entry.release_entry_id for record in delta_records):
        findings.append(
            _finding("release_entry_not_in_delta", "error", "release entry is not represented in delta records", release_entry_id=entry.release_entry_id)
        )
    return findings


def _snapshot_findings(
    snapshot_records: list[OledReleaseCandidateSnapshotRecordSummary],
    prior_records: list[OledReleaseCandidateSnapshotRecordSummary],
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy,
) -> list[OledReleaseCandidateExternalPublicationPreflightFinding]:
    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding] = []
    if not snapshot_records:
        if policy.require_release_snapshot_jsonl:
            findings.append(_finding("missing_release_candidate_snapshot_jsonl", "error", "release snapshot JSONL is required"))
        return findings
    if prior_records and policy.require_prior_snapshot_preserved:
        preserved = _preserved_prior_count(prior_records, snapshot_records)
        if preserved != len(prior_records):
            findings.append(_finding("prior_snapshot_not_preserved", "error", "prior snapshot records are not preserved as a prefix"))
    for record in snapshot_records:
        findings.extend(_source_claim_findings(record.metadata, policy, "release snapshot", release_entry_id=record.release_entry_id))
        findings.extend(_payload_findings(record.model_dump(mode="json"), release_entry_id=record.release_entry_id))
    return findings


def _required_entry_source_findings(
    entry: OledReleaseCandidateEntry,
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy,
    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding],
) -> None:
    required_fields = [
        ("source_global_append_writer_manifest_id", policy.require_source_global_append_writer_manifest_id, "missing_source_global_append_writer_manifest_id"),
        ("source_global_append_entry_id", policy.require_source_global_append_entry_id, "missing_source_global_append_entry_id"),
        ("source_release_preflight_status", policy.require_source_release_preflight_status, "missing_source_release_preflight_status"),
        ("source_final_registry_entry_id", policy.require_source_final_registry_entry_id, "missing_source_final_registry_entry_id"),
        (
            "source_final_registry_writer_manifest_id",
            policy.require_source_final_registry_writer_manifest_id,
            "missing_source_final_registry_writer_manifest_id",
        ),
        ("source_publication_entry_id", policy.require_source_publication_entry_id, "missing_source_publication_entry_id"),
        ("source_publication_writer_manifest_id", policy.require_source_publication_writer_manifest_id, "missing_source_publication_writer_manifest_id"),
        ("source_promoted_entry_id", policy.require_source_promoted_entry_id, "missing_source_promoted_entry_id"),
        ("source_promotion_writer_manifest_id", policy.require_source_promotion_writer_manifest_id, "missing_source_promotion_writer_manifest_id"),
        ("source_registry_entry_id", policy.require_source_registry_entry_id, "missing_source_registry_entry_id"),
        ("source_registry_writer_manifest_id", policy.require_source_registry_writer_manifest_id, "missing_source_registry_writer_manifest_id"),
        ("source_candidate_report_id", policy.require_source_candidate_report_id, "missing_source_candidate_report_id"),
        ("source_benchmark_report_manifest_id", policy.require_source_benchmark_report_manifest_id, "missing_source_benchmark_report_manifest_id"),
    ]
    for field_name, required, code in required_fields:
        if required and not getattr(entry, field_name):
            findings.append(_finding(code, "error", f"release entry lacks {field_name}", release_entry_id=entry.release_entry_id))
    if policy.require_valid_release_preflight_status and entry.source_release_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(
            _finding(
                "invalid_source_release_preflight_status",
                "error",
                "release entry source release preflight status is not valid",
                release_entry_id=entry.release_entry_id,
            )
        )


def _source_claim_findings(
    metadata: dict[str, Any],
    policy: OledReleaseCandidateExternalPublicationPreflightPolicy,
    context: str,
    *,
    release_entry_id: str | None = None,
) -> list[OledReleaseCandidateExternalPublicationPreflightFinding]:
    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding] = []
    if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", f"{context} claims benchmark validation", release_entry_id=release_entry_id))
    if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", metadata):
        findings.append(
            _finding("scientific_claim_validated_source_claim", "error", f"{context} claims scientific validation", release_entry_id=release_entry_id)
        )
    if not policy.require_no_external_publication_claims:
        return findings
    if _truthy_metadata_key("github_release_created", metadata):
        findings.append(_finding("github_release_source_claim", "error", f"{context} claims GitHub release creation", release_entry_id=release_entry_id))
    if _truthy_metadata_key("git_tag_created", metadata):
        findings.append(_finding("git_tag_source_claim", "error", f"{context} claims git tag creation", release_entry_id=release_entry_id))
    if _truthy_metadata_key("artifact_uploaded", metadata):
        findings.append(_finding("artifact_upload_source_claim", "error", f"{context} claims external artifact upload", release_entry_id=release_entry_id))
    if _metadata_claims_external_publication(metadata):
        findings.append(
            _finding("external_publication_source_claim", "error", f"{context} claims publication or global mutation", release_entry_id=release_entry_id)
        )
    return findings


def _payload_findings(
    payload: Any,
    *,
    release_entry_id: str | None = None,
) -> list[OledReleaseCandidateExternalPublicationPreflightFinding]:
    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding] = []
    if _contains_raw_prediction_payload(payload):
        findings.append(_finding("raw_prediction_payload_leaked", "error", "payload leaks raw prediction identifiers", release_entry_id=release_entry_id))
    if _contains_raw_feature_payload(payload):
        findings.append(_finding("raw_feature_payload_leaked", "error", "payload leaks raw feature dictionaries", release_entry_id=release_entry_id))
    if _contains_absolute_path(payload):
        findings.append(_finding("absolute_path_leakage", "error", "payload leaks absolute local path", release_entry_id=release_entry_id))
    return findings


def _snapshot_record_from_payload(payload: dict[str, Any]) -> OledReleaseCandidateSnapshotRecordSummary:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    release_entry_id = payload.get("release_entry_id") or metadata.get("release_entry_id")
    return OledReleaseCandidateSnapshotRecordSummary(
        registry_entry_id=payload.get("registry_entry_id") or payload.get("final_registry_entry_id"),
        release_entry_id=release_entry_id,
        release_status=payload.get("release_status") or payload.get("registry_status") or payload.get("final_registry_status"),
        source_global_append_entry_id=payload.get("source_global_append_entry_id") or metadata.get("source_global_append_entry_id"),
        source_global_append_writer_manifest_id=payload.get("source_global_append_writer_manifest_id")
        or metadata.get("source_global_append_writer_manifest_id"),
        source_final_registry_entry_id=payload.get("source_final_registry_entry_id") or metadata.get("source_final_registry_entry_id"),
        source_publication_entry_id=payload.get("source_publication_entry_id") or metadata.get("source_publication_entry_id"),
        source_candidate_report_id=payload.get("source_candidate_report_id") or metadata.get("source_candidate_report_id"),
        source_benchmark_report_manifest_id=payload.get("source_benchmark_report_manifest_id")
        or metadata.get("source_benchmark_report_manifest_id"),
        metadata=metadata,
    )


def _snapshot_record_matches_entry(record: OledReleaseCandidateSnapshotRecordSummary, entry: OledReleaseCandidateEntry) -> bool:
    return bool(
        record.release_entry_id == entry.release_entry_id
        or record.metadata.get("release_entry_id") == entry.release_entry_id
        or (
            record.source_global_append_entry_id == entry.source_global_append_entry_id
            and record.source_candidate_report_id == entry.source_candidate_report_id
            and record.source_benchmark_report_manifest_id == entry.source_benchmark_report_manifest_id
        )
    )


def _snapshot_record_matches_delta(
    snapshot_record: OledReleaseCandidateSnapshotRecordSummary,
    delta_record: OledReleaseCandidateDeltaRecord,
) -> bool:
    return bool(
        snapshot_record.release_entry_id == delta_record.release_entry_id
        or snapshot_record.metadata.get("release_entry_id") == delta_record.release_entry_id
        or (
            snapshot_record.source_global_append_entry_id == delta_record.source_global_append_entry_id
            and snapshot_record.source_candidate_report_id == delta_record.source_candidate_report_id
            and snapshot_record.source_benchmark_report_manifest_id == delta_record.source_benchmark_report_manifest_id
        )
    )


def _preserved_prior_count(
    prior_records: list[OledReleaseCandidateSnapshotRecordSummary],
    snapshot_records: list[OledReleaseCandidateSnapshotRecordSummary],
) -> int:
    count = 0
    for prior, snapshot in zip(prior_records, snapshot_records):
        if _record_payload(prior) != _record_payload(snapshot):
            break
        count += 1
    return count


def _record_payload(record: OledReleaseCandidateSnapshotRecordSummary) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude_none=True)


def _file_result_for_kind(manifest: OledGlobalAppendReleaseWriterManifest, artifact_kind: str) -> Any | None:
    for file_result in manifest.file_results:
        if file_result.artifact_kind == artifact_kind:
            return file_result
    return None


def _missing_artifact_reason(artifact_kind: str) -> str:
    return {
        "release_candidate_entry_json": "missing_release_candidate_entry_json",
        "release_candidate_delta_jsonl": "missing_release_candidate_delta_jsonl",
        "release_candidate_snapshot_jsonl": "missing_release_candidate_snapshot_jsonl",
    }.get(artifact_kind, f"missing_{artifact_kind}")


def _selected_values(values: Iterable[str], selected: Iterable[str]) -> list[str]:
    selected_set = {str(item).strip() for item in selected if str(item).strip()}
    return sorted({str(item) for item in values if not selected_set or str(item) in selected_set})


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _report_status(
    findings: Iterable[OledReleaseCandidateExternalPublicationPreflightFinding],
) -> OledReleaseCandidateExternalPublicationPreflightStatus:
    severities = [finding.severity for finding in findings]
    if "error" in severities:
        return OledReleaseCandidateExternalPublicationPreflightStatus.FAILED
    if "warning" in severities:
        return OledReleaseCandidateExternalPublicationPreflightStatus.PASSED_WITH_WARNINGS
    return OledReleaseCandidateExternalPublicationPreflightStatus.PASSED


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    artifact_kind: str | None = None,
    release_entry_id: str | None = None,
    source_global_append_entry_id: str | None = None,
    source_final_registry_entry_id: str | None = None,
    source_publication_entry_id: str | None = None,
    source_registry_entry_id: str | None = None,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    output_path: str | None = None,
) -> OledReleaseCandidateExternalPublicationPreflightFinding:
    return OledReleaseCandidateExternalPublicationPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        artifact_kind=artifact_kind,
        release_entry_id=release_entry_id,
        source_global_append_entry_id=source_global_append_entry_id,
        source_final_registry_entry_id=source_final_registry_entry_id,
        source_publication_entry_id=source_publication_entry_id,
        source_registry_entry_id=source_registry_entry_id,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        output_path=output_path,
    )


def _dedup_findings(
    findings: list[OledReleaseCandidateExternalPublicationPreflightFinding],
) -> list[OledReleaseCandidateExternalPublicationPreflightFinding]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[OledReleaseCandidateExternalPublicationPreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.artifact_kind,
            finding.release_entry_id,
            finding.source_global_append_entry_id,
            finding.output_path,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


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
            "global_registry_written",
            "final_registry_written",
            "external_publication_written",
            "externally_published",
            "published",
            "release_published",
            "uploaded",
        )
    )


def _contains_raw_prediction_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"prediction_id", "training_row_id", "predictions", "prediction_payload", "raw_predictions"}:
                return True
            if _contains_raw_prediction_payload(child):
                return True
    elif isinstance(value, list):
        return any(_contains_raw_prediction_payload(item) for item in value)
    return False


def _contains_raw_feature_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"features", "feature_dict", "feature_payload", "raw_features"}:
                return True
            if _contains_raw_feature_payload(child):
                return True
    elif isinstance(value, list):
        return any(_contains_raw_feature_payload(item) for item in value)
    return False


def _contains_forbidden_payload_key(value: Any) -> bool:
    return _contains_raw_prediction_payload(value) or _contains_raw_feature_payload(value) or _contains_raw_text_key(value)


def _contains_raw_text_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"raw_text", "full_text", "paper_text", "ocr_text"}:
                return True
            if _contains_raw_text_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_raw_text_key(item) for item in value)
    return False


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute() or value.startswith("file://") or value.startswith("C:\\")
    return False


def _sanitize_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            if str(key).lower() in {"raw_text", "full_text", "paper_text", "ocr_text", "features", "prediction_id", "training_row_id"}:
                continue
            sanitized[str(key)] = _sanitize_for_output(child)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_output(item) for item in value]
    if isinstance(value, str) and (Path(value).is_absolute() or value.startswith("file://")):
        return redact_oled_mineru_acceptance_path(value)
    return value


def _reject_forbidden_input(path: Path) -> None:
    if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}:
        raise ValueError(f"unsupported_release_candidate_external_publication_input:{redact_oled_mineru_acceptance_path(path)}")


def _resolve_manifest_path(output_path: str, base_dir: str | Path) -> Path:
    candidate = Path(output_path)
    if candidate.is_absolute():
        raise ValueError(f"absolute_path_leakage:{redact_oled_mineru_acceptance_path(candidate)}")
    return Path(base_dir) / candidate


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_cli_values(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _safety_metadata() -> dict[str, Any]:
    return {
        "release_candidate_external_publication_preflight_only": True,
        "external_publication_written": False,
        "github_release_created": False,
        "git_tag_created": False,
        "artifact_uploaded": False,
        "global_registry_mutated": False,
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


if __name__ == "__main__":
    raise SystemExit(main())
