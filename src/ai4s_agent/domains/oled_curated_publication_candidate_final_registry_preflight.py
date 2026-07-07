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

from ai4s_agent.domains.oled_curated_promoted_registry_publication_writer import (
    OledPromotedRegistryPublicationWriteStatus,
    OledPromotedRegistryPublicationWriterManifest,
    OledPublicationCandidateRegistryEntry,
    OledPublicationCandidateRegistryIndexRecord,
    load_oled_publication_candidate_registry_entry_json,
    load_oled_publication_candidate_registry_index_jsonl,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledPublicationCandidateFinalRegistryPreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledPublicationCandidateFinalRegistryArtifactStatus(str, Enum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    FAILED = "failed"
    SKIPPED = "skipped"


class OledPublicationCandidateFinalRegistryPreflightPolicy(BaseModel):
    require_publication_manifest_sha256: bool = True
    require_publication_entry_sha256: bool = True
    require_publication_index_sha256: bool = True

    require_publication_entry_json: bool = True
    require_publication_index_jsonl: bool = True

    require_publication_candidate_status: bool = True
    require_entry_in_index: bool = True
    require_single_publication_index_record: bool = True

    require_source_promotion_writer_manifest_id: bool = True
    require_source_promoted_entry_id: bool = True
    require_source_publication_preflight_status: bool = True
    require_source_registry_entry_id: bool = True
    require_source_registry_writer_manifest_id: bool = True
    require_source_candidate_report_id: bool = True
    require_source_benchmark_report_manifest_id: bool = True

    require_valid_publication_preflight_status: bool = True
    require_caveats: bool = True
    require_run_cards: bool = True
    require_metric_cards: bool = True

    require_no_benchmark_validated_claims: bool = True
    require_no_scientific_claims: bool = True
    require_no_final_registry_claims: bool = True

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


class OledPublicationCandidateFinalRegistryArtifactSummary(BaseModel):
    artifact_kind: str
    status: OledPublicationCandidateFinalRegistryArtifactStatus
    output_path: str | None = None
    output_sha256: str | None = None
    loaded: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class OledPublicationCandidateFinalRegistryEntrySummary(BaseModel):
    publication_entry_id: str
    publication_status: str

    source_promoted_entry_id: str | None = None
    source_promotion_writer_manifest_id: str | None = None
    source_publication_preflight_status: str | None = None
    source_registry_entry_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    run_card_count: int = 0
    metric_card_count: int = 0

    publication_index_record_count: int = 0
    matched_publication_index_record_count: int = 0

    artifact_status: OledPublicationCandidateFinalRegistryArtifactStatus
    reason_codes: list[str] = Field(default_factory=list)


class OledPublicationCandidateFinalRegistryPreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    artifact_kind: str | None = None
    publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    output_path: str | None = None


class OledPublicationCandidateFinalRegistryPreflightReport(BaseModel):
    status: OledPublicationCandidateFinalRegistryPreflightStatus

    source_publication_writer_manifest_id: str | None = None
    source_publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None
    source_publication_preflight_status: str | None = None

    input_publication_entry_count: int = 0
    input_publication_index_record_count: int = 0

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    artifact_summaries: list[OledPublicationCandidateFinalRegistryArtifactSummary] = Field(default_factory=list)
    entry_summaries: list[OledPublicationCandidateFinalRegistryEntrySummary] = Field(default_factory=list)

    caveats: list[str] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    findings: list[OledPublicationCandidateFinalRegistryPreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledPublicationCandidateFinalRegistryPreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_promoted_registry_publication_writer_manifest_json(
    path: str | Path,
) -> OledPromotedRegistryPublicationWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_promoted_registry_publication_writer_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledPromotedRegistryPublicationWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_promoted_registry_publication_writer_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.model_dump(mode="json")):
        raise ValueError("absolute_path_in_promoted_registry_publication_writer_manifest")
    return manifest


def load_oled_publication_candidate_registry_artifacts_from_manifest(
    *,
    manifest: OledPromotedRegistryPublicationWriterManifest,
    base_dir: str | Path,
) -> tuple[OledPublicationCandidateRegistryEntry | None, list[OledPublicationCandidateRegistryIndexRecord]]:
    publication_entry: OledPublicationCandidateRegistryEntry | None = None
    index_records: list[OledPublicationCandidateRegistryIndexRecord] = []
    for file_result in manifest.file_results:
        if _status_value(file_result.status) != OledPromotedRegistryPublicationWriteStatus.WRITTEN.value:
            continue
        if not file_result.output_path:
            continue
        path = _resolve_manifest_path(file_result.output_path, base_dir)
        if file_result.artifact_kind == "publication_candidate_entry_json":
            if not path.exists():
                raise ValueError(f"missing_publication_candidate_registry_entry_json:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"publication_candidate_entry_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            publication_entry = load_oled_publication_candidate_registry_entry_json(path)
        elif file_result.artifact_kind == "publication_candidate_index_jsonl":
            if not path.exists():
                raise ValueError(f"missing_publication_candidate_registry_index_jsonl:{redact_oled_mineru_acceptance_path(path)}")
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"publication_candidate_index_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            index_records = load_oled_publication_candidate_registry_index_jsonl(path)
    return publication_entry, index_records


def run_oled_publication_candidate_final_registry_preflight(
    *,
    publication_writer_manifest: OledPromotedRegistryPublicationWriterManifest,
    publication_entry: OledPublicationCandidateRegistryEntry | None,
    publication_index_records: Iterable[OledPublicationCandidateRegistryIndexRecord],
    policy: OledPublicationCandidateFinalRegistryPreflightPolicy | None = None,
) -> OledPublicationCandidateFinalRegistryPreflightReport:
    preflight_policy = policy or OledPublicationCandidateFinalRegistryPreflightPolicy()
    index_records = list(publication_index_records)
    artifact_summaries = _artifact_summaries(publication_writer_manifest, publication_entry, index_records, preflight_policy)
    entry_summaries = _entry_summaries(publication_entry, index_records, preflight_policy)
    findings: list[OledPublicationCandidateFinalRegistryPreflightFinding] = []
    findings.extend(_manifest_findings(publication_writer_manifest, preflight_policy))
    findings.extend(_artifact_findings(artifact_summaries))
    findings.extend(_entry_findings(publication_entry, index_records, preflight_policy))
    findings.extend(_index_findings(publication_entry, index_records, preflight_policy))
    findings = _dedup_findings(findings)
    status = _report_status(findings)
    status_counts = Counter(_status_value(summary.status) for summary in artifact_summaries)
    status_counts.update(_status_value(summary.artifact_status) for summary in entry_summaries)

    return OledPublicationCandidateFinalRegistryPreflightReport(
        status=status,
        source_publication_writer_manifest_id=publication_writer_manifest.manifest_id,
        source_publication_entry_id=publication_entry.publication_entry_id if publication_entry is not None else None,
        source_promoted_entry_id=(
            publication_entry.source_promoted_entry_id if publication_entry is not None else publication_writer_manifest.source_promoted_entry_id
        ),
        source_registry_entry_id=publication_entry.source_registry_entry_id if publication_entry is not None else None,
        source_candidate_report_id=publication_entry.source_candidate_report_id if publication_entry is not None else None,
        source_benchmark_report_manifest_id=publication_entry.source_benchmark_report_manifest_id if publication_entry is not None else None,
        source_publication_preflight_status=(
            publication_entry.source_publication_preflight_status
            if publication_entry is not None
            else publication_writer_manifest.source_publication_preflight_status
        ),
        input_publication_entry_count=1 if publication_entry is not None else 0,
        input_publication_index_record_count=len(index_records),
        baseline_kinds=_selected_values(
            publication_entry.baseline_kinds if publication_entry is not None else publication_writer_manifest.baseline_kinds,
            preflight_policy.baseline_kinds,
        ),
        target_property_ids=_selected_values(
            publication_entry.target_property_ids if publication_entry is not None else publication_writer_manifest.target_property_ids,
            preflight_policy.target_property_ids,
        ),
        feature_views=_selected_values(publication_entry.feature_views if publication_entry is not None else publication_writer_manifest.feature_views, preflight_policy.feature_views),
        artifact_summaries=artifact_summaries,
        entry_summaries=entry_summaries,
        caveats=sorted(publication_entry.caveats) if publication_entry is not None else [],
        status_counts=dict(sorted(status_counts.items())),
        finding_code_counts=dict(sorted(Counter(finding.code for finding in findings).items())),
        findings=findings,
        metadata=_safety_metadata(),
    )


def run_oled_publication_candidate_final_registry_preflight_from_files(
    *,
    publication_writer_manifest_path: str | Path,
    publication_candidate_base_dir: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledPublicationCandidateFinalRegistryPreflightPolicy | None = None,
) -> OledPublicationCandidateFinalRegistryPreflightReport:
    manifest = load_oled_promoted_registry_publication_writer_manifest_json(publication_writer_manifest_path)
    base_dir = Path(publication_candidate_base_dir) if publication_candidate_base_dir is not None else Path(publication_writer_manifest_path).parent
    entry, index_records = load_oled_publication_candidate_registry_artifacts_from_manifest(manifest=manifest, base_dir=base_dir)
    report = run_oled_publication_candidate_final_registry_preflight(
        publication_writer_manifest=manifest,
        publication_entry=entry,
        publication_index_records=index_records,
        policy=policy,
    )
    if output_report_path is not None:
        write_oled_publication_candidate_final_registry_preflight_report_json(report, output_report_path)
    return report


def write_oled_publication_candidate_final_registry_preflight_report_json(
    report: OledPublicationCandidateFinalRegistryPreflightReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(report.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED publication-candidate final-registry-readiness preflight.")
    parser.add_argument("--publication-writer-manifest", required=True, help="Path to publication writer manifest JSON.")
    parser.add_argument("--publication-candidate-base-dir", help="Base directory for publication-candidate entry/index artifacts.")
    parser.add_argument("--output-report", help="Optional final-registry-readiness preflight report JSON path.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--allow-multiple-publication-index-records", action="store_true", help="Allow more than one publication index record.")
    args = parser.parse_args(argv)
    try:
        policy = OledPublicationCandidateFinalRegistryPreflightPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            require_single_publication_index_record=not args.allow_multiple_publication_index_records,
        )
        report = run_oled_publication_candidate_final_registry_preflight_from_files(
            publication_writer_manifest_path=args.publication_writer_manifest,
            publication_candidate_base_dir=args.publication_candidate_base_dir,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": _status_value(report.status),
            "entry_summary_count": len(report.entry_summaries),
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
    manifest: OledPromotedRegistryPublicationWriterManifest,
    policy: OledPublicationCandidateFinalRegistryPreflightPolicy,
) -> list[OledPublicationCandidateFinalRegistryPreflightFinding]:
    findings: list[OledPublicationCandidateFinalRegistryPreflightFinding] = []
    if policy.require_source_promotion_writer_manifest_id and not manifest.source_promotion_writer_manifest_id:
        findings.append(_finding("missing_source_promotion_writer_manifest_id", "error", "publication writer manifest lacks source promotion writer manifest id"))
    if policy.require_source_promoted_entry_id and not manifest.source_promoted_entry_id:
        findings.append(_finding("missing_source_promoted_entry_id", "error", "publication writer manifest lacks source promoted entry id"))
    if policy.require_source_publication_preflight_status and not manifest.source_publication_preflight_status:
        findings.append(_finding("missing_source_publication_preflight_status", "error", "publication writer manifest lacks source publication preflight status"))
    if policy.require_valid_publication_preflight_status and manifest.source_publication_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_publication_preflight_status", "error", "source publication preflight status is not valid"))
    if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", manifest.metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", "publication writer manifest claims benchmark validation"))
    if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", manifest.metadata):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "publication writer manifest claims scientific validation"))
    if policy.require_no_final_registry_claims and _metadata_claims_final_registry(manifest.metadata):
        findings.append(_finding("final_registry_source_claim", "error", "publication writer manifest claims final registry publication"))
    return findings


def _artifact_summaries(
    manifest: OledPromotedRegistryPublicationWriterManifest,
    entry: OledPublicationCandidateRegistryEntry | None,
    index_records: list[OledPublicationCandidateRegistryIndexRecord],
    policy: OledPublicationCandidateFinalRegistryPreflightPolicy,
) -> list[OledPublicationCandidateFinalRegistryArtifactSummary]:
    summaries: list[OledPublicationCandidateFinalRegistryArtifactSummary] = []
    for artifact_kind, required, loaded, require_sha in (
        ("publication_candidate_entry_json", policy.require_publication_entry_json, entry is not None, policy.require_publication_entry_sha256),
        ("publication_candidate_index_jsonl", policy.require_publication_index_jsonl, bool(index_records), policy.require_publication_index_sha256),
    ):
        file_result = _file_result_for_kind(manifest, artifact_kind)
        reasons: set[str] = set()
        status = OledPublicationCandidateFinalRegistryArtifactStatus.READY
        if loaded:
            reasons.add("artifact_loaded")
        elif required:
            reasons.add(_missing_artifact_reason(artifact_kind))
            status = OledPublicationCandidateFinalRegistryArtifactStatus.FAILED
        else:
            reasons.add("artifact_optional")
            status = OledPublicationCandidateFinalRegistryArtifactStatus.SKIPPED
        if require_sha and file_result is not None and not file_result.output_sha256:
            reasons.add(f"missing_{artifact_kind}_sha256")
            status = OledPublicationCandidateFinalRegistryArtifactStatus.FAILED
        summaries.append(
            OledPublicationCandidateFinalRegistryArtifactSummary(
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
    summaries: list[OledPublicationCandidateFinalRegistryArtifactSummary],
) -> list[OledPublicationCandidateFinalRegistryPreflightFinding]:
    findings: list[OledPublicationCandidateFinalRegistryPreflightFinding] = []
    for summary in summaries:
        if summary.status != OledPublicationCandidateFinalRegistryArtifactStatus.FAILED:
            continue
        for reason in summary.reason_codes:
            findings.append(
                _finding(
                    reason,
                    "error",
                    "publication-candidate final registry artifact is not ready",
                    artifact_kind=summary.artifact_kind,
                    output_path=summary.output_path,
                )
            )
    return findings


def _entry_summaries(
    entry: OledPublicationCandidateRegistryEntry | None,
    index_records: list[OledPublicationCandidateRegistryIndexRecord],
    policy: OledPublicationCandidateFinalRegistryPreflightPolicy,
) -> list[OledPublicationCandidateFinalRegistryEntrySummary]:
    if entry is None:
        return []
    reasons: set[str] = {"entry_loaded"}
    status = OledPublicationCandidateFinalRegistryArtifactStatus.READY
    matched_count = sum(1 for record in index_records if record.publication_entry_id == entry.publication_entry_id)
    if policy.require_publication_candidate_status and _status_value(entry.publication_status) != "publication_candidate":
        reasons.add("publication_status_not_publication_candidate")
        status = OledPublicationCandidateFinalRegistryArtifactStatus.FAILED
    if policy.require_entry_in_index and matched_count == 0:
        reasons.add("publication_entry_not_in_index")
        status = OledPublicationCandidateFinalRegistryArtifactStatus.FAILED
    if policy.require_run_cards and entry.run_card_count <= 0:
        reasons.add("missing_run_cards")
        status = OledPublicationCandidateFinalRegistryArtifactStatus.FAILED
    if policy.require_metric_cards and entry.metric_card_count <= 0:
        reasons.add("missing_metric_cards")
        status = OledPublicationCandidateFinalRegistryArtifactStatus.FAILED
    return [
        OledPublicationCandidateFinalRegistryEntrySummary(
            publication_entry_id=entry.publication_entry_id,
            publication_status=_status_value(entry.publication_status),
            source_promoted_entry_id=entry.source_promoted_entry_id,
            source_promotion_writer_manifest_id=entry.source_promotion_writer_manifest_id,
            source_publication_preflight_status=entry.source_publication_preflight_status,
            source_registry_entry_id=entry.source_registry_entry_id,
            source_candidate_report_id=entry.source_candidate_report_id,
            source_benchmark_report_manifest_id=entry.source_benchmark_report_manifest_id,
            baseline_kinds=_selected_values(entry.baseline_kinds, policy.baseline_kinds),
            target_property_ids=_selected_values(entry.target_property_ids, policy.target_property_ids),
            feature_views=_selected_values(entry.feature_views, policy.feature_views),
            run_card_count=entry.run_card_count,
            metric_card_count=entry.metric_card_count,
            publication_index_record_count=len(index_records),
            matched_publication_index_record_count=matched_count,
            artifact_status=status,
            reason_codes=sorted(reasons),
        )
    ]


def _entry_findings(
    entry: OledPublicationCandidateRegistryEntry | None,
    index_records: list[OledPublicationCandidateRegistryIndexRecord],
    policy: OledPublicationCandidateFinalRegistryPreflightPolicy,
) -> list[OledPublicationCandidateFinalRegistryPreflightFinding]:
    findings: list[OledPublicationCandidateFinalRegistryPreflightFinding] = []
    if entry is None:
        if policy.require_publication_entry_json:
            findings.append(_finding("missing_publication_candidate_registry_entry_json", "error", "publication-candidate entry JSON is required", artifact_kind="publication_candidate_entry_json"))
        return findings
    if policy.require_publication_candidate_status and _status_value(entry.publication_status) != "publication_candidate":
        findings.append(_finding("publication_status_not_publication_candidate", "error", "publication entry status is not publication_candidate", publication_entry_id=entry.publication_entry_id))
    if policy.require_source_promotion_writer_manifest_id and not entry.source_promotion_writer_manifest_id:
        findings.append(_finding("missing_source_promotion_writer_manifest_id", "error", "publication entry lacks source promotion writer manifest id", publication_entry_id=entry.publication_entry_id))
    if policy.require_source_promoted_entry_id and not entry.source_promoted_entry_id:
        findings.append(_finding("missing_source_promoted_entry_id", "error", "publication entry lacks source promoted entry id", publication_entry_id=entry.publication_entry_id))
    if policy.require_source_publication_preflight_status and not entry.source_publication_preflight_status:
        findings.append(_finding("missing_source_publication_preflight_status", "error", "publication entry lacks source publication preflight status", publication_entry_id=entry.publication_entry_id))
    if policy.require_source_registry_entry_id and not entry.source_registry_entry_id:
        findings.append(_finding("missing_source_registry_entry_id", "error", "publication entry lacks source registry entry id", publication_entry_id=entry.publication_entry_id))
    if policy.require_source_registry_writer_manifest_id and not entry.source_registry_writer_manifest_id:
        findings.append(_finding("missing_source_registry_writer_manifest_id", "error", "publication entry lacks source registry writer manifest id", publication_entry_id=entry.publication_entry_id))
    if policy.require_source_candidate_report_id and not entry.source_candidate_report_id:
        findings.append(_finding("missing_source_candidate_report_id", "error", "publication entry lacks source candidate report id", publication_entry_id=entry.publication_entry_id))
    if policy.require_source_benchmark_report_manifest_id and not entry.source_benchmark_report_manifest_id:
        findings.append(_finding("missing_source_benchmark_report_manifest_id", "error", "publication entry lacks source benchmark report manifest id", publication_entry_id=entry.publication_entry_id))
    if policy.require_valid_publication_preflight_status and entry.source_publication_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_publication_preflight_status", "error", "publication entry source publication preflight status is not valid", publication_entry_id=entry.publication_entry_id))
    if policy.require_caveats:
        caveats = set(entry.caveats)
        for caveat in policy.required_caveats:
            if caveat not in caveats:
                findings.append(_finding("missing_required_caveat", "error", "publication entry lacks required caveat", publication_entry_id=entry.publication_entry_id))
    if policy.require_run_cards and entry.run_card_count <= 0:
        findings.append(_finding("missing_run_cards", "error", "publication entry has no run cards", publication_entry_id=entry.publication_entry_id))
    if policy.require_metric_cards and entry.metric_card_count <= 0:
        findings.append(_finding("missing_metric_cards", "error", "publication entry has no metric cards", publication_entry_id=entry.publication_entry_id))
    if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", entry.metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", "publication entry claims benchmark validation", publication_entry_id=entry.publication_entry_id))
    if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", entry.metadata):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "publication entry claims scientific validation", publication_entry_id=entry.publication_entry_id))
    if policy.require_no_final_registry_claims and _metadata_claims_final_registry(entry.metadata):
        findings.append(_finding("final_registry_source_claim", "error", "publication entry claims final/global registry writing", publication_entry_id=entry.publication_entry_id))
    if _contains_raw_prediction_payload_key(entry.model_dump(mode="json")):
        findings.append(_finding("raw_prediction_payload_leaked", "error", "publication entry contains raw prediction payload", publication_entry_id=entry.publication_entry_id))
    if _contains_feature_payload_key(entry.model_dump(mode="json")):
        findings.append(_finding("raw_feature_payload_leaked", "error", "publication entry contains feature payload", publication_entry_id=entry.publication_entry_id))
    if _contains_absolute_path(entry.model_dump(mode="json")):
        findings.append(_finding("absolute_path_leakage", "error", "publication entry contains absolute path", publication_entry_id=entry.publication_entry_id))
    if policy.require_entry_in_index and not any(record.publication_entry_id == entry.publication_entry_id for record in index_records):
        findings.append(_finding("publication_entry_not_in_index", "error", "publication entry is not referenced by index", publication_entry_id=entry.publication_entry_id))
    return findings


def _index_findings(
    entry: OledPublicationCandidateRegistryEntry | None,
    index_records: list[OledPublicationCandidateRegistryIndexRecord],
    policy: OledPublicationCandidateFinalRegistryPreflightPolicy,
) -> list[OledPublicationCandidateFinalRegistryPreflightFinding]:
    findings: list[OledPublicationCandidateFinalRegistryPreflightFinding] = []
    if not index_records:
        if policy.require_publication_index_jsonl:
            findings.append(_finding("missing_publication_candidate_registry_index_jsonl", "error", "publication-candidate index JSONL is required", artifact_kind="publication_candidate_index_jsonl"))
        return findings
    if policy.require_single_publication_index_record and len(index_records) > 1:
        findings.append(_finding("multiple_publication_index_records", "error", "publication-candidate index has multiple records", artifact_kind="publication_candidate_index_jsonl"))
    for record in index_records:
        if policy.require_publication_candidate_status and record.publication_status != "publication_candidate":
            findings.append(_finding("index_status_not_publication_candidate", "error", "publication index status is not publication_candidate", publication_entry_id=record.publication_entry_id))
        if policy.require_no_benchmark_validated_claims and (record.benchmark_validated or _truthy_metadata_key("benchmark_validated", record.metadata)):
            findings.append(_finding("benchmark_validated_source_claim", "error", "publication index claims benchmark validation", publication_entry_id=record.publication_entry_id))
        if policy.require_no_scientific_claims and (record.scientific_claim_validated or _truthy_metadata_key("scientific_claim_validated", record.metadata)):
            findings.append(_finding("scientific_claim_validated_source_claim", "error", "publication index claims scientific validation", publication_entry_id=record.publication_entry_id))
        if policy.require_no_final_registry_claims and (record.benchmark_registered or _metadata_claims_final_registry(record.metadata)):
            findings.append(_finding("final_registry_source_claim", "error", "publication index claims final/global registry writing", publication_entry_id=record.publication_entry_id))
        if _contains_raw_prediction_payload_key(record.model_dump(mode="json")):
            findings.append(_finding("raw_prediction_payload_leaked", "error", "publication index contains raw prediction payload", publication_entry_id=record.publication_entry_id))
        if _contains_feature_payload_key(record.model_dump(mode="json")):
            findings.append(_finding("raw_feature_payload_leaked", "error", "publication index contains feature payload", publication_entry_id=record.publication_entry_id))
        if _contains_absolute_path(record.model_dump(mode="json")):
            findings.append(_finding("absolute_path_leakage", "error", "publication index contains absolute path", publication_entry_id=record.publication_entry_id))
    if entry is not None and policy.require_entry_in_index and not any(record.publication_entry_id == entry.publication_entry_id for record in index_records):
        findings.append(_finding("publication_entry_not_in_index", "error", "publication entry id is absent from index", publication_entry_id=entry.publication_entry_id))
    return findings


def _file_result_for_kind(manifest: OledPromotedRegistryPublicationWriterManifest, artifact_kind: str) -> Any | None:
    for file_result in manifest.file_results:
        if file_result.artifact_kind == artifact_kind:
            return file_result
    return None


def _missing_artifact_reason(artifact_kind: str) -> str:
    if artifact_kind == "publication_candidate_entry_json":
        return "missing_publication_candidate_registry_entry_json"
    if artifact_kind == "publication_candidate_index_jsonl":
        return "missing_publication_candidate_registry_index_jsonl"
    return f"missing_{artifact_kind}"


def _selected_values(values: Iterable[str], selected: Iterable[str]) -> list[str]:
    selected_set = {str(item).strip() for item in selected if str(item).strip()}
    return sorted({str(item) for item in values if not selected_set or str(item) in selected_set})


def _report_status(
    findings: list[OledPublicationCandidateFinalRegistryPreflightFinding],
) -> OledPublicationCandidateFinalRegistryPreflightStatus:
    if any(finding.severity == "error" for finding in findings):
        return OledPublicationCandidateFinalRegistryPreflightStatus.FAILED
    if any(finding.severity == "warning" for finding in findings):
        return OledPublicationCandidateFinalRegistryPreflightStatus.PASSED_WITH_WARNINGS
    return OledPublicationCandidateFinalRegistryPreflightStatus.PASSED


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    artifact_kind: str | None = None,
    publication_entry_id: str | None = None,
    source_promoted_entry_id: str | None = None,
    source_registry_entry_id: str | None = None,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    output_path: str | None = None,
) -> OledPublicationCandidateFinalRegistryPreflightFinding:
    return OledPublicationCandidateFinalRegistryPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        artifact_kind=artifact_kind,
        publication_entry_id=publication_entry_id,
        source_promoted_entry_id=source_promoted_entry_id,
        source_registry_entry_id=source_registry_entry_id,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        output_path=output_path,
    )


def _dedup_findings(
    findings: list[OledPublicationCandidateFinalRegistryPreflightFinding],
) -> list[OledPublicationCandidateFinalRegistryPreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str, str, str, str]] = set()
    output: list[OledPublicationCandidateFinalRegistryPreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.artifact_kind or "",
            finding.publication_entry_id or "",
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
    return sorted(output, key=lambda item: (item.severity, item.code, item.artifact_kind or "", item.publication_entry_id or ""))


def _resolve_manifest_path(output_path: str, base_dir: str | Path) -> Path:
    candidate = Path(output_path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _truthy_metadata_key(key: str, metadata: dict[str, Any]) -> bool:
    return bool(metadata.get(key))


def _metadata_claims_final_registry(metadata: dict[str, Any]) -> bool:
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


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


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


def _contains_raw_prediction_payload_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _RAW_PREDICTION_JSON_KEYS:
                return True
            if _contains_raw_prediction_payload_key(item):
                return True
    if isinstance(value, list):
        return any(_contains_raw_prediction_payload_key(item) for item in value)
    return False


def _contains_feature_payload_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered == "features" or lowered.endswith("_features"):
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
            lowered = key.lower()
            if lowered in _RAW_PREDICTION_JSON_KEYS or lowered == "features" or lowered.endswith("_features"):
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
        "publication_candidate_final_registry_preflight_only": True,
        "final_registry_written": False,
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


_MAX_OUTPUT_STRING_LENGTH = 240

_RAW_PREDICTION_JSON_KEYS = {
    "raw_text",
    "full_text",
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
    "OledPublicationCandidateFinalRegistryPreflightStatus",
    "OledPublicationCandidateFinalRegistryArtifactStatus",
    "OledPublicationCandidateFinalRegistryPreflightPolicy",
    "OledPublicationCandidateFinalRegistryArtifactSummary",
    "OledPublicationCandidateFinalRegistryEntrySummary",
    "OledPublicationCandidateFinalRegistryPreflightFinding",
    "OledPublicationCandidateFinalRegistryPreflightReport",
    "load_oled_promoted_registry_publication_writer_manifest_json",
    "load_oled_publication_candidate_registry_artifacts_from_manifest",
    "run_oled_publication_candidate_final_registry_preflight",
    "run_oled_publication_candidate_final_registry_preflight_from_files",
    "write_oled_publication_candidate_final_registry_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
