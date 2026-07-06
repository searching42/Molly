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

from ai4s_agent.domains.oled_curated_benchmark_registry_writer import (
    OledBenchmarkRegistryEntry,
    OledBenchmarkRegistryIndexRecord,
    OledBenchmarkRegistryWriteStatus,
    OledBenchmarkRegistryWriterManifest,
    load_oled_benchmark_registry_entry_json,
    load_oled_benchmark_registry_index_jsonl,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledBenchmarkRegistryPromotionPreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledBenchmarkRegistryPromotionArtifactStatus(str, Enum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    FAILED = "failed"
    SKIPPED = "skipped"


class OledBenchmarkRegistryPromotionPreflightPolicy(BaseModel):
    require_writer_manifest_sha256: bool = True
    require_registry_entry_sha256: bool = True
    require_registry_index_sha256: bool = True

    require_registry_entry_json: bool = True
    require_registry_index_jsonl: bool = True

    require_candidate_status: bool = True
    require_entry_in_index: bool = True
    require_single_entry_index: bool = True

    require_source_benchmark_report_manifest_id: bool = True
    require_source_benchmark_registry_preflight_status: bool = True
    require_source_candidate_report_id: bool = True
    require_valid_registry_preflight_status: bool = True

    require_caveats: bool = True
    require_run_cards: bool = True
    require_metric_cards: bool = True

    require_no_benchmark_validated_claims: bool = True
    require_no_scientific_claims: bool = True

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


class OledBenchmarkRegistryPromotionArtifactSummary(BaseModel):
    artifact_kind: str

    status: OledBenchmarkRegistryPromotionArtifactStatus
    output_path: str | None = None
    output_sha256: str | None = None

    loaded: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class OledBenchmarkRegistryPromotionEntrySummary(BaseModel):
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

    index_record_count: int = 0
    matched_index_record_count: int = 0

    artifact_status: OledBenchmarkRegistryPromotionArtifactStatus
    reason_codes: list[str] = Field(default_factory=list)


class OledBenchmarkRegistryPromotionPreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    artifact_kind: str | None = None
    registry_entry_id: str | None = None
    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    output_path: str | None = None


class OledBenchmarkRegistryPromotionPreflightReport(BaseModel):
    status: OledBenchmarkRegistryPromotionPreflightStatus

    source_registry_writer_manifest_id: str | None = None
    source_registry_entry_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None
    source_benchmark_registry_preflight_status: str | None = None

    input_registry_entry_count: int = 0
    input_registry_index_record_count: int = 0

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    artifact_summaries: list[OledBenchmarkRegistryPromotionArtifactSummary] = Field(default_factory=list)
    entry_summaries: list[OledBenchmarkRegistryPromotionEntrySummary] = Field(default_factory=list)

    caveats: list[str] = Field(default_factory=list)

    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)

    findings: list[OledBenchmarkRegistryPromotionPreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledBenchmarkRegistryPromotionPreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_benchmark_registry_writer_manifest_json(
    path: str | Path,
) -> OledBenchmarkRegistryWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_benchmark_registry_writer_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledBenchmarkRegistryWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_benchmark_registry_writer_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.metadata):
        raise ValueError("absolute_path_in_benchmark_registry_writer_manifest_metadata")
    return manifest


def load_oled_benchmark_registry_artifacts_from_manifest(
    *,
    manifest: OledBenchmarkRegistryWriterManifest,
    base_dir: str | Path,
) -> tuple[OledBenchmarkRegistryEntry | None, list[OledBenchmarkRegistryIndexRecord]]:
    registry_entry: OledBenchmarkRegistryEntry | None = None
    index_records: list[OledBenchmarkRegistryIndexRecord] = []
    for file_result in manifest.file_results:
        if _status_value(file_result.status) != OledBenchmarkRegistryWriteStatus.WRITTEN.value:
            continue
        if not file_result.output_path:
            continue
        path = _resolve_manifest_path(file_result.output_path, base_dir)
        if file_result.artifact_kind == "registry_entry_json":
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"benchmark_registry_entry_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            registry_entry = load_oled_benchmark_registry_entry_json(path)
        elif file_result.artifact_kind == "registry_index_jsonl":
            if file_result.output_sha256 and _sha256_file(path) != file_result.output_sha256:
                raise ValueError(f"benchmark_registry_index_sha256_mismatch:{redact_oled_mineru_acceptance_path(path)}")
            index_records = load_oled_benchmark_registry_index_jsonl(path)
    return registry_entry, index_records


def run_oled_benchmark_registry_promotion_preflight(
    *,
    registry_writer_manifest: OledBenchmarkRegistryWriterManifest,
    registry_entry: OledBenchmarkRegistryEntry | None,
    registry_index_records: Iterable[OledBenchmarkRegistryIndexRecord],
    policy: OledBenchmarkRegistryPromotionPreflightPolicy | None = None,
) -> OledBenchmarkRegistryPromotionPreflightReport:
    preflight_policy = policy or OledBenchmarkRegistryPromotionPreflightPolicy()
    index_records = list(registry_index_records)
    artifact_summaries = _artifact_summaries(registry_writer_manifest, registry_entry, index_records, preflight_policy)
    entry_summaries = _entry_summaries(registry_entry, index_records, preflight_policy)
    findings: list[OledBenchmarkRegistryPromotionPreflightFinding] = []
    findings.extend(_manifest_findings(registry_writer_manifest, preflight_policy))
    findings.extend(_artifact_findings(artifact_summaries))
    findings.extend(_entry_findings(registry_entry, index_records, preflight_policy))
    findings.extend(_index_findings(registry_entry, index_records, preflight_policy))
    findings = _dedup_findings(findings)
    status = _report_status(findings)
    status_counts = Counter(_status_value(summary.status) for summary in artifact_summaries)
    status_counts.update(_status_value(summary.artifact_status) for summary in entry_summaries)

    return OledBenchmarkRegistryPromotionPreflightReport(
        status=status,
        source_registry_writer_manifest_id=registry_writer_manifest.manifest_id,
        source_registry_entry_id=registry_entry.registry_entry_id if registry_entry is not None else None,
        source_candidate_report_id=registry_entry.source_candidate_report_id if registry_entry is not None else registry_writer_manifest.source_candidate_report_id,
        source_benchmark_report_manifest_id=(
            registry_entry.source_benchmark_report_manifest_id if registry_entry is not None else registry_writer_manifest.source_benchmark_report_manifest_id
        ),
        source_benchmark_registry_preflight_status=(
            registry_entry.source_benchmark_registry_preflight_status if registry_entry is not None else registry_writer_manifest.source_benchmark_registry_preflight_status
        ),
        input_registry_entry_count=1 if registry_entry is not None else 0,
        input_registry_index_record_count=len(index_records),
        baseline_kinds=_selected_values(registry_entry.baseline_kinds if registry_entry is not None else registry_writer_manifest.baseline_kinds, preflight_policy.baseline_kinds),
        target_property_ids=_selected_values(
            registry_entry.target_property_ids if registry_entry is not None else registry_writer_manifest.target_property_ids,
            preflight_policy.target_property_ids,
        ),
        feature_views=_selected_values(registry_entry.feature_views if registry_entry is not None else registry_writer_manifest.feature_views, preflight_policy.feature_views),
        artifact_summaries=artifact_summaries,
        entry_summaries=entry_summaries,
        caveats=sorted(registry_entry.caveats) if registry_entry is not None else [],
        status_counts=dict(sorted(status_counts.items())),
        finding_code_counts=dict(sorted(Counter(finding.code for finding in findings).items())),
        findings=findings,
        metadata=_safety_metadata(),
    )


def run_oled_benchmark_registry_promotion_preflight_from_files(
    *,
    registry_writer_manifest_path: str | Path,
    registry_base_dir: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledBenchmarkRegistryPromotionPreflightPolicy | None = None,
) -> OledBenchmarkRegistryPromotionPreflightReport:
    manifest = load_oled_benchmark_registry_writer_manifest_json(registry_writer_manifest_path)
    base_dir = Path(registry_base_dir) if registry_base_dir is not None else Path(registry_writer_manifest_path).parent
    entry, index_records = load_oled_benchmark_registry_artifacts_from_manifest(manifest=manifest, base_dir=base_dir)
    report = run_oled_benchmark_registry_promotion_preflight(
        registry_writer_manifest=manifest,
        registry_entry=entry,
        registry_index_records=index_records,
        policy=policy,
    )
    if output_report_path is not None:
        write_oled_benchmark_registry_promotion_preflight_report_json(report, output_report_path)
    return report


def write_oled_benchmark_registry_promotion_preflight_report_json(
    report: OledBenchmarkRegistryPromotionPreflightReport,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(report.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED benchmark registry promotion-readiness preflight.")
    parser.add_argument("--registry-writer-manifest", required=True, help="Path to benchmark registry writer manifest JSON.")
    parser.add_argument("--registry-base-dir", help="Base directory for registry entry/index artifacts.")
    parser.add_argument("--output-report", help="Optional promotion-readiness preflight report JSON path.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--allow-multiple-index-records", action="store_true", help="Allow more than one local index record.")
    args = parser.parse_args(argv)
    try:
        policy = OledBenchmarkRegistryPromotionPreflightPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            require_single_entry_index=not args.allow_multiple_index_records,
        )
        report = run_oled_benchmark_registry_promotion_preflight_from_files(
            registry_writer_manifest_path=args.registry_writer_manifest,
            registry_base_dir=args.registry_base_dir,
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
    manifest: OledBenchmarkRegistryWriterManifest,
    policy: OledBenchmarkRegistryPromotionPreflightPolicy,
) -> list[OledBenchmarkRegistryPromotionPreflightFinding]:
    findings: list[OledBenchmarkRegistryPromotionPreflightFinding] = []
    if policy.require_source_benchmark_report_manifest_id and not manifest.source_benchmark_report_manifest_id:
        findings.append(_finding("missing_source_benchmark_report_manifest_id", "error", "registry writer manifest lacks source benchmark report manifest id"))
    if policy.require_source_benchmark_registry_preflight_status and not manifest.source_benchmark_registry_preflight_status:
        findings.append(_finding("missing_source_benchmark_registry_preflight_status", "error", "registry writer manifest lacks source registry preflight status"))
    if policy.require_source_candidate_report_id and not manifest.source_candidate_report_id:
        findings.append(_finding("missing_source_candidate_report_id", "error", "registry writer manifest lacks source candidate report id"))
    if policy.require_valid_registry_preflight_status and manifest.source_benchmark_registry_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_benchmark_registry_preflight_status", "error", "source registry preflight status is not valid"))
    if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", manifest.metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", "registry writer manifest claims benchmark validation"))
    if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", manifest.metadata):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "registry writer manifest claims scientific validation"))
    return findings


def _artifact_summaries(
    manifest: OledBenchmarkRegistryWriterManifest,
    entry: OledBenchmarkRegistryEntry | None,
    index_records: list[OledBenchmarkRegistryIndexRecord],
    policy: OledBenchmarkRegistryPromotionPreflightPolicy,
) -> list[OledBenchmarkRegistryPromotionArtifactSummary]:
    summaries: list[OledBenchmarkRegistryPromotionArtifactSummary] = []
    for artifact_kind, required, loaded, require_sha in (
        ("registry_entry_json", policy.require_registry_entry_json, entry is not None, policy.require_registry_entry_sha256),
        ("registry_index_jsonl", policy.require_registry_index_jsonl, bool(index_records), policy.require_registry_index_sha256),
    ):
        file_result = _file_result_for_kind(manifest, artifact_kind)
        reasons: set[str] = set()
        status = OledBenchmarkRegistryPromotionArtifactStatus.READY
        if loaded:
            reasons.add("artifact_loaded")
        elif required:
            reasons.add(_missing_artifact_reason(artifact_kind))
            status = OledBenchmarkRegistryPromotionArtifactStatus.FAILED
        else:
            reasons.add("artifact_optional")
            status = OledBenchmarkRegistryPromotionArtifactStatus.SKIPPED
        if require_sha and file_result is not None and not file_result.output_sha256:
            reasons.add(f"missing_{artifact_kind}_sha256")
            status = OledBenchmarkRegistryPromotionArtifactStatus.FAILED
        summaries.append(
            OledBenchmarkRegistryPromotionArtifactSummary(
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
    summaries: list[OledBenchmarkRegistryPromotionArtifactSummary],
) -> list[OledBenchmarkRegistryPromotionPreflightFinding]:
    findings: list[OledBenchmarkRegistryPromotionPreflightFinding] = []
    for summary in summaries:
        if summary.status != OledBenchmarkRegistryPromotionArtifactStatus.FAILED:
            continue
        for reason in summary.reason_codes:
            findings.append(
                _finding(
                    _artifact_reason_to_code(reason),
                    "error",
                    "registry promotion artifact is not ready",
                    artifact_kind=summary.artifact_kind,
                    output_path=summary.output_path,
                )
            )
    return findings


def _entry_summaries(
    entry: OledBenchmarkRegistryEntry | None,
    index_records: list[OledBenchmarkRegistryIndexRecord],
    policy: OledBenchmarkRegistryPromotionPreflightPolicy,
) -> list[OledBenchmarkRegistryPromotionEntrySummary]:
    if entry is None:
        return []
    reasons: set[str] = {"entry_loaded"}
    status = OledBenchmarkRegistryPromotionArtifactStatus.READY
    matched_count = sum(1 for record in index_records if record.registry_entry_id == entry.registry_entry_id)
    if policy.require_candidate_status and _status_value(entry.registry_status) != "candidate":
        reasons.add("registry_status_not_candidate")
        status = OledBenchmarkRegistryPromotionArtifactStatus.FAILED
    if policy.require_entry_in_index and matched_count == 0:
        reasons.add("registry_entry_not_in_index")
        status = OledBenchmarkRegistryPromotionArtifactStatus.FAILED
    if policy.require_run_cards and entry.run_card_count <= 0:
        reasons.add("missing_run_cards")
        status = OledBenchmarkRegistryPromotionArtifactStatus.FAILED
    if policy.require_metric_cards and entry.metric_card_count <= 0:
        reasons.add("missing_metric_cards")
        status = OledBenchmarkRegistryPromotionArtifactStatus.FAILED
    return [
        OledBenchmarkRegistryPromotionEntrySummary(
            registry_entry_id=entry.registry_entry_id,
            registry_status=_status_value(entry.registry_status),
            source_candidate_report_id=entry.source_candidate_report_id,
            source_benchmark_report_manifest_id=entry.source_benchmark_report_manifest_id,
            source_benchmark_registry_preflight_status=entry.source_benchmark_registry_preflight_status,
            baseline_kinds=_selected_values(entry.baseline_kinds, policy.baseline_kinds),
            target_property_ids=_selected_values(entry.target_property_ids, policy.target_property_ids),
            feature_views=_selected_values(entry.feature_views, policy.feature_views),
            run_card_count=entry.run_card_count,
            metric_card_count=entry.metric_card_count,
            index_record_count=len(index_records),
            matched_index_record_count=matched_count,
            artifact_status=status,
            reason_codes=sorted(reasons),
        )
    ]


def _entry_findings(
    entry: OledBenchmarkRegistryEntry | None,
    index_records: list[OledBenchmarkRegistryIndexRecord],
    policy: OledBenchmarkRegistryPromotionPreflightPolicy,
) -> list[OledBenchmarkRegistryPromotionPreflightFinding]:
    findings: list[OledBenchmarkRegistryPromotionPreflightFinding] = []
    if entry is None:
        if policy.require_registry_entry_json:
            findings.append(_finding("missing_benchmark_registry_entry_json", "error", "registry entry JSON is required", artifact_kind="registry_entry_json"))
        return findings
    if policy.require_candidate_status and _status_value(entry.registry_status) != "candidate":
        findings.append(_finding("registry_status_not_candidate", "error", "registry entry status is not candidate", registry_entry_id=entry.registry_entry_id))
    if policy.require_source_benchmark_report_manifest_id and not entry.source_benchmark_report_manifest_id:
        findings.append(_finding("missing_source_benchmark_report_manifest_id", "error", "registry entry lacks source benchmark report manifest id"))
    if policy.require_source_benchmark_registry_preflight_status and not entry.source_benchmark_registry_preflight_status:
        findings.append(_finding("missing_source_benchmark_registry_preflight_status", "error", "registry entry lacks source registry preflight status"))
    if policy.require_source_candidate_report_id and not entry.source_candidate_report_id:
        findings.append(_finding("missing_source_candidate_report_id", "error", "registry entry lacks source candidate report id"))
    if policy.require_valid_registry_preflight_status and entry.source_benchmark_registry_preflight_status not in {"passed", "passed_with_warnings"}:
        findings.append(_finding("invalid_source_benchmark_registry_preflight_status", "error", "entry source registry preflight status is not valid"))
    if policy.require_caveats:
        caveats = set(entry.caveats)
        for caveat in policy.required_caveats:
            if caveat not in caveats:
                findings.append(_finding("missing_required_caveat", "error", "registry entry lacks required caveat", registry_entry_id=entry.registry_entry_id))
    if policy.require_run_cards and entry.run_card_count <= 0:
        findings.append(_finding("missing_run_cards", "error", "registry entry has no run cards", registry_entry_id=entry.registry_entry_id))
    if policy.require_metric_cards and entry.metric_card_count <= 0:
        findings.append(_finding("missing_metric_cards", "error", "registry entry has no metric cards", registry_entry_id=entry.registry_entry_id))
    if policy.require_no_benchmark_validated_claims and _truthy_metadata_key("benchmark_validated", entry.metadata):
        findings.append(_finding("benchmark_validated_source_claim", "error", "registry entry claims benchmark validation", registry_entry_id=entry.registry_entry_id))
    if policy.require_no_scientific_claims and _truthy_metadata_key("scientific_claim_validated", entry.metadata):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "registry entry claims scientific validation", registry_entry_id=entry.registry_entry_id))
    if _contains_forbidden_payload_key(entry.model_dump(mode="json")):
        findings.append(_finding("raw_prediction_payload_leaked", "error", "registry entry contains raw prediction payload", registry_entry_id=entry.registry_entry_id))
    if _contains_absolute_path(entry.model_dump(mode="json")):
        findings.append(_finding("absolute_path_leakage", "error", "registry entry contains absolute path", registry_entry_id=entry.registry_entry_id))
    if policy.require_entry_in_index and not any(record.registry_entry_id == entry.registry_entry_id for record in index_records):
        findings.append(_finding("registry_entry_not_in_index", "error", "registry entry is not referenced by index", registry_entry_id=entry.registry_entry_id))
    return findings


def _index_findings(
    entry: OledBenchmarkRegistryEntry | None,
    index_records: list[OledBenchmarkRegistryIndexRecord],
    policy: OledBenchmarkRegistryPromotionPreflightPolicy,
) -> list[OledBenchmarkRegistryPromotionPreflightFinding]:
    findings: list[OledBenchmarkRegistryPromotionPreflightFinding] = []
    if not index_records:
        if policy.require_registry_index_jsonl:
            findings.append(_finding("missing_benchmark_registry_index_jsonl", "error", "registry index JSONL is required", artifact_kind="registry_index_jsonl"))
        return findings
    if policy.require_single_entry_index and len(index_records) > 1:
        findings.append(_finding("multiple_index_records", "error", "registry index has multiple records", artifact_kind="registry_index_jsonl"))
    for record in index_records:
        if policy.require_candidate_status and record.registry_status != "candidate":
            findings.append(_finding("index_status_not_candidate", "error", "registry index status is not candidate", registry_entry_id=record.registry_entry_id))
        if policy.require_no_benchmark_validated_claims and (record.benchmark_validated or _truthy_metadata_key("benchmark_validated", record.metadata)):
            findings.append(_finding("benchmark_validated_source_claim", "error", "registry index claims benchmark validation", registry_entry_id=record.registry_entry_id))
        if policy.require_no_scientific_claims and (record.scientific_claim_validated or _truthy_metadata_key("scientific_claim_validated", record.metadata)):
            findings.append(_finding("scientific_claim_validated_source_claim", "error", "registry index claims scientific validation", registry_entry_id=record.registry_entry_id))
        if _contains_forbidden_payload_key(record.model_dump(mode="json")):
            findings.append(_finding("raw_prediction_payload_leaked", "error", "registry index contains raw prediction payload", registry_entry_id=record.registry_entry_id))
        if _contains_absolute_path(record.model_dump(mode="json")):
            findings.append(_finding("absolute_path_leakage", "error", "registry index contains absolute path", registry_entry_id=record.registry_entry_id))
    if entry is not None and policy.require_entry_in_index and not any(record.registry_entry_id == entry.registry_entry_id for record in index_records):
        findings.append(_finding("registry_entry_not_in_index", "error", "registry entry id is absent from index", registry_entry_id=entry.registry_entry_id))
    return findings


def _file_result_for_kind(manifest: OledBenchmarkRegistryWriterManifest, artifact_kind: str) -> Any | None:
    for file_result in manifest.file_results:
        if file_result.artifact_kind == artifact_kind:
            return file_result
    return None


def _missing_artifact_reason(artifact_kind: str) -> str:
    if artifact_kind == "registry_entry_json":
        return "missing_benchmark_registry_entry_json"
    if artifact_kind == "registry_index_jsonl":
        return "missing_benchmark_registry_index_jsonl"
    return f"missing_{artifact_kind}"


def _artifact_reason_to_code(reason: str) -> str:
    return reason


def _selected_values(values: Iterable[str], selected: Iterable[str]) -> list[str]:
    selected_set = {str(item).strip() for item in selected if str(item).strip()}
    return sorted({str(item) for item in values if not selected_set or str(item) in selected_set})


def _report_status(
    findings: list[OledBenchmarkRegistryPromotionPreflightFinding],
) -> OledBenchmarkRegistryPromotionPreflightStatus:
    if any(finding.severity == "error" for finding in findings):
        return OledBenchmarkRegistryPromotionPreflightStatus.FAILED
    if any(finding.severity == "warning" for finding in findings):
        return OledBenchmarkRegistryPromotionPreflightStatus.PASSED_WITH_WARNINGS
    return OledBenchmarkRegistryPromotionPreflightStatus.PASSED


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    artifact_kind: str | None = None,
    registry_entry_id: str | None = None,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    output_path: str | None = None,
) -> OledBenchmarkRegistryPromotionPreflightFinding:
    return OledBenchmarkRegistryPromotionPreflightFinding(
        code=code,
        severity=severity,
        message=message,
        artifact_kind=artifact_kind,
        registry_entry_id=registry_entry_id,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        output_path=output_path,
    )


def _dedup_findings(
    findings: list[OledBenchmarkRegistryPromotionPreflightFinding],
) -> list[OledBenchmarkRegistryPromotionPreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    output: list[OledBenchmarkRegistryPromotionPreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.artifact_kind or "",
            finding.registry_entry_id or "",
            finding.baseline_kind or "",
            finding.target_property_id or "",
            finding.feature_view or "",
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return sorted(output, key=lambda item: (item.severity, item.code, item.artifact_kind or "", item.registry_entry_id or ""))


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
        "benchmark_registry_promotion_preflight_only": True,
        "benchmark_promotion_written": False,
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
    "OledBenchmarkRegistryPromotionPreflightStatus",
    "OledBenchmarkRegistryPromotionArtifactStatus",
    "OledBenchmarkRegistryPromotionPreflightPolicy",
    "OledBenchmarkRegistryPromotionArtifactSummary",
    "OledBenchmarkRegistryPromotionEntrySummary",
    "OledBenchmarkRegistryPromotionPreflightFinding",
    "OledBenchmarkRegistryPromotionPreflightReport",
    "load_oled_benchmark_registry_writer_manifest_json",
    "load_oled_benchmark_registry_artifacts_from_manifest",
    "run_oled_benchmark_registry_promotion_preflight",
    "run_oled_benchmark_registry_promotion_preflight_from_files",
    "write_oled_benchmark_registry_promotion_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
