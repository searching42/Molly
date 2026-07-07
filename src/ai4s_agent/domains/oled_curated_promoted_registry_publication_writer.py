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

from ai4s_agent.domains.oled_curated_benchmark_registry_promotion_writer import (
    OledBenchmarkPromotedRegistryEntry,
    OledBenchmarkPromotedRegistryEntryStatus,
    OledBenchmarkPromotedRegistryIndexRecord,
    OledBenchmarkRegistryPromotionWriterManifest,
)
from ai4s_agent.domains.oled_curated_promoted_registry_publication_preflight import (
    OledPromotedRegistryPublicationPreflightReport,
    OledPromotedRegistryPublicationPreflightStatus,
    load_oled_benchmark_registry_promotion_writer_manifest_json,
    load_oled_promoted_registry_artifacts_from_manifest,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledPromotedRegistryPublicationWriterPolicy(BaseModel):
    require_confirmation: bool = True
    require_publication_preflight_valid: bool = True
    allow_publication_preflight_warnings: bool = True

    require_promoted_entry: bool = True
    require_promoted_index: bool = True
    require_promoted_entry_sha256: bool = True
    require_promoted_index_sha256: bool = True

    require_promoted_candidate_status: bool = True
    require_source_registry_writer_manifest_id: bool = True
    require_source_registry_entry_id: bool = True
    require_source_registry_promotion_preflight_status: bool = True
    require_source_candidate_report_id: bool = True
    require_source_benchmark_report_manifest_id: bool = True

    require_caveats: bool = True
    require_run_cards: bool = True
    require_metric_cards: bool = True

    require_no_benchmark_validated_claims: bool = True
    require_no_scientific_claims: bool = True
    require_no_publication_claims: bool = True

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)

    write_publication_entry_json: bool = True
    write_publication_index_jsonl: bool = True

    publication_status: Literal["publication_candidate"] = "publication_candidate"
    benchmark_validated: bool = False
    scientific_claim_validated: bool = False
    globally_registered: bool = False


class OledPromotedRegistryPublicationWriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class OledPublicationCandidateRegistryEntryStatus(str, Enum):
    PUBLICATION_CANDIDATE = "publication_candidate"
    REJECTED = "rejected"


class OledPublicationCandidateRegistryEntry(BaseModel):
    publication_entry_id: str
    publication_status: OledPublicationCandidateRegistryEntryStatus = OledPublicationCandidateRegistryEntryStatus.PUBLICATION_CANDIDATE

    source_promotion_writer_manifest_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_publication_preflight_status: str | None = None

    source_registry_entry_id: str | None = None
    source_registry_writer_manifest_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None
    source_registry_promotion_preflight_status: str | None = None

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    run_card_count: int = 0
    metric_card_count: int = 0

    source_promoted_entry_json_path: str | None = None
    source_promoted_entry_json_sha256: str | None = None
    source_promoted_index_jsonl_path: str | None = None
    source_promoted_index_jsonl_sha256: str | None = None

    caveats: list[str] = Field(default_factory=list)
    publication_reason_codes: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class OledPublicationCandidateRegistryIndexRecord(BaseModel):
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

    output_publication_entry_json_path: str | None = None
    output_publication_entry_json_sha256: str | None = None

    benchmark_registered: bool = False
    benchmark_validated: bool = False
    scientific_claim_validated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledPromotedRegistryPublicationFileResult(BaseModel):
    artifact_kind: Literal["publication_candidate_entry_json", "publication_candidate_index_jsonl", "manifest"]

    status: OledPromotedRegistryPublicationWriteStatus
    output_path: str | None = None
    output_sha256: str | None = None

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledPromotedRegistryPublicationWriterFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    output_path: str | None = None


class OledPromotedRegistryPublicationWriterManifest(BaseModel):
    manifest_id: str

    source_promotion_writer_manifest_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_publication_preflight_status: str | None = None

    output_directory: str | None = None
    output_file_count: int = 0

    publication_entry_ids: list[str] = Field(default_factory=list)

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    file_results: list[OledPromotedRegistryPublicationFileResult] = Field(default_factory=list)

    policy: OledPromotedRegistryPublicationWriterPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(result.status == OledPromotedRegistryPublicationWriteStatus.REJECTED for result in self.file_results)


class OledPromotedRegistryPublicationWriterReport(BaseModel):
    manifest: OledPromotedRegistryPublicationWriterManifest
    publication_entry: OledPublicationCandidateRegistryEntry | None = None
    publication_index_records: list[OledPublicationCandidateRegistryIndexRecord] = Field(default_factory=list)
    findings: list[OledPromotedRegistryPublicationWriterFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.error_codes and self.manifest.is_valid

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_promoted_registry_publication_preflight_report_json(
    path: str | Path,
) -> OledPromotedRegistryPublicationPreflightReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_promoted_registry_publication_preflight_report:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = OledPromotedRegistryPublicationPreflightReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_promoted_registry_publication_preflight_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    if _contains_absolute_path(report.metadata):
        raise ValueError("absolute_path_in_promoted_registry_publication_preflight_report_metadata")
    return report


def build_oled_publication_candidate_registry_entry(
    *,
    promotion_writer_manifest: OledBenchmarkRegistryPromotionWriterManifest,
    promoted_entry: OledBenchmarkPromotedRegistryEntry,
    promoted_index_records: Iterable[OledBenchmarkPromotedRegistryIndexRecord],
    publication_preflight_report: OledPromotedRegistryPublicationPreflightReport,
    policy: OledPromotedRegistryPublicationWriterPolicy | None = None,
) -> tuple[OledPublicationCandidateRegistryEntry | None, list[OledPromotedRegistryPublicationWriterFinding]]:
    writer_policy = policy or OledPromotedRegistryPublicationWriterPolicy()
    index_records = list(promoted_index_records)
    findings = _gate_findings(
        promotion_writer_manifest=promotion_writer_manifest,
        promoted_entry=promoted_entry,
        promoted_index_records=index_records,
        publication_preflight_report=publication_preflight_report,
        policy=writer_policy,
    )
    findings = _dedup_findings(findings)
    if any(finding.severity == "error" for finding in findings):
        return None, findings

    entry_result = _file_result_for_kind(promotion_writer_manifest.file_results, "promoted_registry_entry_json")
    index_result = _file_result_for_kind(promotion_writer_manifest.file_results, "promoted_registry_index_jsonl")
    publication_entry = OledPublicationCandidateRegistryEntry(
        publication_entry_id=_publication_entry_id(
            promotion_writer_manifest.manifest_id,
            promoted_entry.promoted_entry_id,
            publication_preflight_report.status,
        ),
        publication_status=OledPublicationCandidateRegistryEntryStatus.PUBLICATION_CANDIDATE,
        source_promotion_writer_manifest_id=promotion_writer_manifest.manifest_id,
        source_promoted_entry_id=promoted_entry.promoted_entry_id,
        source_publication_preflight_status=_status_value(publication_preflight_report.status),
        source_registry_entry_id=promoted_entry.source_registry_entry_id,
        source_registry_writer_manifest_id=promoted_entry.source_registry_writer_manifest_id,
        source_candidate_report_id=promoted_entry.source_candidate_report_id,
        source_benchmark_report_manifest_id=promoted_entry.source_benchmark_report_manifest_id,
        source_registry_promotion_preflight_status=promoted_entry.source_registry_promotion_preflight_status,
        baseline_kinds=_selected_values(promoted_entry.baseline_kinds, writer_policy.baseline_kinds),
        target_property_ids=_selected_values(promoted_entry.target_property_ids, writer_policy.target_property_ids),
        feature_views=_selected_values(promoted_entry.feature_views, writer_policy.feature_views),
        run_card_count=promoted_entry.run_card_count,
        metric_card_count=promoted_entry.metric_card_count,
        source_promoted_entry_json_path=entry_result.output_path if entry_result is not None else None,
        source_promoted_entry_json_sha256=entry_result.output_sha256 if entry_result is not None else None,
        source_promoted_index_jsonl_path=index_result.output_path if index_result is not None else None,
        source_promoted_index_jsonl_sha256=index_result.output_sha256 if index_result is not None else None,
        caveats=sorted(promoted_entry.caveats),
        publication_reason_codes=["selected_for_publication_candidate"],
        metadata=_entry_metadata(),
    )
    return publication_entry, findings


def build_oled_publication_candidate_registry_index_records(
    entry: OledPublicationCandidateRegistryEntry,
) -> list[OledPublicationCandidateRegistryIndexRecord]:
    return [
        OledPublicationCandidateRegistryIndexRecord(
            publication_entry_id=entry.publication_entry_id,
            publication_status=_status_value(entry.publication_status),
            source_promoted_entry_id=entry.source_promoted_entry_id,
            source_promotion_writer_manifest_id=entry.source_promotion_writer_manifest_id,
            source_publication_preflight_status=entry.source_publication_preflight_status,
            source_registry_entry_id=entry.source_registry_entry_id,
            source_candidate_report_id=entry.source_candidate_report_id,
            source_benchmark_report_manifest_id=entry.source_benchmark_report_manifest_id,
            baseline_kinds=list(entry.baseline_kinds),
            target_property_ids=list(entry.target_property_ids),
            feature_views=list(entry.feature_views),
            run_card_count=entry.run_card_count,
            metric_card_count=entry.metric_card_count,
            benchmark_registered=False,
            benchmark_validated=False,
            scientific_claim_validated=False,
            metadata={
                "publication_candidate_registry_index_record": True,
                "publication_status": "publication_candidate",
                "benchmark_registered": False,
                "benchmark_validated": False,
                "scientific_claim_validated": False,
            },
        )
    ]


def select_oled_promoted_registry_publication_for_write(
    *,
    promotion_writer_manifest: OledBenchmarkRegistryPromotionWriterManifest,
    promoted_entry: OledBenchmarkPromotedRegistryEntry,
    promoted_index_records: Iterable[OledBenchmarkPromotedRegistryIndexRecord],
    publication_preflight_report: OledPromotedRegistryPublicationPreflightReport,
    policy: OledPromotedRegistryPublicationWriterPolicy | None = None,
    confirm_promoted_registry_publication_write: bool = False,
) -> OledPromotedRegistryPublicationWriterReport:
    writer_policy = policy or OledPromotedRegistryPublicationWriterPolicy()
    if writer_policy.require_confirmation and not confirm_promoted_registry_publication_write:
        raise ValueError("confirmation_required:promoted_registry_publication_write")
    publication_entry, findings = build_oled_publication_candidate_registry_entry(
        promotion_writer_manifest=promotion_writer_manifest,
        promoted_entry=promoted_entry,
        promoted_index_records=promoted_index_records,
        publication_preflight_report=publication_preflight_report,
        policy=writer_policy,
    )
    publication_index_records = build_oled_publication_candidate_registry_index_records(publication_entry) if publication_entry is not None else []
    manifest = _manifest(
        policy=writer_policy,
        publication_entry=publication_entry,
        findings=findings,
        promotion_writer_manifest=promotion_writer_manifest,
        publication_preflight_report=publication_preflight_report,
        output_directory=None,
        publication_entry_written=False,
        publication_index_written=False,
    )
    return OledPromotedRegistryPublicationWriterReport(
        manifest=manifest,
        publication_entry=publication_entry,
        publication_index_records=publication_index_records,
        findings=findings,
    )


def write_oled_publication_candidate_registry_entry_json(
    entry: OledPublicationCandidateRegistryEntry,
    path: str | Path,
) -> str:
    payload = json.dumps(_sanitize_for_output(entry.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n"
    return _write_bytes(path, payload.encode("utf-8"))


def write_oled_publication_candidate_registry_index_jsonl(
    records: Iterable[OledPublicationCandidateRegistryIndexRecord],
    path: str | Path,
) -> str:
    ordered = sorted(records, key=lambda item: item.publication_entry_id)
    lines = [
        json.dumps(_sanitize_for_output(record.model_dump(mode="json", exclude_none=True)), sort_keys=True, separators=(",", ":"))
        for record in ordered
    ]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    return _write_bytes(path, payload)


def write_oled_promoted_registry_publication_manifest_json(
    manifest: OledPromotedRegistryPublicationWriterManifest,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_oled_publication_candidate_registry_entry_json(
    path: str | Path,
) -> OledPublicationCandidateRegistryEntry:
    entry_path = Path(path)
    _reject_forbidden_input(entry_path)
    if not entry_path.exists():
        raise ValueError(f"missing_publication_candidate_registry_entry_json:{redact_oled_mineru_acceptance_path(entry_path)}")
    try:
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
        if _contains_forbidden_payload_key(payload):
            raise ValueError("forbidden publication candidate registry entry payload")
        entry = OledPublicationCandidateRegistryEntry.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid_publication_candidate_registry_entry_json:{redact_oled_mineru_acceptance_path(entry_path)}") from exc
    if _contains_absolute_path(entry.model_dump(mode="json")):
        raise ValueError("absolute_path_in_publication_candidate_registry_entry")
    return entry


def load_oled_publication_candidate_registry_index_jsonl(
    path: str | Path,
) -> list[OledPublicationCandidateRegistryIndexRecord]:
    index_path = Path(path)
    _reject_forbidden_input(index_path)
    if not index_path.exists():
        raise ValueError(f"missing_publication_candidate_registry_index_jsonl:{redact_oled_mineru_acceptance_path(index_path)}")
    records: list[OledPublicationCandidateRegistryIndexRecord] = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if _contains_forbidden_payload_key(payload):
                raise ValueError("forbidden publication candidate registry index payload")
            record = OledPublicationCandidateRegistryIndexRecord.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"invalid_publication_candidate_registry_index_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(record.model_dump(mode="json")):
            raise ValueError("absolute_path_in_publication_candidate_registry_index")
        records.append(record)
    return records


def oled_publication_candidate_registry_entry_filename() -> str:
    return "oled_publication_candidate_registry_entry.json"


def oled_publication_candidate_registry_index_filename() -> str:
    return "oled_publication_candidate_registry_index.jsonl"


def run_oled_promoted_registry_publication_writer_from_files(
    *,
    promotion_writer_manifest_path: str | Path,
    publication_preflight_report_path: str | Path,
    promoted_registry_base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    policy: OledPromotedRegistryPublicationWriterPolicy | None = None,
    confirm_promoted_registry_publication_write: bool = False,
    dry_run: bool = False,
) -> OledPromotedRegistryPublicationWriterReport:
    writer_policy = policy or OledPromotedRegistryPublicationWriterPolicy()
    if not output_dir and not output_manifest_path:
        raise ValueError("output_required:dir_or_manifest")
    if not dry_run and writer_policy.require_confirmation and not confirm_promoted_registry_publication_write:
        raise ValueError("confirmation_required:promoted_registry_publication_write")

    promotion_writer_manifest = load_oled_benchmark_registry_promotion_writer_manifest_json(promotion_writer_manifest_path)
    base_dir = Path(promoted_registry_base_dir) if promoted_registry_base_dir is not None else Path(promotion_writer_manifest_path).parent
    promoted_entry, promoted_index_records = load_oled_promoted_registry_artifacts_from_manifest(
        manifest=promotion_writer_manifest,
        base_dir=base_dir,
    )
    if promoted_entry is None:
        raise ValueError("missing_promoted_registry_entry_json:from_manifest")
    publication_preflight_report = load_oled_promoted_registry_publication_preflight_report_json(publication_preflight_report_path)

    writer_report = select_oled_promoted_registry_publication_for_write(
        promotion_writer_manifest=promotion_writer_manifest,
        promoted_entry=promoted_entry,
        promoted_index_records=promoted_index_records,
        publication_preflight_report=publication_preflight_report,
        policy=writer_policy,
        confirm_promoted_registry_publication_write=confirm_promoted_registry_publication_write or dry_run,
    )
    if dry_run:
        writer_report = _mark_dry_run(writer_report)
    elif writer_report.publication_entry is not None and writer_report.is_valid:
        if output_dir is None:
            raise ValueError("output_dir_required:promoted_registry_publication_write")
        writer_report = _write_publication_files(writer_report, Path(output_dir))

    if output_manifest_path is not None:
        write_oled_promoted_registry_publication_manifest_json(writer_report.manifest, output_manifest_path)
    return writer_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write local OLED publication-candidate registry artifacts under an explicit gate.")
    parser.add_argument("--promotion-writer-manifest", required=True, help="Path to source promotion writer manifest JSON.")
    parser.add_argument("--publication-preflight-report", required=True, help="Path to publication-readiness preflight report JSON.")
    parser.add_argument("--promoted-registry-base-dir", help="Base directory for source promoted registry artifacts.")
    parser.add_argument("--output-dir", help="Output directory for publication-candidate registry artifacts.")
    parser.add_argument("--output-manifest", help="Optional publication writer manifest JSON path.")
    parser.add_argument("--confirm-promoted-registry-publication-write", action="store_true", help="Confirm publication-candidate artifact writing.")
    parser.add_argument("--dry-run", action="store_true", help="Build publication artifacts in memory and write only manifest if requested.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--entry-only", action="store_true", help="Write only publication-candidate entry JSON.")
    parser.add_argument("--index-only", action="store_true", help="Write only publication-candidate index JSONL.")
    args = parser.parse_args(argv)
    try:
        if not args.output_dir and not args.output_manifest:
            raise ValueError("output_required:dir_or_manifest")
        if args.entry_only and args.index_only:
            raise ValueError("conflicting_output_modes:entry_only,index_only")
        if not args.dry_run and not args.confirm_promoted_registry_publication_write:
            raise ValueError("confirmation_required:promoted_registry_publication_write")
        policy = OledPromotedRegistryPublicationWriterPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            write_publication_entry_json=not args.index_only,
            write_publication_index_jsonl=not args.entry_only,
        )
        report = run_oled_promoted_registry_publication_writer_from_files(
            promotion_writer_manifest_path=args.promotion_writer_manifest,
            publication_preflight_report_path=args.publication_preflight_report,
            promoted_registry_base_dir=args.promoted_registry_base_dir,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            policy=policy,
            confirm_promoted_registry_publication_write=args.confirm_promoted_registry_publication_write,
            dry_run=args.dry_run,
        )
        summary = {
            "status": "valid" if report.is_valid else "invalid",
            "publication_entry_selected": report.publication_entry is not None,
            "publication_index_record_count": len(report.publication_index_records),
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
    promotion_writer_manifest: OledBenchmarkRegistryPromotionWriterManifest,
    promoted_entry: OledBenchmarkPromotedRegistryEntry,
    promoted_index_records: list[OledBenchmarkPromotedRegistryIndexRecord],
    publication_preflight_report: OledPromotedRegistryPublicationPreflightReport,
    policy: OledPromotedRegistryPublicationWriterPolicy,
) -> list[OledPromotedRegistryPublicationWriterFinding]:
    findings: list[OledPromotedRegistryPublicationWriterFinding] = []
    if policy.require_publication_preflight_valid and publication_preflight_report.status == OledPromotedRegistryPublicationPreflightStatus.FAILED:
        findings.append(_finding("publication_preflight_failed", "error", "publication preflight failed"))
    if not policy.allow_publication_preflight_warnings and _warning_codes(publication_preflight_report):
        findings.append(_finding("publication_preflight_warnings_present", "error", "publication preflight has warnings"))
    if policy.require_promoted_entry and not promoted_entry.promoted_entry_id:
        findings.append(_finding("missing_promoted_entry", "error", "source promoted entry is required"))
    if policy.require_promoted_index and not promoted_index_records:
        findings.append(_finding("missing_promoted_index", "error", "source promoted index is required"))
    findings.extend(_artifact_sha_findings(promotion_writer_manifest, policy))
    if policy.require_promoted_candidate_status and _status_value(promoted_entry.promotion_status) != OledBenchmarkPromotedRegistryEntryStatus.PROMOTED_CANDIDATE.value:
        findings.append(
            _finding(
                "promotion_status_not_promoted_candidate",
                "error",
                "source promoted entry status is not promoted_candidate",
                source_promoted_entry_id=promoted_entry.promoted_entry_id,
            )
        )
    if policy.require_source_registry_writer_manifest_id and not promoted_entry.source_registry_writer_manifest_id:
        findings.append(_finding("missing_source_registry_writer_manifest_id", "error", "source registry writer manifest id is required"))
    if policy.require_source_registry_entry_id and not promoted_entry.source_registry_entry_id:
        findings.append(_finding("missing_source_registry_entry_id", "error", "source registry entry id is required"))
    if policy.require_source_registry_promotion_preflight_status and not promoted_entry.source_registry_promotion_preflight_status:
        findings.append(_finding("missing_source_registry_promotion_preflight_status", "error", "source registry promotion preflight status is required"))
    if policy.require_source_candidate_report_id and not promoted_entry.source_candidate_report_id:
        findings.append(_finding("missing_source_candidate_report_id", "error", "source candidate report id is required"))
    if policy.require_source_benchmark_report_manifest_id and not promoted_entry.source_benchmark_report_manifest_id:
        findings.append(_finding("missing_source_benchmark_report_manifest_id", "error", "source benchmark report manifest id is required"))
    if policy.require_caveats:
        caveats = set(promoted_entry.caveats)
        for caveat in _REQUIRED_CAVEATS:
            if caveat not in caveats:
                findings.append(
                    _finding(
                        "missing_required_caveat",
                        "error",
                        "source promoted entry lacks required caveat",
                        source_promoted_entry_id=promoted_entry.promoted_entry_id,
                    )
                )
    if policy.require_run_cards and promoted_entry.run_card_count <= 0:
        findings.append(_finding("missing_run_cards", "error", "source promoted entry has no run cards", source_promoted_entry_id=promoted_entry.promoted_entry_id))
    if policy.require_metric_cards and promoted_entry.metric_card_count <= 0:
        findings.append(_finding("missing_metric_cards", "error", "source promoted entry has no metric cards", source_promoted_entry_id=promoted_entry.promoted_entry_id))
    if bool(policy.benchmark_validated):
        findings.append(_finding("benchmark_validated_source_claim", "error", "publication writer policy cannot benchmark-validate outputs"))
    if bool(policy.scientific_claim_validated):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "publication writer policy cannot validate scientific claims"))
    if bool(policy.globally_registered) or policy.publication_status != "publication_candidate":
        findings.append(_finding("publication_source_claim", "error", "publication writer policy cannot globally register or publish outputs"))
    if policy.require_no_benchmark_validated_claims and (
        _truthy_metadata_key("benchmark_validated", promotion_writer_manifest.metadata)
        or _truthy_metadata_key("benchmark_validated", promoted_entry.metadata)
        or _truthy_metadata_key("benchmark_validated", publication_preflight_report.metadata)
        or any(record.benchmark_validated or _truthy_metadata_key("benchmark_validated", record.metadata) for record in promoted_index_records)
    ):
        findings.append(_finding("benchmark_validated_source_claim", "error", "source metadata claims benchmark validation"))
    if policy.require_no_scientific_claims and (
        _truthy_metadata_key("scientific_claim_validated", promotion_writer_manifest.metadata)
        or _truthy_metadata_key("scientific_claim_validated", promoted_entry.metadata)
        or _truthy_metadata_key("scientific_claim_validated", publication_preflight_report.metadata)
        or any(record.scientific_claim_validated or _truthy_metadata_key("scientific_claim_validated", record.metadata) for record in promoted_index_records)
    ):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "source metadata claims scientific validation"))
    if policy.require_no_publication_claims and (
        _metadata_claims_publication(promotion_writer_manifest.metadata)
        or _metadata_claims_publication(promoted_entry.metadata)
        or _metadata_claims_publication(publication_preflight_report.metadata)
        or any(_metadata_claims_publication(record.metadata) for record in promoted_index_records)
    ):
        findings.append(_finding("publication_source_claim", "error", "source metadata claims publication or global registration"))
    return findings


def _artifact_sha_findings(
    manifest: OledBenchmarkRegistryPromotionWriterManifest,
    policy: OledPromotedRegistryPublicationWriterPolicy,
) -> list[OledPromotedRegistryPublicationWriterFinding]:
    findings: list[OledPromotedRegistryPublicationWriterFinding] = []
    entry_result = _file_result_for_kind(manifest.file_results, "promoted_registry_entry_json")
    index_result = _file_result_for_kind(manifest.file_results, "promoted_registry_index_jsonl")
    if policy.require_promoted_entry_sha256 and (entry_result is None or not entry_result.output_sha256):
        findings.append(_finding("missing_promoted_entry_sha256", "error", "source promoted entry SHA256 is required"))
    if policy.require_promoted_index_sha256 and (index_result is None or not index_result.output_sha256):
        findings.append(_finding("missing_promoted_index_sha256", "error", "source promoted index SHA256 is required"))
    return findings


def _manifest(
    *,
    policy: OledPromotedRegistryPublicationWriterPolicy,
    publication_entry: OledPublicationCandidateRegistryEntry | None,
    findings: list[OledPromotedRegistryPublicationWriterFinding],
    promotion_writer_manifest: OledBenchmarkRegistryPromotionWriterManifest,
    publication_preflight_report: OledPromotedRegistryPublicationPreflightReport,
    output_directory: str | None,
    publication_entry_written: bool,
    publication_index_written: bool,
    file_results: list[OledPromotedRegistryPublicationFileResult] | None = None,
) -> OledPromotedRegistryPublicationWriterManifest:
    return OledPromotedRegistryPublicationWriterManifest(
        manifest_id=_publication_entry_id(
            promotion_writer_manifest.manifest_id,
            publication_entry.source_promoted_entry_id if publication_entry is not None else None,
            publication_preflight_report.status,
        ).replace("entry:", "manifest:"),
        source_promotion_writer_manifest_id=promotion_writer_manifest.manifest_id,
        source_promoted_entry_id=publication_entry.source_promoted_entry_id if publication_entry is not None else None,
        source_publication_preflight_status=_status_value(publication_preflight_report.status),
        output_directory=output_directory,
        output_file_count=sum(1 for result in (file_results or []) if result.status == OledPromotedRegistryPublicationWriteStatus.WRITTEN),
        publication_entry_ids=[publication_entry.publication_entry_id] if publication_entry is not None else [],
        baseline_kinds=publication_entry.baseline_kinds if publication_entry is not None else [],
        target_property_ids=publication_entry.target_property_ids if publication_entry is not None else [],
        feature_views=publication_entry.feature_views if publication_entry is not None else [],
        file_results=file_results or _selection_file_results(publication_entry, findings),
        policy=policy,
        metadata=_safety_metadata(
            publication_candidate_written=publication_entry_written or publication_index_written,
            publication_entry_written=publication_entry_written,
            publication_index_written=publication_index_written,
        ),
    )


def _selection_file_results(
    publication_entry: OledPublicationCandidateRegistryEntry | None,
    findings: list[OledPromotedRegistryPublicationWriterFinding],
) -> list[OledPromotedRegistryPublicationFileResult]:
    if publication_entry is None:
        return [
            OledPromotedRegistryPublicationFileResult(
                artifact_kind="publication_candidate_entry_json",
                status=OledPromotedRegistryPublicationWriteStatus.REJECTED,
                reason_codes=sorted({finding.code for finding in findings} or {"publication_candidate_entry_rejected"}),
            )
        ]
    return [
        OledPromotedRegistryPublicationFileResult(
            artifact_kind="publication_candidate_entry_json",
            status=OledPromotedRegistryPublicationWriteStatus.SKIPPED,
            reason_codes=["selected_for_publication_candidate"],
        ),
        OledPromotedRegistryPublicationFileResult(
            artifact_kind="publication_candidate_index_jsonl",
            status=OledPromotedRegistryPublicationWriteStatus.SKIPPED,
            reason_codes=["selected_for_publication_candidate"],
        ),
    ]


def _write_publication_files(
    writer_report: OledPromotedRegistryPublicationWriterReport,
    output_root: Path,
) -> OledPromotedRegistryPublicationWriterReport:
    assert writer_report.publication_entry is not None
    publication_entry = writer_report.publication_entry
    file_results: list[OledPromotedRegistryPublicationFileResult] = []
    publication_index_records = list(writer_report.publication_index_records)
    if writer_report.manifest.policy.write_publication_entry_json:
        entry_path = output_root / oled_publication_candidate_registry_entry_filename()
        entry_sha = write_oled_publication_candidate_registry_entry_json(
            publication_entry.model_copy(
                update={
                    "metadata": _entry_metadata(
                        publication_candidate_written=True,
                        publication_entry_written=True,
                        publication_index_written=writer_report.manifest.policy.write_publication_index_jsonl,
                    )
                }
            ),
            entry_path,
        )
        file_results.append(
            OledPromotedRegistryPublicationFileResult(
                artifact_kind="publication_candidate_entry_json",
                status=OledPromotedRegistryPublicationWriteStatus.WRITTEN,
                output_path=entry_path.name,
                output_sha256=entry_sha,
                reason_codes=["publication_candidate_entry_json_written", "selected_for_publication_candidate"],
            )
        )
        publication_index_records = [
            record.model_copy(
                update={
                    "output_publication_entry_json_path": entry_path.name,
                    "output_publication_entry_json_sha256": entry_sha,
                }
            )
            for record in publication_index_records
        ]
    if writer_report.manifest.policy.write_publication_index_jsonl:
        index_path = output_root / oled_publication_candidate_registry_index_filename()
        index_sha = write_oled_publication_candidate_registry_index_jsonl(publication_index_records, index_path)
        file_results.append(
            OledPromotedRegistryPublicationFileResult(
                artifact_kind="publication_candidate_index_jsonl",
                status=OledPromotedRegistryPublicationWriteStatus.WRITTEN,
                output_path=index_path.name,
                output_sha256=index_sha,
                reason_codes=["publication_candidate_index_jsonl_written", "selected_for_publication_candidate"],
            )
        )
    updated_entry = publication_entry.model_copy(
        update={
            "metadata": _entry_metadata(
                publication_candidate_written=bool(file_results),
                publication_entry_written=writer_report.manifest.policy.write_publication_entry_json,
                publication_index_written=writer_report.manifest.policy.write_publication_index_jsonl,
            )
        }
    )
    manifest = writer_report.manifest.model_copy(
        update={
            "output_directory": output_root.name,
            "output_file_count": len(file_results),
            "file_results": file_results,
            "metadata": _safety_metadata(
                publication_candidate_written=bool(file_results),
                publication_entry_written=writer_report.manifest.policy.write_publication_entry_json,
                publication_index_written=writer_report.manifest.policy.write_publication_index_jsonl,
            ),
        }
    )
    return writer_report.model_copy(update={"manifest": manifest, "publication_entry": updated_entry, "publication_index_records": publication_index_records})


def _mark_dry_run(
    writer_report: OledPromotedRegistryPublicationWriterReport,
) -> OledPromotedRegistryPublicationWriterReport:
    manifest = writer_report.manifest.model_copy(
        update={
            "metadata": {
                **writer_report.manifest.metadata,
                "dry_run_no_files_written": True,
                "publication_candidate_written": False,
                "publication_candidate_entry_written": False,
                "publication_candidate_index_written": False,
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


def _selected_values(values: Iterable[str], selected: Iterable[str]) -> list[str]:
    selected_set = {str(item).strip() for item in selected if str(item).strip()}
    return sorted({str(item) for item in values if not selected_set or str(item) in selected_set})


def _publication_entry_id(manifest_id: str | None, promoted_entry_id: str | None, status: Enum | str) -> str:
    return "entry:oled-publication-candidate-registry:" + _safe_id_token(f"{manifest_id or 'unknown'}:{promoted_entry_id or 'unknown'}:{_status_value(status)}")


def _safe_id_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", value).strip("-").lower() or "unknown"


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _warning_codes(report: OledPromotedRegistryPublicationPreflightReport) -> list[str]:
    codes: list[str] = []
    for finding in report.findings:
        if isinstance(finding, dict):
            if finding.get("severity") == "warning" and finding.get("code"):
                codes.append(str(finding["code"]))
        elif finding.severity == "warning":
            codes.append(finding.code)
    return codes


def _truthy_metadata_key(key: str, metadata: dict[str, Any]) -> bool:
    return bool(metadata.get(key))


def _metadata_claims_publication(metadata: dict[str, Any]) -> bool:
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


def _finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    *,
    publication_entry_id: str | None = None,
    source_promoted_entry_id: str | None = None,
    source_registry_entry_id: str | None = None,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    output_path: str | None = None,
) -> OledPromotedRegistryPublicationWriterFinding:
    return OledPromotedRegistryPublicationWriterFinding(
        code=code,
        severity=severity,
        message=message,
        publication_entry_id=publication_entry_id,
        source_promoted_entry_id=source_promoted_entry_id,
        source_registry_entry_id=source_registry_entry_id,
        baseline_kind=baseline_kind,
        target_property_id=target_property_id,
        feature_view=feature_view,
        output_path=output_path,
    )


def _dedup_findings(
    findings: list[OledPromotedRegistryPublicationWriterFinding],
) -> list[OledPromotedRegistryPublicationWriterFinding]:
    seen: set[tuple[str, str, str, str, str, str, str, str]] = set()
    output: list[OledPromotedRegistryPublicationWriterFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
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
    return sorted(output, key=lambda item: (item.severity, item.code, item.source_promoted_entry_id or "", item.source_registry_entry_id or ""))


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


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


def _entry_metadata(
    *,
    publication_candidate_written: bool = False,
    publication_entry_written: bool = False,
    publication_index_written: bool = False,
) -> dict[str, Any]:
    return {
        "promoted_registry_publication_writer": True,
        "publication_candidate_written": publication_candidate_written,
        "publication_candidate_entry_written": publication_entry_written,
        "publication_candidate_index_written": publication_index_written,
        "publication_candidate_entry": True,
        "publication_status": "publication_candidate",
        "final_registry_written": False,
        "global_registry_mutated": False,
        "benchmark_published": False,
        "benchmark_registered": False,
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


def _safety_metadata(
    *,
    publication_candidate_written: bool,
    publication_entry_written: bool,
    publication_index_written: bool,
) -> dict[str, Any]:
    return {
        "promoted_registry_publication_writer": True,
        "publication_candidate_written": publication_candidate_written,
        "publication_candidate_entry_written": publication_entry_written,
        "publication_candidate_index_written": publication_index_written,
        "publication_status": "publication_candidate",
        "final_registry_written": False,
        "global_registry_mutated": False,
        "benchmark_published": False,
        "benchmark_registered": False,
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
    "OledPromotedRegistryPublicationWriterPolicy",
    "OledPromotedRegistryPublicationWriteStatus",
    "OledPublicationCandidateRegistryEntryStatus",
    "OledPublicationCandidateRegistryEntry",
    "OledPublicationCandidateRegistryIndexRecord",
    "OledPromotedRegistryPublicationFileResult",
    "OledPromotedRegistryPublicationWriterFinding",
    "OledPromotedRegistryPublicationWriterManifest",
    "OledPromotedRegistryPublicationWriterReport",
    "load_oled_promoted_registry_publication_preflight_report_json",
    "build_oled_publication_candidate_registry_entry",
    "build_oled_publication_candidate_registry_index_records",
    "select_oled_promoted_registry_publication_for_write",
    "write_oled_publication_candidate_registry_entry_json",
    "write_oled_publication_candidate_registry_index_jsonl",
    "write_oled_promoted_registry_publication_manifest_json",
    "load_oled_publication_candidate_registry_entry_json",
    "load_oled_publication_candidate_registry_index_jsonl",
    "oled_publication_candidate_registry_entry_filename",
    "oled_publication_candidate_registry_index_filename",
    "run_oled_promoted_registry_publication_writer_from_files",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
