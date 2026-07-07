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

from ai4s_agent.domains.oled_curated_final_registry_global_append_preflight import (
    OledFinalRegistryExistingRecordSummary,
    load_oled_existing_final_registry_snapshot_jsonl,
)
from ai4s_agent.domains.oled_curated_final_registry_global_append_writer import (
    OledFinalRegistryGlobalAppendWriterManifest,
    OledGlobalAppendCandidateEntry,
    OledGlobalAppendCandidateEntryStatus,
    OledGlobalAppendCandidateIndexRecord,
)
from ai4s_agent.domains.oled_curated_global_append_release_preflight import (
    OledGlobalAppendReleasePreflightReport,
    OledGlobalAppendReleasePreflightStatus,
    load_oled_final_registry_global_append_writer_manifest_json,
    load_oled_global_append_artifacts_from_manifest,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledGlobalAppendReleaseWriterPolicy(BaseModel):
    require_confirmation: bool = True
    require_release_preflight_valid: bool = True
    allow_release_preflight_warnings: bool = True

    require_global_append_entry: bool = True
    require_global_append_delta: bool = True
    require_global_registry_snapshot: bool = True
    require_global_append_entry_sha256: bool = True
    require_global_append_delta_sha256: bool = True
    require_global_registry_snapshot_sha256: bool = True

    require_global_append_candidate_status: bool = True
    require_source_final_registry_writer_manifest_id: bool = True
    require_source_final_registry_entry_id: bool = True
    require_source_global_append_preflight_status: bool = True
    require_source_publication_writer_manifest_id: bool = True
    require_source_publication_entry_id: bool = True
    require_source_promoted_entry_id: bool = True
    require_source_promotion_writer_manifest_id: bool = True
    require_source_registry_entry_id: bool = True
    require_source_registry_writer_manifest_id: bool = True
    require_source_candidate_report_id: bool = True
    require_source_benchmark_report_manifest_id: bool = True

    require_caveats: bool = True
    require_run_cards: bool = True
    require_metric_cards: bool = True

    require_no_benchmark_validated_claims: bool = True
    require_no_scientific_claims: bool = True
    require_no_external_publication_claims: bool = True

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)

    write_release_entry_json: bool = True
    write_release_delta_jsonl: bool = True
    write_release_snapshot_jsonl: bool = True

    release_status: Literal["release_candidate"] = "release_candidate"
    benchmark_validated: bool = False
    scientific_claim_validated: bool = False
    externally_published: bool = False


class OledGlobalAppendReleaseWriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class OledReleaseCandidateEntryStatus(str, Enum):
    RELEASE_CANDIDATE = "release_candidate"
    REJECTED = "rejected"


class OledReleaseCandidateEntry(BaseModel):
    release_entry_id: str
    release_status: OledReleaseCandidateEntryStatus = OledReleaseCandidateEntryStatus.RELEASE_CANDIDATE

    source_global_append_writer_manifest_id: str | None = None
    source_global_append_entry_id: str | None = None
    source_release_preflight_status: str | None = None

    source_final_registry_entry_id: str | None = None
    source_final_registry_writer_manifest_id: str | None = None
    source_publication_entry_id: str | None = None
    source_publication_writer_manifest_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_promotion_writer_manifest_id: str | None = None
    source_registry_entry_id: str | None = None
    source_registry_writer_manifest_id: str | None = None
    source_candidate_report_id: str | None = None
    source_benchmark_report_manifest_id: str | None = None
    source_global_append_preflight_status: str | None = None

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    run_card_count: int = 0
    metric_card_count: int = 0

    source_global_append_entry_json_path: str | None = None
    source_global_append_entry_json_sha256: str | None = None
    source_global_append_delta_jsonl_path: str | None = None
    source_global_append_delta_jsonl_sha256: str | None = None
    source_global_registry_snapshot_jsonl_path: str | None = None
    source_global_registry_snapshot_jsonl_sha256: str | None = None

    caveats: list[str] = Field(default_factory=list)
    release_reason_codes: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class OledReleaseCandidateDeltaRecord(BaseModel):
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

    output_release_entry_json_path: str | None = None
    output_release_entry_json_sha256: str | None = None

    benchmark_published: bool = False
    benchmark_registered: bool = False
    benchmark_validated: bool = False
    scientific_claim_validated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledGlobalAppendReleaseFileResult(BaseModel):
    artifact_kind: Literal[
        "release_candidate_entry_json",
        "release_candidate_delta_jsonl",
        "release_candidate_snapshot_jsonl",
        "manifest",
    ]

    status: OledGlobalAppendReleaseWriteStatus
    output_path: str | None = None
    output_sha256: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledGlobalAppendReleaseWriterFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    release_entry_id: str | None = None
    source_global_append_entry_id: str | None = None
    source_final_registry_entry_id: str | None = None
    source_publication_entry_id: str | None = None
    source_promoted_entry_id: str | None = None
    source_registry_entry_id: str | None = None
    baseline_kind: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    output_path: str | None = None


class OledGlobalAppendReleaseWriterManifest(BaseModel):
    manifest_id: str

    source_global_append_writer_manifest_id: str | None = None
    source_global_append_entry_id: str | None = None
    source_release_preflight_status: str | None = None

    prior_registry_snapshot_path: str | None = None
    prior_registry_record_count: int = 0
    release_snapshot_record_count: int = 0

    output_directory: str | None = None
    output_file_count: int = 0

    release_entry_ids: list[str] = Field(default_factory=list)

    baseline_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    file_results: list[OledGlobalAppendReleaseFileResult] = Field(default_factory=list)

    policy: OledGlobalAppendReleaseWriterPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(result.status == OledGlobalAppendReleaseWriteStatus.REJECTED for result in self.file_results)


class OledGlobalAppendReleaseWriterReport(BaseModel):
    manifest: OledGlobalAppendReleaseWriterManifest
    release_entry: OledReleaseCandidateEntry | None = None
    release_delta_records: list[OledReleaseCandidateDeltaRecord] = Field(default_factory=list)
    findings: list[OledGlobalAppendReleaseWriterFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.error_codes and self.manifest.is_valid

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_global_append_release_preflight_report_json(
    path: str | Path,
) -> OledGlobalAppendReleasePreflightReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_global_append_release_preflight_report:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = OledGlobalAppendReleasePreflightReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_global_append_release_preflight_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    if _contains_absolute_path(report.model_dump(mode="json")):
        raise ValueError("absolute_path_in_global_append_release_preflight_report")
    return report


def build_oled_release_candidate_entry(
    *,
    global_append_writer_manifest: OledFinalRegistryGlobalAppendWriterManifest,
    global_append_entry: OledGlobalAppendCandidateEntry,
    global_append_delta_records: Iterable[OledGlobalAppendCandidateIndexRecord],
    release_preflight_report: OledGlobalAppendReleasePreflightReport,
    existing_registry_records: Iterable[OledFinalRegistryExistingRecordSummary] | None = None,
    policy: OledGlobalAppendReleaseWriterPolicy | None = None,
) -> tuple[OledReleaseCandidateEntry | None, list[OledGlobalAppendReleaseWriterFinding]]:
    writer_policy = policy or OledGlobalAppendReleaseWriterPolicy()
    index_records = list(global_append_delta_records)
    existing_records = list(existing_registry_records or [])
    findings = _gate_findings(
        global_append_writer_manifest=global_append_writer_manifest,
        global_append_entry=global_append_entry,
        global_append_delta_records=index_records,
        release_preflight_report=release_preflight_report,
        existing_registry_records=existing_records,
        policy=writer_policy,
    )
    findings = _dedup_findings(findings)
    if any(finding.severity == "error" for finding in findings):
        return None, findings

    entry_result = _file_result_for_kind(global_append_writer_manifest.file_results, "global_append_entry_json")
    delta_result = _file_result_for_kind(global_append_writer_manifest.file_results, "global_append_delta_jsonl")
    snapshot_result = _file_result_for_kind(global_append_writer_manifest.file_results, "global_registry_snapshot_jsonl")
    release_entry = OledReleaseCandidateEntry(
        release_entry_id=_release_entry_id(
            global_append_writer_manifest.manifest_id,
            global_append_entry.global_append_entry_id,
            release_preflight_report.status,
        ),
        release_status=OledReleaseCandidateEntryStatus.RELEASE_CANDIDATE,
        source_global_append_writer_manifest_id=global_append_writer_manifest.manifest_id,
        source_global_append_entry_id=global_append_entry.global_append_entry_id,
        source_release_preflight_status=_status_value(release_preflight_report.status),
        source_final_registry_entry_id=global_append_entry.source_final_registry_entry_id,
        source_final_registry_writer_manifest_id=global_append_entry.source_final_registry_writer_manifest_id,
        source_publication_entry_id=global_append_entry.source_publication_entry_id,
        source_publication_writer_manifest_id=global_append_entry.source_publication_writer_manifest_id,
        source_promoted_entry_id=global_append_entry.source_promoted_entry_id,
        source_promotion_writer_manifest_id=global_append_entry.source_promotion_writer_manifest_id,
        source_registry_entry_id=global_append_entry.source_registry_entry_id,
        source_registry_writer_manifest_id=global_append_entry.source_registry_writer_manifest_id,
        source_candidate_report_id=global_append_entry.source_candidate_report_id,
        source_benchmark_report_manifest_id=global_append_entry.source_benchmark_report_manifest_id,
        source_global_append_preflight_status=global_append_entry.source_global_append_preflight_status,
        baseline_kinds=_selected_values(global_append_entry.baseline_kinds, writer_policy.baseline_kinds),
        target_property_ids=_selected_values(global_append_entry.target_property_ids, writer_policy.target_property_ids),
        feature_views=_selected_values(global_append_entry.feature_views, writer_policy.feature_views),
        run_card_count=global_append_entry.run_card_count,
        metric_card_count=global_append_entry.metric_card_count,
        source_global_append_entry_json_path=entry_result.output_path if entry_result is not None else None,
        source_global_append_entry_json_sha256=entry_result.output_sha256 if entry_result is not None else None,
        source_global_append_delta_jsonl_path=delta_result.output_path if delta_result is not None else None,
        source_global_append_delta_jsonl_sha256=delta_result.output_sha256 if delta_result is not None else None,
        source_global_registry_snapshot_jsonl_path=snapshot_result.output_path if snapshot_result is not None else None,
        source_global_registry_snapshot_jsonl_sha256=snapshot_result.output_sha256 if snapshot_result is not None else None,
        caveats=sorted(global_append_entry.caveats),
        release_reason_codes=["selected_for_release_candidate"],
        metadata=_entry_metadata(),
    )
    return release_entry, findings


def build_oled_release_candidate_delta_records(
    entry: OledReleaseCandidateEntry,
) -> list[OledReleaseCandidateDeltaRecord]:
    return [
        OledReleaseCandidateDeltaRecord(
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
            benchmark_published=False,
            benchmark_registered=False,
            benchmark_validated=False,
            scientific_claim_validated=False,
            metadata={
                "release_candidate_delta_record": True,
                "release_status": "release_candidate",
                "global_registry_mutated": False,
                "external_publication_written": False,
                "benchmark_published": False,
                "benchmark_registered": False,
                "benchmark_validated": False,
                "scientific_claim_validated": False,
            },
        )
    ]


def select_oled_global_append_release_for_write(
    *,
    global_append_writer_manifest: OledFinalRegistryGlobalAppendWriterManifest,
    global_append_entry: OledGlobalAppendCandidateEntry,
    global_append_delta_records: Iterable[OledGlobalAppendCandidateIndexRecord],
    release_preflight_report: OledGlobalAppendReleasePreflightReport,
    existing_registry_records: Iterable[OledFinalRegistryExistingRecordSummary] | None = None,
    policy: OledGlobalAppendReleaseWriterPolicy | None = None,
    confirm_global_append_release_write: bool = False,
) -> OledGlobalAppendReleaseWriterReport:
    writer_policy = policy or OledGlobalAppendReleaseWriterPolicy()
    if writer_policy.require_confirmation and not confirm_global_append_release_write:
        raise ValueError("confirmation_required:global_append_release_write")
    existing_records = list(existing_registry_records or [])
    release_entry, findings = build_oled_release_candidate_entry(
        global_append_writer_manifest=global_append_writer_manifest,
        global_append_entry=global_append_entry,
        global_append_delta_records=global_append_delta_records,
        release_preflight_report=release_preflight_report,
        existing_registry_records=existing_records,
        policy=writer_policy,
    )
    release_delta_records = build_oled_release_candidate_delta_records(release_entry) if release_entry is not None else []
    manifest = _manifest(
        policy=writer_policy,
        release_entry=release_entry,
        findings=findings,
        global_append_writer_manifest=global_append_writer_manifest,
        release_preflight_report=release_preflight_report,
        prior_registry_snapshot_path=None,
        prior_registry_record_count=len(existing_records),
        output_directory=None,
        release_entry_written=False,
        release_delta_written=False,
        release_snapshot_written=False,
    )
    return OledGlobalAppendReleaseWriterReport(
        manifest=manifest,
        release_entry=release_entry,
        release_delta_records=release_delta_records,
        findings=findings,
    )


def write_oled_release_candidate_entry_json(
    entry: OledReleaseCandidateEntry,
    path: str | Path,
) -> str:
    payload = json.dumps(_sanitize_for_output(entry.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n"
    return _write_bytes(path, payload.encode("utf-8"))


def write_oled_release_candidate_delta_jsonl(
    records: Iterable[OledReleaseCandidateDeltaRecord],
    path: str | Path,
) -> str:
    ordered = sorted(records, key=lambda item: item.release_entry_id)
    lines = [
        json.dumps(_sanitize_for_output(record.model_dump(mode="json", exclude_none=True)), sort_keys=True, separators=(",", ":"))
        for record in ordered
    ]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    return _write_bytes(path, payload)


def write_oled_release_candidate_snapshot_jsonl(
    snapshot_records: Iterable[OledFinalRegistryExistingRecordSummary],
    release_delta_records: Iterable[OledReleaseCandidateDeltaRecord],
    path: str | Path,
) -> str:
    lines: list[str] = []
    for record in snapshot_records:
        lines.append(json.dumps(_sanitize_for_output(record.model_dump(mode="json", exclude_none=True)), sort_keys=True, separators=(",", ":")))
    for record in sorted(release_delta_records, key=lambda item: item.release_entry_id):
        lines.append(json.dumps(_sanitize_for_output(record.model_dump(mode="json", exclude_none=True)), sort_keys=True, separators=(",", ":")))
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    return _write_bytes(path, payload)


def write_oled_global_append_release_manifest_json(
    manifest: OledGlobalAppendReleaseWriterManifest,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_oled_release_candidate_entry_json(
    path: str | Path,
) -> OledReleaseCandidateEntry:
    entry_path = Path(path)
    _reject_forbidden_input(entry_path)
    if not entry_path.exists():
        raise ValueError(f"missing_release_candidate_entry_json:{redact_oled_mineru_acceptance_path(entry_path)}")
    try:
        payload = json.loads(entry_path.read_text(encoding="utf-8"))
        if _contains_forbidden_payload_key(payload):
            raise ValueError("forbidden global append candidate entry payload")
        entry = OledReleaseCandidateEntry.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"invalid_release_candidate_entry_json:{redact_oled_mineru_acceptance_path(entry_path)}") from exc
    if _contains_absolute_path(entry.model_dump(mode="json")):
        raise ValueError("absolute_path_in_release_candidate_entry")
    return entry


def load_oled_release_candidate_delta_jsonl(
    path: str | Path,
) -> list[OledReleaseCandidateDeltaRecord]:
    delta_path = Path(path)
    _reject_forbidden_input(delta_path)
    if not delta_path.exists():
        raise ValueError(f"missing_release_candidate_delta_jsonl:{redact_oled_mineru_acceptance_path(delta_path)}")
    records: list[OledReleaseCandidateDeltaRecord] = []
    for line_number, line in enumerate(delta_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if _contains_forbidden_payload_key(payload):
                raise ValueError("forbidden global append candidate delta payload")
            record = OledReleaseCandidateDeltaRecord.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise ValueError(f"invalid_release_candidate_delta_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(record.model_dump(mode="json")):
            raise ValueError("absolute_path_in_release_candidate_delta")
        records.append(record)
    return records


def oled_release_candidate_entry_filename() -> str:
    return "oled_release_candidate_entry.json"


def oled_release_candidate_delta_filename() -> str:
    return "oled_release_candidate_delta.jsonl"


def oled_release_candidate_snapshot_filename() -> str:
    return "oled_release_candidate_snapshot.jsonl"


def run_oled_global_append_release_writer_from_files(
    *,
    global_append_writer_manifest_path: str | Path,
    release_preflight_report_path: str | Path,
    global_append_base_dir: str | Path | None = None,
    prior_registry_snapshot_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    policy: OledGlobalAppendReleaseWriterPolicy | None = None,
    confirm_global_append_release_write: bool = False,
    dry_run: bool = False,
) -> OledGlobalAppendReleaseWriterReport:
    writer_policy = policy or OledGlobalAppendReleaseWriterPolicy()
    if not output_dir and not output_manifest_path:
        raise ValueError("output_required:dir_or_manifest")
    if not dry_run and writer_policy.require_confirmation and not confirm_global_append_release_write:
        raise ValueError("confirmation_required:global_append_release_write")

    global_append_writer_manifest = load_oled_final_registry_global_append_writer_manifest_json(global_append_writer_manifest_path)
    base_dir = Path(global_append_base_dir) if global_append_base_dir is not None else Path(global_append_writer_manifest_path).parent
    global_append_entry, global_append_delta_records, global_registry_snapshot_records = load_oled_global_append_artifacts_from_manifest(
        manifest=global_append_writer_manifest,
        base_dir=base_dir,
    )
    if global_append_entry is None:
        raise ValueError("missing_global_append_candidate_entry_json:from_manifest")
    release_preflight_report = load_oled_global_append_release_preflight_report_json(release_preflight_report_path)
    snapshot_records = (
        load_oled_existing_final_registry_snapshot_jsonl(prior_registry_snapshot_path)
        if prior_registry_snapshot_path is not None
        else list(global_registry_snapshot_records)
    )

    writer_report = select_oled_global_append_release_for_write(
        global_append_writer_manifest=global_append_writer_manifest,
        global_append_entry=global_append_entry,
        global_append_delta_records=global_append_delta_records,
        release_preflight_report=release_preflight_report,
        existing_registry_records=snapshot_records,
        policy=writer_policy,
        confirm_global_append_release_write=confirm_global_append_release_write or dry_run,
    )
    writer_report = writer_report.model_copy(
        update={
            "manifest": writer_report.manifest.model_copy(
                update={
                    "prior_registry_snapshot_path": Path(prior_registry_snapshot_path).name if prior_registry_snapshot_path is not None else None,
                    "prior_registry_record_count": len(snapshot_records),
                    "release_snapshot_record_count": len(snapshot_records) + len(writer_report.release_delta_records),
                }
            )
        }
    )
    if dry_run:
        writer_report = _mark_dry_run(writer_report)
    elif writer_report.release_entry is not None and writer_report.is_valid:
        if output_dir is None:
            raise ValueError("output_dir_required:global_append_release_write")
        writer_report = _write_release_files(writer_report, Path(output_dir), snapshot_records)

    if output_manifest_path is not None:
        write_oled_global_append_release_manifest_json(writer_report.manifest, output_manifest_path)
    return writer_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write local OLED release-candidate artifacts under an explicit gate.")
    parser.add_argument("--global-append-writer-manifest", required=True, help="Path to source global-append writer manifest JSON.")
    parser.add_argument("--release-preflight-report", required=True, help="Path to release-readiness preflight report JSON.")
    parser.add_argument("--global-append-base-dir", help="Base directory for source global-append candidate artifacts.")
    parser.add_argument("--prior-registry-snapshot", help="Optional prior registry snapshot JSONL.")
    parser.add_argument("--output-dir", help="Output directory for global-append candidate artifacts.")
    parser.add_argument("--output-manifest", help="Optional global append writer manifest JSON path.")
    parser.add_argument(
        "--confirm-global-append-release-write",
        action="store_true",
        help="Confirm local global-append-candidate artifact writing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build global-append artifacts in memory and write only manifest if requested.")
    parser.add_argument("--baseline-kind", action="append", default=[], help="Baseline kind; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--entry-only", action="store_true", help="Write only release candidate entry JSON.")
    parser.add_argument("--delta-only", action="store_true", help="Write only release candidate delta JSONL.")
    parser.add_argument("--snapshot-only", action="store_true", help="Write only release candidate snapshot JSONL.")
    args = parser.parse_args(argv)
    try:
        if not args.output_dir and not args.output_manifest:
            raise ValueError("output_required:dir_or_manifest")
        selected_modes = sum(bool(flag) for flag in (args.entry_only, args.delta_only, args.snapshot_only))
        if selected_modes > 1:
            raise ValueError("conflicting_output_modes:entry_only,delta_only,snapshot_only")
        if not args.dry_run and not args.confirm_global_append_release_write:
            raise ValueError("confirmation_required:global_append_release_write")
        policy = OledGlobalAppendReleaseWriterPolicy(
            baseline_kinds=_split_cli_values(args.baseline_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            write_release_entry_json=not args.delta_only and not args.snapshot_only,
            write_release_delta_jsonl=not args.entry_only and not args.snapshot_only,
            write_release_snapshot_jsonl=not args.entry_only and not args.delta_only,
        )
        report = run_oled_global_append_release_writer_from_files(
            global_append_writer_manifest_path=args.global_append_writer_manifest,
            release_preflight_report_path=args.release_preflight_report,
            global_append_base_dir=args.global_append_base_dir,
            prior_registry_snapshot_path=args.prior_registry_snapshot,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            policy=policy,
            confirm_global_append_release_write=args.confirm_global_append_release_write,
            dry_run=args.dry_run,
        )
        summary = {
            "status": "valid" if report.is_valid else "invalid",
            "release_entry_selected": report.release_entry is not None,
            "release_delta_record_count": len(report.release_delta_records),
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
    global_append_writer_manifest: OledFinalRegistryGlobalAppendWriterManifest,
    global_append_entry: OledGlobalAppendCandidateEntry,
    global_append_delta_records: list[OledGlobalAppendCandidateIndexRecord],
    release_preflight_report: OledGlobalAppendReleasePreflightReport,
    existing_registry_records: list[OledFinalRegistryExistingRecordSummary],
    policy: OledGlobalAppendReleaseWriterPolicy,
) -> list[OledGlobalAppendReleaseWriterFinding]:
    findings: list[OledGlobalAppendReleaseWriterFinding] = []
    if policy.require_release_preflight_valid and release_preflight_report.status == OledGlobalAppendReleasePreflightStatus.FAILED:
        findings.append(_finding("release_preflight_failed", "error", "release preflight failed"))
    if not policy.allow_release_preflight_warnings and _warning_codes(release_preflight_report):
        findings.append(_finding("release_preflight_warnings_present", "error", "release preflight has warnings"))
    if policy.require_global_append_entry and not global_append_entry.global_append_entry_id:
        findings.append(_finding("missing_global_append_entry", "error", "source global-append entry is required"))
    if policy.require_global_append_delta and not global_append_delta_records:
        findings.append(_finding("missing_global_append_delta", "error", "source global-append delta is required"))
    findings.extend(_artifact_sha_findings(global_append_writer_manifest, policy))
    if (
        policy.require_global_append_candidate_status
        and _status_value(global_append_entry.global_append_status) != OledGlobalAppendCandidateEntryStatus.GLOBAL_APPEND_CANDIDATE.value
    ):
        findings.append(
            _finding(
                "global_append_status_not_candidate",
                "error",
                "source global-append entry status is not global_append_candidate",
                source_global_append_entry_id=global_append_entry.global_append_entry_id,
            )
        )
    if policy.require_source_final_registry_writer_manifest_id and not global_append_entry.source_final_registry_writer_manifest_id:
        findings.append(_finding("missing_source_final_registry_writer_manifest_id", "error", "source final-registry writer manifest id is required"))
    if policy.require_source_final_registry_entry_id and not global_append_entry.source_final_registry_entry_id:
        findings.append(_finding("missing_source_final_registry_entry_id", "error", "source final-registry entry id is required"))
    if policy.require_source_global_append_preflight_status and not global_append_entry.source_global_append_preflight_status:
        findings.append(_finding("missing_source_global_append_preflight_status", "error", "source global-append preflight status is required"))
    if policy.require_source_publication_writer_manifest_id and not global_append_entry.source_publication_writer_manifest_id:
        findings.append(_finding("missing_source_publication_writer_manifest_id", "error", "source publication writer manifest id is required"))
    if policy.require_source_publication_entry_id and not global_append_entry.source_publication_entry_id:
        findings.append(_finding("missing_source_publication_entry_id", "error", "source publication entry id is required"))
    if policy.require_source_promoted_entry_id and not global_append_entry.source_promoted_entry_id:
        findings.append(_finding("missing_source_promoted_entry_id", "error", "source promoted entry id is required"))
    if policy.require_source_promotion_writer_manifest_id and not global_append_entry.source_promotion_writer_manifest_id:
        findings.append(_finding("missing_source_promotion_writer_manifest_id", "error", "source promotion writer manifest id is required"))
    if policy.require_source_registry_entry_id and not global_append_entry.source_registry_entry_id:
        findings.append(_finding("missing_source_registry_entry_id", "error", "source registry entry id is required"))
    if policy.require_source_registry_writer_manifest_id and not global_append_entry.source_registry_writer_manifest_id:
        findings.append(_finding("missing_source_registry_writer_manifest_id", "error", "source registry writer manifest id is required"))
    if policy.require_source_candidate_report_id and not global_append_entry.source_candidate_report_id:
        findings.append(_finding("missing_source_candidate_report_id", "error", "source candidate report id is required"))
    if policy.require_source_benchmark_report_manifest_id and not global_append_entry.source_benchmark_report_manifest_id:
        findings.append(_finding("missing_source_benchmark_report_manifest_id", "error", "source benchmark report manifest id is required"))
    if policy.require_caveats:
        caveats = set(global_append_entry.caveats)
        for caveat in _REQUIRED_CAVEATS:
            if caveat not in caveats:
                findings.append(
                    _finding(
                        "missing_required_caveat",
                        "error",
                        "source global-append entry lacks required caveat",
                        source_global_append_entry_id=global_append_entry.global_append_entry_id,
                    )
                )
    if policy.require_run_cards and global_append_entry.run_card_count <= 0:
        findings.append(
            _finding(
                "missing_run_cards",
                "error",
                "source global-append entry has no run cards",
                source_global_append_entry_id=global_append_entry.global_append_entry_id,
            )
        )
    if policy.require_metric_cards and global_append_entry.metric_card_count <= 0:
        findings.append(
            _finding(
                "missing_metric_cards",
                "error",
                "source global-append entry has no metric cards",
                source_global_append_entry_id=global_append_entry.global_append_entry_id,
            )
        )
    if bool(policy.benchmark_validated):
        findings.append(_finding("benchmark_validated_source_claim", "error", "global append writer policy cannot benchmark-validate outputs"))
    if bool(policy.scientific_claim_validated):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "global append writer policy cannot validate scientific claims"))
    if bool(policy.externally_published) or policy.release_status != "release_candidate":
        findings.append(_finding("external_publication_source_claim", "error", "global append writer policy cannot publish outputs externally"))
    if policy.require_no_benchmark_validated_claims and (
        _truthy_metadata_key("benchmark_validated", global_append_writer_manifest.metadata)
        or _truthy_metadata_key("benchmark_validated", global_append_entry.metadata)
        or _truthy_metadata_key("benchmark_validated", release_preflight_report.metadata)
        or any(record.benchmark_validated or _truthy_metadata_key("benchmark_validated", record.metadata) for record in global_append_delta_records)
        or any(_truthy_metadata_key("benchmark_validated", record.metadata) for record in existing_registry_records)
    ):
        findings.append(_finding("benchmark_validated_source_claim", "error", "source metadata claims benchmark validation"))
    if policy.require_no_scientific_claims and (
        _truthy_metadata_key("scientific_claim_validated", global_append_writer_manifest.metadata)
        or _truthy_metadata_key("scientific_claim_validated", global_append_entry.metadata)
        or _truthy_metadata_key("scientific_claim_validated", release_preflight_report.metadata)
        or any(record.scientific_claim_validated or _truthy_metadata_key("scientific_claim_validated", record.metadata) for record in global_append_delta_records)
        or any(_truthy_metadata_key("scientific_claim_validated", record.metadata) for record in existing_registry_records)
    ):
        findings.append(_finding("scientific_claim_validated_source_claim", "error", "source metadata claims scientific validation"))
    if policy.require_no_external_publication_claims and (
        _metadata_claims_external_publication(global_append_writer_manifest.metadata)
        or _metadata_claims_external_publication(global_append_entry.metadata)
        or _metadata_claims_external_publication(release_preflight_report.metadata)
        or any(record.benchmark_published or record.benchmark_registered or _metadata_claims_external_publication(record.metadata) for record in global_append_delta_records)
        or any(_metadata_claims_external_publication(record.metadata) for record in existing_registry_records)
    ):
        findings.append(_finding("external_publication_source_claim", "error", "source metadata claims publication or global registry mutation"))
    return findings


def _artifact_sha_findings(
    manifest: OledFinalRegistryGlobalAppendWriterManifest,
    policy: OledGlobalAppendReleaseWriterPolicy,
) -> list[OledGlobalAppendReleaseWriterFinding]:
    findings: list[OledGlobalAppendReleaseWriterFinding] = []
    entry_result = _file_result_for_kind(manifest.file_results, "global_append_entry_json")
    delta_result = _file_result_for_kind(manifest.file_results, "global_append_delta_jsonl")
    snapshot_result = _file_result_for_kind(manifest.file_results, "global_registry_snapshot_jsonl")
    if policy.require_global_append_entry_sha256 and (entry_result is None or not entry_result.output_sha256):
        findings.append(_finding("missing_global_append_entry_sha256", "error", "source global-append entry SHA256 is required"))
    if policy.require_global_append_delta_sha256 and (delta_result is None or not delta_result.output_sha256):
        findings.append(_finding("missing_global_append_delta_sha256", "error", "source global-append delta SHA256 is required"))
    if policy.require_global_registry_snapshot_sha256 and (snapshot_result is None or not snapshot_result.output_sha256):
        findings.append(_finding("missing_global_registry_snapshot_sha256", "error", "source global registry snapshot SHA256 is required"))
    return findings


def _manifest(
    *,
    policy: OledGlobalAppendReleaseWriterPolicy,
    release_entry: OledReleaseCandidateEntry | None,
    findings: list[OledGlobalAppendReleaseWriterFinding],
    global_append_writer_manifest: OledFinalRegistryGlobalAppendWriterManifest,
    release_preflight_report: OledGlobalAppendReleasePreflightReport,
    prior_registry_snapshot_path: str | None,
    prior_registry_record_count: int,
    output_directory: str | None,
    release_entry_written: bool,
    release_delta_written: bool,
    release_snapshot_written: bool,
    file_results: list[OledGlobalAppendReleaseFileResult] | None = None,
) -> OledGlobalAppendReleaseWriterManifest:
    return OledGlobalAppendReleaseWriterManifest(
        manifest_id=_release_entry_id(
            global_append_writer_manifest.manifest_id,
            release_entry.source_global_append_entry_id if release_entry is not None else None,
            release_preflight_report.status,
        ).replace("entry:", "manifest:"),
        source_global_append_writer_manifest_id=global_append_writer_manifest.manifest_id,
        source_global_append_entry_id=release_entry.source_global_append_entry_id if release_entry is not None else None,
        source_release_preflight_status=_status_value(release_preflight_report.status),
        prior_registry_snapshot_path=prior_registry_snapshot_path,
        prior_registry_record_count=prior_registry_record_count,
        release_snapshot_record_count=prior_registry_record_count + (1 if release_entry is not None else 0),
        output_directory=output_directory,
        output_file_count=sum(1 for result in (file_results or []) if result.status == OledGlobalAppendReleaseWriteStatus.WRITTEN),
        release_entry_ids=[release_entry.release_entry_id] if release_entry is not None else [],
        baseline_kinds=release_entry.baseline_kinds if release_entry is not None else [],
        target_property_ids=release_entry.target_property_ids if release_entry is not None else [],
        feature_views=release_entry.feature_views if release_entry is not None else [],
        file_results=file_results or _selection_file_results(release_entry, findings),
        policy=policy,
        metadata=_safety_metadata(
            release_candidate_written=release_entry_written or release_delta_written or release_snapshot_written,
            release_entry_written=release_entry_written,
            release_delta_written=release_delta_written,
            release_snapshot_written=release_snapshot_written,
        ),
    )


def _selection_file_results(
    release_entry: OledReleaseCandidateEntry | None,
    findings: list[OledGlobalAppendReleaseWriterFinding],
) -> list[OledGlobalAppendReleaseFileResult]:
    if release_entry is None:
        return [
            OledGlobalAppendReleaseFileResult(
                artifact_kind="release_candidate_entry_json",
                status=OledGlobalAppendReleaseWriteStatus.REJECTED,
                reason_codes=sorted({finding.code for finding in findings} or {"release_candidate_rejected"}),
            )
        ]
    return [
        OledGlobalAppendReleaseFileResult(
            artifact_kind="release_candidate_entry_json",
            status=OledGlobalAppendReleaseWriteStatus.SKIPPED,
            reason_codes=["selected_for_release_candidate"],
        ),
        OledGlobalAppendReleaseFileResult(
            artifact_kind="release_candidate_delta_jsonl",
            status=OledGlobalAppendReleaseWriteStatus.SKIPPED,
            reason_codes=["selected_for_release_candidate"],
        ),
        OledGlobalAppendReleaseFileResult(
            artifact_kind="release_candidate_snapshot_jsonl",
            status=OledGlobalAppendReleaseWriteStatus.SKIPPED,
            reason_codes=["selected_for_release_candidate"],
        ),
    ]


def _write_release_files(
    writer_report: OledGlobalAppendReleaseWriterReport,
    output_root: Path,
    existing_records: list[OledFinalRegistryExistingRecordSummary],
) -> OledGlobalAppendReleaseWriterReport:
    assert writer_report.release_entry is not None
    release_entry = writer_report.release_entry
    file_results: list[OledGlobalAppendReleaseFileResult] = []
    release_delta_records = list(writer_report.release_delta_records)
    if writer_report.manifest.policy.write_release_entry_json:
        entry_path = output_root / oled_release_candidate_entry_filename()
        entry_sha = write_oled_release_candidate_entry_json(
            release_entry.model_copy(
                update={
                    "metadata": _entry_metadata(
                        release_candidate_written=True,
                        release_entry_written=True,
                        release_delta_written=writer_report.manifest.policy.write_release_delta_jsonl,
                        release_snapshot_written=writer_report.manifest.policy.write_release_snapshot_jsonl,
                    )
                }
            ),
            entry_path,
        )
        file_results.append(
            OledGlobalAppendReleaseFileResult(
                artifact_kind="release_candidate_entry_json",
                status=OledGlobalAppendReleaseWriteStatus.WRITTEN,
                output_path=entry_path.name,
                output_sha256=entry_sha,
                reason_codes=["release_candidate_entry_json_written", "selected_for_release_candidate"],
            )
        )
        release_delta_records = [
            record.model_copy(
                update={
                    "output_release_entry_json_path": entry_path.name,
                    "output_release_entry_json_sha256": entry_sha,
                }
            )
            for record in release_delta_records
        ]
    if writer_report.manifest.policy.write_release_delta_jsonl:
        delta_path = output_root / oled_release_candidate_delta_filename()
        delta_sha = write_oled_release_candidate_delta_jsonl(release_delta_records, delta_path)
        file_results.append(
            OledGlobalAppendReleaseFileResult(
                artifact_kind="release_candidate_delta_jsonl",
                status=OledGlobalAppendReleaseWriteStatus.WRITTEN,
                output_path=delta_path.name,
                output_sha256=delta_sha,
                reason_codes=["release_candidate_delta_jsonl_written", "selected_for_release_candidate"],
            )
        )
    if writer_report.manifest.policy.write_release_snapshot_jsonl:
        snapshot_path = output_root / oled_release_candidate_snapshot_filename()
        snapshot_sha = write_oled_release_candidate_snapshot_jsonl(
            existing_records,
            release_delta_records,
            snapshot_path,
        )
        file_results.append(
            OledGlobalAppendReleaseFileResult(
                artifact_kind="release_candidate_snapshot_jsonl",
                status=OledGlobalAppendReleaseWriteStatus.WRITTEN,
                output_path=snapshot_path.name,
                output_sha256=snapshot_sha,
                reason_codes=["release_candidate_snapshot_jsonl_written", "selected_for_release_candidate"],
            )
        )
    updated_entry = release_entry.model_copy(
        update={
            "metadata": _entry_metadata(
                release_candidate_written=bool(file_results),
                release_entry_written=writer_report.manifest.policy.write_release_entry_json,
                release_delta_written=writer_report.manifest.policy.write_release_delta_jsonl,
                release_snapshot_written=writer_report.manifest.policy.write_release_snapshot_jsonl,
            )
        }
    )
    manifest = writer_report.manifest.model_copy(
        update={
            "output_directory": output_root.name,
            "output_file_count": len(file_results),
            "release_snapshot_record_count": len(existing_records) + len(release_delta_records),
            "file_results": file_results,
            "metadata": _safety_metadata(
                release_candidate_written=bool(file_results),
                release_entry_written=writer_report.manifest.policy.write_release_entry_json,
                release_delta_written=writer_report.manifest.policy.write_release_delta_jsonl,
                release_snapshot_written=writer_report.manifest.policy.write_release_snapshot_jsonl,
            ),
        }
    )
    return writer_report.model_copy(
        update={"manifest": manifest, "release_entry": updated_entry, "release_delta_records": release_delta_records}
    )


def _mark_dry_run(
    writer_report: OledGlobalAppendReleaseWriterReport,
) -> OledGlobalAppendReleaseWriterReport:
    manifest = writer_report.manifest.model_copy(
        update={
            "metadata": {
                **writer_report.manifest.metadata,
                "dry_run_no_files_written": True,
                "release_candidate_written": False,
                "release_entry_written": False,
                "release_delta_written": False,
                "release_snapshot_written": False,
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


def _release_entry_id(manifest_id: str | None, global_append_entry_id: str | None, status: Enum | str) -> str:
    return "entry:oled-release-candidate:" + _safe_id_token(f"{manifest_id or 'unknown'}:{global_append_entry_id or 'unknown'}:{_status_value(status)}")


def _safe_id_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "-", value).strip("-").lower() or "unknown"


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _warning_codes(report: OledGlobalAppendReleasePreflightReport) -> list[str]:
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


def _metadata_claims_external_publication(metadata: dict[str, Any]) -> bool:
    return any(
        bool(metadata.get(key))
        for key in (
            "benchmark_published",
            "benchmark_registered",
            "globally_registered",
            "global_registry_mutated",
            "final_registry_written",
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
    release_entry_id: str | None = None,
    source_global_append_entry_id: str | None = None,
    source_final_registry_entry_id: str | None = None,
    source_publication_entry_id: str | None = None,
    source_promoted_entry_id: str | None = None,
    source_registry_entry_id: str | None = None,
    baseline_kind: str | None = None,
    target_property_id: str | None = None,
    feature_view: str | None = None,
    output_path: str | None = None,
) -> OledGlobalAppendReleaseWriterFinding:
    return OledGlobalAppendReleaseWriterFinding(
        code=code,
        severity=severity,
        message=message,
        release_entry_id=release_entry_id,
        source_global_append_entry_id=source_global_append_entry_id,
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
    findings: list[OledGlobalAppendReleaseWriterFinding],
) -> list[OledGlobalAppendReleaseWriterFinding]:
    seen: set[tuple[str, str, str, str, str, str, str, str, str, str]] = set()
    output: list[OledGlobalAppendReleaseWriterFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.release_entry_id or "",
            finding.source_global_append_entry_id or "",
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
            item.release_entry_id or "",
            item.source_global_append_entry_id or "",
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
    release_candidate_written: bool = False,
    release_entry_written: bool = False,
    release_delta_written: bool = False,
    release_snapshot_written: bool = False,
) -> dict[str, Any]:
    return {
        "global_append_release_writer": True,
        "release_candidate_entry": True,
        "release_status": "release_candidate",
        "release_candidate_written": release_candidate_written,
        "release_entry_written": release_entry_written,
        "release_delta_written": release_delta_written,
        "release_snapshot_written": release_snapshot_written,
        "global_registry_mutated": False,
        "external_publication_written": False,
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
    release_candidate_written: bool = False,
    release_entry_written: bool = False,
    release_delta_written: bool = False,
    release_snapshot_written: bool = False,
) -> dict[str, Any]:
    return {
        "global_append_release_writer": True,
        "release_candidate_written": release_candidate_written,
        "release_entry_written": release_entry_written,
        "release_delta_written": release_delta_written,
        "release_snapshot_written": release_snapshot_written,
        "global_registry_mutated": False,
        "external_publication_written": False,
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


_REQUIRED_CAVEATS = [
    "baseline_candidate_report_only",
    "not_benchmark_validated",
    "not_scientific_performance_claim",
]
_FORBIDDEN_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
_FORBIDDEN_JSON_KEYS = {"features", "feature_dict", "raw_text", "full_text", "prediction_id", "training_row_id"}
_MAX_OUTPUT_STRING_LENGTH = 512


__all__ = [
    "OledGlobalAppendReleaseWriterPolicy",
    "OledGlobalAppendReleaseWriteStatus",
    "OledReleaseCandidateEntryStatus",
    "OledReleaseCandidateEntry",
    "OledReleaseCandidateDeltaRecord",
    "OledGlobalAppendReleaseFileResult",
    "OledGlobalAppendReleaseWriterFinding",
    "OledGlobalAppendReleaseWriterManifest",
    "OledGlobalAppendReleaseWriterReport",
    "load_oled_global_append_release_preflight_report_json",
    "build_oled_release_candidate_entry",
    "build_oled_release_candidate_delta_records",
    "select_oled_global_append_release_for_write",
    "write_oled_release_candidate_entry_json",
    "write_oled_release_candidate_delta_jsonl",
    "write_oled_release_candidate_snapshot_jsonl",
    "write_oled_global_append_release_manifest_json",
    "load_oled_release_candidate_entry_json",
    "load_oled_release_candidate_delta_jsonl",
    "oled_release_candidate_entry_filename",
    "oled_release_candidate_delta_filename",
    "oled_release_candidate_snapshot_filename",
    "run_oled_global_append_release_writer_from_files",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
