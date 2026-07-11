from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai4s_agent.domains.oled_baseline_loop import OledBaselineFeatureView
from ai4s_agent.domains.oled_curated_gold_view_preflight import load_oled_curated_gold_records_jsonl
from ai4s_agent.domains.oled_curated_split_dataset_view_writer import (
    OledCuratedSplitDatasetViewRowArtifact,
    OledCuratedSplitDatasetViewWriterManifest,
    load_oled_curated_split_dataset_view_rows_jsonl,
)
from ai4s_agent.domains.oled_curated_split_feature_preflight import (
    OledCuratedSplitFeaturePreflightReport,
    OledCuratedSplitFeaturePreflightStatus,
    OledSplitFeatureRowAlignment,
    OledSplitFeatureRowAlignmentStatus,
    load_oled_curated_split_dataset_view_rows_from_manifest,
    load_oled_curated_split_dataset_view_writer_manifest_json,
)
from ai4s_agent.domains.oled_feature_materialization import (
    OledFeatureMaterializationRow,
    materialize_oled_baseline_feature_table,
)
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledCuratedSplitFeatureWriterPolicy(BaseModel):
    require_confirmation: bool = True
    require_feature_preflight_valid: bool = True
    allow_feature_preflight_warnings: bool = True
    require_all_rows_matched: bool = True
    reject_target_mismatch: bool = True
    reject_missing_feature_rows: bool = True
    reject_ambiguous_feature_rows: bool = True
    allow_missing_feature_values: bool = True
    feature_views: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    write_ml_ready_training_data: bool = False
    run_model_backends: bool = False


class OledCuratedSplitFeatureWriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class OledCuratedSplitFeatureRowArtifact(BaseModel):
    feature_row_id: str

    split: str
    split_row_id: str
    row_id: str

    record_id: str
    source_record_ids: list[str] = Field(default_factory=list)

    view_kind: str
    target_property_id: str
    feature_view: str

    target_value: float | int | str | None = None
    target_unit: str | None = None
    target_reported_value_text: str | None = None
    target_reported_decimal_places: int | None = Field(default=None, ge=0)
    target_reported_unit: str | None = None
    condition_hash: str | None = None
    confidence_score: float | None = None

    evidence_refs: list[str] = Field(default_factory=list)

    features: dict[str, Any] = Field(default_factory=dict)
    missing_feature_columns: list[str] = Field(default_factory=list)
    present_feature_columns: list[str] = Field(default_factory=list)

    alignment_status: str
    alignment_reason_codes: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_record_ids", "evidence_refs", "missing_feature_columns", "present_feature_columns", "alignment_reason_codes")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedSplitFeatureFileResult(BaseModel):
    split: str
    target_property_id: str
    feature_view: str

    status: OledCuratedSplitFeatureWriteStatus
    row_count: int = 0

    output_jsonl_path: str | None = None
    output_sha256: str | None = None

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason_codes")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedSplitFeatureWriterFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    split: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    split_row_id: str | None = None
    record_id: str | None = None
    output_jsonl_path: str | None = None


class OledCuratedSplitFeatureWriterManifest(BaseModel):
    manifest_id: str

    source_split_dataset_view_manifest_id: str | None = None
    source_feature_preflight_status: str | None = None

    output_directory: str | None = None
    output_file_count: int = 0
    output_row_count: int = 0

    splits: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    status_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)
    rows_by_split: dict[str, int] = Field(default_factory=dict)

    file_results: list[OledCuratedSplitFeatureFileResult] = Field(default_factory=list)

    policy: OledCuratedSplitFeatureWriterPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(result.status == OledCuratedSplitFeatureWriteStatus.REJECTED for result in self.file_results)


class OledCuratedSplitFeatureWriterReport(BaseModel):
    manifest: OledCuratedSplitFeatureWriterManifest
    feature_row_artifacts: list[OledCuratedSplitFeatureRowArtifact] = Field(default_factory=list)
    findings: list[OledCuratedSplitFeatureWriterFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_curated_split_feature_preflight_report_json(
    path: str | Path,
) -> OledCuratedSplitFeaturePreflightReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_split_feature_preflight_report:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = OledCuratedSplitFeaturePreflightReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_split_feature_preflight_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    if _contains_absolute_path(report.metadata):
        raise ValueError("absolute_path_in_split_feature_preflight_report_metadata")
    return report


def build_oled_curated_split_feature_row_artifacts(
    *,
    gold_records: Iterable[OledGoldDatasetRecord],
    split_rows: Iterable[OledCuratedSplitDatasetViewRowArtifact],
    feature_preflight_report: OledCuratedSplitFeaturePreflightReport,
    policy: OledCuratedSplitFeatureWriterPolicy | None = None,
) -> tuple[list[OledCuratedSplitFeatureRowArtifact], list[OledCuratedSplitFeatureWriterFinding]]:
    writer_policy = policy or OledCuratedSplitFeatureWriterPolicy()
    records = sorted(list(gold_records), key=lambda record: record.record_id)
    rows_by_split_row_id = {row.split_row_id: row for row in split_rows}
    alignments, duplicate_findings = _alignment_allowlist(feature_preflight_report.row_alignments, writer_policy)
    findings: list[OledCuratedSplitFeatureWriterFinding] = list(duplicate_findings)
    feature_rows_by_key, materialization_findings = _materialized_feature_rows(records, alignments)
    findings.extend(materialization_findings)

    artifacts: list[OledCuratedSplitFeatureRowArtifact] = []
    for alignment in alignments:
        row = rows_by_split_row_id.get(alignment.split_row_id)
        if row is None:
            findings.append(_alignment_finding(alignment, code="split_row_missing", severity="error", message="feature preflight alignment references a missing split row"))
            continue
        if alignment.status == OledSplitFeatureRowAlignmentStatus.MISSING_FEATURE_ROW:
            findings.append(
                _alignment_finding(
                    alignment,
                    code="missing_feature_row_rejected",
                    severity="error" if writer_policy.reject_missing_feature_rows or writer_policy.require_all_rows_matched else "warning",
                    message="missing feature row alignment is not materialized",
                )
            )
            continue
        if alignment.status == OledSplitFeatureRowAlignmentStatus.AMBIGUOUS_FEATURE_ROW:
            findings.append(
                _alignment_finding(
                    alignment,
                    code="ambiguous_feature_row_rejected",
                    severity="error" if writer_policy.reject_ambiguous_feature_rows or writer_policy.require_all_rows_matched else "warning",
                    message="ambiguous feature row alignment is not materialized",
                )
            )
            continue
        if alignment.status == OledSplitFeatureRowAlignmentStatus.TARGET_MISMATCH:
            findings.append(
                _alignment_finding(
                    alignment,
                    code="target_mismatch_rejected",
                    severity="error" if writer_policy.reject_target_mismatch else "warning",
                    message="target-mismatched feature row alignment is not materialized",
                )
            )
            continue
        feature_row = _feature_row_for_alignment(
            alignment,
            feature_rows_by_key.get((alignment.target_property_id, alignment.feature_view), []),
        )
        if feature_row is None:
            findings.append(_alignment_finding(alignment, code="matched_feature_row_not_rebuilt", severity="error", message="matched feature preflight row could not be rebuilt"))
            continue
        missing_columns = sorted(set(alignment.missing_feature_columns or _missing_feature_columns(feature_row)))
        present_columns = sorted(set(alignment.present_feature_columns or [key for key in feature_row.features if key not in missing_columns]))
        if missing_columns and not writer_policy.allow_missing_feature_values:
            findings.append(
                _alignment_finding(
                    alignment,
                    code="missing_feature_values_rejected",
                    severity="error",
                    message="matched feature row contains missing feature values and policy rejects missing values",
                )
            )
            continue
        artifacts.append(_feature_artifact(row=row, alignment=alignment, feature_row=feature_row, missing_columns=missing_columns, present_columns=present_columns))
    return sorted(artifacts, key=lambda row: (row.split, row.target_property_id, row.feature_view, row.feature_row_id)), _dedup_findings(findings)


def select_oled_curated_split_feature_rows_for_write(
    *,
    gold_records: Iterable[OledGoldDatasetRecord],
    split_rows: Iterable[OledCuratedSplitDatasetViewRowArtifact],
    feature_preflight_report: OledCuratedSplitFeaturePreflightReport,
    policy: OledCuratedSplitFeatureWriterPolicy | None = None,
    confirm_split_feature_write: bool = False,
) -> OledCuratedSplitFeatureWriterReport:
    writer_policy = policy or OledCuratedSplitFeatureWriterPolicy()
    if writer_policy.require_confirmation and not confirm_split_feature_write:
        raise ValueError("confirmation_required:split_feature_write")

    preflight_findings = _preflight_gate_findings(feature_preflight_report, writer_policy)
    if any(finding.severity == "error" for finding in preflight_findings):
        return OledCuratedSplitFeatureWriterReport(
            manifest=_manifest(
                policy=writer_policy,
                file_results=[],
                feature_row_artifacts=[],
                source_feature_preflight_status=_status_value(feature_preflight_report.status),
            ),
            feature_row_artifacts=[],
            findings=preflight_findings,
        )

    feature_rows, row_findings = build_oled_curated_split_feature_row_artifacts(
        gold_records=gold_records,
        split_rows=split_rows,
        feature_preflight_report=feature_preflight_report,
        policy=writer_policy,
    )
    file_results = _file_results_for_feature_rows(feature_rows)
    return OledCuratedSplitFeatureWriterReport(
        manifest=_manifest(
            policy=writer_policy,
            file_results=file_results,
            feature_row_artifacts=feature_rows,
            source_feature_preflight_status=_status_value(feature_preflight_report.status),
        ),
        feature_row_artifacts=feature_rows,
        findings=_dedup_findings([*preflight_findings, *row_findings]),
    )


def write_oled_curated_split_feature_rows_jsonl(
    rows: Iterable[OledCuratedSplitFeatureRowArtifact],
    path: str | Path,
) -> str:
    lines = [
        json.dumps(
            _sanitize_for_output(row.model_dump(mode="json", exclude_none=True)),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in sorted(rows, key=lambda item: item.feature_row_id)
    ]
    payload = "\n".join(lines) + ("\n" if lines else "")
    encoded = payload.encode("utf-8")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_oled_curated_split_feature_manifest_json(
    manifest: OledCuratedSplitFeatureWriterManifest,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def oled_split_feature_output_filename(
    *,
    split: str,
    target_property_id: str,
    feature_view: str,
) -> str:
    return (
        "oled_split_features__"
        f"{_safe_filename_token(split)}__"
        f"{_safe_filename_token(target_property_id)}__"
        f"{_safe_filename_token(feature_view)}.jsonl"
    )


def run_oled_curated_split_feature_writer_from_files(
    *,
    curated_gold_jsonl_path: str | Path,
    split_dataset_view_manifest_path: str | Path,
    feature_preflight_report_path: str | Path,
    split_dataset_view_base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    policy: OledCuratedSplitFeatureWriterPolicy | None = None,
    confirm_split_feature_write: bool = False,
    dry_run: bool = False,
) -> OledCuratedSplitFeatureWriterReport:
    writer_policy = policy or OledCuratedSplitFeatureWriterPolicy()
    if not dry_run and writer_policy.require_confirmation and not confirm_split_feature_write:
        raise ValueError("confirmation_required:split_feature_write")
    if not dry_run and output_dir is None:
        raise ValueError("output_dir_required:split_feature_write")

    records = load_oled_curated_gold_records_jsonl(curated_gold_jsonl_path)
    split_manifest = load_oled_curated_split_dataset_view_writer_manifest_json(split_dataset_view_manifest_path)
    base_dir = Path(split_dataset_view_base_dir) if split_dataset_view_base_dir is not None else Path(split_dataset_view_manifest_path).parent
    split_rows = load_oled_curated_split_dataset_view_rows_from_manifest(manifest=split_manifest, base_dir=base_dir)
    preflight_report = load_oled_curated_split_feature_preflight_report_json(feature_preflight_report_path)
    selection_policy = writer_policy.model_copy(update={"require_confirmation": not dry_run and writer_policy.require_confirmation})
    report = select_oled_curated_split_feature_rows_for_write(
        gold_records=records,
        split_rows=split_rows,
        feature_preflight_report=preflight_report,
        policy=selection_policy,
        confirm_split_feature_write=confirm_split_feature_write or dry_run,
    )
    report = _attach_source_context(
        report,
        source_split_dataset_view_manifest_id=split_manifest.manifest_id,
        source_feature_preflight_status=_status_value(preflight_report.status),
    )
    if dry_run:
        report = _mark_dry_run(report)
        if output_manifest_path is not None:
            write_oled_curated_split_feature_manifest_json(report.manifest, output_manifest_path)
        return report
    if not report.is_valid:
        if output_manifest_path is not None:
            write_oled_curated_split_feature_manifest_json(report.manifest, output_manifest_path)
        return report

    assert output_dir is not None
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report = _write_selected_feature_files(report, output_root)
    if output_manifest_path is not None:
        write_oled_curated_split_feature_manifest_json(report.manifest, output_manifest_path)
    return report


def load_oled_curated_split_feature_rows_jsonl(
    path: str | Path,
) -> list[OledCuratedSplitFeatureRowArtifact]:
    rows_path = Path(path)
    _reject_forbidden_input(rows_path)
    if not rows_path.exists():
        raise ValueError(f"missing_split_feature_rows_jsonl:{redact_oled_mineru_acceptance_path(rows_path)}")
    rows: list[OledCuratedSplitFeatureRowArtifact] = []
    for line_number, raw_line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            row = OledCuratedSplitFeatureRowArtifact.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid_split_feature_rows_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(row.metadata):
            raise ValueError(f"absolute_path_in_split_feature_row_metadata:{row.feature_row_id}")
        rows.append(row)
    return sorted(rows, key=lambda row: row.feature_row_id)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write aligned OLED split feature row artifacts under explicit gates.")
    parser.add_argument("--curated-gold-jsonl", required=True, help="Path to curated gold-record JSONL.")
    parser.add_argument("--split-dataset-view-manifest", required=True, help="Path to split dataset-view writer manifest JSON.")
    parser.add_argument("--feature-preflight-report", required=True, help="Path to split feature preflight report JSON.")
    parser.add_argument("--split-dataset-view-base-dir", help="Base directory for split dataset-view row JSONL paths.")
    parser.add_argument("--output-dir", help="Directory for split feature row JSONL files.")
    parser.add_argument("--output-manifest", help="Optional path for split feature writer manifest JSON.")
    parser.add_argument("--confirm-split-feature-write", action="store_true", help="Confirm split feature row writing.")
    parser.add_argument("--dry-run", action="store_true", help="Run selection without writing feature row JSONL files.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    args = parser.parse_args(argv)

    if not args.output_dir and not args.output_manifest:
        print("output_required:dir_or_manifest", file=sys.stderr)
        return 1
    if not args.dry_run and not args.confirm_split_feature_write:
        print("confirmation_required:split_feature_write", file=sys.stderr)
        return 1
    try:
        policy = OledCuratedSplitFeatureWriterPolicy(
            require_confirmation=not args.dry_run,
            feature_views=_split_cli_values(args.feature_view),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
        )
        report = run_oled_curated_split_feature_writer_from_files(
            curated_gold_jsonl_path=args.curated_gold_jsonl,
            split_dataset_view_manifest_path=args.split_dataset_view_manifest,
            feature_preflight_report_path=args.feature_preflight_report,
            split_dataset_view_base_dir=args.split_dataset_view_base_dir,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            policy=policy,
            confirm_split_feature_write=args.confirm_split_feature_write,
            dry_run=args.dry_run,
        )
        summary = {
            "output_file_count": report.manifest.output_file_count,
            "output_row_count": report.manifest.output_row_count,
            "rows_by_split": report.manifest.rows_by_split,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _alignment_allowlist(
    alignments: list[OledSplitFeatureRowAlignment],
    policy: OledCuratedSplitFeatureWriterPolicy,
) -> tuple[list[OledSplitFeatureRowAlignment], list[OledCuratedSplitFeatureWriterFinding]]:
    target_property_ids = set(_target_property_ids(policy))
    feature_views = set(_feature_view_values(policy))
    output: dict[tuple[str, str, str], OledSplitFeatureRowAlignment] = {}
    findings: list[OledCuratedSplitFeatureWriterFinding] = []
    for alignment in sorted(alignments, key=lambda item: (item.split_row_id, item.target_property_id, item.feature_view)):
        if alignment.target_property_id not in target_property_ids or alignment.feature_view not in feature_views:
            continue
        key = (alignment.split_row_id, alignment.target_property_id, alignment.feature_view)
        if key in output:
            findings.append(
                _alignment_finding(
                    alignment,
                    code="duplicate_feature_preflight_alignment",
                    severity="error",
                    message="feature preflight report contains duplicate alignment for split row, target, and feature view",
                )
            )
            continue
        output[key] = alignment
    return list(output.values()), findings


def _materialized_feature_rows(
    records: list[OledGoldDatasetRecord],
    alignments: list[OledSplitFeatureRowAlignment],
) -> tuple[dict[tuple[str, str], list[OledFeatureMaterializationRow]], list[OledCuratedSplitFeatureWriterFinding]]:
    grouped_targets = sorted({(alignment.target_property_id, alignment.feature_view) for alignment in alignments})
    output: dict[tuple[str, str], list[OledFeatureMaterializationRow]] = {}
    findings: list[OledCuratedSplitFeatureWriterFinding] = []
    for target_property_id, feature_view_value in grouped_targets:
        try:
            feature_view = OledBaselineFeatureView(feature_view_value)
            table = materialize_oled_baseline_feature_table(
                records,
                feature_view=feature_view,
                target_property_id=target_property_id,
            )
            output[(target_property_id, feature_view_value)] = table.rows
        except Exception as exc:
            message = str(exc).splitlines()[0]
            output[(target_property_id, feature_view_value)] = []
            if message.startswith("no_gold_records_for_target:"):
                continue
            findings.append(
                OledCuratedSplitFeatureWriterFinding(
                    code="feature_materialization_failed",
                    severity="error",
                    message=message,
                    target_property_id=target_property_id,
                    feature_view=feature_view_value,
                )
            )
    return output, findings


def _feature_row_for_alignment(
    alignment: OledSplitFeatureRowAlignment,
    feature_rows: list[OledFeatureMaterializationRow],
) -> OledFeatureMaterializationRow | None:
    record_candidates = [row for row in feature_rows if row.record_id == alignment.record_id]
    if alignment.feature_row_condition_hash:
        candidates = [row for row in record_candidates if row.condition_hash == alignment.feature_row_condition_hash]
    else:
        candidates = record_candidates
    if len(candidates) == 1:
        return candidates[0]
    return None


def _feature_artifact(
    *,
    row: OledCuratedSplitDatasetViewRowArtifact,
    alignment: OledSplitFeatureRowAlignment,
    feature_row: OledFeatureMaterializationRow,
    missing_columns: list[str],
    present_columns: list[str],
) -> OledCuratedSplitFeatureRowArtifact:
    reason_codes = sorted({*alignment.reason_codes, "matched", *(["missing_feature_values_allowed"] if missing_columns else [])})
    metadata = _sanitize_for_output(row.metadata)
    metadata.update(
        {
            "split_feature_row_artifact": True,
            "ml_ready_training_data_record": False,
            "training_package_written": False,
            "condition_hash": feature_row.condition_hash or row.condition_hash,
            "split_dataset_view_row_id": row.split_row_id,
        }
    )
    return OledCuratedSplitFeatureRowArtifact(
        feature_row_id=_feature_row_id(row=row, alignment=alignment, feature_row=feature_row),
        split=alignment.split,
        split_row_id=row.split_row_id,
        row_id=row.row_id,
        record_id=row.record_id,
        source_record_ids=row.source_record_ids or [row.record_id],
        view_kind=row.view_kind,
        target_property_id=row.target_property_id,
        feature_view=alignment.feature_view,
        target_value=feature_row.target_value,
        target_unit=feature_row.target_unit,
        target_reported_value_text=feature_row.target_reported_value_text,
        target_reported_decimal_places=feature_row.target_reported_decimal_places,
        target_reported_unit=feature_row.target_reported_unit,
        condition_hash=feature_row.condition_hash or row.condition_hash,
        confidence_score=feature_row.confidence_score,
        evidence_refs=feature_row.evidence_refs or row.evidence_refs,
        features=_sanitize_for_output(feature_row.features),
        missing_feature_columns=missing_columns,
        present_feature_columns=present_columns,
        alignment_status=_status_value(alignment.status),
        alignment_reason_codes=reason_codes,
        metadata=metadata,
    )


def _feature_row_id(
    *,
    row: OledCuratedSplitDatasetViewRowArtifact,
    alignment: OledSplitFeatureRowAlignment,
    feature_row: OledFeatureMaterializationRow,
) -> str:
    payload = {
        "split": alignment.split,
        "split_row_id": row.split_row_id,
        "row_id": row.row_id,
        "record_id": row.record_id,
        "source_record_ids": row.source_record_ids,
        "target_property_id": row.target_property_id,
        "feature_view": alignment.feature_view,
        "condition_hash": feature_row.condition_hash or row.condition_hash,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-split-feature-row:{digest[:20]}"


def _preflight_gate_findings(
    feature_preflight_report: OledCuratedSplitFeaturePreflightReport,
    policy: OledCuratedSplitFeatureWriterPolicy,
) -> list[OledCuratedSplitFeatureWriterFinding]:
    findings: list[OledCuratedSplitFeatureWriterFinding] = []
    if policy.require_feature_preflight_valid and not _feature_preflight_is_valid(feature_preflight_report):
        findings.append(
            OledCuratedSplitFeatureWriterFinding(
                code="feature_preflight_failed",
                severity="error",
                message="split feature writer blocked because feature preflight report is invalid",
            )
        )
    if not policy.allow_feature_preflight_warnings and feature_preflight_report.warning_codes:
        findings.append(
            OledCuratedSplitFeatureWriterFinding(
                code="feature_preflight_warnings_present",
                severity="error",
                message="split feature writer blocked because feature preflight warnings are disallowed",
            )
        )
    return findings


def _feature_preflight_is_valid(report: OledCuratedSplitFeaturePreflightReport) -> bool:
    return _status_value(report.status) != OledCuratedSplitFeaturePreflightStatus.FAILED.value and not report.error_codes


def _file_results_for_feature_rows(
    feature_rows: list[OledCuratedSplitFeatureRowArtifact],
) -> list[OledCuratedSplitFeatureFileResult]:
    grouped: dict[tuple[str, str, str], list[OledCuratedSplitFeatureRowArtifact]] = defaultdict(list)
    for row in feature_rows:
        grouped[(row.split, row.target_property_id, row.feature_view)].append(row)
    results: list[OledCuratedSplitFeatureFileResult] = []
    for (split, target_property_id, feature_view), group in sorted(grouped.items()):
        results.append(
            OledCuratedSplitFeatureFileResult(
                split=split,
                target_property_id=target_property_id,
                feature_view=feature_view,
                status=OledCuratedSplitFeatureWriteStatus.WRITTEN,
                row_count=len(group),
                reason_codes=sorted({"selected_for_write", *(code for row in group for code in row.alignment_reason_codes)}),
                metadata={"split_feature_rows_written": False},
            )
        )
    return results


def _manifest(
    *,
    policy: OledCuratedSplitFeatureWriterPolicy,
    file_results: list[OledCuratedSplitFeatureFileResult],
    feature_row_artifacts: list[OledCuratedSplitFeatureRowArtifact],
    source_split_dataset_view_manifest_id: str | None = None,
    source_feature_preflight_status: str | None = None,
    output_directory: str | None = None,
    split_feature_rows_written: bool = False,
) -> OledCuratedSplitFeatureWriterManifest:
    return OledCuratedSplitFeatureWriterManifest(
        manifest_id=_manifest_id(policy, file_results),
        source_split_dataset_view_manifest_id=source_split_dataset_view_manifest_id,
        source_feature_preflight_status=source_feature_preflight_status,
        output_directory=output_directory,
        output_file_count=sum(1 for result in file_results if result.status == OledCuratedSplitFeatureWriteStatus.WRITTEN and result.row_count > 0),
        output_row_count=len(feature_row_artifacts),
        splits=sorted({row.split for row in feature_row_artifacts}),
        target_property_ids=sorted({row.target_property_id for row in feature_row_artifacts}),
        feature_views=sorted({row.feature_view for row in feature_row_artifacts}),
        status_counts=dict(sorted(Counter(result.status.value for result in file_results).items())),
        reason_code_counts=dict(sorted(Counter(code for result in file_results for code in result.reason_codes).items())),
        rows_by_split=dict(sorted(Counter(row.split for row in feature_row_artifacts).items())),
        file_results=sorted(file_results, key=lambda item: (item.split, item.target_property_id, item.feature_view)),
        policy=policy,
        metadata=_safety_metadata(split_feature_rows_written=split_feature_rows_written),
    )


def _manifest_id(
    policy: OledCuratedSplitFeatureWriterPolicy,
    file_results: list[OledCuratedSplitFeatureFileResult],
) -> str:
    payload = {
        "policy": policy.model_dump(mode="json"),
        "file_results": [
            {
                "split": result.split,
                "target_property_id": result.target_property_id,
                "feature_view": result.feature_view,
                "row_count": result.row_count,
                "status": result.status.value,
            }
            for result in sorted(file_results, key=lambda item: (item.split, item.target_property_id, item.feature_view))
        ],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-curated-split-feature-writer:{digest[:16]}"


def _attach_source_context(
    report: OledCuratedSplitFeatureWriterReport,
    *,
    source_split_dataset_view_manifest_id: str | None,
    source_feature_preflight_status: str | None,
) -> OledCuratedSplitFeatureWriterReport:
    manifest = report.manifest.model_copy(
        update={
            "source_split_dataset_view_manifest_id": source_split_dataset_view_manifest_id,
            "source_feature_preflight_status": source_feature_preflight_status,
        }
    )
    return report.model_copy(update={"manifest": manifest})


def _mark_dry_run(report: OledCuratedSplitFeatureWriterReport) -> OledCuratedSplitFeatureWriterReport:
    refreshed_results = [
        result.model_copy(
            update={
                "reason_codes": sorted({*result.reason_codes, "dry_run_no_rows_written"}),
                "metadata": {**result.metadata, "split_feature_rows_written": False},
            }
        )
        for result in report.manifest.file_results
    ]
    manifest = _manifest(
        policy=report.manifest.policy,
        file_results=refreshed_results,
        feature_row_artifacts=report.feature_row_artifacts,
        source_split_dataset_view_manifest_id=report.manifest.source_split_dataset_view_manifest_id,
        source_feature_preflight_status=report.manifest.source_feature_preflight_status,
        split_feature_rows_written=False,
    )
    return report.model_copy(update={"manifest": manifest})


def _write_selected_feature_files(
    report: OledCuratedSplitFeatureWriterReport,
    output_dir: Path,
) -> OledCuratedSplitFeatureWriterReport:
    grouped_rows: dict[tuple[str, str, str], list[OledCuratedSplitFeatureRowArtifact]] = defaultdict(list)
    for row in report.feature_row_artifacts:
        grouped_rows[(row.split, row.target_property_id, row.feature_view)].append(row)

    refreshed_results: list[OledCuratedSplitFeatureFileResult] = []
    for result in report.manifest.file_results:
        if result.status != OledCuratedSplitFeatureWriteStatus.WRITTEN or result.row_count <= 0:
            refreshed_results.append(result)
            continue
        filename = oled_split_feature_output_filename(
            split=result.split,
            target_property_id=result.target_property_id,
            feature_view=result.feature_view,
        )
        output_path = output_dir / filename
        output_sha = write_oled_curated_split_feature_rows_jsonl(
            grouped_rows[(result.split, result.target_property_id, result.feature_view)],
            output_path,
        )
        refreshed_results.append(
            result.model_copy(
                update={
                    "output_jsonl_path": filename,
                    "output_sha256": output_sha,
                    "metadata": {**result.metadata, "split_feature_rows_written": True},
                }
            )
        )

    manifest = _manifest(
        policy=report.manifest.policy,
        file_results=refreshed_results,
        feature_row_artifacts=report.feature_row_artifacts,
        source_split_dataset_view_manifest_id=report.manifest.source_split_dataset_view_manifest_id,
        source_feature_preflight_status=report.manifest.source_feature_preflight_status,
        output_directory=redact_oled_mineru_acceptance_path(output_dir),
        split_feature_rows_written=True,
    )
    return report.model_copy(update={"manifest": manifest})


def _alignment_finding(
    alignment: OledSplitFeatureRowAlignment,
    *,
    code: str,
    severity: Literal["error", "warning"],
    message: str,
) -> OledCuratedSplitFeatureWriterFinding:
    return OledCuratedSplitFeatureWriterFinding(
        code=code,
        severity=severity,
        message=message,
        split=alignment.split,
        target_property_id=alignment.target_property_id,
        feature_view=alignment.feature_view,
        split_row_id=alignment.split_row_id,
        record_id=alignment.record_id,
    )


def _target_property_ids(policy: OledCuratedSplitFeatureWriterPolicy) -> list[str]:
    return sorted({str(item).strip() for item in policy.target_property_ids if str(item).strip()})


def _feature_view_values(policy: OledCuratedSplitFeatureWriterPolicy) -> list[str]:
    if not policy.feature_views:
        return [feature_view.value for feature_view in OledBaselineFeatureView]
    return sorted({OledBaselineFeatureView(str(item)).value for item in policy.feature_views})


def _missing_feature_columns(row: OledFeatureMaterializationRow) -> list[str]:
    return sorted(column for column, value in row.features.items() if _is_missing_feature_value(value))


def _is_missing_feature_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _safe_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_") or "unknown"


def _safety_metadata(*, split_feature_rows_written: bool) -> dict[str, Any]:
    return {
        "split_feature_writer": True,
        "split_feature_rows_written": split_feature_rows_written,
        "training_data_written": False,
        "ml_ready_training_data_written": False,
        "model_backends_run": False,
        "baseline_backend_run": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


def _dedup_findings(
    findings: list[OledCuratedSplitFeatureWriterFinding],
) -> list[OledCuratedSplitFeatureWriterFinding]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[OledCuratedSplitFeatureWriterFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.split or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.split_row_id or "",
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
            item.split or "",
            item.target_property_id or "",
            item.feature_view or "",
            item.split_row_id or "",
        ),
    )


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


def _sanitize_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _is_forbidden_payload_key(key):
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
    if isinstance(value, tuple):
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


def _is_forbidden_payload_key(key: str) -> bool:
    normalized = key.lower()
    return any(
        token in normalized
        for token in (
            "raw_text",
            "full_text",
            "parsed_json",
            "table_body",
            "html_table",
            "markdown_table",
            "layered_record",
            "gold_record",
        )
    )


_MAX_OUTPUT_STRING_LENGTH = 240

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
    "OledCuratedSplitFeatureWriterPolicy",
    "OledCuratedSplitFeatureWriteStatus",
    "OledCuratedSplitFeatureRowArtifact",
    "OledCuratedSplitFeatureFileResult",
    "OledCuratedSplitFeatureWriterFinding",
    "OledCuratedSplitFeatureWriterManifest",
    "OledCuratedSplitFeatureWriterReport",
    "load_oled_curated_split_feature_preflight_report_json",
    "build_oled_curated_split_feature_row_artifacts",
    "select_oled_curated_split_feature_rows_for_write",
    "write_oled_curated_split_feature_rows_jsonl",
    "write_oled_curated_split_feature_manifest_json",
    "oled_split_feature_output_filename",
    "run_oled_curated_split_feature_writer_from_files",
    "load_oled_curated_split_feature_rows_jsonl",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
