from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai4s_agent.domains.oled_curated_split_feature_writer import (
    OledCuratedSplitFeatureRowArtifact,
)
from ai4s_agent.domains.oled_curated_split_training_package_preflight import (
    OledCuratedSplitTrainingPackagePreflightReport,
    OledCuratedSplitTrainingPackagePreflightStatus,
    OledTrainingFeatureColumnKind,
    load_oled_curated_split_feature_rows_from_manifest,
    load_oled_curated_split_feature_writer_manifest_json,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path

warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema" in "OledCuratedSplitTrainingPackageWriterReport" shadows an attribute in parent "BaseModel"',
    category=UserWarning,
)


class OledCuratedSplitTrainingPackageWriterPolicy(BaseModel):
    require_confirmation: bool = True
    require_preflight_valid: bool = True
    allow_preflight_warnings: bool = True
    require_manifest_sha256: bool = True
    require_train_split: bool = True
    require_target_values: bool = True
    require_evidence_refs: bool = True
    require_consistent_feature_columns: bool = True

    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])

    output_format: Literal["jsonl"] = "jsonl"

    run_baseline_backend: bool = False
    run_model_backends: bool = False
    benchmark_validated: bool = False


class OledCuratedSplitTrainingPackageWriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class OledCuratedTrainingPackageRow(BaseModel):
    training_row_id: str

    split: str
    feature_row_id: str
    split_row_id: str
    row_id: str
    record_id: str
    source_record_ids: list[str] = Field(default_factory=list)

    target_property_id: str
    target_value: float | int | str
    target_unit: str | None = None

    feature_view: str
    features: dict[str, Any] = Field(default_factory=dict)

    condition_hash: str | None = None
    confidence_score: float | None = None
    evidence_refs: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_record_ids", "evidence_refs")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedTrainingPackageSchema(BaseModel):
    schema_id: str

    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=list)

    target_columns: list[str] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    metadata_columns: list[str] = Field(default_factory=list)

    feature_column_kinds: dict[str, str] = Field(default_factory=dict)
    required_columns: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class OledCuratedTrainingPackageFileResult(BaseModel):
    split: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    artifact_kind: Literal["training_rows", "schema", "manifest"]

    status: OledCuratedSplitTrainingPackageWriteStatus
    row_count: int = 0

    output_path: str | None = None
    output_sha256: str | None = None

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason_codes")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedSplitTrainingPackageWriterFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    split: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    feature_row_id: str | None = None
    training_row_id: str | None = None
    output_path: str | None = None


class OledCuratedSplitTrainingPackageWriterManifest(BaseModel):
    manifest_id: str

    source_split_feature_manifest_id: str | None = None
    source_training_preflight_status: str | None = None

    output_directory: str | None = None
    output_file_count: int = 0
    output_row_count: int = 0

    splits: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)

    rows_by_split: dict[str, int] = Field(default_factory=dict)
    rows_by_target: dict[str, int] = Field(default_factory=dict)
    rows_by_feature_view: dict[str, int] = Field(default_factory=dict)

    file_results: list[OledCuratedTrainingPackageFileResult] = Field(default_factory=list)

    policy: OledCuratedSplitTrainingPackageWriterPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(result.status == OledCuratedSplitTrainingPackageWriteStatus.REJECTED for result in self.file_results)


class OledCuratedSplitTrainingPackageWriterReport(BaseModel):
    manifest: OledCuratedSplitTrainingPackageWriterManifest
    schema: OledCuratedTrainingPackageSchema | None = None
    training_rows: list[OledCuratedTrainingPackageRow] = Field(default_factory=list)
    findings: list[OledCuratedSplitTrainingPackageWriterFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_curated_split_training_package_preflight_report_json(
    path: str | Path,
) -> OledCuratedSplitTrainingPackagePreflightReport:
    report_path = Path(path)
    _reject_forbidden_input(report_path)
    if not report_path.exists():
        raise ValueError(f"missing_training_package_preflight_report:{redact_oled_mineru_acceptance_path(report_path)}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = OledCuratedSplitTrainingPackagePreflightReport.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_training_package_preflight_report_json:{redact_oled_mineru_acceptance_path(report_path)}") from exc
    if _contains_absolute_path(report.metadata):
        raise ValueError("absolute_path_in_training_package_preflight_report_metadata")
    return report


def build_oled_curated_training_package_rows(
    feature_rows: Iterable[OledCuratedSplitFeatureRowArtifact],
    *,
    preflight_report: OledCuratedSplitTrainingPackagePreflightReport,
    policy: OledCuratedSplitTrainingPackageWriterPolicy | None = None,
) -> tuple[list[OledCuratedTrainingPackageRow], list[OledCuratedSplitTrainingPackageWriterFinding]]:
    writer_policy = policy or OledCuratedSplitTrainingPackageWriterPolicy()
    allowed_targets = _target_property_ids(writer_policy)
    allowed_views = _feature_views(writer_policy)
    allowed_splits = _splits(writer_policy)
    preflight_targets = set(preflight_report.target_property_ids)
    preflight_views = set(preflight_report.feature_views)
    training_rows: list[OledCuratedTrainingPackageRow] = []
    findings: list[OledCuratedSplitTrainingPackageWriterFinding] = []

    for row in sorted(feature_rows, key=lambda item: (item.split, item.target_property_id, item.feature_view, item.feature_row_id)):
        if row.target_property_id not in allowed_targets or row.target_property_id not in preflight_targets:
            continue
        if allowed_views and row.feature_view not in allowed_views:
            continue
        if preflight_views and row.feature_view not in preflight_views:
            continue
        if row.split not in allowed_splits:
            findings.append(_row_finding("unknown_split_rejected", "error", "feature row split is not allowed for training package", row))
            continue
        if _is_missing_value(row.target_value):
            findings.append(_row_finding("missing_target_rejected", "error" if writer_policy.require_target_values else "warning", "feature row target is missing", row))
            continue
        if not row.evidence_refs:
            findings.append(_row_finding("missing_evidence_rejected", "error" if writer_policy.require_evidence_refs else "warning", "feature row evidence refs are missing", row))
            continue
        if not row.features:
            findings.append(_row_finding("empty_features_rejected", "error", "feature row has no feature payload", row))
            continue
        if bool(row.metadata.get("benchmark_validated")):
            findings.append(_row_finding("source_claims_benchmark_validated", "error", "source feature row claims benchmark validation before writer gate", row))
            continue
        training_rows.append(_training_row_from_feature_row(row))

    return sorted(training_rows, key=lambda row: row.training_row_id), _dedup_findings(findings)


def build_oled_curated_training_package_schema(
    rows: Iterable[OledCuratedTrainingPackageRow],
    *,
    preflight_report: OledCuratedSplitTrainingPackagePreflightReport | None = None,
) -> OledCuratedTrainingPackageSchema:
    training_rows = sorted(list(rows), key=lambda row: row.training_row_id)
    feature_columns = sorted({column for row in training_rows for column in row.features})
    preflight_kinds = {
        summary.column_name: _status_value(summary.kind)
        for summary in (preflight_report.feature_column_summaries if preflight_report is not None else [])
    }
    feature_column_kinds = {
        column: preflight_kinds.get(column) or _column_kind([row.features.get(column) for row in training_rows if column in row.features]).value
        for column in feature_columns
    }
    schema = OledCuratedTrainingPackageSchema(
        schema_id="",
        target_property_ids=sorted({row.target_property_id for row in training_rows}),
        feature_views=sorted({row.feature_view for row in training_rows}),
        splits=sorted({row.split for row in training_rows}),
        target_columns=["target_property_id", "target_value", "target_unit"],
        feature_columns=feature_columns,
        metadata_columns=[
            "training_row_id",
            "split",
            "record_id",
            "feature_row_id",
            "split_row_id",
            "condition_hash",
            "confidence_score",
            "evidence_refs",
        ],
        feature_column_kinds=dict(sorted(feature_column_kinds.items())),
        required_columns=[],
        metadata={
            "training_package_schema": True,
            "benchmark_validated": False,
            "model_backend_run": False,
        },
    )
    return schema.model_copy(update={"schema_id": _schema_id(schema)})


def select_oled_curated_split_training_package_for_write(
    *,
    feature_rows: Iterable[OledCuratedSplitFeatureRowArtifact],
    preflight_report: OledCuratedSplitTrainingPackagePreflightReport,
    policy: OledCuratedSplitTrainingPackageWriterPolicy | None = None,
    confirm_training_package_write: bool = False,
) -> OledCuratedSplitTrainingPackageWriterReport:
    writer_policy = policy or OledCuratedSplitTrainingPackageWriterPolicy()
    if writer_policy.require_confirmation and not confirm_training_package_write:
        raise ValueError("confirmation_required:training_package_write")
    gate_findings = _preflight_gate_findings(preflight_report, writer_policy)
    if any(finding.severity == "error" for finding in gate_findings):
        return OledCuratedSplitTrainingPackageWriterReport(
            manifest=_manifest(
                policy=writer_policy,
                file_results=[],
                training_rows=[],
                source_training_preflight_status=_status_value(preflight_report.status),
                training_package_written=False,
            ),
            schema=None,
            training_rows=[],
            findings=gate_findings,
        )

    training_rows, row_findings = build_oled_curated_training_package_rows(
        feature_rows,
        preflight_report=preflight_report,
        policy=writer_policy,
    )
    schema = build_oled_curated_training_package_schema(training_rows, preflight_report=preflight_report)
    file_results = _file_results_for_training_rows(training_rows)
    if training_rows:
        file_results.append(
            OledCuratedTrainingPackageFileResult(
                artifact_kind="schema",
                status=OledCuratedSplitTrainingPackageWriteStatus.WRITTEN,
                reason_codes=["selected_for_write"],
                metadata={"training_package_written": False},
            )
        )
    return OledCuratedSplitTrainingPackageWriterReport(
        manifest=_manifest(
            policy=writer_policy,
            file_results=file_results,
            training_rows=training_rows,
            source_training_preflight_status=_status_value(preflight_report.status),
            training_package_written=False,
        ),
        schema=schema,
        training_rows=training_rows,
        findings=_dedup_findings([*gate_findings, *row_findings]),
    )


def write_oled_curated_training_rows_jsonl(
    rows: Iterable[OledCuratedTrainingPackageRow],
    path: str | Path,
) -> str:
    lines = [
        json.dumps(
            _sanitize_for_output(row.model_dump(mode="json", exclude_none=True)),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in sorted(rows, key=lambda item: item.training_row_id)
    ]
    payload = "\n".join(lines) + ("\n" if lines else "")
    encoded = payload.encode("utf-8")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_oled_curated_training_package_schema_json(
    schema: OledCuratedTrainingPackageSchema,
    path: str | Path,
) -> str:
    payload = json.dumps(_sanitize_for_output(schema.model_dump(mode="json", exclude_none=True)), sort_keys=True, indent=2) + "\n"
    encoded = payload.encode("utf-8")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_oled_curated_training_package_manifest_json(
    manifest: OledCuratedSplitTrainingPackageWriterManifest,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def oled_training_rows_output_filename(
    *,
    split: str,
    target_property_id: str,
    feature_view: str,
) -> str:
    return (
        "oled_training_rows__"
        f"{_safe_filename_token(split)}__"
        f"{_safe_filename_token(target_property_id)}__"
        f"{_safe_filename_token(feature_view)}.jsonl"
    )


def oled_training_schema_output_filename() -> str:
    return "oled_training_schema.json"


def run_oled_curated_split_training_package_writer_from_files(
    *,
    split_feature_manifest_path: str | Path,
    training_preflight_report_path: str | Path,
    split_feature_base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    policy: OledCuratedSplitTrainingPackageWriterPolicy | None = None,
    confirm_training_package_write: bool = False,
    dry_run: bool = False,
) -> OledCuratedSplitTrainingPackageWriterReport:
    writer_policy = policy or OledCuratedSplitTrainingPackageWriterPolicy()
    if not dry_run and writer_policy.require_confirmation and not confirm_training_package_write:
        raise ValueError("confirmation_required:training_package_write")
    if not dry_run and output_dir is None:
        raise ValueError("output_dir_required:training_package_write")

    split_feature_manifest = load_oled_curated_split_feature_writer_manifest_json(split_feature_manifest_path)
    base_dir = Path(split_feature_base_dir) if split_feature_base_dir is not None else Path(split_feature_manifest_path).parent
    feature_rows = load_oled_curated_split_feature_rows_from_manifest(manifest=split_feature_manifest, base_dir=base_dir)
    preflight_report = load_oled_curated_split_training_package_preflight_report_json(training_preflight_report_path)
    selection_policy = writer_policy.model_copy(update={"require_confirmation": not dry_run and writer_policy.require_confirmation})
    report = select_oled_curated_split_training_package_for_write(
        feature_rows=feature_rows,
        preflight_report=preflight_report,
        policy=selection_policy,
        confirm_training_package_write=confirm_training_package_write or dry_run,
    )
    report = _attach_source_context(
        report,
        source_split_feature_manifest_id=split_feature_manifest.manifest_id,
        source_training_preflight_status=_status_value(preflight_report.status),
    )
    if dry_run:
        report = _mark_dry_run(report)
        if output_manifest_path is not None:
            write_oled_curated_training_package_manifest_json(report.manifest, output_manifest_path)
        return report
    if not report.is_valid:
        if output_manifest_path is not None:
            write_oled_curated_training_package_manifest_json(report.manifest, output_manifest_path)
        return report

    assert output_dir is not None
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report = _write_training_package_files(report, output_root)
    if output_manifest_path is not None:
        write_oled_curated_training_package_manifest_json(report.manifest, output_manifest_path)
    return report


def load_oled_curated_training_rows_jsonl(
    path: str | Path,
) -> list[OledCuratedTrainingPackageRow]:
    rows_path = Path(path)
    _reject_forbidden_input(rows_path)
    if not rows_path.exists():
        raise ValueError(f"missing_training_rows_jsonl:{redact_oled_mineru_acceptance_path(rows_path)}")
    rows: list[OledCuratedTrainingPackageRow] = []
    for line_number, raw_line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            row = OledCuratedTrainingPackageRow.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid_training_rows_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(row.metadata):
            raise ValueError(f"absolute_path_in_training_row_metadata:{row.training_row_id}")
        rows.append(row)
    return sorted(rows, key=lambda row: row.training_row_id)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write OLED curated split ML-ready training package artifacts.")
    parser.add_argument("--split-feature-manifest", required=True, help="Path to split feature writer manifest JSON.")
    parser.add_argument("--training-preflight-report", required=True, help="Path to training-package preflight report JSON.")
    parser.add_argument("--split-feature-base-dir", help="Base directory for split feature row JSONL files.")
    parser.add_argument("--output-dir", help="Directory for training row and schema artifacts.")
    parser.add_argument("--output-manifest", help="Optional path for training package writer manifest JSON.")
    parser.add_argument("--confirm-training-package-write", action="store_true", help="Confirm ML-ready training package artifact writing.")
    parser.add_argument("--dry-run", action="store_true", help="Run selection without writing training row/schema files.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--split", action="append", default=[], help="Split name; repeat or comma-separate.")
    args = parser.parse_args(argv)

    if not args.output_dir and not args.output_manifest:
        print("output_required:dir_or_manifest", file=sys.stderr)
        return 1
    if not args.dry_run and not args.confirm_training_package_write:
        print("confirmation_required:training_package_write", file=sys.stderr)
        return 1
    try:
        policy = OledCuratedSplitTrainingPackageWriterPolicy(
            require_confirmation=not args.dry_run,
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
            feature_views=_split_cli_values(args.feature_view),
            splits=_split_cli_values(args.split) or ["train", "validation", "test"],
        )
        report = run_oled_curated_split_training_package_writer_from_files(
            split_feature_manifest_path=args.split_feature_manifest,
            training_preflight_report_path=args.training_preflight_report,
            split_feature_base_dir=args.split_feature_base_dir,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            policy=policy,
            confirm_training_package_write=args.confirm_training_package_write,
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


def _training_row_from_feature_row(row: OledCuratedSplitFeatureRowArtifact) -> OledCuratedTrainingPackageRow:
    assert row.target_value is not None
    metadata = _sanitize_for_output(row.metadata)
    metadata.update(
        {
            "ml_ready_training_row": True,
            "benchmark_validated": False,
            "model_backend_run": False,
        }
    )
    return OledCuratedTrainingPackageRow(
        training_row_id=_training_row_id(row),
        split=row.split,
        feature_row_id=row.feature_row_id,
        split_row_id=row.split_row_id,
        row_id=row.row_id,
        record_id=row.record_id,
        source_record_ids=row.source_record_ids,
        target_property_id=row.target_property_id,
        target_value=row.target_value,
        target_unit=row.target_unit,
        feature_view=row.feature_view,
        features=_sanitize_for_output(row.features),
        condition_hash=row.condition_hash,
        confidence_score=row.confidence_score,
        evidence_refs=row.evidence_refs,
        metadata=metadata,
    )


def _training_row_id(row: OledCuratedSplitFeatureRowArtifact) -> str:
    payload = {
        "split": row.split,
        "feature_row_id": row.feature_row_id,
        "target_property_id": row.target_property_id,
        "feature_view": row.feature_view,
        "condition_hash": row.condition_hash,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-training-row:{digest[:20]}"


def _schema_id(schema: OledCuratedTrainingPackageSchema) -> str:
    payload = schema.model_dump(mode="json", exclude={"schema_id"})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-training-schema:{digest[:16]}"


def _preflight_gate_findings(
    preflight_report: OledCuratedSplitTrainingPackagePreflightReport,
    policy: OledCuratedSplitTrainingPackageWriterPolicy,
) -> list[OledCuratedSplitTrainingPackageWriterFinding]:
    findings: list[OledCuratedSplitTrainingPackageWriterFinding] = []
    if policy.require_preflight_valid and not _preflight_is_valid(preflight_report):
        findings.append(
            OledCuratedSplitTrainingPackageWriterFinding(
                code="training_package_preflight_failed",
                severity="error",
                message="training package writer blocked because preflight report is invalid",
            )
        )
    if not policy.allow_preflight_warnings and preflight_report.warning_codes:
        findings.append(
            OledCuratedSplitTrainingPackageWriterFinding(
                code="training_package_preflight_warnings_present",
                severity="error",
                message="training package writer blocked because preflight warnings are disallowed",
            )
        )
    return findings


def _preflight_is_valid(report: OledCuratedSplitTrainingPackagePreflightReport) -> bool:
    return _status_value(report.status) != OledCuratedSplitTrainingPackagePreflightStatus.FAILED.value and not report.error_codes


def _row_finding(
    code: str,
    severity: Literal["error", "warning"],
    message: str,
    row: OledCuratedSplitFeatureRowArtifact,
) -> OledCuratedSplitTrainingPackageWriterFinding:
    return OledCuratedSplitTrainingPackageWriterFinding(
        code=code,
        severity=severity,
        message=message,
        split=row.split,
        target_property_id=row.target_property_id,
        feature_view=row.feature_view,
        feature_row_id=row.feature_row_id,
    )


def _file_results_for_training_rows(
    rows: list[OledCuratedTrainingPackageRow],
) -> list[OledCuratedTrainingPackageFileResult]:
    grouped: dict[tuple[str, str, str], list[OledCuratedTrainingPackageRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.split, row.target_property_id, row.feature_view)].append(row)
    return [
        OledCuratedTrainingPackageFileResult(
            split=split,
            target_property_id=target_property_id,
            feature_view=feature_view,
            artifact_kind="training_rows",
            status=OledCuratedSplitTrainingPackageWriteStatus.WRITTEN,
            row_count=len(group),
            reason_codes=["selected_for_write"],
            metadata={"training_package_written": False},
        )
        for (split, target_property_id, feature_view), group in sorted(grouped.items())
    ]


def _manifest(
    *,
    policy: OledCuratedSplitTrainingPackageWriterPolicy,
    file_results: list[OledCuratedTrainingPackageFileResult],
    training_rows: list[OledCuratedTrainingPackageRow],
    source_split_feature_manifest_id: str | None = None,
    source_training_preflight_status: str | None = None,
    output_directory: str | None = None,
    training_package_written: bool,
) -> OledCuratedSplitTrainingPackageWriterManifest:
    return OledCuratedSplitTrainingPackageWriterManifest(
        manifest_id=_manifest_id(policy, file_results),
        source_split_feature_manifest_id=source_split_feature_manifest_id,
        source_training_preflight_status=source_training_preflight_status,
        output_directory=output_directory,
        output_file_count=sum(1 for result in file_results if result.status == OledCuratedSplitTrainingPackageWriteStatus.WRITTEN),
        output_row_count=len(training_rows),
        splits=sorted({row.split for row in training_rows}),
        target_property_ids=sorted({row.target_property_id for row in training_rows}),
        feature_views=sorted({row.feature_view for row in training_rows}),
        rows_by_split=dict(sorted(Counter(row.split for row in training_rows).items())),
        rows_by_target=dict(sorted(Counter(row.target_property_id for row in training_rows).items())),
        rows_by_feature_view=dict(sorted(Counter(row.feature_view for row in training_rows).items())),
        file_results=sorted(file_results, key=lambda item: (item.artifact_kind, item.split or "", item.target_property_id or "", item.feature_view or "")),
        policy=policy,
        metadata=_safety_metadata(training_package_written=training_package_written),
    )


def _manifest_id(
    policy: OledCuratedSplitTrainingPackageWriterPolicy,
    file_results: list[OledCuratedTrainingPackageFileResult],
) -> str:
    payload = {
        "policy": policy.model_dump(mode="json"),
        "file_results": [
            {
                "split": result.split,
                "target_property_id": result.target_property_id,
                "feature_view": result.feature_view,
                "artifact_kind": result.artifact_kind,
                "row_count": result.row_count,
                "status": result.status.value,
            }
            for result in sorted(file_results, key=lambda item: (item.artifact_kind, item.split or "", item.target_property_id or "", item.feature_view or ""))
        ],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-training-package-writer:{digest[:16]}"


def _attach_source_context(
    report: OledCuratedSplitTrainingPackageWriterReport,
    *,
    source_split_feature_manifest_id: str | None,
    source_training_preflight_status: str | None,
) -> OledCuratedSplitTrainingPackageWriterReport:
    manifest = report.manifest.model_copy(
        update={
            "source_split_feature_manifest_id": source_split_feature_manifest_id,
            "source_training_preflight_status": source_training_preflight_status,
        }
    )
    return report.model_copy(update={"manifest": manifest})


def _mark_dry_run(report: OledCuratedSplitTrainingPackageWriterReport) -> OledCuratedSplitTrainingPackageWriterReport:
    refreshed_results = [
        result.model_copy(
            update={
                "reason_codes": sorted({*result.reason_codes, "dry_run_no_files_written"}),
                "metadata": {**result.metadata, "training_package_written": False},
            }
        )
        for result in report.manifest.file_results
    ]
    manifest = _manifest(
        policy=report.manifest.policy,
        file_results=refreshed_results,
        training_rows=report.training_rows,
        source_split_feature_manifest_id=report.manifest.source_split_feature_manifest_id,
        source_training_preflight_status=report.manifest.source_training_preflight_status,
        training_package_written=False,
    )
    return report.model_copy(update={"manifest": manifest})


def _write_training_package_files(
    report: OledCuratedSplitTrainingPackageWriterReport,
    output_dir: Path,
) -> OledCuratedSplitTrainingPackageWriterReport:
    grouped_rows: dict[tuple[str, str, str], list[OledCuratedTrainingPackageRow]] = defaultdict(list)
    for row in report.training_rows:
        grouped_rows[(row.split, row.target_property_id, row.feature_view)].append(row)
    refreshed_results: list[OledCuratedTrainingPackageFileResult] = []
    for result in report.manifest.file_results:
        if result.artifact_kind == "training_rows":
            assert result.split is not None and result.target_property_id is not None and result.feature_view is not None
            filename = oled_training_rows_output_filename(
                split=result.split,
                target_property_id=result.target_property_id,
                feature_view=result.feature_view,
            )
            output_sha = write_oled_curated_training_rows_jsonl(
                grouped_rows[(result.split, result.target_property_id, result.feature_view)],
                output_dir / filename,
            )
            refreshed_results.append(
                result.model_copy(
                    update={
                        "output_path": filename,
                        "output_sha256": output_sha,
                        "metadata": {**result.metadata, "training_package_written": True},
                    }
                )
            )
        elif result.artifact_kind == "schema":
            assert report.schema is not None
            filename = oled_training_schema_output_filename()
            output_sha = write_oled_curated_training_package_schema_json(report.schema, output_dir / filename)
            refreshed_results.append(
                result.model_copy(
                    update={
                        "output_path": filename,
                        "output_sha256": output_sha,
                        "metadata": {**result.metadata, "training_package_written": True},
                    }
                )
            )
        else:
            refreshed_results.append(result)
    manifest = _manifest(
        policy=report.manifest.policy,
        file_results=refreshed_results,
        training_rows=report.training_rows,
        source_split_feature_manifest_id=report.manifest.source_split_feature_manifest_id,
        source_training_preflight_status=report.manifest.source_training_preflight_status,
        output_directory=redact_oled_mineru_acceptance_path(output_dir),
        training_package_written=True,
    )
    return report.model_copy(update={"manifest": manifest})


def _column_kind(values: list[Any]) -> OledTrainingFeatureColumnKind:
    non_missing = [value for value in values if not _is_missing_value(value)]
    if not non_missing:
        return OledTrainingFeatureColumnKind.MISSING_ONLY
    kinds = {_value_kind(value) for value in non_missing}
    if len(kinds) == 1:
        return next(iter(kinds))
    return OledTrainingFeatureColumnKind.MIXED


def _value_kind(value: Any) -> OledTrainingFeatureColumnKind:
    if isinstance(value, bool):
        return OledTrainingFeatureColumnKind.BOOLEAN
    if isinstance(value, (int, float)):
        return OledTrainingFeatureColumnKind.NUMERIC
    if isinstance(value, str):
        return OledTrainingFeatureColumnKind.CATEGORICAL
    if isinstance(value, (list, tuple, set)):
        return OledTrainingFeatureColumnKind.SEQUENCE
    if isinstance(value, dict):
        return OledTrainingFeatureColumnKind.DICT
    return OledTrainingFeatureColumnKind.MIXED


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _target_property_ids(policy: OledCuratedSplitTrainingPackageWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.target_property_ids if str(item).strip()}


def _feature_views(policy: OledCuratedSplitTrainingPackageWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.feature_views if str(item).strip()}


def _splits(policy: OledCuratedSplitTrainingPackageWriterPolicy) -> set[str]:
    return {str(item).strip() for item in policy.splits if str(item).strip()}


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _status_value(status: Enum | str) -> str:
    return status.value if isinstance(status, Enum) else str(status)


def _safe_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_") or "unknown"


def _safety_metadata(*, training_package_written: bool) -> dict[str, Any]:
    return {
        "training_package_writer": True,
        "training_package_written": training_package_written,
        "ml_ready_training_data_written": training_package_written,
        "benchmark_validated": False,
        "baseline_backend_run": False,
        "model_backends_run": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


def _dedup_findings(
    findings: list[OledCuratedSplitTrainingPackageWriterFinding],
) -> list[OledCuratedSplitTrainingPackageWriterFinding]:
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    deduped: list[OledCuratedSplitTrainingPackageWriterFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.split or "",
            finding.target_property_id or "",
            finding.feature_view or "",
            finding.feature_row_id or "",
            finding.training_row_id or "",
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
            item.feature_row_id or "",
            item.training_row_id or "",
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
    "OledCuratedSplitTrainingPackageWriterPolicy",
    "OledCuratedSplitTrainingPackageWriteStatus",
    "OledCuratedTrainingPackageRow",
    "OledCuratedTrainingPackageSchema",
    "OledCuratedTrainingPackageFileResult",
    "OledCuratedSplitTrainingPackageWriterFinding",
    "OledCuratedSplitTrainingPackageWriterManifest",
    "OledCuratedSplitTrainingPackageWriterReport",
    "load_oled_curated_split_training_package_preflight_report_json",
    "build_oled_curated_training_package_rows",
    "build_oled_curated_training_package_schema",
    "select_oled_curated_split_training_package_for_write",
    "write_oled_curated_training_rows_jsonl",
    "write_oled_curated_training_package_schema_json",
    "write_oled_curated_training_package_manifest_json",
    "oled_training_rows_output_filename",
    "oled_training_schema_output_filename",
    "run_oled_curated_split_training_package_writer_from_files",
    "load_oled_curated_training_rows_jsonl",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
