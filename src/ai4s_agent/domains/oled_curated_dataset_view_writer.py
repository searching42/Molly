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

from ai4s_agent.domains.oled_curated_gold_view_preflight import (
    OledCuratedGoldViewPreflightPolicy,
    OledCuratedGoldViewPreflightReport,
    OledCuratedGoldViewPreflightStatus,
    load_oled_curated_gold_manifest_json,
    load_oled_curated_gold_records_jsonl,
    run_oled_curated_gold_view_preflight_from_files,
)
from ai4s_agent.domains.oled_curated_gold_writer import OledCuratedGoldManifest
from ai4s_agent.domains.oled_dataset_views import (
    OledDatasetViewKind,
    OledDatasetViewReport,
    build_oled_dataset_view,
)
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledCuratedDatasetViewWriterPolicy(BaseModel):
    require_confirmation: bool = True
    require_preflight_valid: bool = True
    allow_preflight_warnings: bool = True
    include_empty_views: bool = False
    include_feature_payload: bool = False
    view_kinds: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    write_training_data: bool = False
    run_leakage_splits: bool = False
    run_feature_materialization: bool = False
    run_model_backends: bool = False


class OledCuratedDatasetViewWriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class OledCuratedDatasetViewRowArtifact(BaseModel):
    row_id: str
    view_kind: str
    target_property_id: str

    record_id: str
    source_record_ids: list[str] = Field(default_factory=list)

    target_value: float | int | str | None = None
    target_unit: str | None = None
    target_reported_value_text: str | None = None
    target_reported_decimal_places: int | None = Field(default=None, ge=0)
    target_reported_unit: str | None = None
    target_layer: str

    condition_hash: str | None = None
    dedup_key_hash: str | None = None

    evidence_refs: list[str] = Field(default_factory=list)
    confidence_score: float | None = None

    feature_view: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_record_ids", "evidence_refs")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedDatasetViewFileResult(BaseModel):
    view_kind: str
    target_property_id: str

    status: OledCuratedDatasetViewWriteStatus
    row_count: int = 0

    output_jsonl_path: str | None = None
    output_sha256: str | None = None

    view_error_codes: list[str] = Field(default_factory=list)
    view_warning_codes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("view_error_codes", "view_warning_codes", "reason_codes")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedDatasetViewWriterFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    view_kind: str | None = None
    target_property_id: str | None = None
    output_jsonl_path: str | None = None


class OledCuratedDatasetViewWriterManifest(BaseModel):
    manifest_id: str

    source_curated_gold_sha256: str | None = None
    source_curated_gold_manifest_id: str | None = None
    source_preflight_status: str | None = None

    output_directory: str | None = None
    output_file_count: int = 0
    output_row_count: int = 0

    status_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)

    file_results: list[OledCuratedDatasetViewFileResult] = Field(default_factory=list)

    policy: OledCuratedDatasetViewWriterPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(result.status == OledCuratedDatasetViewWriteStatus.REJECTED for result in self.file_results)


class OledCuratedDatasetViewWriterReport(BaseModel):
    manifest: OledCuratedDatasetViewWriterManifest
    row_artifacts: list[OledCuratedDatasetViewRowArtifact] = Field(default_factory=list)
    findings: list[OledCuratedDatasetViewWriterFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def build_oled_curated_dataset_view_row_artifacts(
    view_report: OledDatasetViewReport,
    *,
    include_feature_payload: bool = False,
) -> list[OledCuratedDatasetViewRowArtifact]:
    artifacts: list[OledCuratedDatasetViewRowArtifact] = []
    for row in sorted(view_report.rows, key=lambda item: (item.record_id, item.condition_hash or "", item.dedup_key_hash or "")):
        features = _sanitize_for_output(row.features) if include_feature_payload else {}
        metadata = _sanitize_for_output(row.metadata)
        metadata.update(
            {
                "dataset_view_row_artifact": True,
                "training_data_record": False,
                "feature_payload_omitted": not include_feature_payload,
            }
        )
        if include_feature_payload:
            metadata.pop("feature_payload_omitted", None)
        artifacts.append(
            OledCuratedDatasetViewRowArtifact(
                row_id=_row_id(
                    view_kind=row.view_kind.value,
                    target_property_id=row.target_property_id,
                    record_id=row.record_id,
                    condition_hash=row.condition_hash,
                    dedup_key_hash=row.dedup_key_hash,
                    source_record_ids=row.source_record_ids,
                ),
                view_kind=row.view_kind.value,
                target_property_id=row.target_property_id,
                record_id=row.record_id,
                source_record_ids=row.source_record_ids,
                target_value=row.target_value,
                target_unit=row.target_unit,
                target_reported_value_text=row.target_reported_value_text,
                target_reported_decimal_places=row.target_reported_decimal_places,
                target_reported_unit=row.target_reported_unit,
                target_layer=row.target_layer.value,
                condition_hash=row.condition_hash,
                dedup_key_hash=row.dedup_key_hash,
                evidence_refs=row.evidence_refs,
                confidence_score=row.confidence_score,
                feature_view=row.feature_view.value if row.feature_view is not None else None,
                features=features,
                metadata=metadata,
            )
        )
    return artifacts


def select_oled_curated_dataset_views_for_write(
    records: Iterable[OledGoldDatasetRecord],
    *,
    preflight_report: OledCuratedGoldViewPreflightReport | None = None,
    policy: OledCuratedDatasetViewWriterPolicy | None = None,
    confirm_dataset_view_write: bool = False,
) -> OledCuratedDatasetViewWriterReport:
    writer_policy = policy or OledCuratedDatasetViewWriterPolicy()
    if writer_policy.require_confirmation and not confirm_dataset_view_write:
        raise ValueError("confirmation_required:dataset_view_write")

    gold_records = sorted(list(records), key=lambda item: item.record_id)
    preflight_findings = _preflight_gate_findings(preflight_report, writer_policy)
    if any(finding.severity == "error" for finding in preflight_findings):
        file_results = _preflight_blocked_file_results(writer_policy, reason_codes=[finding.code for finding in preflight_findings])
        return OledCuratedDatasetViewWriterReport(
            manifest=_manifest(
                policy=writer_policy,
                file_results=file_results,
                row_artifacts=[],
                source_preflight_status=preflight_report.status.value if preflight_report is not None else None,
                source_curated_gold_sha256=preflight_report.input_sha256 if preflight_report is not None else None,
            ),
            row_artifacts=[],
            findings=preflight_findings,
        )

    file_results: list[OledCuratedDatasetViewFileResult] = []
    row_artifacts: list[OledCuratedDatasetViewRowArtifact] = []
    findings: list[OledCuratedDatasetViewWriterFinding] = list(preflight_findings)
    for view_kind in _view_kinds(writer_policy):
        for target_property_id in _target_property_ids(writer_policy):
            result, artifacts, result_findings = _select_single_view(
                gold_records,
                view_kind=view_kind,
                target_property_id=target_property_id,
                policy=writer_policy,
            )
            file_results.append(result)
            row_artifacts.extend(artifacts)
            findings.extend(result_findings)

    return OledCuratedDatasetViewWriterReport(
        manifest=_manifest(
            policy=writer_policy,
            file_results=file_results,
            row_artifacts=row_artifacts,
            source_preflight_status=preflight_report.status.value if preflight_report is not None else None,
            source_curated_gold_sha256=preflight_report.input_sha256 if preflight_report is not None else None,
        ),
        row_artifacts=sorted(row_artifacts, key=lambda item: item.row_id),
        findings=_dedup_findings(findings),
    )


def write_oled_curated_dataset_view_rows_jsonl(
    rows: Iterable[OledCuratedDatasetViewRowArtifact],
    path: str | Path,
) -> str:
    lines = [
        json.dumps(
            _sanitize_for_output(row.model_dump(mode="json", exclude_none=True)),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in sorted(rows, key=lambda item: item.row_id)
    ]
    payload = "\n".join(lines) + ("\n" if lines else "")
    encoded = payload.encode("utf-8")
    Path(path).write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_oled_curated_dataset_view_manifest_json(
    manifest: OledCuratedDatasetViewWriterManifest,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def oled_dataset_view_output_filename(
    *,
    view_kind: str,
    target_property_id: str,
) -> str:
    return f"oled_view__{_safe_filename_token(view_kind)}__{_safe_filename_token(target_property_id)}.jsonl"


def run_oled_curated_dataset_view_writer_from_files(
    *,
    curated_gold_jsonl_path: str | Path,
    curated_gold_manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    output_preflight_report_path: str | Path | None = None,
    policy: OledCuratedDatasetViewWriterPolicy | None = None,
    confirm_dataset_view_write: bool = False,
    dry_run: bool = False,
) -> OledCuratedDatasetViewWriterReport:
    writer_policy = policy or OledCuratedDatasetViewWriterPolicy()
    if not dry_run and writer_policy.require_confirmation and not confirm_dataset_view_write:
        raise ValueError("confirmation_required:dataset_view_write")
    if not dry_run and output_dir is None:
        raise ValueError("output_dir_required:dataset_view_write")

    records = load_oled_curated_gold_records_jsonl(curated_gold_jsonl_path)
    source_manifest: OledCuratedGoldManifest | None = (
        load_oled_curated_gold_manifest_json(curated_gold_manifest_path)
        if curated_gold_manifest_path is not None
        else None
    )
    preflight = run_oled_curated_gold_view_preflight_from_files(
        curated_gold_jsonl_path=curated_gold_jsonl_path,
        manifest_path=curated_gold_manifest_path,
        output_report_path=output_preflight_report_path,
        policy=OledCuratedGoldViewPreflightPolicy(
            include_empty_views=True,
            target_property_ids=writer_policy.target_property_ids,
            view_kinds=writer_policy.view_kinds,
        ),
    )
    selection_policy = writer_policy.model_copy(update={"require_confirmation": not dry_run and writer_policy.require_confirmation})
    report = select_oled_curated_dataset_views_for_write(
        records,
        preflight_report=preflight,
        policy=selection_policy,
        confirm_dataset_view_write=confirm_dataset_view_write or dry_run,
    )
    report = _attach_source_context(
        report,
        source_curated_gold_sha256=preflight.input_sha256,
        source_curated_gold_manifest_id=source_manifest.manifest_id if source_manifest is not None else None,
        source_preflight_status=preflight.status.value,
    )
    if dry_run:
        report = _mark_dry_run(report)
        if output_manifest_path is not None:
            write_oled_curated_dataset_view_manifest_json(report.manifest, output_manifest_path)
        return report

    assert output_dir is not None
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report = _write_selected_view_files(report, output_root)
    if output_manifest_path is not None:
        write_oled_curated_dataset_view_manifest_json(report.manifest, output_manifest_path)
    return report


def load_oled_curated_dataset_view_rows_jsonl(
    path: str | Path,
) -> list[OledCuratedDatasetViewRowArtifact]:
    rows_path = Path(path)
    _reject_forbidden_input(rows_path)
    if not rows_path.exists():
        raise ValueError(f"missing_dataset_view_rows_jsonl:{redact_oled_mineru_acceptance_path(rows_path)}")
    rows: list[OledCuratedDatasetViewRowArtifact] = []
    for line_number, raw_line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            row = OledCuratedDatasetViewRowArtifact.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid_dataset_view_rows_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(row.metadata):
            raise ValueError(f"absolute_path_in_dataset_view_row_metadata:{row.row_id}")
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write curated OLED dataset-view row artifacts under explicit gates.")
    parser.add_argument("--curated-gold-jsonl", required=True, help="Path to curated gold-record JSONL.")
    parser.add_argument("--curated-gold-manifest", help="Optional curated gold writer manifest JSON.")
    parser.add_argument("--output-dir", help="Directory for dataset-view JSONL files.")
    parser.add_argument("--output-manifest", help="Optional path for dataset-view writer manifest JSON.")
    parser.add_argument("--output-preflight-report", help="Optional path for preflight report JSON.")
    parser.add_argument("--confirm-dataset-view-write", action="store_true", help="Confirm dataset-view row JSONL writing.")
    parser.add_argument("--dry-run", action="store_true", help="Run preflight/selection without writing row JSONL files.")
    parser.add_argument("--view-kind", action="append", default=[], help="View kind to write; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    parser.add_argument("--include-feature-payload", action="store_true", help="Include row feature payloads in JSONL.")
    args = parser.parse_args(argv)

    if not args.output_dir and not args.output_manifest:
        print("output_required:dir_or_manifest", file=sys.stderr)
        return 1
    if not args.dry_run and not args.confirm_dataset_view_write:
        print("confirmation_required:dataset_view_write", file=sys.stderr)
        return 1
    try:
        policy = OledCuratedDatasetViewWriterPolicy(
            require_confirmation=not args.dry_run,
            include_feature_payload=args.include_feature_payload,
            view_kinds=_split_cli_values(args.view_kind),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
        )
        report = run_oled_curated_dataset_view_writer_from_files(
            curated_gold_jsonl_path=args.curated_gold_jsonl,
            curated_gold_manifest_path=args.curated_gold_manifest,
            output_dir=args.output_dir,
            output_manifest_path=args.output_manifest,
            output_preflight_report_path=args.output_preflight_report,
            policy=policy,
            confirm_dataset_view_write=args.confirm_dataset_view_write,
            dry_run=args.dry_run,
        )
        summary = {
            "output_file_count": report.manifest.output_file_count,
            "output_row_count": report.manifest.output_row_count,
            "status_counts": report.manifest.status_counts,
            "reason_code_counts": report.manifest.reason_code_counts,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _select_single_view(
    records: list[OledGoldDatasetRecord],
    *,
    view_kind: OledDatasetViewKind,
    target_property_id: str,
    policy: OledCuratedDatasetViewWriterPolicy,
) -> tuple[OledCuratedDatasetViewFileResult, list[OledCuratedDatasetViewRowArtifact], list[OledCuratedDatasetViewWriterFinding]]:
    try:
        view_report = build_oled_dataset_view(records, view_kind=view_kind, target_property_id=target_property_id)
    except Exception as exc:
        message = str(exc).splitlines()[0]
        if message.startswith("no_gold_records_for_target:"):
            return _empty_view_result(view_kind, target_property_id, policy, message), [], []
        result = OledCuratedDatasetViewFileResult(
            view_kind=view_kind.value,
            target_property_id=target_property_id,
            status=OledCuratedDatasetViewWriteStatus.REJECTED,
            reason_codes=["view_build_failed"],
            metadata={"exception": message},
        )
        return result, [], [
            OledCuratedDatasetViewWriterFinding(
                code="view_build_failed",
                severity="error",
                message=message,
                view_kind=view_kind.value,
                target_property_id=target_property_id,
            )
        ]

    artifacts = build_oled_curated_dataset_view_row_artifacts(
        view_report,
        include_feature_payload=policy.include_feature_payload,
    )
    if not artifacts:
        return _empty_view_result(view_kind, target_property_id, policy, "dataset view produced zero rows"), [], []
    error_codes = view_report.error_codes
    warning_codes = view_report.warning_codes
    if error_codes:
        result = OledCuratedDatasetViewFileResult(
            view_kind=view_kind.value,
            target_property_id=target_property_id,
            status=OledCuratedDatasetViewWriteStatus.REJECTED,
            row_count=0,
            view_error_codes=error_codes,
            view_warning_codes=warning_codes,
            reason_codes=["view_errors_present"],
        )
        findings = [
            OledCuratedDatasetViewWriterFinding(
                code=code,
                severity="error",
                message=f"dataset view error `{code}`",
                view_kind=view_kind.value,
                target_property_id=target_property_id,
            )
            for code in error_codes
        ]
        return result, [], findings

    reason_codes = ["selected_for_write"]
    if not policy.include_feature_payload:
        reason_codes.append("feature_payload_omitted")
    result = OledCuratedDatasetViewFileResult(
        view_kind=view_kind.value,
        target_property_id=target_property_id,
        status=OledCuratedDatasetViewWriteStatus.WRITTEN,
        row_count=len(artifacts),
        view_warning_codes=warning_codes,
        reason_codes=reason_codes,
        metadata={"dataset_view_rows_written": False},
    )
    findings = [
        OledCuratedDatasetViewWriterFinding(
            code=code,
            severity="warning",
            message=f"dataset view warning `{code}`",
            view_kind=view_kind.value,
            target_property_id=target_property_id,
        )
        for code in warning_codes
    ]
    return result, artifacts, findings


def _empty_view_result(
    view_kind: OledDatasetViewKind,
    target_property_id: str,
    policy: OledCuratedDatasetViewWriterPolicy,
    message: str,
) -> OledCuratedDatasetViewFileResult:
    if policy.include_empty_views:
        return OledCuratedDatasetViewFileResult(
            view_kind=view_kind.value,
            target_property_id=target_property_id,
            status=OledCuratedDatasetViewWriteStatus.SKIPPED,
            row_count=0,
            reason_codes=["empty_view_included"],
            metadata={"empty_view_reason": message, "dataset_view_rows_written": False},
        )
    return OledCuratedDatasetViewFileResult(
        view_kind=view_kind.value,
        target_property_id=target_property_id,
        status=OledCuratedDatasetViewWriteStatus.SKIPPED,
        row_count=0,
        reason_codes=["empty_view_skipped"],
        metadata={"empty_view_reason": message, "dataset_view_rows_written": False},
    )


def _preflight_gate_findings(
    preflight_report: OledCuratedGoldViewPreflightReport | None,
    policy: OledCuratedDatasetViewWriterPolicy,
) -> list[OledCuratedDatasetViewWriterFinding]:
    if preflight_report is None:
        return []
    findings: list[OledCuratedDatasetViewWriterFinding] = []
    if policy.require_preflight_valid and not preflight_report.is_valid:
        findings.append(
            OledCuratedDatasetViewWriterFinding(
                code="preflight_failed",
                severity="error",
                message="dataset-view writer blocked because preflight report is invalid",
            )
        )
    if not policy.allow_preflight_warnings and preflight_report.warning_codes:
        findings.append(
            OledCuratedDatasetViewWriterFinding(
                code="preflight_warnings_present",
                severity="error",
                message="dataset-view writer blocked because preflight warnings are disallowed",
            )
        )
    return findings


def _preflight_blocked_file_results(
    policy: OledCuratedDatasetViewWriterPolicy,
    *,
    reason_codes: list[str],
) -> list[OledCuratedDatasetViewFileResult]:
    return [
        OledCuratedDatasetViewFileResult(
            view_kind=view_kind.value,
            target_property_id=target_property_id,
            status=OledCuratedDatasetViewWriteStatus.REJECTED,
            reason_codes=reason_codes,
        )
        for view_kind in _view_kinds(policy)
        for target_property_id in _target_property_ids(policy)
    ]


def _write_selected_view_files(
    report: OledCuratedDatasetViewWriterReport,
    output_dir: Path,
) -> OledCuratedDatasetViewWriterReport:
    grouped_rows: dict[tuple[str, str], list[OledCuratedDatasetViewRowArtifact]] = defaultdict(list)
    for row in report.row_artifacts:
        grouped_rows[(row.view_kind, row.target_property_id)].append(row)

    refreshed_results: list[OledCuratedDatasetViewFileResult] = []
    for result in report.manifest.file_results:
        if result.status != OledCuratedDatasetViewWriteStatus.WRITTEN or result.row_count <= 0:
            refreshed_results.append(result)
            continue
        filename = oled_dataset_view_output_filename(
            view_kind=result.view_kind,
            target_property_id=result.target_property_id,
        )
        output_path = output_dir / filename
        output_hash = write_oled_curated_dataset_view_rows_jsonl(
            grouped_rows[(result.view_kind, result.target_property_id)],
            output_path,
        )
        refreshed_results.append(
            result.model_copy(
                update={
                    "output_jsonl_path": filename,
                    "output_sha256": output_hash,
                    "metadata": {
                        **result.metadata,
                        "dataset_view_rows_written": True,
                    },
                }
            )
        )

    manifest = _manifest(
        policy=report.manifest.policy,
        file_results=refreshed_results,
        row_artifacts=report.row_artifacts,
        source_curated_gold_sha256=report.manifest.source_curated_gold_sha256,
        source_curated_gold_manifest_id=report.manifest.source_curated_gold_manifest_id,
        source_preflight_status=report.manifest.source_preflight_status,
        output_directory=redact_oled_mineru_acceptance_path(output_dir),
        dataset_view_rows_written=True,
    )
    return report.model_copy(update={"manifest": manifest})


def _mark_dry_run(report: OledCuratedDatasetViewWriterReport) -> OledCuratedDatasetViewWriterReport:
    refreshed_results = [
        result.model_copy(
            update={
                "reason_codes": sorted({*result.reason_codes, "dry_run_no_rows_written"}),
                "metadata": {**result.metadata, "dataset_view_rows_written": False},
            }
        )
        for result in report.manifest.file_results
    ]
    manifest = _manifest(
        policy=report.manifest.policy,
        file_results=refreshed_results,
        row_artifacts=report.row_artifacts,
        source_curated_gold_sha256=report.manifest.source_curated_gold_sha256,
        source_curated_gold_manifest_id=report.manifest.source_curated_gold_manifest_id,
        source_preflight_status=report.manifest.source_preflight_status,
        dataset_view_rows_written=False,
    )
    return report.model_copy(update={"manifest": manifest})


def _attach_source_context(
    report: OledCuratedDatasetViewWriterReport,
    *,
    source_curated_gold_sha256: str | None,
    source_curated_gold_manifest_id: str | None,
    source_preflight_status: str | None,
) -> OledCuratedDatasetViewWriterReport:
    manifest = report.manifest.model_copy(
        update={
            "source_curated_gold_sha256": source_curated_gold_sha256,
            "source_curated_gold_manifest_id": source_curated_gold_manifest_id,
            "source_preflight_status": source_preflight_status,
        }
    )
    return report.model_copy(update={"manifest": manifest})


def _manifest(
    *,
    policy: OledCuratedDatasetViewWriterPolicy,
    file_results: list[OledCuratedDatasetViewFileResult],
    row_artifacts: list[OledCuratedDatasetViewRowArtifact],
    source_curated_gold_sha256: str | None = None,
    source_curated_gold_manifest_id: str | None = None,
    source_preflight_status: str | None = None,
    output_directory: str | None = None,
    dataset_view_rows_written: bool = False,
) -> OledCuratedDatasetViewWriterManifest:
    return OledCuratedDatasetViewWriterManifest(
        manifest_id=_manifest_id(policy, file_results),
        source_curated_gold_sha256=source_curated_gold_sha256,
        source_curated_gold_manifest_id=source_curated_gold_manifest_id,
        source_preflight_status=source_preflight_status,
        output_directory=output_directory,
        output_file_count=sum(1 for result in file_results if result.status == OledCuratedDatasetViewWriteStatus.WRITTEN and result.row_count > 0),
        output_row_count=len(row_artifacts),
        status_counts=dict(sorted(Counter(result.status.value for result in file_results).items())),
        reason_code_counts=dict(sorted(Counter(code for result in file_results for code in result.reason_codes).items())),
        file_results=sorted(file_results, key=lambda item: (item.view_kind, item.target_property_id)),
        policy=policy,
        metadata=_safety_metadata(dataset_view_rows_written=dataset_view_rows_written),
    )


def _manifest_id(
    policy: OledCuratedDatasetViewWriterPolicy,
    file_results: list[OledCuratedDatasetViewFileResult],
) -> str:
    payload = {
        "policy": policy.model_dump(mode="json"),
        "file_results": [
            {
                "view_kind": result.view_kind,
                "target_property_id": result.target_property_id,
                "row_count": result.row_count,
                "status": result.status.value,
            }
            for result in sorted(file_results, key=lambda item: (item.view_kind, item.target_property_id))
        ],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-curated-dataset-view-writer:{digest[:16]}"


def _row_id(
    *,
    view_kind: str,
    target_property_id: str,
    record_id: str,
    condition_hash: str | None,
    dedup_key_hash: str | None,
    source_record_ids: list[str],
) -> str:
    payload = {
        "view_kind": view_kind,
        "target_property_id": target_property_id,
        "record_id": record_id,
        "condition_hash": condition_hash,
        "dedup_key_hash": dedup_key_hash,
        "source_record_ids": sorted(source_record_ids),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-view-row:{digest[:20]}"


def _view_kinds(policy: OledCuratedDatasetViewWriterPolicy) -> list[OledDatasetViewKind]:
    if not policy.view_kinds:
        return list(OledDatasetViewKind)
    return [OledDatasetViewKind(str(item)) for item in policy.view_kinds]


def _target_property_ids(policy: OledCuratedDatasetViewWriterPolicy) -> list[str]:
    return sorted({str(item).strip() for item in policy.target_property_ids if str(item).strip()})


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _safe_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_") or "unknown"


def _safety_metadata(*, dataset_view_rows_written: bool) -> dict[str, Any]:
    return {
        "dataset_view_writer": True,
        "dataset_view_rows_written": dataset_view_rows_written,
        "training_data_written": False,
        "leakage_splits_run": False,
        "feature_materialization_outputs_written": False,
        "model_backends_run": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


def _dedup_findings(
    findings: list[OledCuratedDatasetViewWriterFinding],
) -> list[OledCuratedDatasetViewWriterFinding]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[OledCuratedDatasetViewWriterFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.view_kind or "",
            finding.target_property_id or "",
            finding.output_jsonl_path or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return sorted(deduped, key=lambda item: (item.severity, item.code, item.view_kind or "", item.target_property_id or ""))


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
    "OledCuratedDatasetViewWriterPolicy",
    "OledCuratedDatasetViewWriteStatus",
    "OledCuratedDatasetViewRowArtifact",
    "OledCuratedDatasetViewFileResult",
    "OledCuratedDatasetViewWriterFinding",
    "OledCuratedDatasetViewWriterManifest",
    "OledCuratedDatasetViewWriterReport",
    "build_oled_curated_dataset_view_row_artifacts",
    "select_oled_curated_dataset_views_for_write",
    "write_oled_curated_dataset_view_rows_jsonl",
    "write_oled_curated_dataset_view_manifest_json",
    "oled_dataset_view_output_filename",
    "run_oled_curated_dataset_view_writer_from_files",
    "load_oled_curated_dataset_view_rows_jsonl",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
