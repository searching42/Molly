from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from ai4s_agent.domains.oled_baseline_loop import OledBaselineFeatureView
from ai4s_agent.domains.oled_curated_gold_view_preflight import load_oled_curated_gold_records_jsonl
from ai4s_agent.domains.oled_curated_split_dataset_view_writer import (
    OledCuratedSplitDatasetViewRowArtifact,
    OledCuratedSplitDatasetViewWriteStatus,
    OledCuratedSplitDatasetViewWriterManifest,
    load_oled_curated_split_dataset_view_rows_jsonl,
)
from ai4s_agent.domains.oled_feature_materialization import (
    OledFeatureMaterializationRow,
    materialize_oled_baseline_feature_table,
)
from ai4s_agent.domains.oled_gold_validation import OledGoldDatasetRecord, validate_oled_gold_dataset
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledCuratedSplitFeaturePreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledSplitFeatureRowAlignmentStatus(str, Enum):
    MATCHED = "matched"
    MISSING_FEATURE_ROW = "missing_feature_row"
    AMBIGUOUS_FEATURE_ROW = "ambiguous_feature_row"
    TARGET_MISMATCH = "target_mismatch"


class OledCuratedSplitFeaturePreflightPolicy(BaseModel):
    require_gold_validation_success: bool = True
    require_split_row_manifest_sha256: bool = True
    require_all_split_rows_matched: bool = True
    fail_on_target_mismatch: bool = True
    fail_on_missing_features: bool = False
    feature_views: list[str] = Field(default_factory=list)
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])


class OledSplitFeatureRowAlignment(BaseModel):
    split_row_id: str
    row_id: str
    split: str
    view_kind: str
    target_property_id: str
    record_id: str

    feature_view: str
    status: OledSplitFeatureRowAlignmentStatus

    feature_row_record_id: str | None = None
    feature_row_condition_hash: str | None = None

    target_value: float | int | str | None = None
    feature_target_value: float | int | str | None = None
    target_unit: str | None = None
    feature_target_unit: str | None = None

    feature_column_count: int = 0
    missing_feature_columns: list[str] = Field(default_factory=list)
    present_feature_columns: list[str] = Field(default_factory=list)

    evidence_refs: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledSplitFeaturePreflightSummary(BaseModel):
    split: str
    target_property_id: str
    feature_view: str

    input_split_row_count: int
    matched_row_count: int
    missing_feature_row_count: int
    ambiguous_feature_row_count: int
    target_mismatch_count: int

    feature_column_count: int = 0
    missing_feature_column_counts: dict[str, int] = Field(default_factory=dict)
    alignment_status_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)


class OledCuratedSplitFeaturePreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str

    split: str | None = None
    target_property_id: str | None = None
    feature_view: str | None = None
    split_row_id: str | None = None
    record_id: str | None = None


class OledCuratedSplitFeaturePreflightReport(BaseModel):
    status: OledCuratedSplitFeaturePreflightStatus

    input_gold_record_count: int
    input_split_row_count: int

    target_property_ids: list[str] = Field(default_factory=list)
    feature_views: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=list)

    gold_validation_error_codes: list[str] = Field(default_factory=list)
    gold_validation_warning_codes: list[str] = Field(default_factory=list)

    row_alignments: list[OledSplitFeatureRowAlignment] = Field(default_factory=list)
    summaries: list[OledSplitFeaturePreflightSummary] = Field(default_factory=list)

    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)
    rows_by_split: dict[str, int] = Field(default_factory=dict)

    findings: list[OledCuratedSplitFeaturePreflightFinding] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledCuratedSplitFeaturePreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_curated_split_dataset_view_writer_manifest_json(
    path: str | Path,
) -> OledCuratedSplitDatasetViewWriterManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_split_dataset_view_writer_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = OledCuratedSplitDatasetViewWriterManifest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid_split_dataset_view_writer_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    if _contains_absolute_path(manifest.metadata):
        raise ValueError(f"absolute_path_in_split_dataset_view_writer_manifest_metadata:{manifest.manifest_id}")
    for result in manifest.file_results:
        if _contains_absolute_path(result.metadata):
            raise ValueError(
                "absolute_path_in_split_dataset_view_writer_file_result_metadata:"
                f"{result.split}:{result.view_kind}:{result.target_property_id}"
            )
    return manifest


def load_oled_curated_split_dataset_view_rows_from_manifest(
    *,
    manifest: OledCuratedSplitDatasetViewWriterManifest,
    base_dir: str | Path,
) -> list[OledCuratedSplitDatasetViewRowArtifact]:
    rows: list[OledCuratedSplitDatasetViewRowArtifact] = []
    root = Path(base_dir)
    for result in sorted(manifest.file_results, key=lambda item: (item.split, item.view_kind, item.target_property_id)):
        if result.status != OledCuratedSplitDatasetViewWriteStatus.WRITTEN or not result.output_jsonl_path:
            continue
        row_path = Path(result.output_jsonl_path)
        if row_path.suffix.lower() != ".jsonl":
            raise ValueError(f"forbidden_split_dataset_view_rows_input:{redact_oled_mineru_acceptance_path(row_path)}")
        resolved_path = row_path if row_path.is_absolute() else root / row_path
        if result.output_sha256:
            actual_sha = _sha256_file(resolved_path)
            if actual_sha != result.output_sha256:
                raise ValueError(f"split_dataset_view_rows_sha256_mismatch:{redact_oled_mineru_acceptance_path(row_path)}")
        rows.extend(load_oled_curated_split_dataset_view_rows_jsonl(resolved_path))
    return sorted(rows, key=lambda row: row.split_row_id)


def run_oled_curated_split_feature_preflight(
    *,
    gold_records: Iterable[OledGoldDatasetRecord],
    split_rows: Iterable[OledCuratedSplitDatasetViewRowArtifact],
    policy: OledCuratedSplitFeaturePreflightPolicy | None = None,
) -> OledCuratedSplitFeaturePreflightReport:
    preflight_policy = policy or OledCuratedSplitFeaturePreflightPolicy()
    records = sorted(list(gold_records), key=lambda record: record.record_id)
    rows = sorted(list(split_rows), key=lambda row: (row.split, row.target_property_id, row.record_id, row.split_row_id))
    target_property_ids = _target_property_ids(preflight_policy)
    feature_views = _feature_views(preflight_policy)
    findings: list[OledCuratedSplitFeaturePreflightFinding] = []

    gold_report = validate_oled_gold_dataset(records)
    gold_error_codes = gold_report.error_codes
    gold_warning_codes = gold_report.warning_codes
    if gold_error_codes and preflight_policy.require_gold_validation_success:
        findings.append(
            OledCuratedSplitFeaturePreflightFinding(
                code="gold_validation_errors_present",
                severity="error",
                message="gold validation errors block split feature preflight",
            )
        )

    feature_rows_by_key: dict[tuple[str, str], list[OledFeatureMaterializationRow]] = {}
    if not gold_error_codes or not preflight_policy.require_gold_validation_success:
        for target_property_id in target_property_ids:
            for feature_view in feature_views:
                try:
                    table = materialize_oled_baseline_feature_table(
                        records,
                        feature_view=feature_view,
                        target_property_id=target_property_id,
                    )
                    feature_rows_by_key[(target_property_id, feature_view.value)] = table.rows
                except Exception as exc:
                    message = str(exc).splitlines()[0]
                    if not message.startswith("no_gold_records_for_target:"):
                        findings.append(
                            OledCuratedSplitFeaturePreflightFinding(
                                code="feature_materialization_failed",
                                severity="error",
                                message=message,
                                target_property_id=target_property_id,
                                feature_view=feature_view.value,
                            )
                        )
                    feature_rows_by_key[(target_property_id, feature_view.value)] = []

    alignments: list[OledSplitFeatureRowAlignment] = []
    processed_rows = [row for row in rows if row.target_property_id in target_property_ids]
    for row in processed_rows:
        for feature_view in feature_views:
            alignment = _align_split_row(
                row,
                feature_rows_by_key.get((row.target_property_id, feature_view.value), []),
                feature_view=feature_view.value,
                policy=preflight_policy,
            )
            alignments.append(alignment)
            findings.extend(_findings_for_alignment(alignment, preflight_policy))

    report = OledCuratedSplitFeaturePreflightReport(
        status=OledCuratedSplitFeaturePreflightStatus.PASSED,
        input_gold_record_count=len(records),
        input_split_row_count=len(rows),
        target_property_ids=target_property_ids,
        feature_views=[feature_view.value for feature_view in feature_views],
        splits=sorted({row.split for row in rows}),
        gold_validation_error_codes=gold_error_codes,
        gold_validation_warning_codes=gold_warning_codes,
        row_alignments=sorted(alignments, key=lambda item: (item.split, item.target_property_id, item.feature_view, item.split_row_id)),
        summaries=[],
        findings=_dedup_findings(findings),
        metadata=_safety_metadata(),
    )
    return _refresh_report(report, rows)


def run_oled_curated_split_feature_preflight_from_files(
    *,
    curated_gold_jsonl_path: str | Path,
    split_dataset_view_manifest_path: str | Path,
    split_dataset_view_base_dir: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledCuratedSplitFeaturePreflightPolicy | None = None,
) -> OledCuratedSplitFeaturePreflightReport:
    records = load_oled_curated_gold_records_jsonl(curated_gold_jsonl_path)
    manifest = load_oled_curated_split_dataset_view_writer_manifest_json(split_dataset_view_manifest_path)
    base_dir = Path(split_dataset_view_base_dir) if split_dataset_view_base_dir is not None else Path(split_dataset_view_manifest_path).parent
    rows = load_oled_curated_split_dataset_view_rows_from_manifest(manifest=manifest, base_dir=base_dir)
    report = run_oled_curated_split_feature_preflight(
        gold_records=records,
        split_rows=rows,
        policy=policy,
    )
    if output_report_path is not None:
        write_oled_curated_split_feature_preflight_report_json(report, output_report_path)
    return report


def write_oled_curated_split_feature_preflight_report_json(
    report: OledCuratedSplitFeaturePreflightReport,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED curated split feature materialization preflight.")
    parser.add_argument("--curated-gold-jsonl", required=True, help="Path to curated gold-record JSONL.")
    parser.add_argument("--split-dataset-view-manifest", required=True, help="Path to split dataset-view writer manifest JSON.")
    parser.add_argument("--split-dataset-view-base-dir", help="Base directory for split dataset-view row JSONL paths.")
    parser.add_argument("--output-report", help="Optional path for feature preflight report JSON.")
    parser.add_argument("--feature-view", action="append", default=[], help="Feature view; repeat or comma-separate.")
    parser.add_argument("--target-property-id", action="append", default=[], help="Target property id; repeat or comma-separate.")
    args = parser.parse_args(argv)

    try:
        policy = OledCuratedSplitFeaturePreflightPolicy(
            feature_views=_split_cli_values(args.feature_view),
            target_property_ids=_split_cli_values(args.target_property_id) or ["eqe_percent", "plqy", "delta_e_st_ev"],
        )
        report = run_oled_curated_split_feature_preflight_from_files(
            curated_gold_jsonl_path=args.curated_gold_jsonl,
            split_dataset_view_manifest_path=args.split_dataset_view_manifest,
            split_dataset_view_base_dir=args.split_dataset_view_base_dir,
            output_report_path=args.output_report,
            policy=policy,
        )
        summary = {
            "status": report.status.value,
            "input_gold_record_count": report.input_gold_record_count,
            "input_split_row_count": report.input_split_row_count,
            "status_counts": report.status_counts,
            "rows_by_split": report.rows_by_split,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _align_split_row(
    row: OledCuratedSplitDatasetViewRowArtifact,
    feature_rows: list[OledFeatureMaterializationRow],
    *,
    feature_view: str,
    policy: OledCuratedSplitFeaturePreflightPolicy,
) -> OledSplitFeatureRowAlignment:
    record_candidates = [feature_row for feature_row in feature_rows if feature_row.record_id == row.record_id]
    if not record_candidates:
        return _alignment(
            row,
            feature_view=feature_view,
            status=OledSplitFeatureRowAlignmentStatus.MISSING_FEATURE_ROW,
            reason_codes=["missing_feature_row"],
        )

    if row.condition_hash:
        candidates = [feature_row for feature_row in record_candidates if feature_row.condition_hash == row.condition_hash]
    else:
        candidates = record_candidates

    if not candidates:
        return _alignment(
            row,
            feature_view=feature_view,
            status=OledSplitFeatureRowAlignmentStatus.MISSING_FEATURE_ROW,
            reason_codes=["missing_feature_row", "condition_hash_unmatched"],
        )
    if len(candidates) > 1:
        return _alignment(
            row,
            feature_view=feature_view,
            status=OledSplitFeatureRowAlignmentStatus.AMBIGUOUS_FEATURE_ROW,
            reason_codes=["ambiguous_feature_row"],
        )

    feature_row = candidates[0]
    missing_columns = _missing_feature_columns(feature_row)
    present_columns = sorted(column for column in feature_row.features if column not in missing_columns)
    status = OledSplitFeatureRowAlignmentStatus.MATCHED
    reason_codes = ["matched"]
    if not _values_equal(row.target_value, feature_row.target_value) or _clean_unit(row.target_unit) != _clean_unit(feature_row.target_unit):
        status = OledSplitFeatureRowAlignmentStatus.TARGET_MISMATCH
        reason_codes = ["target_mismatch"]
    elif missing_columns:
        reason_codes.append("missing_feature_values")
    return _alignment(
        row,
        feature_view=feature_view,
        status=status,
        feature_row=feature_row,
        feature_column_count=len(feature_row.features),
        missing_feature_columns=missing_columns,
        present_feature_columns=present_columns,
        reason_codes=reason_codes,
    )


def _alignment(
    row: OledCuratedSplitDatasetViewRowArtifact,
    *,
    feature_view: str,
    status: OledSplitFeatureRowAlignmentStatus,
    feature_row: OledFeatureMaterializationRow | None = None,
    feature_column_count: int = 0,
    missing_feature_columns: list[str] | None = None,
    present_feature_columns: list[str] | None = None,
    reason_codes: list[str],
) -> OledSplitFeatureRowAlignment:
    return OledSplitFeatureRowAlignment(
        split_row_id=row.split_row_id,
        row_id=row.row_id,
        split=row.split,
        view_kind=row.view_kind,
        target_property_id=row.target_property_id,
        record_id=row.record_id,
        feature_view=feature_view,
        status=status,
        feature_row_record_id=feature_row.record_id if feature_row is not None else None,
        feature_row_condition_hash=feature_row.condition_hash if feature_row is not None else None,
        target_value=row.target_value,
        feature_target_value=feature_row.target_value if feature_row is not None else None,
        target_unit=row.target_unit,
        feature_target_unit=feature_row.target_unit if feature_row is not None else None,
        feature_column_count=feature_column_count,
        missing_feature_columns=missing_feature_columns or [],
        present_feature_columns=present_feature_columns or [],
        evidence_refs=row.evidence_refs,
        reason_codes=reason_codes,
        metadata={
            "condition_hash": row.condition_hash,
            "dedup_key_hash": row.dedup_key_hash,
            "feature_preflight_only": True,
        },
    )


def _findings_for_alignment(
    alignment: OledSplitFeatureRowAlignment,
    policy: OledCuratedSplitFeaturePreflightPolicy,
) -> list[OledCuratedSplitFeaturePreflightFinding]:
    findings: list[OledCuratedSplitFeaturePreflightFinding] = []
    if alignment.status == OledSplitFeatureRowAlignmentStatus.MISSING_FEATURE_ROW:
        findings.append(
            _alignment_finding(
                alignment,
                code="missing_feature_row",
                severity="error" if policy.require_all_split_rows_matched else "warning",
                message="split dataset-view row has no matching feature materialization row",
            )
        )
    if alignment.status == OledSplitFeatureRowAlignmentStatus.AMBIGUOUS_FEATURE_ROW:
        findings.append(
            _alignment_finding(
                alignment,
                code="ambiguous_feature_row",
                severity="error" if policy.require_all_split_rows_matched else "warning",
                message="split dataset-view row maps to multiple feature materialization rows",
            )
        )
    if alignment.status == OledSplitFeatureRowAlignmentStatus.TARGET_MISMATCH:
        findings.append(
            _alignment_finding(
                alignment,
                code="target_mismatch",
                severity="error" if policy.fail_on_target_mismatch else "warning",
                message="split dataset-view row target differs from feature materialization target",
            )
        )
    if alignment.missing_feature_columns:
        findings.append(
            _alignment_finding(
                alignment,
                code="missing_feature_values",
                severity="error" if policy.fail_on_missing_features else "warning",
                message="feature materialization row contains missing feature values",
            )
        )
    return findings


def _alignment_finding(
    alignment: OledSplitFeatureRowAlignment,
    *,
    code: str,
    severity: Literal["error", "warning"],
    message: str,
) -> OledCuratedSplitFeaturePreflightFinding:
    return OledCuratedSplitFeaturePreflightFinding(
        code=code,
        severity=severity,
        message=message,
        split=alignment.split,
        target_property_id=alignment.target_property_id,
        feature_view=alignment.feature_view,
        split_row_id=alignment.split_row_id,
        record_id=alignment.record_id,
    )


def _refresh_report(
    report: OledCuratedSplitFeaturePreflightReport,
    rows: list[OledCuratedSplitDatasetViewRowArtifact],
) -> OledCuratedSplitFeaturePreflightReport:
    status_counts = Counter(alignment.status.value for alignment in report.row_alignments)
    finding_code_counts = Counter(finding.code for finding in report.findings)
    rows_by_split = Counter(row.split for row in rows)
    has_errors = any(finding.severity == "error" for finding in report.findings)
    has_warnings = any(finding.severity == "warning" for finding in report.findings)
    if has_errors:
        status = OledCuratedSplitFeaturePreflightStatus.FAILED
    elif has_warnings or report.gold_validation_warning_codes:
        status = OledCuratedSplitFeaturePreflightStatus.PASSED_WITH_WARNINGS
    else:
        status = OledCuratedSplitFeaturePreflightStatus.PASSED
    findings = _dedup_findings(report.findings)
    return report.model_copy(
        update={
            "status": status,
            "findings": findings,
            "summaries": _summaries(report.row_alignments),
            "status_counts": dict(sorted(status_counts.items())),
            "finding_code_counts": dict(sorted(finding_code_counts.items())),
            "rows_by_split": dict(sorted(rows_by_split.items())),
        }
    )


def _summaries(
    alignments: list[OledSplitFeatureRowAlignment],
) -> list[OledSplitFeaturePreflightSummary]:
    grouped: dict[tuple[str, str, str], list[OledSplitFeatureRowAlignment]] = defaultdict(list)
    for alignment in alignments:
        grouped[(alignment.split, alignment.target_property_id, alignment.feature_view)].append(alignment)
    summaries: list[OledSplitFeaturePreflightSummary] = []
    for (split, target_property_id, feature_view), group in sorted(grouped.items()):
        status_counts = Counter(alignment.status.value for alignment in group)
        reason_counts = Counter(code for alignment in group for code in alignment.reason_codes)
        missing_counts = Counter(column for alignment in group for column in alignment.missing_feature_columns)
        summaries.append(
            OledSplitFeaturePreflightSummary(
                split=split,
                target_property_id=target_property_id,
                feature_view=feature_view,
                input_split_row_count=len(group),
                matched_row_count=status_counts.get(OledSplitFeatureRowAlignmentStatus.MATCHED.value, 0),
                missing_feature_row_count=status_counts.get(OledSplitFeatureRowAlignmentStatus.MISSING_FEATURE_ROW.value, 0),
                ambiguous_feature_row_count=status_counts.get(OledSplitFeatureRowAlignmentStatus.AMBIGUOUS_FEATURE_ROW.value, 0),
                target_mismatch_count=status_counts.get(OledSplitFeatureRowAlignmentStatus.TARGET_MISMATCH.value, 0),
                feature_column_count=max((alignment.feature_column_count for alignment in group), default=0),
                missing_feature_column_counts=dict(sorted(missing_counts.items())),
                alignment_status_counts=dict(sorted(status_counts.items())),
                reason_code_counts=dict(sorted(reason_counts.items())),
            )
        )
    return summaries


def _feature_views(policy: OledCuratedSplitFeaturePreflightPolicy) -> list[OledBaselineFeatureView]:
    if not policy.feature_views:
        return list(OledBaselineFeatureView)
    return [OledBaselineFeatureView(str(item)) for item in policy.feature_views]


def _target_property_ids(policy: OledCuratedSplitFeaturePreflightPolicy) -> list[str]:
    return sorted({str(item).strip() for item in policy.target_property_ids if str(item).strip()})


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


def _values_equal(left: float | int | str | None, right: float | int | str | None) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return str(left) == str(right)


def _clean_unit(value: str | None) -> str | None:
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _split_cli_values(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip() for part in str(value).split(",") if part.strip())
    return output


def _sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"missing_split_dataset_view_rows_jsonl:{redact_oled_mineru_acceptance_path(file_path)}")
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _dedup_findings(
    findings: list[OledCuratedSplitFeaturePreflightFinding],
) -> list[OledCuratedSplitFeaturePreflightFinding]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[OledCuratedSplitFeaturePreflightFinding] = []
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


def _safety_metadata() -> dict[str, Any]:
    return {
        "feature_preflight_only": True,
        "feature_tables_written": False,
        "training_data_written": False,
        "ml_ready_training_data_written": False,
        "model_backends_run": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


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
    if normalized == "input_gold_record_count":
        return False
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
    "OledCuratedSplitFeaturePreflightStatus",
    "OledSplitFeatureRowAlignmentStatus",
    "OledCuratedSplitFeaturePreflightPolicy",
    "OledSplitFeatureRowAlignment",
    "OledSplitFeaturePreflightSummary",
    "OledCuratedSplitFeaturePreflightFinding",
    "OledCuratedSplitFeaturePreflightReport",
    "load_oled_curated_split_dataset_view_writer_manifest_json",
    "load_oled_curated_split_dataset_view_rows_from_manifest",
    "run_oled_curated_split_feature_preflight",
    "run_oled_curated_split_feature_preflight_from_files",
    "write_oled_curated_split_feature_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
