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

from ai4s_agent.domains.oled_curated_gold_writer import OledCuratedGoldManifest
from ai4s_agent.domains.oled_dataset_views import (
    OledDatasetViewKind,
    OledDatasetViewReport,
    build_oled_dataset_view,
)
from ai4s_agent.domains.oled_gold_validation import (
    OledGoldDatasetRecord,
    validate_oled_gold_dataset,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path


class OledCuratedGoldViewPreflightStatus(str, Enum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class OledCuratedGoldManifestIntegrityStatus(str, Enum):
    NOT_PROVIDED = "not_provided"
    MATCHED = "matched"
    MISMATCHED = "mismatched"
    MISSING_SHA256 = "missing_sha256"
    MISSING_OUTPUT_PATH = "missing_output_path"


class OledCuratedGoldViewPreflightPolicy(BaseModel):
    require_manifest_if_provided: bool = True
    require_sha256_match: bool = True
    require_gold_validation_success: bool = True
    fail_on_view_errors: bool = True
    include_empty_views: bool = True
    target_property_ids: list[str] = Field(default_factory=lambda: ["eqe_percent", "plqy", "delta_e_st_ev"])
    view_kinds: list[str] = Field(default_factory=list)


class OledCuratedGoldDatasetViewPreflightResult(BaseModel):
    view_kind: str
    target_property_id: str
    status: OledCuratedGoldViewPreflightStatus

    input_record_count: int
    row_count: int = 0

    property_ids: list[str] = Field(default_factory=list)
    layer_counts: dict[str, int] = Field(default_factory=dict)
    evidence_anchor_count: int = 0

    view_error_codes: list[str] = Field(default_factory=list)
    view_warning_codes: list[str] = Field(default_factory=list)

    reason_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OledCuratedGoldViewPreflightFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    view_kind: str | None = None
    target_property_id: str | None = None
    record_id: str | None = None


class OledCuratedGoldViewPreflightReport(BaseModel):
    status: OledCuratedGoldViewPreflightStatus

    input_record_count: int
    manifest_integrity_status: OledCuratedGoldManifestIntegrityStatus
    input_sha256: str | None = None
    manifest_sha256: str | None = None

    gold_validation_error_codes: list[str] = Field(default_factory=list)
    gold_validation_warning_codes: list[str] = Field(default_factory=list)

    view_results: list[OledCuratedGoldDatasetViewPreflightResult] = Field(default_factory=list)
    findings: list[OledCuratedGoldViewPreflightFinding] = Field(default_factory=list)

    status_counts: dict[str, int] = Field(default_factory=dict)
    finding_code_counts: dict[str, int] = Field(default_factory=dict)
    view_row_counts: dict[str, int] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status != OledCuratedGoldViewPreflightStatus.FAILED and not self.error_codes

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_curated_gold_records_jsonl(
    path: str | Path,
) -> list[OledGoldDatasetRecord]:
    jsonl_path = Path(path)
    _reject_forbidden_input(jsonl_path)
    if not jsonl_path.exists():
        raise ValueError(f"missing_curated_gold_jsonl:{redact_oled_mineru_acceptance_path(jsonl_path)}")
    records: list[OledGoldDatasetRecord] = []
    for line_number, raw_line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            record = OledGoldDatasetRecord.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid_curated_gold_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(record.metadata):
            raise ValueError(f"absolute_path_in_curated_gold_metadata:{record.record_id}")
        records.append(record)
    return records


def load_oled_curated_gold_manifest_json(
    path: str | Path,
) -> OledCuratedGoldManifest:
    manifest_path = Path(path)
    _reject_forbidden_input(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"missing_curated_gold_manifest:{redact_oled_mineru_acceptance_path(manifest_path)}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return OledCuratedGoldManifest.model_validate(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_curated_gold_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc
    except ValidationError as exc:
        raise ValueError(f"invalid_curated_gold_manifest_json:{redact_oled_mineru_acceptance_path(manifest_path)}") from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_oled_curated_gold_manifest_integrity(
    *,
    input_jsonl_path: str | Path,
    manifest: OledCuratedGoldManifest | None = None,
) -> tuple[OledCuratedGoldManifestIntegrityStatus, list[OledCuratedGoldViewPreflightFinding], str | None]:
    if manifest is None:
        return OledCuratedGoldManifestIntegrityStatus.NOT_PROVIDED, [], None
    if not manifest.output_sha256:
        return (
            OledCuratedGoldManifestIntegrityStatus.MISSING_SHA256,
            [
                OledCuratedGoldViewPreflightFinding(
                    code="manifest_missing_sha256",
                    severity="warning",
                    message="curated gold manifest has no output_sha256",
                )
            ],
            None,
        )
    input_digest = sha256_file(input_jsonl_path)
    if manifest.output_sha256 != input_digest:
        return (
            OledCuratedGoldManifestIntegrityStatus.MISMATCHED,
            [
                OledCuratedGoldViewPreflightFinding(
                    code="manifest_sha256_mismatch",
                    severity="error",
                    message="curated gold JSONL SHA256 does not match manifest output_sha256",
                )
            ],
            input_digest,
        )
    if not manifest.output_jsonl_path:
        return (
            OledCuratedGoldManifestIntegrityStatus.MISSING_OUTPUT_PATH,
            [
                OledCuratedGoldViewPreflightFinding(
                    code="manifest_missing_output_path",
                    severity="warning",
                    message="curated gold manifest has no output_jsonl_path",
                )
            ],
            input_digest,
        )
    return OledCuratedGoldManifestIntegrityStatus.MATCHED, [], input_digest


def run_oled_curated_gold_view_preflight(
    records: Iterable[OledGoldDatasetRecord],
    *,
    manifest: OledCuratedGoldManifest | None = None,
    input_sha256: str | None = None,
    manifest_integrity_status: OledCuratedGoldManifestIntegrityStatus = OledCuratedGoldManifestIntegrityStatus.NOT_PROVIDED,
    policy: OledCuratedGoldViewPreflightPolicy | None = None,
) -> OledCuratedGoldViewPreflightReport:
    preflight_policy = policy or OledCuratedGoldViewPreflightPolicy()
    gold_records = sorted(list(records), key=lambda item: item.record_id)
    findings = _manifest_integrity_findings(manifest_integrity_status, preflight_policy)
    gold_report = validate_oled_gold_dataset(gold_records)
    findings.extend(_gold_validation_findings(gold_report.error_codes, gold_report.warning_codes, preflight_policy))

    if gold_report.error_codes and preflight_policy.require_gold_validation_success:
        return _report(
            records=gold_records,
            manifest=manifest,
            input_sha256=input_sha256,
            manifest_integrity_status=manifest_integrity_status,
            gold_validation_error_codes=gold_report.error_codes,
            gold_validation_warning_codes=gold_report.warning_codes,
            view_results=[],
            findings=findings,
        )

    view_results: list[OledCuratedGoldDatasetViewPreflightResult] = []
    for view_kind in _view_kinds(preflight_policy):
        for target_property_id in _target_property_ids(preflight_policy):
            result, result_findings = _build_view_result(
                gold_records,
                view_kind=view_kind,
                target_property_id=target_property_id,
                policy=preflight_policy,
            )
            view_results.append(result)
            findings.extend(result_findings)

    return _report(
        records=gold_records,
        manifest=manifest,
        input_sha256=input_sha256,
        manifest_integrity_status=manifest_integrity_status,
        gold_validation_error_codes=gold_report.error_codes,
        gold_validation_warning_codes=gold_report.warning_codes,
        view_results=view_results,
        findings=findings,
    )


def run_oled_curated_gold_view_preflight_from_files(
    *,
    curated_gold_jsonl_path: str | Path,
    manifest_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
    policy: OledCuratedGoldViewPreflightPolicy | None = None,
) -> OledCuratedGoldViewPreflightReport:
    records = load_oled_curated_gold_records_jsonl(curated_gold_jsonl_path)
    manifest = load_oled_curated_gold_manifest_json(manifest_path) if manifest_path is not None else None
    input_digest = sha256_file(curated_gold_jsonl_path)
    integrity_status, integrity_findings, computed_digest = check_oled_curated_gold_manifest_integrity(
        input_jsonl_path=curated_gold_jsonl_path,
        manifest=manifest,
    )
    report = run_oled_curated_gold_view_preflight(
        records,
        manifest=manifest,
        input_sha256=computed_digest or input_digest,
        manifest_integrity_status=integrity_status,
        policy=policy,
    )
    if integrity_findings:
        merged_findings = _dedup_findings([*integrity_findings, *report.findings])
        report = _refresh_report_counts(report.model_copy(update={"findings": merged_findings}))
    if output_report_path is not None:
        write_oled_curated_gold_view_preflight_report_json(report, output_report_path)
    return report


def write_oled_curated_gold_view_preflight_report_json(
    report: OledCuratedGoldViewPreflightReport,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(report.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only OLED curated gold dataset-view preflight.")
    parser.add_argument("--curated-gold-jsonl", required=True, help="Path to curated gold-record JSONL.")
    parser.add_argument("--manifest", help="Optional curated gold writer manifest JSON.")
    parser.add_argument("--output-report", help="Optional path to write preflight report JSON.")
    args = parser.parse_args(argv)
    try:
        report = run_oled_curated_gold_view_preflight_from_files(
            curated_gold_jsonl_path=args.curated_gold_jsonl,
            manifest_path=args.manifest,
            output_report_path=args.output_report,
        )
        summary = {
            "status": report.status.value,
            "input_record_count": report.input_record_count,
            "manifest_integrity_status": report.manifest_integrity_status.value,
            "status_counts": report.status_counts,
            "view_row_counts": report.view_row_counts,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _build_view_result(
    records: list[OledGoldDatasetRecord],
    *,
    view_kind: OledDatasetViewKind,
    target_property_id: str,
    policy: OledCuratedGoldViewPreflightPolicy,
) -> tuple[OledCuratedGoldDatasetViewPreflightResult, list[OledCuratedGoldViewPreflightFinding]]:
    try:
        view_report = build_oled_dataset_view(records, view_kind=view_kind, target_property_id=target_property_id)
    except Exception as exc:
        message = str(exc).splitlines()[0]
        if message.startswith("no_gold_records_for_target:") and policy.include_empty_views:
            result = OledCuratedGoldDatasetViewPreflightResult(
                view_kind=view_kind.value,
                target_property_id=target_property_id,
                status=OledCuratedGoldViewPreflightStatus.PASSED_WITH_WARNINGS,
                input_record_count=len(records),
                reason_codes=["empty_view", "view_built"],
                metadata={"empty_view_reason": message, "dataset_view_rows_written": False},
            )
            return result, [
                OledCuratedGoldViewPreflightFinding(
                    code="empty_view",
                    severity="warning",
                    message=message,
                    view_kind=view_kind.value,
                    target_property_id=target_property_id,
                )
            ]
        result = OledCuratedGoldDatasetViewPreflightResult(
            view_kind=view_kind.value,
            target_property_id=target_property_id,
            status=OledCuratedGoldViewPreflightStatus.FAILED,
            input_record_count=len(records),
            reason_codes=["view_build_failed"],
            metadata={"exception": message},
        )
        return result, [
            OledCuratedGoldViewPreflightFinding(
                code="view_build_failed",
                severity="error",
                message=message,
                view_kind=view_kind.value,
                target_property_id=target_property_id,
            )
        ]
    result = _view_result_from_report(view_report, len(records), policy)
    return result, _findings_from_view_report(view_report, result)


def _view_result_from_report(
    view_report: OledDatasetViewReport,
    input_record_count: int,
    policy: OledCuratedGoldViewPreflightPolicy,
) -> OledCuratedGoldDatasetViewPreflightResult:
    row_count = view_report.row_count
    error_codes, warning_codes = _classified_view_codes(view_report, policy)
    reason_codes = ["view_built"]
    if row_count == 0:
        reason_codes.append("empty_view")

    if row_count == 0 and not policy.include_empty_views:
        status = OledCuratedGoldViewPreflightStatus.FAILED
    elif error_codes and policy.fail_on_view_errors:
        status = OledCuratedGoldViewPreflightStatus.FAILED
    elif warning_codes or row_count == 0:
        status = OledCuratedGoldViewPreflightStatus.PASSED_WITH_WARNINGS
    else:
        status = OledCuratedGoldViewPreflightStatus.PASSED

    if status == OledCuratedGoldViewPreflightStatus.FAILED:
        reason_codes.append("view_build_failed")

    return OledCuratedGoldDatasetViewPreflightResult(
        view_kind=view_report.view_kind.value,
        target_property_id=view_report.target_property_id,
        status=status,
        input_record_count=input_record_count,
        row_count=row_count,
        property_ids=sorted({row.target_property_id for row in view_report.rows}),
        layer_counts=dict(sorted(Counter(row.target_layer.value for row in view_report.rows).items())),
        evidence_anchor_count=len({anchor for row in view_report.rows for anchor in row.evidence_refs}),
        view_error_codes=error_codes,
        view_warning_codes=warning_codes,
        reason_codes=sorted(set(reason_codes)),
        metadata={
            "view_metadata": _sanitize_for_output(view_report.metadata),
            "dataset_view_rows_written": False,
        },
    )


def _classified_view_codes(
    view_report: OledDatasetViewReport,
    policy: OledCuratedGoldViewPreflightPolicy,
) -> tuple[list[str], list[str]]:
    error_codes: list[str] = []
    warning_codes: list[str] = []
    for finding in view_report.findings:
        if finding.severity == "error" and not (
            policy.include_empty_views and view_report.row_count == 0 and finding.code in _EMPTY_VIEW_CODES
        ):
            error_codes.append(finding.code)
        else:
            warning_codes.append(finding.code)
    return sorted(set(error_codes)), sorted(set(warning_codes))


def _findings_from_view_report(
    view_report: OledDatasetViewReport,
    result: OledCuratedGoldDatasetViewPreflightResult,
) -> list[OledCuratedGoldViewPreflightFinding]:
    findings: list[OledCuratedGoldViewPreflightFinding] = []
    for finding in view_report.findings:
        severity: Literal["error", "warning"] = "error" if finding.code in result.view_error_codes else "warning"
        findings.append(
            OledCuratedGoldViewPreflightFinding(
                code=finding.code,
                severity=severity,
                message=finding.message,
                view_kind=view_report.view_kind.value,
                target_property_id=view_report.target_property_id,
                record_id=",".join(finding.record_ids) if finding.record_ids else None,
            )
        )
    if result.row_count == 0:
        findings.append(
            OledCuratedGoldViewPreflightFinding(
                code="empty_view",
                severity="warning" if result.status != OledCuratedGoldViewPreflightStatus.FAILED else "error",
                message="dataset view built with zero rows",
                view_kind=view_report.view_kind.value,
                target_property_id=view_report.target_property_id,
            )
        )
    return findings


def _report(
    *,
    records: list[OledGoldDatasetRecord],
    manifest: OledCuratedGoldManifest | None,
    input_sha256: str | None,
    manifest_integrity_status: OledCuratedGoldManifestIntegrityStatus,
    gold_validation_error_codes: list[str],
    gold_validation_warning_codes: list[str],
    view_results: list[OledCuratedGoldDatasetViewPreflightResult],
    findings: list[OledCuratedGoldViewPreflightFinding],
) -> OledCuratedGoldViewPreflightReport:
    report = OledCuratedGoldViewPreflightReport(
        status=_overall_status(gold_validation_error_codes, gold_validation_warning_codes, view_results, findings),
        input_record_count=len(records),
        manifest_integrity_status=manifest_integrity_status,
        input_sha256=input_sha256,
        manifest_sha256=manifest.output_sha256 if manifest is not None else None,
        gold_validation_error_codes=sorted(set(gold_validation_error_codes)),
        gold_validation_warning_codes=sorted(set(gold_validation_warning_codes)),
        view_results=sorted(view_results, key=lambda item: (item.view_kind, item.target_property_id)),
        findings=_dedup_findings(findings),
        metadata=_safety_metadata(),
    )
    return _refresh_report_counts(report)


def _refresh_report_counts(report: OledCuratedGoldViewPreflightReport) -> OledCuratedGoldViewPreflightReport:
    refreshed = report.model_copy(
        update={
            "status": _overall_status(
                report.gold_validation_error_codes,
                report.gold_validation_warning_codes,
                report.view_results,
                report.findings,
            ),
            "status_counts": dict(sorted(Counter(result.status.value for result in report.view_results).items())),
            "finding_code_counts": dict(sorted(Counter(finding.code for finding in report.findings).items())),
            "view_row_counts": dict(
                sorted(
                    (
                        f"{result.view_kind}:{result.target_property_id}",
                        result.row_count,
                    )
                    for result in report.view_results
                )
            ),
        }
    )
    return refreshed


def _overall_status(
    gold_validation_error_codes: list[str],
    gold_validation_warning_codes: list[str],
    view_results: list[OledCuratedGoldDatasetViewPreflightResult],
    findings: list[OledCuratedGoldViewPreflightFinding],
) -> OledCuratedGoldViewPreflightStatus:
    if gold_validation_error_codes or any(finding.severity == "error" for finding in findings):
        return OledCuratedGoldViewPreflightStatus.FAILED
    if any(result.status == OledCuratedGoldViewPreflightStatus.FAILED for result in view_results):
        return OledCuratedGoldViewPreflightStatus.FAILED
    if (
        gold_validation_warning_codes
        or any(finding.severity == "warning" for finding in findings)
        or any(result.status == OledCuratedGoldViewPreflightStatus.PASSED_WITH_WARNINGS for result in view_results)
    ):
        return OledCuratedGoldViewPreflightStatus.PASSED_WITH_WARNINGS
    return OledCuratedGoldViewPreflightStatus.PASSED


def _manifest_integrity_findings(
    status: OledCuratedGoldManifestIntegrityStatus,
    policy: OledCuratedGoldViewPreflightPolicy,
) -> list[OledCuratedGoldViewPreflightFinding]:
    if status == OledCuratedGoldManifestIntegrityStatus.MISMATCHED and policy.require_sha256_match:
        return [
            OledCuratedGoldViewPreflightFinding(
                code="manifest_sha256_mismatch",
                severity="error",
                message="manifest SHA256 does not match curated gold JSONL",
            )
        ]
    if status == OledCuratedGoldManifestIntegrityStatus.MISSING_SHA256 and policy.require_manifest_if_provided:
        return [
            OledCuratedGoldViewPreflightFinding(
                code="manifest_missing_sha256",
                severity="error",
                message="manifest is missing output_sha256",
            )
        ]
    if status == OledCuratedGoldManifestIntegrityStatus.MISSING_OUTPUT_PATH and policy.require_manifest_if_provided:
        return [
            OledCuratedGoldViewPreflightFinding(
                code="manifest_missing_output_path",
                severity="warning",
                message="manifest is missing output_jsonl_path",
            )
        ]
    return []


def _gold_validation_findings(
    error_codes: list[str],
    warning_codes: list[str],
    policy: OledCuratedGoldViewPreflightPolicy,
) -> list[OledCuratedGoldViewPreflightFinding]:
    findings: list[OledCuratedGoldViewPreflightFinding] = []
    if error_codes and policy.require_gold_validation_success:
        findings.append(
            OledCuratedGoldViewPreflightFinding(
                code="gold_validation_errors_present",
                severity="error",
                message="curated gold records failed gold validation",
            )
        )
    for code in sorted(set(error_codes)):
        findings.append(
            OledCuratedGoldViewPreflightFinding(
                code=code,
                severity="error",
                message=f"gold validation error `{code}`",
            )
        )
    for code in sorted(set(warning_codes)):
        findings.append(
            OledCuratedGoldViewPreflightFinding(
                code=code,
                severity="warning",
                message=f"gold validation warning `{code}`",
            )
        )
    return findings


def _view_kinds(policy: OledCuratedGoldViewPreflightPolicy) -> list[OledDatasetViewKind]:
    if not policy.view_kinds:
        return list(OledDatasetViewKind)
    return [OledDatasetViewKind(str(item)) for item in policy.view_kinds]


def _target_property_ids(policy: OledCuratedGoldViewPreflightPolicy) -> list[str]:
    return sorted({str(item).strip() for item in policy.target_property_ids if str(item).strip()})


def _dedup_findings(
    findings: list[OledCuratedGoldViewPreflightFinding],
) -> list[OledCuratedGoldViewPreflightFinding]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[OledCuratedGoldViewPreflightFinding] = []
    for finding in findings:
        key = (
            finding.code,
            finding.severity,
            finding.view_kind or "",
            finding.target_property_id or "",
            finding.record_id or "",
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return sorted(deduped, key=lambda item: (item.severity, item.code, item.view_kind or "", item.target_property_id or ""))


def _safety_metadata() -> dict[str, Any]:
    return {
        "dataset_view_preflight_only": True,
        "dataset_view_rows_written": False,
        "training_data_written": False,
        "leakage_splits_run": False,
        "feature_materialization_run": False,
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
        )
    )


_EMPTY_VIEW_CODES = {
    "no_best_reported_rows",
    "no_intrinsic_target_rows",
}

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
    "OledCuratedGoldViewPreflightStatus",
    "OledCuratedGoldManifestIntegrityStatus",
    "OledCuratedGoldViewPreflightPolicy",
    "OledCuratedGoldDatasetViewPreflightResult",
    "OledCuratedGoldViewPreflightFinding",
    "OledCuratedGoldViewPreflightReport",
    "load_oled_curated_gold_records_jsonl",
    "load_oled_curated_gold_manifest_json",
    "sha256_file",
    "check_oled_curated_gold_manifest_integrity",
    "run_oled_curated_gold_view_preflight",
    "run_oled_curated_gold_view_preflight_from_files",
    "write_oled_curated_gold_view_preflight_report_json",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
