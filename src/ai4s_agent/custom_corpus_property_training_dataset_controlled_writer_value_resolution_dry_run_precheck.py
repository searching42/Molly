from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_materialization import sha256_file


_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck.v1"
_REPORT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_summary.v1"
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_ABSOLUTE_PATH_VALUE_RE = re.compile(r'"(?:/|[A-Za-z]:\\\\)')

_FORBIDDEN_MARKERS = (
    "/Users/",
    "/home/",
    "C:\\",
    "Authorization",
    "Bearer",
    "token",
    "secret",
    "password",
    "cookie",
    "x-api-key",
    ".pdf",
    ".csv",
    ".jsonl",
    ".parquet",
    ".lmdb",
    "x-amz-signature",
    "signature=",
    "signedurl",
    "signed-url",
    "raw article text",
    "raw table",
    "serialized training row",
    "serialized dataset row",
    "conformer block",
    "dpa3 structure block",
    "inchi=",
    "0.72",
    "c1=cc",
)
_COMMON_FIELDS = (
    "value_resolution_dry_run_id",
    "controlled_writer_execution_plan_id",
    "value_source_manifest_id",
    "writer_input_binding_plan_id",
    "writer_execution_request_id",
    "row_contract_id",
    "materialization_plan_id",
    "execution_ledger_id",
    "corpus_id",
    "dataset_name",
)
_SAFE_RECORD_FIELDS = {
    "value_resolution_record_id",
    "writer_request_record_id",
    "writer_input_binding_record_id",
    "row_preview_id",
    "planned_dataset_record_id",
    "candidate_record_id",
    "record_id",
    "document_id",
    "field_name",
    "required_field_resolution_status",
    "optional_field_resolution_status",
    "resolved_required_field_names",
    "missing_required_field_names",
    "resolved_optional_field_names",
    "missing_optional_field_names",
    "value_source_record_ids",
    "source_artifact_labels",
    "source_artifact_sha256s",
    "derivation_rule_labels",
    "controlled_writer_executed",
    "writer_executed",
    "source_payloads_read",
    "values_materialized",
    "serialized_rows_created",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "phase1_status",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
}
_COUNT_FIELDS = (
    "resolution_record_count",
    "resolved_resolution_record_count",
    "binding_record_count",
    "writer_request_record_count",
    "value_source_record_count",
    "missing_required_field_count",
    "missing_optional_field_count",
)


class PropertyTrainingDatasetControlledWriterValueResolutionDryRunPrecheckError(ValueError):
    pass


def precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(
    *,
    controlled_writer_value_resolution_dry_run_report_path: str | Path,
    controlled_writer_value_resolution_dry_run_summary_path: str | Path,
    output_summary_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    require_dry_run_passed: bool = True,
    allow_dry_run_needs_review: bool = False,
    require_values_resolved: bool = True,
    minimum_resolution_records: int = 1,
) -> dict[str, Any]:
    report_path = Path(controlled_writer_value_resolution_dry_run_report_path)
    summary_path = Path(controlled_writer_value_resolution_dry_run_summary_path)
    report = _load_json(report_path)
    dry_run_summary = _load_json(summary_path)
    report_sha = sha256_file(report_path)
    dry_run_summary_sha = sha256_file(summary_path)
    errors: list[str] = []
    warnings: list[str] = []

    _append_errors(errors, _schema_errors(report, dry_run_summary))
    _append_errors(
        errors,
        _status_errors(
            report,
            dry_run_summary,
            require_dry_run_passed=require_dry_run_passed,
            allow_dry_run_needs_review=allow_dry_run_needs_review,
            warnings=warnings,
        ),
    )
    _append_errors(errors, _hash_errors(report, dry_run_summary, report_sha))
    _append_errors(errors, _id_errors(report, dry_run_summary))
    _append_errors(
        errors,
        _count_errors(report, dry_run_summary, minimum_resolution_records=minimum_resolution_records),
    )
    resolution_errors, resolution_warnings = _resolution_errors(
        report,
        require_values_resolved=require_values_resolved,
    )
    if allow_dry_run_needs_review and report.get("dry_run_status") == "needs_review":
        downgraded = {
            "values_not_resolved",
            "missing_required_fields",
            "resolution_record_required_fields_not_resolved",
            "resolution_record_missing_required_fields",
        }
        for warning_code in sorted(downgraded & set(resolution_errors)):
            warnings.append(warning_code)
        resolution_errors = [error for error in resolution_errors if error not in downgraded]
    errors.extend(resolution_errors)
    warnings.extend(resolution_warnings)
    _append_errors(errors, _boundary_errors(report, dry_run_summary))
    _append_errors(errors, _sha_format_errors({"report": report, "summary": dry_run_summary}))
    if _contains_forbidden_material({"report": report, "summary": dry_run_summary}):
        errors.append("controlled_writer_value_resolution_dry_run_package_contains_unsafe_value")

    status = "blocked" if errors else "needs_review" if warnings else "passed"
    summary = _summary(
        status=status,
        report_path=report_path,
        report_sha=report_sha,
        dry_run_summary_path=summary_path,
        dry_run_summary_sha=dry_run_summary_sha,
        report=report,
        dry_run_summary=dry_run_summary,
        errors=_stable_unique(errors),
        warnings=_stable_unique(warnings),
    )
    markdown = _markdown(summary)
    if _contains_forbidden_material({"summary": summary, "markdown": markdown}):
        summary = _minimal_redaction_failure()
        markdown = ""

    if output_summary_path is not None:
        write_json(Path(output_summary_path), summary)
    if output_markdown_path is not None and summary.get("redaction_status") != "failed":
        Path(output_markdown_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_markdown_path).write_text(markdown, encoding="utf-8")
    return summary


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = _parser()
    if stderr is None:
        args = parser.parse_args(argv)
    else:
        with redirect_stderr(stderr):
            args = parser.parse_args(argv)
    try:
        summary = precheck_property_training_dataset_controlled_writer_value_resolution_dry_run(
            controlled_writer_value_resolution_dry_run_report_path=args.controlled_writer_value_resolution_dry_run_report,
            controlled_writer_value_resolution_dry_run_summary_path=args.controlled_writer_value_resolution_dry_run_summary,
            output_summary_path=args.output_summary,
            output_markdown_path=args.output_markdown,
            require_dry_run_passed=args.require_dry_run_passed,
            allow_dry_run_needs_review=args.allow_dry_run_needs_review,
            require_values_resolved=args.require_values_resolved,
            minimum_resolution_records=args.minimum_resolution_records,
        )
    except Exception as exc:
        err.write(
            "property training dataset controlled writer value resolution dry-run precheck invalid: "
            f"{_safe_exception_message(exc)}\n"
        )
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("precheck_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_precheck"
        ),
        description="Precheck controlled writer value resolution dry-run outputs without executing a writer.",
    )
    parser.add_argument("--controlled-writer-value-resolution-dry-run-report", required=True)
    parser.add_argument("--controlled-writer-value-resolution-dry-run-summary", required=True)
    parser.add_argument("--output-summary")
    parser.add_argument("--output-markdown")
    parser.add_argument("--require-dry-run-passed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-dry-run-needs-review", action="store_true")
    parser.add_argument("--require-values-resolved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimum-resolution-records", type=int, default=1)
    return parser


def _schema_errors(report: dict[str, Any], dry_run_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != _REPORT_SCHEMA_VERSION:
        errors.append("controlled_writer_value_resolution_dry_run_report_schema_invalid")
    if dry_run_summary.get("schema_version") != _SUMMARY_SCHEMA_VERSION:
        errors.append("controlled_writer_value_resolution_dry_run_summary_schema_invalid")
    return errors


def _status_errors(
    report: dict[str, Any],
    dry_run_summary: dict[str, Any],
    *,
    require_dry_run_passed: bool,
    allow_dry_run_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    report_status = str(report.get("dry_run_status", ""))
    summary_status = str(dry_run_summary.get("dry_run_status", ""))
    errors: list[str] = []
    if report_status != summary_status:
        errors.append("dry_run_status_mismatch")
    status = report_status or summary_status
    if status == "passed":
        return errors
    if status == "needs_review":
        if allow_dry_run_needs_review or not require_dry_run_passed:
            warnings.append("dry_run_needs_review")
            return errors
        return [*errors, "dry_run_needs_review"]
    if status in {"blocked", "failed"}:
        return [*errors, "dry_run_blocked"]
    return [*errors, "dry_run_status_invalid"]


def _hash_errors(report: dict[str, Any], dry_run_summary: dict[str, Any], report_sha: str) -> list[str]:
    errors: list[str] = []
    if dry_run_summary.get("controlled_writer_value_resolution_dry_run_report_sha256") != report_sha:
        errors.append("controlled_writer_value_resolution_dry_run_report_sha256_mismatch")
    for key, report_value in report.items():
        if not key.endswith("_sha256") or key not in dry_run_summary:
            continue
        if dry_run_summary.get(key) != report_value:
            errors.append(f"{key}_mismatch")
    return _stable_unique(errors)


def _id_errors(report: dict[str, Any], dry_run_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _COMMON_FIELDS:
        report_value = report.get(field)
        summary_value = dry_run_summary.get(field)
        if report_value and summary_value and report_value != summary_value:
            errors.append(f"{field}_mismatch")
        for value in (report_value, summary_value):
            if value and field != "dataset_name" and not _is_safe_id(str(value)):
                errors.append(f"{field}_invalid")
            if value and field == "dataset_name" and not _is_safe_id(str(value)):
                errors.append("dataset_name_invalid")
    return _stable_unique(errors)


def _count_errors(
    report: dict[str, Any],
    dry_run_summary: dict[str, Any],
    *,
    minimum_resolution_records: int,
) -> list[str]:
    errors: list[str] = []
    records = _safe_records(report.get("resolution_records"))
    report_count = report.get("resolution_record_count")
    summary_count = dry_run_summary.get("resolution_record_count")
    if report_count != len(records):
        errors.append("resolution_record_count_mismatch")
    if summary_count != report_count:
        errors.append("resolution_record_count_mismatch")
    if len(records) < max(minimum_resolution_records, 1):
        errors.append("minimum_resolution_records_not_met")
    for field in _COUNT_FIELDS:
        for container in (report, dry_run_summary):
            if field not in container:
                continue
            value = container.get(field)
            if not isinstance(value, int) or value < 0:
                errors.append(f"{field}_invalid")
    resolved_count = report.get("resolved_resolution_record_count")
    if isinstance(resolved_count, int) and isinstance(report_count, int) and resolved_count > report_count:
        errors.append("resolved_resolution_record_count_invalid")
    return _stable_unique(errors)


def _resolution_errors(
    report: dict[str, Any],
    *,
    require_values_resolved: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    records = _safe_records(report.get("resolution_records"))
    if require_values_resolved:
        if report.get("values_resolved") is not True:
            errors.append("values_not_resolved")
        if report.get("missing_required_field_count") != 0:
            errors.append("missing_required_fields")
    else:
        if report.get("values_resolved") is not True or report.get("missing_required_field_count", 0) != 0:
            warnings.append("values_not_resolved")
    for record in records:
        record_errors = _resolution_record_errors(record, require_values_resolved=require_values_resolved)
        if require_values_resolved:
            errors.extend(record_errors)
        else:
            for error in record_errors:
                if error in {"resolution_record_required_fields_not_resolved", "resolution_record_missing_required_fields"}:
                    warnings.append("values_not_resolved")
                else:
                    errors.append(error)
    return _stable_unique(errors), _stable_unique(warnings)


def _resolution_record_errors(record: dict[str, Any], *, require_values_resolved: bool) -> list[str]:
    errors: list[str] = []
    if set(record) - _SAFE_RECORD_FIELDS:
        errors.append("resolution_record_field_not_allowed")
    if _contains_forbidden_material(record):
        errors.append("resolution_record_contains_unsafe_value")
    if require_values_resolved:
        if record.get("required_field_resolution_status") != "resolved":
            errors.append("resolution_record_required_fields_not_resolved")
        if record.get("missing_required_field_names"):
            errors.append("resolution_record_missing_required_fields")
    if record.get("controlled_writer_executed") is True or record.get("writer_executed") is True:
        errors.append("controlled_writer_executed")
    if record.get("source_payloads_read") is not True:
        errors.append("source_payloads_not_read")
    for field, error_code in (
        ("values_materialized", "values_materialized"),
        ("serialized_rows_created", "serialized_rows_created"),
        ("training_dataset_materialized", "training_dataset_materialized"),
        ("dataset_artifact_created", "dataset_artifact_created"),
        ("dataset_confirmation_changed", "dataset_confirmation_changed"),
        ("model_training_run", "model_training_run"),
        ("evaluation_run", "evaluation_run"),
    ):
        if record.get(field) is True:
            errors.append(error_code)
    if record.get("phase1_status") != "not_run":
        errors.append("phase1_ran")
    for field in (
        "resolved_required_field_names",
        "missing_required_field_names",
        "resolved_optional_field_names",
        "missing_optional_field_names",
        "value_source_record_ids",
        "source_artifact_labels",
        "derivation_rule_labels",
    ):
        if any(not _is_safe_id(str(item)) for item in _safe_list(record.get(field))):
            errors.append(f"{field}_invalid")
    for value in _safe_list(record.get("source_artifact_sha256s")):
        if not _is_sha(value):
            errors.append("source_artifact_sha256_invalid")
    return _stable_unique(errors)


def _boundary_errors(report: dict[str, Any], dry_run_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for container in (report, dry_run_summary):
        if container.get("controlled_writer_executed") is True:
            errors.append("controlled_writer_executed")
        if container.get("writer_executed") is True:
            errors.append("writer_executed")
        if container.get("source_payloads_read") is not True:
            errors.append("source_payloads_not_read")
        if container.get("values_materialized") is True:
            errors.append("values_materialized")
        if container.get("serialized_rows_created") is True:
            errors.append("serialized_rows_created")
        if container.get("training_dataset_materialized") is True:
            errors.append("training_dataset_materialized")
        if container.get("dataset_artifact_created") is True:
            errors.append("dataset_artifact_created")
        if container.get("phase1_status") != "not_run":
            errors.append("phase1_ran")
        if container.get("dataset_confirmation_changed") is True:
            errors.append("dataset_confirmation_changed")
        if container.get("model_training_run") is True:
            errors.append("model_training_run")
        if container.get("evaluation_run") is True:
            errors.append("evaluation_run")
    return _stable_unique(errors)


def _sha_format_errors(value: Any) -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.endswith("_sha256") and nested and not _is_sha(str(nested)):
                errors.append("sha256_field_invalid")
            errors.extend(_sha_format_errors(nested))
    elif isinstance(value, list):
        for item in value:
            errors.extend(_sha_format_errors(item))
    return _stable_unique(errors)


def _summary(
    *,
    status: str,
    report_path: Path,
    report_sha: str,
    dry_run_summary_path: Path,
    dry_run_summary_sha: str,
    report: dict[str, Any],
    dry_run_summary: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": status,
        "controlled_writer_value_resolution_dry_run_report_path": report_path.name,
        "controlled_writer_value_resolution_dry_run_report_sha256": report_sha,
        "controlled_writer_value_resolution_dry_run_summary_path": dry_run_summary_path.name,
        "controlled_writer_value_resolution_dry_run_summary_sha256": dry_run_summary_sha,
        "value_resolution_dry_run_id": report.get("value_resolution_dry_run_id", ""),
        "controlled_writer_execution_plan_id": report.get("controlled_writer_execution_plan_id", ""),
        "value_source_manifest_id": report.get("value_source_manifest_id", ""),
        "writer_input_binding_plan_id": report.get("writer_input_binding_plan_id", ""),
        "writer_execution_request_id": report.get("writer_execution_request_id", ""),
        "row_contract_id": report.get("row_contract_id", ""),
        "materialization_plan_id": report.get("materialization_plan_id", ""),
        "execution_ledger_id": report.get("execution_ledger_id", ""),
        "corpus_id": report.get("corpus_id", ""),
        "dataset_name": report.get("dataset_name", ""),
        "resolution_record_count": _safe_int(report.get("resolution_record_count")),
        "resolved_resolution_record_count": _safe_int(report.get("resolved_resolution_record_count")),
        "binding_record_count": _safe_int(report.get("binding_record_count")),
        "writer_request_record_count": _safe_int(report.get("writer_request_record_count")),
        "value_source_record_count": _safe_int(report.get("value_source_record_count")),
        "missing_required_field_count": _safe_int(report.get("missing_required_field_count")),
        "missing_optional_field_count": _safe_int(report.get("missing_optional_field_count")),
        "controlled_writer_executed": bool(report.get("controlled_writer_executed", False)),
        "source_payloads_read": bool(report.get("source_payloads_read", False)),
        "values_resolved": bool(report.get("values_resolved", False)),
        "values_materialized": bool(report.get("values_materialized", False)),
        "serialized_rows_created": bool(report.get("serialized_rows_created", False)),
        "training_dataset_materialized": bool(report.get("training_dataset_materialized", False)),
        "dataset_artifact_created": bool(report.get("dataset_artifact_created", False)),
        "phase1_status": report.get("phase1_status", ""),
        "dataset_confirmation_changed": bool(report.get("dataset_confirmation_changed", False)),
        "model_training_run": bool(report.get("model_training_run", False)),
        "evaluation_run": bool(report.get("evaluation_run", False)),
        "precheck_errors": errors,
        "warnings": warnings,
        "redaction_status": "passed",
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Property Training Dataset Controlled Writer Value Resolution Dry-Run Precheck",
        "",
        f"- Precheck status: {summary.get('precheck_status', '')}",
        f"- Value resolution dry-run id: {summary.get('value_resolution_dry_run_id', '')}",
        f"- Report: {summary.get('controlled_writer_value_resolution_dry_run_report_path', '')}",
        f"- Report SHA: {summary.get('controlled_writer_value_resolution_dry_run_report_sha256', '')}",
        f"- Summary: {summary.get('controlled_writer_value_resolution_dry_run_summary_path', '')}",
        f"- Summary SHA: {summary.get('controlled_writer_value_resolution_dry_run_summary_sha256', '')}",
        f"- Resolution records: {summary.get('resolution_record_count', 0)}",
        f"- Missing required fields: {summary.get('missing_required_field_count', 0)}",
        f"- Missing optional fields: {summary.get('missing_optional_field_count', 0)}",
        f"- Errors: {', '.join(summary.get('precheck_errors', [])) or 'none'}",
        f"- Warnings: {', '.join(summary.get('warnings', [])) or 'none'}",
        "",
        "## Boundary Statement",
        "",
        "this is a value resolution dry-run precheck only.",
        "The controlled writer was not executed.",
        "authorized source payloads were not re-read by this precheck.",
        "Values were not emitted or materialized.",
        "No row serialization occurred.",
        "No training dataset artifact was created.",
        "No training or candidate CSV/JSONL/Parquet/LMDB artifact was created.",
        "No conformers or DPA3 structures were generated.",
        "Phase 1 did not run and DatasetConfirmation was not changed.",
        "No model training or evaluation was run.",
        "No LLM, MinerU, PDF parser, or corpus workflow was called.",
    ]
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetControlledWriterValueResolutionDryRunPrecheckError("json_object_required")
    return payload


def _safe_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def _safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _safe_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _append_errors(errors: list[str], new_errors: list[str]) -> None:
    errors.extend(new_errors)


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _is_safe_id(value: str) -> bool:
    return bool(value) and bool(_SAFE_ID_RE.match(value))


def _is_sha(value: str) -> bool:
    return bool(_SHA_RE.match(value))


def _contains_forbidden_material(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(text))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": "blocked",
        "precheck_errors": [
            "property_training_dataset_controlled_writer_value_resolution_dry_run_precheck_redaction_failed"
        ],
        "redaction_status": "failed",
    }


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc)
    if _contains_forbidden_material(message):
        return "redacted"
    return message or exc.__class__.__name__


if __name__ == "__main__":
    raise SystemExit(main())
