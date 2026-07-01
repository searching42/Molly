from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json


_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck.v1"
_REPORT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_dry_run_report.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_dry_run_summary.v1"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ABSOLUTE_PATH_VALUE_RE = re.compile(r'"(?:/|[A-Za-z]:\\\\)|/tmp/|/private/')
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
    ".pdf",
    ".csv",
    ".jsonl",
    ".parquet",
    ".lmdb",
    "raw article text",
    "raw table",
    "serialized training row",
    "serialized dataset row",
    "conformer block",
    "dpa3 structure block",
    "inchi=",
    "inchikey",
    "smiles",
    "c1=cc",
    "0.72",
)
_COMMON_FIELDS = (
    "dry_run_id",
    "dry_run_status",
    "corpus_id",
    "dataset_name",
    "accepted_candidate_record_count",
    "needs_review_candidate_record_count",
    "blocked_candidate_record_count",
    "required_field_count",
    "resolved_required_field_count",
    "missing_required_field_count",
    "would_write_row_count",
    "would_write_field_count",
    "controlled_writer_executed",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "serialized_rows_created",
    "phase1_status",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
)
_NON_NEGATIVE_COUNT_FIELDS = (
    "accepted_candidate_record_count",
    "needs_review_candidate_record_count",
    "blocked_candidate_record_count",
    "required_field_count",
    "resolved_required_field_count",
    "missing_required_field_count",
    "would_write_row_count",
    "would_write_field_count",
)
_FALSE_REPORT_FLAGS = (
    "would_create_training_dataset_artifact",
    "would_create_csv_jsonl_parquet_lmdb",
    "would_serialize_rows",
    "would_materialize_values",
    "controlled_writer_executed",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "serialized_rows_created",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
)
_FALSE_SUMMARY_FLAGS = (
    "controlled_writer_executed",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "serialized_rows_created",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
)
_SAFE_OUTPUT_KEYS = {
    "schema_version",
    "precheck_status",
    "dry_run_id",
    "dry_run_status",
    "dry_run_report_basename",
    "dry_run_report_sha256",
    "dry_run_summary_basename",
    "corpus_id",
    "dataset_name",
    "accepted_candidate_record_count",
    "needs_review_candidate_record_count",
    "blocked_candidate_record_count",
    "required_field_count",
    "resolved_required_field_count",
    "missing_required_field_count",
    "would_write_row_count",
    "would_write_field_count",
    "would_create_training_dataset_artifact",
    "would_create_csv_jsonl_parquet_lmdb",
    "would_serialize_rows",
    "would_materialize_values",
    "controlled_writer_executed",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "serialized_rows_created",
    "phase1_status",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
    "redaction_status",
    "precheck_errors",
    "precheck_warnings",
}


class PropertyTrainingDatasetControlledWriterDryRunPrecheckError(ValueError):
    pass


def precheck_property_training_dataset_controlled_writer_dry_run(
    *,
    controlled_writer_dry_run_report_path: str | Path,
    controlled_writer_dry_run_summary_path: str | Path,
    controlled_writer_dry_run_evidence_path: str | Path | None = None,
    output_summary_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    require_dry_run_passed: bool = True,
    allow_dry_run_needs_review: bool = False,
    require_zero_missing_required_fields: bool = True,
    minimum_would_write_row_count: int = 1,
) -> dict[str, Any]:
    report_path = Path(controlled_writer_dry_run_report_path)
    summary_path = Path(controlled_writer_dry_run_summary_path)
    evidence_path = Path(controlled_writer_dry_run_evidence_path) if controlled_writer_dry_run_evidence_path else None

    report, report_errors = _load_json_safe(report_path, "controlled_writer_dry_run_report")
    dry_run_summary, summary_errors = _load_json_safe(summary_path, "controlled_writer_dry_run_summary")
    errors = [*report_errors, *summary_errors]
    warnings: list[str] = []
    report_sha = _sha256_file(report_path) if report_path.exists() and report_path.is_file() else ""

    if isinstance(report, dict) and isinstance(dry_run_summary, dict):
        _append_errors(errors, _schema_errors(report, dry_run_summary))
        _append_errors(errors, _hash_errors(report_path, report_sha, dry_run_summary))
        _append_errors(errors, _id_and_count_consistency_errors(report, dry_run_summary))
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
        _append_errors(
            errors,
            _count_errors(
                report,
                minimum_would_write_row_count=minimum_would_write_row_count,
                allow_dry_run_needs_review=allow_dry_run_needs_review,
                warnings=warnings,
            ),
        )
        _append_errors(
            errors,
            _missing_required_field_errors(
                report,
                require_zero_missing_required_fields=require_zero_missing_required_fields,
                warnings=warnings,
            ),
        )
        _append_errors(errors, _would_write_and_boundary_errors(report, dry_run_summary))
        _append_errors(errors, _redaction_status_errors(report, dry_run_summary))
        _append_errors(errors, _basename_reference_errors(dry_run_summary))
        if _contains_forbidden_material({"report": report, "summary": dry_run_summary}):
            errors.append("controlled_writer_dry_run_package_contains_unsafe_material")
    if evidence_path is not None:
        evidence_text, evidence_errors = _load_text_safe(evidence_path)
        errors.extend(evidence_errors)
        if evidence_text and _contains_forbidden_material(evidence_text):
            errors.append("controlled_writer_dry_run_evidence_contains_unsafe_material")

    status = "blocked" if errors else "needs_review" if warnings else "passed"
    summary = _summary(
        status=status,
        report_path=report_path,
        summary_path=summary_path,
        report_sha=report_sha,
        report=report if isinstance(report, dict) else {},
        dry_run_summary=dry_run_summary if isinstance(dry_run_summary, dict) else {},
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
        summary = precheck_property_training_dataset_controlled_writer_dry_run(
            controlled_writer_dry_run_report_path=args.controlled_writer_dry_run_report,
            controlled_writer_dry_run_summary_path=args.controlled_writer_dry_run_summary,
            controlled_writer_dry_run_evidence_path=args.controlled_writer_dry_run_evidence,
            output_summary_path=args.output_summary,
            output_markdown_path=args.output_markdown,
            require_dry_run_passed=args.require_dry_run_passed,
            allow_dry_run_needs_review=args.allow_dry_run_needs_review,
            require_zero_missing_required_fields=args.require_zero_missing_required_fields,
            minimum_would_write_row_count=args.minimum_would_write_row_count,
        )
    except Exception as exc:
        err.write(f"controlled writer dry-run precheck invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("precheck_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck"
        ),
        description="Precheck controlled writer dry-run report/summary outputs without rerunning the dry-run.",
    )
    parser.add_argument("--controlled-writer-dry-run-report", required=True)
    parser.add_argument("--controlled-writer-dry-run-summary", required=True)
    parser.add_argument("--controlled-writer-dry-run-evidence")
    parser.add_argument("--output-summary")
    parser.add_argument("--output-markdown")
    parser.add_argument("--require-dry-run-passed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-dry-run-needs-review", action="store_true")
    parser.add_argument(
        "--require-zero-missing-required-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum-would-write-row-count", type=int, default=1)
    return parser


def _load_json_safe(path: Path, prefix: str) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists() or not path.is_file():
        return None, [f"{prefix}_missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, [f"{prefix}_invalid_json"]
    if not isinstance(payload, dict):
        return None, [f"{prefix}_invalid_json"]
    return payload, []


def _load_text_safe(path: Path) -> tuple[str, list[str]]:
    if not path.exists() or not path.is_file():
        return "", ["controlled_writer_dry_run_evidence_missing"]
    try:
        return path.read_text(encoding="utf-8"), []
    except Exception:
        return "", ["controlled_writer_dry_run_evidence_invalid"]


def _schema_errors(report: dict[str, Any], dry_run_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != _REPORT_SCHEMA_VERSION:
        errors.append("controlled_writer_dry_run_report_schema_invalid")
    if dry_run_summary.get("schema_version") != _SUMMARY_SCHEMA_VERSION:
        errors.append("controlled_writer_dry_run_summary_schema_invalid")
    return errors


def _hash_errors(report_path: Path, report_sha: str, dry_run_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _is_sha(str(dry_run_summary.get("dry_run_report_sha256", ""))):
        errors.append("dry_run_report_sha256_invalid")
    elif dry_run_summary.get("dry_run_report_sha256") != report_sha:
        errors.append("dry_run_report_sha256_mismatch")
    if dry_run_summary.get("dry_run_report_basename") != report_path.name:
        errors.append("dry_run_report_basename_mismatch")
    return errors


def _id_and_count_consistency_errors(report: dict[str, Any], dry_run_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _COMMON_FIELDS:
        if report.get(field) != dry_run_summary.get(field):
            errors.append(f"{field}_mismatch")
    for field in ("dry_run_id", "corpus_id", "dataset_name"):
        for container in (report, dry_run_summary):
            if field in container and not _is_safe_id(container.get(field)):
                errors.append(f"{field}_unsafe")
    for field in _NON_NEGATIVE_COUNT_FIELDS:
        for container in (report, dry_run_summary):
            if field in container and not _is_non_negative_int(container.get(field)):
                errors.append(f"{field}_invalid")
    return _stable_unique(errors)


def _status_errors(
    report: dict[str, Any],
    dry_run_summary: dict[str, Any],
    *,
    require_dry_run_passed: bool,
    allow_dry_run_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    if report.get("dry_run_status") != dry_run_summary.get("dry_run_status"):
        return ["dry_run_status_mismatch"]
    status = report.get("dry_run_status")
    if status == "passed":
        return []
    if status == "needs_review":
        if allow_dry_run_needs_review or not require_dry_run_passed:
            warnings.append("dry_run_needs_review")
            return []
        return ["dry_run_needs_review"]
    if status in {"blocked", "failed"}:
        return ["dry_run_blocked"]
    return ["dry_run_status_invalid"]


def _count_errors(
    report: dict[str, Any],
    *,
    minimum_would_write_row_count: int,
    allow_dry_run_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if report.get("would_write_row_count", 0) < max(minimum_would_write_row_count, 1):
        errors.append("minimum_would_write_row_count_not_met")
    if report.get("would_write_field_count", 0) <= 0:
        errors.append("would_write_field_count_invalid")
    if report.get("blocked_candidate_record_count", 0) > 0:
        errors.append("blocked_candidate_records_present")
    if report.get("needs_review_candidate_record_count", 0) > 0:
        if allow_dry_run_needs_review:
            warnings.append("needs_review_candidate_records_present")
        else:
            errors.append("needs_review_candidate_records_present")
    return errors


def _missing_required_field_errors(
    report: dict[str, Any],
    *,
    require_zero_missing_required_fields: bool,
    warnings: list[str],
) -> list[str]:
    if report.get("missing_required_field_count", 0) == 0:
        return []
    if require_zero_missing_required_fields:
        return ["missing_required_fields"]
    warnings.append("missing_required_fields")
    return []


def _would_write_and_boundary_errors(report: dict[str, Any], dry_run_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _FALSE_REPORT_FLAGS:
        if report.get(field) is True:
            errors.append(field if field != "phase1_status" else "phase1_ran")
    for field in _FALSE_SUMMARY_FLAGS:
        if dry_run_summary.get(field) is True:
            errors.append(field)
    if report.get("phase1_status") != "not_run" or dry_run_summary.get("phase1_status") != "not_run":
        errors.append("phase1_ran")
    return _stable_unique(errors)


def _redaction_status_errors(report: dict[str, Any], dry_run_summary: dict[str, Any]) -> list[str]:
    if report.get("redaction_status") != "passed" or dry_run_summary.get("redaction_status") != "passed":
        return ["redaction_status_failed"]
    return []


def _basename_reference_errors(dry_run_summary: dict[str, Any]) -> list[str]:
    basename = dry_run_summary.get("dry_run_report_basename")
    if not isinstance(basename, str) or not basename or "/" in basename or "\\" in basename:
        return ["dry_run_report_basename_mismatch"]
    return []


def _summary(
    *,
    status: str,
    report_path: Path,
    summary_path: Path,
    report_sha: str,
    report: dict[str, Any],
    dry_run_summary: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": status,
        "dry_run_id": _safe_summary_value(report.get("dry_run_id")),
        "dry_run_status": _safe_summary_value(report.get("dry_run_status")),
        "dry_run_report_basename": report_path.name,
        "dry_run_report_sha256": report_sha if _is_sha(report_sha) else "",
        "dry_run_summary_basename": summary_path.name,
        "corpus_id": _safe_summary_value(report.get("corpus_id")),
        "dataset_name": _safe_summary_value(report.get("dataset_name")),
        "accepted_candidate_record_count": _safe_int(report.get("accepted_candidate_record_count")),
        "needs_review_candidate_record_count": _safe_int(report.get("needs_review_candidate_record_count")),
        "blocked_candidate_record_count": _safe_int(report.get("blocked_candidate_record_count")),
        "required_field_count": _safe_int(report.get("required_field_count")),
        "resolved_required_field_count": _safe_int(report.get("resolved_required_field_count")),
        "missing_required_field_count": _safe_int(report.get("missing_required_field_count")),
        "would_write_row_count": _safe_int(report.get("would_write_row_count")),
        "would_write_field_count": _safe_int(report.get("would_write_field_count")),
        "would_create_training_dataset_artifact": report.get("would_create_training_dataset_artifact") is True,
        "would_create_csv_jsonl_parquet_lmdb": report.get("would_create_csv_jsonl_parquet_lmdb") is True,
        "would_serialize_rows": report.get("would_serialize_rows") is True,
        "would_materialize_values": report.get("would_materialize_values") is True,
        "controlled_writer_executed": report.get("controlled_writer_executed") is True,
        "training_dataset_materialized": report.get("training_dataset_materialized") is True,
        "dataset_artifact_created": report.get("dataset_artifact_created") is True,
        "serialized_rows_created": report.get("serialized_rows_created") is True,
        "phase1_status": _safe_summary_value(report.get("phase1_status")),
        "dataset_confirmation_changed": report.get("dataset_confirmation_changed") is True,
        "model_training_run": report.get("model_training_run") is True,
        "evaluation_run": report.get("evaluation_run") is True,
        "redaction_status": "passed",
        "precheck_errors": _stable_unique(errors),
        "precheck_warnings": _stable_unique(warnings),
    }


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Property Training Dataset Controlled Writer Dry-Run Precheck Evidence",
            "",
            f"- precheck_status: {summary.get('precheck_status', '')}",
            f"- dry_run_id: {summary.get('dry_run_id', '')}",
            f"- dry_run_status: {summary.get('dry_run_status', '')}",
            f"- dry_run_report_basename: {summary.get('dry_run_report_basename', '')}",
            f"- dry_run_report_sha256: {summary.get('dry_run_report_sha256', '')}",
            f"- dry_run_summary_basename: {summary.get('dry_run_summary_basename', '')}",
            f"- accepted_candidate_record_count: {summary.get('accepted_candidate_record_count', 0)}",
            f"- needs_review_candidate_record_count: {summary.get('needs_review_candidate_record_count', 0)}",
            f"- blocked_candidate_record_count: {summary.get('blocked_candidate_record_count', 0)}",
            f"- required_field_count: {summary.get('required_field_count', 0)}",
            f"- resolved_required_field_count: {summary.get('resolved_required_field_count', 0)}",
            f"- missing_required_field_count: {summary.get('missing_required_field_count', 0)}",
            f"- would_write_row_count: {summary.get('would_write_row_count', 0)}",
            f"- would_write_field_count: {summary.get('would_write_field_count', 0)}",
            f"- precheck_errors: {', '.join(summary.get('precheck_errors', [])) or 'none'}",
            f"- precheck_warnings: {', '.join(summary.get('precheck_warnings', [])) or 'none'}",
            "",
            "## Boundary Statement",
            "",
            "This controlled writer dry-run precheck does not rerun the dry-run.",
            "This controlled writer dry-run precheck does not execute the controlled writer.",
            "This controlled writer dry-run precheck does not emit raw values.",
            "This controlled writer dry-run precheck does not materialize values.",
            "This controlled writer dry-run precheck does not serialize training rows.",
            "This controlled writer dry-run precheck does not create training dataset artifacts.",
            "This controlled writer dry-run precheck does not create CSV/JSONL/Parquet/LMDB artifacts.",
            "This controlled writer dry-run precheck does not generate conformers.",
            "This controlled writer dry-run precheck does not generate DPA3 structures.",
            "This controlled writer dry-run precheck does not run Phase 1.",
            "This controlled writer dry-run precheck does not modify DatasetConfirmation.",
            "This controlled writer dry-run precheck does not run model training or evaluation.",
            "",
        ]
    )


def _contains_forbidden_material(value: Any) -> bool:
    sanitized = _sanitize_allowed_terms(value)
    text = sanitized if isinstance(sanitized, str) else json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    lowered = text.lower()
    if _ABSOLUTE_PATH_VALUE_RE.search(text):
        return True
    return any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS)


def _sanitize_allowed_terms(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_allowed_terms(nested) for key, nested in value.items() if key in _SAFE_OUTPUT_KEYS or key not in {"would_create_csv_jsonl_parquet_lmdb"}}
    if isinstance(value, list):
        return [_sanitize_allowed_terms(item) for item in value]
    if not isinstance(value, str):
        return value
    return (
        value.replace(
            "This controlled writer dry-run precheck does not create CSV/JSONL/Parquet/LMDB artifacts.",
            "This controlled writer dry-run precheck does not create FORMAT-LABEL artifacts.",
        )
        .replace("CSV/JSONL/Parquet/LMDB artifacts", "FORMAT-LABEL artifacts")
        .replace("No training or candidate CSV/JSONL/Parquet/LMDB artifact was created.", "No format-label artifact was created.")
    )


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_dataset_controlled_writer_dry_run_precheck_redaction_failed"],
        "redaction_status": "failed",
    }


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc)
    if _contains_forbidden_material(message):
        return "redacted"
    return message or exc.__class__.__name__


def _safe_summary_value(value: Any) -> str:
    if isinstance(value, str) and _is_safe_id(value):
        return value
    return ""


def _safe_int(value: Any) -> int:
    return value if _is_non_negative_int(value) else 0


def _is_safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and bool(_SAFE_ID_RE.fullmatch(value))


def _is_sha(value: str) -> bool:
    return bool(_SHA_RE.fullmatch(value))


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _append_errors(errors: list[str], new_errors: list[str]) -> None:
    errors.extend(new_errors)


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


if __name__ == "__main__":
    raise SystemExit(main())
