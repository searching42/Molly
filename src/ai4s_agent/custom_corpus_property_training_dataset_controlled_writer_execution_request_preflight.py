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


_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight.v1"
_REQUEST_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_execution_request.v1"
_REQUEST_SUMMARY_SCHEMA_VERSION = (
    "custom_corpus_property_training_dataset_controlled_writer_execution_request_summary.v1"
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ABSOLUTE_PATH_VALUE_RE = re.compile(r'"(?:/|[A-Za-z]:\\\\)|/tmp/|/private/')
_FORBIDDEN_MARKERS = (
    "/Users/",
    "/home/",
    "C:\\",
    "Authorization",
    "Bearer",
    "token=",
    "secret=",
    "password=",
    "cookie=",
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
    "smiles=",
    "c1=cc",
    "0.72",
)
_COMMON_FIELDS = (
    "request_id",
    "request_status",
    "requested_by",
    "corpus_id",
    "dataset_name",
    "dry_run_precheck_summary_basename",
    "dry_run_precheck_summary_sha256",
    "dry_run_precheck_status",
    "dry_run_status",
    "accepted_candidate_record_count",
    "would_write_row_count",
    "would_write_field_count",
    "missing_required_field_count",
    "redaction_status",
    "explicit_confirmation_required",
    "writer_execution_authorized",
    "controlled_writer_executed",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "serialized_rows_created",
    "phase1_status",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
)
_OPTIONAL_COMMON_ZERO_FIELDS = (
    "needs_review_candidate_record_count",
    "blocked_candidate_record_count",
)
_COUNT_FIELDS = (
    "accepted_candidate_record_count",
    "needs_review_candidate_record_count",
    "blocked_candidate_record_count",
    "missing_required_field_count",
    "would_write_row_count",
    "would_write_field_count",
)
_FALSE_FLAGS = (
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
    "preflight_status",
    "request_id",
    "request_status",
    "request_basename",
    "request_sha256",
    "request_summary_basename",
    "requested_by",
    "corpus_id",
    "dataset_name",
    "dry_run_precheck_summary_basename",
    "dry_run_precheck_summary_sha256",
    "dry_run_precheck_status",
    "dry_run_status",
    "accepted_candidate_record_count",
    "needs_review_candidate_record_count",
    "blocked_candidate_record_count",
    "missing_required_field_count",
    "would_write_row_count",
    "would_write_field_count",
    "redaction_status",
    "explicit_confirmation_required",
    "writer_execution_authorized",
    "controlled_writer_executed",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "serialized_rows_created",
    "phase1_status",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
    "next_gate",
    "preflight_errors",
    "preflight_warnings",
}


class PropertyTrainingDatasetControlledWriterExecutionRequestPreflightError(ValueError):
    pass


def preflight_property_training_dataset_controlled_writer_execution_request(
    *,
    controlled_writer_execution_request_path: str | Path,
    controlled_writer_execution_request_summary_path: str | Path,
    controlled_writer_execution_request_evidence_path: str | Path | None = None,
    output_summary_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    require_request_ready_for_preflight: bool = True,
    allow_request_needs_review: bool = False,
    require_explicit_confirmation_required: bool = True,
    require_writer_execution_unauthorized: bool = True,
    require_zero_missing_required_fields: bool = True,
    minimum_accepted_candidate_records: int = 1,
) -> dict[str, Any]:
    request_path = Path(controlled_writer_execution_request_path)
    request_summary_path = Path(controlled_writer_execution_request_summary_path)
    evidence_path = (
        Path(controlled_writer_execution_request_evidence_path)
        if controlled_writer_execution_request_evidence_path
        else None
    )
    request, request_errors = _load_json_safe(request_path, "controlled_writer_execution_request")
    request_summary, summary_errors = _load_json_safe(
        request_summary_path, "controlled_writer_execution_request_summary"
    )
    errors = [*request_errors, *summary_errors]
    warnings: list[str] = []
    request_sha = _sha256_file(request_path)

    if isinstance(request, dict) and isinstance(request_summary, dict):
        _append_errors(errors, _schema_errors(request, request_summary))
        _append_errors(errors, _hash_errors(request_path, request_sha, request_summary))
        _append_errors(errors, _consistency_errors(request, request_summary))
        _append_errors(
            errors,
            _status_errors(
                request,
                request_summary,
                require_request_ready_for_preflight=require_request_ready_for_preflight,
                allow_request_needs_review=allow_request_needs_review,
                warnings=warnings,
            ),
        )
        _append_errors(
            errors,
            _count_errors(
                request,
                minimum_accepted_candidate_records=minimum_accepted_candidate_records,
                allow_request_needs_review=allow_request_needs_review,
                warnings=warnings,
            ),
        )
        _append_errors(
            errors,
            _missing_required_field_errors(
                request,
                require_zero_missing_required_fields=require_zero_missing_required_fields,
                warnings=warnings,
            ),
        )
        _append_errors(
            errors,
            _authorization_errors(
                request,
                require_explicit_confirmation_required=require_explicit_confirmation_required,
                require_writer_execution_unauthorized=require_writer_execution_unauthorized,
                warnings=warnings,
            ),
        )
        _append_errors(errors, _boundary_errors(request))
        _append_errors(errors, _basename_and_hash_errors(request, request_summary))
        _append_errors(errors, _redaction_status_errors(request, request_summary))
        if _contains_forbidden_material({"request": request, "summary": request_summary}):
            errors.append("controlled_writer_execution_request_package_contains_unsafe_material")
    if evidence_path is not None:
        evidence_text, evidence_errors = _load_text_safe(evidence_path)
        errors.extend(evidence_errors)
        if evidence_text and _contains_forbidden_material(evidence_text):
            errors.append("controlled_writer_execution_request_evidence_contains_unsafe_material")

    status = "preflight_blocked" if errors else "preflight_needs_review" if warnings else "preflight_passed"
    summary = _summary(
        status=status,
        request_path=request_path,
        request_summary_path=request_summary_path,
        request_sha=request_sha,
        request=request if isinstance(request, dict) else {},
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
        summary = preflight_property_training_dataset_controlled_writer_execution_request(
            controlled_writer_execution_request_path=args.controlled_writer_execution_request,
            controlled_writer_execution_request_summary_path=args.controlled_writer_execution_request_summary,
            controlled_writer_execution_request_evidence_path=args.controlled_writer_execution_request_evidence,
            output_summary_path=args.output_summary,
            output_markdown_path=args.output_markdown,
            require_request_ready_for_preflight=args.require_request_ready_for_preflight,
            allow_request_needs_review=args.allow_request_needs_review,
            require_explicit_confirmation_required=args.require_explicit_confirmation_required,
            require_writer_execution_unauthorized=args.require_writer_execution_unauthorized,
            require_zero_missing_required_fields=args.require_zero_missing_required_fields,
            minimum_accepted_candidate_records=args.minimum_accepted_candidate_records,
        )
    except Exception as exc:
        err.write(f"controlled writer execution request preflight invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("preflight_status") == "preflight_blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_request_preflight"
        ),
        description="Preflight controlled writer execution request artifacts without executing a writer.",
    )
    parser.add_argument("--controlled-writer-execution-request", required=True)
    parser.add_argument("--controlled-writer-execution-request-summary", required=True)
    parser.add_argument("--controlled-writer-execution-request-evidence")
    parser.add_argument("--output-summary")
    parser.add_argument("--output-markdown")
    parser.add_argument("--allow-request-needs-review", action="store_true")
    parser.add_argument(
        "--require-request-ready-for-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-explicit-confirmation-required",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-writer-execution-unauthorized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-zero-missing-required-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum-accepted-candidate-records", type=int, default=1)
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
        return "", ["controlled_writer_execution_request_evidence_missing"]
    try:
        return path.read_text(encoding="utf-8"), []
    except Exception:
        return "", ["controlled_writer_execution_request_evidence_invalid"]


def _schema_errors(request: dict[str, Any], request_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if request.get("schema_version") != _REQUEST_SCHEMA_VERSION:
        errors.append("controlled_writer_execution_request_schema_invalid")
    if request_summary.get("schema_version") != _REQUEST_SUMMARY_SCHEMA_VERSION:
        errors.append("controlled_writer_execution_request_summary_schema_invalid")
    return errors


def _hash_errors(request_path: Path, request_sha: str, request_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _is_sha(str(request_summary.get("request_sha256", ""))):
        errors.append("request_sha256_invalid")
    elif request_summary.get("request_sha256") != request_sha:
        errors.append("request_sha256_mismatch")
    if request_summary.get("request_basename") != request_path.name:
        errors.append("request_basename_mismatch")
    return errors


def _consistency_errors(request: dict[str, Any], request_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _COMMON_FIELDS:
        if request.get(field) != request_summary.get(field):
            errors.append(f"{field}_mismatch")
    for field in _OPTIONAL_COMMON_ZERO_FIELDS:
        if _safe_int(request.get(field)) != _safe_int(request_summary.get(field)):
            errors.append(f"{field}_mismatch")
    for field in ("request_id", "requested_by", "corpus_id", "dataset_name"):
        for container in (request, request_summary):
            if field in container and not _is_safe_id(container.get(field)):
                errors.append(f"{field}_unsafe")
    for field in _COUNT_FIELDS:
        for container in (request, request_summary):
            if field in container and not _is_non_negative_int(container.get(field)):
                errors.append(f"{field}_invalid")
    return _stable_unique(errors)


def _status_errors(
    request: dict[str, Any],
    request_summary: dict[str, Any],
    *,
    require_request_ready_for_preflight: bool,
    allow_request_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    if request.get("request_status") != request_summary.get("request_status"):
        return ["request_status_mismatch"]
    status = request.get("request_status")
    if status == "request_ready_for_preflight":
        return []
    if status == "request_needs_review":
        if allow_request_needs_review or not require_request_ready_for_preflight:
            warnings.append("request_needs_review")
            return []
        return ["request_needs_review"]
    if status in {"request_blocked", "blocked", "failed"}:
        return ["request_blocked"]
    return ["request_status_invalid"]


def _count_errors(
    request: dict[str, Any],
    *,
    minimum_accepted_candidate_records: int,
    allow_request_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    minimum = max(minimum_accepted_candidate_records, 1)
    if request.get("accepted_candidate_record_count", 0) < minimum:
        errors.append("minimum_accepted_candidate_records_not_met")
    if request.get("would_write_row_count", 0) < minimum:
        errors.append("minimum_would_write_row_count_not_met")
    if request.get("would_write_field_count", 0) <= 0:
        errors.append("would_write_field_count_invalid")
    if request.get("blocked_candidate_record_count", 0) > 0:
        errors.append("blocked_candidate_records_present")
    if request.get("needs_review_candidate_record_count", 0) > 0:
        if allow_request_needs_review:
            warnings.append("needs_review_candidate_records_present")
        else:
            errors.append("needs_review_candidate_records_present")
    return errors


def _missing_required_field_errors(
    request: dict[str, Any],
    *,
    require_zero_missing_required_fields: bool,
    warnings: list[str],
) -> list[str]:
    if request.get("missing_required_field_count", 0) == 0:
        return []
    if require_zero_missing_required_fields:
        return ["missing_required_fields"]
    warnings.append("missing_required_fields")
    return []


def _authorization_errors(
    request: dict[str, Any],
    *,
    require_explicit_confirmation_required: bool,
    require_writer_execution_unauthorized: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if request.get("writer_execution_authorized") is True:
        if require_writer_execution_unauthorized:
            errors.append("writer_execution_authorized")
        else:
            warnings.append("writer_execution_authorized")
    if request.get("explicit_confirmation_required") is not True:
        if require_explicit_confirmation_required:
            errors.append("explicit_confirmation_required")
        else:
            warnings.append("explicit_confirmation_required")
    if request.get("requested_next_gate") != "controlled_writer_execution_request_preflight":
        errors.append("requested_next_gate_invalid")
    return errors


def _boundary_errors(request: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _FALSE_FLAGS:
        if request.get(field) is True:
            errors.append(field)
    if request.get("phase1_status") != "not_run":
        errors.append("phase1_ran")
    return _stable_unique(errors)


def _basename_and_hash_errors(request: dict[str, Any], request_summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "dry_run_precheck_summary_basename",
        "dry_run_report_basename",
        "dry_run_summary_basename",
    ):
        if field in request and not _is_safe_basename(request.get(field)):
            errors.append(f"{field}_invalid")
    for field in ("request_basename",):
        if field in request_summary and not _is_safe_basename(request_summary.get(field)):
            errors.append(f"{field}_invalid")
    for field in ("dry_run_precheck_summary_sha256", "dry_run_report_sha256"):
        if field in request and not _is_sha(str(request.get(field, ""))):
            errors.append(f"{field}_invalid")
    return errors


def _redaction_status_errors(request: dict[str, Any], request_summary: dict[str, Any]) -> list[str]:
    if request.get("redaction_status") != "passed" or request_summary.get("redaction_status") != "passed":
        return ["redaction_status_failed"]
    return []


def _summary(
    *,
    status: str,
    request_path: Path,
    request_summary_path: Path,
    request_sha: str,
    request: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": status,
        "request_id": _safe_summary_value(request.get("request_id")),
        "request_status": _safe_summary_value(request.get("request_status")),
        "request_basename": request_path.name,
        "request_sha256": request_sha if _is_sha(request_sha) else "",
        "request_summary_basename": request_summary_path.name,
        "requested_by": _safe_summary_value(request.get("requested_by")),
        "corpus_id": _safe_summary_value(request.get("corpus_id")),
        "dataset_name": _safe_summary_value(request.get("dataset_name")),
        "dry_run_precheck_summary_basename": _safe_basename_value(request.get("dry_run_precheck_summary_basename")),
        "dry_run_precheck_summary_sha256": request.get("dry_run_precheck_summary_sha256", "")
        if _is_sha(str(request.get("dry_run_precheck_summary_sha256", "")))
        else "",
        "dry_run_precheck_status": _safe_summary_value(request.get("dry_run_precheck_status")),
        "dry_run_status": _safe_summary_value(request.get("dry_run_status")),
        "accepted_candidate_record_count": _safe_int(request.get("accepted_candidate_record_count")),
        "needs_review_candidate_record_count": _safe_int(request.get("needs_review_candidate_record_count")),
        "blocked_candidate_record_count": _safe_int(request.get("blocked_candidate_record_count")),
        "missing_required_field_count": _safe_int(request.get("missing_required_field_count")),
        "would_write_row_count": _safe_int(request.get("would_write_row_count")),
        "would_write_field_count": _safe_int(request.get("would_write_field_count")),
        "redaction_status": "passed",
        "explicit_confirmation_required": request.get("explicit_confirmation_required") is True,
        "writer_execution_authorized": request.get("writer_execution_authorized") is True,
        "controlled_writer_executed": request.get("controlled_writer_executed") is True,
        "training_dataset_materialized": request.get("training_dataset_materialized") is True,
        "dataset_artifact_created": request.get("dataset_artifact_created") is True,
        "serialized_rows_created": request.get("serialized_rows_created") is True,
        "phase1_status": _safe_summary_value(request.get("phase1_status")),
        "dataset_confirmation_changed": request.get("dataset_confirmation_changed") is True,
        "model_training_run": request.get("model_training_run") is True,
        "evaluation_run": request.get("evaluation_run") is True,
        "next_gate": "future_explicit_confirmation",
        "preflight_errors": _stable_unique(errors),
        "preflight_warnings": _stable_unique(warnings),
    }


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Property Training Dataset Controlled Writer Execution Request Preflight Evidence",
            "",
            f"- preflight_status: {summary.get('preflight_status', '')}",
            f"- request_id: {summary.get('request_id', '')}",
            f"- request_status: {summary.get('request_status', '')}",
            f"- request_basename: {summary.get('request_basename', '')}",
            f"- request_sha256: {summary.get('request_sha256', '')}",
            f"- request_summary_basename: {summary.get('request_summary_basename', '')}",
            f"- requested_by: {summary.get('requested_by', '')}",
            f"- dry_run_precheck_summary_basename: {summary.get('dry_run_precheck_summary_basename', '')}",
            f"- dry_run_precheck_summary_sha256: {summary.get('dry_run_precheck_summary_sha256', '')}",
            f"- dry_run_precheck_status: {summary.get('dry_run_precheck_status', '')}",
            f"- dry_run_status: {summary.get('dry_run_status', '')}",
            f"- accepted_candidate_record_count: {summary.get('accepted_candidate_record_count', 0)}",
            f"- needs_review_candidate_record_count: {summary.get('needs_review_candidate_record_count', 0)}",
            f"- blocked_candidate_record_count: {summary.get('blocked_candidate_record_count', 0)}",
            f"- missing_required_field_count: {summary.get('missing_required_field_count', 0)}",
            f"- would_write_row_count: {summary.get('would_write_row_count', 0)}",
            f"- would_write_field_count: {summary.get('would_write_field_count', 0)}",
            f"- writer_execution_authorized: {summary.get('writer_execution_authorized', False)}",
            f"- explicit_confirmation_required: {summary.get('explicit_confirmation_required', True)}",
            f"- next_gate: {summary.get('next_gate', '')}",
            f"- preflight_errors: {', '.join(summary.get('preflight_errors', [])) or 'none'}",
            f"- preflight_warnings: {', '.join(summary.get('preflight_warnings', [])) or 'none'}",
            "",
            "## Boundary Statement",
            "",
            "This controlled writer execution request preflight does not explicitly confirm execution.",
            "This controlled writer execution request preflight does not execute the controlled writer.",
            "This controlled writer execution request preflight does not authorize writer execution by itself.",
            "This controlled writer execution request preflight keeps explicit confirmation required.",
            "This controlled writer execution request preflight does not emit raw values.",
            "This controlled writer execution request preflight does not materialize values.",
            "This controlled writer execution request preflight does not serialize training rows.",
            "This controlled writer execution request preflight does not create training dataset artifacts.",
            "This controlled writer execution request preflight does not create CSV/JSONL/Parquet/LMDB artifacts.",
            "This controlled writer execution request preflight does not generate conformers.",
            "This controlled writer execution request preflight does not generate DPA3 structures.",
            "This controlled writer execution request preflight does not run Phase 1.",
            "This controlled writer execution request preflight does not modify DatasetConfirmation.",
            "This controlled writer execution request preflight does not run model training or evaluation.",
            "",
        ]
    )


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": "preflight_blocked",
        "preflight_errors": [
            "property_training_dataset_controlled_writer_execution_request_preflight_redaction_failed"
        ],
        "redaction_status": "failed",
        "writer_execution_authorized": False,
        "explicit_confirmation_required": True,
    }


def _contains_forbidden_material(value: Any) -> bool:
    sanitized = _sanitize_allowed_terms(value)
    text = sanitized if isinstance(sanitized, str) else json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    lowered = text.lower()
    if _ABSOLUTE_PATH_VALUE_RE.search(text):
        return True
    return any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS)


def _sanitize_allowed_terms(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_allowed_terms(nested)
            for key, nested in value.items()
            if key in _SAFE_OUTPUT_KEYS or key not in {"would_create_csv_jsonl_parquet_lmdb"}
        }
    if isinstance(value, list):
        return [_sanitize_allowed_terms(item) for item in value]
    if not isinstance(value, str):
        return value
    return (
        value.replace(
            "This controlled writer execution request preflight does not create CSV/JSONL/Parquet/LMDB artifacts.",
            "This controlled writer execution request preflight does not create FORMAT-LABEL artifacts.",
        )
        .replace("CSV/JSONL/Parquet/LMDB artifacts", "FORMAT-LABEL artifacts")
        .replace("would_create_csv_jsonl_parquet_lmdb", "would_create_format_label")
    )


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


def _safe_basename_value(value: Any) -> str:
    if _is_safe_basename(value):
        return str(value)
    return ""


def _safe_int(value: Any) -> int:
    return value if _is_non_negative_int(value) else 0


def _is_safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and bool(_SAFE_ID_RE.fullmatch(value))


def _is_sha(value: str) -> bool:
    return bool(_SHA_RE.fullmatch(value))


def _is_safe_basename(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and "/" not in value and "\\" not in value


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _append_errors(errors: list[str], new_errors: list[str]) -> None:
    errors.extend(new_errors)


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


if __name__ == "__main__":
    raise SystemExit(main())
