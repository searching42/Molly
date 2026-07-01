from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO


_INPUT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_dry_run_precheck.v1"
_REQUEST_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_execution_request.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_execution_request_summary.v1"

_REQUEST_BASENAME = "property_training_dataset_controlled_writer_execution_request.json"
_SUMMARY_BASENAME = "property_training_dataset_controlled_writer_execution_request_summary.json"
_EVIDENCE_BASENAME = "redacted_property_training_dataset_controlled_writer_execution_request_evidence.md"

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
_SAFE_OUTPUT_KEYS = {
    "schema_version",
    "request_id",
    "request_status",
    "request_basename",
    "request_sha256",
    "requested_by",
    "request_purpose",
    "corpus_id",
    "dataset_name",
    "dry_run_precheck_summary_basename",
    "dry_run_precheck_summary_sha256",
    "dry_run_precheck_status",
    "dry_run_status",
    "dry_run_report_basename",
    "dry_run_report_sha256",
    "dry_run_summary_basename",
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
    "redaction_status",
    "requested_next_gate",
    "explicit_confirmation_required",
    "writer_execution_authorized",
    "request_errors",
    "request_warnings",
}
_COUNT_FIELDS = (
    "accepted_candidate_record_count",
    "needs_review_candidate_record_count",
    "blocked_candidate_record_count",
    "required_field_count",
    "resolved_required_field_count",
    "missing_required_field_count",
    "would_write_row_count",
    "would_write_field_count",
)
_FALSE_FLAGS = (
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


class PropertyTrainingDatasetControlledWriterExecutionRequestError(ValueError):
    pass


def create_property_training_dataset_controlled_writer_execution_request(
    *,
    controlled_writer_dry_run_precheck_summary_path: str | Path,
    output_dir: str | Path,
    request_id: str,
    requested_by: str,
    request_purpose: str = "controlled_writer_execution_request_for_preflight",
    require_dry_run_precheck_passed: bool = True,
    require_dry_run_passed: bool = True,
    require_zero_missing_required_fields: bool = True,
    allow_needs_review_candidates: bool = False,
    minimum_accepted_candidate_records: int = 1,
) -> dict[str, Any]:
    input_path = Path(controlled_writer_dry_run_precheck_summary_path)
    payload, load_errors = _load_json_safe(input_path)
    input_sha = _sha256_file(input_path)
    errors = [*load_errors]
    warnings: list[str] = []

    if payload is not None:
        _append_errors(errors, _schema_errors(payload))
        _append_errors(errors, _request_metadata_errors(request_id, requested_by, request_purpose))
        _append_errors(errors, _safe_input_id_errors(payload))
        _append_errors(
            errors,
            _status_errors(
                payload,
                require_dry_run_precheck_passed=require_dry_run_precheck_passed,
                require_dry_run_passed=require_dry_run_passed,
                warnings=warnings,
            ),
        )
        _append_errors(
            errors,
            _count_errors(
                payload,
                minimum_accepted_candidate_records=minimum_accepted_candidate_records,
                allow_needs_review_candidates=allow_needs_review_candidates,
                warnings=warnings,
            ),
        )
        _append_errors(
            errors,
            _missing_required_field_errors(
                payload,
                require_zero_missing_required_fields=require_zero_missing_required_fields,
                warnings=warnings,
            ),
        )
        _append_errors(errors, _hash_and_basename_errors(payload))
        _append_errors(errors, _boundary_errors(payload))
        _append_errors(errors, _redaction_status_errors(payload))
        if _contains_forbidden_material(payload):
            errors.append("controlled_writer_dry_run_precheck_summary_contains_unsafe_material")
    else:
        _append_errors(errors, _request_metadata_errors(request_id, requested_by, request_purpose))

    run_dir: Path | None = None
    if _is_safe_id(request_id):
        run_dir = Path(output_dir) / request_id
        if run_dir.exists():
            errors.append("controlled_writer_execution_request_output_dir_not_clean")

    if errors:
        return _blocked_summary(payload=payload or {}, errors=_stable_unique(errors), warnings=_stable_unique(warnings))

    if payload is None or run_dir is None:
        return _blocked_summary(payload=payload or {}, errors=["request_input_invalid"], warnings=_stable_unique(warnings))

    request_status = "request_needs_review" if warnings else "request_ready_for_preflight"
    request = _request(
        status=request_status,
        payload=payload,
        input_path=input_path,
        input_sha=input_sha,
        request_id=request_id,
        requested_by=requested_by,
        request_purpose=request_purpose,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    request_bytes = _json_bytes(request)
    request_sha = _sha256_bytes(request_bytes)
    summary = _summary(
        status=request_status,
        payload=payload,
        input_path=input_path,
        input_sha=input_sha,
        request_sha=request_sha,
        request_id=request_id,
        requested_by=requested_by,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    markdown = _markdown(request, summary)

    if _contains_forbidden_material({"request": request, "summary": summary, "markdown": markdown}):
        return _minimal_redaction_failure()

    run_dir.mkdir(parents=True)
    (run_dir / _REQUEST_BASENAME).write_bytes(request_bytes)
    (run_dir / _SUMMARY_BASENAME).write_bytes(_json_bytes(summary))
    (run_dir / _EVIDENCE_BASENAME).write_text(markdown, encoding="utf-8")
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
        summary = create_property_training_dataset_controlled_writer_execution_request(
            controlled_writer_dry_run_precheck_summary_path=args.controlled_writer_dry_run_precheck_summary,
            output_dir=args.output_dir,
            request_id=args.request_id,
            requested_by=args.requested_by,
            request_purpose=args.request_purpose,
            require_dry_run_precheck_passed=args.require_dry_run_precheck_passed,
            require_dry_run_passed=args.require_dry_run_passed,
            require_zero_missing_required_fields=args.require_zero_missing_required_fields,
            allow_needs_review_candidates=args.allow_needs_review_candidates,
            minimum_accepted_candidate_records=args.minimum_accepted_candidate_records,
        )
    except Exception as exc:
        err.write(f"controlled writer execution request invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("request_status") == "request_blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_request"
        ),
        description="Create a safe property training dataset controlled writer execution request artifact.",
    )
    parser.add_argument("--controlled-writer-dry-run-precheck-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--requested-by", required=True)
    parser.add_argument(
        "--request-purpose",
        default="controlled_writer_execution_request_for_preflight",
    )
    parser.add_argument("--allow-needs-review-candidates", action="store_true")
    parser.add_argument("--require-dry-run-precheck-passed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-dry-run-passed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-zero-missing-required-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum-accepted-candidate-records", type=int, default=1)
    return parser


def _load_json_safe(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists() or not path.is_file():
        return None, ["controlled_writer_dry_run_precheck_summary_missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, ["controlled_writer_dry_run_precheck_summary_invalid_json"]
    if not isinstance(payload, dict):
        return None, ["controlled_writer_dry_run_precheck_summary_invalid_json"]
    return payload, []


def _schema_errors(payload: dict[str, Any]) -> list[str]:
    if payload.get("schema_version") != _INPUT_SCHEMA_VERSION:
        return ["controlled_writer_dry_run_precheck_schema_invalid"]
    return []


def _request_metadata_errors(request_id: str, requested_by: str, request_purpose: str) -> list[str]:
    errors: list[str] = []
    if not _is_safe_id(request_id):
        errors.append("request_id_unsafe")
    if not _is_safe_id(requested_by):
        errors.append("requested_by_unsafe")
    if not _is_safe_id(request_purpose):
        errors.append("request_purpose_unsafe")
    return errors


def _safe_input_id_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("dry_run_id", "corpus_id", "dataset_name"):
        if field in payload and not _is_safe_id(payload.get(field)):
            errors.append(f"{field}_unsafe")
    return errors


def _status_errors(
    payload: dict[str, Any],
    *,
    require_dry_run_precheck_passed: bool,
    require_dry_run_passed: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    precheck_status = payload.get("precheck_status")
    if precheck_status == "passed":
        pass
    elif precheck_status == "needs_review":
        if require_dry_run_precheck_passed:
            errors.append("dry_run_precheck_needs_review")
        else:
            warnings.append("dry_run_precheck_needs_review")
    elif precheck_status in {"blocked", "failed"}:
        errors.append("dry_run_precheck_blocked")
    else:
        errors.append("dry_run_precheck_status_invalid")

    dry_run_status = payload.get("dry_run_status")
    if dry_run_status == "passed":
        pass
    elif dry_run_status == "needs_review":
        if require_dry_run_passed:
            errors.append("dry_run_needs_review")
        else:
            warnings.append("dry_run_needs_review")
    elif dry_run_status in {"blocked", "failed"}:
        errors.append("dry_run_blocked")
    else:
        errors.append("dry_run_status_invalid")
    return errors


def _count_errors(
    payload: dict[str, Any],
    *,
    minimum_accepted_candidate_records: int,
    allow_needs_review_candidates: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    for field in _COUNT_FIELDS:
        if field in payload and not _is_non_negative_int(payload.get(field)):
            errors.append(f"{field}_invalid")
    if payload.get("accepted_candidate_record_count", 0) < max(minimum_accepted_candidate_records, 1):
        errors.append("minimum_accepted_candidate_records_not_met")
    if payload.get("would_write_row_count", 0) < max(minimum_accepted_candidate_records, 1):
        errors.append("minimum_would_write_row_count_not_met")
    if payload.get("would_write_field_count", 0) <= 0:
        errors.append("would_write_field_count_invalid")
    if payload.get("blocked_candidate_record_count", 0) > 0:
        errors.append("blocked_candidate_records_present")
    if payload.get("needs_review_candidate_record_count", 0) > 0:
        if allow_needs_review_candidates:
            warnings.append("needs_review_candidate_records_present")
        else:
            errors.append("needs_review_candidate_records_present")
    return errors


def _missing_required_field_errors(
    payload: dict[str, Any],
    *,
    require_zero_missing_required_fields: bool,
    warnings: list[str],
) -> list[str]:
    if payload.get("missing_required_field_count", 0) == 0:
        return []
    if require_zero_missing_required_fields:
        return ["missing_required_fields"]
    warnings.append("missing_required_fields")
    return []


def _hash_and_basename_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _is_sha(str(payload.get("dry_run_report_sha256", ""))):
        errors.append("dry_run_report_sha256_invalid")
    for field in ("dry_run_report_basename", "dry_run_summary_basename"):
        if not _is_safe_basename(payload.get(field)):
            errors.append(f"{field}_invalid")
    return errors


def _boundary_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _FALSE_FLAGS:
        if payload.get(field) is True:
            errors.append(field)
    if payload.get("phase1_status") != "not_run":
        errors.append("phase1_ran")
    return _stable_unique(errors)


def _redaction_status_errors(payload: dict[str, Any]) -> list[str]:
    if payload.get("redaction_status") != "passed":
        return ["redaction_status_failed"]
    return []


def _request(
    *,
    status: str,
    payload: dict[str, Any],
    input_path: Path,
    input_sha: str,
    request_id: str,
    requested_by: str,
    request_purpose: str,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _REQUEST_SCHEMA_VERSION,
        "request_id": request_id,
        "request_status": status,
        "requested_by": requested_by,
        "request_purpose": request_purpose,
        "corpus_id": _safe_summary_value(payload.get("corpus_id")),
        "dataset_name": _safe_summary_value(payload.get("dataset_name")),
        "dry_run_precheck_summary_basename": input_path.name,
        "dry_run_precheck_summary_sha256": input_sha,
        "dry_run_precheck_status": _safe_summary_value(payload.get("precheck_status")),
        "dry_run_status": _safe_summary_value(payload.get("dry_run_status")),
        "dry_run_report_basename": _safe_basename_value(payload.get("dry_run_report_basename")),
        "dry_run_report_sha256": payload.get("dry_run_report_sha256", "") if _is_sha(str(payload.get("dry_run_report_sha256", ""))) else "",
        "dry_run_summary_basename": _safe_basename_value(payload.get("dry_run_summary_basename")),
        "accepted_candidate_record_count": _safe_int(payload.get("accepted_candidate_record_count")),
        "needs_review_candidate_record_count": _safe_int(payload.get("needs_review_candidate_record_count")),
        "blocked_candidate_record_count": _safe_int(payload.get("blocked_candidate_record_count")),
        "required_field_count": _safe_int(payload.get("required_field_count")),
        "resolved_required_field_count": _safe_int(payload.get("resolved_required_field_count")),
        "missing_required_field_count": _safe_int(payload.get("missing_required_field_count")),
        "would_write_row_count": _safe_int(payload.get("would_write_row_count")),
        "would_write_field_count": _safe_int(payload.get("would_write_field_count")),
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "serialized_rows_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "redaction_status": "passed",
        "requested_next_gate": "controlled_writer_execution_request_preflight",
        "explicit_confirmation_required": True,
        "writer_execution_authorized": False,
        "request_errors": _stable_unique(errors),
        "request_warnings": _stable_unique(warnings),
    }


def _summary(
    *,
    status: str,
    payload: dict[str, Any],
    input_path: Path,
    input_sha: str,
    request_sha: str,
    request_id: str,
    requested_by: str,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "request_id": request_id,
        "request_status": status,
        "request_basename": _REQUEST_BASENAME,
        "request_sha256": request_sha,
        "requested_by": requested_by,
        "corpus_id": _safe_summary_value(payload.get("corpus_id")),
        "dataset_name": _safe_summary_value(payload.get("dataset_name")),
        "dry_run_precheck_summary_basename": input_path.name,
        "dry_run_precheck_summary_sha256": input_sha,
        "dry_run_precheck_status": _safe_summary_value(payload.get("precheck_status")),
        "dry_run_status": _safe_summary_value(payload.get("dry_run_status")),
        "accepted_candidate_record_count": _safe_int(payload.get("accepted_candidate_record_count")),
        "would_write_row_count": _safe_int(payload.get("would_write_row_count")),
        "would_write_field_count": _safe_int(payload.get("would_write_field_count")),
        "missing_required_field_count": _safe_int(payload.get("missing_required_field_count")),
        "redaction_status": "passed",
        "explicit_confirmation_required": True,
        "writer_execution_authorized": False,
        "controlled_writer_executed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "serialized_rows_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "request_errors": _stable_unique(errors),
        "request_warnings": _stable_unique(warnings),
    }


def _blocked_summary(*, payload: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "request_status": "request_blocked",
        "corpus_id": _safe_summary_value(payload.get("corpus_id")),
        "dataset_name": _safe_summary_value(payload.get("dataset_name")),
        "redaction_status": "passed" if payload.get("redaction_status") != "failed" else "failed",
        "writer_execution_authorized": False,
        "explicit_confirmation_required": True,
        "request_errors": _stable_unique(errors),
        "request_warnings": _stable_unique(warnings),
    }


def _markdown(request: dict[str, Any], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Property Training Dataset Controlled Writer Execution Request Evidence",
            "",
            f"- request_id: {summary.get('request_id', '')}",
            f"- request_status: {summary.get('request_status', '')}",
            f"- requested_by: {summary.get('requested_by', '')}",
            f"- request_purpose: {request.get('request_purpose', '')}",
            f"- corpus_id: {summary.get('corpus_id', '')}",
            f"- dataset_name: {summary.get('dataset_name', '')}",
            f"- dry_run_precheck_summary_basename: {summary.get('dry_run_precheck_summary_basename', '')}",
            f"- dry_run_precheck_summary_sha256: {summary.get('dry_run_precheck_summary_sha256', '')}",
            f"- dry_run_precheck_status: {summary.get('dry_run_precheck_status', '')}",
            f"- dry_run_status: {summary.get('dry_run_status', '')}",
            f"- accepted_candidate_record_count: {summary.get('accepted_candidate_record_count', 0)}",
            f"- would_write_row_count: {summary.get('would_write_row_count', 0)}",
            f"- would_write_field_count: {summary.get('would_write_field_count', 0)}",
            f"- missing_required_field_count: {summary.get('missing_required_field_count', 0)}",
            f"- redaction_status: {summary.get('redaction_status', '')}",
            f"- writer_execution_authorized: {summary.get('writer_execution_authorized', False)}",
            f"- explicit_confirmation_required: {summary.get('explicit_confirmation_required', True)}",
            f"- request_errors: {', '.join(summary.get('request_errors', [])) or 'none'}",
            f"- request_warnings: {', '.join(summary.get('request_warnings', [])) or 'none'}",
            "",
            "## Boundary Statement",
            "",
            "This controlled writer execution request does not implement execution request preflight.",
            "This controlled writer execution request does not explicitly confirm execution.",
            "This controlled writer execution request does not execute the controlled writer.",
            "This controlled writer execution request does not authorize writer execution by itself.",
            "This controlled writer execution request keeps explicit confirmation required.",
            "This controlled writer execution request does not emit raw values.",
            "This controlled writer execution request does not materialize values.",
            "This controlled writer execution request does not serialize training rows.",
            "This controlled writer execution request does not create training dataset artifacts.",
            "This controlled writer execution request does not create CSV/JSONL/Parquet/LMDB artifacts.",
            "This controlled writer execution request does not generate conformers.",
            "This controlled writer execution request does not generate DPA3 structures.",
            "This controlled writer execution request does not run Phase 1.",
            "This controlled writer execution request does not modify DatasetConfirmation.",
            "This controlled writer execution request does not run model training or evaluation.",
            "",
        ]
    )


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "request_status": "request_blocked",
        "request_errors": ["property_training_dataset_controlled_writer_execution_request_redaction_failed"],
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
            "This controlled writer execution request does not create CSV/JSONL/Parquet/LMDB artifacts.",
            "This controlled writer execution request does not create FORMAT-LABEL artifacts.",
        )
        .replace("CSV/JSONL/Parquet/LMDB artifacts", "FORMAT-LABEL artifacts")
        .replace("would_create_csv_jsonl_parquet_lmdb", "would_create_format_label")
    )


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


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
