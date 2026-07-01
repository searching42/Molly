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


_INPUT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_dry_run_input.v1"
_REPORT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_dry_run_report.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_dry_run_summary.v1"

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
_TOP_LEVEL_REQUIRED_FIELDS = (
    "schema_version",
    "dry_run_id",
    "corpus_id",
    "dataset_name",
    "controlled_writer_design_plan_preflight_status",
    "controlled_writer_design_plan_preflight_id",
    "domain_validation_boundary_status",
    "controlled_writer_value_resolution_dry_run_precheck_status",
    "accepted_candidate_record_count",
    "needs_review_candidate_record_count",
    "blocked_candidate_record_count",
    "field_coverage",
    "would_write",
    "boundary_flags",
    "redaction_status",
)
_SAFE_ID_FIELDS = (
    "dry_run_id",
    "corpus_id",
    "dataset_name",
    "controlled_writer_design_plan_preflight_id",
)
_FALSE_WOULD_WRITE_FLAGS = (
    "would_create_training_dataset_artifact",
    "would_create_csv_jsonl_parquet_lmdb",
    "would_serialize_rows",
    "would_materialize_values",
)
_FALSE_BOUNDARY_FLAGS = (
    "controlled_writer_executed",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "serialized_rows_created",
    "dataset_confirmation_changed",
    "model_training_run",
    "evaluation_run",
)


class PropertyTrainingDatasetControlledWriterDryRunError(ValueError):
    pass


def run_property_training_dataset_controlled_writer_dry_run(
    *,
    controlled_writer_dry_run_input_path: str | Path,
    output_dir: str | Path,
    require_design_plan_preflight_passed: bool = True,
    require_domain_validation_passed: bool = True,
    require_value_resolution_precheck_passed: bool = True,
    require_values_resolved: bool = True,
    allow_needs_review_candidates: bool = False,
    minimum_accepted_candidate_records: int = 1,
) -> dict[str, Any]:
    input_path = Path(controlled_writer_dry_run_input_path)
    payload = _load_json(input_path)
    errors: list[str] = []
    warnings: list[str] = []

    _append_errors(errors, _top_level_errors(payload))
    _append_errors(errors, _schema_errors(payload))
    _append_errors(errors, _safe_id_errors(payload))
    _append_errors(
        errors,
        _status_errors(
            payload,
            require_design_plan_preflight_passed=require_design_plan_preflight_passed,
            require_domain_validation_passed=require_domain_validation_passed,
            require_value_resolution_precheck_passed=require_value_resolution_precheck_passed,
            warnings=warnings,
        ),
    )
    _append_errors(
        errors,
        _candidate_count_errors(
            payload,
            minimum_accepted_candidate_records=minimum_accepted_candidate_records,
            allow_needs_review_candidates=allow_needs_review_candidates,
            warnings=warnings,
        ),
    )
    _append_errors(
        errors,
        _field_coverage_errors(payload, require_values_resolved=require_values_resolved, warnings=warnings),
    )
    _append_errors(errors, _would_write_errors(payload))
    _append_errors(errors, _boundary_errors(payload))
    _append_errors(errors, _redaction_status_errors(payload))
    if _contains_forbidden_material(payload):
        errors.append("controlled_writer_dry_run_input_contains_unsafe_material")

    dry_run_id = payload.get("dry_run_id")
    run_dir: Path | None = None
    if _is_safe_id(dry_run_id):
        run_dir = Path(output_dir) / str(dry_run_id)
        if run_dir.exists():
            errors.append("controlled_writer_dry_run_output_dir_not_clean")

    if errors:
        return _blocked_summary(payload=payload, errors=_stable_unique(errors), warnings=_stable_unique(warnings))

    status = "needs_review" if warnings else "passed"
    report = _report(status=status, payload=payload, errors=[], warnings=_stable_unique(warnings))
    report_sha = _sha256_bytes(_json_bytes(report))
    summary = _summary(status=status, payload=payload, report_sha=report_sha, errors=[], warnings=_stable_unique(warnings))
    markdown = _markdown(report, summary)

    if _contains_forbidden_material({"report": report, "summary": summary, "markdown": markdown}):
        return _minimal_redaction_failure()

    if run_dir is None:
        return _blocked_summary(
            payload=payload,
            errors=["dry_run_id_unsafe"],
            warnings=_stable_unique(warnings),
        )
    if run_dir.exists():
        return _blocked_summary(
            payload=payload,
            errors=["controlled_writer_dry_run_output_dir_not_clean"],
            warnings=_stable_unique(warnings),
        )
    run_dir.mkdir(parents=True)

    report_path = run_dir / "property_training_dataset_controlled_writer_dry_run_report.json"
    summary_path = run_dir / "property_training_dataset_controlled_writer_dry_run_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_controlled_writer_dry_run_evidence.md"
    write_json(report_path, report)
    write_json(summary_path, summary)
    markdown_path.write_text(markdown, encoding="utf-8")
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
        summary = run_property_training_dataset_controlled_writer_dry_run(
            controlled_writer_dry_run_input_path=args.controlled_writer_dry_run_input,
            output_dir=args.output_dir,
            require_design_plan_preflight_passed=args.require_design_plan_preflight_passed,
            require_domain_validation_passed=args.require_domain_validation_passed,
            require_value_resolution_precheck_passed=args.require_value_resolution_precheck_passed,
            require_values_resolved=args.require_values_resolved,
            allow_needs_review_candidates=args.allow_needs_review_candidates,
            minimum_accepted_candidate_records=args.minimum_accepted_candidate_records,
        )
    except Exception as exc:
        err.write(f"controlled writer dry-run invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("dry_run_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_dry_run"
        ),
        description="Run the aggregate-only property training dataset controlled writer dry-run.",
    )
    parser.add_argument("--controlled-writer-dry-run-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--require-design-plan-preflight-passed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-domain-validation-passed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-value-resolution-precheck-passed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-values-resolved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-needs-review-candidates", action="store_true")
    parser.add_argument("--minimum-accepted-candidate-records", type=int, default=1)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetControlledWriterDryRunError("json_object_required")
    return payload


def _top_level_errors(payload: dict[str, Any]) -> list[str]:
    return [f"{field}_missing" for field in _TOP_LEVEL_REQUIRED_FIELDS if field not in payload]


def _schema_errors(payload: dict[str, Any]) -> list[str]:
    if payload.get("schema_version") != _INPUT_SCHEMA_VERSION:
        return ["controlled_writer_dry_run_input_schema_invalid"]
    return []


def _safe_id_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _SAFE_ID_FIELDS:
        if field in payload and not _is_safe_id(payload.get(field)):
            errors.append(f"{field}_unsafe")
    return errors


def _status_errors(
    payload: dict[str, Any],
    *,
    require_design_plan_preflight_passed: bool,
    require_domain_validation_passed: bool,
    require_value_resolution_precheck_passed: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    status_rules = (
        (
            "controlled_writer_design_plan_preflight_status",
            require_design_plan_preflight_passed,
        ),
        ("domain_validation_boundary_status", require_domain_validation_passed),
        (
            "controlled_writer_value_resolution_dry_run_precheck_status",
            require_value_resolution_precheck_passed,
        ),
    )
    for field, require_passed in status_rules:
        if field not in payload:
            continue
        status = payload.get(field)
        if status == "passed":
            continue
        if status == "needs_review":
            code = f"{field}_needs_review"
            if require_passed:
                errors.append(code)
            else:
                warnings.append(code)
            continue
        if status in {"blocked", "failed"}:
            errors.append(f"{field}_blocked")
        else:
            errors.append(f"{field}_invalid")
    return errors


def _candidate_count_errors(
    payload: dict[str, Any],
    *,
    minimum_accepted_candidate_records: int,
    allow_needs_review_candidates: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    for field in (
        "accepted_candidate_record_count",
        "needs_review_candidate_record_count",
        "blocked_candidate_record_count",
    ):
        if field not in payload:
            continue
        if not _is_non_negative_int(payload.get(field)):
            errors.append(f"{field}_invalid")
    if errors:
        return errors
    accepted = payload.get("accepted_candidate_record_count")
    needs_review = payload.get("needs_review_candidate_record_count")
    blocked = payload.get("blocked_candidate_record_count")
    if _is_non_negative_int(accepted) and accepted < max(minimum_accepted_candidate_records, 1):
        errors.append("minimum_accepted_candidate_records_not_met")
    if _is_non_negative_int(blocked) and blocked > 0:
        errors.append("blocked_candidate_records_present")
    if _is_non_negative_int(needs_review) and needs_review > 0:
        if allow_needs_review_candidates:
            warnings.append("needs_review_candidate_records_present")
        else:
            errors.append("needs_review_candidate_records_present")
    return errors


def _field_coverage_errors(
    payload: dict[str, Any],
    *,
    require_values_resolved: bool,
    warnings: list[str],
) -> list[str]:
    coverage = payload.get("field_coverage")
    if not isinstance(coverage, dict):
        return ["field_coverage_invalid"]
    errors: list[str] = []
    for field in (
        "required_field_count",
        "resolved_required_field_count",
        "missing_required_field_count",
    ):
        if field not in coverage:
            errors.append(f"{field}_missing")
        elif not _is_non_negative_int(coverage.get(field)):
            errors.append(f"{field}_invalid")
    if errors:
        return errors
    required = coverage["required_field_count"]
    resolved = coverage["resolved_required_field_count"]
    missing = coverage["missing_required_field_count"]
    if required <= 0:
        errors.append("required_field_count_invalid")
    if resolved > required:
        errors.append("resolved_required_field_count_invalid")
    if require_values_resolved:
        if missing > 0 or resolved != required:
            errors.append("missing_required_fields")
    elif missing > 0 or resolved != required:
        warnings.append("missing_required_fields")
    return errors


def _would_write_errors(payload: dict[str, Any]) -> list[str]:
    would_write = payload.get("would_write")
    if not isinstance(would_write, dict):
        return ["would_write_invalid"]
    errors: list[str] = []
    for field in ("would_write_row_count", "would_write_field_count"):
        if field not in would_write:
            errors.append(f"{field}_missing")
        elif not _is_non_negative_int(would_write.get(field)):
            errors.append(f"{field}_invalid")
    if errors:
        return errors
    accepted = payload.get("accepted_candidate_record_count")
    if _is_non_negative_int(accepted) and would_write["would_write_row_count"] < accepted:
        errors.append("would_write_row_count_invalid")
    if would_write["would_write_field_count"] <= 0:
        errors.append("would_write_field_count_invalid")
    for field in _FALSE_WOULD_WRITE_FLAGS:
        if field not in would_write:
            errors.append(f"{field}_missing")
        elif would_write.get(field) is not False:
            errors.append(field)
    return errors


def _boundary_errors(payload: dict[str, Any]) -> list[str]:
    flags = payload.get("boundary_flags")
    if not isinstance(flags, dict):
        return ["boundary_flags_invalid"]
    errors: list[str] = []
    for field in _FALSE_BOUNDARY_FLAGS:
        if field not in flags:
            errors.append(f"{field}_missing")
        elif flags.get(field) is not False:
            errors.append(field)
    if flags.get("phase1_status") != "not_run":
        errors.append("phase1_ran")
    return errors


def _redaction_status_errors(payload: dict[str, Any]) -> list[str]:
    if payload.get("redaction_status") != "passed":
        return ["redaction_status_blocked"]
    return []


def _report(
    *,
    status: str,
    payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    coverage = payload.get("field_coverage") if isinstance(payload.get("field_coverage"), dict) else {}
    would_write = payload.get("would_write") if isinstance(payload.get("would_write"), dict) else {}
    flags = payload.get("boundary_flags") if isinstance(payload.get("boundary_flags"), dict) else {}
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "dry_run_id": payload.get("dry_run_id", ""),
        "dry_run_status": status,
        "corpus_id": payload.get("corpus_id", ""),
        "dataset_name": payload.get("dataset_name", ""),
        "controlled_writer_design_plan_preflight_status": payload.get(
            "controlled_writer_design_plan_preflight_status", ""
        ),
        "domain_validation_boundary_status": payload.get("domain_validation_boundary_status", ""),
        "controlled_writer_value_resolution_dry_run_precheck_status": payload.get(
            "controlled_writer_value_resolution_dry_run_precheck_status", ""
        ),
        "accepted_candidate_record_count": _safe_int(payload.get("accepted_candidate_record_count")),
        "needs_review_candidate_record_count": _safe_int(payload.get("needs_review_candidate_record_count")),
        "blocked_candidate_record_count": _safe_int(payload.get("blocked_candidate_record_count")),
        "required_field_count": _safe_int(coverage.get("required_field_count")),
        "resolved_required_field_count": _safe_int(coverage.get("resolved_required_field_count")),
        "missing_required_field_count": _safe_int(coverage.get("missing_required_field_count")),
        "would_write_row_count": _safe_int(would_write.get("would_write_row_count")),
        "would_write_field_count": _safe_int(would_write.get("would_write_field_count")),
        "would_create_training_dataset_artifact": would_write.get("would_create_training_dataset_artifact")
        is True,
        "would_create_csv_jsonl_parquet_lmdb": would_write.get("would_create_csv_jsonl_parquet_lmdb") is True,
        "would_serialize_rows": would_write.get("would_serialize_rows") is True,
        "would_materialize_values": would_write.get("would_materialize_values") is True,
        "controlled_writer_executed": flags.get("controlled_writer_executed") is True,
        "training_dataset_materialized": flags.get("training_dataset_materialized") is True,
        "dataset_artifact_created": flags.get("dataset_artifact_created") is True,
        "serialized_rows_created": flags.get("serialized_rows_created") is True,
        "phase1_status": flags.get("phase1_status", ""),
        "dataset_confirmation_changed": flags.get("dataset_confirmation_changed") is True,
        "model_training_run": flags.get("model_training_run") is True,
        "evaluation_run": flags.get("evaluation_run") is True,
        "redaction_status": "passed",
        "dry_run_errors": errors,
        "dry_run_warnings": warnings,
    }


def _summary(
    *,
    status: str,
    payload: dict[str, Any],
    report_sha: str,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    coverage = payload.get("field_coverage") if isinstance(payload.get("field_coverage"), dict) else {}
    would_write = payload.get("would_write") if isinstance(payload.get("would_write"), dict) else {}
    flags = payload.get("boundary_flags") if isinstance(payload.get("boundary_flags"), dict) else {}
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "dry_run_id": payload.get("dry_run_id", ""),
        "dry_run_status": status,
        "dry_run_report_basename": "property_training_dataset_controlled_writer_dry_run_report.json",
        "dry_run_report_sha256": report_sha,
        "corpus_id": payload.get("corpus_id", ""),
        "dataset_name": payload.get("dataset_name", ""),
        "accepted_candidate_record_count": _safe_int(payload.get("accepted_candidate_record_count")),
        "needs_review_candidate_record_count": _safe_int(payload.get("needs_review_candidate_record_count")),
        "blocked_candidate_record_count": _safe_int(payload.get("blocked_candidate_record_count")),
        "required_field_count": _safe_int(coverage.get("required_field_count")),
        "resolved_required_field_count": _safe_int(coverage.get("resolved_required_field_count")),
        "missing_required_field_count": _safe_int(coverage.get("missing_required_field_count")),
        "would_write_row_count": _safe_int(would_write.get("would_write_row_count")),
        "would_write_field_count": _safe_int(would_write.get("would_write_field_count")),
        "redaction_status": "passed",
        "controlled_writer_executed": flags.get("controlled_writer_executed") is True,
        "training_dataset_materialized": flags.get("training_dataset_materialized") is True,
        "dataset_artifact_created": flags.get("dataset_artifact_created") is True,
        "serialized_rows_created": flags.get("serialized_rows_created") is True,
        "phase1_status": flags.get("phase1_status", ""),
        "dataset_confirmation_changed": flags.get("dataset_confirmation_changed") is True,
        "model_training_run": flags.get("model_training_run") is True,
        "evaluation_run": flags.get("evaluation_run") is True,
        "dry_run_errors": errors,
        "dry_run_warnings": warnings,
    }


def _blocked_summary(
    *,
    payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    flags = payload.get("boundary_flags") if isinstance(payload.get("boundary_flags"), dict) else {}
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "dry_run_status": "blocked",
        "dry_run_id": _safe_summary_value(payload.get("dry_run_id")),
        "corpus_id": _safe_summary_value(payload.get("corpus_id")),
        "dataset_name": _safe_summary_value(payload.get("dataset_name")),
        "redaction_status": "passed",
        "controlled_writer_executed": flags.get("controlled_writer_executed") is True,
        "training_dataset_materialized": flags.get("training_dataset_materialized") is True,
        "dataset_artifact_created": flags.get("dataset_artifact_created") is True,
        "serialized_rows_created": flags.get("serialized_rows_created") is True,
        "phase1_status": flags.get("phase1_status", ""),
        "dataset_confirmation_changed": flags.get("dataset_confirmation_changed") is True,
        "model_training_run": flags.get("model_training_run") is True,
        "evaluation_run": flags.get("evaluation_run") is True,
        "dry_run_errors": errors,
        "dry_run_warnings": warnings,
    }


def _markdown(report: dict[str, Any], summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Property Training Dataset Controlled Writer Dry-Run Evidence",
            "",
            "This is a controlled writer dry-run only.",
            "",
            f"- dry_run_id: {summary.get('dry_run_id', '')}",
            f"- dry_run_status: {summary.get('dry_run_status', '')}",
            f"- corpus_id: {summary.get('corpus_id', '')}",
            f"- dataset_name: {summary.get('dataset_name', '')}",
            f"- accepted_candidate_record_count: {summary.get('accepted_candidate_record_count', 0)}",
            f"- needs_review_candidate_record_count: {summary.get('needs_review_candidate_record_count', 0)}",
            f"- blocked_candidate_record_count: {summary.get('blocked_candidate_record_count', 0)}",
            f"- required_field_count: {summary.get('required_field_count', 0)}",
            f"- resolved_required_field_count: {summary.get('resolved_required_field_count', 0)}",
            f"- missing_required_field_count: {summary.get('missing_required_field_count', 0)}",
            f"- would_write_row_count: {summary.get('would_write_row_count', 0)}",
            f"- would_write_field_count: {summary.get('would_write_field_count', 0)}",
            f"- dry_run_errors: {', '.join(report.get('dry_run_errors', [])) or 'none'}",
            f"- dry_run_warnings: {', '.join(report.get('dry_run_warnings', [])) or 'none'}",
            "",
            "## Boundary Statement",
            "",
            "This controlled writer dry-run does not execute the controlled writer.",
            "This controlled writer dry-run does not emit raw values.",
            "This controlled writer dry-run does not materialize values.",
            "This controlled writer dry-run does not serialize training rows.",
            "This controlled writer dry-run does not create training dataset artifacts.",
            "This controlled writer dry-run does not create CSV/JSONL/Parquet/LMDB artifacts.",
            "This controlled writer dry-run does not generate conformers.",
            "This controlled writer dry-run does not generate DPA3 structures.",
            "This controlled writer dry-run does not run Phase 1.",
            "This controlled writer dry-run does not modify DatasetConfirmation.",
            "This controlled writer dry-run does not run model training or evaluation.",
            "",
        ]
    )


def _contains_forbidden_material(value: Any) -> bool:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    lowered = text.lower()
    if _ABSOLUTE_PATH_VALUE_RE.search(text):
        return True
    return any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS)


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "dry_run_status": "blocked",
        "dry_run_errors": ["property_training_dataset_controlled_writer_dry_run_redaction_failed"],
        "redaction_status": "failed",
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc)
    if _contains_forbidden_material(message):
        return "redacted"
    return message or exc.__class__.__name__


def _safe_summary_value(value: Any) -> str:
    if _is_safe_id(value):
        return str(value)
    return ""


def _safe_int(value: Any) -> int:
    return value if _is_non_negative_int(value) else 0


def _is_safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and bool(_SAFE_ID_RE.fullmatch(value))


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _append_errors(errors: list[str], new_errors: list[str]) -> None:
    errors.extend(new_errors)


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


if __name__ == "__main__":
    raise SystemExit(main())
