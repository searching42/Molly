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


_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_preflight.v1"
_REQUEST_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_plan.v1"
_READINESS_SCHEMA_VERSION = "custom_corpus_property_training_admission_readiness.v1"
_QUARANTINE_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_quarantine_candidate_preflight.v1"
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
)


class CustomCorpusPropertyTrainingAdmissionRequestPreflightError(ValueError):
    pass


def preflight_property_training_admission_request(
    *,
    training_admission_request_plan_path: str | Path,
    training_admission_readiness_summary_path: str | Path,
    quarantine_candidate_preflight_summary_path: str | Path,
    output_summary_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    request_plan = _read_safe_json_dict(training_admission_request_plan_path, "training admission request plan")
    readiness = _read_safe_json_dict(training_admission_readiness_summary_path, "training admission readiness summary")
    quarantine_preflight = _read_safe_json_dict(quarantine_candidate_preflight_summary_path, "quarantine candidate preflight summary")
    hashes = {
        "training_admission_request_plan_sha256": _safe_sha_for_path(training_admission_request_plan_path),
        "training_admission_readiness_summary_sha256": _safe_sha_for_path(training_admission_readiness_summary_path),
        "quarantine_candidate_preflight_summary_sha256": _safe_sha_for_path(quarantine_candidate_preflight_summary_path),
    }
    warnings: list[str] = []
    errors = _consistency_errors(
        request_plan=request_plan,
        readiness=readiness,
        quarantine_preflight=quarantine_preflight,
        hashes=hashes,
        warnings=warnings,
    )
    status = "blocked" if errors else "partial" if warnings else "passed"
    summary = _summary(
        preflight_status=status,
        request_plan=request_plan,
        readiness=readiness,
        quarantine_preflight=quarantine_preflight,
        hashes=hashes,
        paths={
            "training_admission_request_plan_path": training_admission_request_plan_path,
            "training_admission_readiness_summary_path": training_admission_readiness_summary_path,
            "quarantine_candidate_preflight_summary_path": quarantine_candidate_preflight_summary_path,
        },
        errors=_stable_unique(errors),
        warnings=_stable_unique(warnings),
    )
    markdown = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "markdown": markdown}):
        minimal = _minimal_redaction_failure()
        if output_summary_path is not None:
            write_json(Path(output_summary_path).expanduser(), minimal)
        return minimal
    if output_summary_path is not None:
        write_json(Path(output_summary_path).expanduser(), summary)
    if output_markdown_path is not None:
        Path(output_markdown_path).expanduser().write_text(markdown, encoding="utf-8")
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
        summary = preflight_property_training_admission_request(
            training_admission_request_plan_path=args.training_admission_request_plan,
            training_admission_readiness_summary_path=args.training_admission_readiness_summary,
            quarantine_candidate_preflight_summary_path=args.quarantine_candidate_preflight_summary,
            output_summary_path=args.output_summary or None,
            output_markdown_path=args.output_markdown or None,
        )
    except Exception as exc:
        err.write(f"property training admission request preflight invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("preflight_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_admission_request_preflight",
        description="Preflight a future training admission request plan for property quarantine candidates.",
    )
    parser.add_argument("--training-admission-request-plan", required=True)
    parser.add_argument("--training-admission-readiness-summary", required=True)
    parser.add_argument("--quarantine-candidate-preflight-summary", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-markdown", default="")
    return parser


def _consistency_errors(
    *,
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    hashes: dict[str, str],
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    _append_errors(errors, _schema_errors(request_plan, readiness, quarantine_preflight))
    _append_errors(errors, _status_errors(request_plan, readiness, quarantine_preflight, warnings))
    _append_errors(errors, _hash_errors(request_plan, readiness, quarantine_preflight, hashes))
    _append_errors(errors, _id_errors(request_plan, readiness, quarantine_preflight))
    _append_errors(errors, _record_errors(request_plan, readiness, quarantine_preflight))
    return _stable_unique(errors)


def _schema_errors(
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if request_plan.get("schema_version") != _REQUEST_PLAN_SCHEMA_VERSION:
        errors.append("training_admission_request_plan_schema_invalid")
    if readiness.get("schema_version") != _READINESS_SCHEMA_VERSION:
        errors.append("training_admission_readiness_schema_invalid")
    if quarantine_preflight.get("schema_version") != _QUARANTINE_PREFLIGHT_SCHEMA_VERSION:
        errors.append("quarantine_candidate_preflight_schema_invalid")
    return errors


def _status_errors(
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    request_status = str(request_plan.get("planner_status", ""))
    readiness_status = str(readiness.get("readiness_status", ""))
    quarantine_status = str(quarantine_preflight.get("preflight_status", request_plan.get("quarantine_candidate_preflight_status", "")))

    if request_status == "blocked":
        errors.append("training_admission_request_plan_blocked")
    elif request_status == "partial":
        warnings.append("training_admission_request_plan_partial")
    elif request_status != "planned":
        errors.append("training_admission_request_plan_status_invalid")

    if readiness_status == "blocked":
        errors.append("training_admission_readiness_blocked")
    elif readiness_status == "partial":
        warnings.append("training_admission_readiness_partial")
    elif readiness_status != "ready":
        errors.append("training_admission_readiness_status_invalid")

    if quarantine_status in {"failed", "blocked"}:
        errors.append("quarantine_candidate_preflight_failed")
    elif quarantine_status == "needs_review":
        warnings.append("quarantine_candidate_preflight_needs_review")
    elif quarantine_status != "passed":
        errors.append("quarantine_candidate_preflight_status_invalid")

    if request_plan.get("planning_errors"):
        errors.append("training_admission_request_plan_has_errors")
    if readiness.get("readiness_errors"):
        errors.append("training_admission_readiness_has_errors")
    if quarantine_preflight.get("preflight_errors"):
        errors.append("quarantine_candidate_preflight_has_errors")

    if (
        request_plan.get("training_admitted") is not False
        or readiness.get("training_admitted") is not False
        or quarantine_preflight.get("training_admitted") is not False
    ):
        errors.append("training_admitted")
    if (
        request_plan.get("phase1_status") != "not_run"
        or readiness.get("phase1_status") != "not_run"
        or quarantine_preflight.get("phase1_status") != "not_run"
    ):
        errors.append("phase1_ran")
    if (
        request_plan.get("dataset_confirmation_changed") is not False
        or readiness.get("dataset_confirmation_changed") is not False
        or quarantine_preflight.get("dataset_confirmation_changed") is not False
    ):
        errors.append("dataset_confirmation_changed")
    return errors


def _hash_errors(
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    hashes: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    if request_plan.get("training_admission_readiness_summary_sha256") != hashes["training_admission_readiness_summary_sha256"]:
        errors.append("training_admission_readiness_summary_sha256_mismatch")
    if request_plan.get("quarantine_candidate_preflight_summary_sha256") != hashes["quarantine_candidate_preflight_summary_sha256"]:
        errors.append("quarantine_candidate_preflight_summary_sha256_mismatch")
    if readiness.get("quarantine_candidate_preflight_summary_sha256") != hashes["quarantine_candidate_preflight_summary_sha256"]:
        errors.append("quarantine_candidate_preflight_summary_sha256_mismatch")
    declared_readiness_sha = str(readiness.get("training_admission_readiness_summary_sha256", ""))
    if declared_readiness_sha and declared_readiness_sha != hashes["training_admission_readiness_summary_sha256"]:
        errors.append("training_admission_readiness_summary_sha256_mismatch")

    for record in _safe_records(request_plan.get("planned_request_record_summaries")):
        if record.get("training_admission_readiness_sha256") != hashes["training_admission_readiness_summary_sha256"]:
            errors.append("training_admission_readiness_summary_sha256_mismatch")

    common_hash_fields = (
        "manifest_sha256",
        "dry_run_report_sha256",
        "review_manifest_sha256",
        "admission_request_sha256",
        "formal_package_validation_sha256",
        "property_package_binding_summary_sha256",
        "materialization_plan_sha256",
        "materialization_plan_preflight_summary_sha256",
        "offline_planner_output_sha256",
        "property_planner_summary_sha256",
        "materialization_dry_run_report_sha256",
        "execution_request_sha256",
        "execution_request_summary_sha256",
        "execution_preflight_summary_sha256",
        "quarantine_candidate_records_sha256",
        "quarantine_materializer_summary_sha256",
    )
    for field in common_hash_fields:
        values = [str(container.get(field, "")) for container in (request_plan, readiness, quarantine_preflight) if container.get(field)]
        if len(set(values)) > 1:
            errors.append(f"{field}_mismatch")
    return _stable_unique(errors)


def _id_errors(
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    id_fields = (
        "corpus_id",
        "source_dry_run_id",
        "review_manifest_id",
        "admission_request_id",
        "materialization_plan_id",
        "execution_request_id",
        "quarantine_run_id",
        "review_queue_id",
        "property_candidate_manifest_id",
    )
    for field in id_fields:
        values = [str(container.get(field, "")) for container in (request_plan, readiness, quarantine_preflight) if container.get(field)]
        if len(set(values)) > 1:
            errors.append(f"{field}_mismatch")
    return errors


def _record_errors(
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    candidate_ids = _safe_list(request_plan.get("candidate_record_ids"))
    readiness_candidate_ids = _safe_list(readiness.get("candidate_record_ids"))
    quarantine_candidate_ids = _safe_list(quarantine_preflight.get("candidate_record_ids"))
    planned_ids = _safe_list(request_plan.get("planned_training_admission_candidate_record_ids"))
    readiness_planned_ids = _safe_list(readiness.get("planned_training_admission_candidate_record_ids"))
    blocked_ids = set(_safe_list(request_plan.get("blocked_from_training_admission_record_ids")))
    blocked_ids.update(_safe_list(readiness.get("blocked_from_training_admission_record_ids")))
    blocked_record_ids = set(_safe_list(request_plan.get("blocked_record_ids")))
    exclude_record_ids = set(_safe_list(request_plan.get("exclude_record_ids")))
    admit_record_ids = set(_safe_list(request_plan.get("admit_record_ids")))

    if not candidate_ids:
        errors.append("no_candidate_records")
    if not planned_ids:
        errors.append("no_planned_candidates")
    if candidate_ids != readiness_candidate_ids or candidate_ids != quarantine_candidate_ids:
        errors.append("candidate_record_ids_mismatch")
    if planned_ids != readiness_planned_ids:
        errors.append("planned_candidate_ids_mismatch")
    if not set(planned_ids).issubset(set(candidate_ids)):
        errors.append("planned_candidate_ids_unknown")
    if set(planned_ids).intersection(blocked_ids):
        errors.append("planned_candidate_from_blocked_record")
    for container in (request_plan, readiness, quarantine_preflight):
        count = container.get("candidate_record_count")
        if count is not None and int(count) != len(candidate_ids):
            errors.append("candidate_record_count_mismatch")

    for record in _safe_records(request_plan.get("planned_request_record_summaries")):
        candidate_id = str(record.get("candidate_record_id", ""))
        record_id = str(record.get("record_id", ""))
        if candidate_id and candidate_id not in planned_ids:
            errors.append("planned_candidate_summary_unknown")
        if record_id in exclude_record_ids:
            errors.append("planned_candidate_from_excluded_record")
        if record_id in blocked_record_ids:
            errors.append("planned_candidate_from_blocked_record")
        if record_id and record_id not in admit_record_ids:
            errors.append("planned_candidate_not_admitted")
    return _stable_unique(errors)


def _summary(
    *,
    preflight_status: str,
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    hashes: dict[str, str],
    paths: dict[str, str | Path],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    candidate_ids = _safe_list(request_plan.get("candidate_record_ids"))
    planned_ids = _safe_list(request_plan.get("planned_training_admission_candidate_record_ids"))
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": preflight_status,
        "training_admission_request_plan_path": _basename(paths["training_admission_request_plan_path"], "property_training_admission_request_plan_summary.json"),
        "training_admission_request_plan_sha256": hashes["training_admission_request_plan_sha256"],
        "training_admission_readiness_summary_path": _basename(paths["training_admission_readiness_summary_path"], "property_training_admission_readiness_summary.json"),
        "training_admission_readiness_summary_sha256": hashes["training_admission_readiness_summary_sha256"],
        "quarantine_candidate_preflight_summary_path": _basename(paths["quarantine_candidate_preflight_summary_path"], "property_quarantine_candidate_preflight_summary.json"),
        "quarantine_candidate_preflight_summary_sha256": hashes["quarantine_candidate_preflight_summary_sha256"],
        "request_plan_status": str(request_plan.get("planner_status", "")),
        "readiness_status": str(readiness.get("readiness_status", "")),
        "quarantine_candidate_preflight_status": str(quarantine_preflight.get("preflight_status", request_plan.get("quarantine_candidate_preflight_status", ""))),
        "corpus_id": str(request_plan.get("corpus_id", readiness.get("corpus_id", ""))),
        "source_dry_run_id": str(request_plan.get("source_dry_run_id", readiness.get("source_dry_run_id", ""))),
        "review_manifest_id": str(request_plan.get("review_manifest_id", readiness.get("review_manifest_id", ""))),
        "admission_request_id": str(request_plan.get("admission_request_id", readiness.get("admission_request_id", ""))),
        "materialization_plan_id": str(request_plan.get("materialization_plan_id", readiness.get("materialization_plan_id", ""))),
        "execution_request_id": str(request_plan.get("execution_request_id", readiness.get("execution_request_id", ""))),
        "quarantine_run_id": str(request_plan.get("quarantine_run_id", readiness.get("quarantine_run_id", ""))),
        "review_queue_id": str(request_plan.get("review_queue_id", readiness.get("review_queue_id", ""))),
        "property_candidate_manifest_id": str(request_plan.get("property_candidate_manifest_id", readiness.get("property_candidate_manifest_id", ""))),
        "training_admitted": request_plan.get("training_admitted"),
        "phase1_status": str(request_plan.get("phase1_status", "")),
        "dataset_confirmation_changed": request_plan.get("dataset_confirmation_changed"),
        "candidate_record_count": len(candidate_ids),
        "planned_candidate_count": len(planned_ids),
        "candidate_record_ids": candidate_ids,
        "planned_training_admission_candidate_record_ids": planned_ids,
        "blocked_from_training_admission_record_ids": _safe_list(readiness.get("blocked_from_training_admission_record_ids")),
        "preflight_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Admission Request Preflight Evidence",
            "",
            f"- Preflight status: `{summary['preflight_status']}`",
            f"- Request plan status: `{summary['request_plan_status']}`",
            f"- Readiness status: `{summary['readiness_status']}`",
            f"- Quarantine candidate preflight status: `{summary['quarantine_candidate_preflight_status']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Source dry-run id: `{summary['source_dry_run_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Quarantine run id: `{summary['quarantine_run_id']}`",
            f"- Candidate record count: `{summary['candidate_record_count']}`",
            f"- Planned candidate ids: `{json.dumps(summary['planned_training_admission_candidate_record_ids'])}`",
            f"- Preflight errors: `{json.dumps(summary['preflight_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- no training admission executed.",
            "- no training data created.",
            "- no dataset materialization.",
            "- no Phase 1 execution.",
            "- no DatasetConfirmation change.",
            "- no model training or evaluation.",
            "",
        ]
    )


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyTrainingAdmissionRequestPreflightError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyTrainingAdmissionRequestPreflightError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyTrainingAdmissionRequestPreflightError(f"{label} contains forbidden material")
    return payload


def _safe_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def _safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _safe_sha_for_path(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except Exception:
        return ""


def _basename(path: str | Path, fallback: str) -> str:
    return Path(path).name or fallback


def _append_errors(errors: list[str], additions: list[str]) -> None:
    for error in additions:
        if error and error not in errors:
            errors.append(error)


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _input_contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    allowed_input_markers = {".csv", ".jsonl", ".parquet", ".lmdb"}
    markers = tuple(marker for marker in _FORBIDDEN_MARKERS if marker not in allowed_input_markers)
    if any(marker.lower() in lowered for marker in markers):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": "blocked",
        "preflight_errors": ["property_training_admission_request_preflight_redaction_failed"],
        "redaction_status": "failed",
    }


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    for marker in _FORBIDDEN_MARKERS:
        if marker.lower() in lowered:
            return "invalid input contained forbidden material"
    if _ABSOLUTE_PATH_VALUE_RE.search(json.dumps(message)):
        return "invalid input"
    return message or "invalid input"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
