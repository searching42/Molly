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


_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_draft_precheck.v1"
_DRAFT_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_draft.v1"
_DRAFT_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_draft_builder.v1"
_REQUEST_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_plan.v1"
_REQUEST_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_preflight.v1"
_READINESS_SCHEMA_VERSION = "custom_corpus_property_training_admission_readiness.v1"
_QUARANTINE_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_quarantine_candidate_preflight.v1"
_QUARANTINE_CANDIDATE_SCHEMA_VERSION = "custom_corpus_property_quarantine_materialization.v1"
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
)


class CustomCorpusPropertyTrainingAdmissionRequestDraftPrecheckError(ValueError):
    pass


def precheck_property_training_admission_request_draft_package(
    *,
    training_admission_request_draft_path: str | Path,
    training_admission_request_draft_summary_path: str | Path,
    training_admission_request_plan_path: str | Path,
    training_admission_request_preflight_path: str | Path,
    training_admission_readiness_summary_path: str | Path,
    quarantine_candidate_preflight_summary_path: str | Path,
    quarantine_candidate_records_path: str | Path,
    output_summary_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    require_draft_written: bool = True,
    allow_draft_needs_review: bool = False,
    minimum_draft_records: int = 1,
) -> dict[str, Any]:
    draft = _read_safe_json_dict(training_admission_request_draft_path, "training admission request draft")
    draft_summary = _read_safe_json_dict(
        training_admission_request_draft_summary_path,
        "training admission request draft summary",
    )
    request_plan = _read_safe_json_dict(training_admission_request_plan_path, "training admission request plan")
    request_preflight = _read_safe_json_dict(
        training_admission_request_preflight_path,
        "training admission request preflight",
    )
    readiness = _read_safe_json_dict(
        training_admission_readiness_summary_path,
        "training admission readiness summary",
    )
    quarantine_preflight = _read_safe_json_dict(
        quarantine_candidate_preflight_summary_path,
        "quarantine candidate preflight summary",
    )
    quarantine_candidate = _read_safe_json_dict(quarantine_candidate_records_path, "quarantine candidate records")
    paths = {
        "training_admission_request_draft_path": training_admission_request_draft_path,
        "training_admission_request_draft_summary_path": training_admission_request_draft_summary_path,
        "training_admission_request_plan_path": training_admission_request_plan_path,
        "training_admission_request_preflight_path": training_admission_request_preflight_path,
        "training_admission_readiness_summary_path": training_admission_readiness_summary_path,
        "quarantine_candidate_preflight_summary_path": quarantine_candidate_preflight_summary_path,
        "quarantine_candidate_records_path": quarantine_candidate_records_path,
    }
    hashes = {
        "training_admission_request_draft_sha256": _safe_sha_for_path(training_admission_request_draft_path),
        "training_admission_request_draft_summary_sha256": _safe_sha_for_path(training_admission_request_draft_summary_path),
        "training_admission_request_plan_sha256": _safe_sha_for_path(training_admission_request_plan_path),
        "training_admission_request_preflight_sha256": _safe_sha_for_path(training_admission_request_preflight_path),
        "training_admission_readiness_summary_sha256": _safe_sha_for_path(training_admission_readiness_summary_path),
        "quarantine_candidate_preflight_summary_sha256": _safe_sha_for_path(quarantine_candidate_preflight_summary_path),
        "quarantine_candidate_records_sha256": _safe_sha_for_path(quarantine_candidate_records_path),
    }
    warnings: list[str] = []
    errors = _consistency_errors(
        draft=draft,
        draft_summary=draft_summary,
        request_plan=request_plan,
        request_preflight=request_preflight,
        readiness=readiness,
        quarantine_preflight=quarantine_preflight,
        quarantine_candidate=quarantine_candidate,
        hashes=hashes,
        require_draft_written=require_draft_written,
        allow_draft_needs_review=allow_draft_needs_review,
        minimum_draft_records=max(int(minimum_draft_records), 0),
        warnings=warnings,
    )
    status = "blocked" if errors else "needs_review" if warnings else "passed"
    summary = _summary(
        precheck_status=status,
        paths=paths,
        hashes=hashes,
        draft=draft,
        draft_summary=draft_summary,
        request_plan=request_plan,
        request_preflight=request_preflight,
        readiness=readiness,
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
        summary = precheck_property_training_admission_request_draft_package(
            training_admission_request_draft_path=args.training_admission_request_draft,
            training_admission_request_draft_summary_path=args.training_admission_request_draft_summary,
            training_admission_request_plan_path=args.training_admission_request_plan,
            training_admission_request_preflight_path=args.training_admission_request_preflight,
            training_admission_readiness_summary_path=args.training_admission_readiness_summary,
            quarantine_candidate_preflight_summary_path=args.quarantine_candidate_preflight_summary,
            quarantine_candidate_records_path=args.quarantine_candidate_records,
            output_summary_path=args.output_summary or None,
            output_markdown_path=args.output_markdown or None,
            require_draft_written=args.require_draft_written,
            allow_draft_needs_review=args.allow_draft_needs_review,
            minimum_draft_records=args.minimum_draft_records,
        )
    except Exception as exc:
        err.write(f"property training admission request draft precheck invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("precheck_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_admission_request_draft_precheck",
        description="Precheck a property training admission request draft package.",
    )
    parser.add_argument("--training-admission-request-draft", required=True)
    parser.add_argument("--training-admission-request-draft-summary", required=True)
    parser.add_argument("--training-admission-request-plan", required=True)
    parser.add_argument("--training-admission-request-preflight", required=True)
    parser.add_argument("--training-admission-readiness-summary", required=True)
    parser.add_argument("--quarantine-candidate-preflight-summary", required=True)
    parser.add_argument("--quarantine-candidate-records", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-markdown", default="")
    parser.add_argument("--require-draft-written", action="store_true", default=True)
    parser.add_argument("--allow-draft-needs-review", action="store_true")
    parser.add_argument("--minimum-draft-records", type=int, default=1)
    return parser


def _consistency_errors(
    *,
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    hashes: dict[str, str],
    require_draft_written: bool,
    allow_draft_needs_review: bool,
    minimum_draft_records: int,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    _append_errors(errors, _schema_errors(draft, draft_summary, request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate))
    _append_errors(errors, _status_errors(draft, draft_summary, request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate, require_draft_written, allow_draft_needs_review, warnings))
    _append_errors(errors, _hash_errors(draft, draft_summary, request_plan, request_preflight, readiness, hashes))
    _append_errors(errors, _id_errors(draft, draft_summary, request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate))
    _append_errors(errors, _record_errors(draft, draft_summary, request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate, minimum_draft_records))
    _append_errors(errors, _sha_format_errors(draft, draft_summary, request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate))
    return _stable_unique(errors)


def _schema_errors(
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if draft.get("schema_version") != _DRAFT_SCHEMA_VERSION:
        errors.append("training_admission_request_draft_schema_invalid")
    if draft_summary.get("schema_version") != _DRAFT_SUMMARY_SCHEMA_VERSION:
        errors.append("training_admission_request_draft_summary_schema_invalid")
    if request_plan.get("schema_version") != _REQUEST_PLAN_SCHEMA_VERSION:
        errors.append("training_admission_request_plan_schema_invalid")
    if request_preflight.get("schema_version") != _REQUEST_PREFLIGHT_SCHEMA_VERSION:
        errors.append("training_admission_request_preflight_schema_invalid")
    if readiness.get("schema_version") != _READINESS_SCHEMA_VERSION:
        errors.append("training_admission_readiness_schema_invalid")
    if quarantine_preflight.get("schema_version") != _QUARANTINE_PREFLIGHT_SCHEMA_VERSION:
        errors.append("quarantine_candidate_preflight_schema_invalid")
    if quarantine_candidate.get("schema_version") != _QUARANTINE_CANDIDATE_SCHEMA_VERSION:
        errors.append("quarantine_candidate_schema_invalid")
    return errors


def _status_errors(
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    require_draft_written: bool,
    allow_draft_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    _append_errors(errors, _draft_status_errors(draft, "training_admission_request_draft", require_draft_written, allow_draft_needs_review, warnings))
    _append_errors(errors, _draft_status_errors(draft_summary, "training_admission_request_draft_summary", require_draft_written, allow_draft_needs_review, warnings))
    _append_errors(errors, _planned_status_errors(request_plan, "planner_status", "training_admission_request_plan", "planned", allow_draft_needs_review, warnings))
    _append_errors(errors, _planned_status_errors(request_preflight, "preflight_status", "training_admission_request_preflight", "passed", allow_draft_needs_review, warnings))
    _append_errors(errors, _planned_status_errors(readiness, "readiness_status", "training_admission_readiness", "ready", allow_draft_needs_review, warnings))

    quarantine_status = str(quarantine_preflight.get("preflight_status", ""))
    if quarantine_status in {"failed", "blocked"}:
        errors.append("quarantine_candidate_preflight_failed")
    elif quarantine_status in {"needs_review", "partial"}:
        if allow_draft_needs_review:
            warnings.append("quarantine_candidate_preflight_needs_review")
        else:
            errors.append("quarantine_candidate_preflight_needs_review")
    elif quarantine_status != "passed":
        errors.append("quarantine_candidate_preflight_status_invalid")

    if draft.get("request_mode") != "draft_only":
        errors.append("training_admission_request_draft_mode_invalid")
    if draft_summary.get("draft_errors"):
        errors.append("training_admission_request_draft_summary_has_errors")
    if request_preflight.get("preflight_errors"):
        errors.append("training_admission_request_preflight_has_errors")
    if request_plan.get("planning_errors"):
        errors.append("training_admission_request_plan_has_errors")
    if readiness.get("readiness_errors"):
        errors.append("training_admission_readiness_has_errors")
    if quarantine_preflight.get("preflight_errors"):
        errors.append("quarantine_candidate_preflight_has_errors")
    if quarantine_candidate.get("materialization_mode") != "candidate_quarantine":
        errors.append("quarantine_materialization_mode_invalid")

    if any(container.get("training_admitted") is not False for container in (draft, draft_summary, request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate)):
        errors.append("training_admitted")
    if any(container.get("phase1_status") != "not_run" for container in (draft, draft_summary, request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate)):
        errors.append("phase1_ran")
    if any(container.get("dataset_confirmation_changed") is not False for container in (draft, draft_summary, request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate)):
        errors.append("dataset_confirmation_changed")
    return _stable_unique(errors)


def _draft_status_errors(
    container: dict[str, Any],
    prefix: str,
    require_draft_written: bool,
    allow_draft_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    status = str(container.get("draft_status", ""))
    if status == "written":
        return []
    if status == "needs_review":
        if allow_draft_needs_review:
            warnings.append(f"{prefix}_needs_review")
            return []
        return [f"{prefix}_needs_review"]
    if status == "blocked":
        return [f"{prefix}_blocked"]
    if require_draft_written:
        return [f"{prefix}_status_invalid"]
    return []


def _planned_status_errors(
    container: dict[str, Any],
    field: str,
    prefix: str,
    required: str,
    allow_draft_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    status = str(container.get(field, ""))
    if status == required:
        return []
    if status in {"partial", "needs_review"}:
        if allow_draft_needs_review:
            warnings.append(f"{prefix}_{status}")
            return []
        return [f"{prefix}_{status}"]
    if status == "blocked":
        return [f"{prefix}_blocked"]
    return [f"{prefix}_status_invalid"]


def _hash_errors(
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    hashes: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    direct_pairs = (
        (draft_summary, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (draft, "source_training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (draft_summary, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (request_preflight, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (draft, "source_training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (draft_summary, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (draft, "source_training_admission_readiness_sha256", "training_admission_readiness_summary_sha256"),
        (draft_summary, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (request_plan, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (request_preflight, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (draft, "source_quarantine_candidate_preflight_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (draft_summary, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (request_plan, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (request_preflight, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (readiness, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (draft, "source_quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (draft_summary, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (request_plan, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (readiness, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
    )
    for container, field, hash_key in direct_pairs:
        if container.get(field) and container.get(field) != hashes[hash_key]:
            errors.append(f"{hash_key}_mismatch")
    for record in _safe_records(draft.get("draft_records")):
        if record.get("training_admission_request_plan_sha256") != hashes["training_admission_request_plan_sha256"]:
            errors.append("training_admission_request_plan_sha256_mismatch")
        if record.get("training_admission_request_preflight_sha256") != hashes["training_admission_request_preflight_sha256"]:
            errors.append("training_admission_request_preflight_sha256_mismatch")
        if record.get("training_admission_readiness_sha256") != hashes["training_admission_readiness_summary_sha256"]:
            errors.append("training_admission_readiness_summary_sha256_mismatch")
        if record.get("quarantine_candidate_records_sha256") != hashes["quarantine_candidate_records_sha256"]:
            errors.append("quarantine_candidate_records_sha256_mismatch")
    for record in _safe_records(request_plan.get("planned_request_record_summaries")):
        if record.get("training_admission_readiness_sha256") != hashes["training_admission_readiness_summary_sha256"]:
            errors.append("training_admission_readiness_summary_sha256_mismatch")
        if record.get("quarantine_candidate_records_sha256") != hashes["quarantine_candidate_records_sha256"]:
            errors.append("quarantine_candidate_records_sha256_mismatch")
    return _stable_unique(errors)


def _id_errors(*containers: dict[str, Any]) -> list[str]:
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
        "dataset_target",
    )
    for field in id_fields:
        values = [str(container.get(field, "")) for container in containers if container.get(field)]
        if len(set(values)) > 1:
            errors.append(f"{field}_mismatch")
    request_draft_ids = [str(container.get("request_draft_id", "")) for container in containers if container.get("request_draft_id")]
    if len(set(request_draft_ids)) > 1:
        errors.append("request_draft_id_mismatch")
    for field in id_fields[:-1] + ("request_draft_id",):
        for container in containers:
            value = str(container.get(field, ""))
            if value and not _is_safe_id(value):
                errors.append(f"{field}_invalid")
    return _stable_unique(errors)


def _record_errors(
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    minimum_draft_records: int,
) -> list[str]:
    errors: list[str] = []
    draft_records = _safe_records(draft.get("draft_records"))
    draft_record_ids = [str(record.get("draft_record_id", "")) for record in draft_records]
    draft_candidate_ids = [str(record.get("candidate_record_id", "")) for record in draft_records]
    top_draft_ids = _safe_list(draft.get("draft_record_ids"))
    planned_ids = _safe_list(draft.get("planned_training_admission_candidate_record_ids"))
    summary_planned_ids = _safe_list(draft_summary.get("planned_training_admission_candidate_record_ids"))
    plan_planned_ids = _safe_list(request_plan.get("planned_training_admission_candidate_record_ids"))
    preflight_planned_ids = _safe_list(request_preflight.get("planned_training_admission_candidate_record_ids"))
    readiness_planned_ids = _safe_list(readiness.get("planned_training_admission_candidate_record_ids"))
    quarantine_candidate_ids = _safe_list(quarantine_candidate.get("candidate_record_ids"))

    if not draft_records:
        errors.append("no_draft_records")
    if not planned_ids:
        errors.append("no_planned_candidates")
    if len(draft_records) < max(minimum_draft_records, 1):
        errors.append("minimum_draft_record_count_not_met")
    if int(draft.get("draft_record_count", len(draft_records))) != len(draft_records):
        errors.append("draft_record_count_mismatch")
    if int(draft_summary.get("draft_record_count", len(draft_records))) != len(draft_records):
        errors.append("draft_record_count_mismatch")
    if top_draft_ids != draft_record_ids:
        errors.append("draft_record_ids_mismatch")
    if _safe_list(draft_summary.get("draft_record_ids")) != draft_record_ids:
        errors.append("draft_record_ids_mismatch")
    if set(planned_ids) != set(draft_candidate_ids):
        errors.append("planned_candidate_not_in_draft")
    for ids in (summary_planned_ids, plan_planned_ids, preflight_planned_ids):
        if ids != planned_ids:
            errors.append("planned_candidate_ids_mismatch")
    if readiness_planned_ids and readiness_planned_ids != planned_ids:
        errors.append("planned_candidate_ids_mismatch")
    if not set(planned_ids).issubset(set(quarantine_candidate_ids)):
        errors.append("planned_candidate_ids_unknown")
    if int(draft_summary.get("planned_candidate_count", len(planned_ids))) != len(planned_ids):
        errors.append("planned_candidate_count_mismatch")
    if int(request_plan.get("planned_candidate_count", len(plan_planned_ids))) != len(plan_planned_ids):
        errors.append("planned_candidate_count_mismatch")
    if int(request_preflight.get("planned_candidate_count", len(preflight_planned_ids))) != len(preflight_planned_ids):
        errors.append("planned_candidate_count_mismatch")

    excluded = set(_safe_list(request_plan.get("exclude_record_ids")))
    excluded.update(_safe_list(request_preflight.get("exclude_record_ids")))
    blocked = set(_safe_list(request_plan.get("blocked_record_ids")))
    blocked.update(_safe_list(request_plan.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(request_preflight.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(readiness.get("blocked_from_training_admission_record_ids")))
    needs_review = set(_safe_list(request_plan.get("needs_review_record_ids")))
    needs_review.update(_safe_list(request_preflight.get("needs_review_record_ids")))
    for record in draft_records:
        draft_record_id = str(record.get("draft_record_id", ""))
        candidate_id = str(record.get("candidate_record_id", ""))
        record_id = str(record.get("record_id", ""))
        if not draft_record_id or not _is_safe_id(draft_record_id):
            errors.append("draft_record_id_invalid")
        if candidate_id in excluded or record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if candidate_id in blocked or record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if candidate_id in needs_review or record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
        if not _draft_record_values_are_safe(record):
            errors.append("draft_record_contains_unsafe_value")
    return _stable_unique(errors)


def _sha_format_errors(*containers: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for container in containers:
        for key, value in container.items():
            if key.endswith("_sha256") and value and not _SHA_RE.match(str(value)):
                errors.append(f"{key}_invalid")
    return _stable_unique(errors)


def _summary(
    *,
    precheck_status: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    planned_ids = _safe_list(draft.get("planned_training_admission_candidate_record_ids"))
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": precheck_status,
        "training_admission_request_draft_path": _basename(paths["training_admission_request_draft_path"], "property_training_admission_request.draft.json"),
        "training_admission_request_draft_sha256": hashes["training_admission_request_draft_sha256"],
        "training_admission_request_draft_summary_path": _basename(paths["training_admission_request_draft_summary_path"], "property_training_admission_request_draft_summary.json"),
        "training_admission_request_draft_summary_sha256": hashes["training_admission_request_draft_summary_sha256"],
        "training_admission_request_plan_path": _basename(paths["training_admission_request_plan_path"], "property_training_admission_request_plan_summary.json"),
        "training_admission_request_plan_sha256": hashes["training_admission_request_plan_sha256"],
        "training_admission_request_preflight_path": _basename(paths["training_admission_request_preflight_path"], "property_training_admission_request_preflight_summary.json"),
        "training_admission_request_preflight_sha256": hashes["training_admission_request_preflight_sha256"],
        "training_admission_readiness_summary_path": _basename(paths["training_admission_readiness_summary_path"], "property_training_admission_readiness_summary.json"),
        "training_admission_readiness_summary_sha256": hashes["training_admission_readiness_summary_sha256"],
        "quarantine_candidate_preflight_summary_path": _basename(paths["quarantine_candidate_preflight_summary_path"], "property_quarantine_candidate_preflight_summary.json"),
        "quarantine_candidate_preflight_summary_sha256": hashes["quarantine_candidate_preflight_summary_sha256"],
        "quarantine_candidate_records_path": _basename(paths["quarantine_candidate_records_path"], "property_quarantine_candidate_records.json"),
        "quarantine_candidate_records_sha256": hashes["quarantine_candidate_records_sha256"],
        "request_draft_id": str(draft.get("request_draft_id", draft_summary.get("request_draft_id", ""))),
        "corpus_id": str(draft.get("corpus_id", request_plan.get("corpus_id", ""))),
        "source_dry_run_id": str(draft.get("source_dry_run_id", request_plan.get("source_dry_run_id", ""))),
        "review_manifest_id": str(draft.get("review_manifest_id", request_plan.get("review_manifest_id", ""))),
        "admission_request_id": str(draft.get("admission_request_id", request_plan.get("admission_request_id", ""))),
        "materialization_plan_id": str(draft.get("materialization_plan_id", request_plan.get("materialization_plan_id", ""))),
        "execution_request_id": str(draft.get("execution_request_id", request_plan.get("execution_request_id", ""))),
        "quarantine_run_id": str(draft.get("quarantine_run_id", request_plan.get("quarantine_run_id", ""))),
        "review_queue_id": str(draft.get("review_queue_id", request_plan.get("review_queue_id", ""))),
        "property_candidate_manifest_id": str(draft.get("property_candidate_manifest_id", request_plan.get("property_candidate_manifest_id", ""))),
        "dataset_target": str(draft.get("dataset_target", request_plan.get("dataset_target", ""))),
        "draft_status": str(draft.get("draft_status", draft_summary.get("draft_status", ""))),
        "request_plan_status": str(request_plan.get("planner_status", "")),
        "request_preflight_status": str(request_preflight.get("preflight_status", "")),
        "readiness_status": str(readiness.get("readiness_status", "")),
        "training_admitted": draft.get("training_admitted"),
        "phase1_status": str(draft.get("phase1_status", "")),
        "dataset_confirmation_changed": draft.get("dataset_confirmation_changed"),
        "draft_record_count": len(_safe_records(draft.get("draft_records"))),
        "planned_candidate_count": len(planned_ids),
        "draft_record_ids": _safe_list(draft.get("draft_record_ids")),
        "planned_training_admission_candidate_record_ids": planned_ids,
        "blocked_from_training_admission_record_ids": _safe_list(request_plan.get("blocked_from_training_admission_record_ids")),
        "precheck_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Admission Request Draft Package Precheck Evidence",
            "",
            f"- Precheck status: `{summary['precheck_status']}`",
            f"- Request draft id: `{summary['request_draft_id']}`",
            f"- Draft status: `{summary['draft_status']}`",
            f"- Request plan status: `{summary['request_plan_status']}`",
            f"- Request preflight status: `{summary['request_preflight_status']}`",
            f"- Readiness status: `{summary['readiness_status']}`",
            f"- Quarantine run id: `{summary['quarantine_run_id']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Source dry-run id: `{summary['source_dry_run_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Execution request id: `{summary['execution_request_id']}`",
            f"- Planned candidate count: `{summary['planned_candidate_count']}`",
            f"- Draft record count: `{summary['draft_record_count']}`",
            f"- Draft record ids: `{json.dumps(summary['draft_record_ids'])}`",
            f"- Planned candidate ids: `{json.dumps(summary['planned_training_admission_candidate_record_ids'])}`",
            f"- Precheck errors: `{json.dumps(summary['precheck_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training admission request draft package precheck only.",
            "- no training admission was executed.",
            "- no training data was admitted.",
            "- no training CSV/JSONL/Parquet/LMDB was created.",
            "- no candidate CSV/JSONL/Parquet/LMDB was created.",
            "- no Phase 1 was run.",
            "- DatasetConfirmation was not changed.",
            "- no model training or evaluation was run.",
            "",
        ]
    )


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyTrainingAdmissionRequestDraftPrecheckError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyTrainingAdmissionRequestDraftPrecheckError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyTrainingAdmissionRequestDraftPrecheckError(f"{label} contains forbidden material")
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


def _is_safe_id(value: str) -> bool:
    return bool(_SAFE_ID_RE.match(value))


def _draft_record_values_are_safe(record: dict[str, Any]) -> bool:
    for key, value in record.items():
        if key.endswith("_sha256"):
            if value and not _SHA_RE.match(str(value)):
                return False
            continue
        if isinstance(value, str) and key != "requested_action":
            if value and not _is_safe_id(value):
                return False
    return not _contains_forbidden_material(record)


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
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_admission_request_draft_precheck_redaction_failed"],
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
