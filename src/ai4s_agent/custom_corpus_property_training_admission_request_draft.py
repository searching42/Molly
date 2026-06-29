from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.custom_corpus_materialization import sha256_file


_DRAFT_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_draft.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_draft_builder.v1"
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


class CustomCorpusPropertyTrainingAdmissionRequestDraftError(ValueError):
    pass


def build_property_training_admission_request_draft(
    *,
    training_admission_request_plan_path: str | Path,
    training_admission_request_preflight_path: str | Path,
    training_admission_readiness_summary_path: str | Path,
    quarantine_candidate_preflight_summary_path: str | Path,
    quarantine_candidate_records_path: str | Path,
    output_dir: str | Path,
    request_draft_id: str,
    created_by: str,
    confirm_training_admission_request_draft_output: bool = False,
    allow_preflight_partial: bool = False,
    minimum_draft_records: int = 1,
) -> dict[str, Any]:
    request_plan = _read_safe_json_dict(training_admission_request_plan_path, "training admission request plan")
    request_preflight = _read_safe_json_dict(training_admission_request_preflight_path, "training admission request preflight")
    readiness = _read_safe_json_dict(training_admission_readiness_summary_path, "training admission readiness summary")
    quarantine_preflight = _read_safe_json_dict(quarantine_candidate_preflight_summary_path, "quarantine candidate preflight summary")
    quarantine_candidate = _read_safe_json_dict(quarantine_candidate_records_path, "quarantine candidate records")
    run_dir = Path(output_dir).expanduser() / request_draft_id
    hashes = {
        "training_admission_request_plan_sha256": _safe_sha_for_path(training_admission_request_plan_path),
        "training_admission_request_preflight_sha256": _safe_sha_for_path(training_admission_request_preflight_path),
        "training_admission_readiness_summary_sha256": _safe_sha_for_path(training_admission_readiness_summary_path),
        "quarantine_candidate_preflight_summary_sha256": _safe_sha_for_path(quarantine_candidate_preflight_summary_path),
        "quarantine_candidate_records_sha256": _safe_sha_for_path(quarantine_candidate_records_path),
    }
    warnings: list[str] = []
    errors = _consistency_errors(
        request_plan=request_plan,
        request_preflight=request_preflight,
        readiness=readiness,
        quarantine_preflight=quarantine_preflight,
        quarantine_candidate=quarantine_candidate,
        hashes=hashes,
        request_draft_id=request_draft_id,
        created_by=created_by,
        run_dir=run_dir,
        confirm_training_admission_request_draft_output=confirm_training_admission_request_draft_output,
        allow_preflight_partial=allow_preflight_partial,
        minimum_draft_records=max(int(minimum_draft_records), 0),
        warnings=warnings,
    )
    draft_status = "blocked" if errors else "needs_review" if warnings else "written"
    draft_records = _draft_records(request_plan, hashes, request_draft_id)
    draft = _draft_payload(
        request_draft_id=request_draft_id,
        created_by=created_by,
        draft_status="needs_review" if draft_status == "needs_review" else "written",
        request_plan=request_plan,
        request_preflight=request_preflight,
        readiness=readiness,
        hashes=hashes,
        draft_records=draft_records,
    )
    paths = {
        "training_admission_request_plan_path": training_admission_request_plan_path,
        "training_admission_request_preflight_path": training_admission_request_preflight_path,
        "training_admission_readiness_summary_path": training_admission_readiness_summary_path,
        "quarantine_candidate_preflight_summary_path": quarantine_candidate_preflight_summary_path,
        "quarantine_candidate_records_path": quarantine_candidate_records_path,
    }
    if draft_status == "blocked":
        return _summary(
            draft_status="blocked",
            request_draft_id=request_draft_id,
            paths=paths,
            hashes=hashes,
            draft_sha256="",
            request_plan=request_plan,
            request_preflight=request_preflight,
            readiness=readiness,
            draft_records=[],
            errors=_stable_unique(errors),
            warnings=_stable_unique(warnings),
        )
    draft_path = run_dir / "property_training_admission_request.draft.json"
    summary_path = run_dir / "property_training_admission_request_draft_summary.json"
    evidence_path = run_dir / "redacted_property_training_admission_request_draft_evidence.md"
    early_summary = _summary(
        draft_status=draft_status,
        request_draft_id=request_draft_id,
        paths=paths,
        hashes=hashes,
        draft_sha256="",
        request_plan=request_plan,
        request_preflight=request_preflight,
        readiness=readiness,
        draft_records=draft_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    evidence = _evidence_markdown(early_summary)
    if _contains_forbidden_material({"draft": draft, "summary": early_summary, "evidence": evidence}):
        return _minimal_redaction_failure()

    write_json(draft_path, draft)
    draft_sha256 = sha256_file(draft_path)
    summary = _summary(
        draft_status=draft_status,
        request_draft_id=request_draft_id,
        paths=paths,
        hashes=hashes,
        draft_sha256=draft_sha256,
        request_plan=request_plan,
        request_preflight=request_preflight,
        readiness=readiness,
        draft_records=draft_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "evidence": evidence}):
        draft_path.unlink(missing_ok=True)
        return _minimal_redaction_failure()
    write_json(summary_path, summary)
    evidence_path.write_text(evidence, encoding="utf-8")
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
        summary = build_property_training_admission_request_draft(
            training_admission_request_plan_path=args.training_admission_request_plan,
            training_admission_request_preflight_path=args.training_admission_request_preflight,
            training_admission_readiness_summary_path=args.training_admission_readiness_summary,
            quarantine_candidate_preflight_summary_path=args.quarantine_candidate_preflight_summary,
            quarantine_candidate_records_path=args.quarantine_candidate_records,
            output_dir=args.output_dir,
            request_draft_id=args.request_draft_id,
            created_by=args.created_by,
            confirm_training_admission_request_draft_output=args.confirm_training_admission_request_draft_output,
            allow_preflight_partial=args.allow_preflight_partial,
            minimum_draft_records=args.minimum_draft_records,
        )
    except Exception as exc:
        err.write(f"property training admission request draft invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("draft_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_admission_request_draft",
        description="Build a reviewable property training admission request draft.",
    )
    parser.add_argument("--training-admission-request-plan", required=True)
    parser.add_argument("--training-admission-request-preflight", required=True)
    parser.add_argument("--training-admission-readiness-summary", required=True)
    parser.add_argument("--quarantine-candidate-preflight-summary", required=True)
    parser.add_argument("--quarantine-candidate-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-draft-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-training-admission-request-draft-output", action="store_true")
    parser.add_argument("--allow-preflight-partial", action="store_true")
    parser.add_argument("--minimum-draft-records", type=int, default=1)
    return parser


def _consistency_errors(
    *,
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    hashes: dict[str, str],
    request_draft_id: str,
    created_by: str,
    run_dir: Path,
    confirm_training_admission_request_draft_output: bool,
    allow_preflight_partial: bool,
    minimum_draft_records: int,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if not confirm_training_admission_request_draft_output:
        errors.append("confirmation_required")
    if not _is_safe_id(request_draft_id):
        errors.append("request_draft_id_invalid")
    if not created_by or "@" in created_by:
        errors.append("created_by_invalid")
    if run_dir.exists() and any(run_dir.iterdir()):
        errors.append("output_directory_not_clean")
    _append_errors(errors, _schema_errors(request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate))
    _append_errors(errors, _status_errors(request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate, allow_preflight_partial, warnings))
    _append_errors(errors, _hash_errors(request_plan, request_preflight, readiness, quarantine_preflight, hashes))
    _append_errors(errors, _id_errors(request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate))
    _append_errors(errors, _record_errors(request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate, minimum_draft_records))
    _append_errors(errors, _sha_format_errors(request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate))
    return _stable_unique(errors)


def _schema_errors(
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
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
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    allow_preflight_partial: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    preflight_status = str(request_preflight.get("preflight_status", ""))
    request_status = str(request_plan.get("planner_status", ""))
    readiness_status = str(readiness.get("readiness_status", ""))
    if preflight_status == "blocked":
        errors.append("training_admission_request_preflight_blocked")
    elif preflight_status == "partial":
        if allow_preflight_partial:
            warnings.append("training_admission_request_preflight_partial")
        else:
            errors.append("training_admission_request_preflight_partial")
    elif preflight_status != "passed":
        errors.append("training_admission_request_preflight_status_invalid")
    if request_status == "blocked":
        errors.append("training_admission_request_plan_blocked")
    elif request_status == "partial":
        if allow_preflight_partial:
            warnings.append("training_admission_request_plan_partial")
        else:
            errors.append("training_admission_request_plan_partial")
    elif request_status != "planned":
        errors.append("training_admission_request_plan_status_invalid")
    if readiness_status == "blocked":
        errors.append("training_admission_readiness_blocked")
    elif readiness_status == "partial":
        if allow_preflight_partial:
            warnings.append("training_admission_readiness_partial")
        else:
            errors.append("training_admission_readiness_partial")
    elif readiness_status != "ready":
        errors.append("training_admission_readiness_status_invalid")
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
    if (
        request_plan.get("training_admitted") is not False
        or request_preflight.get("training_admitted") is not False
        or readiness.get("training_admitted") is not False
        or quarantine_candidate.get("training_admitted") is not False
        or quarantine_preflight.get("training_admitted") is not False
    ):
        errors.append("training_admitted")
    if (
        request_plan.get("phase1_status") != "not_run"
        or request_preflight.get("phase1_status") != "not_run"
        or readiness.get("phase1_status") != "not_run"
        or quarantine_candidate.get("phase1_status") != "not_run"
        or quarantine_preflight.get("phase1_status") != "not_run"
    ):
        errors.append("phase1_ran")
    if (
        request_plan.get("dataset_confirmation_changed") is not False
        or request_preflight.get("dataset_confirmation_changed") is not False
        or readiness.get("dataset_confirmation_changed") is not False
        or quarantine_candidate.get("dataset_confirmation_changed") is not False
        or quarantine_preflight.get("dataset_confirmation_changed") is not False
    ):
        errors.append("dataset_confirmation_changed")
    return errors


def _hash_errors(
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    hashes: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    direct_pairs = (
        (request_preflight, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (request_preflight, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (request_preflight, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (request_plan, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (request_plan, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (request_plan, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (readiness, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (readiness, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
    )
    for container, field, hash_key in direct_pairs:
        if container.get(field) and container.get(field) != hashes[hash_key]:
            errors.append(f"{hash_key}_mismatch")
    for record in _safe_records(request_plan.get("planned_request_record_summaries")):
        if record.get("training_admission_readiness_sha256") != hashes["training_admission_readiness_summary_sha256"]:
            errors.append("training_admission_readiness_summary_sha256_mismatch")
        if record.get("quarantine_candidate_records_sha256") != hashes["quarantine_candidate_records_sha256"]:
            errors.append("quarantine_candidate_records_sha256_mismatch")
    return _stable_unique(errors)


def _id_errors(
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
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
        values = [str(container.get(field, "")) for container in (request_plan, request_preflight, readiness, quarantine_preflight, quarantine_candidate) if container.get(field)]
        if len(set(values)) > 1:
            errors.append(f"{field}_mismatch")
    return errors


def _record_errors(
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    minimum_draft_records: int,
) -> list[str]:
    errors: list[str] = []
    planned_ids = _safe_list(request_plan.get("planned_training_admission_candidate_record_ids"))
    preflight_planned_ids = _safe_list(request_preflight.get("planned_training_admission_candidate_record_ids"))
    candidate_ids = _safe_list(quarantine_candidate.get("candidate_record_ids"))
    plan_candidate_ids = _safe_list(request_plan.get("candidate_record_ids"))
    readiness_candidate_ids = _safe_list(readiness.get("candidate_record_ids"))
    preflight_candidate_ids = _safe_list(quarantine_preflight.get("candidate_record_ids"))
    records = _safe_records(request_plan.get("planned_request_record_summaries"))
    draft_record_ids = [str(record.get("candidate_record_id", "")) for record in records]
    if not planned_ids:
        errors.append("no_planned_candidates")
    if len(planned_ids) < max(minimum_draft_records, 1):
        errors.append("minimum_draft_record_count_not_met")
    if planned_ids != preflight_planned_ids:
        errors.append("planned_candidate_ids_mismatch")
    if planned_ids != draft_record_ids:
        errors.append("planned_candidate_ids_mismatch")
    if not set(planned_ids).issubset(set(candidate_ids)):
        errors.append("planned_candidate_ids_unknown")
    for ids in (plan_candidate_ids, readiness_candidate_ids, preflight_candidate_ids):
        if ids and ids != candidate_ids:
            errors.append("candidate_record_ids_mismatch")
    if int(request_plan.get("planned_candidate_count", len(planned_ids))) != len(planned_ids):
        errors.append("planned_candidate_count_mismatch")
    if int(request_preflight.get("planned_candidate_count", len(planned_ids))) != len(planned_ids):
        errors.append("planned_candidate_count_mismatch")

    excluded = set(_safe_list(request_plan.get("exclude_record_ids")))
    blocked = set(_safe_list(request_plan.get("blocked_record_ids")))
    blocked.update(_safe_list(request_plan.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(readiness.get("blocked_from_training_admission_record_ids")))
    needs_review = set(_safe_list(request_plan.get("needs_review_record_ids")))
    admit = set(_safe_list(request_plan.get("admit_record_ids")))
    for record in records:
        record_id = str(record.get("record_id", ""))
        if record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
        if record_id and record_id not in admit:
            errors.append("planned_candidate_not_admitted")
    return _stable_unique(errors)


def _sha_format_errors(*containers: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for container in containers:
        for key, value in container.items():
            if key.endswith("_sha256") and value and not _SHA_RE.match(str(value)):
                errors.append(f"{key}_invalid")
    return _stable_unique(errors)


def _draft_payload(
    *,
    request_draft_id: str,
    created_by: str,
    draft_status: str,
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    hashes: dict[str, str],
    draft_records: list[dict[str, Any]],
) -> dict[str, Any]:
    draft_record_ids = [record["draft_record_id"] for record in draft_records]
    planned_ids = _safe_list(request_plan.get("planned_training_admission_candidate_record_ids"))
    return {
        "schema_version": _DRAFT_SCHEMA_VERSION,
        "request_draft_id": request_draft_id,
        "created_at": now_iso(),
        "created_by": created_by,
        "draft_status": draft_status,
        "request_mode": "draft_only",
        "training_admitted": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "corpus_id": str(request_plan.get("corpus_id", "")),
        "source_dry_run_id": str(request_plan.get("source_dry_run_id", "")),
        "review_manifest_id": str(request_plan.get("review_manifest_id", "")),
        "admission_request_id": str(request_plan.get("admission_request_id", "")),
        "materialization_plan_id": str(request_plan.get("materialization_plan_id", "")),
        "execution_request_id": str(request_plan.get("execution_request_id", "")),
        "quarantine_run_id": str(request_plan.get("quarantine_run_id", "")),
        "review_queue_id": str(request_plan.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(request_plan.get("property_candidate_manifest_id", "")),
        "dataset_target": str(request_plan.get("dataset_target", "")),
        "source_training_admission_request_plan_sha256": hashes["training_admission_request_plan_sha256"],
        "source_training_admission_request_preflight_sha256": hashes["training_admission_request_preflight_sha256"],
        "source_training_admission_readiness_sha256": hashes["training_admission_readiness_summary_sha256"],
        "source_quarantine_candidate_preflight_sha256": hashes["quarantine_candidate_preflight_summary_sha256"],
        "source_quarantine_candidate_records_sha256": hashes["quarantine_candidate_records_sha256"],
        "draft_record_count": len(draft_records),
        "draft_record_ids": draft_record_ids,
        "planned_training_admission_candidate_record_ids": planned_ids,
        "draft_records": draft_records,
        "boundary_statement": [
            "draft only",
            "no training data admission",
            "no training artifact creation",
            "no Phase 1",
            "DatasetConfirmation unchanged",
        ],
    }


def _draft_records(
    request_plan: dict[str, Any],
    hashes: dict[str, str],
    request_draft_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _safe_records(request_plan.get("planned_request_record_summaries")):
        candidate_id = str(record.get("candidate_record_id", ""))
        records.append(
            {
                "draft_record_id": f"{request_draft_id}-{candidate_id}",
                "candidate_record_id": candidate_id,
                "record_id": str(record.get("record_id", "")),
                "materialization_record_id": str(record.get("materialization_record_id", "")),
                "execution_record_id": str(record.get("execution_record_id", "")),
                "admission_record_id": str(record.get("admission_record_id", "")),
                "review_id": str(record.get("review_id", "")),
                "document_id": str(record.get("document_id", "")),
                "field_name": str(record.get("field_name", "")),
                "requested_action": "request_training_admission",
                "request_status": "drafted",
                "source_artifact_sha256": str(record.get("source_artifact_sha256", "")),
                "review_artifact_sha256": str(record.get("review_artifact_sha256", "")),
                "admission_request_sha256": str(record.get("admission_request_sha256", "")),
                "package_validation_sha256": str(record.get("package_validation_sha256", "")),
                "materialization_plan_sha256": str(record.get("materialization_plan_sha256", "")),
                "quarantine_candidate_records_sha256": str(record.get("quarantine_candidate_records_sha256", "")),
                "training_admission_readiness_sha256": str(record.get("training_admission_readiness_sha256", "")),
                "training_admission_request_plan_sha256": hashes["training_admission_request_plan_sha256"],
                "training_admission_request_preflight_sha256": hashes["training_admission_request_preflight_sha256"],
            }
        )
    return records


def _summary(
    *,
    draft_status: str,
    request_draft_id: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    draft_sha256: str,
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    draft_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    planned_ids = _safe_list(request_plan.get("planned_training_admission_candidate_record_ids"))
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "draft_status": draft_status,
        "request_draft_id": request_draft_id,
        "training_admission_request_draft_path": "property_training_admission_request.draft.json" if draft_sha256 else "",
        "training_admission_request_draft_sha256": draft_sha256,
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
        "corpus_id": str(request_plan.get("corpus_id", "")),
        "source_dry_run_id": str(request_plan.get("source_dry_run_id", "")),
        "review_manifest_id": str(request_plan.get("review_manifest_id", "")),
        "admission_request_id": str(request_plan.get("admission_request_id", "")),
        "materialization_plan_id": str(request_plan.get("materialization_plan_id", "")),
        "execution_request_id": str(request_plan.get("execution_request_id", "")),
        "quarantine_run_id": str(request_plan.get("quarantine_run_id", "")),
        "review_queue_id": str(request_plan.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(request_plan.get("property_candidate_manifest_id", "")),
        "dataset_target": str(request_plan.get("dataset_target", "")),
        "request_plan_status": str(request_plan.get("planner_status", "")),
        "request_preflight_status": str(request_preflight.get("preflight_status", "")),
        "readiness_status": str(readiness.get("readiness_status", "")),
        "training_admitted": request_plan.get("training_admitted"),
        "phase1_status": str(request_plan.get("phase1_status", "")),
        "dataset_confirmation_changed": request_plan.get("dataset_confirmation_changed"),
        "candidate_record_count": len(_safe_list(request_plan.get("candidate_record_ids"))),
        "planned_candidate_count": len(planned_ids),
        "draft_record_count": len(draft_records),
        "draft_record_ids": [record["draft_record_id"] for record in draft_records],
        "planned_training_admission_candidate_record_ids": planned_ids,
        "blocked_from_training_admission_record_ids": _safe_list(request_plan.get("blocked_from_training_admission_record_ids")),
        "draft_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Admission Request Draft Evidence",
            "",
            f"- Draft status: `{summary['draft_status']}`",
            f"- Request draft id: `{summary['request_draft_id']}`",
            f"- Request plan status: `{summary['request_plan_status']}`",
            f"- Request preflight status: `{summary['request_preflight_status']}`",
            f"- Readiness status: `{summary['readiness_status']}`",
            f"- Quarantine run id: `{summary['quarantine_run_id']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Source dry-run id: `{summary['source_dry_run_id']}`",
            f"- Review manifest id: `{summary['review_manifest_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Execution request id: `{summary['execution_request_id']}`",
            f"- Planned candidate count: `{summary['planned_candidate_count']}`",
            f"- Draft record count: `{summary['draft_record_count']}`",
            f"- Draft record ids: `{json.dumps(summary['draft_record_ids'])}`",
            f"- Planned candidate ids: `{json.dumps(summary['planned_training_admission_candidate_record_ids'])}`",
            f"- Draft errors: `{json.dumps(summary['draft_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training admission request draft only.",
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
        raise CustomCorpusPropertyTrainingAdmissionRequestDraftError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyTrainingAdmissionRequestDraftError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyTrainingAdmissionRequestDraftError(f"{label} contains forbidden material")
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
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "draft_status": "blocked",
        "draft_errors": ["property_training_admission_request_draft_redaction_failed"],
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
