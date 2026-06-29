from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.custom_corpus_materialization import sha256_file


_LEDGER_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger_summary.v1"
_DRY_RUN_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_dry_run_precheck.v1"
_DRY_RUN_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_dry_run.v1"
_REQUEST_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_request.v1"
_REQUEST_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_request_builder.v1"
_EXECUTION_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_request_preflight.v1"
_DRAFT_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_draft.v1"
_DRAFT_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_draft_builder.v1"
_DRAFT_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_admission_request_draft_precheck.v1"
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


class CustomCorpusPropertyTrainingAdmissionExecutionLedgerError(ValueError):
    pass


def build_property_training_admission_execution_ledger(
    *,
    training_admission_execution_dry_run_precheck_path: str | Path,
    training_admission_execution_dry_run_report_path: str | Path,
    training_admission_execution_request_path: str | Path,
    training_admission_execution_request_summary_path: str | Path,
    training_admission_execution_request_preflight_path: str | Path,
    training_admission_request_draft_path: str | Path,
    training_admission_request_draft_summary_path: str | Path,
    training_admission_request_draft_precheck_path: str | Path,
    training_admission_request_plan_path: str | Path,
    training_admission_request_preflight_path: str | Path,
    training_admission_readiness_summary_path: str | Path,
    quarantine_candidate_preflight_summary_path: str | Path,
    quarantine_candidate_records_path: str | Path,
    output_dir: str | Path,
    execution_ledger_id: str,
    created_by: str,
    confirm_training_admission_ledger_write: bool = False,
    allow_dry_run_precheck_needs_review: bool = False,
    minimum_ledger_records: int = 1,
) -> dict[str, Any]:
    dry_run_precheck = _read_safe_json_dict(
        training_admission_execution_dry_run_precheck_path,
        "training admission execution dry-run precheck",
    )
    dry_run = _read_safe_json_dict(training_admission_execution_dry_run_report_path, "training admission execution dry-run report")
    request = _read_safe_json_dict(training_admission_execution_request_path, "training admission execution request")
    request_summary = _read_safe_json_dict(
        training_admission_execution_request_summary_path,
        "training admission execution request summary",
    )
    execution_preflight = _read_safe_json_dict(
        training_admission_execution_request_preflight_path,
        "training admission execution request preflight",
    )
    draft = _read_safe_json_dict(training_admission_request_draft_path, "training admission request draft")
    draft_summary = _read_safe_json_dict(training_admission_request_draft_summary_path, "training admission request draft summary")
    draft_precheck = _read_safe_json_dict(
        training_admission_request_draft_precheck_path,
        "training admission request draft precheck",
    )
    request_plan = _read_safe_json_dict(training_admission_request_plan_path, "training admission request plan")
    request_preflight = _read_safe_json_dict(training_admission_request_preflight_path, "training admission request preflight")
    readiness = _read_safe_json_dict(training_admission_readiness_summary_path, "training admission readiness summary")
    quarantine_preflight = _read_safe_json_dict(
        quarantine_candidate_preflight_summary_path,
        "quarantine candidate preflight summary",
    )
    quarantine_candidate = _read_safe_json_dict(quarantine_candidate_records_path, "quarantine candidate records")
    run_dir = Path(output_dir).expanduser() / execution_ledger_id
    paths = {
        "training_admission_execution_dry_run_precheck_path": training_admission_execution_dry_run_precheck_path,
        "training_admission_execution_dry_run_report_path": training_admission_execution_dry_run_report_path,
        "training_admission_execution_request_path": training_admission_execution_request_path,
        "training_admission_execution_request_summary_path": training_admission_execution_request_summary_path,
        "training_admission_execution_request_preflight_path": training_admission_execution_request_preflight_path,
        "training_admission_request_draft_path": training_admission_request_draft_path,
        "training_admission_request_draft_summary_path": training_admission_request_draft_summary_path,
        "training_admission_request_draft_precheck_path": training_admission_request_draft_precheck_path,
        "training_admission_request_plan_path": training_admission_request_plan_path,
        "training_admission_request_preflight_path": training_admission_request_preflight_path,
        "training_admission_readiness_summary_path": training_admission_readiness_summary_path,
        "quarantine_candidate_preflight_summary_path": quarantine_candidate_preflight_summary_path,
        "quarantine_candidate_records_path": quarantine_candidate_records_path,
    }
    hashes = {
        "training_admission_execution_dry_run_precheck_sha256": _safe_sha_for_path(
            training_admission_execution_dry_run_precheck_path
        ),
        "training_admission_execution_dry_run_report_sha256": _safe_sha_for_path(
            training_admission_execution_dry_run_report_path
        ),
        "training_admission_execution_request_sha256": _safe_sha_for_path(training_admission_execution_request_path),
        "training_admission_execution_request_summary_sha256": _safe_sha_for_path(
            training_admission_execution_request_summary_path
        ),
        "training_admission_execution_request_preflight_sha256": _safe_sha_for_path(
            training_admission_execution_request_preflight_path
        ),
        "training_admission_request_draft_sha256": _safe_sha_for_path(training_admission_request_draft_path),
        "training_admission_request_draft_summary_sha256": _safe_sha_for_path(training_admission_request_draft_summary_path),
        "training_admission_request_draft_precheck_sha256": _safe_sha_for_path(training_admission_request_draft_precheck_path),
        "training_admission_request_plan_sha256": _safe_sha_for_path(training_admission_request_plan_path),
        "training_admission_request_preflight_sha256": _safe_sha_for_path(training_admission_request_preflight_path),
        "training_admission_readiness_summary_sha256": _safe_sha_for_path(training_admission_readiness_summary_path),
        "quarantine_candidate_preflight_summary_sha256": _safe_sha_for_path(quarantine_candidate_preflight_summary_path),
        "quarantine_candidate_records_sha256": _safe_sha_for_path(quarantine_candidate_records_path),
    }
    warnings: list[str] = []
    errors = _consistency_errors(
        dry_run_precheck=dry_run_precheck,
        dry_run=dry_run,
        request=request,
        request_summary=request_summary,
        execution_preflight=execution_preflight,
        draft=draft,
        draft_summary=draft_summary,
        draft_precheck=draft_precheck,
        request_plan=request_plan,
        request_preflight=request_preflight,
        readiness=readiness,
        quarantine_preflight=quarantine_preflight,
        quarantine_candidate=quarantine_candidate,
        hashes=hashes,
        run_dir=run_dir,
        execution_ledger_id=execution_ledger_id,
        created_by=created_by,
        confirm_training_admission_ledger_write=confirm_training_admission_ledger_write,
        allow_dry_run_precheck_needs_review=allow_dry_run_precheck_needs_review,
        minimum_ledger_records=max(int(minimum_ledger_records), 0),
        warnings=warnings,
    )
    execution_status = "blocked" if errors else "needs_review" if warnings else "committed"
    ledger_records = [] if execution_status == "blocked" else _ledger_records(dry_run, hashes, execution_ledger_id)
    ledger_path = run_dir / "property_training_admission_execution_ledger.json"
    summary_path = run_dir / "property_training_admission_execution_ledger_summary.json"
    evidence_path = run_dir / "redacted_property_training_admission_execution_ledger_evidence.md"
    ledger = _ledger(
        execution_status=execution_status,
        execution_ledger_id=execution_ledger_id,
        created_by=created_by,
        paths=paths,
        hashes=hashes,
        dry_run_precheck=dry_run_precheck,
        dry_run=dry_run,
        request=request,
        request_summary=request_summary,
        execution_preflight=execution_preflight,
        draft=draft,
        draft_precheck=draft_precheck,
        request_plan=request_plan,
        request_preflight=request_preflight,
        readiness=readiness,
        ledger_records=ledger_records,
        errors=_stable_unique(errors),
        warnings=_stable_unique(warnings),
    )
    if execution_status == "blocked":
        return _summary(
            execution_status="blocked",
            execution_ledger_id=execution_ledger_id,
            paths=paths,
            hashes=hashes,
            ledger_path=ledger_path,
            ledger_sha256="",
            dry_run_precheck=dry_run_precheck,
            dry_run=dry_run,
            request=request,
            request_summary=request_summary,
            execution_preflight=execution_preflight,
            draft=draft,
            draft_precheck=draft_precheck,
            request_plan=request_plan,
            request_preflight=request_preflight,
            readiness=readiness,
            ledger_records=[],
            errors=_stable_unique(errors),
            warnings=_stable_unique(warnings),
        )

    ledger_sha256 = _sha256_payload(ledger)
    summary = _summary(
        execution_status=execution_status,
        execution_ledger_id=execution_ledger_id,
        paths=paths,
        hashes=hashes,
        ledger_path=ledger_path,
        ledger_sha256=ledger_sha256,
        dry_run_precheck=dry_run_precheck,
        dry_run=dry_run,
        request=request,
        request_summary=request_summary,
        execution_preflight=execution_preflight,
        draft=draft,
        draft_precheck=draft_precheck,
        request_plan=request_plan,
        request_preflight=request_preflight,
        readiness=readiness,
        ledger_records=ledger_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    markdown = _evidence_markdown(summary)
    if _contains_forbidden_material({"ledger": ledger, "summary": summary, "markdown": markdown}):
        return _minimal_redaction_failure()
    write_json(ledger_path, ledger)
    write_json(summary_path, summary)
    evidence_path.write_text(markdown, encoding="utf-8")
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
        summary = build_property_training_admission_execution_ledger(
            training_admission_execution_dry_run_precheck_path=args.training_admission_execution_dry_run_precheck,
            training_admission_execution_dry_run_report_path=args.training_admission_execution_dry_run_report,
            training_admission_execution_request_path=args.training_admission_execution_request,
            training_admission_execution_request_summary_path=args.training_admission_execution_request_summary,
            training_admission_execution_request_preflight_path=args.training_admission_execution_request_preflight,
            training_admission_request_draft_path=args.training_admission_request_draft,
            training_admission_request_draft_summary_path=args.training_admission_request_draft_summary,
            training_admission_request_draft_precheck_path=args.training_admission_request_draft_precheck,
            training_admission_request_plan_path=args.training_admission_request_plan,
            training_admission_request_preflight_path=args.training_admission_request_preflight,
            training_admission_readiness_summary_path=args.training_admission_readiness_summary,
            quarantine_candidate_preflight_summary_path=args.quarantine_candidate_preflight_summary,
            quarantine_candidate_records_path=args.quarantine_candidate_records,
            output_dir=args.output_dir,
            execution_ledger_id=args.execution_ledger_id,
            created_by=args.created_by,
            confirm_training_admission_ledger_write=args.confirm_training_admission_ledger_write,
            allow_dry_run_precheck_needs_review=args.allow_dry_run_precheck_needs_review,
            minimum_ledger_records=args.minimum_ledger_records,
        )
    except Exception as exc:
        err.write(f"property training admission execution ledger invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("execution_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_admission_execution_ledger",
        description="Write a property training admission execution ledger.",
    )
    parser.add_argument("--training-admission-execution-dry-run-precheck", required=True)
    parser.add_argument("--training-admission-execution-dry-run-report", required=True)
    parser.add_argument("--training-admission-execution-request", required=True)
    parser.add_argument("--training-admission-execution-request-summary", required=True)
    parser.add_argument("--training-admission-execution-request-preflight", required=True)
    parser.add_argument("--training-admission-request-draft", required=True)
    parser.add_argument("--training-admission-request-draft-summary", required=True)
    parser.add_argument("--training-admission-request-draft-precheck", required=True)
    parser.add_argument("--training-admission-request-plan", required=True)
    parser.add_argument("--training-admission-request-preflight", required=True)
    parser.add_argument("--training-admission-readiness-summary", required=True)
    parser.add_argument("--quarantine-candidate-preflight-summary", required=True)
    parser.add_argument("--quarantine-candidate-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-ledger-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-training-admission-ledger-write", action="store_true")
    parser.add_argument("--allow-dry-run-precheck-needs-review", action="store_true")
    parser.add_argument("--minimum-ledger-records", type=int, default=1)
    return parser


def _consistency_errors(
    *,
    dry_run_precheck: dict[str, Any],
    dry_run: dict[str, Any],
    request: dict[str, Any],
    request_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    draft_precheck: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    hashes: dict[str, str],
    run_dir: Path,
    execution_ledger_id: str,
    created_by: str,
    confirm_training_admission_ledger_write: bool,
    allow_dry_run_precheck_needs_review: bool,
    minimum_ledger_records: int,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if not confirm_training_admission_ledger_write:
        errors.append("confirmation_required")
    if not _is_safe_id(execution_ledger_id):
        errors.append("execution_ledger_id_invalid")
    if not created_by or "@" in created_by:
        errors.append("created_by_invalid")
    if run_dir.exists() and any(run_dir.iterdir()):
        errors.append("output_directory_not_clean")
    _append_errors(
        errors,
        _schema_errors(
            dry_run_precheck,
            dry_run,
            request,
            request_summary,
            execution_preflight,
            draft,
            draft_summary,
            draft_precheck,
            request_plan,
            request_preflight,
            readiness,
            quarantine_preflight,
            quarantine_candidate,
        ),
    )
    _append_errors(
        errors,
        _status_errors(
            dry_run_precheck,
            dry_run,
            request,
            request_summary,
            execution_preflight,
            draft,
            draft_summary,
            draft_precheck,
            request_plan,
            request_preflight,
            readiness,
            quarantine_preflight,
            quarantine_candidate,
            allow_dry_run_precheck_needs_review,
            warnings,
        ),
    )
    _append_errors(
        errors,
        _hash_errors(
            dry_run_precheck,
            dry_run,
            request,
            request_summary,
            execution_preflight,
            draft,
            draft_summary,
            draft_precheck,
            request_plan,
            request_preflight,
            readiness,
            hashes,
        ),
    )
    _append_errors(
        errors,
        _id_errors(
            dry_run_precheck,
            dry_run,
            request,
            request_summary,
            execution_preflight,
            draft,
            draft_summary,
            draft_precheck,
            request_plan,
            request_preflight,
            readiness,
            quarantine_preflight,
            quarantine_candidate,
        ),
    )
    _append_errors(
        errors,
        _record_errors(
            dry_run_precheck,
            dry_run,
            request,
            request_summary,
            execution_preflight,
            draft,
            draft_summary,
            draft_precheck,
            request_plan,
            request_preflight,
            readiness,
            quarantine_candidate,
            hashes,
            minimum_ledger_records,
        ),
    )
    _append_errors(
        errors,
        _sha_format_errors(
            dry_run_precheck,
            dry_run,
            request,
            request_summary,
            execution_preflight,
            draft,
            draft_summary,
            draft_precheck,
            request_plan,
            request_preflight,
            readiness,
            quarantine_preflight,
            quarantine_candidate,
        ),
    )
    return _stable_unique(errors)


def _schema_errors(*containers: dict[str, Any]) -> list[str]:
    (
        dry_run_precheck,
        dry_run,
        request,
        request_summary,
        execution_preflight,
        draft,
        draft_summary,
        draft_precheck,
        request_plan,
        request_preflight,
        readiness,
        quarantine_preflight,
        quarantine_candidate,
    ) = containers
    errors: list[str] = []
    if dry_run_precheck.get("schema_version") != _DRY_RUN_PREFLIGHT_SCHEMA_VERSION:
        errors.append("training_admission_execution_dry_run_precheck_schema_invalid")
    if dry_run.get("schema_version") != _DRY_RUN_SCHEMA_VERSION:
        errors.append("training_admission_execution_dry_run_schema_invalid")
    if request.get("schema_version") != _REQUEST_SCHEMA_VERSION:
        errors.append("training_admission_execution_request_schema_invalid")
    if request_summary.get("schema_version") != _REQUEST_SUMMARY_SCHEMA_VERSION:
        errors.append("training_admission_execution_request_summary_schema_invalid")
    if execution_preflight.get("schema_version") != _EXECUTION_PREFLIGHT_SCHEMA_VERSION:
        errors.append("training_admission_execution_request_preflight_schema_invalid")
    if draft.get("schema_version") != _DRAFT_SCHEMA_VERSION:
        errors.append("training_admission_request_draft_schema_invalid")
    if draft_summary.get("schema_version") != _DRAFT_SUMMARY_SCHEMA_VERSION:
        errors.append("training_admission_request_draft_summary_schema_invalid")
    if draft_precheck.get("schema_version") != _DRAFT_PREFLIGHT_SCHEMA_VERSION:
        errors.append("training_admission_request_draft_precheck_schema_invalid")
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
    dry_run_precheck: dict[str, Any],
    dry_run: dict[str, Any],
    request: dict[str, Any],
    request_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    draft_precheck: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    allow: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    _append_errors(errors, _status_field_errors(dry_run_precheck, "preflight_status", "training_admission_execution_dry_run_precheck", "passed", allow, warnings))
    _append_errors(errors, _status_field_errors(dry_run, "dry_run_status", "training_admission_execution_dry_run", "passed", allow, warnings))
    _append_errors(errors, _status_field_errors(execution_preflight, "preflight_status", "training_admission_execution_request_preflight", "passed", allow, warnings))
    _append_errors(errors, _status_field_errors(request, "request_status", "training_admission_execution_request", "written", allow, warnings))
    _append_errors(errors, _status_field_errors(request_summary, "request_status", "training_admission_execution_request_summary", "written", allow, warnings))
    _append_errors(errors, _status_field_errors(draft_precheck, "precheck_status", "training_admission_request_draft_precheck", "passed", allow, warnings))
    _append_errors(errors, _status_field_errors(draft, "draft_status", "training_admission_request_draft", "written", allow, warnings))
    _append_errors(errors, _status_field_errors(draft_summary, "draft_status", "training_admission_request_draft_summary", "written", allow, warnings))
    _append_errors(errors, _status_field_errors(request_plan, "planner_status", "training_admission_request_plan", "planned", allow, warnings))
    _append_errors(errors, _status_field_errors(request_preflight, "preflight_status", "training_admission_request_preflight", "passed", allow, warnings))
    _append_errors(errors, _status_field_errors(readiness, "readiness_status", "training_admission_readiness", "ready", allow, warnings))
    quarantine_status = str(quarantine_preflight.get("preflight_status", ""))
    if quarantine_status in {"failed", "blocked"}:
        errors.append("quarantine_candidate_preflight_failed")
    elif quarantine_status in {"needs_review", "partial"}:
        if allow:
            warnings.append("quarantine_candidate_preflight_needs_review")
        else:
            errors.append("quarantine_candidate_preflight_needs_review")
    elif quarantine_status != "passed":
        errors.append("quarantine_candidate_preflight_status_invalid")

    if dry_run_precheck.get("preflight_errors"):
        errors.append("training_admission_execution_dry_run_precheck_has_errors")
    if dry_run.get("dry_run_errors"):
        errors.append("training_admission_execution_dry_run_has_errors")
    if dry_run.get("dry_run_mode") != "execution_simulation_only":
        errors.append("training_admission_execution_dry_run_mode_invalid")
    if request.get("request_mode") != "execution_request_only":
        errors.append("training_admission_execution_request_mode_invalid")
    if draft.get("request_mode") != "draft_only":
        errors.append("training_admission_request_draft_mode_invalid")
    if quarantine_candidate.get("materialization_mode") != "candidate_quarantine":
        errors.append("quarantine_materialization_mode_invalid")
    if request_summary.get("request_errors"):
        errors.append("training_admission_execution_request_summary_has_errors")
    if execution_preflight.get("preflight_errors"):
        errors.append("training_admission_execution_request_preflight_has_errors")
    if draft_summary.get("draft_errors"):
        errors.append("training_admission_request_draft_summary_has_errors")
    if draft_precheck.get("precheck_errors"):
        errors.append("training_admission_request_draft_precheck_has_errors")
    if request_preflight.get("preflight_errors"):
        errors.append("training_admission_request_preflight_has_errors")
    if request_plan.get("planning_errors"):
        errors.append("training_admission_request_plan_has_errors")
    if readiness.get("readiness_errors"):
        errors.append("training_admission_readiness_has_errors")
    if quarantine_preflight.get("preflight_errors"):
        errors.append("quarantine_candidate_preflight_has_errors")

    pre_ledger_containers = (
        dry_run_precheck,
        dry_run,
        request,
        request_summary,
        execution_preflight,
        draft,
        draft_summary,
        draft_precheck,
        request_plan,
        request_preflight,
        readiness,
        quarantine_preflight,
        quarantine_candidate,
    )
    if any(container.get("training_admitted") is not False for container in pre_ledger_containers):
        errors.append("training_admitted_before_ledger")
    if any(container.get("phase1_status") != "not_run" for container in pre_ledger_containers):
        errors.append("phase1_ran")
    if any(container.get("dataset_confirmation_changed") is not False for container in pre_ledger_containers):
        errors.append("dataset_confirmation_changed")
    return _stable_unique(errors)


def _status_field_errors(
    container: dict[str, Any],
    field: str,
    prefix: str,
    required: str,
    allow: bool,
    warnings: list[str],
) -> list[str]:
    status = str(container.get(field, ""))
    if status == required:
        return []
    if status in {"needs_review", "partial"}:
        if allow:
            warnings.append(f"{prefix}_{status}")
            return []
        return [f"{prefix}_{status}"]
    if status == "blocked":
        return [f"{prefix}_blocked"]
    return [f"{prefix}_status_invalid"]


def _hash_errors(
    dry_run_precheck: dict[str, Any],
    dry_run: dict[str, Any],
    request: dict[str, Any],
    request_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    draft_precheck: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    hashes: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    direct_pairs = (
        (dry_run_precheck, "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256"),
        (dry_run_precheck, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (dry_run_precheck, "training_admission_execution_request_summary_sha256", "training_admission_execution_request_summary_sha256"),
        (dry_run_precheck, "training_admission_execution_request_preflight_sha256", "training_admission_execution_request_preflight_sha256"),
        (dry_run_precheck, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (dry_run_precheck, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (dry_run_precheck, "training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (dry_run_precheck, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (dry_run_precheck, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (dry_run_precheck, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (dry_run_precheck, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (dry_run_precheck, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (dry_run, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (dry_run, "training_admission_execution_request_summary_sha256", "training_admission_execution_request_summary_sha256"),
        (dry_run, "training_admission_execution_request_preflight_sha256", "training_admission_execution_request_preflight_sha256"),
        (dry_run, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (dry_run, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (dry_run, "training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (dry_run, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (dry_run, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (dry_run, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (dry_run, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (dry_run, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (request_summary, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (execution_preflight, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (execution_preflight, "training_admission_execution_request_summary_sha256", "training_admission_execution_request_summary_sha256"),
        (request, "source_training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (request_summary, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (execution_preflight, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (draft_summary, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (draft_precheck, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (request, "source_training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (request_summary, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (execution_preflight, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (draft_precheck, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (request, "source_training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (request_summary, "training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (execution_preflight, "training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (request, "source_training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (request_summary, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (execution_preflight, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (draft, "source_training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (draft_summary, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (draft_precheck, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (request_preflight, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (request, "source_training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (request_summary, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (execution_preflight, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (draft, "source_training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (draft_summary, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (draft_precheck, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (request, "source_training_admission_readiness_sha256", "training_admission_readiness_summary_sha256"),
        (request_summary, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (execution_preflight, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (draft, "source_training_admission_readiness_sha256", "training_admission_readiness_summary_sha256"),
        (draft_summary, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (draft_precheck, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (request_plan, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (request_preflight, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (request, "source_quarantine_candidate_preflight_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (request_summary, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (execution_preflight, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (draft, "source_quarantine_candidate_preflight_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (draft_summary, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (draft_precheck, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (request_plan, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (request_preflight, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (readiness, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (request, "source_quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (request_summary, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (execution_preflight, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (draft, "source_quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (draft_summary, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (draft_precheck, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (request_plan, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (readiness, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
    )
    for container, field, hash_key in direct_pairs:
        if container.get(field) and container.get(field) != hashes[hash_key]:
            errors.append(f"{hash_key}_mismatch")
    for record in _safe_records(dry_run.get("dry_run_records")):
        if record.get("training_admission_execution_request_sha256") != hashes["training_admission_execution_request_sha256"]:
            errors.append("training_admission_execution_request_sha256_mismatch")
        if record.get("training_admission_execution_request_preflight_sha256") != hashes[
            "training_admission_execution_request_preflight_sha256"
        ]:
            errors.append("training_admission_execution_request_preflight_sha256_mismatch")
        if record.get("training_admission_request_draft_sha256") != hashes["training_admission_request_draft_sha256"]:
            errors.append("training_admission_request_draft_sha256_mismatch")
        if record.get("training_admission_request_draft_precheck_sha256") != hashes["training_admission_request_draft_precheck_sha256"]:
            errors.append("training_admission_request_draft_precheck_sha256_mismatch")
        if record.get("training_admission_request_plan_sha256") != hashes["training_admission_request_plan_sha256"]:
            errors.append("training_admission_request_plan_sha256_mismatch")
        if record.get("training_admission_request_preflight_sha256") != hashes["training_admission_request_preflight_sha256"]:
            errors.append("training_admission_request_preflight_sha256_mismatch")
        if record.get("training_admission_readiness_sha256") != hashes["training_admission_readiness_summary_sha256"]:
            errors.append("training_admission_readiness_summary_sha256_mismatch")
        if record.get("quarantine_candidate_records_sha256") != hashes["quarantine_candidate_records_sha256"]:
            errors.append("quarantine_candidate_records_sha256_mismatch")
    return _stable_unique(errors)


def _id_errors(*containers: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    logical_fields = {
        "corpus_id": ("corpus_id",),
        "source_dry_run_id": ("source_dry_run_id",),
        "review_manifest_id": ("review_manifest_id",),
        "admission_request_id": ("admission_request_id",),
        "materialization_plan_id": ("materialization_plan_id",),
        "source_execution_request_id": ("source_execution_request_id", "execution_request_id"),
        "quarantine_run_id": ("quarantine_run_id",),
        "review_queue_id": ("review_queue_id",),
        "property_candidate_manifest_id": ("property_candidate_manifest_id",),
        "dataset_target": ("dataset_target",),
    }
    for logical, keys in logical_fields.items():
        values: list[str] = []
        for container in containers:
            for key in keys:
                value = str(container.get(key, ""))
                if value:
                    values.append(value)
                    break
        if len(set(values)) > 1:
            errors.append(f"{logical}_mismatch")
    execution_request_ids = [
        str(container.get("execution_request_id", ""))
        for container in containers[:5]
        if container.get("execution_request_id")
    ]
    if len(set(execution_request_ids)) > 1:
        errors.append("execution_request_id_mismatch")
    for logical, keys in logical_fields.items():
        if logical == "dataset_target":
            continue
        for container in containers:
            for key in keys:
                value = str(container.get(key, ""))
                if value and not _is_safe_id(value):
                    errors.append(f"{logical}_invalid")
    for value in execution_request_ids:
        if value and not _is_safe_id(value):
            errors.append("execution_request_id_invalid")
    return _stable_unique(errors)


def _record_errors(
    dry_run_precheck: dict[str, Any],
    dry_run: dict[str, Any],
    request: dict[str, Any],
    request_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    draft: dict[str, Any],
    draft_summary: dict[str, Any],
    draft_precheck: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    hashes: dict[str, str],
    minimum_ledger_records: int,
) -> list[str]:
    errors: list[str] = []
    dry_run_records = _safe_records(dry_run.get("dry_run_records"))
    dry_run_record_ids = [str(record.get("dry_run_record_id", "")) for record in dry_run_records]
    execution_records = _safe_records(request.get("execution_records"))
    execution_record_ids = [str(record.get("execution_record_id", "")) for record in execution_records]
    execution_candidate_ids = [str(record.get("candidate_record_id", "")) for record in execution_records]
    request_planned_ids = _safe_list(request.get("planned_training_admission_candidate_record_ids"))
    dry_run_planned_ids = _safe_list(dry_run.get("planned_training_admission_candidate_record_ids"))
    draft_records = _safe_records(draft.get("draft_records"))
    draft_record_ids = [str(record.get("draft_record_id", "")) for record in draft_records]
    draft_candidate_ids = [str(record.get("candidate_record_id", "")) for record in draft_records]
    quarantine_candidate_ids = _safe_list(quarantine_candidate.get("candidate_record_ids"))
    if not dry_run_records:
        errors.append("no_dry_run_records")
    if len(dry_run_records) < max(minimum_ledger_records, 1):
        errors.append("minimum_ledger_record_count_not_met")
    if not execution_records:
        errors.append("no_execution_records")
    if not request_planned_ids or not dry_run_planned_ids:
        errors.append("no_planned_candidates")
    if int(dry_run.get("dry_run_record_count", len(dry_run_records))) != len(dry_run_records):
        errors.append("dry_run_record_count_mismatch")
    if int(dry_run_precheck.get("dry_run_record_count", len(dry_run_records))) != len(dry_run_records):
        errors.append("dry_run_record_count_mismatch")
    if int(dry_run.get("execution_record_count", len(execution_records))) != len(execution_records):
        errors.append("execution_record_count_mismatch")
    if int(dry_run_precheck.get("execution_record_count", len(execution_records))) != len(execution_records):
        errors.append("execution_record_count_mismatch")
    if int(dry_run.get("planned_candidate_count", len(request_planned_ids))) != len(request_planned_ids):
        errors.append("planned_candidate_count_mismatch")
    if _safe_list(dry_run.get("dry_run_record_ids")) != dry_run_record_ids:
        errors.append("dry_run_record_ids_mismatch")
    if _safe_list(dry_run_precheck.get("dry_run_record_ids")) != dry_run_record_ids:
        errors.append("dry_run_record_ids_mismatch")
    if _safe_list(dry_run.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(dry_run_precheck.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if dry_run_planned_ids != request_planned_ids:
        errors.append("planned_candidate_ids_mismatch")
    if _safe_list(dry_run_precheck.get("planned_training_admission_candidate_record_ids")) != request_planned_ids:
        errors.append("planned_candidate_ids_mismatch")
    if int(request.get("execution_record_count", len(execution_records))) != len(execution_records):
        errors.append("execution_record_count_mismatch")
    if int(request_summary.get("execution_record_count", len(execution_records))) != len(execution_records):
        errors.append("execution_record_count_mismatch")
    if int(execution_preflight.get("execution_record_count", len(execution_records))) != len(execution_records):
        errors.append("execution_record_count_mismatch")
    if _safe_list(request.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(request_summary.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(execution_preflight.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if set(request_planned_ids) != set(execution_candidate_ids):
        errors.append("planned_candidate_not_in_execution_request")
    if set(request_planned_ids) != set(draft_candidate_ids):
        errors.append("planned_candidate_not_in_draft")
    if len(execution_records) != len(draft_records):
        errors.append("execution_record_count_mismatch")
    for ids in (
        _safe_list(request_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(execution_preflight.get("planned_training_admission_candidate_record_ids")),
        _safe_list(draft.get("planned_training_admission_candidate_record_ids")),
        _safe_list(draft_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(draft_precheck.get("planned_training_admission_candidate_record_ids")),
        _safe_list(request_plan.get("planned_training_admission_candidate_record_ids")),
        _safe_list(request_preflight.get("planned_training_admission_candidate_record_ids")),
    ):
        if ids != request_planned_ids:
            errors.append("planned_candidate_ids_mismatch")
    readiness_planned = _safe_list(readiness.get("planned_training_admission_candidate_record_ids"))
    if readiness_planned and readiness_planned != request_planned_ids:
        errors.append("planned_candidate_ids_mismatch")
    if not set(request_planned_ids).issubset(set(quarantine_candidate_ids)):
        errors.append("planned_candidate_ids_unknown")
    if int(draft.get("draft_record_count", len(draft_records))) != len(draft_records):
        errors.append("draft_record_count_mismatch")
    if int(execution_preflight.get("draft_record_count", len(draft_records))) != len(draft_records):
        errors.append("draft_record_count_mismatch")
    if _safe_list(draft.get("draft_record_ids")) != draft_record_ids:
        errors.append("draft_record_ids_mismatch")
    excluded = set(_safe_list(request_plan.get("exclude_record_ids")))
    excluded.update(_safe_list(request_preflight.get("exclude_record_ids")))
    blocked = set(_safe_list(request_plan.get("blocked_record_ids")))
    blocked.update(_safe_list(request_plan.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(request_preflight.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(readiness.get("blocked_from_training_admission_record_ids")))
    needs_review = set(_safe_list(request_plan.get("needs_review_record_ids")))
    needs_review.update(_safe_list(request_preflight.get("needs_review_record_ids")))
    execution_record_id_set = set(execution_record_ids)
    draft_record_id_set = set(draft_record_ids)
    for record in dry_run_records:
        dry_run_record_id = str(record.get("dry_run_record_id", ""))
        execution_record_id = str(record.get("execution_record_id", ""))
        draft_record_id = str(record.get("draft_record_id", ""))
        candidate_id = str(record.get("candidate_record_id", ""))
        record_id = str(record.get("record_id", ""))
        if not dry_run_record_id or not _is_safe_id(dry_run_record_id):
            errors.append("dry_run_record_id_invalid")
        if execution_record_id not in execution_record_id_set:
            errors.append("dry_run_execution_record_id_unknown")
        if draft_record_id not in draft_record_id_set:
            errors.append("dry_run_draft_record_id_unknown")
        if candidate_id not in request_planned_ids:
            errors.append("dry_run_candidate_id_unknown")
        if candidate_id in excluded or record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if candidate_id in blocked or record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if candidate_id in needs_review or record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
        if record.get("would_execute_action") != "would_admit_training_candidate":
            errors.append("dry_run_record_action_invalid")
        if record.get("dry_run_record_status") != "would_admit":
            errors.append("dry_run_record_status_invalid")
        if record.get("training_admitted") is not False:
            errors.append("training_admitted_before_ledger")
        if record.get("phase1_status") != "not_run":
            errors.append("phase1_ran")
        if record.get("dataset_confirmation_changed") is not False:
            errors.append("dataset_confirmation_changed")
        if record.get("training_admission_execution_request_sha256") != hashes["training_admission_execution_request_sha256"]:
            errors.append("training_admission_execution_request_sha256_mismatch")
        if record.get("training_admission_execution_request_preflight_sha256") != hashes[
            "training_admission_execution_request_preflight_sha256"
        ]:
            errors.append("training_admission_execution_request_preflight_sha256_mismatch")
        if not _record_values_are_safe(record, allowed_labels={"would_execute_action", "dry_run_record_status", "phase1_status"}):
            errors.append("dry_run_record_contains_unsafe_value")
    return _stable_unique(errors)


def _ledger_records(dry_run: dict[str, Any], hashes: dict[str, str], execution_ledger_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _safe_records(dry_run.get("dry_run_records")):
        dry_run_record_id = str(record.get("dry_run_record_id", ""))
        records.append(
            {
                "ledger_record_id": f"{execution_ledger_id}-{dry_run_record_id}",
                "dry_run_record_id": dry_run_record_id,
                "execution_record_id": str(record.get("execution_record_id", "")),
                "draft_record_id": str(record.get("draft_record_id", "")),
                "candidate_record_id": str(record.get("candidate_record_id", "")),
                "record_id": str(record.get("record_id", "")),
                "materialization_record_id": str(record.get("materialization_record_id", "")),
                "execution_record_id_from_materializer": str(record.get("execution_record_id_from_materializer", "")),
                "admission_record_id": str(record.get("admission_record_id", "")),
                "review_id": str(record.get("review_id", "")),
                "document_id": str(record.get("document_id", "")),
                "field_name": str(record.get("field_name", "")),
                "admission_action": "admit_training_candidate",
                "ledger_record_status": "admitted_to_training_ledger",
                "training_admitted": True,
                "phase1_status": "not_run",
                "dataset_confirmation_changed": False,
                "source_artifact_sha256": str(record.get("source_artifact_sha256", "")),
                "review_artifact_sha256": str(record.get("review_artifact_sha256", "")),
                "admission_request_sha256": str(record.get("admission_request_sha256", "")),
                "package_validation_sha256": str(record.get("package_validation_sha256", "")),
                "materialization_plan_sha256": str(record.get("materialization_plan_sha256", "")),
                "quarantine_candidate_records_sha256": str(record.get("quarantine_candidate_records_sha256", "")),
                "training_admission_readiness_sha256": str(record.get("training_admission_readiness_sha256", "")),
                "training_admission_request_plan_sha256": str(record.get("training_admission_request_plan_sha256", "")),
                "training_admission_request_preflight_sha256": str(record.get("training_admission_request_preflight_sha256", "")),
                "training_admission_request_draft_sha256": str(record.get("training_admission_request_draft_sha256", "")),
                "training_admission_request_draft_precheck_sha256": str(record.get("training_admission_request_draft_precheck_sha256", "")),
                "training_admission_execution_request_sha256": str(record.get("training_admission_execution_request_sha256", "")),
                "training_admission_execution_request_preflight_sha256": str(
                    record.get("training_admission_execution_request_preflight_sha256", "")
                ),
                "training_admission_execution_dry_run_sha256": hashes["training_admission_execution_dry_run_report_sha256"],
                "training_admission_execution_dry_run_precheck_sha256": hashes[
                    "training_admission_execution_dry_run_precheck_sha256"
                ],
            }
        )
    return records


def _ledger(
    *,
    execution_status: str,
    execution_ledger_id: str,
    created_by: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    dry_run_precheck: dict[str, Any],
    dry_run: dict[str, Any],
    request: dict[str, Any],
    request_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    draft: dict[str, Any],
    draft_precheck: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    ledger_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    execution_records = _safe_records(request.get("execution_records"))
    planned_ids = _safe_list(request.get("planned_training_admission_candidate_record_ids"))
    return {
        "schema_version": _LEDGER_SCHEMA_VERSION,
        "execution_ledger_id": execution_ledger_id,
        "created_at": now_iso(),
        "created_by": created_by,
        "execution_status": execution_status,
        "execution_mode": "training_admission_ledger_only",
        "training_admitted": True,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        **_source_fields(paths, hashes),
        **_id_status_fields(dry_run_precheck, dry_run, request, request_summary, execution_preflight, draft, draft_precheck, request_plan, request_preflight, readiness),
        "ledger_record_count": len(ledger_records),
        "dry_run_record_count": len(_safe_records(dry_run.get("dry_run_records"))),
        "execution_record_count": len(execution_records),
        "draft_record_count": len(_safe_records(draft.get("draft_records"))),
        "planned_candidate_count": len(planned_ids),
        "ledger_record_ids": [record["ledger_record_id"] for record in ledger_records],
        "dry_run_record_ids": _safe_list(dry_run.get("dry_run_record_ids")),
        "execution_record_ids": [str(record.get("execution_record_id", "")) for record in execution_records],
        "planned_training_admission_candidate_record_ids": planned_ids,
        "ledger_records": ledger_records,
        "execution_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
        "boundary_statement": [
            "training admission ledger only",
            "no training dataset artifact",
            "no training CSV/JSONL/Parquet/LMDB",
            "no Phase 1",
            "DatasetConfirmation unchanged",
            "no model training/evaluation",
        ],
    }


def _summary(
    *,
    execution_status: str,
    execution_ledger_id: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    ledger_path: Path,
    ledger_sha256: str,
    dry_run_precheck: dict[str, Any],
    dry_run: dict[str, Any],
    request: dict[str, Any],
    request_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    draft: dict[str, Any],
    draft_precheck: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
    ledger_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    execution_records = _safe_records(request.get("execution_records"))
    planned_ids = _safe_list(request.get("planned_training_admission_candidate_record_ids"))
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "execution_status": execution_status,
        "execution_ledger_id": execution_ledger_id,
        "training_admission_execution_ledger_path": ledger_path.name,
        "training_admission_execution_ledger_sha256": ledger_sha256,
        **_source_fields(paths, hashes),
        **_id_status_fields(dry_run_precheck, dry_run, request, request_summary, execution_preflight, draft, draft_precheck, request_plan, request_preflight, readiness),
        "training_admitted": True if execution_status in {"committed", "needs_review"} else False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "ledger_record_count": len(ledger_records),
        "dry_run_record_count": len(_safe_records(dry_run.get("dry_run_records"))),
        "execution_record_count": len(execution_records),
        "draft_record_count": len(_safe_records(draft.get("draft_records"))),
        "planned_candidate_count": len(planned_ids),
        "ledger_record_ids": [record["ledger_record_id"] for record in ledger_records],
        "dry_run_record_ids": _safe_list(dry_run.get("dry_run_record_ids")),
        "execution_record_ids": [str(record.get("execution_record_id", "")) for record in execution_records],
        "planned_training_admission_candidate_record_ids": planned_ids,
        "execution_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _source_fields(paths: dict[str, str | Path], hashes: dict[str, str]) -> dict[str, str]:
    return {
        "training_admission_execution_dry_run_precheck_path": _basename(
            paths["training_admission_execution_dry_run_precheck_path"],
            "property_training_admission_execution_dry_run_precheck_summary.json",
        ),
        "training_admission_execution_dry_run_precheck_sha256": hashes["training_admission_execution_dry_run_precheck_sha256"],
        "training_admission_execution_dry_run_report_path": _basename(
            paths["training_admission_execution_dry_run_report_path"],
            "property_training_admission_execution_dry_run_report.json",
        ),
        "training_admission_execution_dry_run_report_sha256": hashes["training_admission_execution_dry_run_report_sha256"],
        "training_admission_execution_request_path": _basename(
            paths["training_admission_execution_request_path"],
            "property_training_admission_execution_request.json",
        ),
        "training_admission_execution_request_sha256": hashes["training_admission_execution_request_sha256"],
        "training_admission_execution_request_summary_path": _basename(
            paths["training_admission_execution_request_summary_path"],
            "property_training_admission_execution_request_summary.json",
        ),
        "training_admission_execution_request_summary_sha256": hashes["training_admission_execution_request_summary_sha256"],
        "training_admission_execution_request_preflight_path": _basename(
            paths["training_admission_execution_request_preflight_path"],
            "property_training_admission_execution_request_preflight_summary.json",
        ),
        "training_admission_execution_request_preflight_sha256": hashes["training_admission_execution_request_preflight_sha256"],
        "training_admission_request_draft_path": _basename(paths["training_admission_request_draft_path"], "property_training_admission_request.draft.json"),
        "training_admission_request_draft_sha256": hashes["training_admission_request_draft_sha256"],
        "training_admission_request_draft_summary_path": _basename(
            paths["training_admission_request_draft_summary_path"],
            "property_training_admission_request_draft_summary.json",
        ),
        "training_admission_request_draft_summary_sha256": hashes["training_admission_request_draft_summary_sha256"],
        "training_admission_request_draft_precheck_path": _basename(
            paths["training_admission_request_draft_precheck_path"],
            "property_training_admission_request_draft_precheck_summary.json",
        ),
        "training_admission_request_draft_precheck_sha256": hashes["training_admission_request_draft_precheck_sha256"],
        "training_admission_request_plan_path": _basename(
            paths["training_admission_request_plan_path"],
            "property_training_admission_request_plan_summary.json",
        ),
        "training_admission_request_plan_sha256": hashes["training_admission_request_plan_sha256"],
        "training_admission_request_preflight_path": _basename(
            paths["training_admission_request_preflight_path"],
            "property_training_admission_request_preflight_summary.json",
        ),
        "training_admission_request_preflight_sha256": hashes["training_admission_request_preflight_sha256"],
        "training_admission_readiness_summary_path": _basename(
            paths["training_admission_readiness_summary_path"],
            "property_training_admission_readiness_summary.json",
        ),
        "training_admission_readiness_summary_sha256": hashes["training_admission_readiness_summary_sha256"],
        "quarantine_candidate_preflight_summary_path": _basename(
            paths["quarantine_candidate_preflight_summary_path"],
            "property_quarantine_candidate_preflight_summary.json",
        ),
        "quarantine_candidate_preflight_summary_sha256": hashes["quarantine_candidate_preflight_summary_sha256"],
        "quarantine_candidate_records_path": _basename(paths["quarantine_candidate_records_path"], "property_quarantine_candidate_records.json"),
        "quarantine_candidate_records_sha256": hashes["quarantine_candidate_records_sha256"],
    }


def _id_status_fields(
    dry_run_precheck: dict[str, Any],
    dry_run: dict[str, Any],
    request: dict[str, Any],
    request_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    draft: dict[str, Any],
    draft_precheck: dict[str, Any],
    request_plan: dict[str, Any],
    request_preflight: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, str]:
    return {
        "execution_request_id": str(request.get("execution_request_id", dry_run.get("execution_request_id", ""))),
        "dry_run_id": str(dry_run.get("dry_run_id", "")),
        "corpus_id": str(dry_run.get("corpus_id", request_plan.get("corpus_id", ""))),
        "source_dry_run_id": str(dry_run.get("source_dry_run_id", request_plan.get("source_dry_run_id", ""))),
        "review_manifest_id": str(dry_run.get("review_manifest_id", request_plan.get("review_manifest_id", ""))),
        "admission_request_id": str(dry_run.get("admission_request_id", request_plan.get("admission_request_id", ""))),
        "materialization_plan_id": str(dry_run.get("materialization_plan_id", request_plan.get("materialization_plan_id", ""))),
        "source_execution_request_id": str(dry_run.get("source_execution_request_id", request_plan.get("execution_request_id", ""))),
        "quarantine_run_id": str(dry_run.get("quarantine_run_id", request_plan.get("quarantine_run_id", ""))),
        "review_queue_id": str(dry_run.get("review_queue_id", request_plan.get("review_queue_id", ""))),
        "property_candidate_manifest_id": str(
            dry_run.get("property_candidate_manifest_id", request_plan.get("property_candidate_manifest_id", ""))
        ),
        "dataset_target": str(dry_run.get("dataset_target", request_plan.get("dataset_target", ""))),
        "dry_run_precheck_status": str(dry_run_precheck.get("preflight_status", "")),
        "dry_run_status": str(dry_run.get("dry_run_status", "")),
        "execution_request_status": str(request.get("request_status", request_summary.get("request_status", ""))),
        "execution_request_preflight_status": str(execution_preflight.get("preflight_status", "")),
        "draft_precheck_status": str(draft_precheck.get("precheck_status", "")),
        "draft_status": str(draft.get("draft_status", "")),
        "request_plan_status": str(request_plan.get("planner_status", "")),
        "request_preflight_status": str(request_preflight.get("preflight_status", "")),
        "readiness_status": str(readiness.get("readiness_status", "")),
    }


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Admission Execution Ledger Evidence",
            "",
            f"- Execution status: `{summary['execution_status']}`",
            f"- Execution ledger id: `{summary['execution_ledger_id']}`",
            f"- Dry-run precheck status: `{summary['dry_run_precheck_status']}`",
            f"- Dry-run status: `{summary['dry_run_status']}`",
            f"- Execution request id: `{summary['execution_request_id']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Source dry-run id: `{summary['source_dry_run_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Planned candidate count: `{summary['planned_candidate_count']}`",
            f"- Ledger record count: `{summary['ledger_record_count']}`",
            f"- Ledger record ids: `{json.dumps(summary['ledger_record_ids'])}`",
            f"- Planned candidate ids: `{json.dumps(summary['planned_training_admission_candidate_record_ids'])}`",
            f"- Execution errors: `{json.dumps(summary['execution_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training admission execution ledger only.",
            "- training candidates were admitted to the ledger.",
            "- no training dataset artifact was created.",
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
        raise CustomCorpusPropertyTrainingAdmissionExecutionLedgerError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyTrainingAdmissionExecutionLedgerError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyTrainingAdmissionExecutionLedgerError(f"{label} contains forbidden material")
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


def _sha256_payload(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    return "sha256:" + hashlib.sha256(data).hexdigest()


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


def _record_values_are_safe(record: dict[str, Any], *, allowed_labels: set[str]) -> bool:
    for key, value in record.items():
        if key.endswith("_sha256"):
            if value and not _SHA_RE.match(str(value)):
                return False
            continue
        if isinstance(value, str) and key not in allowed_labels:
            if value and not _is_safe_id(value):
                return False
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return False
    return not bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _sha_format_errors(*containers: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for container in containers:
        for key, value in container.items():
            if key.endswith("_sha256") and value and not _SHA_RE.match(str(value)):
                errors.append(f"{key}_invalid")
    return _stable_unique(errors)


def _contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _input_contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "execution_status": "blocked",
        "execution_errors": ["property_training_admission_execution_ledger_redaction_failed"],
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
