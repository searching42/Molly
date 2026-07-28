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


_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger_precheck.v1"
_LEDGER_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger.v1"
_LEDGER_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger_summary.v1"
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


class CustomCorpusPropertyTrainingAdmissionExecutionLedgerPrecheckError(ValueError):
    pass


def precheck_property_training_admission_execution_ledger_package(
    *,
    training_admission_execution_ledger_path: str | Path,
    training_admission_execution_ledger_summary_path: str | Path,
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
    output_summary_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    require_ledger_committed: bool = True,
    allow_ledger_needs_review: bool = False,
    minimum_ledger_records: int = 1,
) -> dict[str, Any]:
    ledger = _read_safe_json_dict(training_admission_execution_ledger_path, "training admission execution ledger")
    ledger_summary = _read_safe_json_dict(
        training_admission_execution_ledger_summary_path,
        "training admission execution ledger summary",
    )
    dry_run_precheck = _read_safe_json_dict(
        training_admission_execution_dry_run_precheck_path,
        "training admission execution dry-run precheck",
    )
    dry_run = _read_safe_json_dict(
        training_admission_execution_dry_run_report_path,
        "training admission execution dry-run report",
    )
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
    draft_summary = _read_safe_json_dict(
        training_admission_request_draft_summary_path,
        "training admission request draft summary",
    )
    draft_precheck = _read_safe_json_dict(
        training_admission_request_draft_precheck_path,
        "training admission request draft precheck",
    )
    request_plan = _read_safe_json_dict(training_admission_request_plan_path, "training admission request plan")
    request_preflight = _read_safe_json_dict(
        training_admission_request_preflight_path,
        "training admission request preflight",
    )
    readiness = _read_safe_json_dict(training_admission_readiness_summary_path, "training admission readiness summary")
    quarantine_preflight = _read_safe_json_dict(
        quarantine_candidate_preflight_summary_path,
        "quarantine candidate preflight summary",
    )
    quarantine_candidate = _read_safe_json_dict(quarantine_candidate_records_path, "quarantine candidate records")
    paths = {
        "training_admission_execution_ledger_path": training_admission_execution_ledger_path,
        "training_admission_execution_ledger_summary_path": training_admission_execution_ledger_summary_path,
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
    hashes = {key.replace("_path", "_sha256"): _safe_sha_for_path(path) for key, path in paths.items()}
    warnings: list[str] = []
    errors = _consistency_errors(
        ledger=ledger,
        ledger_summary=ledger_summary,
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
        require_ledger_committed=require_ledger_committed,
        allow_ledger_needs_review=allow_ledger_needs_review,
        minimum_ledger_records=max(int(minimum_ledger_records), 0),
        warnings=warnings,
    )
    precheck_status = "blocked" if errors else "needs_review" if warnings else "passed"
    summary = _summary(
        precheck_status=precheck_status,
        paths=paths,
        hashes=hashes,
        ledger=ledger,
        ledger_summary=ledger_summary,
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
        precheck_errors=_stable_unique(errors),
        warnings=_stable_unique(warnings),
    )
    markdown = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "markdown": markdown}):
        failure = _minimal_redaction_failure()
        if output_summary_path:
            write_json(Path(output_summary_path), failure)
        return failure
    if output_summary_path:
        write_json(Path(output_summary_path), summary)
    if output_markdown_path:
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
        summary = precheck_property_training_admission_execution_ledger_package(
            training_admission_execution_ledger_path=args.training_admission_execution_ledger,
            training_admission_execution_ledger_summary_path=args.training_admission_execution_ledger_summary,
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
            output_summary_path=args.output_summary,
            output_markdown_path=args.output_markdown,
            require_ledger_committed=args.require_ledger_committed,
            allow_ledger_needs_review=args.allow_ledger_needs_review,
            minimum_ledger_records=args.minimum_ledger_records,
        )
    except Exception as exc:
        err.write(f"property training admission execution ledger precheck invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("precheck_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_admission_execution_ledger_precheck",
        description="Preflight a property training admission execution ledger package.",
    )
    parser.add_argument("--training-admission-execution-ledger", required=True)
    parser.add_argument("--training-admission-execution-ledger-summary", required=True)
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
    parser.add_argument("--output-summary")
    parser.add_argument("--output-markdown")
    parser.add_argument("--require-ledger-committed", action="store_true", default=True)
    parser.add_argument("--allow-ledger-needs-review", action="store_true")
    parser.add_argument("--minimum-ledger-records", type=int, default=1)
    return parser


def _consistency_errors(
    *,
    ledger: dict[str, Any],
    ledger_summary: dict[str, Any],
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
    require_ledger_committed: bool,
    allow_ledger_needs_review: bool,
    minimum_ledger_records: int,
    warnings: list[str],
) -> list[str]:
    containers = (
        ledger,
        ledger_summary,
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
    errors: list[str] = []
    _append_errors(errors, _schema_errors(*containers))
    _append_errors(
        errors,
        _status_errors(
            ledger,
            ledger_summary,
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
            require_ledger_committed,
            allow_ledger_needs_review,
            warnings,
        ),
    )
    _append_errors(errors, _hash_errors(*containers, hashes=hashes))
    _append_errors(errors, _id_errors(*containers))
    _append_errors(
        errors,
        _record_errors(
            ledger,
            ledger_summary,
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
    _append_errors(errors, _sha_format_errors(*containers))
    return _stable_unique(errors)


def _schema_errors(*containers: dict[str, Any]) -> list[str]:
    expected = (
        ("training_admission_execution_ledger_schema_invalid", _LEDGER_SCHEMA_VERSION),
        ("training_admission_execution_ledger_summary_schema_invalid", _LEDGER_SUMMARY_SCHEMA_VERSION),
        ("training_admission_execution_dry_run_precheck_schema_invalid", _DRY_RUN_PREFLIGHT_SCHEMA_VERSION),
        ("training_admission_execution_dry_run_schema_invalid", _DRY_RUN_SCHEMA_VERSION),
        ("training_admission_execution_request_schema_invalid", _REQUEST_SCHEMA_VERSION),
        ("training_admission_execution_request_summary_schema_invalid", _REQUEST_SUMMARY_SCHEMA_VERSION),
        ("training_admission_execution_request_preflight_schema_invalid", _EXECUTION_PREFLIGHT_SCHEMA_VERSION),
        ("training_admission_request_draft_schema_invalid", _DRAFT_SCHEMA_VERSION),
        ("training_admission_request_draft_summary_schema_invalid", _DRAFT_SUMMARY_SCHEMA_VERSION),
        ("training_admission_request_draft_precheck_schema_invalid", _DRAFT_PREFLIGHT_SCHEMA_VERSION),
        ("training_admission_request_plan_schema_invalid", _REQUEST_PLAN_SCHEMA_VERSION),
        ("training_admission_request_preflight_schema_invalid", _REQUEST_PREFLIGHT_SCHEMA_VERSION),
        ("training_admission_readiness_schema_invalid", _READINESS_SCHEMA_VERSION),
        ("quarantine_candidate_preflight_schema_invalid", _QUARANTINE_PREFLIGHT_SCHEMA_VERSION),
        ("quarantine_candidate_schema_invalid", _QUARANTINE_CANDIDATE_SCHEMA_VERSION),
    )
    return [error for container, (error, schema) in zip(containers, expected) if container.get("schema_version") != schema]


def _status_errors(
    ledger: dict[str, Any],
    ledger_summary: dict[str, Any],
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
    require_ledger_committed: bool,
    allow: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    _append_errors(errors, _status_field_errors(ledger, "execution_status", "training_admission_execution_ledger", "committed", allow, warnings))
    _append_errors(errors, _status_field_errors(ledger_summary, "execution_status", "training_admission_execution_ledger_summary", "committed", allow, warnings))
    if require_ledger_committed and ledger.get("execution_status") == "needs_review" and not allow:
        errors.append("training_admission_execution_ledger_requires_committed")
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
    if ledger.get("execution_mode") != "training_admission_ledger_only":
        errors.append("training_admission_execution_ledger_mode_invalid")
    if request.get("request_mode") != "execution_request_only":
        errors.append("training_admission_execution_request_mode_invalid")
    if draft.get("request_mode") != "draft_only":
        errors.append("training_admission_request_draft_mode_invalid")
    if quarantine_candidate.get("materialization_mode") != "candidate_quarantine":
        errors.append("quarantine_materialization_mode_invalid")
    if ledger.get("execution_errors") or ledger_summary.get("execution_errors"):
        errors.append("training_admission_execution_ledger_has_errors")
    if dry_run_precheck.get("preflight_errors"):
        errors.append("training_admission_execution_dry_run_precheck_has_errors")
    if dry_run.get("dry_run_errors"):
        errors.append("training_admission_execution_dry_run_has_errors")
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
    if ledger.get("training_admitted") is not True or ledger_summary.get("training_admitted") is not True:
        errors.append("training_admission_ledger_not_committed")
    if ledger.get("phase1_status") != "not_run" or ledger_summary.get("phase1_status") != "not_run":
        errors.append("phase1_ran")
    if ledger.get("dataset_confirmation_changed") is not False or ledger_summary.get("dataset_confirmation_changed") is not False:
        errors.append("dataset_confirmation_changed")
    if ledger.get("training_dataset_materialized") is not False or ledger_summary.get("training_dataset_materialized") is not False:
        errors.append("training_dataset_materialized")
    if ledger.get("dataset_artifact_created") is not False or ledger_summary.get("dataset_artifact_created") is not False:
        errors.append("dataset_artifact_created")
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
    if status in {"blocked", "failed"}:
        return [f"{prefix}_blocked"]
    return [f"{prefix}_status_invalid"]


def _hash_errors(*containers: dict[str, Any], hashes: dict[str, str]) -> list[str]:
    (
        ledger,
        ledger_summary,
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
        _quarantine_preflight,
        _quarantine_candidate,
    ) = containers
    errors: list[str] = []
    if ledger_summary.get("training_admission_execution_ledger_sha256") != hashes["training_admission_execution_ledger_sha256"]:
        errors.append("training_admission_execution_ledger_sha256_mismatch")
    direct_pairs = (
        (ledger, "training_admission_execution_dry_run_precheck_sha256", "training_admission_execution_dry_run_precheck_sha256"),
        (ledger, "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256"),
        (ledger, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (ledger, "training_admission_execution_request_summary_sha256", "training_admission_execution_request_summary_sha256"),
        (ledger, "training_admission_execution_request_preflight_sha256", "training_admission_execution_request_preflight_sha256"),
        (ledger, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (ledger, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (ledger, "training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (ledger, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (ledger, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (ledger, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (ledger, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (ledger, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (ledger_summary, "training_admission_execution_dry_run_precheck_sha256", "training_admission_execution_dry_run_precheck_sha256"),
        (ledger_summary, "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256"),
        (ledger_summary, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (ledger_summary, "training_admission_execution_request_summary_sha256", "training_admission_execution_request_summary_sha256"),
        (ledger_summary, "training_admission_execution_request_preflight_sha256", "training_admission_execution_request_preflight_sha256"),
        (ledger_summary, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (ledger_summary, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (ledger_summary, "training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (ledger_summary, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (ledger_summary, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (ledger_summary, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (ledger_summary, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (ledger_summary, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
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
        (request_summary, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (execution_preflight, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (execution_preflight, "training_admission_execution_request_summary_sha256", "training_admission_execution_request_summary_sha256"),
        (request, "source_training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (request, "source_training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (request, "source_training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (request, "source_training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (request, "source_training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (request, "source_training_admission_readiness_sha256", "training_admission_readiness_summary_sha256"),
        (request, "source_quarantine_candidate_preflight_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (request, "source_quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (draft_summary, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (draft_precheck, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (request_plan, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (request_preflight, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (readiness, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
    )
    for container, field, hash_key in direct_pairs:
        if container.get(field) and container.get(field) != hashes[hash_key]:
            errors.append(f"{hash_key}_mismatch")
    for record in _safe_records(ledger.get("ledger_records")):
        record_pairs = (
            ("training_admission_execution_dry_run_sha256", "training_admission_execution_dry_run_report_sha256"),
            ("training_admission_execution_dry_run_precheck_sha256", "training_admission_execution_dry_run_precheck_sha256"),
            ("training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
            ("training_admission_execution_request_preflight_sha256", "training_admission_execution_request_preflight_sha256"),
            ("training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
            ("training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
            ("training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
            ("training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
            ("training_admission_readiness_sha256", "training_admission_readiness_summary_sha256"),
            ("quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        )
        for record_field, hash_key in record_pairs:
            if record.get(record_field) != hashes[hash_key]:
                errors.append(f"{hash_key}_mismatch")
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
    ledger_ids = [
        str(container.get("execution_ledger_id", ""))
        for container in containers[:2]
        if container.get("execution_ledger_id")
    ]
    if len(set(ledger_ids)) > 1:
        errors.append("execution_ledger_id_mismatch")
    for logical, keys in logical_fields.items():
        if logical == "dataset_target":
            continue
        for container in containers:
            for key in keys:
                value = str(container.get(key, ""))
                if value and not _is_safe_id(value):
                    errors.append(f"{logical}_invalid")
    for value in ledger_ids:
        if value and not _is_safe_id(value):
            errors.append("execution_ledger_id_invalid")
    return _stable_unique(errors)


def _record_errors(
    ledger: dict[str, Any],
    ledger_summary: dict[str, Any],
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
    ledger_records = _safe_records(ledger.get("ledger_records"))
    ledger_record_ids = [str(record.get("ledger_record_id", "")) for record in ledger_records]
    dry_run_records = _safe_records(dry_run.get("dry_run_records"))
    dry_run_record_ids = [str(record.get("dry_run_record_id", "")) for record in dry_run_records]
    execution_records = _safe_records(request.get("execution_records"))
    execution_record_ids = [str(record.get("execution_record_id", "")) for record in execution_records]
    execution_candidate_ids = [str(record.get("candidate_record_id", "")) for record in execution_records]
    draft_records = _safe_records(draft.get("draft_records"))
    draft_record_ids = [str(record.get("draft_record_id", "")) for record in draft_records]
    draft_candidate_ids = [str(record.get("candidate_record_id", "")) for record in draft_records]
    planned_ids = _safe_list(ledger.get("planned_training_admission_candidate_record_ids"))
    request_planned_ids = _safe_list(request.get("planned_training_admission_candidate_record_ids"))
    quarantine_candidate_ids = _safe_list(quarantine_candidate.get("candidate_record_ids"))
    if not ledger_records:
        errors.append("no_ledger_records")
    if len(ledger_records) < max(minimum_ledger_records, 1):
        errors.append("minimum_ledger_record_count_not_met")
    if not dry_run_records:
        errors.append("no_dry_run_records")
    if not execution_records:
        errors.append("no_execution_records")
    if not planned_ids or not request_planned_ids:
        errors.append("no_planned_candidates")
    _count_errors(errors, ledger, "ledger_record_count", ledger_records)
    _count_errors(errors, ledger_summary, "ledger_record_count", ledger_records)
    _count_errors(errors, dry_run, "dry_run_record_count", dry_run_records)
    _count_errors(errors, request, "execution_record_count", execution_records)
    _count_errors(errors, request_summary, "execution_record_count", execution_records)
    _count_errors(errors, execution_preflight, "execution_record_count", execution_records)
    _count_errors(errors, draft, "draft_record_count", draft_records)
    if _safe_list(ledger.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(ledger_summary.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(ledger.get("dry_run_record_ids")) != dry_run_record_ids:
        errors.append("dry_run_record_ids_mismatch")
    if _safe_list(ledger_summary.get("dry_run_record_ids")) != dry_run_record_ids:
        errors.append("dry_run_record_ids_mismatch")
    if _safe_list(dry_run.get("dry_run_record_ids")) != dry_run_record_ids:
        errors.append("dry_run_record_ids_mismatch")
    if _safe_list(ledger.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(ledger_summary.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(request.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(request_summary.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if planned_ids != request_planned_ids:
        errors.append("planned_candidate_ids_mismatch")
    for ids in (
        _safe_list(ledger_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(dry_run_precheck.get("planned_training_admission_candidate_record_ids")),
        _safe_list(dry_run.get("planned_training_admission_candidate_record_ids")),
        _safe_list(request_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(execution_preflight.get("planned_training_admission_candidate_record_ids")),
        _safe_list(draft.get("planned_training_admission_candidate_record_ids")),
        _safe_list(draft_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(draft_precheck.get("planned_training_admission_candidate_record_ids")),
        _safe_list(request_plan.get("planned_training_admission_candidate_record_ids")),
        _safe_list(request_preflight.get("planned_training_admission_candidate_record_ids")),
    ):
        if ids != planned_ids:
            errors.append("planned_candidate_ids_mismatch")
    readiness_planned = _safe_list(readiness.get("planned_training_admission_candidate_record_ids"))
    if readiness_planned and readiness_planned != planned_ids:
        errors.append("planned_candidate_ids_mismatch")
    if set(planned_ids) != set(execution_candidate_ids):
        errors.append("planned_candidate_not_in_execution_request")
    if set(planned_ids) != set(draft_candidate_ids):
        errors.append("planned_candidate_not_in_draft")
    if not set(planned_ids).issubset(set(quarantine_candidate_ids)):
        errors.append("planned_candidate_ids_unknown")
    excluded = set(_safe_list(request_plan.get("exclude_record_ids")))
    excluded.update(_safe_list(request_preflight.get("exclude_record_ids")))
    blocked = set(_safe_list(request_plan.get("blocked_record_ids")))
    blocked.update(_safe_list(request_plan.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(request_preflight.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(readiness.get("blocked_from_training_admission_record_ids")))
    needs_review = set(_safe_list(request_plan.get("needs_review_record_ids")))
    needs_review.update(_safe_list(request_preflight.get("needs_review_record_ids")))
    dry_run_id_set = set(dry_run_record_ids)
    execution_id_set = set(execution_record_ids)
    draft_id_set = set(draft_record_ids)
    for record in ledger_records:
        ledger_record_id = str(record.get("ledger_record_id", ""))
        dry_run_record_id = str(record.get("dry_run_record_id", ""))
        execution_record_id = str(record.get("execution_record_id", ""))
        draft_record_id = str(record.get("draft_record_id", ""))
        candidate_id = str(record.get("candidate_record_id", ""))
        record_id = str(record.get("record_id", ""))
        if not ledger_record_id or not _is_safe_id(ledger_record_id):
            errors.append("ledger_record_id_invalid")
        if dry_run_record_id not in dry_run_id_set:
            errors.append("ledger_dry_run_record_id_unknown")
        if execution_record_id not in execution_id_set:
            errors.append("ledger_execution_record_id_unknown")
        if draft_record_id not in draft_id_set:
            errors.append("ledger_draft_record_id_unknown")
        if candidate_id not in planned_ids:
            errors.append("ledger_candidate_id_unknown")
        if candidate_id in excluded or record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if candidate_id in blocked or record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if candidate_id in needs_review or record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
        if record.get("admission_action") != "admit_training_candidate":
            errors.append("ledger_record_action_invalid")
        if record.get("ledger_record_status") != "admitted_to_training_ledger":
            errors.append("ledger_record_status_invalid")
        if record.get("training_admitted") is not True:
            errors.append("ledger_record_not_training_admitted")
        if record.get("phase1_status") != "not_run":
            errors.append("phase1_ran")
        if record.get("dataset_confirmation_changed") is not False:
            errors.append("dataset_confirmation_changed")
        if record.get("training_admission_execution_dry_run_sha256") != hashes["training_admission_execution_dry_run_report_sha256"]:
            errors.append("training_admission_execution_dry_run_report_sha256_mismatch")
        if not _record_values_are_safe(record, allowed_labels={"admission_action", "ledger_record_status", "phase1_status"}):
            errors.append("ledger_record_contains_unsafe_value")
    return _stable_unique(errors)


def _count_errors(errors: list[str], container: dict[str, Any], field: str, records: list[dict[str, Any]]) -> None:
    try:
        count = int(container.get(field, len(records)))
    except Exception:
        errors.append(f"{field}_mismatch")
        return
    if count != len(records):
        errors.append(f"{field}_mismatch")


def _summary(
    *,
    precheck_status: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    ledger: dict[str, Any],
    ledger_summary: dict[str, Any],
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
    precheck_errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    ledger_records = _safe_records(ledger.get("ledger_records"))
    planned_ids = _safe_list(ledger.get("planned_training_admission_candidate_record_ids"))
    blocked_ids = _stable_unique(
        _safe_list(request_plan.get("blocked_record_ids"))
        + _safe_list(request_plan.get("blocked_from_training_admission_record_ids"))
        + _safe_list(request_preflight.get("blocked_from_training_admission_record_ids"))
        + _safe_list(readiness.get("blocked_from_training_admission_record_ids"))
    )
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": precheck_status,
        **_source_fields(paths, hashes),
        "execution_ledger_id": str(ledger.get("execution_ledger_id", ledger_summary.get("execution_ledger_id", ""))),
        "dry_run_id": str(ledger.get("dry_run_id", dry_run.get("dry_run_id", ""))),
        "execution_request_id": str(ledger.get("execution_request_id", request.get("execution_request_id", ""))),
        "corpus_id": str(ledger.get("corpus_id", request_plan.get("corpus_id", ""))),
        "source_dry_run_id": str(ledger.get("source_dry_run_id", request_plan.get("source_dry_run_id", ""))),
        "review_manifest_id": str(ledger.get("review_manifest_id", request_plan.get("review_manifest_id", ""))),
        "admission_request_id": str(ledger.get("admission_request_id", request_plan.get("admission_request_id", ""))),
        "materialization_plan_id": str(ledger.get("materialization_plan_id", request_plan.get("materialization_plan_id", ""))),
        "source_execution_request_id": str(ledger.get("source_execution_request_id", request_plan.get("execution_request_id", ""))),
        "quarantine_run_id": str(ledger.get("quarantine_run_id", request_plan.get("quarantine_run_id", ""))),
        "review_queue_id": str(ledger.get("review_queue_id", request_plan.get("review_queue_id", ""))),
        "property_candidate_manifest_id": str(
            ledger.get("property_candidate_manifest_id", request_plan.get("property_candidate_manifest_id", ""))
        ),
        "dataset_target": str(ledger.get("dataset_target", request_plan.get("dataset_target", ""))),
        "execution_ledger_status": str(ledger.get("execution_status", "")),
        "dry_run_precheck_status": str(dry_run_precheck.get("preflight_status", "")),
        "dry_run_status": str(dry_run.get("dry_run_status", "")),
        "execution_request_status": str(request.get("request_status", request_summary.get("request_status", ""))),
        "execution_request_preflight_status": str(execution_preflight.get("preflight_status", "")),
        "draft_precheck_status": str(draft_precheck.get("precheck_status", "")),
        "draft_status": str(draft.get("draft_status", "")),
        "request_plan_status": str(request_plan.get("planner_status", "")),
        "request_preflight_status": str(request_preflight.get("preflight_status", "")),
        "readiness_status": str(readiness.get("readiness_status", "")),
        "training_admitted": bool(ledger.get("training_admitted") is True and ledger_summary.get("training_admitted") is True),
        "phase1_status": str(ledger.get("phase1_status", "")),
        "dataset_confirmation_changed": bool(ledger.get("dataset_confirmation_changed") is True),
        "training_dataset_materialized": bool(ledger.get("training_dataset_materialized") is True),
        "dataset_artifact_created": bool(ledger.get("dataset_artifact_created") is True),
        "ledger_record_count": len(ledger_records),
        "dry_run_record_count": len(_safe_records(dry_run.get("dry_run_records"))),
        "execution_record_count": len(_safe_records(request.get("execution_records"))),
        "draft_record_count": len(_safe_records(draft.get("draft_records"))),
        "planned_candidate_count": len(planned_ids),
        "ledger_record_ids": [str(record.get("ledger_record_id", "")) for record in ledger_records],
        "dry_run_record_ids": _safe_list(dry_run.get("dry_run_record_ids")),
        "execution_record_ids": _safe_list(request.get("execution_record_ids")),
        "planned_training_admission_candidate_record_ids": planned_ids,
        "blocked_from_training_admission_record_ids": blocked_ids,
        "precheck_errors": _stable_unique(precheck_errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
        "boundary_statement": [
            "training admission execution ledger precheck only",
            "no future training dataset materialization",
            "no training CSV/JSONL/Parquet/LMDB",
            "no candidate CSV/JSONL/Parquet/LMDB",
            "no Phase 1",
            "DatasetConfirmation unchanged",
            "no model training/evaluation",
        ],
    }
    return summary


def _source_fields(paths: dict[str, str | Path], hashes: dict[str, str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, path in paths.items():
        fields[key] = _basename(path, key)
        fields[key.replace("_path", "_sha256")] = hashes[key.replace("_path", "_sha256")]
    return fields


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Admission Execution Ledger Precheck Evidence",
            "",
            f"- Precheck status: `{summary['precheck_status']}`",
            f"- Execution ledger id: `{summary['execution_ledger_id']}`",
            f"- Execution ledger status: `{summary['execution_ledger_status']}`",
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
            f"- Precheck errors: `{json.dumps(summary['precheck_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training admission execution ledger precheck only.",
            "- future training dataset materialization was not run.",
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
        raise CustomCorpusPropertyTrainingAdmissionExecutionLedgerPrecheckError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyTrainingAdmissionExecutionLedgerPrecheckError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyTrainingAdmissionExecutionLedgerPrecheckError(f"{label} contains forbidden material")
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
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_admission_execution_ledger_precheck_redaction_failed"],
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
