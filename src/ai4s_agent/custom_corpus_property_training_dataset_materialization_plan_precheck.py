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


_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_plan_precheck.v1"
_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_plan.v1"
_PLANNER_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_planner.v1"
_LEDGER_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger_precheck.v1"
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
_ALLOWED_OUTPUT_FORMATS = {"jsonl", "parquet", "lmdb", "csv"}
_ALLOWED_TARGET_MODEL_FAMILIES = {"unimol", "dpa3", "generic_property_predictor"}
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
)


class CustomCorpusPropertyTrainingDatasetMaterializationPlanPrecheckError(ValueError):
    pass


def precheck_property_training_dataset_materialization_plan(
    *,
    training_dataset_materialization_plan_path: str | Path,
    training_dataset_materialization_planner_summary_path: str | Path,
    training_admission_execution_ledger_precheck_path: str | Path,
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
    require_plan_planned: bool = True,
    allow_plan_needs_review: bool = False,
    minimum_planned_records: int = 1,
) -> dict[str, Any]:
    plan = _read_safe_json_dict(training_dataset_materialization_plan_path, "training dataset materialization plan")
    planner_summary = _read_safe_json_dict(
        training_dataset_materialization_planner_summary_path,
        "training dataset materialization planner summary",
    )
    ledger_precheck = _read_safe_json_dict(
        training_admission_execution_ledger_precheck_path,
        "training admission execution ledger precheck",
    )
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
        "training_dataset_materialization_plan_path": training_dataset_materialization_plan_path,
        "training_dataset_materialization_planner_summary_path": training_dataset_materialization_planner_summary_path,
        "training_admission_execution_ledger_precheck_path": training_admission_execution_ledger_precheck_path,
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
        plan=plan,
        planner_summary=planner_summary,
        ledger_precheck=ledger_precheck,
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
        require_plan_planned=require_plan_planned,
        allow_plan_needs_review=allow_plan_needs_review,
        minimum_planned_records=max(int(minimum_planned_records), 0),
        warnings=warnings,
    )
    precheck_status = "blocked" if errors else "needs_review" if warnings else "passed"
    summary = _summary(
        precheck_status=precheck_status,
        paths=paths,
        hashes=hashes,
        plan=plan,
        planner_summary=planner_summary,
        ledger_precheck=ledger_precheck,
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
        errors=_stable_unique(errors),
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
        summary = precheck_property_training_dataset_materialization_plan(
            training_dataset_materialization_plan_path=args.training_dataset_materialization_plan,
            training_dataset_materialization_planner_summary_path=args.training_dataset_materialization_planner_summary,
            training_admission_execution_ledger_precheck_path=args.training_admission_execution_ledger_precheck,
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
            require_plan_planned=args.require_plan_planned,
            allow_plan_needs_review=args.allow_plan_needs_review,
            minimum_planned_records=args.minimum_planned_records,
        )
    except Exception as exc:
        err.write(f"property training dataset materialization plan precheck invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("precheck_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_dataset_materialization_plan_precheck",
        description="Precheck a property training dataset materialization plan package.",
    )
    parser.add_argument("--training-dataset-materialization-plan", required=True)
    parser.add_argument("--training-dataset-materialization-planner-summary", required=True)
    parser.add_argument("--training-admission-execution-ledger-precheck", required=True)
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
    parser.add_argument("--require-plan-planned", action="store_true", default=True)
    parser.add_argument("--allow-plan-needs-review", action="store_true")
    parser.add_argument("--minimum-planned-records", type=int, default=1)
    return parser


def _consistency_errors(
    *,
    plan: dict[str, Any],
    planner_summary: dict[str, Any],
    ledger_precheck: dict[str, Any],
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
    require_plan_planned: bool,
    allow_plan_needs_review: bool,
    minimum_planned_records: int,
    warnings: list[str],
) -> list[str]:
    containers = (
        plan,
        planner_summary,
        ledger_precheck,
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
            plan,
            planner_summary,
            ledger_precheck,
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
            require_plan_planned,
            allow_plan_needs_review,
            warnings,
        ),
    )
    _append_errors(errors, _hash_errors(*containers, hashes=hashes))
    _append_errors(errors, _id_errors(*containers))
    _append_errors(
        errors,
        _record_errors(
            plan,
            planner_summary,
            ledger_precheck,
            ledger,
            ledger_summary,
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
            minimum_planned_records,
        ),
    )
    _append_errors(errors, _sha_format_errors(*containers))
    return _stable_unique(errors)


def _schema_errors(*containers: dict[str, Any]) -> list[str]:
    expected = (
        ("training_dataset_materialization_plan_schema_invalid", _PLAN_SCHEMA_VERSION),
        ("training_dataset_materialization_planner_summary_schema_invalid", _PLANNER_SUMMARY_SCHEMA_VERSION),
        ("training_admission_execution_ledger_precheck_schema_invalid", _LEDGER_PREFLIGHT_SCHEMA_VERSION),
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
    plan: dict[str, Any],
    planner_summary: dict[str, Any],
    ledger_precheck: dict[str, Any],
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
    require_plan_planned: bool,
    allow: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    _append_errors(errors, _status_field_errors(plan, "plan_status", "training_dataset_materialization_plan", "planned", allow, warnings))
    _append_errors(errors, _status_field_errors(planner_summary, "plan_status", "training_dataset_materialization_planner_summary", "planned", allow, warnings))
    _append_errors(errors, _status_field_errors(ledger_precheck, "precheck_status", "training_admission_execution_ledger_precheck", "passed", allow, warnings))
    _append_errors(errors, _status_field_errors(ledger, "execution_status", "training_admission_execution_ledger", "committed", allow, warnings))
    _append_errors(errors, _status_field_errors(ledger_summary, "execution_status", "training_admission_execution_ledger_summary", "committed", allow, warnings))
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
    if require_plan_planned and plan.get("plan_status") == "needs_review" and not allow:
        errors.append("training_dataset_materialization_plan_requires_planned")
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
    if plan.get("plan_mode") != "training_dataset_materialization_plan_only":
        errors.append("training_dataset_materialization_plan_mode_invalid")
    if plan.get("planning_errors"):
        errors.append("training_dataset_materialization_plan_has_errors")
    if planner_summary.get("planning_errors"):
        errors.append("training_dataset_materialization_planner_summary_has_errors")
    if ledger_precheck.get("precheck_errors"):
        errors.append("training_admission_execution_ledger_precheck_has_errors")
    if ledger.get("execution_errors") or ledger_summary.get("execution_errors"):
        errors.append("training_admission_execution_ledger_has_errors")
    if dry_run_precheck.get("preflight_errors"):
        errors.append("training_admission_execution_dry_run_precheck_has_errors")
    if dry_run.get("dry_run_errors"):
        errors.append("training_admission_execution_dry_run_has_errors")
    if request.get("request_mode") != "execution_request_only":
        errors.append("training_admission_execution_request_mode_invalid")
    if request_summary.get("request_errors"):
        errors.append("training_admission_execution_request_summary_has_errors")
    if execution_preflight.get("preflight_errors"):
        errors.append("training_admission_execution_request_preflight_has_errors")
    if draft.get("request_mode") != "draft_only":
        errors.append("training_admission_request_draft_mode_invalid")
    if draft_summary.get("draft_errors"):
        errors.append("training_admission_request_draft_summary_has_errors")
    if draft_precheck.get("precheck_errors"):
        errors.append("training_admission_request_draft_precheck_has_errors")
    if request_plan.get("planning_errors"):
        errors.append("training_admission_request_plan_has_errors")
    if request_preflight.get("preflight_errors"):
        errors.append("training_admission_request_preflight_has_errors")
    if readiness.get("readiness_errors"):
        errors.append("training_admission_readiness_has_errors")
    if quarantine_candidate.get("materialization_mode") != "candidate_quarantine":
        errors.append("quarantine_materialization_mode_invalid")
    if quarantine_preflight.get("preflight_errors"):
        errors.append("quarantine_candidate_preflight_has_errors")
    if plan.get("training_admitted") is not True:
        errors.append("training_not_admitted")
    if ledger_precheck.get("training_admitted") is not True or ledger.get("training_admitted") is not True or ledger_summary.get("training_admitted") is not True:
        errors.append("training_admission_ledger_not_committed")
    if any(container.get("training_dataset_materialized") is not False for container in (plan, planner_summary, ledger_precheck, ledger, ledger_summary)):
        errors.append("training_dataset_materialized")
    if any(container.get("dataset_artifact_created") is not False for container in (plan, planner_summary, ledger_precheck, ledger, ledger_summary)):
        errors.append("dataset_artifact_created")
    boundary_containers = (
        plan,
        planner_summary,
        ledger_precheck,
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
    if any(container.get("phase1_status") != "not_run" for container in boundary_containers):
        errors.append("phase1_ran")
    if any(container.get("dataset_confirmation_changed") is not False for container in boundary_containers):
        errors.append("dataset_confirmation_changed")
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
    dataset_name = str(plan.get("dataset_name", ""))
    if not dataset_name or not _is_safe_id(dataset_name):
        errors.append("dataset_name_invalid")
    output_formats = _safe_list(plan.get("planned_output_formats"))
    if not output_formats or any(fmt not in _ALLOWED_OUTPUT_FORMATS for fmt in output_formats):
        errors.append("planned_output_format_invalid")
    model_families = _safe_list(plan.get("target_model_families"))
    if not model_families or any(label not in _ALLOWED_TARGET_MODEL_FAMILIES for label in model_families):
        errors.append("target_model_family_invalid")
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
        plan,
        planner_summary,
        ledger_precheck,
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
    direct_pairs = (
        (planner_summary, "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256"),
        (plan, "training_admission_execution_ledger_precheck_sha256", "training_admission_execution_ledger_precheck_sha256"),
        (plan, "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256"),
        (plan, "training_admission_execution_ledger_summary_sha256", "training_admission_execution_ledger_summary_sha256"),
        (plan, "training_admission_execution_dry_run_precheck_sha256", "training_admission_execution_dry_run_precheck_sha256"),
        (plan, "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256"),
        (plan, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (plan, "training_admission_execution_request_summary_sha256", "training_admission_execution_request_summary_sha256"),
        (plan, "training_admission_execution_request_preflight_sha256", "training_admission_execution_request_preflight_sha256"),
        (plan, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (plan, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (plan, "training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (plan, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (plan, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (plan, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (plan, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (plan, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (planner_summary, "training_admission_execution_ledger_precheck_sha256", "training_admission_execution_ledger_precheck_sha256"),
        (planner_summary, "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256"),
        (planner_summary, "training_admission_execution_ledger_summary_sha256", "training_admission_execution_ledger_summary_sha256"),
        (planner_summary, "training_admission_execution_dry_run_precheck_sha256", "training_admission_execution_dry_run_precheck_sha256"),
        (planner_summary, "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256"),
        (planner_summary, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (planner_summary, "training_admission_execution_request_summary_sha256", "training_admission_execution_request_summary_sha256"),
        (planner_summary, "training_admission_execution_request_preflight_sha256", "training_admission_execution_request_preflight_sha256"),
        (planner_summary, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (planner_summary, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (planner_summary, "training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (planner_summary, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (planner_summary, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (planner_summary, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (planner_summary, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (planner_summary, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (ledger_precheck, "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256"),
        (ledger_summary, "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256"),
        (dry_run_precheck, "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256"),
        (request_summary, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (execution_preflight, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (draft_summary, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (draft_precheck, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (request_plan, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (request_preflight, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (readiness, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
    )
    for container, field, hash_key in direct_pairs:
        if container.get(field) and container.get(field) != hashes[hash_key]:
            errors.append(f"{hash_key}_mismatch")
    for record in _safe_records(plan.get("planned_dataset_records")):
        record_pairs = (
            ("training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256"),
            ("training_admission_execution_ledger_precheck_sha256", "training_admission_execution_ledger_precheck_sha256"),
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
    plan, planner_summary, *_rest = containers
    if plan.get("materialization_plan_id") != planner_summary.get("materialization_plan_id"):
        errors.append("materialization_plan_id_mismatch")
    logical_fields = {
        "execution_ledger_id": ("execution_ledger_id",),
        "dry_run_id": ("dry_run_id",),
        "corpus_id": ("corpus_id",),
        "source_dry_run_id": ("source_dry_run_id",),
        "review_manifest_id": ("review_manifest_id",),
        "admission_request_id": ("admission_request_id",),
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
    for value in (str(plan.get("materialization_plan_id", "")), str(planner_summary.get("materialization_plan_id", ""))):
        if value and not _is_safe_id(value):
            errors.append("materialization_plan_id_invalid")
    for logical, keys in logical_fields.items():
        if logical == "dataset_target":
            continue
        for container in containers:
            for key in keys:
                value = str(container.get(key, ""))
                if value and not _is_safe_id(value):
                    errors.append(f"{logical}_invalid")
    training_execution_request_values = [
        str(container.get("execution_request_id", ""))
        for container in containers[:10]
        if container.get("execution_request_id")
    ]
    if len(set(training_execution_request_values)) > 1:
        errors.append("execution_request_id_mismatch")
    for value in training_execution_request_values:
        if value and not _is_safe_id(value):
            errors.append("execution_request_id_invalid")
    source_execution_request_values: list[str] = []
    for index, container in enumerate(containers):
        value = str(container.get("source_execution_request_id", ""))
        if not value and index >= 10:
            value = str(container.get("execution_request_id", ""))
        if value:
            source_execution_request_values.append(value)
    if len(set(source_execution_request_values)) > 1:
        errors.append("source_execution_request_id_mismatch")
    for value in source_execution_request_values:
        if value and not _is_safe_id(value):
            errors.append("source_execution_request_id_invalid")
    if plan.get("plan_status") != planner_summary.get("plan_status"):
        errors.append("plan_status_mismatch")
    return _stable_unique(errors)


def _record_errors(
    plan: dict[str, Any],
    planner_summary: dict[str, Any],
    ledger_precheck: dict[str, Any],
    ledger: dict[str, Any],
    ledger_summary: dict[str, Any],
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
    minimum_planned_records: int,
) -> list[str]:
    errors: list[str] = []
    plan_records = _safe_records(plan.get("planned_dataset_records"))
    plan_record_ids = [str(record.get("planned_dataset_record_id", "")) for record in plan_records]
    ledger_records = _safe_records(ledger.get("ledger_records"))
    ledger_record_ids = [str(record.get("ledger_record_id", "")) for record in ledger_records]
    dry_run_records = _safe_records(dry_run.get("dry_run_records"))
    dry_run_record_ids = [str(record.get("dry_run_record_id", "")) for record in dry_run_records]
    execution_records = _safe_records(request.get("execution_records"))
    execution_record_ids = [str(record.get("execution_record_id", "")) for record in execution_records]
    draft_records = _safe_records(draft.get("draft_records"))
    draft_record_ids = [str(record.get("draft_record_id", "")) for record in draft_records]
    planned_ids = _safe_list(plan.get("planned_training_admission_candidate_record_ids"))
    ledger_planned_ids = _safe_list(ledger.get("planned_training_admission_candidate_record_ids"))
    quarantine_candidate_ids = _safe_list(quarantine_candidate.get("candidate_record_ids"))
    if not plan_records:
        errors.append("no_planned_dataset_records")
    if len(plan_records) < max(minimum_planned_records, 1):
        errors.append("minimum_planned_record_count_not_met")
    if not ledger_records or not _safe_list(plan.get("ledger_record_ids")):
        errors.append("no_ledger_records")
    if not planned_ids:
        errors.append("no_planned_candidates")
    _count_errors(errors, plan, "planned_dataset_record_count", plan_records)
    _count_errors(errors, planner_summary, "planned_dataset_record_count", plan_records)
    _count_errors(errors, plan, "ledger_record_count", ledger_records)
    _count_errors(errors, planner_summary, "ledger_record_count", ledger_records)
    _count_errors(errors, ledger_precheck, "ledger_record_count", ledger_records)
    _count_errors(errors, ledger, "ledger_record_count", ledger_records)
    _count_errors(errors, ledger_summary, "ledger_record_count", ledger_records)
    _count_errors(errors, dry_run, "dry_run_record_count", dry_run_records)
    _count_errors(errors, request, "execution_record_count", execution_records)
    _count_errors(errors, request_summary, "execution_record_count", execution_records)
    _count_errors(errors, execution_preflight, "execution_record_count", execution_records)
    _count_errors(errors, draft, "draft_record_count", draft_records)
    if int(ledger_precheck.get("planned_candidate_count", len(ledger_planned_ids))) != len(ledger_planned_ids):
        errors.append("planned_candidate_count_mismatch")
    if int(planner_summary.get("planned_candidate_count", len(planned_ids))) != len(planned_ids):
        errors.append("planned_candidate_count_mismatch")
    if _safe_list(plan.get("planned_dataset_record_ids")) != plan_record_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    if _safe_list(planner_summary.get("planned_dataset_record_ids")) != plan_record_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    if _safe_list(plan.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(planner_summary.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(ledger.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(ledger_summary.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(ledger_precheck.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(dry_run.get("dry_run_record_ids")) != dry_run_record_ids:
        errors.append("dry_run_record_ids_mismatch")
    if _safe_list(request.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(request_summary.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(draft.get("draft_record_ids")) != draft_record_ids:
        errors.append("draft_record_ids_mismatch")
    if planned_ids != ledger_planned_ids:
        errors.append("planned_candidate_ids_mismatch")
    for ids in (
        _safe_list(planner_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(ledger_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(ledger_precheck.get("planned_training_admission_candidate_record_ids")),
        _safe_list(dry_run.get("planned_training_admission_candidate_record_ids")),
        _safe_list(request.get("planned_training_admission_candidate_record_ids")),
        _safe_list(request_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(execution_preflight.get("planned_training_admission_candidate_record_ids")),
        _safe_list(draft.get("planned_training_admission_candidate_record_ids")),
        _safe_list(draft_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(draft_precheck.get("planned_training_admission_candidate_record_ids")),
        _safe_list(request_plan.get("planned_training_admission_candidate_record_ids")),
        _safe_list(request_preflight.get("planned_training_admission_candidate_record_ids")),
    ):
        if ids and ids != planned_ids:
            errors.append("planned_candidate_ids_mismatch")
    readiness_planned = _safe_list(readiness.get("planned_training_admission_candidate_record_ids"))
    if readiness_planned and readiness_planned != planned_ids:
        errors.append("planned_candidate_ids_mismatch")
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
    ledger_id_set = set(ledger_record_ids)
    dry_run_id_set = set(dry_run_record_ids)
    execution_id_set = set(execution_record_ids)
    draft_id_set = set(draft_record_ids)
    plan_candidate_ids = [str(record.get("candidate_record_id", "")) for record in plan_records]
    if set(plan_candidate_ids) != set(planned_ids):
        errors.append("planned_candidate_ids_mismatch")
    for record in plan_records:
        plan_record_id = str(record.get("planned_dataset_record_id", ""))
        ledger_record_id = str(record.get("ledger_record_id", ""))
        dry_run_record_id = str(record.get("dry_run_record_id", ""))
        execution_record_id = str(record.get("execution_record_id", ""))
        draft_record_id = str(record.get("draft_record_id", ""))
        candidate_id = str(record.get("candidate_record_id", ""))
        record_id = str(record.get("record_id", ""))
        if not plan_record_id or not _is_safe_id(plan_record_id):
            errors.append("planned_dataset_record_id_invalid")
        if ledger_record_id not in ledger_id_set:
            errors.append("planned_dataset_record_ledger_id_unknown")
        if dry_run_record_id and dry_run_record_id not in dry_run_id_set:
            errors.append("planned_dataset_record_dry_run_record_id_unknown")
        if execution_record_id and execution_record_id not in execution_id_set:
            errors.append("planned_dataset_record_execution_record_id_unknown")
        if draft_record_id and draft_record_id not in draft_id_set:
            errors.append("planned_dataset_record_draft_record_id_unknown")
        if candidate_id not in planned_ids:
            errors.append("planned_candidate_ids_mismatch")
        if candidate_id in excluded or record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if candidate_id in blocked or record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if candidate_id in needs_review or record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
        if record.get("planned_action") != "materialize_training_dataset_record":
            errors.append("planned_dataset_record_action_invalid")
        if record.get("planned_record_status") != "planned":
            errors.append("planned_dataset_record_status_invalid")
        if record.get("training_admitted") is not True:
            errors.append("planned_dataset_record_not_training_admitted")
        if record.get("training_dataset_materialized") is not False:
            errors.append("training_dataset_materialized")
        if record.get("dataset_artifact_created") is not False:
            errors.append("dataset_artifact_created")
        if record.get("phase1_status") != "not_run":
            errors.append("phase1_ran")
        if record.get("dataset_confirmation_changed") is not False:
            errors.append("dataset_confirmation_changed")
        if _safe_list(record.get("target_model_families")) != _safe_list(plan.get("target_model_families")):
            errors.append("target_model_family_mismatch")
        if _safe_list(record.get("planned_output_formats")) != _safe_list(plan.get("planned_output_formats")):
            errors.append("planned_output_format_mismatch")
        if not _record_values_are_safe(
            record,
            allowed_labels={"planned_action", "planned_record_status", "phase1_status"},
            allowed_list_fields={"target_model_families", "planned_output_formats"},
        ):
            errors.append("planned_dataset_record_contains_unsafe_value")
    return _stable_unique(errors)


def _summary(
    *,
    precheck_status: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    plan: dict[str, Any],
    planner_summary: dict[str, Any],
    ledger_precheck: dict[str, Any],
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
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    plan_records = _safe_records(plan.get("planned_dataset_records"))
    ledger_records = _safe_records(ledger.get("ledger_records"))
    dry_run_records = _safe_records(dry_run.get("dry_run_records"))
    execution_records = _safe_records(request.get("execution_records"))
    draft_records = _safe_records(draft.get("draft_records"))
    blocked_ids = _stable_unique(
        _safe_list(request_plan.get("blocked_record_ids"))
        + _safe_list(request_plan.get("blocked_from_training_admission_record_ids"))
        + _safe_list(request_preflight.get("blocked_from_training_admission_record_ids"))
        + _safe_list(readiness.get("blocked_from_training_admission_record_ids"))
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": precheck_status,
        **_source_fields(paths, hashes),
        "materialization_plan_id": str(plan.get("materialization_plan_id", "")),
        "dataset_name": str(plan.get("dataset_name", "")),
        "target_model_families": _safe_target_model_labels(plan.get("target_model_families")),
        "planned_output_formats": _safe_output_format_labels(plan.get("planned_output_formats")),
        "execution_ledger_id": str(ledger.get("execution_ledger_id", ledger_precheck.get("execution_ledger_id", ""))),
        "dry_run_id": str(ledger.get("dry_run_id", dry_run.get("dry_run_id", ""))),
        "execution_request_id": str(ledger.get("execution_request_id", request.get("execution_request_id", ""))),
        "corpus_id": str(ledger.get("corpus_id", request_plan.get("corpus_id", ""))),
        "source_dry_run_id": str(ledger.get("source_dry_run_id", request_plan.get("source_dry_run_id", ""))),
        "review_manifest_id": str(ledger.get("review_manifest_id", request_plan.get("review_manifest_id", ""))),
        "admission_request_id": str(ledger.get("admission_request_id", request_plan.get("admission_request_id", ""))),
        "source_execution_request_id": str(
            ledger.get("source_execution_request_id", request_plan.get("execution_request_id", ""))
        ),
        "quarantine_run_id": str(ledger.get("quarantine_run_id", request_plan.get("quarantine_run_id", ""))),
        "review_queue_id": str(ledger.get("review_queue_id", request_plan.get("review_queue_id", ""))),
        "property_candidate_manifest_id": str(
            ledger.get("property_candidate_manifest_id", request_plan.get("property_candidate_manifest_id", ""))
        ),
        "dataset_target": str(ledger.get("dataset_target", request_plan.get("dataset_target", ""))),
        "plan_status": str(plan.get("plan_status", "")),
        "ledger_precheck_status": str(ledger_precheck.get("precheck_status", "")),
        "ledger_status": str(ledger.get("execution_status", "")),
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
        "training_admitted": bool(plan.get("training_admitted") is True and ledger.get("training_admitted") is True),
        "training_dataset_materialized": bool(plan.get("training_dataset_materialized") is True),
        "dataset_artifact_created": bool(plan.get("dataset_artifact_created") is True),
        "phase1_status": str(plan.get("phase1_status", "")),
        "dataset_confirmation_changed": bool(plan.get("dataset_confirmation_changed") is True),
        "planned_dataset_record_count": len(plan_records),
        "ledger_record_count": len(ledger_records),
        "dry_run_record_count": len(dry_run_records),
        "execution_record_count": len(execution_records),
        "draft_record_count": len(draft_records),
        "planned_candidate_count": len(_safe_list(plan.get("planned_training_admission_candidate_record_ids"))),
        "planned_dataset_record_ids": [str(record.get("planned_dataset_record_id", "")) for record in plan_records],
        "ledger_record_ids": _safe_list(plan.get("ledger_record_ids")),
        "dry_run_record_ids": [str(record.get("dry_run_record_id", "")) for record in dry_run_records],
        "execution_record_ids": [str(record.get("execution_record_id", "")) for record in execution_records],
        "draft_record_ids": [str(record.get("draft_record_id", "")) for record in draft_records],
        "planned_training_admission_candidate_record_ids": _safe_list(
            plan.get("planned_training_admission_candidate_record_ids")
        ),
        "blocked_from_training_admission_record_ids": blocked_ids,
        "precheck_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _source_fields(paths: dict[str, str | Path], hashes: dict[str, str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, path in paths.items():
        fields[key] = Path(path).name or key
        fields[key.replace("_path", "_sha256")] = hashes[key.replace("_path", "_sha256")]
    return fields


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Dataset Materialization Plan Precheck Evidence",
            "",
            f"- Precheck status: `{summary['precheck_status']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Plan status: `{summary['plan_status']}`",
            f"- Dataset name: `{summary['dataset_name']}`",
            f"- Target model families: `{json.dumps(summary['target_model_families'])}`",
            f"- Planned output formats: `{json.dumps(summary['planned_output_formats'])}`",
            f"- Execution ledger id: `{summary['execution_ledger_id']}`",
            f"- Ledger status: `{summary['ledger_status']}`",
            f"- Dry-run id: `{summary['dry_run_id']}`",
            f"- Dry-run status: `{summary['dry_run_status']}`",
            f"- Execution request id: `{summary['execution_request_id']}`",
            f"- Readiness status: `{summary['readiness_status']}`",
            f"- Quarantine run id: `{summary['quarantine_run_id']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Source dry-run id: `{summary['source_dry_run_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Source execution request id: `{summary['source_execution_request_id']}`",
            f"- Planned candidate count: `{summary['planned_candidate_count']}`",
            f"- Planned dataset record count: `{summary['planned_dataset_record_count']}`",
            f"- Planned dataset record ids: `{json.dumps(summary['planned_dataset_record_ids'])}`",
            f"- Planned candidate ids: `{json.dumps(summary['planned_training_admission_candidate_record_ids'])}`",
            f"- Precheck errors: `{json.dumps(summary['precheck_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training dataset materialization plan precheck only.",
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
        raise CustomCorpusPropertyTrainingDatasetMaterializationPlanPrecheckError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyTrainingDatasetMaterializationPlanPrecheckError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyTrainingDatasetMaterializationPlanPrecheckError(f"{label} contains forbidden material")
    return payload


def _safe_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def _safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _safe_output_format_labels(value: Any) -> list[str]:
    return [item for item in _safe_list(value) if item in _ALLOWED_OUTPUT_FORMATS]


def _safe_target_model_labels(value: Any) -> list[str]:
    return [item for item in _safe_list(value) if item in _ALLOWED_TARGET_MODEL_FAMILIES]


def _safe_sha_for_path(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except Exception:
        return ""


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


def _count_errors(errors: list[str], container: dict[str, Any], field: str, records: list[dict[str, Any]]) -> None:
    try:
        count = int(container.get(field, len(records)))
    except Exception:
        errors.append(f"{field}_mismatch")
        return
    if count != len(records):
        errors.append(f"{field}_mismatch")


def _record_values_are_safe(
    record: dict[str, Any],
    *,
    allowed_labels: set[str],
    allowed_list_fields: set[str],
) -> bool:
    for key, value in record.items():
        if key.endswith("_sha256"):
            if value and not _SHA_RE.match(str(value)):
                return False
            continue
        if isinstance(value, str) and key not in allowed_labels:
            if value and not _is_safe_id(value):
                return False
        if isinstance(value, list):
            if key not in allowed_list_fields:
                return False
            if not all(isinstance(item, str) and _is_safe_id(item) for item in value):
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
    input_markers = tuple(
        marker
        for marker in _FORBIDDEN_MARKERS
        if marker not in {".csv", ".jsonl", ".parquet", ".lmdb"}
    )
    if any(marker.lower() in lowered for marker in input_markers):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_dataset_materialization_plan_precheck_redaction_failed"],
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
