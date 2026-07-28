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


_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_dry_run_precheck.v1"
_DRY_RUN_REPORT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_dry_run.v1"
_DRY_RUN_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_dry_run_summary.v1"
_ROW_CONTRACT_PRECHECK_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract_precheck.v1"
_ROW_CONTRACT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract.v1"
_ROW_CONTRACT_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract_builder.v1"
_PLAN_PRECHECK_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_plan_precheck.v1"
_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_plan.v1"
_PLANNER_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_planner.v1"
_LEDGER_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger_precheck.v1"
_LEDGER_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger.v1"
_LEDGER_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger_summary.v1"
_TRAINING_DRY_RUN_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_dry_run_precheck.v1"
_TRAINING_DRY_RUN_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_dry_run.v1"
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

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
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
)
_INPUT_FORBIDDEN_MARKERS = (
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
    "x-amz-signature",
    "signature=",
    "signedurl",
    "signed-url",
)
_MODEL_FAMILY_LABELS = {"generic_property_predictor", "unimol", "dpa3"}
_OUTPUT_FORMAT_LABELS = {"jsonl", "parquet", "lmdb", "csv"}
_ALLOWED_ROW_PREVIEW_KEYS = {
    "row_preview_id",
    "contract_record_reference_id",
    "planned_dataset_record_id",
    "ledger_record_id",
    "candidate_record_id",
    "record_id",
    "document_id",
    "field_name",
    "dataset_name",
    "contract_version_label",
    "row_contract_id",
    "would_materialize_row",
    "row_preview_status",
    "required_field_count",
    "optional_field_count",
    "required_row_fields",
    "optional_row_fields",
    "missing_required_fields",
    "missing_optional_fields",
    "quality_flag_labels",
    "target_model_families",
    "planned_output_formats",
    "training_admitted",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "phase1_status",
    "dataset_confirmation_changed",
    "source_artifact_sha256",
    "review_artifact_sha256",
    "admission_request_sha256",
    "training_admission_execution_ledger_sha256",
    "training_dataset_materialization_plan_sha256",
    "training_dataset_row_contract_sha256",
    "training_dataset_row_contract_precheck_sha256",
    "quarantine_candidate_records_sha256",
}


class PropertyTrainingDatasetMaterializationDryRunPrecheckError(ValueError):
    pass


def precheck_property_training_dataset_materialization_dry_run(
    *,
    training_dataset_materialization_dry_run_report_path: str | Path,
    training_dataset_materialization_dry_run_summary_path: str | Path,
    training_dataset_row_contract_precheck_path: str | Path,
    training_dataset_row_contract_path: str | Path,
    training_dataset_row_contract_summary_path: str | Path,
    training_dataset_materialization_plan_precheck_path: str | Path,
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
    require_dry_run_passed: bool = True,
    allow_dry_run_needs_review: bool = False,
    minimum_row_previews: int = 1,
) -> dict[str, Any]:
    payloads = _load_payloads(
        training_dataset_materialization_dry_run_report_path=training_dataset_materialization_dry_run_report_path,
        training_dataset_materialization_dry_run_summary_path=training_dataset_materialization_dry_run_summary_path,
        training_dataset_row_contract_precheck_path=training_dataset_row_contract_precheck_path,
        training_dataset_row_contract_path=training_dataset_row_contract_path,
        training_dataset_row_contract_summary_path=training_dataset_row_contract_summary_path,
        training_dataset_materialization_plan_precheck_path=training_dataset_materialization_plan_precheck_path,
        training_dataset_materialization_plan_path=training_dataset_materialization_plan_path,
        training_dataset_materialization_planner_summary_path=training_dataset_materialization_planner_summary_path,
        training_admission_execution_ledger_precheck_path=training_admission_execution_ledger_precheck_path,
        training_admission_execution_ledger_path=training_admission_execution_ledger_path,
        training_admission_execution_ledger_summary_path=training_admission_execution_ledger_summary_path,
        training_admission_execution_dry_run_precheck_path=training_admission_execution_dry_run_precheck_path,
        training_admission_execution_dry_run_report_path=training_admission_execution_dry_run_report_path,
        training_admission_execution_request_path=training_admission_execution_request_path,
        training_admission_execution_request_summary_path=training_admission_execution_request_summary_path,
        training_admission_execution_request_preflight_path=training_admission_execution_request_preflight_path,
        training_admission_request_draft_path=training_admission_request_draft_path,
        training_admission_request_draft_summary_path=training_admission_request_draft_summary_path,
        training_admission_request_draft_precheck_path=training_admission_request_draft_precheck_path,
        training_admission_request_plan_path=training_admission_request_plan_path,
        training_admission_request_preflight_path=training_admission_request_preflight_path,
        training_admission_readiness_summary_path=training_admission_readiness_summary_path,
        quarantine_candidate_preflight_summary_path=quarantine_candidate_preflight_summary_path,
        quarantine_candidate_records_path=quarantine_candidate_records_path,
    )
    paths = payloads["paths"]
    hashes = {key.replace("_path", "_sha256"): _safe_sha_for_path(path) for key, path in paths.items()}
    errors: list[str] = []
    warnings: list[str] = []
    _append_errors(errors, _schema_errors(payloads))
    _append_errors(
        errors,
        _status_errors(
            payloads,
            require_dry_run_passed=require_dry_run_passed,
            allow_dry_run_needs_review=allow_dry_run_needs_review,
            warnings=warnings,
        ),
    )
    _append_errors(errors, _summary_section_errors(payloads))
    _append_errors(errors, _hash_errors(payloads, hashes))
    _append_errors(errors, _id_errors(payloads))
    _append_errors(errors, _record_errors(payloads, minimum_row_previews))
    _append_errors(errors, _sha_format_errors(payloads))
    status = "blocked" if errors else "needs_review" if warnings else "passed"
    summary = _summary(status, payloads, paths, hashes, _stable_unique(errors), _stable_unique(warnings))
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
        summary = precheck_property_training_dataset_materialization_dry_run(
            training_dataset_materialization_dry_run_report_path=args.training_dataset_materialization_dry_run_report,
            training_dataset_materialization_dry_run_summary_path=args.training_dataset_materialization_dry_run_summary,
            training_dataset_row_contract_precheck_path=args.training_dataset_row_contract_precheck,
            training_dataset_row_contract_path=args.training_dataset_row_contract,
            training_dataset_row_contract_summary_path=args.training_dataset_row_contract_summary,
            training_dataset_materialization_plan_precheck_path=args.training_dataset_materialization_plan_precheck,
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
            require_dry_run_passed=args.require_dry_run_passed,
            allow_dry_run_needs_review=args.allow_dry_run_needs_review,
            minimum_row_previews=args.minimum_row_previews,
        )
    except Exception as exc:
        err.write(f"property training dataset materialization dry-run precheck invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("precheck_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run_precheck",
        description="Precheck a property training dataset materialization dry-run package without writing datasets.",
    )
    parser.add_argument("--training-dataset-materialization-dry-run-report", required=True)
    parser.add_argument("--training-dataset-materialization-dry-run-summary", required=True)
    parser.add_argument("--training-dataset-row-contract-precheck", required=True)
    parser.add_argument("--training-dataset-row-contract", required=True)
    parser.add_argument("--training-dataset-row-contract-summary", required=True)
    parser.add_argument("--training-dataset-materialization-plan-precheck", required=True)
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
    parser.add_argument("--require-dry-run-passed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-dry-run-needs-review", action="store_true")
    parser.add_argument("--minimum-row-previews", type=int, default=1)
    return parser


def _load_payloads(**paths: str | Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {"paths": paths}
    for key, path in paths.items():
        payloads[key.removesuffix("_path")] = _read_safe_json_dict(path, key.removesuffix("_path"))
    return payloads


def _schema_errors(payloads: dict[str, Any]) -> list[str]:
    expected = {
        "training_dataset_materialization_dry_run_report": (
            "training_dataset_materialization_dry_run_report_schema_invalid",
            _DRY_RUN_REPORT_SCHEMA_VERSION,
        ),
        "training_dataset_materialization_dry_run_summary": (
            "training_dataset_materialization_dry_run_summary_schema_invalid",
            _DRY_RUN_SUMMARY_SCHEMA_VERSION,
        ),
        "training_dataset_row_contract_precheck": (
            "training_dataset_row_contract_precheck_schema_invalid",
            _ROW_CONTRACT_PRECHECK_SCHEMA_VERSION,
        ),
        "training_dataset_row_contract": ("training_dataset_row_contract_schema_invalid", _ROW_CONTRACT_SCHEMA_VERSION),
        "training_dataset_row_contract_summary": (
            "training_dataset_row_contract_summary_schema_invalid",
            _ROW_CONTRACT_SUMMARY_SCHEMA_VERSION,
        ),
        "training_dataset_materialization_plan_precheck": (
            "training_dataset_materialization_plan_precheck_schema_invalid",
            _PLAN_PRECHECK_SCHEMA_VERSION,
        ),
        "training_dataset_materialization_plan": ("training_dataset_materialization_plan_schema_invalid", _PLAN_SCHEMA_VERSION),
        "training_dataset_materialization_planner_summary": (
            "training_dataset_materialization_planner_summary_schema_invalid",
            _PLANNER_SUMMARY_SCHEMA_VERSION,
        ),
        "training_admission_execution_ledger_precheck": (
            "training_admission_execution_ledger_precheck_schema_invalid",
            _LEDGER_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_admission_execution_ledger": ("training_admission_execution_ledger_schema_invalid", _LEDGER_SCHEMA_VERSION),
        "training_admission_execution_ledger_summary": (
            "training_admission_execution_ledger_summary_schema_invalid",
            _LEDGER_SUMMARY_SCHEMA_VERSION,
        ),
        "training_admission_execution_dry_run_precheck": (
            "training_admission_execution_dry_run_precheck_schema_invalid",
            _TRAINING_DRY_RUN_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_admission_execution_dry_run_report": (
            "training_admission_execution_dry_run_schema_invalid",
            _TRAINING_DRY_RUN_SCHEMA_VERSION,
        ),
        "training_admission_execution_request": ("training_admission_execution_request_schema_invalid", _REQUEST_SCHEMA_VERSION),
        "training_admission_execution_request_summary": (
            "training_admission_execution_request_summary_schema_invalid",
            _REQUEST_SUMMARY_SCHEMA_VERSION,
        ),
        "training_admission_execution_request_preflight": (
            "training_admission_execution_request_preflight_schema_invalid",
            _EXECUTION_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_admission_request_draft": ("training_admission_request_draft_schema_invalid", _DRAFT_SCHEMA_VERSION),
        "training_admission_request_draft_summary": (
            "training_admission_request_draft_summary_schema_invalid",
            _DRAFT_SUMMARY_SCHEMA_VERSION,
        ),
        "training_admission_request_draft_precheck": (
            "training_admission_request_draft_precheck_schema_invalid",
            _DRAFT_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_admission_request_plan": ("training_admission_request_plan_schema_invalid", _REQUEST_PLAN_SCHEMA_VERSION),
        "training_admission_request_preflight": (
            "training_admission_request_preflight_schema_invalid",
            _REQUEST_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_admission_readiness_summary": ("training_admission_readiness_schema_invalid", _READINESS_SCHEMA_VERSION),
        "quarantine_candidate_preflight_summary": (
            "quarantine_candidate_preflight_schema_invalid",
            _QUARANTINE_PREFLIGHT_SCHEMA_VERSION,
        ),
        "quarantine_candidate_records": ("quarantine_candidate_schema_invalid", _QUARANTINE_CANDIDATE_SCHEMA_VERSION),
    }
    return [
        error
        for key, (error, schema) in expected.items()
        if payloads[key].get("schema_version") != schema
    ]


def _status_errors(
    payloads: dict[str, Any],
    *,
    require_dry_run_passed: bool,
    allow_dry_run_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    status_checks = (
        ("training_dataset_materialization_dry_run_report", "dry_run_status", "training_dataset_materialization_dry_run", "passed"),
        ("training_dataset_materialization_dry_run_summary", "dry_run_status", "training_dataset_materialization_dry_run_summary", "passed"),
        ("training_dataset_row_contract_precheck", "precheck_status", "training_dataset_row_contract_precheck", "passed"),
        ("training_dataset_row_contract", "contract_status", "training_dataset_row_contract", "written"),
        ("training_dataset_row_contract_summary", "contract_status", "training_dataset_row_contract_summary", "written"),
        ("training_dataset_materialization_plan_precheck", "precheck_status", "training_dataset_materialization_plan_precheck", "passed"),
        ("training_dataset_materialization_plan", "plan_status", "training_dataset_materialization_plan", "planned"),
        ("training_dataset_materialization_planner_summary", "plan_status", "training_dataset_materialization_planner_summary", "planned"),
        ("training_admission_execution_ledger_precheck", "precheck_status", "training_admission_execution_ledger_precheck", "passed"),
        ("training_admission_execution_ledger", "execution_status", "training_admission_execution_ledger", "committed"),
        ("training_admission_execution_ledger_summary", "execution_status", "training_admission_execution_ledger_summary", "committed"),
        ("training_admission_execution_dry_run_precheck", "preflight_status", "training_admission_execution_dry_run_precheck", "passed"),
        ("training_admission_execution_dry_run_report", "dry_run_status", "training_admission_execution_dry_run", "passed"),
        ("training_admission_execution_request", "request_status", "training_admission_execution_request", "written"),
        ("training_admission_execution_request_summary", "request_status", "training_admission_execution_request_summary", "written"),
        ("training_admission_execution_request_preflight", "preflight_status", "training_admission_execution_request_preflight", "passed"),
        ("training_admission_request_draft", "draft_status", "training_admission_request_draft", "written"),
        ("training_admission_request_draft_summary", "draft_status", "training_admission_request_draft_summary", "written"),
        ("training_admission_request_draft_precheck", "precheck_status", "training_admission_request_draft_precheck", "passed"),
        ("training_admission_request_plan", "planner_status", "training_admission_request_plan", "planned"),
        ("training_admission_request_preflight", "preflight_status", "training_admission_request_preflight", "passed"),
        ("training_admission_readiness_summary", "readiness_status", "training_admission_readiness", "ready"),
        ("quarantine_candidate_preflight_summary", "preflight_status", "quarantine_candidate_preflight", "passed"),
    )
    allow = allow_dry_run_needs_review or not require_dry_run_passed
    for key, field, prefix, required in status_checks:
        _append_errors(errors, _status_field_errors(payloads[key], field, prefix, required, allow, warnings))
    report = payloads["training_dataset_materialization_dry_run_report"]
    summary = payloads["training_dataset_materialization_dry_run_summary"]
    if report.get("dry_run_status") != summary.get("dry_run_status"):
        errors.append("dry_run_status_mismatch")
    if report.get("dry_run_mode") != "training_dataset_materialization_dry_run_only":
        errors.append("training_dataset_materialization_dry_run_mode_invalid")
    if payloads["training_dataset_row_contract"].get("contract_mode") != "training_dataset_row_contract_only":
        errors.append("training_dataset_row_contract_mode_invalid")
    if payloads["training_dataset_materialization_plan"].get("plan_mode") != "training_dataset_materialization_plan_only":
        errors.append("training_dataset_materialization_plan_mode_invalid")
    for key in _payload_keys(payloads):
        container = payloads[key]
        if container.get("training_dataset_materialized") is True:
            errors.append("training_dataset_materialized")
        if container.get("dataset_artifact_created") is True:
            errors.append("dataset_artifact_created")
        if container.get("phase1_status") not in {"", None, "not_run"}:
            errors.append("phase1_ran")
        if container.get("dataset_confirmation_changed") not in {"", None, False}:
            errors.append("dataset_confirmation_changed")
    for key in (
        "training_dataset_materialization_dry_run_report",
        "training_dataset_materialization_dry_run_summary",
        "training_dataset_row_contract_precheck",
        "training_dataset_row_contract",
        "training_dataset_row_contract_summary",
        "training_dataset_materialization_plan_precheck",
        "training_dataset_materialization_plan",
        "training_dataset_materialization_planner_summary",
        "training_admission_execution_ledger_precheck",
        "training_admission_execution_ledger",
        "training_admission_execution_ledger_summary",
    ):
        if payloads[key].get("training_admitted") not in {"", None, True}:
            errors.append("training_not_admitted")
    for key, error_field, error_code in (
        ("training_dataset_materialization_dry_run_report", "dry_run_errors", "training_dataset_materialization_dry_run_has_errors"),
        ("training_dataset_materialization_dry_run_summary", "dry_run_errors", "training_dataset_materialization_dry_run_summary_has_errors"),
        ("training_dataset_row_contract_precheck", "precheck_errors", "training_dataset_row_contract_precheck_has_errors"),
        ("training_dataset_row_contract", "contract_errors", "training_dataset_row_contract_has_errors"),
        ("training_dataset_materialization_plan", "planning_errors", "training_dataset_materialization_plan_has_errors"),
        ("training_admission_execution_ledger", "execution_errors", "training_admission_execution_ledger_has_errors"),
    ):
        if payloads[key].get(error_field):
            errors.append(error_code)
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


def _summary_section_errors(payloads: dict[str, Any]) -> list[str]:
    report = payloads["training_dataset_materialization_dry_run_report"]
    contract = payloads["training_dataset_row_contract"]
    errors: list[str] = []
    field_summary = report.get("field_coverage_summary")
    if not isinstance(field_summary, dict):
        errors.append("field_coverage_summary_missing")
    else:
        if set(_safe_list(field_summary.get("required_row_fields"))) != set(_safe_list(contract.get("required_row_fields"))):
            errors.append("field_coverage_required_fields_mismatch")
        if set(_safe_list(field_summary.get("optional_row_fields"))) != set(_safe_list(contract.get("optional_row_fields"))):
            errors.append("field_coverage_optional_fields_mismatch")
        if _value_contains_unsafe_material(field_summary):
            errors.append("field_coverage_summary_contains_unsafe_value")
    model_summary = report.get("model_family_compatibility_summary")
    if not isinstance(model_summary, dict):
        errors.append("model_family_compatibility_summary_missing")
    else:
        model_counts = model_summary.get("counts_by_model_family")
        if not isinstance(model_counts, dict) or not _MODEL_FAMILY_LABELS.issubset(set(model_counts)):
            errors.append("model_family_compatibility_summary_missing_label")
        if model_summary.get("conformers_generated") is not False:
            errors.append("conformers_generated")
        if model_summary.get("dpa3_structures_generated") is not False:
            errors.append("dpa3_structures_generated")
    output_summary = report.get("output_format_compatibility_summary")
    if not isinstance(output_summary, dict):
        errors.append("output_format_compatibility_summary_missing")
    else:
        output_counts = output_summary.get("counts_by_output_format")
        if not isinstance(output_counts, dict) or not _OUTPUT_FORMAT_LABELS.issubset(set(output_counts)):
            errors.append("output_format_compatibility_summary_missing_label")
        for label in sorted(_OUTPUT_FORMAT_LABELS):
            if output_summary.get(f"{label}_created") is not False:
                errors.append(f"{label}_created")
    return _stable_unique(errors)


def _hash_errors(payloads: dict[str, Any], hashes: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for key in _payload_keys(payloads):
        container = payloads[key]
        for hash_key, actual_hash in hashes.items():
            if container.get(hash_key) and container.get(hash_key) != actual_hash:
                errors.append(f"{hash_key}_mismatch")
    return _stable_unique(errors)


def _id_errors(payloads: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    groups = {
        "corpus_id": (_payload_keys(payloads), ("corpus_id",)),
        "materialization_dry_run_id": (
            ("training_dataset_materialization_dry_run_report", "training_dataset_materialization_dry_run_summary"),
            ("materialization_dry_run_id",),
        ),
        "materialization_plan_id": (
            (
                "training_dataset_materialization_dry_run_report",
                "training_dataset_materialization_dry_run_summary",
                "training_dataset_row_contract_precheck",
                "training_dataset_row_contract",
                "training_dataset_row_contract_summary",
                "training_dataset_materialization_plan_precheck",
                "training_dataset_materialization_plan",
                "training_dataset_materialization_planner_summary",
            ),
            ("materialization_plan_id",),
        ),
        "row_contract_id": (
            (
                "training_dataset_materialization_dry_run_report",
                "training_dataset_materialization_dry_run_summary",
                "training_dataset_row_contract_precheck",
                "training_dataset_row_contract",
                "training_dataset_row_contract_summary",
            ),
            ("row_contract_id",),
        ),
        "execution_ledger_id": (_payload_keys(payloads), ("execution_ledger_id",)),
        "execution_request_id": (
            (
                "training_dataset_materialization_dry_run_report",
                "training_dataset_materialization_dry_run_summary",
                "training_dataset_row_contract_precheck",
                "training_dataset_row_contract",
                "training_dataset_row_contract_summary",
                "training_dataset_materialization_plan_precheck",
                "training_dataset_materialization_plan",
                "training_dataset_materialization_planner_summary",
                "training_admission_execution_ledger_precheck",
                "training_admission_execution_ledger",
                "training_admission_execution_ledger_summary",
                "training_admission_execution_dry_run_precheck",
                "training_admission_execution_dry_run_report",
                "training_admission_execution_request",
                "training_admission_execution_request_summary",
                "training_admission_execution_request_preflight",
            ),
            ("execution_request_id",),
        ),
        "dataset_name": (
            (
                "training_dataset_materialization_dry_run_report",
                "training_dataset_materialization_dry_run_summary",
                "training_dataset_row_contract_precheck",
                "training_dataset_row_contract",
                "training_dataset_row_contract_summary",
                "training_dataset_materialization_plan_precheck",
                "training_dataset_materialization_plan",
            ),
            ("dataset_name",),
        ),
    }
    for logical, (payload_keys, fields) in groups.items():
        values = _group_values(payloads, payload_keys, fields)
        if len(set(values)) > 1:
            errors.append(f"{logical}_mismatch")
        if any(value and not _is_safe_id(value) for value in values):
            errors.append(f"{logical}_invalid")
    return _stable_unique(errors)


def _record_errors(payloads: dict[str, Any], minimum_row_previews: int) -> list[str]:
    errors: list[str] = []
    report = payloads["training_dataset_materialization_dry_run_report"]
    summary = payloads["training_dataset_materialization_dry_run_summary"]
    plan = payloads["training_dataset_materialization_plan"]
    contract = payloads["training_dataset_row_contract"]
    ledger = payloads["training_admission_execution_ledger"]
    request_plan = payloads["training_admission_request_plan"]
    request_preflight = payloads["training_admission_request_preflight"]
    readiness = payloads["training_admission_readiness_summary"]
    previews = _safe_records(report.get("row_previews"))
    references = _safe_records(contract.get("contract_record_references"))
    plan_records = _safe_records(plan.get("planned_dataset_records"))
    ledger_records = _safe_records(ledger.get("ledger_records"))
    if len(previews) < max(minimum_row_previews, 1):
        errors.append("minimum_row_preview_count_not_met")
    if not previews:
        errors.append("no_row_previews")
    if int(report.get("row_preview_count", -1)) != len(previews):
        errors.append("row_preview_count_mismatch")
    if int(summary.get("row_preview_count", -1)) != len(previews):
        errors.append("row_preview_count_mismatch")
    preview_ids = [str(preview.get("row_preview_id", "")) for preview in previews]
    if _safe_list(report.get("row_preview_ids")) != preview_ids:
        errors.append("row_preview_ids_mismatch")
    if _safe_list(summary.get("row_preview_ids")) != preview_ids:
        errors.append("row_preview_ids_mismatch")
    if int(report.get("contract_record_reference_count", -1)) != len(references):
        errors.append("contract_record_reference_count_mismatch")
    if int(report.get("planned_dataset_record_count", -1)) != len(plan_records):
        errors.append("planned_dataset_record_count_mismatch")
    preview_reference_ids = [str(preview.get("contract_record_reference_id", "")) for preview in previews]
    reference_ids = [str(reference.get("contract_record_reference_id", "")) for reference in references]
    if preview_reference_ids != reference_ids:
        errors.append("contract_record_reference_ids_mismatch")
    plan_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in plan_records]
    ref_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in references]
    preview_dataset_ids = [str(preview.get("planned_dataset_record_id", "")) for preview in previews]
    if ref_dataset_ids != plan_dataset_ids or preview_dataset_ids != plan_dataset_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    ledger_ids = [str(record.get("ledger_record_id", "")) for record in ledger_records]
    ref_ledger_ids = [str(record.get("ledger_record_id", "")) for record in references]
    preview_ledger_ids = [str(preview.get("ledger_record_id", "")) for preview in previews]
    if set(ref_ledger_ids) != set(ledger_ids) or set(preview_ledger_ids) != set(ledger_ids):
        errors.append("ledger_record_ids_mismatch")
    planned_candidates = _safe_list(plan.get("planned_training_admission_candidate_record_ids"))
    report_candidates = _safe_list(report.get("planned_training_admission_candidate_record_ids"))
    summary_candidates = _safe_list(summary.get("planned_training_admission_candidate_record_ids"))
    ref_candidates = [str(record.get("candidate_record_id", "")) for record in references]
    preview_candidates = [str(preview.get("candidate_record_id", "")) for preview in previews]
    ledger_candidates = [str(record.get("candidate_record_id", "")) for record in ledger_records]
    if (
        set(report_candidates) != set(planned_candidates)
        or set(summary_candidates) != set(planned_candidates)
        or set(ref_candidates) != set(planned_candidates)
        or set(preview_candidates) != set(planned_candidates)
        or set(ledger_candidates) != set(planned_candidates)
    ):
        errors.append("planned_candidate_ids_mismatch")
    excluded = set(_safe_list(request_plan.get("exclude_record_ids")))
    excluded.update(_safe_list(request_preflight.get("exclude_record_ids")))
    blocked = set(_safe_list(request_plan.get("blocked_record_ids")))
    blocked.update(_safe_list(request_plan.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(request_preflight.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(readiness.get("blocked_from_training_admission_record_ids")))
    needs_review = set(_safe_list(request_plan.get("needs_review_record_ids")))
    needs_review.update(_safe_list(request_preflight.get("needs_review_record_ids")))
    for preview in previews:
        candidate_id = str(preview.get("candidate_record_id", ""))
        record_id = str(preview.get("record_id", ""))
        if preview.get("would_materialize_row") is not True:
            errors.append("row_preview_not_marked_would_materialize")
        if preview.get("row_preview_status") != "would_materialize":
            errors.append("row_preview_status_invalid")
        if preview.get("training_admitted") is not True:
            errors.append("training_not_admitted")
        if preview.get("training_dataset_materialized") is not False:
            errors.append("training_dataset_materialized")
        if preview.get("dataset_artifact_created") is not False:
            errors.append("dataset_artifact_created")
        if preview.get("phase1_status") != "not_run":
            errors.append("phase1_ran")
        if preview.get("dataset_confirmation_changed") is not False:
            errors.append("dataset_confirmation_changed")
        if candidate_id in excluded or record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if candidate_id in blocked or record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if candidate_id in needs_review or record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
        if not _row_preview_is_safe(preview):
            errors.append("row_preview_contains_unsafe_value")
    return _stable_unique(errors)


def _summary(
    status: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    report = payloads["training_dataset_materialization_dry_run_report"]
    summary = payloads["training_dataset_materialization_dry_run_summary"]
    plan = payloads["training_dataset_materialization_plan"]
    contract = payloads["training_dataset_row_contract"]
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": status,
        **_source_fields(paths, hashes),
        **_source_status_fields(payloads),
        "materialization_dry_run_id": str(report.get("materialization_dry_run_id", "")),
        "dataset_name": str(report.get("dataset_name", "")),
        "row_contract_id": str(report.get("row_contract_id", "")),
        "contract_version_label": str(report.get("contract_version_label", "")),
        "materialization_plan_id": str(report.get("materialization_plan_id", "")),
        "execution_ledger_id": str(report.get("execution_ledger_id", "")),
        "execution_request_id": str(report.get("execution_request_id", "")),
        "corpus_id": str(report.get("corpus_id", "")),
        "admission_request_id": str(report.get("admission_request_id", "")),
        "review_manifest_id": str(report.get("review_manifest_id", "")),
        "review_queue_id": str(report.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(report.get("property_candidate_manifest_id", "")),
        "training_admitted": True if status in {"passed", "needs_review"} else bool(report.get("training_admitted") is True),
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "row_preview_count": len(_safe_records(report.get("row_previews"))),
        "planned_dataset_record_count": len(_safe_records(plan.get("planned_dataset_records"))),
        "contract_record_reference_count": len(_safe_records(contract.get("contract_record_references"))),
        "row_preview_ids": _safe_list(report.get("row_preview_ids")),
        "planned_dataset_record_ids": [
            str(record.get("planned_dataset_record_id", ""))
            for record in _safe_records(plan.get("planned_dataset_records"))
        ],
        "planned_training_admission_candidate_record_ids": _safe_list(
            summary.get("planned_training_admission_candidate_record_ids")
        ),
        "field_coverage_summary": _safe_field_coverage_summary(report.get("field_coverage_summary")),
        "model_family_compatibility_summary": _safe_model_summary(report.get("model_family_compatibility_summary")),
        "output_format_compatibility_summary": _safe_output_summary(report.get("output_format_compatibility_summary")),
        "precheck_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _safe_field_coverage_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "required_field_count": value.get("required_field_count", 0),
        "optional_field_count": value.get("optional_field_count", 0),
        "required_row_fields": _safe_list(value.get("required_row_fields")),
        "optional_row_fields": _safe_list(value.get("optional_row_fields")),
        "row_preview_count": value.get("row_preview_count", 0),
        "missing_required_field_counts": value.get("missing_required_field_counts", {}),
        "missing_optional_field_counts": value.get("missing_optional_field_counts", {}),
    }


def _safe_model_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "counts_by_model_family": value.get("counts_by_model_family", {}),
        "unimol_requires_future_conformer_generation": value.get(
            "unimol_requires_future_conformer_generation",
            False,
        ),
        "dpa3_requires_future_structure_generation": value.get(
            "dpa3_requires_future_structure_generation",
            False,
        ),
        "conformers_generated": value.get("conformers_generated", False),
        "dpa3_structures_generated": value.get("dpa3_structures_generated", False),
    }


def _safe_output_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "counts_by_output_format": value.get("counts_by_output_format", {}),
        "jsonl_created": value.get("jsonl_created", False),
        "parquet_created": value.get("parquet_created", False),
        "lmdb_created": value.get("lmdb_created", False),
        "csv_created": value.get("csv_created", False),
    }


def _source_status_fields(payloads: dict[str, Any]) -> dict[str, Any]:
    report = payloads["training_dataset_materialization_dry_run_report"]
    return {
        "dry_run_status": str(report.get("dry_run_status", "")),
        "row_contract_precheck_status": str(report.get("row_contract_precheck_status", "")),
        "contract_status": str(report.get("contract_status", "")),
        "materialization_plan_precheck_status": str(report.get("materialization_plan_precheck_status", "")),
        "plan_status": str(report.get("plan_status", "")),
        "ledger_status": str(report.get("ledger_status", "")),
    }


def _source_fields(paths: dict[str, str | Path], hashes: dict[str, str]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, path in paths.items():
        fields[key] = Path(path).name or key
        fields[key.replace("_path", "_sha256")] = hashes[key.replace("_path", "_sha256")]
    return fields


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Dataset Materialization Dry-Run Precheck Evidence",
            "",
            f"- Precheck status: `{summary['precheck_status']}`",
            f"- Materialization dry-run id: `{summary['materialization_dry_run_id']}`",
            f"- Dataset name: `{summary['dataset_name']}`",
            f"- Row contract id: `{summary['row_contract_id']}`",
            f"- Row preview count: `{summary['row_preview_count']}`",
            f"- Precheck errors: `{json.dumps(summary['precheck_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training dataset materialization dry-run precheck only.",
            "- row previews are summaries only, not serialized training rows.",
            "- no training dataset artifact was created.",
            "- no training CSV/JSONL/Parquet/LMDB was created.",
            "- no candidate CSV/JSONL/Parquet/LMDB was created.",
            "- no conformers were generated.",
            "- no DPA3 structures were generated.",
            "- no Phase 1 was run.",
            "- DatasetConfirmation was not changed.",
            "- no model training or evaluation was run.",
            "",
        ]
    )


def _row_preview_is_safe(preview: dict[str, Any]) -> bool:
    if set(preview) - _ALLOWED_ROW_PREVIEW_KEYS:
        return False
    for key, value in preview.items():
        if key.endswith("_sha256"):
            if value and not _SHA_RE.match(str(value)):
                return False
            continue
        if key in {
            "would_materialize_row",
            "training_admitted",
            "training_dataset_materialized",
            "dataset_artifact_created",
            "dataset_confirmation_changed",
        }:
            if not isinstance(value, bool):
                return False
            continue
        if key in {"required_field_count", "optional_field_count"}:
            if not isinstance(value, int):
                return False
            continue
        if isinstance(value, list):
            if key in {"target_model_families", "planned_output_formats"}:
                if not all(isinstance(item, str) and _is_safe_id(item) for item in value):
                    return False
            elif key in {
                "required_row_fields",
                "optional_row_fields",
                "missing_required_fields",
                "missing_optional_fields",
                "quality_flag_labels",
            }:
                if not all(isinstance(item, str) and _is_safe_id(item) for item in value):
                    return False
            else:
                return False
            continue
        if isinstance(value, str) and value and not _is_safe_id(value):
            return False
        if _value_contains_unsafe_material(value):
            return False
    return True


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise PropertyTrainingDatasetMaterializationDryRunPrecheckError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetMaterializationDryRunPrecheckError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise PropertyTrainingDatasetMaterializationDryRunPrecheckError(f"{label} contains forbidden material")
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


def _group_values(payloads: dict[str, Any], payload_keys: tuple[str, ...] | list[str], fields: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for payload_key in payload_keys:
        for field in fields:
            value = str(payloads[payload_key].get(field, ""))
            if value:
                values.append(value)
                break
    return values


def _sha_format_errors(payloads: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in _payload_keys(payloads):
        for field, value in payloads[key].items():
            if field.endswith("_sha256") and value and not _SHA_RE.match(str(value)):
                errors.append(f"{field}_invalid")
    return _stable_unique(errors)


def _payload_keys(payloads: dict[str, Any]) -> list[str]:
    return [key for key in payloads if key != "paths"]


def _is_safe_id(value: str) -> bool:
    return bool(_SAFE_ID_RE.match(value))


def _contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _value_contains_unsafe_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    markers = (
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
        "raw article text",
        "raw table",
        "serialized training row",
        "serialized dataset row",
        "conformer block",
        "dpa3 structure",
    )
    if any(marker.lower() in lowered for marker in markers):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _input_contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _INPUT_FORBIDDEN_MARKERS):
        return True
    return False


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": "blocked",
        "precheck_errors": ["property_training_dataset_materialization_dry_run_precheck_redaction_failed"],
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
