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
from ai4s_agent.custom_corpus_materialization import sha256_file


_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_input_binding_plan_preflight.v1"
_BINDING_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_input_binding_plan.v1"
_BINDING_PLANNER_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_input_binding_planner.v1"
_WRITER_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_execution_request_preflight.v1"
_WRITER_REQUEST_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_execution_request.v1"
_WRITER_REQUEST_SUMMARY_SCHEMA_VERSION = (
    "custom_corpus_property_training_dataset_writer_execution_request_builder.v1"
)
_DRY_RUN_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_dry_run_precheck.v1"
_DRY_RUN_REPORT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_dry_run.v1"
_DRY_RUN_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_dry_run_summary.v1"
_ROW_CONTRACT_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract_precheck.v1"
_ROW_CONTRACT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract.v1"
_ROW_CONTRACT_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract_builder.v1"
_MATERIALIZATION_PLAN_PREFLIGHT_SCHEMA_VERSION = (
    "custom_corpus_property_training_dataset_materialization_plan_precheck.v1"
)
_MATERIALIZATION_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_plan.v1"
_MATERIALIZATION_PLANNER_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_planner.v1"
_LEDGER_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger_precheck.v1"
_LEDGER_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger.v1"
_LEDGER_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_ledger_summary.v1"
_TRAINING_DRY_RUN_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_dry_run_precheck.v1"
_TRAINING_DRY_RUN_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_dry_run.v1"
_TRAINING_EXECUTION_REQUEST_SCHEMA_VERSION = "custom_corpus_property_training_admission_execution_request.v1"
_TRAINING_EXECUTION_REQUEST_SUMMARY_SCHEMA_VERSION = (
    "custom_corpus_property_training_admission_execution_request_builder.v1"
)
_TRAINING_EXECUTION_PREFLIGHT_SCHEMA_VERSION = (
    "custom_corpus_property_training_admission_execution_request_preflight.v1"
)
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
_OUTPUT_FORMAT_LABELS = {"jsonl", "parquet", "lmdb", "csv"}
_ALLOWED_SOURCE_ARTIFACT_LABELS = {
    "writer_execution_request",
    "materialization_dry_run_report",
    "row_contract",
    "materialization_plan",
    "training_admission_execution_ledger",
    "quarantine_candidate_records",
}
_FIELD_BINDING_STATUSES = {"bound", "missing_source", "derived_later"}
_FINAL_FORBIDDEN_MARKERS = (
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
    "conformer block",
    "inchi=",
)
_VALUE_FORBIDDEN_MARKERS = _FINAL_FORBIDDEN_MARKERS + (
    "0.72",
    "c1=cc",
    "canonical_smiles_value",
    "serialized training row",
    "serialized dataset row",
    "full candidate payload",
    "full materialized payload",
    "full writer request payload",
    "dpa3 structure block",
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


class PropertyTrainingDatasetWriterInputBindingPlanPreflightError(ValueError):
    pass


def preflight_property_training_dataset_writer_input_binding_plan(
    *,
    training_dataset_writer_input_binding_plan_path: str | Path,
    training_dataset_writer_input_binding_planner_summary_path: str | Path,
    training_dataset_writer_execution_request_preflight_path: str | Path,
    training_dataset_writer_execution_request_path: str | Path,
    training_dataset_writer_execution_request_summary_path: str | Path,
    training_dataset_materialization_dry_run_precheck_path: str | Path,
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
    require_binding_plan_planned: bool = True,
    allow_binding_plan_needs_review: bool = False,
    minimum_binding_records: int = 1,
    require_all_required_fields_bound: bool = True,
) -> dict[str, Any]:
    payloads = _load_payloads(
        training_dataset_writer_input_binding_plan_path=training_dataset_writer_input_binding_plan_path,
        training_dataset_writer_input_binding_planner_summary_path=training_dataset_writer_input_binding_planner_summary_path,
        training_dataset_writer_execution_request_preflight_path=training_dataset_writer_execution_request_preflight_path,
        training_dataset_writer_execution_request_path=training_dataset_writer_execution_request_path,
        training_dataset_writer_execution_request_summary_path=training_dataset_writer_execution_request_summary_path,
        training_dataset_materialization_dry_run_precheck_path=training_dataset_materialization_dry_run_precheck_path,
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
    warnings: list[str] = []
    errors: list[str] = []
    _append_errors(errors, _schema_errors(payloads))
    _append_errors(
        errors,
        _status_errors(
            payloads,
            require_binding_plan_planned=require_binding_plan_planned,
            allow_needs_review=allow_binding_plan_needs_review,
            warnings=warnings,
        ),
    )
    _append_errors(errors, _hash_errors(payloads, hashes))
    _append_errors(errors, _id_errors(payloads))
    _append_errors(errors, _record_errors(payloads, minimum_binding_records))
    _append_errors(errors, _binding_record_errors(payloads, require_all_required_fields_bound, warnings))
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
        summary = preflight_property_training_dataset_writer_input_binding_plan(
            training_dataset_writer_input_binding_plan_path=args.training_dataset_writer_input_binding_plan,
            training_dataset_writer_input_binding_planner_summary_path=args.training_dataset_writer_input_binding_planner_summary,
            training_dataset_writer_execution_request_preflight_path=args.training_dataset_writer_execution_request_preflight,
            training_dataset_writer_execution_request_path=args.training_dataset_writer_execution_request,
            training_dataset_writer_execution_request_summary_path=args.training_dataset_writer_execution_request_summary,
            training_dataset_materialization_dry_run_precheck_path=args.training_dataset_materialization_dry_run_precheck,
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
            require_binding_plan_planned=args.require_binding_plan_planned,
            allow_binding_plan_needs_review=args.allow_binding_plan_needs_review,
            minimum_binding_records=args.minimum_binding_records,
            require_all_required_fields_bound=args.require_all_required_fields_bound,
        )
    except Exception as exc:
        err.write(f"property training dataset writer input binding plan preflight invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("preflight_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_plan_preflight",
        description="Preflight a property training dataset writer input binding plan without executing a writer.",
    )
    parser.add_argument("--training-dataset-writer-input-binding-plan", required=True)
    parser.add_argument("--training-dataset-writer-input-binding-planner-summary", required=True)
    parser.add_argument("--training-dataset-writer-execution-request-preflight", required=True)
    parser.add_argument("--training-dataset-writer-execution-request", required=True)
    parser.add_argument("--training-dataset-writer-execution-request-summary", required=True)
    parser.add_argument("--training-dataset-materialization-dry-run-precheck", required=True)
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
    parser.add_argument("--require-binding-plan-planned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-binding-plan-needs-review", action="store_true")
    parser.add_argument("--minimum-binding-records", type=int, default=1)
    parser.add_argument("--require-all-required-fields-bound", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _load_payloads(**paths: str | Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {"paths": paths}
    for key, path in paths.items():
        payloads[key.removesuffix("_path")] = _read_safe_json_dict(path, key.removesuffix("_path"))
    return payloads


def _schema_errors(payloads: dict[str, Any]) -> list[str]:
    expected = {
        "training_dataset_writer_input_binding_plan": (
            "training_dataset_writer_input_binding_plan_schema_invalid",
            _BINDING_PLAN_SCHEMA_VERSION,
        ),
        "training_dataset_writer_input_binding_planner_summary": (
            "training_dataset_writer_input_binding_planner_summary_schema_invalid",
            _BINDING_PLANNER_SCHEMA_VERSION,
        ),
        "training_dataset_writer_execution_request_preflight": (
            "training_dataset_writer_execution_request_preflight_schema_invalid",
            _WRITER_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_dataset_writer_execution_request": (
            "training_dataset_writer_execution_request_schema_invalid",
            _WRITER_REQUEST_SCHEMA_VERSION,
        ),
        "training_dataset_writer_execution_request_summary": (
            "training_dataset_writer_execution_request_summary_schema_invalid",
            _WRITER_REQUEST_SUMMARY_SCHEMA_VERSION,
        ),
        "training_dataset_materialization_dry_run_precheck": (
            "training_dataset_materialization_dry_run_precheck_schema_invalid",
            _DRY_RUN_PREFLIGHT_SCHEMA_VERSION,
        ),
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
            _ROW_CONTRACT_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_dataset_row_contract": ("training_dataset_row_contract_schema_invalid", _ROW_CONTRACT_SCHEMA_VERSION),
        "training_dataset_row_contract_summary": (
            "training_dataset_row_contract_summary_schema_invalid",
            _ROW_CONTRACT_SUMMARY_SCHEMA_VERSION,
        ),
        "training_dataset_materialization_plan_precheck": (
            "training_dataset_materialization_plan_precheck_schema_invalid",
            _MATERIALIZATION_PLAN_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_dataset_materialization_plan": (
            "training_dataset_materialization_plan_schema_invalid",
            _MATERIALIZATION_PLAN_SCHEMA_VERSION,
        ),
        "training_dataset_materialization_planner_summary": (
            "training_dataset_materialization_planner_summary_schema_invalid",
            _MATERIALIZATION_PLANNER_SCHEMA_VERSION,
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
        "training_admission_execution_request": (
            "training_admission_execution_request_schema_invalid",
            _TRAINING_EXECUTION_REQUEST_SCHEMA_VERSION,
        ),
        "training_admission_execution_request_summary": (
            "training_admission_execution_request_summary_schema_invalid",
            _TRAINING_EXECUTION_REQUEST_SUMMARY_SCHEMA_VERSION,
        ),
        "training_admission_execution_request_preflight": (
            "training_admission_execution_request_preflight_schema_invalid",
            _TRAINING_EXECUTION_PREFLIGHT_SCHEMA_VERSION,
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
    return [error for key, (error, schema) in expected.items() if payloads[key].get("schema_version") != schema]


def _status_errors(
    payloads: dict[str, Any],
    *,
    require_binding_plan_planned: bool,
    allow_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    allow = allow_needs_review or not require_binding_plan_planned
    status_checks = (
        ("training_dataset_writer_input_binding_plan", "planner_status", "training_dataset_writer_input_binding_plan", "planned"),
        ("training_dataset_writer_input_binding_planner_summary", "planner_status", "training_dataset_writer_input_binding_planner_summary", "planned"),
        ("training_dataset_writer_execution_request_preflight", "preflight_status", "training_dataset_writer_execution_request_preflight", "passed"),
        ("training_dataset_writer_execution_request", "request_status", "training_dataset_writer_execution_request", "written"),
        ("training_dataset_writer_execution_request_summary", "request_status", "training_dataset_writer_execution_request_summary", "written"),
        ("training_dataset_materialization_dry_run_precheck", "precheck_status", "training_dataset_materialization_dry_run_precheck", "passed"),
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
    for key, field, prefix, required in status_checks:
        _append_errors(errors, _status_field_errors(payloads[key], field, prefix, required, allow, warnings))
    plan = payloads["training_dataset_writer_input_binding_plan"]
    planner = payloads["training_dataset_writer_input_binding_planner_summary"]
    if plan.get("planner_status") != planner.get("planner_status"):
        errors.append("writer_input_binding_plan_status_mismatch")
    if plan.get("plan_mode") != "training_dataset_writer_input_binding_plan_only":
        errors.append("training_dataset_writer_input_binding_plan_mode_invalid")
    if plan.get("planner_errors") or planner.get("planner_errors"):
        errors.append("training_dataset_writer_input_binding_plan_has_errors")
    for key in _payload_keys(payloads):
        container = payloads[key]
        if container.get("writer_executed") is True:
            errors.append("writer_executed")
        if container.get("values_materialized") is True:
            errors.append("values_materialized")
        if container.get("training_dataset_materialized") is True:
            errors.append("training_dataset_materialized")
        if container.get("dataset_artifact_created") is True:
            errors.append("dataset_artifact_created")
        if container.get("phase1_status") not in {"", None, "not_run"}:
            errors.append("phase1_ran")
        if container.get("dataset_confirmation_changed") not in {"", None, False}:
            errors.append("dataset_confirmation_changed")
    for key in (
        "training_dataset_writer_input_binding_plan",
        "training_dataset_writer_input_binding_planner_summary",
        "training_dataset_writer_execution_request_preflight",
        "training_dataset_writer_execution_request",
        "training_dataset_writer_execution_request_summary",
        "training_dataset_materialization_dry_run_precheck",
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


def _hash_errors(payloads: dict[str, Any], hashes: dict[str, str]) -> list[str]:
    errors: list[str] = []
    canonical_hashes = {
        "training_dataset_writer_input_binding_plan_sha256": _sha256_payload(
            payloads["training_dataset_writer_input_binding_plan"]
        ),
        "training_dataset_writer_execution_request_sha256": _sha256_payload(
            payloads["training_dataset_writer_execution_request"]
        ),
    }
    for key in _payload_keys(payloads):
        container = payloads[key]
        for hash_key, actual_hash in hashes.items():
            if not container.get(hash_key):
                continue
            allowed_hashes = {actual_hash}
            if hash_key in canonical_hashes:
                allowed_hashes.add(canonical_hashes[hash_key])
            if container.get(hash_key) not in allowed_hashes:
                errors.append(f"{hash_key}_mismatch")
    return _stable_unique(errors)


def _id_errors(payloads: dict[str, Any]) -> list[str]:
    groups = {
        "corpus_id": (_payload_keys(payloads), ("corpus_id",)),
        "writer_input_binding_plan_id": (
            ("training_dataset_writer_input_binding_plan", "training_dataset_writer_input_binding_planner_summary"),
            ("writer_input_binding_plan_id",),
        ),
        "writer_execution_request_id": (
            (
                "training_dataset_writer_input_binding_plan",
                "training_dataset_writer_input_binding_planner_summary",
                "training_dataset_writer_execution_request_preflight",
                "training_dataset_writer_execution_request",
                "training_dataset_writer_execution_request_summary",
            ),
            ("writer_execution_request_id",),
        ),
        "materialization_dry_run_id": (
            (
                "training_dataset_writer_input_binding_plan",
                "training_dataset_writer_input_binding_planner_summary",
                "training_dataset_writer_execution_request_preflight",
                "training_dataset_writer_execution_request",
                "training_dataset_writer_execution_request_summary",
                "training_dataset_materialization_dry_run_precheck",
                "training_dataset_materialization_dry_run_report",
                "training_dataset_materialization_dry_run_summary",
            ),
            ("materialization_dry_run_id",),
        ),
        "materialization_plan_id": (
            (
                "training_dataset_writer_input_binding_plan",
                "training_dataset_writer_input_binding_planner_summary",
                "training_dataset_writer_execution_request_preflight",
                "training_dataset_writer_execution_request",
                "training_dataset_writer_execution_request_summary",
                "training_dataset_materialization_dry_run_precheck",
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
                "training_dataset_writer_input_binding_plan",
                "training_dataset_writer_input_binding_planner_summary",
                "training_dataset_writer_execution_request_preflight",
                "training_dataset_writer_execution_request",
                "training_dataset_writer_execution_request_summary",
                "training_dataset_materialization_dry_run_precheck",
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
                "training_dataset_writer_input_binding_plan",
                "training_dataset_writer_input_binding_planner_summary",
                "training_dataset_writer_execution_request_preflight",
                "training_dataset_writer_execution_request",
                "training_dataset_writer_execution_request_summary",
                "training_dataset_materialization_dry_run_precheck",
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
                "training_dataset_writer_input_binding_plan",
                "training_dataset_writer_input_binding_planner_summary",
                "training_dataset_writer_execution_request_preflight",
                "training_dataset_writer_execution_request",
                "training_dataset_writer_execution_request_summary",
                "training_dataset_materialization_dry_run_precheck",
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
    errors: list[str] = []
    for logical, (payload_keys, fields) in groups.items():
        values = _group_values(payloads, payload_keys, fields)
        if len(set(values)) > 1:
            errors.append(f"{logical}_mismatch")
        if any(value and not _is_safe_id(value) for value in values):
            errors.append(f"{logical}_invalid")
    return _stable_unique(errors)


def _record_errors(payloads: dict[str, Any], minimum_binding_records: int) -> list[str]:
    errors: list[str] = []
    plan = payloads["training_dataset_writer_input_binding_plan"]
    planner = payloads["training_dataset_writer_input_binding_planner_summary"]
    request = payloads["training_dataset_writer_execution_request"]
    request_summary = payloads["training_dataset_writer_execution_request_summary"]
    report = payloads["training_dataset_materialization_dry_run_report"]
    dry_summary = payloads["training_dataset_materialization_dry_run_summary"]
    contract = payloads["training_dataset_row_contract"]
    dataset_plan = payloads["training_dataset_materialization_plan"]
    ledger = payloads["training_admission_execution_ledger"]
    request_plan = payloads["training_admission_request_plan"]
    request_preflight = payloads["training_admission_request_preflight"]
    readiness = payloads["training_admission_readiness_summary"]
    binding_records = _safe_records(plan.get("binding_records"))
    writer_records = _safe_records(request.get("writer_request_records"))
    previews = _safe_records(report.get("row_previews"))
    references = _safe_records(contract.get("contract_record_references"))
    dataset_plan_records = _safe_records(dataset_plan.get("planned_dataset_records"))
    ledger_records = _safe_records(ledger.get("ledger_records"))
    if len(binding_records) < max(minimum_binding_records, 1):
        errors.append("minimum_binding_record_count_not_met")
    if not binding_records:
        errors.append("no_binding_records")
    if int(plan.get("binding_record_count", -1)) != len(binding_records):
        errors.append("binding_record_count_mismatch")
    if int(planner.get("binding_record_count", -1)) != len(binding_records):
        errors.append("binding_record_count_mismatch")
    binding_ids = [str(record.get("writer_input_binding_record_id", "")) for record in binding_records]
    if _safe_list(plan.get("binding_record_ids")) != binding_ids:
        errors.append("binding_record_ids_mismatch")
    if planner.get("binding_record_ids") and _safe_list(planner.get("binding_record_ids")) != binding_ids:
        errors.append("binding_record_ids_mismatch")
    writer_ids = [str(record.get("writer_request_record_id", "")) for record in writer_records]
    binding_writer_ids = [str(record.get("writer_request_record_id", "")) for record in binding_records]
    if binding_writer_ids != writer_ids:
        errors.append("writer_request_record_ids_mismatch")
    if _safe_list(plan.get("writer_request_record_ids")) != binding_writer_ids:
        errors.append("writer_request_record_ids_mismatch")
    if _safe_list(request.get("writer_request_record_ids")) != writer_ids:
        errors.append("writer_request_record_ids_mismatch")
    if _safe_list(request_summary.get("writer_request_record_ids")) != writer_ids:
        errors.append("writer_request_record_ids_mismatch")
    row_preview_ids = [str(record.get("row_preview_id", "")) for record in binding_records]
    report_preview_ids = [str(preview.get("row_preview_id", "")) for preview in previews]
    if row_preview_ids != report_preview_ids:
        errors.append("row_preview_ids_mismatch")
    if _safe_list(plan.get("row_preview_ids")) != row_preview_ids:
        errors.append("row_preview_ids_mismatch")
    if _safe_list(request.get("row_preview_ids")) != row_preview_ids:
        errors.append("row_preview_ids_mismatch")
    plan_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in dataset_plan_records]
    binding_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in binding_records]
    request_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in writer_records]
    reference_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in references]
    preview_dataset_ids = [str(preview.get("planned_dataset_record_id", "")) for preview in previews]
    if (
        binding_dataset_ids != plan_dataset_ids
        or request_dataset_ids != plan_dataset_ids
        or reference_dataset_ids != plan_dataset_ids
        or preview_dataset_ids != plan_dataset_ids
    ):
        errors.append("planned_dataset_record_ids_mismatch")
    ledger_ids = [str(record.get("ledger_record_id", "")) for record in ledger_records]
    binding_ledger_ids = [str(record.get("ledger_record_id", "")) for record in binding_records]
    if set(binding_ledger_ids) != set(ledger_ids):
        errors.append("ledger_record_ids_mismatch")
    planned_candidates = _safe_list(plan.get("planned_training_admission_candidate_record_ids"))
    binding_candidates = [str(record.get("candidate_record_id", "")) for record in binding_records]
    request_candidates = [str(record.get("candidate_record_id", "")) for record in writer_records]
    preview_candidates = [str(preview.get("candidate_record_id", "")) for preview in previews]
    reference_candidates = [str(reference.get("candidate_record_id", "")) for reference in references]
    ledger_candidates = [str(record.get("candidate_record_id", "")) for record in ledger_records]
    if not planned_candidates:
        errors.append("no_planned_candidates")
    if (
        set(binding_candidates) != set(planned_candidates)
        or set(request_candidates) != set(planned_candidates)
        or set(preview_candidates) != set(planned_candidates)
        or set(reference_candidates) != set(planned_candidates)
        or set(ledger_candidates) != set(planned_candidates)
        or set(_safe_list(request_summary.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates)
        or set(_safe_list(dry_summary.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates)
        or set(_safe_list(dataset_plan.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates)
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
    for record in binding_records:
        candidate_id = str(record.get("candidate_record_id", ""))
        record_id = str(record.get("record_id", ""))
        if candidate_id in excluded or record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if candidate_id in blocked or record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if candidate_id in needs_review or record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
    return _stable_unique(errors)


def _binding_record_errors(
    payloads: dict[str, Any],
    require_all_required_fields_bound: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    plan = payloads["training_dataset_writer_input_binding_plan"]
    contract = payloads["training_dataset_row_contract"]
    required_fields = set(_safe_list(contract.get("required_row_fields")))
    optional_fields = set(_safe_list(contract.get("optional_row_fields")))
    for record in _safe_records(plan.get("binding_records")):
        if record.get("binding_record_status") != "planned":
            errors.append("binding_record_status_invalid")
        if record.get("writer_executed") is not False:
            errors.append("writer_executed")
        if record.get("training_admitted") is not True:
            errors.append("training_not_admitted")
        if record.get("training_dataset_materialized") is not False:
            errors.append("training_dataset_materialized")
        if record.get("dataset_artifact_created") is not False:
            errors.append("dataset_artifact_created")
        if record.get("phase1_status") != "not_run":
            errors.append("phase1_ran")
        if record.get("dataset_confirmation_changed") is not False:
            errors.append("dataset_confirmation_changed")
        if not _record_contains_required_binding_sections(record):
            errors.append("binding_record_missing_sections")
        if not _binding_record_is_safe(record):
            errors.append("binding_record_contains_unsafe_value")
        required_bindings = _safe_records(record.get("required_field_bindings"))
        optional_bindings = _safe_records(record.get("optional_field_bindings"))
        required_names = {str(binding.get("field_name", "")) for binding in required_bindings}
        optional_names = {str(binding.get("field_name", "")) for binding in optional_bindings}
        if required_names != required_fields:
            errors.append("required_field_bindings_missing")
        if not optional_names.issubset(optional_fields):
            errors.append("optional_field_binding_names_invalid")
        for binding in required_bindings:
            _append_errors(errors, _field_binding_errors(binding, required=True))
            if binding.get("binding_status") == "missing_source":
                if require_all_required_fields_bound:
                    errors.append("required_field_source_missing")
                else:
                    warnings.append("required_field_source_missing")
        for binding in optional_bindings:
            _append_errors(errors, _field_binding_errors(binding, required=False))
        _append_errors(errors, _dedup_split_errors(record.get("dedup_split_binding")))
    return _stable_unique(errors)


def _record_contains_required_binding_sections(record: dict[str, Any]) -> bool:
    return all(
        key in record
        for key in (
            "required_field_bindings",
            "optional_field_bindings",
            "dedup_split_binding",
            "requested_output_formats",
            "target_model_families",
        )
    )


def _field_binding_errors(binding: dict[str, Any], *, required: bool) -> list[str]:
    errors: list[str] = []
    if not {"field_name", "binding_status", "source_artifact_label", "source_artifact_sha256", "source_record_id", "derivation_rule", "value_materialized"}.issubset(binding):
        errors.append("required_field_binding_missing_fields" if required else "optional_field_binding_missing_fields")
    status = str(binding.get("binding_status", ""))
    label = str(binding.get("source_artifact_label", ""))
    source_sha = str(binding.get("source_artifact_sha256", ""))
    source_record_id = str(binding.get("source_record_id", ""))
    derivation_rule = str(binding.get("derivation_rule", ""))
    if status not in _FIELD_BINDING_STATUSES:
        errors.append("required_field_binding_status_invalid" if required else "optional_field_binding_status_invalid")
    if status == "missing_source" and label:
        errors.append("source_artifact_label_invalid")
    if status != "missing_source" and label not in _ALLOWED_SOURCE_ARTIFACT_LABELS:
        errors.append("source_artifact_label_invalid")
    if source_sha and not _SHA_RE.match(source_sha):
        errors.append("source_artifact_sha256_invalid")
    if source_record_id and not _is_safe_id(source_record_id):
        errors.append("source_record_id_invalid")
    if not derivation_rule or not _is_safe_id(derivation_rule):
        errors.append("derivation_rule_label_invalid")
    if binding.get("value_materialized") is not False:
        errors.append("value_materialized")
    if _value_contains_unsafe_material(binding):
        errors.append("binding_record_contains_unsafe_value")
    return _stable_unique(errors)


def _dedup_split_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["dedup_split_binding_missing"]
    errors: list[str] = []
    if not value.get("dedup_key_rule"):
        errors.append("dedup_key_rule_missing")
    if not value.get("split_group_key_rule"):
        errors.append("split_group_key_rule_missing")
    if value.get("dedup_key_materialized") is not False:
        errors.append("dedup_key_materialized")
    if value.get("split_group_key_materialized") is not False:
        errors.append("split_group_key_materialized")
    if value.get("split_group_key_default") != "canonical_molecule_identity":
        errors.append("split_group_key_default_invalid")
    if value.get("row_id_split_forbidden") is not True:
        errors.append("row_id_split_not_forbidden")
    return errors


def _summary(
    status: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    plan = payloads["training_dataset_writer_input_binding_plan"]
    planner = payloads["training_dataset_writer_input_binding_planner_summary"]
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": status,
        **_source_fields(paths, hashes),
        **_source_status_fields(payloads),
        "writer_input_binding_plan_id": str(plan.get("writer_input_binding_plan_id", "")),
        "dataset_name": str(plan.get("dataset_name", "")),
        "row_contract_id": str(plan.get("row_contract_id", "")),
        "contract_version_label": str(plan.get("contract_version_label", "")),
        "materialization_plan_id": str(plan.get("materialization_plan_id", "")),
        "materialization_dry_run_id": str(plan.get("materialization_dry_run_id", "")),
        "writer_execution_request_id": str(plan.get("writer_execution_request_id", "")),
        "execution_ledger_id": str(plan.get("execution_ledger_id", "")),
        "execution_request_id": str(plan.get("execution_request_id", "")),
        "corpus_id": str(plan.get("corpus_id", "")),
        "admission_request_id": str(plan.get("admission_request_id", "")),
        "review_manifest_id": str(plan.get("review_manifest_id", "")),
        "review_queue_id": str(plan.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(plan.get("property_candidate_manifest_id", "")),
        "writer_executed": False,
        "values_materialized": False,
        "training_admitted": True if status in {"passed", "needs_review"} else plan.get("training_admitted") is True,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "binding_record_count": len(_safe_records(plan.get("binding_records"))),
        "binding_record_ids": _safe_list(plan.get("binding_record_ids")),
        "writer_request_record_ids": _safe_list(plan.get("writer_request_record_ids")),
        "row_preview_ids": _safe_list(plan.get("row_preview_ids")),
        "planned_training_admission_candidate_record_ids": _safe_list(
            plan.get("planned_training_admission_candidate_record_ids")
        ),
        "required_field_names": _safe_list(plan.get("required_field_names")),
        "optional_field_names": _safe_list(plan.get("optional_field_names")),
        "missing_required_field_counts": plan.get("missing_required_field_counts", {})
        if isinstance(plan.get("missing_required_field_counts"), dict)
        else {},
        "missing_optional_field_counts": plan.get("missing_optional_field_counts", {})
        if isinstance(plan.get("missing_optional_field_counts"), dict)
        else {},
        "dedup_split_summary": _dedup_split_summary(payloads),
        "preflight_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _source_status_fields(payloads: dict[str, Any]) -> dict[str, str]:
    return {
        "binding_plan_status": str(payloads["training_dataset_writer_input_binding_plan"].get("planner_status", "")),
        "binding_planner_summary_status": str(
            payloads["training_dataset_writer_input_binding_planner_summary"].get("planner_status", "")
        ),
        "writer_request_preflight_status": str(
            payloads["training_dataset_writer_execution_request_preflight"].get("preflight_status", "")
        ),
        "writer_request_status": str(payloads["training_dataset_writer_execution_request"].get("request_status", "")),
        "materialization_dry_run_precheck_status": str(
            payloads["training_dataset_materialization_dry_run_precheck"].get("precheck_status", "")
        ),
        "materialization_dry_run_status": str(
            payloads["training_dataset_materialization_dry_run_report"].get("dry_run_status", "")
        ),
        "row_contract_precheck_status": str(payloads["training_dataset_row_contract_precheck"].get("precheck_status", "")),
        "contract_status": str(payloads["training_dataset_row_contract"].get("contract_status", "")),
        "materialization_plan_precheck_status": str(
            payloads["training_dataset_materialization_plan_precheck"].get("precheck_status", "")
        ),
        "plan_status": str(payloads["training_dataset_materialization_plan"].get("plan_status", "")),
        "ledger_status": str(payloads["training_admission_execution_ledger"].get("execution_status", "")),
    }


def _dedup_split_summary(payloads: dict[str, Any]) -> dict[str, Any]:
    records = _safe_records(payloads["training_dataset_writer_input_binding_plan"].get("binding_records"))
    if not records:
        return {}
    binding = records[0].get("dedup_split_binding")
    if not isinstance(binding, dict):
        return {}
    return {
        "dedup_key_rule": str(binding.get("dedup_key_rule", "")),
        "split_group_key_rule": str(binding.get("split_group_key_rule", "")),
        "dedup_key_materialized": binding.get("dedup_key_materialized") is True,
        "split_group_key_materialized": binding.get("split_group_key_materialized") is True,
        "split_group_key_default": str(binding.get("split_group_key_default", "")),
        "row_id_split_forbidden": binding.get("row_id_split_forbidden") is True,
    }


def _source_fields(paths: dict[str, str | Path], hashes: dict[str, str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, path in paths.items():
        fields[key] = Path(path).name or key
        fields[key.replace("_path", "_sha256")] = hashes[key.replace("_path", "_sha256")]
    return fields


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Dataset Writer Input Binding Plan Preflight Evidence",
            "",
            f"- Preflight status: `{summary['preflight_status']}`",
            f"- Writer input binding plan id: `{summary['writer_input_binding_plan_id']}`",
            f"- Dataset name: `{summary['dataset_name']}`",
            f"- Row contract id: `{summary['row_contract_id']}`",
            f"- Writer execution request id: `{summary['writer_execution_request_id']}`",
            f"- Binding record count: `{summary['binding_record_count']}`",
            f"- Missing required fields: `{json.dumps(summary['missing_required_field_counts'], sort_keys=True)}`",
            f"- Missing optional fields: `{json.dumps(summary['missing_optional_field_counts'], sort_keys=True)}`",
            f"- Preflight errors: `{json.dumps(summary['preflight_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training dataset writer input binding plan preflight only.",
            "- no dataset writer was executed.",
            "- no values were materialized.",
            "- no serialized training rows were created.",
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


def _binding_record_is_safe(record: dict[str, Any]) -> bool:
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    if _value_contains_unsafe_material(serialized):
        return False
    for key, value in record.items():
        if key.endswith("_sha256") and value and not _SHA_RE.match(str(value)):
            return False
        if key in {"writer_executed", "training_admitted", "training_dataset_materialized", "dataset_artifact_created", "dataset_confirmation_changed"}:
            if not isinstance(value, bool):
                return False
        if key in {"requested_output_formats", "target_model_families"}:
            if not isinstance(value, list) or not all(isinstance(item, str) and _is_safe_id(item) for item in value):
                return False
            if key == "requested_output_formats" and not set(value).issubset(_OUTPUT_FORMAT_LABELS):
                return False
    return True


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise PropertyTrainingDatasetWriterInputBindingPlanPreflightError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetWriterInputBindingPlanPreflightError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise PropertyTrainingDatasetWriterInputBindingPlanPreflightError(f"{label} contains forbidden material")
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
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


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
    if any(marker.lower() in lowered for marker in _FINAL_FORBIDDEN_MARKERS):
        return True
    if any(marker in serialized for marker in ("C1=CC", "C1CC", "N#N", "InChI=")):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _value_contains_unsafe_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _VALUE_FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _input_contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    return any(marker.lower() in lowered for marker in _INPUT_FORBIDDEN_MARKERS)


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": "blocked",
        "preflight_errors": ["property_training_dataset_writer_input_binding_plan_preflight_redaction_failed"],
        "redaction_status": "failed",
    }


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    for marker in _FINAL_FORBIDDEN_MARKERS:
        if marker.lower() in lowered:
            return "invalid input contained forbidden material"
    if _ABSOLUTE_PATH_VALUE_RE.search(json.dumps(message)):
        return "invalid input"
    return message or "invalid input"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
