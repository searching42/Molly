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


_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight.v1"
_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_execution_plan.v1"
_PLANNER_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_execution_planner.v1"
_VALUE_SOURCE_PREFLIGHT_SCHEMA_VERSION = (
    "custom_corpus_property_training_dataset_writer_value_source_manifest_preflight.v1"
)
_VALUE_SOURCE_MANIFEST_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_value_source_manifest.v1"
_VALUE_SOURCE_PLANNER_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_value_source_manifest_planner.v1"
_INPUT_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_input_binding_plan_preflight.v1"
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
_OUTPUT_FORMAT_LABELS = {"csv", "jsonl", "lmdb", "parquet"}
_VALUE_FIELD_NAMES = {
    "property_name",
    "property_value",
    "property_unit",
    "property_value_normalized",
    "property_unit_normalized",
    "compound_id",
    "canonical_smiles",
}
_ALLOWED_SOURCE_ARTIFACT_LABELS = {
    "writer_execution_request",
    "materialization_dry_run_report",
    "row_contract",
    "materialization_plan",
    "training_admission_execution_ledger",
    "quarantine_candidate_records",
}
_SOURCE_LABEL_PATH_KEYS = {
    "writer_execution_request": "training_dataset_writer_execution_request_path",
    "materialization_dry_run_report": "training_dataset_materialization_dry_run_report_path",
    "row_contract": "training_dataset_row_contract_path",
    "materialization_plan": "training_dataset_materialization_plan_path",
    "training_admission_execution_ledger": "training_admission_execution_ledger_path",
    "quarantine_candidate_records": "quarantine_candidate_records_path",
}
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
    "serialized training row",
    "serialized dataset row",
    "conformer block",
    "dpa3 structure block",
    "inchi=",
)
_VALUE_FORBIDDEN_MARKERS = _FINAL_FORBIDDEN_MARKERS + (
    "0.72",
    "c1=cc",
    "canonical_smiles_value",
    "full candidate payload",
    "full materialized payload",
    "full draft payload",
    "full execution request payload",
    "full dry-run payload",
    "full ledger payload",
    "full materialization plan payload",
    "full row contract payload",
    "full row preview payload",
    "full writer request payload",
    "full binding plan payload",
    "full value source payload",
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


class PropertyTrainingDatasetControlledWriterExecutionPlanPreflightError(ValueError):
    pass


def preflight_property_training_dataset_controlled_writer_execution_plan(
    *,
    training_dataset_controlled_writer_execution_plan_path: str | Path,
    training_dataset_controlled_writer_execution_planner_summary_path: str | Path,
    training_dataset_writer_value_source_manifest_preflight_path: str | Path,
    training_dataset_writer_value_source_manifest_path: str | Path,
    training_dataset_writer_value_source_manifest_planner_summary_path: str | Path,
    training_dataset_writer_input_binding_plan_preflight_path: str | Path,
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
    require_controlled_writer_execution_plan_planned: bool = True,
    allow_controlled_writer_execution_plan_needs_review: bool = False,
    minimum_value_source_records: int = 1,
) -> dict[str, Any]:
    payloads = _load_payloads(
        training_dataset_controlled_writer_execution_plan_path=training_dataset_controlled_writer_execution_plan_path,
        training_dataset_controlled_writer_execution_planner_summary_path=training_dataset_controlled_writer_execution_planner_summary_path,
        training_dataset_writer_value_source_manifest_preflight_path=training_dataset_writer_value_source_manifest_preflight_path,
        training_dataset_writer_value_source_manifest_path=training_dataset_writer_value_source_manifest_path,
        training_dataset_writer_value_source_manifest_planner_summary_path=training_dataset_writer_value_source_manifest_planner_summary_path,
        training_dataset_writer_input_binding_plan_preflight_path=training_dataset_writer_input_binding_plan_preflight_path,
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
            require_controlled_writer_execution_plan_planned=require_controlled_writer_execution_plan_planned,
            allow_plan_needs_review=allow_controlled_writer_execution_plan_needs_review,
            warnings=warnings,
        ),
    )
    _append_errors(errors, _hash_errors(payloads, hashes))
    _append_errors(errors, _id_errors(payloads))
    record_errors = _record_errors(payloads, hashes, minimum_value_source_records)
    if allow_controlled_writer_execution_plan_needs_review and "no_value_source_records" in record_errors:
        warnings.append("no_value_source_records")
        record_errors = [error for error in record_errors if error != "no_value_source_records"]
    _append_errors(errors, record_errors)
    _append_errors(errors, _plan_content_errors(payloads, hashes))
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
        summary = preflight_property_training_dataset_controlled_writer_execution_plan(
            training_dataset_controlled_writer_execution_plan_path=args.training_dataset_controlled_writer_execution_plan,
            training_dataset_controlled_writer_execution_planner_summary_path=args.training_dataset_controlled_writer_execution_planner_summary,
            training_dataset_writer_value_source_manifest_preflight_path=args.training_dataset_writer_value_source_manifest_preflight,
            training_dataset_writer_value_source_manifest_path=args.training_dataset_writer_value_source_manifest,
            training_dataset_writer_value_source_manifest_planner_summary_path=args.training_dataset_writer_value_source_manifest_planner_summary,
            training_dataset_writer_input_binding_plan_preflight_path=args.training_dataset_writer_input_binding_plan_preflight,
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
            require_controlled_writer_execution_plan_planned=args.require_controlled_writer_execution_plan_planned,
            allow_controlled_writer_execution_plan_needs_review=args.allow_controlled_writer_execution_plan_needs_review,
            minimum_value_source_records=args.minimum_value_source_records,
        )
    except Exception as exc:
        err.write(
            "property training dataset controlled writer execution plan preflight invalid: "
            f"{_safe_exception_message(exc)}\n"
        )
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("preflight_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight"
        ),
        description="Preflight a controlled property training dataset writer execution plan without running a writer.",
    )
    parser.add_argument("--training-dataset-controlled-writer-execution-plan", required=True)
    parser.add_argument("--training-dataset-controlled-writer-execution-planner-summary", required=True)
    parser.add_argument("--training-dataset-writer-value-source-manifest-preflight", required=True)
    parser.add_argument("--training-dataset-writer-value-source-manifest", required=True)
    parser.add_argument("--training-dataset-writer-value-source-manifest-planner-summary", required=True)
    parser.add_argument("--training-dataset-writer-input-binding-plan-preflight", required=True)
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
    parser.add_argument("--require-controlled-writer-execution-plan-planned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-controlled-writer-execution-plan-needs-review", action="store_true")
    parser.add_argument("--minimum-value-source-records", type=int, default=1)
    return parser


def _schema_errors(payloads: dict[str, Any]) -> list[str]:
    expected = {
        "training_dataset_controlled_writer_execution_plan": (
            "training_dataset_controlled_writer_execution_plan_schema_invalid",
            _PLAN_SCHEMA_VERSION,
        ),
        "training_dataset_controlled_writer_execution_planner_summary": (
            "training_dataset_controlled_writer_execution_planner_summary_schema_invalid",
            _PLANNER_SCHEMA_VERSION,
        ),
        "training_dataset_writer_value_source_manifest_preflight": (
            "training_dataset_writer_value_source_manifest_preflight_schema_invalid",
            _VALUE_SOURCE_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_dataset_writer_value_source_manifest": (
            "training_dataset_writer_value_source_manifest_schema_invalid",
            _VALUE_SOURCE_MANIFEST_SCHEMA_VERSION,
        ),
        "training_dataset_writer_value_source_manifest_planner_summary": (
            "training_dataset_writer_value_source_manifest_planner_summary_schema_invalid",
            _VALUE_SOURCE_PLANNER_SCHEMA_VERSION,
        ),
        "training_dataset_writer_input_binding_plan_preflight": (
            "training_dataset_writer_input_binding_plan_preflight_schema_invalid",
            _INPUT_PREFLIGHT_SCHEMA_VERSION,
        ),
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
        "training_admission_readiness_summary": (
            "training_admission_readiness_summary_schema_invalid",
            _READINESS_SCHEMA_VERSION,
        ),
        "quarantine_candidate_preflight_summary": (
            "quarantine_candidate_preflight_summary_schema_invalid",
            _QUARANTINE_PREFLIGHT_SCHEMA_VERSION,
        ),
        "quarantine_candidate_records": ("quarantine_candidate_records_schema_invalid", _QUARANTINE_CANDIDATE_SCHEMA_VERSION),
    }
    return [error for key, (error, schema) in expected.items() if payloads[key].get("schema_version") != schema]


def _status_errors(
    payloads: dict[str, Any],
    *,
    require_controlled_writer_execution_plan_planned: bool,
    allow_plan_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    plan_allow = allow_plan_needs_review or not require_controlled_writer_execution_plan_planned
    for key in (
        "training_dataset_controlled_writer_execution_plan",
        "training_dataset_controlled_writer_execution_planner_summary",
    ):
        errors.extend(
            _status_field_errors(
                payloads[key],
                "planner_status",
                "training_dataset_controlled_writer_execution_plan",
                "planned",
                plan_allow,
                warnings,
            )
        )
    checks = (
        ("training_dataset_writer_value_source_manifest_preflight", "preflight_status", "training_dataset_writer_value_source_manifest_preflight", "passed"),
        ("training_dataset_writer_value_source_manifest", "planner_status", "training_dataset_writer_value_source_manifest", "planned"),
        ("training_dataset_writer_value_source_manifest_planner_summary", "planner_status", "training_dataset_writer_value_source_manifest_planner_summary", "planned"),
        ("training_dataset_writer_input_binding_plan_preflight", "preflight_status", "training_dataset_writer_input_binding_plan_preflight", "passed"),
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
        ("training_admission_execution_dry_run_precheck", "dry_run_status", "training_admission_execution_dry_run_precheck", "passed"),
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
    for payload_key, field, prefix, required in checks:
        errors.extend(_status_field_errors(payloads[payload_key], field, prefix, required, plan_allow, warnings))
    plan = payloads["training_dataset_controlled_writer_execution_plan"]
    if plan.get("writer_execution_mode") != "controlled_writer_execution_plan_only":
        errors.append("writer_execution_mode_invalid")
    if plan.get("planner_errors"):
        errors.append("training_dataset_controlled_writer_execution_plan_has_errors")
    if payloads["training_dataset_controlled_writer_execution_planner_summary"].get("planner_errors"):
        errors.append("training_dataset_controlled_writer_execution_planner_summary_has_errors")
    for key in _payload_keys(payloads):
        container = payloads[key]
        if container.get("writer_executed") is True:
            errors.append("writer_executed")
        if container.get("source_payloads_read") is True:
            errors.append("source_payloads_read")
        if container.get("source_payload_read") is True:
            errors.append("source_payload_read")
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
        if container.get("model_training_run") is True:
            errors.append("model_training_run")
        if container.get("evaluation_run") is True:
            errors.append("evaluation_run")
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
    if payloads["training_dataset_controlled_writer_execution_planner_summary"].get(
        "controlled_writer_execution_plan_sha256"
    ) != hashes.get("training_dataset_controlled_writer_execution_plan_sha256"):
        errors.append("controlled_writer_execution_plan_sha256_mismatch")
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
    value_source_keys = (
        "training_dataset_controlled_writer_execution_plan",
        "training_dataset_controlled_writer_execution_planner_summary",
        "training_dataset_writer_value_source_manifest_preflight",
        "training_dataset_writer_value_source_manifest",
        "training_dataset_writer_value_source_manifest_planner_summary",
    )
    writer_chain_keys = (
        *value_source_keys,
        "training_dataset_writer_input_binding_plan_preflight",
        "training_dataset_writer_input_binding_plan",
        "training_dataset_writer_input_binding_planner_summary",
    )
    writer_request_keys = (
        *writer_chain_keys,
        "training_dataset_writer_execution_request_preflight",
        "training_dataset_writer_execution_request",
        "training_dataset_writer_execution_request_summary",
    )
    materialization_keys = (
        *writer_request_keys,
        "training_dataset_materialization_dry_run_precheck",
        "training_dataset_materialization_dry_run_report",
        "training_dataset_materialization_dry_run_summary",
        "training_dataset_row_contract_precheck",
        "training_dataset_row_contract",
        "training_dataset_row_contract_summary",
        "training_dataset_materialization_plan_precheck",
        "training_dataset_materialization_plan",
        "training_dataset_materialization_planner_summary",
    )
    groups = {
        "corpus_id": (_payload_keys(payloads), ("corpus_id",)),
        "controlled_writer_execution_plan_id": (
            (
                "training_dataset_controlled_writer_execution_plan",
                "training_dataset_controlled_writer_execution_planner_summary",
            ),
            ("controlled_writer_execution_plan_id",),
        ),
        "value_source_manifest_id": (value_source_keys, ("value_source_manifest_id",)),
        "writer_input_binding_plan_id": (writer_chain_keys, ("writer_input_binding_plan_id",)),
        "writer_execution_request_id": (writer_request_keys, ("writer_execution_request_id",)),
        "materialization_plan_id": (materialization_keys, ("materialization_plan_id",)),
        "row_contract_id": (materialization_keys, ("row_contract_id",)),
        "execution_ledger_id": (_payload_keys(payloads), ("execution_ledger_id",)),
        "execution_request_id": (
            (
                *materialization_keys,
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
        "dataset_name": (materialization_keys, ("dataset_name",)),
    }
    errors: list[str] = []
    for logical, (payload_keys, fields) in groups.items():
        values = _group_values(payloads, payload_keys, fields)
        if len(set(values)) > 1:
            errors.append(f"{logical}_mismatch")
        if any(value and not _is_safe_id(value) for value in values):
            errors.append(f"{logical}_invalid")
    return _stable_unique(errors)


def _record_errors(payloads: dict[str, Any], hashes: dict[str, str], minimum_value_source_records: int) -> list[str]:
    errors: list[str] = []
    plan = payloads["training_dataset_controlled_writer_execution_plan"]
    planner = payloads["training_dataset_controlled_writer_execution_planner_summary"]
    manifest = payloads["training_dataset_writer_value_source_manifest"]
    preflight = payloads["training_dataset_writer_value_source_manifest_preflight"]
    value_records = _safe_records(manifest.get("value_source_records"))
    value_record_ids = [str(record.get("value_source_record_id", "")) for record in value_records]
    if len(value_records) < max(minimum_value_source_records, 1):
        errors.append("no_value_source_records")
    for container in (plan, planner, manifest, preflight, payloads["training_dataset_writer_value_source_manifest_planner_summary"]):
        if int(container.get("value_source_record_count", -1)) != len(value_records):
            errors.append("value_source_record_count_mismatch")
        if _safe_list(container.get("value_source_record_ids")) != value_record_ids:
            errors.append("value_source_record_ids_mismatch")
    for record in value_records:
        if _value_contains_unsafe_material(record):
            errors.append("value_source_record_contains_unsafe_value")
        if record.get("value_field_name") not in _VALUE_FIELD_NAMES:
            errors.append("value_field_name_invalid")
    binding_records = _safe_records(payloads["training_dataset_writer_input_binding_plan"].get("binding_records"))
    binding_ids = [str(record.get("writer_input_binding_record_id", "")) for record in binding_records]
    writer_request_ids = [str(record.get("writer_request_record_id", "")) for record in binding_records]
    planned_candidates = [str(record.get("candidate_record_id", "")) for record in binding_records]
    for container in (
        plan,
        planner,
        manifest,
        payloads["training_dataset_writer_input_binding_plan"],
        payloads["training_dataset_writer_input_binding_planner_summary"],
    ):
        if container.get("binding_record_ids") is not None and _safe_list(container.get("binding_record_ids")) != binding_ids:
            errors.append("binding_record_ids_mismatch")
    for container in (
        plan,
        planner,
        payloads["training_dataset_writer_execution_request"],
        payloads["training_dataset_writer_execution_request_summary"],
    ):
        if container.get("writer_request_record_ids") is not None and _safe_list(container.get("writer_request_record_ids")) != writer_request_ids:
            errors.append("writer_request_record_ids_mismatch")
    if plan.get("row_count_expectations", {}).get("value_source_record_count") != len(value_records):
        errors.append("row_count_expectations_mismatch")
    if plan.get("row_count_expectations", {}).get("writer_request_record_count") != len(writer_request_ids):
        errors.append("row_count_expectations_mismatch")
    if plan.get("row_count_expectations", {}).get("binding_record_count") != len(binding_ids):
        errors.append("row_count_expectations_mismatch")
    request_plan = payloads["training_admission_request_plan"]
    excluded = set(_safe_list(request_plan.get("planned_exclusion_record_ids"))) | set(
        _safe_list(request_plan.get("exclude_record_ids"))
    )
    blocked = set(_safe_list(request_plan.get("blocked_from_admission_record_ids"))) | set(
        _safe_list(request_plan.get("blocked_record_ids"))
    )
    needs_review = set(_safe_list(payloads["training_admission_readiness_summary"].get("blocked_from_admission_record_ids")))
    candidate_set = set(planned_candidates)
    if candidate_set & excluded:
        errors.append("planned_candidate_from_excluded_record")
    if candidate_set & blocked:
        errors.append("planned_candidate_from_blocked_record")
    if candidate_set & needs_review:
        errors.append("planned_candidate_from_needs_review_record")
    if _safe_list(plan.get("planned_training_admission_candidate_record_ids")) != planned_candidates:
        errors.append("planned_candidate_record_ids_mismatch")
    for artifact in _safe_records(plan.get("allowed_source_artifacts")):
        if _value_contains_unsafe_material(artifact):
            errors.append("allowed_source_artifact_contains_unsafe_value")
        label = str(artifact.get("source_artifact_label", ""))
        if label not in _ALLOWED_SOURCE_ARTIFACT_LABELS:
            errors.append("source_artifact_label_invalid")
        basename = str(artifact.get("source_artifact_basename", ""))
        if not basename or Path(basename).name != basename or "/" in basename or "\\" in basename:
            errors.append("source_artifact_basename_not_safe")
        path_key = _SOURCE_LABEL_PATH_KEYS.get(label, "")
        if path_key:
            if artifact.get("source_artifact_sha256") != hashes.get(path_key.replace("_path", "_sha256")):
                errors.append("source_artifact_sha256_mismatch")
            if basename != Path(payloads["paths"][path_key]).name:
                errors.append("source_artifact_basename_mismatch")
    return _stable_unique(errors)


def _plan_content_errors(payloads: dict[str, Any], hashes: dict[str, str]) -> list[str]:
    errors: list[str] = []
    plan = payloads["training_dataset_controlled_writer_execution_plan"]
    planner = payloads["training_dataset_controlled_writer_execution_planner_summary"]
    requested = _safe_list(plan.get("requested_output_formats"))
    if set(requested) - _OUTPUT_FORMAT_LABELS or not requested:
        errors.append("output_format_label_invalid")
    if sorted(requested) != sorted(_safe_list(planner.get("requested_output_formats"))):
        errors.append("requested_output_formats_mismatch")
    labels = _safe_list(plan.get("planned_output_artifact_labels"))
    if not labels or any(not _is_label_only(label) or _value_contains_unsafe_material(label) for label in labels):
        errors.append("planned_output_artifact_label_invalid")
    if sorted(labels) != sorted(_safe_list(planner.get("planned_output_artifact_labels"))):
        errors.append("planned_output_artifact_labels_mismatch")
    for policy_field, error in (
        ("output_directory_policy", "output_directory_policy_invalid"),
        ("file_naming_policy", "file_naming_policy_invalid"),
        ("provenance_preservation_requirements", "provenance_preservation_requirements_invalid"),
        ("redaction_policy", "redaction_policy_invalid"),
    ):
        values = _safe_list(plan.get(policy_field))
        if not values or any(not _is_label_only(value) or _value_contains_unsafe_material(value) for value in values):
            errors.append(error)
    allowed_fields = _safe_list(plan.get("allowed_value_field_names"))
    value_fields = sorted(
        {str(record.get("value_field_name", "")) for record in _safe_records(payloads["training_dataset_writer_value_source_manifest"].get("value_source_records"))}
    )
    if sorted(allowed_fields) != value_fields:
        errors.append("allowed_value_field_names_mismatch")
    plan_without_boundary = dict(plan)
    plan_without_boundary.pop("boundary_statement", None)
    if _value_contains_unsafe_material(plan_without_boundary):
        errors.append("controlled_writer_execution_plan_contains_unsafe_value")
    return _stable_unique(errors)


def _sha_format_errors(payloads: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for container in (payloads[key] for key in _payload_keys(payloads)):
        for key, value in container.items():
            if key.endswith("_sha256") and value and not _is_sha(value):
                errors.append(f"{key}_invalid")
    return _stable_unique(errors)


def _summary(
    status: str,
    payloads: dict[str, Any],
    paths: dict[str, Path],
    hashes: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    plan = payloads["training_dataset_controlled_writer_execution_plan"]
    planner = payloads["training_dataset_controlled_writer_execution_planner_summary"]
    manifest = payloads["training_dataset_writer_value_source_manifest"]
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": status,
        **{f"{key.replace('_path', '')}_path": Path(path).name for key, path in paths.items()},
        **hashes,
        "controlled_writer_execution_plan_id": plan.get("controlled_writer_execution_plan_id", ""),
        "value_source_manifest_id": plan.get("value_source_manifest_id", ""),
        "writer_input_binding_plan_id": plan.get("writer_input_binding_plan_id", ""),
        "writer_execution_request_id": plan.get("writer_execution_request_id", ""),
        "row_contract_id": plan.get("row_contract_id", ""),
        "materialization_plan_id": plan.get("materialization_plan_id", ""),
        "execution_ledger_id": plan.get("execution_ledger_id", ""),
        "execution_request_id": plan.get("execution_request_id", ""),
        "corpus_id": plan.get("corpus_id", ""),
        "dataset_name": plan.get("dataset_name", ""),
        "planner_status": plan.get("planner_status", ""),
        "value_source_preflight_status": plan.get("value_source_preflight_status", ""),
        "value_source_manifest_status": plan.get("value_source_manifest_status", ""),
        "input_binding_preflight_status": plan.get("input_binding_preflight_status", ""),
        "binding_plan_status": plan.get("binding_plan_status", ""),
        "writer_request_preflight_status": plan.get("writer_request_preflight_status", ""),
        "writer_request_status": plan.get("writer_request_status", ""),
        "materialization_dry_run_status": plan.get("materialization_dry_run_status", ""),
        "contract_status": plan.get("contract_status", ""),
        "plan_status": plan.get("plan_status", ""),
        "ledger_status": plan.get("ledger_status", ""),
        "writer_execution_mode": plan.get("writer_execution_mode", ""),
        "requested_output_formats": _safe_list(plan.get("requested_output_formats")),
        "planned_output_artifact_labels": _safe_list(plan.get("planned_output_artifact_labels")),
        "allowed_value_field_names": _safe_list(plan.get("allowed_value_field_names")),
        "allowed_source_artifact_basenames": [
            str(artifact.get("source_artifact_basename", ""))
            for artifact in _safe_records(plan.get("allowed_source_artifacts"))
        ],
        "value_source_record_count": len(_safe_records(manifest.get("value_source_records"))),
        "value_source_record_ids": _safe_list(plan.get("value_source_record_ids")),
        "binding_record_ids": _safe_list(plan.get("binding_record_ids")),
        "writer_request_record_ids": _safe_list(plan.get("writer_request_record_ids")),
        "planned_training_admission_candidate_record_ids": _safe_list(
            plan.get("planned_training_admission_candidate_record_ids")
        ),
        "writer_executed": bool(plan.get("writer_executed", False)),
        "source_payloads_read": bool(plan.get("source_payloads_read", False)),
        "values_materialized": bool(plan.get("values_materialized", False)),
        "training_dataset_materialized": bool(plan.get("training_dataset_materialized", False)),
        "dataset_artifact_created": bool(plan.get("dataset_artifact_created", False)),
        "phase1_status": plan.get("phase1_status", ""),
        "dataset_confirmation_changed": bool(plan.get("dataset_confirmation_changed", False)),
        "model_training_run": bool(plan.get("model_training_run", False)),
        "evaluation_run": bool(plan.get("evaluation_run", False)),
        "row_count_expectations": dict(plan.get("row_count_expectations") or {}),
        "preflight_errors": errors,
        "warnings": warnings,
        "redaction_status": "passed",
    }
    if planner.get("controlled_writer_execution_plan_sha256"):
        summary["controlled_writer_execution_plan_sha256"] = planner["controlled_writer_execution_plan_sha256"]
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Property Training Dataset Controlled Writer Execution Plan Preflight",
        "",
        f"- Preflight status: {summary.get('preflight_status', '')}",
        f"- Controlled writer execution plan id: {summary.get('controlled_writer_execution_plan_id', '')}",
        f"- Corpus id: {summary.get('corpus_id', '')}",
        f"- Dataset name: {summary.get('dataset_name', '')}",
        f"- Value source record count: {summary.get('value_source_record_count', 0)}",
        f"- Requested output formats: {', '.join(summary.get('requested_output_formats', []))}",
        f"- Planned output artifact labels: {', '.join(summary.get('planned_output_artifact_labels', []))}",
        f"- Preflight errors: {', '.join(summary.get('preflight_errors', [])) or 'none'}",
        f"- Warnings: {', '.join(summary.get('warnings', [])) or 'none'}",
        "",
        "## Boundary Statement",
        "",
        "this is a controlled writer execution plan preflight only.",
        "The writer was not executed; source payloads were not read; values were not materialized.",
        "No row serialization occurred.",
        "no training CSV/JSONL/Parquet/LMDB was created.",
        "No candidate CSV/JSONL/Parquet/LMDB was created.",
        "No conformers or DPA3 structures were generated.",
        "Phase 1 did not run and DatasetConfirmation was not changed.",
        "No model training or evaluation was run.",
    ]
    return "\n".join(lines) + "\n"


def _load_payloads(**path_kwargs: str | Path) -> dict[str, Any]:
    paths = {key: Path(value) for key, value in path_kwargs.items()}
    payloads: dict[str, Any] = {"paths": paths}
    for key, path in paths.items():
        payloads[key.replace("_path", "")] = _load_json(path)
    return payloads


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if _input_contains_forbidden_material(text):
        raise PropertyTrainingDatasetControlledWriterExecutionPlanPreflightError("input_redaction_failed")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetControlledWriterExecutionPlanPreflightError("json_object_required")
    return payload


def _payload_keys(payloads: dict[str, Any]) -> list[str]:
    return [key for key in payloads.keys() if key != "paths"]


def _group_values(payloads: dict[str, Any], payload_keys: tuple[str, ...], fields: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for key in payload_keys:
        container = payloads.get(key, {})
        if not isinstance(container, dict):
            continue
        for field in fields:
            value = container.get(field)
            if value:
                values.append(str(value))
    return values


def _safe_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]


def _safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _append_errors(errors: list[str], new_errors: list[str]) -> None:
    errors.extend(new_errors)


def _stable_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _is_safe_id(value: str) -> bool:
    return bool(value) and bool(_SAFE_ID_RE.match(value))


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA_RE.match(value))


def _is_label_only(value: str) -> bool:
    return _is_safe_id(value) and "/" not in value and "\\" not in value and "." not in value


def _safe_sha_for_path(path: Path) -> str:
    return sha256_file(path)


def _sha256_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _input_contains_forbidden_material(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in _INPUT_FORBIDDEN_MARKERS)


def _value_contains_unsafe_material(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in _VALUE_FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(text))


def _contains_forbidden_material(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in _FINAL_FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(text))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": "blocked",
        "preflight_errors": [
            "property_training_dataset_controlled_writer_execution_plan_preflight_redaction_failed"
        ],
        "redaction_status": "failed",
    }


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc)
    if _input_contains_forbidden_material(message):
        return "redacted"
    return message or exc.__class__.__name__


if __name__ == "__main__":
    raise SystemExit(main())
