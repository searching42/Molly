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


_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_execution_plan.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_execution_planner.v1"
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
    "conformer block",
    "dpa3 structure block",
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


class PropertyTrainingDatasetControlledWriterExecutionPlanError(ValueError):
    pass


def plan_property_training_dataset_controlled_writer_execution(
    *,
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
    output_dir: str | Path,
    controlled_writer_execution_plan_id: str,
    created_by: str,
    confirm_training_dataset_controlled_writer_execution_plan: bool = False,
    allow_value_source_manifest_preflight_needs_review: bool = False,
    minimum_value_source_records: int = 1,
) -> dict[str, Any]:
    payloads = _load_payloads(
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
    run_dir = Path(output_dir).expanduser() / controlled_writer_execution_plan_id
    warnings: list[str] = []
    errors: list[str] = []
    if not confirm_training_dataset_controlled_writer_execution_plan:
        errors.append("confirmation_required")
    if not _is_safe_id(controlled_writer_execution_plan_id):
        errors.append("controlled_writer_execution_plan_id_invalid")
    if created_by and not _is_safe_id(created_by):
        errors.append("created_by_invalid")
    if run_dir.exists() and any(run_dir.iterdir()):
        errors.append("output_directory_not_clean")
    _append_errors(errors, _schema_errors(payloads))
    _append_errors(errors, _status_errors(payloads, allow_value_source_manifest_preflight_needs_review, warnings))
    _append_errors(errors, _hash_errors(payloads, hashes))
    _append_errors(errors, _id_errors(payloads))
    record_errors = _record_errors(payloads, hashes, minimum_value_source_records)
    if allow_value_source_manifest_preflight_needs_review and "no_value_source_records" in record_errors:
        warnings.append("no_value_source_records")
        record_errors = [error for error in record_errors if error != "no_value_source_records"]
    _append_errors(errors, record_errors)
    _append_errors(errors, _output_errors(payloads))
    _append_errors(errors, _sha_format_errors(payloads))
    status = "blocked" if errors else "needs_review" if warnings else "planned"
    plan_path = run_dir / "property_training_dataset_controlled_writer_execution_plan.json"
    summary_path = run_dir / "property_training_dataset_controlled_writer_execution_planner_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_controlled_writer_execution_plan_evidence.md"
    plan = _plan(
        status=status,
        controlled_writer_execution_plan_id=controlled_writer_execution_plan_id,
        created_by=created_by,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        errors=errors,
        warnings=warnings,
    )
    summary = _summary(
        status=status,
        controlled_writer_execution_plan_id=controlled_writer_execution_plan_id,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        plan_path=plan_path,
        plan_sha256="",
        errors=errors,
        warnings=warnings,
    )
    markdown = _markdown(summary)
    if _contains_forbidden_material({"plan": plan, "summary": summary, "markdown": markdown}):
        return _minimal_redaction_failure()
    if status == "blocked":
        return summary
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    plan_sha256 = sha256_file(plan_path)
    summary = _summary(
        status=status,
        controlled_writer_execution_plan_id=controlled_writer_execution_plan_id,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        errors=errors,
        warnings=warnings,
    )
    markdown = _markdown(summary)
    if _contains_forbidden_material({"summary": summary, "markdown": markdown}):
        plan_path.unlink(missing_ok=True)
        return _minimal_redaction_failure()
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
        summary = plan_property_training_dataset_controlled_writer_execution(
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
            output_dir=args.output_dir,
            controlled_writer_execution_plan_id=args.controlled_writer_execution_plan_id,
            created_by=args.created_by,
            confirm_training_dataset_controlled_writer_execution_plan=args.confirm_training_dataset_controlled_writer_execution_plan,
            allow_value_source_manifest_preflight_needs_review=args.allow_value_source_manifest_preflight_needs_review,
            minimum_value_source_records=args.minimum_value_source_records,
        )
    except Exception as exc:
        err.write(f"property training dataset controlled writer execution plan invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("planner_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_execution_plan",
        description="Plan controlled property training dataset writer execution without running a writer.",
    )
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--controlled-writer-execution-plan-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-training-dataset-controlled-writer-execution-plan", action="store_true")
    parser.add_argument("--allow-value-source-manifest-preflight-needs-review", action="store_true")
    parser.add_argument("--minimum-value-source-records", type=int, default=1)
    return parser


def _schema_errors(payloads: dict[str, Any]) -> list[str]:
    expected = {
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


def _status_errors(payloads: dict[str, Any], allow_needs_review: bool, warnings: list[str]) -> list[str]:
    errors: list[str] = []
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
        errors.extend(_status_field_errors(payloads[payload_key], field, prefix, required, allow_needs_review, warnings))
    for key in _payload_keys(payloads):
        container = payloads[key]
        if container.get("writer_executed") is True:
            errors.append("writer_executed")
        if container.get("values_materialized") is True:
            errors.append("values_materialized")
        if container.get("source_payloads_read") is True:
            errors.append("source_payloads_read")
        if container.get("source_payload_read") is True:
            errors.append("source_payload_read")
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
    materialization_plan_keys = (
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
        "value_source_manifest_id": (value_source_keys, ("value_source_manifest_id",)),
        "writer_input_binding_plan_id": (writer_chain_keys, ("writer_input_binding_plan_id",)),
        "writer_execution_request_id": (writer_request_keys, ("writer_execution_request_id",)),
        "materialization_plan_id": (materialization_plan_keys, ("materialization_plan_id",)),
        "row_contract_id": (materialization_plan_keys, ("row_contract_id",)),
        "execution_ledger_id": (_payload_keys(payloads), ("execution_ledger_id",)),
        "dataset_name": (materialization_plan_keys, ("dataset_name",)),
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
    manifest = payloads["training_dataset_writer_value_source_manifest"]
    preflight = payloads["training_dataset_writer_value_source_manifest_preflight"]
    planner = payloads["training_dataset_writer_value_source_manifest_planner_summary"]
    records = _safe_records(manifest.get("value_source_records"))
    record_ids = [str(record.get("value_source_record_id", "")) for record in records]
    if len(records) < max(minimum_value_source_records, 1):
        errors.append("no_value_source_records")
    for container in (manifest, preflight, planner):
        if int(container.get("value_source_record_count", -1)) != len(records):
            errors.append("value_source_record_count_mismatch")
        if _safe_list(container.get("value_source_record_ids")) != record_ids:
            errors.append("value_source_record_ids_mismatch")
    for record in records:
        if _value_contains_unsafe_material(record):
            errors.append("value_source_record_contains_unsafe_value")
        if record.get("value_field_name") not in _VALUE_FIELD_NAMES:
            errors.append("value_field_name_invalid")
        label = str(record.get("source_artifact_label", ""))
        if label not in _ALLOWED_SOURCE_ARTIFACT_LABELS:
            errors.append("source_artifact_label_invalid")
        basename = str(record.get("source_artifact_basename", ""))
        if not basename or Path(basename).name != basename or "/" in basename or "\\" in basename:
            errors.append("source_artifact_basename_not_safe")
        path_key = _SOURCE_LABEL_PATH_KEYS.get(label, "")
        if path_key:
            if record.get("source_artifact_sha256") != hashes.get(path_key.replace("_path", "_sha256")):
                errors.append("source_artifact_sha256_mismatch")
            if basename != Path(payloads["paths"][path_key]).name:
                errors.append("source_artifact_basename_mismatch")
        for field in (
            "source_payload_read",
            "value_materialized",
            "writer_executed",
            "training_dataset_materialized",
            "dataset_artifact_created",
            "dataset_confirmation_changed",
        ):
            if record.get(field) is not False:
                errors.append(field)
        if record.get("phase1_status") != "not_run":
            errors.append("phase1_ran")
    plan_candidates = set(_safe_list(payloads["training_dataset_writer_input_binding_plan"].get("planned_training_admission_candidate_record_ids")))
    request_plan = payloads["training_admission_request_plan"]
    if plan_candidates & (set(_safe_list(request_plan.get("planned_exclusion_record_ids"))) | set(_safe_list(request_plan.get("exclude_record_ids")))):
        errors.append("planned_candidate_from_excluded_record")
    if plan_candidates & (set(_safe_list(request_plan.get("blocked_from_training_admission_record_ids"))) | set(_safe_list(request_plan.get("blocked_record_ids")))):
        errors.append("planned_candidate_from_blocked_record")
    if plan_candidates & set(_safe_list(request_plan.get("needs_review_record_ids"))):
        errors.append("planned_candidate_from_needs_review_record")
    return _stable_unique(errors)


def _output_errors(payloads: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    request = payloads["training_dataset_writer_execution_request"]
    formats = _safe_list(request.get("requested_output_formats"))
    if not formats or not set(formats).issubset(_OUTPUT_FORMAT_LABELS):
        errors.append("output_format_label_invalid")
    for label in _safe_list(request.get("planned_output_artifact_labels")):
        if not label or "/" in label or "\\" in label or _value_contains_unsafe_material(label):
            errors.append("planned_output_artifact_label_invalid")
    return _stable_unique(errors)


def _plan(
    *,
    status: str,
    controlled_writer_execution_plan_id: str,
    created_by: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    manifest = payloads["training_dataset_writer_value_source_manifest"]
    request = payloads["training_dataset_writer_execution_request"]
    records = _safe_records(manifest.get("value_source_records"))
    formats = sorted(set(_safe_list(request.get("requested_output_formats"))) & _OUTPUT_FORMAT_LABELS)
    return {
        "schema_version": _PLAN_SCHEMA_VERSION,
        "controlled_writer_execution_plan_id": controlled_writer_execution_plan_id,
        "created_at": now_iso(),
        "created_by": created_by,
        "planner_status": status,
        "writer_execution_mode": "controlled_writer_execution_plan_only",
        **_source_fields(paths, hashes),
        **_source_ids(manifest),
        **_source_status_fields(payloads),
        "dataset_name": str(manifest.get("dataset_name", "")),
        "row_contract_id": str(manifest.get("row_contract_id", "")),
        "value_source_manifest_id": str(manifest.get("value_source_manifest_id", "")),
        "writer_input_binding_plan_id": str(manifest.get("writer_input_binding_plan_id", "")),
        "writer_execution_request_id": str(manifest.get("writer_execution_request_id", "")),
        "writer_executed": False,
        "source_payloads_read": False,
        "values_materialized": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "requested_output_formats": formats,
        "planned_output_artifact_labels": [f"property_training_dataset_{label}" for label in formats],
        "output_directory_policy": ["future_writer_supplied_output_directory", "no_paths_in_plan"],
        "file_naming_policy": ["dataset_name_plus_format_label", "collision_free_writer_required"],
        "row_count_expectations": {
            "value_source_record_count": len(records),
            "writer_request_record_count": len(_safe_list(manifest.get("writer_request_record_ids"))),
            "binding_record_count": len(_safe_list(manifest.get("binding_record_ids"))),
        },
        "allowed_value_field_names": sorted({str(record.get("value_field_name", "")) for record in records}),
        "allowed_source_artifacts": _allowed_source_artifacts(records),
        "value_source_record_count": len(records),
        "value_source_record_ids": [str(record.get("value_source_record_id", "")) for record in records],
        "binding_record_ids": _safe_list(manifest.get("binding_record_ids")),
        "writer_request_record_ids": _safe_list(manifest.get("writer_request_record_ids")),
        "planned_training_admission_candidate_record_ids": _safe_list(
            manifest.get("planned_training_admission_candidate_record_ids")
        ),
        "provenance_preservation_requirements": [
            "preserve_value_source_record_id",
            "preserve_writer_input_binding_record_id",
            "preserve_writer_request_record_id",
            "preserve_row_contract_sha256",
            "preserve_value_source_manifest_sha256",
            "preserve_source_artifact_sha256",
        ],
        "redaction_policy": [
            "no_source_payload_values_in_plan",
            "no_serialized_rows_in_plan",
            "no_output_paths_in_plan",
            "no_pdf_names_or_paths",
            "no_credentials",
        ],
        "planner_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
        "boundary_statement": [
            "controlled writer execution plan only",
            "writer not executed",
            "source payloads not read",
            "values not materialized",
            "no serialized training rows",
            "no training dataset materialization",
            "no training CSV/JSONL/Parquet/LMDB",
            "no candidate CSV/JSONL/Parquet/LMDB",
            "no conformer generation",
            "no DPA3 structure generation",
            "no Phase 1",
            "DatasetConfirmation unchanged",
            "no model training/evaluation",
        ],
    }


def _summary(
    *,
    status: str,
    controlled_writer_execution_plan_id: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    plan_path: Path,
    plan_sha256: str,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    manifest = payloads["training_dataset_writer_value_source_manifest"]
    records = _safe_records(manifest.get("value_source_records"))
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "planner_status": status,
        "controlled_writer_execution_plan_id": controlled_writer_execution_plan_id,
        "controlled_writer_execution_plan_path": plan_path.name,
        "controlled_writer_execution_plan_sha256": plan_sha256,
        **_source_fields(paths, hashes),
        **_source_ids(manifest),
        **_source_status_fields(payloads),
        "dataset_name": str(manifest.get("dataset_name", "")),
        "row_contract_id": str(manifest.get("row_contract_id", "")),
        "value_source_manifest_id": str(manifest.get("value_source_manifest_id", "")),
        "writer_input_binding_plan_id": str(manifest.get("writer_input_binding_plan_id", "")),
        "writer_execution_request_id": str(manifest.get("writer_execution_request_id", "")),
        "writer_executed": False,
        "source_payloads_read": False,
        "values_materialized": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "value_source_record_count": len(records),
        "value_source_record_ids": [str(record.get("value_source_record_id", "")) for record in records],
        "binding_record_ids": _safe_list(manifest.get("binding_record_ids")),
        "writer_request_record_ids": _safe_list(manifest.get("writer_request_record_ids")),
        "requested_output_formats": sorted(set(_safe_list(payloads["training_dataset_writer_execution_request"].get("requested_output_formats"))) & _OUTPUT_FORMAT_LABELS),
        "planned_output_artifact_labels": [
            f"property_training_dataset_{label}"
            for label in sorted(
                set(_safe_list(payloads["training_dataset_writer_execution_request"].get("requested_output_formats")))
                & _OUTPUT_FORMAT_LABELS
            )
        ],
        "planner_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _allowed_source_artifacts(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for record in records:
        key = (
            str(record.get("source_artifact_label", "")),
            str(record.get("source_artifact_basename", "")),
            str(record.get("source_artifact_sha256", "")),
        )
        by_key[key] = {
            "source_artifact_label": key[0],
            "source_artifact_basename": key[1],
            "source_artifact_sha256": key[2],
        }
    return [by_key[key] for key in sorted(by_key)]


def _source_ids(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        "corpus_id": str(manifest.get("corpus_id", "")),
        "admission_request_id": str(manifest.get("admission_request_id", "")),
        "review_manifest_id": str(manifest.get("review_manifest_id", "")),
        "review_queue_id": str(manifest.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(manifest.get("property_candidate_manifest_id", "")),
        "execution_ledger_id": str(manifest.get("execution_ledger_id", "")),
        "execution_request_id": str(manifest.get("execution_request_id", "")),
        "source_execution_request_id": str(manifest.get("source_execution_request_id", "")),
    }


def _source_status_fields(payloads: dict[str, Any]) -> dict[str, str]:
    return {
        "value_source_preflight_status": str(
            payloads["training_dataset_writer_value_source_manifest_preflight"].get("preflight_status", "")
        ),
        "value_source_manifest_status": str(payloads["training_dataset_writer_value_source_manifest"].get("planner_status", "")),
        "input_binding_preflight_status": str(
            payloads["training_dataset_writer_input_binding_plan_preflight"].get("preflight_status", "")
        ),
        "binding_plan_status": str(payloads["training_dataset_writer_input_binding_plan"].get("planner_status", "")),
        "writer_request_preflight_status": str(
            payloads["training_dataset_writer_execution_request_preflight"].get("preflight_status", "")
        ),
        "writer_request_status": str(payloads["training_dataset_writer_execution_request"].get("request_status", "")),
        "materialization_dry_run_status": str(
            payloads["training_dataset_materialization_dry_run_report"].get("dry_run_status", "")
        ),
        "contract_status": str(payloads["training_dataset_row_contract"].get("contract_status", "")),
        "plan_status": str(payloads["training_dataset_materialization_plan"].get("plan_status", "")),
        "ledger_status": str(payloads["training_admission_execution_ledger"].get("execution_status", "")),
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
            "# Custom Corpus Property Training Dataset Controlled Writer Execution Plan Evidence",
            "",
            f"- Planner status: `{summary['planner_status']}`",
            f"- Controlled writer execution plan id: `{summary['controlled_writer_execution_plan_id']}`",
            f"- Dataset name: `{summary['dataset_name']}`",
            f"- Value source manifest id: `{summary['value_source_manifest_id']}`",
            f"- Row contract id: `{summary['row_contract_id']}`",
            f"- Value source record count: `{summary['value_source_record_count']}`",
            f"- Requested output formats: `{json.dumps(summary['requested_output_formats'])}`",
            f"- Planner errors: `{json.dumps(summary['planner_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a controlled writer execution plan only.",
            "- no dataset writer was executed.",
            "- source payloads were not read.",
            "- no values were materialized.",
            "- no serialized training rows were created.",
            "- no training dataset materialization occurred.",
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


def _load_payloads(**paths: str | Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {"paths": paths}
    for key, path in paths.items():
        payloads[key.removesuffix("_path")] = _read_safe_json_dict(path, key.removesuffix("_path"))
    return payloads


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise PropertyTrainingDatasetControlledWriterExecutionPlanError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetControlledWriterExecutionPlanError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise PropertyTrainingDatasetControlledWriterExecutionPlanError(f"{label} contains forbidden material")
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
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "planner_status": "blocked",
        "planner_errors": ["property_training_dataset_controlled_writer_execution_plan_redaction_failed"],
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
