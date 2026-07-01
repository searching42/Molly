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


_REPORT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run.v1"
_SUMMARY_SCHEMA_VERSION = (
    "custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run_summary.v1"
)
_PLAN_PREFLIGHT_SCHEMA_VERSION = (
    "custom_corpus_property_training_dataset_controlled_writer_execution_plan_preflight.v1"
)
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
    "0.72",
    "c1=cc",
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
_VALUE_FORBIDDEN_MARKERS = _FINAL_FORBIDDEN_MARKERS + (
    "full source payload",
    "full upstream payload",
    "full candidate payload",
    "full writer request payload",
)


class PropertyTrainingDatasetControlledWriterValueResolutionDryRunError(ValueError):
    pass


def run_property_training_dataset_controlled_writer_value_resolution_dry_run(
    *,
    training_dataset_controlled_writer_execution_plan_preflight_path: str | Path,
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
    output_dir: str | Path,
    value_resolution_dry_run_id: str,
    created_by: str,
    confirm_training_dataset_controlled_writer_value_resolution_dry_run: bool = False,
    allow_controlled_writer_execution_plan_preflight_needs_review: bool = False,
    minimum_resolution_records: int = 1,
    require_all_required_fields_resolved: bool = True,
) -> dict[str, Any]:
    payloads = _load_payloads(
        training_dataset_controlled_writer_execution_plan_preflight_path=training_dataset_controlled_writer_execution_plan_preflight_path,
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
    run_dir = Path(output_dir).expanduser() / value_resolution_dry_run_id
    warnings: list[str] = []
    errors: list[str] = []
    if not confirm_training_dataset_controlled_writer_value_resolution_dry_run:
        errors.append("confirmation_required")
    if not _is_safe_id(value_resolution_dry_run_id):
        errors.append("value_resolution_dry_run_id_invalid")
    if created_by and not _is_safe_id(created_by):
        errors.append("created_by_invalid")
    if run_dir.exists() and any(run_dir.iterdir()):
        errors.append("output_directory_not_clean")
    _append_errors(errors, _schema_errors(payloads))
    _append_errors(
        errors,
        _status_errors(payloads, allow_controlled_writer_execution_plan_preflight_needs_review, warnings),
    )
    _append_errors(errors, _hash_errors(payloads, hashes))
    _append_errors(errors, _id_errors(payloads))
    plan_content_errors = _plan_content_errors(payloads, hashes)
    if allow_controlled_writer_execution_plan_preflight_needs_review:
        for warning_code in sorted({"source_artifact_label_unauthorized"} & set(plan_content_errors)):
            warnings.append(warning_code)
        plan_content_errors = [error for error in plan_content_errors if error != "source_artifact_label_unauthorized"]
    _append_errors(errors, plan_content_errors)
    source_payloads = _authorized_source_payloads(payloads, hashes, errors)
    resolution_records, resolution_errors, resolution_warnings = _resolution_records(
        payloads,
        source_payloads,
        value_resolution_dry_run_id,
        require_all_required_fields_resolved=require_all_required_fields_resolved,
        minimum_resolution_records=minimum_resolution_records,
    )
    if allow_controlled_writer_execution_plan_preflight_needs_review:
        downgraded = {
            "required_field_unresolved",
            "source_artifact_label_unauthorized",
            "source_payload_record_missing",
        }
        for warning_code in sorted(downgraded & set(resolution_errors)):
            warnings.append(warning_code)
        resolution_errors = [error for error in resolution_errors if error not in downgraded]
    if resolution_errors and not require_all_required_fields_resolved:
        warnings.extend(resolution_errors)
    else:
        errors.extend(resolution_errors)
    warnings.extend(resolution_warnings)
    _append_errors(errors, _unsafe_input_errors(payloads))
    status = "blocked" if errors else "needs_review" if warnings else "passed"
    report_path = run_dir / "property_training_dataset_controlled_writer_value_resolution_dry_run_report.json"
    summary_path = run_dir / "property_training_dataset_controlled_writer_value_resolution_dry_run_summary.json"
    markdown_path = run_dir / "redacted_property_training_dataset_controlled_writer_value_resolution_dry_run_evidence.md"
    if status == "blocked":
        return _summary(
            status=status,
            value_resolution_dry_run_id=value_resolution_dry_run_id,
            payloads=payloads,
            paths=paths,
            hashes=hashes,
            report_path=report_path,
            report_sha256="",
            resolution_records=[],
            errors=_stable_unique(errors),
            warnings=_stable_unique(warnings),
        )
    report = _report(
        status=status,
        value_resolution_dry_run_id=value_resolution_dry_run_id,
        created_by=created_by,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        resolution_records=resolution_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    summary = _summary(
        status=status,
        value_resolution_dry_run_id=value_resolution_dry_run_id,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        report_path=report_path,
        report_sha256=_sha256_payload(report),
        resolution_records=resolution_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    markdown = _markdown(summary)
    if _contains_forbidden_material({"report": report, "summary": summary, "markdown": markdown}):
        return _minimal_redaction_failure()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    report_sha256 = sha256_file(report_path)
    summary = _summary(
        status=status,
        value_resolution_dry_run_id=value_resolution_dry_run_id,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        report_path=report_path,
        report_sha256=report_sha256,
        resolution_records=resolution_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    markdown = _markdown(summary)
    if _contains_forbidden_material({"summary": summary, "markdown": markdown}):
        report_path.unlink(missing_ok=True)
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
        summary = run_property_training_dataset_controlled_writer_value_resolution_dry_run(
            training_dataset_controlled_writer_execution_plan_preflight_path=args.training_dataset_controlled_writer_execution_plan_preflight,
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
            output_dir=args.output_dir,
            value_resolution_dry_run_id=args.value_resolution_dry_run_id,
            created_by=args.created_by,
            confirm_training_dataset_controlled_writer_value_resolution_dry_run=args.confirm_training_dataset_controlled_writer_value_resolution_dry_run,
            allow_controlled_writer_execution_plan_preflight_needs_review=args.allow_controlled_writer_execution_plan_preflight_needs_review,
            minimum_resolution_records=args.minimum_resolution_records,
            require_all_required_fields_resolved=args.require_all_required_fields_resolved,
        )
    except Exception as exc:
        err.write(f"property training dataset controlled writer value resolution dry-run invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("dry_run_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "ai4s_agent.custom_corpus_property_training_dataset_controlled_writer_value_resolution_dry_run"
        ),
        description="Dry-run controlled value resolution without writing a property training dataset.",
    )
    parser.add_argument("--training-dataset-controlled-writer-execution-plan-preflight", required=True)
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--value-resolution-dry-run-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-training-dataset-controlled-writer-value-resolution-dry-run", action="store_true")
    parser.add_argument("--allow-controlled-writer-execution-plan-preflight-needs-review", action="store_true")
    parser.add_argument("--minimum-resolution-records", type=int, default=1)
    parser.add_argument("--require-all-required-fields-resolved", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _schema_errors(payloads: dict[str, Any]) -> list[str]:
    expected = {
        "training_dataset_controlled_writer_execution_plan_preflight": ("controlled_writer_execution_plan_preflight_schema_invalid", _PLAN_PREFLIGHT_SCHEMA_VERSION),
        "training_dataset_controlled_writer_execution_plan": ("controlled_writer_execution_plan_schema_invalid", _PLAN_SCHEMA_VERSION),
        "training_dataset_controlled_writer_execution_planner_summary": ("controlled_writer_execution_planner_summary_schema_invalid", _PLANNER_SCHEMA_VERSION),
        "training_dataset_writer_value_source_manifest_preflight": ("training_dataset_writer_value_source_manifest_preflight_schema_invalid", _VALUE_SOURCE_PREFLIGHT_SCHEMA_VERSION),
        "training_dataset_writer_value_source_manifest": ("training_dataset_writer_value_source_manifest_schema_invalid", _VALUE_SOURCE_MANIFEST_SCHEMA_VERSION),
        "training_dataset_writer_value_source_manifest_planner_summary": ("training_dataset_writer_value_source_manifest_planner_summary_schema_invalid", _VALUE_SOURCE_PLANNER_SCHEMA_VERSION),
        "training_dataset_writer_input_binding_plan_preflight": ("training_dataset_writer_input_binding_plan_preflight_schema_invalid", _INPUT_PREFLIGHT_SCHEMA_VERSION),
        "training_dataset_writer_input_binding_plan": ("training_dataset_writer_input_binding_plan_schema_invalid", _BINDING_PLAN_SCHEMA_VERSION),
        "training_dataset_writer_input_binding_planner_summary": ("training_dataset_writer_input_binding_planner_summary_schema_invalid", _BINDING_PLANNER_SCHEMA_VERSION),
        "training_dataset_writer_execution_request_preflight": ("training_dataset_writer_execution_request_preflight_schema_invalid", _WRITER_PREFLIGHT_SCHEMA_VERSION),
        "training_dataset_writer_execution_request": ("training_dataset_writer_execution_request_schema_invalid", _WRITER_REQUEST_SCHEMA_VERSION),
        "training_dataset_writer_execution_request_summary": ("training_dataset_writer_execution_request_summary_schema_invalid", _WRITER_REQUEST_SUMMARY_SCHEMA_VERSION),
        "training_dataset_materialization_dry_run_precheck": ("training_dataset_materialization_dry_run_precheck_schema_invalid", _DRY_RUN_PREFLIGHT_SCHEMA_VERSION),
        "training_dataset_materialization_dry_run_report": ("training_dataset_materialization_dry_run_report_schema_invalid", _DRY_RUN_REPORT_SCHEMA_VERSION),
        "training_dataset_materialization_dry_run_summary": ("training_dataset_materialization_dry_run_summary_schema_invalid", _DRY_RUN_SUMMARY_SCHEMA_VERSION),
        "training_dataset_row_contract_precheck": ("training_dataset_row_contract_precheck_schema_invalid", _ROW_CONTRACT_PREFLIGHT_SCHEMA_VERSION),
        "training_dataset_row_contract": ("training_dataset_row_contract_schema_invalid", _ROW_CONTRACT_SCHEMA_VERSION),
        "training_dataset_row_contract_summary": ("training_dataset_row_contract_summary_schema_invalid", _ROW_CONTRACT_SUMMARY_SCHEMA_VERSION),
        "training_dataset_materialization_plan_precheck": ("training_dataset_materialization_plan_precheck_schema_invalid", _MATERIALIZATION_PLAN_PREFLIGHT_SCHEMA_VERSION),
        "training_dataset_materialization_plan": ("training_dataset_materialization_plan_schema_invalid", _MATERIALIZATION_PLAN_SCHEMA_VERSION),
        "training_dataset_materialization_planner_summary": ("training_dataset_materialization_planner_summary_schema_invalid", _MATERIALIZATION_PLANNER_SCHEMA_VERSION),
        "training_admission_execution_ledger_precheck": ("training_admission_execution_ledger_precheck_schema_invalid", _LEDGER_PREFLIGHT_SCHEMA_VERSION),
        "training_admission_execution_ledger": ("training_admission_execution_ledger_schema_invalid", _LEDGER_SCHEMA_VERSION),
        "training_admission_execution_ledger_summary": ("training_admission_execution_ledger_summary_schema_invalid", _LEDGER_SUMMARY_SCHEMA_VERSION),
        "training_admission_execution_dry_run_precheck": ("training_admission_execution_dry_run_precheck_schema_invalid", _TRAINING_DRY_RUN_PREFLIGHT_SCHEMA_VERSION),
        "training_admission_execution_dry_run_report": ("training_admission_execution_dry_run_schema_invalid", _TRAINING_DRY_RUN_SCHEMA_VERSION),
        "training_admission_execution_request": ("training_admission_execution_request_schema_invalid", _TRAINING_EXECUTION_REQUEST_SCHEMA_VERSION),
        "training_admission_execution_request_summary": ("training_admission_execution_request_summary_schema_invalid", _TRAINING_EXECUTION_REQUEST_SUMMARY_SCHEMA_VERSION),
        "training_admission_execution_request_preflight": ("training_admission_execution_request_preflight_schema_invalid", _TRAINING_EXECUTION_PREFLIGHT_SCHEMA_VERSION),
        "training_admission_request_draft": ("training_admission_request_draft_schema_invalid", _DRAFT_SCHEMA_VERSION),
        "training_admission_request_draft_summary": ("training_admission_request_draft_summary_schema_invalid", _DRAFT_SUMMARY_SCHEMA_VERSION),
        "training_admission_request_draft_precheck": ("training_admission_request_draft_precheck_schema_invalid", _DRAFT_PREFLIGHT_SCHEMA_VERSION),
        "training_admission_request_plan": ("training_admission_request_plan_schema_invalid", _REQUEST_PLAN_SCHEMA_VERSION),
        "training_admission_request_preflight": ("training_admission_request_preflight_schema_invalid", _REQUEST_PREFLIGHT_SCHEMA_VERSION),
        "training_admission_readiness_summary": ("training_admission_readiness_summary_schema_invalid", _READINESS_SCHEMA_VERSION),
        "quarantine_candidate_preflight_summary": ("quarantine_candidate_preflight_summary_schema_invalid", _QUARANTINE_PREFLIGHT_SCHEMA_VERSION),
        "quarantine_candidate_records": ("quarantine_candidate_records_schema_invalid", _QUARANTINE_CANDIDATE_SCHEMA_VERSION),
    }
    return [error for key, (error, schema) in expected.items() if payloads[key].get("schema_version") != schema]


def _status_errors(payloads: dict[str, Any], allow_preflight_needs_review: bool, warnings: list[str]) -> list[str]:
    errors: list[str] = []
    checks = (
        ("training_dataset_controlled_writer_execution_plan_preflight", "preflight_status", "controlled_writer_execution_plan_preflight", "passed"),
        ("training_dataset_controlled_writer_execution_plan", "planner_status", "controlled_writer_execution_plan", "planned"),
        ("training_dataset_controlled_writer_execution_planner_summary", "planner_status", "controlled_writer_execution_planner_summary", "planned"),
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
        errors.extend(_status_field_errors(payloads[payload_key], field, prefix, required, allow_preflight_needs_review, warnings))
    plan = payloads["training_dataset_controlled_writer_execution_plan"]
    if plan.get("writer_execution_mode") != "controlled_writer_execution_plan_only":
        errors.append("writer_execution_mode_invalid")
    for key in _payload_keys(payloads):
        container = payloads[key]
        if container.get("controlled_writer_executed") is True:
            errors.append("controlled_writer_executed")
        if container.get("writer_executed") is True:
            errors.append("writer_executed")
        if container.get("values_materialized") is True:
            errors.append("values_materialized")
        if container.get("serialized_rows_created") is True:
            errors.append("serialized_rows_created")
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
    groups = {
        "corpus_id": (_payload_keys(payloads), ("corpus_id",)),
        "controlled_writer_execution_plan_id": (
            (
                "training_dataset_controlled_writer_execution_plan_preflight",
                "training_dataset_controlled_writer_execution_plan",
                "training_dataset_controlled_writer_execution_planner_summary",
            ),
            ("controlled_writer_execution_plan_id",),
        ),
        "value_source_manifest_id": (
            (
                "training_dataset_controlled_writer_execution_plan_preflight",
                "training_dataset_controlled_writer_execution_plan",
                "training_dataset_controlled_writer_execution_planner_summary",
                "training_dataset_writer_value_source_manifest_preflight",
                "training_dataset_writer_value_source_manifest",
                "training_dataset_writer_value_source_manifest_planner_summary",
            ),
            ("value_source_manifest_id",),
        ),
        "writer_input_binding_plan_id": (_payload_keys(payloads), ("writer_input_binding_plan_id",)),
        "writer_execution_request_id": (_payload_keys(payloads), ("writer_execution_request_id",)),
        "row_contract_id": (_payload_keys(payloads), ("row_contract_id",)),
        "execution_ledger_id": (_payload_keys(payloads), ("execution_ledger_id",)),
        "dataset_name": (_payload_keys(payloads), ("dataset_name",)),
    }
    errors: list[str] = []
    for logical, (payload_keys, fields) in groups.items():
        values = _group_values(payloads, payload_keys, fields)
        if len(set(values)) > 1:
            errors.append(f"{logical}_mismatch")
        if any(value and not _is_safe_id(value) for value in values):
            errors.append(f"{logical}_invalid")
    return _stable_unique(errors)


def _plan_content_errors(payloads: dict[str, Any], hashes: dict[str, str]) -> list[str]:
    errors: list[str] = []
    plan = payloads["training_dataset_controlled_writer_execution_plan"]
    requested = _safe_list(plan.get("requested_output_formats"))
    if not requested or set(requested) - _OUTPUT_FORMAT_LABELS:
        errors.append("output_format_label_invalid")
    labels = _safe_list(plan.get("planned_output_artifact_labels"))
    if not labels or any(not _is_label_only(label) or _value_contains_unsafe_material(label) for label in labels):
        errors.append("planned_output_artifact_label_invalid")
    allowed_sources = _safe_records(plan.get("allowed_source_artifacts"))
    if not allowed_sources:
        errors.append("source_artifact_label_unauthorized")
    for artifact in allowed_sources:
        label = str(artifact.get("source_artifact_label", ""))
        basename = str(artifact.get("source_artifact_basename", ""))
        if label not in _SOURCE_LABEL_PATH_KEYS:
            errors.append("source_artifact_label_invalid")
            continue
        if not basename or Path(basename).name != basename or "/" in basename or "\\" in basename:
            errors.append("source_artifact_basename_not_safe")
        path_key = _SOURCE_LABEL_PATH_KEYS[label]
        if basename != Path(payloads["paths"][path_key]).name:
            errors.append("source_artifact_basename_mismatch")
        if artifact.get("source_artifact_sha256") != hashes.get(path_key.replace("_path", "_sha256")):
            errors.append("source_artifact_sha256_mismatch")
    return _stable_unique(errors)


def _authorized_source_payloads(
    payloads: dict[str, Any],
    hashes: dict[str, str],
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    plan = payloads["training_dataset_controlled_writer_execution_plan"]
    authorized: dict[str, dict[str, Any]] = {}
    for artifact in _safe_records(plan.get("allowed_source_artifacts")):
        label = str(artifact.get("source_artifact_label", ""))
        path_key = _SOURCE_LABEL_PATH_KEYS.get(label)
        if not path_key:
            continue
        authorized[label] = payloads[path_key.replace("_path", "")]
    return authorized


def _resolution_records(
    payloads: dict[str, Any],
    source_payloads: dict[str, dict[str, Any]],
    value_resolution_dry_run_id: str,
    *,
    require_all_required_fields_resolved: bool,
    minimum_resolution_records: int,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    binding_records = _safe_records(payloads["training_dataset_writer_input_binding_plan"].get("binding_records"))
    value_source_records = _safe_records(payloads["training_dataset_writer_value_source_manifest"].get("value_source_records"))
    if len(binding_records) < max(minimum_resolution_records, 1):
        errors.append("no_resolution_records")
    records: list[dict[str, Any]] = []
    for index, binding in enumerate(binding_records, start=1):
        writer_binding_id = str(binding.get("writer_input_binding_record_id", ""))
        required_bindings = _safe_records(binding.get("required_field_bindings"))
        optional_bindings = _safe_records(binding.get("optional_field_bindings"))
        required_fields = [str(item.get("field_name", "")) for item in required_bindings]
        optional_fields = [str(item.get("field_name", "")) for item in optional_bindings]
        required_result = _resolve_bindings(
            required_bindings,
            writer_binding_id,
            value_source_records,
            source_payloads,
        )
        optional_result = _resolve_bindings(
            optional_bindings,
            writer_binding_id,
            value_source_records,
            source_payloads,
        )
        missing_required = sorted(set(required_fields) - set(required_result["resolved_fields"]))
        missing_optional = sorted(set(optional_fields) - set(optional_result["resolved_fields"]))
        if missing_required:
            errors.append("required_field_unresolved")
        if required_result["missing_payload_record"]:
            errors.append("source_payload_record_missing")
        if required_result["unauthorized_source"]:
            errors.append("source_artifact_label_unauthorized")
        if optional_result["missing_payload_record"] or missing_optional:
            warnings.append("optional_field_unresolved")
        records.append(
            {
                "value_resolution_record_id": f"{value_resolution_dry_run_id}-resolution-{index:03d}",
                "writer_request_record_id": str(binding.get("writer_request_record_id", "")),
                "writer_input_binding_record_id": writer_binding_id,
                "row_preview_id": str(binding.get("row_preview_id", "")),
                "planned_dataset_record_id": str(binding.get("planned_dataset_record_id", "")),
                "candidate_record_id": str(binding.get("candidate_record_id", "")),
                "record_id": str(binding.get("record_id", "")),
                "document_id": str(binding.get("document_id", "")),
                "field_name": str(binding.get("field_name", "")),
                "required_field_resolution_status": "resolved" if not missing_required else "missing",
                "optional_field_resolution_status": "resolved" if not missing_optional else "partial",
                "resolved_required_field_names": sorted(required_result["resolved_fields"]),
                "missing_required_field_names": missing_required,
                "resolved_optional_field_names": sorted(optional_result["resolved_fields"]),
                "missing_optional_field_names": missing_optional,
                "value_source_record_ids": sorted(required_result["value_source_record_ids"]),
                "source_artifact_labels": sorted(required_result["source_artifact_labels"]),
                "source_artifact_sha256s": sorted(required_result["source_artifact_sha256s"]),
                "derivation_rule_labels": sorted(required_result["derivation_rule_labels"]),
                "controlled_writer_executed": False,
                "source_payloads_read": True,
                "values_materialized": False,
                "serialized_rows_created": False,
                "training_dataset_materialized": False,
                "dataset_artifact_created": False,
                "phase1_status": "not_run",
                "dataset_confirmation_changed": False,
            }
        )
    if errors and not require_all_required_fields_resolved:
        return records, ["required_field_unresolved"] if "required_field_unresolved" in errors else [], warnings
    return records, _stable_unique(errors), _stable_unique(warnings)


def _resolve_bindings(
    bindings: list[dict[str, Any]],
    writer_binding_id: str,
    value_source_records: list[dict[str, Any]],
    source_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resolved_fields: set[str] = set()
    value_source_record_ids: set[str] = set()
    source_artifact_labels: set[str] = set()
    source_artifact_sha256s: set[str] = set()
    derivation_rule_labels: set[str] = set()
    missing_payload_record = False
    unauthorized_source = False
    value_records_by_field = {
        str(record.get("value_field_name", "")): record
        for record in value_source_records
        if record.get("writer_input_binding_record_id") == writer_binding_id
    }
    for binding in bindings:
        field_name = str(binding.get("field_name", ""))
        if binding.get("binding_status") not in {"bound", "derived_later"}:
            continue
        source_label = str(binding.get("source_artifact_label", ""))
        source_record_id = str(binding.get("source_record_id", ""))
        derivation_rule = str(binding.get("derivation_rule", ""))
        source_sha = str(binding.get("source_artifact_sha256", ""))
        if field_name in _VALUE_FIELD_NAMES:
            value_record = value_records_by_field.get(field_name)
            if not value_record:
                continue
            source_label = str(value_record.get("source_artifact_label", source_label))
            source_record_id = str(value_record.get("source_record_id", source_record_id))
            derivation_rule = str(value_record.get("derivation_rule", derivation_rule))
            source_sha = str(value_record.get("source_artifact_sha256", source_sha))
            value_source_record_ids.add(str(value_record.get("value_source_record_id", "")))
        if source_label not in source_payloads:
            if field_name not in _VALUE_FIELD_NAMES and source_label in _SOURCE_LABEL_PATH_KEYS:
                resolved_fields.add(field_name)
                source_artifact_labels.add(source_label)
                if source_sha:
                    source_artifact_sha256s.add(source_sha)
                if derivation_rule:
                    derivation_rule_labels.add(derivation_rule)
                continue
            unauthorized_source = True
            continue
        if source_record_id and not _payload_has_record_id(source_payloads[source_label], source_record_id):
            missing_payload_record = True
            continue
        resolved_fields.add(field_name)
        source_artifact_labels.add(source_label)
        if source_sha:
            source_artifact_sha256s.add(source_sha)
        if derivation_rule:
            derivation_rule_labels.add(derivation_rule)
    return {
        "resolved_fields": resolved_fields,
        "value_source_record_ids": value_source_record_ids,
        "source_artifact_labels": source_artifact_labels,
        "source_artifact_sha256s": source_artifact_sha256s,
        "derivation_rule_labels": derivation_rule_labels,
        "missing_payload_record": missing_payload_record,
        "unauthorized_source": unauthorized_source,
    }


def _payload_has_record_id(payload: Any, record_id: str) -> bool:
    if not record_id:
        return True
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.endswith("_id") and value == record_id:
                return True
            if isinstance(value, (dict, list)) and _payload_has_record_id(value, record_id):
                return True
    if isinstance(payload, list):
        return any(_payload_has_record_id(item, record_id) for item in payload)
    return False


def _unsafe_input_errors(payloads: dict[str, Any]) -> list[str]:
    plan = dict(payloads["training_dataset_controlled_writer_execution_plan"])
    plan.pop("boundary_statement", None)
    if _value_contains_unsafe_material(plan):
        return ["controlled_writer_value_resolution_input_contains_unsafe_value"]
    return []


def _report(
    *,
    status: str,
    value_resolution_dry_run_id: str,
    created_by: str,
    payloads: dict[str, Any],
    paths: dict[str, Path],
    hashes: dict[str, str],
    resolution_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    base = _safe_base_fields(status, value_resolution_dry_run_id, payloads, paths, hashes, resolution_records, errors, warnings)
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "value_resolution_dry_run_id": value_resolution_dry_run_id,
        "created_at": now_iso(),
        "created_by": created_by,
        "dry_run_status": status,
        "dry_run_mode": "controlled_writer_value_resolution_dry_run_only",
        **base,
        "resolution_record_ids": [record["value_resolution_record_id"] for record in resolution_records],
        "resolution_records": resolution_records,
        "source_artifact_read_summary": _source_artifact_read_summary(payloads, hashes),
        "boundary_statement": [
            "controlled_writer_value_resolution_dry_run_only",
            "controlled_writer_not_executed",
            "authorized_source_payloads_may_be_read",
            "values_not_emitted",
            "values_not_materialized",
            "serialized_rows_not_created",
            "training_dataset_not_materialized",
            "dataset_artifact_not_created",
            "phase1_not_run",
            "dataset_confirmation_unchanged",
        ],
    }


def _summary(
    *,
    status: str,
    value_resolution_dry_run_id: str,
    payloads: dict[str, Any],
    paths: dict[str, Path],
    hashes: dict[str, str],
    report_path: Path,
    report_sha256: str,
    resolution_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "dry_run_status": status,
        "value_resolution_dry_run_id": value_resolution_dry_run_id,
        "controlled_writer_value_resolution_dry_run_report_path": report_path.name,
        "controlled_writer_value_resolution_dry_run_report_sha256": report_sha256,
        **_safe_base_fields(status, value_resolution_dry_run_id, payloads, paths, hashes, resolution_records, errors, warnings),
    }


def _safe_base_fields(
    status: str,
    value_resolution_dry_run_id: str,
    payloads: dict[str, Any],
    paths: dict[str, Path],
    hashes: dict[str, str],
    resolution_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    plan = payloads["training_dataset_controlled_writer_execution_plan"]
    preflight = payloads["training_dataset_controlled_writer_execution_plan_preflight"]
    missing_required = sum(len(record.get("missing_required_field_names", [])) for record in resolution_records)
    missing_optional = sum(len(record.get("missing_optional_field_names", [])) for record in resolution_records)
    path_fields = {f"{key.replace('_path', '')}_path": Path(path).name for key, path in paths.items()}
    path_fields["controlled_writer_execution_plan_preflight_path"] = Path(
        paths["training_dataset_controlled_writer_execution_plan_preflight_path"]
    ).name
    return {
        **path_fields,
        **hashes,
        "controlled_writer_execution_plan_preflight_status": preflight.get("preflight_status", ""),
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
        "requested_output_formats": _safe_list(plan.get("requested_output_formats")),
        "planned_output_artifact_labels": _safe_list(plan.get("planned_output_artifact_labels")),
        "value_source_record_count": int(plan.get("value_source_record_count", 0) or 0),
        "binding_record_count": len(_safe_records(payloads["training_dataset_writer_input_binding_plan"].get("binding_records"))),
        "writer_request_record_count": len(
            _safe_records(payloads["training_dataset_writer_execution_request"].get("writer_request_records"))
        ),
        "resolution_record_count": len(resolution_records),
        "resolved_resolution_record_count": sum(
            1 for record in resolution_records if record.get("required_field_resolution_status") == "resolved"
        ),
        "missing_required_field_count": missing_required,
        "missing_optional_field_count": missing_optional,
        "controlled_writer_executed": False,
        "source_payloads_read": True,
        "values_resolved": missing_required == 0 and bool(resolution_records),
        "values_materialized": False,
        "serialized_rows_created": False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "model_training_run": False,
        "evaluation_run": False,
        "dry_run_errors": errors,
        "warnings": warnings,
        "redaction_status": "passed",
    }


def _source_artifact_read_summary(payloads: dict[str, Any], hashes: dict[str, str]) -> list[dict[str, str]]:
    summary = []
    for artifact in _safe_records(payloads["training_dataset_controlled_writer_execution_plan"].get("allowed_source_artifacts")):
        label = str(artifact.get("source_artifact_label", ""))
        path_key = _SOURCE_LABEL_PATH_KEYS.get(label)
        if not path_key:
            continue
        summary.append(
            {
                "source_artifact_label": label,
                "source_artifact_basename": Path(payloads["paths"][path_key]).name,
                "source_artifact_sha256": hashes.get(path_key.replace("_path", "_sha256"), ""),
                "source_payload_read": True,
            }
        )
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Property Training Dataset Controlled Writer Value Resolution Dry-Run",
        "",
        f"- Dry-run status: {summary.get('dry_run_status', '')}",
        f"- Value resolution dry-run id: {summary.get('value_resolution_dry_run_id', '')}",
        f"- Controlled writer execution plan id: {summary.get('controlled_writer_execution_plan_id', '')}",
        f"- Dataset name: {summary.get('dataset_name', '')}",
        f"- Resolution record count: {summary.get('resolution_record_count', 0)}",
        f"- Missing required field count: {summary.get('missing_required_field_count', 0)}",
        f"- Missing optional field count: {summary.get('missing_optional_field_count', 0)}",
        f"- Dry-run errors: {', '.join(summary.get('dry_run_errors', [])) or 'none'}",
        f"- Warnings: {', '.join(summary.get('warnings', [])) or 'none'}",
        "",
        "## Boundary Statement",
        "",
        "this is a controlled writer value resolution dry-run only.",
        "The controlled writer was not executed, but authorized source payloads may be read.",
        "Values may be resolved internally but were not emitted.",
        "Values were not materialized into rows.",
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
        raise PropertyTrainingDatasetControlledWriterValueResolutionDryRunError("input_redaction_failed")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetControlledWriterValueResolutionDryRunError("json_object_required")
    return payload


def _payload_keys(payloads: dict[str, Any]) -> list[str]:
    return [key for key in payloads if key != "paths"]


def _group_values(payloads: dict[str, Any], payload_keys: list[str] | tuple[str, ...], fields: tuple[str, ...]) -> list[str]:
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
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "dry_run_status": "blocked",
        "dry_run_errors": [
            "property_training_dataset_controlled_writer_value_resolution_dry_run_redaction_failed"
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
