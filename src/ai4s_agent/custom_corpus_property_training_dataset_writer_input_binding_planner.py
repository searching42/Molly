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


_PLAN_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_input_binding_plan.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_input_binding_planner.v1"
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
_VALUE_FIELD_NAMES = {
    "property_name",
    "property_value",
    "property_unit",
    "property_value_normalized",
    "property_unit_normalized",
    "compound_id",
    "canonical_smiles",
}
_BOOLEAN_BOUNDARY_FIELDS = {
    "writer_executed",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "dataset_confirmation_changed",
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
    "inchi=",
)
_VALUE_FORBIDDEN_MARKERS = _FINAL_FORBIDDEN_MARKERS + (
    "canonical_smiles_value",
    "serialized training row",
    "serialized dataset row",
    "full candidate payload",
    "full materialized payload",
    "full row preview payload",
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


class PropertyTrainingDatasetWriterInputBindingPlannerError(ValueError):
    pass


def build_property_training_dataset_writer_input_binding_plan(
    *,
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
    writer_input_binding_plan_id: str,
    created_by: str,
    confirm_training_dataset_writer_input_binding_plan: bool = False,
    allow_writer_request_preflight_needs_review: bool = False,
    minimum_binding_records: int = 1,
    require_all_required_fields_bound: bool = True,
) -> dict[str, Any]:
    payloads = _load_payloads(
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
    run_dir = Path(output_dir).expanduser() / writer_input_binding_plan_id
    warnings: list[str] = []
    errors = _validation_errors(
        payloads,
        hashes,
        run_dir=run_dir,
        writer_input_binding_plan_id=writer_input_binding_plan_id,
        created_by=created_by,
        confirm=confirm_training_dataset_writer_input_binding_plan,
        allow_needs_review=allow_writer_request_preflight_needs_review,
        minimum_binding_records=minimum_binding_records,
        require_all_required_fields_bound=require_all_required_fields_bound,
        warnings=warnings,
    )
    plan_path = run_dir / "property_training_dataset_writer_input_binding_plan.json"
    summary_path = run_dir / "property_training_dataset_writer_input_binding_planner_summary.json"
    evidence_path = run_dir / "redacted_property_training_dataset_writer_input_binding_plan_evidence.md"
    if errors:
        return _summary(
            status="blocked",
            writer_input_binding_plan_id=writer_input_binding_plan_id,
            payloads=payloads,
            paths=paths,
            hashes=hashes,
            plan_path=plan_path,
            plan_sha256="",
            binding_records=[],
            errors=_stable_unique(errors),
            warnings=_stable_unique(warnings),
        )
    binding_records = _binding_records(payloads, hashes)
    status = "needs_review" if warnings else "planned"
    plan = _plan(
        status=status,
        writer_input_binding_plan_id=writer_input_binding_plan_id,
        created_by=created_by,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        binding_records=binding_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    plan_sha256 = _sha256_payload(plan)
    summary = _summary(
        status=status,
        writer_input_binding_plan_id=writer_input_binding_plan_id,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        binding_records=binding_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    markdown = _markdown(summary)
    if _contains_forbidden_material({"plan": plan, "summary": summary, "markdown": markdown}):
        return _minimal_redaction_failure()
    write_json(plan_path, plan)
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
        summary = build_property_training_dataset_writer_input_binding_plan(
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
            writer_input_binding_plan_id=args.writer_input_binding_plan_id,
            created_by=args.created_by,
            confirm_training_dataset_writer_input_binding_plan=args.confirm_training_dataset_writer_input_binding_plan,
            allow_writer_request_preflight_needs_review=args.allow_writer_request_preflight_needs_review,
            minimum_binding_records=args.minimum_binding_records,
            require_all_required_fields_bound=args.require_all_required_fields_bound,
        )
    except Exception as exc:
        err.write(f"property training dataset writer input binding planner invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("planner_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_dataset_writer_input_binding_planner",
        description="Plan safe input bindings for future property training dataset writing.",
    )
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
    parser.add_argument("--writer-input-binding-plan-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-training-dataset-writer-input-binding-plan", action="store_true")
    parser.add_argument("--allow-writer-request-preflight-needs-review", action="store_true")
    parser.add_argument("--minimum-binding-records", type=int, default=1)
    parser.add_argument("--require-all-required-fields-bound", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _load_payloads(**paths: str | Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {"paths": paths}
    for key, path in paths.items():
        payloads[key.removesuffix("_path")] = _read_safe_json_dict(path, key.removesuffix("_path"))
    return payloads


def _validation_errors(
    payloads: dict[str, Any],
    hashes: dict[str, str],
    *,
    run_dir: Path,
    writer_input_binding_plan_id: str,
    created_by: str,
    confirm: bool,
    allow_needs_review: bool,
    minimum_binding_records: int,
    require_all_required_fields_bound: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if not confirm:
        errors.append("confirmation_required")
    if not _is_safe_id(writer_input_binding_plan_id):
        errors.append("writer_input_binding_plan_id_invalid")
    if not created_by or "@" in created_by:
        errors.append("created_by_invalid")
    if run_dir.exists() and any(run_dir.iterdir()):
        errors.append("output_directory_not_clean")
    _append_errors(errors, _schema_errors(payloads))
    _append_errors(errors, _status_errors(payloads, allow_needs_review, warnings))
    _append_errors(errors, _hash_errors(payloads, hashes))
    _append_errors(errors, _id_errors(payloads))
    _append_errors(errors, _record_errors(payloads, minimum_binding_records))
    _append_errors(errors, _field_binding_errors(payloads, require_all_required_fields_bound, warnings))
    _append_errors(errors, _sha_format_errors(payloads))
    return _stable_unique(errors)


def _schema_errors(payloads: dict[str, Any]) -> list[str]:
    expected = {
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
    return [
        error
        for key, (error, schema) in expected.items()
        if payloads[key].get("schema_version") != schema
    ]


def _status_errors(payloads: dict[str, Any], allow_needs_review: bool, warnings: list[str]) -> list[str]:
    errors: list[str] = []
    status_checks = (
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
        _append_errors(errors, _status_field_errors(payloads[key], field, prefix, required, allow_needs_review, warnings))
    request = payloads["training_dataset_writer_execution_request"]
    if request.get("request_mode") != "training_dataset_writer_execution_request_only":
        errors.append("training_dataset_writer_execution_request_mode_invalid")
    if payloads["training_dataset_materialization_dry_run_report"].get("dry_run_mode") != "training_dataset_materialization_dry_run_only":
        errors.append("training_dataset_materialization_dry_run_mode_invalid")
    if payloads["training_dataset_row_contract"].get("contract_mode") != "training_dataset_row_contract_only":
        errors.append("training_dataset_row_contract_mode_invalid")
    if payloads["training_dataset_materialization_plan"].get("plan_mode") != "training_dataset_materialization_plan_only":
        errors.append("training_dataset_materialization_plan_mode_invalid")
    if request.get("writer_executed") is not False:
        errors.append("writer_executed")
    if not set(_safe_list(request.get("requested_output_formats"))).issubset(_OUTPUT_FORMAT_LABELS):
        errors.append("requested_output_format_invalid")
    for key in _payload_keys(payloads):
        container = payloads[key]
        if container.get("writer_executed") is True:
            errors.append("writer_executed")
        if container.get("training_dataset_materialized") is True:
            errors.append("training_dataset_materialized")
        if container.get("dataset_artifact_created") is True:
            errors.append("dataset_artifact_created")
        if container.get("phase1_status") not in {"", None, "not_run"}:
            errors.append("phase1_ran")
        if container.get("dataset_confirmation_changed") not in {"", None, False}:
            errors.append("dataset_confirmation_changed")
    for key in (
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
    for key, error_field, error_code in (
        ("training_dataset_writer_execution_request_preflight", "preflight_errors", "training_dataset_writer_execution_request_preflight_has_errors"),
        ("training_dataset_writer_execution_request", "request_errors", "training_dataset_writer_execution_request_has_errors"),
        ("training_dataset_writer_execution_request_summary", "request_errors", "training_dataset_writer_execution_request_summary_has_errors"),
        ("training_dataset_materialization_dry_run_precheck", "precheck_errors", "training_dataset_materialization_dry_run_precheck_has_errors"),
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


def _hash_errors(payloads: dict[str, Any], hashes: dict[str, str]) -> list[str]:
    errors: list[str] = []
    canonical_hashes = {
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
        "writer_execution_request_id": (
            (
                "training_dataset_writer_execution_request_preflight",
                "training_dataset_writer_execution_request",
                "training_dataset_writer_execution_request_summary",
            ),
            ("writer_execution_request_id",),
        ),
        "materialization_dry_run_id": (
            (
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
        "source_execution_request_id": (_payload_keys(payloads), ("source_execution_request_id",)),
        "dataset_name": (
            (
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
    preflight = payloads["training_dataset_writer_execution_request_preflight"]
    request = payloads["training_dataset_writer_execution_request"]
    summary = payloads["training_dataset_writer_execution_request_summary"]
    report = payloads["training_dataset_materialization_dry_run_report"]
    dry_summary = payloads["training_dataset_materialization_dry_run_summary"]
    contract = payloads["training_dataset_row_contract"]
    plan = payloads["training_dataset_materialization_plan"]
    ledger = payloads["training_admission_execution_ledger"]
    request_plan = payloads["training_admission_request_plan"]
    request_preflight = payloads["training_admission_request_preflight"]
    readiness = payloads["training_admission_readiness_summary"]
    records = _safe_records(request.get("writer_request_records"))
    previews = _safe_records(report.get("row_previews"))
    references = _safe_records(contract.get("contract_record_references"))
    plan_records = _safe_records(plan.get("planned_dataset_records"))
    ledger_records = _safe_records(ledger.get("ledger_records"))
    if len(records) < max(minimum_binding_records, 1):
        errors.append("minimum_binding_record_count_not_met")
    if not records:
        errors.append("no_writer_request_records")
    if not previews:
        errors.append("no_row_previews")
    planned_candidates = _safe_list(request.get("planned_training_admission_candidate_record_ids"))
    if not planned_candidates:
        errors.append("no_planned_candidates")
    if int(request.get("writer_request_record_count", -1)) != len(records):
        errors.append("writer_request_record_count_mismatch")
    if int(summary.get("writer_request_record_count", -1)) != len(records):
        errors.append("writer_request_record_count_mismatch")
    if request.get("binding_record_count") not in {"", None, len(records)}:
        errors.append("binding_record_count_mismatch")
    if preflight.get("binding_record_count") not in {"", None, len(records)}:
        errors.append("binding_record_count_mismatch")
    writer_ids = [str(record.get("writer_request_record_id", "")) for record in records]
    if _safe_list(request.get("writer_request_record_ids")) != writer_ids:
        errors.append("writer_request_record_ids_mismatch")
    if _safe_list(summary.get("writer_request_record_ids")) != writer_ids:
        errors.append("writer_request_record_ids_mismatch")
    if preflight.get("writer_request_record_ids") and _safe_list(preflight.get("writer_request_record_ids")) != writer_ids:
        errors.append("writer_request_record_ids_mismatch")
    row_preview_ids = [str(record.get("row_preview_id", "")) for record in records]
    report_preview_ids = [str(preview.get("row_preview_id", "")) for preview in previews]
    if row_preview_ids != report_preview_ids:
        errors.append("row_preview_ids_mismatch")
    if _safe_list(request.get("row_preview_ids")) != row_preview_ids:
        errors.append("row_preview_ids_mismatch")
    if _safe_list(summary.get("row_preview_ids")) != row_preview_ids:
        errors.append("row_preview_ids_mismatch")
    if int(report.get("row_preview_count", -1)) != len(previews):
        errors.append("row_preview_count_mismatch")
    if int(dry_summary.get("row_preview_count", -1)) != len(previews):
        errors.append("row_preview_count_mismatch")
    plan_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in plan_records]
    request_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in records]
    reference_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in references]
    preview_dataset_ids = [str(preview.get("planned_dataset_record_id", "")) for preview in previews]
    if request_dataset_ids != plan_dataset_ids or reference_dataset_ids != plan_dataset_ids or preview_dataset_ids != plan_dataset_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    if _safe_list(request.get("planned_dataset_record_ids")) != request_dataset_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    ledger_ids = [str(record.get("ledger_record_id", "")) for record in ledger_records]
    request_ledger_ids = [str(record.get("ledger_record_id", "")) for record in records]
    preview_ledger_ids = [str(preview.get("ledger_record_id", "")) for preview in previews]
    if set(request_ledger_ids) != set(ledger_ids) or set(preview_ledger_ids) != set(ledger_ids):
        errors.append("ledger_record_ids_mismatch")
    request_candidates = [str(record.get("candidate_record_id", "")) for record in records]
    preview_candidates = [str(preview.get("candidate_record_id", "")) for preview in previews]
    reference_candidates = [str(reference.get("candidate_record_id", "")) for reference in references]
    ledger_candidates = [str(record.get("candidate_record_id", "")) for record in ledger_records]
    if (
        set(request_candidates) != set(planned_candidates)
        or set(preview_candidates) != set(planned_candidates)
        or set(reference_candidates) != set(planned_candidates)
        or set(ledger_candidates) != set(planned_candidates)
        or set(_safe_list(summary.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates)
        or set(_safe_list(report.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates)
        or set(_safe_list(dry_summary.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates)
        or set(_safe_list(plan.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates)
        or set(_safe_list(preflight.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates)
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
    for record in records:
        candidate_id = str(record.get("candidate_record_id", ""))
        record_id = str(record.get("record_id", ""))
        if record.get("requested_action") != "write_training_dataset_row_later":
            errors.append("writer_request_record_action_invalid")
        if record.get("writer_request_record_status") != "requested":
            errors.append("writer_request_record_status_invalid")
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
        if candidate_id in excluded or record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if candidate_id in blocked or record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if candidate_id in needs_review or record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
        if not _writer_record_is_safe(record):
            errors.append("writer_request_record_contains_unsafe_value")
    return _stable_unique(errors)


def _field_binding_errors(
    payloads: dict[str, Any],
    require_all_required_fields_bound: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    missing = _missing_required_field_counts(payloads)
    if missing:
        if require_all_required_fields_bound:
            errors.append("required_field_source_missing")
        else:
            warnings.append("required_field_source_missing")
    return errors


def _binding_records(payloads: dict[str, Any], hashes: dict[str, str]) -> list[dict[str, Any]]:
    request = payloads["training_dataset_writer_execution_request"]
    contract = payloads["training_dataset_row_contract"]
    required_fields = _safe_list(contract.get("required_row_fields"))
    optional_fields = _safe_list(contract.get("optional_row_fields"))
    declarations = _source_declarations(payloads)
    records: list[dict[str, Any]] = []
    for index, record in enumerate(_safe_records(request.get("writer_request_records")), start=1):
        writer_request_record_id = str(record.get("writer_request_record_id", ""))
        binding_record_id = f"{request.get('writer_execution_request_id', 'writer-request')}-input-binding-{index:03d}"
        required_bindings = [
            _field_binding(field_name, record, declarations.get((writer_request_record_id, field_name)), hashes, required=True)
            for field_name in required_fields
        ]
        optional_bindings = [
            _field_binding(field_name, record, declarations.get((writer_request_record_id, field_name)), hashes, required=False)
            for field_name in optional_fields
        ]
        records.append(
            {
                "writer_input_binding_record_id": binding_record_id,
                "writer_request_record_id": writer_request_record_id,
                "row_preview_id": str(record.get("row_preview_id", "")),
                "contract_record_reference_id": str(record.get("contract_record_reference_id", "")),
                "planned_dataset_record_id": str(record.get("planned_dataset_record_id", "")),
                "ledger_record_id": str(record.get("ledger_record_id", "")),
                "candidate_record_id": str(record.get("candidate_record_id", "")),
                "record_id": str(record.get("record_id", "")),
                "document_id": str(record.get("document_id", "")),
                "field_name": str(record.get("field_name", "")),
                "dataset_name": str(record.get("dataset_name", "")),
                "row_contract_id": str(record.get("row_contract_id", "")),
                "materialization_plan_id": str(record.get("materialization_plan_id", "")),
                "materialization_dry_run_id": str(record.get("materialization_dry_run_id", "")),
                "writer_execution_request_id": str(request.get("writer_execution_request_id", "")),
                "binding_record_status": "planned",
                "writer_executed": False,
                "training_admitted": True,
                "training_dataset_materialized": False,
                "dataset_artifact_created": False,
                "phase1_status": "not_run",
                "dataset_confirmation_changed": False,
                "required_field_bindings": required_bindings,
                "optional_field_bindings": optional_bindings,
                "dedup_split_binding": _dedup_split_binding(),
                "requested_output_formats": _safe_list(record.get("requested_output_formats")),
                "target_model_families": _safe_list(record.get("target_model_families")),
                **_record_sha_fields(record),
            }
        )
    return records


def _field_binding(
    field_name: str,
    record: dict[str, Any],
    declaration: dict[str, Any] | None,
    hashes: dict[str, str],
    *,
    required: bool,
) -> dict[str, Any]:
    if declaration:
        return {
            "field_name": field_name,
            "binding_status": "bound",
            "source_artifact_label": str(declaration.get("source_artifact_label", "")),
            "source_artifact_sha256": str(declaration.get("source_artifact_sha256", "")),
            "source_record_id": str(declaration.get("source_record_id", "")),
            "derivation_rule": str(declaration.get("derivation_rule", "")),
            "value_materialized": False,
        }
    if field_name in _VALUE_FIELD_NAMES:
        return {
            "field_name": field_name,
            "binding_status": "missing_source",
            "source_artifact_label": "",
            "source_artifact_sha256": "",
            "source_record_id": str(record.get("candidate_record_id", "")),
            "derivation_rule": f"future_safe_binding_required_for_{field_name}",
            "value_materialized": False,
        }
    if field_name == "dataset_record_id":
        label, sha, source_record_id, rule = (
            "materialization_plan",
            hashes.get("training_dataset_materialization_plan_sha256", ""),
            str(record.get("planned_dataset_record_id", "")),
            "bind_from_planned_dataset_record_id",
        )
    elif field_name == "task_type":
        label, sha, source_record_id, rule = (
            "row_contract",
            hashes.get("training_dataset_row_contract_sha256", ""),
            str(record.get("row_contract_id", "")),
            "derive_property_prediction_task_type_from_row_contract",
        )
    elif field_name in {"source_artifact_sha256", "review_artifact_sha256", "admission_request_sha256"}:
        label, sha, source_record_id, rule = (
            "writer_execution_request",
            str(record.get(field_name, "")),
            str(record.get("writer_request_record_id", "")),
            f"bind_from_{field_name}",
        )
    elif field_name == "training_admission_execution_ledger_sha256":
        label, sha, source_record_id, rule = (
            "training_admission_execution_ledger",
            str(record.get("training_admission_execution_ledger_sha256", "")),
            str(record.get("ledger_record_id", "")),
            "bind_from_training_admission_execution_ledger_sha256",
        )
    elif field_name == "training_dataset_materialization_plan_sha256":
        label, sha, source_record_id, rule = (
            "materialization_plan",
            str(record.get("training_dataset_materialization_plan_sha256", "")),
            str(record.get("planned_dataset_record_id", "")),
            "bind_from_training_dataset_materialization_plan_sha256",
        )
    else:
        label, sha, source_record_id, rule = (
            "writer_execution_request",
            hashes.get("training_dataset_writer_execution_request_sha256", ""),
            str(record.get(field_name, record.get("writer_request_record_id", ""))),
            f"bind_from_writer_request_{field_name}",
        )
    return {
        "field_name": field_name,
        "binding_status": "derived_later" if field_name == "task_type" else "bound",
        "source_artifact_label": label,
        "source_artifact_sha256": sha,
        "source_record_id": source_record_id,
        "derivation_rule": rule,
        "value_materialized": False,
    }


def _source_declarations(payloads: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    hashes = {
        "writer_execution_request": _sha256_payload(payloads["training_dataset_writer_execution_request"]),
        "materialization_dry_run_report": _safe_sha_for_path(payloads["paths"]["training_dataset_materialization_dry_run_report_path"]),
        "row_contract": _safe_sha_for_path(payloads["paths"]["training_dataset_row_contract_path"]),
        "materialization_plan": _safe_sha_for_path(payloads["paths"]["training_dataset_materialization_plan_path"]),
        "training_admission_execution_ledger": _safe_sha_for_path(payloads["paths"]["training_admission_execution_ledger_path"]),
        "quarantine_candidate_records": _safe_sha_for_path(payloads["paths"]["quarantine_candidate_records_path"]),
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for declaration in _safe_records(
        payloads["training_dataset_writer_execution_request_preflight"].get("writer_input_field_source_declarations")
    ):
        writer_request_record_id = str(declaration.get("writer_request_record_id", ""))
        field_name = str(declaration.get("field_name", ""))
        label = str(declaration.get("source_artifact_label", ""))
        source_sha = str(declaration.get("source_artifact_sha256", ""))
        source_record_id = str(declaration.get("source_record_id", ""))
        derivation_rule = str(declaration.get("derivation_rule", ""))
        if (
            declaration.get("field_available") is True
            and _is_safe_id(writer_request_record_id)
            and _is_safe_id(field_name)
            and label in _ALLOWED_SOURCE_ARTIFACT_LABELS
            and source_sha == hashes.get(label, "")
            and _SHA_RE.match(source_sha)
            and source_record_id
            and _is_safe_id(source_record_id)
            and derivation_rule
            and _is_safe_id(derivation_rule)
        ):
            result[(writer_request_record_id, field_name)] = {
                "source_artifact_label": label,
                "source_artifact_sha256": source_sha,
                "source_record_id": source_record_id,
                "derivation_rule": derivation_rule,
            }
    return result


def _dedup_split_binding() -> dict[str, Any]:
    return {
        "dedup_key_rule": (
            "derive_from_canonical_molecule_identity_property_name_normalized_value_normalized_unit_source_artifact_hash"
        ),
        "split_group_key_rule": "default_to_canonical_molecule_identity_to_reduce_molecule_level_leakage",
        "dedup_key_materialized": False,
        "split_group_key_materialized": False,
        "split_group_key_default": "canonical_molecule_identity",
        "row_id_split_forbidden": True,
    }


def _plan(
    *,
    status: str,
    writer_input_binding_plan_id: str,
    created_by: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    binding_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    request = payloads["training_dataset_writer_execution_request"]
    contract = payloads["training_dataset_row_contract"]
    record_ids = [str(record.get("writer_input_binding_record_id", "")) for record in binding_records]
    writer_ids = [str(record.get("writer_request_record_id", "")) for record in binding_records]
    row_preview_ids = [str(record.get("row_preview_id", "")) for record in binding_records]
    return {
        "schema_version": _PLAN_SCHEMA_VERSION,
        "writer_input_binding_plan_id": writer_input_binding_plan_id,
        "created_at": now_iso(),
        "created_by": created_by,
        "planner_status": status,
        "plan_mode": "training_dataset_writer_input_binding_plan_only",
        "dataset_name": str(request.get("dataset_name", "")),
        "row_contract_id": str(request.get("row_contract_id", "")),
        "contract_version_label": str(request.get("contract_version_label", "")),
        "materialization_plan_id": str(request.get("materialization_plan_id", "")),
        "materialization_dry_run_id": str(request.get("materialization_dry_run_id", "")),
        "writer_execution_request_id": str(request.get("writer_execution_request_id", "")),
        **_source_fields(paths, hashes),
        **_source_ids(request),
        **_source_status_fields(payloads),
        "writer_executed": False,
        "training_admitted": True,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "binding_record_count": len(binding_records),
        "writer_request_record_count": len(_safe_records(request.get("writer_request_records"))),
        "row_preview_count": len(_safe_records(payloads["training_dataset_materialization_dry_run_report"].get("row_previews"))),
        "binding_record_ids": record_ids,
        "writer_request_record_ids": writer_ids,
        "row_preview_ids": row_preview_ids,
        "planned_training_admission_candidate_record_ids": _safe_list(
            request.get("planned_training_admission_candidate_record_ids")
        ),
        "required_field_names": _safe_list(contract.get("required_row_fields")),
        "optional_field_names": _safe_list(contract.get("optional_row_fields")),
        "binding_records": binding_records,
        "missing_required_field_counts": _missing_required_field_counts(payloads),
        "missing_optional_field_counts": _missing_optional_field_counts(payloads),
        "planner_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
        "boundary_statement": [
            "writer input binding plan only",
            "writer not executed",
            "values not materialized",
            "no serialized training rows",
            "no training dataset artifact",
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
    writer_input_binding_plan_id: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    plan_path: Path,
    plan_sha256: str,
    binding_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    request = payloads["training_dataset_writer_execution_request"]
    contract = payloads["training_dataset_row_contract"]
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "planner_status": status,
        "writer_input_binding_plan_id": writer_input_binding_plan_id,
        "training_dataset_writer_input_binding_plan_path": plan_path.name,
        "training_dataset_writer_input_binding_plan_sha256": plan_sha256,
        **_source_fields(paths, hashes),
        **_source_ids(request),
        **_source_status_fields(payloads),
        "dataset_name": str(request.get("dataset_name", "")),
        "row_contract_id": str(request.get("row_contract_id", "")),
        "materialization_plan_id": str(request.get("materialization_plan_id", "")),
        "materialization_dry_run_id": str(request.get("materialization_dry_run_id", "")),
        "writer_execution_request_id": str(request.get("writer_execution_request_id", "")),
        "writer_executed": False,
        "training_admitted": True if status in {"planned", "needs_review"} else request.get("training_admitted") is True,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "binding_record_count": len(binding_records),
        "binding_record_ids": [str(record.get("writer_input_binding_record_id", "")) for record in binding_records],
        "writer_request_record_ids": _safe_list(request.get("writer_request_record_ids")),
        "row_preview_ids": _safe_list(request.get("row_preview_ids")),
        "planned_training_admission_candidate_record_ids": _safe_list(
            request.get("planned_training_admission_candidate_record_ids")
        ),
        "required_field_names": _safe_list(contract.get("required_row_fields")),
        "optional_field_names": _safe_list(contract.get("optional_row_fields")),
        "missing_required_field_counts": _missing_required_field_counts(payloads),
        "missing_optional_field_counts": _missing_optional_field_counts(payloads),
        "planner_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _source_ids(request: dict[str, Any]) -> dict[str, str]:
    return {
        "corpus_id": str(request.get("corpus_id", "")),
        "admission_request_id": str(request.get("admission_request_id", "")),
        "review_manifest_id": str(request.get("review_manifest_id", "")),
        "review_queue_id": str(request.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(request.get("property_candidate_manifest_id", "")),
        "execution_ledger_id": str(request.get("execution_ledger_id", "")),
        "execution_request_id": str(request.get("execution_request_id", "")),
        "source_execution_request_id": str(request.get("source_execution_request_id", "")),
    }


def _source_status_fields(payloads: dict[str, Any]) -> dict[str, str]:
    return {
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


def _source_fields(paths: dict[str, str | Path], hashes: dict[str, str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, path in paths.items():
        fields[key] = Path(path).name or key
        fields[key.replace("_path", "_sha256")] = hashes[key.replace("_path", "_sha256")]
    return fields


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Dataset Writer Input Binding Plan Evidence",
            "",
            f"- Planner status: `{summary['planner_status']}`",
            f"- Writer input binding plan id: `{summary['writer_input_binding_plan_id']}`",
            f"- Dataset name: `{summary['dataset_name']}`",
            f"- Row contract id: `{summary['row_contract_id']}`",
            f"- Writer execution request id: `{summary['writer_execution_request_id']}`",
            f"- Binding record count: `{summary['binding_record_count']}`",
            f"- Missing required fields: `{json.dumps(summary['missing_required_field_counts'], sort_keys=True)}`",
            f"- Missing optional fields: `{json.dumps(summary['missing_optional_field_counts'], sort_keys=True)}`",
            f"- Planner errors: `{json.dumps(summary['planner_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training dataset writer input binding plan only.",
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


def _missing_required_field_counts(payloads: dict[str, Any]) -> dict[str, int]:
    return _missing_field_counts(payloads, required=True)


def _missing_optional_field_counts(payloads: dict[str, Any]) -> dict[str, int]:
    return _missing_field_counts(payloads, required=False)


def _missing_field_counts(payloads: dict[str, Any], *, required: bool) -> dict[str, int]:
    request = payloads["training_dataset_writer_execution_request"]
    contract = payloads["training_dataset_row_contract"]
    fields = _safe_list(contract.get("required_row_fields" if required else "optional_row_fields"))
    declarations = _source_declarations(payloads)
    counts: dict[str, int] = {}
    for record in _safe_records(request.get("writer_request_records")):
        writer_request_record_id = str(record.get("writer_request_record_id", ""))
        for field_name in fields:
            if field_name in _VALUE_FIELD_NAMES and (writer_request_record_id, field_name) not in declarations:
                counts[field_name] = counts.get(field_name, 0) + 1
            elif not required and field_name not in _VALUE_FIELD_NAMES:
                counts[field_name] = counts.get(field_name, 0) + 1
    return dict(sorted(counts.items()))


def _record_sha_fields(record: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in record.items()
        if key.endswith("_sha256") and isinstance(value, str) and _SHA_RE.match(value)
    }


def _writer_record_is_safe(record: dict[str, Any]) -> bool:
    for key, value in record.items():
        if key.endswith("_sha256"):
            if value and not _SHA_RE.match(str(value)):
                return False
            continue
        if key in _BOOLEAN_BOUNDARY_FIELDS or key == "training_admitted":
            if not isinstance(value, bool):
                return False
            continue
        if isinstance(value, list):
            if not all(isinstance(item, str) and _is_safe_id(item) for item in value):
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
        raise PropertyTrainingDatasetWriterInputBindingPlannerError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetWriterInputBindingPlannerError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise PropertyTrainingDatasetWriterInputBindingPlannerError(f"{label} contains forbidden material")
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
        "planner_errors": ["property_training_dataset_writer_input_binding_planner_redaction_failed"],
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
