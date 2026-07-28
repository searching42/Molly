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


_REPORT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_dry_run.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_dry_run_summary.v1"
_ROW_CONTRACT_PRECHECK_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract_precheck.v1"
_ROW_CONTRACT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract.v1"
_ROW_CONTRACT_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract_builder.v1"
_PLAN_PRECHECK_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_plan_precheck.v1"
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
_ALLOWED_FIELD_TYPES = {
    "string",
    "number",
    "boolean",
    "array[string]",
    "nullable[string]",
    "nullable[number]",
}
_ALLOWED_QUALITY_FLAGS = {
    "unit_normalized",
    "value_normalized",
    "source_reviewed",
    "human_review_bound",
    "ledger_admitted",
    "needs_unit_review",
    "needs_structure_review",
    "needs_property_review",
}
_MODEL_FAMILY_LABELS = {"generic_property_predictor", "unimol", "dpa3"}
_OUTPUT_FORMAT_LABELS = {"jsonl", "parquet", "lmdb", "csv"}


class PropertyTrainingDatasetMaterializationDryRunError(ValueError):
    pass


def run_property_training_dataset_materialization_dry_run(
    *,
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
    materialization_dry_run_id: str,
    created_by: str,
    confirm_training_dataset_materialization_dry_run: bool = False,
    allow_row_contract_precheck_needs_review: bool = False,
    minimum_dry_run_records: int = 1,
) -> dict[str, Any]:
    payloads = _load_payloads(
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
    run_dir = Path(output_dir).expanduser() / materialization_dry_run_id
    warnings: list[str] = []
    errors = _validation_errors(
        payloads,
        hashes,
        run_dir=run_dir,
        materialization_dry_run_id=materialization_dry_run_id,
        created_by=created_by,
        confirm=confirm_training_dataset_materialization_dry_run,
        allow_needs_review=allow_row_contract_precheck_needs_review,
        minimum_dry_run_records=minimum_dry_run_records,
        warnings=warnings,
    )
    status = "blocked" if errors else "needs_review" if warnings else "passed"
    report_path = run_dir / "property_training_dataset_materialization_dry_run_report.json"
    summary_path = run_dir / "property_training_dataset_materialization_dry_run_summary.json"
    evidence_path = run_dir / "redacted_property_training_dataset_materialization_dry_run_evidence.md"
    if status == "blocked":
        return _summary(
            status=status,
            materialization_dry_run_id=materialization_dry_run_id,
            payloads=payloads,
            paths=paths,
            hashes=hashes,
            report_path=report_path,
            report_sha256="",
            row_previews=[],
            errors=_stable_unique(errors),
            warnings=_stable_unique(warnings),
        )
    row_previews = _row_previews(payloads, hashes, materialization_dry_run_id)
    report = _report(
        status=status,
        materialization_dry_run_id=materialization_dry_run_id,
        created_by=created_by,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        row_previews=row_previews,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    report_sha256 = _sha256_payload(report)
    summary = _summary(
        status=status,
        materialization_dry_run_id=materialization_dry_run_id,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        report_path=report_path,
        report_sha256=report_sha256,
        row_previews=row_previews,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    markdown = _markdown(summary)
    if _contains_forbidden_material({"report": report, "summary": summary, "markdown": markdown}):
        return _minimal_redaction_failure()
    write_json(report_path, report)
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
        summary = run_property_training_dataset_materialization_dry_run(
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
            materialization_dry_run_id=args.materialization_dry_run_id,
            created_by=args.created_by,
            confirm_training_dataset_materialization_dry_run=args.confirm_training_dataset_materialization_dry_run,
            allow_row_contract_precheck_needs_review=args.allow_row_contract_precheck_needs_review,
            minimum_dry_run_records=args.minimum_dry_run_records,
        )
    except Exception as exc:
        err.write(f"property training dataset materialization dry-run invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("dry_run_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_dataset_materialization_dry_run",
        description="Run a property training dataset materialization dry-run without writing dataset artifacts.",
    )
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
    parser.add_argument("--materialization-dry-run-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-training-dataset-materialization-dry-run", action="store_true")
    parser.add_argument("--allow-row-contract-precheck-needs-review", action="store_true")
    parser.add_argument("--minimum-dry-run-records", type=int, default=1)
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
    materialization_dry_run_id: str,
    created_by: str,
    confirm: bool,
    allow_needs_review: bool,
    minimum_dry_run_records: int,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if not confirm:
        errors.append("confirmation_required")
    if not _is_safe_id(materialization_dry_run_id):
        errors.append("materialization_dry_run_id_invalid")
    if not created_by or "@" in created_by:
        errors.append("created_by_invalid")
    if run_dir.exists() and any(run_dir.iterdir()):
        errors.append("output_directory_not_clean")
    _append_errors(errors, _schema_errors(payloads))
    _append_errors(errors, _status_errors(payloads, allow_needs_review, warnings))
    _append_errors(errors, _contract_structure_errors(payloads))
    _append_errors(errors, _hash_errors(payloads, hashes))
    _append_errors(errors, _id_errors(payloads))
    _append_errors(errors, _record_errors(payloads, minimum_dry_run_records))
    _append_errors(errors, _sha_format_errors(payloads))
    return _stable_unique(errors)


def _schema_errors(payloads: dict[str, Any]) -> list[str]:
    expected = {
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
            _DRY_RUN_PREFLIGHT_SCHEMA_VERSION,
        ),
        "training_admission_execution_dry_run_report": (
            "training_admission_execution_dry_run_schema_invalid",
            _DRY_RUN_SCHEMA_VERSION,
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


def _status_errors(payloads: dict[str, Any], allow: bool, warnings: list[str]) -> list[str]:
    errors: list[str] = []
    status_checks = (
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
    if payloads["training_dataset_row_contract"].get("contract_mode") != "training_dataset_row_contract_only":
        errors.append("training_dataset_row_contract_mode_invalid")
    if payloads["training_dataset_materialization_plan"].get("plan_mode") != "training_dataset_materialization_plan_only":
        errors.append("training_dataset_materialization_plan_mode_invalid")
    if payloads["training_admission_execution_request"].get("request_mode") != "execution_request_only":
        errors.append("training_admission_execution_request_mode_invalid")
    if payloads["training_admission_request_draft"].get("request_mode") != "draft_only":
        errors.append("training_admission_request_draft_mode_invalid")
    if payloads["quarantine_candidate_records"].get("materialization_mode") != "candidate_quarantine":
        errors.append("quarantine_materialization_mode_invalid")
    for key in ("training_dataset_row_contract", "training_dataset_row_contract_summary", "training_dataset_row_contract_precheck"):
        if payloads[key].get("training_admitted") is not True:
            errors.append("training_not_admitted")
        if payloads[key].get("training_dataset_materialized") is not False:
            errors.append("training_dataset_materialized")
        if payloads[key].get("dataset_artifact_created") is not False:
            errors.append("dataset_artifact_created")
    for key in _payload_keys(payloads):
        if payloads[key].get("phase1_status") != "not_run":
            errors.append("phase1_ran")
        if payloads[key].get("dataset_confirmation_changed") is not False:
            errors.append("dataset_confirmation_changed")
    for key, error_field, error_code in (
        ("training_dataset_row_contract", "contract_errors", "training_dataset_row_contract_has_errors"),
        ("training_dataset_row_contract_precheck", "precheck_errors", "training_dataset_row_contract_precheck_has_errors"),
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


def _contract_structure_errors(payloads: dict[str, Any]) -> list[str]:
    contract = payloads["training_dataset_row_contract"]
    plan = payloads["training_dataset_materialization_plan"]
    errors: list[str] = []
    if not _safe_list(contract.get("required_row_fields")):
        errors.append("required_row_fields_missing")
    if not _safe_list(contract.get("optional_row_fields")):
        errors.append("optional_row_fields_missing")
    field_contract = contract.get("field_type_contract")
    if not isinstance(field_contract, dict) or any(value not in _ALLOWED_FIELD_TYPES for value in field_contract.values()):
        errors.append("field_type_descriptor_invalid")
    quality = contract.get("quality_flag_contract")
    if not isinstance(quality, dict) or not set(_safe_list(quality.get("allowed_quality_flags"))).issubset(_ALLOWED_QUALITY_FLAGS):
        errors.append("quality_flag_contract_invalid")
    split_dedup = contract.get("split_dedup_contract")
    if not isinstance(split_dedup, dict) or not {"dedup_key", "split_group_key"}.issubset(set(_safe_list(split_dedup.get("required_keys")))):
        errors.append("split_dedup_contract_invalid")
    model_contract = contract.get("model_family_compatibility_contract")
    if not isinstance(model_contract, dict) or not _MODEL_FAMILY_LABELS.issubset(set(model_contract)):
        errors.append("model_family_compatibility_invalid")
    output_contract = contract.get("output_format_compatibility_contract")
    if (
        not isinstance(output_contract, dict)
        or not _OUTPUT_FORMAT_LABELS.issubset(set(output_contract))
    ):
        errors.append("output_format_compatibility_invalid")
    if not _safe_records(plan.get("planned_dataset_records")):
        errors.append("no_planned_dataset_records")
    if not _safe_list(plan.get("planned_training_admission_candidate_record_ids")):
        errors.append("no_planned_candidates")
    if not _safe_records(contract.get("contract_record_references")):
        errors.append("no_contract_record_references")
    return _stable_unique(errors)


def _hash_errors(payloads: dict[str, Any], hashes: dict[str, str]) -> list[str]:
    errors: list[str] = []
    row_precheck = payloads["training_dataset_row_contract_precheck"]
    contract = payloads["training_dataset_row_contract"]
    contract_summary = payloads["training_dataset_row_contract_summary"]
    if row_precheck.get("training_dataset_row_contract_sha256") != hashes["training_dataset_row_contract_sha256"]:
        errors.append("training_dataset_row_contract_sha256_mismatch")
    if contract_summary.get("training_dataset_row_contract_sha256") != hashes["training_dataset_row_contract_sha256"]:
        errors.append("training_dataset_row_contract_sha256_mismatch")
    source_hash_keys = [
        key.replace("_path", "_sha256")
        for key in payloads["paths"]
        if key
        not in {
            "training_dataset_row_contract_precheck_path",
            "training_dataset_row_contract_path",
            "training_dataset_row_contract_summary_path",
        }
    ]
    for hash_key in source_hash_keys:
        for container in (row_precheck, contract, contract_summary):
            if container.get(hash_key) and container.get(hash_key) != hashes[hash_key]:
                errors.append(f"{hash_key}_mismatch")
    direct_pairs = (
        ("training_dataset_materialization_plan_precheck", "training_dataset_materialization_plan_sha256"),
        ("training_dataset_materialization_plan_precheck", "training_admission_execution_ledger_sha256"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_sha256"),
        ("training_dataset_materialization_plan", "training_admission_execution_ledger_precheck_sha256"),
        ("training_dataset_materialization_planner_summary", "training_dataset_materialization_plan_sha256"),
        ("training_admission_execution_ledger_precheck", "training_admission_execution_ledger_sha256"),
        ("training_admission_execution_ledger_summary", "training_admission_execution_ledger_sha256"),
        ("training_admission_execution_dry_run_precheck", "training_admission_execution_dry_run_report_sha256"),
        ("training_admission_execution_request_summary", "training_admission_execution_request_sha256"),
        ("training_admission_execution_request_preflight", "training_admission_execution_request_sha256"),
        ("training_admission_request_draft_summary", "training_admission_request_draft_sha256"),
        ("training_admission_request_draft_precheck", "training_admission_request_draft_sha256"),
        ("training_admission_request_plan", "training_admission_readiness_summary_sha256"),
        ("training_admission_request_preflight", "training_admission_request_plan_sha256"),
        ("training_admission_readiness_summary", "quarantine_candidate_records_sha256"),
    )
    for key, hash_key in direct_pairs:
        container = payloads[key]
        if container.get(hash_key) and container.get(hash_key) != hashes[hash_key]:
            errors.append(f"{hash_key}_mismatch")
    return _stable_unique(errors)


def _id_errors(payloads: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    groups = {
        "corpus_id": (_payload_keys(payloads), ("corpus_id",)),
        "materialization_plan_id": (
            (
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
            ("training_dataset_row_contract_precheck", "training_dataset_row_contract", "training_dataset_row_contract_summary"),
            ("row_contract_id",),
        ),
        "execution_ledger_id": (
            (
                "training_dataset_row_contract_precheck",
                "training_dataset_row_contract",
                "training_dataset_row_contract_summary",
                "training_dataset_materialization_plan_precheck",
                "training_dataset_materialization_plan",
                "training_dataset_materialization_planner_summary",
                "training_admission_execution_ledger_precheck",
                "training_admission_execution_ledger",
                "training_admission_execution_ledger_summary",
            ),
            ("execution_ledger_id",),
        ),
        "execution_request_id": (
            (
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


def _record_errors(payloads: dict[str, Any], minimum_dry_run_records: int) -> list[str]:
    errors: list[str] = []
    contract = payloads["training_dataset_row_contract"]
    plan = payloads["training_dataset_materialization_plan"]
    ledger = payloads["training_admission_execution_ledger"]
    request_plan = payloads["training_admission_request_plan"]
    request_preflight = payloads["training_admission_request_preflight"]
    readiness = payloads["training_admission_readiness_summary"]
    quarantine_candidate = payloads["quarantine_candidate_records"]
    references = _safe_records(contract.get("contract_record_references"))
    plan_records = _safe_records(plan.get("planned_dataset_records"))
    ledger_records = _safe_records(ledger.get("ledger_records"))
    if len(references) < max(minimum_dry_run_records, 1):
        errors.append("minimum_dry_run_record_count_not_met")
    if int(contract.get("contract_record_reference_count", -1)) != len(references):
        errors.append("contract_record_reference_count_mismatch")
    if len(references) != len(plan_records):
        errors.append("row_preview_count_mismatch")
    plan_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in plan_records]
    ref_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in references]
    if ref_dataset_ids != plan_dataset_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    ledger_ids = [str(record.get("ledger_record_id", "")) for record in ledger_records]
    ref_ledger_ids = [str(record.get("ledger_record_id", "")) for record in references]
    if set(ref_ledger_ids) != set(ledger_ids):
        errors.append("ledger_record_ids_mismatch")
    planned_candidates = _safe_list(plan.get("planned_training_admission_candidate_record_ids"))
    ref_candidates = [str(record.get("candidate_record_id", "")) for record in references]
    if set(ref_candidates) != set(planned_candidates):
        errors.append("planned_candidate_ids_mismatch")
    if not set(planned_candidates).issubset(set(_safe_list(quarantine_candidate.get("candidate_record_ids")))):
        errors.append("planned_candidate_ids_unknown")
    excluded = set(_safe_list(request_plan.get("exclude_record_ids")))
    excluded.update(_safe_list(request_preflight.get("exclude_record_ids")))
    blocked = set(_safe_list(request_plan.get("blocked_record_ids")))
    blocked.update(_safe_list(request_plan.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(request_preflight.get("blocked_from_training_admission_record_ids")))
    blocked.update(_safe_list(readiness.get("blocked_from_training_admission_record_ids")))
    needs_review = set(_safe_list(request_plan.get("needs_review_record_ids")))
    needs_review.update(_safe_list(request_preflight.get("needs_review_record_ids")))
    for record in references:
        candidate_id = str(record.get("candidate_record_id", ""))
        record_id = str(record.get("record_id", ""))
        if candidate_id in excluded or record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if candidate_id in blocked or record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if candidate_id in needs_review or record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
        if not _reference_is_safe(record):
            errors.append("contract_record_reference_contains_unsafe_value")
    return _stable_unique(errors)


def _row_previews(payloads: dict[str, Any], hashes: dict[str, str], dry_run_id: str) -> list[dict[str, Any]]:
    contract = payloads["training_dataset_row_contract"]
    previews: list[dict[str, Any]] = []
    required = _safe_list(contract.get("required_row_fields"))
    optional = _safe_list(contract.get("optional_row_fields"))
    quality_flags = _safe_list(contract.get("quality_flag_contract", {}).get("allowed_quality_flags"))
    for index, reference in enumerate(_safe_records(contract.get("contract_record_references")), start=1):
        target_model_families = _safe_list(reference.get("target_model_families"))
        planned_output_formats = _safe_list(reference.get("planned_output_formats"))
        previews.append(
            {
                "row_preview_id": f"{dry_run_id}-row-preview-{index:03d}",
                "contract_record_reference_id": str(reference.get("contract_record_reference_id", "")),
                "planned_dataset_record_id": str(reference.get("planned_dataset_record_id", "")),
                "ledger_record_id": str(reference.get("ledger_record_id", "")),
                "candidate_record_id": str(reference.get("candidate_record_id", "")),
                "record_id": str(reference.get("record_id", "")),
                "document_id": str(reference.get("document_id", "")),
                "field_name": str(reference.get("field_name", "")),
                "dataset_name": str(contract.get("dataset_name", "")),
                "contract_version_label": str(contract.get("contract_version_label", "")),
                "row_contract_id": str(contract.get("row_contract_id", "")),
                "would_materialize_row": True,
                "row_preview_status": "would_materialize",
                "required_field_count": len(required),
                "optional_field_count": len(optional),
                "required_row_fields": required,
                "optional_row_fields": optional,
                "missing_required_fields": [],
                "missing_optional_fields": list(optional),
                "quality_flag_labels": quality_flags,
                "target_model_families": target_model_families,
                "planned_output_formats": planned_output_formats,
                "training_admitted": True,
                "training_dataset_materialized": False,
                "dataset_artifact_created": False,
                "phase1_status": "not_run",
                "dataset_confirmation_changed": False,
                "source_artifact_sha256": str(reference.get("source_artifact_sha256", "")),
                "review_artifact_sha256": str(reference.get("review_artifact_sha256", "")),
                "admission_request_sha256": str(reference.get("admission_request_sha256", "")),
                "training_admission_execution_ledger_sha256": hashes["training_admission_execution_ledger_sha256"],
                "training_dataset_materialization_plan_sha256": hashes["training_dataset_materialization_plan_sha256"],
                "training_dataset_row_contract_sha256": hashes["training_dataset_row_contract_sha256"],
                "training_dataset_row_contract_precheck_sha256": hashes[
                    "training_dataset_row_contract_precheck_sha256"
                ],
                "quarantine_candidate_records_sha256": hashes["quarantine_candidate_records_sha256"],
            }
        )
    return previews


def _report(
    *,
    status: str,
    materialization_dry_run_id: str,
    created_by: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    row_previews: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    contract = payloads["training_dataset_row_contract"]
    plan = payloads["training_dataset_materialization_plan"]
    source = _source_fields(paths, hashes)
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "materialization_dry_run_id": materialization_dry_run_id,
        "created_at": now_iso(),
        "created_by": created_by,
        "dry_run_status": status,
        "dry_run_mode": "training_dataset_materialization_dry_run_only",
        "dataset_name": str(contract.get("dataset_name", "")),
        "row_contract_id": str(contract.get("row_contract_id", "")),
        "contract_version_label": str(contract.get("contract_version_label", "")),
        "materialization_plan_id": str(plan.get("materialization_plan_id", "")),
        **source,
        **_source_status_fields(payloads),
        "training_admitted": True,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "row_preview_count": len(row_previews),
        "planned_dataset_record_count": len(_safe_records(plan.get("planned_dataset_records"))),
        "contract_record_reference_count": len(_safe_records(contract.get("contract_record_references"))),
        "planned_training_admission_candidate_record_ids": _safe_list(
            plan.get("planned_training_admission_candidate_record_ids")
        ),
        "row_preview_ids": [preview["row_preview_id"] for preview in row_previews],
        "row_previews": row_previews,
        "field_coverage_summary": _field_coverage_summary(contract, row_previews),
        "model_family_compatibility_summary": _model_family_summary(row_previews),
        "output_format_compatibility_summary": _output_format_summary(row_previews),
        "dry_run_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
        "boundary_statement": [
            "training dataset materialization dry-run only",
            "row previews are summaries only",
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
    materialization_dry_run_id: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    report_path: Path,
    report_sha256: str,
    row_previews: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    contract = payloads["training_dataset_row_contract"]
    plan = payloads["training_dataset_materialization_plan"]
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "dry_run_status": status,
        "materialization_dry_run_id": materialization_dry_run_id,
        "training_dataset_materialization_dry_run_report_path": report_path.name,
        "training_dataset_materialization_dry_run_report_sha256": report_sha256,
        **_source_fields(paths, hashes),
        **_source_status_fields(payloads),
        "dataset_name": str(contract.get("dataset_name", "")),
        "row_contract_id": str(contract.get("row_contract_id", "")),
        "contract_version_label": str(contract.get("contract_version_label", "")),
        "materialization_plan_id": str(plan.get("materialization_plan_id", "")),
        "row_preview_count": len(row_previews),
        "row_preview_ids": [preview["row_preview_id"] for preview in row_previews],
        "planned_dataset_record_count": len(_safe_records(plan.get("planned_dataset_records"))),
        "contract_record_reference_count": len(_safe_records(contract.get("contract_record_references"))),
        "planned_training_admission_candidate_record_ids": _safe_list(
            plan.get("planned_training_admission_candidate_record_ids")
        ),
        "training_admitted": True if status in {"passed", "needs_review"} else False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "dry_run_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _field_coverage_summary(contract: dict[str, Any], row_previews: list[dict[str, Any]]) -> dict[str, Any]:
    required = _safe_list(contract.get("required_row_fields"))
    optional = _safe_list(contract.get("optional_row_fields"))
    missing_required: dict[str, int] = {}
    missing_optional = {field: 0 for field in optional}
    for preview in row_previews:
        for field in _safe_list(preview.get("missing_required_fields")):
            missing_required[field] = missing_required.get(field, 0) + 1
        for field in _safe_list(preview.get("missing_optional_fields")):
            missing_optional[field] = missing_optional.get(field, 0) + 1
    return {
        "required_field_count": len(required),
        "optional_field_count": len(optional),
        "required_row_fields": required,
        "optional_row_fields": optional,
        "row_preview_count": len(row_previews),
        "missing_required_field_counts": missing_required,
        "missing_optional_field_counts": missing_optional,
    }


def _model_family_summary(row_previews: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {label: 0 for label in sorted(_MODEL_FAMILY_LABELS)}
    for preview in row_previews:
        for label in _safe_list(preview.get("target_model_families")):
            if label in counts:
                counts[label] += 1
    return {
        "counts_by_model_family": counts,
        "unimol_requires_future_conformer_generation": True,
        "dpa3_requires_future_structure_generation": True,
        "conformers_generated": False,
        "dpa3_structures_generated": False,
    }


def _output_format_summary(row_previews: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {label: 0 for label in sorted(_OUTPUT_FORMAT_LABELS)}
    for preview in row_previews:
        for label in _safe_list(preview.get("planned_output_formats")):
            if label in counts:
                counts[label] += 1
    return {
        "counts_by_output_format": counts,
        "jsonl_created": False,
        "parquet_created": False,
        "lmdb_created": False,
        "csv_created": False,
    }


def _source_status_fields(payloads: dict[str, Any]) -> dict[str, str]:
    return {
        "row_contract_precheck_status": str(payloads["training_dataset_row_contract_precheck"].get("precheck_status", "")),
        "contract_status": str(payloads["training_dataset_row_contract"].get("contract_status", "")),
        "materialization_plan_precheck_status": str(
            payloads["training_dataset_materialization_plan_precheck"].get("precheck_status", "")
        ),
        "plan_status": str(payloads["training_dataset_materialization_plan"].get("plan_status", "")),
        "ledger_status": str(payloads["training_admission_execution_ledger"].get("execution_status", "")),
        "corpus_id": str(payloads["training_dataset_materialization_plan"].get("corpus_id", "")),
        "execution_ledger_id": str(payloads["training_dataset_materialization_plan"].get("execution_ledger_id", "")),
        "execution_request_id": str(payloads["training_dataset_materialization_plan"].get("execution_request_id", "")),
        "source_execution_request_id": str(
            payloads["training_dataset_materialization_plan"].get("source_execution_request_id", "")
        ),
        "review_manifest_id": str(payloads["training_dataset_materialization_plan"].get("review_manifest_id", "")),
        "admission_request_id": str(payloads["training_dataset_materialization_plan"].get("admission_request_id", "")),
        "review_queue_id": str(payloads["training_dataset_materialization_plan"].get("review_queue_id", "")),
        "property_candidate_manifest_id": str(
            payloads["training_dataset_materialization_plan"].get("property_candidate_manifest_id", "")
        ),
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
            "# Custom Corpus Property Training Dataset Materialization Dry-Run Evidence",
            "",
            f"- Dry-run status: `{summary['dry_run_status']}`",
            f"- Materialization dry-run id: `{summary['materialization_dry_run_id']}`",
            f"- Dataset name: `{summary['dataset_name']}`",
            f"- Row contract id: `{summary['row_contract_id']}`",
            f"- Row preview count: `{summary['row_preview_count']}`",
            f"- Dry-run errors: `{json.dumps(summary['dry_run_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training dataset materialization dry-run only.",
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


def _schema_status_label(status: str) -> str:
    return status


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise PropertyTrainingDatasetMaterializationDryRunError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetMaterializationDryRunError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise PropertyTrainingDatasetMaterializationDryRunError(f"{label} contains forbidden material")
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


def _reference_is_safe(record: dict[str, Any]) -> bool:
    for key, value in record.items():
        if key.endswith("_sha256"):
            if value and not _SHA_RE.match(str(value)):
                return False
            continue
        if isinstance(value, list):
            if key not in {"target_model_families", "planned_output_formats"}:
                return False
            if not all(isinstance(item, str) and _is_safe_id(item) for item in value):
                return False
            continue
        if isinstance(value, str) and value and not _is_safe_id(value):
            return False
    return True


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


def _input_contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    input_markers = (
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
    if any(marker.lower() in lowered for marker in input_markers):
        return True
    return False


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "dry_run_status": "blocked",
        "dry_run_errors": ["property_training_dataset_materialization_dry_run_redaction_failed"],
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
