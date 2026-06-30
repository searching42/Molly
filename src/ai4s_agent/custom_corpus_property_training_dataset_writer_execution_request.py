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


_REQUEST_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_execution_request.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_writer_execution_request_builder.v1"
_DRY_RUN_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_materialization_dry_run_precheck.v1"
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
_WRITER_MODES = {"jsonl_first", "format_label_only"}
_REQUEST_RECORD_KEYS = {
    "writer_request_record_id",
    "row_preview_id",
    "contract_record_reference_id",
    "planned_dataset_record_id",
    "ledger_record_id",
    "candidate_record_id",
    "record_id",
    "document_id",
    "field_name",
    "dataset_name",
    "row_contract_id",
    "materialization_plan_id",
    "materialization_dry_run_id",
    "requested_action",
    "writer_request_record_status",
    "writer_executed",
    "training_admitted",
    "training_dataset_materialized",
    "dataset_artifact_created",
    "phase1_status",
    "dataset_confirmation_changed",
    "requested_output_formats",
    "target_model_families",
    "source_artifact_sha256",
    "review_artifact_sha256",
    "admission_request_sha256",
    "package_validation_sha256",
    "materialization_plan_sha256",
    "quarantine_candidate_records_sha256",
    "training_admission_readiness_summary_sha256",
    "training_admission_request_plan_sha256",
    "training_admission_request_preflight_sha256",
    "training_admission_request_draft_sha256",
    "training_admission_request_draft_precheck_sha256",
    "training_admission_execution_request_sha256",
    "training_admission_execution_request_preflight_sha256",
    "training_admission_execution_dry_run_report_sha256",
    "training_admission_execution_ledger_sha256",
    "training_dataset_materialization_plan_sha256",
    "training_dataset_row_contract_sha256",
    "training_dataset_materialization_dry_run_report_sha256",
    "training_dataset_materialization_dry_run_precheck_sha256",
}
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


class PropertyTrainingDatasetWriterExecutionRequestError(ValueError):
    pass


def build_property_training_dataset_writer_execution_request(
    *,
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
    writer_execution_request_id: str,
    created_by: str,
    confirm_training_dataset_writer_execution_request: bool = False,
    allow_dry_run_precheck_needs_review: bool = False,
    minimum_request_records: int = 1,
    requested_output_formats: list[str] | None = None,
    requested_writer_mode: str = "format_label_only",
) -> dict[str, Any]:
    payloads = _load_payloads(
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
    run_dir = Path(output_dir).expanduser() / writer_execution_request_id
    warnings: list[str] = []
    resolved_output_formats = _resolve_requested_output_formats(
        payloads,
        requested_output_formats=requested_output_formats,
    )
    errors = _validation_errors(
        payloads,
        hashes,
        run_dir=run_dir,
        writer_execution_request_id=writer_execution_request_id,
        created_by=created_by,
        confirm=confirm_training_dataset_writer_execution_request,
        allow_needs_review=allow_dry_run_precheck_needs_review,
        minimum_request_records=minimum_request_records,
        requested_output_formats=resolved_output_formats,
        requested_writer_mode=requested_writer_mode,
        warnings=warnings,
    )
    status = "blocked" if errors else "needs_review" if warnings else "written"
    request_path = run_dir / "property_training_dataset_writer_execution_request.json"
    summary_path = run_dir / "property_training_dataset_writer_execution_request_summary.json"
    evidence_path = run_dir / "redacted_property_training_dataset_writer_execution_request_evidence.md"
    if status == "blocked":
        return _summary(
            status=status,
            writer_execution_request_id=writer_execution_request_id,
            payloads=payloads,
            paths=paths,
            hashes=hashes,
            request_path=request_path,
            request_sha256="",
            requested_output_formats=resolved_output_formats,
            requested_writer_mode=requested_writer_mode,
            writer_request_records=[],
            errors=_stable_unique(errors),
            warnings=_stable_unique(warnings),
        )
    writer_request_records = _writer_request_records(
        payloads,
        hashes,
        writer_execution_request_id=writer_execution_request_id,
        requested_output_formats=resolved_output_formats,
    )
    request = _request(
        status=status,
        writer_execution_request_id=writer_execution_request_id,
        created_by=created_by,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        requested_output_formats=resolved_output_formats,
        requested_writer_mode=requested_writer_mode,
        writer_request_records=writer_request_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    request_sha256 = _sha256_payload(request)
    summary = _summary(
        status=status,
        writer_execution_request_id=writer_execution_request_id,
        payloads=payloads,
        paths=paths,
        hashes=hashes,
        request_path=request_path,
        request_sha256=request_sha256,
        requested_output_formats=resolved_output_formats,
        requested_writer_mode=requested_writer_mode,
        writer_request_records=writer_request_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    markdown = _markdown(summary)
    if _contains_forbidden_material({"request": request, "summary": summary, "markdown": markdown}):
        return _minimal_redaction_failure()
    write_json(request_path, request)
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
        summary = build_property_training_dataset_writer_execution_request(
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
            writer_execution_request_id=args.writer_execution_request_id,
            created_by=args.created_by,
            confirm_training_dataset_writer_execution_request=args.confirm_training_dataset_writer_execution_request,
            allow_dry_run_precheck_needs_review=args.allow_dry_run_precheck_needs_review,
            minimum_request_records=args.minimum_request_records,
            requested_output_formats=args.requested_output_format,
            requested_writer_mode=args.requested_writer_mode,
        )
    except Exception as exc:
        err.write(f"property training dataset writer execution request invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("request_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_dataset_writer_execution_request",
        description="Build a property training dataset writer execution request without executing a dataset writer.",
    )
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
    parser.add_argument("--writer-execution-request-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-training-dataset-writer-execution-request", action="store_true")
    parser.add_argument("--allow-dry-run-precheck-needs-review", action="store_true")
    parser.add_argument("--minimum-request-records", type=int, default=1)
    parser.add_argument("--requested-output-format", action="append", choices=sorted(_OUTPUT_FORMAT_LABELS))
    parser.add_argument("--requested-writer-mode", choices=sorted(_WRITER_MODES), default="format_label_only")
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
    writer_execution_request_id: str,
    created_by: str,
    confirm: bool,
    allow_needs_review: bool,
    minimum_request_records: int,
    requested_output_formats: list[str],
    requested_writer_mode: str,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if not confirm:
        errors.append("confirmation_required")
    if not _is_safe_id(writer_execution_request_id):
        errors.append("writer_execution_request_id_invalid")
    if not created_by or "@" in created_by:
        errors.append("created_by_invalid")
    if run_dir.exists() and any(run_dir.iterdir()):
        errors.append("output_directory_not_clean")
    if requested_writer_mode not in _WRITER_MODES:
        errors.append("requested_writer_mode_invalid")
    if not requested_output_formats or any(label not in _OUTPUT_FORMAT_LABELS for label in requested_output_formats):
        errors.append("requested_output_format_invalid")
    planned_formats = _planned_output_formats(payloads)
    if any(label not in planned_formats for label in requested_output_formats):
        errors.append("requested_output_format_not_planned")
    _append_errors(errors, _schema_errors(payloads))
    _append_errors(errors, _status_errors(payloads, allow_needs_review, warnings))
    _append_errors(errors, _hash_errors(payloads, hashes))
    _append_errors(errors, _id_errors(payloads))
    _append_errors(errors, _record_errors(payloads, minimum_request_records))
    _append_errors(errors, _sha_format_errors(payloads))
    return _stable_unique(errors)


def _schema_errors(payloads: dict[str, Any]) -> list[str]:
    expected = {
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


def _status_errors(payloads: dict[str, Any], allow: bool, warnings: list[str]) -> list[str]:
    errors: list[str] = []
    status_checks = (
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
    report = payloads["training_dataset_materialization_dry_run_report"]
    if report.get("dry_run_mode") != "training_dataset_materialization_dry_run_only":
        errors.append("training_dataset_materialization_dry_run_mode_invalid")
    if payloads["training_dataset_row_contract"].get("contract_mode") != "training_dataset_row_contract_only":
        errors.append("training_dataset_row_contract_mode_invalid")
    if payloads["training_dataset_materialization_plan"].get("plan_mode") != "training_dataset_materialization_plan_only":
        errors.append("training_dataset_materialization_plan_mode_invalid")
    if payloads["training_admission_execution_request"].get("request_mode") != "execution_request_only":
        errors.append("training_admission_execution_request_mode_invalid")
    if payloads["training_admission_request_draft"].get("request_mode") != "draft_only":
        errors.append("training_admission_request_draft_mode_invalid")
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
    for key in _payload_keys(payloads):
        container = payloads[key]
        for hash_key, actual_hash in hashes.items():
            if container.get(hash_key) and container.get(hash_key) != actual_hash:
                errors.append(f"{hash_key}_mismatch")
    return _stable_unique(errors)


def _id_errors(payloads: dict[str, Any]) -> list[str]:
    groups = {
        "corpus_id": (_payload_keys(payloads), ("corpus_id",)),
        "materialization_dry_run_id": (
            (
                "training_dataset_materialization_dry_run_precheck",
                "training_dataset_materialization_dry_run_report",
                "training_dataset_materialization_dry_run_summary",
            ),
            ("materialization_dry_run_id",),
        ),
        "materialization_plan_id": (
            (
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


def _record_errors(payloads: dict[str, Any], minimum_request_records: int) -> list[str]:
    errors: list[str] = []
    report = payloads["training_dataset_materialization_dry_run_report"]
    summary = payloads["training_dataset_materialization_dry_run_summary"]
    plan = payloads["training_dataset_materialization_plan"]
    ledger = payloads["training_admission_execution_ledger"]
    contract = payloads["training_dataset_row_contract"]
    request_plan = payloads["training_admission_request_plan"]
    request_preflight = payloads["training_admission_request_preflight"]
    readiness = payloads["training_admission_readiness_summary"]
    previews = _safe_records(report.get("row_previews"))
    references = _safe_records(contract.get("contract_record_references"))
    plan_records = _safe_records(plan.get("planned_dataset_records"))
    ledger_records = _safe_records(ledger.get("ledger_records"))
    if len(previews) < max(minimum_request_records, 1):
        errors.append("minimum_request_record_count_not_met")
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
    planned_candidates = _safe_list(report.get("planned_training_admission_candidate_record_ids"))
    if not planned_candidates:
        errors.append("no_planned_candidates")
    if set(_safe_list(summary.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates):
        errors.append("planned_candidate_ids_mismatch")
    if set(_safe_list(plan.get("planned_training_admission_candidate_record_ids"))) != set(planned_candidates):
        errors.append("planned_candidate_ids_mismatch")
    plan_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in plan_records]
    reference_dataset_ids = [str(record.get("planned_dataset_record_id", "")) for record in references]
    preview_dataset_ids = [str(preview.get("planned_dataset_record_id", "")) for preview in previews]
    if plan_dataset_ids != reference_dataset_ids or plan_dataset_ids != preview_dataset_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    ledger_ids = [str(record.get("ledger_record_id", "")) for record in ledger_records]
    preview_ledger_ids = [str(preview.get("ledger_record_id", "")) for preview in previews]
    if set(preview_ledger_ids) != set(ledger_ids):
        errors.append("ledger_record_ids_mismatch")
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


def _resolve_requested_output_formats(
    payloads: dict[str, Any],
    *,
    requested_output_formats: list[str] | None,
) -> list[str]:
    if requested_output_formats:
        return _stable_unique([str(label) for label in requested_output_formats])
    planned = _planned_output_formats(payloads)
    return sorted(planned) if planned else sorted(_OUTPUT_FORMAT_LABELS)


def _planned_output_formats(payloads: dict[str, Any]) -> set[str]:
    formats: set[str] = set()
    for preview in _safe_records(payloads["training_dataset_materialization_dry_run_report"].get("row_previews")):
        formats.update(label for label in _safe_list(preview.get("planned_output_formats")) if label in _OUTPUT_FORMAT_LABELS)
    output_counts = payloads["training_dataset_materialization_dry_run_report"].get(
        "output_format_compatibility_summary",
        {},
    ).get("counts_by_output_format", {})
    if isinstance(output_counts, dict):
        formats.update(label for label in output_counts if label in _OUTPUT_FORMAT_LABELS)
    return formats


def _writer_request_records(
    payloads: dict[str, Any],
    hashes: dict[str, str],
    *,
    writer_execution_request_id: str,
    requested_output_formats: list[str],
) -> list[dict[str, Any]]:
    report = payloads["training_dataset_materialization_dry_run_report"]
    plan = payloads["training_dataset_materialization_plan"]
    records: list[dict[str, Any]] = []
    for index, preview in enumerate(_safe_records(report.get("row_previews")), start=1):
        records.append(
            {
                "writer_request_record_id": f"{writer_execution_request_id}-record-{index:03d}",
                "row_preview_id": str(preview.get("row_preview_id", "")),
                "contract_record_reference_id": str(preview.get("contract_record_reference_id", "")),
                "planned_dataset_record_id": str(preview.get("planned_dataset_record_id", "")),
                "ledger_record_id": str(preview.get("ledger_record_id", "")),
                "candidate_record_id": str(preview.get("candidate_record_id", "")),
                "record_id": str(preview.get("record_id", "")),
                "document_id": str(preview.get("document_id", "")),
                "field_name": str(preview.get("field_name", "")),
                "dataset_name": str(report.get("dataset_name", "")),
                "row_contract_id": str(report.get("row_contract_id", "")),
                "materialization_plan_id": str(plan.get("materialization_plan_id", "")),
                "materialization_dry_run_id": str(report.get("materialization_dry_run_id", "")),
                "requested_action": "write_training_dataset_row_later",
                "writer_request_record_status": "requested",
                "writer_executed": False,
                "training_admitted": True,
                "training_dataset_materialized": False,
                "dataset_artifact_created": False,
                "phase1_status": "not_run",
                "dataset_confirmation_changed": False,
                "requested_output_formats": requested_output_formats,
                "target_model_families": _safe_list(preview.get("target_model_families")),
                "source_artifact_sha256": str(preview.get("source_artifact_sha256", "")),
                "review_artifact_sha256": str(preview.get("review_artifact_sha256", "")),
                "admission_request_sha256": str(preview.get("admission_request_sha256", "")),
                "package_validation_sha256": str(plan.get("formal_package_validation_sha256", "")),
                "materialization_plan_sha256": str(preview.get("training_dataset_materialization_plan_sha256", "")),
                "quarantine_candidate_records_sha256": hashes["quarantine_candidate_records_sha256"],
                "training_admission_readiness_summary_sha256": hashes["training_admission_readiness_summary_sha256"],
                "training_admission_request_plan_sha256": hashes["training_admission_request_plan_sha256"],
                "training_admission_request_preflight_sha256": hashes["training_admission_request_preflight_sha256"],
                "training_admission_request_draft_sha256": hashes["training_admission_request_draft_sha256"],
                "training_admission_request_draft_precheck_sha256": hashes[
                    "training_admission_request_draft_precheck_sha256"
                ],
                "training_admission_execution_request_sha256": hashes["training_admission_execution_request_sha256"],
                "training_admission_execution_request_preflight_sha256": hashes[
                    "training_admission_execution_request_preflight_sha256"
                ],
                "training_admission_execution_dry_run_report_sha256": hashes[
                    "training_admission_execution_dry_run_report_sha256"
                ],
                "training_admission_execution_ledger_sha256": hashes["training_admission_execution_ledger_sha256"],
                "training_dataset_materialization_plan_sha256": hashes["training_dataset_materialization_plan_sha256"],
                "training_dataset_row_contract_sha256": hashes["training_dataset_row_contract_sha256"],
                "training_dataset_materialization_dry_run_report_sha256": hashes[
                    "training_dataset_materialization_dry_run_report_sha256"
                ],
                "training_dataset_materialization_dry_run_precheck_sha256": hashes[
                    "training_dataset_materialization_dry_run_precheck_sha256"
                ],
            }
        )
    return records


def _request(
    *,
    status: str,
    writer_execution_request_id: str,
    created_by: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    requested_output_formats: list[str],
    requested_writer_mode: str,
    writer_request_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    report = payloads["training_dataset_materialization_dry_run_report"]
    plan = payloads["training_dataset_materialization_plan"]
    return {
        "schema_version": _REQUEST_SCHEMA_VERSION,
        "writer_execution_request_id": writer_execution_request_id,
        "created_at": now_iso(),
        "created_by": created_by,
        "request_status": status,
        "request_mode": "training_dataset_writer_execution_request_only",
        "requested_writer_mode": requested_writer_mode,
        "writer_executed": False,
        "training_admitted": True,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        **_source_fields(paths, hashes),
        **_source_status_fields(payloads),
        "dataset_name": str(report.get("dataset_name", "")),
        "row_contract_id": str(report.get("row_contract_id", "")),
        "contract_version_label": str(report.get("contract_version_label", "")),
        "materialization_plan_id": str(plan.get("materialization_plan_id", "")),
        "materialization_dry_run_id": str(report.get("materialization_dry_run_id", "")),
        "requested_output_formats": requested_output_formats,
        "writer_request_record_count": len(writer_request_records),
        "writer_request_record_ids": [record["writer_request_record_id"] for record in writer_request_records],
        "row_preview_ids": [record["row_preview_id"] for record in writer_request_records],
        "planned_dataset_record_ids": [record["planned_dataset_record_id"] for record in writer_request_records],
        "planned_training_admission_candidate_record_ids": [
            record["candidate_record_id"] for record in writer_request_records
        ],
        "writer_request_records": writer_request_records,
        "request_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
        "boundary_statement": [
            "training dataset writer execution request only",
            "no dataset writer execution",
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
    writer_execution_request_id: str,
    payloads: dict[str, Any],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    request_path: Path,
    request_sha256: str,
    requested_output_formats: list[str],
    requested_writer_mode: str,
    writer_request_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    report = payloads["training_dataset_materialization_dry_run_report"]
    plan = payloads["training_dataset_materialization_plan"]
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "request_status": status,
        "writer_execution_request_id": writer_execution_request_id,
        "training_dataset_writer_execution_request_path": request_path.name,
        "training_dataset_writer_execution_request_sha256": request_sha256,
        **_source_fields(paths, hashes),
        **_source_status_fields(payloads),
        "dataset_name": str(report.get("dataset_name", "")),
        "row_contract_id": str(report.get("row_contract_id", "")),
        "contract_version_label": str(report.get("contract_version_label", "")),
        "materialization_plan_id": str(plan.get("materialization_plan_id", "")),
        "materialization_dry_run_id": str(report.get("materialization_dry_run_id", "")),
        "requested_writer_mode": requested_writer_mode,
        "requested_output_formats": requested_output_formats,
        "writer_executed": False,
        "training_admitted": True if status in {"written", "needs_review"} else False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "writer_request_record_count": len(writer_request_records),
        "writer_request_record_ids": [record["writer_request_record_id"] for record in writer_request_records],
        "row_preview_count": len(_safe_records(report.get("row_previews"))),
        "row_preview_ids": [record["row_preview_id"] for record in writer_request_records],
        "planned_dataset_record_count": len(_safe_records(plan.get("planned_dataset_records"))),
        "planned_dataset_record_ids": [record["planned_dataset_record_id"] for record in writer_request_records],
        "planned_training_admission_candidate_record_ids": [
            record["candidate_record_id"] for record in writer_request_records
        ],
        "request_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _source_status_fields(payloads: dict[str, Any]) -> dict[str, str]:
    report = payloads["training_dataset_materialization_dry_run_report"]
    return {
        "materialization_dry_run_precheck_status": str(
            payloads["training_dataset_materialization_dry_run_precheck"].get("precheck_status", "")
        ),
        "materialization_dry_run_status": str(report.get("dry_run_status", "")),
        "row_contract_precheck_status": str(report.get("row_contract_precheck_status", "")),
        "contract_status": str(report.get("contract_status", "")),
        "materialization_plan_precheck_status": str(report.get("materialization_plan_precheck_status", "")),
        "plan_status": str(report.get("plan_status", "")),
        "ledger_status": str(report.get("ledger_status", "")),
        "corpus_id": str(report.get("corpus_id", "")),
        "execution_ledger_id": str(report.get("execution_ledger_id", "")),
        "execution_request_id": str(report.get("execution_request_id", "")),
        "source_execution_request_id": str(report.get("source_execution_request_id", "")),
        "review_manifest_id": str(report.get("review_manifest_id", "")),
        "admission_request_id": str(report.get("admission_request_id", "")),
        "review_queue_id": str(report.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(report.get("property_candidate_manifest_id", "")),
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
            "# Custom Corpus Property Training Dataset Writer Execution Request Evidence",
            "",
            f"- Request status: `{summary['request_status']}`",
            f"- Writer execution request id: `{summary['writer_execution_request_id']}`",
            f"- Dataset name: `{summary['dataset_name']}`",
            f"- Row contract id: `{summary['row_contract_id']}`",
            f"- Writer request record count: `{summary['writer_request_record_count']}`",
            f"- Requested output formats: `{json.dumps(summary['requested_output_formats'])}`",
            f"- Request errors: `{json.dumps(summary['request_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training dataset writer execution request only.",
            "- no dataset writer was executed.",
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
            if not all(isinstance(item, str) and _is_safe_id(item) for item in value):
                return False
            continue
        if isinstance(value, str) and value and not _is_safe_id(value):
            return False
        if _value_contains_unsafe_material(value):
            return False
    return True


def _writer_record_is_safe(record: dict[str, Any]) -> bool:
    if set(record) - _REQUEST_RECORD_KEYS:
        return False
    return _row_preview_is_safe(record)


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise PropertyTrainingDatasetWriterExecutionRequestError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise PropertyTrainingDatasetWriterExecutionRequestError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise PropertyTrainingDatasetWriterExecutionRequestError(f"{label} contains forbidden material")
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
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _value_contains_unsafe_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    value_markers = _FORBIDDEN_MARKERS + ("conformer block", "dpa3 structure")
    if any(marker.lower() in lowered for marker in value_markers):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _input_contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    return any(marker.lower() in lowered for marker in _INPUT_FORBIDDEN_MARKERS)


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "request_status": "blocked",
        "request_errors": ["property_training_dataset_writer_execution_request_redaction_failed"],
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
