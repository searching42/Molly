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


_CONTRACT_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_training_dataset_row_contract_builder.v1"
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
_ALLOWED_OUTPUT_FORMATS = {"jsonl", "parquet", "lmdb", "csv"}
_ALLOWED_TARGET_MODEL_FAMILIES = {"unimol", "dpa3", "generic_property_predictor"}
_FIELD_TYPE_DESCRIPTORS = {
    "string",
    "number",
    "boolean",
    "array[string]",
    "nullable[string]",
    "nullable[number]",
}
_REQUIRED_ROW_FIELDS = [
    "dataset_record_id",
    "candidate_record_id",
    "record_id",
    "document_id",
    "field_name",
    "property_name",
    "property_value",
    "property_unit",
    "property_value_normalized",
    "property_unit_normalized",
    "task_type",
    "compound_id",
    "canonical_smiles",
    "source_artifact_sha256",
    "review_artifact_sha256",
    "admission_request_sha256",
    "training_admission_execution_ledger_sha256",
    "training_dataset_materialization_plan_sha256",
]
_OPTIONAL_ROW_FIELDS = [
    "inchi",
    "inchi_key",
    "molecular_formula",
    "molecular_weight",
    "temperature",
    "solvent",
    "method",
    "aggregation_state",
    "device_context",
    "paper_id",
    "doi",
    "property_uncertainty",
    "quality_flags",
    "split_group_key",
    "dedup_key",
    "model_family_compatibility",
]
_QUALITY_FLAGS = [
    "unit_normalized",
    "value_normalized",
    "source_reviewed",
    "human_review_bound",
    "ledger_admitted",
    "needs_unit_review",
    "needs_structure_review",
    "needs_property_review",
]
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


class CustomCorpusPropertyTrainingDatasetRowContractError(ValueError):
    pass


def build_property_training_dataset_row_contract(
    *,
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
    row_contract_id: str,
    created_by: str,
    confirm_training_dataset_row_contract: bool = False,
    allow_materialization_plan_precheck_needs_review: bool = False,
    minimum_contract_records: int = 1,
    dataset_name: str | None = None,
    contract_version_label: str = "v1",
    target_model_families: list[str] | None = None,
    planned_output_formats: list[str] | None = None,
) -> dict[str, Any]:
    plan_precheck = _read_safe_json_dict(
        training_dataset_materialization_plan_precheck_path,
        "training dataset materialization plan precheck",
    )
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
    run_dir = Path(output_dir).expanduser() / row_contract_id
    paths = {
        "training_dataset_materialization_plan_precheck_path": training_dataset_materialization_plan_precheck_path,
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
    effective_dataset_name = dataset_name or str(plan.get("dataset_name", ""))
    effective_model_families = _stable_unique(
        target_model_families or _safe_list(plan.get("target_model_families"))
    )
    effective_output_formats = _stable_unique(
        planned_output_formats or _safe_list(plan.get("planned_output_formats"))
    )
    warnings: list[str] = []
    errors = _consistency_errors(
        plan_precheck=plan_precheck,
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
        run_dir=run_dir,
        row_contract_id=row_contract_id,
        created_by=created_by,
        confirm_training_dataset_row_contract=confirm_training_dataset_row_contract,
        allow_materialization_plan_precheck_needs_review=allow_materialization_plan_precheck_needs_review,
        minimum_contract_records=max(int(minimum_contract_records), 0),
        dataset_name=effective_dataset_name,
        contract_version_label=contract_version_label,
        target_model_families=effective_model_families,
        planned_output_formats=effective_output_formats,
        warnings=warnings,
    )
    contract_status = "blocked" if errors else "needs_review" if warnings else "written"
    contract_path = run_dir / "property_training_dataset_row_contract.json"
    summary_path = run_dir / "property_training_dataset_row_contract_summary.json"
    evidence_path = run_dir / "redacted_property_training_dataset_row_contract_evidence.md"
    if contract_status == "blocked":
        return _summary(
            contract_status="blocked",
            row_contract_id=row_contract_id,
            paths=paths,
            hashes=hashes,
            contract_path=contract_path,
            contract_sha256="",
            plan_precheck=plan_precheck,
            plan=plan,
            ledger_precheck=ledger_precheck,
            ledger=ledger,
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
            dataset_name=effective_dataset_name,
            contract_version_label=contract_version_label,
            contract_references=[],
            errors=_stable_unique(errors),
            warnings=_stable_unique(warnings),
        )
    contract_references = _contract_record_references(
        plan,
        hashes,
        row_contract_id,
        target_model_families=effective_model_families,
        planned_output_formats=effective_output_formats,
    )
    contract = _contract(
        contract_status=contract_status,
        row_contract_id=row_contract_id,
        created_by=created_by,
        dataset_name=effective_dataset_name,
        contract_version_label=contract_version_label,
        paths=paths,
        hashes=hashes,
        plan_precheck=plan_precheck,
        plan=plan,
        ledger_precheck=ledger_precheck,
        ledger=ledger,
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
        contract_references=contract_references,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    contract_sha256 = _sha256_payload(contract)
    summary = _summary(
        contract_status=contract_status,
        row_contract_id=row_contract_id,
        paths=paths,
        hashes=hashes,
        contract_path=contract_path,
        contract_sha256=contract_sha256,
        plan_precheck=plan_precheck,
        plan=plan,
        ledger_precheck=ledger_precheck,
        ledger=ledger,
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
        dataset_name=effective_dataset_name,
        contract_version_label=contract_version_label,
        contract_references=contract_references,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    markdown = _evidence_markdown(summary)
    if _contains_forbidden_material({"contract": contract, "summary": summary, "markdown": markdown}):
        return _minimal_redaction_failure()
    write_json(contract_path, contract)
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
        summary = build_property_training_dataset_row_contract(
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
            row_contract_id=args.row_contract_id,
            created_by=args.created_by,
            confirm_training_dataset_row_contract=args.confirm_training_dataset_row_contract,
            allow_materialization_plan_precheck_needs_review=args.allow_materialization_plan_precheck_needs_review,
            minimum_contract_records=args.minimum_contract_records,
            dataset_name=args.dataset_name,
            contract_version_label=args.contract_version_label,
        )
    except Exception as exc:
        err.write(f"property training dataset row contract invalid: {_safe_exception_message(exc)}\n")
        return 1
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("contract_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_dataset_row_contract",
        description="Build a property training dataset row contract without writing dataset artifacts.",
    )
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
    parser.add_argument("--row-contract-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-training-dataset-row-contract", action="store_true")
    parser.add_argument("--allow-materialization-plan-precheck-needs-review", action="store_true")
    parser.add_argument("--minimum-contract-records", type=int, default=1)
    parser.add_argument("--dataset-name")
    parser.add_argument("--contract-version-label", default="v1")
    return parser


def _consistency_errors(
    *,
    plan_precheck: dict[str, Any],
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
    run_dir: Path,
    row_contract_id: str,
    created_by: str,
    confirm_training_dataset_row_contract: bool,
    allow_materialization_plan_precheck_needs_review: bool,
    minimum_contract_records: int,
    dataset_name: str,
    contract_version_label: str,
    target_model_families: list[str],
    planned_output_formats: list[str],
    warnings: list[str],
) -> list[str]:
    containers = (
        plan_precheck,
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
    if not confirm_training_dataset_row_contract:
        errors.append("confirmation_required")
    if not _is_safe_id(row_contract_id):
        errors.append("row_contract_id_invalid")
    if not created_by or "@" in created_by:
        errors.append("created_by_invalid")
    if not contract_version_label or not _is_safe_id(contract_version_label):
        errors.append("contract_version_label_invalid")
    if not dataset_name or not _is_safe_id(dataset_name):
        errors.append("dataset_name_invalid")
    if run_dir.exists() and any(run_dir.iterdir()):
        errors.append("output_directory_not_clean")
    if not target_model_families or any(label not in _ALLOWED_TARGET_MODEL_FAMILIES for label in target_model_families):
        errors.append("target_model_family_invalid")
    if not planned_output_formats or any(label not in _ALLOWED_OUTPUT_FORMATS for label in planned_output_formats):
        errors.append("planned_output_format_invalid")
    _append_errors(errors, _schema_errors(*containers))
    _append_errors(
        errors,
        _status_errors(
            plan_precheck,
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
            allow_materialization_plan_precheck_needs_review,
            warnings,
        ),
    )
    _append_errors(errors, _hash_errors(*containers, hashes=hashes))
    _append_errors(errors, _id_errors(*containers))
    _append_errors(
        errors,
        _record_errors(
            plan_precheck,
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
            minimum_contract_records,
        ),
    )
    _append_errors(errors, _sha_format_errors(*containers))
    return _stable_unique(errors)


def _schema_errors(*containers: dict[str, Any]) -> list[str]:
    expected = (
        ("training_dataset_materialization_plan_precheck_schema_invalid", _PLAN_PRECHECK_SCHEMA_VERSION),
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
    plan_precheck: dict[str, Any],
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
    allow: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    _append_errors(errors, _status_field_errors(plan_precheck, "precheck_status", "training_dataset_materialization_plan_precheck", "passed", allow, warnings))
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
    if plan_precheck.get("precheck_errors"):
        errors.append("training_dataset_materialization_plan_precheck_has_errors")
    if plan.get("plan_mode") != "training_dataset_materialization_plan_only":
        errors.append("training_dataset_materialization_plan_mode_invalid")
    if plan.get("planning_errors") or planner_summary.get("planning_errors"):
        errors.append("training_dataset_materialization_plan_has_errors")
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
    if any(container.get("training_admitted") is not True for container in (plan_precheck, plan, planner_summary, ledger_precheck, ledger, ledger_summary)):
        errors.append("training_not_admitted")
    if any(container.get("training_dataset_materialized") is not False for container in (plan_precheck, plan, planner_summary, ledger_precheck, ledger, ledger_summary)):
        errors.append("training_dataset_materialized")
    if any(container.get("dataset_artifact_created") is not False for container in (plan_precheck, plan, planner_summary, ledger_precheck, ledger, ledger_summary)):
        errors.append("dataset_artifact_created")
    boundary_containers = (
        plan_precheck,
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
        plan_precheck,
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
    direct_pairs = (
        (plan_precheck, "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256"),
        (plan_precheck, "training_dataset_materialization_planner_summary_sha256", "training_dataset_materialization_planner_summary_sha256"),
        (plan_precheck, "training_admission_execution_ledger_precheck_sha256", "training_admission_execution_ledger_precheck_sha256"),
        (plan_precheck, "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256"),
        (plan_precheck, "training_admission_execution_ledger_summary_sha256", "training_admission_execution_ledger_summary_sha256"),
        (plan_precheck, "training_admission_execution_dry_run_precheck_sha256", "training_admission_execution_dry_run_precheck_sha256"),
        (plan_precheck, "training_admission_execution_dry_run_report_sha256", "training_admission_execution_dry_run_report_sha256"),
        (plan_precheck, "training_admission_execution_request_sha256", "training_admission_execution_request_sha256"),
        (plan_precheck, "training_admission_execution_request_summary_sha256", "training_admission_execution_request_summary_sha256"),
        (plan_precheck, "training_admission_execution_request_preflight_sha256", "training_admission_execution_request_preflight_sha256"),
        (plan_precheck, "training_admission_request_draft_sha256", "training_admission_request_draft_sha256"),
        (plan_precheck, "training_admission_request_draft_summary_sha256", "training_admission_request_draft_summary_sha256"),
        (plan_precheck, "training_admission_request_draft_precheck_sha256", "training_admission_request_draft_precheck_sha256"),
        (plan_precheck, "training_admission_request_plan_sha256", "training_admission_request_plan_sha256"),
        (plan_precheck, "training_admission_request_preflight_sha256", "training_admission_request_preflight_sha256"),
        (plan_precheck, "training_admission_readiness_summary_sha256", "training_admission_readiness_summary_sha256"),
        (plan_precheck, "quarantine_candidate_preflight_summary_sha256", "quarantine_candidate_preflight_summary_sha256"),
        (plan_precheck, "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256"),
        (plan, "training_admission_execution_ledger_precheck_sha256", "training_admission_execution_ledger_precheck_sha256"),
        (plan, "training_admission_execution_ledger_sha256", "training_admission_execution_ledger_sha256"),
        (planner_summary, "training_dataset_materialization_plan_sha256", "training_dataset_materialization_plan_sha256"),
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
    errors: list[str] = []
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
    plan_precheck, plan, planner_summary, *_rest = containers
    plan_id_values = [
        str(container.get("materialization_plan_id", ""))
        for container in (plan_precheck, plan, planner_summary)
        if container.get("materialization_plan_id")
    ]
    if len(set(plan_id_values)) > 1:
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
    for value in plan_id_values:
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
    source_execution_request_values: list[str] = []
    for index, container in enumerate(containers):
        value = str(container.get("source_execution_request_id", ""))
        if not value and index >= 11:
            value = str(container.get("execution_request_id", ""))
        if value:
            source_execution_request_values.append(value)
    if len(set(source_execution_request_values)) > 1:
        errors.append("source_execution_request_id_mismatch")
    return _stable_unique(errors)


def _record_errors(
    plan_precheck: dict[str, Any],
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
    minimum_contract_records: int,
) -> list[str]:
    errors: list[str] = []
    plan_records = _safe_records(plan.get("planned_dataset_records"))
    plan_record_ids = [str(record.get("planned_dataset_record_id", "")) for record in plan_records]
    ledger_records = _safe_records(ledger.get("ledger_records"))
    ledger_record_ids = [str(record.get("ledger_record_id", "")) for record in ledger_records]
    planned_ids = _safe_list(plan.get("planned_training_admission_candidate_record_ids"))
    if not plan_records:
        errors.append("no_planned_dataset_records")
    if len(plan_records) < max(minimum_contract_records, 1):
        errors.append("minimum_contract_record_count_not_met")
    if not planned_ids:
        errors.append("no_planned_candidates")
    _count_errors(errors, plan, "planned_dataset_record_count", plan_records)
    _count_errors(errors, plan_precheck, "planned_dataset_record_count", plan_records)
    _count_errors(errors, planner_summary, "planned_dataset_record_count", plan_records)
    _count_errors(errors, ledger_precheck, "ledger_record_count", ledger_records)
    _count_errors(errors, ledger, "ledger_record_count", ledger_records)
    _count_errors(errors, ledger_summary, "ledger_record_count", ledger_records)
    _count_errors(errors, dry_run, "dry_run_record_count", _safe_records(dry_run.get("dry_run_records")))
    _count_errors(errors, request, "execution_record_count", _safe_records(request.get("execution_records")))
    _count_errors(errors, request_summary, "execution_record_count", _safe_records(request.get("execution_records")))
    _count_errors(errors, execution_preflight, "execution_record_count", _safe_records(request.get("execution_records")))
    _count_errors(errors, draft, "draft_record_count", _safe_records(draft.get("draft_records")))
    if _safe_list(plan.get("planned_dataset_record_ids")) != plan_record_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    if _safe_list(plan_precheck.get("planned_dataset_record_ids")) != plan_record_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    if _safe_list(planner_summary.get("planned_dataset_record_ids")) != plan_record_ids:
        errors.append("planned_dataset_record_ids_mismatch")
    if _safe_list(plan.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(plan_precheck.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(ledger.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(ledger_summary.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    if _safe_list(ledger_precheck.get("ledger_record_ids")) != ledger_record_ids:
        errors.append("ledger_record_ids_mismatch")
    for ids in (
        _safe_list(plan_precheck.get("planned_training_admission_candidate_record_ids")),
        _safe_list(planner_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(ledger.get("planned_training_admission_candidate_record_ids")),
        _safe_list(ledger_summary.get("planned_training_admission_candidate_record_ids")),
        _safe_list(ledger_precheck.get("planned_training_admission_candidate_record_ids")),
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
    quarantine_candidate_ids = _safe_list(quarantine_candidate.get("candidate_record_ids"))
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
    plan_candidate_ids = [str(record.get("candidate_record_id", "")) for record in plan_records]
    if set(plan_candidate_ids) != set(planned_ids):
        errors.append("planned_candidate_ids_mismatch")
    for record in plan_records:
        candidate_id = str(record.get("candidate_record_id", ""))
        record_id = str(record.get("record_id", ""))
        ledger_record_id = str(record.get("ledger_record_id", ""))
        if ledger_record_id not in ledger_id_set:
            errors.append("planned_dataset_record_ledger_id_unknown")
        if candidate_id in excluded or record_id in excluded:
            errors.append("planned_candidate_from_excluded_record")
        if candidate_id in blocked or record_id in blocked:
            errors.append("planned_candidate_from_blocked_record")
        if candidate_id in needs_review or record_id in needs_review:
            errors.append("planned_candidate_from_needs_review_record")
        if not _record_values_are_safe(
            record,
            allowed_labels={"planned_action", "planned_record_status", "phase1_status"},
            allowed_list_fields={"target_model_families", "planned_output_formats"},
        ):
            errors.append("planned_dataset_record_contains_unsafe_value")
    return _stable_unique(errors)


def _contract_record_references(
    plan: dict[str, Any],
    hashes: dict[str, str],
    row_contract_id: str,
    *,
    target_model_families: list[str],
    planned_output_formats: list[str],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for record in _safe_records(plan.get("planned_dataset_records")):
        planned_id = str(record.get("planned_dataset_record_id", ""))
        references.append(
            {
                "contract_record_reference_id": f"{row_contract_id}-{planned_id}",
                "planned_dataset_record_id": planned_id,
                "ledger_record_id": str(record.get("ledger_record_id", "")),
                "candidate_record_id": str(record.get("candidate_record_id", "")),
                "record_id": str(record.get("record_id", "")),
                "document_id": str(record.get("document_id", "")),
                "field_name": str(record.get("field_name", "")),
                "target_model_families": list(target_model_families),
                "planned_output_formats": list(planned_output_formats),
                "source_artifact_sha256": str(record.get("source_artifact_sha256", "")),
                "review_artifact_sha256": str(record.get("review_artifact_sha256", "")),
                "admission_request_sha256": str(record.get("admission_request_sha256", "")),
                "package_validation_sha256": str(record.get("package_validation_sha256", "")),
                "materialization_plan_sha256": str(record.get("materialization_plan_sha256", "")),
                "quarantine_candidate_records_sha256": str(record.get("quarantine_candidate_records_sha256", "")),
                "training_admission_readiness_sha256": str(record.get("training_admission_readiness_sha256", "")),
                "training_admission_request_plan_sha256": str(record.get("training_admission_request_plan_sha256", "")),
                "training_admission_request_preflight_sha256": str(record.get("training_admission_request_preflight_sha256", "")),
                "training_admission_request_draft_sha256": str(record.get("training_admission_request_draft_sha256", "")),
                "training_admission_request_draft_precheck_sha256": str(
                    record.get("training_admission_request_draft_precheck_sha256", "")
                ),
                "training_admission_execution_request_sha256": str(
                    record.get("training_admission_execution_request_sha256", "")
                ),
                "training_admission_execution_request_preflight_sha256": str(
                    record.get("training_admission_execution_request_preflight_sha256", "")
                ),
                "training_admission_execution_dry_run_sha256": str(
                    record.get("training_admission_execution_dry_run_sha256", "")
                ),
                "training_admission_execution_dry_run_precheck_sha256": str(
                    record.get("training_admission_execution_dry_run_precheck_sha256", "")
                ),
                "training_admission_execution_ledger_sha256": hashes["training_admission_execution_ledger_sha256"],
                "training_admission_execution_ledger_precheck_sha256": hashes[
                    "training_admission_execution_ledger_precheck_sha256"
                ],
                "training_dataset_materialization_plan_sha256": hashes["training_dataset_materialization_plan_sha256"],
                "training_dataset_materialization_plan_precheck_sha256": hashes[
                    "training_dataset_materialization_plan_precheck_sha256"
                ],
            }
        )
    return references


def _contract(
    *,
    contract_status: str,
    row_contract_id: str,
    created_by: str,
    dataset_name: str,
    contract_version_label: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    plan_precheck: dict[str, Any],
    plan: dict[str, Any],
    ledger_precheck: dict[str, Any],
    ledger: dict[str, Any],
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
    contract_references: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _CONTRACT_SCHEMA_VERSION,
        "row_contract_id": row_contract_id,
        "created_at": now_iso(),
        "created_by": created_by,
        "contract_status": contract_status,
        "contract_mode": "training_dataset_row_contract_only",
        "dataset_name": dataset_name,
        "contract_version_label": contract_version_label,
        "training_admitted": True,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        **_source_fields(paths, hashes),
        **_id_status_fields(plan_precheck, plan, ledger_precheck, ledger, dry_run_precheck, dry_run, request, request_summary, execution_preflight, draft, draft_precheck, request_plan, request_preflight, readiness),
        "planned_dataset_record_count": len(contract_references),
        "planned_dataset_record_ids": [record["planned_dataset_record_id"] for record in contract_references],
        "planned_training_admission_candidate_record_ids": _safe_list(
            plan.get("planned_training_admission_candidate_record_ids")
        ),
        "required_row_fields": list(_REQUIRED_ROW_FIELDS),
        "optional_row_fields": list(_OPTIONAL_ROW_FIELDS),
        "field_type_contract": _field_type_contract(),
        "provenance_field_contract": _provenance_field_contract(),
        "quality_flag_contract": {"allowed_quality_flags": list(_QUALITY_FLAGS)},
        "split_dedup_contract": _split_dedup_contract(),
        "model_family_compatibility_contract": _model_family_compatibility_contract(),
        "output_format_compatibility_contract": _output_format_compatibility_contract(),
        "contract_record_reference_count": len(contract_references),
        "contract_record_references": contract_references,
        "contract_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
        "boundary_statement": [
            "training dataset row contract only",
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
    contract_status: str,
    row_contract_id: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    contract_path: Path,
    contract_sha256: str,
    plan_precheck: dict[str, Any],
    plan: dict[str, Any],
    ledger_precheck: dict[str, Any],
    ledger: dict[str, Any],
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
    dataset_name: str,
    contract_version_label: str,
    contract_references: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "contract_status": contract_status,
        "row_contract_id": row_contract_id,
        "training_dataset_row_contract_path": contract_path.name,
        "training_dataset_row_contract_sha256": contract_sha256,
        **_source_fields(paths, hashes),
        **_id_status_fields(plan_precheck, plan, ledger_precheck, ledger, dry_run_precheck, dry_run, request, request_summary, execution_preflight, draft, draft_precheck, request_plan, request_preflight, readiness),
        "dataset_name": dataset_name,
        "contract_version_label": contract_version_label,
        "training_admitted": True if contract_status in {"written", "needs_review"} else False,
        "training_dataset_materialized": False,
        "dataset_artifact_created": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "required_row_fields": list(_REQUIRED_ROW_FIELDS),
        "optional_row_fields": list(_OPTIONAL_ROW_FIELDS),
        "planned_dataset_record_count": len(_safe_records(plan.get("planned_dataset_records"))),
        "contract_record_reference_count": len(contract_references),
        "planned_dataset_record_ids": [record["planned_dataset_record_id"] for record in contract_references],
        "planned_training_admission_candidate_record_ids": _safe_list(
            plan.get("planned_training_admission_candidate_record_ids")
        ),
        "contract_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _source_fields(paths: dict[str, str | Path], hashes: dict[str, str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, path in paths.items():
        fields[key] = Path(path).name or key
        fields[key.replace("_path", "_sha256")] = hashes[key.replace("_path", "_sha256")]
    return fields


def _id_status_fields(
    plan_precheck: dict[str, Any],
    plan: dict[str, Any],
    ledger_precheck: dict[str, Any],
    ledger: dict[str, Any],
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
) -> dict[str, str]:
    return {
        "materialization_plan_id": str(plan.get("materialization_plan_id", plan_precheck.get("materialization_plan_id", ""))),
        "execution_ledger_id": str(ledger.get("execution_ledger_id", plan_precheck.get("execution_ledger_id", ""))),
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
        "materialization_plan_precheck_status": str(plan_precheck.get("precheck_status", "")),
        "plan_status": str(plan.get("plan_status", "")),
        "ledger_precheck_status": str(ledger_precheck.get("precheck_status", "")),
        "ledger_status": str(ledger.get("execution_status", "")),
        "dry_run_precheck_status": str(dry_run_precheck.get("preflight_status", "")),
        "dry_run_status": str(dry_run.get("dry_run_status", "")),
        "execution_request_status": str(request.get("request_status", request_summary.get("request_status", ""))),
        "execution_request_preflight_status": str(execution_preflight.get("preflight_status", "")),
        "draft_precheck_status": str(draft_precheck.get("precheck_status", "")),
        "draft_status": str(draft.get("draft_status", "")),
        "request_plan_status": str(request_plan.get("planner_status", "")),
        "request_preflight_status": str(request_preflight.get("preflight_status", "")),
        "readiness_status": str(readiness.get("readiness_status", "")),
    }


def _field_type_contract() -> dict[str, str]:
    contract = {field: "string" for field in _REQUIRED_ROW_FIELDS + _OPTIONAL_ROW_FIELDS}
    contract.update(
        {
            "property_value": "number",
            "property_value_normalized": "number",
            "molecular_weight": "nullable[number]",
            "temperature": "nullable[number]",
            "property_uncertainty": "nullable[number]",
            "quality_flags": "array[string]",
            "model_family_compatibility": "array[string]",
            "inchi": "nullable[string]",
            "inchi_key": "nullable[string]",
            "molecular_formula": "nullable[string]",
            "solvent": "nullable[string]",
            "method": "nullable[string]",
            "aggregation_state": "nullable[string]",
            "device_context": "nullable[string]",
            "paper_id": "nullable[string]",
            "doi": "nullable[string]",
            "split_group_key": "nullable[string]",
            "dedup_key": "nullable[string]",
        }
    )
    assert set(contract.values()).issubset(_FIELD_TYPE_DESCRIPTORS)
    return contract


def _provenance_field_contract() -> dict[str, list[str]]:
    return {
        "required_id_fields": [
            "ledger_record_id",
            "planned_dataset_record_id",
            "review_id",
            "admission_record_id",
        ],
        "required_sha_fields": [
            "source_artifact_sha256",
            "review_artifact_sha256",
            "admission_request_sha256",
            "materialization_plan_sha256",
            "training_admission_execution_ledger_sha256",
            "training_dataset_materialization_plan_sha256",
            "row_contract_sha256",
        ],
    }


def _split_dedup_contract() -> dict[str, str | list[str]]:
    return {
        "required_keys": ["dedup_key", "split_group_key"],
        "dedup_key_rule": (
            "dedup_key must derive from canonical molecule identity plus property name, "
            "normalized value, normalized unit, and source artifact hash"
        ),
        "split_group_key_rule": (
            "split_group_key must default to canonical molecule identity, not row id, "
            "to reduce molecule-level leakage"
        ),
    }


def _model_family_compatibility_contract() -> dict[str, str]:
    return {
        "generic_property_predictor": "requires canonical SMILES and scalar property value",
        "unimol": "requires canonical SMILES and later conformer generation or reference; this contract must not generate conformers",
        "dpa3": "requires future structure or geometry-compatible data; this contract must not create DPA3 artifacts",
    }


def _output_format_compatibility_contract() -> dict[str, str]:
    return {
        "jsonl": "future writer label only; this contract creates no jsonl file",
        "parquet": "future writer label only; this contract creates no parquet file",
        "lmdb": "future writer label only; this contract creates no lmdb file",
        "csv": "future writer label only; this contract creates no csv file",
    }


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Dataset Row Contract Evidence",
            "",
            f"- Contract status: `{summary['contract_status']}`",
            f"- Row contract id: `{summary['row_contract_id']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Dataset name: `{summary['dataset_name']}`",
            f"- Contract version label: `{summary['contract_version_label']}`",
            f"- Planned dataset record count: `{summary['planned_dataset_record_count']}`",
            f"- Contract record reference count: `{summary['contract_record_reference_count']}`",
            f"- Required row fields: `{json.dumps(summary['required_row_fields'])}`",
            f"- Optional row fields: `{json.dumps(summary['optional_row_fields'])}`",
            f"- Contract errors: `{json.dumps(summary['contract_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a training dataset row contract only.",
            "- no training dataset artifact was created.",
            "- no training CSV/JSONL/Parquet/LMDB was created.",
            "- no candidate CSV/JSONL/Parquet/LMDB was created.",
            "- no Phase 1 was run.",
            "- DatasetConfirmation was not changed.",
            "- no model training or evaluation was run.",
            "- no conformers or DPA3 structures were generated.",
            "",
        ]
    )


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyTrainingDatasetRowContractError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyTrainingDatasetRowContractError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyTrainingDatasetRowContractError(f"{label} contains forbidden material")
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
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "contract_status": "blocked",
        "contract_errors": ["property_training_dataset_row_contract_redaction_failed"],
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
