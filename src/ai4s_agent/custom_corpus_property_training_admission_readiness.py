from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_admission import AdmissionRequest, load_admission_request
from ai4s_agent.custom_corpus_dry_run import CustomCorpusDryRunReport
from ai4s_agent.custom_corpus_manifest import CustomCorpusManifest, load_custom_corpus_manifest
from ai4s_agent.custom_corpus_materialization import (
    CustomCorpusMaterializationError,
    MaterializationPlan,
    load_materialization_plan,
    sha256_file,
)
from ai4s_agent.custom_corpus_review import ReviewManifest, load_review_manifest


_SCHEMA_VERSION = "custom_corpus_property_training_admission_readiness.v1"
_QUARANTINE_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_quarantine_candidate_preflight.v1"
_QUARANTINE_CANDIDATE_SCHEMA_VERSION = "custom_corpus_property_quarantine_materialization.v1"
_QUARANTINE_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_quarantine_materializer.v1"
_EXECUTION_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_materializer_execution_preflight.v1"
_REQUEST_SCHEMA_VERSION = "custom_corpus_property_materializer_execution_request.v1"
_REQUEST_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_materializer_execution_request_builder.v1"
_MATERIALIZATION_DRY_RUN_SCHEMA_VERSION = "custom_corpus_property_materialization_dry_run.v1"
_PROPERTY_PLANNER_SCHEMA_VERSION = "custom_corpus_property_materialization_planner_runner.v1"
_OFFLINE_PLANNER_SCHEMA_VERSION = "custom_corpus_materialization_planner.v1"
_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_materialization_plan_preflight.v1"
_PROPERTY_BINDING_SCHEMA_VERSION = "custom_corpus_property_package_binding.v1"
_FORMAL_SCHEMA_VERSION = "custom_corpus_admission_package_validation.v1"
_MATERIALIZATION_SCHEMA_VERSION = "custom_corpus_materialization.v1"
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
)


class CustomCorpusPropertyTrainingAdmissionReadinessError(ValueError):
    pass


def plan_property_training_admission_readiness(
    *,
    manifest_path: str | Path,
    dry_run_report_path: str | Path,
    review_manifest_path: str | Path,
    admission_request_path: str | Path,
    formal_package_validation_path: str | Path,
    property_package_binding_summary_path: str | Path,
    materialization_plan_path: str | Path,
    materialization_plan_preflight_summary_path: str | Path,
    offline_planner_output_path: str | Path,
    property_planner_summary_path: str | Path,
    materialization_dry_run_report_path: str | Path,
    execution_request_path: str | Path,
    execution_request_summary_path: str | Path,
    execution_preflight_summary_path: str | Path,
    quarantine_candidate_records_path: str | Path,
    quarantine_materializer_summary_path: str | Path,
    quarantine_candidate_preflight_summary_path: str | Path,
    output_summary_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    require_quarantine_candidate_preflight_passed: bool = False,
    minimum_candidate_records: int = 1,
) -> dict[str, Any]:
    manifest = load_custom_corpus_manifest(manifest_path)
    source_dry_run = _load_source_dry_run_report(dry_run_report_path)
    review = load_review_manifest(review_manifest_path)
    admission = load_admission_request(admission_request_path)
    formal = _read_safe_json_dict(formal_package_validation_path, "formal package validation")
    binding = _read_safe_json_dict(property_package_binding_summary_path, "property package binding summary")
    plan_payload, plan = _load_materialization_plan_payload(materialization_plan_path)
    plan_preflight = _read_safe_json_dict(materialization_plan_preflight_summary_path, "materialization plan preflight summary")
    offline_planner = _read_safe_json_dict(offline_planner_output_path, "offline planner output")
    property_planner = _read_safe_json_dict(property_planner_summary_path, "property planner summary")
    materialization_dry_run = _read_safe_json_dict(materialization_dry_run_report_path, "materialization dry-run report")
    execution_request = _read_safe_json_dict(execution_request_path, "execution request")
    execution_summary = _read_safe_json_dict(execution_request_summary_path, "execution request summary")
    execution_preflight = _read_safe_json_dict(execution_preflight_summary_path, "execution preflight summary")
    quarantine_candidate = _read_safe_json_dict(quarantine_candidate_records_path, "quarantine candidate records")
    quarantine_summary = _read_safe_json_dict(quarantine_materializer_summary_path, "quarantine materializer summary")
    quarantine_preflight = _read_safe_json_dict(quarantine_candidate_preflight_summary_path, "quarantine candidate preflight summary")

    paths = {
        "manifest_path": manifest_path,
        "dry_run_report_path": dry_run_report_path,
        "review_manifest_path": review_manifest_path,
        "admission_request_path": admission_request_path,
        "formal_package_validation_path": formal_package_validation_path,
        "property_package_binding_summary_path": property_package_binding_summary_path,
        "materialization_plan_path": materialization_plan_path,
        "materialization_plan_preflight_summary_path": materialization_plan_preflight_summary_path,
        "offline_planner_output_path": offline_planner_output_path,
        "property_planner_summary_path": property_planner_summary_path,
        "materialization_dry_run_report_path": materialization_dry_run_report_path,
        "execution_request_path": execution_request_path,
        "execution_request_summary_path": execution_request_summary_path,
        "execution_preflight_summary_path": execution_preflight_summary_path,
        "quarantine_candidate_records_path": quarantine_candidate_records_path,
        "quarantine_materializer_summary_path": quarantine_materializer_summary_path,
        "quarantine_candidate_preflight_summary_path": quarantine_candidate_preflight_summary_path,
    }
    hashes = {
        "manifest_sha256": _safe_sha_for_path(manifest_path),
        "dry_run_report_sha256": _safe_sha_for_path(dry_run_report_path),
        "review_manifest_sha256": _safe_sha_for_path(review_manifest_path),
        "admission_request_sha256": _safe_sha_for_path(admission_request_path),
        "formal_package_validation_sha256": _safe_sha_for_path(formal_package_validation_path),
        "property_package_binding_summary_sha256": _safe_sha_for_path(property_package_binding_summary_path),
        "materialization_plan_sha256": _safe_sha_for_path(materialization_plan_path),
        "materialization_plan_preflight_summary_sha256": _safe_sha_for_path(materialization_plan_preflight_summary_path),
        "offline_planner_output_sha256": _safe_sha_for_path(offline_planner_output_path),
        "property_planner_summary_sha256": _safe_sha_for_path(property_planner_summary_path),
        "materialization_dry_run_report_sha256": _safe_sha_for_path(materialization_dry_run_report_path),
        "execution_request_sha256": _safe_sha_for_path(execution_request_path),
        "execution_request_summary_sha256": _safe_sha_for_path(execution_request_summary_path),
        "execution_preflight_summary_sha256": _safe_sha_for_path(execution_preflight_summary_path),
        "quarantine_candidate_records_sha256": _safe_sha_for_path(quarantine_candidate_records_path),
        "quarantine_materializer_summary_sha256": _safe_sha_for_path(quarantine_materializer_summary_path),
        "quarantine_candidate_preflight_summary_sha256": _safe_sha_for_path(quarantine_candidate_preflight_summary_path),
    }

    warnings: list[str] = []
    errors = _consistency_errors(
        manifest=manifest,
        source_dry_run=source_dry_run,
        review=review,
        admission=admission,
        formal=formal,
        binding=binding,
        plan_payload=plan_payload,
        plan=plan,
        plan_preflight=plan_preflight,
        offline_planner=offline_planner,
        property_planner=property_planner,
        materialization_dry_run=materialization_dry_run,
        execution_request=execution_request,
        execution_summary=execution_summary,
        execution_preflight=execution_preflight,
        quarantine_candidate=quarantine_candidate,
        quarantine_summary=quarantine_summary,
        quarantine_preflight=quarantine_preflight,
        hashes=hashes,
        require_quarantine_candidate_preflight_passed=require_quarantine_candidate_preflight_passed,
        minimum_candidate_records=max(int(minimum_candidate_records), 0),
        warnings=warnings,
    )
    status = "blocked" if errors else "partial" if warnings else "ready"
    summary = _summary(
        readiness_status=status,
        paths=paths,
        hashes=hashes,
        manifest=manifest,
        source_dry_run=source_dry_run,
        review=review,
        admission=admission,
        formal=formal,
        binding=binding,
        plan_payload=plan_payload,
        plan_preflight=plan_preflight,
        offline_planner=offline_planner,
        property_planner=property_planner,
        materialization_dry_run=materialization_dry_run,
        execution_request=execution_request,
        execution_preflight=execution_preflight,
        quarantine_candidate=quarantine_candidate,
        quarantine_summary=quarantine_summary,
        quarantine_preflight=quarantine_preflight,
        errors=_stable_unique(errors),
        warnings=_stable_unique(warnings),
    )
    markdown = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "markdown": markdown}):
        minimal = _minimal_redaction_failure()
        if output_summary_path is not None:
            write_json(Path(output_summary_path).expanduser(), minimal)
        return minimal
    if output_summary_path is not None:
        write_json(Path(output_summary_path).expanduser(), summary)
    if output_markdown_path is not None:
        Path(output_markdown_path).expanduser().write_text(markdown, encoding="utf-8")
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
        summary = plan_property_training_admission_readiness(
            manifest_path=args.manifest,
            dry_run_report_path=args.dry_run_report,
            review_manifest_path=args.review_manifest,
            admission_request_path=args.admission_request,
            formal_package_validation_path=args.formal_package_validation,
            property_package_binding_summary_path=args.property_package_binding_summary,
            materialization_plan_path=args.materialization_plan,
            materialization_plan_preflight_summary_path=args.materialization_plan_preflight_summary,
            offline_planner_output_path=args.offline_planner_output,
            property_planner_summary_path=args.property_planner_summary,
            materialization_dry_run_report_path=args.materialization_dry_run_report,
            execution_request_path=args.execution_request,
            execution_request_summary_path=args.execution_request_summary,
            execution_preflight_summary_path=args.execution_preflight_summary,
            quarantine_candidate_records_path=args.quarantine_candidate_records,
            quarantine_materializer_summary_path=args.quarantine_materializer_summary,
            quarantine_candidate_preflight_summary_path=args.quarantine_candidate_preflight_summary,
            output_summary_path=args.output_summary or None,
            output_markdown_path=args.output_markdown or None,
            require_quarantine_candidate_preflight_passed=args.require_quarantine_candidate_preflight_passed,
            minimum_candidate_records=args.minimum_candidate_records,
        )
    except Exception as exc:
        err.write(f"property training admission readiness invalid: {_safe_exception_message(exc)}\n")
        return 1

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("readiness_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_training_admission_readiness",
        description="Plan future training admission readiness for property quarantine candidates.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run-report", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--admission-request", required=True)
    parser.add_argument("--formal-package-validation", required=True)
    parser.add_argument("--property-package-binding-summary", required=True)
    parser.add_argument("--materialization-plan", required=True)
    parser.add_argument("--materialization-plan-preflight-summary", required=True)
    parser.add_argument("--offline-planner-output", required=True)
    parser.add_argument("--property-planner-summary", required=True)
    parser.add_argument("--materialization-dry-run-report", required=True)
    parser.add_argument("--execution-request", required=True)
    parser.add_argument("--execution-request-summary", required=True)
    parser.add_argument("--execution-preflight-summary", required=True)
    parser.add_argument("--quarantine-candidate-records", required=True)
    parser.add_argument("--quarantine-materializer-summary", required=True)
    parser.add_argument("--quarantine-candidate-preflight-summary", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-markdown", default="")
    parser.add_argument("--require-quarantine-candidate-preflight-passed", action="store_true")
    parser.add_argument("--minimum-candidate-records", type=int, default=1)
    return parser


def _consistency_errors(
    *,
    manifest: CustomCorpusManifest,
    source_dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    formal: dict[str, Any],
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    plan: MaterializationPlan | None,
    plan_preflight: dict[str, Any],
    offline_planner: dict[str, Any],
    property_planner: dict[str, Any],
    materialization_dry_run: dict[str, Any],
    execution_request: dict[str, Any],
    execution_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    quarantine_summary: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    hashes: dict[str, str],
    require_quarantine_candidate_preflight_passed: bool,
    minimum_candidate_records: int,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    _append_errors(errors, _schema_errors(formal, binding, plan_payload, plan, plan_preflight, offline_planner, property_planner, materialization_dry_run, execution_request, execution_summary, execution_preflight, quarantine_candidate, quarantine_summary, quarantine_preflight))
    _append_errors(errors, _status_errors(source_dry_run, materialization_dry_run, execution_request, execution_preflight, quarantine_candidate, quarantine_summary, quarantine_preflight, require_quarantine_candidate_preflight_passed, minimum_candidate_records, warnings))
    _append_errors(errors, _hash_errors(quarantine_candidate, quarantine_summary, execution_preflight, quarantine_preflight, hashes))
    _append_errors(errors, _id_errors(manifest, source_dry_run, review, admission, formal, binding, plan_payload, plan_preflight, offline_planner, property_planner, materialization_dry_run, execution_request, execution_summary, execution_preflight, quarantine_candidate, quarantine_summary, quarantine_preflight))
    _append_errors(errors, _record_errors(plan_payload, execution_request, execution_summary, execution_preflight, quarantine_candidate, quarantine_summary, quarantine_preflight, admission, binding, minimum_candidate_records))
    return _stable_unique(errors)


def _schema_errors(
    formal: dict[str, Any],
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    plan: MaterializationPlan | None,
    plan_preflight: dict[str, Any],
    offline_planner: dict[str, Any],
    property_planner: dict[str, Any],
    materialization_dry_run: dict[str, Any],
    execution_request: dict[str, Any],
    execution_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    quarantine_summary: dict[str, Any],
    quarantine_preflight: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if quarantine_preflight.get("schema_version") != _QUARANTINE_PREFLIGHT_SCHEMA_VERSION:
        errors.append("quarantine_candidate_preflight_schema_invalid")
    if quarantine_candidate.get("schema_version") != _QUARANTINE_CANDIDATE_SCHEMA_VERSION:
        errors.append("quarantine_candidate_schema_invalid")
    if quarantine_summary.get("schema_version") != _QUARANTINE_SUMMARY_SCHEMA_VERSION:
        errors.append("quarantine_materializer_summary_schema_invalid")
    if execution_preflight.get("schema_version") != _EXECUTION_PREFLIGHT_SCHEMA_VERSION:
        errors.append("execution_preflight_schema_invalid")
    if execution_request.get("schema_version") != _REQUEST_SCHEMA_VERSION:
        errors.append("execution_request_schema_invalid")
    if execution_summary.get("schema_version") != _REQUEST_SUMMARY_SCHEMA_VERSION:
        errors.append("execution_request_summary_schema_invalid")
    if materialization_dry_run.get("schema_version") != _MATERIALIZATION_DRY_RUN_SCHEMA_VERSION:
        errors.append("materialization_dry_run_report_schema_invalid")
    if property_planner.get("schema_version") != _PROPERTY_PLANNER_SCHEMA_VERSION:
        errors.append("property_planner_schema_invalid")
    if offline_planner.get("schema_version") != _OFFLINE_PLANNER_SCHEMA_VERSION:
        errors.append("offline_planner_schema_invalid")
    if plan_preflight.get("schema_version") != _PREFLIGHT_SCHEMA_VERSION:
        errors.append("preflight_schema_invalid")
    if binding.get("schema_version") != _PROPERTY_BINDING_SCHEMA_VERSION:
        errors.append("property_package_binding_schema_invalid")
    if formal.get("schema_version") != _FORMAL_SCHEMA_VERSION:
        errors.append("formal_package_validation_schema_invalid")
    if plan_payload.get("schema_version") != _MATERIALIZATION_SCHEMA_VERSION:
        errors.append("materialization_plan_schema_invalid")
    if plan is None and plan_payload.get("schema_version") == _MATERIALIZATION_SCHEMA_VERSION:
        errors.append("materialization_plan_invalid")
    return errors


def _status_errors(
    source_dry_run: CustomCorpusDryRunReport,
    materialization_dry_run: dict[str, Any],
    execution_request: dict[str, Any],
    execution_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    quarantine_summary: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    require_quarantine_candidate_preflight_passed: bool,
    minimum_candidate_records: int,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    preflight_status = str(quarantine_preflight.get("preflight_status", ""))
    if preflight_status == "failed":
        errors.append("quarantine_candidate_preflight_failed")
    elif preflight_status == "needs_review":
        if require_quarantine_candidate_preflight_passed:
            errors.append("quarantine_candidate_preflight_needs_review")
        else:
            warnings.append("quarantine_candidate_preflight_needs_review")
    elif preflight_status != "passed":
        errors.append("quarantine_candidate_preflight_status_invalid")
    if quarantine_preflight.get("preflight_errors"):
        errors.append("quarantine_candidate_preflight_has_errors")

    for status in (str(quarantine_candidate.get("materializer_status", "")), str(quarantine_summary.get("materializer_status", "")), str(quarantine_preflight.get("quarantine_materializer_status", ""))):
        if status == "failed":
            errors.append("quarantine_materializer_failed")
        elif status == "needs_review":
            if require_quarantine_candidate_preflight_passed:
                errors.append("quarantine_materializer_needs_review")
            else:
                warnings.append("quarantine_materializer_needs_review")
        elif status != "written":
            errors.append("quarantine_materializer_status_invalid")
    if quarantine_summary.get("materializer_errors"):
        errors.append("quarantine_materializer_has_errors")
    if quarantine_candidate.get("materialization_mode") != "candidate_quarantine":
        errors.append("quarantine_materialization_mode_invalid")

    if execution_preflight.get("preflight_status") == "failed":
        errors.append("execution_preflight_failed")
    if execution_preflight.get("preflight_errors"):
        errors.append("execution_preflight_has_errors")
    if execution_request.get("request_status") != "written" or execution_preflight.get("request_status") != "written":
        errors.append("execution_request_not_written")
    if execution_request.get("execution_mode") != "request_only":
        errors.append("execution_mode_invalid")
    if execution_preflight.get("materializer_status") != "not_run" or execution_request.get("materializer_status") != "not_run":
        errors.append("materializer_status_not_run")
    if materialization_dry_run.get("dry_run_status") == "failed" or execution_preflight.get("dry_run_status") == "failed":
        errors.append("materialization_dry_run_failed")
    if materialization_dry_run.get("planner_status") == "failed" or execution_preflight.get("planner_status") == "failed":
        errors.append("planner_summary_failed")
    if materialization_dry_run.get("offline_planner_status") != "planned" or execution_preflight.get("offline_planner_status") != "planned":
        errors.append("offline_planner_failed")
    if materialization_dry_run.get("materialization_decision") != "planned" or execution_preflight.get("materialization_decision") != "planned":
        errors.append("materialization_decision_not_planned")

    candidate_count = len(_raw_candidate_records(quarantine_candidate))
    if candidate_count < max(minimum_candidate_records, 1):
        errors.append("minimum_candidate_record_count_not_met")
    if (
        quarantine_candidate.get("training_admitted") is not False
        or quarantine_summary.get("training_admitted") is not False
        or quarantine_preflight.get("training_admitted") is not False
        or execution_preflight.get("training_admitted") is not False
        or execution_request.get("training_admitted") is not False
        or source_dry_run.confirmation_boundary.training_dataset_admitted is not False
    ):
        errors.append("training_admitted")
    if (
        quarantine_candidate.get("phase1_status") != "not_run"
        or quarantine_summary.get("phase1_status") != "not_run"
        or quarantine_preflight.get("phase1_status") != "not_run"
        or execution_preflight.get("phase1_status") != "not_run"
        or execution_request.get("phase1_status") != "not_run"
        or source_dry_run.confirmation_boundary.phase1_status != "not_run"
    ):
        errors.append("phase1_ran")
    if (
        quarantine_candidate.get("dataset_confirmation_changed") is not False
        or quarantine_summary.get("dataset_confirmation_changed") is not False
        or quarantine_preflight.get("dataset_confirmation_changed") is not False
        or execution_preflight.get("dataset_confirmation_changed") is not False
        or execution_request.get("dataset_confirmation_changed") is not False
        or source_dry_run.confirmation_boundary.dataset_confirmation_confirmed is not False
    ):
        errors.append("dataset_confirmation_changed")
    return errors


def _hash_errors(
    quarantine_candidate: dict[str, Any],
    quarantine_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    hashes: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    summary_pairs = (
        ("manifest_sha256", "manifest_sha256", "manifest_sha256_mismatch"),
        ("dry_run_report_sha256", "dry_run_report_sha256", "dry_run_report_sha256_mismatch"),
        ("review_manifest_sha256", "review_manifest_sha256", "review_manifest_sha256_mismatch"),
        ("admission_request_sha256", "admission_request_sha256", "admission_request_sha256_mismatch"),
        ("formal_package_validation_sha256", "formal_package_validation_sha256", "formal_package_validation_sha256_mismatch"),
        ("property_package_binding_summary_sha256", "property_package_binding_summary_sha256", "property_package_binding_summary_sha256_mismatch"),
        ("materialization_plan_sha256", "materialization_plan_sha256", "materialization_plan_sha256_mismatch"),
        ("materialization_plan_preflight_summary_sha256", "materialization_plan_preflight_summary_sha256", "materialization_plan_preflight_summary_sha256_mismatch"),
        ("offline_planner_output_sha256", "offline_planner_output_sha256", "offline_planner_output_sha256_mismatch"),
        ("property_planner_summary_sha256", "property_planner_summary_sha256", "property_planner_summary_sha256_mismatch"),
        ("materialization_dry_run_report_sha256", "materialization_dry_run_report_sha256", "materialization_dry_run_report_sha256_mismatch"),
        ("execution_request_sha256", "execution_request_sha256", "execution_request_sha256_mismatch"),
        ("execution_request_summary_sha256", "execution_request_summary_sha256", "execution_request_summary_sha256_mismatch"),
        ("execution_preflight_summary_sha256", "execution_preflight_summary_sha256", "execution_preflight_summary_sha256_mismatch"),
        ("quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256", "quarantine_candidate_records_sha256_mismatch"),
        ("quarantine_materializer_summary_sha256", "quarantine_materializer_summary_sha256", "quarantine_materializer_summary_sha256_mismatch"),
    )
    candidate_pairs = (
        ("source_manifest_sha256", "manifest_sha256", "manifest_sha256_mismatch"),
        ("source_dry_run_report_sha256", "dry_run_report_sha256", "dry_run_report_sha256_mismatch"),
        ("source_review_manifest_sha256", "review_manifest_sha256", "review_manifest_sha256_mismatch"),
        ("source_admission_request_sha256", "admission_request_sha256", "admission_request_sha256_mismatch"),
        ("source_formal_package_validation_sha256", "formal_package_validation_sha256", "formal_package_validation_sha256_mismatch"),
        ("source_property_package_binding_summary_sha256", "property_package_binding_summary_sha256", "property_package_binding_summary_sha256_mismatch"),
        ("source_materialization_plan_sha256", "materialization_plan_sha256", "materialization_plan_sha256_mismatch"),
        ("source_materialization_preflight_summary_sha256", "materialization_plan_preflight_summary_sha256", "materialization_plan_preflight_summary_sha256_mismatch"),
        ("source_offline_planner_output_sha256", "offline_planner_output_sha256", "offline_planner_output_sha256_mismatch"),
        ("source_property_planner_summary_sha256", "property_planner_summary_sha256", "property_planner_summary_sha256_mismatch"),
        ("source_materialization_dry_run_report_sha256", "materialization_dry_run_report_sha256", "materialization_dry_run_report_sha256_mismatch"),
        ("source_execution_request_sha256", "execution_request_sha256", "execution_request_sha256_mismatch"),
        ("source_execution_request_summary_sha256", "execution_request_summary_sha256", "execution_request_summary_sha256_mismatch"),
        ("source_execution_preflight_summary_sha256", "execution_preflight_summary_sha256", "execution_preflight_summary_sha256_mismatch"),
    )
    for field, hash_field, error in candidate_pairs:
        if quarantine_candidate.get(field) and quarantine_candidate.get(field) != hashes[hash_field]:
            errors.append(error)
    for container in (quarantine_summary, execution_preflight, quarantine_preflight):
        for field, hash_field, error in summary_pairs:
            if container.get(field) and container.get(field) != hashes[hash_field]:
                errors.append(error)
    if quarantine_summary.get("quarantine_candidate_records_sha256") != hashes["quarantine_candidate_records_sha256"]:
        errors.append("quarantine_candidate_records_sha256_mismatch")
    declared_preflight_sha = str(quarantine_preflight.get("quarantine_candidate_preflight_summary_sha256", ""))
    if declared_preflight_sha and declared_preflight_sha != hashes["quarantine_candidate_preflight_summary_sha256"]:
        errors.append("quarantine_candidate_preflight_summary_sha256_mismatch")
    return _stable_unique(errors)


def _id_errors(
    manifest: CustomCorpusManifest,
    source_dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    formal: dict[str, Any],
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    plan_preflight: dict[str, Any],
    offline_planner: dict[str, Any],
    property_planner: dict[str, Any],
    materialization_dry_run: dict[str, Any],
    execution_request: dict[str, Any],
    execution_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    quarantine_summary: dict[str, Any],
    quarantine_preflight: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    id_sets = (
        (
            manifest.corpus_id,
            quarantine_candidate.get("corpus_id"),
            quarantine_summary.get("corpus_id"),
            quarantine_preflight.get("corpus_id"),
            execution_preflight.get("corpus_id"),
            execution_request.get("corpus_id"),
            execution_summary.get("corpus_id"),
            materialization_dry_run.get("corpus_id"),
            property_planner.get("corpus_id"),
            plan_preflight.get("corpus_id"),
            binding.get("corpus_id"),
            formal.get("corpus_id"),
            plan_payload.get("corpus_id"),
        ),
        (
            source_dry_run.run_id,
            quarantine_candidate.get("source_dry_run_id"),
            quarantine_summary.get("source_dry_run_id"),
            quarantine_preflight.get("source_dry_run_id"),
            execution_preflight.get("source_dry_run_id"),
            execution_request.get("source_dry_run_id"),
            execution_summary.get("source_dry_run_id"),
            materialization_dry_run.get("corpus_dry_run_id"),
            property_planner.get("dry_run_id"),
            plan_preflight.get("dry_run_id"),
            binding.get("dry_run_id"),
            formal.get("dry_run_id"),
            plan_payload.get("dry_run_id"),
        ),
        (
            review.review_manifest_id,
            quarantine_candidate.get("review_manifest_id"),
            quarantine_summary.get("review_manifest_id"),
            quarantine_preflight.get("review_manifest_id"),
            execution_preflight.get("review_manifest_id"),
            execution_request.get("review_manifest_id"),
            execution_summary.get("review_manifest_id"),
            materialization_dry_run.get("review_manifest_id"),
            property_planner.get("review_manifest_id"),
            plan_preflight.get("review_manifest_id"),
            binding.get("review_manifest_id"),
            formal.get("review_manifest_id"),
            plan_payload.get("review_manifest_id"),
        ),
        (
            admission.admission_request_id,
            quarantine_candidate.get("admission_request_id"),
            quarantine_summary.get("admission_request_id"),
            quarantine_preflight.get("admission_request_id"),
            execution_preflight.get("admission_request_id"),
            execution_request.get("admission_request_id"),
            execution_summary.get("admission_request_id"),
            materialization_dry_run.get("admission_request_id"),
            property_planner.get("admission_request_id"),
            plan_preflight.get("admission_request_id"),
            binding.get("admission_request_id"),
            formal.get("admission_request_id"),
            plan_payload.get("admission_request_id"),
        ),
        (
            plan_payload.get("materialization_plan_id"),
            quarantine_candidate.get("materialization_plan_id"),
            quarantine_summary.get("materialization_plan_id"),
            quarantine_preflight.get("materialization_plan_id"),
            execution_preflight.get("materialization_plan_id"),
            execution_request.get("materialization_plan_id"),
            execution_summary.get("materialization_plan_id"),
            materialization_dry_run.get("materialization_plan_id"),
            property_planner.get("materialization_plan_id"),
            plan_preflight.get("materialization_plan_id"),
            offline_planner.get("materialization_plan_id"),
        ),
        (
            execution_preflight.get("execution_request_id"),
            quarantine_candidate.get("execution_request_id"),
            quarantine_summary.get("execution_request_id"),
            quarantine_preflight.get("execution_request_id"),
            execution_request.get("execution_request_id"),
            execution_summary.get("execution_request_id"),
        ),
        (
            quarantine_candidate.get("quarantine_run_id"),
            quarantine_summary.get("quarantine_run_id"),
            quarantine_preflight.get("quarantine_run_id"),
        ),
        (
            binding.get("review_queue_id"),
            quarantine_candidate.get("review_queue_id"),
            quarantine_summary.get("review_queue_id"),
            quarantine_preflight.get("review_queue_id"),
        ),
        (
            binding.get("property_candidate_manifest_id"),
            quarantine_candidate.get("property_candidate_manifest_id"),
            quarantine_summary.get("property_candidate_manifest_id"),
            quarantine_preflight.get("property_candidate_manifest_id"),
        ),
    )
    error_codes = (
        "corpus_id_mismatch",
        "dry_run_id_mismatch",
        "review_manifest_id_mismatch",
        "admission_request_id_mismatch",
        "materialization_plan_id_mismatch",
        "execution_request_id_mismatch",
        "quarantine_run_id_mismatch",
        "review_queue_id_mismatch",
        "property_candidate_manifest_id_mismatch",
    )
    for values, error in zip(id_sets, error_codes, strict=True):
        present = [str(value) for value in values if value not in {None, ""}]
        if len(set(present)) > 1:
            errors.append(error)
    return errors


def _record_errors(
    plan_payload: dict[str, Any],
    execution_request: dict[str, Any],
    execution_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    quarantine_summary: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    admission: AdmissionRequest,
    binding: dict[str, Any],
    minimum_candidate_records: int,
) -> list[str]:
    errors: list[str] = []
    materialization_records = _materialize_records(plan_payload)
    execution_records = _raw_execution_records(execution_request)
    candidate_records = _raw_candidate_records(quarantine_candidate)
    materialization_ids = [str(record.get("materialization_record_id", "")) for record in materialization_records]
    execution_ids = [str(record.get("execution_record_id", "")) for record in execution_records]
    candidate_ids = [str(record.get("candidate_record_id", "")) for record in candidate_records]
    if not materialization_records:
        errors.append("no_materialization_records")
    if not execution_records:
        errors.append("no_execution_records")
    if not candidate_records:
        errors.append("no_candidate_records")
    if len(candidate_records) < max(minimum_candidate_records, 1):
        errors.append("minimum_candidate_record_count_not_met")
    for container in (quarantine_candidate, quarantine_summary, quarantine_preflight):
        if int(container.get("candidate_record_count", -1)) != len(candidate_records):
            errors.append("candidate_record_count_mismatch")
        if _safe_list(container.get("candidate_record_ids")) != candidate_ids:
            errors.append("candidate_record_ids_mismatch")
    for container in (quarantine_summary, quarantine_preflight):
        if int(container.get("materialization_record_count", -1)) != len(materialization_records):
            errors.append("materialization_record_count_mismatch")
        if int(container.get("execution_record_count", -1)) != len(execution_records):
            errors.append("execution_record_count_mismatch")
        if _safe_list(container.get("materialization_record_ids")) != materialization_ids:
            errors.append("materialization_record_ids_mismatch")
        if _safe_list(container.get("execution_record_ids")) != execution_ids:
            errors.append("execution_record_ids_mismatch")
    if _safe_list(execution_preflight.get("materialization_record_ids")) != materialization_ids:
        errors.append("materialization_record_ids_mismatch")
    if _safe_list(execution_preflight.get("execution_record_ids")) != execution_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(execution_summary.get("materialization_record_ids")) != materialization_ids:
        errors.append("materialization_record_ids_mismatch")
    if _safe_list(execution_summary.get("execution_record_ids")) != execution_ids:
        errors.append("execution_record_ids_mismatch")

    materialization_by_id = {str(record.get("materialization_record_id", "")): record for record in materialization_records}
    execution_by_id = {str(record.get("execution_record_id", "")): record for record in execution_records}
    for record in candidate_records:
        if record.get("candidate_status") != "quarantined":
            errors.append("candidate_status_invalid")
        boundary = record.get("materialization_boundary")
        if not isinstance(boundary, list) or not {"candidate_only", "not_training", "not_phase1", "dataset_confirmation_unchanged"}.issubset({str(item) for item in boundary}):
            errors.append("candidate_boundary_invalid")
        materialization_record = materialization_by_id.get(str(record.get("materialization_record_id", "")))
        execution_record = execution_by_id.get(str(record.get("execution_record_id", "")))
        if materialization_record is None:
            errors.append("candidate_materialization_record_unknown")
        if execution_record is None:
            errors.append("candidate_execution_record_unknown")
        if materialization_record and str(materialization_record.get("record_id", "")) != str(record.get("record_id", "")):
            errors.append("candidate_record_id_mismatch")
        if execution_record and str(execution_record.get("record_id", "")) != str(record.get("record_id", "")):
            errors.append("candidate_record_id_mismatch")
    _append_errors(errors, _candidate_record_source_errors(candidate_records, admission, binding))
    return _stable_unique(errors)


def _summary(
    *,
    readiness_status: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    manifest: CustomCorpusManifest,
    source_dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    formal: dict[str, Any],
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    plan_preflight: dict[str, Any],
    offline_planner: dict[str, Any],
    property_planner: dict[str, Any],
    materialization_dry_run: dict[str, Any],
    execution_request: dict[str, Any],
    execution_preflight: dict[str, Any],
    quarantine_candidate: dict[str, Any],
    quarantine_summary: dict[str, Any],
    quarantine_preflight: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    materialization_records = _materialize_records(plan_payload)
    execution_records = _raw_execution_records(execution_request)
    candidate_records = _raw_candidate_records(quarantine_candidate)
    candidate_ids = [str(record.get("candidate_record_id", "")) for record in candidate_records]
    return {
        "schema_version": _SCHEMA_VERSION,
        "readiness_status": readiness_status,
        "manifest_path": _basename(paths["manifest_path"], "manifest.json"),
        "manifest_sha256": hashes["manifest_sha256"],
        "dry_run_report_path": _basename(paths["dry_run_report_path"], "dry_run_report.json"),
        "dry_run_report_sha256": hashes["dry_run_report_sha256"],
        "review_manifest_path": _basename(paths["review_manifest_path"], "review_manifest.json"),
        "review_manifest_sha256": hashes["review_manifest_sha256"],
        "admission_request_path": _basename(paths["admission_request_path"], "custom_corpus_admission.draft.json"),
        "admission_request_sha256": hashes["admission_request_sha256"],
        "formal_package_validation_path": _basename(paths["formal_package_validation_path"], "custom_corpus_admission_package_validation.json"),
        "formal_package_validation_sha256": hashes["formal_package_validation_sha256"],
        "property_package_binding_summary_path": _basename(paths["property_package_binding_summary_path"], "property_package_binding_summary.json"),
        "property_package_binding_summary_sha256": hashes["property_package_binding_summary_sha256"],
        "materialization_plan_path": _basename(paths["materialization_plan_path"], "custom_corpus_materialization.draft.json"),
        "materialization_plan_sha256": hashes["materialization_plan_sha256"],
        "materialization_plan_preflight_summary_path": _basename(paths["materialization_plan_preflight_summary_path"], "materialization_plan_preflight_summary.json"),
        "materialization_plan_preflight_summary_sha256": hashes["materialization_plan_preflight_summary_sha256"],
        "offline_planner_output_path": _basename(paths["offline_planner_output_path"], "offline_materialization_planner_output.json"),
        "offline_planner_output_sha256": hashes["offline_planner_output_sha256"],
        "property_planner_summary_path": _basename(paths["property_planner_summary_path"], "property_materialization_planner_summary.json"),
        "property_planner_summary_sha256": hashes["property_planner_summary_sha256"],
        "materialization_dry_run_report_path": _basename(paths["materialization_dry_run_report_path"], "property_materialization_dry_run_report.json"),
        "materialization_dry_run_report_sha256": hashes["materialization_dry_run_report_sha256"],
        "execution_request_path": _basename(paths["execution_request_path"], "property_materializer_execution_request.json"),
        "execution_request_sha256": hashes["execution_request_sha256"],
        "execution_request_summary_path": _basename(paths["execution_request_summary_path"], "property_materializer_execution_request_summary.json"),
        "execution_request_summary_sha256": hashes["execution_request_summary_sha256"],
        "execution_preflight_summary_path": _basename(paths["execution_preflight_summary_path"], "execution_preflight_summary.json"),
        "execution_preflight_summary_sha256": hashes["execution_preflight_summary_sha256"],
        "quarantine_candidate_records_path": _basename(paths["quarantine_candidate_records_path"], "property_quarantine_candidate_records.json"),
        "quarantine_candidate_records_sha256": hashes["quarantine_candidate_records_sha256"],
        "quarantine_materializer_summary_path": _basename(paths["quarantine_materializer_summary_path"], "property_quarantine_materializer_summary.json"),
        "quarantine_materializer_summary_sha256": hashes["quarantine_materializer_summary_sha256"],
        "quarantine_candidate_preflight_summary_path": _basename(paths["quarantine_candidate_preflight_summary_path"], "property_quarantine_candidate_preflight_summary.json"),
        "quarantine_candidate_preflight_summary_sha256": hashes["quarantine_candidate_preflight_summary_sha256"],
        "corpus_id": manifest.corpus_id,
        "source_dry_run_id": source_dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "materialization_plan_id": str(plan_payload.get("materialization_plan_id", "")),
        "execution_request_id": str(execution_request.get("execution_request_id", "")),
        "quarantine_run_id": str(quarantine_candidate.get("quarantine_run_id", quarantine_summary.get("quarantine_run_id", ""))),
        "review_queue_id": str(binding.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(binding.get("property_candidate_manifest_id", "")),
        "dataset_target": str(plan_payload.get("dataset_target", property_planner.get("dataset_target", ""))),
        "quarantine_candidate_preflight_status": str(quarantine_preflight.get("preflight_status", "")),
        "quarantine_materializer_status": str(quarantine_summary.get("materializer_status", quarantine_candidate.get("materializer_status", ""))),
        "training_admitted": source_dry_run.confirmation_boundary.training_dataset_admitted,
        "phase1_status": source_dry_run.confirmation_boundary.phase1_status,
        "dataset_confirmation_changed": source_dry_run.confirmation_boundary.dataset_confirmation_confirmed,
        "admission_record_count": len(admission.admission_records),
        "admit_count": len([record for record in admission.admission_records if record.action == "admit"]),
        "exclude_count": len([record for record in admission.admission_records if record.action == "exclude"]),
        "blocked_record_count": len(_safe_list(binding.get("blocked_record_ids"))),
        "materialization_record_count": len(materialization_records),
        "execution_record_count": len(execution_records),
        "candidate_record_count": len(candidate_records),
        "candidate_record_ids": candidate_ids,
        "materialization_record_ids": [str(record.get("materialization_record_id", "")) for record in materialization_records],
        "execution_record_ids": [str(record.get("execution_record_id", "")) for record in execution_records],
        "admit_record_ids": [record.record_id for record in admission.admission_records if record.action == "admit"],
        "exclude_record_ids": [record.record_id for record in admission.admission_records if record.action == "exclude"],
        "blocked_record_ids": _safe_list(binding.get("blocked_record_ids")),
        "readiness_record_summaries": _readiness_record_summaries(candidate_records, hashes["quarantine_candidate_records_sha256"]),
        "planned_training_admission_candidate_record_ids": [] if errors else candidate_ids,
        "blocked_from_training_admission_record_ids": candidate_ids if errors else [],
        "readiness_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _readiness_record_summaries(candidate_records: list[dict[str, Any]], quarantine_candidate_sha256: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for record in candidate_records:
        summaries.append(
            {
                "candidate_record_id": str(record.get("candidate_record_id", "")),
                "record_id": str(record.get("record_id", "")),
                "materialization_record_id": str(record.get("materialization_record_id", "")),
                "execution_record_id": str(record.get("execution_record_id", "")),
                "admission_record_id": str(record.get("admission_record_id", "")),
                "review_id": str(record.get("review_id", "")),
                "document_id": str(record.get("document_id", "")),
                "field_name": str(record.get("field_name", "")),
                "readiness_action": "candidate_for_future_training_admission",
                "source_artifact_sha256": str(record.get("source_artifact_sha256", "")),
                "review_artifact_sha256": str(record.get("review_artifact_sha256", "")),
                "admission_request_sha256": str(record.get("admission_request_sha256", "")),
                "package_validation_sha256": str(record.get("package_validation_sha256", "")),
                "materialization_plan_sha256": str(record.get("materialization_plan_sha256", "")),
                "quarantine_candidate_records_sha256": quarantine_candidate_sha256,
            }
        )
    return summaries


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Training Admission Readiness Evidence",
            "",
            f"- Readiness status: `{summary['readiness_status']}`",
            f"- Quarantine run id: `{summary['quarantine_run_id']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Source dry-run id: `{summary['source_dry_run_id']}`",
            f"- Review manifest id: `{summary['review_manifest_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Execution request id: `{summary['execution_request_id']}`",
            f"- Quarantine candidate preflight status: `{summary['quarantine_candidate_preflight_status']}`",
            f"- Quarantine materializer status: `{summary['quarantine_materializer_status']}`",
            f"- Candidate records: `{summary['candidate_record_count']}`",
            f"- Materialization records: `{summary['materialization_record_count']}`",
            f"- Execution records: `{summary['execution_record_count']}`",
            f"- Planned training admission candidate ids: `{json.dumps(summary['planned_training_admission_candidate_record_ids'])}`",
            f"- Blocked candidate ids: `{json.dumps(summary['blocked_from_training_admission_record_ids'])}`",
            f"- Readiness errors: `{json.dumps(summary['readiness_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is training admission readiness only.",
            "- No training data was admitted.",
            "- No training CSV/JSONL/Parquet/LMDB was created.",
            "- No candidate CSV/JSONL/Parquet/LMDB was created.",
            "- No Phase 1 was run.",
            "- DatasetConfirmation was not changed.",
            "- No model training or evaluation was run.",
            "",
        ]
    )


def _candidate_record_source_errors(
    candidate_records: list[dict[str, Any]],
    admission: AdmissionRequest,
    binding: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    admit_ids = [record.record_id for record in admission.admission_records if record.action == "admit" and record.review_decision == "accept"]
    exclude_ids = [record.record_id for record in admission.admission_records if record.action == "exclude"]
    needs_review_ids = [record.record_id for record in admission.admission_records if record.action == "needs_review" or record.review_decision == "needs_review"]
    blocked_ids = _safe_list(binding.get("blocked_record_ids"))
    for record in candidate_records:
        record_id = str(record.get("record_id", ""))
        if record_id in exclude_ids:
            errors.append("candidate_record_from_excluded_record")
        if record_id in blocked_ids:
            errors.append("candidate_record_from_blocked_record")
        if record_id in needs_review_ids:
            errors.append("candidate_record_from_needs_review_record")
        if record_id not in admit_ids:
            errors.append("candidate_record_not_admitted")
    return _stable_unique(errors)


def _load_source_dry_run_report(path: str | Path) -> CustomCorpusDryRunReport:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return CustomCorpusDryRunReport.model_validate(payload)
    except Exception as exc:
        raise CustomCorpusPropertyTrainingAdmissionReadinessError("dry-run report invalid") from exc


def _load_materialization_plan_payload(path: str | Path) -> tuple[dict[str, Any], MaterializationPlan | None]:
    payload = _read_safe_json_dict(path, "materialization plan")
    if payload.get("schema_version") != _MATERIALIZATION_SCHEMA_VERSION:
        return payload, None
    try:
        return payload, load_materialization_plan(path)
    except CustomCorpusMaterializationError:
        return payload, None


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyTrainingAdmissionReadinessError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyTrainingAdmissionReadinessError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyTrainingAdmissionReadinessError(f"{label} invalid")
    return payload


def _materialize_records(plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = plan_payload.get("materialization_records", [])
    return [record for record in records if isinstance(record, dict) and record.get("action") == "materialize_candidate"] if isinstance(records, list) else []


def _raw_execution_records(request: dict[str, Any]) -> list[dict[str, Any]]:
    records = request.get("execution_records", [])
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _raw_candidate_records(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    records = candidate.get("candidate_records", [])
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _safe_sha_for_path(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except Exception:
        return ""


def _basename(path: str | Path, fallback: str) -> str:
    return Path(path).name or fallback


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


def _contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _input_contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    allowed_input_markers = {".csv", ".jsonl", ".parquet", ".lmdb"}
    markers = tuple(marker for marker in _FORBIDDEN_MARKERS if marker not in allowed_input_markers)
    if any(marker.lower() in lowered for marker in markers):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "readiness_status": "blocked",
        "readiness_errors": ["property_training_admission_readiness_redaction_failed"],
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
