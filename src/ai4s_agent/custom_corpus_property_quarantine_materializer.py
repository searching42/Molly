from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import redirect_stderr
from datetime import UTC, datetime
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


_CANDIDATE_SCHEMA_VERSION = "custom_corpus_property_quarantine_materialization.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_quarantine_materializer.v1"
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
)
_ARTIFACTS = {
    "quarantine_candidate_records_json": "property_quarantine_candidate_records.json",
    "quarantine_materializer_summary_json": "property_quarantine_materializer_summary.json",
    "redacted_quarantine_materializer_evidence_md": "redacted_property_quarantine_materializer_evidence.md",
}


class CustomCorpusPropertyQuarantineMaterializerError(ValueError):
    pass


def run_property_quarantine_materializer(
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
    output_dir: str | Path,
    quarantine_run_id: str,
    created_by: str,
    confirm_quarantine_materialization: bool,
    allow_execution_preflight_needs_review: bool = False,
) -> dict[str, Any]:
    quarantine_run_id = _required_safe_id(quarantine_run_id, "quarantine_run_id")
    created_by = _required_safe_label(created_by, "created_by")
    manifest = load_custom_corpus_manifest(manifest_path)
    source_dry_run = _load_source_dry_run_report(dry_run_report_path)
    review = load_review_manifest(review_manifest_path)
    admission = load_admission_request(admission_request_path)
    formal = _read_safe_json_dict(formal_package_validation_path, "formal package validation")
    binding = _read_safe_json_dict(property_package_binding_summary_path, "property package binding summary")
    plan_payload, plan = _load_materialization_plan_payload(materialization_plan_path)
    preflight = _read_safe_json_dict(materialization_plan_preflight_summary_path, "materialization plan preflight summary")
    offline_planner = _read_safe_json_dict(offline_planner_output_path, "offline planner output")
    property_planner = _read_safe_json_dict(property_planner_summary_path, "property planner summary")
    materialization_dry_run = _read_safe_json_dict(materialization_dry_run_report_path, "materialization dry-run report")
    execution_request = _read_safe_json_dict(execution_request_path, "execution request")
    execution_summary = _read_safe_json_dict(execution_request_summary_path, "execution request summary")
    execution_preflight = _read_safe_json_dict(execution_preflight_summary_path, "execution preflight summary")

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
    }
    run_dir = Path(output_dir).expanduser() / quarantine_run_id
    errors: list[str] = []
    warnings: list[str] = []
    if not confirm_quarantine_materialization:
        errors.append("quarantine_materialization_not_confirmed")
    if run_dir.exists() and any(run_dir.iterdir()):
        errors.append("output_directory_not_clean")
    _append_errors(
        errors,
        _consistency_errors(
            manifest=manifest,
            source_dry_run=source_dry_run,
            review=review,
            admission=admission,
            formal=formal,
            binding=binding,
            plan_payload=plan_payload,
            plan=plan,
            preflight=preflight,
            offline_planner=offline_planner,
            property_planner=property_planner,
            materialization_dry_run=materialization_dry_run,
            execution_request=execution_request,
            execution_summary=execution_summary,
            execution_preflight=execution_preflight,
            hashes=hashes,
            allow_execution_preflight_needs_review=allow_execution_preflight_needs_review,
            warnings=warnings,
        ),
    )
    candidate_records = _candidate_records(
        quarantine_run_id=quarantine_run_id,
        plan_payload=plan_payload,
        execution_request=execution_request,
        hashes=hashes,
    )
    if not candidate_records:
        errors.append("no_candidate_records")
    _append_errors(errors, _candidate_record_source_errors(candidate_records, admission, binding))

    if errors:
        summary = _summary(
            materializer_status="failed",
            quarantine_run_id=quarantine_run_id,
            paths=paths,
            hashes=hashes,
            quarantine_candidate_sha256="",
            manifest=manifest,
            source_dry_run=source_dry_run,
            review=review,
            admission=admission,
            formal=formal,
            binding=binding,
            plan_payload=plan_payload,
            preflight=preflight,
            offline_planner=offline_planner,
            property_planner=property_planner,
            materialization_dry_run=materialization_dry_run,
            execution_request=execution_request,
            execution_summary=execution_summary,
            execution_preflight=execution_preflight,
            candidate_records=candidate_records,
            errors=_stable_unique(errors),
            warnings=_stable_unique(warnings),
        )
        return _minimal_redaction_failure() if _contains_forbidden_material(summary) else summary

    status = "needs_review" if warnings else "written"
    candidate_artifact = _candidate_artifact(
        materializer_status=status,
        quarantine_run_id=quarantine_run_id,
        created_by=created_by,
        hashes=hashes,
        manifest=manifest,
        source_dry_run=source_dry_run,
        review=review,
        admission=admission,
        binding=binding,
        plan_payload=plan_payload,
        candidate_records=candidate_records,
    )
    summary = _summary(
        materializer_status=status,
        quarantine_run_id=quarantine_run_id,
        paths=paths,
        hashes=hashes,
        quarantine_candidate_sha256="",
        manifest=manifest,
        source_dry_run=source_dry_run,
        review=review,
        admission=admission,
        formal=formal,
        binding=binding,
        plan_payload=plan_payload,
        preflight=preflight,
        offline_planner=offline_planner,
        property_planner=property_planner,
        materialization_dry_run=materialization_dry_run,
        execution_request=execution_request,
        execution_summary=execution_summary,
        execution_preflight=execution_preflight,
        candidate_records=candidate_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"candidate_artifact": candidate_artifact, "summary": summary, "evidence": evidence}):
        return _minimal_redaction_failure()

    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = run_dir / _ARTIFACTS["quarantine_candidate_records_json"]
    write_json(candidate_path, candidate_artifact)
    quarantine_candidate_sha256 = _safe_sha_for_path(candidate_path)
    summary = _summary(
        materializer_status=status,
        quarantine_run_id=quarantine_run_id,
        paths=paths,
        hashes=hashes,
        quarantine_candidate_sha256=quarantine_candidate_sha256,
        manifest=manifest,
        source_dry_run=source_dry_run,
        review=review,
        admission=admission,
        formal=formal,
        binding=binding,
        plan_payload=plan_payload,
        preflight=preflight,
        offline_planner=offline_planner,
        property_planner=property_planner,
        materialization_dry_run=materialization_dry_run,
        execution_request=execution_request,
        execution_summary=execution_summary,
        execution_preflight=execution_preflight,
        candidate_records=candidate_records,
        errors=[],
        warnings=_stable_unique(warnings),
    )
    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "evidence": evidence}):
        _safe_unlink(candidate_path)
        return _minimal_redaction_failure()
    write_json(run_dir / _ARTIFACTS["quarantine_materializer_summary_json"], summary)
    (run_dir / _ARTIFACTS["redacted_quarantine_materializer_evidence_md"]).write_text(
        evidence,
        encoding="utf-8",
    )
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
        summary = run_property_quarantine_materializer(
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
            output_dir=args.output_dir,
            quarantine_run_id=args.quarantine_run_id,
            created_by=args.created_by,
            confirm_quarantine_materialization=args.confirm_quarantine_materialization,
            allow_execution_preflight_needs_review=args.allow_execution_preflight_needs_review,
        )
    except Exception as exc:
        err.write(f"property quarantine materializer invalid: {_safe_exception_message(exc)}\n")
        return 1

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("materializer_status") == "failed" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_quarantine_materializer",
        description="Write candidate-only property quarantine materialization artifacts from a passed execution preflight.",
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quarantine-run-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-quarantine-materialization", action="store_true")
    parser.add_argument("--allow-execution-preflight-needs-review", action="store_true")
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
    preflight: dict[str, Any],
    offline_planner: dict[str, Any],
    property_planner: dict[str, Any],
    materialization_dry_run: dict[str, Any],
    execution_request: dict[str, Any],
    execution_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    hashes: dict[str, str],
    allow_execution_preflight_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if execution_preflight.get("schema_version") != _EXECUTION_PREFLIGHT_SCHEMA_VERSION:
        errors.append("execution_preflight_schema_invalid")
    if formal.get("schema_version") != _FORMAL_SCHEMA_VERSION:
        errors.append("formal_package_validation_schema_invalid")
    if binding.get("schema_version") != _PROPERTY_BINDING_SCHEMA_VERSION:
        errors.append("property_package_binding_schema_invalid")
    if preflight.get("schema_version") != _PREFLIGHT_SCHEMA_VERSION:
        errors.append("preflight_schema_invalid")
    if offline_planner.get("schema_version") != _OFFLINE_PLANNER_SCHEMA_VERSION:
        errors.append("offline_planner_schema_invalid")
    if property_planner.get("schema_version") != _PROPERTY_PLANNER_SCHEMA_VERSION:
        errors.append("property_planner_schema_invalid")
    if materialization_dry_run.get("schema_version") != _MATERIALIZATION_DRY_RUN_SCHEMA_VERSION:
        errors.append("materialization_dry_run_report_schema_invalid")
    if execution_request.get("schema_version") != _REQUEST_SCHEMA_VERSION:
        errors.append("execution_request_schema_invalid")
    if execution_summary.get("schema_version") != _REQUEST_SUMMARY_SCHEMA_VERSION:
        errors.append("execution_request_summary_schema_invalid")
    if plan_payload.get("schema_version") != _MATERIALIZATION_SCHEMA_VERSION:
        errors.append("materialization_plan_schema_invalid")
    if plan is None and plan_payload.get("schema_version") == _MATERIALIZATION_SCHEMA_VERSION:
        errors.append("materialization_plan_invalid")

    preflight_status = execution_preflight.get("preflight_status")
    if preflight_status == "failed":
        errors.append("execution_preflight_failed")
    if preflight_status == "needs_review":
        if allow_execution_preflight_needs_review:
            warnings.append("execution_preflight_needs_review_allowed")
        else:
            errors.append("execution_preflight_needs_review")
    if preflight_status not in {"passed", "needs_review", "failed"}:
        errors.append("execution_preflight_status_invalid")
    if execution_preflight.get("preflight_errors"):
        errors.append("execution_preflight_has_errors")
    if execution_preflight.get("request_status") != "written" or execution_request.get("request_status") != "written":
        errors.append("execution_request_not_written")
    if execution_request.get("execution_mode") != "request_only":
        errors.append("execution_mode_invalid")
    if execution_request.get("materializer_status") != "not_run" or execution_preflight.get("materializer_status") != "not_run":
        errors.append("materializer_status_not_run")
    if execution_request.get("phase1_status") != "not_run" or execution_preflight.get("phase1_status") != "not_run":
        errors.append("phase1_ran")
    if execution_request.get("training_admitted") is not False or execution_preflight.get("training_admitted") is not False:
        errors.append("training_admitted")
    if execution_request.get("dataset_confirmation_changed") is not False or execution_preflight.get("dataset_confirmation_changed") is not False:
        errors.append("dataset_confirmation_changed")
    if materialization_dry_run.get("dry_run_status") == "failed" or execution_preflight.get("dry_run_status") == "failed":
        errors.append("materialization_dry_run_failed")
    if materialization_dry_run.get("planner_status") == "failed" or execution_preflight.get("planner_status") == "failed":
        errors.append("planner_summary_failed")
    if materialization_dry_run.get("offline_planner_status") != "planned" or execution_preflight.get("offline_planner_status") != "planned":
        errors.append("offline_planner_failed")
    if materialization_dry_run.get("materialization_decision") != "planned" or execution_preflight.get("materialization_decision") != "planned":
        errors.append("materialization_decision_not_planned")

    _check_preflight_hashes(errors, execution_preflight, hashes)
    if execution_preflight.get("quarantine_candidate_records_sha256"):
        errors.append("execution_preflight_claimed_quarantine_artifact")

    id_sets = (
        (manifest.corpus_id, execution_preflight.get("corpus_id"), execution_request.get("corpus_id"), execution_summary.get("corpus_id"), materialization_dry_run.get("corpus_id"), property_planner.get("corpus_id"), preflight.get("corpus_id"), binding.get("corpus_id"), formal.get("corpus_id"), plan_payload.get("corpus_id")),
        (source_dry_run.run_id, execution_preflight.get("source_dry_run_id"), execution_request.get("source_dry_run_id"), execution_summary.get("source_dry_run_id"), materialization_dry_run.get("corpus_dry_run_id"), property_planner.get("dry_run_id"), preflight.get("dry_run_id"), binding.get("dry_run_id"), formal.get("dry_run_id"), plan_payload.get("dry_run_id")),
        (review.review_manifest_id, execution_preflight.get("review_manifest_id"), execution_request.get("review_manifest_id"), execution_summary.get("review_manifest_id"), materialization_dry_run.get("review_manifest_id"), property_planner.get("review_manifest_id"), preflight.get("review_manifest_id"), binding.get("review_manifest_id"), formal.get("review_manifest_id"), plan_payload.get("review_manifest_id")),
        (admission.admission_request_id, execution_preflight.get("admission_request_id"), execution_request.get("admission_request_id"), execution_summary.get("admission_request_id"), materialization_dry_run.get("admission_request_id"), property_planner.get("admission_request_id"), preflight.get("admission_request_id"), binding.get("admission_request_id"), formal.get("admission_request_id"), plan_payload.get("admission_request_id")),
        (plan_payload.get("materialization_plan_id"), execution_preflight.get("materialization_plan_id"), execution_request.get("materialization_plan_id"), execution_summary.get("materialization_plan_id"), materialization_dry_run.get("materialization_plan_id"), property_planner.get("materialization_plan_id"), preflight.get("materialization_plan_id"), offline_planner.get("materialization_plan_id")),
        (execution_preflight.get("execution_request_id"), execution_request.get("execution_request_id"), execution_summary.get("execution_request_id")),
    )
    error_codes = (
        "corpus_id_mismatch",
        "dry_run_id_mismatch",
        "review_manifest_id_mismatch",
        "admission_request_id_mismatch",
        "materialization_plan_id_mismatch",
        "execution_request_id_mismatch",
    )
    for values, error in zip(id_sets, error_codes, strict=True):
        if len({str(value or "") for value in values}) != 1:
            errors.append(error)

    materialization_records = _materialize_records(plan_payload)
    execution_records = _raw_execution_records(execution_request)
    materialization_record_ids = [str(record.get("materialization_record_id", "")) for record in materialization_records]
    execution_record_ids = [str(record.get("execution_record_id", "")) for record in execution_records]
    if not materialization_records:
        errors.append("no_materialization_records")
    if not execution_records:
        errors.append("no_execution_records")
    if int(execution_preflight.get("materialization_record_count", -1)) != len(materialization_records):
        errors.append("materialization_record_count_mismatch")
    if int(execution_preflight.get("execution_record_count", -1)) != len(execution_records):
        errors.append("execution_record_count_mismatch")
    if _safe_list(execution_preflight.get("materialization_record_ids")) != materialization_record_ids:
        errors.append("materialization_record_ids_mismatch")
    if _safe_list(execution_preflight.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    if _safe_list(execution_summary.get("materialization_record_ids")) != materialization_record_ids:
        errors.append("materialization_record_ids_mismatch")
    if _safe_list(execution_summary.get("execution_record_ids")) != execution_record_ids:
        errors.append("execution_record_ids_mismatch")
    errors.extend(_candidate_record_source_errors(_candidate_records_from_inputs(materialization_records, execution_records), admission, binding))
    return _stable_unique(errors)


def _candidate_artifact(
    *,
    materializer_status: str,
    quarantine_run_id: str,
    created_by: str,
    hashes: dict[str, str],
    manifest: CustomCorpusManifest,
    source_dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": _CANDIDATE_SCHEMA_VERSION,
        "quarantine_run_id": quarantine_run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "created_by": created_by,
        "materializer_status": materializer_status,
        "materialization_mode": "candidate_quarantine",
        "training_admitted": False,
        "phase1_status": "not_run",
        "dataset_confirmation_changed": False,
        "corpus_id": manifest.corpus_id,
        "source_dry_run_id": source_dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "materialization_plan_id": str(plan_payload.get("materialization_plan_id", "")),
        "execution_request_id": str(candidate_records[0].get("execution_record_id", "")).split("-property-materialization-plan", 1)[0] if candidate_records else "",
        "review_queue_id": str(binding.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(binding.get("property_candidate_manifest_id", "")),
        "dataset_target": str(plan_payload.get("dataset_target", "")),
        "source_manifest_sha256": hashes["manifest_sha256"],
        "source_dry_run_report_sha256": hashes["dry_run_report_sha256"],
        "source_review_manifest_sha256": hashes["review_manifest_sha256"],
        "source_admission_request_sha256": hashes["admission_request_sha256"],
        "source_formal_package_validation_sha256": hashes["formal_package_validation_sha256"],
        "source_property_package_binding_summary_sha256": hashes["property_package_binding_summary_sha256"],
        "source_materialization_plan_sha256": hashes["materialization_plan_sha256"],
        "source_materialization_preflight_summary_sha256": hashes["materialization_plan_preflight_summary_sha256"],
        "source_offline_planner_output_sha256": hashes["offline_planner_output_sha256"],
        "source_property_planner_summary_sha256": hashes["property_planner_summary_sha256"],
        "source_materialization_dry_run_report_sha256": hashes["materialization_dry_run_report_sha256"],
        "source_execution_request_sha256": hashes["execution_request_sha256"],
        "source_execution_request_summary_sha256": hashes["execution_request_summary_sha256"],
        "source_execution_preflight_summary_sha256": hashes["execution_preflight_summary_sha256"],
        "candidate_record_count": len(candidate_records),
        "candidate_record_ids": [record["candidate_record_id"] for record in candidate_records],
        "candidate_records": candidate_records,
        "boundary_statement": "candidate quarantine only; no training data admission, no Phase 1, and DatasetConfirmation unchanged.",
    }


def _summary(
    *,
    materializer_status: str,
    quarantine_run_id: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    quarantine_candidate_sha256: str,
    manifest: CustomCorpusManifest,
    source_dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    formal: dict[str, Any],
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    preflight: dict[str, Any],
    offline_planner: dict[str, Any],
    property_planner: dict[str, Any],
    materialization_dry_run: dict[str, Any],
    execution_request: dict[str, Any],
    execution_summary: dict[str, Any],
    execution_preflight: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    materialization_records = _materialize_records(plan_payload)
    execution_records = _raw_execution_records(execution_request)
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "materializer_status": materializer_status,
        "quarantine_run_id": quarantine_run_id,
        "quarantine_candidate_records_path": _ARTIFACTS["quarantine_candidate_records_json"] if materializer_status in {"written", "needs_review"} else "",
        "quarantine_candidate_records_sha256": quarantine_candidate_sha256,
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
        "corpus_id": manifest.corpus_id,
        "source_dry_run_id": source_dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "materialization_plan_id": str(plan_payload.get("materialization_plan_id", "")),
        "execution_request_id": str(execution_request.get("execution_request_id", "")),
        "review_queue_id": str(binding.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(binding.get("property_candidate_manifest_id", "")),
        "dataset_target": str(plan_payload.get("dataset_target", property_planner.get("dataset_target", ""))),
        "execution_preflight_status": str(execution_preflight.get("preflight_status", "")),
        "dry_run_status": str(materialization_dry_run.get("dry_run_status", execution_preflight.get("dry_run_status", ""))),
        "planner_status": str(property_planner.get("planner_status", execution_preflight.get("planner_status", ""))),
        "offline_planner_status": str(offline_planner.get("planner_status", execution_preflight.get("offline_planner_status", ""))),
        "package_binding_status": str(binding.get("binding_status", execution_preflight.get("package_binding_status", ""))),
        "formal_package_validation_status": str(formal.get("validation_status", execution_preflight.get("formal_package_validation_status", ""))),
        "materialization_decision": str(plan_payload.get("materialization_decision", execution_preflight.get("materialization_decision", ""))),
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
        "candidate_record_ids": [record["candidate_record_id"] for record in candidate_records],
        "materialization_record_ids": [str(record.get("materialization_record_id", "")) for record in materialization_records],
        "execution_record_ids": [str(record.get("execution_record_id", "")) for record in execution_records],
        "admit_record_ids": [record.record_id for record in admission.admission_records if record.action == "admit"],
        "exclude_record_ids": [record.record_id for record in admission.admission_records if record.action == "exclude"],
        "blocked_record_ids": _safe_list(binding.get("blocked_record_ids")),
        "materializer_errors": _stable_unique(errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _candidate_records(
    *,
    quarantine_run_id: str,
    plan_payload: dict[str, Any],
    execution_request: dict[str, Any],
    hashes: dict[str, str],
) -> list[dict[str, Any]]:
    return _candidate_records_from_inputs(
        _materialize_records(plan_payload),
        _raw_execution_records(execution_request),
        quarantine_run_id=quarantine_run_id,
        hashes=hashes,
    )


def _candidate_records_from_inputs(
    materialization_records: list[dict[str, Any]],
    execution_records: list[dict[str, Any]],
    *,
    quarantine_run_id: str = "",
    hashes: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    materialization_by_id = {str(record.get("materialization_record_id", "")): record for record in materialization_records}
    records: list[dict[str, Any]] = []
    for execution in execution_records:
        materialization_id = str(execution.get("materialization_record_id", ""))
        materialization = materialization_by_id.get(materialization_id, {})
        if not materialization:
            continue
        record = {
            "candidate_record_id": f"{quarantine_run_id}-{materialization_id}" if quarantine_run_id else materialization_id,
            "quarantine_run_id": quarantine_run_id,
            "execution_record_id": str(execution.get("execution_record_id", "")),
            "materialization_record_id": materialization_id,
            "record_id": str(execution.get("record_id", "")),
            "admission_record_id": str(execution.get("admission_record_id", "")),
            "review_id": str(execution.get("review_id", "")),
            "document_id": str(execution.get("document_id", "")),
            "field_name": str(execution.get("field_name", "")),
            "candidate_status": "quarantined",
            "source_artifact_sha256": str(execution.get("source_artifact_sha256", "")),
            "review_artifact_sha256": str(execution.get("review_artifact_sha256", "")),
            "admission_request_sha256": str(execution.get("admission_request_sha256", "")),
            "package_validation_sha256": str(execution.get("package_validation_sha256", "")),
            "materialization_plan_sha256": str(execution.get("materialization_plan_sha256", "")),
            "offline_planner_output_sha256": str(execution.get("offline_planner_output_sha256", "")),
            "materialization_dry_run_report_sha256": str(execution.get("dry_run_report_sha256", "")),
            "execution_request_sha256": hashes["execution_request_sha256"] if hashes else "",
            "execution_preflight_summary_sha256": hashes["execution_preflight_summary_sha256"] if hashes else "",
            "normalized_value_summary": str(materialization.get("normalized_value_summary", "")),
            "provenance_summary": str(materialization.get("provenance_summary", "")),
            "materialization_boundary": [
                "candidate_only",
                "not_training",
                "not_phase1",
                "dataset_confirmation_unchanged",
            ],
        }
        records.append(record)
    return records


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


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Quarantine Materializer Evidence",
            "",
            f"- Materializer status: `{summary['materializer_status']}`",
            f"- Quarantine run id: `{summary['quarantine_run_id']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Source dry-run id: `{summary['source_dry_run_id']}`",
            f"- Review manifest id: `{summary['review_manifest_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Execution request id: `{summary['execution_request_id']}`",
            f"- Execution preflight status: `{summary['execution_preflight_status']}`",
            f"- Candidate records: `{summary['candidate_record_count']}`",
            f"- Materialization records: `{summary['materialization_record_count']}`",
            f"- Execution records: `{summary['execution_record_count']}`",
            f"- Candidate record ids: `{json.dumps(summary['candidate_record_ids'])}`",
            f"- Materialization record ids: `{json.dumps(summary['materialization_record_ids'])}`",
            f"- Execution record ids: `{json.dumps(summary['execution_record_ids'])}`",
            f"- Admit record ids: `{json.dumps(summary['admit_record_ids'])}`",
            f"- Exclude record ids: `{json.dumps(summary['exclude_record_ids'])}`",
            f"- Blocked record ids: `{json.dumps(summary['blocked_record_ids'])}`",
            f"- Materializer errors: `{json.dumps(summary['materializer_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is candidate quarantine materialization only.",
            "- No training data was admitted.",
            "- No training CSV/JSONL/Parquet/LMDB was created.",
            "- No Phase 1 was run.",
            "- DatasetConfirmation was not changed.",
            "- No model training or evaluation was run.",
            "",
        ]
    )


def _check_preflight_hashes(errors: list[str], preflight: dict[str, Any], hashes: dict[str, str]) -> None:
    pairs = (
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
    )
    for field, hash_field, error in pairs:
        if preflight.get(field) and preflight.get(field) != hashes[hash_field]:
            errors.append(error)


def _load_source_dry_run_report(path: str | Path) -> CustomCorpusDryRunReport:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return CustomCorpusDryRunReport.model_validate(payload)
    except Exception as exc:
        raise CustomCorpusPropertyQuarantineMaterializerError("dry-run report invalid") from exc


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
        raise CustomCorpusPropertyQuarantineMaterializerError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyQuarantineMaterializerError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyQuarantineMaterializerError(f"{label} invalid")
    return payload


def _materialize_records(plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = plan_payload.get("materialization_records", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict) and record.get("action") == "materialize_candidate"]


def _raw_execution_records(request: dict[str, Any]) -> list[dict[str, Any]]:
    records = request.get("execution_records", [])
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


def _required_safe_id(value: str, field_name: str) -> str:
    if not value or not _SAFE_ID_RE.fullmatch(value):
        raise CustomCorpusPropertyQuarantineMaterializerError(f"{field_name} invalid")
    return value


def _required_safe_label(value: str, field_name: str) -> str:
    if not value or len(value) > 200 or _input_contains_forbidden_material(value):
        raise CustomCorpusPropertyQuarantineMaterializerError(f"{field_name} invalid")
    if "@" in value and "redacted" not in value.lower():
        raise CustomCorpusPropertyQuarantineMaterializerError(f"{field_name} invalid")
    return value


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
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "materializer_status": "failed",
        "materializer_errors": ["property_quarantine_materializer_redaction_failed"],
        "redaction_status": "failed",
    }


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


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
