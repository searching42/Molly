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
from ai4s_agent.custom_corpus_property_materialization_dry_run import (
    _local_consistency_errors as _dry_run_local_consistency_errors,
)
from ai4s_agent.custom_corpus_review import ReviewManifest, load_review_manifest


_REQUEST_SCHEMA_VERSION = "custom_corpus_property_materializer_execution_request.v1"
_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_materializer_execution_request_builder.v1"
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
    "property_materializer_execution_request_json": "property_materializer_execution_request.json",
    "property_materializer_execution_request_summary_json": "property_materializer_execution_request_summary.json",
    "redacted_property_materializer_execution_request_evidence_md": "redacted_property_materializer_execution_request_evidence.md",
}


class CustomCorpusPropertyMaterializerExecutionRequestError(ValueError):
    pass


def build_property_materializer_execution_request(
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
    output_dir: str | Path,
    execution_request_id: str,
    created_by: str,
    confirm_materializer_execution_request_output: bool,
    allow_dry_run_needs_review: bool = False,
) -> dict[str, Any]:
    execution_request_id = _required_safe_id(execution_request_id, "execution_request_id")
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
    }
    run_dir = Path(output_dir).expanduser() / execution_request_id
    request_errors: list[str] = []
    warnings: list[str] = []
    if not confirm_materializer_execution_request_output:
        request_errors.append("materializer_execution_request_not_confirmed")
    if run_dir.exists() and any(run_dir.iterdir()):
        request_errors.append("output_directory_not_clean")

    _append_errors(
        request_errors,
        _dry_run_local_consistency_errors(
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
            hashes={key: hashes[key] for key in hashes if key != "materialization_dry_run_report_sha256"},
            allow_planner_needs_review=allow_dry_run_needs_review,
            warnings=warnings,
        ),
    )
    _append_errors(
        request_errors,
        _materialization_dry_run_errors(
            materialization_dry_run=materialization_dry_run,
            manifest=manifest,
            source_dry_run=source_dry_run,
            review=review,
            admission=admission,
            binding=binding,
            plan_payload=plan_payload,
            preflight=preflight,
            offline_planner=offline_planner,
            property_planner=property_planner,
            hashes=hashes,
            allow_dry_run_needs_review=allow_dry_run_needs_review,
            warnings=warnings,
        ),
    )
    execution_records = _execution_records(
        execution_request_id=execution_request_id,
        plan_payload=plan_payload,
        hashes=hashes,
    )
    if not execution_records:
        request_errors.append("no_execution_records")
    _append_errors(request_errors, _execution_record_source_errors(execution_records, admission, binding))

    if request_errors:
        summary = _summary(
            request_status="blocked",
            execution_request_id=execution_request_id,
            paths=paths,
            hashes=hashes,
            request_sha256="",
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
            execution_records=execution_records,
            request_errors=_stable_unique(request_errors),
            warnings=_stable_unique(warnings),
        )
        return _minimal_redaction_failure() if _contains_forbidden_material(summary) else summary

    request = _execution_request(
        execution_request_id=execution_request_id,
        created_by=created_by,
        hashes=hashes,
        manifest=manifest,
        source_dry_run=source_dry_run,
        review=review,
        admission=admission,
        binding=binding,
        plan_payload=plan_payload,
        execution_records=execution_records,
    )
    summary = _summary(
        request_status="written",
        execution_request_id=execution_request_id,
        paths=paths,
        hashes=hashes,
        request_sha256="",
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
        execution_records=execution_records,
        request_errors=[],
        warnings=_stable_unique(warnings),
    )
    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"request": request, "summary": summary, "evidence": evidence}):
        return _minimal_redaction_failure()

    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / _ARTIFACTS["property_materializer_execution_request_json"]
    write_json(request_path, request)
    request_sha256 = _safe_sha_for_path(request_path)
    summary = _summary(
        request_status="written",
        execution_request_id=execution_request_id,
        paths=paths,
        hashes=hashes,
        request_sha256=request_sha256,
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
        execution_records=execution_records,
        request_errors=[],
        warnings=_stable_unique(warnings),
    )
    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "evidence": evidence}):
        _safe_unlink(request_path)
        return _minimal_redaction_failure()
    write_json(run_dir / _ARTIFACTS["property_materializer_execution_request_summary_json"], summary)
    (run_dir / _ARTIFACTS["redacted_property_materializer_execution_request_evidence_md"]).write_text(
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
        summary = build_property_materializer_execution_request(
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
            output_dir=args.output_dir,
            execution_request_id=args.execution_request_id,
            created_by=args.created_by,
            confirm_materializer_execution_request_output=args.confirm_materializer_execution_request_output,
            allow_dry_run_needs_review=args.allow_dry_run_needs_review,
        )
    except Exception as exc:
        err.write(f"property materializer execution request invalid: {_safe_exception_message(exc)}\n")
        return 1

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 0 if summary.get("request_status") == "written" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_materializer_execution_request",
        description="Build a request-only property materializer execution packet from a passed dry-run.",
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execution-request-id", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-materializer-execution-request-output", action="store_true")
    parser.add_argument("--allow-dry-run-needs-review", action="store_true")
    return parser


def _materialization_dry_run_errors(
    *,
    materialization_dry_run: dict[str, Any],
    manifest: CustomCorpusManifest,
    source_dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    preflight: dict[str, Any],
    offline_planner: dict[str, Any],
    property_planner: dict[str, Any],
    hashes: dict[str, str],
    allow_dry_run_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if materialization_dry_run.get("schema_version") != _MATERIALIZATION_DRY_RUN_SCHEMA_VERSION:
        errors.append("materialization_dry_run_report_schema_invalid")
    dry_run_status = materialization_dry_run.get("dry_run_status")
    if dry_run_status == "failed":
        errors.append("materialization_dry_run_failed")
    if dry_run_status == "needs_review":
        if allow_dry_run_needs_review:
            warnings.append("materialization_dry_run_needs_review_allowed")
        else:
            errors.append("materialization_dry_run_needs_review")
    if dry_run_status not in {"passed", "needs_review", "failed"}:
        errors.append("materialization_dry_run_status_invalid")
    if materialization_dry_run.get("dry_run_errors"):
        errors.append("materialization_dry_run_has_errors")
    if materialization_dry_run.get("planner_status") == "failed":
        errors.append("planner_summary_failed")
    if materialization_dry_run.get("planner_status") == "needs_review" and not allow_dry_run_needs_review:
        errors.append("planner_summary_needs_review")
    if materialization_dry_run.get("offline_planner_status") != "planned":
        errors.append("offline_planner_failed")
    if materialization_dry_run.get("preflight_status") == "failed":
        errors.append("preflight_failed")
    if materialization_dry_run.get("preflight_status") == "needs_review" and not allow_dry_run_needs_review:
        errors.append("preflight_needs_review")
    if materialization_dry_run.get("package_binding_status") == "failed":
        errors.append("package_binding_failed")
    if materialization_dry_run.get("package_binding_status") == "needs_review" and not allow_dry_run_needs_review:
        errors.append("package_binding_needs_review")
    if materialization_dry_run.get("formal_package_validation_status") != "passed":
        errors.append("formal_package_validation_failed")
    if materialization_dry_run.get("materialization_decision") != "planned":
        errors.append("materialization_decision_not_planned")
    if materialization_dry_run.get("source_dry_run_decision") != "passed":
        errors.append("dry_run_not_passed")
    if materialization_dry_run.get("phase1_status") != "not_run":
        errors.append("dry_run_phase1_ran")
    if materialization_dry_run.get("training_admitted") is not False:
        errors.append("dry_run_training_admitted")

    _check_hashes(errors, materialization_dry_run, hashes)
    if materialization_dry_run.get("materialization_plan_preflight_summary_sha256") and materialization_dry_run.get("materialization_plan_preflight_summary_sha256") != hashes["materialization_plan_preflight_summary_sha256"]:
        errors.append("materialization_plan_preflight_summary_sha256_mismatch")
    if materialization_dry_run.get("offline_planner_output_sha256") and materialization_dry_run.get("offline_planner_output_sha256") != hashes["offline_planner_output_sha256"]:
        errors.append("offline_planner_output_sha256_mismatch")
    if materialization_dry_run.get("property_planner_summary_sha256") and materialization_dry_run.get("property_planner_summary_sha256") != hashes["property_planner_summary_sha256"]:
        errors.append("property_planner_summary_sha256_mismatch")

    if materialization_dry_run.get("corpus_id") != manifest.corpus_id:
        errors.append("corpus_id_mismatch")
    if materialization_dry_run.get("corpus_dry_run_id") != source_dry_run.run_id:
        errors.append("dry_run_id_mismatch")
    if materialization_dry_run.get("review_manifest_id") != review.review_manifest_id:
        errors.append("review_manifest_id_mismatch")
    if materialization_dry_run.get("admission_request_id") != admission.admission_request_id:
        errors.append("admission_request_id_mismatch")
    if materialization_dry_run.get("materialization_plan_id") != plan_payload.get("materialization_plan_id"):
        errors.append("materialization_plan_id_mismatch")
    if materialization_dry_run.get("review_queue_id") != binding.get("review_queue_id"):
        errors.append("review_queue_id_mismatch")
    if materialization_dry_run.get("property_candidate_manifest_id") != binding.get("property_candidate_manifest_id"):
        errors.append("property_candidate_manifest_id_mismatch")
    if materialization_dry_run.get("planner_status") != property_planner.get("planner_status"):
        errors.append("planner_status_mismatch")
    if materialization_dry_run.get("offline_planner_status") != offline_planner.get("planner_status"):
        errors.append("offline_planner_status_mismatch")
    if materialization_dry_run.get("preflight_status") != preflight.get("preflight_status"):
        errors.append("preflight_status_mismatch")

    materialization_records = _raw_materialization_records(plan_payload)
    materialization_record_ids = [str(record.get("materialization_record_id", "")) for record in materialization_records]
    if int(materialization_dry_run.get("materialization_record_count", -1)) != len(materialization_records):
        errors.append("materialization_record_count_mismatch")
    if _safe_list(materialization_dry_run.get("materialization_record_ids")) != materialization_record_ids:
        errors.append("materialization_record_ids_mismatch")
    if _safe_list(materialization_dry_run.get("dry_run_record_summaries")):
        dry_run_ids = [
            str(record.get("materialization_record_id", ""))
            for record in materialization_dry_run.get("dry_run_record_summaries", [])
            if isinstance(record, dict)
        ]
        if dry_run_ids != materialization_record_ids:
            errors.append("materialization_record_ids_mismatch")
    if _safe_list(materialization_dry_run.get("admit_record_ids")) != [record.record_id for record in admission.admission_records if record.action == "admit"]:
        errors.append("admit_record_ids_mismatch")
    if _safe_list(materialization_dry_run.get("exclude_record_ids")) != [record.record_id for record in admission.admission_records if record.action == "exclude"]:
        errors.append("exclude_record_ids_mismatch")
    if _safe_list(materialization_dry_run.get("blocked_record_ids")) != _safe_list(binding.get("blocked_record_ids")):
        errors.append("blocked_record_ids_mismatch")
    return _stable_unique(errors)


def _execution_record_source_errors(
    execution_records: list[dict[str, Any]],
    admission: AdmissionRequest,
    binding: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    admit_ids = [record.record_id for record in admission.admission_records if record.action == "admit" and record.review_decision == "accept"]
    exclude_ids = [record.record_id for record in admission.admission_records if record.action == "exclude"]
    needs_review_ids = [record.record_id for record in admission.admission_records if record.action == "needs_review" or record.review_decision == "needs_review"]
    blocked_ids = _safe_list(binding.get("blocked_record_ids"))
    for record in execution_records:
        record_id = str(record.get("record_id", ""))
        if record_id in exclude_ids:
            errors.append("execution_record_from_excluded_record")
        if record_id in blocked_ids:
            errors.append("execution_record_from_blocked_record")
        if record_id in needs_review_ids:
            errors.append("execution_record_from_needs_review_record")
        if record_id not in admit_ids:
            errors.append("execution_record_not_admitted")
    return _stable_unique(errors)


def _execution_request(
    *,
    execution_request_id: str,
    created_by: str,
    hashes: dict[str, str],
    manifest: CustomCorpusManifest,
    source_dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    execution_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": _REQUEST_SCHEMA_VERSION,
        "execution_request_id": execution_request_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "created_by": created_by,
        "request_status": "written",
        "corpus_id": manifest.corpus_id,
        "source_dry_run_id": source_dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "materialization_plan_id": str(plan_payload.get("materialization_plan_id", "")),
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
        "execution_mode": "request_only",
        "materializer_status": "not_run",
        "phase1_status": "not_run",
        "training_admitted": False,
        "dataset_confirmation_changed": False,
        "record_count": len(execution_records),
        "execution_record_ids": [record["execution_record_id"] for record in execution_records],
        "execution_records": execution_records,
        "boundary_statement": "this is a materializer execution request only; no materializer was run, no materialization was executed, no candidate or training artifacts were created, Phase 1 did not run, and DatasetConfirmation was not changed.",
    }


def _summary(
    *,
    request_status: str,
    execution_request_id: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    request_sha256: str,
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
    execution_records: list[dict[str, Any]],
    request_errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    materialization_records = _raw_materialization_records(plan_payload)
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "request_status": request_status,
        "execution_request_id": execution_request_id,
        "execution_request_path": _ARTIFACTS["property_materializer_execution_request_json"] if request_status == "written" else "",
        "execution_request_sha256": request_sha256,
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
        "corpus_id": manifest.corpus_id,
        "source_dry_run_id": source_dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "materialization_plan_id": str(plan_payload.get("materialization_plan_id", "")),
        "review_queue_id": str(binding.get("review_queue_id", "")),
        "property_candidate_manifest_id": str(binding.get("property_candidate_manifest_id", "")),
        "dataset_target": str(plan_payload.get("dataset_target", property_planner.get("dataset_target", ""))),
        "dry_run_status": str(materialization_dry_run.get("dry_run_status", "")),
        "planner_status": str(property_planner.get("planner_status", materialization_dry_run.get("planner_status", ""))),
        "offline_planner_status": str(offline_planner.get("planner_status", materialization_dry_run.get("offline_planner_status", ""))),
        "preflight_status": str(preflight.get("preflight_status", materialization_dry_run.get("preflight_status", ""))),
        "package_binding_status": str(binding.get("binding_status", materialization_dry_run.get("package_binding_status", ""))),
        "formal_package_validation_status": str(formal.get("validation_status", materialization_dry_run.get("formal_package_validation_status", ""))),
        "materialization_decision": str(plan_payload.get("materialization_decision", materialization_dry_run.get("materialization_decision", ""))),
        "phase1_status": source_dry_run.confirmation_boundary.phase1_status,
        "training_admitted": source_dry_run.confirmation_boundary.training_dataset_admitted,
        "admission_record_count": len(admission.admission_records),
        "admit_count": len([record for record in admission.admission_records if record.action == "admit"]),
        "exclude_count": len([record for record in admission.admission_records if record.action == "exclude"]),
        "blocked_record_count": len(_safe_list(binding.get("blocked_record_ids"))),
        "materialization_record_count": len(materialization_records),
        "execution_record_count": len(execution_records),
        "materialization_record_ids": [str(record.get("materialization_record_id", "")) for record in materialization_records],
        "execution_record_ids": [record["execution_record_id"] for record in execution_records],
        "admit_record_ids": [record.record_id for record in admission.admission_records if record.action == "admit"],
        "exclude_record_ids": [record.record_id for record in admission.admission_records if record.action == "exclude"],
        "blocked_record_ids": _safe_list(binding.get("blocked_record_ids")),
        "request_errors": _stable_unique(request_errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _execution_records(
    *,
    execution_request_id: str,
    plan_payload: dict[str, Any],
    hashes: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _raw_materialization_records(plan_payload):
        materialization_record_id = str(record.get("materialization_record_id", ""))
        if record.get("action") != "materialize_candidate":
            continue
        records.append(
            {
                "execution_record_id": f"{execution_request_id}-{materialization_record_id}",
                "materialization_record_id": materialization_record_id,
                "record_id": str(record.get("record_id", "")),
                "admission_record_id": str(record.get("admission_record_id", "")),
                "review_id": str(record.get("review_id", "")),
                "document_id": str(record.get("document_id", "")),
                "field_name": str(record.get("field_name", "")),
                "planned_action": "request_materialize_candidate",
                "source_artifact_sha256": str(record.get("source_artifact_sha256", "")),
                "review_artifact_sha256": str(record.get("review_artifact_sha256", "")),
                "admission_request_sha256": str(record.get("admission_request_sha256", hashes["admission_request_sha256"])),
                "package_validation_sha256": str(record.get("package_validation_sha256", hashes["formal_package_validation_sha256"])),
                "materialization_plan_sha256": hashes["materialization_plan_sha256"],
                "offline_planner_output_sha256": hashes["offline_planner_output_sha256"],
                "dry_run_report_sha256": hashes["materialization_dry_run_report_sha256"],
            }
        )
    return records


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Materializer Execution Request Evidence",
            "",
            f"- Request status: `{summary['request_status']}`",
            f"- Execution request id: `{summary['execution_request_id']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Source dry-run id: `{summary['source_dry_run_id']}`",
            f"- Review manifest id: `{summary['review_manifest_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Dry-run status: `{summary['dry_run_status']}`",
            f"- Planner status: `{summary['planner_status']}`",
            f"- Offline planner status: `{summary['offline_planner_status']}`",
            f"- Preflight status: `{summary['preflight_status']}`",
            f"- Package binding status: `{summary['package_binding_status']}`",
            f"- Formal package validation status: `{summary['formal_package_validation_status']}`",
            f"- Materialization records: `{summary['materialization_record_count']}`",
            f"- Execution records: `{summary['execution_record_count']}`",
            f"- Materialization record ids: `{json.dumps(summary['materialization_record_ids'])}`",
            f"- Execution record ids: `{json.dumps(summary['execution_record_ids'])}`",
            f"- Admit record ids: `{json.dumps(summary['admit_record_ids'])}`",
            f"- Exclude record ids: `{json.dumps(summary['exclude_record_ids'])}`",
            f"- Blocked record ids: `{json.dumps(summary['blocked_record_ids'])}`",
            f"- Request errors: `{json.dumps(summary['request_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a materializer execution request only.",
            "- No real materializer was run.",
            "- No materialization was executed.",
            "- No materialized records were created.",
            "- No candidate/training CSV was created.",
            "- No candidate/training JSONL/Parquet/LMDB was created.",
            "- No training data was admitted.",
            "- Phase 1 did not run.",
            "- DatasetConfirmation was not changed.",
            "",
        ]
    )


def _load_source_dry_run_report(path: str | Path) -> CustomCorpusDryRunReport:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return CustomCorpusDryRunReport.model_validate(payload)
    except Exception as exc:
        raise CustomCorpusPropertyMaterializerExecutionRequestError("dry-run report invalid") from exc


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
        raise CustomCorpusPropertyMaterializerExecutionRequestError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyMaterializerExecutionRequestError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyMaterializerExecutionRequestError(f"{label} invalid")
    return payload


def _check_hashes(errors: list[str], payload: dict[str, Any], hashes: dict[str, str]) -> None:
    pairs = (
        ("manifest_sha256", "manifest_sha256", "manifest_sha256_mismatch"),
        ("dry_run_report_sha256", "dry_run_report_sha256", "dry_run_report_sha256_mismatch"),
        ("review_manifest_sha256", "review_manifest_sha256", "review_manifest_sha256_mismatch"),
        ("admission_request_sha256", "admission_request_sha256", "admission_request_sha256_mismatch"),
        ("formal_package_validation_sha256", "formal_package_validation_sha256", "formal_package_validation_sha256_mismatch"),
        (
            "property_package_binding_summary_sha256",
            "property_package_binding_summary_sha256",
            "property_package_binding_summary_sha256_mismatch",
        ),
        ("materialization_plan_sha256", "materialization_plan_sha256", "materialization_plan_sha256_mismatch"),
    )
    for field, hash_field, error in pairs:
        if payload.get(field) and payload.get(field) != hashes[hash_field]:
            errors.append(error)


def _required_safe_id(value: str, field_name: str) -> str:
    if not value or not _SAFE_ID_RE.fullmatch(value):
        raise CustomCorpusPropertyMaterializerExecutionRequestError(f"{field_name} invalid")
    return value


def _required_safe_label(value: str, field_name: str) -> str:
    if not value or len(value) > 200 or _input_contains_forbidden_material(value):
        raise CustomCorpusPropertyMaterializerExecutionRequestError(f"{field_name} invalid")
    if "@" in value and "redacted" not in value.lower():
        raise CustomCorpusPropertyMaterializerExecutionRequestError(f"{field_name} invalid")
    return value


def _raw_materialization_records(plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = plan_payload.get("materialization_records", [])
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
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "request_status": "blocked",
        "request_errors": ["property_materializer_execution_request_redaction_failed"],
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
