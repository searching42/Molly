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
from ai4s_agent.custom_corpus_materialization_planner import plan_materialization
from ai4s_agent.custom_corpus_review import ReviewManifest, load_review_manifest


_SCHEMA_VERSION = "custom_corpus_property_materialization_planner_runner.v1"
_PREFLIGHT_SCHEMA_VERSION = "custom_corpus_property_materialization_plan_preflight.v1"
_DRAFT_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_materialization_plan_draft_builder.v1"
_PROPERTY_BINDING_SCHEMA_VERSION = "custom_corpus_property_package_binding.v1"
_FORMAL_SCHEMA_VERSION = "custom_corpus_admission_package_validation.v1"
_MATERIALIZATION_SCHEMA_VERSION = "custom_corpus_materialization.v1"
_OFFLINE_PLANNER_SCHEMA_VERSION = "custom_corpus_materialization_planner.v1"
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
    "x-amz-signature",
    "signature=",
    "signedurl",
    "signed-url",
)
_ARTIFACTS = {
    "offline_materialization_planner_output_json": "offline_materialization_planner_output.json",
    "property_materialization_planner_summary_json": "property_materialization_planner_summary.json",
    "redacted_property_materialization_planner_evidence_md": "redacted_property_materialization_planner_evidence.md",
}


class CustomCorpusPropertyMaterializationPlannerRunnerError(ValueError):
    pass


def run_property_materialization_planner(
    *,
    manifest_path: str | Path,
    dry_run_report_path: str | Path,
    review_manifest_path: str | Path,
    admission_request_path: str | Path,
    formal_package_validation_path: str | Path,
    property_package_binding_summary_path: str | Path,
    materialization_plan_path: str | Path,
    materialization_plan_draft_summary_path: str | Path,
    materialization_plan_preflight_summary_path: str | Path,
    output_dir: str | Path,
    planner_run_id: str,
    confirm_offline_materialization_planner: bool,
    allow_preflight_needs_review: bool = False,
) -> dict[str, Any]:
    planner_run_id = _required_safe_id(planner_run_id, "planner_run_id")
    manifest = load_custom_corpus_manifest(manifest_path)
    dry_run = _load_dry_run_report(dry_run_report_path)
    review = load_review_manifest(review_manifest_path)
    admission = load_admission_request(admission_request_path)
    formal = _read_safe_json_dict(formal_package_validation_path, "formal package validation")
    binding = _read_safe_json_dict(property_package_binding_summary_path, "property package binding summary")
    plan_payload, plan = _load_materialization_plan_payload(materialization_plan_path)
    draft_summary = _read_safe_json_dict(materialization_plan_draft_summary_path, "materialization draft summary")
    preflight = _read_safe_json_dict(materialization_plan_preflight_summary_path, "materialization plan preflight summary")

    paths = {
        "manifest_path": manifest_path,
        "dry_run_report_path": dry_run_report_path,
        "review_manifest_path": review_manifest_path,
        "admission_request_path": admission_request_path,
        "formal_package_validation_path": formal_package_validation_path,
        "property_package_binding_summary_path": property_package_binding_summary_path,
        "materialization_plan_path": materialization_plan_path,
        "materialization_plan_draft_summary_path": materialization_plan_draft_summary_path,
        "materialization_plan_preflight_summary_path": materialization_plan_preflight_summary_path,
    }
    hashes = {
        "manifest_sha256": _safe_sha_for_path(manifest_path),
        "dry_run_report_sha256": _safe_sha_for_path(dry_run_report_path),
        "review_manifest_sha256": _safe_sha_for_path(review_manifest_path),
        "admission_request_sha256": _safe_sha_for_path(admission_request_path),
        "formal_package_validation_sha256": _safe_sha_for_path(formal_package_validation_path),
        "property_package_binding_summary_sha256": _safe_sha_for_path(property_package_binding_summary_path),
        "materialization_plan_sha256": _safe_sha_for_path(materialization_plan_path),
        "materialization_plan_draft_summary_sha256": _safe_sha_for_path(materialization_plan_draft_summary_path),
        "materialization_plan_preflight_summary_sha256": _safe_sha_for_path(materialization_plan_preflight_summary_path),
    }
    run_dir = Path(output_dir).expanduser() / planner_run_id

    planner_errors: list[str] = []
    warnings: list[str] = []
    if not confirm_offline_materialization_planner:
        planner_errors.append("offline_materialization_planner_not_confirmed")
    if run_dir.exists() and any(run_dir.iterdir()):
        planner_errors.append("output_directory_not_clean")
    _append_errors(
        planner_errors,
        _local_consistency_errors(
            manifest=manifest,
            dry_run=dry_run,
            review=review,
            admission=admission,
            formal=formal,
            binding=binding,
            plan_payload=plan_payload,
            plan=plan,
            draft_summary=draft_summary,
            preflight=preflight,
            hashes=hashes,
            allow_preflight_needs_review=allow_preflight_needs_review,
            warnings=warnings,
        ),
    )

    if planner_errors:
        summary = _summary(
            planner_status="failed",
            planner_run_id=planner_run_id,
            paths=paths,
            hashes=hashes,
            manifest=manifest,
            dry_run=dry_run,
            review=review,
            admission=admission,
            formal=formal,
            binding=binding,
            plan_payload=plan_payload,
            draft_summary=draft_summary,
            preflight=preflight,
            offline_planner_output_sha256="",
            planner_errors=_stable_unique(planner_errors),
            warnings=_stable_unique(warnings),
        )
        return _minimal_redaction_failure() if _contains_forbidden_material(summary) else summary

    planner_output = plan_materialization(materialization_plan_path)
    planner_output_errors = _offline_planner_output_errors(planner_output)
    if planner_output_errors:
        summary = _summary(
            planner_status="failed",
            planner_run_id=planner_run_id,
            paths=paths,
            hashes=hashes,
            manifest=manifest,
            dry_run=dry_run,
            review=review,
            admission=admission,
            formal=formal,
            binding=binding,
            plan_payload=plan_payload,
            draft_summary=draft_summary,
            preflight=preflight,
            offline_planner_output_sha256="",
            planner_errors=planner_output_errors,
            warnings=_stable_unique(warnings),
            offline_planner_status=str(planner_output.get("planner_status", "")),
        )
        return _minimal_redaction_failure() if _contains_forbidden_material(summary) else summary

    run_dir.mkdir(parents=True, exist_ok=True)
    offline_output_path = run_dir / _ARTIFACTS["offline_materialization_planner_output_json"]
    write_json(offline_output_path, planner_output)
    hashes["offline_planner_output_sha256"] = _safe_sha_for_path(offline_output_path)
    planner_status = "needs_review" if preflight.get("preflight_status") == "needs_review" else "planned"
    if preflight.get("preflight_status") == "needs_review":
        _append_unique(warnings, "preflight_needs_review_allowed")
    summary = _summary(
        planner_status=planner_status,
        planner_run_id=planner_run_id,
        paths=paths,
        hashes=hashes,
        manifest=manifest,
        dry_run=dry_run,
        review=review,
        admission=admission,
        formal=formal,
        binding=binding,
        plan_payload=plan_payload,
        draft_summary=draft_summary,
        preflight=preflight,
        offline_planner_output_sha256=hashes["offline_planner_output_sha256"],
        planner_errors=[],
        warnings=_stable_unique(warnings),
        offline_planner_status=str(planner_output.get("planner_status", "")),
    )
    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "evidence": evidence}):
        _safe_unlink(offline_output_path)
        return _minimal_redaction_failure()
    write_json(run_dir / _ARTIFACTS["property_materialization_planner_summary_json"], summary)
    (run_dir / _ARTIFACTS["redacted_property_materialization_planner_evidence_md"]).write_text(evidence, encoding="utf-8")
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
        summary = run_property_materialization_planner(
            manifest_path=args.manifest,
            dry_run_report_path=args.dry_run_report,
            review_manifest_path=args.review_manifest,
            admission_request_path=args.admission_request,
            formal_package_validation_path=args.formal_package_validation,
            property_package_binding_summary_path=args.property_package_binding_summary,
            materialization_plan_path=args.materialization_plan,
            materialization_plan_draft_summary_path=args.materialization_plan_draft_summary,
            materialization_plan_preflight_summary_path=args.materialization_plan_preflight_summary,
            output_dir=args.output_dir,
            planner_run_id=args.planner_run_id,
            confirm_offline_materialization_planner=args.confirm_offline_materialization_planner,
            allow_preflight_needs_review=args.allow_preflight_needs_review,
        )
    except Exception as exc:
        err.write(f"property materialization planner runner invalid: {_safe_exception_message(exc)}\n")
        return 1

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("planner_status") == "failed" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_materialization_planner_runner",
        description="Run the offline materialization planner with property preflight gating.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run-report", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--admission-request", required=True)
    parser.add_argument("--formal-package-validation", required=True)
    parser.add_argument("--property-package-binding-summary", required=True)
    parser.add_argument("--materialization-plan", required=True)
    parser.add_argument("--materialization-plan-draft-summary", required=True)
    parser.add_argument("--materialization-plan-preflight-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--planner-run-id", required=True)
    parser.add_argument("--confirm-offline-materialization-planner", action="store_true")
    parser.add_argument("--allow-preflight-needs-review", action="store_true")
    return parser


def _local_consistency_errors(
    *,
    manifest: CustomCorpusManifest,
    dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    formal: dict[str, Any],
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    plan: MaterializationPlan | None,
    draft_summary: dict[str, Any],
    preflight: dict[str, Any],
    hashes: dict[str, str],
    allow_preflight_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if preflight.get("schema_version") != _PREFLIGHT_SCHEMA_VERSION:
        errors.append("preflight_schema_invalid")
    if draft_summary.get("schema_version") != _DRAFT_SUMMARY_SCHEMA_VERSION:
        errors.append("materialization_draft_summary_schema_invalid")
    if binding.get("schema_version") != _PROPERTY_BINDING_SCHEMA_VERSION:
        errors.append("property_package_binding_schema_invalid")
    if formal.get("schema_version") != _FORMAL_SCHEMA_VERSION:
        errors.append("formal_package_validation_schema_invalid")
    if plan_payload.get("schema_version") != _MATERIALIZATION_SCHEMA_VERSION:
        errors.append("materialization_plan_schema_invalid")
    if plan is None and plan_payload.get("schema_version") == _MATERIALIZATION_SCHEMA_VERSION:
        errors.append("materialization_plan_invalid")

    if preflight.get("preflight_status") == "failed":
        errors.append("preflight_failed")
    if preflight.get("preflight_status") == "needs_review":
        if allow_preflight_needs_review:
            warnings.append("preflight_needs_review_allowed")
        else:
            errors.append("preflight_needs_review")
    if preflight.get("preflight_errors"):
        errors.append("preflight_has_errors")
    if binding.get("binding_status") == "failed" or preflight.get("package_binding_status") == "failed":
        errors.append("package_binding_failed")
    if (
        formal.get("validation_status") != "passed"
        or binding.get("formal_package_validation_status") != "passed"
        or preflight.get("formal_package_validation_status") != "passed"
    ):
        errors.append("formal_package_validation_failed")
    if draft_summary.get("draft_status") != "written" or preflight.get("materialization_draft_status") != "written":
        errors.append("materialization_draft_not_written")
    if (
        plan_payload.get("materialization_decision") != "planned"
        or preflight.get("materialization_decision") != "planned"
        or (draft_summary.get("materialization_decision") and draft_summary.get("materialization_decision") != "planned")
    ):
        errors.append("materialization_decision_not_planned")
    if dry_run.decision != "passed" or binding.get("dry_run_decision") != "passed" or preflight.get("dry_run_decision") != "passed":
        errors.append("dry_run_not_passed")
    if dry_run.confirmation_boundary.phase1_status != "not_run" or binding.get("phase1_status") != "not_run" or preflight.get("phase1_status") != "not_run":
        errors.append("dry_run_phase1_ran")
    if dry_run.confirmation_boundary.training_dataset_admitted is not False or binding.get("training_admitted") is True or preflight.get("training_admitted") is True:
        errors.append("dry_run_training_admitted")
    if dry_run.confirmation_boundary.dataset_confirmation_confirmed is not False or plan_payload.get("dry_run_dataset_confirmation_confirmed") is not False:
        errors.append("dry_run_dataset_confirmed")
    if binding.get("binding_errors"):
        errors.append("package_binding_has_errors")

    _check_summary_hashes(errors, preflight, hashes)
    _check_summary_hashes(errors, draft_summary, hashes)
    _check_summary_hashes(errors, binding, hashes)
    if plan_payload.get("source_manifest_sha256") and plan_payload.get("source_manifest_sha256") != hashes["manifest_sha256"]:
        errors.append("manifest_sha256_mismatch")
    if plan_payload.get("source_dry_run_report_sha256") and plan_payload.get("source_dry_run_report_sha256") != hashes["dry_run_report_sha256"]:
        errors.append("dry_run_report_sha256_mismatch")
    if plan_payload.get("source_review_manifest_sha256") and plan_payload.get("source_review_manifest_sha256") != hashes["review_manifest_sha256"]:
        errors.append("review_manifest_sha256_mismatch")
    if plan_payload.get("source_admission_request_sha256") and plan_payload.get("source_admission_request_sha256") != hashes["admission_request_sha256"]:
        errors.append("admission_request_sha256_mismatch")
    if plan_payload.get("source_package_validation_sha256") and plan_payload.get("source_package_validation_sha256") != hashes["formal_package_validation_sha256"]:
        errors.append("formal_package_validation_sha256_mismatch")
    if preflight.get("materialization_plan_draft_sha256") and preflight.get("materialization_plan_draft_sha256") != hashes["materialization_plan_sha256"]:
        errors.append("materialization_plan_sha256_mismatch")
    if draft_summary.get("materialization_plan_draft_sha256") and draft_summary.get("materialization_plan_draft_sha256") != hashes["materialization_plan_sha256"]:
        errors.append("materialization_plan_sha256_mismatch")

    if len({manifest.corpus_id, dry_run.corpus_id, review.corpus_id, admission.corpus_id, str(formal.get("corpus_id", "")), str(binding.get("corpus_id", "")), str(plan_payload.get("corpus_id", "")), str(draft_summary.get("corpus_id", "")), str(preflight.get("corpus_id", ""))}) != 1:
        errors.append("corpus_id_mismatch")
    if len({dry_run.run_id, review.dry_run_id, admission.dry_run_id, str(formal.get("dry_run_id", "")), str(binding.get("dry_run_id", "")), str(plan_payload.get("dry_run_id", "")), str(draft_summary.get("dry_run_id", "")), str(preflight.get("dry_run_id", ""))}) != 1:
        errors.append("dry_run_id_mismatch")
    if len({review.review_manifest_id, admission.review_manifest_id, str(formal.get("review_manifest_id", "")), str(binding.get("review_manifest_id", "")), str(plan_payload.get("review_manifest_id", "")), str(draft_summary.get("review_manifest_id", "")), str(preflight.get("review_manifest_id", ""))}) != 1:
        errors.append("review_manifest_id_mismatch")
    if len({admission.admission_request_id, str(formal.get("admission_request_id", "")), str(binding.get("admission_request_id", "")), str(plan_payload.get("admission_request_id", "")), str(draft_summary.get("admission_request_id", "")), str(preflight.get("admission_request_id", ""))}) != 1:
        errors.append("admission_request_id_mismatch")
    if str(plan_payload.get("materialization_plan_id", "")) != str(draft_summary.get("materialization_plan_id", "")) or str(plan_payload.get("materialization_plan_id", "")) != str(preflight.get("materialization_plan_id", "")):
        errors.append("materialization_plan_id_mismatch")
    if _safe_str(binding.get("review_queue_id")) != _safe_str(preflight.get("review_queue_id")):
        errors.append("review_queue_id_mismatch")
    if _safe_str(binding.get("property_candidate_manifest_id")) != _safe_str(preflight.get("property_candidate_manifest_id")):
        errors.append("property_candidate_manifest_id_mismatch")

    materialization_records = _raw_materialization_records(plan_payload)
    materialization_record_ids = [str(record.get("materialization_record_id", "")) for record in materialization_records]
    materialized_record_ids = [str(record.get("record_id", "")) for record in materialization_records]
    admit_ids = [record.record_id for record in admission.admission_records if record.action == "admit"]
    exclude_ids = [record.record_id for record in admission.admission_records if record.action == "exclude"]
    needs_review_ids = [record.record_id for record in admission.admission_records if record.action == "needs_review" or record.review_decision == "needs_review"]
    blocked_ids = _safe_list(binding.get("blocked_record_ids"))
    if not materialization_records:
        errors.append("no_materialization_records")
    if int(preflight.get("materialization_record_count", -1)) == 0:
        errors.append("no_materialization_records")
    if _safe_list(preflight.get("materialization_record_ids")) != materialization_record_ids:
        errors.append("materialization_record_ids_mismatch")
    if _safe_list(draft_summary.get("materialization_record_ids")) != materialization_record_ids:
        errors.append("materialization_record_ids_mismatch")
    if int(preflight.get("materialization_record_count", -1)) != len(materialization_records):
        errors.append("materialization_record_count_mismatch")
    if _safe_list(binding.get("admit_record_ids")) != _safe_list(draft_summary.get("admit_record_ids")) or _safe_list(binding.get("admit_record_ids")) != _safe_list(preflight.get("admit_record_ids")):
        errors.append("admit_record_ids_mismatch")
    if _safe_list(binding.get("exclude_record_ids")) != _safe_list(draft_summary.get("exclude_record_ids")) or _safe_list(binding.get("exclude_record_ids")) != _safe_list(preflight.get("exclude_record_ids")):
        errors.append("exclude_record_ids_mismatch")
    if blocked_ids != _safe_list(draft_summary.get("blocked_record_ids")) or blocked_ids != _safe_list(preflight.get("blocked_record_ids")):
        errors.append("blocked_record_ids_mismatch")
    for record_id in materialized_record_ids:
        if record_id in exclude_ids:
            errors.append("materialization_record_from_excluded_record")
        if record_id in blocked_ids:
            errors.append("materialization_record_from_blocked_record")
        if record_id in needs_review_ids:
            errors.append("materialization_record_from_needs_review_record")
        if record_id not in admit_ids:
            errors.append("materialization_record_not_admitted")
    return _stable_unique(errors)


def _summary(
    *,
    planner_status: str,
    planner_run_id: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    manifest: CustomCorpusManifest,
    dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    formal: dict[str, Any],
    binding: dict[str, Any],
    plan_payload: dict[str, Any],
    draft_summary: dict[str, Any],
    preflight: dict[str, Any],
    offline_planner_output_sha256: str,
    planner_errors: list[str],
    warnings: list[str],
    offline_planner_status: str = "",
) -> dict[str, Any]:
    materialization_records = _raw_materialization_records(plan_payload)
    return {
        "schema_version": _SCHEMA_VERSION,
        "planner_status": planner_status,
        "planner_run_id": planner_run_id,
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
        "materialization_plan_draft_summary_path": _basename(paths["materialization_plan_draft_summary_path"], "property_materialization_plan_draft_summary.json"),
        "materialization_plan_draft_summary_sha256": hashes["materialization_plan_draft_summary_sha256"],
        "materialization_plan_preflight_summary_path": _basename(paths["materialization_plan_preflight_summary_path"], "materialization_plan_preflight_summary.json"),
        "materialization_plan_preflight_summary_sha256": hashes["materialization_plan_preflight_summary_sha256"],
        "offline_planner_output_path": _ARTIFACTS["offline_materialization_planner_output_json"] if offline_planner_output_sha256 else "",
        "offline_planner_output_sha256": offline_planner_output_sha256,
        "corpus_id": manifest.corpus_id,
        "dry_run_id": dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "materialization_plan_id": str(plan_payload.get("materialization_plan_id", draft_summary.get("materialization_plan_id", ""))),
        "review_queue_id": str(binding.get("review_queue_id", preflight.get("review_queue_id", ""))),
        "property_candidate_manifest_id": str(binding.get("property_candidate_manifest_id", preflight.get("property_candidate_manifest_id", ""))),
        "dataset_target": str(plan_payload.get("dataset_target", preflight.get("dataset_target", ""))),
        "preflight_status": str(preflight.get("preflight_status", "")),
        "package_binding_status": str(binding.get("binding_status", preflight.get("package_binding_status", ""))),
        "formal_package_validation_status": str(formal.get("validation_status", preflight.get("formal_package_validation_status", ""))),
        "materialization_draft_status": str(draft_summary.get("draft_status", preflight.get("materialization_draft_status", ""))),
        "materialization_decision": str(plan_payload.get("materialization_decision", preflight.get("materialization_decision", ""))),
        "offline_planner_status": offline_planner_status,
        "dry_run_decision": dry_run.decision,
        "phase1_status": dry_run.confirmation_boundary.phase1_status,
        "training_admitted": dry_run.confirmation_boundary.training_dataset_admitted,
        "admission_record_count": len(admission.admission_records),
        "admit_count": len([record for record in admission.admission_records if record.action == "admit"]),
        "exclude_count": len([record for record in admission.admission_records if record.action == "exclude"]),
        "blocked_record_count": len(_safe_list(binding.get("blocked_record_ids"))),
        "materialization_record_count": len(materialization_records),
        "materialization_record_ids": [str(record.get("materialization_record_id", "")) for record in materialization_records],
        "admit_record_ids": [record.record_id for record in admission.admission_records if record.action == "admit"],
        "exclude_record_ids": [record.record_id for record in admission.admission_records if record.action == "exclude"],
        "blocked_record_ids": _safe_list(binding.get("blocked_record_ids")),
        "planner_artifacts": {
            "offline_materialization_planner_output_json": _ARTIFACTS["offline_materialization_planner_output_json"],
            "property_materialization_planner_summary_json": _ARTIFACTS["property_materialization_planner_summary_json"],
            "redacted_property_materialization_planner_evidence_md": _ARTIFACTS["redacted_property_materialization_planner_evidence_md"],
        },
        "planner_errors": _stable_unique(planner_errors),
        "warnings": _stable_unique(warnings),
        "redaction_status": "passed",
    }


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Materialization Planner Evidence",
            "",
            f"- Planner status: `{summary['planner_status']}`",
            f"- Planner run id: `{summary['planner_run_id']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Dry-run id: `{summary['dry_run_id']}`",
            f"- Review manifest id: `{summary['review_manifest_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Preflight status: `{summary['preflight_status']}`",
            f"- Package binding status: `{summary['package_binding_status']}`",
            f"- Formal package validation status: `{summary['formal_package_validation_status']}`",
            f"- Offline planner status: `{summary['offline_planner_status']}`",
            f"- Materialization records: `{summary['materialization_record_count']}`",
            f"- Materialization record ids: `{json.dumps(summary['materialization_record_ids'])}`",
            f"- Admit record ids: `{json.dumps(summary['admit_record_ids'])}`",
            f"- Exclude record ids: `{json.dumps(summary['exclude_record_ids'])}`",
            f"- Blocked record ids: `{json.dumps(summary['blocked_record_ids'])}`",
            f"- Planner errors: `{json.dumps(summary['planner_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- formal package binding was completed upstream.",
            "- the offline materialization planner was run.",
            "- No materializer was run.",
            "- No materialization was executed.",
            "- No materialization output records were created.",
            "- No candidate/training CSV was created.",
            "- No training data was admitted.",
            "- Phase 1 did not run.",
            "- DatasetConfirmation was not changed.",
            "",
        ]
    )


def _offline_planner_output_errors(planner_output: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if planner_output.get("schema_version") != _OFFLINE_PLANNER_SCHEMA_VERSION:
        errors.append("offline_planner_schema_invalid")
    if planner_output.get("planner_status") != "planned":
        errors.append("offline_planner_failed")
    if planner_output.get("dry_run_phase1_status") not in {"", None, "not_run"} or planner_output.get("phase1_status") not in {"", None, "not_run"}:
        errors.append("offline_planner_claimed_phase1_run")
    if planner_output.get("dry_run_dataset_confirmation_confirmed") is True or planner_output.get("dataset_confirmation_changed") is True:
        errors.append("offline_planner_claimed_dataset_confirmation_change")
    if planner_output.get("dry_run_training_dataset_admitted") is True or planner_output.get("training_admitted") is True:
        errors.append("offline_planner_claimed_training_admission")
    for key, value in _walk_items(planner_output):
        lowered_key = key.lower()
        lowered_value = str(value).lower()
        if lowered_key.endswith("_path") and ".csv" in lowered_value:
            errors.append("offline_planner_claimed_csv_output")
        if lowered_key in {"materialized_records", "materialization_outputs", "candidate_artifacts", "training_artifacts"} and value:
            errors.append("offline_planner_claimed_materialization_output")
    return _stable_unique(errors)


def _load_dry_run_report(path: str | Path) -> CustomCorpusDryRunReport:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return CustomCorpusDryRunReport.model_validate(payload)
    except Exception as exc:
        raise CustomCorpusPropertyMaterializationPlannerRunnerError("dry-run report invalid") from exc


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
        raise CustomCorpusPropertyMaterializationPlannerRunnerError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyMaterializationPlannerRunnerError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyMaterializationPlannerRunnerError(f"{label} invalid")
    return payload


def _check_summary_hashes(errors: list[str], payload: dict[str, Any], hashes: dict[str, str]) -> None:
    pairs = (
        ("manifest_sha256", "manifest_sha256", "manifest_sha256_mismatch"),
        ("dry_run_report_sha256", "dry_run_report_sha256", "dry_run_report_sha256_mismatch"),
        ("review_manifest_sha256", "review_manifest_sha256", "review_manifest_sha256_mismatch"),
        ("admission_request_sha256", "admission_request_sha256", "admission_request_sha256_mismatch"),
        ("formal_package_validation_sha256", "formal_package_validation_sha256", "formal_package_validation_sha256_mismatch"),
    )
    for field, hash_field, error in pairs:
        if payload.get(field) and payload.get(field) != hashes[hash_field]:
            errors.append(error)


def _required_safe_id(value: str, field_name: str) -> str:
    if not value or not _SAFE_ID_RE.fullmatch(value):
        raise CustomCorpusPropertyMaterializationPlannerRunnerError(f"{field_name} invalid")
    return value


def _raw_materialization_records(plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = plan_payload.get("materialization_records", [])
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _safe_str(value: Any) -> str:
    return str(value or "")


def _safe_sha_for_path(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except Exception:
        return ""


def _basename(path: str | Path, fallback: str) -> str:
    return Path(path).name or fallback


def _append_errors(errors: list[str], additions: list[str]) -> None:
    for error in additions:
        _append_unique(errors, error)


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _walk_items(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            items.extend(_walk_items(nested, key_text))
    elif isinstance(value, list):
        for nested in value:
            items.extend(_walk_items(nested, prefix))
    else:
        items.append((prefix, value))
    return items


def _contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _input_contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    markers = tuple(marker for marker in _FORBIDDEN_MARKERS if marker != ".csv")
    if any(marker.lower() in lowered for marker in markers):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "planner_status": "failed",
        "planner_errors": ["property_materialization_planner_redaction_failed"],
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


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
