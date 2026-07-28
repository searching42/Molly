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


_SCHEMA_VERSION = "custom_corpus_property_materialization_plan_preflight.v1"
_FORMAL_SCHEMA_VERSION = "custom_corpus_admission_package_validation.v1"
_PROPERTY_BINDING_SCHEMA_VERSION = "custom_corpus_property_package_binding.v1"
_MATERIALIZATION_SCHEMA_VERSION = "custom_corpus_materialization.v1"
_DRAFT_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_materialization_plan_draft_builder.v1"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^(sha256:)?([0-9a-fA-F]{64})$")
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
    "x-amz-signature",
    "signature=",
    "signedurl",
    "signed-url",
)


class CustomCorpusPropertyMaterializationPlanPreflightError(ValueError):
    pass


def preflight_property_materialization_plan(
    *,
    manifest_path: str | Path,
    dry_run_report_path: str | Path,
    review_manifest_path: str | Path,
    admission_request_path: str | Path,
    formal_package_validation_path: str | Path,
    property_package_binding_summary_path: str | Path,
    materialization_plan_draft_path: str | Path,
    materialization_plan_draft_summary_path: str | Path,
    require_package_binding_passed: bool = False,
    require_draft_written: bool = True,
) -> dict[str, Any]:
    manifest = load_custom_corpus_manifest(manifest_path)
    dry_run = _load_dry_run_report(dry_run_report_path)
    review = load_review_manifest(review_manifest_path)
    admission = load_admission_request(admission_request_path)
    formal, formal_errors = _load_formal_package_validation(formal_package_validation_path)
    binding, binding_errors = _load_property_package_binding_summary(property_package_binding_summary_path)
    plan_payload, plan, plan_errors = _load_materialization_plan_payload(materialization_plan_draft_path)
    draft_summary, draft_summary_errors = _load_materialization_draft_summary(materialization_plan_draft_summary_path)

    paths = {
        "manifest_path": manifest_path,
        "dry_run_report_path": dry_run_report_path,
        "review_manifest_path": review_manifest_path,
        "admission_request_path": admission_request_path,
        "formal_package_validation_path": formal_package_validation_path,
        "property_package_binding_summary_path": property_package_binding_summary_path,
        "materialization_plan_draft_path": materialization_plan_draft_path,
        "materialization_plan_draft_summary_path": materialization_plan_draft_summary_path,
    }
    hashes = {
        "manifest_sha256": _safe_sha_for_path(manifest_path),
        "dry_run_report_sha256": _safe_sha_for_path(dry_run_report_path),
        "review_manifest_sha256": _safe_sha_for_path(review_manifest_path),
        "admission_request_sha256": _safe_sha_for_path(admission_request_path),
        "formal_package_validation_sha256": _safe_sha_for_path(formal_package_validation_path),
        "property_package_binding_summary_sha256": _safe_sha_for_path(property_package_binding_summary_path),
        "materialization_plan_draft_sha256": _safe_sha_for_path(materialization_plan_draft_path),
        "materialization_plan_draft_summary_sha256": _safe_sha_for_path(materialization_plan_draft_summary_path),
    }

    preflight_errors: list[str] = []
    warnings: list[str] = []
    _append_errors(preflight_errors, formal_errors)
    _append_errors(preflight_errors, binding_errors)
    _append_errors(preflight_errors, plan_errors)
    _append_errors(preflight_errors, draft_summary_errors)
    _append_errors(
        preflight_errors,
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
            hashes=hashes,
            require_package_binding_passed=require_package_binding_passed,
            require_draft_written=require_draft_written,
            warnings=warnings,
        ),
    )

    materialization_records = _raw_materialization_records(plan_payload)
    materialization_record_ids = [str(record.get("materialization_record_id", "")) for record in materialization_records]
    status = "failed" if preflight_errors else "needs_review" if warnings else "passed"
    summary = _summary(
        preflight_status=status,
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
        materialization_record_ids=materialization_record_ids,
        preflight_errors=_stable_unique(preflight_errors),
        warnings=_stable_unique(warnings),
    )
    if _contains_forbidden_material(summary):
        return _minimal_redaction_failure()
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
        summary = preflight_property_materialization_plan(
            manifest_path=args.manifest,
            dry_run_report_path=args.dry_run_report,
            review_manifest_path=args.review_manifest,
            admission_request_path=args.admission_request,
            formal_package_validation_path=args.formal_package_validation,
            property_package_binding_summary_path=args.property_package_binding_summary,
            materialization_plan_draft_path=args.materialization_plan_draft,
            materialization_plan_draft_summary_path=args.materialization_plan_draft_summary,
            require_package_binding_passed=args.require_package_binding_passed,
            require_draft_written=args.require_draft_written,
        )
    except Exception as exc:
        err.write(f"property materialization plan preflight invalid: {_safe_exception_message(exc)}\n")
        return 1

    if args.output_summary and summary.get("redaction_status") != "failed":
        write_json(Path(args.output_summary).expanduser(), summary)
    if args.output_markdown and summary.get("redaction_status") != "failed":
        markdown = _markdown(summary)
        if _contains_forbidden_material(markdown):
            summary = _minimal_redaction_failure()
        else:
            Path(args.output_markdown).expanduser().parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_markdown).expanduser().write_text(markdown, encoding="utf-8")

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("preflight_status") == "failed" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_materialization_plan_preflight",
        description="Preflight a property materialization plan draft before offline planner submission.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run-report", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--admission-request", required=True)
    parser.add_argument("--formal-package-validation", required=True)
    parser.add_argument("--property-package-binding-summary", required=True)
    parser.add_argument("--materialization-plan-draft", required=True)
    parser.add_argument("--materialization-plan-draft-summary", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-markdown", default="")
    parser.add_argument("--require-package-binding-passed", action="store_true")
    parser.add_argument("--require-draft-written", action="store_true", default=True)
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
    hashes: dict[str, str],
    require_package_binding_passed: bool,
    require_draft_written: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if formal.get("schema_version") != _FORMAL_SCHEMA_VERSION:
        errors.append("formal_package_validation_schema_invalid")
    if formal.get("validation_status") != "passed":
        errors.append("formal_package_validation_failed")
    if formal.get("admission_decision", "eligible") != "eligible":
        errors.append("formal_package_not_eligible")

    if binding.get("schema_version") != _PROPERTY_BINDING_SCHEMA_VERSION:
        errors.append("property_package_binding_schema_invalid")
    if binding.get("binding_status") == "failed":
        errors.append("package_binding_failed")
    if binding.get("binding_status") == "needs_review":
        if require_package_binding_passed:
            errors.append("package_binding_needs_review")
        else:
            warnings.append("package_binding_needs_review")
    if binding.get("formal_package_validation_status") != "passed":
        errors.append("formal_package_validation_failed")
    if binding.get("dry_run_decision") != "passed":
        errors.append("dry_run_not_passed")
    if binding.get("phase1_status") != "not_run":
        errors.append("dry_run_phase1_ran")
    if binding.get("training_admitted") is True:
        errors.append("dry_run_training_admitted")
    if binding.get("binding_errors"):
        errors.append("package_binding_has_errors")

    if draft_summary.get("schema_version") != _DRAFT_SUMMARY_SCHEMA_VERSION:
        errors.append("materialization_draft_summary_schema_invalid")
    if require_draft_written and draft_summary.get("draft_status") != "written":
        errors.append("materialization_draft_not_written")
    if draft_summary.get("draft_errors"):
        errors.append("materialization_draft_has_errors")

    if dry_run.decision != "passed":
        errors.append("dry_run_not_passed")
    if dry_run.confirmation_boundary.phase1_status != "not_run":
        errors.append("dry_run_phase1_ran")
    if dry_run.confirmation_boundary.training_dataset_admitted is not False:
        errors.append("dry_run_training_admitted")
    if dry_run.confirmation_boundary.dataset_confirmation_confirmed is not False:
        errors.append("dry_run_dataset_confirmed")

    _check_hashes(errors, formal, hashes, formal_prefix=False)
    _check_hashes(errors, binding, hashes, formal_prefix=False)
    _check_hashes(errors, draft_summary, hashes, formal_prefix=False)
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
    if draft_summary.get("materialization_plan_draft_sha256") and draft_summary.get("materialization_plan_draft_sha256") != hashes["materialization_plan_draft_sha256"]:
        errors.append("materialization_plan_draft_sha256_mismatch")

    if len({manifest.corpus_id, dry_run.corpus_id, review.corpus_id, admission.corpus_id, str(formal.get("corpus_id", "")), str(binding.get("corpus_id", "")), str(plan_payload.get("corpus_id", "")), str(draft_summary.get("corpus_id", ""))}) != 1:
        errors.append("corpus_id_mismatch")
    if len({dry_run.run_id, review.dry_run_id, admission.dry_run_id, str(formal.get("dry_run_id", "")), str(binding.get("dry_run_id", "")), str(plan_payload.get("dry_run_id", "")), str(draft_summary.get("dry_run_id", ""))}) != 1:
        errors.append("dry_run_id_mismatch")
    if len({review.review_manifest_id, admission.review_manifest_id, str(formal.get("review_manifest_id", "")), str(binding.get("review_manifest_id", "")), str(plan_payload.get("review_manifest_id", "")), str(draft_summary.get("review_manifest_id", ""))}) != 1:
        errors.append("review_manifest_id_mismatch")
    if len({admission.admission_request_id, str(formal.get("admission_request_id", "")), str(binding.get("admission_request_id", "")), str(plan_payload.get("admission_request_id", "")), str(draft_summary.get("admission_request_id", ""))}) != 1:
        errors.append("admission_request_id_mismatch")
    if str(plan_payload.get("materialization_plan_id", "")) != str(draft_summary.get("materialization_plan_id", "")):
        errors.append("materialization_plan_id_mismatch")

    if plan_payload.get("materialization_decision") != "planned":
        errors.append("materialization_decision_not_planned")
    if plan_payload.get("package_validation_status") != "passed":
        errors.append("package_validation_failed")
    if plan_payload.get("package_admission_decision") != "eligible":
        errors.append("package_admission_not_eligible")
    if plan_payload.get("materialization_mode") != "candidate_only":
        errors.append("materialization_mode_invalid")
    if plan_payload.get("dry_run_phase1_status") != "not_run":
        errors.append("dry_run_phase1_ran")
    if plan_payload.get("dry_run_dataset_confirmation_confirmed") is not False:
        errors.append("dry_run_dataset_confirmed")
    if plan_payload.get("dry_run_training_dataset_admitted") is not False:
        errors.append("dry_run_training_admitted")

    admission_records = admission.admission_records
    admit_ids = [record.record_id for record in admission_records if record.action == "admit"]
    exclude_ids = [record.record_id for record in admission_records if record.action == "exclude"]
    needs_review_ids = [record.record_id for record in admission_records if record.review_decision == "needs_review" or record.action == "needs_review"]
    blocked_ids = _safe_list(binding.get("blocked_record_ids"))
    if not admit_ids:
        errors.append("no_admitted_records")
    if int(formal.get("admission_record_count", -1)) != len(admission_records):
        errors.append("admission_record_count_mismatch")
    if int(formal.get("admit_count", -1)) != len(admit_ids):
        errors.append("admit_count_mismatch")
    if int(formal.get("exclude_count", -1)) != len(exclude_ids):
        errors.append("exclude_count_mismatch")
    if _safe_list(binding.get("admit_record_ids")) != _safe_list(draft_summary.get("admit_record_ids")):
        errors.append("admit_record_ids_mismatch")
    if _safe_list(binding.get("exclude_record_ids")) != _safe_list(draft_summary.get("exclude_record_ids")):
        errors.append("exclude_record_ids_mismatch")
    if blocked_ids != _safe_list(draft_summary.get("blocked_record_ids")):
        errors.append("blocked_record_ids_mismatch")

    materialization_records = _raw_materialization_records(plan_payload)
    materialized_record_ids = [str(record.get("record_id", "")) for record in materialization_records]
    materialization_record_ids = [str(record.get("materialization_record_id", "")) for record in materialization_records]
    if not materialization_records:
        errors.append("no_materialization_records")
    if int(draft_summary.get("materialization_record_count", -1)) != len(materialization_records):
        errors.append("materialization_record_count_mismatch")
    if _safe_list(draft_summary.get("materialization_record_ids")) != materialization_record_ids:
        errors.append("materialization_record_ids_mismatch")
    for record_id in materialized_record_ids:
        if record_id in exclude_ids:
            errors.append("materialization_record_from_excluded_record")
        if record_id in blocked_ids:
            errors.append("materialization_record_from_blocked_record")
        if record_id in needs_review_ids:
            errors.append("materialization_record_from_needs_review_record")
        if record_id not in admit_ids:
            errors.append("materialization_record_not_admitted")
    admission_by_id = {record.record_id: record for record in admission_records}
    for record in materialization_records:
        admission_record = admission_by_id.get(str(record.get("record_id", "")))
        if admission_record and (admission_record.action != "admit" or admission_record.review_decision != "accept"):
            if admission_record.review_decision == "needs_review" or admission_record.action == "needs_review":
                errors.append("materialization_record_from_needs_review_record")
            elif admission_record.action == "exclude":
                errors.append("materialization_record_from_excluded_record")
            else:
                errors.append("materialization_record_not_admitted")
        if record.get("admission_action") != "admit" or record.get("review_decision") != "accept":
            errors.append("materialization_record_not_accepted_admit")
    if plan is None and _MATERIALIZATION_SCHEMA_VERSION == plan_payload.get("schema_version"):
        errors.append("materialization_plan_invalid")
    return _stable_unique(errors)


def _summary(
    *,
    preflight_status: str,
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
    materialization_record_ids: list[str],
    preflight_errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": preflight_status,
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
        "materialization_plan_draft_path": _basename(paths["materialization_plan_draft_path"], "custom_corpus_materialization.draft.json"),
        "materialization_plan_draft_sha256": hashes["materialization_plan_draft_sha256"],
        "materialization_plan_draft_summary_path": _basename(paths["materialization_plan_draft_summary_path"], "property_materialization_plan_draft_summary.json"),
        "materialization_plan_draft_summary_sha256": hashes["materialization_plan_draft_summary_sha256"],
        "corpus_id": manifest.corpus_id,
        "dry_run_id": dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "materialization_plan_id": str(plan_payload.get("materialization_plan_id", draft_summary.get("materialization_plan_id", ""))),
        "review_queue_id": str(binding.get("review_queue_id", draft_summary.get("review_queue_id", ""))),
        "property_candidate_manifest_id": str(binding.get("property_candidate_manifest_id", draft_summary.get("property_candidate_manifest_id", ""))),
        "dataset_target": str(plan_payload.get("dataset_target", draft_summary.get("dataset_target", ""))),
        "package_binding_status": str(binding.get("binding_status", "")),
        "formal_package_validation_status": str(formal.get("validation_status", "")),
        "materialization_draft_status": str(draft_summary.get("draft_status", "")),
        "materialization_decision": str(plan_payload.get("materialization_decision", "")),
        "dry_run_decision": dry_run.decision,
        "phase1_status": dry_run.confirmation_boundary.phase1_status,
        "training_admitted": dry_run.confirmation_boundary.training_dataset_admitted,
        "admission_record_count": len(admission.admission_records),
        "admit_count": len([record for record in admission.admission_records if record.action == "admit"]),
        "exclude_count": len([record for record in admission.admission_records if record.action == "exclude"]),
        "blocked_record_count": len(_safe_list(binding.get("blocked_record_ids"))),
        "materialization_record_count": len(_raw_materialization_records(plan_payload)),
        "materialization_record_ids": materialization_record_ids,
        "admit_record_ids": [record.record_id for record in admission.admission_records if record.action == "admit"],
        "exclude_record_ids": [record.record_id for record in admission.admission_records if record.action == "exclude"],
        "blocked_record_ids": _safe_list(binding.get("blocked_record_ids")),
        "preflight_errors": preflight_errors,
        "warnings": warnings,
        "redaction_status": "passed",
    }


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Materialization Plan Preflight Summary",
            "",
            f"- Preflight status: `{summary['preflight_status']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Dry-run id: `{summary['dry_run_id']}`",
            f"- Review manifest id: `{summary['review_manifest_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Package binding status: `{summary['package_binding_status']}`",
            f"- Formal package validation status: `{summary['formal_package_validation_status']}`",
            f"- Materialization draft status: `{summary['materialization_draft_status']}`",
            f"- Materialization decision: `{summary['materialization_decision']}`",
            f"- Materialization records: `{summary['materialization_record_count']}`",
            f"- Materialization record ids: `{json.dumps(summary['materialization_record_ids'])}`",
            f"- Admit record ids: `{json.dumps(summary['admit_record_ids'])}`",
            f"- Exclude record ids: `{json.dumps(summary['exclude_record_ids'])}`",
            f"- Blocked record ids: `{json.dumps(summary['blocked_record_ids'])}`",
            f"- Preflight errors: `{json.dumps(summary['preflight_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a materialization plan preflight only.",
            "- The offline materialization planner was not run.",
            "- No materializer was run.",
            "- No materialization was executed.",
            "- No candidate/training CSV was created.",
            "- No training data was admitted.",
            "- Phase 1 did not run.",
            "- DatasetConfirmation was not changed.",
            "",
        ]
    )


def _load_dry_run_report(path: str | Path) -> CustomCorpusDryRunReport:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return CustomCorpusDryRunReport.model_validate(payload)
    except Exception as exc:
        raise CustomCorpusPropertyMaterializationPlanPreflightError("dry-run report invalid") from exc


def _load_formal_package_validation(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    payload = _read_safe_json_dict(path, "formal package validation")
    errors = []
    if payload.get("schema_version") != _FORMAL_SCHEMA_VERSION:
        errors.append("formal_package_validation_schema_invalid")
    return payload, errors


def _load_property_package_binding_summary(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    payload = _read_safe_json_dict(path, "property package binding summary")
    errors = []
    if payload.get("schema_version") != _PROPERTY_BINDING_SCHEMA_VERSION:
        errors.append("property_package_binding_schema_invalid")
    _validate_safe_id_fields(
        payload,
        ("corpus_id", "dry_run_id", "review_manifest_id", "admission_request_id", "review_queue_id", "property_candidate_manifest_id"),
    )
    return payload, errors


def _load_materialization_draft_summary(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    payload = _read_safe_json_dict(path, "materialization draft summary")
    errors = []
    if payload.get("schema_version") != _DRAFT_SUMMARY_SCHEMA_VERSION:
        errors.append("materialization_draft_summary_schema_invalid")
    _validate_safe_id_fields(
        payload,
        ("corpus_id", "dry_run_id", "review_manifest_id", "admission_request_id", "materialization_plan_id", "review_queue_id", "property_candidate_manifest_id"),
    )
    return payload, errors


def _load_materialization_plan_payload(path: str | Path) -> tuple[dict[str, Any], MaterializationPlan | None, list[str]]:
    payload = _read_safe_json_dict(path, "materialization plan draft")
    errors: list[str] = []
    plan: MaterializationPlan | None = None
    if payload.get("schema_version") != _MATERIALIZATION_SCHEMA_VERSION:
        errors.append("materialization_plan_schema_invalid")
    else:
        try:
            plan = load_materialization_plan(path)
        except CustomCorpusMaterializationError:
            plan = None
    return payload, plan, errors


def _read_safe_json_dict(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyMaterializationPlanPreflightError(f"{label} invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyMaterializationPlanPreflightError(f"{label} invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyMaterializationPlanPreflightError(f"{label} invalid")
    return payload


def _check_hashes(errors: list[str], payload: dict[str, Any], hashes: dict[str, str], *, formal_prefix: bool) -> None:
    del formal_prefix
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


def _validate_safe_id_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        value = str(payload.get(field, "") or "")
        if value and not _SAFE_ID_RE.fullmatch(value):
            raise CustomCorpusPropertyMaterializationPlanPreflightError(f"{field} invalid")


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
        if error not in errors:
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
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "preflight_status": "failed",
        "preflight_errors": ["property_materialization_plan_preflight_redaction_failed"],
        "redaction_status": "failed",
    }


def _safe_exception_message(exc: BaseException) -> str:
    message = str(exc or "").lower()
    if any(marker.lower() in message for marker in _FORBIDDEN_MARKERS):
        return "artifact invalid"
    if _ABSOLUTE_PATH_VALUE_RE.search(json.dumps(message)):
        return "artifact invalid"
    if "schema" in message:
        return "schema_version invalid"
    if "invalid" in message:
        return "artifact invalid"
    return exc.__class__.__name__


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
