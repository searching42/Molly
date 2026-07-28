from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_admission import AdmissionRequest, load_admission_request
from ai4s_agent.custom_corpus_dry_run import CustomCorpusDryRunReport
from ai4s_agent.custom_corpus_manifest import CustomCorpusManifest, load_custom_corpus_manifest, sha256_file
from ai4s_agent.custom_corpus_materialization import validate_materialization_plan
from ai4s_agent.custom_corpus_review import ReviewManifest, load_review_manifest


_SCHEMA_VERSION = "custom_corpus_property_materialization_plan_draft_builder.v1"
_BINDING_SCHEMA_VERSION = "custom_corpus_property_package_binding.v1"
_FORMAL_SCHEMA_VERSION = "custom_corpus_admission_package_validation.v1"
_MATERIALIZATION_SCHEMA_VERSION = "custom_corpus_materialization.v1"
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
_ARTIFACTS = {
    "custom_corpus_materialization_draft_json": "custom_corpus_materialization.draft.json",
    "property_materialization_plan_draft_summary_json": "property_materialization_plan_draft_summary.json",
    "redacted_property_materialization_plan_draft_evidence_md": "redacted_property_materialization_plan_draft_evidence.md",
}


class CustomCorpusPropertyMaterializationPlanDraftError(ValueError):
    pass


def build_property_materialization_plan_draft(
    *,
    manifest_path: str | Path,
    dry_run_report_path: str | Path,
    review_manifest_path: str | Path,
    admission_request_path: str | Path,
    formal_package_validation_path: str | Path,
    property_package_binding_summary_path: str | Path,
    output_dir: str | Path,
    materialization_plan_id: str,
    dataset_target: str,
    created_by: str,
    confirm_materialization_plan_draft_output: bool,
    allow_package_binding_needs_review: bool = False,
) -> dict[str, Any]:
    materialization_plan_id = _required_safe_id(materialization_plan_id, field_name="materialization_plan_id")
    dataset_target = _required_safe_id(dataset_target, field_name="dataset_target")
    created_by = _safe_created_by(created_by)
    manifest = load_custom_corpus_manifest(manifest_path)
    dry_run = _load_dry_run_report(dry_run_report_path)
    review = load_review_manifest(review_manifest_path)
    admission = load_admission_request(admission_request_path)
    formal = _load_formal_package_validation(formal_package_validation_path)
    binding = _load_property_package_binding_summary(property_package_binding_summary_path)

    paths = {
        "manifest_path": manifest_path,
        "dry_run_report_path": dry_run_report_path,
        "review_manifest_path": review_manifest_path,
        "admission_request_path": admission_request_path,
        "formal_package_validation_path": formal_package_validation_path,
        "property_package_binding_summary_path": property_package_binding_summary_path,
    }
    hashes = {
        "manifest_sha256": _safe_sha_for_path(manifest_path),
        "dry_run_report_sha256": _safe_sha_for_path(dry_run_report_path),
        "review_manifest_sha256": _safe_sha_for_path(review_manifest_path),
        "admission_request_sha256": _safe_sha_for_path(admission_request_path),
        "formal_package_validation_sha256": _safe_sha_for_path(formal_package_validation_path),
        "property_package_binding_summary_sha256": _safe_sha_for_path(property_package_binding_summary_path),
    }
    run_dir = Path(output_dir).expanduser() / materialization_plan_id
    draft_errors: list[str] = []
    warnings: list[str] = []

    if not confirm_materialization_plan_draft_output:
        draft_errors.append("materialization_plan_draft_output_not_confirmed")
    if run_dir.exists() and any(run_dir.iterdir()):
        draft_errors.append("output_directory_not_clean")
    _append_errors(
        draft_errors,
        _local_consistency_errors(
            manifest=manifest,
            dry_run=dry_run,
            review=review,
            admission=admission,
            formal=formal,
            binding=binding,
            hashes=hashes,
            allow_package_binding_needs_review=allow_package_binding_needs_review,
            warnings=warnings,
        ),
    )

    materialization_records = _materialization_records(
        materialization_plan_id=materialization_plan_id,
        admission=admission,
        admission_sha=hashes["admission_request_sha256"],
        package_sha=hashes["formal_package_validation_sha256"],
        excluded_ids=binding["exclude_record_ids"],
        blocked_ids=binding["blocked_record_ids"],
    )
    if not materialization_records:
        _append_unique(draft_errors, "no_materialization_records")

    draft_payload = _draft_payload(
        materialization_plan_id=materialization_plan_id,
        dataset_target=dataset_target,
        created_by=created_by,
        manifest=manifest,
        dry_run=dry_run,
        review=review,
        admission=admission,
        formal=formal,
        hashes=hashes,
        materialization_records=materialization_records,
    )
    if materialization_records:
        try:
            validate_materialization_plan(draft_payload)
        except Exception:
            _append_unique(draft_errors, "generated_materialization_plan_invalid")

    summary = _summary(
        draft_status="blocked" if draft_errors else "written",
        materialization_plan_id=materialization_plan_id,
        paths=paths,
        hashes=hashes,
        manifest=manifest,
        dry_run=dry_run,
        review=review,
        admission=admission,
        formal=formal,
        binding=binding,
        dataset_target=dataset_target,
        materialization_records=materialization_records,
        draft_errors=_stable_unique(draft_errors),
        warnings=_stable_unique(warnings),
    )
    if summary["draft_status"] == "blocked":
        return _minimal_redaction_failure() if _contains_forbidden_material(summary) else summary

    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "draft": draft_payload, "evidence": evidence}):
        return _minimal_redaction_failure()

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / _ARTIFACTS["custom_corpus_materialization_draft_json"], draft_payload)
    write_json(run_dir / _ARTIFACTS["property_materialization_plan_draft_summary_json"], summary)
    (run_dir / _ARTIFACTS["redacted_property_materialization_plan_draft_evidence_md"]).write_text(
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
    args = _parser().parse_args(argv)
    try:
        summary = build_property_materialization_plan_draft(
            manifest_path=args.manifest,
            dry_run_report_path=args.dry_run_report,
            review_manifest_path=args.review_manifest,
            admission_request_path=args.admission_request,
            formal_package_validation_path=args.formal_package_validation,
            property_package_binding_summary_path=args.property_package_binding_summary,
            output_dir=args.output_dir,
            materialization_plan_id=args.materialization_plan_id,
            dataset_target=args.dataset_target,
            created_by=args.created_by,
            confirm_materialization_plan_draft_output=args.confirm_materialization_plan_draft_output,
            allow_package_binding_needs_review=args.allow_package_binding_needs_review,
        )
    except Exception as exc:
        err.write(f"property materialization plan draft invalid: {_safe_exception_message(exc)}\n")
        return 1

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("draft_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_materialization_plan_draft",
        description="Build a reviewable property materialization plan draft from package-validated admission artifacts.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run-report", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--admission-request", required=True)
    parser.add_argument("--formal-package-validation", required=True)
    parser.add_argument("--property-package-binding-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--materialization-plan-id", required=True)
    parser.add_argument("--dataset-target", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-materialization-plan-draft-output", action="store_true")
    parser.add_argument("--allow-package-binding-needs-review", action="store_true")
    return parser


def _local_consistency_errors(
    *,
    manifest: CustomCorpusManifest,
    dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    formal: dict[str, Any],
    binding: dict[str, Any],
    hashes: dict[str, str],
    allow_package_binding_needs_review: bool,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if binding["binding_status"] == "failed":
        errors.append("package_binding_failed")
    if binding["binding_status"] == "needs_review":
        if allow_package_binding_needs_review:
            warnings.append("package_binding_needs_review_allowed")
        else:
            errors.append("package_binding_needs_review")
    if formal.get("schema_version") != _FORMAL_SCHEMA_VERSION:
        errors.append("formal_package_validation_schema_invalid")
    if formal.get("validation_status") != "passed":
        errors.append("formal_package_validation_failed")
    if formal.get("admission_decision") != "eligible":
        errors.append("formal_package_not_eligible")
    if binding["formal_package_validation_status"] != "passed":
        errors.append("formal_package_validation_failed")
    if binding["dry_run_decision"] != "passed" or dry_run.decision != "passed":
        errors.append("dry_run_not_passed")
    if binding["phase1_status"] != "not_run" or dry_run.confirmation_boundary.phase1_status != "not_run":
        errors.append("dry_run_phase1_ran")
    if binding["training_admitted"] is not False or dry_run.confirmation_boundary.training_dataset_admitted is not False:
        errors.append("dry_run_training_admitted")
    if dry_run.confirmation_boundary.dataset_confirmation_confirmed is not False:
        errors.append("dry_run_dataset_confirmed")
    if binding["binding_errors"]:
        errors.append("package_binding_has_errors")

    if binding["manifest_sha256"] != hashes["manifest_sha256"]:
        errors.append("manifest_sha256_mismatch")
    if binding["dry_run_report_sha256"] != hashes["dry_run_report_sha256"]:
        errors.append("dry_run_report_sha256_mismatch")
    if binding["review_manifest_sha256"] != hashes["review_manifest_sha256"]:
        errors.append("review_manifest_sha256_mismatch")
    if binding["admission_request_sha256"] != hashes["admission_request_sha256"]:
        errors.append("admission_request_sha256_mismatch")
    if binding["formal_package_validation_sha256"] != hashes["formal_package_validation_sha256"]:
        errors.append("formal_package_validation_sha256_mismatch")

    if len({manifest.corpus_id, dry_run.corpus_id, review.corpus_id, admission.corpus_id, binding["corpus_id"], str(formal.get("corpus_id", ""))}) != 1:
        errors.append("corpus_id_mismatch")
    if len({dry_run.run_id, review.dry_run_id, admission.dry_run_id, binding["dry_run_id"], str(formal.get("dry_run_id", ""))}) != 1:
        errors.append("dry_run_id_mismatch")
    if len({review.review_manifest_id, admission.review_manifest_id, binding["review_manifest_id"], str(formal.get("review_manifest_id", ""))}) != 1:
        errors.append("review_manifest_id_mismatch")
    if len({admission.admission_request_id, binding["admission_request_id"], str(formal.get("admission_request_id", ""))}) != 1:
        errors.append("admission_request_id_mismatch")

    if int(formal.get("admission_record_count", -1)) != len(admission.admission_records):
        errors.append("admission_record_count_mismatch")
    admit_ids = [record.record_id for record in admission.admission_records if record.action == "admit"]
    exclude_ids = [record.record_id for record in admission.admission_records if record.action == "exclude"]
    if int(formal.get("admit_count", -1)) != len(admit_ids) or binding["admit_count"] != len(admit_ids):
        errors.append("admit_count_mismatch")
    if int(formal.get("exclude_count", -1)) != len(exclude_ids) or binding["exclude_count"] != len(exclude_ids):
        errors.append("exclude_count_mismatch")
    return _stable_unique(errors)


def _draft_payload(
    *,
    materialization_plan_id: str,
    dataset_target: str,
    created_by: str,
    manifest: CustomCorpusManifest,
    dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    formal: dict[str, Any],
    hashes: dict[str, str],
    materialization_records: list[dict[str, Any]],
) -> dict[str, Any]:
    generated = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": _MATERIALIZATION_SCHEMA_VERSION,
        "materialization_plan_id": materialization_plan_id,
        "materialization_run_id": materialization_plan_id,
        "created_at": generated,
        "created_by": created_by,
        "corpus_id": manifest.corpus_id,
        "dry_run_id": dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "materialization_mode": "candidate_only",
        "materialization_decision": "planned",
        "dataset_target": dataset_target,
        "source_manifest_sha256": hashes["manifest_sha256"],
        "source_dry_run_report_sha256": hashes["dry_run_report_sha256"],
        "source_review_manifest_sha256": hashes["review_manifest_sha256"],
        "source_admission_request_sha256": hashes["admission_request_sha256"],
        "source_package_validation_sha256": hashes["formal_package_validation_sha256"],
        "package_validation_status": "passed",
        "package_admission_decision": str(formal.get("admission_decision", "")),
        "dry_run_phase1_status": dry_run.confirmation_boundary.phase1_status,
        "dry_run_dataset_confirmation_confirmed": dry_run.confirmation_boundary.dataset_confirmation_confirmed,
        "dry_run_training_dataset_admitted": dry_run.confirmation_boundary.training_dataset_admitted,
        "confirmation": {
            "confirmed": True,
            "confirmed_by": created_by,
            "confirmed_at": generated,
            "confirmation_source": "property-materialization-plan-draft-builder",
            "manifest_sha256": hashes["manifest_sha256"],
            "dry_run_report_sha256": hashes["dry_run_report_sha256"],
            "review_manifest_sha256": hashes["review_manifest_sha256"],
            "admission_request_sha256": hashes["admission_request_sha256"],
            "package_validation_sha256": hashes["formal_package_validation_sha256"],
            "corpus_id": manifest.corpus_id,
            "dry_run_id": dry_run.run_id,
            "review_manifest_id": review.review_manifest_id,
            "admission_request_id": admission.admission_request_id,
            "reason": "operator confirmed reviewable materialization plan draft output",
        },
        "materialization_records": materialization_records,
        "rollback_policy": "delete generated candidate artifacts only",
        "redaction_policy": "redacted evidence only",
    }


def _materialization_records(
    *,
    materialization_plan_id: str,
    admission: AdmissionRequest,
    admission_sha: str,
    package_sha: str,
    excluded_ids: list[str],
    blocked_ids: list[str],
) -> list[dict[str, Any]]:
    excluded = set(excluded_ids)
    blocked = set(blocked_ids)
    records: list[dict[str, Any]] = []
    for record in admission.admission_records:
        if record.record_id in excluded or record.record_id in blocked:
            continue
        if record.action != "admit" or record.review_decision != "accept":
            continue
        records.append(
            {
                "materialization_record_id": f"{materialization_plan_id}-{record.record_id}",
                "corpus_id": admission.corpus_id,
                "dry_run_id": admission.dry_run_id,
                "review_manifest_id": admission.review_manifest_id,
                "admission_request_id": admission.admission_request_id,
                "admission_record_id": record.admission_record_id,
                "review_id": record.review_id,
                "document_id": record.document_id,
                "record_id": record.record_id,
                "field_name": record.field_name,
                "action": "materialize_candidate",
                "admission_action": record.action,
                "review_decision": record.review_decision,
                "source_artifact_sha256": record.source_artifact_sha256,
                "review_artifact_sha256": record.review_artifact_sha256,
                "admission_request_sha256": admission_sha,
                "package_validation_sha256": package_sha,
                "normalized_value_summary": record.normalized_value_summary,
                "provenance_summary": record.provenance_summary,
                "materialization_reason": "draft materialization plan generated from package-validated property admission record",
                "exclusion_reason": "",
                "notes": "reviewable materialization plan draft only",
            }
        )
    return records


def _summary(
    *,
    draft_status: str,
    materialization_plan_id: str,
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    manifest: CustomCorpusManifest,
    dry_run: CustomCorpusDryRunReport,
    review: ReviewManifest,
    admission: AdmissionRequest,
    formal: dict[str, Any],
    binding: dict[str, Any],
    dataset_target: str,
    materialization_records: list[dict[str, Any]],
    draft_errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "draft_status": draft_status,
        "materialization_plan_id": materialization_plan_id,
        "manifest_path": _basename(paths["manifest_path"], "manifest.json"),
        "manifest_sha256": hashes["manifest_sha256"],
        "dry_run_report_path": _basename(paths["dry_run_report_path"], "dry_run_report.json"),
        "dry_run_report_sha256": hashes["dry_run_report_sha256"],
        "review_manifest_path": _basename(paths["review_manifest_path"], "review_manifest.json"),
        "review_manifest_sha256": hashes["review_manifest_sha256"],
        "admission_request_path": _basename(paths["admission_request_path"], "custom_corpus_admission.draft.json"),
        "admission_request_sha256": hashes["admission_request_sha256"],
        "formal_package_validation_path": _basename(
            paths["formal_package_validation_path"], "custom_corpus_admission_package_validation.json"
        ),
        "formal_package_validation_sha256": hashes["formal_package_validation_sha256"],
        "property_package_binding_summary_path": _basename(
            paths["property_package_binding_summary_path"], "property_package_binding_summary.json"
        ),
        "property_package_binding_summary_sha256": hashes["property_package_binding_summary_sha256"],
        "corpus_id": manifest.corpus_id,
        "dry_run_id": dry_run.run_id,
        "review_manifest_id": review.review_manifest_id,
        "admission_request_id": admission.admission_request_id,
        "review_queue_id": binding.get("review_queue_id", ""),
        "property_candidate_manifest_id": binding.get("property_candidate_manifest_id", ""),
        "dataset_target": dataset_target,
        "package_binding_status": binding["binding_status"],
        "formal_package_validation_status": str(formal.get("validation_status", "")),
        "dry_run_decision": dry_run.decision,
        "phase1_status": dry_run.confirmation_boundary.phase1_status,
        "training_admitted": dry_run.confirmation_boundary.training_dataset_admitted,
        "admission_record_count": len(admission.admission_records),
        "admit_count": len([record for record in admission.admission_records if record.action == "admit"]),
        "exclude_count": len([record for record in admission.admission_records if record.action == "exclude"]),
        "blocked_record_count": len(binding["blocked_record_ids"]),
        "materialization_record_count": len(materialization_records),
        "materialization_record_ids": [record["materialization_record_id"] for record in materialization_records],
        "admit_record_ids": [record.record_id for record in admission.admission_records if record.action == "admit"],
        "exclude_record_ids": [record.record_id for record in admission.admission_records if record.action == "exclude"],
        "blocked_record_ids": binding["blocked_record_ids"],
        "draft_artifacts": dict(_ARTIFACTS),
        "draft_errors": draft_errors,
        "warnings": warnings,
        "redaction_status": "passed",
    }


def _load_dry_run_report(path: str | Path) -> CustomCorpusDryRunReport:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return CustomCorpusDryRunReport.model_validate(payload)
    except Exception as exc:
        raise CustomCorpusPropertyMaterializationPlanDraftError("dry-run report invalid") from exc


def _load_formal_package_validation(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyMaterializationPlanDraftError("formal package validation invalid") from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyMaterializationPlanDraftError("formal package validation invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyMaterializationPlanDraftError("formal package validation invalid")
    return payload


def _load_property_package_binding_summary(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyMaterializationPlanDraftError("package binding summary invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _BINDING_SCHEMA_VERSION:
        raise CustomCorpusPropertyMaterializationPlanDraftError("package binding summary schema_version invalid")
    if _input_contains_forbidden_material(payload):
        raise CustomCorpusPropertyMaterializationPlanDraftError("package binding summary invalid")
    clean = dict(payload)
    if clean.get("binding_status") not in {"passed", "needs_review", "failed"}:
        raise CustomCorpusPropertyMaterializationPlanDraftError("binding_status invalid")
    for field in (
        "corpus_id",
        "dry_run_id",
        "review_manifest_id",
        "admission_request_id",
        "review_queue_id",
        "property_candidate_manifest_id",
    ):
        clean[field] = _optional_safe_id(clean.get(field), field_name=field)
        if field in {"corpus_id", "dry_run_id", "review_manifest_id", "admission_request_id"} and not clean[field]:
            raise CustomCorpusPropertyMaterializationPlanDraftError(f"{field} invalid")
    for field in (
        "manifest_sha256",
        "dry_run_report_sha256",
        "review_manifest_sha256",
        "admission_request_sha256",
        "formal_package_validation_sha256",
        "property_package_binding_summary_sha256",
    ):
        if field in clean:
            clean[field] = _optional_sha(clean.get(field), field_name=field)
    clean["formal_package_validation_status"] = _safe_label(
        clean.get("formal_package_validation_status"), field_name="formal_package_validation_status"
    )
    clean["dry_run_decision"] = _safe_label(clean.get("dry_run_decision"), field_name="dry_run_decision")
    clean["phase1_status"] = _safe_label(clean.get("phase1_status"), field_name="phase1_status")
    clean["training_admitted"] = bool(clean.get("training_admitted"))
    clean["admission_record_count"] = int(clean.get("admission_record_count", 0))
    clean["admit_count"] = int(clean.get("admit_count", 0))
    clean["exclude_count"] = int(clean.get("exclude_count", 0))
    clean["blocked_record_count"] = int(clean.get("blocked_record_count", 0))
    clean["admit_record_ids"] = _safe_id_list(clean.get("admit_record_ids"), field_name="admit_record_ids")
    clean["exclude_record_ids"] = _safe_id_list(clean.get("exclude_record_ids"), field_name="exclude_record_ids")
    clean["blocked_record_ids"] = _safe_id_list(clean.get("blocked_record_ids"), field_name="blocked_record_ids")
    clean["binding_errors"] = _safe_id_list(clean.get("binding_errors"), field_name="binding_errors")
    clean["warnings"] = _safe_id_list(clean.get("warnings"), field_name="warnings")
    return clean


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Materialization Plan Draft Evidence",
            "",
            f"- Materialization plan id: `{summary['materialization_plan_id']}`",
            f"- Draft status: `{summary['draft_status']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Dry-run id: `{summary['dry_run_id']}`",
            f"- Review manifest id: `{summary['review_manifest_id']}`",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Package binding status: `{summary['package_binding_status']}`",
            f"- Formal package validation status: `{summary['formal_package_validation_status']}`",
            f"- Materialization draft record count: `{summary['materialization_record_count']}`",
            f"- Materialization record ids: `{json.dumps(summary['materialization_record_ids'])}`",
            f"- Admit record ids: `{json.dumps(summary['admit_record_ids'])}`",
            f"- Exclude record ids: `{json.dumps(summary['exclude_record_ids'])}`",
            f"- Blocked record ids: `{json.dumps(summary['blocked_record_ids'])}`",
            f"- Draft errors: `{json.dumps(summary['draft_errors'])}`",
            f"- Warnings: `{json.dumps(summary['warnings'])}`",
            "",
            "## Boundary Statement",
            "",
            "- this is a materialization plan draft only.",
            "- No materialization was run.",
            "- No materialization planner was run.",
            "- No materializer was run.",
            "- No candidate/training CSV was created.",
            "- No training data was admitted.",
            "- Phase 1 did not run.",
            "- DatasetConfirmation was not changed.",
            "",
        ]
    )


def _required_safe_id(value: Any, *, field_name: str) -> str:
    clean = _optional_safe_id(value, field_name=field_name)
    if not clean:
        raise CustomCorpusPropertyMaterializationPlanDraftError(f"{field_name} invalid")
    return clean


def _optional_safe_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    if not _SAFE_ID_RE.fullmatch(clean):
        raise CustomCorpusPropertyMaterializationPlanDraftError(f"{field_name} invalid")
    if any(marker.lower() in clean.lower() for marker in _FORBIDDEN_MARKERS):
        raise CustomCorpusPropertyMaterializationPlanDraftError(f"{field_name} invalid")
    return clean


def _safe_created_by(value: Any) -> str:
    clean = _required_safe_id(value, field_name="created_by")
    if "@" in clean and "redacted" not in clean.lower():
        raise CustomCorpusPropertyMaterializationPlanDraftError("created_by invalid")
    return clean


def _safe_label(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_ID_RE.fullmatch(clean):
        raise CustomCorpusPropertyMaterializationPlanDraftError(f"{field_name} invalid")
    return clean


def _safe_id_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CustomCorpusPropertyMaterializationPlanDraftError(f"{field_name} invalid")
    return [_required_safe_id(item, field_name=field_name) for item in value]


def _optional_sha(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise CustomCorpusPropertyMaterializationPlanDraftError(f"{field_name} invalid")
    return f"sha256:{match.group(2).lower()}"


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


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


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
        "draft_status": "blocked",
        "draft_errors": ["property_materialization_plan_draft_redaction_failed"],
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
