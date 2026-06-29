from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_admission import (
    AdmissionRequest,
    CustomCorpusAdmissionError,
    load_admission_request,
)
from ai4s_agent.custom_corpus_dry_run import CustomCorpusDryRunReport
from ai4s_agent.custom_corpus_manifest import CustomCorpusManifest, load_custom_corpus_manifest, sha256_file
from ai4s_agent.custom_corpus_review import ReviewManifest, load_review_manifest


_SCHEMA_VERSION = "custom_corpus_property_admission_draft_package_precheck.v1"
_DRAFT_SUMMARY_SCHEMA_VERSION = "custom_corpus_property_admission_draft_builder.v1"
_REQUEST_PLAN_SCHEMA_VERSION = "custom_corpus_property_admission_request_plan.v1"
_READINESS_SCHEMA_VERSION = "custom_corpus_property_admission_readiness.v1"
_REVIEW_BINDING_SCHEMA_VERSION = "custom_corpus_property_review_binding.v1"
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


class CustomCorpusPropertyAdmissionDraftPackagePrecheckError(ValueError):
    pass


@dataclass(frozen=True)
class _LooseAdmissionRecord:
    admission_record_id: str = ""
    corpus_id: str = ""
    dry_run_id: str = ""
    review_manifest_id: str = ""
    document_id: str = ""
    record_id: str = ""
    field_name: str = ""
    review_id: str = ""
    review_decision: str = ""
    action: str = ""


@dataclass(frozen=True)
class _LooseAdmissionRequest:
    admission_request_id: str = ""
    corpus_id: str = ""
    dry_run_id: str = ""
    source_manifest_sha256: str = ""
    source_dry_run_report_sha256: str = ""
    source_review_manifest_sha256: str = ""
    review_manifest_id: str = ""
    admission_records: list[_LooseAdmissionRecord] | None = None


def precheck_property_admission_draft_package(
    *,
    manifest_path: str | Path,
    dry_run_report_path: str | Path,
    review_manifest_path: str | Path,
    admission_draft_path: str | Path,
    draft_summary_path: str | Path,
    request_plan_summary_path: str | Path,
    readiness_summary_path: str | Path,
    review_binding_summary_path: str | Path,
    require_written_draft: bool = True,
    require_planned_request: bool = False,
    require_ready_readiness: bool = False,
) -> dict[str, Any]:
    manifest = load_custom_corpus_manifest(manifest_path)
    dry_run_report = _load_dry_run_report(dry_run_report_path)
    review_manifest = load_review_manifest(review_manifest_path)
    admission_draft, admission_draft_errors = _load_admission_draft_for_precheck(admission_draft_path)
    draft_summary = _load_summary(draft_summary_path, _DRAFT_SUMMARY_SCHEMA_VERSION, "draft_summary")
    request_plan = _load_summary(request_plan_summary_path, _REQUEST_PLAN_SCHEMA_VERSION, "request_plan_summary")
    readiness = _load_summary(readiness_summary_path, _READINESS_SCHEMA_VERSION, "readiness_summary")
    review_binding = _load_summary(review_binding_summary_path, _REVIEW_BINDING_SCHEMA_VERSION, "review_binding_summary")

    hashes = {
        "manifest_sha256": _safe_sha_for_path(manifest_path),
        "dry_run_report_sha256": _safe_sha_for_path(dry_run_report_path),
        "review_manifest_sha256": _safe_sha_for_path(review_manifest_path),
        "admission_draft_sha256": _safe_sha_for_path(admission_draft_path),
        "draft_summary_sha256": _safe_sha_for_path(draft_summary_path),
        "request_plan_summary_sha256": _safe_sha_for_path(request_plan_summary_path),
        "readiness_summary_sha256": _safe_sha_for_path(readiness_summary_path),
        "review_binding_summary_sha256": _safe_sha_for_path(review_binding_summary_path),
    }

    summary = _build_precheck_summary(
        manifest=manifest,
        dry_run_report=dry_run_report,
        review_manifest=review_manifest,
        admission_draft=admission_draft,
        draft_summary=draft_summary,
        request_plan=request_plan,
        readiness=readiness,
        review_binding=review_binding,
        initial_errors=admission_draft_errors,
        paths={
            "manifest_path": manifest_path,
            "dry_run_report_path": dry_run_report_path,
            "review_manifest_path": review_manifest_path,
            "admission_draft_path": admission_draft_path,
            "draft_summary_path": draft_summary_path,
            "request_plan_summary_path": request_plan_summary_path,
            "readiness_summary_path": readiness_summary_path,
            "review_binding_summary_path": review_binding_summary_path,
        },
        hashes=hashes,
        require_written_draft=require_written_draft,
        require_planned_request=require_planned_request,
        require_ready_readiness=require_ready_readiness,
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
    args = _parser().parse_args(argv)
    try:
        summary = precheck_property_admission_draft_package(
            manifest_path=args.manifest,
            dry_run_report_path=args.dry_run_report,
            review_manifest_path=args.review_manifest,
            admission_draft_path=args.admission_draft,
            draft_summary_path=args.draft_summary,
            request_plan_summary_path=args.request_plan_summary,
            readiness_summary_path=args.readiness_summary,
            review_binding_summary_path=args.review_binding_summary,
            require_written_draft=args.require_written_draft,
            require_planned_request=args.require_planned_request,
            require_ready_readiness=args.require_ready_readiness,
        )
    except Exception as exc:
        err.write(f"property admission draft package precheck invalid: {_safe_exception_message(exc)}\n")
        return 1

    if args.output_summary:
        write_json(Path(args.output_summary).expanduser(), summary)
    if args.output_markdown:
        markdown = _summary_markdown(summary)
        if _contains_forbidden_material(markdown):
            summary = _minimal_redaction_failure()
            output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            output.write("\n")
            return 1
        Path(args.output_markdown).expanduser().write_text(markdown, encoding="utf-8")

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("precheck_status") == "failed" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_admission_draft_package_precheck",
        description="Precheck a property admission draft before formal custom corpus package binding.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run-report", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--admission-draft", required=True)
    parser.add_argument("--draft-summary", required=True)
    parser.add_argument("--request-plan-summary", required=True)
    parser.add_argument("--readiness-summary", required=True)
    parser.add_argument("--review-binding-summary", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-markdown", default="")
    parser.add_argument("--require-written-draft", action="store_true", default=True)
    parser.add_argument("--require-planned-request", action="store_true")
    parser.add_argument("--require-ready-readiness", action="store_true")
    return parser


def _build_precheck_summary(
    *,
    manifest: CustomCorpusManifest,
    dry_run_report: CustomCorpusDryRunReport,
    review_manifest: ReviewManifest,
    admission_draft: AdmissionRequest,
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    review_binding: dict[str, Any],
    initial_errors: list[str],
    paths: dict[str, str | Path],
    hashes: dict[str, str],
    require_written_draft: bool,
    require_planned_request: bool,
    require_ready_readiness: bool,
) -> dict[str, Any]:
    errors: list[str] = list(initial_errors)
    warnings: list[str] = []

    confirmation_boundary = dry_run_report.confirmation_boundary
    admission_ids = [record.record_id for record in admission_draft.admission_records]
    admit_ids = [record.record_id for record in admission_draft.admission_records if record.action == "admit"]
    exclude_ids = [record.record_id for record in admission_draft.admission_records if record.action == "exclude"]
    draft_blocked_ids = _safe_id_list(draft_summary.get("blocked_record_ids"), "blocked_record_ids")

    if dry_run_report.decision != "passed":
        _append_unique(errors, "dry_run_not_passed")
    if confirmation_boundary.dataset_confirmation_confirmed:
        _append_unique(errors, "dry_run_dataset_confirmed")
    if confirmation_boundary.phase1_status != "not_run":
        _append_unique(errors, "dry_run_phase1_ran")
    if confirmation_boundary.training_dataset_admitted:
        _append_unique(errors, "dry_run_training_admitted")

    _check_ids(manifest, dry_run_report, review_manifest, admission_draft, draft_summary, request_plan, readiness, review_binding, errors)
    _check_hashes(
        dry_run_report=dry_run_report,
        review_manifest=review_manifest,
        admission_draft=admission_draft,
        draft_summary=draft_summary,
        request_plan=request_plan,
        readiness=readiness,
        review_binding=review_binding,
        hashes=hashes,
        errors=errors,
    )
    _check_upstream_statuses(
        draft_summary=draft_summary,
        request_plan=request_plan,
        readiness=readiness,
        review_binding=review_binding,
        require_written_draft=require_written_draft,
        require_planned_request=require_planned_request,
        require_ready_readiness=require_ready_readiness,
        errors=errors,
        warnings=warnings,
    )
    _check_record_consistency(
        review_manifest=review_manifest,
        admission_draft=admission_draft,
        draft_summary=draft_summary,
        request_plan=request_plan,
        readiness=readiness,
        review_binding=review_binding,
        admit_ids=admit_ids,
        exclude_ids=exclude_ids,
        admission_ids=admission_ids,
        draft_blocked_ids=draft_blocked_ids,
        errors=errors,
    )

    if errors:
        status = "failed"
    elif warnings:
        status = "needs_review"
    else:
        status = "passed"

    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": status,
        "manifest_path": _basename(paths["manifest_path"], "custom_corpus_manifest.json"),
        "manifest_sha256": hashes["manifest_sha256"],
        "dry_run_report_path": _basename(paths["dry_run_report_path"], "dry_run_report.json"),
        "dry_run_report_sha256": hashes["dry_run_report_sha256"],
        "review_manifest_path": _basename(paths["review_manifest_path"], "property_review_manifest.json"),
        "review_manifest_sha256": hashes["review_manifest_sha256"],
        "admission_draft_path": _basename(paths["admission_draft_path"], "custom_corpus_admission.draft.json"),
        "admission_draft_sha256": hashes["admission_draft_sha256"],
        "draft_summary_path": _basename(paths["draft_summary_path"], "property_admission_draft_summary.json"),
        "draft_summary_sha256": hashes["draft_summary_sha256"],
        "request_plan_summary_path": _basename(
            paths["request_plan_summary_path"], "property_admission_request_plan_summary.json"
        ),
        "request_plan_summary_sha256": hashes["request_plan_summary_sha256"],
        "readiness_summary_path": _basename(paths["readiness_summary_path"], "property_admission_readiness_summary.json"),
        "readiness_summary_sha256": hashes["readiness_summary_sha256"],
        "review_binding_summary_path": _basename(
            paths["review_binding_summary_path"], "property_review_binding_summary.json"
        ),
        "review_binding_summary_sha256": hashes["review_binding_summary_sha256"],
        "corpus_id": manifest.corpus_id,
        "dry_run_id": dry_run_report.run_id,
        "review_manifest_id": review_manifest.review_manifest_id,
        "admission_request_id": admission_draft.admission_request_id,
        "review_queue_id": str(review_binding.get("review_queue_id") or request_plan.get("review_queue_id") or ""),
        "property_candidate_manifest_id": str(
            review_binding.get("property_candidate_manifest_id") or request_plan.get("property_candidate_manifest_id") or ""
        ),
        "dry_run_decision": dry_run_report.decision,
        "phase1_status": confirmation_boundary.phase1_status,
        "training_admitted": confirmation_boundary.training_dataset_admitted,
        "draft_status": str(draft_summary.get("draft_status") or ""),
        "planner_status": str(request_plan.get("planner_status") or ""),
        "readiness_status": str(readiness.get("readiness_status") or ""),
        "binding_status": str(review_binding.get("binding_status") or ""),
        "draft_record_count": len(admission_draft.admission_records),
        "admit_count": len(admit_ids),
        "exclude_count": len(exclude_ids),
        "blocked_record_count": len(draft_blocked_ids),
        "admit_record_ids": admit_ids,
        "exclude_record_ids": exclude_ids,
        "blocked_record_ids": draft_blocked_ids,
        "precheck_errors": errors,
        "warnings": warnings,
        "redaction_status": "passed",
    }


def _check_ids(
    manifest: CustomCorpusManifest,
    dry_run_report: CustomCorpusDryRunReport,
    review_manifest: ReviewManifest,
    admission_draft: AdmissionRequest,
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    review_binding: dict[str, Any],
    errors: list[str],
) -> None:
    corpus_values = [
        manifest.corpus_id,
        dry_run_report.corpus_id,
        review_manifest.corpus_id,
        admission_draft.corpus_id,
        _string_field(draft_summary, "corpus_id"),
        _string_field(request_plan, "corpus_id"),
        _string_field(readiness, "corpus_id"),
        _string_field(review_binding, "corpus_id"),
    ]
    if len(set(corpus_values)) != 1:
        _append_unique(errors, "corpus_id_mismatch")

    dry_run_values = [
        dry_run_report.run_id,
        review_manifest.dry_run_id,
        admission_draft.dry_run_id,
        _string_field(draft_summary, "dry_run_id"),
        _string_field(request_plan, "dry_run_id"),
        _string_field(readiness, "dry_run_id"),
        _string_field(review_binding, "dry_run_id"),
    ]
    if len(set(dry_run_values)) != 1:
        _append_unique(errors, "dry_run_id_mismatch")

    review_values = [
        review_manifest.review_manifest_id,
        admission_draft.review_manifest_id,
        _string_field(draft_summary, "review_manifest_id"),
        _string_field(request_plan, "review_manifest_id"),
        _string_field(readiness, "review_manifest_id"),
        _string_field(review_binding, "review_manifest_id"),
    ]
    if len(set(review_values)) != 1:
        _append_unique(errors, "review_manifest_id_mismatch")

    if admission_draft.admission_request_id != _string_field(draft_summary, "admission_request_id"):
        _append_unique(errors, "admission_request_id_mismatch")

    review_queue_values = [
        _string_field(draft_summary, "review_queue_id"),
        _string_field(request_plan, "review_queue_id"),
        _string_field(readiness, "review_queue_id"),
        _string_field(review_binding, "review_queue_id"),
    ]
    if len(set(review_queue_values)) != 1:
        _append_unique(errors, "review_queue_id_mismatch")

    candidate_manifest_values = [
        _string_field(draft_summary, "property_candidate_manifest_id"),
        _string_field(request_plan, "property_candidate_manifest_id"),
        _string_field(readiness, "property_candidate_manifest_id"),
        _string_field(review_binding, "property_candidate_manifest_id"),
    ]
    if len(set(candidate_manifest_values)) != 1:
        _append_unique(errors, "property_candidate_manifest_id_mismatch")


def _check_hashes(
    *,
    dry_run_report: CustomCorpusDryRunReport,
    review_manifest: ReviewManifest,
    admission_draft: AdmissionRequest,
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    review_binding: dict[str, Any],
    hashes: dict[str, str],
    errors: list[str],
) -> None:
    manifest_sha = hashes["manifest_sha256"]
    dry_sha = hashes["dry_run_report_sha256"]
    review_sha = hashes["review_manifest_sha256"]

    _check_hash_value(dry_run_report.manifest_summary.manifest_sha256, manifest_sha, "source_manifest_sha256_mismatch", errors)
    _check_hash_value(review_manifest.source_manifest_sha256, manifest_sha, "source_manifest_sha256_mismatch", errors)
    _check_hash_value(review_manifest.source_dry_run_report_sha256, dry_sha, "source_dry_run_report_sha256_mismatch", errors)

    _check_hash_value(admission_draft.source_manifest_sha256, manifest_sha, "source_manifest_sha256_mismatch", errors)
    _check_hash_value(admission_draft.source_dry_run_report_sha256, dry_sha, "source_dry_run_report_sha256_mismatch", errors)
    _check_hash_value(admission_draft.source_review_manifest_sha256, review_sha, "source_review_manifest_sha256_mismatch", errors)

    for payload in (draft_summary, request_plan, readiness, review_binding):
        _check_hash_value(payload.get("source_manifest_sha256"), manifest_sha, "source_manifest_sha256_mismatch", errors)
        _check_hash_value(payload.get("source_dry_run_report_sha256"), dry_sha, "source_dry_run_report_sha256_mismatch", errors)
        _check_hash_value(payload.get("review_manifest_sha256"), review_sha, "source_review_manifest_sha256_mismatch", errors)


def _check_upstream_statuses(
    *,
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    review_binding: dict[str, Any],
    require_written_draft: bool,
    require_planned_request: bool,
    require_ready_readiness: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    if require_written_draft and draft_summary.get("draft_status") != "written":
        _append_unique(errors, "draft_not_written")
    if request_plan.get("planner_status") == "blocked":
        _append_unique(errors, "request_plan_blocked")
    if readiness.get("readiness_status") == "blocked":
        _append_unique(errors, "readiness_blocked")
    if review_binding.get("binding_status") == "failed":
        _append_unique(errors, "review_binding_failed")

    if request_plan.get("planner_status") == "partial":
        if require_planned_request:
            _append_unique(errors, "request_plan_not_planned")
        else:
            _append_unique(warnings, "request_plan_partial")
    if readiness.get("readiness_status") == "partial":
        if require_ready_readiness:
            _append_unique(errors, "readiness_not_ready")
        else:
            _append_unique(warnings, "readiness_partial")
    if review_binding.get("binding_status") == "needs_review":
        _append_unique(warnings, "review_binding_needs_review")
    if _safe_id_list(draft_summary.get("draft_errors"), "draft_errors"):
        _append_unique(errors, "draft_summary_has_errors")
    if _safe_id_list(request_plan.get("planning_errors"), "planning_errors"):
        _append_unique(errors, "request_plan_has_planning_errors")
    if _safe_id_list(readiness.get("readiness_errors"), "readiness_errors"):
        _append_unique(errors, "readiness_has_errors")
    if _safe_id_list(review_binding.get("binding_errors"), "binding_errors"):
        _append_unique(errors, "review_binding_has_errors")


def _check_record_consistency(
    *,
    review_manifest: ReviewManifest,
    admission_draft: AdmissionRequest,
    draft_summary: dict[str, Any],
    request_plan: dict[str, Any],
    readiness: dict[str, Any],
    review_binding: dict[str, Any],
    admit_ids: list[str],
    exclude_ids: list[str],
    admission_ids: list[str],
    draft_blocked_ids: list[str],
    errors: list[str],
) -> None:
    admission_id_set = set(admission_ids)
    draft_admit_ids = _safe_id_list(draft_summary.get("draft_admit_record_ids"), "draft_admit_record_ids")
    draft_exclude_ids = _safe_id_list(draft_summary.get("draft_exclude_record_ids"), "draft_exclude_record_ids")
    request_admit_ids = _safe_id_list(request_plan.get("planned_admit_record_ids"), "planned_admit_record_ids")
    request_exclude_ids = _safe_id_list(request_plan.get("planned_exclude_record_ids"), "planned_exclude_record_ids")
    request_blocked_ids = _safe_id_list(request_plan.get("blocked_record_ids"), "blocked_record_ids")
    readiness_admit_ids = _safe_id_list(
        readiness.get("planned_admission_candidate_record_ids"),
        "planned_admission_candidate_record_ids",
    )
    readiness_exclude_ids = _safe_id_list(readiness.get("planned_exclusion_record_ids"), "planned_exclusion_record_ids")
    readiness_blocked_ids = _safe_id_list(
        readiness.get("blocked_from_admission_record_ids"),
        "blocked_from_admission_record_ids",
    )
    reviewed_queue_ids = _safe_id_list(review_binding.get("reviewed_queue_record_ids"), "reviewed_queue_record_ids")
    reviewed_blocked_ids = _safe_id_list(review_binding.get("reviewed_blocked_record_ids"), "reviewed_blocked_record_ids")
    unknown_review_ids = _safe_id_list(review_binding.get("unknown_review_record_ids"), "unknown_review_record_ids")

    if set(admit_ids) != set(draft_admit_ids) or set(exclude_ids) != set(draft_exclude_ids):
        _append_unique(errors, "draft_summary_record_ids_mismatch")
    if set(request_admit_ids) != set(draft_admit_ids) or set(request_exclude_ids) != set(draft_exclude_ids):
        _append_unique(errors, "request_plan_draft_ids_mismatch")
    if set(readiness_admit_ids) != set(request_admit_ids) or set(readiness_exclude_ids) != set(request_exclude_ids):
        _append_unique(errors, "readiness_request_plan_ids_mismatch")
    if not set(draft_blocked_ids).issubset(set(request_blocked_ids) | set(readiness_blocked_ids)):
        _append_unique(errors, "blocked_record_ids_mismatch")
    if admission_id_set & set(draft_blocked_ids):
        _append_unique(errors, "blocked_record_in_admission_draft")
    if not admission_id_set.issubset(set(reviewed_queue_ids)):
        _append_unique(errors, "review_binding_missing_admission_records")
    if admission_id_set & set(reviewed_blocked_ids):
        _append_unique(errors, "reviewed_blocked_record_in_admission_draft")
    if admission_id_set & set(unknown_review_ids):
        _append_unique(errors, "unknown_review_record_in_admission_draft")

    review_by_id = {record.review_id: record for record in review_manifest.review_records}
    review_by_record_id = {record.record_id: record for record in review_manifest.review_records}
    if not admission_draft.admission_records:
        _append_unique(errors, "invalid_admission_draft")
    for admission_record in admission_draft.admission_records:
        review_record = review_by_id.get(admission_record.review_id)
        if review_record is None:
            _append_unique(errors, "admission_review_record_missing")
            continue
        if review_by_record_id.get(admission_record.record_id) is None:
            _append_unique(errors, "admission_review_record_missing")
        if review_record.record_id != admission_record.record_id:
            _append_unique(errors, "review_record_target_mismatch")
        if review_record.decision != admission_record.review_decision:
            _append_unique(errors, "invalid_admission_draft")
        if admission_record.action == "admit" and admission_record.review_decision != "accept":
            _append_unique(errors, "invalid_admission_draft")
        if admission_record.action == "exclude" and admission_record.review_decision != "reject":
            _append_unique(errors, "invalid_admission_draft")
        if review_record.decision == "needs_review" and admission_record.action in {"admit", "exclude"}:
            _append_unique(errors, "needs_review_record_in_admission_draft")


def _load_admission_draft_for_precheck(path: str | Path) -> tuple[AdmissionRequest | _LooseAdmissionRequest, list[str]]:
    try:
        return load_admission_request(path), []
    except CustomCorpusAdmissionError as exc:
        try:
            payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        except Exception as read_exc:
            raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(
                f"could not read admission draft: {read_exc.__class__.__name__}"
            ) from read_exc
        if _contains_forbidden_material(payload):
            raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError("admission draft invalid") from exc
        if not isinstance(payload, dict):
            raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError("admission draft invalid") from exc
        records = []
        for item in payload.get("admission_records") or []:
            if not isinstance(item, dict):
                continue
            records.append(
                _LooseAdmissionRecord(
                    admission_record_id=_loose_safe_text(item.get("admission_record_id")),
                    corpus_id=_loose_safe_text(item.get("corpus_id")),
                    dry_run_id=_loose_safe_text(item.get("dry_run_id")),
                    review_manifest_id=_loose_safe_text(item.get("review_manifest_id")),
                    document_id=_loose_safe_text(item.get("document_id")),
                    record_id=_loose_safe_text(item.get("record_id")),
                    field_name=_loose_safe_text(item.get("field_name")),
                    review_id=_loose_safe_text(item.get("review_id")),
                    review_decision=_loose_safe_text(item.get("review_decision")),
                    action=_loose_safe_text(item.get("action")),
                )
            )
        return (
            _LooseAdmissionRequest(
                admission_request_id=_loose_safe_text(payload.get("admission_request_id")),
                corpus_id=_loose_safe_text(payload.get("corpus_id")),
                dry_run_id=_loose_safe_text(payload.get("dry_run_id")),
                source_manifest_sha256=_loose_safe_text(payload.get("source_manifest_sha256")),
                source_dry_run_report_sha256=_loose_safe_text(payload.get("source_dry_run_report_sha256")),
                source_review_manifest_sha256=_loose_safe_text(payload.get("source_review_manifest_sha256")),
                review_manifest_id=_loose_safe_text(payload.get("review_manifest_id")),
                admission_records=records,
            ),
            ["invalid_admission_draft"],
        )


def _load_dry_run_report(path: str | Path) -> CustomCorpusDryRunReport:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(
            f"could not read dry-run report: {exc.__class__.__name__}"
        ) from exc
    try:
        return CustomCorpusDryRunReport.model_validate(payload)
    except Exception as exc:
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(_safe_exception_message(exc)) from exc


def _load_summary(path: str | Path, expected_schema: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(
            f"could not read {label}: {exc.__class__.__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{label} is invalid")
    if payload.get("schema_version") != expected_schema:
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{label} schema_version is invalid")
    _validate_summary_payload(payload, label=label)
    return payload


def _validate_summary_payload(payload: dict[str, Any], *, label: str) -> None:
    for key, value in payload.items():
        if key.endswith("_id") or key in {
            "review_queue_id",
            "property_candidate_manifest_id",
            "corpus_id",
            "dry_run_id",
            "review_manifest_id",
            "admission_request_id",
        }:
            if isinstance(value, str) and value and not _SAFE_ID_RE.fullmatch(value):
                raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{label} contains unsafe id")
        if key.endswith("_sha256") and value:
            _normalize_sha(value, field_name=key)
        if key.endswith("_ids") or key.endswith("_errors") or key == "warnings":
            if not isinstance(value, list):
                raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{label} contains invalid list")
            for item in value:
                if not isinstance(item, str) or not item or not _SAFE_ID_RE.fullmatch(item):
                    raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{label} contains unsafe list value")


def _string_field(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{field_name} is invalid")
    if not _SAFE_ID_RE.fullmatch(value):
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{field_name} is invalid")
    return value


def _safe_id_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{field_name} is invalid")
    result: list[str] = []
    for item in value:
        clean = str(item or "").strip()
        if not clean or not _SAFE_ID_RE.fullmatch(clean):
            raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{field_name} is invalid")
        result.append(clean)
    return result


def _loose_safe_text(value: Any) -> str:
    clean = str(value or "").strip()
    if any(marker.lower() in clean.lower() for marker in _FORBIDDEN_MARKERS):
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError("admission draft invalid")
    if _ABSOLUTE_PATH_VALUE_RE.search(json.dumps(clean)):
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError("admission draft invalid")
    return clean


def _check_hash_value(value: Any, expected: str, error_code: str, errors: list[str]) -> None:
    clean = _normalize_sha(value, field_name="sha256", allow_empty=True)
    if clean and clean != expected:
        _append_unique(errors, error_code)


def _normalize_sha(value: Any, *, field_name: str, allow_empty: bool = False) -> str:
    clean = str(value or "").strip()
    if not clean and allow_empty:
        return ""
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise CustomCorpusPropertyAdmissionDraftPackagePrecheckError(f"{field_name} is invalid")
    return f"sha256:{match.group(2).lower()}"


def _safe_sha_for_path(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except Exception:
        return ""


def _basename(path: str | Path, fallback: str) -> str:
    return Path(path).name or fallback


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(json.dumps(serialized)))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "precheck_status": "failed",
        "precheck_errors": ["property_admission_draft_package_precheck_redaction_failed"],
        "redaction_status": "failed",
    }


def _safe_exception_message(exc: BaseException) -> str:
    if isinstance(exc, CustomCorpusAdmissionError):
        return "admission draft invalid"
    message = str(exc or "").lower()
    if any(marker.lower() in message for marker in _FORBIDDEN_MARKERS):
        return "artifact invalid"
    if _ABSOLUTE_PATH_VALUE_RE.search(json.dumps(message)):
        return "artifact invalid"
    if "schema_version" in message:
        return "schema_version invalid"
    if "invalid" in message:
        return "artifact invalid"
    return exc.__class__.__name__


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Admission Draft Package Precheck",
            "",
            f"- Precheck status: `{summary['precheck_status']}`",
            f"- Corpus id: `{summary.get('corpus_id', '')}`",
            f"- Dry-run id: `{summary.get('dry_run_id', '')}`",
            f"- Review manifest id: `{summary.get('review_manifest_id', '')}`",
            f"- Admission request id: `{summary.get('admission_request_id', '')}`",
            f"- Draft status: `{summary.get('draft_status', '')}`",
            f"- Planner status: `{summary.get('planner_status', '')}`",
            f"- Readiness status: `{summary.get('readiness_status', '')}`",
            f"- Binding status: `{summary.get('binding_status', '')}`",
            f"- Draft record count: `{summary.get('draft_record_count', 0)}`",
            f"- Admit record ids: `{json.dumps(summary.get('admit_record_ids', []))}`",
            f"- Exclude record ids: `{json.dumps(summary.get('exclude_record_ids', []))}`",
            f"- Blocked record ids: `{json.dumps(summary.get('blocked_record_ids', []))}`",
            f"- Precheck errors: `{json.dumps(summary.get('precheck_errors', []))}`",
            f"- Warnings: `{json.dumps(summary.get('warnings', []))}`",
            "",
            "## Boundary Statement",
            "",
            "- This is a package precheck only.",
            "- formal package binding was not run.",
            "- No `custom_corpus_admission_package_validation.v1` was created.",
            "- No materialization was run.",
            "- No candidate/training CSV was created.",
            "- Phase 1 did not run.",
            "- DatasetConfirmation was not changed.",
            "- No training data was admitted.",
            "",
        ]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
