from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_review import (
    CustomCorpusReviewError,
    ReviewManifest,
    load_review_manifest,
    sha256_file,
)


_SCHEMA_VERSION = "custom_corpus_property_admission_request_plan.v1"
_READINESS_SCHEMA_VERSION = "custom_corpus_property_admission_readiness.v1"
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


class CustomCorpusPropertyAdmissionRequestPlannerError(ValueError):
    pass


def plan_property_admission_request(
    *,
    admission_readiness_summary_path: str | Path,
    review_manifest_path: str | Path,
    require_ready_status: bool = False,
) -> dict[str, Any]:
    readiness = _load_readiness_summary(admission_readiness_summary_path)
    review_manifest = load_review_manifest(review_manifest_path)
    summary = _request_plan_summary(
        readiness=readiness,
        review_manifest=review_manifest,
        admission_readiness_summary_path=admission_readiness_summary_path,
        review_manifest_path=review_manifest_path,
        require_ready_status=require_ready_status,
    )
    return _fail_closed_if_unsafe(summary)


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
        summary = plan_property_admission_request(
            admission_readiness_summary_path=args.admission_readiness_summary,
            review_manifest_path=args.review_manifest,
            require_ready_status=args.require_ready_status,
        )
    except CustomCorpusReviewError as exc:
        err.write(f"property admission request plan review manifest invalid: {exc}\n")
        return 1
    except CustomCorpusPropertyAdmissionRequestPlannerError as exc:
        err.write(f"property admission request plan readiness summary invalid: {exc}\n")
        return 1

    if summary.get("redaction_status") == "failed":
        output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        output.write("\n")
        return 1
    if args.output_summary:
        write_json(Path(args.output_summary).expanduser(), summary)
    if args.output_markdown:
        markdown = _markdown_summary(summary)
        if _contains_forbidden_material(markdown):
            summary = _minimal_redaction_failure()
            output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            output.write("\n")
            return 1
        Path(args.output_markdown).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_markdown).expanduser().write_text(markdown, encoding="utf-8")
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("planner_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_admission_request_planner",
        description="Plan a future property admission request from readiness and human review artifacts.",
    )
    parser.add_argument("--admission-readiness-summary", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-markdown", default="")
    parser.add_argument("--require-ready-status", action="store_true")
    return parser


def _request_plan_summary(
    *,
    readiness: dict[str, Any],
    review_manifest: ReviewManifest,
    admission_readiness_summary_path: str | Path,
    review_manifest_path: str | Path,
    require_ready_status: bool,
) -> dict[str, Any]:
    planning_errors: list[str] = []
    warnings: list[str] = []

    if readiness["readiness_status"] == "blocked":
        planning_errors.append("readiness_status_blocked")
    if readiness["readiness_status"] == "partial" and require_ready_status:
        planning_errors.append("readiness_status_not_ready")
    if readiness["readiness_errors"]:
        planning_errors.append("readiness_errors_present")
    if review_manifest.review_manifest_id != readiness["review_manifest_id"]:
        planning_errors.append("review_manifest_id_mismatch")
    if review_manifest.corpus_id != readiness["corpus_id"]:
        planning_errors.append("corpus_id_mismatch")
    if review_manifest.dry_run_id != readiness["dry_run_id"]:
        planning_errors.append("dry_run_id_mismatch")
    if readiness["review_manifest_sha256"] and readiness["review_manifest_sha256"] != _safe_sha_for_path(review_manifest_path):
        planning_errors.append("review_manifest_sha256_mismatch")
    if review_manifest.source_manifest_sha256 and review_manifest.source_manifest_sha256 != readiness["source_manifest_sha256"]:
        planning_errors.append("source_manifest_sha256_mismatch")
    if review_manifest.source_dry_run_report_sha256 != readiness["source_dry_run_report_sha256"]:
        planning_errors.append("source_dry_run_report_sha256_mismatch")

    planned_admission_candidate_ids = set(readiness["planned_admission_candidate_record_ids"])
    planned_exclusion_ids = set(readiness["planned_exclusion_record_ids"])
    blocked_from_admission_ids = set(readiness["blocked_from_admission_record_ids"])
    planned_admit_ids: list[str] = []
    planned_exclude_ids: list[str] = []
    blocked_ids: list[str] = []
    planned_record_summaries: list[dict[str, Any]] = []

    for record in review_manifest.review_records:
        record_id = record.record_id
        if record_id in planned_admission_candidate_ids:
            action = "admit"
            blocking_reason = ""
            planned_reason = "accepted review record is ready for future admission request planning"
            if record_id in planned_exclusion_ids and record.decision != "reject":
                _append_unique(planning_errors, "planned_exclusion_review_decision_invalid")
                blocking_reason = "planned_exclusion_review_decision_invalid"
            if record.decision != "accept":
                _append_unique(planning_errors, "planned_admit_review_decision_invalid")
                blocking_reason = blocking_reason or "planned_admit_review_decision_invalid"
            if not record.extracted_value_summary:
                _append_unique(planning_errors, "planned_admit_missing_extracted_value_summary")
                blocking_reason = blocking_reason or "planned_admit_missing_extracted_value_summary"
            if not record.normalized_value_summary:
                _append_unique(planning_errors, "planned_admit_missing_normalized_value_summary")
                blocking_reason = blocking_reason or "planned_admit_missing_normalized_value_summary"
            if not record.provenance_note:
                _append_unique(planning_errors, "planned_admit_missing_provenance_note")
                blocking_reason = blocking_reason or "planned_admit_missing_provenance_note"
            if not record.source_artifact_sha256:
                _append_unique(planning_errors, "planned_admit_missing_source_artifact_sha256")
                blocking_reason = blocking_reason or "planned_admit_missing_source_artifact_sha256"
            if blocking_reason:
                action = "blocked"
                _append_unique(blocked_ids, record_id)
            else:
                _append_unique(planned_admit_ids, record_id)
            planned_record_summaries.append(
                _planned_record_summary(
                    review_manifest_sha256=_safe_sha_for_path(review_manifest_path),
                    review_id=record.review_id,
                    record_id=record_id,
                    document_id=record.document_id,
                    field_name=record.field_name,
                    review_decision=record.decision,
                    planned_action=action,
                    planned_reason=planned_reason if action == "admit" else "",
                    source_artifact_sha256=record.source_artifact_sha256,
                    normalized_value_summary=record.normalized_value_summary if action == "admit" else "",
                    provenance_summary=record.provenance_note if action == "admit" else "",
                    blocking_reason=blocking_reason,
                )
            )
            continue

        if record_id in planned_exclusion_ids:
            action = "exclude"
            blocking_reason = ""
            planned_reason = record.rejection_reason or "reviewed record is planned for future exclusion"
            if record.decision != "reject":
                _append_unique(planning_errors, "planned_exclusion_review_decision_invalid")
                action = "blocked"
                blocking_reason = "planned_exclusion_review_decision_invalid"
                _append_unique(blocked_ids, record_id)
            else:
                _append_unique(planned_exclude_ids, record_id)
            planned_record_summaries.append(
                _planned_record_summary(
                    review_manifest_sha256=_safe_sha_for_path(review_manifest_path),
                    review_id=record.review_id,
                    record_id=record_id,
                    document_id=record.document_id,
                    field_name=record.field_name,
                    review_decision=record.decision,
                    planned_action=action,
                    planned_reason=planned_reason if action == "exclude" else "",
                    source_artifact_sha256=record.source_artifact_sha256,
                    normalized_value_summary="",
                    provenance_summary="",
                    blocking_reason=blocking_reason,
                )
            )
            continue

        if record_id in blocked_from_admission_ids or record.decision == "needs_review":
            _append_unique(blocked_ids, record_id)
            planned_record_summaries.append(
                _planned_record_summary(
                    review_manifest_sha256=_safe_sha_for_path(review_manifest_path),
                    review_id=record.review_id,
                    record_id=record_id,
                    document_id=record.document_id,
                    field_name=record.field_name,
                    review_decision=record.decision,
                    planned_action="blocked",
                    planned_reason="",
                    source_artifact_sha256=record.source_artifact_sha256,
                    normalized_value_summary="",
                    provenance_summary="",
                    blocking_reason="blocked_from_admission_or_needs_review",
                )
            )
            continue

        warnings.append("review_record_not_in_readiness_plan")

    if not planned_admit_ids and not planned_exclude_ids:
        _append_unique(planning_errors, "no_planned_admission_records")

    if planning_errors:
        planner_status = "blocked"
    elif readiness["readiness_status"] == "partial":
        planner_status = "partial"
    else:
        planner_status = "planned"

    decisions = [record.decision for record in review_manifest.review_records]
    return {
        "schema_version": _SCHEMA_VERSION,
        "planner_status": planner_status,
        "admission_readiness_summary_path": Path(admission_readiness_summary_path).name
        or "property_admission_readiness_summary.json",
        "admission_readiness_summary_sha256": _safe_sha_for_path(admission_readiness_summary_path),
        "review_manifest_path": Path(review_manifest_path).name or "property_review_manifest.json",
        "review_manifest_sha256": _safe_sha_for_path(review_manifest_path),
        "review_queue_id": readiness["review_queue_id"],
        "property_candidate_manifest_id": readiness["property_candidate_manifest_id"],
        "review_manifest_id": readiness["review_manifest_id"],
        "corpus_id": readiness["corpus_id"],
        "dry_run_id": readiness["dry_run_id"],
        "readiness_status": readiness["readiness_status"],
        "binding_status": readiness["binding_status"],
        "require_ready_status": require_ready_status,
        "review_record_count": len(review_manifest.review_records),
        "accepted_review_count": decisions.count("accept"),
        "rejected_review_count": decisions.count("reject"),
        "needs_review_count": decisions.count("needs_review"),
        "planned_admit_count": len(planned_admit_ids),
        "planned_exclude_count": len(planned_exclude_ids),
        "blocked_count": len(blocked_ids),
        "planned_admit_record_ids": planned_admit_ids,
        "planned_exclude_record_ids": planned_exclude_ids,
        "blocked_record_ids": blocked_ids,
        "unreviewed_queue_record_ids": list(readiness["unreviewed_queue_record_ids"]),
        "readiness_errors": list(readiness["readiness_errors"]),
        "planning_errors": planning_errors,
        "warnings": warnings,
        "source_manifest_sha256": readiness["source_manifest_sha256"],
        "source_dry_run_report_sha256": readiness["source_dry_run_report_sha256"],
        "planned_record_summaries": planned_record_summaries,
        "redaction_status": "passed",
    }


def _planned_record_summary(
    *,
    review_manifest_sha256: str,
    review_id: str,
    record_id: str,
    document_id: str,
    field_name: str,
    review_decision: str,
    planned_action: str,
    planned_reason: str,
    source_artifact_sha256: str,
    normalized_value_summary: str,
    provenance_summary: str,
    blocking_reason: str,
) -> dict[str, Any]:
    return {
        "planned_admission_plan_record_id": f"plan-{review_id}",
        "source_review_id": review_id,
        "record_id": record_id,
        "document_id": document_id,
        "field_name": field_name,
        "review_decision": review_decision,
        "planned_action": planned_action,
        "planned_reason": planned_reason,
        "source_artifact_sha256": source_artifact_sha256,
        "review_manifest_sha256": review_manifest_sha256,
        "normalized_value_summary": normalized_value_summary,
        "provenance_summary": provenance_summary,
        "blocking_reason": blocking_reason,
    }


def _load_readiness_summary(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyAdmissionRequestPlannerError(
            f"could not read readiness summary: {exc.__class__.__name__}"
        ) from exc
    return _validate_readiness_summary(payload)


def _validate_readiness_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CustomCorpusPropertyAdmissionRequestPlannerError("readiness summary must be an object")
    if value.get("schema_version") != _READINESS_SCHEMA_VERSION:
        raise CustomCorpusPropertyAdmissionRequestPlannerError("schema_version is invalid")
    clean = dict(value)
    if clean.get("readiness_status") not in {"ready", "partial", "blocked"}:
        raise CustomCorpusPropertyAdmissionRequestPlannerError("readiness_status is invalid")
    if clean.get("binding_status") not in {"passed", "needs_review", "failed"}:
        raise CustomCorpusPropertyAdmissionRequestPlannerError("binding_status is invalid")
    for field in ("review_queue_id", "property_candidate_manifest_id", "review_manifest_id", "corpus_id", "dry_run_id"):
        clean[field] = _required_safe_id(clean.get(field), field_name=field)
    clean["review_binding_summary_sha256"] = _optional_sha(
        clean.get("review_binding_summary_sha256"),
        field_name="review_binding_summary_sha256",
    )
    clean["review_manifest_sha256"] = _optional_sha(
        clean.get("review_manifest_sha256"),
        field_name="review_manifest_sha256",
    )
    clean["source_manifest_sha256"] = _optional_sha(clean.get("source_manifest_sha256"), field_name="source_manifest_sha256")
    clean["source_dry_run_report_sha256"] = _required_sha(
        clean.get("source_dry_run_report_sha256"),
        field_name="source_dry_run_report_sha256",
    )
    for field in (
        "planned_admission_candidate_record_ids",
        "planned_exclusion_record_ids",
        "blocked_from_admission_record_ids",
        "unreviewed_queue_record_ids",
        "reviewed_blocked_record_ids",
        "unknown_review_record_ids",
        "readiness_errors",
    ):
        clean[field] = _safe_string_list(clean.get(field), field_name=field)
    return clean


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Custom Corpus Property Admission Request Plan",
        "",
        f"- Planner status: `{summary['planner_status']}`",
        f"- Readiness status: `{summary['readiness_status']}`",
        f"- Binding status: `{summary['binding_status']}`",
        f"- Review manifest id: `{summary['review_manifest_id']}`",
        f"- Review queue id: `{summary['review_queue_id']}`",
        f"- Review record count: `{summary['review_record_count']}`",
        f"- Planned admit count: `{summary['planned_admit_count']}`",
        f"- Planned exclude count: `{summary['planned_exclude_count']}`",
        f"- Blocked count: `{summary['blocked_count']}`",
        f"- Planned admit record IDs: `{json.dumps(summary['planned_admit_record_ids'])}`",
        f"- Planned exclude record IDs: `{json.dumps(summary['planned_exclude_record_ids'])}`",
        f"- Blocked record IDs: `{json.dumps(summary['blocked_record_ids'])}`",
        f"- Planning errors: `{json.dumps(summary['planning_errors'])}`",
        "",
        "## Boundary Statement",
        "",
        "- No admission request created.",
        "- No admission action created.",
        "- No `custom_corpus_admission.v1` created.",
        "- No materialization.",
        "- No candidate/training CSV.",
        "- No Phase 1.",
        "- No DatasetConfirmation change.",
        "",
    ]
    return "\n".join(lines)


def _safe_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise CustomCorpusPropertyAdmissionRequestPlannerError(f"{field_name} is invalid")
    return [_required_safe_id(item, field_name=field_name) for item in value]


def _required_safe_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_ID_RE.fullmatch(clean):
        raise CustomCorpusPropertyAdmissionRequestPlannerError(f"{field_name} is invalid")
    if any(marker.lower() in clean.lower() for marker in _FORBIDDEN_MARKERS):
        raise CustomCorpusPropertyAdmissionRequestPlannerError(f"{field_name} is invalid")
    return clean


def _required_sha(value: Any, *, field_name: str) -> str:
    clean = _optional_sha(value, field_name=field_name)
    if not clean:
        raise CustomCorpusPropertyAdmissionRequestPlannerError(f"{field_name} is required")
    return clean


def _optional_sha(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise CustomCorpusPropertyAdmissionRequestPlannerError(f"{field_name} is invalid")
    return f"sha256:{match.group(2).lower()}"


def _safe_sha_for_path(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except Exception:
        return ""


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _fail_closed_if_unsafe(summary: dict[str, Any]) -> dict[str, Any]:
    if _contains_forbidden_material(summary):
        return _minimal_redaction_failure()
    return summary


def _contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(serialized))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "planner_status": "blocked",
        "planning_errors": ["property_admission_request_plan_summary_redaction_failed"],
        "redaction_status": "failed",
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
