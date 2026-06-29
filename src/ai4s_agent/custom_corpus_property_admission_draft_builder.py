from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_admission import CustomCorpusAdmissionError, validate_admission_request
from ai4s_agent.custom_corpus_review import (
    CustomCorpusReviewError,
    ReviewManifest,
    load_review_manifest,
    sha256_file,
)


_SCHEMA_VERSION = "custom_corpus_property_admission_draft_builder.v1"
_PLAN_SCHEMA_VERSION = "custom_corpus_property_admission_request_plan.v1"
_ADMISSION_SCHEMA_VERSION = "custom_corpus_admission.v1"
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
_FREE_TEXT_LIMIT = 500
_ARTIFACTS = {
    "custom_corpus_admission_draft_json": "custom_corpus_admission.draft.json",
    "property_admission_draft_summary_json": "property_admission_draft_summary.json",
    "redacted_property_admission_draft_evidence_md": "redacted_property_admission_draft_evidence.md",
}


class CustomCorpusPropertyAdmissionDraftBuilderError(ValueError):
    pass


def build_property_admission_draft(
    *,
    admission_request_plan_path: str | Path,
    review_manifest_path: str | Path,
    output_dir: str | Path,
    admission_request_id: str,
    dataset_target: str,
    created_by: str,
    confirm_admission_draft_output: bool,
    allow_partial_plan: bool = False,
) -> dict[str, Any]:
    plan = _load_request_plan(admission_request_plan_path)
    review_manifest = load_review_manifest(review_manifest_path)
    admission_request_id = _required_safe_id(admission_request_id, field_name="admission_request_id")
    dataset_target = _required_safe_id(dataset_target, field_name="dataset_target")
    created_by = _safe_created_by(created_by)
    run_dir = Path(output_dir).expanduser() / admission_request_id

    summary, draft_payload = _build_summary_and_draft(
        plan=plan,
        review_manifest=review_manifest,
        admission_request_plan_path=admission_request_plan_path,
        review_manifest_path=review_manifest_path,
        admission_request_id=admission_request_id,
        dataset_target=dataset_target,
        created_by=created_by,
        confirm_admission_draft_output=confirm_admission_draft_output,
        allow_partial_plan=allow_partial_plan,
        run_dir=run_dir,
    )

    if summary["draft_status"] == "blocked":
        return _minimal_redaction_failure() if _contains_forbidden_material(summary) else summary

    evidence = _evidence_markdown(summary)
    if _contains_forbidden_material({"summary": summary, "draft": draft_payload, "evidence": evidence}):
        return _minimal_redaction_failure()

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / _ARTIFACTS["custom_corpus_admission_draft_json"], draft_payload)
    write_json(run_dir / _ARTIFACTS["property_admission_draft_summary_json"], summary)
    (run_dir / _ARTIFACTS["redacted_property_admission_draft_evidence_md"]).write_text(evidence, encoding="utf-8")
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
        summary = build_property_admission_draft(
            admission_request_plan_path=args.admission_request_plan,
            review_manifest_path=args.review_manifest,
            output_dir=args.output_dir,
            admission_request_id=args.admission_request_id,
            dataset_target=args.dataset_target,
            created_by=args.created_by,
            confirm_admission_draft_output=args.confirm_admission_draft_output,
            allow_partial_plan=args.allow_partial_plan,
        )
    except CustomCorpusReviewError as exc:
        err.write(f"property admission draft review manifest invalid: {exc}\n")
        return 1
    except CustomCorpusPropertyAdmissionDraftBuilderError as exc:
        err.write(f"property admission draft request plan invalid: {exc}\n")
        return 1

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("draft_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_admission_draft_builder",
        description="Build a reviewable draft custom corpus admission request from a property request plan.",
    )
    parser.add_argument("--admission-request-plan", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--admission-request-id", required=True)
    parser.add_argument("--dataset-target", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--confirm-admission-draft-output", action="store_true")
    parser.add_argument("--allow-partial-plan", action="store_true")
    return parser


def _build_summary_and_draft(
    *,
    plan: dict[str, Any],
    review_manifest: ReviewManifest,
    admission_request_plan_path: str | Path,
    review_manifest_path: str | Path,
    admission_request_id: str,
    dataset_target: str,
    created_by: str,
    confirm_admission_draft_output: bool,
    allow_partial_plan: bool,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    draft_errors: list[str] = []
    warnings: list[str] = []
    actual_review_sha = _safe_sha_for_path(review_manifest_path)

    if not confirm_admission_draft_output:
        draft_errors.append("admission_draft_output_not_confirmed")
    if run_dir.exists() and any(run_dir.iterdir()):
        draft_errors.append("output_directory_not_clean")
    if plan["planner_status"] == "blocked":
        draft_errors.append("request_plan_blocked")
    if plan["planner_status"] == "partial" and not allow_partial_plan:
        draft_errors.append("partial_plan_requires_allow_partial_plan")
    if plan["planning_errors"]:
        draft_errors.append("request_plan_has_planning_errors")
    if review_manifest.review_manifest_id != plan["review_manifest_id"]:
        draft_errors.append("review_manifest_id_mismatch")
    if review_manifest.corpus_id != plan["corpus_id"]:
        draft_errors.append("corpus_id_mismatch")
    if review_manifest.dry_run_id != plan["dry_run_id"]:
        draft_errors.append("dry_run_id_mismatch")
    if plan["review_manifest_sha256"] and plan["review_manifest_sha256"] != actual_review_sha:
        draft_errors.append("review_manifest_sha256_mismatch")
    if review_manifest.source_manifest_sha256 and review_manifest.source_manifest_sha256 != plan["source_manifest_sha256"]:
        draft_errors.append("source_manifest_sha256_mismatch")
    if review_manifest.source_dry_run_report_sha256 != plan["source_dry_run_report_sha256"]:
        draft_errors.append("source_dry_run_report_sha256_mismatch")

    review_by_id = {record.record_id: record for record in review_manifest.review_records}
    admission_records: list[dict[str, Any]] = []
    draft_admit_ids: list[str] = []
    draft_exclude_ids: list[str] = []
    blocked_ids: list[str] = []
    for record in plan["planned_record_summaries"]:
        record_id = record["record_id"]
        planned_action = record["planned_action"]
        review_record = review_by_id.get(record_id)
        if review_record is None:
            _append_unique(draft_errors, "review_record_missing_for_plan_record")
            _append_unique(blocked_ids, record_id)
            continue
        if review_record.document_id != record["document_id"] or review_record.field_name != record["field_name"]:
            _append_unique(draft_errors, "review_record_target_mismatch")
            _append_unique(blocked_ids, record_id)
            continue
        if review_record.review_id != record["source_review_id"]:
            _append_unique(draft_errors, "review_id_mismatch")
            _append_unique(blocked_ids, record_id)
            continue
        if review_record.decision != record["review_decision"]:
            if planned_action == "admit":
                _append_unique(draft_errors, "planned_admit_review_decision_invalid")
            elif planned_action == "exclude":
                _append_unique(draft_errors, "planned_exclude_review_decision_invalid")
            _append_unique(draft_errors, "review_decision_mismatch")
            _append_unique(blocked_ids, record_id)
            continue

        if planned_action == "admit":
            blocking_reason = _validate_planned_admit(record)
            if blocking_reason:
                _append_unique(draft_errors, blocking_reason)
                _append_unique(blocked_ids, record_id)
                continue
            admission_records.append(
                _admission_record_from_plan(
                    plan=plan,
                    record=record,
                    admission_request_id=admission_request_id,
                    action="admit",
                    review_artifact_sha256=actual_review_sha,
                )
            )
            _append_unique(draft_admit_ids, record_id)
        elif planned_action == "exclude":
            blocking_reason = _validate_planned_exclude(record)
            if blocking_reason:
                _append_unique(draft_errors, blocking_reason)
                _append_unique(blocked_ids, record_id)
                continue
            admission_records.append(
                _admission_record_from_plan(
                    plan=plan,
                    record=record,
                    admission_request_id=admission_request_id,
                    action="exclude",
                    review_artifact_sha256=actual_review_sha,
                )
            )
            _append_unique(draft_exclude_ids, record_id)
        elif planned_action == "blocked":
            _append_unique(blocked_ids, record_id)
        else:
            _append_unique(draft_errors, "planned_action_invalid")
            _append_unique(blocked_ids, record_id)

    if not admission_records:
        _append_unique(draft_errors, "no_draft_admission_records")

    draft_payload = {
        "schema_version": _ADMISSION_SCHEMA_VERSION,
        "admission_request_id": admission_request_id,
        "corpus_id": plan["corpus_id"],
        "dry_run_id": plan["dry_run_id"],
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "created_by": created_by,
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "source_dry_run_report_sha256": plan["source_dry_run_report_sha256"],
        "source_review_manifest_sha256": actual_review_sha,
        "review_manifest_id": plan["review_manifest_id"],
        "admission_policy": "draft-property-admission-request-from-plan",
        "dataset_target": dataset_target,
        "admission_records": admission_records,
    }
    if admission_records:
        try:
            validate_admission_request(draft_payload)
        except CustomCorpusAdmissionError:
            _append_unique(draft_errors, "generated_admission_draft_invalid")

    summary = {
        "schema_version": _SCHEMA_VERSION,
        "draft_status": "blocked" if draft_errors else "written",
        "admission_request_plan_path": Path(admission_request_plan_path).name
        or "property_admission_request_plan_summary.json",
        "admission_request_plan_sha256": _safe_sha_for_path(admission_request_plan_path),
        "review_manifest_path": Path(review_manifest_path).name or "property_review_manifest.json",
        "review_manifest_sha256": actual_review_sha,
        "admission_request_id": admission_request_id,
        "review_queue_id": plan["review_queue_id"],
        "property_candidate_manifest_id": plan["property_candidate_manifest_id"],
        "review_manifest_id": plan["review_manifest_id"],
        "corpus_id": plan["corpus_id"],
        "dry_run_id": plan["dry_run_id"],
        "dataset_target": dataset_target,
        "planner_status": plan["planner_status"],
        "allow_partial_plan": allow_partial_plan,
        "draft_record_count": len(admission_records),
        "draft_admit_count": len(draft_admit_ids),
        "draft_exclude_count": len(draft_exclude_ids),
        "blocked_record_count": len(blocked_ids),
        "draft_admit_record_ids": draft_admit_ids,
        "draft_exclude_record_ids": draft_exclude_ids,
        "blocked_record_ids": blocked_ids,
        "draft_artifacts": dict(_ARTIFACTS),
        "draft_errors": draft_errors,
        "warnings": warnings,
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "source_dry_run_report_sha256": plan["source_dry_run_report_sha256"],
        "redaction_status": "passed",
    }

    if draft_errors:
        return summary, {}
    if _contains_forbidden_material({"summary": summary, "draft": draft_payload}):
        return _minimal_redaction_failure(), {}
    return summary, draft_payload


def _admission_record_from_plan(
    *,
    plan: dict[str, Any],
    record: dict[str, Any],
    admission_request_id: str,
    action: str,
    review_artifact_sha256: str,
) -> dict[str, Any]:
    if action == "admit":
        admission_reason = "draft request generated from property admission request plan; review accepted the normalized value"
        exclusion_reason = ""
        provenance_summary = record["provenance_summary"]
        normalized_value_summary = record["normalized_value_summary"]
    else:
        admission_reason = ""
        exclusion_reason = "draft request generated from property admission request plan; review rejected this record"
        provenance_summary = ""
        normalized_value_summary = ""
    return {
        "admission_record_id": f"{admission_request_id}-{record['record_id']}",
        "corpus_id": plan["corpus_id"],
        "dry_run_id": plan["dry_run_id"],
        "review_manifest_id": plan["review_manifest_id"],
        "document_id": record["document_id"],
        "record_id": record["record_id"],
        "field_name": record["field_name"],
        "admission_scope": "record",
        "review_id": record["source_review_id"],
        "review_decision": record["review_decision"],
        "action": action,
        "admission_reason": admission_reason,
        "exclusion_reason": exclusion_reason,
        "source_artifact_sha256": record["source_artifact_sha256"],
        "review_artifact_sha256": review_artifact_sha256,
        "provenance_summary": provenance_summary,
        "normalized_value_summary": normalized_value_summary,
        "notes": "draft admission request artifact only; not training admission",
    }


def _validate_planned_admit(record: dict[str, Any]) -> str:
    if record["review_decision"] != "accept":
        return "planned_admit_review_decision_invalid"
    if not record["normalized_value_summary"]:
        return "planned_admit_missing_normalized_value_summary"
    if not record["provenance_summary"]:
        return "planned_admit_missing_provenance_summary"
    if not record["source_artifact_sha256"]:
        return "planned_admit_missing_source_artifact_sha256"
    if not record["review_manifest_sha256"]:
        return "planned_admit_missing_review_manifest_sha256"
    if not record["planned_reason"]:
        return "planned_admit_missing_planned_reason"
    return ""


def _validate_planned_exclude(record: dict[str, Any]) -> str:
    if record["review_decision"] != "reject":
        return "planned_exclude_review_decision_invalid"
    if not record["planned_reason"]:
        return "planned_exclude_missing_planned_reason"
    if not record["source_artifact_sha256"]:
        return "planned_exclude_missing_source_artifact_sha256"
    if not record["review_manifest_sha256"]:
        return "planned_exclude_missing_review_manifest_sha256"
    return ""


def _load_request_plan(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyAdmissionDraftBuilderError(
            f"could not read request plan: {exc.__class__.__name__}"
        ) from exc
    return _validate_request_plan(payload)


def _validate_request_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CustomCorpusPropertyAdmissionDraftBuilderError("request plan must be an object")
    if value.get("schema_version") != _PLAN_SCHEMA_VERSION:
        raise CustomCorpusPropertyAdmissionDraftBuilderError("schema_version is invalid")
    clean = dict(value)
    if clean.get("planner_status") not in {"planned", "partial", "blocked"}:
        raise CustomCorpusPropertyAdmissionDraftBuilderError("planner_status is invalid")
    for field in ("review_queue_id", "property_candidate_manifest_id", "review_manifest_id", "corpus_id", "dry_run_id"):
        clean[field] = _required_safe_id(clean.get(field), field_name=field)
    clean["admission_readiness_summary_sha256"] = _optional_sha(
        clean.get("admission_readiness_summary_sha256"),
        field_name="admission_readiness_summary_sha256",
    )
    clean["review_manifest_sha256"] = _optional_sha(clean.get("review_manifest_sha256"), field_name="review_manifest_sha256")
    clean["source_manifest_sha256"] = _optional_sha(clean.get("source_manifest_sha256"), field_name="source_manifest_sha256")
    clean["source_dry_run_report_sha256"] = _required_sha(
        clean.get("source_dry_run_report_sha256"),
        field_name="source_dry_run_report_sha256",
    )
    for field in (
        "planned_admit_record_ids",
        "planned_exclude_record_ids",
        "blocked_record_ids",
        "unreviewed_queue_record_ids",
        "readiness_errors",
        "planning_errors",
    ):
        clean[field] = _safe_string_list(clean.get(field), field_name=field)
    summaries = clean.get("planned_record_summaries")
    if not isinstance(summaries, list):
        raise CustomCorpusPropertyAdmissionDraftBuilderError("planned_record_summaries is invalid")
    clean["planned_record_summaries"] = [_validate_planned_record_summary(record) for record in summaries]
    return clean


def _validate_planned_record_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CustomCorpusPropertyAdmissionDraftBuilderError("planned record summary is invalid")
    record = dict(value)
    for field in (
        "planned_admission_plan_record_id",
        "source_review_id",
        "record_id",
        "document_id",
        "field_name",
    ):
        record[field] = _required_safe_id(record.get(field), field_name=field)
    if record.get("review_decision") not in {"accept", "reject", "needs_review"}:
        raise CustomCorpusPropertyAdmissionDraftBuilderError("review_decision is invalid")
    if record.get("planned_action") not in {"admit", "exclude", "blocked"}:
        raise CustomCorpusPropertyAdmissionDraftBuilderError("planned_action is invalid")
    record["planned_reason"] = _safe_text(record.get("planned_reason"), field_name="planned_reason")
    record["source_artifact_sha256"] = _optional_sha(
        record.get("source_artifact_sha256"),
        field_name="source_artifact_sha256",
    )
    record["review_manifest_sha256"] = _optional_sha(
        record.get("review_manifest_sha256"),
        field_name="review_manifest_sha256",
    )
    record["normalized_value_summary"] = _safe_text(
        record.get("normalized_value_summary"),
        field_name="normalized_value_summary",
    )
    record["provenance_summary"] = _safe_text(record.get("provenance_summary"), field_name="provenance_summary")
    record["blocking_reason"] = _safe_text(record.get("blocking_reason"), field_name="blocking_reason")
    return record


def _evidence_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Custom Corpus Property Admission Draft Evidence",
            "",
            f"- Admission request id: `{summary['admission_request_id']}`",
            f"- Draft status: `{summary['draft_status']}`",
            f"- Corpus id: `{summary['corpus_id']}`",
            f"- Dry-run id: `{summary['dry_run_id']}`",
            f"- Review manifest id: `{summary['review_manifest_id']}`",
            f"- Planner status: `{summary['planner_status']}`",
            f"- Draft record count: `{summary['draft_record_count']}`",
            f"- Admit record ids: `{json.dumps(summary['draft_admit_record_ids'])}`",
            f"- Exclude record ids: `{json.dumps(summary['draft_exclude_record_ids'])}`",
            f"- Blocked record ids: `{json.dumps(summary['blocked_record_ids'])}`",
            "",
            "## Boundary Statement",
            "",
            "- This is a draft admission request artifact.",
            "- No training data was admitted.",
            "- No package binding was run.",
            "- No materialization was run.",
            "- No candidate/training CSV was created.",
            "- Phase 1 did not run.",
            "- DatasetConfirmation was not changed.",
            "",
        ]
    )


def _safe_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise CustomCorpusPropertyAdmissionDraftBuilderError(f"{field_name} is invalid")
    return [_required_safe_id(item, field_name=field_name) for item in value]


def _required_safe_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_ID_RE.fullmatch(clean):
        raise CustomCorpusPropertyAdmissionDraftBuilderError(f"{field_name} is invalid")
    if any(marker.lower() in clean.lower() for marker in _FORBIDDEN_MARKERS):
        raise CustomCorpusPropertyAdmissionDraftBuilderError(f"{field_name} is invalid")
    return clean


def _safe_text(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if len(clean) > _FREE_TEXT_LIMIT:
        raise CustomCorpusPropertyAdmissionDraftBuilderError(f"{field_name} is invalid")
    lowered = clean.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        raise CustomCorpusPropertyAdmissionDraftBuilderError(f"{field_name} is invalid")
    if _ABSOLUTE_PATH_VALUE_RE.search(json.dumps(clean)):
        raise CustomCorpusPropertyAdmissionDraftBuilderError(f"{field_name} is invalid")
    return clean


def _safe_created_by(value: Any) -> str:
    clean = _safe_text(value, field_name="created_by")
    if not clean:
        raise CustomCorpusPropertyAdmissionDraftBuilderError("created_by is invalid")
    if "@" in clean and "redacted" not in clean.lower():
        raise CustomCorpusPropertyAdmissionDraftBuilderError("created_by is invalid")
    return clean


def _required_sha(value: Any, *, field_name: str) -> str:
    clean = _optional_sha(value, field_name=field_name)
    if not clean:
        raise CustomCorpusPropertyAdmissionDraftBuilderError(f"{field_name} is required")
    return clean


def _optional_sha(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise CustomCorpusPropertyAdmissionDraftBuilderError(f"{field_name} is invalid")
    return f"sha256:{match.group(2).lower()}"


def _safe_sha_for_path(path: str | Path) -> str:
    try:
        return sha256_file(path)
    except Exception:
        return ""


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _contains_forbidden_material(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(serialized))


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "draft_status": "blocked",
        "draft_errors": ["property_admission_draft_redaction_failed"],
        "redaction_status": "failed",
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
