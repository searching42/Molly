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


_SCHEMA_VERSION = "custom_corpus_property_admission_readiness.v1"
_BINDING_SCHEMA_VERSION = "custom_corpus_property_review_binding.v1"
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


class CustomCorpusPropertyAdmissionReadinessError(ValueError):
    pass


def plan_property_admission_readiness(
    *,
    review_binding_summary_path: str | Path,
    review_manifest_path: str | Path,
    require_complete_binding: bool = False,
) -> dict[str, Any]:
    binding = _load_binding_summary(review_binding_summary_path)
    review_manifest = load_review_manifest(review_manifest_path)
    summary = _readiness_summary(
        binding=binding,
        review_manifest=review_manifest,
        review_binding_summary_path=review_binding_summary_path,
        review_manifest_path=review_manifest_path,
        require_complete_binding=require_complete_binding,
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
        summary = plan_property_admission_readiness(
            review_binding_summary_path=args.review_binding_summary,
            review_manifest_path=args.review_manifest,
            require_complete_binding=args.require_complete_binding,
        )
    except CustomCorpusReviewError as exc:
        err.write(f"property admission readiness invalid: {exc}\n")
        return 1
    except CustomCorpusPropertyAdmissionReadinessError as exc:
        err.write(f"property admission readiness binding summary invalid: {exc}\n")
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
    return 1 if summary.get("readiness_status") == "blocked" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_admission_readiness",
        description="Plan admission readiness from property review binding and human review artifacts.",
    )
    parser.add_argument("--review-binding-summary", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-markdown", default="")
    parser.add_argument("--require-complete-binding", action="store_true")
    return parser


def _readiness_summary(
    *,
    binding: dict[str, Any],
    review_manifest: ReviewManifest,
    review_binding_summary_path: str | Path,
    review_manifest_path: str | Path,
    require_complete_binding: bool,
) -> dict[str, Any]:
    readiness_errors: list[str] = []
    if binding["binding_status"] == "failed":
        readiness_errors.append("binding_status_failed")
    if binding["binding_status"] == "needs_review" and require_complete_binding:
        readiness_errors.append("binding_incomplete")
    if review_manifest.review_manifest_id != binding["review_manifest_id"]:
        readiness_errors.append("review_manifest_id_mismatch")
    if review_manifest.corpus_id != binding["corpus_id"]:
        readiness_errors.append("corpus_id_mismatch")
    if review_manifest.dry_run_id != binding["dry_run_id"]:
        readiness_errors.append("dry_run_id_mismatch")
    if binding["review_manifest_sha256"] and binding["review_manifest_sha256"] != _safe_sha_for_path(review_manifest_path):
        readiness_errors.append("review_manifest_sha256_mismatch")
    if review_manifest.source_manifest_sha256 and review_manifest.source_manifest_sha256 != binding["source_manifest_sha256"]:
        readiness_errors.append("source_manifest_sha256_mismatch")
    if review_manifest.source_dry_run_report_sha256 != binding["source_dry_run_report_sha256"]:
        readiness_errors.append("source_dry_run_report_sha256_mismatch")

    reviewed_ids = set(binding["reviewed_queue_record_ids"])
    reviewed_blocked_ids = set(binding["reviewed_blocked_record_ids"])
    unknown_ids = set(binding["unknown_review_record_ids"])
    planned_candidates: list[str] = []
    planned_exclusions: list[str] = []
    blocked_ids: list[str] = []
    for record in review_manifest.review_records:
        record_id = record.record_id
        if record.decision == "accept":
            blocked = False
            if record_id not in reviewed_ids:
                _append_unique(readiness_errors, "accepted_record_not_in_reviewed_queue_ids")
                blocked = True
            if record_id in reviewed_blocked_ids:
                _append_unique(readiness_errors, "accepted_record_in_reviewed_blocked_record_ids")
                blocked = True
            if record_id in unknown_ids:
                _append_unique(readiness_errors, "accepted_record_in_unknown_review_record_ids")
                blocked = True
            if not record.extracted_value_summary:
                _append_unique(readiness_errors, "accepted_review_missing_extracted_value_summary")
                blocked = True
            if not record.normalized_value_summary:
                _append_unique(readiness_errors, "accepted_review_missing_normalized_value_summary")
                blocked = True
            if not record.provenance_note:
                _append_unique(readiness_errors, "accepted_review_missing_provenance_note")
                blocked = True
            if blocked:
                _append_unique(blocked_ids, record_id)
            else:
                planned_candidates.append(record_id)
        elif record.decision == "reject":
            planned_exclusions.append(record_id)
        elif record.decision == "needs_review":
            _append_unique(blocked_ids, record_id)

    if not planned_candidates:
        _append_unique(readiness_errors, "no_admission_ready_records")

    decisions = [record.decision for record in review_manifest.review_records]
    if readiness_errors:
        readiness_status = "blocked"
    elif binding["binding_status"] == "needs_review":
        readiness_status = "partial"
    else:
        readiness_status = "ready"

    return {
        "schema_version": _SCHEMA_VERSION,
        "readiness_status": readiness_status,
        "review_binding_summary_path": Path(review_binding_summary_path).name or "review_binding_summary.json",
        "review_binding_summary_sha256": _safe_sha_for_path(review_binding_summary_path),
        "review_manifest_path": Path(review_manifest_path).name or "review_manifest.json",
        "review_manifest_sha256": _safe_sha_for_path(review_manifest_path),
        "review_queue_id": binding["review_queue_id"],
        "property_candidate_manifest_id": binding["property_candidate_manifest_id"],
        "review_manifest_id": binding["review_manifest_id"],
        "corpus_id": binding["corpus_id"],
        "dry_run_id": binding["dry_run_id"],
        "binding_status": binding["binding_status"],
        "require_complete_binding": require_complete_binding,
        "review_record_count": len(review_manifest.review_records),
        "accepted_review_count": decisions.count("accept"),
        "rejected_review_count": decisions.count("reject"),
        "needs_review_count": decisions.count("needs_review"),
        "admission_ready_record_count": len(planned_candidates),
        "planned_admission_candidate_record_ids": planned_candidates,
        "planned_exclusion_record_ids": planned_exclusions,
        "blocked_from_admission_record_ids": blocked_ids,
        "unreviewed_queue_record_ids": list(binding["unreviewed_queue_record_ids"]),
        "reviewed_blocked_record_ids": list(binding["reviewed_blocked_record_ids"]),
        "unknown_review_record_ids": list(binding["unknown_review_record_ids"]),
        "readiness_errors": readiness_errors,
        "warnings": [],
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "source_dry_run_report_sha256": binding["source_dry_run_report_sha256"],
        "redaction_status": "passed",
    }


def _load_binding_summary(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyAdmissionReadinessError(
            f"could not read binding summary: {exc.__class__.__name__}"
        ) from exc
    return _validate_binding_summary(payload)


def _validate_binding_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CustomCorpusPropertyAdmissionReadinessError("binding summary must be an object")
    if value.get("schema_version") != _BINDING_SCHEMA_VERSION:
        raise CustomCorpusPropertyAdmissionReadinessError("schema_version is invalid")
    clean = dict(value)
    if clean.get("binding_status") not in {"passed", "needs_review", "failed"}:
        raise CustomCorpusPropertyAdmissionReadinessError("binding_status is invalid")
    for field in ("review_queue_id", "property_candidate_manifest_id", "review_manifest_id", "corpus_id", "dry_run_id"):
        clean[field] = _required_safe_id(clean.get(field), field_name=field)
    clean["review_queue_sha256"] = _optional_sha(clean.get("review_queue_sha256"), field_name="review_queue_sha256")
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
        "reviewed_queue_record_ids",
        "unreviewed_queue_record_ids",
        "reviewed_blocked_record_ids",
        "unknown_review_record_ids",
        "binding_errors",
    ):
        clean[field] = _safe_string_list(clean.get(field), field_name=field)
    return clean


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Custom Corpus Property Admission Readiness",
        "",
        f"- Readiness status: `{summary['readiness_status']}`",
        f"- Review manifest id: `{summary['review_manifest_id']}`",
        f"- Review queue id: `{summary['review_queue_id']}`",
        f"- Binding status: `{summary['binding_status']}`",
        f"- Review record count: `{summary['review_record_count']}`",
        f"- Accepted review count: `{summary['accepted_review_count']}`",
        f"- Rejected review count: `{summary['rejected_review_count']}`",
        f"- Needs-review count: `{summary['needs_review_count']}`",
        f"- Admission-ready record count: `{summary['admission_ready_record_count']}`",
        f"- Planned admission candidate IDs: `{json.dumps(summary['planned_admission_candidate_record_ids'])}`",
        f"- Planned exclusion IDs: `{json.dumps(summary['planned_exclusion_record_ids'])}`",
        f"- Blocked-from-admission IDs: `{json.dumps(summary['blocked_from_admission_record_ids'])}`",
        "",
        "## Boundary Statement",
        "",
        "- No admission request created.",
        "- No admission action created.",
        "- No materialization.",
        "- No candidate/training CSV.",
        "- No Phase 1.",
        "- No DatasetConfirmation change.",
        "",
    ]
    return "\n".join(lines)


def _safe_string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise CustomCorpusPropertyAdmissionReadinessError(f"{field_name} is invalid")
    return [_required_safe_id(item, field_name=field_name) for item in value]


def _required_safe_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_ID_RE.fullmatch(clean):
        raise CustomCorpusPropertyAdmissionReadinessError(f"{field_name} is invalid")
    if any(marker.lower() in clean.lower() for marker in _FORBIDDEN_MARKERS):
        raise CustomCorpusPropertyAdmissionReadinessError(f"{field_name} is invalid")
    return clean


def _required_sha(value: Any, *, field_name: str) -> str:
    clean = _optional_sha(value, field_name=field_name)
    if not clean:
        raise CustomCorpusPropertyAdmissionReadinessError(f"{field_name} is required")
    return clean


def _optional_sha(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise CustomCorpusPropertyAdmissionReadinessError(f"{field_name} is invalid")
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
        "readiness_status": "blocked",
        "readiness_errors": ["property_admission_readiness_summary_redaction_failed"],
        "redaction_status": "failed",
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
