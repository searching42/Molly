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


_SCHEMA_VERSION = "custom_corpus_property_review_binding.v1"
_QUEUE_SCHEMA_VERSION = "custom_corpus_property_candidate_review_queue.v1"
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
_REQUIRED_QUEUE_RECORD_FIELDS = (
    "property_candidate_id",
    "corpus_id",
    "dry_run_id",
    "document_id",
    "field_name",
    "source_artifact_sha256",
    "parsed_document_sha256",
    "value_kind",
    "unit_status",
    "entity_id",
    "entity_type",
    "trainability_decision",
    "review_required",
    "review_instruction",
)


class CustomCorpusPropertyReviewBindingError(ValueError):
    pass


def bind_property_review_manifest(
    *,
    review_queue_path: str | Path,
    review_manifest_path: str | Path,
    require_complete_queue: bool = False,
) -> dict[str, Any]:
    queue = _load_review_queue(review_queue_path)
    review_manifest = load_review_manifest(review_manifest_path)
    summary = _binding_summary(
        queue=queue,
        review_manifest=review_manifest,
        review_queue_path=review_queue_path,
        review_manifest_path=review_manifest_path,
        require_complete_queue=require_complete_queue,
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
        summary = bind_property_review_manifest(
            review_queue_path=args.review_queue,
            review_manifest_path=args.review_manifest,
            require_complete_queue=args.require_complete_queue,
        )
    except CustomCorpusReviewError as exc:
        err.write(f"property review binding invalid: {exc}\n")
        return 1
    except CustomCorpusPropertyReviewBindingError as exc:
        err.write(f"property review binding review queue invalid: {exc}\n")
        return 1

    if summary.get("redaction_status") == "failed":
        output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        output.write("\n")
        return 1
    if args.output_summary:
        write_json(Path(args.output_summary).expanduser(), summary)
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 1 if summary.get("binding_status") == "failed" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_review_binding",
        description="Validate binding between a property candidate review queue and a human review manifest.",
    )
    parser.add_argument("--review-queue", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--require-complete-queue", action="store_true")
    return parser


def _binding_summary(
    *,
    queue: dict[str, Any],
    review_manifest: ReviewManifest,
    review_queue_path: str | Path,
    review_manifest_path: str | Path,
    require_complete_queue: bool,
) -> dict[str, Any]:
    queue_records = list(queue["queue_records"])
    queue_by_id = {str(record["property_candidate_id"]): record for record in queue_records}
    queue_ids = list(queue_by_id)
    blocked_ids = list(queue["blocked_record_ids"])
    blocked_id_set = set(blocked_ids)
    binding_errors: list[str] = []
    reviewed_blocked_record_ids: list[str] = []
    unknown_review_record_ids: list[str] = []
    reviewed_valid_ids: list[str] = []

    if review_manifest.corpus_id != queue["corpus_id"]:
        binding_errors.append("corpus_id_mismatch")
    if review_manifest.dry_run_id != queue["dry_run_id"]:
        binding_errors.append("dry_run_id_mismatch")
    if review_manifest.source_dry_run_report_sha256 != queue["source_dry_run_report_sha256"]:
        binding_errors.append("source_dry_run_report_sha256_mismatch")
    if review_manifest.source_manifest_sha256 and review_manifest.source_manifest_sha256 != queue["source_manifest_sha256"]:
        binding_errors.append("source_manifest_sha256_mismatch")

    for record in review_manifest.review_records:
        record_id = record.record_id
        if record_id in blocked_id_set:
            _append_unique(reviewed_blocked_record_ids, record_id)
            _append_unique(binding_errors, "reviewed_blocked_record")
            continue
        queue_record = queue_by_id.get(record_id)
        if queue_record is None:
            _append_unique(unknown_review_record_ids, record_id)
            _append_unique(binding_errors, "unknown_review_record")
            continue
        _append_unique(reviewed_valid_ids, record_id)
        if record.review_scope not in {"field", "record"}:
            _append_unique(binding_errors, "review_scope_invalid")
        if record.document_id != queue_record["document_id"]:
            _append_unique(binding_errors, "review_record_document_id_mismatch")
        if record.field_name != queue_record["field_name"]:
            _append_unique(binding_errors, "review_record_field_name_mismatch")
        if record.corpus_id != queue_record["corpus_id"]:
            _append_unique(binding_errors, "review_record_corpus_id_mismatch")
        if record.dry_run_id != queue_record["dry_run_id"]:
            _append_unique(binding_errors, "review_record_dry_run_id_mismatch")
        if record.source_artifact_sha256 != queue_record["source_artifact_sha256"]:
            _append_unique(binding_errors, "review_record_source_artifact_sha256_mismatch")
        if record.decision == "accept":
            if not record.extracted_value_summary:
                _append_unique(binding_errors, "accepted_review_missing_extracted_value_summary")
            if not record.normalized_value_summary:
                _append_unique(binding_errors, "accepted_review_missing_normalized_value_summary")
            if not record.provenance_note:
                _append_unique(binding_errors, "accepted_review_missing_provenance_note")

    unreviewed_ids = [record_id for record_id in queue_ids if record_id not in set(reviewed_valid_ids)]
    if require_complete_queue and unreviewed_ids:
        _append_unique(binding_errors, "queue_review_incomplete")

    decisions = [record.decision for record in review_manifest.review_records]
    if binding_errors:
        binding_status = "failed"
    elif unreviewed_ids:
        binding_status = "needs_review"
    else:
        binding_status = "passed"

    return {
        "schema_version": _SCHEMA_VERSION,
        "binding_status": binding_status,
        "review_queue_path": Path(review_queue_path).name or "property_candidate_review_queue.json",
        "review_queue_sha256": _safe_sha_for_path(review_queue_path),
        "review_manifest_path": Path(review_manifest_path).name or "review_manifest.json",
        "review_manifest_sha256": _safe_sha_for_path(review_manifest_path),
        "review_queue_id": queue["review_queue_id"],
        "property_candidate_manifest_id": queue["property_candidate_manifest_id"],
        "review_manifest_id": review_manifest.review_manifest_id,
        "corpus_id": queue["corpus_id"],
        "dry_run_id": queue["dry_run_id"],
        "queue_record_count": len(queue_ids),
        "blocked_record_count": len(blocked_ids),
        "review_record_count": len(review_manifest.review_records),
        "reviewed_queue_record_count": len(reviewed_valid_ids),
        "accepted_count": decisions.count("accept"),
        "rejected_count": decisions.count("reject"),
        "needs_review_count": decisions.count("needs_review"),
        "unreviewed_queue_record_count": len(unreviewed_ids),
        "reviewed_queue_record_ids": reviewed_valid_ids,
        "unreviewed_queue_record_ids": unreviewed_ids,
        "reviewed_blocked_record_ids": reviewed_blocked_record_ids,
        "unknown_review_record_ids": unknown_review_record_ids,
        "binding_errors": binding_errors,
        "warnings": [],
        "require_complete_queue": require_complete_queue,
        "source_manifest_sha256": queue["source_manifest_sha256"],
        "source_dry_run_report_sha256": queue["source_dry_run_report_sha256"],
        "redaction_status": "passed",
    }


def _load_review_queue(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusPropertyReviewBindingError(f"could not read review queue: {exc.__class__.__name__}") from exc
    return _validate_review_queue(payload)


def _validate_review_queue(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CustomCorpusPropertyReviewBindingError("review queue must be an object")
    if value.get("schema_version") != _QUEUE_SCHEMA_VERSION:
        raise CustomCorpusPropertyReviewBindingError("schema_version is invalid")
    clean = dict(value)
    for field in ("review_queue_id", "property_candidate_manifest_id", "corpus_id", "dry_run_id"):
        clean[field] = _required_safe_id(clean.get(field), field_name=field)
    clean["property_candidate_manifest_sha256"] = _required_sha(
        clean.get("property_candidate_manifest_sha256"),
        field_name="property_candidate_manifest_sha256",
    )
    clean["source_manifest_sha256"] = _optional_sha(clean.get("source_manifest_sha256"), field_name="source_manifest_sha256")
    clean["source_dry_run_report_sha256"] = _required_sha(
        clean.get("source_dry_run_report_sha256"),
        field_name="source_dry_run_report_sha256",
    )
    queue_records = clean.get("queue_records")
    if not isinstance(queue_records, list):
        raise CustomCorpusPropertyReviewBindingError("queue_records is invalid")
    clean["queue_records"] = [_validate_queue_record(record) for record in queue_records]
    blocked_ids = clean.get("blocked_record_ids")
    if not isinstance(blocked_ids, list):
        raise CustomCorpusPropertyReviewBindingError("blocked_record_ids is invalid")
    clean["blocked_record_ids"] = [_required_safe_id(item, field_name="blocked_record_id") for item in blocked_ids]
    return clean


def _validate_queue_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CustomCorpusPropertyReviewBindingError("queue record is invalid")
    for field in _REQUIRED_QUEUE_RECORD_FIELDS:
        if field not in value:
            raise CustomCorpusPropertyReviewBindingError(f"{field} is missing")
    record = dict(value)
    for field in ("property_candidate_id", "corpus_id", "dry_run_id", "document_id", "field_name", "entity_id"):
        record[field] = _required_safe_id(record.get(field), field_name=field)
    record["source_artifact_sha256"] = _required_sha(record.get("source_artifact_sha256"), field_name="source_artifact_sha256")
    record["parsed_document_sha256"] = _optional_sha(
        record.get("parsed_document_sha256"),
        field_name="parsed_document_sha256",
    )
    if not isinstance(record.get("review_required"), bool):
        raise CustomCorpusPropertyReviewBindingError("review_required is invalid")
    for field in ("value_kind", "unit_status", "entity_type", "trainability_decision", "review_instruction"):
        if not str(record.get(field) or "").strip():
            raise CustomCorpusPropertyReviewBindingError(f"{field} is invalid")
    return record


def _required_safe_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_ID_RE.fullmatch(clean):
        raise CustomCorpusPropertyReviewBindingError(f"{field_name} is invalid")
    if any(marker.lower() in clean.lower() for marker in _FORBIDDEN_MARKERS):
        raise CustomCorpusPropertyReviewBindingError(f"{field_name} is invalid")
    return clean


def _required_sha(value: Any, *, field_name: str) -> str:
    clean = _optional_sha(value, field_name=field_name)
    if not clean:
        raise CustomCorpusPropertyReviewBindingError(f"{field_name} is required")
    return clean


def _optional_sha(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise CustomCorpusPropertyReviewBindingError(f"{field_name} is invalid")
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
    serialized = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return _minimal_redaction_failure()
    if _ABSOLUTE_PATH_VALUE_RE.search(serialized):
        return _minimal_redaction_failure()
    return summary


def _minimal_redaction_failure() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "binding_status": "failed",
        "binding_errors": ["property_review_binding_summary_redaction_failed"],
        "redaction_status": "failed",
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
