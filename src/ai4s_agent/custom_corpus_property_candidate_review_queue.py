from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_property_candidate import (
    CustomCorpusPropertyCandidateError,
    PropertyCandidateRecord,
    load_property_candidate_manifest,
)
from ai4s_agent.custom_corpus_property_candidate_planner import plan_property_candidates


_SCHEMA_VERSION = "custom_corpus_property_candidate_review_queue.v1"
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
    "x-amz-signature",
    "signature=",
    "signedurl",
    "signed-url",
)

REVIEW_INSTRUCTION = "review_property_candidate_for_future_custom_corpus_review_manifest"
REVIEW_QUEUE_ARTIFACTS = (
    "property_candidate_review_queue.json",
    "property_candidate_review_queue.md",
    "property_candidate_review_summary.json",
    "redacted_property_candidate_evidence.md",
)


class CustomCorpusPropertyCandidateReviewQueueError(ValueError):
    pass


def build_property_candidate_review_queue(
    *,
    property_candidates_path: str | Path,
    output_dir: str | Path,
    review_queue_id: str,
    allow_empty_queue: bool = False,
) -> dict[str, Any]:
    safe_review_queue_id = _validate_safe_label(review_queue_id, field_name="review_queue_id")
    run_dir = Path(output_dir).expanduser() / safe_review_queue_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise CustomCorpusPropertyCandidateReviewQueueError("output directory is not empty")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_property_candidate_manifest(property_candidates_path)
    planner_summary = plan_property_candidates(property_candidates_path)
    if planner_summary.get("redaction_status") == "failed":
        summary = _minimal_blocked_summary(
            review_queue_id=safe_review_queue_id,
            reason="property_candidate_review_queue_redaction_failed",
        )
        _write_safe_summary(run_dir, summary)
        return summary

    records_by_id = {record.property_candidate_id: record for record in manifest.records}
    review_queue_ids = list(planner_summary["review_queue_record_ids"])
    queue_records = [_queue_record(records_by_id[record_id]) for record_id in review_queue_ids]
    blocked_ids = list(planner_summary["blocked_record_ids"])

    summary = {
        "schema_version": _SCHEMA_VERSION,
        "review_queue_id": safe_review_queue_id,
        "review_queue_status": "prepared"
        if planner_summary["planner_status"] == "planned" and queue_records
        else "blocked",
        "property_candidate_manifest_path": planner_summary["property_candidate_manifest_path"],
        "property_candidate_manifest_sha256": planner_summary["property_candidate_manifest_sha256"],
        "property_candidate_manifest_id": planner_summary["property_candidate_manifest_id"],
        "corpus_id": planner_summary["corpus_id"],
        "dry_run_id": planner_summary["dry_run_id"],
        "record_count": planner_summary["record_count"],
        "review_queue_count": len(queue_records),
        "blocked_record_count": len(blocked_ids),
        "candidate_count": planner_summary["candidate_count"],
        "needs_review_count": planner_summary["needs_review_count"],
        "rejected_count": planner_summary["rejected_count"],
        "review_queue_record_ids": review_queue_ids,
        "blocked_record_ids": blocked_ids,
        "field_name_counts": planner_summary["field_name_counts"],
        "property_family_counts": planner_summary["property_family_counts"],
        "value_kind_counts": planner_summary["value_kind_counts"],
        "unit_status_counts": planner_summary["unit_status_counts"],
        "extraction_source_counts": planner_summary["extraction_source_counts"],
        "source_manifest_sha256": planner_summary["source_manifest_sha256"],
        "source_dry_run_report_sha256": planner_summary["source_dry_run_report_sha256"],
        "artifacts": {
            "property_candidate_review_queue_json": f"{safe_review_queue_id}/property_candidate_review_queue.json",
            "property_candidate_review_queue_md": f"{safe_review_queue_id}/property_candidate_review_queue.md",
            "property_candidate_review_summary_json": f"{safe_review_queue_id}/property_candidate_review_summary.json",
            "redacted_property_candidate_evidence_md": f"{safe_review_queue_id}/redacted_property_candidate_evidence.md",
        },
        "blocking_reasons": list(planner_summary["blocking_reasons"]),
        "warnings": list(planner_summary["warnings"]),
        "redaction_status": "passed",
    }
    if not queue_records and "no_reviewable_property_candidates" not in summary["blocking_reasons"]:
        summary["blocking_reasons"].append("no_reviewable_property_candidates")

    queue_payload = {
        "schema_version": _SCHEMA_VERSION,
        "review_queue_id": safe_review_queue_id,
        "property_candidate_manifest_id": manifest.property_candidate_manifest_id,
        "corpus_id": manifest.corpus_id,
        "dry_run_id": manifest.dry_run_id,
        "property_candidate_manifest_sha256": summary["property_candidate_manifest_sha256"],
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "source_dry_run_report_sha256": manifest.source_dry_run_report_sha256,
        "queue_records": queue_records,
        "blocked_record_ids": blocked_ids,
        "boundary_statement": [
            "review queue preparation only",
            "no property extraction",
            "no llm or agent call",
            "no custom_corpus_review.v1 manifest created",
            "no review decisions",
            "no admission",
            "no materialization",
            "no phase 1",
            "no datasetconfirmation change",
        ],
    }
    queue_markdown = _queue_markdown(summary, queue_records)
    evidence_markdown = _evidence_markdown(summary)

    if _contains_forbidden_material([summary, queue_payload, queue_markdown, evidence_markdown]):
        blocked_summary = _minimal_blocked_summary(
            review_queue_id=safe_review_queue_id,
            reason="property_candidate_review_queue_redaction_failed",
        )
        _write_safe_summary(run_dir, blocked_summary)
        return blocked_summary

    if not queue_records and not allow_empty_queue:
        _write_safe_summary(run_dir, summary)
        return summary

    write_json(run_dir / "property_candidate_review_queue.json", queue_payload)
    (run_dir / "property_candidate_review_queue.md").write_text(queue_markdown, encoding="utf-8")
    write_json(run_dir / "property_candidate_review_summary.json", summary)
    (run_dir / "redacted_property_candidate_evidence.md").write_text(evidence_markdown, encoding="utf-8")
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
        summary = build_property_candidate_review_queue(
            property_candidates_path=args.property_candidates,
            output_dir=args.output_dir,
            review_queue_id=args.review_queue_id,
            allow_empty_queue=args.allow_empty_queue,
        )
    except CustomCorpusPropertyCandidateError as exc:
        err.write(f"property candidate review queue invalid: {exc}\n")
        return 1
    except CustomCorpusPropertyCandidateReviewQueueError as exc:
        err.write(f"property candidate review queue invalid: {exc}\n")
        return 1

    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    if summary.get("redaction_status") == "failed":
        return 1
    if summary.get("review_queue_count", 0) == 0 and not args.allow_empty_queue:
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_candidate_review_queue",
        description="Build offline review-preparation artifacts for custom corpus property candidates.",
    )
    parser.add_argument("--property-candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--review-queue-id", required=True)
    parser.add_argument("--allow-empty-queue", action="store_true")
    return parser


def _queue_record(record: PropertyCandidateRecord) -> dict[str, Any]:
    return {
        "review_queue_record_id": f"review-queue-{record.property_candidate_id}",
        "property_candidate_id": record.property_candidate_id,
        "corpus_id": record.corpus_id,
        "dry_run_id": record.dry_run_id,
        "document_id": record.document_id,
        "source_record_id": record.source_record_id,
        "source_artifact_sha256": record.source_artifact_sha256,
        "parsed_document_sha256": record.parsed_document_sha256,
        "page": record.page,
        "table_id": record.table_id,
        "row_id": record.row_id,
        "column_name": record.column_name,
        "raw_property_label": record.raw_property_label,
        "canonical_property_guess": record.canonical_property_guess,
        "property_family": record.property_family,
        "field_name": record.field_name,
        "value_kind": record.value_kind,
        "value_raw_summary": record.value_raw_summary,
        "value_normalized": record.value_normalized,
        "value_min": record.value_min,
        "value_max": record.value_max,
        "value_tuple": record.value_tuple,
        "unit_raw": record.unit_raw,
        "unit_normalized": record.unit_normalized,
        "unit_status": record.unit_status,
        "entity_id": record.entity_id,
        "entity_type": record.entity_type,
        "entity_label_summary": record.entity_label_summary,
        "method_summary": record.method_summary,
        "condition_summary": record.condition_summary,
        "provenance_summary": record.provenance_summary,
        "extraction_source": record.extraction_source,
        "extractor_label": record.extractor_label,
        "confidence": record.confidence,
        "trainability_decision": record.trainability_decision,
        "decision_reason": record.decision_reason,
        "review_required": record.review_required,
        "review_instruction": REVIEW_INSTRUCTION,
    }


def _queue_markdown(summary: dict[str, Any], queue_records: list[dict[str, Any]]) -> str:
    lines = [
        "# Custom Corpus Property Candidate Review Queue",
        "",
        f"- Review queue id: `{summary['review_queue_id']}`",
        f"- Queue status: `{summary['review_queue_status']}`",
        f"- Property candidate manifest id: `{summary['property_candidate_manifest_id']}`",
        f"- Corpus id: `{summary['corpus_id']}`",
        f"- Dry-run id: `{summary['dry_run_id']}`",
        f"- Review queue count: `{summary['review_queue_count']}`",
        f"- Blocked record count: `{summary['blocked_record_count']}`",
        "",
        "## Queue Records",
        "",
    ]
    if not queue_records:
        lines.append("No reviewable property candidates were queued.")
    for record in queue_records:
        lines.extend(
            [
                f"### {record['property_candidate_id']}",
                "",
                f"- Document id: `{record['document_id']}`",
                f"- Field name: `{record['field_name']}`",
                f"- Raw property label: `{record['raw_property_label']}`",
                f"- Canonical property guess: `{record['canonical_property_guess']}`",
                f"- Property family: `{record['property_family']}`",
                f"- Value kind: `{record['value_kind']}`",
                f"- Value raw summary: `{record['value_raw_summary']}`",
                f"- Value normalized: `{record['value_normalized']}`",
                f"- Value min: `{record['value_min']}`",
                f"- Value max: `{record['value_max']}`",
                f"- Value tuple: `{json.dumps(record['value_tuple'], sort_keys=True)}`",
                f"- Unit raw: `{record['unit_raw']}`",
                f"- Unit normalized: `{record['unit_normalized']}`",
                f"- Unit status: `{record['unit_status']}`",
                f"- Entity id: `{record['entity_id']}`",
                f"- Entity type: `{record['entity_type']}`",
                f"- Entity label summary: `{record['entity_label_summary']}`",
                f"- Method summary: `{record['method_summary']}`",
                f"- Condition summary: `{record['condition_summary']}`",
                f"- Provenance summary: `{record['provenance_summary']}`",
                f"- Extraction source: `{record['extraction_source']}`",
                f"- Extractor label: `{record['extractor_label']}`",
                f"- Confidence: `{record['confidence']}`",
                f"- Trainability decision: `{record['trainability_decision']}`",
                f"- Decision reason: `{record['decision_reason']}`",
                f"- Review instruction: `{record['review_instruction']}`",
                "",
            ]
        )
    lines.extend(_boundary_lines())
    return "\n".join(lines) + "\n"


def _evidence_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Redacted Property Candidate Review Queue Evidence",
        "",
        "This was a review-preparation run only.",
        "",
        "## Queue Preparation Summary",
        "",
        f"- Review queue id: `{summary['review_queue_id']}`",
        f"- Queue status: `{summary['review_queue_status']}`",
        f"- Property candidate manifest id: `{summary['property_candidate_manifest_id']}`",
        f"- Corpus id: `{summary['corpus_id']}`",
        f"- Dry-run id: `{summary['dry_run_id']}`",
        f"- Review queue count: `{summary['review_queue_count']}`",
        f"- Blocked record count: `{summary['blocked_record_count']}`",
        "",
        "## Boundary Statement",
        "",
        "- No review decisions were created.",
        "- No admission was performed.",
        "- No materialization was performed.",
        "- No Phase 1 execution.",
        "- No DatasetConfirmation change.",
        "- No raw source documents, ParsedDocument outputs, MinerU bundles, or raw article text are included.",
        "- No private paths or credential material are included.",
        "",
    ]
    return "\n".join(lines)


def _boundary_lines() -> list[str]:
    return [
        "## Boundary Statement",
        "",
        "- No property extraction.",
        "- No LLM or agent call.",
        "- No human review performed.",
        "- No custom_corpus_review.v1 manifest created.",
        "- No review decisions created.",
        "- No admission.",
        "- No materialization.",
        "- No dataset candidate or training CSV.",
        "- No Phase 1 execution.",
        "- No DatasetConfirmation change.",
    ]


def _write_safe_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    if _contains_forbidden_material([summary]):
        summary = _minimal_blocked_summary(
            review_queue_id=str(summary.get("review_queue_id") or "redacted-review-queue"),
            reason="property_candidate_review_queue_redaction_failed",
        )
    write_json(run_dir / "property_candidate_review_summary.json", summary)


def _minimal_blocked_summary(*, review_queue_id: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "review_queue_id": _safe_label_or_redacted(review_queue_id),
        "review_queue_status": "blocked",
        "review_queue_count": 0,
        "blocked_record_count": 0,
        "blocking_reasons": [reason],
        "warnings": [],
        "redaction_status": "failed" if "redaction" in reason else "passed",
    }


def _validate_safe_label(value: str, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_ID_RE.fullmatch(clean):
        raise CustomCorpusPropertyCandidateReviewQueueError(f"{field_name} must be a safe identifier")
    lowered = clean.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        raise CustomCorpusPropertyCandidateReviewQueueError(f"{field_name} contains forbidden material")
    return clean


def _safe_label_or_redacted(value: str) -> str:
    clean = str(value or "").strip()
    if clean and _SAFE_ID_RE.fullmatch(clean) and not any(marker.lower() in clean.lower() for marker in _FORBIDDEN_MARKERS):
        return clean
    return "redacted-review-queue"


def _contains_forbidden_material(values: list[Any]) -> bool:
    serialized = json.dumps(values, ensure_ascii=False, sort_keys=True)
    lowered = serialized.lower()
    if any(marker.lower() in lowered for marker in _FORBIDDEN_MARKERS):
        return True
    return bool(_ABSOLUTE_PATH_VALUE_RE.search(serialized))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
