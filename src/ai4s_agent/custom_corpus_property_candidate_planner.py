from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_property_candidate import (
    CustomCorpusPropertyCandidateError,
    load_property_candidate_manifest,
    property_candidate_manifest_summary,
)


_SCHEMA_VERSION = "custom_corpus_property_candidate_planner.v1"

PLANNED_REVIEW_OUTPUT_LABELS = (
    "property_candidate_review_queue.json",
    "property_candidate_review_queue.md",
    "property_candidate_review_summary.json",
    "redacted_property_candidate_evidence.md",
)

_FORBIDDEN_SUMMARY_MARKERS = (
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
)
_ABSOLUTE_PATH_VALUE_RE = re.compile(r'"(?:/|[A-Za-z]:\\\\)')


def plan_property_candidates(property_candidates_path: str | Path) -> dict[str, Any]:
    manifest = load_property_candidate_manifest(property_candidates_path)
    base_summary = property_candidate_manifest_summary(manifest, path=property_candidates_path)

    candidate_ids = [record.property_candidate_id for record in manifest.records if record.trainability_decision == "candidate"]
    needs_review_ids = [
        record.property_candidate_id for record in manifest.records if record.trainability_decision == "needs_review"
    ]
    rejected_ids = [record.property_candidate_id for record in manifest.records if record.trainability_decision == "reject"]
    review_queue_ids = [
        record.property_candidate_id
        for record in manifest.records
        if record.review_required and record.trainability_decision in {"candidate", "needs_review"}
    ]
    blocked_ids = [
        record.property_candidate_id
        for record in manifest.records
        if (not record.review_required) or record.trainability_decision == "reject"
    ]
    blocking_reasons = []
    if not review_queue_ids:
        blocking_reasons.append("no_reviewable_property_candidates")

    summary = {
        "schema_version": _SCHEMA_VERSION,
        "planner_status": "blocked" if blocking_reasons else "planned",
        "property_candidate_manifest_path": base_summary["property_candidate_manifest_path"],
        "property_candidate_manifest_sha256": base_summary["property_candidate_manifest_sha256"],
        "property_candidate_manifest_id": manifest.property_candidate_manifest_id,
        "corpus_id": manifest.corpus_id,
        "dry_run_id": manifest.dry_run_id,
        "record_count": len(manifest.records),
        "candidate_count": len(candidate_ids),
        "needs_review_count": len(needs_review_ids),
        "rejected_count": len(rejected_ids),
        "review_queue_count": len(review_queue_ids),
        "blocked_record_count": len(blocked_ids),
        "review_queue_record_ids": review_queue_ids,
        "blocked_record_ids": blocked_ids,
        "candidate_record_ids": candidate_ids,
        "needs_review_record_ids": needs_review_ids,
        "rejected_record_ids": rejected_ids,
        "unique_document_count": base_summary["unique_document_count"],
        "unique_entity_count": base_summary["unique_entity_count"],
        "unique_field_count": base_summary["unique_field_count"],
        "field_name_counts": dict(sorted(Counter(record.field_name for record in manifest.records).items())),
        "property_family_counts": base_summary["property_family_counts"],
        "value_kind_counts": base_summary["value_kind_counts"],
        "unit_status_counts": base_summary["unit_status_counts"],
        "extraction_source_counts": base_summary["extraction_source_counts"],
        "candidate_policy": manifest.candidate_policy,
        "extraction_scope": manifest.extraction_scope,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "source_dry_run_report_sha256": manifest.source_dry_run_report_sha256,
        "planned_review_output_labels": list(PLANNED_REVIEW_OUTPUT_LABELS),
        "blocking_reasons": blocking_reasons,
        "warnings": [],
        "redaction_status": "passed",
    }
    return _fail_closed_if_unsafe(summary)


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
        summary = plan_property_candidates(args.property_candidates)
    except CustomCorpusPropertyCandidateError as exc:
        err.write(f"property candidate planner invalid: {exc}\n")
        return 1

    if summary.get("redaction_status") == "failed":
        err.write("property candidate planner summary redaction failed\n")
        output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        output.write("\n")
        return 1

    if args.output_summary:
        write_json(Path(args.output_summary).expanduser(), summary)
    if args.output_markdown:
        _write_markdown(Path(args.output_markdown).expanduser(), summary)
    output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_property_candidate_planner",
        description="Plan review handling for custom corpus property candidates without creating review artifacts.",
    )
    parser.add_argument("--property-candidates", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-markdown", default="")
    return parser


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Custom Corpus Property Candidate Planner Summary",
        "",
        f"- Property candidate manifest id: `{summary['property_candidate_manifest_id']}`",
        f"- Corpus id: `{summary['corpus_id']}`",
        f"- Dry-run id: `{summary['dry_run_id']}`",
        f"- Planner status: `{summary['planner_status']}`",
        f"- Record count: `{summary['record_count']}`",
        f"- Candidate count: `{summary['candidate_count']}`",
        f"- Needs-review count: `{summary['needs_review_count']}`",
        f"- Rejected count: `{summary['rejected_count']}`",
        f"- Review queue count: `{summary['review_queue_count']}`",
        f"- Blocked record count: `{summary['blocked_record_count']}`",
        "",
        "## Numeric Property Summary",
        "",
        f"- Field name counts: `{json.dumps(summary['field_name_counts'], sort_keys=True)}`",
        f"- Property family counts: `{json.dumps(summary['property_family_counts'], sort_keys=True)}`",
        f"- Value kind counts: `{json.dumps(summary['value_kind_counts'], sort_keys=True)}`",
        f"- Unit status counts: `{json.dumps(summary['unit_status_counts'], sort_keys=True)}`",
        f"- Extraction source counts: `{json.dumps(summary['extraction_source_counts'], sort_keys=True)}`",
        "",
        "## Planned Review Output Labels",
        "",
    ]
    lines.extend(f"- `{label}`" for label in summary["planned_review_output_labels"])
    lines.extend(
        [
            "",
            "## Boundary Statement",
            "",
            "- No property extraction.",
            "- No LLM or agent call.",
            "- No human review manifest created.",
            "- No admission request.",
            "- No materialization.",
            "- No Phase 1 execution.",
            "- No DatasetConfirmation change.",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    safe = _fail_closed_if_unsafe({"markdown": markdown})
    if safe.get("redaction_status") == "failed":
        path.write_text("# Custom Corpus Property Candidate Planner Summary\n\nRedaction failed.\n", encoding="utf-8")
        return
    path.write_text(markdown, encoding="utf-8")


def _fail_closed_if_unsafe(summary: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    lowered = serialized.lower()
    for marker in _FORBIDDEN_SUMMARY_MARKERS:
        if marker.lower() in lowered:
            return {
                "schema_version": _SCHEMA_VERSION,
                "planner_status": "blocked",
                "blocking_reasons": ["property_candidate_planner_summary_redaction_failed"],
                "redaction_status": "failed",
            }
    if _ABSOLUTE_PATH_VALUE_RE.search(serialized):
        return {
            "schema_version": _SCHEMA_VERSION,
            "planner_status": "blocked",
            "blocking_reasons": ["property_candidate_planner_summary_redaction_failed"],
            "redaction_status": "failed",
        }
    return summary


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
