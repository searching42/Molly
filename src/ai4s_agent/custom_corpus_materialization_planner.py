from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO

from ai4s_agent._utils import write_json
from ai4s_agent.custom_corpus_materialization import (
    CustomCorpusMaterializationError,
    load_materialization_plan,
    materialization_plan_summary,
)


_SCHEMA_VERSION = "custom_corpus_materialization_planner.v1"

PLANNED_OUTPUT_LABELS = (
    "materialization_summary.json",
    "materialized_records.jsonl",
    "materialized_records.csv",
    "provenance_bindings.jsonl",
    "rollback_manifest.json",
    "redacted_evidence_summary.md",
)

PLANNED_ROLLBACK_LABELS = (
    "rollback_manifest.json",
    "delete_generated_candidate_artifacts_only",
    "do_not_delete_source_pdfs",
    "do_not_delete_external_original_corpora",
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


def plan_materialization(materialization_plan_path: str | Path) -> dict[str, Any]:
    plan = load_materialization_plan(materialization_plan_path)
    plan_summary = materialization_plan_summary(plan, path=materialization_plan_path)
    candidate_ids = [
        record.materialization_record_id for record in plan.materialization_records if record.action == "materialize_candidate"
    ]
    excluded_ids = [record.materialization_record_id for record in plan.materialization_records if record.action == "exclude"]
    blocking_reasons = []
    if plan.materialization_decision == "blocked":
        blocking_reasons.append("materialization_decision_blocked")

    summary = {
        "schema_version": _SCHEMA_VERSION,
        "planner_status": "planned" if plan.materialization_decision == "planned" else "blocked",
        "materialization_plan_path": plan_summary["materialization_plan_path"],
        "materialization_plan_sha256": plan_summary["materialization_plan_sha256"],
        "materialization_plan_id": plan.materialization_plan_id,
        "materialization_run_id": plan.materialization_run_id,
        "corpus_id": plan.corpus_id,
        "dry_run_id": plan.dry_run_id,
        "review_manifest_id": plan.review_manifest_id,
        "admission_request_id": plan.admission_request_id,
        "dataset_target": plan.dataset_target,
        "materialization_mode": plan.materialization_mode,
        "materialization_decision": plan.materialization_decision,
        "package_validation_status": plan.package_validation_status,
        "package_admission_decision": plan.package_admission_decision,
        "dry_run_phase1_status": plan.dry_run_phase1_status,
        "dry_run_dataset_confirmation_confirmed": plan.dry_run_dataset_confirmation_confirmed,
        "dry_run_training_dataset_admitted": plan.dry_run_training_dataset_admitted,
        "confirmation_present": bool(plan.confirmation.confirmed),
        "candidate_record_count": len(candidate_ids),
        "excluded_record_count": len(excluded_ids),
        "planned_output_labels": list(PLANNED_OUTPUT_LABELS),
        "planned_rollback_labels": list(PLANNED_ROLLBACK_LABELS),
        "candidate_record_ids": candidate_ids,
        "excluded_record_ids": excluded_ids,
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
        summary = plan_materialization(args.materialization_plan)
    except CustomCorpusMaterializationError as exc:
        err.write(f"materialization planner invalid: {exc}\n")
        return 1

    if summary.get("redaction_status") == "failed":
        err.write("materialization planner summary redaction failed\n")
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
        prog="python -m ai4s_agent.custom_corpus_materialization_planner",
        description="Plan custom corpus materialization from a validated plan without creating candidate artifacts.",
    )
    parser.add_argument("--materialization-plan", required=True)
    parser.add_argument("--output-summary", default="")
    parser.add_argument("--output-markdown", default="")
    return parser


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Custom Corpus Materialization Planner Summary",
        "",
        f"- Materialization plan id: `{summary['materialization_plan_id']}`",
        f"- Materialization run id: `{summary['materialization_run_id']}`",
        f"- Planner status: `{summary['planner_status']}`",
        f"- Candidate records: `{summary['candidate_record_count']}`",
        f"- Excluded records: `{summary['excluded_record_count']}`",
        "",
        "## Source Boundary",
        "",
        f"- Package validation status: `{summary['package_validation_status']}`",
        f"- Package admission decision: `{summary['package_admission_decision']}`",
        f"- Phase 1 status: `{summary['dry_run_phase1_status']}`",
        f"- DatasetConfirmation confirmed: `{summary['dry_run_dataset_confirmation_confirmed']}`",
        f"- Training dataset admitted: `{summary['dry_run_training_dataset_admitted']}`",
        "",
        "## Planned Output Labels",
        "",
    ]
    lines.extend(f"- `{label}`" for label in summary["planned_output_labels"])
    lines.extend(["", "## Rollback Labels", ""])
    lines.extend(f"- `{label}`" for label in summary["planned_rollback_labels"])
    lines.extend(
        [
            "",
            "## Boundary Statement",
            "",
            "- No candidate artifacts created.",
            "- No training data admitted.",
            "- No Phase 1 execution.",
            "- No DatasetConfirmation change.",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    safe = _fail_closed_if_unsafe({"markdown": markdown})
    if safe.get("redaction_status") == "failed":
        path.write_text("# Custom Corpus Materialization Planner Summary\n\nRedaction failed.\n", encoding="utf-8")
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
                "blocking_reasons": ["planner_summary_redaction_failed"],
                "redaction_status": "failed",
            }
    if _ABSOLUTE_PATH_VALUE_RE.search(serialized):
        return {
            "schema_version": _SCHEMA_VERSION,
            "planner_status": "blocked",
            "blocking_reasons": ["planner_summary_redaction_failed"],
            "redaction_status": "failed",
        }
    return summary


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
