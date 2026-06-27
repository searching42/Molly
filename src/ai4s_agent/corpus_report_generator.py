from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json


@dataclass(frozen=True)
class CorpusReportResult:
    corpus_report_json: str
    corpus_report_md: str
    corpus_summary_json: str


def generate_corpus_report(
    *,
    conflict_summary_json: str | Path,
    phase1_pipeline_json: str | Path,
    reproducibility_report_json: str | Path,
    ranked_candidates: list[dict[str, Any]],
    output_dir: str | Path,
    run_id: str,
    generated_at: str | None = None,
) -> CorpusReportResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    conflict_summary = _load_json(conflict_summary_json)
    phase1_pipeline = _load_json(phase1_pipeline_json)
    reproducibility = _load_json(reproducibility_report_json)
    top_candidates = ranked_candidates[:10]

    payload = {
        "run_id": run_id,
        "generated_at": generated,
        "document_count": int(conflict_summary.get("document_count") or 0),
        "extracted_record_count": int(conflict_summary.get("input_record_count") or 0),
        "accepted_record_count": int(conflict_summary.get("accepted_record_count") or 0),
        "rejected_record_count": int(conflict_summary.get("rejected_record_count") or 0),
        "duplicate_count": int(conflict_summary.get("consistent_duplicate_count") or 0),
        "conflict_count": int(conflict_summary.get("conflict_count") or 0),
        "unresolved_conflict_count": int(conflict_summary.get("unresolved_conflict_count") or 0),
        "training_record_count": int(conflict_summary.get("training_record_count") or 0),
        "candidate_record_count": int(conflict_summary.get("candidate_record_count") or 0),
        "phase1_status": str(phase1_pipeline.get("status") or "not_run"),
        "top_ranked_candidates": top_candidates,
        "provenance_coverage": {
            "mandatory_fields": ["paper_id", "page", "table_id", "row_id"],
            "source": "corpus_conflict_audit_and_dataset_manifest",
        },
        "reproducibility_hashes": reproducibility.get("hashes", {}),
        "phase1_hashes": phase1_pipeline.get("hashes", {}),
        "external_services_required": False,
    }
    summary = {
        "run_id": run_id,
        "status": "success",
        "document_count": payload["document_count"],
        "conflict_count": payload["conflict_count"],
        "unresolved_conflict_count": payload["unresolved_conflict_count"],
        "training_record_count": payload["training_record_count"],
        "candidate_record_count": payload["candidate_record_count"],
        "phase1_status": payload["phase1_status"],
        "top_ranked_candidate_count": len(top_candidates),
    }

    report_json = output_path / "corpus_report.json"
    summary_json = output_path / "corpus_summary.json"
    report_md = output_path / "corpus_report.md"
    write_json(report_json, payload)
    write_json(summary_json, summary)
    report_md.write_text(_render_markdown(payload), encoding="utf-8")
    return CorpusReportResult(
        corpus_report_json=str(report_json),
        corpus_report_md=str(report_md),
        corpus_summary_json=str(summary_json),
    )


def _load_json(path_like: str | Path) -> dict[str, Any]:
    if not str(path_like or "").strip():
        return {}
    path = Path(path_like).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Corpus Evaluation And Reproducibility Audit",
        "",
        f"- Documents: {payload['document_count']}",
        f"- Extracted records: {payload['extracted_record_count']}",
        f"- Accepted records: {payload['accepted_record_count']}",
        f"- Rejected records: {payload['rejected_record_count']}",
        f"- Conflicts: {payload['conflict_count']}",
        f"- Unresolved conflicts: {payload['unresolved_conflict_count']}",
        f"- Phase 1 status: {payload['phase1_status']}",
        "",
        "## Top Ranked Candidates",
    ]
    candidates = payload.get("top_ranked_candidates") or []
    if not candidates:
        lines.append("- None")
    else:
        for index, row in enumerate(candidates, start=1):
            smiles = row.get("SMILES") or row.get("smiles") or ""
            score = row.get("weighted_score") or row.get("score") or ""
            lines.append(f"{index}. {smiles} {score}".rstrip())
    lines.extend(
        [
            "",
            "## Reproducibility",
            "- External services required: false",
            "- Replay boundary: ParsedDocument fixtures through corpus report",
        ]
    )
    return "\n".join(lines) + "\n"
