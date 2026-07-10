from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.corpus_report_generator import generate_corpus_report


GENERATED_AT = "2026-06-27T00:00:00Z"


def test_corpus_report_generator_summarizes_corpus_phase1_and_reproducibility(tmp_path: Path) -> None:
    conflict_summary = tmp_path / "conflict_summary.json"
    phase1_pipeline = tmp_path / "full_phase1_pipeline.json"
    reproducibility = tmp_path / "corpus_reproducibility_report.json"
    review_summary = tmp_path / "oled_review_summary.json"
    conflict_summary.write_text(
        json.dumps(
            {
                "document_count": 3,
                "input_record_count": 9,
                "accepted_record_count": 5,
                "rejected_record_count": 4,
                "consistent_duplicate_count": 1,
                "conflict_count": 1,
                "unresolved_conflict_count": 1,
                "oled_candidate_count": 6,
                "oled_text_evidence_candidate_count": 17,
                "oled_schema_candidate_count": 24,
                "oled_compiled_record_count": 5,
                "oled_compiled_admission_item_count": 3,
                "candidate_record_count": 5,
                "training_record_count": 5,
            }
        ),
        encoding="utf-8",
    )
    phase1_pipeline.write_text(
        json.dumps(
            {
                "status": "success",
                "artifacts": {"ranked_candidates_csv": "ranked_candidates.csv"},
                "hashes": {"dataset_hash": "sha256:data"},
            }
        ),
        encoding="utf-8",
    )
    reproducibility.write_text(
        json.dumps({"status": "success", "hashes": {"corpus_records_json": "sha256:records"}}),
        encoding="utf-8",
    )
    review_summary.write_text(
        json.dumps(
            {
                "run_id": "corpus-report",
                "review_item_count": 12,
                "counts_by_candidate_type": {
                    "oled_compiled_record": 3,
                    "oled_schema_candidate": 5,
                    "oled_text_evidence": 4,
                },
                "counts_by_priority": {"high": 5, "medium": 4, "low": 3},
                "governance_notes": ["candidate_only_review_packet"],
            }
        ),
        encoding="utf-8",
    )

    report = generate_corpus_report(
        conflict_summary_json=conflict_summary,
        phase1_pipeline_json=phase1_pipeline,
        reproducibility_report_json=reproducibility,
        oled_review_summary_json=review_summary,
        ranked_candidates=[{"SMILES": "CCO", "weighted_score": "1.0"}],
        output_dir=tmp_path / "report",
        run_id="corpus-report",
        generated_at=GENERATED_AT,
    )
    payload = _read_json(Path(report.corpus_report_json))
    summary = _read_json(Path(report.corpus_summary_json))
    markdown = Path(report.corpus_report_md).read_text(encoding="utf-8")

    assert payload["document_count"] == 3
    assert payload["oled_text_evidence_candidate_count"] == 17
    assert payload["oled_schema_candidate_count"] == 24
    assert payload["oled_review_item_count"] == 12
    assert payload["oled_compiled_admission_item_count"] == 3
    assert payload["oled_review_counts_by_priority"] == {"high": 5, "medium": 4, "low": 3}
    assert payload["phase1_status"] == "success"
    assert payload["top_ranked_candidates"][0]["SMILES"] == "CCO"
    assert summary["conflict_count"] == 1
    assert summary["oled_text_evidence_candidate_count"] == 17
    assert summary["oled_compiled_record_count"] == 5
    assert summary["oled_review_item_count"] == 12
    assert summary["oled_compiled_admission_item_count"] == 3
    assert "Corpus Evaluation And Reproducibility Audit" in markdown
    assert "OLED text evidence candidates: 17" in markdown
    assert "OLED schema candidates: 24" in markdown
    assert "OLED review items: 12" in markdown
    assert "OLED compiled admission items: 3" in markdown
    assert "High priority review items: 5" in markdown


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
