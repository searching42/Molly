from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.corpus_report_generator import generate_corpus_report


GENERATED_AT = "2026-06-27T00:00:00Z"


def test_corpus_report_generator_summarizes_corpus_phase1_and_reproducibility(tmp_path: Path) -> None:
    conflict_summary = tmp_path / "conflict_summary.json"
    phase1_pipeline = tmp_path / "full_phase1_pipeline.json"
    reproducibility = tmp_path / "corpus_reproducibility_report.json"
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

    report = generate_corpus_report(
        conflict_summary_json=conflict_summary,
        phase1_pipeline_json=phase1_pipeline,
        reproducibility_report_json=reproducibility,
        ranked_candidates=[{"SMILES": "CCO", "weighted_score": "1.0"}],
        output_dir=tmp_path / "report",
        run_id="corpus-report",
        generated_at=GENERATED_AT,
    )
    payload = _read_json(Path(report.corpus_report_json))
    summary = _read_json(Path(report.corpus_summary_json))
    markdown = Path(report.corpus_report_md).read_text(encoding="utf-8")

    assert payload["document_count"] == 3
    assert payload["phase1_status"] == "success"
    assert payload["top_ranked_candidates"][0]["SMILES"] == "CCO"
    assert summary["conflict_count"] == 1
    assert "Corpus Evaluation And Reproducibility Audit" in markdown


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
