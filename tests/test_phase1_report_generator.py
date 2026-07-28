from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.phase1_candidate_ranker import rank_phase1_candidates
from ai4s_agent.phase1_report_generator import generate_phase1_report
from ai4s_agent.phase1_training_orchestrator import run_phase1_training
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase1_training_and_ranking"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_phase1_report_generator_includes_confirmation_provenance_and_hashes(tmp_path: Path) -> None:
    training = run_phase1_training(
        confirmed_training_dataset_csv=FIXTURE_DIR / "confirmed_training_dataset.csv",
        dataset_manifest_json=FIXTURE_DIR / "dataset_manifest.json",
        output_dir=tmp_path / "train",
        run_id="phase1-report-train",
        confirmation=_confirmation(),
        property_ids=["plqy"],
        n_bits=64,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )
    ranking = rank_phase1_candidates(
        candidate_dataset_csv=FIXTURE_DIR / "candidate_dataset.csv",
        training_metadata_json=training.training_metadata_json,
        output_dir=tmp_path / "rank",
        run_id="phase1-report-rank",
        topn=5,
        generated_at=GENERATED_AT,
    )

    report = generate_phase1_report(
        training_metadata_json=training.training_metadata_json,
        ranking_metadata_json=ranking.ranking_metadata_json,
        dataset_manifest_json=FIXTURE_DIR / "dataset_manifest.json",
        output_dir=tmp_path / "report",
        run_id="phase1-report",
        generated_at=GENERATED_AT,
    )
    payload = json.loads(Path(report.report_json).read_text(encoding="utf-8"))
    summary = json.loads(Path(report.report_summary_json).read_text(encoding="utf-8"))
    markdown = Path(report.report_md).read_text(encoding="utf-8")

    assert payload["confirmation"]["confirmed"] is True
    assert payload["dataset_provenance"]["provenance_fields"] == ["paper_id", "page", "table_id", "row_id"]
    assert payload["model_configuration"]["feature_type"] == "rdkit_morgan_fingerprint"
    assert payload["reproducibility"]["dataset_hash"] == training.hashes["dataset_hash"]
    assert payload["ranking_summary"]["topn"] == 5
    assert summary["status"] == "success"
    assert summary["confirmation_confirmed"] is True
    assert "Phase 1 Scientific Modeling Report" in markdown
    assert "Dataset provenance" in markdown


def _confirmation() -> DatasetConfirmation:
    return DatasetConfirmation(
        confirmed=True,
        confirmed_by="test-fixture",
        confirmation_source="phase3-to-phase1-fixture",
        confirmation_timestamp=GENERATED_AT,
    )
