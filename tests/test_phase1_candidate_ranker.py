from __future__ import annotations

import csv
import json
from pathlib import Path

from ai4s_agent.phase1_candidate_ranker import rank_phase1_candidates
from ai4s_agent.phase1_training_orchestrator import run_phase1_training
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase1_training_and_ranking"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_phase1_candidate_ranker_produces_stable_model_based_ranking(tmp_path: Path) -> None:
    training = run_phase1_training(
        confirmed_training_dataset_csv=FIXTURE_DIR / "confirmed_training_dataset.csv",
        dataset_manifest_json=FIXTURE_DIR / "dataset_manifest.json",
        output_dir=tmp_path / "train",
        run_id="phase1-ranker-train",
        confirmation=_confirmation(),
        property_ids=["plqy"],
        n_bits=64,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )

    first = rank_phase1_candidates(
        candidate_dataset_csv=FIXTURE_DIR / "candidate_dataset.csv",
        training_metadata_json=training.training_metadata_json,
        output_dir=tmp_path / "rank-first",
        run_id="phase1-rank",
        topn=5,
        generated_at=GENERATED_AT,
    )
    second = rank_phase1_candidates(
        candidate_dataset_csv=FIXTURE_DIR / "candidate_dataset.csv",
        training_metadata_json=training.training_metadata_json,
        output_dir=tmp_path / "rank-second",
        run_id="phase1-rank",
        topn=5,
        generated_at=GENERATED_AT,
    )

    first_rows = _csv_rows(Path(first.ranked_candidates_csv))
    second_rows = _csv_rows(Path(second.ranked_candidates_csv))
    metadata = json.loads(Path(first.ranking_metadata_json).read_text(encoding="utf-8"))

    assert first.status == "success"
    assert first_rows == second_rows
    assert len(first_rows) == 5
    assert "plqy_pred" in first_rows[0]
    assert "weighted_score" in first_rows[0]
    assert metadata["scoring"]["model_based"] is True
    assert metadata["hashes"]["ranking_hash"] == second.hashes["ranking_hash"]
    assert metadata["property_ids"] == ["plqy"]


def _confirmation() -> DatasetConfirmation:
    return DatasetConfirmation(
        confirmed=True,
        confirmed_by="test-fixture",
        confirmation_source="phase3-to-phase1-fixture",
        confirmation_timestamp=GENERATED_AT,
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
