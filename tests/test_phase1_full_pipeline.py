from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.phase1_training_orchestrator import DatasetNotConfirmedError
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation
from ai4s_agent.workflows.phase1_full_pipeline import run_phase1_full_pipeline


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase1_training_and_ranking"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_phase1_full_pipeline_executes_end_to_end_with_confirmed_dataset(tmp_path: Path) -> None:
    result = run_phase1_full_pipeline(
        confirmed_training_dataset_csv=FIXTURE_DIR / "confirmed_training_dataset.csv",
        candidate_dataset_csv=FIXTURE_DIR / "candidate_dataset.csv",
        dataset_manifest_json=FIXTURE_DIR / "dataset_manifest.json",
        output_dir=tmp_path / "pipeline",
        run_id="phase1-full",
        confirmation=_confirmation(),
        property_ids=["plqy"],
        n_bits=64,
        topn=5,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )

    pipeline = json.loads(Path(result.full_phase1_pipeline_json).read_text(encoding="utf-8"))

    assert result.status == "success"
    assert Path(result.trained_model_paths["plqy"]).exists()
    assert Path(result.ranked_candidates_csv).exists()
    assert Path(result.report_json).exists()
    assert pipeline["confirmation"]["confirmed"] is True
    assert pipeline["hashes"]["dataset_hash"] == result.hashes["dataset_hash"]
    assert pipeline["artifacts"]["ranked_candidates_csv"] == result.ranked_candidates_csv


def test_phase1_full_pipeline_hard_blocks_unconfirmed_dataset(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotConfirmedError):
        run_phase1_full_pipeline(
            confirmed_training_dataset_csv=FIXTURE_DIR / "confirmed_training_dataset.csv",
            candidate_dataset_csv=FIXTURE_DIR / "candidate_dataset.csv",
            dataset_manifest_json=FIXTURE_DIR / "dataset_manifest.json",
            output_dir=tmp_path / "pipeline",
            run_id="phase1-full-unconfirmed",
            confirmation=DatasetConfirmation(
                confirmed=False,
                confirmed_by="",
                confirmation_source="test",
                confirmation_timestamp=GENERATED_AT,
            ),
            generated_at=GENERATED_AT,
        )
    assert not (tmp_path / "pipeline" / "full_phase1_pipeline.json").exists()


def _confirmation() -> DatasetConfirmation:
    return DatasetConfirmation(
        confirmed=True,
        confirmed_by="test-fixture",
        confirmation_source="phase3-to-phase1-fixture",
        confirmation_timestamp=GENERATED_AT,
    )
