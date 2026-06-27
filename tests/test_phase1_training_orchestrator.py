from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.phase1_training_orchestrator import (
    DatasetNotConfirmedError,
    run_phase1_training,
)
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase1_training_and_ranking"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_phase1_training_orchestrator_persists_reproducible_model_artifacts(tmp_path: Path) -> None:
    first = run_phase1_training(
        confirmed_training_dataset_csv=FIXTURE_DIR / "confirmed_training_dataset.csv",
        dataset_manifest_json=FIXTURE_DIR / "dataset_manifest.json",
        output_dir=tmp_path / "first",
        run_id="phase1-train-fixture",
        confirmation=_confirmation(),
        property_ids=["plqy"],
        n_bits=64,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )
    second = run_phase1_training(
        confirmed_training_dataset_csv=FIXTURE_DIR / "confirmed_training_dataset.csv",
        dataset_manifest_json=FIXTURE_DIR / "dataset_manifest.json",
        output_dir=tmp_path / "second",
        run_id="phase1-train-fixture",
        confirmation=_confirmation(),
        property_ids=["plqy"],
        n_bits=64,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )

    first_meta = json.loads(Path(first.training_metadata_json).read_text(encoding="utf-8"))
    second_meta = json.loads(Path(second.training_metadata_json).read_text(encoding="utf-8"))

    assert first.status == "success"
    assert Path(first.models["plqy"]["model_path"]).exists()
    assert Path(first.feature_config_json).exists()
    assert first_meta["feature_config"]["feature_type"] == "rdkit_morgan_fingerprint"
    assert first_meta["feature_config"]["random_seed"] == 0
    assert first_meta["hashes"]["dataset_hash"] == second_meta["hashes"]["dataset_hash"]
    assert first_meta["hashes"]["config_hash"] == second_meta["hashes"]["config_hash"]
    assert first_meta["models"]["plqy"]["model_hash"] == second_meta["models"]["plqy"]["model_hash"]
    assert first_meta["confirmation"]["confirmed"] is True
    assert first_meta["dataset_manifest"]["provenance_fields"] == ["paper_id", "page", "table_id", "row_id"]


def test_phase1_training_orchestrator_rejects_unconfirmed_dataset(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotConfirmedError):
        run_phase1_training(
            confirmed_training_dataset_csv=FIXTURE_DIR / "confirmed_training_dataset.csv",
            dataset_manifest_json=FIXTURE_DIR / "dataset_manifest.json",
            output_dir=tmp_path,
            run_id="phase1-unconfirmed",
            confirmation=DatasetConfirmation(
                confirmed=False,
                confirmed_by="",
                confirmation_source="test",
                confirmation_timestamp=GENERATED_AT,
            ),
            generated_at=GENERATED_AT,
        )


def _confirmation() -> DatasetConfirmation:
    return DatasetConfirmation(
        confirmed=True,
        confirmed_by="test-fixture",
        confirmation_source="phase3-to-phase1-fixture",
        confirmation_timestamp=GENERATED_AT,
    )
