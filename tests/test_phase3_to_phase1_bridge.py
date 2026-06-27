from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.phase3_scientific_extractor import extract_scientific_records
from ai4s_agent.phase3_to_phase1_bridge import run_phase3_to_phase1_bridge
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation, build_scientific_dataset
from ai4s_agent.schemas import ParsedDocument


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase3_to_phase1"
RUN_ID = "phase3-to-phase1-fixture"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_phase3_to_phase1_bridge_runs_existing_phase1_adapters(tmp_path: Path) -> None:
    dataset = _build_confirmed_dataset(tmp_path / "dataset")
    confirmation = DatasetConfirmation(
        confirmed=True,
        confirmed_by="test-fixture",
        confirmation_source="bridge-test",
        confirmation_timestamp=GENERATED_AT,
    )

    result = run_phase3_to_phase1_bridge(
        training_dataset_csv=dataset.training_dataset_csv,
        candidate_dataset_csv=dataset.candidate_dataset_csv,
        output_dir=tmp_path / "phase1",
        run_id=RUN_ID,
        confirmation=confirmation,
        property_id="plqy",
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
        n_bits=64,
        topn=3,
    )

    assert result.status == "success"
    assert result.adapter_statuses["inspect_dataset"] == "success"
    assert result.adapter_statuses["clean_dataset"] == "success"
    assert result.adapter_statuses["check_trainability"] == "success"
    assert result.adapter_statuses["run_baseline"] == "success"
    assert result.adapter_statuses["train_model"] == "success"
    assert result.adapter_statuses["predict_candidates"] == "success"
    assert result.adapter_statuses["filter_rank"] == "success"
    assert result.adapter_statuses["render_report"] == "success"
    assert Path(result.phase1_baseline_report_json).exists()
    assert Path(result.candidate_predictions_csv).exists()
    assert Path(result.candidate_ranking_json).exists()
    ranking = _read_json(Path(result.candidate_ranking_json))
    assert ranking["topn"] == 3
    assert len(ranking["candidates"]) == 3
    assert all("weighted_score" in row for row in ranking["candidates"])


def test_phase3_to_phase1_bridge_hard_blocks_unconfirmed_dataset(tmp_path: Path) -> None:
    dataset = _build_confirmed_dataset(tmp_path / "dataset")
    confirmation = DatasetConfirmation(
        confirmed=False,
        confirmed_by="",
        confirmation_source="bridge-test",
        confirmation_timestamp=GENERATED_AT,
    )

    result = run_phase3_to_phase1_bridge(
        training_dataset_csv=dataset.training_dataset_csv,
        candidate_dataset_csv=dataset.candidate_dataset_csv,
        output_dir=tmp_path / "phase1",
        run_id=RUN_ID,
        confirmation=confirmation,
        property_id="plqy",
        generated_at=GENERATED_AT,
    )

    assert result.status == "blocked_confirmation_required"
    assert result.adapter_statuses == {}
    assert result.candidate_ranking_json == ""
    assert not (tmp_path / "phase1" / f"{RUN_ID}_phase1_bridge_report.json").exists()


def _build_confirmed_dataset(output_dir: Path):
    extraction = extract_scientific_records(_load_parsed_document(), run_id=RUN_ID, generated_at=GENERATED_AT)
    confirmation = DatasetConfirmation(
        confirmed=True,
        confirmed_by="test-fixture",
        confirmation_source="bridge-test",
        confirmation_timestamp=GENERATED_AT,
    )
    return build_scientific_dataset(
        extraction.records,
        output_dir=output_dir,
        run_id=RUN_ID,
        confirmation=confirmation,
        generated_at=GENERATED_AT,
    )


def _load_parsed_document() -> ParsedDocument:
    return ParsedDocument.model_validate(_read_json(FIXTURE_DIR / "parsed_document.json"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
