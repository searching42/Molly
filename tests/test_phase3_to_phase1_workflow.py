from __future__ import annotations

import csv
import json
from pathlib import Path

from ai4s_agent.scientific_dataset_builder import DatasetConfirmation
from ai4s_agent.schemas import ParsedDocument
from ai4s_agent.workflows.phase3_to_phase1_workflow import run_phase3_to_phase1_workflow


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase3_to_phase1"
RUN_ID = "phase3-to-phase1-fixture"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_phase3_to_phase1_workflow_runs_end_to_end_offline(tmp_path: Path) -> None:
    confirmation = DatasetConfirmation(
        confirmed=True,
        confirmed_by="test-fixture",
        confirmation_source="workflow-test",
        confirmation_timestamp=GENERATED_AT,
    )

    result = run_phase3_to_phase1_workflow(
        parsed_document=_load_parsed_document(),
        output_dir=tmp_path / "run",
        run_id=RUN_ID,
        confirmation=confirmation,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
        n_bits=64,
        topn=3,
    )

    assert result.status == "success"
    assert Path(result.full_pipeline_report_json).exists()
    assert Path(result.scientific_dataset_manifest_json).exists()
    assert Path(result.phase1_baseline_report_json).exists()
    assert Path(result.candidate_ranking_json).exists()
    report = _read_json(Path(result.full_pipeline_report_json))
    assert report["summary"]["extracted_record_count"] == 15
    assert report["summary"]["training_record_count"] == 9
    assert report["summary"]["candidate_ranking_count"] == 3


def test_phase3_to_phase1_workflow_is_deterministic_for_fixed_fixture(tmp_path: Path) -> None:
    confirmation = DatasetConfirmation(
        confirmed=True,
        confirmed_by="test-fixture",
        confirmation_source="workflow-test",
        confirmation_timestamp=GENERATED_AT,
    )

    first = run_phase3_to_phase1_workflow(
        parsed_document=_load_parsed_document(),
        output_dir=tmp_path / "first",
        run_id=RUN_ID,
        confirmation=confirmation,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
        n_bits=64,
        topn=3,
    )
    second = run_phase3_to_phase1_workflow(
        parsed_document=_load_parsed_document(),
        output_dir=tmp_path / "second",
        run_id=RUN_ID,
        confirmation=confirmation,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
        n_bits=64,
        topn=3,
    )

    first_report = _read_json(Path(first.full_pipeline_report_json))
    second_report = _read_json(Path(second.full_pipeline_report_json))
    assert first_report["summary"] == second_report["summary"]
    assert _csv_rows(Path(first.training_dataset_csv)) == _csv_rows(Path(second.training_dataset_csv))
    assert _ranking_smiles(Path(first.candidate_ranking_json)) == _ranking_smiles(Path(second.candidate_ranking_json))


def test_phase3_to_phase1_workflow_stops_before_phase1_without_confirmation(tmp_path: Path) -> None:
    confirmation = DatasetConfirmation(
        confirmed=False,
        confirmed_by="",
        confirmation_source="workflow-test",
        confirmation_timestamp=GENERATED_AT,
    )

    result = run_phase3_to_phase1_workflow(
        parsed_document=_load_parsed_document(),
        output_dir=tmp_path / "run",
        run_id=RUN_ID,
        confirmation=confirmation,
        generated_at=GENERATED_AT,
        min_numeric_ratio=0.5,
        min_nonempty=1,
        n_bits=64,
        topn=3,
    )

    assert result.status == "awaiting_confirmation"
    assert Path(result.full_pipeline_report_json).exists()
    assert Path(result.scientific_dataset_manifest_json).exists()
    assert result.phase1_baseline_report_json == ""
    assert result.candidate_ranking_json == ""


def _load_parsed_document() -> ParsedDocument:
    return ParsedDocument.model_validate(_read_json(FIXTURE_DIR / "parsed_document.json"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _ranking_smiles(path: Path) -> list[str]:
    payload = _read_json(path)
    return [str(row["SMILES"]) for row in payload["candidates"]]
