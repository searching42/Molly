from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.scientific_dataset_builder import DatasetConfirmation
from ai4s_agent.workflows.corpus_to_phase1_workflow import run_corpus_to_phase1_workflow


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus_multi_paper"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_corpus_to_phase1_workflow_runs_confirmed_corpus_end_to_end(tmp_path: Path) -> None:
    result = run_corpus_to_phase1_workflow(
        parsed_document_paths=_document_paths(),
        output_dir=tmp_path / "corpus",
        run_id="corpus-fixture",
        confirmation=_confirmation(True),
        generated_at=GENERATED_AT,
        property_ids=["plqy"],
        n_bits=64,
        topn=3,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )
    expected_manifest = _read_json(FIXTURE_DIR / "expected_dataset_manifest.json")
    workflow = _read_json(Path(result.corpus_workflow_report_json))
    manifest = _read_json(Path(result.dataset_manifest_json))
    replay = _read_json(Path(result.corpus_replay_manifest_json))

    assert result.status == "success"
    assert manifest["candidate_record_count"] == expected_manifest["candidate_record_count"]
    assert manifest["training_record_count"] == expected_manifest["training_record_count"]
    assert manifest["rejected_record_count"] == expected_manifest["rejected_record_count"]
    assert manifest["artifacts"]["training_dataset_csv"] == result.training_dataset_csv
    assert Path(result.full_phase1_pipeline_json).exists()
    assert Path(result.corpus_report_json).exists()
    assert replay["external_services_required"] is False
    assert workflow["summary"]["phase1_status"] == "success"
    assert workflow["summary"]["top_ranked_candidate_count"] == 3


def test_corpus_to_phase1_workflow_stops_before_phase1_without_confirmation(tmp_path: Path) -> None:
    result = run_corpus_to_phase1_workflow(
        parsed_document_paths=_document_paths(),
        output_dir=tmp_path / "corpus",
        run_id="corpus-unconfirmed",
        confirmation=_confirmation(False),
        generated_at=GENERATED_AT,
        property_ids=["plqy"],
        n_bits=64,
        topn=3,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )

    assert result.status == "awaiting_confirmation"
    assert Path(result.corpus_workflow_report_json).exists()
    assert result.full_phase1_pipeline_json == ""
    assert result.report_json == ""


def test_corpus_to_phase1_workflow_reproducibility_hashes_are_stable(tmp_path: Path) -> None:
    first = run_corpus_to_phase1_workflow(
        parsed_document_paths=_document_paths(),
        output_dir=tmp_path / "first",
        run_id="corpus-fixture",
        confirmation=_confirmation(True),
        generated_at=GENERATED_AT,
        property_ids=["plqy"],
        n_bits=64,
        topn=3,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )
    second = run_corpus_to_phase1_workflow(
        parsed_document_paths=_document_paths(),
        output_dir=tmp_path / "second",
        run_id="corpus-fixture",
        confirmation=_confirmation(True),
        generated_at=GENERATED_AT,
        property_ids=["plqy"],
        n_bits=64,
        topn=3,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )

    first_replay = _read_json(Path(first.corpus_replay_manifest_json))
    second_replay = _read_json(Path(second.corpus_replay_manifest_json))
    assert first_replay["hashes"]["corpus_records_json"] == second_replay["hashes"]["corpus_records_json"]
    assert first_replay["hashes"]["training_dataset_csv"] == second_replay["hashes"]["training_dataset_csv"]
    assert first_replay["hashes"]["ranked_candidates_csv"] == second_replay["hashes"]["ranked_candidates_csv"]


def _document_paths() -> list[Path]:
    return [
        FIXTURE_DIR / "paper_a_parsed_document.json",
        FIXTURE_DIR / "paper_b_parsed_document.json",
        FIXTURE_DIR / "paper_c_parsed_document.json",
    ]


def _confirmation(confirmed: bool) -> DatasetConfirmation:
    return DatasetConfirmation(
        confirmed=confirmed,
        confirmed_by="test-fixture" if confirmed else "",
        confirmation_source="corpus-fixture",
        confirmation_timestamp=GENERATED_AT,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
