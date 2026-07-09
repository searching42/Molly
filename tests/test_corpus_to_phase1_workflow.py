from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.scientific_dataset_builder import DatasetConfirmation
from ai4s_agent.schemas import ParsedDocument
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


def test_corpus_to_phase1_workflow_surfaces_oled_schema_artifacts_without_confirmation(tmp_path: Path) -> None:
    parsed_path = tmp_path / "paper_oled_parsed_document.json"
    parsed_path.write_text(_oled_parsed_document().model_dump_json(), encoding="utf-8")

    result = run_corpus_to_phase1_workflow(
        parsed_document_paths=[parsed_path],
        output_dir=tmp_path / "corpus",
        run_id="corpus-oled-unconfirmed",
        confirmation=_confirmation(False),
        generated_at=GENERATED_AT,
        property_ids=["plqy"],
        n_bits=64,
        topn=3,
        min_numeric_ratio=0.5,
        min_nonempty=1,
    )

    assert result.status == "awaiting_confirmation"
    assert Path(result.oled_schema_candidates_json).exists()
    assert Path(result.oled_compiled_records_json).exists()

    workflow = _read_json(Path(result.corpus_workflow_report_json))
    manifest = _read_json(Path(result.dataset_manifest_json))
    replay = _read_json(Path(result.corpus_replay_manifest_json))

    assert workflow["summary"]["oled_schema_candidate_count"] >= 4
    assert workflow["artifacts"]["oled_schema_candidates_json"] == result.oled_schema_candidates_json
    assert manifest["corpus"]["oled_schema_candidate_count"] == workflow["summary"]["oled_schema_candidate_count"]
    assert "oled_schema_candidates_json" in replay["hashes"]


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


def _oled_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="paper-oled-workflow",
        source_path="/tmp/paper-oled-workflow.pdf",
        parser_backend="mineru_api:hybrid-engine",
        pages=[{"page": 1}],
        tables=[
            {
                "table_id": "table_oled_components",
                "caption": "Table 1 | Components of the emitter layers of the four colour OLEDs.",
                "headers": [
                    "EL colour",
                    "Host",
                    "Assistant dopant",
                    "Assistant dopant concentration (wt%)",
                    "$\\Delta E_{ST}$  (eV)",
                    "Emitter dopant",
                    "Emitter dopant concentration (wt%)",
                    "$\\Phi_{PL}$  (%)",
                ],
                "rows": [
                    {
                        "EL colour": "Blue",
                        "Host": "DPEPO",
                        "Assistant dopant": "ACRSA",
                        "Assistant dopant concentration (wt%)": "15",
                        "$\\Delta E_{ST}$  (eV)": "0.03",
                        "Emitter dopant": "TBPe",
                        "Emitter dopant concentration (wt%)": "1",
                        "$\\Phi_{PL}$  (%)": "80",
                    }
                ],
                "page": 4,
            }
        ],
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
