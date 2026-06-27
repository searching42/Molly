from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from ai4s_agent.phase3_scientific_extractor import extract_scientific_records
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation, build_scientific_dataset
from ai4s_agent.schemas import ParsedDocument


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase3_to_phase1"
RUN_ID = "phase3-to-phase1-fixture"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_builds_confirmed_training_dataset_with_provenance(tmp_path: Path) -> None:
    extraction = extract_scientific_records(_load_parsed_document(), run_id=RUN_ID, generated_at=GENERATED_AT)
    confirmation = DatasetConfirmation(
        confirmed=True,
        confirmed_by="test-fixture",
        confirmation_source="workflow",
        confirmation_timestamp=GENERATED_AT,
    )

    result = build_scientific_dataset(
        extraction.records,
        output_dir=tmp_path,
        run_id=RUN_ID,
        confirmation=confirmation,
        generated_at=GENERATED_AT,
    )

    actual_rows = _csv_rows(Path(result.training_dataset_csv))
    expected_rows = _csv_rows(FIXTURE_DIR / "expected_dataset.csv")
    rejected = _read_json(Path(result.rejected_records_json))["records"]
    rejected_reasons = Counter(item["reason"] for item in rejected)
    manifest = _read_json(Path(result.dataset_manifest_json))

    assert actual_rows == expected_rows
    assert result.training_record_count == len(expected_rows)
    assert result.candidate_record_count == len(expected_rows)
    assert rejected_reasons["invalid_smiles"] == 1
    assert rejected_reasons["invalid_plqy_range"] == 1
    assert rejected_reasons["invalid_lambda_em_nm_range"] == 1
    assert rejected_reasons["duplicate_conflict"] == 2
    assert manifest["confirmation"]["confirmed"] is True
    assert manifest["confirmation"]["confirmed_by"] == "test-fixture"
    assert manifest["training_record_count"] == len(expected_rows)
    assert manifest["provenance_fields"] == ["paper_id", "page", "table_id", "row_id"]


def test_unconfirmed_dataset_never_enters_training_set(tmp_path: Path) -> None:
    extraction = extract_scientific_records(_load_parsed_document(), run_id=RUN_ID, generated_at=GENERATED_AT)
    confirmation = DatasetConfirmation(
        confirmed=False,
        confirmed_by="",
        confirmation_source="test-unconfirmed",
        confirmation_timestamp=GENERATED_AT,
    )

    result = build_scientific_dataset(
        extraction.records,
        output_dir=tmp_path,
        run_id=RUN_ID,
        confirmation=confirmation,
        generated_at=GENERATED_AT,
    )

    manifest = _read_json(Path(result.dataset_manifest_json))

    assert _csv_rows(Path(result.candidate_dataset_csv))
    assert _csv_rows(Path(result.training_dataset_csv)) == []
    assert result.training_record_count == 0
    assert manifest["status"] == "awaiting_confirmation"
    assert manifest["confirmation"]["confirmed"] is False


def _load_parsed_document() -> ParsedDocument:
    return ParsedDocument.model_validate(_read_json(FIXTURE_DIR / "parsed_document.json"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
