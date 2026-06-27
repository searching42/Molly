from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ai4s_agent.phase3_scientific_extractor import extract_scientific_records
from ai4s_agent.schemas import ConflictReport, ParsedDocument


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase3_to_phase1"
RUN_ID = "phase3-to-phase1-fixture"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_extracts_smiles_properties_and_provenance_from_parsed_document() -> None:
    parsed = _load_parsed_document()
    expected = _read_json(FIXTURE_DIR / "expected_extraction.json")

    result = extract_scientific_records(parsed, run_id=RUN_ID, generated_at=GENERATED_AT)

    assert result.extraction_report.selected_table_count == expected["selected_table_count"]
    assert result.extraction_report.extracted_record_count == expected["extracted_record_count"]
    assert result.extraction_report.rejected_record_count == expected["rejected_record_count"]
    assert result.extraction_report.duplicate_smiles_count == expected["duplicate_smiles_count"]

    records_by_id = {record.record_id: record for record in result.records}
    for expected_record in expected["records"]:
        record = records_by_id[expected_record["record_id"]]
        assert record.smiles == expected_record["smiles"]
        assert record.plqy == expected_record["plqy"]
        assert record.lambda_em_nm == expected_record["lambda_em_nm"]
        assert record.paper_id == expected_record["paper_id"]
        assert record.page == expected_record["page"]
        assert record.table_id == expected_record["table_id"]
        assert record.row_id == expected_record["row_id"]
        assert record.evidence_ref.endswith(f":{record.row_id}")
        assert record.provenance["paper_id"] == expected_record["paper_id"]
        assert record.confidence >= 0.8


def test_reports_rejections_and_duplicate_conflicts_deterministically() -> None:
    parsed = _load_parsed_document()
    expected_extraction = _read_json(FIXTURE_DIR / "expected_extraction.json")
    expected_conflicts = _read_json(FIXTURE_DIR / "expected_conflicts.json")

    result = extract_scientific_records(parsed, run_id=RUN_ID, generated_at=GENERATED_AT)
    reasons = Counter(item["reason"] for item in result.rejected_records)
    conflict_report = ConflictReport.model_validate(result.conflict_report.model_dump(mode="json"))

    assert reasons == expected_extraction["expected_rejection_reasons"]
    assert conflict_report.conflict_count == expected_conflicts["conflict_count"]
    assert conflict_report.input_record_count == expected_conflicts["input_record_count"]
    assert conflict_report.merged_record_count == expected_conflicts["merged_record_count"]
    assert conflict_report.non_conflicting_record_count == expected_conflicts["non_conflicting_record_count"]
    conflict = conflict_report.conflicts[0]
    assert conflict.smiles == expected_conflicts["conflicts"][0]["smiles"]
    assert conflict.property_id == expected_conflicts["conflicts"][0]["property_id"]
    assert conflict.min_value == expected_conflicts["conflicts"][0]["min_value"]
    assert conflict.max_value == expected_conflicts["conflicts"][0]["max_value"]
    assert conflict.tolerance == expected_conflicts["conflicts"][0]["tolerance"]


def _load_parsed_document() -> ParsedDocument:
    return ParsedDocument.model_validate(_read_json(FIXTURE_DIR / "parsed_document.json"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
