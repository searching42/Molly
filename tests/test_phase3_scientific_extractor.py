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


def test_extracts_oled_schema_candidates_from_non_smiles_oled_tables() -> None:
    parsed = ParsedDocument(
        paper_id="paper-oled-real-shape",
        source_path="/tmp/paper-oled-real-shape.pdf",
        parser_backend="mineru_api:hybrid-engine",
        pages=[{"page": 1, "width": 595.0, "height": 779.0}],
        elements=[],
        tables=[
            {
                "table_id": "table_p4_0043",
                "caption": "Table 1 | Components of the emitter layers of the four colour OLEDs.",
                "headers": [
                    "EL colour",
                    "Host $(S_1^H, T_1^H)^*$  (eV)",
                    "Assistant dopant $(S_1^A, T_1^A)^*$  (eV)",
                    "Assistant dopant concentration (wt%)",
                    "$\\Delta E_{ST}$  (eV)",
                    "Emitter dopant  $S_1^{E*}$  (eV)",
                    "Emitter dopant concentration (wt%)",
                    "$\\Phi_{PL}$  (%)",
                ],
                "rows": [
                    {
                        "EL colour": "Blue",
                        "Host $(S_1^H, T_1^H)^*$  (eV)": "DPEPO(3.50, 3.00)",
                        "Assistant dopant $(S_1^A, T_1^A)^*$  (eV)": "ACRSA(2.55, 2.52)",
                        "Assistant dopant concentration (wt%)": "15",
                        "$\\Delta E_{ST}$  (eV)": "0.03",
                        "Emitter dopant  $S_1^{E*}$  (eV)": "TBPe(2.69)",
                        "Emitter dopant concentration (wt%)": "1",
                        "$\\Phi_{PL}$  (%)": "80 ± 2",
                    }
                ],
                "page": 4,
                "markdown": "",
                "source_bbox": {"x0": 73.0, "y0": 85.0, "x1": 921.0, "y1": 237.0},
            },
            {
                "table_id": "table_p5_0060",
                "caption": "Table 2 | Device performance of the four colour OLEDs with assistant dopants.",
                "headers": [
                    "Device",
                    "Turn on voltage (V)",
                    "Max EQE (%)",
                    "Max CE (cd A $^{-1}$ )",
                    "CIE",
                    "Performance at 1,000 cd m $^{-2}$",
                ],
                "rows": [
                    {
                        "Device": "Blue",
                        "Turn on voltage (V)": "4.7",
                        "Max EQE (%)": "13.4",
                        "Max CE (cd A $^{-1}$ )": "27",
                        "CIE": "(0.17, 0.30)",
                        "Performance at 1,000 cd m $^{-2}$": "7",
                    }
                ],
                "page": 5,
                "markdown": "",
                "source_bbox": {"x0": 73.0, "y0": 620.0, "x1": 922.0, "y1": 730.0},
            },
        ],
    )

    result = extract_scientific_records(parsed, run_id=RUN_ID, generated_at=GENERATED_AT)

    assert result.records == []
    assert result.extraction_report.selected_table_count == 0
    assert result.extraction_report.oled_candidate_count == 2
    assert result.extraction_report.oled_schema_candidate_count >= 8
    assert result.extraction_report.oled_compiled_record_count == 2
    property_ids = {
        candidate.property_id
        for candidate in result.oled_schema_candidates
        if candidate.candidate_type.value == "property_observation"
    }
    assert {"plqy", "delta_e_st_ev", "eqe_percent", "doping_ratio_percent"}.issubset(property_ids)
    material_roles = {
        candidate.material_role
        for candidate in result.oled_schema_candidates
        if candidate.candidate_type.value == "material_role"
    }
    assert {"host", "assistant_dopant", "emitter_dopant"}.issubset(material_roles)
    assert all(record.layered_record is not None for record in result.oled_compiled_records)


def test_oled_schema_extraction_skips_repeated_header_rows_in_parsed_tables() -> None:
    parsed = ParsedDocument(
        paper_id="paper-oled-header-row",
        source_path="/tmp/paper-oled-header-row.pdf",
        parser_backend="mineru_api:hybrid-engine",
        pages=[{"page": 1}],
        tables=[
            {
                "table_id": "table_p5_0060",
                "caption": "Table 2 | Device performance of the four colour OLEDs with assistant dopants.",
                "headers": [
                    "Device",
                    "Turn on voltage (V)",
                    "Max EQE (%)",
                    "CIE",
                    "Performance at 1,000 cd m $^{-2}$",
                ],
                "rows": [
                    {
                        "Device": "Device",
                        "Turn on voltage (V)": "Turn on voltage (V)",
                        "Max EQE (%)": "Max EQE (%)",
                        "CIE": "CIE",
                        "Performance at 1,000 cd m $^{-2}$": "PE (lm W $^{-1}$ )",
                    },
                    {
                        "Device": "Blue",
                        "Turn on voltage (V)": "4.7",
                        "Max EQE (%)": "13.4",
                        "CIE": "(0.17, 0.30)",
                        "Performance at 1,000 cd m $^{-2}$": "7",
                    },
                ],
                "page": 5,
            }
        ],
    )

    result = extract_scientific_records(parsed, run_id=RUN_ID, generated_at=GENERATED_AT)

    assert result.extraction_report.oled_candidate_count == 1
    assert result.extraction_report.oled_compiled_record_count == 1
    condition_values = [
        candidate.condition_value
        for candidate in result.oled_schema_candidates
        if candidate.candidate_type.value == "measurement_condition"
    ]
    assert "Turn on voltage (V)" not in condition_values
    assert 4.7 in condition_values


def test_extracts_oled_text_candidates_from_no_table_parsed_documents() -> None:
    parsed = ParsedDocument(
        paper_id="paper-oled-text-only",
        source_path="/tmp/paper-oled-text-only.pdf",
        parser_backend="mineru_api:hybrid-engine",
        pages=[{"page": 1}],
        elements=[
            {
                "element_id": "el_p1_0001",
                "page": 1,
                "type": "paragraph",
                "text": "4CzIPN has intense green emission at 507 nm and photoluminescence quantum yield of 94 ± 2% in toluene.",
                "markdown": "",
                "bbox": [72.0, 90.0, 520.0, 120.0],
            }
        ],
        tables=[],
    )

    result = extract_scientific_records(parsed, run_id=RUN_ID, generated_at=GENERATED_AT)

    assert result.records == []
    assert result.extraction_report.input_table_count == 0
    assert result.extraction_report.oled_candidate_count == 1
    assert result.extraction_report.oled_text_evidence_candidate_count == 2
    assert result.oled_text_evidence_candidates[0].property_id == "emission_wavelength_nm"
    assert result.oled_text_evidence_candidates[1].property_id == "plqy"
    assert result.extraction_report.oled_schema_candidate_count == 0
    assert result.extraction_report.oled_compiled_record_count == 0
    assert result.oled_candidates[0].candidate_type.value == "text"
    assert "photoluminescence_quantum_yield" in result.oled_candidates[0].matched_terms
    assert "property_keyword" in {signal.value for signal in result.oled_candidates[0].relevance_signals}


def _load_parsed_document() -> ParsedDocument:
    return ParsedDocument.model_validate(_read_json(FIXTURE_DIR / "parsed_document.json"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
