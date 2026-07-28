from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.phase3_corpus_extractor import extract_corpus_records
from ai4s_agent.schemas import ParsedDocument


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus_multi_paper"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_phase3_corpus_extractor_consumes_multiple_parsed_documents_with_provenance(tmp_path: Path) -> None:
    expected = _read_json(FIXTURE_DIR / "expected_corpus_records.json")

    result = extract_corpus_records(
        parsed_documents=_document_paths(),
        output_dir=tmp_path,
        run_id="corpus-fixture",
        generated_at=GENERATED_AT,
    )

    assert result.report.document_count == expected["document_count"]
    assert result.report.extracted_record_count == expected["extracted_record_count"]
    assert result.report.paper_ids == expected["paper_ids"]
    assert [record.smiles for record in result.records] == expected["ordered_smiles"]
    assert result.report.record_counts_by_paper == expected["record_counts_by_paper"]
    assert result.report.extraction_rejection_counts_by_paper == expected["extraction_rejection_counts_by_paper"]
    assert all(record.provenance["source_document_id"] for record in result.records)
    assert all(record.provenance["parsed_document_path"].endswith("_parsed_document.json") for record in result.records)
    assert Path(result.corpus_records_json).exists()
    assert Path(result.per_document_extraction_reports_json).exists()
    assert Path(result.corpus_extraction_manifest_json).exists()


def test_phase3_corpus_extractor_writes_oled_candidate_artifacts(tmp_path: Path) -> None:
    result = extract_corpus_records(
        parsed_documents=[_oled_parsed_document()],
        output_dir=tmp_path,
        run_id="corpus-oled-fixture",
        generated_at=GENERATED_AT,
    )

    assert result.report.extracted_record_count == 0
    assert result.report.oled_candidate_count == 1
    assert result.report.oled_schema_candidate_count >= 4
    assert result.report.oled_compiled_record_count == 1
    assert Path(result.oled_candidates_json).exists()
    assert Path(result.oled_schema_candidates_json).exists()
    assert Path(result.oled_compiled_records_json).exists()

    schema_payload = _read_json(Path(result.oled_schema_candidates_json))
    property_ids = {
        item["property_id"]
        for item in schema_payload["schema_candidates"]
        if item["candidate_type"] == "property_observation"
    }
    assert {"plqy", "delta_e_st_ev", "doping_ratio_percent"}.issubset(property_ids)

    manifest = _read_json(Path(result.corpus_extraction_manifest_json))
    assert manifest["report"]["oled_schema_candidate_count"] == result.report.oled_schema_candidate_count
    assert manifest["artifacts"]["oled_schema_candidates_json"] == result.oled_schema_candidates_json


def test_phase3_corpus_extractor_writes_oled_text_evidence_artifact(tmp_path: Path) -> None:
    result = extract_corpus_records(
        parsed_documents=[_oled_text_parsed_document()],
        output_dir=tmp_path,
        run_id="corpus-oled-text-fixture",
        generated_at=GENERATED_AT,
    )

    assert result.records == []
    assert result.report.oled_text_evidence_candidate_count == 3
    assert result.report.oled_text_evidence_candidate_counts_by_paper == {"paper-oled-text-corpus": 3}
    assert Path(result.oled_text_evidence_candidates_json).exists()

    payload = _read_json(Path(result.oled_text_evidence_candidates_json))
    assert payload["run_id"] == "corpus-oled-text-fixture"
    assert {item["property_id"] for item in payload["text_evidence_candidates"]} == {
        "plqy",
        "emission_wavelength_nm",
        "eqe_percent",
    }

    manifest = _read_json(Path(result.corpus_extraction_manifest_json))
    assert manifest["report"]["oled_text_evidence_candidate_count"] == 3
    assert manifest["documents"][0]["oled_text_evidence_candidate_count"] == 3
    assert manifest["artifacts"]["oled_text_evidence_candidates_json"] == result.oled_text_evidence_candidates_json


def _document_paths() -> list[Path]:
    return [
        FIXTURE_DIR / "paper_a_parsed_document.json",
        FIXTURE_DIR / "paper_b_parsed_document.json",
        FIXTURE_DIR / "paper_c_parsed_document.json",
    ]


def _oled_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="paper-oled-corpus",
        source_path="/tmp/paper-oled-corpus.pdf",
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


def _oled_text_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        paper_id="paper-oled-text-corpus",
        source_path="/tmp/paper-oled-text-corpus.pdf",
        parser_backend="mineru_api:hybrid-engine",
        pages=[{"page": 1}],
        elements=[
            {
                "element_id": "el_p1_0001",
                "page": 1,
                "type": "paragraph",
                "text": (
                    "4CzIPN showed a photoluminescence quantum yield of 94 ± 2% in toluene. "
                    "The emission maximum was observed at 507 nm in doped film."
                ),
            },
            {
                "element_id": "el_p2_0001",
                "page": 2,
                "type": "paragraph",
                "text": "The device achieved a maximum EQE of 21.3% at 1000 cd m−2.",
            },
        ],
        tables=[],
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
