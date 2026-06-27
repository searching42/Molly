from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent.phase3_corpus_extractor import extract_corpus_records


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


def _document_paths() -> list[Path]:
    return [
        FIXTURE_DIR / "paper_a_parsed_document.json",
        FIXTURE_DIR / "paper_b_parsed_document.json",
        FIXTURE_DIR / "paper_c_parsed_document.json",
    ]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
