from __future__ import annotations

import csv
import json
from pathlib import Path

from ai4s_agent.corpus_conflict_auditor import audit_corpus_conflicts
from ai4s_agent.phase3_corpus_extractor import extract_corpus_records


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "corpus_multi_paper"
GENERATED_AT = "2026-06-27T00:00:00Z"


def test_corpus_conflict_auditor_detects_duplicates_conflicts_and_invalid_records(tmp_path: Path) -> None:
    extraction = extract_corpus_records(
        parsed_documents=_document_paths(),
        output_dir=tmp_path / "extract",
        run_id="corpus-fixture",
        generated_at=GENERATED_AT,
    )

    audit = audit_corpus_conflicts(
        records=extraction.records,
        extraction_rejections=extraction.rejected_records,
        output_dir=tmp_path / "audit",
        run_id="corpus-fixture",
        generated_at=GENERATED_AT,
    )
    expected = _read_json(FIXTURE_DIR / "expected_conflict_summary.json")
    summary = _read_json(Path(audit.conflict_summary_json))
    conflict_rows = _csv_rows(Path(audit.conflict_table_csv))

    assert summary["input_record_count"] == expected["input_record_count"]
    assert summary["accepted_record_count"] == expected["accepted_record_count"]
    assert summary["rejected_record_count"] == expected["rejected_record_count"]
    assert summary["consistent_duplicate_count"] == expected["consistent_duplicate_count"]
    assert summary["conflict_count"] == expected["conflict_count"]
    assert summary["unresolved_conflict_count"] == expected["unresolved_conflict_count"]
    assert summary["conflicted_smiles"] == expected["conflicted_smiles"]
    assert summary["consistent_duplicate_smiles"] == expected["consistent_duplicate_smiles"]
    assert summary["reason_counts"] == expected["reason_counts"]
    assert any(row["smiles"] == "CCN" and row["status"] == "rejected" for row in conflict_rows)
    assert any(row["smiles"] == "CCCC" and row["reason"] == "missing_plqy" for row in conflict_rows)
    assert any(row["smiles"] == "CCS" and row["reason"] == "missing_lambda_em_nm" for row in conflict_rows)
    assert all("paper_id" in item["provenance"] for item in audit.rejected_records)


def _document_paths() -> list[Path]:
    return [
        FIXTURE_DIR / "paper_a_parsed_document.json",
        FIXTURE_DIR / "paper_b_parsed_document.json",
        FIXTURE_DIR / "paper_c_parsed_document.json",
    ]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
