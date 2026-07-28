from __future__ import annotations

from pathlib import Path

from ai4s_agent.document_parse_pdfplumber import PdfPlumberDocumentParseProvider
from ai4s_agent.document_parse_provider import DocumentParseRequest
from document_parse_test_helpers import write_synthetic_pdf


def test_pdfplumber_document_parse_provider_extracts_text_and_table(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    request = DocumentParseRequest(
        run_id="pdfplumber-baseline",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="pdfplumber",
    )

    result = PdfPlumberDocumentParseProvider().parse(request)

    assert result.ok is True
    assert result.status == "success"
    assert result.provider == "pdfplumber"
    assert result.parser_backend == "pdfplumber_local"
    assert result.parsed_document is not None
    assert any("Synthetic OLED Paper" in element.text for element in result.parsed_document.elements)
    assert result.parsed_document.tables[0].headers == ["SMILES", "PLQY", "lambda_em"]
    assert result.parsed_document.tables[0].rows == [{"SMILES": "CCO", "PLQY": "0.65", "lambda_em": "520"}]
    assert Path(result.outputs.parsed_document_json).exists()
    assert Path(result.outputs.parsed_document_markdown).exists()
    assert Path(result.outputs.parser_audit_json).exists()
    assert result.audit.request_provider == "pdfplumber"
    assert result.audit.selected_provider == "pdfplumber"
