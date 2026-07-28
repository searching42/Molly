from __future__ import annotations

from pathlib import Path

from ai4s_agent.document_parse_benchmark import evaluate_document_parse_against_gold
from ai4s_agent.document_parse_pdfplumber import PdfPlumberDocumentParseProvider
from ai4s_agent.document_parse_provider import DocumentParseRequest
from ai4s_agent.mineru_output_normalizer import discover_mineru_output_bundle, normalize_mineru_output_bundle
from document_parse_test_helpers import fixture_gold, fixture_mineru_output_dir, write_synthetic_pdf


def test_document_parse_benchmark_reports_pdfplumber_baseline_metrics(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    request = DocumentParseRequest(
        run_id="benchmark-pdfplumber",
        input_pdf=str(pdf),
        output_dir=str(tmp_path / "out"),
        provider="pdfplumber",
    )
    result = PdfPlumberDocumentParseProvider().parse(request)

    report = evaluate_document_parse_against_gold(
        parsed_document=result.parsed_document,
        gold=fixture_gold(),
        provider="pdfplumber",
    )

    assert report.provider == "pdfplumber"
    assert report.expected_page_count == 1
    assert report.observed_page_count == 1
    assert report.expected_table_count == 1
    assert report.observed_table_count == 1
    assert report.normalized_text_token_recall > 0.5
    assert report.header_match_rate == 1.0
    assert report.row_count_match is True
    assert report.simple_cell_exact_match_rate == 1.0


def test_document_parse_benchmark_reports_mineru_normalizer_metrics(tmp_path: Path) -> None:
    pdf = write_synthetic_pdf(tmp_path / "paper.pdf")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    for path in fixture_mineru_output_dir().iterdir():
        (bundle_dir / path.name).write_bytes(path.read_bytes())
    normalized = normalize_mineru_output_bundle(
        input_pdf=pdf,
        bundle=discover_mineru_output_bundle(bundle_dir),
        parser_backend="mineru_api:hybrid-engine",
    )

    report = evaluate_document_parse_against_gold(
        parsed_document=normalized.parsed_document,
        gold=fixture_gold(),
        provider="mineru_api_fixture",
    )

    assert report.provider == "mineru_api_fixture"
    assert report.observed_page_count == 1
    assert report.observed_table_count == 1
    assert report.header_match_rate == 1.0
    assert report.simple_cell_exact_match_rate == 1.0
