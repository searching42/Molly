from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai4s_agent.adapters.phase3 import _sha256_file


@dataclass(frozen=True)
class SyntheticLiveCorpusPdf:
    document_id: str
    paper_id: str
    pdf_path: str
    sha256: str
    expected_record_count: int


_CORPUS_TABLES = [
    {
        "document_id": "paper_a",
        "paper_id": "corpus-paper-a",
        "title": "Synthetic OLED Corpus Paper A",
        "rows": [
            ["a-001", "CCO", "65%", "512", "0.96"],
            ["a-002", "CCN", "0.58", "498", "0.93"],
            ["a-003", "CCCO", "72%", "530", "0.92"],
            ["a-004", "c1ccccc1", "0.44", "410", "0.90"],
        ],
    },
    {
        "document_id": "paper_b",
        "paper_id": "corpus-paper-b",
        "title": "Synthetic OLED Corpus Paper B",
        "rows": [
            ["b-001", "CCO", "66%", "513", "0.95"],
            ["b-002", "CCN", "0.91", "500", "0.94"],
            ["b-003", "CCCl", "36%", "460", "0.89"],
        ],
    },
    {
        "document_id": "paper_c",
        "paper_id": "corpus-paper-c",
        "title": "Synthetic OLED Corpus Paper C",
        "rows": [
            ["c-001", "not-a-smiles", "0.42", "515", "0.88"],
            ["c-002", "CCCC", "", "510", "0.86"],
            ["c-003", "CCS", "0.57", "", "0.85"],
            ["c-004", "CCBr", "0.61", "545", "0.90"],
        ],
    },
]


def write_synthetic_live_corpus_pdfs(output_dir: str | Path) -> list[SyntheticLiveCorpusPdf]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[SyntheticLiveCorpusPdf] = []
    for item in _CORPUS_TABLES:
        pdf_path = output_path / f"{item['document_id']}.pdf"
        _write_table_pdf(
            pdf_path,
            title=str(item["title"]),
            paper_id=str(item["paper_id"]),
            rows=list(item["rows"]),
        )
        results.append(
            SyntheticLiveCorpusPdf(
                document_id=str(item["document_id"]),
                paper_id=str(item["paper_id"]),
                pdf_path=str(pdf_path),
                sha256=_sha256_file(pdf_path),
                expected_record_count=len(item["rows"]),
            )
        )
    return results


def _write_table_pdf(path: Path, *, title: str, paper_id: str, rows: list[list[str]]) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("reportlab is required to generate the live corpus acceptance PDFs") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pdf = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    except TypeError:  # pragma: no cover - older reportlab
        pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    left = 54
    top = height - 54
    pdf.setTitle(title)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(left, top, title)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(left, top - 22, f"paper_id: {paper_id}")
    pdf.drawString(left, top - 40, "Synthetic-data notice: generated locally for parser acceptance only.")
    pdf.drawString(left, top - 58, "Table 1 OLED measurements")

    headers = ["molecule_id", "SMILES", "PLQY", "lambda_em_nm", "confidence"]
    col_widths = [82, 112, 72, 96, 82]
    row_height = 22
    table_top = top - 86
    x_positions = [left]
    for width_item in col_widths[:-1]:
        x_positions.append(x_positions[-1] + width_item)
    all_rows = [headers, *rows]
    table_width = sum(col_widths)
    table_height = row_height * len(all_rows)
    pdf.setLineWidth(0.75)
    for row_index, row in enumerate(all_rows):
        y = table_top - row_index * row_height
        pdf.line(left, y, left + table_width, y)
        pdf.setFont("Helvetica-Bold" if row_index == 0 else "Helvetica", 9)
        for col_index, value in enumerate(row):
            pdf.drawString(x_positions[col_index] + 4, y - 15, str(value))
    pdf.line(left, table_top - table_height, left + table_width, table_top - table_height)
    x = left
    for col_width in col_widths:
        pdf.line(x, table_top, x, table_top - table_height)
        x += col_width
    pdf.line(left + table_width, table_top, left + table_width, table_top - table_height)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(left, table_top - table_height - 24, "Synthetic fixture values. Do not use as scientific evidence.")
    pdf.showPage()
    pdf.save()
