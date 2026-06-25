from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "document_parse_provider"


def write_synthetic_pdf(path: Path) -> Path:
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    content = [
        Paragraph("Synthetic OLED Paper", styles["Title"]),
        Spacer(1, 12),
        Paragraph("PLQY values are summarized for OLED emitters.", styles["BodyText"]),
        Spacer(1, 12),
    ]
    table = Table([["SMILES", "PLQY", "lambda_em"], ["CCO", "0.65", "520"]])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    content.append(table)
    doc.build(content)
    return path


def fixture_gold() -> dict[str, Any]:
    return json.loads((FIXTURE_ROOT / "expected_gold.json").read_text(encoding="utf-8"))


def fixture_mineru_output_dir() -> Path:
    return FIXTURE_ROOT / "mineru_output"


def build_zip_from_dir(root: Path) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, arcname=str(path.relative_to(root)))
    return payload.getvalue()
