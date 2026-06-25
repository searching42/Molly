from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai4s_agent.schemas import ParsedDocument


class DocumentParseBenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    parser_backend: str
    expected_page_count: int
    observed_page_count: int
    expected_table_count: int
    observed_table_count: int
    normalized_text_token_recall: float
    header_match_rate: float
    row_count_match: bool
    simple_cell_exact_match_rate: float
    provenance_completeness: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("provenance_completeness")
    @classmethod
    def validate_provenance(cls, value: dict[str, float]) -> dict[str, float]:
        return {str(key): float(item) for key, item in value.items()}


def evaluate_document_parse_against_gold(
    *,
    parsed_document: ParsedDocument,
    gold: dict[str, Any],
    provider: str,
) -> DocumentParseBenchmarkReport:
    expected_tokens = _tokens(" ".join(str(item) for item in gold.get("text_contains", [])))
    observed_tokens = set(_tokens(" ".join(element.text for element in parsed_document.elements)))
    matched = sum(1 for token in expected_tokens if token in observed_tokens)
    expected_tables = gold.get("tables") if isinstance(gold.get("tables"), list) else []
    first_expected = expected_tables[0] if expected_tables and isinstance(expected_tables[0], dict) else {}
    first_observed = parsed_document.tables[0] if parsed_document.tables else None
    expected_headers = [str(item) for item in first_expected.get("headers", [])] if isinstance(first_expected.get("headers"), list) else []
    observed_headers = list(first_observed.headers) if first_observed is not None else []
    header_matches = sum(1 for index, header in enumerate(expected_headers) if index < len(observed_headers) and observed_headers[index] == header)
    expected_rows = first_expected.get("rows") if isinstance(first_expected.get("rows"), list) else []
    observed_rows = list(first_observed.rows) if first_observed is not None else []
    cell_total = 0
    cell_matches = 0
    for index, row in enumerate(expected_rows):
        if not isinstance(row, dict):
            continue
        observed_row = observed_rows[index] if index < len(observed_rows) else {}
        for key, value in row.items():
            cell_total += 1
            if str(observed_row.get(str(key), "")) == str(value):
                cell_matches += 1
    page_target = int(gold.get("page_count") or 0)
    provenance = {
        "page_present": _fraction(sum(1 for element in parsed_document.elements if int(element.page) > 0), len(parsed_document.elements)),
        "element_id_present": _fraction(sum(1 for element in parsed_document.elements if str(element.element_id).strip()), len(parsed_document.elements)),
        "bbox_present": _fraction(sum(1 for element in parsed_document.elements if element.bbox is not None), len(parsed_document.elements)),
        "table_page_present": _fraction(sum(1 for table in parsed_document.tables if int(table.page) > 0), len(parsed_document.tables)),
    }
    return DocumentParseBenchmarkReport(
        provider=provider,
        parser_backend=parsed_document.parser_backend,
        expected_page_count=page_target,
        observed_page_count=len(parsed_document.pages),
        expected_table_count=len(expected_tables),
        observed_table_count=len(parsed_document.tables),
        normalized_text_token_recall=_fraction(matched, len(expected_tokens)),
        header_match_rate=_fraction(header_matches, len(expected_headers)),
        row_count_match=len(expected_rows) == len(observed_rows),
        simple_cell_exact_match_rate=_fraction(cell_matches, cell_total),
        provenance_completeness=provenance,
        warnings=[],
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_@.+%-]+", text.lower())


def _fraction(numerator: int | bool, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return float(int(numerator)) / float(denominator)
