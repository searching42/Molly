from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.domains.oled_mineru_candidates import (
    OledMineruCandidate,
    extract_oled_mineru_candidates_from_document,
)
from ai4s_agent.domains.oled_mineru_semantic_mapping import (
    OledSchemaCandidate,
    map_oled_mineru_candidates_to_schema_candidates,
)
from ai4s_agent.domains.oled_schema_candidate_compiler import (
    OledCompiledLayeredRecordCandidate,
    compile_oled_schema_candidates_to_layered_records,
)
from ai4s_agent.domains.oled_text_evidence_candidates import (
    OledTextEvidenceCandidate,
    extract_oled_text_evidence_candidates_from_document,
)
from ai4s_agent.schemas import ConflictGroup, ConflictReport, ParsedDocument, ParsedTable


class StructuredScientificRecord(BaseModel):
    record_id: str
    smiles: str
    plqy: float | None = None
    lambda_em_nm: float | None = None
    paper_id: str
    source_id: str
    page: int
    table_id: str
    row_index: int
    row_id: str
    evidence_ref: str
    confidence: float
    confidence_factors: dict[str, Any] = Field(default_factory=dict)
    raw_values: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    status: str = "extracted"

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value


class ExtractionReport(BaseModel):
    run_id: str
    paper_id: str
    input_table_count: int
    selected_table_count: int
    oled_candidate_count: int = 0
    oled_text_evidence_candidate_count: int = 0
    oled_schema_candidate_count: int = 0
    oled_compiled_record_count: int = 0
    input_row_count: int
    extracted_record_count: int
    rejected_record_count: int
    duplicate_smiles_count: int
    generated_at: str
    notes: list[str] = Field(default_factory=list)


class ScientificExtractionResult(BaseModel):
    records: list[StructuredScientificRecord]
    rejected_records: list[dict[str, Any]]
    extraction_report: ExtractionReport
    conflict_report: ConflictReport
    oled_candidates: list[OledMineruCandidate] = Field(default_factory=list)
    oled_text_evidence_candidates: list[OledTextEvidenceCandidate] = Field(default_factory=list)
    oled_schema_candidates: list[OledSchemaCandidate] = Field(default_factory=list)
    oled_compiled_records: list[OledCompiledLayeredRecordCandidate] = Field(default_factory=list)


_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_SMILES_ALIASES = {"smiles", "canonicalsmiles", "molsmiles", "structure", "chromophore"}
_PLQY_TOKENS = {"plqy", "photoluminescencequantumyield", "quantumyield"}
_LAMBDA_TOKENS = {
    "lambdaem",
    "lambdaemnm",
    "emissionwavelength",
    "emissionmax",
    "emmax",
    "lambda",
}


def extract_scientific_records(
    parsed_document: ParsedDocument,
    *,
    run_id: str = "phase3-scientific-extraction",
    confidence_threshold: float = 0.6,
    plqy_conflict_tolerance: float = 0.03,
    lambda_conflict_tolerance_nm: float = 5.0,
    generated_at: str | None = None,
) -> ScientificExtractionResult:
    generated = generated_at or now_iso()
    selected_tables = [_table_columns(table) for table in parsed_document.tables]
    selected_tables = [item for item in selected_tables if item["smiles_col"] and (item["plqy_col"] or item["lambda_col"])]

    records: list[StructuredScientificRecord] = []
    rejected: list[dict[str, Any]] = []
    input_row_count = 0
    for table_info in selected_tables:
        table = table_info["table"]
        smiles_col = str(table_info["smiles_col"])
        plqy_col = str(table_info["plqy_col"] or "")
        lambda_col = str(table_info["lambda_col"] or "")
        confidence_col = str(table_info["confidence_col"] or "")
        for row_index, row in enumerate(table.rows):
            input_row_count += 1
            row_id = f"row_{row_index + 1:06d}"
            raw_values = {str(key): str(value) for key, value in row.items()}
            base_attempt = {
                "paper_id": parsed_document.paper_id,
                "source_id": parsed_document.paper_id,
                "page": table.page,
                "table_id": table.table_id,
                "row_index": row_index,
                "row_id": row_id,
                "raw_values": raw_values,
            }
            smiles = _row_value(row, smiles_col).strip()
            if not smiles:
                rejected.append({**base_attempt, "reason": "missing_smiles"})
                continue

            plqy = _parse_plqy(_row_value(row, plqy_col)) if plqy_col else None
            lambda_em_nm = _parse_lambda_em_nm(_row_value(row, lambda_col)) if lambda_col else None
            confidence = _row_confidence(row, confidence_col, has_plqy=plqy is not None, has_lambda=lambda_em_nm is not None)
            if confidence < confidence_threshold:
                rejected.append({**base_attempt, "smiles": smiles, "confidence": confidence, "reason": "low_confidence"})
                continue
            if plqy is None or lambda_em_nm is None:
                rejected.append(
                    {
                        **base_attempt,
                        "smiles": smiles,
                        "confidence": confidence,
                        "reason": "missing_required_properties",
                    }
                )
                continue

            evidence_ref = f"{parsed_document.paper_id}:{table.table_id}:{row_id}"
            provenance = {
                "paper_id": parsed_document.paper_id,
                "source_path": parsed_document.source_path,
                "parser_backend": parsed_document.parser_backend,
                "page": table.page,
                "table_id": table.table_id,
                "row_index": row_index,
                "row_id": row_id,
                "bbox": table.source_bbox,
            }
            records.append(
                StructuredScientificRecord(
                    record_id=f"sci_{len(records) + 1:06d}",
                    smiles=smiles,
                    plqy=plqy,
                    lambda_em_nm=lambda_em_nm,
                    paper_id=parsed_document.paper_id,
                    source_id=parsed_document.paper_id,
                    page=table.page,
                    table_id=table.table_id,
                    row_index=row_index,
                    row_id=row_id,
                    evidence_ref=evidence_ref,
                    confidence=confidence,
                    confidence_factors={
                        "has_smiles": True,
                        "has_plqy": plqy is not None,
                        "has_lambda_em_nm": lambda_em_nm is not None,
                        "source": "phase3_scientific_extractor",
                    },
                    raw_values=raw_values,
                    provenance=provenance,
                )
            )

    duplicate_smiles_count = sum(1 for group in _records_by_smiles(records).values() if len(group) > 1)
    oled_candidates = _extract_oled_candidates(parsed_document)
    oled_text_evidence_candidates = extract_oled_text_evidence_candidates_from_document(parsed_document)
    oled_mapping = map_oled_mineru_candidates_to_schema_candidates(oled_candidates)
    oled_compilation = compile_oled_schema_candidates_to_layered_records(
        oled_mapping.schema_candidates,
        require_measurement_condition=False,
        require_device_context_for_measurement=False,
    )
    conflict_report = _build_conflict_report(
        records,
        run_id=run_id,
        generated_at=generated,
        plqy_tolerance=plqy_conflict_tolerance,
        lambda_tolerance=lambda_conflict_tolerance_nm,
    )
    report = ExtractionReport(
        run_id=run_id,
        paper_id=parsed_document.paper_id,
        input_table_count=len(parsed_document.tables),
        selected_table_count=len(selected_tables),
        oled_candidate_count=len(oled_candidates),
        oled_text_evidence_candidate_count=len(oled_text_evidence_candidates),
        oled_schema_candidate_count=len(oled_mapping.schema_candidates),
        oled_compiled_record_count=len(oled_compilation.compiled_records),
        input_row_count=input_row_count,
        extracted_record_count=len(records),
        rejected_record_count=len(rejected),
        duplicate_smiles_count=duplicate_smiles_count,
        generated_at=generated,
        notes=[
            "deterministic_table_extraction",
            "deterministic_oled_evidence_schema_candidates",
            "deterministic_oled_text_evidence_candidates",
            "no_llm_calls",
            "no_external_services",
        ],
    )
    return ScientificExtractionResult(
        records=records,
        rejected_records=rejected,
        extraction_report=report,
        conflict_report=conflict_report,
        oled_candidates=oled_candidates,
        oled_text_evidence_candidates=oled_text_evidence_candidates,
        oled_schema_candidates=oled_mapping.schema_candidates,
        oled_compiled_records=oled_compilation.compiled_records,
    )


def _extract_oled_candidates(parsed_document: ParsedDocument) -> list[OledMineruCandidate]:
    table_blocks = [_table_to_mineru_block(table) for table in parsed_document.tables]
    element_blocks = [_element_to_mineru_block(element) for element in parsed_document.elements]
    return extract_oled_mineru_candidates_from_document(
        [*table_blocks, *element_blocks],
        paper_id=parsed_document.paper_id,
        source_path=parsed_document.source_path,
    )


def _table_to_mineru_block(table: ParsedTable) -> dict[str, Any]:
    return {
        "type": "table",
        "table_caption": table.caption,
        "table_body": table.markdown or _markdown_from_table(table),
        "table_footnote": "\n".join(table.footnotes),
        "bbox": _bbox_list(table.source_bbox),
        "page_idx": table.page,
    }


def _element_to_mineru_block(element: Any) -> dict[str, Any]:
    element_type = _mineru_element_type(element)
    text = str(getattr(element, "markdown", "") or getattr(element, "text", "") or "")
    block: dict[str, Any] = {
        "type": element_type,
        "page_idx": getattr(element, "page", None),
        "bbox": getattr(element, "bbox", None),
    }
    if element_type == "image":
        block["image_caption"] = text
        image_path = _element_image_path(element)
        if image_path:
            block["img_path"] = image_path
    elif element_type == "chart":
        block["chart_caption"] = text
        image_path = _element_image_path(element)
        if image_path:
            block["img_path"] = image_path
    else:
        block["text"] = text
    return block


def _mineru_element_type(element: Any) -> str:
    raw_type = str(getattr(element, "metadata", {}).get("raw_type") or getattr(element, "type", "") or "").lower()
    if "chart" in raw_type:
        return "chart"
    if "image" in raw_type or "figure" in raw_type:
        return "image"
    if str(getattr(element, "metadata", {}).get("text_level") or "").strip():
        return "title"
    return "text"


def _element_image_path(element: Any) -> str:
    metadata = getattr(element, "metadata", {})
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("image_path") or "").strip()


def _markdown_from_table(table: ParsedTable) -> str:
    headers = list(table.headers)
    if not headers and table.rows:
        headers = list(table.rows[0].keys())
    if not headers:
        return ""
    lines = [
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.rows:
        lines.append("| " + " | ".join(_markdown_cell(str(row.get(header) or "")) for header in headers) + " |")
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return str(value or "").replace("\n", " ").replace("|", "/").strip()


def _bbox_list(source_bbox: dict[str, float] | None) -> list[float] | None:
    if not isinstance(source_bbox, dict):
        return None
    keys = ("x0", "y0", "x1", "y1")
    if all(key in source_bbox for key in keys):
        return [float(source_bbox[key]) for key in keys]
    alternate_keys = ("x0", "top", "x1", "bottom")
    if all(key in source_bbox for key in alternate_keys):
        return [float(source_bbox[key]) for key in alternate_keys]
    return None


def _table_columns(table: ParsedTable) -> dict[str, Any]:
    return {
        "table": table,
        "smiles_col": _find_column(table.headers, exact_aliases=_SMILES_ALIASES),
        "plqy_col": _find_column(table.headers, contains_tokens=_PLQY_TOKENS),
        "lambda_col": _find_column(table.headers, contains_tokens=_LAMBDA_TOKENS),
        "confidence_col": _find_column(table.headers, exact_aliases={"confidence", "score", "extractconfidence"}),
    }


def _find_column(
    headers: list[str],
    *,
    exact_aliases: set[str] | None = None,
    contains_tokens: set[str] | None = None,
) -> str:
    normalized = [(header, _normalize_header(header)) for header in headers]
    aliases = exact_aliases or set()
    for header, token in normalized:
        if token in aliases:
            return header
    tokens = contains_tokens or set()
    for header, token in normalized:
        if any(item in token for item in tokens):
            return header
    return ""


def _row_value(row: dict[str, str], column: str) -> str:
    if not column:
        return ""
    if column in row:
        return str(row.get(column) or "")
    target = _normalize_header(column)
    for key, value in row.items():
        if _normalize_header(str(key)) == target:
            return str(value or "")
    return ""


def _normalize_header(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _parse_number(raw: str) -> float | None:
    text = str(raw or "").strip()
    if not text or text.lower() in {"na", "n/a", "nan", "none", "null", "-"}:
        return None
    match = _NUMBER_PATTERN.search(text.replace(",", ""))
    if match is None:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _parse_plqy(raw: str) -> float | None:
    value = _parse_number(raw)
    if value is None:
        return None
    text = str(raw or "")
    if "%" in text or value > 1.0:
        value = value / 100.0
    return round(value, 12)


def _parse_lambda_em_nm(raw: str) -> float | None:
    value = _parse_number(raw)
    if value is None:
        return None
    text = str(raw or "").lower()
    if "um" in text or "micrometer" in text:
        value *= 1000.0
    return round(value, 12)


def _row_confidence(row: dict[str, str], confidence_col: str, *, has_plqy: bool, has_lambda: bool) -> float:
    explicit = _parse_number(_row_value(row, confidence_col)) if confidence_col else None
    if explicit is not None:
        return round(max(0.0, min(1.0, explicit / 100.0 if explicit > 1.0 else explicit)), 6)
    score = 0.5
    if has_plqy:
        score += 0.2
    if has_lambda:
        score += 0.2
    return round(min(score, 0.95), 6)


def _records_by_smiles(records: list[StructuredScientificRecord]) -> dict[str, list[StructuredScientificRecord]]:
    grouped: dict[str, list[StructuredScientificRecord]] = defaultdict(list)
    for record in records:
        grouped[record.smiles].append(record)
    return dict(grouped)


def _build_conflict_report(
    records: list[StructuredScientificRecord],
    *,
    run_id: str,
    generated_at: str,
    plqy_tolerance: float,
    lambda_tolerance: float,
) -> ConflictReport:
    conflicts: list[ConflictGroup] = []
    for smiles, group in sorted(_records_by_smiles(records).items()):
        if len(group) <= 1:
            continue
        for property_id, tolerance in [("plqy", plqy_tolerance), ("lambda_em_nm", lambda_tolerance)]:
            values = [(record, getattr(record, property_id)) for record in group if getattr(record, property_id) is not None]
            if len(values) <= 1:
                continue
            numeric = [float(value) for _, value in values]
            min_value = min(numeric)
            max_value = max(numeric)
            if max_value - min_value <= tolerance:
                continue
            conflict_id = f"conflict_{len(conflicts) + 1:06d}"
            conflicts.append(
                ConflictGroup(
                    conflict_id=conflict_id,
                    smiles=smiles,
                    property_id=property_id,
                    min_value=round(min_value, 12),
                    max_value=round(max_value, 12),
                    tolerance=tolerance,
                    observations=[
                        {
                            "record_id": record.record_id,
                            "value": float(value),
                            "paper_id": record.paper_id,
                            "page": record.page,
                            "table_id": record.table_id,
                            "row_id": record.row_id,
                        }
                        for record, value in values
                    ],
                    status="needs_review",
                )
            )
    unique_smiles = {record.smiles for record in records}
    conflicted_smiles = {conflict.smiles for conflict in conflicts}
    return ConflictReport(
        run_id=run_id,
        input_record_count=len(records),
        merged_record_count=len(unique_smiles),
        conflict_count=len(conflicts),
        non_conflicting_record_count=len(unique_smiles - conflicted_smiles),
        conflicts=conflicts,
        generated_at=generated_at,
        notes=["duplicate_smiles_conflict_detection", "deterministic_tolerance_rules"],
    )
