from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.phase3_scientific_extractor import (
    ExtractionReport,
    StructuredScientificRecord,
    extract_scientific_records,
)
from ai4s_agent.domains.oled_mineru_candidates import OledMineruCandidate
from ai4s_agent.domains.oled_mineru_semantic_mapping import OledSchemaCandidate
from ai4s_agent.domains.oled_schema_candidate_compiler import OledCompiledLayeredRecordCandidate
from ai4s_agent.schemas import ParsedDocument


class CorpusExtractionReport(BaseModel):
    run_id: str
    document_count: int
    extracted_record_count: int
    rejected_record_count: int
    oled_candidate_count: int = 0
    oled_schema_candidate_count: int = 0
    oled_compiled_record_count: int = 0
    paper_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    record_counts_by_paper: dict[str, int] = Field(default_factory=dict)
    extraction_rejection_counts_by_paper: dict[str, int] = Field(default_factory=dict)
    oled_candidate_counts_by_paper: dict[str, int] = Field(default_factory=dict)
    oled_schema_candidate_counts_by_paper: dict[str, int] = Field(default_factory=dict)
    oled_compiled_record_counts_by_paper: dict[str, int] = Field(default_factory=dict)
    generated_at: str
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class CorpusExtractionResult:
    records: list[StructuredScientificRecord]
    rejected_records: list[dict[str, Any]]
    oled_candidates: list[OledMineruCandidate]
    oled_schema_candidates: list[OledSchemaCandidate]
    oled_compiled_records: list[OledCompiledLayeredRecordCandidate]
    report: CorpusExtractionReport
    per_document_reports: list[ExtractionReport] = field(default_factory=list)
    corpus_records_json: str = ""
    oled_candidates_json: str = ""
    oled_schema_candidates_json: str = ""
    oled_compiled_records_json: str = ""
    per_document_extraction_reports_json: str = ""
    corpus_extraction_manifest_json: str = ""


def extract_corpus_records(
    *,
    parsed_documents: Iterable[str | Path | ParsedDocument],
    output_dir: str | Path,
    run_id: str,
    generated_at: str | None = None,
) -> CorpusExtractionResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    records: list[StructuredScientificRecord] = []
    rejected_records: list[dict[str, Any]] = []
    oled_candidates: list[OledMineruCandidate] = []
    oled_schema_candidates: list[OledSchemaCandidate] = []
    oled_compiled_records: list[OledCompiledLayeredRecordCandidate] = []
    per_document_reports: list[ExtractionReport] = []
    document_refs: list[dict[str, Any]] = []
    record_counts_by_paper: dict[str, int] = {}
    rejection_counts_by_paper: dict[str, int] = {}
    oled_candidate_counts_by_paper: dict[str, int] = {}
    oled_schema_candidate_counts_by_paper: dict[str, int] = {}
    oled_compiled_record_counts_by_paper: dict[str, int] = {}
    paper_ids: list[str] = []
    source_document_ids: list[str] = []

    for document_index, item in enumerate(parsed_documents, start=1):
        parsed_document, parsed_document_path = _load_parsed_document(item)
        paper_id = parsed_document.paper_id
        source_document_id = str(parsed_document.metadata.get("source_document_id") or paper_id)
        parser_provider = str(parsed_document.metadata.get("parser_provider") or "")
        parser_backend = parsed_document.parser_backend
        paper_ids.append(paper_id)
        source_document_ids.append(source_document_id)

        extraction = extract_scientific_records(
            parsed_document,
            run_id=f"{run_id}-{document_index:03d}",
            generated_at=generated,
        )
        per_document_reports.append(extraction.extraction_report)
        record_counts_by_paper[paper_id] = len(extraction.records)
        oled_candidate_counts_by_paper[paper_id] = len(extraction.oled_candidates)
        oled_schema_candidate_counts_by_paper[paper_id] = len(extraction.oled_schema_candidates)
        oled_compiled_record_counts_by_paper[paper_id] = len(extraction.oled_compiled_records)
        if extraction.rejected_records:
            rejection_counts_by_paper[paper_id] = len(extraction.rejected_records)

        for record_index, record in enumerate(extraction.records, start=1):
            provenance = {
                **record.provenance,
                "paper_id": paper_id,
                "source_document_id": source_document_id,
                "parsed_document_path": str(parsed_document_path),
                "parser_provider": parser_provider,
                "parser_backend": parser_backend,
            }
            records.append(
                record.model_copy(
                    update={
                        "record_id": f"corpus_{len(records) + 1:06d}",
                        "paper_id": paper_id,
                        "source_id": source_document_id,
                        "evidence_ref": f"{paper_id}:{record.table_id}:{record.row_id}",
                        "provenance": provenance,
                    }
                )
            )

        for rejected in extraction.rejected_records:
            raw_values = rejected.get("raw_values") if isinstance(rejected.get("raw_values"), dict) else {}
            rejected_records.append(
                {
                    **rejected,
                    "record_id": f"corpus_rejected_{len(rejected_records) + 1:06d}",
                    "paper_id": paper_id,
                    "source_id": source_document_id,
                    "provenance": {
                        "paper_id": paper_id,
                        "source_document_id": source_document_id,
                        "parsed_document_path": str(parsed_document_path),
                        "parser_provider": parser_provider,
                        "parser_backend": parser_backend,
                        "page": rejected.get("page"),
                        "table_id": rejected.get("table_id"),
                        "row_id": rejected.get("row_id"),
                    },
                    "raw_values": raw_values,
                }
            )

        oled_candidates.extend(extraction.oled_candidates)
        oled_schema_candidates.extend(extraction.oled_schema_candidates)
        oled_compiled_records.extend(extraction.oled_compiled_records)

        document_refs.append(
            {
                "paper_id": paper_id,
                "source_document_id": source_document_id,
                "parsed_document_path": str(parsed_document_path),
                "parser_provider": parser_provider,
                "parser_backend": parser_backend,
                "extracted_record_count": len(extraction.records),
                "rejected_record_count": len(extraction.rejected_records),
                "oled_candidate_count": len(extraction.oled_candidates),
                "oled_schema_candidate_count": len(extraction.oled_schema_candidates),
                "oled_compiled_record_count": len(extraction.oled_compiled_records),
            }
        )

    report = CorpusExtractionReport(
        run_id=run_id,
        document_count=len(document_refs),
        extracted_record_count=len(records),
        rejected_record_count=len(rejected_records),
        oled_candidate_count=len(oled_candidates),
        oled_schema_candidate_count=len(oled_schema_candidates),
        oled_compiled_record_count=len(oled_compiled_records),
        paper_ids=paper_ids,
        source_document_ids=source_document_ids,
        record_counts_by_paper=record_counts_by_paper,
        extraction_rejection_counts_by_paper=rejection_counts_by_paper,
        oled_candidate_counts_by_paper=oled_candidate_counts_by_paper,
        oled_schema_candidate_counts_by_paper=oled_schema_candidate_counts_by_paper,
        oled_compiled_record_counts_by_paper=oled_compiled_record_counts_by_paper,
        generated_at=generated,
        notes=[
            "deterministic_multi_document_extraction",
            "reuses_phase3_scientific_extractor",
            "includes_oled_table_schema_candidates",
            "no_llm_calls",
            "no_external_services",
        ],
    )

    corpus_records_json = output_path / "corpus_records.json"
    oled_candidates_json = output_path / "oled_candidates.json"
    oled_schema_candidates_json = output_path / "oled_schema_candidates.json"
    oled_compiled_records_json = output_path / "oled_compiled_records.json"
    per_document_reports_json = output_path / "per_document_extraction_reports.json"
    manifest_json = output_path / "corpus_extraction_manifest.json"
    write_json(
        corpus_records_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "records": [record.model_dump(mode="json") for record in records],
        },
    )
    write_json(
        oled_candidates_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "candidates": [candidate.model_dump(mode="json") for candidate in oled_candidates],
        },
    )
    write_json(
        oled_schema_candidates_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "schema_candidates": [candidate.model_dump(mode="json") for candidate in oled_schema_candidates],
        },
    )
    write_json(
        oled_compiled_records_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "compiled_records": [record.model_dump(mode="json") for record in oled_compiled_records],
        },
    )
    write_json(
        per_document_reports_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "reports": [report_item.model_dump(mode="json") for report_item in per_document_reports],
            "rejected_records": rejected_records,
        },
    )
    write_json(
        manifest_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "report": report.model_dump(mode="json"),
            "documents": document_refs,
            "artifacts": {
                "corpus_records_json": str(corpus_records_json),
                "oled_candidates_json": str(oled_candidates_json),
                "oled_schema_candidates_json": str(oled_schema_candidates_json),
                "oled_compiled_records_json": str(oled_compiled_records_json),
                "per_document_extraction_reports_json": str(per_document_reports_json),
            },
        },
    )
    return CorpusExtractionResult(
        records=records,
        rejected_records=rejected_records,
        oled_candidates=oled_candidates,
        oled_schema_candidates=oled_schema_candidates,
        oled_compiled_records=oled_compiled_records,
        report=report,
        per_document_reports=per_document_reports,
        corpus_records_json=str(corpus_records_json),
        oled_candidates_json=str(oled_candidates_json),
        oled_schema_candidates_json=str(oled_schema_candidates_json),
        oled_compiled_records_json=str(oled_compiled_records_json),
        per_document_extraction_reports_json=str(per_document_reports_json),
        corpus_extraction_manifest_json=str(manifest_json),
    )


def _load_parsed_document(item: str | Path | ParsedDocument) -> tuple[ParsedDocument, Path | str]:
    if isinstance(item, ParsedDocument):
        return item, ""
    path = Path(item).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ParsedDocument.model_validate(payload), path
