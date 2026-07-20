"""Adapter interfaces for Phase 1 workflows and legacy script bridges."""

from typing import Any

from ai4s_agent._utils import strict_bool
from ai4s_agent.adapters.phase1 import (
    check_trainability_service,
    draft_cleaning_rules_adapter,
    execute_cleaning_adapter,
    filter_rank_adapter,
    generate_candidates_stub_adapter,
    inspect_dataset_service,
    iterative_generate_predict_filter_adapter,
    legacy_full_flow_adapter,
    parse_task_adapter,
    predict_candidates_baseline_adapter,
    predict_candidates_domain_model_adapter,
    predict_candidates_unimol_legacy_adapter,
    recommend_backend_service,
    render_report_adapter,
    run_baseline_service,
    train_model_baseline_adapter,
    train_model_unimol_legacy_adapter,
)
from ai4s_agent.adapters.phase3 import (
    acquire_literature_sources_adapter,
    build_dense_index_adapter,
    build_multi_index_adapter,
    check_public_dataset_leakage_adapter,
    confirm_extracted_dataset_adapter,
    evaluate_extraction_benchmark_adapter,
    extract_records_adapter,
    index_corpus_adapter,
    literature_to_dataset_workflow_adapter,
    merge_extracted_records_adapter,
    normalize_extracted_units_adapter,
    parse_document_grobid_adapter as _parse_document_grobid_adapter,
    parse_document_mineru_adapter as _parse_document_mineru_adapter,
    parse_document_pdfplumber_adapter,
    parse_document_pymupdf_adapter,
    parse_pdf_folder_mineru_adapter as _parse_pdf_folder_mineru_adapter,
    prepare_literature_corpus_sources_adapter,
    retrieve_evidence_adapter,
    track_citation_provenance_adapter,
)
from ai4s_agent.adapters.oled_demo import execute_oled_local_demo_adapter
from ai4s_agent.adapters.oled_registry_screening import (
    execute_oled_registry_candidate_screening_adapter,
)
from ai4s_agent.adapters.runtime import AdapterRuntimeError


def _strict_execute_error(payload: dict[str, Any], *, adapter: str) -> dict[str, Any] | None:
    """Validate JSON boolean execution flags at the package boundary."""
    try:
        strict_bool(payload.get("execute", False), key="execute")
    except ValueError as exc:
        return {
            "status": "failed",
            "adapter": adapter,
            "error": {"code": "invalid_execute_flag", "message": str(exc)},
        }
    return None


def parse_document_mineru_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    error = _strict_execute_error(payload, adapter="parse_document_mineru")
    return error or _parse_document_mineru_adapter(payload)


def parse_pdf_folder_mineru_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    error = _strict_execute_error(payload, adapter="parse_pdf_folder_mineru")
    return error or _parse_pdf_folder_mineru_adapter(payload)


def parse_document_grobid_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    error = _strict_execute_error(payload, adapter="parse_document_grobid")
    return error or _parse_document_grobid_adapter(payload)


__all__ = [
    "AdapterRuntimeError",
    "legacy_full_flow_adapter",
    "parse_task_adapter",
    "inspect_dataset_service",
    "draft_cleaning_rules_adapter",
    "execute_cleaning_adapter",
    "check_trainability_service",
    "run_baseline_service",
    "recommend_backend_service",
    "train_model_baseline_adapter",
    "train_model_unimol_legacy_adapter",
    "generate_candidates_stub_adapter",
    "iterative_generate_predict_filter_adapter",
    "predict_candidates_baseline_adapter",
    "predict_candidates_domain_model_adapter",
    "predict_candidates_unimol_legacy_adapter",
    "filter_rank_adapter",
    "render_report_adapter",
    "acquire_literature_sources_adapter",
    "check_public_dataset_leakage_adapter",
    "evaluate_extraction_benchmark_adapter",
    "build_dense_index_adapter",
    "build_multi_index_adapter",
    "parse_document_grobid_adapter",
    "parse_document_mineru_adapter",
    "parse_document_pdfplumber_adapter",
    "parse_document_pymupdf_adapter",
    "parse_pdf_folder_mineru_adapter",
    "prepare_literature_corpus_sources_adapter",
    "index_corpus_adapter",
    "retrieve_evidence_adapter",
    "extract_records_adapter",
    "track_citation_provenance_adapter",
    "merge_extracted_records_adapter",
    "confirm_extracted_dataset_adapter",
    "literature_to_dataset_workflow_adapter",
    "normalize_extracted_units_adapter",
    "execute_oled_local_demo_adapter",
    "execute_oled_registry_candidate_screening_adapter",
]
