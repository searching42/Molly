from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_PHASE3_OUTPUT_DIRS: dict[str, str] = {
    "prepare_literature_corpus_sources": "20_literature_sources",
    "acquire_literature_sources": "21_literature_acquisition",
    "parse_document": "22_parse_document",
    "parse_document_pdfplumber": "22_parse_document",
    "parse_document_pymupdf": "22_parse_document",
    "parse_document_grobid": "22_parse_document",
    "index_corpus": "23_corpus_index",
    "build_multi_index": "24_multi_index",
    "build_dense_index": "24_dense_index",
    "retrieve_evidence": "25_retrieval",
    "extract_records": "26_extraction",
    "normalize_extracted_units": "27_normalization",
    "track_citation_provenance": "28_provenance",
    "merge_extracted_records": "29_merge",
    "evaluate_extraction_benchmark": "30_benchmark",
    "confirm_extracted_dataset": "31_confirmation",
    "literature_to_dataset_workflow": "32_literature_workflow",
    "check_public_dataset_leakage": "33_leakage",
}

_OUTPUT_ARTIFACTS: dict[str, dict[str, str]] = {
    "prepare_literature_corpus_sources": {
        "corpus_source_manifest_json": "corpus_source_manifest",
        "corpus_source_manifest_md": "corpus_source_manifest_md",
    },
    "acquire_literature_sources": {
        "acquisition_manifest_json": "acquisition_manifest",
        "acquisition_plan_md": "acquisition_plan",
        "acquired_pdf_dir": "pdf_corpus",
        "acquired_dataset_dir": "structured_datasets",
    },
    "parse_document": {
        "parsed_document_json": "parsed_document",
        "parsed_document_markdown": "parsed_document_markdown",
        "parser_audit_json": "parser_audit",
    },
    "parse_document_pdfplumber": {
        "parsed_document_json": "parsed_document",
        "parsed_document_markdown": "parsed_document_markdown",
        "parser_audit_json": "parser_audit",
    },
    "parse_document_pymupdf": {
        "parsed_document_json": "parsed_document",
        "parsed_document_markdown": "parsed_document_markdown",
        "parser_audit_json": "parser_audit",
    },
    "parse_document_grobid": {
        "parsed_document_json": "parsed_document",
        "parsed_document_markdown": "parsed_document_markdown",
        "parser_audit_json": "parser_audit",
    },
    "index_corpus": {
        "corpus_index_json": "corpus_index",
        "chunks_jsonl": "evidence_chunks",
        "index_report_json": "corpus_index_report",
    },
    "build_multi_index": {
        "multi_index_json": "multi_index",
        "multi_index_summary_md": "multi_index_summary",
    },
    "build_dense_index": {
        "dense_index_json": "dense_index",
        "dense_index_summary_md": "dense_index_summary",
    },
    "retrieve_evidence": {
        "evidence_hits_json": "evidence_hits",
        "retrieval_log_jsonl": "retrieval_log",
    },
    "extract_records": {
        "extracted_records_jsonl": "extracted_records",
        "rejected_records_jsonl": "rejected_records",
        "extraction_confidence_report_json": "extraction_confidence_report",
        "candidate_training_dataset_csv": "candidate_training_dataset",
    },
    "normalize_extracted_units": {
        "normalized_extracted_records_jsonl": "normalized_extracted_records",
        "normalized_candidate_training_dataset_csv": "candidate_training_dataset",
        "unit_normalization_report_json": "unit_normalization_report",
    },
    "track_citation_provenance": {
        "citation_provenance_report_json": "citation_provenance_report",
        "audit_summary_md": "audit_summary",
    },
    "merge_extracted_records": {
        "merged_records_jsonl": "merged_records",
        "conflict_report_json": "conflict_report",
        "candidate_training_dataset_csv": "candidate_training_dataset",
    },
    "evaluate_extraction_benchmark": {
        "extraction_benchmark_report_json": "extraction_benchmark_report",
    },
    "confirm_extracted_dataset": {
        "confirmed_training_dataset_csv": "confirmed_training_dataset",
        "confirmation_record_json": "extraction_confirmation_record",
    },
    "literature_to_dataset_workflow": {
        "corpus_manifest_json": "corpus_manifest",
        "corpus_index_json": "corpus_index",
        "evidence_hits_json": "evidence_hits",
        "extracted_records_jsonl": "extracted_records",
        "unit_normalization_report_json": "unit_normalization_report",
        "citation_provenance_report_json": "citation_provenance_report",
        "conflict_report_json": "conflict_report",
        "extraction_benchmark_report_json": "extraction_benchmark_report",
        "candidate_training_dataset_csv": "candidate_training_dataset",
        "workflow_report_json": "workflow_report",
    },
    "check_public_dataset_leakage": {
        "benchmark_contamination_report_json": "benchmark_contamination_report",
    },
}


def install_phase3_executor_support() -> None:
    """Install Phase 3 payload and artifact support for RunPlanExecutor."""

    from ai4s_agent.executor import RunPlanExecutor

    original_payload_for = RunPlanExecutor._payload_for
    if getattr(original_payload_for, "_phase3_payload_support", False):
        return
    original_collect_artifacts = RunPlanExecutor._collect_artifacts

    def payload_for_with_phase3(
        self: Any,
        task_id: str,
        *,
        run_id: str,
        run_dir: Path,
        artifact_paths: dict[str, str],
        actor: str = "",
        approved_gates: set[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if task_id in _PHASE3_OUTPUT_DIRS:
            return _phase3_payload_for(
                self,
                task_id=task_id,
                run_id=run_id,
                run_dir=run_dir,
                artifact_paths=artifact_paths,
                actor=actor,
                approved_gates=approved_gates or set(),
                options=options or {},
            )
        return original_payload_for(
            self,
            task_id,
            run_id=run_id,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            actor=actor,
            approved_gates=approved_gates,
            options=options,
        )

    def collect_artifacts_with_phase3(
        self: Any,
        *,
        project_id: str,
        run_id: str,
        run_dir: Path,
        task_id: str,
        result: dict[str, Any],
        result_path: Path,
        artifact_paths: dict[str, str],
    ) -> None:
        if task_id in _PHASE3_OUTPUT_DIRS:
            _collect_phase3_artifacts(
                self,
                project_id=project_id,
                run_id=run_id,
                run_dir=run_dir,
                task_id=task_id,
                result=result,
                result_path=result_path,
                artifact_paths=artifact_paths,
            )
            return
        return original_collect_artifacts(
            self,
            project_id=project_id,
            run_id=run_id,
            run_dir=run_dir,
            task_id=task_id,
            result=result,
            result_path=result_path,
            artifact_paths=artifact_paths,
        )

    payload_for_with_phase3._phase3_payload_support = True  # type: ignore[attr-defined]
    RunPlanExecutor._payload_for = payload_for_with_phase3  # type: ignore[method-assign]
    RunPlanExecutor._collect_artifacts = collect_artifacts_with_phase3  # type: ignore[method-assign]


def _phase3_payload_for(
    executor: Any,
    *,
    task_id: str,
    run_id: str,
    run_dir: Path,
    artifact_paths: dict[str, str],
    actor: str,
    approved_gates: set[str],
    options: dict[str, Any],
) -> dict[str, Any]:
    task_options = executor._payload_options(options)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "output_dir": str(run_dir / _PHASE3_OUTPUT_DIRS[task_id]),
    }

    if task_id == "prepare_literature_corpus_sources":
        payload.update(task_options)
        return payload
    if task_id == "acquire_literature_sources":
        payload["corpus_source_manifest_json"] = _require_artifact(artifact_paths, "corpus_source_manifest")
        payload.update(task_options)
        return payload
    if task_id in {"parse_document", "parse_document_pdfplumber", "parse_document_pymupdf", "parse_document_grobid"}:
        payload["input_pdf"] = _first_pdf(_require_artifact(artifact_paths, "pdf_corpus"))
        payload.setdefault("execute", False)
        payload.update(task_options)
        return payload
    if task_id == "index_corpus":
        _add_optional_artifact(payload, artifact_paths, "parsed_document", "parsed_document_json")
        _add_optional_artifact(payload, artifact_paths, "corpus_manifest", "corpus_manifest_json")
        payload.update(task_options)
        return payload
    if task_id == "build_multi_index":
        _add_optional_artifact(payload, artifact_paths, "evidence_chunks", "chunks_jsonl")
        _add_optional_artifact(payload, artifact_paths, "corpus_index", "corpus_index_json")
        payload.update(task_options)
        return payload
    if task_id == "build_dense_index":
        _add_optional_artifact(payload, artifact_paths, "evidence_chunks", "chunks_jsonl")
        _add_optional_artifact(payload, artifact_paths, "corpus_index", "corpus_index_json")
        payload.update(task_options)
        return payload
    if task_id == "retrieve_evidence":
        payload["query"] = str(task_options.pop("query", "SMILES PLQY OLED property table") or "SMILES PLQY OLED property table")
        payload["topk"] = int(task_options.pop("topk", 20) or 20)
        payload["corpus_index_json"] = _require_artifact(artifact_paths, "corpus_index")
        _add_optional_artifact(payload, artifact_paths, "multi_index", "multi_index_json")
        _add_optional_artifact(payload, artifact_paths, "dense_index", "dense_index_json")
        payload.update(task_options)
        return payload
    if task_id == "extract_records":
        payload["evidence_hits_json"] = _require_artifact(artifact_paths, "evidence_hits")
        _add_optional_artifact(payload, artifact_paths, "evidence_chunks", "chunks_jsonl")
        _add_chunks_from_corpus_index(payload)
        payload.update(task_options)
        return payload
    if task_id == "normalize_extracted_units":
        payload["extracted_records_jsonl"] = _require_artifact(artifact_paths, "extracted_records")
        payload.update(task_options)
        return payload
    if task_id == "track_citation_provenance":
        _add_optional_artifact(payload, artifact_paths, "parsed_document", "parsed_document_json")
        payload["evidence_hits_json"] = _require_artifact(artifact_paths, "evidence_hits")
        payload["extracted_records_jsonl"] = _require_artifact(artifact_paths, "extracted_records")
        payload.update(task_options)
        return payload
    if task_id == "merge_extracted_records":
        records_path = str(artifact_paths.get("normalized_extracted_records") or artifact_paths.get("extracted_records") or "").strip()
        if not records_path:
            raise ValueError("missing artifact path: normalized_extracted_records or extracted_records")
        payload["extracted_records_jsonl_list"] = [records_path]
        _add_optional_artifact(payload, artifact_paths, "citation_provenance_report", "citation_provenance_report_json")
        payload.update(task_options)
        return payload
    if task_id == "evaluate_extraction_benchmark":
        payload["evidence_hits_json"] = _require_artifact(artifact_paths, "evidence_hits")
        payload["extracted_records_jsonl"] = _require_artifact(artifact_paths, "extracted_records")
        _add_optional_artifact(payload, artifact_paths, "conflict_report", "conflict_report_json")
        payload.update(task_options)
        return payload
    if task_id == "confirm_extracted_dataset":
        payload["candidate_training_dataset_csv"] = _require_artifact(artifact_paths, "candidate_training_dataset")
        payload["conflict_report_json"] = _require_artifact(artifact_paths, "conflict_report")
        payload["citation_provenance_report_json"] = _require_artifact(artifact_paths, "citation_provenance_report")
        payload["actor"] = actor
        payload["confirmed"] = "gate_2_data_mining" in approved_gates
        payload.update(task_options)
        return payload
    if task_id == "literature_to_dataset_workflow":
        pdf_corpus = _require_artifact(artifact_paths, "pdf_corpus")
        path = Path(pdf_corpus).expanduser()
        payload["input_pdf_dir"] = str(path if path.is_dir() else path.parent)
        payload.update(task_options)
        return payload
    if task_id == "check_public_dataset_leakage":
        payload["candidate_training_dataset_csv"] = _require_artifact(artifact_paths, "confirmed_training_dataset")
        payload.update(task_options)
        return payload
    payload.update(task_options)
    return payload


def _collect_phase3_artifacts(
    executor: Any,
    *,
    project_id: str,
    run_id: str,
    run_dir: Path,
    task_id: str,
    result: dict[str, Any],
    result_path: Path,
    artifact_paths: dict[str, str],
) -> None:
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    for output_key, artifact_id in _OUTPUT_ARTIFACTS.get(task_id, {}).items():
        raw = str(outputs.get(output_key) or "").strip()
        if raw:
            _register_path(executor, project_id, run_id, run_dir, artifact_id, raw, artifact_paths)
    if task_id in {"parse_document", "parse_document_pdfplumber", "parse_document_pymupdf", "parse_document_grobid"}:
        parsed_json = str(outputs.get("parsed_document_json") or "").strip()
        if parsed_json:
            _register_path(executor, project_id, run_id, run_dir, "parsed_tables", parsed_json, artifact_paths)
    executor._register(project_id, run_id, f"{task_id}_result", executor._relative(run_dir, result_path))


def _register_path(
    executor: Any,
    project_id: str,
    run_id: str,
    run_dir: Path,
    artifact_id: str,
    raw_path: str,
    artifact_paths: dict[str, str],
) -> None:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (run_dir / path).resolve()
    rel = executor._relative(run_dir, path)
    executor._register(project_id, run_id, artifact_id, rel)
    artifact_paths[artifact_id] = str(path)


def _require_artifact(artifact_paths: dict[str, str], artifact_id: str) -> str:
    value = str(artifact_paths.get(artifact_id) or "").strip()
    if not value:
        raise ValueError(f"missing artifact path: {artifact_id}")
    return value


def _add_optional_artifact(payload: dict[str, Any], artifact_paths: dict[str, str], artifact_id: str, payload_key: str) -> None:
    value = str(artifact_paths.get(artifact_id) or "").strip()
    if value:
        payload[payload_key] = value


def _first_pdf(path_raw: str) -> str:
    path = Path(path_raw).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        for child in sorted(path.glob("*.pdf")):
            if child.is_file():
                return str(child)
    raise FileNotFoundError(f"pdf_corpus does not contain a PDF: {path}")


def _add_chunks_from_corpus_index(payload: dict[str, Any]) -> None:
    if payload.get("chunks_jsonl"):
        return
    index_raw = str(payload.get("corpus_index_json") or "").strip()
    if not index_raw:
        return
    try:
        index = json.loads(Path(index_raw).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    chunks = str(index.get("chunks_jsonl") or "").strip() if isinstance(index, dict) else ""
    if chunks:
        payload["chunks_jsonl"] = chunks
