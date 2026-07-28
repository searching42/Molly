from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.corpus_conflict_auditor import audit_corpus_conflicts
from ai4s_agent.corpus_report_generator import generate_corpus_report
from ai4s_agent.corpus_reproducibility_auditor import audit_corpus_reproducibility
from ai4s_agent.oled_review_packet_generator import generate_oled_review_packet
from ai4s_agent.phase3_corpus_extractor import extract_corpus_records
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation, build_scientific_dataset
from ai4s_agent.workflows.phase1_full_pipeline import run_phase1_full_pipeline


@dataclass(frozen=True)
class CorpusToPhase1WorkflowResult:
    status: str
    corpus_workflow_report_json: str
    corpus_extraction_manifest_json: str
    corpus_conflict_report_json: str
    candidate_dataset_csv: str
    training_dataset_csv: str
    rejected_records_json: str
    dataset_manifest_json: str
    oled_candidates_json: str = ""
    oled_text_evidence_candidates_json: str = ""
    oled_schema_candidates_json: str = ""
    oled_compiled_records_json: str = ""
    oled_review_packet_json: str = ""
    oled_review_packet_md: str = ""
    oled_reviewer_decision_template_json: str = ""
    oled_review_summary_json: str = ""
    oled_compiled_admission_packet_json: str = ""
    oled_compiled_admission_packet_md: str = ""
    oled_compiled_admission_decision_template_json: str = ""
    oled_compiled_admission_summary_json: str = ""
    oled_review_item_count: int = 0
    oled_compiled_admission_item_count: int = 0
    oled_review_high_priority_count: int = 0
    oled_review_medium_priority_count: int = 0
    oled_review_low_priority_count: int = 0
    full_phase1_pipeline_json: str = ""
    report_json: str = ""
    report_md: str = ""
    ranked_candidates_csv: str = ""
    corpus_report_json: str = ""
    corpus_report_md: str = ""
    corpus_summary_json: str = ""
    corpus_lineage_manifest_json: str = ""
    corpus_replay_manifest_json: str = ""
    corpus_reproducibility_report_json: str = ""


def run_corpus_to_phase1_workflow(
    *,
    parsed_document_paths: list[str | Path],
    output_dir: str | Path,
    run_id: str,
    confirmation: DatasetConfirmation,
    generated_at: str | None = None,
    property_ids: list[str] | None = None,
    n_bits: int = 256,
    topn: int = 10,
    min_numeric_ratio: float = 0.6,
    min_nonempty: int = 30,
) -> CorpusToPhase1WorkflowResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    document_paths = [Path(path).expanduser().resolve() for path in parsed_document_paths]

    extraction = extract_corpus_records(
        parsed_documents=document_paths,
        output_dir=output_path / "extraction",
        run_id=run_id,
        generated_at=generated,
    )
    conflict_audit = audit_corpus_conflicts(
        records=extraction.records,
        extraction_rejections=extraction.rejected_records,
        output_dir=output_path / "conflicts",
        run_id=run_id,
        generated_at=generated,
    )
    dataset = build_scientific_dataset(
        conflict_audit.accepted_records,
        output_dir=output_path / "dataset",
        run_id=run_id,
        confirmation=confirmation,
        generated_at=generated,
    )
    builder_rejected_count = _builder_rejection_count(dataset.rejected_records_json)
    total_rejected_count = len(conflict_audit.rejected_records) + builder_rejected_count
    rejected_records_json = _merge_rejected_records(
        builder_rejected_records_json=dataset.rejected_records_json,
        audit_rejected_records=conflict_audit.rejected_records,
        run_id=run_id,
        generated_at=generated,
    )
    review_packet = generate_oled_review_packet(
        run_id=run_id,
        output_dir=output_path / "review",
        oled_candidates_json=extraction.oled_candidates_json,
        oled_text_evidence_candidates_json=extraction.oled_text_evidence_candidates_json,
        oled_schema_candidates_json=extraction.oled_schema_candidates_json,
        oled_compiled_records_json=extraction.oled_compiled_records_json,
        corpus_extraction_manifest_json=extraction.corpus_extraction_manifest_json,
        generated_at=generated,
    )
    dataset_manifest_json = _update_dataset_manifest(
        dataset_manifest_json=dataset.dataset_manifest_json,
        corpus_extraction=extraction,
        conflict_summary=conflict_audit.summary,
        rejected_records_json=rejected_records_json,
        rejected_record_count=total_rejected_count,
        review_packet=review_packet,
    )
    _update_conflict_summary(
        conflict_summary_json=conflict_audit.conflict_summary_json,
        document_count=extraction.report.document_count,
        candidate_record_count=dataset.candidate_record_count,
        training_record_count=dataset.training_record_count,
        oled_candidate_count=extraction.report.oled_candidate_count,
        oled_text_evidence_candidate_count=extraction.report.oled_text_evidence_candidate_count,
        oled_schema_candidate_count=extraction.report.oled_schema_candidate_count,
        oled_compiled_record_count=extraction.report.oled_compiled_record_count,
        oled_review_item_count=review_packet.review_item_count,
        oled_compiled_admission_item_count=review_packet.compiled_admission_item_count,
        oled_review_high_priority_count=review_packet.high_priority_count,
        oled_review_medium_priority_count=review_packet.medium_priority_count,
        oled_review_low_priority_count=review_packet.low_priority_count,
    )

    phase1_status = "not_run"
    full_phase1_pipeline_json = ""
    ranked_candidates_csv = ""
    phase1_report_json = ""
    phase1_report_md = ""
    ranked_candidates: list[dict[str, Any]] = []

    artifact_paths: dict[str, str | Path] = {
        "corpus_records_json": extraction.corpus_records_json,
        "oled_candidates_json": extraction.oled_candidates_json,
        "oled_text_evidence_candidates_json": extraction.oled_text_evidence_candidates_json,
        "oled_schema_candidates_json": extraction.oled_schema_candidates_json,
        "oled_compiled_records_json": extraction.oled_compiled_records_json,
        "oled_review_packet_json": review_packet.review_packet_json,
        "oled_review_packet_md": review_packet.review_packet_md,
        "oled_reviewer_decision_template_json": review_packet.reviewer_decision_template_json,
        "oled_review_summary_json": review_packet.review_summary_json,
        "oled_compiled_admission_packet_json": review_packet.compiled_admission_packet_json,
        "oled_compiled_admission_packet_md": review_packet.compiled_admission_packet_md,
        "oled_compiled_admission_decision_template_json": review_packet.compiled_admission_decision_template_json,
        "oled_compiled_admission_summary_json": review_packet.compiled_admission_summary_json,
        "corpus_extraction_manifest_json": extraction.corpus_extraction_manifest_json,
        "corpus_conflict_report_json": conflict_audit.conflict_report_json,
        "conflict_summary_json": conflict_audit.conflict_summary_json,
        "candidate_dataset_csv": dataset.candidate_dataset_csv,
        "training_dataset_csv": dataset.training_dataset_csv,
        "rejected_records_json": rejected_records_json,
        "dataset_manifest_json": dataset_manifest_json,
    }

    if confirmation.confirmed:
        phase1 = run_phase1_full_pipeline(
            confirmed_training_dataset_csv=dataset.training_dataset_csv,
            candidate_dataset_csv=dataset.candidate_dataset_csv,
            dataset_manifest_json=dataset_manifest_json,
            output_dir=output_path / "phase1",
            run_id=run_id,
            confirmation=confirmation,
            property_ids=property_ids,
            n_bits=n_bits,
            topn=topn,
            generated_at=generated,
            min_numeric_ratio=min_numeric_ratio,
            min_nonempty=min_nonempty,
        )
        phase1_status = phase1.status
        full_phase1_pipeline_json = phase1.full_phase1_pipeline_json
        ranked_candidates_csv = phase1.ranked_candidates_csv
        phase1_report_json = phase1.report_json
        phase1_report_md = phase1.report_md
        ranked_candidates = _read_csv_rows(ranked_candidates_csv)
        artifact_paths.update(
            {
                "full_phase1_pipeline_json": full_phase1_pipeline_json,
                "ranked_candidates_csv": ranked_candidates_csv,
                "phase1_report_json": phase1_report_json,
                "phase1_report_md": phase1_report_md,
            }
        )

    reproducibility = audit_corpus_reproducibility(
        input_document_paths=document_paths,
        artifact_paths=artifact_paths,
        output_dir=output_path / "reproducibility",
        run_id=run_id,
        generated_at=generated,
    )
    corpus_report = generate_corpus_report(
        conflict_summary_json=conflict_audit.conflict_summary_json,
        phase1_pipeline_json=full_phase1_pipeline_json,
        reproducibility_report_json=reproducibility.corpus_reproducibility_report_json,
        oled_review_summary_json=review_packet.review_summary_json,
        ranked_candidates=ranked_candidates,
        output_dir=output_path / "report",
        run_id=run_id,
        generated_at=generated,
    )
    status = "success" if confirmation.confirmed else "awaiting_confirmation"
    workflow_report_json = output_path / "corpus_workflow_report.json"
    write_json(
        workflow_report_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "status": status,
            "confirmation": confirmation.to_dict(),
            "summary": {
                "document_count": extraction.report.document_count,
                "extracted_record_count": extraction.report.extracted_record_count,
                "accepted_record_count": conflict_audit.summary["accepted_record_count"],
                "rejected_record_count": total_rejected_count,
                "conflict_count": conflict_audit.summary["conflict_count"],
                "unresolved_conflict_count": conflict_audit.summary["unresolved_conflict_count"],
                "oled_candidate_count": extraction.report.oled_candidate_count,
                "oled_text_evidence_candidate_count": extraction.report.oled_text_evidence_candidate_count,
                "oled_schema_candidate_count": extraction.report.oled_schema_candidate_count,
                "oled_compiled_record_count": extraction.report.oled_compiled_record_count,
                "oled_review_item_count": review_packet.review_item_count,
                "oled_compiled_admission_item_count": review_packet.compiled_admission_item_count,
                "oled_review_high_priority_count": review_packet.high_priority_count,
                "oled_review_medium_priority_count": review_packet.medium_priority_count,
                "oled_review_low_priority_count": review_packet.low_priority_count,
                "candidate_record_count": dataset.candidate_record_count,
                "training_record_count": dataset.training_record_count,
                "phase1_status": phase1_status,
                "top_ranked_candidate_count": len(ranked_candidates),
            },
            "artifacts": {
                "corpus_extraction_manifest_json": extraction.corpus_extraction_manifest_json,
                "oled_candidates_json": extraction.oled_candidates_json,
                "oled_text_evidence_candidates_json": extraction.oled_text_evidence_candidates_json,
                "oled_schema_candidates_json": extraction.oled_schema_candidates_json,
                "oled_compiled_records_json": extraction.oled_compiled_records_json,
                "oled_review_packet_json": review_packet.review_packet_json,
                "oled_review_packet_md": review_packet.review_packet_md,
                "oled_reviewer_decision_template_json": review_packet.reviewer_decision_template_json,
                "oled_review_summary_json": review_packet.review_summary_json,
                "oled_compiled_admission_packet_json": review_packet.compiled_admission_packet_json,
                "oled_compiled_admission_packet_md": review_packet.compiled_admission_packet_md,
                "oled_compiled_admission_decision_template_json": review_packet.compiled_admission_decision_template_json,
                "oled_compiled_admission_summary_json": review_packet.compiled_admission_summary_json,
                "corpus_conflict_report_json": conflict_audit.conflict_report_json,
                "candidate_dataset_csv": dataset.candidate_dataset_csv,
                "training_dataset_csv": dataset.training_dataset_csv,
                "rejected_records_json": rejected_records_json,
                "dataset_manifest_json": dataset_manifest_json,
                "full_phase1_pipeline_json": full_phase1_pipeline_json,
                "ranked_candidates_csv": ranked_candidates_csv,
                "phase1_report_json": phase1_report_json,
                "phase1_report_md": phase1_report_md,
                "corpus_report_json": corpus_report.corpus_report_json,
                "corpus_replay_manifest_json": reproducibility.corpus_replay_manifest_json,
            },
            "external_services_required": False,
        },
    )
    return CorpusToPhase1WorkflowResult(
        status=status,
        corpus_workflow_report_json=str(workflow_report_json),
        corpus_extraction_manifest_json=extraction.corpus_extraction_manifest_json,
        corpus_conflict_report_json=conflict_audit.conflict_report_json,
        oled_candidates_json=extraction.oled_candidates_json,
        oled_text_evidence_candidates_json=extraction.oled_text_evidence_candidates_json,
        oled_schema_candidates_json=extraction.oled_schema_candidates_json,
        oled_compiled_records_json=extraction.oled_compiled_records_json,
        oled_review_packet_json=review_packet.review_packet_json,
        oled_review_packet_md=review_packet.review_packet_md,
        oled_reviewer_decision_template_json=review_packet.reviewer_decision_template_json,
        oled_review_summary_json=review_packet.review_summary_json,
        oled_compiled_admission_packet_json=review_packet.compiled_admission_packet_json,
        oled_compiled_admission_packet_md=review_packet.compiled_admission_packet_md,
        oled_compiled_admission_decision_template_json=review_packet.compiled_admission_decision_template_json,
        oled_compiled_admission_summary_json=review_packet.compiled_admission_summary_json,
        oled_review_item_count=review_packet.review_item_count,
        oled_compiled_admission_item_count=review_packet.compiled_admission_item_count,
        oled_review_high_priority_count=review_packet.high_priority_count,
        oled_review_medium_priority_count=review_packet.medium_priority_count,
        oled_review_low_priority_count=review_packet.low_priority_count,
        candidate_dataset_csv=dataset.candidate_dataset_csv,
        training_dataset_csv=dataset.training_dataset_csv,
        rejected_records_json=rejected_records_json,
        dataset_manifest_json=dataset_manifest_json,
        full_phase1_pipeline_json=full_phase1_pipeline_json,
        report_json=phase1_report_json,
        report_md=phase1_report_md,
        ranked_candidates_csv=ranked_candidates_csv,
        corpus_report_json=corpus_report.corpus_report_json,
        corpus_report_md=corpus_report.corpus_report_md,
        corpus_summary_json=corpus_report.corpus_summary_json,
        corpus_lineage_manifest_json=reproducibility.corpus_lineage_manifest_json,
        corpus_replay_manifest_json=reproducibility.corpus_replay_manifest_json,
        corpus_reproducibility_report_json=reproducibility.corpus_reproducibility_report_json,
    )


def _builder_rejection_count(path_like: str | Path) -> int:
    payload = _load_json(path_like)
    records = payload.get("records")
    return len(records) if isinstance(records, list) else 0


def _merge_rejected_records(
    *,
    builder_rejected_records_json: str | Path,
    audit_rejected_records: list[dict[str, Any]],
    run_id: str,
    generated_at: str,
) -> str:
    path = Path(builder_rejected_records_json).expanduser().resolve()
    payload = _load_json(path)
    builder_records = payload.get("records") if isinstance(payload.get("records"), list) else []
    combined = [*audit_rejected_records, *builder_records]
    write_json(
        path,
        {
            "run_id": run_id,
            "generated_at": generated_at,
            "records": combined,
            "source": "corpus_conflict_audit_plus_dataset_validation",
        },
    )
    return str(path)


def _update_dataset_manifest(
    *,
    dataset_manifest_json: str | Path,
    corpus_extraction: Any,
    conflict_summary: dict[str, Any],
    rejected_records_json: str,
    rejected_record_count: int,
    review_packet: Any,
) -> str:
    path = Path(dataset_manifest_json).expanduser().resolve()
    manifest = _load_json(path)
    manifest["rejected_record_count"] = int(rejected_record_count)
    manifest["corpus"] = {
        "document_count": corpus_extraction.report.document_count,
        "paper_ids": corpus_extraction.report.paper_ids,
        "conflict_count": conflict_summary.get("conflict_count", 0),
        "unresolved_conflict_count": conflict_summary.get("unresolved_conflict_count", 0),
        "consistent_duplicate_count": conflict_summary.get("consistent_duplicate_count", 0),
        "oled_candidate_count": corpus_extraction.report.oled_candidate_count,
        "oled_text_evidence_candidate_count": corpus_extraction.report.oled_text_evidence_candidate_count,
        "oled_schema_candidate_count": corpus_extraction.report.oled_schema_candidate_count,
        "oled_compiled_record_count": corpus_extraction.report.oled_compiled_record_count,
        "oled_review_item_count": review_packet.review_item_count,
        "oled_compiled_admission_item_count": review_packet.compiled_admission_item_count,
        "oled_review_high_priority_count": review_packet.high_priority_count,
        "oled_review_medium_priority_count": review_packet.medium_priority_count,
        "oled_review_low_priority_count": review_packet.low_priority_count,
    }
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    artifacts["rejected_records_json"] = rejected_records_json
    artifacts["oled_candidates_json"] = corpus_extraction.oled_candidates_json
    artifacts["oled_text_evidence_candidates_json"] = corpus_extraction.oled_text_evidence_candidates_json
    artifacts["oled_schema_candidates_json"] = corpus_extraction.oled_schema_candidates_json
    artifacts["oled_compiled_records_json"] = corpus_extraction.oled_compiled_records_json
    artifacts["oled_review_packet_json"] = review_packet.review_packet_json
    artifacts["oled_review_packet_md"] = review_packet.review_packet_md
    artifacts["oled_reviewer_decision_template_json"] = review_packet.reviewer_decision_template_json
    artifacts["oled_review_summary_json"] = review_packet.review_summary_json
    artifacts["oled_compiled_admission_packet_json"] = review_packet.compiled_admission_packet_json
    artifacts["oled_compiled_admission_packet_md"] = review_packet.compiled_admission_packet_md
    artifacts["oled_compiled_admission_decision_template_json"] = review_packet.compiled_admission_decision_template_json
    artifacts["oled_compiled_admission_summary_json"] = review_packet.compiled_admission_summary_json
    manifest["artifacts"] = artifacts
    write_json(path, manifest)
    return str(path)


def _update_conflict_summary(
    *,
    conflict_summary_json: str | Path,
    document_count: int,
    candidate_record_count: int,
    training_record_count: int,
    oled_candidate_count: int,
    oled_text_evidence_candidate_count: int,
    oled_schema_candidate_count: int,
    oled_compiled_record_count: int,
    oled_review_item_count: int,
    oled_compiled_admission_item_count: int,
    oled_review_high_priority_count: int,
    oled_review_medium_priority_count: int,
    oled_review_low_priority_count: int,
) -> None:
    path = Path(conflict_summary_json).expanduser().resolve()
    summary = _load_json(path)
    summary["document_count"] = document_count
    summary["candidate_record_count"] = candidate_record_count
    summary["training_record_count"] = training_record_count
    summary["oled_candidate_count"] = oled_candidate_count
    summary["oled_text_evidence_candidate_count"] = oled_text_evidence_candidate_count
    summary["oled_schema_candidate_count"] = oled_schema_candidate_count
    summary["oled_compiled_record_count"] = oled_compiled_record_count
    summary["oled_review_item_count"] = oled_review_item_count
    summary["oled_compiled_admission_item_count"] = oled_compiled_admission_item_count
    summary["oled_review_high_priority_count"] = oled_review_high_priority_count
    summary["oled_review_medium_priority_count"] = oled_review_medium_priority_count
    summary["oled_review_low_priority_count"] = oled_review_low_priority_count
    write_json(path, summary)


def _read_csv_rows(path_like: str | Path) -> list[dict[str, str]]:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
