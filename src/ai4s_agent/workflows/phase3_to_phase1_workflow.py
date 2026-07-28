from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.phase3_scientific_extractor import extract_scientific_records
from ai4s_agent.phase3_to_phase1_bridge import Phase3ToPhase1BridgeResult, run_phase3_to_phase1_bridge
from ai4s_agent.schemas import ParsedDocument
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation, build_scientific_dataset


@dataclass(frozen=True)
class Phase3ToPhase1WorkflowResult:
    status: str
    full_pipeline_report_json: str
    scientific_dataset_manifest_json: str
    phase1_baseline_report_json: str
    candidate_ranking_json: str
    training_dataset_csv: str
    candidate_dataset_csv: str


def run_phase3_to_phase1_workflow(
    *,
    parsed_document: ParsedDocument,
    output_dir: str | Path,
    run_id: str,
    confirmation: DatasetConfirmation,
    generated_at: str | None = None,
    property_id: str = "plqy",
    min_numeric_ratio: float = 0.6,
    min_nonempty: int = 30,
    n_bits: int = 256,
    topn: int = 10,
) -> Phase3ToPhase1WorkflowResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    extraction = extract_scientific_records(parsed_document, run_id=run_id, generated_at=generated)
    dataset = build_scientific_dataset(
        extraction.records,
        output_dir=output_path / "dataset",
        run_id=run_id,
        confirmation=confirmation,
        generated_at=generated,
    )

    dataset_manifest = _read_json(Path(dataset.dataset_manifest_json))
    scientific_manifest_json = output_path / "scientific_dataset_manifest.json"
    write_json(scientific_manifest_json, dataset_manifest)

    bridge_result: Phase3ToPhase1BridgeResult | None = None
    phase1_baseline_report_json = ""
    candidate_ranking_json = ""
    if confirmation.confirmed:
        bridge_result = run_phase3_to_phase1_bridge(
            training_dataset_csv=dataset.training_dataset_csv,
            candidate_dataset_csv=dataset.candidate_dataset_csv,
            output_dir=output_path / "phase1",
            run_id=run_id,
            confirmation=confirmation,
            property_id=property_id,
            generated_at=generated,
            min_numeric_ratio=min_numeric_ratio,
            min_nonempty=min_nonempty,
            n_bits=n_bits,
            topn=topn,
        )
        if bridge_result.status == "success":
            phase1_baseline_report_json = str(output_path / "phase1_baseline_report.json")
            candidate_ranking_json = str(output_path / "candidate_ranking.json")
            write_json(Path(phase1_baseline_report_json), _read_json(Path(bridge_result.phase1_baseline_report_json)))
            write_json(Path(candidate_ranking_json), _read_json(Path(bridge_result.candidate_ranking_json)))

    status = "success" if bridge_result is not None and bridge_result.status == "success" else "awaiting_confirmation"
    ranking_count = 0
    if candidate_ranking_json:
        ranking_count = len(_read_json(Path(candidate_ranking_json)).get("candidates", []))

    full_report_json = output_path / "full_pipeline_report.json"
    write_json(
        full_report_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "status": status,
            "summary": {
                "extracted_record_count": extraction.extraction_report.extracted_record_count,
                "extraction_rejected_record_count": extraction.extraction_report.rejected_record_count,
                "conflict_count": extraction.conflict_report.conflict_count,
                "candidate_record_count": dataset.candidate_record_count,
                "training_record_count": dataset.training_record_count,
                "dataset_rejected_record_count": dataset.rejected_record_count,
                "candidate_ranking_count": ranking_count,
            },
            "confirmation": confirmation.to_dict(),
            "artifacts": {
                "scientific_dataset_manifest_json": str(scientific_manifest_json),
                "candidate_dataset_csv": dataset.candidate_dataset_csv,
                "training_dataset_csv": dataset.training_dataset_csv,
                "phase1_baseline_report_json": phase1_baseline_report_json,
                "candidate_ranking_json": candidate_ranking_json,
            },
            "phase1": {
                "status": bridge_result.status if bridge_result is not None else "not_started",
                "adapter_statuses": bridge_result.adapter_statuses if bridge_result is not None else {},
            },
        },
    )
    return Phase3ToPhase1WorkflowResult(
        status=status,
        full_pipeline_report_json=str(full_report_json),
        scientific_dataset_manifest_json=str(scientific_manifest_json),
        phase1_baseline_report_json=phase1_baseline_report_json,
        candidate_ranking_json=candidate_ranking_json,
        training_dataset_csv=dataset.training_dataset_csv,
        candidate_dataset_csv=dataset.candidate_dataset_csv,
    )


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}
