from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.phase1_candidate_ranker import rank_phase1_candidates
from ai4s_agent.phase1_report_generator import generate_phase1_report
from ai4s_agent.phase1_training_orchestrator import run_phase1_training
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation


@dataclass(frozen=True)
class Phase1FullPipelineResult:
    status: str
    full_phase1_pipeline_json: str
    trained_model_paths: dict[str, str] = field(default_factory=dict)
    ranked_candidates_csv: str = ""
    report_json: str = ""
    report_md: str = ""
    hashes: dict[str, str] = field(default_factory=dict)


def run_phase1_full_pipeline(
    *,
    confirmed_training_dataset_csv: str | Path,
    candidate_dataset_csv: str | Path,
    dataset_manifest_json: str | Path,
    output_dir: str | Path,
    run_id: str,
    confirmation: DatasetConfirmation,
    property_ids: list[str] | None = None,
    n_bits: int = 256,
    topn: int = 10,
    generated_at: str | None = None,
    min_numeric_ratio: float = 0.6,
    min_nonempty: int = 30,
) -> Phase1FullPipelineResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    props = property_ids or ["plqy"]

    training = run_phase1_training(
        confirmed_training_dataset_csv=confirmed_training_dataset_csv,
        dataset_manifest_json=dataset_manifest_json,
        output_dir=output_path / "training",
        run_id=run_id,
        confirmation=confirmation,
        property_ids=props,
        n_bits=n_bits,
        generated_at=generated,
        min_numeric_ratio=min_numeric_ratio,
        min_nonempty=min_nonempty,
    )
    ranking = rank_phase1_candidates(
        candidate_dataset_csv=candidate_dataset_csv,
        training_metadata_json=training.training_metadata_json,
        output_dir=output_path / "ranking",
        run_id=run_id,
        property_ids=props,
        topn=topn,
        generated_at=generated,
    )
    report = generate_phase1_report(
        training_metadata_json=training.training_metadata_json,
        ranking_metadata_json=ranking.ranking_metadata_json,
        dataset_manifest_json=dataset_manifest_json,
        output_dir=output_path / "report",
        run_id=run_id,
        generated_at=generated,
    )
    trained_model_paths = {
        prop: str(model["model_path"])
        for prop, model in training.models.items()
    }
    hashes = {**training.hashes, **ranking.hashes}
    pipeline_json = output_path / "full_phase1_pipeline.json"
    write_json(
        pipeline_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "status": "success",
            "confirmation": confirmation.to_dict(),
            "property_ids": props,
            "hashes": hashes,
            "artifacts": {
                "training_metadata_json": training.training_metadata_json,
                "ranking_metadata_json": ranking.ranking_metadata_json,
                "full_phase1_pipeline_json": str(pipeline_json),
                "ranked_candidates_csv": ranking.ranked_candidates_csv,
                "report_json": report.report_json,
                "report_md": report.report_md,
                "trained_model_paths": trained_model_paths,
            },
        },
    )
    return Phase1FullPipelineResult(
        status="success",
        full_phase1_pipeline_json=str(pipeline_json),
        trained_model_paths=trained_model_paths,
        ranked_candidates_csv=ranking.ranked_candidates_csv,
        report_json=report.report_json,
        report_md=report.report_md,
        hashes=hashes,
    )
