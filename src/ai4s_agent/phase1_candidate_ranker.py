from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.adapters.phase1 import filter_rank_adapter, predict_candidates_baseline_adapter
from ai4s_agent.phase1_training_orchestrator import (
    DatasetNotConfirmedError,
    Phase1TrainingError,
    _load_json,
    _sha256_file,
    _sha256_json,
)


@dataclass(frozen=True)
class Phase1CandidateRankingResult:
    status: str
    ranked_candidates_csv: str
    ranking_metadata_json: str
    prediction_csv: str
    hashes: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


def rank_phase1_candidates(
    *,
    candidate_dataset_csv: str | Path,
    training_metadata_json: str | Path,
    output_dir: str | Path,
    run_id: str,
    property_ids: list[str] | None = None,
    topn: int = 10,
    generated_at: str | None = None,
    weights: dict[str, float] | None = None,
) -> Phase1CandidateRankingResult:
    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    candidate_csv = Path(candidate_dataset_csv).expanduser().resolve()
    training_metadata_path = Path(training_metadata_json).expanduser().resolve()
    training_metadata = _load_json(training_metadata_path)
    confirmation = training_metadata.get("confirmation") if isinstance(training_metadata.get("confirmation"), dict) else {}
    if confirmation.get("confirmed") is not True:
        raise DatasetNotConfirmedError("candidate ranking requires confirmed Phase 1 training metadata")

    models = training_metadata.get("models") if isinstance(training_metadata.get("models"), dict) else {}
    props = property_ids or list(models.keys())
    if not props:
        raise Phase1TrainingError("no trained properties available for candidate ranking")

    current_csv = candidate_csv
    prediction_columns: list[str] = []
    prediction_outputs: dict[str, str] = {}
    for prop in props:
        model_info = models.get(prop)
        if not isinstance(model_info, dict):
            raise Phase1TrainingError(f"missing trained model for property: {prop}")
        prediction_csv = output_path / f"{run_id}_{prop}_predictions.csv"
        predicted = predict_candidates_baseline_adapter(
            {
                "candidate_csv": str(current_csv),
                "property_id": prop,
                "model_path": str(model_info["model_path"]),
                "output_csv": str(prediction_csv),
            }
        )
        if predicted.get("status") != "success":
            raise Phase1TrainingError(f"predict_candidates:{prop} failed: {predicted.get('error') or predicted}")
        prediction_columns.append(str(predicted["score_column"]))
        prediction_outputs[prop] = str(prediction_csv)
        current_csv = prediction_csv

    ranked_csv = output_path / "ranked_candidates.csv"
    score_weights = weights or {column: 1.0 for column in prediction_columns}
    ranked = filter_rank_adapter(
        {
            "run_id": run_id,
            "prediction_csv": str(current_csv),
            "output_csv": str(ranked_csv),
            "topn": topn,
            "score_columns": prediction_columns,
            "directions": {column: "maximize" for column in prediction_columns},
            "weights": score_weights,
        }
    )
    if ranked.get("status") != "success":
        raise Phase1TrainingError(f"filter_rank failed: {ranked.get('error') or ranked}")

    candidate_hash = _sha256_file(candidate_csv)
    training_metadata_hash = _sha256_file(training_metadata_path)
    ranking_hash = _sha256_file(ranked_csv)
    scoring_config = {
        "model_based": True,
        "score_columns": prediction_columns,
        "directions": {column: "maximize" for column in prediction_columns},
        "weights": score_weights,
        "function": "weighted_score_over_model_predictions",
        "topn": int(topn),
    }
    hashes = {
        "candidate_dataset_hash": candidate_hash,
        "training_metadata_hash": training_metadata_hash,
        "ranking_hash": ranking_hash,
        "scoring_config_hash": _sha256_json(scoring_config),
    }
    rows = _read_csv_rows(ranked_csv)
    metadata = {
        "run_id": run_id,
        "generated_at": generated,
        "status": "success",
        "property_ids": props,
        "prediction_columns": prediction_columns,
        "scoring": scoring_config,
        "hashes": hashes,
        "topn": len(rows),
        "top_candidates": rows,
        "artifacts": {
            "candidate_dataset_csv": str(candidate_csv),
            "prediction_csv": str(current_csv),
            "ranked_candidates_csv": str(ranked_csv),
            "ranked_report_markdown": ranked.get("outputs", {}).get("markdown", ""),
        },
        "prediction_outputs": prediction_outputs,
    }
    ranking_metadata_json = output_path / "ranking_metadata.json"
    write_json(ranking_metadata_json, metadata)
    return Phase1CandidateRankingResult(
        status="success",
        ranked_candidates_csv=str(ranked_csv),
        ranking_metadata_json=str(ranking_metadata_json),
        prediction_csv=str(current_csv),
        hashes=hashes,
        outputs={
            "ranked_candidates_csv": str(ranked_csv),
            "ranking_metadata_json": str(ranking_metadata_json),
            "prediction_csv": str(current_csv),
        },
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
