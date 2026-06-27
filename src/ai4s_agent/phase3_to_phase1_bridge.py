from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.adapters.phase1 import (
    check_trainability_service,
    draft_cleaning_rules_adapter,
    execute_cleaning_adapter,
    filter_rank_adapter,
    inspect_dataset_service,
    predict_candidates_baseline_adapter,
    render_report_adapter,
    run_baseline_service,
    train_model_baseline_adapter,
)
from ai4s_agent.scientific_dataset_builder import DatasetConfirmation


@dataclass(frozen=True)
class Phase3ToPhase1BridgeResult:
    status: str
    adapter_statuses: dict[str, str] = field(default_factory=dict)
    phase1_baseline_report_json: str = ""
    candidate_predictions_csv: str = ""
    candidate_ranking_json: str = ""
    phase1_bridge_report_json: str = ""
    report_json: str = ""
    outputs: dict[str, str] = field(default_factory=dict)


def run_phase3_to_phase1_bridge(
    *,
    training_dataset_csv: str | Path,
    candidate_dataset_csv: str | Path,
    output_dir: str | Path,
    run_id: str,
    confirmation: DatasetConfirmation,
    property_id: str = "plqy",
    generated_at: str | None = None,
    min_numeric_ratio: float = 0.6,
    min_nonempty: int = 30,
    n_bits: int = 256,
    topn: int = 10,
    strict_smiles_cleaning: bool = True,
) -> Phase3ToPhase1BridgeResult:
    if not confirmation.confirmed:
        return Phase3ToPhase1BridgeResult(status="blocked_confirmation_required")

    generated = generated_at or now_iso()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    training_csv = Path(training_dataset_csv).expanduser().resolve()
    candidate_csv = Path(candidate_dataset_csv).expanduser().resolve()
    clean_dir = output_path / "clean"
    baseline_dir = output_path / "baseline"
    model_root = output_path / "models"

    adapter_statuses: dict[str, str] = {}

    inspect = inspect_dataset_service(
        {
            "input_csv": str(training_csv),
            "min_numeric_ratio": min_numeric_ratio,
            "min_nonempty": min_nonempty,
        }
    )
    _capture_status(adapter_statuses, "inspect_dataset", inspect)
    _ensure_success("inspect_dataset", inspect)

    draft = draft_cleaning_rules_adapter(
        {
            "inspect_result": inspect,
            "strict_smiles_cleaning": strict_smiles_cleaning,
        }
    )
    _ensure_success("draft_cleaning_rules", draft)
    mapping = dict(draft["cleaning_rules_draft"])
    mapping["properties"] = [
        {"property_id": property_id, "source_column": property_id},
        {"property_id": "lambda_em_nm", "source_column": "lambda_em_nm"},
    ]

    cleaned = execute_cleaning_adapter(
        {
            "run_id": run_id,
            "input_csv": str(training_csv),
            "output_dir": str(clean_dir),
            "mapping": mapping,
            "properties": [property_id, "lambda_em_nm"],
            "min_numeric_ratio": min_numeric_ratio,
            "min_nonempty": min_nonempty,
            "strict_smiles_cleaning": strict_smiles_cleaning,
        }
    )
    _capture_status(adapter_statuses, "clean_dataset", cleaned)
    _ensure_success("clean_dataset", cleaned)
    cleaned_master_csv = str(cleaned["outputs"]["cleaned_master_csv"])
    property_catalog_json = str(cleaned["outputs"]["property_catalog_json"])

    trainability = check_trainability_service(
        {
            "run_id": run_id,
            "property_catalog_json": property_catalog_json,
            "output_dir": str(output_path),
        }
    )
    _capture_status(adapter_statuses, "check_trainability", trainability)
    _ensure_success("check_trainability", trainability)

    baseline = run_baseline_service(
        {
            "run_id": run_id,
            "cleaned_master_csv": cleaned_master_csv,
            "output_dir": str(baseline_dir),
            "properties": [property_id],
        }
    )
    _capture_status(adapter_statuses, "run_baseline", baseline)
    _ensure_success("run_baseline", baseline)
    baseline_report_json = str(baseline["outputs"]["baseline_report_json"])

    train_model = train_model_baseline_adapter(
        {
            "run_id": run_id,
            "cleaned_master_csv": cleaned_master_csv,
            "property_id": property_id,
            "model_root": str(model_root),
            "n_bits": n_bits,
            "domain": "photophysical_oled",
        }
    )
    _capture_status(adapter_statuses, "train_model", train_model)
    _ensure_success("train_model", train_model)
    model_path = str(train_model["model_metadata"]["model_path"])

    prediction_csv = output_path / "candidate_predictions.csv"
    predicted = predict_candidates_baseline_adapter(
        {
            "candidate_csv": str(candidate_csv),
            "property_id": property_id,
            "model_path": model_path,
            "output_csv": str(prediction_csv),
        }
    )
    _capture_status(adapter_statuses, "predict_candidates", predicted)
    _ensure_success("predict_candidates", predicted)

    ranked_csv = output_path / "ranked_candidates.csv"
    ranked = filter_rank_adapter(
        {
            "run_id": run_id,
            "prediction_csv": str(prediction_csv),
            "output_csv": str(ranked_csv),
            "topn": topn,
            "score_columns": [predicted["score_column"]],
            "directions": {predicted["score_column"]: "maximize"},
            "weights": {predicted["score_column"]: 1.0},
        }
    )
    _capture_status(adapter_statuses, "filter_rank", ranked)
    _ensure_success("filter_rank", ranked)

    ranking_json = output_path / "candidate_ranking.json"
    ranking_rows = _read_csv_rows(ranked_csv)
    write_json(
        ranking_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "topn": len(ranking_rows),
            "score_columns": [predicted["score_column"]],
            "candidates": ranking_rows,
            "source_csv": str(ranked_csv),
        },
    )

    rendered = render_report_adapter(
        {
            "run_id": run_id,
            "output_dir": str(output_path / "reports"),
            "sections": {
                "Summary": [
                    "Phase 3 confirmed scientific dataset entered Phase 1 baseline pipeline.",
                    f"Confirmation source: {confirmation.confirmation_source}",
                ],
                "Adapters": [f"{key}: {value}" for key, value in adapter_statuses.items()],
            },
            "artifacts": {
                "baseline_report_json": baseline_report_json,
                "candidate_predictions_csv": str(prediction_csv),
                "candidate_ranking_json": str(ranking_json),
            },
        }
    )
    _capture_status(adapter_statuses, "render_report", rendered)
    _ensure_success("render_report", rendered)

    bridge_report_json = output_path / f"{run_id}_phase1_bridge_report.json"
    write_json(
        bridge_report_json,
        {
            "run_id": run_id,
            "generated_at": generated,
            "status": "success",
            "confirmation": confirmation.to_dict(),
            "adapter_statuses": adapter_statuses,
            "trainability_report": trainability["trainability_report"],
            "baseline_report_json": baseline_report_json,
            "candidate_ranking_json": str(ranking_json),
        },
    )
    outputs = {
        "cleaned_master_csv": cleaned_master_csv,
        "property_catalog_json": property_catalog_json,
        "phase1_baseline_report_json": baseline_report_json,
        "model_path": model_path,
        "candidate_predictions_csv": str(prediction_csv),
        "ranked_candidates_csv": str(ranked_csv),
        "candidate_ranking_json": str(ranking_json),
        "rendered_report_json": str(rendered["outputs"]["json"]),
        "phase1_bridge_report_json": str(bridge_report_json),
    }
    return Phase3ToPhase1BridgeResult(
        status="success",
        adapter_statuses=adapter_statuses,
        phase1_baseline_report_json=baseline_report_json,
        candidate_predictions_csv=str(prediction_csv),
        candidate_ranking_json=str(ranking_json),
        phase1_bridge_report_json=str(bridge_report_json),
        report_json=str(rendered["outputs"]["json"]),
        outputs=outputs,
    )


def _capture_status(adapter_statuses: dict[str, str], name: str, result: dict[str, Any]) -> None:
    adapter_statuses[name] = str(result.get("status") or "unknown")


def _ensure_success(name: str, result: dict[str, Any]) -> None:
    if result.get("status") != "success":
        raise RuntimeError(f"{name} failed: {result.get('error') or result}")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
