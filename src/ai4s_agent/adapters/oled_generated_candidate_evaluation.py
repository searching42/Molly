from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.adapters.contract_validation import validate_adapter_output_shape
from ai4s_agent.oled_generated_candidate_evaluation import (
    run_oled_generated_candidate_evaluation_from_files,
)


_ADAPTER_NAME = "execute_oled_generated_candidate_evaluation_adapter"
_OUTPUT_FILENAMES = {
    "oled_generated_evaluation_receipt": "evaluation.json",
    "oled_generated_evaluation_predictions": "complete_predictions.jsonl",
    "oled_generated_evaluation_shortlist": "ranked_shortlist.csv",
    "oled_generated_evaluation_exclusions": "generated_candidate_exclusions.jsonl",
    "oled_generated_evaluation_report": "report.md",
}


def execute_oled_generated_candidate_evaluation_adapter(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Run the local PR-AT controlled-prediction successor."""

    required = {
        key: _required_string(payload, key)
        for key in (
            "run_id",
            "inverse_design_json",
            "batch_selection_json",
            "screening_receipt_json",
            "ranked_shortlist_csv",
            "phase1_execution_dir",
            "dataset_snapshot_json",
            "registry_snapshot_json",
            "output_root",
        )
    }
    if any(not value for value in required.values()):
        return _failed(
            "missing_required_fields",
            "Exact PR-AS, PR-ARb, PR-AP, PR-AO, PR-AI, and Registry inputs are required.",
        )
    if Path(required["output_root"]).expanduser().name != "oled_generated_evaluation":
        return _failed(
            "invalid_output_root",
            "Generated-candidate evaluation output root is not executor-owned.",
        )
    try:
        result = run_oled_generated_candidate_evaluation_from_files(
            inverse_design_json=required["inverse_design_json"],
            batch_selection_json=required["batch_selection_json"],
            screening_receipt_json=required["screening_receipt_json"],
            ranked_shortlist_csv=required["ranked_shortlist_csv"],
            phase1_execution_dir=required["phase1_execution_dir"],
            dataset_snapshot_json=required["dataset_snapshot_json"],
            registry_snapshot_json=required["registry_snapshot_json"],
            candidate_cost_manifest_json=(
                _optional_string(payload.get("candidate_cost_manifest_json")) or None
            ),
            remote_known_hosts=(
                _optional_string(payload.get("remote_known_hosts")) or None
            ),
            output_root=required["output_root"],
        )
    except Exception:
        return _failed(
            "generated_candidate_evaluation_failed",
            "OLED generated-candidate evaluation failed before publication.",
        )
    outputs = {
        artifact_id: str(result.output_dir / filename)
        for artifact_id, filename in _OUTPUT_FILENAMES.items()
    }
    if not all(Path(path).is_file() for path in outputs.values()):
        return _failed(
            "incomplete_generated_candidate_evaluation",
            "Generated-candidate evaluation publication is incomplete.",
        )
    return validate_adapter_output_shape(
        {
            "status": "success",
            "adapter": _ADAPTER_NAME,
            "outputs": outputs,
            "summary": {
                "evaluation_id": result.evaluation_id,
                "registry_prediction_count": result.registry_prediction_count,
                "generated_prediction_count": result.generated_prediction_count,
                "generated_exclusion_count": result.generated_exclusion_count,
                "shortlist_count": result.shortlist_count,
                "controlled_prediction_executed": True,
                "global_ranking_executed": True,
                "experimental_validation_claimed": False,
                "registry_mutated": False,
            },
        },
        required_top_level_keys=["status", "adapter", "outputs", "summary"],
    )


def _required_string(payload: dict[str, Any], key: str) -> str:
    return str(payload.get(key) or "").strip()


def _optional_string(value: Any) -> str:
    return str(value or "").strip()


def _failed(code: str, message: str) -> dict[str, Any]:
    return validate_adapter_output_shape(
        {
            "status": "failed",
            "adapter": _ADAPTER_NAME,
            "error": {"code": code, "message": message},
        },
        required_top_level_keys=["status", "adapter"],
    )
