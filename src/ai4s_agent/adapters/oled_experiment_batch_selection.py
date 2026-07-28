from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ai4s_agent._utils import strict_bool
from ai4s_agent.adapters.contract_validation import validate_adapter_output_shape
from ai4s_agent.oled_experiment_batch_selection import (
    run_oled_experiment_batch_selection_from_files,
)


_ADAPTER_NAME = "execute_oled_experiment_batch_selection_adapter"
_OUTPUT_FILENAMES: dict[str, str] = {
    "oled_experiment_batch_receipt": "batch_selection.json",
    "oled_experiment_batch_handoff": "experiment_batch.csv",
    "oled_candidate_decision_dossier": "candidate_decision_dossier.csv",
    "oled_experiment_batch_report": "experiment_handoff.md",
}


def execute_oled_experiment_batch_selection_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Produce a recommendation-only candidate batch from a PR-AP shortlist.

    The controlled RunPlan executor owns the output root and supplies only its
    run-local frozen input copies.  This adapter deliberately performs no
    procurement, synthesis, instrument, Registry, Gold, dataset, or model
    registration action.
    """

    run_id = _required_string(payload, "run_id")
    screening_receipt_json = _required_string(payload, "screening_receipt_json")
    ranked_shortlist_csv = _required_string(payload, "ranked_shortlist_csv")
    phase1_execution_dir = _required_string(payload, "phase1_execution_dir")
    dataset_snapshot_json = _required_string(payload, "dataset_snapshot_json")
    registry_snapshot_json = _required_string(payload, "registry_snapshot_json")
    output_root = _required_string(payload, "output_root")
    actor = str(payload.get("actor") or "").strip()
    if not all(
        (
            run_id,
            screening_receipt_json,
            ranked_shortlist_csv,
            phase1_execution_dir,
            dataset_snapshot_json,
            registry_snapshot_json,
            output_root,
        )
    ):
        return _failed(
            "missing_required_fields",
            "Exact screening publication and replay-anchor inputs are required.",
        )
    if Path(output_root).expanduser().name != "oled_experiment_batch":
        return _failed("invalid_output_root", "Experiment batch output root is not executor-owned.")
    try:
        confirmed = strict_bool(payload.get("confirmed"), key="confirmed")
    except ValueError as exc:
        return _failed("invalid_confirmation", str(exc))
    if not confirmed or not actor:
        return _failed("confirmation_required", "Approved gate confirmation and actor are required.")

    try:
        target_batch_size = _positive_int(payload.get("target_batch_size"), key="target_batch_size")
        minimums = _string_list(payload.get("minimums", []), key="minimums")
        maximums = _string_list(payload.get("maximums", []), key="maximums")
        max_budget_minor = _optional_nonnegative_int(
            payload.get("max_budget_minor"), key="max_budget_minor"
        )
        max_pairwise_tanimoto = _optional_probability(
            payload.get("max_pairwise_tanimoto"), key="max_pairwise_tanimoto"
        )
    except ValueError as exc:
        return _failed("invalid_selection_constraints", str(exc))
    if target_batch_size > 1 and max_pairwise_tanimoto is None:
        return _failed(
            "diversity_threshold_required",
            "max_pairwise_tanimoto is required when target_batch_size is greater than one.",
        )

    candidate_cost_manifest_json = _optional_string(payload.get("candidate_cost_manifest_json"))
    if max_budget_minor is not None and not candidate_cost_manifest_json:
        return _failed(
            "cost_manifest_required",
            "A local candidate cost manifest is required when max_budget_minor is set.",
        )

    try:
        result = run_oled_experiment_batch_selection_from_files(
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            output_root=output_root,
            target_batch_size=target_batch_size,
            minimums=minimums,
            maximums=maximums,
            max_budget_minor=max_budget_minor,
            candidate_cost_manifest_json=candidate_cost_manifest_json or None,
            max_pairwise_tanimoto=max_pairwise_tanimoto,
        )
    except Exception:
        # Persisted adapter receipts are user-facing.  Keep host paths and
        # implementation details out of stable agent error responses.
        return _failed("experiment_batch_selection_failed", "Experiment batch selection failed.")

    outputs = {
        artifact_id: str(result.output_dir / filename)
        for artifact_id, filename in _OUTPUT_FILENAMES.items()
    }
    if not all(Path(path).is_file() for path in outputs.values()):
        return _failed("incomplete_experiment_batch_publication", "Experiment batch publication is incomplete.")
    return validate_adapter_output_shape(
        {
            "status": "success",
            "adapter": _ADAPTER_NAME,
            "outputs": outputs,
            "summary": {
                "batch_id": result.batch_id,
                "batch_status": result.status,
                "selected_count": result.selected_count,
                "eligible_count": result.eligible_count,
                "excluded_count": result.excluded_count,
                "total_cost_minor": result.total_cost_minor,
                "candidate_supply_count": result.candidate_supply_count,
                "inverse_design_should_trigger": result.inverse_design_should_trigger,
                "generation_executed": False,
                "recommendation_only": True,
                "experimental_validation_claimed": False,
                "experiment_started": False,
                "experiment_completed": False,
                "experiment_executed": False,
                "procurement_started": False,
                "procurement_performed": False,
                "synthesis_performed": False,
                "measurement_performed": False,
                "registry_mutated": False,
                "gold_written": False,
                "dataset_written": False,
                "model_registered": False,
            },
        },
        required_top_level_keys=["status", "adapter", "outputs", "summary"],
    )


def _required_string(payload: dict[str, Any], key: str) -> str:
    return str(payload.get(key) or "").strip()


def _optional_string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any, *, key: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of non-empty strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _optional_nonnegative_int(value: Any, *, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _optional_probability(value: Any, *, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number between 0 and 1")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{key} must be a finite number between 0 and 1")
    return parsed


def _failed(code: str, message: str) -> dict[str, Any]:
    return validate_adapter_output_shape(
        {
            "status": "failed",
            "adapter": _ADAPTER_NAME,
            "error": {"code": code, "message": message},
        },
        required_top_level_keys=["status", "adapter"],
    )
