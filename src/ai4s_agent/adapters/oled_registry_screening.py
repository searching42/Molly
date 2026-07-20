from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent._utils import strict_bool
from ai4s_agent.adapters.contract_validation import validate_adapter_output_shape
from ai4s_agent.oled_registry_candidate_screening import (
    run_oled_registry_candidate_screening_from_files,
)


_ADAPTER_NAME = "execute_oled_registry_candidate_screening_adapter"
_OUTPUT_FILENAMES: dict[str, str] = {
    "oled_registry_screening_receipt": "screening.json",
    "oled_registry_screening_shortlist": "ranked_shortlist.csv",
    "oled_registry_screening_predictions": "predictions.jsonl",
    "oled_registry_screening_exclusions": "excluded_candidates.jsonl",
    "oled_registry_screening_eligible_candidates": "eligible_candidates.csv",
    "oled_registry_screening_report": "report.md",
}


def execute_oled_registry_candidate_screening_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the exact-bound PR-AP Registry screening after a RunPlan gate.

    The RunPlan executor supplies all input paths from explicit artifacts and
    fixes the output root inside its run directory.  This adapter deliberately
    has no overwrite mode and never discovers a latest Registry, dataset, or
    PR-AO execution artifact.
    """

    run_id = _required_string(payload, "run_id")
    phase1_execution_dir = _required_string(payload, "phase1_execution_dir")
    dataset_snapshot_json = _required_string(payload, "dataset_snapshot_json")
    registry_snapshot_json = _required_string(payload, "registry_snapshot_json")
    output_root = _required_string(payload, "output_root")
    actor = str(payload.get("actor") or "").strip()
    if not all((run_id, phase1_execution_dir, dataset_snapshot_json, registry_snapshot_json, output_root)):
        return _failed("missing_required_fields", "Exact Registry screening inputs are required.")
    if Path(output_root).expanduser().resolve().name != "oled_registry_screening":
        return _failed("invalid_output_root", "Registry screening output root is not executor-owned.")
    try:
        confirmed = strict_bool(payload.get("confirmed"), key="confirmed")
    except ValueError as exc:
        return _failed("invalid_confirmation", str(exc))
    if not confirmed or not actor:
        return _failed("confirmation_required", "Approved gate confirmation and actor are required.")

    try:
        minimums = _string_list(payload.get("minimums", []), key="minimums")
        maximums = _string_list(payload.get("maximums", []), key="maximums")
    except ValueError as exc:
        return _failed("invalid_constraints", str(exc))

    try:
        result = run_oled_registry_candidate_screening_from_files(
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            output_root=output_root,
            minimums=minimums,
            maximums=maximums,
        )
    except Exception:
        # The lower-level runner may include local path details in its errors.
        # Adapter results are persisted and surfaced in the agent UI, so keep
        # this boundary stable and free of host-specific paths.
        return _failed("registry_candidate_screening_failed", "Registry candidate screening failed.")

    outputs = {
        artifact_id: str(result.output_dir / filename)
        for artifact_id, filename in _OUTPUT_FILENAMES.items()
    }
    if not all(Path(path).is_file() for path in outputs.values()):
        return _failed("incomplete_screening_publication", "Registry screening publication is incomplete.")
    return validate_adapter_output_shape(
        {
            "status": "success",
            "adapter": _ADAPTER_NAME,
            "outputs": outputs,
            "summary": {
                "screening_id": result.screening_id,
                "eligible_candidate_count": result.eligible_candidate_count,
                "excluded_candidate_count": result.excluded_candidate_count,
                "prediction_count": result.prediction_count,
                "shortlist_count": result.shortlist_count,
                "registry_mutated": False,
                "experimental_validation_claimed": False,
                "model_registered": False,
            },
        },
        required_top_level_keys=["status", "adapter", "outputs", "summary"],
    )


def _required_string(payload: dict[str, Any], key: str) -> str:
    return str(payload.get(key) or "").strip()


def _string_list(value: Any, *, key: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list of non-empty strings")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _failed(code: str, message: str) -> dict[str, Any]:
    return validate_adapter_output_shape(
        {
            "status": "failed",
            "adapter": _ADAPTER_NAME,
            "error": {"code": code, "message": message},
        },
        required_top_level_keys=["status", "adapter"],
    )
