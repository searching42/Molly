from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent._utils import strict_bool
from ai4s_agent.adapters.contract_validation import validate_adapter_output_shape
from ai4s_agent.oled_inverse_design import run_oled_inverse_design_from_files


_ADAPTER_NAME = "execute_oled_inverse_design_adapter"
_OUTPUT_FILENAMES: dict[str, str] = {
    "oled_inverse_design_receipt": "inverse_design.json",
    "oled_inverse_design_candidates": "generated_candidates.csv",
    "oled_inverse_design_exclusions": "excluded_candidates.jsonl",
    "oled_inverse_design_report": "report.md",
}
_MODES = frozenset({"existing_output", "remote"})


def execute_oled_inverse_design_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a PR-ARb-authorized candidate-generation boundary.

    The RunPlan executor owns and freezes the inputs.  This adapter never
    declares generated structures property-qualified and never writes Registry,
    Gold, dataset, model, or experimental state.
    """

    run_id = _required_string(payload, "run_id")
    batch_selection_json = _required_string(payload, "batch_selection_json")
    screening_receipt_json = _required_string(payload, "screening_receipt_json")
    ranked_shortlist_csv = _required_string(payload, "ranked_shortlist_csv")
    phase1_execution_dir = _required_string(payload, "phase1_execution_dir")
    dataset_snapshot_json = _required_string(payload, "dataset_snapshot_json")
    registry_snapshot_json = _required_string(payload, "registry_snapshot_json")
    reinvent4_config = _required_string(payload, "reinvent4_config")
    output_root = _required_string(payload, "output_root")
    actor = _required_string(payload, "actor")
    if not all(
        (
            run_id,
            batch_selection_json,
            screening_receipt_json,
            ranked_shortlist_csv,
            phase1_execution_dir,
            dataset_snapshot_json,
            registry_snapshot_json,
            reinvent4_config,
            output_root,
            actor,
        )
    ):
        return _failed(
            "missing_required_fields",
            "Exact PR-ARb, PR-AP, PR-AO, PR-AI, Registry, and REINVENT4 inputs are required.",
        )
    if Path(output_root).expanduser().name != "oled_inverse_design":
        return _failed(
            "invalid_output_root",
            "Inverse-design output root is not executor-owned.",
        )
    try:
        confirmed = strict_bool(payload.get("confirmed"), key="confirmed")
    except ValueError as exc:
        return _failed("invalid_confirmation", str(exc))
    if not confirmed:
        return _failed(
            "confirmation_required",
            "Approved gate confirmation and actor are required.",
        )

    mode = str(payload.get("reinvent4_mode") or "").strip().lower()
    if mode not in _MODES:
        return _failed("invalid_reinvent4_mode", "REINVENT4 mode is invalid.")
    existing_output = _optional_string(payload.get("reinvent4_output_csv"))
    if mode == "existing_output" and not existing_output:
        return _failed(
            "reinvent4_output_required",
            "existing_output mode requires a frozen REINVENT4 output CSV.",
        )
    if mode == "remote" and existing_output:
        return _failed(
            "reinvent4_output_not_allowed",
            "remote mode must not consume an existing REINVENT4 output CSV.",
        )
    remote_profile_id = _optional_string(payload.get("remote_profile_id"))
    remote_known_hosts = _optional_string(payload.get("remote_known_hosts"))
    if mode == "remote" and not remote_known_hosts:
        return _failed(
            "remote_known_hosts_required",
            "remote mode requires an executor-frozen pinned known-hosts file.",
        )
    if mode != "remote" and (remote_profile_id or remote_known_hosts):
        return _failed(
            "remote_transport_not_allowed",
            "remote transport inputs are only allowed for remote mode.",
        )
    try:
        seed = _nonnegative_int(payload.get("seed", 0), key="seed")
        timeout_sec = _positive_int(payload.get("timeout_sec", 7200), key="timeout_sec")
    except ValueError as exc:
        return _failed("invalid_inverse_design_options", str(exc))

    try:
        result = run_oled_inverse_design_from_files(
            batch_selection_json=batch_selection_json,
            screening_receipt_json=screening_receipt_json,
            ranked_shortlist_csv=ranked_shortlist_csv,
            phase1_execution_dir=phase1_execution_dir,
            dataset_snapshot_json=dataset_snapshot_json,
            registry_snapshot_json=registry_snapshot_json,
            candidate_cost_manifest_json=(
                _optional_string(payload.get("candidate_cost_manifest_json")) or None
            ),
            reinvent4_config=reinvent4_config,
            reinvent4_mode=mode,
            reinvent4_output_csv=existing_output or None,
            output_root=output_root,
            seed=seed,
            remote_profile_id=remote_profile_id or None,
            remote_known_hosts=remote_known_hosts or None,
            timeout_sec=timeout_sec,
        )
    except Exception:
        # Persisted adapter results are visible to users and must not reveal
        # local/remote paths, command output, or transport details.
        return _failed(
            "inverse_design_execution_failed",
            "OLED inverse-design execution failed before publication.",
        )

    outputs = {
        artifact_id: str(result.output_dir / filename)
        for artifact_id, filename in _OUTPUT_FILENAMES.items()
    }
    if not all(Path(path).is_file() for path in outputs.values()):
        return _failed(
            "incomplete_inverse_design_publication",
            "Inverse-design publication is incomplete.",
        )
    return validate_adapter_output_shape(
        {
            "status": "success",
            "adapter": _ADAPTER_NAME,
            "outputs": outputs,
            "summary": {
                "design_request_id": result.design_request_id,
                "publication_id": result.publication_id,
                "requested_candidate_count": result.requested_candidate_count,
                "accepted_candidate_count": result.accepted_candidate_count,
                "excluded_candidate_count": result.excluded_candidate_count,
                "backend_mode": result.backend_mode,
                "generation_executed": result.backend_mode == "remote",
                "existing_generator_output_imported": result.backend_mode == "existing_output",
                "property_qualification_claimed": False,
                "controlled_prediction_executed": False,
                "screening_executed": False,
                "ranking_executed": False,
                "experimental_validation_claimed": False,
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


def _positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _failed(code: str, message: str) -> dict[str, Any]:
    return validate_adapter_output_shape(
        {
            "status": "failed",
            "adapter": _ADAPTER_NAME,
            "error": {"code": code, "message": message},
        },
        required_top_level_keys=["status", "adapter"],
    )
