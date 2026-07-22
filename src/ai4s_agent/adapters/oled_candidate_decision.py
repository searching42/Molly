from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.adapters.contract_validation import validate_adapter_output_shape
from ai4s_agent.oled_candidate_decision import run_oled_candidate_decision_from_files


_ADAPTER_NAME = "execute_oled_candidate_decision_adapter"
_OUTPUTS = {
    "oled_final_candidate_decision_receipt": "candidate_decision.json",
    "oled_final_candidate_decision_top_n": "top_candidates.csv",
    "oled_final_candidate_decision_dossier": "candidate_decision_dossier.csv",
    "oled_final_candidate_decision_report": "report.md",
}


def execute_oled_candidate_decision_adapter(payload: dict[str, Any]) -> dict[str, Any]:
    required_keys = (
        "run_id",
        "evaluation_json",
        "inverse_design_json",
        "batch_selection_json",
        "screening_receipt_json",
        "ranked_shortlist_csv",
        "phase1_execution_dir",
        "dataset_snapshot_json",
        "registry_snapshot_json",
        "output_root",
    )
    required = {key: str(payload.get(key) or "").strip() for key in required_keys}
    if any(not value for value in required.values()):
        return _failed(
            "missing_required_fields",
            "Exact PR-AT and upstream replay inputs are required.",
        )
    if Path(required["output_root"]).expanduser().name != "oled_candidate_decision":
        return _failed(
            "invalid_output_root",
            "Candidate-decision output root is not executor-owned.",
        )
    try:
        result = run_oled_candidate_decision_from_files(
            evaluation_json=required["evaluation_json"],
            inverse_design_json=required["inverse_design_json"],
            batch_selection_json=required["batch_selection_json"],
            screening_receipt_json=required["screening_receipt_json"],
            ranked_shortlist_csv=required["ranked_shortlist_csv"],
            phase1_execution_dir=required["phase1_execution_dir"],
            dataset_snapshot_json=required["dataset_snapshot_json"],
            registry_snapshot_json=required["registry_snapshot_json"],
            candidate_cost_manifest_json=(
                str(payload.get("candidate_cost_manifest_json") or "").strip() or None
            ),
            remote_known_hosts=(
                str(payload.get("remote_known_hosts") or "").strip() or None
            ),
            controller_request_json=(
                str(payload.get("controller_request_json") or "").strip() or None
            ),
            controller_json=(
                str(payload.get("controller_json") or "").strip() or None
            ),
            generation_authorization_json=(
                str(payload.get("generation_authorization_json") or "").strip()
                or None
            ),
            controller_report_md=(
                str(payload.get("controller_report_md") or "").strip() or None
            ),
            generation_roster_json=(
                str(payload.get("generation_roster_json") or "").strip() or None
            ),
            output_root=required["output_root"],
        )
    except Exception:
        return _failed(
            "candidate_decision_failed",
            "Final candidate decision failed before publication.",
        )
    outputs = {
        artifact_id: str(result.output_dir / filename)
        for artifact_id, filename in _OUTPUTS.items()
    }
    return validate_adapter_output_shape(
        {
            "status": "success",
            "adapter": _ADAPTER_NAME,
            "outputs": outputs,
            "summary": {
                "decision_id": result.decision_id,
                "decision_status": result.status,
                "target_top_n": result.target_count,
                "selected_candidate_count": result.selected_count,
                "supported_candidate_sources": ["registry", "generated"],
                "recommendation_only": True,
                "human_candidate_adjudication_performed": False,
                "validation_claimed": False,
            },
        },
        required_top_level_keys=["status", "adapter", "outputs", "summary"],
    )


def _failed(code: str, message: str) -> dict[str, Any]:
    return validate_adapter_output_shape(
        {
            "status": "failed",
            "adapter": _ADAPTER_NAME,
            "error": {"code": code, "message": message},
        },
        required_top_level_keys=["status", "adapter"],
    )


__all__ = ["execute_oled_candidate_decision_adapter"]
