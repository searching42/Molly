"""Path-redacted, exact-replayed presentation for one PR-AV session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.oled_bounded_discovery_session import (
    WAITING_USER,
    _child_receipt,
    _read_session_json,
    _read_spec,
    _read_state,
    _reconcile_waiting_child,
    _result_from_state,
    _session_dir,
    _session_lock,
    _validate_external_state,
)
from ai4s_agent.storage import ProjectStorage


def build_oled_bounded_discovery_session_view(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Validate external facts and return only user-facing session data."""

    clean_project = validated_oled_bounded_project_id(project_id)
    session_dir = _session_dir(storage, clean_project, session_id)
    with _session_lock(session_dir):
        spec = _read_spec(session_dir)
        state = _read_state(session_dir)
        state = _reconcile_waiting_child(
            storage, clean_project, session_dir, spec, state
        )
        _validate_external_state(storage, clean_project, session_dir, spec, state)
        result = _result_from_state(session_dir, state)
        usage = _usage(storage=storage, project_id=clean_project, state=state)
        terminal = _terminal_view(
            storage=storage,
            project_id=clean_project,
            session_dir=session_dir,
            state=state,
        )
        limits = dict(spec["controller_limits"])
        return {
            "session_id": result.session_id,
            "revision": result.revision,
            "status": result.status,
            "current_step": result.current_step,
            "gate": (
                {
                    "required": True,
                    "gate": "gate_5_final_threshold",
                    "run_id": result.waiting_run_id,
                    "task_id": result.waiting_task_id,
                }
                if result.status == WAITING_USER
                else {"required": False}
            ),
            "limits": limits,
            "usage": usage,
            "remaining": {
                "iterations": max(0, limits["max_iterations"] - usage["iterations"]),
                "generation_rounds": max(
                    0,
                    limits["max_generation_rounds"] - usage["generation_rounds"],
                ),
                "generated_candidates": max(
                    0,
                    limits["max_generated_candidates"]
                    - usage["generated_candidates"],
                ),
            },
            "children": [
                {
                    "label": str(child["label"]),
                    "run_id": str(child["run_id"]),
                    "task_id": str(child["task_id"]),
                    "status": str(child["status"]),
                }
                for child in state["children"]
            ],
            "failure": dict(state["failure"]) if isinstance(state.get("failure"), dict) else None,
            "terminal": terminal,
            "claims": {
                "recommendation_only": True,
                "experimental_validation_claimed": False,
                "computational_validation_claimed": False,
                "registry_mutated": False,
                "human_candidate_adjudication_performed": False,
            },
        }


def validated_oled_bounded_project_id(value: str) -> str:
    """Return one unmodified safe project ID or reject it before any read."""

    if not isinstance(value, str) or value != value.strip():
        raise ValueError("PR-AW project_id must be canonical")
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError("PR-AW project_id is invalid")
    return value


def _usage(
    *, storage: ProjectStorage, project_id: str, state: dict[str, Any]
) -> dict[str, int]:
    iterations = sum(
        1
        for child in state["children"]
        if str(child["label"]).startswith("controller_")
        and child["status"] == "succeeded"
    )
    generated = 0
    rounds = 0
    for child in state["children"]:
        if not (
            str(child["label"]).startswith("generation_")
            and child["status"] == "succeeded"
        ):
            continue
        rounds += 1
        receipt = _child_receipt(
            storage,
            project_id,
            state,
            str(child["label"]),
            "oled_inverse_design_receipt",
        )
        counts = receipt.get("counts")
        if not isinstance(counts, dict):
            raise ValueError("PR-AW inverse-design counts are invalid")
        accepted = counts.get("accepted_candidate_count")
        if isinstance(accepted, bool) or not isinstance(accepted, int) or accepted < 0:
            raise ValueError("PR-AW inverse-design accepted count is invalid")
        generated += accepted
    return {
        "iterations": iterations,
        "generation_rounds": rounds,
        "generated_candidates": generated,
    }


def _terminal_view(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(state.get("result"), dict):
        return None
    result = _read_session_json(session_dir, "session_result.json")
    source_run_id = str(result.get("source_child_run_id") or "")
    matches = [
        child for child in state["children"] if child.get("run_id") == source_run_id
    ]
    if len(matches) != 1:
        raise ValueError("PR-AW terminal source child is invalid")
    child = matches[0]
    source = str(result.get("result_source") or "")
    if source == "pr_arb_v2":
        receipt = _child_receipt(
            storage,
            project_id,
            state,
            str(child["label"]),
            "oled_final_candidate_decision_receipt",
        )
        candidates = [_v2_candidate(item) for item in _list(receipt, "selected_candidates")]
    elif source == "pr_arb_v1":
        receipt = _child_receipt(
            storage,
            project_id,
            state,
            str(child["label"]),
            "oled_experiment_batch_receipt",
        )
        selection = receipt.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("PR-AW Registry-only selection is invalid")
        candidates = [
            _v1_candidate(item) for item in _list(selection, "selected_candidates")
        ]
    else:
        candidates = []
    return {
        "result_id": str(result.get("result_id") or ""),
        "status": str(result.get("status") or ""),
        "has_complete_top_n": result.get("has_complete_top_n") is True,
        "result_source": source,
        "stop_reason": str(result.get("stop_reason") or ""),
        "usage": dict(result.get("usage") or {}),
        "top_candidates": candidates,
    }


def _v2_candidate(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("PR-AW generated candidate is invalid")
    candidate = raw.get("candidate")
    properties = raw.get("properties")
    if not isinstance(candidate, dict) or not isinstance(properties, dict):
        raise ValueError("PR-AW generated candidate presentation is invalid")
    return {
        "selection_order": raw.get("selection_order"),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_kind": str(candidate.get("source_kind") or ""),
        "canonical_name": str(candidate.get("canonical_name") or ""),
        "canonical_isomeric_smiles": str(
            candidate.get("canonical_isomeric_smiles") or ""
        ),
        "inchikey": str(candidate.get("inchikey") or ""),
        "aggregate_percentile": candidate.get("aggregate_percentile"),
        "properties": {
            str(property_id): {
                "display_name": str(value.get("display_name") or property_id),
                "unit": str(value.get("unit") or ""),
                "direction": str(value.get("objective_direction") or ""),
                "predicted_value": value.get("predicted_value"),
                "screening_status": value.get("screening_constraint"),
                "decision_status": value.get("decision_constraint"),
            }
            for property_id, value in sorted(properties.items())
            if isinstance(value, dict)
        },
        "reason_codes": [str(item) for item in raw.get("reason_codes") or []],
    }


def _v1_candidate(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("PR-AW Registry candidate is invalid")
    predictions = raw.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("PR-AW Registry candidate predictions are invalid")
    return {
        "selection_order": raw.get("selection_order"),
        "candidate_id": str(raw.get("material_id") or ""),
        "source_kind": "registry",
        "canonical_name": str(raw.get("canonical_name") or ""),
        "canonical_isomeric_smiles": str(
            raw.get("canonical_isomeric_smiles") or ""
        ),
        "inchikey": "",
        "aggregate_percentile": raw.get("aggregate_percentile"),
        "properties": {
            str(property_id): {"predicted_value": value}
            for property_id, value in sorted(predictions.items())
        },
        "reason_codes": [
            str(item) for item in raw.get("selection_reason_codes") or []
        ],
    }


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"PR-AW {key} is invalid")
    return value


__all__ = [
    "build_oled_bounded_discovery_session_view",
    "validated_oled_bounded_project_id",
]
