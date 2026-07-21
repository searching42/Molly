from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.oled_bounded_discovery_controller import (
    _route,
    _verified_oled_bounded_discovery_controller_from_files,
    run_oled_bounded_discovery_controller_from_files,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes
from tests.test_oled_candidate_decision import _inputs, _run as _run_decision


def _request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    limits: dict[str, int] | None = None,
) -> Path:
    publication, batch_receipt, inverse, evaluation = _inputs(tmp_path, monkeypatch)
    decision = _run_decision(
        tmp_path,
        publication,
        batch_receipt,
        inverse,
        evaluation,
    )
    payload = {
        "request_version": "oled_bounded_discovery_controller_request.v1",
        "limits": limits
        or {
            "max_iterations": 3,
            "max_generation_rounds": 2,
            "max_generated_candidates": 512,
        },
        "iterations": [
            {
                "decision_json": str(decision.output_dir / "candidate_decision.json"),
                "evaluation_json": str(evaluation.output_dir / "evaluation.json"),
                "inverse_design_json": str(inverse.output_dir / "inverse_design.json"),
                "batch_selection_json": str(batch_receipt),
                "screening_receipt_json": str(publication.screening_receipt),
                "ranked_shortlist_csv": str(publication.ranked_shortlist),
                "phase1_execution_dir": str(publication.phase1_execution_dir),
                "dataset_snapshot_json": str(publication.dataset_snapshot),
                "registry_snapshot_json": str(publication.registry_snapshot),
                "candidate_cost_manifest_json": None,
                "remote_known_hosts": None,
            }
        ],
    }
    path = tmp_path / "controller-request.json"
    path.write_bytes(_json_bytes(payload))
    return path


def test_controller_exact_replays_chain_and_never_executes_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, monkeypatch)
    result = run_oled_bounded_discovery_controller_from_files(
        controller_request_json=request,
        output_root=tmp_path / "controllers",
        generated_at="2026-07-21T21:00:00+08:00",
    )

    receipt = json.loads(
        (result.output_dir / "controller.json").read_text(encoding="utf-8")
    )
    assert receipt["limits"] == {
        "max_iterations": 3,
        "max_generation_rounds": 2,
        "max_generated_candidates": 512,
    }
    assert receipt["usage"]["iterations"] == 1
    assert receipt["usage"]["generation_rounds"] == 1
    assert receipt["claims"]["generation_executed"] is False
    assert receipt["claims"]["gate_bypassed"] is False
    if receipt["route"]["next_action"] == "request_generation_approval":
        assert receipt["route"]["requires_human_approval"] is True
        assert receipt["route"]["required_gate"] == "gate_5_final_threshold"
    else:
        assert receipt["route"]["next_action"] == "stop"

    with _verified_oled_bounded_discovery_controller_from_files(
        controller_json=result.output_dir / "controller.json",
        controller_request_json=request,
    ) as bound:
        assert bound.result.controller_id == result.controller_id


def test_controller_rejects_limits_above_hard_ceilings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        tmp_path,
        monkeypatch,
        limits={
            "max_iterations": 4,
            "max_generation_rounds": 2,
            "max_generated_candidates": 512,
        },
    )
    with pytest.raises(ValueError, match="hard ceilings"):
        run_oled_bounded_discovery_controller_from_files(
            controller_request_json=request,
            output_root=tmp_path / "controllers",
        )


def _incomplete_decision(target: int = 3) -> dict[str, object]:
    return {
        "status": "incomplete",
        "config": {"target_top_n": target, "constraints": {}},
    }


def test_route_requests_only_gated_generation_for_true_supply_shortfall() -> None:
    status, action, reason, count = _route(
        decision=_incomplete_decision(),
        predictions=[{"predictions": {"p1": 1.0}, "hard_constraints_passed": True}],
        iterations_used=1,
        generation_rounds_used=1,
        generated_candidates_used=10,
        limits={
            "max_iterations": 3,
            "max_generation_rounds": 2,
            "max_generated_candidates": 512,
        },
    )
    assert (status, action, reason, count) == (
        "waiting_user",
        "request_generation_approval",
        "property_eligible_candidate_shortfall",
        2,
    )


@pytest.mark.parametrize(
    ("predictions", "iterations", "rounds", "generated", "reason"),
    [
        (
            [{"predictions": {"p1": 1.0}, "hard_constraints_passed": True}] * 3,
            1,
            1,
            10,
            "non_supply_policy_prevented_complete_top_n",
        ),
        (
            [{"predictions": {"p1": 1.0}, "hard_constraints_passed": True}],
            3,
            1,
            10,
            "max_iterations_reached",
        ),
        (
            [{"predictions": {"p1": 1.0}, "hard_constraints_passed": True}],
            1,
            2,
            10,
            "max_generation_rounds_reached",
        ),
        (
            [{"predictions": {"p1": 1.0}, "hard_constraints_passed": True}],
            1,
            1,
            511,
            "max_generated_candidates_would_be_exceeded",
        ),
    ],
)
def test_route_stops_at_non_supply_or_budget_boundaries(
    predictions: list[dict[str, object]],
    iterations: int,
    rounds: int,
    generated: int,
    reason: str,
) -> None:
    status, action, observed_reason, count = _route(
        decision=_incomplete_decision(),
        predictions=predictions,
        iterations_used=iterations,
        generation_rounds_used=rounds,
        generated_candidates_used=generated,
        limits={
            "max_iterations": 3,
            "max_generation_rounds": 2,
            "max_generated_candidates": 512,
        },
    )
    assert (status, action, observed_reason, count) == (
        "stopped",
        "stop",
        reason,
        0,
    )
