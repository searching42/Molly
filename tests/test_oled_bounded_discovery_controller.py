from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ai4s_agent.oled_bounded_discovery_controller import (
    _accumulate_chemical_identity_ledger,
    _validate_iteration_predecessor_authorization,
    _loop_fingerprint_payload,
    _route,
    _verified_oled_bounded_discovery_controller_from_files,
    run_oled_bounded_discovery_controller_from_files,
    validate_oled_bounded_generation_authorization_bundle,
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
    assert (result.output_dir / "generation_authorization.json").is_file()
    if receipt["route"]["next_action"] == "request_generation_approval":
        assert receipt["route"]["requires_human_approval"] is True
        assert receipt["route"]["required_gate"] == "gate_5_final_threshold"
        authorization = validate_oled_bounded_generation_authorization_bundle(
            controller_request_json=request,
            controller_json=result.output_dir / "controller.json",
            generation_authorization_json=(
                result.output_dir / "generation_authorization.json"
            ),
            controller_report_md=result.output_dir / "report.md",
        )
        assert authorization.controller_id == result.controller_id
        assert authorization.target_task == "execute_oled_inverse_design"
        assert (
            authorization.requested_candidate_count
            == receipt["route"]["requested_candidate_count"]
        )
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


def test_controller_rejects_a_second_legacy_inverse_design_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the first history entry may lack a previous controller grant."""

    request = _request(tmp_path, monkeypatch)
    payload = json.loads(request.read_text(encoding="utf-8"))
    payload["iterations"].append(dict(payload["iterations"][0]))
    request.write_bytes(_json_bytes(payload))

    with pytest.raises(ValueError, match="requires the previous controller authorization"):
        run_oled_bounded_discovery_controller_from_files(
            controller_request_json=request,
            output_root=tmp_path / "controllers",
            generated_at="2026-07-21T21:00:00+08:00",
        )


def test_controller_rejects_a_second_iteration_with_the_wrong_previous_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_oled_inverse_design_runplan import (
        _controller_authorized_input_artifacts,
    )

    artifacts = _controller_authorized_input_artifacts(tmp_path, monkeypatch)
    current_request = json.loads(
        Path(artifacts["oled_bounded_controller_request_snapshot"]).read_text(
            encoding="utf-8"
        )
    )
    current_request["iterations"].append(dict(current_request["iterations"][0]))
    paths = {
        "controller_request_json": artifacts[
            "oled_bounded_controller_request_snapshot"
        ],
        "controller_json": artifacts["oled_bounded_controller_receipt"],
        "generation_authorization_json": artifacts[
            "oled_bounded_controller_generation_authorization"
        ],
        "controller_report_md": artifacts["oled_bounded_controller_report"],
    }

    with pytest.raises(ValueError, match="not bound to the previous controller state"):
        _validate_iteration_predecessor_authorization(
            current_request=current_request,
            current_iterations=current_request["iterations"],
            iteration_index=2,
            paths=paths,
            inverse_receipt={"controller_authorization": {}},
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


def test_route_counts_actual_generated_source_supply_not_prediction_successes() -> None:
    """A 500-source PR-AS publication leaves only 12 candidates of the 512 cap."""

    status, action, reason, count = _route(
        decision=_incomplete_decision(target=23),
        predictions=[
            {"predictions": {"p1": 1.0}, "hard_constraints_passed": True}
        ] * 10,
        iterations_used=1,
        generation_rounds_used=1,
        generated_candidates_used=500,
        limits={
            "max_iterations": 3,
            "max_generation_rounds": 2,
            "max_generated_candidates": 512,
        },
    )
    assert (status, action, reason, count) == (
        "stopped",
        "stop",
        "max_generated_candidates_would_be_exceeded",
        0,
    )


def test_chemical_identity_ledger_rejects_reissued_or_rebound_candidate_ids() -> None:
    by_id: dict[str, tuple[str, str, str]] = {}
    owners = {
        "canonical_isomeric_smiles": {},
        "standard_inchi": {},
        "inchikey": {},
    }
    first = {
        "candidate_id": "oled-generated:publication-a",
        "canonical_isomeric_smiles": "CCO",
        "standard_inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
    }
    _accumulate_chemical_identity_ledger(
        predictions=[first],
        candidate_identity_by_id=by_id,
        identity_owner_by_field=owners,
    )

    reissued = {**first, "candidate_id": "oled-generated:publication-b"}
    with pytest.raises(ValueError, match="duplicated across candidate IDs"):
        _accumulate_chemical_identity_ledger(
            predictions=[first, reissued],
            candidate_identity_by_id=by_id,
            identity_owner_by_field=owners,
        )

    rebound = {
        **first,
        "canonical_isomeric_smiles": "CCCO",
        "standard_inchi": "InChI=1S/C3H8O/c1-2-3-4/h4H,2-3H2,1H3",
        "inchikey": "BDERNNFJNOPAEC-UHFFFAOYSA-N",
    }
    with pytest.raises(ValueError, match="rebound to a different chemical identity"):
        _accumulate_chemical_identity_ledger(
            predictions=[rebound],
            candidate_identity_by_id=by_id,
            identity_owner_by_field=owners,
        )


def test_loop_fingerprint_payload_changes_when_scientific_context_changes() -> None:
    decision = {
        "config": {
            "target_top_n": 3,
            "constraints": {"s1_ev": {"min": 2.0}},
            "directions": {"s1_ev": "maximize"},
            "max_pairwise_tanimoto": 0.7,
            "max_budget_minor": 100,
            "currency": "USD",
            "selection_policy": "rank_anchored_greedy_max_min_tanimoto.v1",
            "property_presentation": {"s1_ev": {"unit": "eV"}},
        }
    }
    evaluation = {
        "sources": {
            "pr_ap_screening_id": "screening-a",
            "pr_ap_screening_sha256": "sha256:screening",
            "pr_ap_ranked_shortlist_sha256": "sha256:shortlist",
            "model_sha256": {"s1_ev": "sha256:model"},
            "phase1_execution_id": "execution-a",
            "phase1_execution_sha256": "sha256:execution",
            "dataset_snapshot_id": "dataset-a",
            "dataset_snapshot_sha256": "sha256:dataset",
            "registry_id": "registry-a",
            "registry_snapshot_sha256": "sha256:registry",
        },
        "config": {
            "constraints": {"s1_ev": {"min": 2.0}},
            "scoring_policy": "global_pareto_then_mean_rank_percentile.v1",
        },
    }
    baseline = _loop_fingerprint_payload(decision=decision, evaluation=evaluation)
    variants = []

    changed_target = deepcopy(decision)
    changed_target["config"]["target_top_n"] = 4
    variants.append((changed_target, evaluation))

    changed_constraints = deepcopy(decision)
    changed_constraints["config"]["constraints"]["s1_ev"]["min"] = 2.5
    variants.append((changed_constraints, evaluation))

    changed_budget = deepcopy(decision)
    changed_budget["config"]["max_budget_minor"] = 101
    variants.append((changed_budget, evaluation))

    changed_diversity = deepcopy(decision)
    changed_diversity["config"]["max_pairwise_tanimoto"] = 0.6
    variants.append((changed_diversity, evaluation))

    changed_policy = deepcopy(decision)
    changed_policy["config"]["selection_policy"] = "different-policy"
    variants.append((changed_policy, evaluation))

    changed_model = deepcopy(evaluation)
    changed_model["sources"]["model_sha256"]["s1_ev"] = "sha256:other-model"
    variants.append((decision, changed_model))

    changed_dataset = deepcopy(evaluation)
    changed_dataset["sources"]["dataset_snapshot_sha256"] = "sha256:other-dataset"
    variants.append((decision, changed_dataset))

    changed_registry = deepcopy(evaluation)
    changed_registry["sources"]["registry_snapshot_sha256"] = "sha256:other-registry"
    variants.append((decision, changed_registry))

    for changed_decision, changed_evaluation in variants:
        assert (
            _loop_fingerprint_payload(
                decision=changed_decision,
                evaluation=changed_evaluation,
            )
            != baseline
        )
