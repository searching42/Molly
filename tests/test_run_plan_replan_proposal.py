from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from ai4s_agent.run_plan_artifact_verifier import RunPlanArtifactFinding, RunPlanArtifactVerification
from ai4s_agent.run_plan_replan_proposal import (
    RunPlanReplanProposal,
    propose_replan_from_verification,
)


def _finding(
    category: str,
    decision: str,
    *,
    severity: str = "warning",
    evidence: dict[str, Any] | None = None,
    message: str | None = None,
) -> RunPlanArtifactFinding:
    return RunPlanArtifactFinding(
        finding_id=f"{category}-1",
        category=category,
        severity=severity,  # type: ignore[arg-type]
        decision=decision,  # type: ignore[arg-type]
        message=message or f"{category} requires attention.",
        evidence=evidence or {},
    )


def _verification(
    decision: str,
    findings: list[RunPlanArtifactFinding] | None = None,
) -> RunPlanArtifactVerification:
    return RunPlanArtifactVerification(
        project_id="proj-a",
        run_id="run-a",
        generated_at="2026-06-24T00:00:00Z",
        decision=decision,  # type: ignore[arg-type]
        summary=f"Verifier decision is {decision}.",
        findings=findings or [],
        observed={"queue": {"summary": {"queued_job_id": "job-a"}}},
    )


def test_replan_proposal_continues_without_patch_operations() -> None:
    proposal = propose_replan_from_verification(_verification("continue"))

    assert isinstance(proposal, RunPlanReplanProposal)
    assert proposal.decision_source == "verifier"
    assert proposal.verifier_decision == "continue"
    assert proposal.proposed_action == "continue"
    assert proposal.executable is False
    assert proposal.affected_tasks == []
    assert proposal.required_user_decisions == []
    assert proposal.proposed_run_plan_patch == {
        "schema_version": "reviewable_run_plan_patch.v1",
        "applied": False,
        "operations": [],
    }


def test_replan_proposal_maps_bad_metrics_to_reviewable_rerun_task() -> None:
    proposal = propose_replan_from_verification(
        _verification(
            "rerun_recommended",
            [
                _finding(
                    "poor_model_metrics",
                    "rerun_recommended",
                    evidence={"weak_metrics": [{"property_id": "plqy", "r2": -0.2}]},
                    message="Model metrics are weak enough to recommend a rerun.",
                )
            ],
        )
    )

    assert proposal.proposed_action == "rerun_task"
    assert proposal.affected_tasks == ["train_model"]
    assert proposal.source_finding_ids == ["poor_model_metrics-1"]
    assert proposal.executable is False
    assert any("Approve rerun" in item for item in proposal.required_user_decisions)
    assert proposal.proposed_run_plan_patch["applied"] is False
    assert proposal.proposed_run_plan_patch["operations"] == [
        {
            "op": "rerun_task",
            "task_id": "train_model",
            "source_finding_id": "poor_model_metrics-1",
            "category": "poor_model_metrics",
            "reason": "Model metrics are weak enough to recommend a rerun.",
        }
    ]


def test_replan_proposal_maps_waiting_user_to_request_review() -> None:
    proposal = propose_replan_from_verification(
        _verification(
            "needs_review",
            [
                _finding(
                    "waiting_user",
                    "needs_review",
                    evidence={
                        "waiting_task": "approve_training",
                        "required_gates": ["gate_3_train_config"],
                    },
                    message="Queued execution is waiting for user review.",
                )
            ],
        )
    )

    assert proposal.proposed_action == "request_review"
    assert proposal.affected_tasks == ["approve_training"]
    assert any("gate_3_train_config" in item for item in proposal.required_user_decisions)
    assert proposal.proposed_run_plan_patch["operations"][0]["op"] == "request_review"
    assert proposal.executable is False


def test_replan_proposal_maps_data_gaps_to_collect_more_data() -> None:
    proposal = propose_replan_from_verification(
        _verification(
            "needs_review",
            [
                _finding(
                    "missing_trainability_properties",
                    "needs_review",
                    evidence={"overall_status": "READY"},
                    message="Trainability report does not list any properties.",
                )
            ],
        )
    )

    assert proposal.proposed_action == "collect_more_data"
    assert proposal.affected_tasks == ["inspect_dataset", "check_trainability"]
    assert any("additional data" in item.lower() for item in proposal.required_user_decisions)
    assert proposal.proposed_run_plan_patch["operations"][0]["op"] == "collect_more_data"
    assert proposal.executable is False


def test_replan_proposal_blocks_critical_findings() -> None:
    proposal = propose_replan_from_verification(
        _verification(
            "blocked",
            [
                _finding(
                    "missing_artifact",
                    "blocked",
                    severity="critical",
                    evidence={"missing_artifacts": ["metrics/model_metrics.json"]},
                    message="Registered artifacts are missing on disk.",
                )
            ],
        )
    )

    assert proposal.proposed_action == "block"
    assert proposal.affected_tasks == ["artifact_registry"]
    assert any("Resolve blocked verifier finding" in item for item in proposal.required_user_decisions)
    assert proposal.proposed_run_plan_patch["operations"][0]["op"] == "block"
    assert proposal.executable is False


def test_replan_proposal_rejects_executable_true() -> None:
    payload = {
        "decision_source": "verifier",
        "verifier_decision": "continue",
        "proposed_action": "continue",
        "affected_tasks": [],
        "rationale": [],
        "required_user_decisions": [],
        "proposed_run_plan_patch": {
            "schema_version": "reviewable_run_plan_patch.v1",
            "applied": False,
            "operations": [],
        },
        "executable": True,
    }

    with pytest.raises(ValueError, match="executable must remain false"):
        RunPlanReplanProposal.model_validate(payload)


def test_replan_proposal_accepts_dict_verification_and_does_not_mutate_input() -> None:
    verification = _verification(
        "needs_review",
        [_finding("empty_generation", "needs_review", evidence={"generated_count": 0})],
    ).model_dump(mode="json")
    before = deepcopy(verification)

    proposal = propose_replan_from_verification(verification)

    assert proposal.proposed_action == "collect_more_data"
    assert verification == before
    assert proposal.executable is False
