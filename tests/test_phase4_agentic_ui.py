import pytest

from ai4s_agent._utils import now_iso
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.agents.recovery import RecoveryAgent
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import (
    ProjectMemoryRecord,
    ReplanRequest,
    RunPlanRevision,
    VerificationFinding,
    VerificationReport,
)
from ai4s_agent.ui_cards import build_agent_review_card


def _verification_report(run_id: str) -> VerificationReport:
    return VerificationReport(
        project_id="proj-agentic-ui",
        run_id=run_id,
        generated_at=now_iso(),
        overall_decision="replan",
        findings=[
            VerificationFinding(
                finding_id="weak_metric_1",
                category="abnormal_model_metrics",
                severity="warning",
                decision="replan",
                message="Model R2 is negative.",
                evidence={"property_id": "plqy", "r2": -0.2},
            )
        ],
    )


def _revision(run_id: str) -> RunPlanRevision:
    previous = expand_run_plan(
        run_id=run_id,
        requested_tasks=["run_baseline"],
        available_artifacts=["cleaned_train_dataset", "property_catalog", "trainability_report"],
    )
    return RecoveryAgent().propose_revision(
        request=ReplanRequest(
            project_id="proj-agentic-ui",
            run_id=run_id,
            trigger="new_user_constraints",
            reason="User asked to retrain after weak metrics.",
            new_constraints=["switch backend", "retrain model"],
        ),
        previous_plan=previous,
        verification_report=_verification_report(run_id),
    )


def test_agent_review_card_summarizes_plan_findings_replan_and_approvals() -> None:
    memory_record = ProjectMemoryRecord(
        record_id="backend-rf",
        category="backend_choice",
        summary="Use random forest for small OLED datasets.",
        value={"backend": "random_forest"},
        decision="confirmed_backend_choice",
        confirmed_by="user",
    )
    proposal = PlannerAgent(memory_records=[memory_record]).propose_plan(
        run_id="run-agentic-ui",
        goal="Train a model for PLQY.",
        available_artifacts=["cleaned_train_dataset", "property_catalog", "trainability_report"],
    )
    revision = _revision("run-agentic-ui")

    card = build_agent_review_card(
        {
            "plan_proposal": proposal.model_dump(mode="json"),
            "verification_report": _verification_report("run-agentic-ui").model_dump(mode="json"),
            "run_plan_revision": revision.model_dump(mode="json"),
        }
    )

    assert card["run_id"] == "run-agentic-ui"
    assert card["requires_confirmation"] is True
    assert card["sections"]["plan_explanation"]["rationales"][0]["task_id"] == "train_model"
    assert card["sections"]["task_timeline"][0]["task_id"] == "train_model"
    assert card["sections"]["missing_information"] == []
    assert card["sections"]["memory_use"][0]["record_id"] == "backend-rf"
    assert card["sections"]["verifier_findings"][0]["category"] == "abnormal_model_metrics"
    assert card["sections"]["replan_compare"]["added_tasks"] == ["train_model"]
    assert card["sections"]["autonomy_metrics"]["tasks_selected_by_agent"] == 1
    assert card["sections"]["autonomy_metrics"]["replans_proposed"] == 1

    controls = card["approval_controls"]
    assert any(control["target_type"] == "plan" and control["action"] == "confirm_plan" for control in controls)
    assert any(control["target_type"] == "gate" and control["target_id"] == "gate_3_train_config" for control in controls)
    assert any(control["target_type"] == "replan" and control["action"] == "confirm_replan" for control in controls)
    assert any(control["target_type"] == "memory" and control["action"] == "approve_memory_use" for control in controls)
    assert all(control["target_type"] in {"plan", "task", "gate", "permission", "memory", "replan"} for control in controls)


def test_agent_review_card_rejects_present_malformed_artifact() -> None:
    with pytest.raises(ValueError, match="plan_proposal must be an object"):
        build_agent_review_card({"plan_proposal": "not-an-object"})
