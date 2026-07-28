from ai4s_agent.agents.report import ReportAgent
from ai4s_agent.schemas import (
    ReportSynthesisProposal,
    VerificationFinding,
    VerificationReport,
)
from ai4s_agent.storage import ProjectStorage


def _verification_report() -> VerificationReport:
    return VerificationReport(
        project_id="proj-report",
        run_id="run-report",
        generated_at="2026-06-05T10:00:00Z",
        overall_decision="replan",
        findings=[
            VerificationFinding(
                finding_id="weak_model_metrics_1",
                category="abnormal_model_metrics",
                severity="warning",
                message="Model R2 is negative.",
                decision="replan",
                evidence={"property_id": "plqy"},
            )
        ],
        summary="Verifier recommends replanning before downstream screening.",
    )


def test_report_agent_synthesizes_run_state_limitations_and_next_steps() -> None:
    proposal = ReportAgent().synthesize_run(
        run_id="run-report",
        goal="Summarize this AI4S run for review.",
        observation={
            "stage_state": {"stage": "train_model", "status": "FAILED"},
            "artifacts": [{"artifact_id": "verification_report_json"}],
            "notes": ["No stage.json found for an older sub-run."],
        },
        verification_report=_verification_report(),
        modeling_proposal={
            "status": "needs_confirmation",
            "retry_proposals": [{"action": "adjust_split", "reason": "Weak metrics."}],
        },
        generation_proposal={
            "status": "needs_clarification",
            "required_permissions": ["generate_candidates_expensive"],
            "questions": [{"question_id": "q_generation_expensive_confirmation", "blocks_execution": True}],
        },
    )

    assert proposal.run_id == "run-report"
    assert proposal.status == "needs_clarification"
    assert proposal.executable is False
    assert "Verifier recommends" in proposal.executive_summary
    assert any(section.title == "Verification" for section in proposal.sections)
    assert any("abnormal_model_metrics" in limitation for limitation in proposal.limitations)
    assert any(step.action == "propose_replan" for step in proposal.next_steps)
    assert any(step.action == "confirm_generation_budget" for step in proposal.next_steps)

    restored = ReportSynthesisProposal.model_validate_json(proposal.model_dump_json())
    assert restored.model_dump(mode="json") == proposal.model_dump(mode="json")


def test_report_agent_writes_report_synthesis_artifact(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    agent = ReportAgent()
    proposal = agent.synthesize_run(
        run_id="run-report-artifact",
        goal="Write a paper-style audit summary.",
        verification_report=_verification_report(),
    )

    json_path, md_path = agent.write_proposal(storage, "proj-report", "run-report-artifact", proposal)

    assert json_path.name == "report_synthesis_proposal.json"
    assert md_path.name == "report_synthesis_proposal.md"
    assert json_path.exists()
    assert md_path.exists()
    registry = storage.read_artifact_registry("proj-report", "run-report-artifact")
    assert registry["report_synthesis_proposal_json"] == "report_synthesis_proposal.json"
    assert registry["report_synthesis_proposal_md"] == "report_synthesis_proposal.md"
