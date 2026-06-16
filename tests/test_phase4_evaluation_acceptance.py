from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.agents.evaluation import compute_autonomy_metrics
from ai4s_agent.agents.modeling import ModelingAgent
from ai4s_agent.agents.observer import ObserverAgent
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.agents.recovery import RecoveryAgent
from ai4s_agent.agents.research import ResearchAgent
from ai4s_agent.agents.verifier import VerifierAgent
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import ReplanRequest, RunStatus, StageState, VerificationFinding, VerificationReport
from ai4s_agent.storage import ProjectStorage


def test_observe_verify_replan_loop_for_parser_failure(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-loop"
    run_id = "run-loop"
    run_dir = storage.run_dir(project_id, run_id)
    storage.write_stage_state(
        project_id,
        run_id,
        StageState(
            stage="parse_document",
            status=RunStatus.FAILED,
            started_at="2026-06-05T10:00:00Z",
            updated_at="2026-06-05T10:01:00Z",
            ended_at="2026-06-05T10:01:00Z",
            error={"retryable": True, "category": "parser_failed"},
        ),
    )
    write_json(
        run_dir / "extraction_confidence_report.json",
        {
            "attempted_hit_count": 12,
            "extracted_record_count": 0,
            "high_confidence_count": 0,
            "low_confidence_count": 5,
        },
    )

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent().verify(observation)
    previous = expand_run_plan(
        run_id=run_id,
        requested_tasks=["parse_document", "index_corpus", "retrieve_evidence", "extract_records"],
        available_artifacts=["pdf_corpus"],
    )
    revision = RecoveryAgent().propose_revision(
        request=ReplanRequest(
            project_id=project_id,
            run_id=run_id,
            trigger="failure",
            failed_stage="parse_document",
            reason="Primary parser produced no extractable records.",
        ),
        previous_plan=previous,
        verification_report=report,
    )

    assert report.overall_decision in {"retry", "ask_user"}
    assert "empty_extraction" in {finding.category for finding in report.findings}
    assert "parse_document_pdfplumber" in revision.revised_plan.requested_tasks
    assert revision.executable is False


def test_acceptance_broad_research_goal_produces_safe_dry_run_plan() -> None:
    proposal = PlannerAgent().propose_plan(
        run_id="run-broad-research",
        goal="Find OLED papers and build a dataset for PLQY screening.",
    )
    research = ResearchAgent().propose_sources(
        run_id="run-broad-research",
        goal="Find OLED papers and build a dataset for PLQY screening.",
    )

    assert proposal.executable is False
    assert "literature_to_dataset_workflow" in proposal.run_plan.requested_tasks
    assert proposal.status in {"needs_confirmation", "needs_clarification"}
    assert research.executable is False
    assert research.status == "needs_clarification"
    assert research.questions[0].blocks_execution is True


def test_acceptance_high_conflict_extraction_asks_for_human_review(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    project_id = "proj-conflict-acceptance"
    run_id = "run-conflict-acceptance"
    run_dir = storage.run_dir(project_id, run_id)
    write_json(
        run_dir / "conflict_report.json",
        {
            "input_record_count": 10,
            "merged_record_count": 6,
            "conflict_count": 4,
            "non_conflicting_record_count": 6,
            "conflicts": [],
            "generated_at": now_iso(),
        },
    )

    observation = ObserverAgent(storage=storage).observe_run(project_id, run_id)
    report = VerifierAgent(conflict_rate_threshold=0.25).verify(observation)

    assert report.overall_decision == "ask_user"
    assert "high_conflict_rate" in {finding.category for finding in report.findings}


def test_acceptance_weak_model_metrics_propose_more_data_or_backend_change() -> None:
    proposal = ModelingAgent().propose_modeling_plan(
        run_id="run-weak-model",
        goal="Train a reliable model for PLQY.",
        trainability_report={
            "overall_status": "READY",
            "properties": [
                {
                    "property_id": "plqy",
                    "effective_labels": 42,
                    "status": "TRAIN_READY",
                }
            ],
        },
        model_metrics={"properties": [{"property_id": "plqy", "metrics": {"r2": -0.3}}]},
    )

    actions = {item.action for item in proposal.retry_proposals}
    assert proposal.metric_interpretations[0].status == "weak"
    assert actions >= {"adjust_split", "switch_backend_or_features"}


def test_autonomy_metrics_count_agent_decisions_and_safety_events() -> None:
    plan = PlannerAgent().propose_plan(
        run_id="run-autonomy",
        goal="Train a model for PLQY.",
        available_artifacts=["cleaned_train_dataset", "property_catalog", "trainability_report"],
    )
    report = VerificationReport(
        project_id="proj-autonomy",
        run_id="run-autonomy",
        generated_at=now_iso(),
        overall_decision="replan",
        findings=[
            VerificationFinding(
                finding_id="metric_1",
                category="abnormal_model_metrics",
                severity="warning",
                decision="replan",
                message="Model metric is weak.",
                evidence={},
            )
        ],
    )
    previous = expand_run_plan(
        run_id="run-autonomy",
        requested_tasks=["run_baseline"],
        available_artifacts=["cleaned_train_dataset", "property_catalog", "trainability_report"],
    )
    revision = RecoveryAgent().propose_revision(
        request=ReplanRequest(
            project_id="proj-autonomy",
            run_id="run-autonomy",
            trigger="new_user_constraints",
            reason="User requested retraining after weak metrics.",
            new_constraints=["switch backend", "retrain model"],
        ),
        previous_plan=previous,
        verification_report=report,
    )

    metrics = compute_autonomy_metrics(
        {
            "plan_proposal": plan.model_dump(mode="json"),
            "verification_report": report.model_dump(mode="json"),
            "run_plan_revision": revision.model_dump(mode="json"),
        }
    )

    assert metrics["tasks_selected_by_agent"] == 1
    assert metrics["replans_proposed"] == 1
    assert metrics["user_confirmations_required"] >= 2
    assert metrics["verifier_catches"] == 1
    assert metrics["failed_autonomous_decisions"] == 0
