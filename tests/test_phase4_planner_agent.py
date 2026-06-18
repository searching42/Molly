from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.schemas import AgentPlanProposal, PlanQuestion, PlanRationale, ProjectMemoryRecord


def test_agent_plan_proposal_schema_roundtrip() -> None:
    rationale = PlanRationale(
        task_id="retrieve_evidence",
        reason="Need evidence before extracting literature records.",
        risk_level="low",
        required_gates=[],
    )
    question = PlanQuestion(
        question_id="q_dataset",
        prompt="Which dataset should be used?",
        reason="Training requires a confirmed dataset.",
        choices=["upload_dataset", "use_existing_asset"],
        blocks_execution=True,
    )
    proposal = AgentPlanProposal(
        run_id="r-phase4",
        goal="Build a literature-derived training dataset.",
        planner_backend="rule_based",
        status="needs_confirmation",
        run_plan={
            "run_id": "r-phase4",
            "requested_tasks": ["retrieve_evidence"],
            "tasks": [],
            "available_artifacts": [],
            "missing_artifacts": [],
        },
        rationales=[rationale],
        assumptions=["No adapters are executed during proposal generation."],
        questions=[question],
        required_gates=["gate_2_data_mining"],
        executable=False,
    )

    restored = AgentPlanProposal.model_validate_json(proposal.model_dump_json())
    assert restored.model_dump(mode="json") == proposal.model_dump(mode="json")


def test_planner_agent_proposes_literature_to_dataset_dry_run_plan() -> None:
    proposal = PlannerAgent().propose_plan(
        run_id="r-lit-agent",
        goal="Mine OLED papers and build a confirmed training dataset from literature evidence.",
        available_artifacts=["pdf_corpus"],
    )

    assert proposal.status == "needs_confirmation"
    assert proposal.executable is False
    assert proposal.planner_backend == "rule_based"
    assert proposal.run_plan.requested_tasks == ["literature_to_dataset_workflow"]
    task_ids = [task.task_id for task in proposal.run_plan.tasks]
    assert "literature_to_dataset_workflow" in task_ids
    assert "pdf_corpus" not in proposal.run_plan.missing_artifacts
    assert any(r.task_id == "literature_to_dataset_workflow" for r in proposal.rationales)
    assert proposal.rationales[0].required_gates == ["gate_2_data_mining"]
    assert any("No adapters are executed" in item for item in proposal.assumptions)
    assert "gate_2_data_mining" in proposal.required_gates


def test_planner_agent_asks_question_for_underspecified_goal() -> None:
    proposal = PlannerAgent().propose_plan(run_id="r-unclear", goal="Help me improve materials.")

    assert proposal.status == "needs_clarification"
    assert proposal.executable is False
    assert proposal.run_plan.tasks == []
    assert proposal.questions
    assert proposal.questions[0].blocks_execution is True


def test_planner_agent_surfaces_project_memory_without_hiding_assumptions() -> None:
    memory_records = [
        ProjectMemoryRecord(
            record_id="backend-rf",
            category="backend_choice",
            summary="Use random forest for small OLED datasets.",
            value={"backend": "random_forest"},
            source_refs=["run:baseline"],
            decision="confirmed_backend_choice",
            confirmed_by="user",
        ),
        ProjectMemoryRecord(
            record_id="alias-plqy",
            category="property_alias",
            summary="Treat PLQY and quantum yield as plqy.",
            value={"aliases": {"PLQY": "plqy", "quantum yield": "plqy"}},
            decision="confirmed_property_alias",
            confirmed_by="user",
        ),
    ]

    proposal = PlannerAgent(memory_records=memory_records).propose_plan(
        run_id="r-memory",
        goal="Train a model for PLQY.",
        available_artifacts=["cleaned_train_dataset", "trainability_report"],
    )

    assert proposal.run_plan.requested_tasks == ["train_model"]
    assert {item.record_id for item in proposal.memory_references} == {"backend-rf", "alias-plqy"}
    assert all(item.reason for item in proposal.memory_references)
    assert any("Project memory used" in assumption for assumption in proposal.assumptions)


def test_planner_agent_does_not_apply_unrelated_property_alias_memory() -> None:
    alias_record = ProjectMemoryRecord(
        record_id="alias-homo",
        category="property_alias",
        summary="Treat HOMO as homo_ev.",
        value={"aliases": {"HOMO": "homo_ev"}},
        decision="confirmed_property_alias",
        confirmed_by="user",
    )

    proposal = PlannerAgent(memory_records=[alias_record]).propose_plan(
        run_id="r-unrelated-memory",
        goal="Mine OLED papers from PDFs.",
        available_artifacts=["pdf_corpus"],
    )

    assert proposal.memory_references == []
