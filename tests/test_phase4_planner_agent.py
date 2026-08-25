from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.planner import DEFAULT_ATOMIC_TASKS, br2_contextual_mapping_task_registry_v1
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


def test_planner_agent_proposes_review_only_oled_candidate_chain() -> None:
    proposal = PlannerAgent(registry=br2_contextual_mapping_task_registry_v1()).propose_plan(
        run_id="r-br2-review",
        goal="Parse this OLED paper, extract the evidence, and organize a candidate raw dataset for me to review.",
        available_artifacts=["pdf_corpus"],
    )

    assert proposal.status == "needs_confirmation"
    assert proposal.run_plan.requested_tasks == ["prepare_oled_candidate_raw_dataset"]
    assert [task.task_id for task in proposal.run_plan.tasks] == [
        "parse_document",
        "extract_oled_evidence",
        "map_oled_contextual_semantics",
        "prepare_oled_candidate_raw_dataset",
    ]
    assert proposal.required_gates == ["gate_2_data_mining"]


def test_planner_agent_routes_plain_oled_organize_request_to_review_chain() -> None:
    proposal = PlannerAgent(registry=br2_contextual_mapping_task_registry_v1()).propose_plan(
        run_id="r-br2-plain-review",
        goal="帮我从这篇 OLED 文献中整理可用于后续建模的数据。",
        available_artifacts=["pdf_corpus"],
    )

    assert proposal.run_plan.requested_tasks == ["prepare_oled_candidate_raw_dataset"]
    assert "train_model" not in [task.task_id for task in proposal.run_plan.tasks]


def test_br2_registry_only_preapproves_existing_parse_gate() -> None:
    default_parse = next(task for task in DEFAULT_ATOMIC_TASKS if task.task_id == "parse_document")
    br2_parse = br2_contextual_mapping_task_registry_v1().get("parse_document")

    assert default_parse.supports_plan_preapproval is False
    assert br2_parse.supports_plan_preapproval is True
    assert br2_parse.gates == default_parse.gates
    assert br2_parse.default_adapter == default_parse.default_adapter


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
