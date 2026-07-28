from ai4s_agent.agents.recovery import RecoveryAgent
from ai4s_agent.planner import expand_run_plan
import pytest
from pydantic import ValidationError

from ai4s_agent.storage import ProjectStorage
from ai4s_agent.schemas import ReplanRequest, RunPlanRevision, VerificationFinding, VerificationReport


def _finding(category: str, *, decision: str = "replan") -> VerificationFinding:
    return VerificationFinding(
        finding_id=f"{category}_1",
        category=category,
        severity="warning",
        decision=decision,
        message=f"{category} occurred",
        evidence={},
    )


def _report(run_id: str, *categories: str) -> VerificationReport:
    return VerificationReport(
        project_id="proj-recovery",
        run_id=run_id,
        generated_at="2026-06-04T10:00:00Z",
        overall_decision="replan",
        findings=[_finding(category) for category in categories],
    )


def test_replan_request_and_run_plan_revision_schema_roundtrip() -> None:
    previous = expand_run_plan(
        run_id="run-recovery-schema",
        requested_tasks=["run_baseline"],
        available_artifacts=["trainability_report"],
    )
    revised = expand_run_plan(
        run_id="run-recovery-schema",
        requested_tasks=["check_trainability", "run_baseline"],
        available_artifacts=["cleaned_train_dataset", "property_catalog"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-recovery-schema",
        trigger="failure",
        reason="Training data failed trainability checks.",
        failed_stage="train_model",
        available_artifacts=["cleaned_train_dataset", "property_catalog"],
        new_constraints=["run baseline only until more data is available"],
    )
    revision = RunPlanRevision(
        revision_id="rev-run-recovery-schema",
        project_id=request.project_id,
        run_id=request.run_id,
        created_at="2026-06-04T10:01:00Z",
        previous_plan=previous,
        revised_plan=revised,
        diff={"added_tasks": ["check_trainability"], "unchanged_tasks": ["run_baseline"]},
        reason=request.reason,
        recovery_actions=["run_baseline_only"],
        approvals_required=[],
    )

    assert ReplanRequest.model_validate_json(request.model_dump_json()).model_dump(mode="json") == request.model_dump(
        mode="json"
    )
    assert RunPlanRevision.model_validate_json(revision.model_dump_json()).model_dump(
        mode="json"
    ) == revision.model_dump(mode="json")


def test_replan_request_rejects_blank_run_id() -> None:
    with pytest.raises(ValidationError):
        ReplanRequest(run_id="   ", trigger="failure", reason="broken run id")


def test_recovery_agent_replaces_mineru_parse_with_parser_fallbacks() -> None:
    previous = expand_run_plan(
        run_id="run-parser-recovery",
        requested_tasks=["parse_document", "index_corpus", "retrieve_evidence", "extract_records"],
        available_artifacts=["pdf_corpus"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-parser-recovery",
        trigger="degraded_output",
        failed_stage="parse_document",
        reason="MinerU output has no extractable tables.",
    )

    revision = RecoveryAgent().propose_revision(
        request=request,
        previous_plan=previous,
        verification_report=_report("run-parser-recovery", "empty_extraction", "low_confidence"),
    )

    assert "parse_document" not in revision.revised_plan.requested_tasks
    task_ids = [task.task_id for task in revision.revised_plan.tasks]
    assert "parse_document" not in task_ids
    assert "parse_document_pdfplumber" in revision.revised_plan.requested_tasks
    assert "parse_document_pymupdf" not in revision.revised_plan.requested_tasks
    assert "parse_document_grobid" not in revision.revised_plan.requested_tasks
    index_task = next(task for task in revision.revised_plan.tasks if task.task_id == "index_corpus")
    assert index_task.depends_on == ["parse_document_pdfplumber"]
    assert "parser_fallback" in revision.recovery_actions
    assert revision.executable is False


def test_recovery_agent_selects_one_parser_fallback_to_avoid_output_overwrite() -> None:
    previous = expand_run_plan(
        run_id="run-parser-single-fallback",
        requested_tasks=["parse_document", "index_corpus", "retrieve_evidence"],
        available_artifacts=["pdf_corpus"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-parser-single-fallback",
        trigger="degraded_output",
        failed_stage="parse_document",
        reason="MinerU output has no extractable tables.",
    )

    revision = RecoveryAgent().propose_revision(request=request, previous_plan=previous)

    fallback_tasks = [
        task_id
        for task_id in revision.revised_plan.requested_tasks
        if task_id in {"parse_document_pdfplumber", "parse_document_pymupdf", "parse_document_grobid"}
    ]
    assert fallback_tasks == ["parse_document_pdfplumber"]


def test_recovery_agent_does_not_trigger_parser_fallback_from_sparse_text() -> None:
    previous = expand_run_plan(
        run_id="run-sparse-modeling-failure",
        requested_tasks=["train_model"],
        available_artifacts=["cleaned_train_dataset", "trainability_report"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-sparse-modeling-failure",
        trigger="failure",
        failed_stage="train_model",
        reason="Sparse labels caused unstable training.",
    )

    revision = RecoveryAgent().propose_revision(request=request, previous_plan=previous)

    assert "parser_fallback" not in revision.recovery_actions
    assert all(not task_id.startswith("parse_document_") for task_id in revision.revised_plan.requested_tasks)


def test_recovery_agent_replaces_monolithic_literature_workflow_with_fallback_pipeline() -> None:
    previous = expand_run_plan(
        run_id="run-workflow-parser-recovery",
        requested_tasks=["literature_to_dataset_workflow"],
        available_artifacts=["pdf_corpus"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-workflow-parser-recovery",
        trigger="degraded_output",
        failed_stage="parse_document",
        reason="Parser output has no extractable records.",
    )

    revision = RecoveryAgent().propose_revision(
        request=request,
        previous_plan=previous,
        verification_report=_report("run-workflow-parser-recovery", "empty_extraction"),
    )

    task_ids = [task.task_id for task in revision.revised_plan.tasks]
    assert "literature_to_dataset_workflow" not in task_ids
    assert "parse_document" not in task_ids
    assert "parse_document_pdfplumber" in revision.revised_plan.requested_tasks
    assert "merge_extracted_records" in revision.revised_plan.requested_tasks
    index_task = next(task for task in revision.revised_plan.tasks if task.task_id == "index_corpus")
    assert index_task.depends_on == ["parse_document_pdfplumber"]


def test_recovery_agent_falls_back_for_plain_parser_failure_without_report_categories() -> None:
    previous = expand_run_plan(
        run_id="run-parser-timeout",
        requested_tasks=["parse_document", "index_corpus"],
        available_artifacts=["pdf_corpus"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-parser-timeout",
        trigger="failure",
        failed_stage="parse_document",
        reason="Parser timed out.",
    )

    revision = RecoveryAgent().propose_revision(request=request, previous_plan=previous)

    assert "parser_fallback" in revision.recovery_actions
    assert "parse_document" not in revision.revised_plan.requested_tasks
    assert "parse_document_pdfplumber" in revision.revised_plan.requested_tasks


def test_recovery_agent_uses_request_available_artifacts_as_trusted_replan_inputs() -> None:
    previous = expand_run_plan(
        run_id="run-available-recovery",
        requested_tasks=["parse_document", "index_corpus", "retrieve_evidence"],
        available_artifacts=["pdf_corpus"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-available-recovery",
        trigger="changed_artifacts",
        reason="A confirmed corpus index is now available.",
        available_artifacts=["corpus_index"],
    )

    revision = RecoveryAgent().propose_revision(
        request=request,
        previous_plan=previous,
        verification_report=_report("run-available-recovery", "missing_provenance"),
    )

    retrieve_task = next(task for task in revision.revised_plan.tasks if task.task_id == "retrieve_evidence")
    assert retrieve_task.depends_on == []
    assert "parse_document" not in [task.task_id for task in revision.revised_plan.tasks]


def test_recovery_agent_requires_approval_for_data_mining_network_expansion() -> None:
    previous = expand_run_plan(
        run_id="run-mining-recovery",
        requested_tasks=["retrieve_evidence", "extract_records"],
        available_artifacts=["corpus_index", "evidence_chunks"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-mining-recovery",
        trigger="failure",
        failed_stage="retrieve_evidence",
        reason="No evidence hits were found for the requested property.",
    )

    revision = RecoveryAgent().propose_revision(
        request=request,
        previous_plan=previous,
        verification_report=_report("run-mining-recovery", "empty_extraction", "missing_provenance"),
    )

    assert "prepare_literature_corpus_sources" in revision.revised_plan.requested_tasks
    assert "acquire_literature_sources" in revision.revised_plan.requested_tasks
    assert "expand_literature_query" in revision.recovery_actions
    assert revision.external_network_added is True
    assert revision.user_approval_required is True
    assert "external_network_action" in revision.approvals_required
    assert revision.questions[0].question_id == "q_literature_sources"


def test_recovery_agent_makes_high_risk_modeling_downgrade_explicit() -> None:
    previous = expand_run_plan(
        run_id="run-model-downgrade",
        requested_tasks=["train_model"],
        available_artifacts=["cleaned_train_dataset", "trainability_report"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-model-downgrade",
        trigger="degraded_output",
        failed_stage="train_model",
        reason="Training data is not ready for model training.",
    )

    revision = RecoveryAgent().propose_revision(
        request=request,
        previous_plan=previous,
        verification_report=_report("run-model-downgrade", "poor_trainability"),
    )

    assert "train_model" not in revision.revised_plan.requested_tasks
    assert "run_baseline" in revision.revised_plan.requested_tasks
    assert "run_baseline_only" in revision.recovery_actions
    assert "train_model" in revision.removed_high_risk_tasks
    assert "high_risk_downgrade_review" in revision.approvals_required
    assert revision.user_approval_required is True


def test_recovery_agent_requires_approval_when_replan_adds_high_risk_training() -> None:
    previous = expand_run_plan(
        run_id="run-model-retrain",
        requested_tasks=["run_baseline"],
        available_artifacts=["cleaned_train_dataset", "property_catalog", "trainability_report"],
    )
    request = ReplanRequest(
        project_id="proj-recovery",
        run_id="run-model-retrain",
        trigger="new_user_constraints",
        reason="User asked to retrain with a different backend after abnormal metrics.",
        new_constraints=["switch backend", "retrain model"],
    )

    revision = RecoveryAgent().propose_revision(
        request=request,
        previous_plan=previous,
        verification_report=_report("run-model-retrain", "abnormal_model_metrics"),
    )

    assert "train_model" in revision.diff.added_tasks
    assert revision.high_risk_added is True
    assert "gate_3_train_config" in revision.approvals_required
    assert revision.user_approval_required is True


def test_recovery_agent_writes_revision_history_without_overwriting(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    agent = RecoveryAgent()
    previous = expand_run_plan(
        run_id="run-revision-history",
        requested_tasks=["run_baseline"],
        available_artifacts=["trainability_report"],
    )
    first = agent.propose_revision(
        request=ReplanRequest(
            project_id="proj-recovery",
            run_id="run-revision-history",
            trigger="failure",
            reason="First recovery attempt.",
        ),
        previous_plan=previous,
        verification_report=_report("run-revision-history", "poor_trainability"),
    )
    second = agent.propose_revision(
        request=ReplanRequest(
            project_id="proj-recovery",
            run_id="run-revision-history",
            trigger="new_user_constraints",
            reason="Second recovery attempt.",
            new_constraints=["switch backend", "retrain model"],
        ),
        previous_plan=previous,
        verification_report=_report("run-revision-history", "abnormal_model_metrics"),
    )

    first_json, _ = agent.write_revision(storage, "proj-recovery", "run-revision-history", first)
    second_json, _ = agent.write_revision(storage, "proj-recovery", "run-revision-history", second)

    assert first_json != second_json
    assert first_json.exists()
    assert second_json.exists()
    index = (storage.run_dir("proj-recovery", "run-revision-history") / "run_plan_revisions.json").read_text(
        encoding="utf-8"
    )
    assert first.revision_id in index
    assert second.revision_id in index
    registry = storage.read_artifact_registry("proj-recovery", "run-revision-history")
    assert registry["run_plan_revision_json"] == second_json.name
    assert registry["run_plan_revisions_index"] == "run_plan_revisions.json"
