from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.memory import ProjectMemory
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_artifact_verifier import RunPlanArtifactFinding, RunPlanArtifactVerification
from ai4s_agent.run_plan_replan_application_artifacts import proposal_artifact_hash
from ai4s_agent.run_plan_resume_intent_validation import ResumeIntentValidationResult
from ai4s_agent.run_plan_resume_intent_validation_audit_memory import RESUME_INTENT_VALIDATION_AUDIT_REF
from ai4s_agent.run_plan_review_artifacts import write_run_plan_review_artifacts
from ai4s_agent.run_plan_review_card import read_run_plan_review_card
from ai4s_agent.run_plan_review_memory import save_run_plan_review_card_summary_to_memory
from ai4s_agent.schemas import GateName, RunStatus
from ai4s_agent.server_permissions import ServerPermissionStore
from ai4s_agent.storage import ProjectStorage


PROJECT_ID = "proj-resume-loop"
RUN_ID = "run-resume-loop"


def _write_training_csv(path: Path) -> None:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _prepare_waiting_run(workspace: Path) -> dict[str, Any]:
    storage = ProjectStorage(workspace)
    dataset = workspace / "input" / "train.csv"
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id=RUN_ID, requested_tasks=["train_model"], available_artifacts=[])
    first = RunPlanExecutor(storage=storage).execute(
        project_id=PROJECT_ID,
        run_plan=run_plan,
        input_artifacts={"uploaded_dataset": str(dataset)},
    )
    assert first["status"] == RunStatus.WAITING_USER.value
    assert first["waiting_task"] == "train_model"
    assert first["required_gates"] == [GateName.TRAIN_CONFIG.value]
    stage_state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    assert stage_state is not None
    assert stage_state.status == RunStatus.WAITING_USER
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    (run_dir / "run_plan.json").write_text(
        json.dumps(run_plan.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "run_plan": run_plan,
        "stage_state": stage_state,
        "first_execution": first,
    }


def _write_waiting_review_artifacts(workspace: Path, *, waiting_execution: dict[str, Any]) -> dict[str, Any]:
    verification = RunPlanArtifactVerification(
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        generated_at="2026-01-01T00:00:00+00:00",
        decision="rerun_recommended",
        summary="Run is waiting for user approval at the train_model gate.",
        findings=[
            RunPlanArtifactFinding(
                finding_id="waiting_train_gate",
                category="waiting_user",
                severity="warning",
                decision="rerun_recommended",
                message="The run is waiting for train_model gate approval.",
                evidence={
                    "waiting_task": "train_model",
                    "required_gates": [GateName.TRAIN_CONFIG.value],
                    "terminal_record": waiting_execution,
                },
            )
        ],
        observed={"stage": {"status": RunStatus.WAITING_USER.value}},
    )
    bundle = write_run_plan_review_artifacts(
        workspace_dir=workspace,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        verification=verification,
    )
    operation_ids = [
        operation["operation_id"]
        for operation in bundle.proposal.proposed_run_plan_patch["operations"]
    ]
    return {
        "bundle": bundle,
        "proposal_path": ProjectStorage(workspace).run_dir(PROJECT_ID, RUN_ID) / "review" / "replan_proposal.json",
        "operation_ids": operation_ids,
    }


def _grant(workspace: Path, action: str) -> dict[str, Any]:
    return ServerPermissionStore(workspace).create_grant(
        PROJECT_ID,
        action,
        actor="admin",
        actor_source="test",
        run_id=RUN_ID,
        reason=f"test grant for {action}",
    )


def _grant_resume_loop_permissions(workspace: Path) -> None:
    _grant(workspace, "run_plan_replan_apply")
    _grant(workspace, "run_plan_resume_intent_use")
    _grant(workspace, "run_plan_resume_execute")


def _enable_internal_routes(app) -> None:
    app.config["AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"] = True
    app.config["AI4S_ENABLE_INTERNAL_RESUME_INTENT_VALIDATION_ROUTE"] = True
    app.config["AI4S_ENABLE_INTERNAL_RESUME_INTENT_EXECUTE_ROUTE"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = lambda storage: (_ for _ in ()).throw(
        AssertionError("run-plan queue executor factory must not be used by resume loop e2e")
    )


def _audit_records(workspace: Path) -> list[dict[str, Any]]:
    path = ProjectStorage(workspace).run_dir(PROJECT_ID, RUN_ID) / RESUME_INTENT_VALIDATION_AUDIT_REF
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _post_apply_review(client, review: dict[str, Any]):
    proposal_path = review["proposal_path"]
    return client.post(
        "/api/internal/run-plan/replan/apply-review",
        json={
            "project_id": PROJECT_ID,
            "run_id": RUN_ID,
            "proposal_artifact_ref": "review/replan_proposal.json",
            "proposal_hash": proposal_artifact_hash(proposal_path),
            "selected_action": "rerun_task",
            "selected_operation_ids": review["operation_ids"],
            "reason": "User confirmed rerun after review.",
        },
        headers={"X-Actor": "review-user"},
    )


def _post_validate_resume(client):
    return client.post(
        "/api/internal/run-plan/resume-intent/validate",
        json={
            "project_id": PROJECT_ID,
            "run_id": RUN_ID,
            "approved_gates": [GateName.TRAIN_CONFIG.value],
        },
        headers={"X-Actor": "validator-user"},
    )


def _post_execute_resume(client):
    return client.post(
        "/api/internal/run-plan/resume-intent/execute",
        json={
            "project_id": PROJECT_ID,
            "run_id": RUN_ID,
            "approved_gates": [GateName.TRAIN_CONFIG.value],
        },
        headers={"X-Actor": "resume-user"},
    )


def test_user_confirmed_resume_loop_end_to_end_review_after_resume(tmp_path: Path) -> None:
    waiting = _prepare_waiting_run(tmp_path)
    review = _write_waiting_review_artifacts(tmp_path, waiting_execution=waiting["first_execution"])
    _grant_resume_loop_permissions(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_internal_routes(app)
    client = app.test_client()

    default_execute_response = client.post("/api/run-plan/execute", json={})
    default_resume_response = client.post("/api/run-plan/resume", json={})
    assert default_execute_response.status_code != 404
    assert default_resume_response.status_code != 404

    apply_response = _post_apply_review(client, review)

    assert apply_response.status_code == 200
    apply_payload = apply_response.get_json()
    assert apply_payload["ok"] is True
    assert apply_payload["application"]["result_type"] == "resume_intent"
    assert apply_payload["application"]["selected_operation_ids"] == review["operation_ids"]
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()

    validate_response = _post_validate_resume(client)

    assert validate_response.status_code == 200
    validate_payload = validate_response.get_json()
    validation = ResumeIntentValidationResult.model_validate(validate_payload["validation"])
    assert validation.decision == "resume_eligible"
    assert validation.required_gates == [GateName.TRAIN_CONFIG.value]
    assert validation.approved_gates == [GateName.TRAIN_CONFIG.value]
    assert validation.rerun_tasks == ["train_model"]
    assert validation.resume_from_task == "train_model"

    resume_response = _post_execute_resume(client)

    assert resume_response.status_code == 200
    resume_payload = resume_response.get_json()
    assert resume_payload["ok"] is True
    assert resume_payload["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert resume_payload["execution"]["executed_tasks"][-1] == "train_model"
    storage = ProjectStorage(tmp_path)
    stage_state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    assert stage_state is not None
    assert stage_state.status == RunStatus.SUCCEEDED
    gate_decisions = storage.read_gate_decisions(PROJECT_ID, RUN_ID)
    assert len(gate_decisions) == 1
    assert gate_decisions[0]["gate"] == GateName.TRAIN_CONFIG.value
    assert gate_decisions[0]["actor"] == "resume-user"
    assert gate_decisions[0]["approved_snapshot_id"]
    assert gate_decisions[0]["approved_snapshot_hash"]

    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    for artifact_id in [
        "trained_model",
        "model_metadata",
        "model_manifest",
        "domain_model_manifest",
        "model_diagnostics_report",
        "model_package_review",
        "trainability_report",
    ]:
        relative_path = registry[artifact_id]
        assert (storage.run_dir(PROJECT_ID, RUN_ID) / relative_path).exists()

    second_resume = _post_execute_resume(client)

    assert second_resume.status_code == 409
    second_payload = second_resume.get_json()
    assert second_payload["ok"] is False
    second_validation = ResumeIntentValidationResult.model_validate(second_payload["validation"])
    assert second_validation.decision == "stale_intent"
    assert second_validation.error is not None
    assert second_validation.error["type"] == "resume_intent_already_consumed"
    assert len(storage.read_gate_decisions(PROJECT_ID, RUN_ID)) == 1

    post_resume_review = write_run_plan_review_artifacts(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
    )
    observed_artifacts = post_resume_review.verification.observed["artifacts"]["artifacts"]
    assert observed_artifacts["trained_model"]["exists"] is True
    assert observed_artifacts["model_manifest"]["exists"] is True
    assert observed_artifacts["model_diagnostics_report"]["exists"] is True

    review_card = read_run_plan_review_card(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
    )
    assert review_card.executable is False
    assert review_card.artifacts["observer_verification"] == "review/observer_verification.json"
    memory_save = save_run_plan_review_card_summary_to_memory(
        workspace_dir=tmp_path,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        card=review_card,
        confirmed_by="review-user",
    )
    assert memory_save.saved is True
    memory_records = ProjectMemory(tmp_path).list_project_records(PROJECT_ID)
    categories = {record.category for record in memory_records}
    assert "run_plan_replan_application" in categories
    assert "run_plan_resume_intent_validation" in categories
    assert "run_plan_review" in categories

    audit_events = [record["event"] for record in _audit_records(tmp_path)]
    assert audit_events.count("resume_intent_consumed") == 1
    assert "run_plan_resume_completed" in audit_events
    assert "run_plan_resume_failed" in audit_events
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()
