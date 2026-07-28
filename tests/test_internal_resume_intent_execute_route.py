from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.memory import ProjectMemory
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_replan_application import ReplanApplicationRequest
from ai4s_agent.run_plan_replan_application_artifacts import (
    proposal_artifact_hash,
    write_replan_application_artifacts,
)
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.run_plan_resume_intent_validation import ResumeIntentValidationResult
from ai4s_agent.run_plan_resume_intent_validation_audit_memory import RESUME_INTENT_VALIDATION_AUDIT_REF
from ai4s_agent.schemas import GateName, RunStatus
from ai4s_agent.server_permissions import ServerPermissionStore
from ai4s_agent.storage import ProjectStorage


PROJECT_ID = "proj-resume-execute"
RUN_ID = "run-resume-execute"
PERMISSION_ACTION = "run_plan_resume_execute"


def _write_training_csv(path: Path) -> None:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _proposal_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _prepare_waiting_resume_intent(workspace: Path) -> dict[str, Any]:
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
    stage_state = storage.read_stage_state(PROJECT_ID, RUN_ID)
    assert stage_state is not None
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    (run_dir / "run_plan.json").write_text(
        json.dumps(run_plan.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    proposal = RunPlanReplanProposal(
        verifier_decision="rerun_recommended",
        proposed_action="rerun_task",
        affected_tasks=["train_model"],
        rationale=["Model metrics are weak enough to recommend a rerun."],
        required_user_decisions=["Approve rerun before any queued execution."],
        proposed_run_plan_patch={
            "schema_version": "reviewable_run_plan_patch.v1",
            "applied": False,
            "operations": [
                {
                    "operation_id": "op_000001",
                    "op": "rerun_task",
                    "task_id": "train_model",
                    "source_finding_id": "finding_1",
                    "category": "poor_model_metrics",
                    "reason": "Model metrics are weak enough to recommend a rerun.",
                }
            ],
        },
        executable=False,
        source_finding_ids=["finding_1"],
    )
    proposal_path = run_dir / "review" / "replan_proposal.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(json.dumps(proposal.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    request = ReplanApplicationRequest(
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        proposal_artifact_ref="review/replan_proposal.json",
        proposal_hash=proposal_artifact_hash(proposal_path),
        selected_action="rerun_task",
        selected_operation_ids=["op_000001"],
        executable=False,
    )
    bundle = write_replan_application_artifacts(
        workspace_dir=workspace,
        request=request,
        actor="review-user",
        actor_source="test",
        current_run_plan=run_plan,
        stage_state=stage_state,
    )
    return {
        "run_plan": run_plan,
        "stage_state": stage_state,
        "bundle": bundle,
    }


def _grant_permission(workspace: Path) -> dict[str, Any]:
    return ServerPermissionStore(workspace).create_grant(
        PROJECT_ID,
        PERMISSION_ACTION,
        actor="admin",
        actor_source="test",
        run_id=RUN_ID,
        reason="test grant",
    )


def _enable_route(app) -> None:
    app.config["AI4S_ENABLE_INTERNAL_RESUME_INTENT_EXECUTE_ROUTE"] = True


def _payload(*, actor: str | None = "json-user", approved_gates: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "run_id": RUN_ID,
        "approved_gates": approved_gates if approved_gates is not None else [GateName.TRAIN_CONFIG.value],
    }
    if actor is not None:
        payload["actor"] = actor
    return payload


def _audit_records(workspace: Path) -> list[dict[str, Any]]:
    path = ProjectStorage(workspace).run_dir(PROJECT_ID, RUN_ID) / RESUME_INTENT_VALIDATION_AUDIT_REF
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_internal_resume_intent_execute_route_is_disabled_by_default(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/resume-intent/execute", json=_payload())

    assert response.status_code == 404


def test_internal_resume_intent_execute_route_resumes_once_with_audit_and_memory(tmp_path: Path) -> None:
    _prepare_waiting_resume_intent(tmp_path)
    grant = _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/resume-intent/execute",
        json=_payload(actor=None),
        headers={"X-Actor": "resume-user"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["executable"] is False
    validation = ResumeIntentValidationResult.model_validate(payload["validation"])
    assert validation.decision == "resume_eligible"
    assert validation.approved_gates == [GateName.TRAIN_CONFIG.value]
    assert payload["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert payload["permission"]["allowed"] is True
    assert payload["permission"]["action"] == PERMISSION_ACTION
    assert payload["permission"]["grant_id"] == grant["grant_id"]
    storage = ProjectStorage(tmp_path)
    decisions = storage.read_gate_decisions(PROJECT_ID, RUN_ID)
    assert len(decisions) == 1
    assert decisions[0]["gate"] == GateName.TRAIN_CONFIG.value
    assert decisions[0]["actor"] == "resume-user"
    audit = _audit_records(tmp_path)
    assert [record["event"] for record in audit] == [
        "resume_intent_consumed",
        "run_plan_resume_completed",
    ]
    assert audit[0]["intent_id"] == validation.intent_id
    assert audit[1]["validation_decision"] == "resume_eligible"
    records = ProjectMemory(tmp_path).list_project_records(PROJECT_ID)
    assert len(records) == 1
    assert records[0].value["validation_decision"] == "resume_eligible"

    second = client.post(
        "/api/internal/run-plan/resume-intent/execute",
        json=_payload(actor=None),
        headers={"X-Actor": "resume-user"},
    )

    assert second.status_code == 409
    second_payload = second.get_json()
    assert second_payload["ok"] is False
    assert second_payload["execution"] is None
    second_validation = ResumeIntentValidationResult.model_validate(second_payload["validation"])
    assert second_validation.decision == "stale_intent"
    assert second_validation.error is not None
    assert second_validation.error["type"] == "resume_intent_already_consumed"
    assert len(storage.read_gate_decisions(PROJECT_ID, RUN_ID)) == 1


def test_internal_resume_intent_execute_route_requires_execute_permission(tmp_path: Path) -> None:
    _prepare_waiting_resume_intent(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/resume-intent/execute", json=_payload())

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["validation"] is None
    assert payload["execution"] is None
    assert payload["error"]["type"] == "permission_denied"
    assert ProjectStorage(tmp_path).read_gate_decisions(PROJECT_ID, RUN_ID) == []


def test_internal_resume_intent_execute_route_requires_resume_eligible_validation(tmp_path: Path) -> None:
    _prepare_waiting_resume_intent(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/resume-intent/execute",
        json=_payload(approved_gates=[]),
    )

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["execution"] is None
    validation = ResumeIntentValidationResult.model_validate(payload["validation"])
    assert validation.decision == "needs_gate_approval"
    assert payload["error"]["type"] == "resume_intent_not_eligible"
    assert ProjectStorage(tmp_path).read_gate_decisions(PROJECT_ID, RUN_ID) == []
    assert [record["event"] for record in _audit_records(tmp_path)] == ["run_plan_resume_failed"]


def test_internal_resume_intent_execute_route_records_failed_audit_when_executor_returns_failed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _prepare_waiting_resume_intent(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    def failed_resume(self, **kwargs):
        return {
            "ok": False,
            "status": RunStatus.FAILED.value,
            "failed_task": "train_model",
            "error": {"message": "simulated failure"},
        }

    monkeypatch.setattr("ai4s_agent.routes.internal_run_plan_queue.RunPlanExecutor.resume_after_gate", failed_resume)

    response = client.post("/api/internal/run-plan/resume-intent/execute", json=_payload())

    assert response.status_code == 500
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["execution"]["status"] == RunStatus.FAILED.value
    assert payload["error"] == {
        "type": "run_plan_resume_execution_failed",
        "message": "simulated failure",
        "status": RunStatus.FAILED.value,
        "failed_task": "train_model",
    }
    assert [record["event"] for record in _audit_records(tmp_path)] == [
        "resume_intent_consumed",
        "run_plan_resume_failed",
    ]
    assert ProjectMemory(tmp_path).list_project_records(PROJECT_ID) == []

    second = client.post("/api/internal/run-plan/resume-intent/execute", json=_payload())

    assert second.status_code == 409
    second_validation = ResumeIntentValidationResult.model_validate(second.get_json()["validation"])
    assert second_validation.decision == "stale_intent"
    assert second_validation.error is not None
    assert second_validation.error["type"] == "resume_intent_already_consumed"


def test_internal_resume_intent_execute_route_audit_write_failure_fails_before_executor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _prepare_waiting_resume_intent(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    def raise_oserror(*args, **kwargs):
        raise OSError("audit unavailable")

    monkeypatch.setattr(
        "ai4s_agent.routes.internal_run_plan_queue.append_resume_intent_validation_audit_record",
        raise_oserror,
    )

    response = client.post("/api/internal/run-plan/resume-intent/execute", json=_payload())

    assert response.status_code == 500
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"]["type"] == "audit_write_failed"
    assert ProjectStorage(tmp_path).read_gate_decisions(PROJECT_ID, RUN_ID) == []
    state = ProjectStorage(tmp_path).read_stage_state(PROJECT_ID, RUN_ID)
    assert state is not None
    assert state.status == RunStatus.WAITING_USER
