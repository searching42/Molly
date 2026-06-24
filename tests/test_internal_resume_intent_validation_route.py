from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.app import create_app
from ai4s_agent.memory import ProjectMemory
from ai4s_agent.run_plan_replan_application import ReplanApplicationRequest
from ai4s_agent.run_plan_replan_application_artifacts import write_replan_application_artifacts
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.run_plan_resume_intent_validation import ResumeIntentValidationResult
from ai4s_agent.run_plan_resume_intent_validation_audit_memory import RESUME_INTENT_VALIDATION_AUDIT_REF
from ai4s_agent.schemas import PlannedTask, RunPlan, RunStatus, StageState
from ai4s_agent.server_permissions import ServerPermissionStore
from ai4s_agent.storage import ProjectStorage


PROJECT_ID = "proj-resume"
RUN_ID = "run-resume"
PERMISSION_ACTION = "run_plan_resume_intent_use"


def _proposal_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _run_plan(*task_ids: str) -> RunPlan:
    tasks = list(task_ids) or ["inspect_dataset", "train_model", "render_report"]
    return RunPlan(
        run_id=RUN_ID,
        requested_tasks=tasks,
        tasks=[PlannedTask(task_id=task_id) for task_id in tasks],
    )


def _write_current_state(workspace: Path, *, run_plan: RunPlan | None = None) -> None:
    storage = ProjectStorage(workspace)
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    write_json(run_dir / "run_plan.json", (run_plan or _run_plan()).model_dump(mode="json"))
    now = now_iso()
    storage.write_stage_state(
        PROJECT_ID,
        RUN_ID,
        StageState(
            stage="train_model",
            status=RunStatus.WAITING_USER,
            started_at=now,
            ended_at=now,
            updated_at=now,
            details={"required_gates": ["gate_replan_rerun_task"]},
        ),
    )


def _write_resume_intent_artifacts(workspace: Path) -> Path:
    run_dir = ProjectStorage(workspace).run_dir(PROJECT_ID, RUN_ID)
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
    proposal_path = write_json(run_dir / "review" / "replan_proposal.json", proposal.model_dump(mode="json"))
    request = ReplanApplicationRequest(
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        proposal_artifact_ref="review/replan_proposal.json",
        proposal_hash=_proposal_hash(proposal_path),
        selected_action="rerun_task",
        selected_operation_ids=["op_000001"],
        executable=False,
    )
    write_replan_application_artifacts(
        workspace_dir=workspace,
        request=request,
        actor="review-user",
        actor_source="test",
    )
    return proposal_path


def _enable_route(app) -> None:
    app.config["AI4S_ENABLE_INTERNAL_RESUME_INTENT_VALIDATION_ROUTE"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = lambda storage: (_ for _ in ()).throw(
        AssertionError("executor factory must not be called by resume intent validation route")
    )


def _grant_permission(workspace: Path) -> dict[str, Any]:
    return ServerPermissionStore(workspace).create_grant(
        PROJECT_ID,
        PERMISSION_ACTION,
        actor="admin",
        actor_source="test",
        run_id=RUN_ID,
        reason="test grant",
    )


def _payload(*, actor: str | None = "json-user", approved_gates: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "run_id": RUN_ID,
    }
    if actor is not None:
        payload["actor"] = actor
    if approved_gates is not None:
        payload["approved_gates"] = approved_gates
    return payload


def _audit_records(workspace: Path) -> list[dict[str, Any]]:
    path = ProjectStorage(workspace).run_dir(PROJECT_ID, RUN_ID) / RESUME_INTENT_VALIDATION_AUDIT_REF
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_internal_resume_intent_validation_route_is_disabled_by_default(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/resume-intent/validate", json=_payload())

    assert response.status_code == 404


def test_internal_resume_intent_validation_route_requires_actor(tmp_path: Path) -> None:
    _write_resume_intent_artifacts(tmp_path)
    _write_current_state(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/resume-intent/validate", json=_payload(actor=None))

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    assert "actor required" in payload["error"]["message"]
    assert _audit_records(tmp_path) == []
    assert ProjectMemory(tmp_path).list_project_records(PROJECT_ID) == []
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_resume_intent_validation_route_requires_permission_grant(tmp_path: Path) -> None:
    _write_resume_intent_artifacts(tmp_path)
    _write_current_state(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/resume-intent/validate", json=_payload())

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["validation"] is None
    assert payload["error"]["type"] == "permission_denied"
    assert payload["permission"]["action"] == PERMISSION_ACTION
    audit = _audit_records(tmp_path)
    assert [record["event"] for record in audit] == ["resume_intent_validation_failed"]
    assert audit[0]["actor"] == "json-user"
    assert audit[0]["error"]["type"] == "permission_denied"
    assert ProjectMemory(tmp_path).list_project_records(PROJECT_ID) == []
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_resume_intent_validation_route_returns_result_audit_and_memory(tmp_path: Path) -> None:
    _write_resume_intent_artifacts(tmp_path)
    _write_current_state(tmp_path)
    grant = _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/resume-intent/validate",
        json=_payload(actor=None, approved_gates=["gate_replan_rerun_task"]),
        headers={"X-Actor": "review-user"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["project_id"] == PROJECT_ID
    assert payload["run_id"] == RUN_ID
    assert payload["executable"] is False
    validation = ResumeIntentValidationResult.model_validate(payload["validation"])
    assert validation.ok is True
    assert validation.decision == "resume_eligible"
    assert validation.approved_gates == ["gate_replan_rerun_task"]
    assert validation.rerun_tasks == ["train_model"]
    assert validation.resume_from_task == "train_model"
    assert payload["audit_refs"] == [RESUME_INTENT_VALIDATION_AUDIT_REF]
    assert payload["memory"]["category"] == "run_plan_resume_intent_validation"
    assert payload["memory"]["value"]["validation_decision"] == "resume_eligible"
    assert payload["memory"]["value"]["audit_refs"] == [RESUME_INTENT_VALIDATION_AUDIT_REF]
    assert payload["permission"]["allowed"] is True
    assert payload["permission"]["action"] == PERMISSION_ACTION
    assert payload["permission"]["grant_id"] == grant["grant_id"]
    audit = _audit_records(tmp_path)
    assert [record["event"] for record in audit] == [
        "resume_intent_validation_requested",
        "resume_intent_validation_completed",
    ]
    assert audit[0]["actor"] == "review-user"
    assert audit[0]["actor_source"] == "header:X-Actor"
    assert audit[0]["validation_decision"] == ""
    assert audit[1]["validation_decision"] == "resume_eligible"
    assert audit[1]["approved_gates"] == ["gate_replan_rerun_task"]
    records = ProjectMemory(tmp_path).list_project_records(PROJECT_ID)
    assert len(records) == 1
    assert records[0].value["intent_id"] == validation.intent_id
    assert "validation_findings" not in json.dumps(records[0].model_dump(mode="json"))
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_resume_intent_validation_route_keeps_artifact_approved_gates_when_payload_omits_override(
    tmp_path: Path,
) -> None:
    _write_resume_intent_artifacts(tmp_path)
    _write_current_state(tmp_path)
    _grant_permission(tmp_path)
    run_dir = ProjectStorage(tmp_path).run_dir(PROJECT_ID, RUN_ID)
    intent_path = run_dir / "review" / "replan_resume_intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["approved_gates"] = ["gate_replan_rerun_task"]
    intent_path.write_text(json.dumps(intent, indent=2, sort_keys=True), encoding="utf-8")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/resume-intent/validate", json=_payload())

    assert response.status_code == 200
    validation = ResumeIntentValidationResult.model_validate(response.get_json()["validation"])
    assert validation.decision == "resume_eligible"
    assert validation.approved_gates == ["gate_replan_rerun_task"]


def test_internal_resume_intent_validation_route_records_stale_result_without_execution(tmp_path: Path) -> None:
    proposal_path = _write_resume_intent_artifacts(tmp_path)
    _write_current_state(tmp_path)
    _grant_permission(tmp_path)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["rationale"].append("mutated after application")
    proposal_path.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/resume-intent/validate", json=_payload())

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is False
    validation = ResumeIntentValidationResult.model_validate(payload["validation"])
    assert validation.decision == "stale_intent"
    assert validation.error is not None
    assert validation.error["type"] == "proposal_hash_mismatch"
    assert payload["memory"]["value"]["validation_decision"] == "stale_intent"
    audit = _audit_records(tmp_path)
    assert [record["event"] for record in audit] == [
        "resume_intent_validation_requested",
        "resume_intent_validation_completed",
    ]
    assert audit[-1]["validation_decision"] == "stale_intent"
    assert ProjectMemory(tmp_path).list_project_records(PROJECT_ID)[0].value["error"]["type"] == "proposal_hash_mismatch"
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_resume_intent_validation_route_writes_failed_audit_on_missing_run_plan(tmp_path: Path) -> None:
    _write_resume_intent_artifacts(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/resume-intent/validate", json=_payload())

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["validation"] is None
    assert payload["error"]["type"] == "validation_error"
    assert "current RunPlan not found" in payload["error"]["message"]
    audit = _audit_records(tmp_path)
    assert [record["event"] for record in audit] == [
        "resume_intent_validation_requested",
        "resume_intent_validation_failed",
    ]
    assert audit[-1]["error"]["type"] == "validation_error"
    assert ProjectMemory(tmp_path).list_project_records(PROJECT_ID) == []
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_resume_intent_validation_route_audit_write_failure_fails_before_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_resume_intent_artifacts(tmp_path)
    _write_current_state(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    def raise_oserror(*args: Any, **kwargs: Any) -> None:
        raise OSError("audit unavailable")

    monkeypatch.setattr(
        "ai4s_agent.routes.internal_run_plan_queue.append_resume_intent_validation_audit_record",
        raise_oserror,
    )

    response = client.post("/api/internal/run-plan/resume-intent/validate", json=_payload())

    assert response.status_code == 500
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["validation"] is None
    assert payload["error"]["type"] == "audit_write_failed"
    assert "audit unavailable" in payload["error"]["message"]
    assert ProjectMemory(tmp_path).list_project_records(PROJECT_ID) == []
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_resume_intent_validation_route_does_not_replace_default_resume(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post("/api/run-plan/resume", json={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "project_id required"
