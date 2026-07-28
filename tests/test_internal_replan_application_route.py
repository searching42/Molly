from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.app import create_app
from ai4s_agent.memory import ProjectMemory
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.run_plan_replan_application_audit_memory import REPLAN_APPLICATION_AUDIT_REF
from ai4s_agent.run_plan_replan_proposal import RunPlanReplanProposal
from ai4s_agent.run_plan_state_fingerprint import build_resume_state_binding
from ai4s_agent.schemas import PlannedTask, RunPlan, RunStatus, StageState
from ai4s_agent.server_permissions import ServerPermissionStore
from ai4s_agent.storage import ProjectStorage


PERMISSION_ACTION = "run_plan_replan_apply"


def _proposal_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _run_plan(*task_ids: str, run_id: str = "run-apply") -> RunPlan:
    tasks = list(task_ids) or ["inspect_dataset", "train_model", "render_report"]
    return RunPlan(
        run_id=run_id,
        requested_tasks=tasks,
        tasks=[PlannedTask(task_id=task_id) for task_id in tasks],
        available_artifacts=["uploaded_dataset"],
    )


def _execution_snapshot(run_plan: RunPlan, *, task_id: str = "train_model") -> dict[str, Any]:
    gates = sorted(AtomicTaskRegistry().get(task_id).gates)
    material = {
        "schema_version": 1,
        "run_id": run_plan.run_id,
        "task_id": task_id,
        "adapter": "train_model_baseline_adapter",
        "run_plan": run_plan.model_dump(mode="json"),
        "task_options": {},
        "payload": {},
        "input_artifacts": {},
        "approved_gates": gates,
    }
    digest = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "snapshot_id": f"{run_plan.run_id}:{task_id}:{digest[:16]}",
        "snapshot_hash": digest,
        **material,
    }


def _stage_state(run_plan: RunPlan | None = None) -> StageState:
    plan = run_plan or _run_plan()
    now = now_iso()
    return StageState(
        stage="train_model",
        status=RunStatus.WAITING_USER,
        started_at=now,
        ended_at=now,
        updated_at=now,
        details={
            "required_gates": list(AtomicTaskRegistry().get("train_model").gates),
            "executed_tasks": ["inspect_dataset"],
            "execution_snapshot": _execution_snapshot(plan),
        },
    )


def _write_current_state(workspace: Path, *, project_id: str = "proj-apply", run_id: str = "run-apply") -> None:
    storage = ProjectStorage(workspace)
    run_dir = storage.run_dir(project_id, run_id)
    run_plan = _run_plan(run_id=run_id)
    write_json(run_dir / "run_plan.json", run_plan.model_dump(mode="json"))
    storage.write_stage_state(project_id, run_id, _stage_state(run_plan))


def _write_proposal(
    workspace: Path,
    *,
    project_id: str = "proj-apply",
    run_id: str = "run-apply",
) -> Path:
    run_dir = ProjectStorage(workspace).run_dir(project_id, run_id)
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
    return write_json(run_dir / "review" / "replan_proposal.json", proposal.model_dump(mode="json"))


def _payload(
    proposal_path: Path,
    *,
    project_id: str = "proj-apply",
    run_id: str = "run-apply",
    actor: str | None = "json-user",
    proposal_hash: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_id": project_id,
        "run_id": run_id,
        "proposal_artifact_ref": "review/replan_proposal.json",
        "proposal_hash": proposal_hash if proposal_hash is not None else _proposal_hash(proposal_path),
        "selected_action": "rerun_task",
        "selected_operation_ids": ["op_000001"],
        "reason": "User approved review-only rerun application.",
    }
    if actor is not None:
        payload["actor"] = actor
    return payload


def _enable_route(app) -> None:
    app.config["AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = lambda storage: (_ for _ in ()).throw(
        AssertionError("executor factory must not be called by replan application route")
    )


def _grant_permission(
    workspace: Path,
    *,
    project_id: str = "proj-apply",
    run_id: str = "run-apply",
    actor: str = "admin",
) -> dict[str, Any]:
    return ServerPermissionStore(workspace).create_grant(
        project_id,
        PERMISSION_ACTION,
        actor=actor,
        actor_source="test",
        run_id=run_id,
        reason="test grant",
    )


def _audit_records(workspace: Path, *, project_id: str = "proj-apply", run_id: str = "run-apply") -> list[dict[str, Any]]:
    path = ProjectStorage(workspace).run_dir(project_id, run_id) / REPLAN_APPLICATION_AUDIT_REF
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_internal_replan_application_route_is_disabled_by_default(tmp_path: Path) -> None:
    proposal_path = _write_proposal(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/replan/apply-review", json=_payload(proposal_path))

    assert response.status_code == 404


def test_internal_replan_application_route_requires_actor(tmp_path: Path) -> None:
    proposal_path = _write_proposal(tmp_path)
    _write_current_state(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/replan/apply-review", json=_payload(proposal_path, actor=None))

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    assert "actor required" in payload["error"]["message"]
    assert _audit_records(tmp_path) == []
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_replan_application_route_requires_permission_grant(tmp_path: Path) -> None:
    proposal_path = _write_proposal(tmp_path)
    _write_current_state(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/replan/apply-review", json=_payload(proposal_path))

    assert response.status_code == 403
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"]["type"] == "permission_denied"
    assert payload["permission"]["action"] == PERMISSION_ACTION
    audit = _audit_records(tmp_path)
    assert [record["event"] for record in audit] == ["replan_application_failed"]
    assert audit[0]["actor"] == "json-user"
    assert audit[0]["error"]["type"] == "permission_denied"
    assert not (ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply") / "review" / "replan_application_record.json").exists()
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_replan_application_route_writes_artifacts_audit_and_memory(tmp_path: Path) -> None:
    proposal_path = _write_proposal(tmp_path)
    _write_current_state(tmp_path)
    grant = _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/replan/apply-review",
        json=_payload(proposal_path, actor=None),
        headers={"X-Actor": "review-user"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["project_id"] == "proj-apply"
    assert payload["run_id"] == "run-apply"
    assert payload["executable"] is False
    assert payload["application"]["result_type"] == "resume_intent"
    assert payload["application"]["selected_operation_ids"] == ["op_000001"]
    assert payload["application"]["artifacts"]["replan_application_record"] == "review/replan_application_record.json"
    assert payload["application"]["artifacts"]["replan_resume_intent"] == "review/replan_resume_intent.json"
    assert payload["audit_refs"] == [REPLAN_APPLICATION_AUDIT_REF]
    assert payload["memory"]["category"] == "run_plan_replan_application"
    assert payload["memory"]["value"]["artifact_refs"] == payload["application"]["artifacts"]
    assert payload["memory"]["value"]["audit_refs"] == [REPLAN_APPLICATION_AUDIT_REF]
    assert payload["permission"]["allowed"] is True
    assert payload["permission"]["action"] == PERMISSION_ACTION
    assert payload["permission"]["grant_id"] == grant["grant_id"]
    run_dir = ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply")
    assert (run_dir / "review" / "replan_application_record.json").exists()
    assert (run_dir / "review" / "replan_resume_intent.json").exists()
    application_record = json.loads((run_dir / "review" / "replan_application_record.json").read_text(encoding="utf-8"))
    resume_intent = json.loads((run_dir / "review" / "replan_resume_intent.json").read_text(encoding="utf-8"))
    assert application_record["resume_state_binding"] == resume_intent["resume_state_binding"]
    assert application_record["resume_state_binding"] == build_resume_state_binding(
        _run_plan(),
        _stage_state(),
    ).model_dump(mode="json")
    audit = _audit_records(tmp_path)
    assert [record["event"] for record in audit] == [
        "replan_application_requested",
        "replan_application_completed",
    ]
    assert audit[0]["actor"] == "review-user"
    assert audit[0]["actor_source"] == "header:X-Actor"
    assert audit[0]["application_id"] == ""
    assert audit[1]["application_id"] == payload["application"]["application_id"]
    assert audit[1]["artifact_refs"] == payload["application"]["artifacts"]
    records = ProjectMemory(tmp_path).list_project_records("proj-apply")
    assert len(records) == 1
    assert records[0].value["selected_operation_ids"] == ["op_000001"]
    assert "Model metrics are weak enough" not in json.dumps(records[0].model_dump(mode="json"))
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_replan_application_route_rejects_client_supplied_state_binding(tmp_path: Path) -> None:
    proposal_path = _write_proposal(tmp_path)
    _write_current_state(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()
    payload = _payload(proposal_path)
    payload["resume_state_binding"] = {
        "schema_version": "resume_state_binding.v1",
        "run_plan_fingerprint": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "stage_fingerprint": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "stage": "train_model",
        "stage_status": "WAITING_USER",
    }

    response = client.post("/api/internal/run-plan/replan/apply-review", json=payload)

    assert response.status_code == 400
    body = response.get_json()
    assert body["ok"] is False
    assert body["error"]["type"] == "validation_error"
    assert not (ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply") / "review" / "replan_application_record.json").exists()
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_replan_application_route_writes_failed_audit_on_validation_error(tmp_path: Path) -> None:
    proposal_path = _write_proposal(tmp_path)
    _write_current_state(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/replan/apply-review",
        json=_payload(proposal_path, proposal_hash="sha256:not-the-proposal"),
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    audit = _audit_records(tmp_path)
    assert [record["event"] for record in audit] == [
        "replan_application_requested",
        "replan_application_failed",
    ]
    assert audit[-1]["error"]["type"] == "validation_error"
    assert "proposal_hash mismatch" in audit[-1]["error"]["message"]
    assert ProjectMemory(tmp_path).list_project_records("proj-apply") == []
    assert not (ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply") / "review" / "replan_application_record.json").exists()
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_replan_application_route_audit_write_failure_fails_before_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    proposal_path = _write_proposal(tmp_path)
    _write_current_state(tmp_path)
    _grant_permission(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    def raise_oserror(*args: Any, **kwargs: Any) -> None:
        raise OSError("audit unavailable")

    monkeypatch.setattr(
        "ai4s_agent.routes.internal_run_plan_queue.append_replan_application_audit_record",
        raise_oserror,
    )

    response = client.post("/api/internal/run-plan/replan/apply-review", json=_payload(proposal_path))

    assert response.status_code == 500
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"]["type"] == "audit_write_failed"
    assert "audit unavailable" in payload["error"]["message"]
    assert not (ProjectStorage(tmp_path).run_dir("proj-apply", "run-apply") / "review" / "replan_application_record.json").exists()
    assert ProjectMemory(tmp_path).list_project_records("proj-apply") == []
    assert not (tmp_path / ".ai4s_internal" / "run_plan_queues").exists()


def test_internal_replan_application_route_rejects_unsafe_project_id_without_audit(tmp_path: Path) -> None:
    proposal_path = _write_proposal(tmp_path)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    _enable_route(app)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/replan/apply-review",
        json=_payload(proposal_path, project_id="../outside"),
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    assert "project_id must be a safe path component" in payload["error"]["message"]
    assert _audit_records(tmp_path) == []
