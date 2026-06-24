from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.app import create_app
from ai4s_agent.run_plan_queue_summary import RunPlanQueueExecutionSummary
from ai4s_agent.schemas import PlannedTask, RunPlan
from ai4s_agent.server_permissions import ServerPermissionStore
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


PERMISSION_ACTION = "run_plan_queue_execute"


class FakeRunPlanExecutor:
    def __init__(self, execution: dict[str, Any], calls: list[dict[str, Any]]) -> None:
        self.execution = dict(execution)
        self.calls = calls

    def execute(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        input_artifacts: dict[str, Any] | None = None,
        task_options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "project_id": project_id,
                "run_plan": run_plan,
                "input_artifacts": input_artifacts,
                "task_options": task_options,
            }
        )
        return dict(self.execution)


def _run_plan(run_id: str = "run-a") -> RunPlan:
    return RunPlan(
        run_id=run_id,
        requested_tasks=["train_model"],
        tasks=[PlannedTask(task_id="train_model")],
        available_artifacts=[],
        missing_artifacts=[],
    )


def _payload(*, run_plan: RunPlan | None = None, actor: str | None = "json-user") -> dict[str, Any]:
    payload = {
        "project_id": "proj-a",
        "run_plan": (run_plan or _run_plan()).model_dump(mode="json"),
        "input_artifacts": {"dataset": "datasets/input.csv"},
        "task_options": {"train_model": {"epochs": 1}},
        "max_iterations": 10,
    }
    if actor is not None:
        payload["actor"] = actor
    return payload


def _default_queue_dir(workspace: Path, project_id: str = "proj-a", run_id: str = "run-a") -> Path:
    return workspace / ".ai4s_internal" / "run_plan_queues" / project_id / run_id


def _audit_path(workspace: Path) -> Path:
    return workspace / ".ai4s_internal" / "audit" / "internal_run_plan_queue_audit.jsonl"


def _audit_records(workspace: Path) -> list[dict[str, Any]]:
    path = _audit_path(workspace)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _enable_queue_route(app, *, execution: dict[str, Any], calls: list[dict[str, Any]]) -> None:
    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return FakeRunPlanExecutor(execution, calls)

    app.config["AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = factory


def _grant_run_plan_queue_permission(workspace: Path, *, project_id: str = "proj-a", run_id: str = "run-a", actor: str = "admin") -> dict[str, Any]:
    return ServerPermissionStore(workspace).create_grant(
        project_id,
        PERMISSION_ACTION,
        actor=actor,
        actor_source="test",
        run_id=run_id,
        reason="test grant",
    )


def _iso_at(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat().replace("+00:00", "Z")


def test_internal_run_plan_queue_route_is_disabled_by_default(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 404


def test_internal_run_plan_queue_route_executes_when_feature_flag_enabled(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    grant = _grant_run_plan_queue_permission(tmp_path)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 200
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is True
    assert summary.terminal is True
    assert summary.final_job is not None
    assert summary.final_job["status"] == "succeeded"
    assert summary.final_lease is not None
    assert summary.final_lease["status"] == "completed"
    assert summary.loop_results == ["completed", "idle"]
    assert calls[0]["project_id"] == "proj-a"
    assert calls[0]["input_artifacts"] == {"dataset": "datasets/input.csv"}
    assert calls[0]["task_options"] == {"train_model": {"epochs": 1}}
    assert (_default_queue_dir(tmp_path) / "worker_queue.json").exists()
    assert (_default_queue_dir(tmp_path) / "worker_leases.json").exists()
    audit = _audit_records(tmp_path)
    assert audit[-2]["outcome"] == "requested"
    assert audit[-2]["status_code"] == 202
    assert audit[-2]["queued_job_id"] == ""
    assert audit[-2]["permission_allowed"] is True
    assert audit[-2]["permission_reason"] == "SERVER_GRANT"
    assert audit[-2]["permission_action"] == PERMISSION_ACTION
    assert audit[-2]["permission_resource"] == "project:proj-a:run:run-a"
    assert audit[-2]["permission_grant_id"] == grant["grant_id"]
    assert audit[-1]["event"] == "internal_run_plan_queue_execute"
    assert audit[-1]["actor"] == "json-user"
    assert audit[-1]["actor_source"] == "json:actor"
    assert audit[-1]["project_id"] == "proj-a"
    assert audit[-1]["run_id"] == "run-a"
    assert audit[-1]["route"] == "/api/internal/run-plan/queue/execute"
    assert audit[-1]["feature_flag_enabled"] is True
    assert audit[-1]["outcome"] == "succeeded"
    assert audit[-1]["status_code"] == 200
    assert audit[-1]["queued_job_id"] == summary.queued_job_id
    assert audit[-1]["permission_allowed"] is True
    assert audit[-1]["permission_reason"] == "SERVER_GRANT"
    assert audit[-1]["permission_resource"] == "project:proj-a:run:run-a"
    assert str(audit[-1]["timestamp"]).endswith("Z")


def test_internal_run_plan_queue_route_requires_actor_without_executor(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload(actor=None))

    assert response.status_code == 403
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.error is not None
    assert summary.error["type"] == "validation_error"
    assert "actor required" in summary.error["message"]
    assert calls == []
    audit = _audit_records(tmp_path)
    assert audit[-1]["actor"] == ""
    assert audit[-1]["actor_source"] == "missing"
    assert audit[-1]["project_id"] == "proj-a"
    assert audit[-1]["run_id"] == "run-a"
    assert audit[-1]["outcome"] == "validation_error"
    assert audit[-1]["status_code"] == 403
    assert audit[-1]["error"]["message"] == "actor required"


def test_internal_run_plan_queue_route_audit_write_failure_fails_before_executor(monkeypatch, tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    _grant_run_plan_queue_permission(tmp_path)
    client = app.test_client()

    def raise_oserror(*args, **kwargs):
        raise OSError("audit unavailable")

    monkeypatch.setattr("ai4s_agent.routes.internal_run_plan_queue._append_audit_event", raise_oserror)

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 500
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.error is not None
    assert summary.error["type"] == "audit_write_failed"
    assert summary.error["message"] == "audit unavailable"
    assert calls == []
    assert not (_default_queue_dir(tmp_path) / "worker_queue.json").exists()
    assert not (_default_queue_dir(tmp_path) / "worker_leases.json").exists()


def test_internal_run_plan_queue_route_denies_actor_without_permission_grant(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 403
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.error is not None
    assert summary.error["type"] == "permission_denied"
    assert "SERVER_GRANT_REQUIRED" in summary.error["message"]
    assert calls == []
    assert not (_default_queue_dir(tmp_path) / "worker_queue.json").exists()
    audit = _audit_records(tmp_path)
    assert audit[-1]["actor"] == "json-user"
    assert audit[-1]["actor_source"] == "json:actor"
    assert audit[-1]["outcome"] == "permission_denied"
    assert audit[-1]["status_code"] == 403
    assert audit[-1]["permission_allowed"] is False
    assert audit[-1]["permission_reason"] == "SERVER_GRANT_REQUIRED"
    assert audit[-1]["permission_action"] == PERMISSION_ACTION
    assert audit[-1]["permission_resource"] == "project:proj-a:run:run-a"
    assert audit[-1]["permission_scope"] == "project:proj-a:run:run-a"
    assert audit[-1]["error"]["type"] == "permission_denied"


def test_internal_run_plan_queue_route_denies_expired_permission_grant(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    store = ServerPermissionStore(tmp_path)
    grant = store.create_grant("proj-a", PERMISSION_ACTION, actor="admin", run_id="run-a", expires_at=_iso_at(timedelta(hours=1)))
    grant["expires_at"] = _iso_at(timedelta(minutes=-1))
    write_json(store._grants_path("proj-a"), {"project_id": "proj-a", "updated_at": now_iso(), "grants": [grant]})
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 403
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.error is not None
    assert summary.error["type"] == "permission_denied"
    assert "EXPIRED_GRANT" in summary.error["message"]
    assert calls == []
    audit = _audit_records(tmp_path)
    assert audit[-1]["outcome"] == "permission_denied"
    assert audit[-1]["permission_reason"] == "EXPIRED_GRANT"
    assert audit[-1]["permission_grant_id"] == grant["grant_id"]


def test_internal_run_plan_queue_route_denies_revoked_permission_grant(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    store = ServerPermissionStore(tmp_path)
    grant = store.create_grant("proj-a", PERMISSION_ACTION, actor="admin", run_id="run-a")
    store.revoke_grant("proj-a", grant["grant_id"], revoked_by="admin", revoke_reason="test revoke")
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 403
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.error is not None
    assert summary.error["type"] == "permission_denied"
    assert "REVOKED_GRANT" in summary.error["message"]
    assert calls == []
    audit = _audit_records(tmp_path)
    assert audit[-1]["outcome"] == "permission_denied"
    assert audit[-1]["permission_reason"] == "REVOKED_GRANT"
    assert audit[-1]["permission_grant_id"] == grant["grant_id"]


def test_internal_run_plan_queue_route_audits_x_actor_success(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    _grant_run_plan_queue_permission(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/queue/execute",
        json=_payload(actor=None),
        headers={"X-Actor": "test-user"},
    )

    assert response.status_code == 200
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is True
    audit = _audit_records(tmp_path)
    assert audit[-2]["actor"] == "test-user"
    assert audit[-2]["actor_source"] == "header:X-Actor"
    assert audit[-2]["outcome"] == "requested"
    assert audit[-1]["actor"] == "test-user"
    assert audit[-1]["actor_source"] == "header:X-Actor"
    assert audit[-1]["outcome"] == "succeeded"
    assert audit[-1]["queued_job_id"] == summary.queued_job_id


def test_internal_run_plan_queue_route_can_be_enabled_by_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE", "1")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []

    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls)

    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = factory
    _grant_run_plan_queue_permission(tmp_path)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 200
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is True
    assert calls


def test_internal_run_plan_queue_route_returns_summary_for_invalid_payload(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json={"project_id": "proj-a", "actor": "json-user"})

    assert response.status_code == 400
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.queued_job_id == ""
    assert summary.final_job is None
    assert summary.final_lease is None
    assert summary.loop_results == []
    assert summary.error is not None
    assert summary.error["type"] == "validation_error"
    assert "run_plan object required" in summary.error["message"]
    assert calls == []
    audit = _audit_records(tmp_path)
    assert audit[-1]["actor"] == "json-user"
    assert audit[-1]["actor_source"] == "json:actor"
    assert audit[-1]["project_id"] == "proj-a"
    assert audit[-1]["outcome"] == "validation_error"
    assert audit[-1]["status_code"] == 400
    assert audit[-1]["error"]["message"] == "run_plan object required"


def test_internal_run_plan_queue_route_audits_failed_executor(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": False, "run_id": "run-a", "status": "FAILED", "error": {"message": "adapter failed"}}, calls=calls)
    _grant_run_plan_queue_permission(tmp_path)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 400
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is True
    assert summary.final_job is not None
    assert summary.final_job["status"] == "failed"
    assert summary.final_job["error"] == {"reason": "adapter failed"}
    audit = _audit_records(tmp_path)
    assert audit[-2]["outcome"] == "requested"
    assert audit[-2]["status_code"] == 202
    assert audit[-2]["permission_allowed"] is True
    assert audit[-1]["outcome"] == "failed"
    assert audit[-1]["status_code"] == 400
    assert audit[-1]["queued_job_id"] == summary.queued_job_id
    assert audit[-1]["error"]["message"] == "adapter failed"


def test_internal_run_plan_queue_route_rejects_absolute_queue_dir_without_executor(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    client = app.test_client()

    payload = _payload()
    payload["queue_dir"] = str(tmp_path / "external-queue")
    response = client.post("/api/internal/run-plan/queue/execute", json=payload)

    assert response.status_code == 400
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.queued_job_id == ""
    assert summary.final_job is None
    assert summary.final_lease is None
    assert summary.loop_results == []
    assert summary.error is not None
    assert "queue_dir is not accepted" in summary.error["message"]
    assert calls == []


def test_internal_run_plan_queue_route_rejects_queue_dir_path_escape_without_executor(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    client = app.test_client()

    payload = _payload()
    payload["queue_dir"] = "../external-queue"
    response = client.post("/api/internal/run-plan/queue/execute", json=payload)

    assert response.status_code == 400
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.error is not None
    assert "queue_dir is not accepted" in summary.error["message"]
    assert calls == []
    audit = _audit_records(tmp_path)
    assert audit[-1]["outcome"] == "validation_error"
    assert audit[-1]["error"]["message"] == "queue_dir is not accepted by the internal run-plan queue route"


def test_internal_run_plan_queue_route_rejects_project_id_path_escape_without_executor(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    client = app.test_client()

    payload = _payload()
    payload["project_id"] = "../outside"
    response = client.post("/api/internal/run-plan/queue/execute", json=payload)

    assert response.status_code == 400
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.error is not None
    assert "project_id must be a safe path component" in summary.error["message"]
    assert calls == []


def test_internal_run_plan_queue_route_rejects_project_id_nested_path_without_executor(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    client = app.test_client()

    payload = _payload()
    payload["project_id"] = "a/b"
    response = client.post("/api/internal/run-plan/queue/execute", json=payload)

    assert response.status_code == 400
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.error is not None
    assert "project_id must be a safe path component" in summary.error["message"]
    assert calls == []


def test_internal_run_plan_queue_route_rejects_run_id_path_escape_without_executor(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "../outside", "status": "WAITING_USER"}, calls=calls)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload(run_plan=_run_plan("../outside")))

    assert response.status_code == 400
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.error is not None
    assert "run_id must be a safe path component" in summary.error["message"]
    assert calls == []


def test_internal_run_plan_queue_route_rejects_run_id_nested_path_without_executor(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "a/b", "status": "WAITING_USER"}, calls=calls)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload(run_plan=_run_plan("a/b")))

    assert response.status_code == 400
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.error is not None
    assert "run_id must be a safe path component" in summary.error["message"]
    assert calls == []


def test_internal_run_plan_queue_route_rejects_non_dedicated_queue_without_executor(tmp_path: Path) -> None:
    queue_dir = _default_queue_dir(tmp_path)
    queue = WorkerQueue(JsonWorkerQueueStore(queue_dir))
    queue.enqueue("proj-old", "run-old", {"task_id": "some_other_task"})
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
    _grant_run_plan_queue_permission(tmp_path)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 400
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.terminal is False
    assert summary.queued_job_id == ""
    assert summary.final_job is None
    assert summary.final_lease is None
    assert summary.loop_results == []
    assert summary.error is not None
    assert "empty/dedicated queue" in summary.error["message"]
    assert calls == []


def test_internal_run_plan_queue_route_does_not_replace_default_execute_route(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post("/api/run-plan/execute", json={})

    assert response.status_code == 400
    assert response.get_json() == {"ok": False, "error": "project_id required"}
