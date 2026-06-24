from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.run_plan_queue_summary import RunPlanQueueExecutionSummary
from ai4s_agent.schemas import PlannedTask, RunPlan
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


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


def _payload(*, run_plan: RunPlan | None = None) -> dict[str, Any]:
    return {
        "project_id": "proj-a",
        "run_plan": (run_plan or _run_plan()).model_dump(mode="json"),
        "input_artifacts": {"dataset": "datasets/input.csv"},
        "task_options": {"train_model": {"epochs": 1}},
        "max_iterations": 10,
    }


def _default_queue_dir(workspace: Path, project_id: str = "proj-a", run_id: str = "run-a") -> Path:
    return workspace / ".ai4s_internal" / "run_plan_queues" / project_id / run_id


def _enable_queue_route(app, *, execution: dict[str, Any], calls: list[dict[str, Any]]) -> None:
    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return FakeRunPlanExecutor(execution, calls)

    app.config["AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = factory


def test_internal_run_plan_queue_route_is_disabled_by_default(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.post("/api/internal/run-plan/queue/execute", json=_payload())

    assert response.status_code == 404


def test_internal_run_plan_queue_route_executes_when_feature_flag_enabled(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    _enable_queue_route(app, execution={"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls=calls)
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


def test_internal_run_plan_queue_route_can_be_enabled_by_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE", "1")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []

    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"}, calls)

    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = factory
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

    response = client.post("/api/internal/run-plan/queue/execute", json={"project_id": "proj-a"})

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
