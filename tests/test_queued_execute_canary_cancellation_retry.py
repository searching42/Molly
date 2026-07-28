from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller
from ai4s_agent.worker_task_runner import TaskRunResult


def _write_training_csv(path: Path) -> None:
    rows = ["SMILES,plqy,lambda_em,split_group"]
    for idx in range(36):
        split = "train" if idx < 24 else "valid" if idx < 30 else "test"
        rows.append(f"CC{'C' * (idx % 5)}O,{0.45 + idx * 0.01:.3f},{500 + idx},{split}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _expanded_baseline_plan(run_id: str) -> RunPlan:
    return expand_run_plan(run_id=run_id, requested_tasks=["run_baseline"], available_artifacts=[])


def _post_execute(client: Any, *, project_id: str, run_plan: RunPlan, dataset: Path):
    return client.post(
        "/api/run-plan/execute",
        json={
            "project_id": project_id,
            "run_plan": run_plan.model_dump(mode="json"),
            "input_artifacts": {"uploaded_dataset": str(dataset)},
        },
    )


def _default_queue_dir(workspace: Path, project_id: str, run_id: str) -> Path:
    return workspace / ".ai4s_internal" / "run_plan_queues" / project_id / run_id


def _log_messages(client: Any, run_id: str) -> list[str]:
    logs = client.get(f"/api/runs/{run_id}/logs?limit=50")
    assert logs.status_code == 200
    return [entry["message"] for entry in logs.json["logs"]]


class _SpyCancellationRunner:
    def __init__(self) -> None:
        self.start_calls = 0
        self.poll_calls = 0
        self.cancel_calls = 0

    def start(self, job: dict[str, Any]) -> TaskRunResult:
        self.start_calls += 1
        return TaskRunResult(state="running", message="task started", output={"task_id": str(job["task"]["task_id"])})

    def poll(self, job: dict[str, Any]) -> TaskRunResult:
        self.poll_calls += 1
        return TaskRunResult(state="running", message="task still running", output={"task_id": str(job["task"]["task_id"])})

    def cancel(self, job: dict[str, Any]) -> TaskRunResult:
        self.cancel_calls += 1
        return TaskRunResult(state="cancelled", message="task cancelled", output={"task_id": str(job["task"]["task_id"])})


def test_worker_queue_targeted_acquire_skips_cancelled_queued_job(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    cancelled = queue.enqueue("proj-cancel", "r-target", {"task_id": "old_cancelled"})
    target = queue.enqueue("proj-cancel", "r-target", {"task_id": "target"})

    cancelled_state = queue.cancel(cancelled["job_id"])
    acquired = queue.acquire(
        "worker-target",
        target_project_id="proj-cancel",
        target_run_id="r-target",
    )

    assert cancelled_state["status"] == "cancelled"
    assert cancelled_state["cancellation_requested"] is True
    assert acquired is not None
    assert acquired["job_id"] == target["job_id"]
    assert acquired["status"] == "running"
    current_cancelled = queue.status(cancelled["job_id"])
    assert current_cancelled is not None
    assert current_cancelled["status"] == "cancelled"
    assert current_cancelled["cancellation_requested"] is True


def test_worker_queue_poller_cancellation_calls_cancel_without_poll_or_heartbeat(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    queued = queue.enqueue("proj-cancel", "r-cancel", {"task_id": "train_model"})
    runner = _SpyCancellationRunner()
    poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=runner)

    acquired = poller.poll_once(now="2026-01-01T00:00:00Z")
    assert acquired.acquired_job is not None
    original_heartbeat = acquired.acquired_job["heartbeat_at"]
    poller.cancel(queued["job_id"], now="2026-01-01T00:00:01Z")

    result = poller.poll_once(now="2026-01-01T00:00:10Z")

    assert result.action == "cancelled"
    assert result.cancellation_requested is True
    assert result.runner_result == TaskRunResult(
        state="cancelled",
        message="task cancelled",
        output={"task_id": "train_model"},
    )
    assert runner.start_calls == 1
    assert runner.poll_calls == 0
    assert runner.cancel_calls == 1
    status = queue.status(queued["job_id"])
    assert status is not None
    assert status["status"] == "failed"
    assert status["heartbeat_at"] == original_heartbeat
    assert status["error"] == {"reason": "cancelled"}


def test_run_plan_execute_queued_canary_does_not_consume_cancelled_old_job(tmp_path: Path) -> None:
    project_id = "proj-canary-cancelled-old"
    run_id = "r-canary-cancelled-old"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, run_id)))
    old_job = queue.enqueue(project_id, run_id, {"task_id": "old_job"})
    queue.cancel(old_job["job_id"])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    client = app.test_client()

    response = _post_execute(
        client,
        project_id=project_id,
        run_plan=_expanded_baseline_plan(run_id),
        dataset=dataset,
    )

    assert response.status_code == 200
    payload = response.json
    assert payload["ok"] is True
    assert payload["execution_backend"] == "queued_canary"
    assert payload["queue_summary"]["ok"] is True
    assert payload["queue_summary"]["queued_job_id"] != old_job["job_id"]
    assert payload["queue_summary"]["final_job"]["job_id"] == payload["queue_summary"]["queued_job_id"]
    assert payload["queue_summary"]["final_job"]["status"] == "succeeded"
    assert payload["queue_summary"]["final_lease"]["status"] == "completed"
    old_status = queue.status(old_job["job_id"])
    assert old_status is not None
    assert old_status["status"] == "cancelled"
    assert old_status["cancellation_requested"] is True

    registry = ProjectStorage(tmp_path).read_artifact_registry(project_id, run_id)
    for artifact_id in [
        "dataset_profile",
        "cleaned_train_dataset",
        "trainability_report",
        "baseline_report",
    ]:
        assert artifact_id in registry


def test_run_plan_execute_sync_fallback_does_not_touch_cancelled_queue_job(tmp_path: Path) -> None:
    project_id = "proj-sync-fallback-cancelled-old"
    run_id = "r-sync-fallback-cancelled-old"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, run_id)))
    old_job = queue.enqueue(project_id, run_id, {"task_id": "old_job"})
    queue.cancel(old_job["job_id"])
    run_plan = expand_run_plan(run_id=run_id, requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True

    def fail_if_queued(_storage: ProjectStorage) -> object:
        raise AssertionError("train_model must not use queued canary")

    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = fail_if_queued
    client = app.test_client()

    response = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)

    assert response.status_code == 200
    assert response.json["execution"]["status"] == RunStatus.WAITING_USER.value
    assert "execution_backend" not in response.json
    assert "queue_summary" not in response.json
    old_status = queue.status(old_job["job_id"])
    assert old_status is not None
    assert old_status["status"] == "cancelled"
    assert old_status["cancellation_requested"] is True
    messages = _log_messages(client, run_id)
    assert any("RunPlan execution backend: sync_fallback_not_allowlisted" in message for message in messages)
    assert any("disallowed_tasks=train_model" in message for message in messages)
