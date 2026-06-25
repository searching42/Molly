from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


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


def _stale_running_job(
    queue: WorkerQueue,
    *,
    project_id: str,
    run_id: str,
    task_id: str,
    acquired_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    job = queue.enqueue(project_id, run_id, {"task_id": task_id}, created_at=acquired_at)
    acquired = queue.acquire("worker-stale", now=acquired_at)
    assert acquired is not None
    return job, acquired


def test_worker_queue_recovers_stale_running_job_before_targeted_canary_acquire(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path), lease_ttl_sec=10)
    base = _iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    future = _iso(datetime(2026, 1, 1, 0, 0, 11, tzinfo=timezone.utc))
    old_job, old_acquired = _stale_running_job(
        queue,
        project_id="proj-recovery",
        run_id="r-old-stale",
        task_id="old_stale_job",
        acquired_at=base,
    )
    target_job = queue.enqueue("proj-recovery", "r-target", {"task_id": "target_job"}, created_at=base)

    acquired = queue.acquire(
        "worker-target",
        now=future,
        target_project_id="proj-recovery",
        target_run_id="r-target",
    )

    assert acquired is not None
    assert acquired["job_id"] == target_job["job_id"]
    assert acquired["run_id"] == "r-target"
    assert acquired["status"] == "running"
    old_status = queue.status(old_job["job_id"])
    assert old_status is not None
    assert old_status["status"] == "queued"
    stale_lease = queue.lease_status(old_acquired["lease_id"])
    assert stale_lease is not None
    assert stale_lease["status"] == "stale"


def test_worker_queue_poller_recovers_stale_lease_but_acquires_only_target_job(tmp_path: Path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path), lease_ttl_sec=10)
    base = _iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    future = _iso(datetime(2026, 1, 1, 0, 0, 11, tzinfo=timezone.utc))
    old_job, _old_acquired = _stale_running_job(
        queue,
        project_id="proj-recovery",
        run_id="r-old-stale",
        task_id="old_stale_job",
        acquired_at=base,
    )
    target_job = queue.enqueue("proj-recovery", "r-target", {"task_id": "target_job"}, created_at=base)
    poller = WorkerQueuePoller(
        queue,
        worker_id="worker-target",
        target_project_id="proj-recovery",
        target_run_id="r-target",
    )

    result = poller.poll_once(now=future)

    assert result.action == "acquired"
    assert result.recovered_job_ids == [old_job["job_id"]]
    assert result.acquired_job is not None
    assert result.acquired_job["job_id"] == target_job["job_id"]
    assert result.active_lease is not None
    assert result.active_lease["job_id"] == target_job["job_id"]
    old_status = queue.status(old_job["job_id"])
    assert old_status is not None
    assert old_status["status"] == "queued"


def test_run_plan_execute_queued_canary_ignores_recovered_stale_old_job_and_runs_new_target_job(tmp_path: Path) -> None:
    project_id = "proj-canary-recovery"
    run_id = "r-canary-recovery"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, run_id)), lease_ttl_sec=10)
    base = _iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    old_job, old_acquired = _stale_running_job(
        queue,
        project_id=project_id,
        run_id=run_id,
        task_id="old_stale_job",
        acquired_at=base,
    )
    old_lease_id = old_acquired["lease_id"]
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
    assert payload["execution_backend"] == "queued_canary"
    assert payload["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert payload["queue_summary"]["ok"] is True
    assert payload["queue_summary"]["queued_job_id"] != old_job["job_id"]
    assert payload["queue_summary"]["final_job"]["job_id"] == payload["queue_summary"]["queued_job_id"]
    assert payload["queue_summary"]["final_job"]["run_id"] == run_id
    assert payload["queue_summary"]["final_job"]["status"] == "succeeded"
    assert payload["queue_summary"]["final_lease"]["status"] == "completed"
    old_status = queue.status(old_job["job_id"])
    assert old_status is not None
    assert old_status["status"] == "queued"
    old_lease = queue.lease_status(old_lease_id)
    assert old_lease is not None
    assert old_lease["status"] == "stale"

    registry = ProjectStorage(tmp_path).read_artifact_registry(project_id, run_id)
    for artifact_id in [
        "dataset_profile",
        "cleaned_train_dataset",
        "trainability_report",
        "baseline_report",
    ]:
        assert artifact_id in registry


def test_run_plan_execute_sync_fallback_ignores_existing_stale_queue_state(tmp_path: Path) -> None:
    project_id = "proj-sync-recovery"
    run_id = "r-sync-recovery"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, run_id)), lease_ttl_sec=10)
    base = _iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    old_job, old_acquired = _stale_running_job(
        queue,
        project_id=project_id,
        run_id=run_id,
        task_id="old_stale_job",
        acquired_at=base,
    )
    old_lease_id = old_acquired["lease_id"]
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = _post_execute(
        client,
        project_id=project_id,
        run_plan=_expanded_baseline_plan(run_id),
        dataset=dataset,
    )

    assert response.status_code == 200
    assert response.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert "execution_backend" not in response.json
    assert "queue_summary" not in response.json
    old_status = queue.status(old_job["job_id"])
    assert old_status is not None
    assert old_status["status"] == "running"
    old_lease = queue.lease_status(old_lease_id)
    assert old_lease is not None
    assert old_lease["status"] == "active"


def test_run_plan_execute_train_model_sync_fallback_ignores_existing_stale_queue_state(tmp_path: Path) -> None:
    project_id = "proj-train-recovery"
    run_id = "r-train-recovery"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, run_id)), lease_ttl_sec=10)
    base = _iso(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
    old_job, old_acquired = _stale_running_job(
        queue,
        project_id=project_id,
        run_id=run_id,
        task_id="old_stale_job",
        acquired_at=base,
    )
    old_lease_id = old_acquired["lease_id"]
    run_plan = expand_run_plan(run_id=run_id, requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True

    def fail_if_queued(_storage: Any) -> object:
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
    assert old_status["status"] == "running"
    old_lease = queue.lease_status(old_lease_id)
    assert old_lease is not None
    assert old_lease["status"] == "active"
