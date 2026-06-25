from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue_retry import enqueue_queued_canary_retry
from ai4s_agent.schemas import RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


class FakeRunPlanExecutor:
    def __init__(self, executions: dict[str, dict[str, Any]], calls: list[dict[str, Any]]) -> None:
        self.executions = {key: dict(value) for key, value in executions.items()}
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
                "run_id": run_plan.run_id,
                "input_artifacts": dict(input_artifacts or {}),
                "task_options": dict(task_options or {}),
            }
        )
        return dict(self.executions[run_plan.run_id])


def _fake_executor_factory(executions: dict[str, dict[str, Any]], calls: list[dict[str, Any]]):
    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return FakeRunPlanExecutor(executions, calls)

    return factory


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


def _queue_snapshot(queue: WorkerQueue) -> dict[str, Any]:
    jobs = sorted(queue.list_jobs(), key=lambda job: str(job.get("job_id") or ""))
    leases = sorted(queue.list_leases(), key=lambda lease: str(lease.get("lease_id") or ""))
    return {
        "jobs": json.loads(json.dumps(jobs, sort_keys=True)),
        "leases": json.loads(json.dumps(leases, sort_keys=True)),
    }


def _log_messages(client: Any, run_id: str) -> list[str]:
    logs = client.get(f"/api/runs/{run_id}/logs?limit=100")
    assert logs.status_code == 200
    return [entry["message"] for entry in logs.json["logs"]]


def _queued_canary_telemetry(client: Any, run_id: str) -> list[dict[str, Any]]:
    prefix = "RunPlan queued canary telemetry: "
    payloads: list[dict[str, Any]] = []
    for message in _log_messages(client, run_id):
        if message.startswith(prefix):
            payloads.append(json.loads(message[len(prefix):]))
    return payloads


def test_operational_rollback_flag_off_routes_new_requests_to_sync_without_mutating_existing_queue_state(
    tmp_path: Path,
) -> None:
    project_id = "proj-rollback-drill"
    success_run_id = "r-rollback-success-before-flag-off"
    failed_run_id = "r-rollback-failed-before-flag-off"
    old_queued_run_id = "r-rollback-old-queued"
    stale_run_id = "r-rollback-stale-lease"
    sync_run_id = "r-rollback-sync-after-flag-off"
    reenabled_run_id = "r-rollback-queued-after-reenable"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)

    executions = {
        success_run_id: {
            "ok": True,
            "run_id": success_run_id,
            "status": RunStatus.SUCCEEDED.value,
            "executed_tasks": ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"],
        },
        failed_run_id: {
            "ok": False,
            "run_id": failed_run_id,
            "status": RunStatus.FAILED.value,
            "failed_task": "run_baseline",
            "error": {"message": "simulated canary failure"},
        },
        reenabled_run_id: {
            "ok": True,
            "run_id": reenabled_run_id,
            "status": RunStatus.SUCCEEDED.value,
            "executed_tasks": ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"],
        },
    }
    calls: list[dict[str, Any]] = []
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(executions, calls)
    client = app.test_client()

    success_response = _post_execute(
        client,
        project_id=project_id,
        run_plan=_expanded_baseline_plan(success_run_id),
        dataset=dataset,
    )
    assert success_response.status_code == 200
    assert success_response.json["execution_backend"] == "queued_canary"
    assert success_response.json["queue_summary"]["ok"] is True

    failed_response = _post_execute(
        client,
        project_id=project_id,
        run_plan=_expanded_baseline_plan(failed_run_id),
        dataset=dataset,
    )
    assert failed_response.status_code == 400
    assert failed_response.json["execution_backend"] == "queued_canary"
    assert failed_response.json["queue_summary"]["ok"] is False
    failed_queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, failed_run_id)))
    failed_source_job_id = failed_response.json["queue_summary"]["queued_job_id"]
    failed_source_before_retry = failed_queue.status(failed_source_job_id)
    assert failed_source_before_retry is not None
    retry_child = enqueue_queued_canary_retry(
        failed_queue,
        source_job_id=failed_source_job_id,
        retry_request_id="retry-rollback-001",
        actor="operator",
        reason="preserve failed job for rollback drill",
        now="2026-01-01T00:20:00Z",
    )
    assert retry_child["status"] == "queued"

    old_queued_queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, old_queued_run_id)))
    old_queued_job = old_queued_queue.enqueue(project_id, old_queued_run_id, {"task_id": "old_queued_job"})

    stale_queue = WorkerQueue(
        JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, stale_run_id)),
        lease_ttl_sec=10,
    )
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stale_job = stale_queue.enqueue(project_id, stale_run_id, {"task_id": "old_stale_job"}, created_at=_iso(base))
    stale_acquired = stale_queue.acquire("worker-stale", now=_iso(base))
    assert stale_acquired is not None
    stale_queue.recover_stale_leases(now=_iso(base + timedelta(seconds=11)))

    success_queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, success_run_id)))
    pre_rollback_snapshots = {
        success_run_id: _queue_snapshot(success_queue),
        failed_run_id: _queue_snapshot(failed_queue),
        old_queued_run_id: _queue_snapshot(old_queued_queue),
        stale_run_id: _queue_snapshot(stale_queue),
    }

    failed_telemetry = _queued_canary_telemetry(client, failed_run_id)
    assert len(failed_telemetry) == 1
    assert failed_telemetry[0]["execution_backend"] == "queued_canary"
    assert failed_telemetry[0]["ok"] is False
    assert failed_telemetry[0]["final_job_status"] == "failed"

    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = False
    sync_response = _post_execute(
        client,
        project_id=project_id,
        run_plan=_expanded_baseline_plan(sync_run_id),
        dataset=dataset,
    )

    assert sync_response.status_code == 200
    assert sync_response.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert "execution_backend" not in sync_response.json
    assert "queue_summary" not in sync_response.json
    assert not _default_queue_dir(tmp_path, project_id, sync_run_id).exists()
    sync_messages = _log_messages(client, sync_run_id)
    assert any("RunPlan execution backend: sync" in message for message in sync_messages)

    post_rollback_snapshots = {
        success_run_id: _queue_snapshot(success_queue),
        failed_run_id: _queue_snapshot(failed_queue),
        old_queued_run_id: _queue_snapshot(old_queued_queue),
        stale_run_id: _queue_snapshot(stale_queue),
    }
    assert post_rollback_snapshots == pre_rollback_snapshots

    failed_source_after = failed_queue.status(failed_source_job_id)
    assert failed_source_after == failed_source_before_retry
    assert failed_queue.status(retry_child["job_id"])["status"] == "queued"  # type: ignore[index]
    assert old_queued_queue.status(old_queued_job["job_id"])["status"] == "queued"  # type: ignore[index]
    assert stale_queue.status(stale_job["job_id"])["status"] == "queued"  # type: ignore[index]
    assert stale_queue.lease_status(stale_acquired["lease_id"])["status"] == "stale"  # type: ignore[index]
    assert len([job for job in failed_queue.list_jobs() if job.get("retry_of_job_id") == failed_source_job_id]) == 1

    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    reenabled_response = _post_execute(
        client,
        project_id=project_id,
        run_plan=_expanded_baseline_plan(reenabled_run_id),
        dataset=dataset,
    )

    assert reenabled_response.status_code == 200
    assert reenabled_response.json["execution_backend"] == "queued_canary"
    assert reenabled_response.json["queue_summary"]["ok"] is True
    reenabled_job_id = reenabled_response.json["queue_summary"]["queued_job_id"]
    assert reenabled_job_id not in {
        failed_source_job_id,
        retry_child["job_id"],
        old_queued_job["job_id"],
        stale_job["job_id"],
    }
    assert _queue_snapshot(failed_queue) == pre_rollback_snapshots[failed_run_id]
    assert _queue_snapshot(old_queued_queue) == pre_rollback_snapshots[old_queued_run_id]
    assert _queue_snapshot(stale_queue) == pre_rollback_snapshots[stale_run_id]

    assert len(calls) == 3
    assert [call["run_id"] for call in calls] == [success_run_id, failed_run_id, reenabled_run_id]
