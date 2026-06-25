from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


def _write_invalid_training_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not_smiles,wrong_property\nabc,123\n", encoding="utf-8")


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


def _failure_summary(payload: dict[str, Any]) -> dict[str, str]:
    execution = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    execution_error = execution.get("error") if isinstance(execution.get("error"), dict) else {}
    result = execution.get("result") if isinstance(execution.get("result"), dict) else {}
    result_error = result.get("error") if isinstance(result.get("error"), dict) else {}
    message = str(
        execution_error.get("message")
        or result_error.get("message")
        or result_error.get("stderr")
        or execution.get("message")
        or error.get("message")
        or payload.get("error")
        or ""
    )
    status = str(execution.get("status") or payload.get("status") or "").upper()
    failed_task = str(execution.get("failed_task") or error.get("failed_task") or "")
    assert payload.get("ok") is False or status == RunStatus.FAILED.value
    assert message
    return {"status": status, "failed_task": failed_task, "message": message}


def test_run_plan_execute_allowlisted_chain_failure_classification_matches_sync_and_queued_canary(
    tmp_path: Path,
) -> None:
    project_id = "proj-failure-parity"
    sync_run_id = "r-failure-parity-sync"
    queued_run_id = "r-failure-parity-queued"
    dataset = tmp_path / "input" / "invalid.csv"
    _write_invalid_training_csv(dataset)
    sync_plan = _expanded_baseline_plan(sync_run_id)
    queued_plan = _expanded_baseline_plan(queued_run_id)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    sync_resp = _post_execute(client, project_id=project_id, run_plan=sync_plan, dataset=dataset)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    queued_resp = _post_execute(client, project_id=project_id, run_plan=queued_plan, dataset=dataset)

    assert "execution_backend" not in sync_resp.json
    assert "queue_summary" not in sync_resp.json
    assert not _default_queue_dir(tmp_path, project_id, sync_run_id).exists()

    assert queued_resp.status_code == 400
    assert queued_resp.json["ok"] is False
    assert queued_resp.json["execution_backend"] == "queued_canary"
    assert queued_resp.json["queue_summary"]["ok"] is False
    assert queued_resp.json["queue_summary"]["final_job"]["status"] == "failed"
    assert queued_resp.json["queue_summary"]["final_job"]["project_id"] == project_id
    assert queued_resp.json["queue_summary"]["final_job"]["run_id"] == queued_run_id

    sync_failure = _failure_summary(sync_resp.json)
    queued_failure = _failure_summary(queued_resp.json)
    assert queued_failure["status"] == RunStatus.FAILED.value
    if sync_failure["status"]:
        assert sync_failure["status"] == queued_failure["status"]
    if sync_failure["failed_task"]:
        assert sync_failure["failed_task"] == queued_failure["failed_task"]
    assert "clean_dataset.py" in sync_failure["message"]
    assert "clean_dataset.py" in queued_failure["message"]
    assert not queued_resp.json["queue_summary"]["final_job"].get("result")


def test_run_plan_execute_failed_canary_does_not_consume_existing_job(tmp_path: Path) -> None:
    project_id = "proj-failure-parity-existing"
    run_id = "r-failure-parity-existing"
    dataset = tmp_path / "input" / "invalid.csv"
    _write_invalid_training_csv(dataset)
    run_plan = _expanded_baseline_plan(run_id)
    queue_dir = _default_queue_dir(tmp_path, project_id, run_id)
    queue = WorkerQueue(JsonWorkerQueueStore(queue_dir))
    old_job = queue.enqueue(project_id, run_id, {"task_id": "old_job"})
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    client = app.test_client()

    resp = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)

    assert resp.status_code == 400
    assert resp.json["ok"] is False
    assert resp.json["queue_summary"]["queued_job_id"] != old_job["job_id"]
    assert queue.status(old_job["job_id"])["status"] == "queued"
    assert resp.json["queue_summary"]["final_job"]["job_id"] == resp.json["queue_summary"]["queued_job_id"]
    assert resp.json["queue_summary"]["final_job"]["status"] == "failed"


def test_run_plan_execute_sync_failure_does_not_create_queue_files(tmp_path: Path) -> None:
    project_id = "proj-failure-parity-sync-only"
    run_id = "r-failure-parity-sync-only"
    dataset = tmp_path / "input" / "invalid.csv"
    _write_invalid_training_csv(dataset)
    run_plan = _expanded_baseline_plan(run_id)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    resp = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)

    _failure_summary(resp.json)
    assert "execution_backend" not in resp.json
    assert "queue_summary" not in resp.json
    assert not _default_queue_dir(tmp_path, project_id, run_id).exists()


def test_failure_parity_fixture_does_not_put_train_model_on_queued_canary(tmp_path: Path) -> None:
    project_id = "proj-failure-parity-train"
    run_id = "r-failure-parity-train"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id=run_id, requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True

    def fail_if_queued(storage: ProjectStorage) -> object:
        raise AssertionError("train_model must not use queued canary")

    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = fail_if_queued
    client = app.test_client()

    resp = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)

    assert resp.status_code == 200
    assert resp.json["execution"]["status"] == RunStatus.WAITING_USER.value
    assert "execution_backend" not in resp.json
    assert "queue_summary" not in resp.json
    assert not _default_queue_dir(tmp_path, project_id, run_id).exists()
    messages = _log_messages(client, run_id)
    assert any("sync_fallback_not_allowlisted" in message for message in messages)
    assert any("train_model" in message for message in messages)
