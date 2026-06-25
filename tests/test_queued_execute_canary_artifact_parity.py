from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


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


def _registry_paths(storage: ProjectStorage, project_id: str, run_id: str) -> dict[str, Path]:
    registry = storage.read_artifact_registry(project_id, run_id)
    run_dir = storage.run_dir(project_id, run_id)
    return {artifact_id: run_dir / relative_path for artifact_id, relative_path in registry.items()}


def _log_messages(client: Any, run_id: str) -> list[str]:
    logs = client.get(f"/api/runs/{run_id}/logs?limit=50")
    assert logs.status_code == 200
    return [entry["message"] for entry in logs.json["logs"]]


def test_run_plan_execute_allowlisted_chain_artifact_registry_matches_sync_and_queued_canary(tmp_path: Path) -> None:
    project_id = "proj-artifact-parity"
    sync_run_id = "r-artifact-parity-sync"
    queued_run_id = "r-artifact-parity-queued"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    sync_plan = _expanded_baseline_plan(sync_run_id)
    queued_plan = _expanded_baseline_plan(queued_run_id)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    sync_resp = _post_execute(client, project_id=project_id, run_plan=sync_plan, dataset=dataset)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    queued_resp = _post_execute(client, project_id=project_id, run_plan=queued_plan, dataset=dataset)

    assert sync_resp.status_code == 200
    assert sync_resp.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert "execution_backend" not in sync_resp.json
    assert "queue_summary" not in sync_resp.json
    assert not _default_queue_dir(tmp_path, project_id, sync_run_id).exists()

    assert queued_resp.status_code == 200
    assert queued_resp.json["execution_backend"] == "queued_canary"
    assert queued_resp.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert queued_resp.json["queue_summary"]["ok"] is True
    assert queued_resp.json["queue_summary"]["final_job"]["status"] == "succeeded"
    assert queued_resp.json["queue_summary"]["final_job"]["project_id"] == project_id
    assert queued_resp.json["queue_summary"]["final_job"]["run_id"] == queued_run_id
    assert queued_resp.json["queue_summary"]["final_lease"]["status"] == "completed"

    assert sync_resp.json["execution"]["executed_tasks"] == queued_resp.json["execution"]["executed_tasks"]
    assert "failed_task" not in sync_resp.json["execution"]
    assert "failed_task" not in queued_resp.json["execution"]

    storage = ProjectStorage(tmp_path)
    sync_paths = _registry_paths(storage, project_id, sync_run_id)
    queued_paths = _registry_paths(storage, project_id, queued_run_id)

    assert set(sync_paths) == set(queued_paths)
    for artifact_id in [
        "dataset_profile",
        "property_catalog",
        "cleaned_train_dataset",
        "cleaning_rules",
        "trainability_report",
        "baseline_report",
        "backend_recommendation",
    ]:
        assert artifact_id in sync_paths

    for artifact_id, sync_path in sync_paths.items():
        assert sync_path.exists(), artifact_id
        assert queued_paths[artifact_id].exists(), artifact_id


def test_run_plan_execute_artifact_parity_canary_does_not_consume_existing_job(tmp_path: Path) -> None:
    project_id = "proj-artifact-parity-existing"
    sync_run_id = "r-artifact-parity-existing-sync"
    queued_run_id = "r-artifact-parity-existing-queued"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    sync_plan = _expanded_baseline_plan(sync_run_id)
    queued_plan = _expanded_baseline_plan(queued_run_id)
    queue_dir = _default_queue_dir(tmp_path, project_id, queued_run_id)
    queue = WorkerQueue(JsonWorkerQueueStore(queue_dir))
    old_job = queue.enqueue(project_id, queued_run_id, {"task_id": "old_job"})
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    sync_resp = _post_execute(client, project_id=project_id, run_plan=sync_plan, dataset=dataset)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    queued_resp = _post_execute(client, project_id=project_id, run_plan=queued_plan, dataset=dataset)

    assert sync_resp.status_code == 200
    assert queued_resp.status_code == 200
    assert queued_resp.json["queue_summary"]["queued_job_id"] != old_job["job_id"]
    assert queue.status(old_job["job_id"])["status"] == "queued"

    storage = ProjectStorage(tmp_path)
    sync_paths = _registry_paths(storage, project_id, sync_run_id)
    queued_paths = _registry_paths(storage, project_id, queued_run_id)
    assert set(sync_paths) == set(queued_paths)
    assert queued_paths["baseline_report"].exists()


def test_artifact_parity_fixture_does_not_put_train_model_on_queued_canary(tmp_path: Path) -> None:
    project_id = "proj-artifact-parity-train"
    run_id = "r-artifact-parity-train"
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
