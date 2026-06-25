from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


KEY_ARTIFACTS = {
    "dataset_profile",
    "property_catalog",
    "cleaned_train_dataset",
    "cleaning_rules",
    "trainability_report",
    "baseline_report",
    "backend_recommendation",
}


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


def _assert_successful_queued_response(payload: dict[str, Any], *, project_id: str, run_id: str) -> dict[str, Any]:
    assert payload["ok"] is True
    assert payload["execution_backend"] == "queued_canary"
    assert payload["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert payload["queue_summary"]["ok"] is True
    assert payload["queue_summary"]["queued_job_id"] == payload["queue_summary"]["final_job"]["job_id"]
    assert payload["queue_summary"]["final_job"]["status"] == "succeeded"
    assert payload["queue_summary"]["final_job"]["project_id"] == project_id
    assert payload["queue_summary"]["final_job"]["run_id"] == run_id
    assert payload["queue_summary"]["final_lease"]["status"] == "completed"
    return payload["queue_summary"]["final_job"]


def test_run_plan_execute_queued_canary_repeated_runs_are_isolated_and_stable(tmp_path: Path) -> None:
    project_id = "proj-repeat-canary"
    run_ids = ["r-repeat-canary-1", "r-repeat-canary-2", "r-repeat-canary-3"]
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    client = app.test_client()

    responses: list[dict[str, Any]] = []
    final_job_ids: set[str] = set()
    queue_dirs: set[Path] = set()
    for run_id in run_ids:
        response = _post_execute(client, project_id=project_id, run_plan=_expanded_baseline_plan(run_id), dataset=dataset)
        assert response.status_code == 200
        payload = response.json
        responses.append(payload)
        final_job = _assert_successful_queued_response(payload, project_id=project_id, run_id=run_id)
        final_job_ids.add(final_job["job_id"])
        queue_dir = _default_queue_dir(tmp_path, project_id, run_id)
        queue_dirs.add(queue_dir)
        assert queue_dir.exists()
        assert (queue_dir / "worker_queue.json").exists()

    assert len(final_job_ids) == len(run_ids)
    assert len(queue_dirs) == len(run_ids)
    assert len({frozenset(response.keys()) for response in responses}) == 1
    assert len({frozenset(response["queue_summary"].keys()) for response in responses}) == 1
    assert len({tuple(response["execution"]["executed_tasks"]) for response in responses}) == 1

    storage = ProjectStorage(tmp_path)
    registry_key_sets: list[set[str]] = []
    for run_id in run_ids:
        paths = _registry_paths(storage, project_id, run_id)
        assert paths
        registry_key_sets.append(set(paths))
        assert KEY_ARTIFACTS <= set(paths)
        for artifact_id, path in paths.items():
            assert path.exists(), artifact_id

    assert all(keys == registry_key_sets[0] for keys in registry_key_sets)


def test_run_plan_execute_repeated_canary_runs_do_not_consume_old_jobs(tmp_path: Path) -> None:
    project_id = "proj-repeat-canary-old-jobs"
    run_ids = ["r-repeat-old-1", "r-repeat-old-2"]
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    client = app.test_client()

    old_jobs: dict[str, tuple[WorkerQueue, dict[str, Any]]] = {}
    for run_id in run_ids:
        queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, run_id)))
        old_jobs[run_id] = (queue, queue.enqueue(project_id, run_id, {"task_id": "old_job"}))

    storage = ProjectStorage(tmp_path)
    for run_id in run_ids:
        response = _post_execute(client, project_id=project_id, run_plan=_expanded_baseline_plan(run_id), dataset=dataset)
        assert response.status_code == 200
        payload = response.json
        final_job = _assert_successful_queued_response(payload, project_id=project_id, run_id=run_id)
        queue, old_job = old_jobs[run_id]
        assert final_job["job_id"] != old_job["job_id"]
        assert queue.status(old_job["job_id"])["status"] == "queued"
        paths = _registry_paths(storage, project_id, run_id)
        assert KEY_ARTIFACTS <= set(paths)
        assert paths["baseline_report"].exists()


def test_run_plan_execute_repeated_canary_rollback_to_sync_does_not_touch_queued_jobs(tmp_path: Path) -> None:
    project_id = "proj-repeat-canary-rollback"
    queued_run_id = "r-repeat-queued-before-rollback"
    old_job_run_id = "r-repeat-old-during-rollback"
    sync_run_id = "r-repeat-sync-after-rollback"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    queued_response = _post_execute(
        client,
        project_id=project_id,
        run_plan=_expanded_baseline_plan(queued_run_id),
        dataset=dataset,
    )
    assert queued_response.status_code == 200
    _assert_successful_queued_response(queued_response.json, project_id=project_id, run_id=queued_run_id)
    queued_job_id = queued_response.json["queue_summary"]["queued_job_id"]

    old_queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, old_job_run_id)))
    old_job = old_queue.enqueue(project_id, old_job_run_id, {"task_id": "old_job"})

    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = False
    sync_response = _post_execute(client, project_id=project_id, run_plan=_expanded_baseline_plan(sync_run_id), dataset=dataset)

    assert sync_response.status_code == 200
    assert sync_response.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert "execution_backend" not in sync_response.json
    assert "queue_summary" not in sync_response.json
    assert not _default_queue_dir(tmp_path, project_id, sync_run_id).exists()
    assert old_queue.status(old_job["job_id"])["status"] == "queued"

    queued_queue = WorkerQueue(JsonWorkerQueueStore(_default_queue_dir(tmp_path, project_id, queued_run_id)))
    assert queued_queue.status(queued_job_id)["status"] == "succeeded"


def test_repeated_run_stability_does_not_put_train_model_on_queued_canary(tmp_path: Path) -> None:
    project_id = "proj-repeat-canary-train"
    run_ids = ["r-repeat-train-1", "r-repeat-train-2"]
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True

    def fail_if_queued(storage: ProjectStorage) -> object:
        raise AssertionError("train_model must not use queued canary")

    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = fail_if_queued
    client = app.test_client()

    for run_id in run_ids:
        run_plan = expand_run_plan(run_id=run_id, requested_tasks=["train_model"], available_artifacts=[])
        response = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)
        assert response.status_code == 200
        assert response.json["execution"]["status"] == RunStatus.WAITING_USER.value
        assert "execution_backend" not in response.json
        assert "queue_summary" not in response.json
        assert not _default_queue_dir(tmp_path, project_id, run_id).exists()
        messages = _log_messages(client, run_id)
        assert any("sync_fallback_not_allowlisted" in message for message in messages)
        assert any("train_model" in message for message in messages)
