from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent.app import create_app
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.schemas import GateName, RunPlan, RunStatus
from ai4s_agent.storage import ProjectStorage


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


def _fake_executor_factory(execution: dict[str, Any], calls: list[dict[str, Any]]):
    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return FakeRunPlanExecutor(execution, calls)

    return factory


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


def _log_messages(client: Any, run_id: str) -> list[str]:
    logs = client.get(f"/api/runs/{run_id}/logs?limit=100")
    assert logs.status_code == 200
    return [entry["message"] for entry in logs.json["logs"]]


def _queued_canary_telemetry(client: Any, run_id: str) -> list[dict[str, Any]]:
    prefix = "RunPlan queued canary telemetry: "
    parsed: list[dict[str, Any]] = []
    for message in _log_messages(client, run_id):
        if message.startswith(prefix):
            parsed.append(json.loads(message[len(prefix):]))
    return parsed


def test_queued_execute_canary_emits_minimal_structured_telemetry_for_allowlisted_run(tmp_path: Path) -> None:
    run_id = "r-canary-telemetry"
    project_id = "proj-canary-telemetry"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    run_plan = _expanded_baseline_plan(run_id)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {
            "ok": True,
            "run_id": run_id,
            "status": RunStatus.SUCCEEDED.value,
            "executed_tasks": ["inspect_dataset", "clean_dataset", "check_trainability", "run_baseline"],
        },
        calls,
    )
    client = app.test_client()

    response = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)

    assert response.status_code == 200
    assert response.json["execution_backend"] == "queued_canary"
    telemetry = _queued_canary_telemetry(client, run_id)
    assert len(telemetry) == 1
    event = telemetry[0]
    assert event["project_id"] == project_id
    assert event["run_id"] == run_id
    assert event["execution_backend"] == "queued_canary"
    assert event["queued_job_id"] == response.json["queue_summary"]["queued_job_id"]
    assert event["job_id"] == response.json["queue_summary"]["final_job"]["job_id"]
    assert event["lease_id"] == response.json["queue_summary"]["final_lease"]["lease_id"]
    assert event["worker_id"] == response.json["queue_summary"]["final_lease"]["worker_id"]
    assert event["ok"] is True
    assert event["terminal"] is True
    assert event["final_job_status"] == "succeeded"
    assert event["final_lease_status"] == "completed"
    assert event["waiting_user"] is False
    assert event["waiting_task"] == ""
    assert event["required_gates"] == []
    assert event["loop_results"] == response.json["queue_summary"]["loop_results"]
    assert len(calls) == 1


def test_queued_execute_canary_telemetry_surfaces_waiting_user_metadata(tmp_path: Path) -> None:
    run_id = "r-canary-telemetry-waiting"
    project_id = "proj-canary-telemetry-waiting"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    run_plan = _expanded_baseline_plan(run_id)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {
            "ok": True,
            "run_id": run_id,
            "status": RunStatus.WAITING_USER.value,
            "waiting_task": "run_baseline",
            "required_gates": [GateName.TRAIN_CONFIG.value],
        },
        calls,
    )
    client = app.test_client()

    response = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)

    assert response.status_code == 200
    assert response.json["execution_backend"] == "queued_canary"
    telemetry = _queued_canary_telemetry(client, run_id)
    assert len(telemetry) == 1
    event = telemetry[0]
    assert event["ok"] is True
    assert event["terminal"] is True
    assert event["final_job_status"] == "succeeded"
    assert event["waiting_user"] is True
    assert event["waiting_task"] == "run_baseline"
    assert event["required_gates"] == [GateName.TRAIN_CONFIG.value]
    assert len(calls) == 1


def test_queued_execute_canary_telemetry_surfaces_failure_evidence(tmp_path: Path) -> None:
    run_id = "r-canary-telemetry-failed"
    project_id = "proj-canary-telemetry-failed"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    run_plan = _expanded_baseline_plan(run_id)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = _fake_executor_factory(
        {
            "ok": False,
            "run_id": run_id,
            "status": RunStatus.FAILED.value,
            "failed_task": "run_baseline",
            "error": {"message": "adapter failed"},
        },
        calls,
    )
    client = app.test_client()

    response = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)

    assert response.status_code == 400
    assert response.json["execution_backend"] == "queued_canary"
    telemetry = _queued_canary_telemetry(client, run_id)
    assert len(telemetry) == 1
    event = telemetry[0]
    assert event["ok"] is False
    assert event["terminal"] is True
    assert event["final_job_status"] == "failed"
    assert event["final_lease_status"] == "failed"
    assert event["failed_task"] == "run_baseline"
    assert event["error_message_present"] is True
    assert len(calls) == 1


def test_allowlisted_sync_path_without_canary_flag_does_not_emit_queued_telemetry(tmp_path: Path) -> None:
    run_id = "r-sync-no-telemetry"
    project_id = "proj-sync-no-telemetry"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    run_plan = _expanded_baseline_plan(run_id)
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)

    assert response.status_code == 200
    assert response.json["execution"]["status"] == RunStatus.SUCCEEDED.value
    assert "execution_backend" not in response.json
    assert "queue_summary" not in response.json
    assert _queued_canary_telemetry(client, run_id) == []


def test_non_allowlisted_sync_fallback_does_not_emit_queued_telemetry(tmp_path: Path) -> None:
    run_id = "r-fallback-no-telemetry"
    project_id = "proj-fallback-no-telemetry"
    dataset = tmp_path / "input" / "train.csv"
    _write_training_csv(dataset)
    run_plan = expand_run_plan(run_id=run_id, requested_tasks=["train_model"], available_artifacts=[])
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_RUN_PLAN_EXECUTE_QUEUED_CANARY"] = True

    def fail_if_queued(storage: ProjectStorage) -> object:
        raise AssertionError("train_model must not use queued canary")

    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = fail_if_queued
    client = app.test_client()

    response = _post_execute(client, project_id=project_id, run_plan=run_plan, dataset=dataset)

    assert response.status_code == 200
    assert response.json["execution"]["status"] == RunStatus.WAITING_USER.value
    assert "execution_backend" not in response.json
    assert "queue_summary" not in response.json
    assert _queued_canary_telemetry(client, run_id) == []
