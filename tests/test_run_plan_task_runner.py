from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent.run_plan_queue import build_run_plan_execute_task
from ai4s_agent.run_plan_task_runner import RunPlanExecutorTaskRunner
from ai4s_agent.schemas import PlannedTask, RunPlan
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_task_runner import TaskRunResult


class FakeRunPlanExecutor:
    def __init__(self, execution: dict[str, Any]) -> None:
        self.execution = execution
        self.calls: list[dict[str, Any]] = []

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


def _job(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": "job-a",
        "project_id": "outer-project",
        "run_id": "outer-run",
        "task": task,
    }


def _runner(tmp_path: Path, execution: dict[str, Any]) -> tuple[RunPlanExecutorTaskRunner, FakeRunPlanExecutor]:
    fake = FakeRunPlanExecutor(execution)

    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return fake

    return RunPlanExecutorTaskRunner(storage=ProjectStorage(tmp_path), executor_factory=factory), fake


def test_run_plan_task_runner_start_executes_valid_queue_task(tmp_path: Path) -> None:
    execution = {"ok": True, "run_id": "run-a", "status": "WAITING_USER", "waiting_task": "train_model"}
    expected_output = {
        **execution,
        "waiting_user": True,
        "required_gates": [],
    }
    runner, fake = _runner(tmp_path, execution)
    run_plan = _run_plan()
    task = build_run_plan_execute_task(
        project_id="proj-a",
        run_id="run-a",
        run_plan=run_plan,
        input_artifacts={"dataset": "datasets/input.csv"},
        task_options={"train_model": {"epochs": 1}},
    )

    result = runner.start(_job(task))

    assert result == TaskRunResult(state="succeeded", message="run-plan execution completed", output=expected_output)
    assert fake.calls == [
        {
            "project_id": "proj-a",
            "run_plan": run_plan,
            "input_artifacts": {"dataset": "datasets/input.csv"},
            "task_options": {"train_model": {"epochs": 1}},
        }
    ]


def test_run_plan_task_runner_start_uses_worker_job_task_envelope(tmp_path: Path) -> None:
    execution = {"ok": True, "run_id": "run-a", "status": "SUCCEEDED"}
    runner, fake = _runner(tmp_path, execution)
    task = build_run_plan_execute_task(project_id="inner-project", run_id="run-a", run_plan=_run_plan())

    result = runner.start(_job(task))

    assert result.state == "succeeded"
    assert fake.calls[0]["project_id"] == "inner-project"


def test_run_plan_task_runner_start_maps_failed_execution_to_failed(tmp_path: Path) -> None:
    execution = {
        "ok": False,
        "run_id": "run-a",
        "status": "FAILED",
        "error": {"message": "adapter failed"},
    }
    runner, _fake = _runner(tmp_path, execution)
    task = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())

    result = runner.start(_job(task))

    assert result.state == "failed"
    assert result.message == "adapter failed"
    assert result.output == execution


def test_run_plan_task_runner_rejects_invalid_task_envelope(tmp_path: Path) -> None:
    runner, _fake = _runner(tmp_path, {"ok": True, "status": "SUCCEEDED"})
    task = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())
    task["command"] = ["python", "-c", "print('unsafe')"]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        runner.start(_job(task))


def test_run_plan_task_runner_rejects_missing_worker_job_task(tmp_path: Path) -> None:
    runner, _fake = _runner(tmp_path, {"ok": True, "status": "SUCCEEDED"})

    with pytest.raises(ValueError, match="job task must be an object"):
        runner.start({"job_id": "job-a"})


def test_run_plan_task_runner_poll_is_unsupported_one_shot(tmp_path: Path) -> None:
    runner, _fake = _runner(tmp_path, {"ok": True, "status": "SUCCEEDED"})

    result = runner.poll(_job(build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())))

    assert result == TaskRunResult(
        state="failed",
        message="run-plan task runner is one-shot; poll is unsupported",
    )


def test_run_plan_task_runner_cancel_is_unsupported_one_shot(tmp_path: Path) -> None:
    runner, _fake = _runner(tmp_path, {"ok": True, "status": "SUCCEEDED"})

    result = runner.cancel(_job(build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())))

    assert result == TaskRunResult(
        state="cancelled",
        message="run-plan execution cancellation is not supported by one-shot runner",
    )
