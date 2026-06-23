from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai4s_agent.run_plan_queue import RUN_PLAN_EXECUTE_TASK_ID, enqueue_run_plan_execute_job
from ai4s_agent.schemas import PlannedTask, RunPlan
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


def _run_plan(run_id: str = "run-a") -> RunPlan:
    return RunPlan(
        run_id=run_id,
        requested_tasks=["train_model"],
        tasks=[PlannedTask(task_id="train_model")],
        available_artifacts=[],
        missing_artifacts=[],
    )


def _queue(tmp_path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(tmp_path))


def test_enqueue_run_plan_execute_job_enqueues_valid_run_plan(tmp_path) -> None:
    queue = _queue(tmp_path)
    run_plan = _run_plan()

    job = enqueue_run_plan_execute_job(
        queue,
        project_id="proj-a",
        run_plan=run_plan,
        input_artifacts={"dataset": "datasets/input.csv"},
        task_options={"train_model": {"epochs": 1}},
    )

    assert job["status"] == "queued"
    assert job["project_id"] == "proj-a"
    assert job["run_id"] == "run-a"
    assert job["task"]["task_id"] == RUN_PLAN_EXECUTE_TASK_ID
    assert job["task"]["kind"] == "run_plan_execute"
    assert job["task"]["project_id"] == "proj-a"
    assert job["task"]["run_id"] == "run-a"
    assert job["task"]["input_artifacts"] == {"dataset": "datasets/input.csv"}
    assert job["task"]["task_options"] == {"train_model": {"epochs": 1}}


def test_enqueue_run_plan_execute_job_accepts_run_plan_dict(tmp_path) -> None:
    queue = _queue(tmp_path)

    job = enqueue_run_plan_execute_job(
        queue,
        project_id="proj-a",
        run_plan=_run_plan().model_dump(mode="json"),
    )

    assert job["run_id"] == "run-a"
    assert job["task"]["run_plan"]["run_id"] == "run-a"


def test_enqueue_run_plan_execute_job_uses_run_plan_run_id(tmp_path) -> None:
    queue = _queue(tmp_path)

    job = enqueue_run_plan_execute_job(queue, project_id="proj-a", run_plan=_run_plan("run-from-plan"))

    assert job["run_id"] == "run-from-plan"
    assert job["task"]["run_id"] == "run-from-plan"


def test_enqueue_run_plan_execute_job_rejects_invalid_input_artifacts_without_enqueue(tmp_path) -> None:
    queue = _queue(tmp_path)

    with pytest.raises(ValidationError, match="input_artifacts"):
        enqueue_run_plan_execute_job(
            queue,
            project_id="proj-a",
            run_plan=_run_plan(),
            input_artifacts=[],  # type: ignore[arg-type]
        )

    assert queue.list_jobs() == []


def test_enqueue_run_plan_execute_job_rejects_invalid_task_options_without_enqueue(tmp_path) -> None:
    queue = _queue(tmp_path)

    with pytest.raises(ValidationError, match="task_options"):
        enqueue_run_plan_execute_job(
            queue,
            project_id="proj-a",
            run_plan=_run_plan(),
            task_options=[],  # type: ignore[arg-type]
        )

    assert queue.list_jobs() == []


def test_enqueue_run_plan_execute_job_does_not_include_command_or_argv(tmp_path) -> None:
    queue = _queue(tmp_path)

    job = enqueue_run_plan_execute_job(queue, project_id="proj-a", run_plan=_run_plan())

    assert "command" not in job["task"]
    assert "argv" not in job["task"]


def test_enqueue_run_plan_execute_job_does_not_call_run_plan_executor(tmp_path, monkeypatch) -> None:
    queue = _queue(tmp_path)

    def fail_if_executor_is_touched(*args, **kwargs):
        raise AssertionError("RunPlanExecutor must not be touched by enqueue helper")

    monkeypatch.setattr("ai4s_agent.executor.RunPlanExecutor", fail_if_executor_is_touched)

    job = enqueue_run_plan_execute_job(queue, project_id="proj-a", run_plan=_run_plan())

    assert job["status"] == "queued"
