from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai4s_agent.run_plan_queue import (
    RUN_PLAN_EXECUTE_TASK_ID,
    build_run_plan_execute_task,
    validate_run_plan_execute_task,
)
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


def test_build_run_plan_execute_task_generates_valid_serializable_task() -> None:
    task = build_run_plan_execute_task(
        project_id="proj-a",
        run_id="run-a",
        run_plan=_run_plan(),
        input_artifacts={"dataset": "datasets/input.csv"},
        task_options={"train_model": {"epochs": 1}},
    )

    assert task == {
        "task_id": RUN_PLAN_EXECUTE_TASK_ID,
        "kind": "run_plan_execute",
        "project_id": "proj-a",
        "run_id": "run-a",
        "run_plan": _run_plan().model_dump(mode="json"),
        "input_artifacts": {"dataset": "datasets/input.csv"},
        "task_options": {"train_model": {"epochs": 1}},
    }
    assert "command" not in task
    assert "argv" not in task


@pytest.mark.parametrize(
    ("field", "payload"),
    [
        ("project_id", {"run_id": "run-a", "run_plan": _run_plan().model_dump(mode="json")}),
        ("project_id", {"project_id": "", "run_id": "run-a", "run_plan": _run_plan().model_dump(mode="json")}),
        ("run_id", {"project_id": "proj-a", "run_plan": _run_plan().model_dump(mode="json")}),
        ("run_id", {"project_id": "proj-a", "run_id": "", "run_plan": _run_plan().model_dump(mode="json")}),
        ("run_plan", {"project_id": "proj-a", "run_id": "run-a"}),
    ],
)
def test_validate_run_plan_execute_task_rejects_missing_required_fields(field: str, payload: dict) -> None:
    payload.setdefault("task_id", RUN_PLAN_EXECUTE_TASK_ID)
    payload.setdefault("kind", "run_plan_execute")

    with pytest.raises(ValueError, match=field):
        validate_run_plan_execute_task(payload)


def test_validate_run_plan_execute_task_rejects_non_object_input_artifacts() -> None:
    payload = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())
    payload["input_artifacts"] = ["dataset.csv"]

    with pytest.raises(ValidationError, match="input_artifacts"):
        validate_run_plan_execute_task(payload)


def test_build_run_plan_execute_task_rejects_empty_list_input_artifacts() -> None:
    with pytest.raises(ValidationError, match="input_artifacts"):
        build_run_plan_execute_task(
            project_id="proj-a",
            run_id="run-a",
            run_plan=_run_plan(),
            input_artifacts=[],  # type: ignore[arg-type]
        )


def test_validate_run_plan_execute_task_rejects_non_object_task_options() -> None:
    payload = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())
    payload["task_options"] = ["epochs"]

    with pytest.raises(ValidationError, match="task_options"):
        validate_run_plan_execute_task(payload)


def test_build_run_plan_execute_task_rejects_empty_list_task_options() -> None:
    with pytest.raises(ValidationError, match="task_options"):
        build_run_plan_execute_task(
            project_id="proj-a",
            run_id="run-a",
            run_plan=_run_plan(),
            task_options=[],  # type: ignore[arg-type]
        )


def test_validate_run_plan_execute_task_rejects_command_fields() -> None:
    payload = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())
    payload["command"] = ["python", "-c", "print('unsafe')"]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_run_plan_execute_task(payload)


def test_validate_run_plan_execute_task_rejects_non_json_safe_options() -> None:
    with pytest.raises(ValidationError, match="non-JSON value"):
        build_run_plan_execute_task(
            project_id="proj-a",
            run_id="run-a",
            run_plan=_run_plan(),
            task_options={"train_model": {"callback": object()}},
        )


def test_build_run_plan_execute_task_rejects_run_id_mismatch() -> None:
    with pytest.raises(ValueError, match="run_id must match run_plan.run_id"):
        build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan("run-b"))


def test_generated_task_can_be_enqueued_by_worker_queue(tmp_path) -> None:
    queue = WorkerQueue(JsonWorkerQueueStore(tmp_path))
    task = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())

    job = queue.enqueue("proj-a", "run-a", task)

    assert job["status"] == "queued"
    assert job["project_id"] == "proj-a"
    assert job["run_id"] == "run-a"
    assert job["task"]["task_id"] == RUN_PLAN_EXECUTE_TASK_ID
    assert "command" not in job["task"]
