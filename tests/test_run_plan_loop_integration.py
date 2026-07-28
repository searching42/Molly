from __future__ import annotations

from pathlib import Path
from typing import Any

from ai4s_agent.local_worker_loop import LocalWorkerLoop
from ai4s_agent.run_plan_queue import build_run_plan_execute_task
from ai4s_agent.run_plan_task_runner import RunPlanExecutorTaskRunner
from ai4s_agent.schemas import PlannedTask, RunPlan
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller


class FakeRunPlanExecutor:
    def __init__(self, execution: dict[str, Any]) -> None:
        self.execution = dict(execution)
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


def _queue(tmp_path: Path) -> WorkerQueue:
    return WorkerQueue(JsonWorkerQueueStore(tmp_path / "queue"))


def _loop(queue: WorkerQueue, tmp_path: Path, execution: dict[str, Any]) -> tuple[LocalWorkerLoop, FakeRunPlanExecutor]:
    fake = FakeRunPlanExecutor(execution)

    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return fake

    runner = RunPlanExecutorTaskRunner(
        storage=ProjectStorage(tmp_path / "workspace"),
        executor_factory=factory,
    )
    poller = WorkerQueuePoller(queue, worker_id="worker-a", runner=runner)
    return LocalWorkerLoop(poller), fake


def test_run_plan_execute_queue_task_completes_through_local_worker_loop(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    run_plan = _run_plan()
    task = build_run_plan_execute_task(
        project_id="proj-a",
        run_id="run-a",
        run_plan=run_plan,
        input_artifacts={"dataset": "datasets/input.csv"},
        task_options={"train_model": {"epochs": 1}},
    )
    job = queue.enqueue("proj-a", "run-a", task)
    loop, fake = _loop(queue, tmp_path, {"ok": True, "run_id": "run-a", "status": "WAITING_USER"})

    result = loop.run_until_idle(max_iterations=3)

    status = queue.status(job["job_id"])
    assert status is not None
    assert [item.action for item in result.results] == ["completed", "idle"]
    assert status["status"] == "succeeded"
    lease = queue.lease_status(str(status["lease_id"]))
    assert lease is not None
    assert lease["status"] == "completed"
    assert fake.calls == [
        {
            "project_id": "proj-a",
            "run_plan": run_plan,
            "input_artifacts": {"dataset": "datasets/input.csv"},
            "task_options": {"train_model": {"epochs": 1}},
        }
    ]


def test_run_plan_execute_queue_task_failure_writes_terminal_state(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    task = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())
    job = queue.enqueue("proj-a", "run-a", task)
    loop, _fake = _loop(
        queue,
        tmp_path,
        {
            "ok": False,
            "run_id": "run-a",
            "status": "FAILED",
            "error": {"message": "adapter failed"},
        },
    )

    result = loop.run_until_idle(max_iterations=3)

    status = queue.status(job["job_id"])
    assert status is not None
    assert [item.action for item in result.results] == ["failed", "idle"]
    assert status["status"] == "failed"
    assert status["error"] == {"reason": "adapter failed"}
    lease = queue.lease_status(str(status["lease_id"]))
    assert lease is not None
    assert lease["status"] == "failed"


def test_invalid_run_plan_execute_envelope_fails_through_local_worker_loop(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    task = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())
    task["command"] = ["python", "-c", "print('unsafe')"]
    job = queue.enqueue("proj-a", "run-a", task)
    loop, fake = _loop(queue, tmp_path, {"ok": True, "run_id": "run-a", "status": "SUCCEEDED"})

    result = loop.run_until_idle(max_iterations=3)

    status = queue.status(job["job_id"])
    assert status is not None
    assert [item.action for item in result.results] == ["failed", "idle"]
    assert status["status"] == "failed"
    assert "Extra inputs are not permitted" in status["error"]["reason"]
    lease = queue.lease_status(str(status["lease_id"]))
    assert lease is not None
    assert lease["status"] == "failed"
    assert fake.calls == []


def test_run_plan_execute_task_schema_does_not_include_command_or_argv() -> None:
    task = build_run_plan_execute_task(project_id="proj-a", run_id="run-a", run_plan=_run_plan())

    assert "command" not in task
    assert "argv" not in task
