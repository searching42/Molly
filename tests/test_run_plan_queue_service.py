from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ai4s_agent.run_plan_queue_service import run_run_plan_via_local_queue
from ai4s_agent.schemas import PlannedTask, RunPlan
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


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


def _storage(tmp_path: Path) -> ProjectStorage:
    return ProjectStorage(tmp_path / "workspace")


def _factory(fake: FakeRunPlanExecutor):
    def factory(storage: ProjectStorage) -> FakeRunPlanExecutor:
        assert isinstance(storage, ProjectStorage)
        return fake

    return factory


def test_run_plan_via_local_queue_completes_waiting_user_execution(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    storage = _storage(tmp_path)
    run_plan = _run_plan()
    fake = FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"})

    result = run_run_plan_via_local_queue(
        queue=queue,
        storage=storage,
        project_id="proj-a",
        run_plan=run_plan,
        input_artifacts={"dataset": "datasets/input.csv"},
        task_options={"train_model": {"epochs": 1}},
        executor_factory=_factory(fake),
    )

    assert result["ok"] is True
    assert result["terminal"] is True
    assert result["waiting_user"] is True
    assert result["waiting_task"] == ""
    assert result["required_gates"] == []
    assert result["loop_results"] == ["completed", "idle"]
    assert result["final_job"]["status"] == "succeeded"
    assert result["final_job"]["result"]["waiting_user"] is True
    assert result["final_lease"]["status"] == "completed"
    assert result["queued_job_id"] == result["final_job"]["job_id"]
    assert fake.calls == [
        {
            "project_id": "proj-a",
            "run_plan": run_plan,
            "input_artifacts": {"dataset": "datasets/input.csv"},
            "task_options": {"train_model": {"epochs": 1}},
        }
    ]


def test_run_plan_via_local_queue_surfaces_waiting_user_metadata(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    fake = FakeRunPlanExecutor(
        {
            "ok": True,
            "run_id": "run-a",
            "status": "WAITING_USER",
            "waiting_task": "train_model",
            "required_gates": ["gate_3_train_config"],
        }
    )

    result = run_run_plan_via_local_queue(
        queue=queue,
        storage=_storage(tmp_path),
        project_id="proj-a",
        run_plan=_run_plan(),
        executor_factory=_factory(fake),
    )

    assert result["ok"] is True
    assert result["terminal"] is True
    assert result["waiting_user"] is True
    assert result["waiting_task"] == "train_model"
    assert result["required_gates"] == ["gate_3_train_config"]
    assert result["final_job"]["status"] == "succeeded"
    assert result["final_job"]["result"]["status"] == "WAITING_USER"
    assert result["final_job"]["result"]["waiting_user"] is True
    assert result["final_job"]["result"]["waiting_task"] == "train_model"


def test_run_plan_via_local_queue_reports_failed_execution(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    fake = FakeRunPlanExecutor(
        {
            "ok": False,
            "run_id": "run-a",
            "status": "FAILED",
            "error": {"message": "adapter failed"},
        }
    )

    result = run_run_plan_via_local_queue(
        queue=queue,
        storage=_storage(tmp_path),
        project_id="proj-a",
        run_plan=_run_plan(),
        executor_factory=_factory(fake),
    )

    assert result["ok"] is False
    assert result["terminal"] is True
    assert result["loop_results"] == ["failed", "idle"]
    assert result["final_job"]["status"] == "failed"
    assert result["final_job"]["error"] == {"reason": "adapter failed"}
    assert result["final_lease"]["status"] == "failed"


def test_run_plan_via_local_queue_rejects_invalid_options_without_enqueue(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    fake = FakeRunPlanExecutor({"ok": True, "status": "WAITING_USER"})

    with pytest.raises(ValidationError, match="task_options"):
        run_run_plan_via_local_queue(
            queue=queue,
            storage=_storage(tmp_path),
            project_id="proj-a",
            run_plan=_run_plan(),
            task_options=[],  # type: ignore[arg-type]
            executor_factory=_factory(fake),
        )

    assert queue.list_jobs() == []
    assert fake.calls == []


def test_run_plan_via_local_queue_rejects_invalid_input_artifacts_without_enqueue(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    fake = FakeRunPlanExecutor({"ok": True, "status": "WAITING_USER"})

    with pytest.raises(ValidationError, match="input_artifacts"):
        run_run_plan_via_local_queue(
            queue=queue,
            storage=_storage(tmp_path),
            project_id="proj-a",
            run_plan=_run_plan(),
            input_artifacts=[],  # type: ignore[arg-type]
            executor_factory=_factory(fake),
        )

    assert queue.list_jobs() == []
    assert fake.calls == []


def test_run_plan_via_local_queue_rejects_existing_queued_job_without_enqueue(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    queue.enqueue("proj-old", "run-old", {"task_id": "some_other_task"})
    fake = FakeRunPlanExecutor({"ok": True, "status": "WAITING_USER"})

    with pytest.raises(ValueError, match="empty/dedicated queue"):
        run_run_plan_via_local_queue(
            queue=queue,
            storage=_storage(tmp_path),
            project_id="proj-a",
            run_plan=_run_plan(),
            executor_factory=_factory(fake),
        )

    jobs = queue.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["run_id"] == "run-old"
    assert fake.calls == []


def test_run_plan_via_local_queue_rejects_existing_running_job_without_enqueue(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    old = queue.enqueue("proj-old", "run-old", {"task_id": "some_other_task"})
    queue.acquire("other-worker")
    fake = FakeRunPlanExecutor({"ok": True, "status": "WAITING_USER"})

    with pytest.raises(ValueError, match="empty/dedicated queue"):
        run_run_plan_via_local_queue(
            queue=queue,
            storage=_storage(tmp_path),
            project_id="proj-a",
            run_plan=_run_plan(),
            executor_factory=_factory(fake),
        )

    jobs = queue.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == old["job_id"]
    assert jobs[0]["status"] == "running"
    assert fake.calls == []


def test_run_plan_via_local_queue_runs_targeted_job_when_queue_not_empty(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    existing = queue.enqueue("proj-a", "run-a", {"task_id": "existing_job"})
    run_plan = _run_plan()
    fake = FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"})

    result = run_run_plan_via_local_queue(
        queue=queue,
        storage=_storage(tmp_path),
        project_id="proj-a",
        run_plan=run_plan,
        require_empty_queue=False,
        executor_factory=_factory(fake),
    )

    assert result["ok"] is True
    assert result["queued_job_id"] != existing["job_id"]
    assert result["final_job"]["job_id"] == result["queued_job_id"]
    assert result["final_job"]["status"] == "succeeded"
    assert queue.status(existing["job_id"]) is not None
    assert queue.status(existing["job_id"])["status"] == "queued"
    assert fake.calls == [
        {
            "project_id": "proj-a",
            "run_plan": run_plan,
            "input_artifacts": {},
            "task_options": {},
        }
    ]


def test_run_plan_via_local_queue_rejects_invalid_target_selector(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    fake = FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"})

    with pytest.raises(ValueError, match="single safe path segment"):
        run_run_plan_via_local_queue(
            queue=queue,
            storage=_storage(tmp_path),
            project_id="proj-a",
            run_plan=_run_plan(),
            require_empty_queue=False,
            target_project_id="proj/a",
            executor_factory=_factory(fake),
        )

    assert queue.list_jobs() == []
    assert fake.calls == []


def test_run_plan_via_local_queue_rejects_explicit_target_job_id_without_enqueue(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    fake = FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"})

    with pytest.raises(ValueError, match="target_job_id is managed internally"):
        run_run_plan_via_local_queue(
            queue=queue,
            storage=_storage(tmp_path),
            project_id="proj-a",
            run_plan=_run_plan(),
            require_empty_queue=False,
            target_job_id="job-proj-a-run-a-000001",
            executor_factory=_factory(fake),
        )

    assert queue.list_jobs() == []
    assert fake.calls == []


def test_run_plan_via_local_queue_rejects_mismatched_target_project_without_enqueue(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    fake = FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"})

    with pytest.raises(ValueError, match="target_project_id must match project_id"):
        run_run_plan_via_local_queue(
            queue=queue,
            storage=_storage(tmp_path),
            project_id="proj-a",
            run_plan=_run_plan(),
            require_empty_queue=False,
            target_project_id="proj-b",
            executor_factory=_factory(fake),
        )

    assert queue.list_jobs() == []
    assert fake.calls == []


def test_run_plan_via_local_queue_rejects_mismatched_target_run_without_enqueue(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    fake = FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"})

    with pytest.raises(ValueError, match="target_run_id must match run_plan.run_id"):
        run_run_plan_via_local_queue(
            queue=queue,
            storage=_storage(tmp_path),
            project_id="proj-a",
            run_plan=_run_plan(),
            require_empty_queue=False,
            target_run_id="run-b",
            executor_factory=_factory(fake),
        )

    assert queue.list_jobs() == []
    assert fake.calls == []


def test_run_plan_via_local_queue_respects_max_iterations(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    fake = FakeRunPlanExecutor({"ok": True, "run_id": "run-a", "status": "WAITING_USER"})

    result = run_run_plan_via_local_queue(
        queue=queue,
        storage=_storage(tmp_path),
        project_id="proj-a",
        run_plan=_run_plan(),
        max_iterations=1,
        executor_factory=_factory(fake),
    )

    assert result["loop_results"] == ["completed"]
    assert result["terminal"] is False
    assert result["final_job"]["status"] == "succeeded"
    assert result["final_lease"]["status"] == "completed"


def test_run_plan_via_local_queue_task_has_no_command_or_argv(tmp_path: Path) -> None:
    queue = _queue(tmp_path)

    result = run_run_plan_via_local_queue(
        queue=queue,
        storage=_storage(tmp_path),
        project_id="proj-a",
        run_plan=_run_plan(),
        executor_factory=_factory(FakeRunPlanExecutor({"ok": True, "status": "WAITING_USER"})),
    )

    task = result["final_job"]["task"]
    assert "command" not in task
    assert "argv" not in task
