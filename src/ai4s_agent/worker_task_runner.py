from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

TaskState = Literal["running", "succeeded", "failed", "cancelled"]


@dataclass(frozen=True)
class TaskRunResult:
    state: TaskState
    message: str = ""
    output: dict[str, Any] | None = None


class WorkerTaskRunner(Protocol):
    def start(self, job: dict[str, Any]) -> TaskRunResult:
        ...

    def poll(self, job: dict[str, Any]) -> TaskRunResult:
        ...

    def cancel(self, job: dict[str, Any]) -> TaskRunResult:
        ...


class FakeWorkerTaskRunner:
    """In-process fake runner for control-plane tests.

    This fake validates the job task envelope but does not execute commands,
    start processes, call RunPlanExecutor, or contact remote workers.
    """

    def __init__(self, *, poll_results: dict[str, TaskRunResult] | None = None) -> None:
        self._poll_results = dict(poll_results or {})

    def start(self, job: dict[str, Any]) -> TaskRunResult:
        task = _validate_job(job)
        return TaskRunResult(state="running", message="task started", output={"task_id": str(task["task_id"])})

    def poll(self, job: dict[str, Any]) -> TaskRunResult:
        task = _validate_job(job)
        job_id = str(job["job_id"])
        return self._poll_results.get(
            job_id,
            TaskRunResult(state="running", message="task running", output={"task_id": str(task["task_id"])}),
        )

    def cancel(self, job: dict[str, Any]) -> TaskRunResult:
        task = _validate_job(job)
        return TaskRunResult(state="cancelled", message="task cancelled", output={"task_id": str(task["task_id"])})


def _validate_job(job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        raise ValueError("job_id required")
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job task must be an object")
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("job task_id required")
    return task
