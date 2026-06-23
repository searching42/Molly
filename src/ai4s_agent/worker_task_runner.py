from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from ai4s_agent.worker_supervisor import WorkerHeartbeat, WorkerSupervisor

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


class WorkerSupervisorTaskRunner:
    """WorkerTaskRunner adapter backed by WorkerSupervisor.

    This adapter starts local dummy/process commands through WorkerSupervisor.
    It does not call RunPlanExecutor, use shell strings, contact remote workers,
    or change queue storage.
    """

    def __init__(self, *, supervisor: WorkerSupervisor, stop_timeout_sec: int = 10) -> None:
        self.supervisor = supervisor
        if stop_timeout_sec < 0:
            raise ValueError("stop_timeout_sec must be non-negative")
        self.stop_timeout_sec = stop_timeout_sec

    def start(self, job: dict[str, Any]) -> TaskRunResult:
        project_id, run_id, task = _validate_supervisor_job(job)
        heartbeat = self.supervisor.start(
            project_id=project_id,
            run_id=run_id,
            command=_validate_command(task),
            cwd=_task_cwd(task),
        )
        return TaskRunResult(state="running", message="worker started", output=_heartbeat_output(heartbeat))

    def poll(self, job: dict[str, Any]) -> TaskRunResult:
        project_id, run_id, _task = _validate_supervisor_job(job)
        heartbeat = self.supervisor.status(project_id, run_id)
        if heartbeat.status in {"pending", "running"}:
            return TaskRunResult(state="running", message="worker running", output=_heartbeat_output(heartbeat))
        if heartbeat.status == "stopped" and heartbeat.exit_code == 0:
            return TaskRunResult(state="succeeded", message="worker stopped", output=_heartbeat_output(heartbeat))
        return TaskRunResult(state="failed", message="worker failed", output=_heartbeat_output(heartbeat))

    def cancel(self, job: dict[str, Any]) -> TaskRunResult:
        project_id, run_id, _task = _validate_supervisor_job(job)
        heartbeat = self.supervisor.stop(project_id, run_id, timeout_sec=self.stop_timeout_sec)
        return TaskRunResult(state="cancelled", message="worker cancelled", output=_heartbeat_output(heartbeat))


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


def _validate_supervisor_job(job: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    task = _validate_job(job)
    project_id = str(job.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("project_id required")
    run_id = str(job.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_id required")
    return project_id, run_id, task


def _validate_command(task: dict[str, Any]) -> list[str]:
    command = task.get("command")
    if not isinstance(command, list):
        raise ValueError("job task command must be a list")
    if not command:
        raise ValueError("job task command must not be empty")
    clean: list[str] = []
    for index, item in enumerate(command):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"job task command[{index}] must be a non-empty string")
        clean.append(item)
    return clean


def _task_cwd(task: dict[str, Any]) -> Path | None:
    cwd = str(task.get("cwd") or "").strip()
    return Path(cwd) if cwd else None


def _heartbeat_output(heartbeat: WorkerHeartbeat) -> dict[str, Any]:
    return {
        "run_id": heartbeat.run_id,
        "pid": heartbeat.pid,
        "status": heartbeat.status,
        "exit_code": heartbeat.exit_code,
        "command": list(heartbeat.command),
        "cwd": heartbeat.cwd,
    }
