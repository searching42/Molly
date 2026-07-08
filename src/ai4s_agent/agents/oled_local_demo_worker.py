from __future__ import annotations

from typing import Any

from ai4s_agent._utils import strict_bool
from ai4s_agent.agents.oled_local_demo_runplan import execute_oled_local_demo_runplan
from ai4s_agent.worker_task_runner import TaskRunResult


TASK_ID = "execute_oled_local_demo_runplan"


class OLEDLocalDemoRunPlanWorkerTaskRunner:
    """WorkerTaskRunner that executes the OLED local demo RunPlanExecutor path."""

    def __init__(self) -> None:
        self._completed: dict[str, TaskRunResult] = {}

    def start(self, job: dict[str, Any]) -> TaskRunResult:
        try:
            project_id, run_id, task = _validate_job(job)
            result = execute_oled_local_demo_runplan(
                project_root=_required_task_value(task, "project_root"),
                project_id=project_id,
                run_id=run_id,
                input_bundle=_required_task_value(task, "input_bundle"),
                output_dir=_required_task_value(task, "output_dir"),
                goal=_optional_task_value(task, "goal"),
                overwrite=strict_bool(task.get("overwrite", False), key="overwrite"),
            )
            run_result = TaskRunResult(
                state="succeeded",
                message="oled local demo runplan succeeded",
                output=result,
            )
        except Exception as exc:
            task_id = _job_task_id(job)
            run_result = TaskRunResult(
                state="failed",
                message=str(exc) or exc.__class__.__name__,
                output={"task_id": task_id},
            )
        job_id = str(job.get("job_id") or "").strip() if isinstance(job, dict) else ""
        if job_id:
            self._completed[job_id] = run_result
        return run_result

    def poll(self, job: dict[str, Any]) -> TaskRunResult:
        job_id = str(job.get("job_id") or "").strip() if isinstance(job, dict) else ""
        if job_id and job_id in self._completed:
            return self._completed[job_id]
        task_id = _job_task_id(job)
        return TaskRunResult(
            state="succeeded",
            message="oled local demo runplan already completed",
            output={"task_id": task_id},
        )

    def cancel(self, job: dict[str, Any]) -> TaskRunResult:
        return TaskRunResult(
            state="cancelled",
            message="oled local demo runplan cancellation acknowledged",
            output={"task_id": _job_task_id(job)},
        )


def _validate_job(job: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    project_id = str(job.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("project_id required")
    run_id = str(job.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_id required")
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job task must be an object")
    task_id = str(task.get("task_id") or "").strip()
    if task_id != TASK_ID:
        raise ValueError(f"unsupported_oled_local_demo_worker_task:{task_id}")
    _required_task_value(task, "project_root")
    _required_task_value(task, "input_bundle")
    _required_task_value(task, "output_dir")
    return project_id, run_id, task


def _required_task_value(task: dict[str, Any], key: str) -> str:
    value = str(task.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} required")
    return value


def _optional_task_value(task: dict[str, Any], key: str) -> str | None:
    value = task.get(key)
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _job_task_id(job: dict[str, Any]) -> str:
    if not isinstance(job, dict):
        return TASK_ID
    task = job.get("task")
    if isinstance(task, dict):
        task_id = str(task.get("task_id") or "").strip()
        if task_id:
            return task_id
    return TASK_ID
