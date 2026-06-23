from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.run_plan_queue import validate_run_plan_execute_task
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_task_runner import TaskRunResult


class _RunPlanExecutorLike(Protocol):
    def execute(
        self,
        *,
        project_id: str,
        run_plan: Any,
        input_artifacts: dict[str, Any] | None = None,
        task_options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ...


ExecutorFactory = Callable[[ProjectStorage], _RunPlanExecutorLike]


class RunPlanExecutorTaskRunner:
    """One-shot WorkerTaskRunner adapter for run-plan queue jobs.

    This runner consumes the `run_plan_execute` queue task envelope only. It
    does not expose API routes, start background threads, connect remote
    workers, migrate storage, or change the default `/api/run-plan/execute`
    synchronous path.
    """

    def __init__(
        self,
        *,
        storage: ProjectStorage,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self.storage = storage
        self.executor_factory = executor_factory or (lambda storage: RunPlanExecutor(storage=storage))

    def start(self, job: dict[str, Any]) -> TaskRunResult:
        task = validate_run_plan_execute_task(_task_envelope(job))
        executor = self.executor_factory(self.storage)
        execution = executor.execute(
            project_id=task.project_id,
            run_plan=task.run_plan,
            input_artifacts={str(key): str(value) for key, value in task.input_artifacts.items()},
            task_options=_task_options(task.task_options),
        )
        if _execution_succeeded(execution):
            return TaskRunResult(
                state="succeeded",
                message="run-plan execution completed",
                output=execution,
            )
        return TaskRunResult(
            state="failed",
            message=_failure_message(execution),
            output=execution,
        )

    def poll(self, job: dict[str, Any]) -> TaskRunResult:
        return TaskRunResult(
            state="failed",
            message="run-plan task runner is one-shot; poll is unsupported",
        )

    def cancel(self, job: dict[str, Any]) -> TaskRunResult:
        return TaskRunResult(
            state="cancelled",
            message="run-plan execution cancellation is not supported by one-shot runner",
        )


def _task_envelope(job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job task must be an object")
    return task


def _execution_succeeded(execution: dict[str, Any]) -> bool:
    if execution.get("ok") is False:
        return False
    status = str(execution.get("status") or "").strip().lower()
    return status in {"succeeded", "done", "waiting_user"}


def _failure_message(execution: dict[str, Any]) -> str:
    error = execution.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        if message:
            return message
    message = str(execution.get("message") or "").strip()
    return message or "run-plan execution failed"


def _task_options(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for task_id, options in value.items():
        if isinstance(options, dict):
            normalized[str(task_id)] = {str(key): item for key, item in options.items()}
    return normalized
