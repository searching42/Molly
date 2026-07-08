from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.agents.oled_local_demo_generic_allowlist import (
    ALLOWLISTED_RUN_PLAN_TASKS,
    validate_oled_local_demo_run_plan_execute_job,
)
from ai4s_agent.local_worker_loop import LocalWorkerLoop
from ai4s_agent.run_plan_queue import RUN_PLAN_EXECUTE_TASK_ID
from ai4s_agent.run_plan_task_runner import RunPlanExecutorTaskRunner
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePollResult, WorkerQueuePoller
from ai4s_agent.worker_task_runner import TaskRunResult


class AllowlistedOLEDRunPlanExecutorTaskRunner:
    """WorkerTaskRunner wrapper for OLED local demo generic run-plan jobs."""

    def __init__(self, *, storage: ProjectStorage) -> None:
        self.delegate = RunPlanExecutorTaskRunner(storage=storage)

    def start(self, job: dict[str, Any]) -> TaskRunResult:
        task = _validate_allowlisted_task(job)
        if task is None:
            return TaskRunResult(
                state="failed",
                message="generic_run_plan_not_allowlisted_for_oled_local_demo",
                output={"task_id": RUN_PLAN_EXECUTE_TASK_ID},
            )
        return self.delegate.start(job)

    def poll(self, job: dict[str, Any]) -> TaskRunResult:
        return self.delegate.poll(job)

    def cancel(self, job: dict[str, Any]) -> TaskRunResult:
        return self.delegate.cancel(job)


def run_oled_local_demo_generic_worker_loop(
    *,
    queue_root: Path | str,
    project_root: Path | str,
    worker_id: str,
    max_iterations: int,
    target_job_id: str | None = None,
    target_project_id: str | None = None,
    target_run_id: str | None = None,
    now: str = "",
) -> dict[str, Any]:
    """Run a bounded worker loop over existing allowlisted generic RunPlan jobs."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    runner = AllowlistedOLEDRunPlanExecutorTaskRunner(storage=ProjectStorage(Path(project_root)))
    poller = WorkerQueuePoller(
        queue,
        worker_id=worker_id,
        runner=runner,
        target_job_id=target_job_id,
        target_project_id=target_project_id,
        target_run_id=target_run_id,
    )
    loop_result = LocalWorkerLoop(poller).run_until_idle(max_iterations=max_iterations, now=now)
    actions = [item.action for item in loop_result.results]
    return {
        "ok": True,
        "queue_root": str(queue_root),
        "project_root": str(project_root),
        "worker_id": worker_id,
        "max_iterations": max_iterations,
        "iterations": loop_result.iterations,
        "actions": actions,
        "completed_jobs": _job_ids_for_actions(loop_result.results, {"completed"}),
        "failed_jobs": _job_ids_for_actions(loop_result.results, {"failed"}),
        "cancelled_jobs": _job_ids_for_actions(loop_result.results, {"cancelled"}),
        "executed_tasks": _executed_tasks(loop_result.results),
        "generic_run_plan_queue": True,
        "allowlist": list(ALLOWLISTED_RUN_PLAN_TASKS),
        "scientific_adapters_executed": False,
        "executable": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded OLED local demo generic RunPlan worker loop.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--target-job-id")
    parser.add_argument("--target-project-id")
    parser.add_argument("--target-run-id")
    parser.add_argument("--now", default="")
    args = parser.parse_args(argv)

    result = run_oled_local_demo_generic_worker_loop(
        queue_root=args.queue_root,
        project_root=args.project_root,
        worker_id=args.worker_id,
        max_iterations=args.max_iterations,
        target_job_id=args.target_job_id,
        target_project_id=args.target_project_id,
        target_run_id=args.target_run_id,
        now=args.now,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _validate_allowlisted_task(job: dict[str, Any]) -> object | None:
    try:
        return validate_oled_local_demo_run_plan_execute_job(job)
    except ValueError:
        return None


def _job_ids_for_actions(results: list[WorkerQueuePollResult], actions: set[str]) -> list[str]:
    job_ids: list[str] = []
    for item in results:
        if item.action not in actions:
            continue
        job_id = _result_job_id(item)
        if job_id and job_id not in job_ids:
            job_ids.append(job_id)
    return job_ids


def _result_job_id(result: WorkerQueuePollResult) -> str:
    for payload in (result.heartbeat_job, result.acquired_job, result.active_lease):
        if isinstance(payload, dict):
            job_id = str(payload.get("job_id") or "").strip()
            if job_id:
                return job_id
    return ""


def _executed_tasks(results: list[WorkerQueuePollResult]) -> list[str]:
    tasks: list[str] = []
    for result in results:
        output = result.runner_result.output if result.runner_result is not None else None
        if not isinstance(output, dict):
            continue
        values = output.get("executed_tasks")
        if not isinstance(values, list):
            continue
        for value in values:
            task_id = str(value).strip()
            if task_id and task_id not in tasks:
                tasks.append(task_id)
    return tasks


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
