from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.agents.oled_local_demo_worker import OLEDLocalDemoRunPlanWorkerTaskRunner
from ai4s_agent.local_worker_loop import LocalWorkerLoop
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePollResult, WorkerQueuePoller


def run_oled_local_demo_worker_loop(
    *,
    queue_root: Path | str,
    worker_id: str,
    max_iterations: int,
    target_job_id: str | None = None,
    target_project_id: str | None = None,
    target_run_id: str | None = None,
    now: str = "",
) -> dict[str, Any]:
    """Run a bounded local worker loop over existing OLED local demo jobs."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    poller = WorkerQueuePoller(
        queue,
        worker_id=worker_id,
        runner=OLEDLocalDemoRunPlanWorkerTaskRunner(),
        target_job_id=target_job_id,
        target_project_id=target_project_id,
        target_run_id=target_run_id,
    )
    loop_result = LocalWorkerLoop(poller).run_until_idle(max_iterations=max_iterations, now=now)
    actions = [item.action for item in loop_result.results]
    completed_jobs: list[str] = []
    failed_jobs: list[str] = []
    cancelled_jobs: list[str] = []
    recovered_jobs: list[str] = []
    executed_tasks: list[str] = []
    failure_messages: dict[str, str] = {}

    for item in loop_result.results:
        _extend_unique(recovered_jobs, item.recovered_job_ids)
        job_id = _result_job_id(item)
        if item.action == "completed" and job_id:
            _append_unique(completed_jobs, job_id)
            _extend_unique(executed_tasks, _executed_tasks(item))
        elif item.action == "failed" and job_id:
            _append_unique(failed_jobs, job_id)
            message = item.runner_result.message if item.runner_result is not None else ""
            failure_messages[job_id] = message
        elif item.action == "cancelled" and job_id:
            _append_unique(cancelled_jobs, job_id)

    return {
        "ok": not failed_jobs and not cancelled_jobs,
        "queue_root": str(queue_root),
        "worker_id": worker_id,
        "iterations": loop_result.iterations,
        "actions": actions,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "cancelled_jobs": cancelled_jobs,
        "recovered_jobs": recovered_jobs,
        "executed_tasks": executed_tasks,
        "failure_messages": failure_messages,
        "max_iterations": max_iterations,
        "idle": bool(actions) and actions[-1] == "idle",
        "executable": True,
        "adapters_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded OLED local demo worker loop.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--target-job-id")
    parser.add_argument("--target-project-id")
    parser.add_argument("--target-run-id")
    parser.add_argument("--now", default="")
    args = parser.parse_args(argv)

    result = run_oled_local_demo_worker_loop(
        queue_root=args.queue_root,
        worker_id=args.worker_id,
        max_iterations=args.max_iterations,
        target_job_id=args.target_job_id,
        target_project_id=args.target_project_id,
        target_run_id=args.target_run_id,
        now=args.now,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _result_job_id(result: WorkerQueuePollResult) -> str:
    for payload in (result.heartbeat_job, result.acquired_job, result.active_lease):
        if isinstance(payload, dict):
            job_id = str(payload.get("job_id") or "").strip()
            if job_id:
                return job_id
    return ""


def _executed_tasks(result: WorkerQueuePollResult) -> list[str]:
    if result.runner_result is None or not isinstance(result.runner_result.output, dict):
        return []
    tasks = result.runner_result.output.get("executed_tasks")
    if not isinstance(tasks, list):
        return []
    return [str(task).strip() for task in tasks if str(task).strip()]


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _extend_unique(values: list[str], additions: list[str]) -> None:
    for value in additions:
        _append_unique(values, str(value).strip())


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
