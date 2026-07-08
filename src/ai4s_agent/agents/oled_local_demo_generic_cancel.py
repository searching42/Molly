from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.run_plan_queue import RUN_PLAN_EXECUTE_KIND, RUN_PLAN_EXECUTE_TASK_ID, validate_run_plan_execute_task
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


LOCAL_DEMO_TASK_ID = "execute_oled_local_demo"
ALLOWLISTED_RUN_PLAN_TASKS = (LOCAL_DEMO_TASK_ID,)


def cancel_oled_local_demo_generic_run_plan_job(
    *,
    queue_root: Path | str,
    job_id: str,
    now: str = "",
) -> dict[str, Any]:
    """Cancel or request cancellation for an allowlisted generic OLED RunPlan job."""
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    job = queue.status(job_id)
    if job is None:
        raise KeyError(f"job not found: {job_id}")
    previous_status = str(job.get("status") or "")
    task_payload = job.get("task")
    if not isinstance(task_payload, dict):
        raise ValueError("job task must be an object")
    task_id = str(task_payload.get("task_id") or "").strip()
    if task_id != RUN_PLAN_EXECUTE_TASK_ID:
        raise ValueError("job task_id must be run_plan_execute")
    task_kind = str(task_payload.get("kind") or "").strip()
    if task_kind != RUN_PLAN_EXECUTE_KIND:
        raise ValueError("job kind must be run_plan_execute")
    task = validate_run_plan_execute_task(task_payload)
    run_plan_tasks = [planned.task_id for planned in task.run_plan.tasks]
    if run_plan_tasks != list(ALLOWLISTED_RUN_PLAN_TASKS):
        raise ValueError("generic_run_plan_not_allowlisted_for_oled_local_demo")
    if task.run_plan.requested_tasks != list(ALLOWLISTED_RUN_PLAN_TASKS):
        raise ValueError("generic_run_plan_not_allowlisted_for_oled_local_demo")

    cancelled = queue.cancel(job_id, now=now)
    return {
        "ok": True,
        "queue_root": str(queue_root),
        "job_id": job_id,
        "previous_status": previous_status,
        "job_status": str(cancelled.get("status") or ""),
        "cancellation_requested": bool(cancelled.get("cancellation_requested")),
        "queue_task_id": task.task_id,
        "queue_task_kind": task.kind,
        "run_plan_tasks": run_plan_tasks,
        "generic_run_plan_queue": True,
        "cancelled": True,
        "executed": False,
        "executable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cancel or request cancellation for a generic OLED RunPlan job.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--now", default="")
    args = parser.parse_args(argv)

    result = cancel_oled_local_demo_generic_run_plan_job(
        queue_root=args.queue_root,
        job_id=args.job_id,
        now=args.now,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
