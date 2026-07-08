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


def retry_failed_oled_local_demo_generic_run_plan_job(
    *,
    queue_root: Path | str,
    source_job_id: str,
    retry_request_id: str,
    requested_by: str,
    reason: str,
    now: str = "",
) -> dict[str, Any]:
    """Enqueue a retry child for a failed generic OLED local demo RunPlan job."""
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    source = queue.status(source_job_id)
    if source is None:
        raise KeyError(f"source job not found: {source_job_id}")
    source_status = str(source.get("status") or "")
    if source_status != "failed":
        raise ValueError("source job status must be failed")
    task_payload = source.get("task")
    if not isinstance(task_payload, dict):
        raise ValueError("source job task must be an object")
    task_id = str(task_payload.get("task_id") or "").strip()
    if task_id != RUN_PLAN_EXECUTE_TASK_ID:
        raise ValueError("source job task_id must be run_plan_execute")
    task_kind = str(task_payload.get("kind") or "").strip()
    if task_kind != RUN_PLAN_EXECUTE_KIND:
        raise ValueError("source job kind must be run_plan_execute")
    task = validate_run_plan_execute_task(task_payload)
    run_plan_tasks = [planned.task_id for planned in task.run_plan.tasks]
    if run_plan_tasks != list(ALLOWLISTED_RUN_PLAN_TASKS):
        raise ValueError("generic_run_plan_not_allowlisted_for_oled_local_demo")
    if task.run_plan.requested_tasks != list(ALLOWLISTED_RUN_PLAN_TASKS):
        raise ValueError("generic_run_plan_not_allowlisted_for_oled_local_demo")

    retry = queue.enqueue_retry_of_failed_job(
        source_job_id,
        retry_request_id=retry_request_id,
        requested_by=requested_by,
        reason=reason,
        now=now,
    )
    return {
        "ok": True,
        "queue_root": str(queue_root),
        "source_job_id": source_job_id,
        "source_status": source_status,
        "retry_job_id": str(retry.get("job_id") or ""),
        "retry_status": str(retry.get("status") or ""),
        "retry_of_job_id": str(retry.get("retry_of_job_id") or ""),
        "retry_root_job_id": str(retry.get("retry_root_job_id") or ""),
        "retry_request_id": str(retry.get("retry_request_id") or ""),
        "retry_requested_by": str(retry.get("retry_requested_by") or ""),
        "retry_reason": str(retry.get("retry_reason") or ""),
        "queue_task_id": task.task_id,
        "queue_task_kind": task.kind,
        "run_plan_tasks": run_plan_tasks,
        "generic_run_plan_queue": True,
        "enqueued": True,
        "executed": False,
        "executable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enqueue a retry child for a failed generic OLED local demo job.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--retry-request-id", required=True)
    parser.add_argument("--requested-by", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--now", default="")
    args = parser.parse_args(argv)

    result = retry_failed_oled_local_demo_generic_run_plan_job(
        queue_root=args.queue_root,
        source_job_id=args.source_job_id,
        retry_request_id=args.retry_request_id,
        requested_by=args.requested_by,
        reason=args.reason,
        now=args.now,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
