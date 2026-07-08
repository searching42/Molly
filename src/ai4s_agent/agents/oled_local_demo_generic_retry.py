from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.agents.oled_local_demo_generic_allowlist import (
    validate_oled_local_demo_run_plan_execute_job,
)
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


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
    try:
        task = validate_oled_local_demo_run_plan_execute_job(source)
    except ValueError as exc:
        raise _source_validation_error(exc) from exc
    run_plan_tasks = [planned.task_id for planned in task.run_plan.tasks]

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


def _source_validation_error(exc: ValueError) -> ValueError:
    message = str(exc)
    if message == "job task must be an object":
        return ValueError("source job task must be an object")
    if message == "task_id must be run_plan_execute":
        return ValueError("source job task_id must be run_plan_execute")
    if message == "task kind must be run_plan_execute":
        return ValueError("source job kind must be run_plan_execute")
    return ValueError(message)


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
