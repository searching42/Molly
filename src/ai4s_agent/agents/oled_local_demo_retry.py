from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "execute_oled_local_demo_runplan"


def retry_failed_oled_local_demo_worker_job(
    *,
    queue_root: Path | str,
    source_job_id: str,
    retry_request_id: str,
    requested_by: str,
    reason: str,
    now: str = "",
) -> dict[str, Any]:
    """Enqueue a retry child for a failed OLED local demo worker job."""
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    source = queue.status(source_job_id)
    if source is None:
        raise KeyError(f"source job not found: {source_job_id}")
    source_status = str(source.get("status") or "")
    if source_status != "failed":
        raise ValueError("source job status must be failed")
    task = source.get("task")
    if not isinstance(task, dict):
        raise ValueError("source job task must be an object")
    task_id = str(task.get("task_id") or "").strip()
    if task_id != TASK_ID:
        raise ValueError(f"unsupported_oled_local_demo_retry_task:{task_id}")

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
        "retry_request_id": str(retry.get("retry_request_id") or ""),
        "retry_requested_by": str(retry.get("retry_requested_by") or ""),
        "retry_reason": str(retry.get("retry_reason") or ""),
        "task_id": task_id,
        "enqueued": True,
        "executed": False,
        "executable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enqueue a retry child for a failed OLED local demo worker job.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--retry-request-id", required=True)
    parser.add_argument("--requested-by", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--now", default="")
    args = parser.parse_args(argv)

    result = retry_failed_oled_local_demo_worker_job(
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
