from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "execute_oled_local_demo_runplan"


def cancel_oled_local_demo_worker_job(
    *,
    queue_root: Path | str,
    job_id: str,
    now: str = "",
) -> dict[str, Any]:
    """Cancel or request cancellation for one OLED local demo worker job."""
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    job = queue.status(job_id)
    if job is None:
        raise KeyError(f"job not found: {job_id}")
    previous_status = str(job.get("status") or "")
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job task must be an object")
    task_id = str(task.get("task_id") or "").strip()
    if task_id != TASK_ID:
        raise ValueError(f"unsupported_oled_local_demo_cancel_task:{task_id}")

    cancelled = queue.cancel(job_id, now=now)
    return {
        "ok": True,
        "queue_root": str(queue_root),
        "job_id": job_id,
        "previous_status": previous_status,
        "job_status": str(cancelled.get("status") or ""),
        "cancellation_requested": bool(cancelled.get("cancellation_requested")),
        "task_id": task_id,
        "cancelled": True,
        "executed": False,
        "executable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cancel one OLED local demo worker job without executing it.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--now", default="")
    args = parser.parse_args(argv)

    result = cancel_oled_local_demo_worker_job(
        queue_root=args.queue_root,
        job_id=args.job_id,
        now=args.now,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
