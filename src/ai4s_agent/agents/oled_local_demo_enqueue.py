from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "execute_oled_local_demo_runplan"


def enqueue_oled_local_demo_worker_job(
    *,
    queue_root: Path | str,
    project_root: Path | str,
    project_id: str,
    run_id: str,
    input_bundle: Path | str,
    output_dir: Path | str,
    goal: str | None = None,
    overwrite: bool = False,
    created_at: str = "",
) -> dict[str, Any]:
    """Enqueue one OLED local demo worker job without executing it."""
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    job = queue.enqueue(
        project_id,
        run_id,
        {
            "task_id": TASK_ID,
            "project_root": str(project_root),
            "input_bundle": str(input_bundle),
            "output_dir": str(output_dir),
            "goal": goal,
            "overwrite": overwrite,
        },
        created_at=created_at,
    )
    return {
        "ok": True,
        "queue_root": str(queue_root),
        "project_id": project_id,
        "run_id": run_id,
        "job_id": str(job.get("job_id") or ""),
        "job_status": str(job.get("status") or ""),
        "task_id": TASK_ID,
        "project_root": str(project_root),
        "input_bundle": str(input_bundle),
        "output_dir": str(output_dir),
        "overwrite": overwrite,
        "enqueued": True,
        "executed": False,
        "executable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enqueue one OLED local demo worker job without executing it.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--created-at", default="")
    args = parser.parse_args(argv)

    result = enqueue_oled_local_demo_worker_job(
        queue_root=args.queue_root,
        project_root=args.project_root,
        project_id=args.project_id,
        run_id=args.run_id,
        input_bundle=args.input_bundle,
        output_dir=args.output_dir,
        goal=args.goal,
        overwrite=args.overwrite,
        created_at=args.created_at,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
