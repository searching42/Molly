from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue import enqueue_run_plan_execute_job
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


LOCAL_DEMO_TASK_ID = "execute_oled_local_demo"


def enqueue_oled_local_demo_generic_run_plan_job(
    *,
    queue_root: Path | str,
    project_id: str,
    run_id: str,
    input_bundle: Path | str,
    output_dir: Path | str,
    goal: str | None = None,
    overwrite: bool = False,
    created_at: str = "",
) -> dict[str, Any]:
    """Enqueue one generic RunPlan queue job for the OLED local demo task."""
    _ = created_at
    run_plan = expand_run_plan(
        run_id=run_id,
        requested_tasks=[LOCAL_DEMO_TASK_ID],
    )
    task_options = {
        LOCAL_DEMO_TASK_ID: {
            "input_bundle": str(input_bundle),
            "output_dir": str(output_dir),
            "overwrite": overwrite,
            "goal": goal,
            "project_id": project_id,
        }
    }
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    job = enqueue_run_plan_execute_job(
        queue,
        project_id=project_id,
        run_plan=run_plan,
        input_artifacts={},
        task_options=task_options,
    )
    task = job.get("task") if isinstance(job.get("task"), dict) else {}
    return {
        "ok": True,
        "queue_root": str(queue_root),
        "project_id": project_id,
        "run_id": run_id,
        "job_id": str(job.get("job_id") or ""),
        "job_status": str(job.get("status") or ""),
        "queue_task_id": str(task.get("task_id") or ""),
        "queue_task_kind": str(task.get("kind") or ""),
        "run_plan_tasks": [task.task_id for task in run_plan.tasks],
        "input_bundle": str(input_bundle),
        "output_dir": str(output_dir),
        "overwrite": overwrite,
        "generic_run_plan_queue": True,
        "enqueued": True,
        "executed": False,
        "executable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enqueue one generic RunPlan OLED local demo job.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = enqueue_oled_local_demo_generic_run_plan_job(
        queue_root=args.queue_root,
        project_id=args.project_id,
        run_id=args.run_id,
        input_bundle=args.input_bundle,
        output_dir=args.output_dir,
        goal=args.goal,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
