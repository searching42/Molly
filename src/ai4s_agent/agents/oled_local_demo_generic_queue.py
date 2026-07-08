from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.local_worker_loop import LocalWorkerLoop
from ai4s_agent.planner import expand_run_plan
from ai4s_agent.run_plan_queue import enqueue_run_plan_execute_job
from ai4s_agent.run_plan_task_runner import RunPlanExecutorTaskRunner
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePollResult, WorkerQueuePoller


LOCAL_DEMO_TASK_ID = "execute_oled_local_demo"
QUEUE_TASK_ID = "run_plan_execute"
ARTIFACT_IDS = [
    "oled_demo_bundle_report",
    "oled_demo_bundle_markdown",
    "oled_local_demo_execution_manifest",
]


def execute_oled_local_demo_via_generic_run_plan_queue(
    *,
    queue_root: Path | str,
    project_root: Path | str,
    project_id: str,
    run_id: str,
    input_bundle: Path | str,
    output_dir: Path | str,
    worker_id: str,
    max_iterations: int,
    goal: str | None = None,
    overwrite: bool = False,
    now: str = "",
) -> dict[str, Any]:
    """Execute the OLED local demo through the generic run-plan queue envelope."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

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
    job_id = str(job.get("job_id") or "")
    poller = WorkerQueuePoller(
        queue,
        worker_id=worker_id,
        runner=RunPlanExecutorTaskRunner(storage=ProjectStorage(Path(project_root))),
        target_job_id=job_id,
    )
    loop_result = LocalWorkerLoop(poller).run_until_idle(max_iterations=max_iterations, now=now)
    actions = [item.action for item in loop_result.results]
    final_job = queue.status(job_id)
    if final_job is None:
        raise ValueError("generic_run_plan_queue_job_missing")
    final_status = str(final_job.get("status") or "")
    if final_status != "succeeded":
        raise ValueError(_failure_message(final_job, actions))

    runner_output = final_job.get("result")
    output = runner_output if isinstance(runner_output, dict) else {}
    return {
        "ok": True,
        "queue_root": str(queue_root),
        "project_root": str(project_root),
        "project_id": project_id,
        "run_id": run_id,
        "worker_id": worker_id,
        "job_id": job_id,
        "queue_task_id": str(final_job.get("task", {}).get("task_id") if isinstance(final_job.get("task"), dict) else ""),
        "run_plan_tasks": [task.task_id for task in run_plan.tasks],
        "actions": actions,
        "status": final_status,
        "completed_jobs": _job_ids_for_actions(loop_result.results, {"completed"}),
        "failed_jobs": _job_ids_for_actions(loop_result.results, {"failed"}),
        "executed_tasks": [str(item) for item in output.get("executed_tasks", [])] if isinstance(output.get("executed_tasks"), list) else [],
        "output_dir": str(output_dir),
        "artifacts": list(ARTIFACT_IDS),
        "generic_run_plan_queue": True,
        "scientific_adapters_executed": False,
        "executable": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the OLED local demo through the generic run-plan queue.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--goal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--now", default="")
    args = parser.parse_args(argv)

    result = execute_oled_local_demo_via_generic_run_plan_queue(
        queue_root=args.queue_root,
        project_root=args.project_root,
        project_id=args.project_id,
        run_id=args.run_id,
        input_bundle=args.input_bundle,
        output_dir=args.output_dir,
        worker_id=args.worker_id,
        max_iterations=args.max_iterations,
        goal=args.goal,
        overwrite=args.overwrite,
        now=args.now,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


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


def _failure_message(job: dict[str, Any], actions: list[str]) -> str:
    error = job.get("error")
    if isinstance(error, dict):
        reason = str(error.get("reason") or "").strip()
        if reason:
            return reason
    status = str(job.get("status") or "").strip()
    action = actions[-1] if actions else ""
    return f"generic_run_plan_queue_not_succeeded:{status or action or 'unknown'}"


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
