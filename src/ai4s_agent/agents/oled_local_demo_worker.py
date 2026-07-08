from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent._utils import strict_bool
from ai4s_agent.agents.oled_local_demo_runplan import execute_oled_local_demo_runplan
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue
from ai4s_agent.worker_queue_poller import WorkerQueuePoller
from ai4s_agent.worker_task_runner import TaskRunResult


TASK_ID = "execute_oled_local_demo_runplan"
OUTPUT_ARTIFACT_IDS = [
    "oled_demo_bundle_report",
    "oled_demo_bundle_markdown",
    "oled_local_demo_execution_manifest",
]


class OLEDLocalDemoRunPlanWorkerTaskRunner:
    """WorkerTaskRunner that executes the OLED local demo RunPlanExecutor path."""

    def __init__(self) -> None:
        self._completed: dict[str, TaskRunResult] = {}

    def start(self, job: dict[str, Any]) -> TaskRunResult:
        try:
            project_id, run_id, task = _validate_job(job)
            result = execute_oled_local_demo_runplan(
                project_root=_required_task_value(task, "project_root"),
                project_id=project_id,
                run_id=run_id,
                input_bundle=_required_task_value(task, "input_bundle"),
                output_dir=_required_task_value(task, "output_dir"),
                goal=_optional_task_value(task, "goal"),
                overwrite=strict_bool(task.get("overwrite", False), key="overwrite"),
            )
            run_result = TaskRunResult(
                state="succeeded",
                message="oled local demo runplan succeeded",
                output=result,
            )
        except Exception as exc:
            task_id = _job_task_id(job)
            run_result = TaskRunResult(
                state="failed",
                message=str(exc) or exc.__class__.__name__,
                output={"task_id": task_id},
            )
        job_id = str(job.get("job_id") or "").strip() if isinstance(job, dict) else ""
        if job_id:
            self._completed[job_id] = run_result
        return run_result

    def poll(self, job: dict[str, Any]) -> TaskRunResult:
        job_id = str(job.get("job_id") or "").strip() if isinstance(job, dict) else ""
        if job_id and job_id in self._completed:
            return self._completed[job_id]
        task_id = _job_task_id(job)
        return TaskRunResult(
            state="succeeded",
            message="oled local demo runplan already completed",
            output={"task_id": task_id},
        )

    def cancel(self, job: dict[str, Any]) -> TaskRunResult:
        return TaskRunResult(
            state="cancelled",
            message="oled local demo runplan cancellation acknowledged",
            output={"task_id": _job_task_id(job)},
        )


def execute_oled_local_demo_worker_queue(
    *,
    queue_root: Path | str,
    project_root: Path | str,
    project_id: str,
    run_id: str,
    input_bundle: Path | str,
    output_dir: Path | str,
    worker_id: str,
    goal: str | None = None,
    overwrite: bool = False,
    now: str = "",
) -> dict[str, Any]:
    """Enqueue and execute one OLED local demo job through the local worker queue."""
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
        created_at=now,
    )
    poller = WorkerQueuePoller(
        queue,
        worker_id=worker_id,
        runner=OLEDLocalDemoRunPlanWorkerTaskRunner(),
    )
    poll_result = poller.poll_once(now=now)
    job_id = str(job.get("job_id") or "")
    job_status = queue.status(job_id)
    lease = poll_result.active_lease
    if poll_result.action != "completed" or job_status is None or str(job_status.get("status") or "") != "succeeded":
        message = ""
        if poll_result.runner_result is not None:
            message = poll_result.runner_result.message
        if not message and job_status is not None:
            error = job_status.get("error")
            if isinstance(error, dict):
                message = str(error.get("reason") or "")
        raise ValueError(message or f"oled_local_demo_worker_queue_failed:{poll_result.action}")

    runner_output = (
        poll_result.runner_result.output
        if poll_result.runner_result is not None and isinstance(poll_result.runner_result.output, dict)
        else {}
    )
    artifacts = runner_output.get("artifacts")
    return {
        "ok": True,
        "queue_root": str(queue_root),
        "project_root": str(project_root),
        "project_id": project_id,
        "run_id": run_id,
        "worker_id": worker_id,
        "job_id": job_id,
        "job_status": str(job_status.get("status") or ""),
        "poll_action": poll_result.action,
        "lease_status": str(lease.get("status") or "") if isinstance(lease, dict) else "",
        "executed_tasks": list(runner_output.get("executed_tasks") or []),
        "output_dir": str(output_dir),
        "artifacts": list(artifacts) if isinstance(artifacts, list) else list(OUTPUT_ARTIFACT_IDS),
        "executable": True,
        "adapters_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one OLED local demo job through the local worker queue.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--now", default="")
    args = parser.parse_args(argv)

    result = execute_oled_local_demo_worker_queue(
        queue_root=args.queue_root,
        project_root=args.project_root,
        project_id=args.project_id,
        run_id=args.run_id,
        input_bundle=args.input_bundle,
        output_dir=args.output_dir,
        worker_id=args.worker_id,
        goal=args.goal,
        overwrite=args.overwrite,
        now=args.now,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _validate_job(job: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    project_id = str(job.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("project_id required")
    run_id = str(job.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_id required")
    task = job.get("task")
    if not isinstance(task, dict):
        raise ValueError("job task must be an object")
    task_id = str(task.get("task_id") or "").strip()
    if task_id != TASK_ID:
        raise ValueError(f"unsupported_oled_local_demo_worker_task:{task_id}")
    _required_task_value(task, "project_root")
    _required_task_value(task, "input_bundle")
    _required_task_value(task, "output_dir")
    return project_id, run_id, task


def _required_task_value(task: dict[str, Any], key: str) -> str:
    value = str(task.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} required")
    return value


def _optional_task_value(task: dict[str, Any], key: str) -> str | None:
    value = task.get(key)
    if value is None:
        return None
    clean = str(value).strip()
    return clean or None


def _job_task_id(job: dict[str, Any]) -> str:
    if not isinstance(job, dict):
        return TASK_ID
    task = job.get("task")
    if isinstance(task, dict):
        task_id = str(task.get("task_id") or "").strip()
        if task_id:
            return task_id
    return TASK_ID


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
