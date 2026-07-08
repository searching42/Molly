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


def inspect_oled_local_demo_generic_run_plan_jobs(
    *,
    queue_root: Path | str,
    project_root: Path | str | None = None,
    job_id: str | None = None,
    project_id: str | None = None,
    run_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Inspect generic OLED local demo run-plan queue jobs without mutation."""
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    leases = queue.list_leases()
    filters = {
        "job_id": _clean_optional(job_id),
        "project_id": _clean_optional(project_id),
        "run_id": _clean_optional(run_id),
        "status": _clean_optional(status),
    }
    jobs: list[dict[str, Any]] = []
    for job in queue.list_jobs():
        parsed_task = _allowlisted_task(job, selected_job_id=filters["job_id"])
        if parsed_task is None or not _matches_filters(job, filters):
            continue
        jobs.append(_summarize_job(job, parsed_task=parsed_task, leases=leases, project_root=project_root))
    counts: dict[str, int] = {}
    for item in jobs:
        job_status = str(item.get("status") or "")
        if job_status:
            counts[job_status] = counts.get(job_status, 0) + 1
    return {
        "ok": True,
        "queue_root": str(queue_root),
        "project_root": str(project_root) if project_root is not None else "",
        "filters": filters,
        "job_count": len(jobs),
        "status_counts": {key: counts[key] for key in sorted(counts)},
        "jobs": jobs,
        "generic_run_plan_queue": True,
        "allowlist": list(ALLOWLISTED_RUN_PLAN_TASKS),
        "executed": False,
        "executable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect generic OLED local demo run-plan queue jobs.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--project-root")
    parser.add_argument("--job-id")
    parser.add_argument("--project-id")
    parser.add_argument("--run-id")
    parser.add_argument("--status")
    args = parser.parse_args(argv)

    result = inspect_oled_local_demo_generic_run_plan_jobs(
        queue_root=args.queue_root,
        project_root=args.project_root,
        job_id=args.job_id,
        project_id=args.project_id,
        run_id=args.run_id,
        status=args.status,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _allowlisted_task(job: dict[str, Any], *, selected_job_id: str):
    task_payload = job.get("task")
    if not isinstance(task_payload, dict):
        return None
    if str(task_payload.get("task_id") or "") != RUN_PLAN_EXECUTE_TASK_ID:
        return None
    if str(task_payload.get("kind") or "") != RUN_PLAN_EXECUTE_KIND:
        return None
    try:
        task = validate_run_plan_execute_task(task_payload)
    except ValueError as exc:
        if selected_job_id and selected_job_id == str(job.get("job_id") or ""):
            raise ValueError("malformed_run_plan_execute_job") from exc
        return None
    task_ids = [planned.task_id for planned in task.run_plan.tasks]
    requested = list(task.run_plan.requested_tasks)
    if task_ids != list(ALLOWLISTED_RUN_PLAN_TASKS):
        return None
    if requested != list(ALLOWLISTED_RUN_PLAN_TASKS):
        return None
    return task


def _matches_filters(job: dict[str, Any], filters: dict[str, str]) -> bool:
    if filters["job_id"] and str(job.get("job_id") or "") != filters["job_id"]:
        return False
    if filters["project_id"] and str(job.get("project_id") or "") != filters["project_id"]:
        return False
    if filters["run_id"] and str(job.get("run_id") or "") != filters["run_id"]:
        return False
    if filters["status"] and str(job.get("status") or "") != filters["status"]:
        return False
    return True


def _summarize_job(
    job: dict[str, Any],
    *,
    parsed_task: Any,
    leases: list[dict[str, Any]],
    project_root: Path | str | None,
) -> dict[str, Any]:
    task_options = parsed_task.task_options.get(LOCAL_DEMO_TASK_ID)
    options = task_options if isinstance(task_options, dict) else {}
    summary: dict[str, Any] = {
        "job_id": str(job.get("job_id") or ""),
        "project_id": str(job.get("project_id") or ""),
        "run_id": str(job.get("run_id") or ""),
        "status": str(job.get("status") or ""),
        "attempts": int(job.get("attempts") or 0),
        "worker_id": str(job.get("worker_id") or ""),
        "lease_id": str(job.get("lease_id") or ""),
        "cancellation_requested": bool(job.get("cancellation_requested")),
        "retry_of_job_id": str(job.get("retry_of_job_id") or ""),
        "retry_root_job_id": str(job.get("retry_root_job_id") or ""),
        "retry_request_id": str(job.get("retry_request_id") or ""),
        "queue_task_id": parsed_task.task_id,
        "queue_task_kind": parsed_task.kind,
        "run_plan_tasks": [planned.task_id for planned in parsed_task.run_plan.tasks],
        "requested_tasks": list(parsed_task.run_plan.requested_tasks),
        "input_bundle": str(options.get("input_bundle") or ""),
        "output_dir": str(options.get("output_dir") or ""),
        "overwrite": bool(options.get("overwrite")),
        "lease": _matching_lease_summary(job, leases),
    }
    error = job.get("error")
    if isinstance(error, dict) and error:
        summary["error"] = {str(key): value for key, value in error.items()}
    result = job.get("result")
    if isinstance(result, dict):
        executed_tasks = result.get("executed_tasks")
        if isinstance(executed_tasks, list):
            summary["executed_tasks"] = [str(item) for item in executed_tasks]
    if project_root is not None:
        _add_project_metadata(summary, project_root=project_root)
    return summary


def _matching_lease_summary(job: dict[str, Any], leases: list[dict[str, Any]]) -> dict[str, Any]:
    job_id = str(job.get("job_id") or "")
    lease_id = str(job.get("lease_id") or "")
    candidates = [
        lease for lease in leases
        if str(lease.get("lease_id") or "") == lease_id or str(lease.get("job_id") or "") == job_id
    ]
    if not candidates:
        return {}
    lease = sorted(candidates, key=lambda item: (str(item.get("acquired_at") or ""), str(item.get("lease_id") or "")))[-1]
    return {
        "lease_id": str(lease.get("lease_id") or ""),
        "status": str(lease.get("status") or ""),
        "worker_id": str(lease.get("worker_id") or ""),
        "heartbeat_at": str(lease.get("heartbeat_at") or ""),
        "expires_at": str(lease.get("expires_at") or ""),
    }


def _add_project_metadata(summary: dict[str, Any], *, project_root: Path | str) -> None:
    run_path = (
        Path(project_root)
        / "projects"
        / str(summary.get("project_id") or "")
        / "runs"
        / str(summary.get("run_id") or "")
    )
    stage = _read_json_object(run_path / "stage.json")
    if stage:
        summary["stage_state"] = {
            "stage": str(stage.get("stage") or ""),
            "status": str(stage.get("status") or "").lower(),
            "updated_at": str(stage.get("updated_at") or ""),
        }
    registry = _read_json_object(run_path / "artifact_registry.json")
    artifacts = registry.get("artifacts")
    if isinstance(artifacts, dict):
        summary["artifact_ids"] = [str(key) for key in artifacts]


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _clean_optional(value: str | None) -> str:
    return str(value or "").strip()


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
