from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


TASK_ID = "execute_oled_local_demo_runplan"


def inspect_oled_local_demo_worker_jobs(
    *,
    queue_root: Path | str,
    project_root: Path | str | None = None,
    job_id: str | None = None,
    project_id: str | None = None,
    run_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Inspect OLED local demo worker jobs without mutating queue or run state."""
    queue = WorkerQueue(JsonWorkerQueueStore(queue_root))
    leases = queue.list_leases()
    filters = {
        "job_id": _clean_optional(job_id),
        "project_id": _clean_optional(project_id),
        "run_id": _clean_optional(run_id),
        "status": _clean_optional(status),
    }
    jobs = [
        _summarize_job(job, leases=leases, project_root=project_root)
        for job in queue.list_jobs()
        if _is_oled_local_demo_job(job) and _matches_filters(job, filters)
    ]
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
        "executed": False,
        "executable": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect OLED local demo worker jobs without mutating them.")
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--project-root")
    parser.add_argument("--job-id")
    parser.add_argument("--project-id")
    parser.add_argument("--run-id")
    parser.add_argument("--status")
    args = parser.parse_args(argv)

    result = inspect_oled_local_demo_worker_jobs(
        queue_root=args.queue_root,
        project_root=args.project_root,
        job_id=args.job_id,
        project_id=args.project_id,
        run_id=args.run_id,
        status=args.status,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def _summarize_job(
    job: dict[str, Any],
    *,
    leases: list[dict[str, Any]],
    project_root: Path | str | None,
) -> dict[str, Any]:
    task = job.get("task")
    task_payload = task if isinstance(task, dict) else {}
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
        "task_id": TASK_ID,
        "input_bundle": str(task_payload.get("input_bundle") or ""),
        "output_dir": str(task_payload.get("output_dir") or ""),
        "overwrite": bool(task_payload.get("overwrite")),
        "lease": _matching_lease_summary(job, leases),
    }
    error = job.get("error")
    if isinstance(error, dict) and error:
        summary["error"] = {str(key): value for key, value in error.items()}
    if project_root is not None:
        _add_project_storage_metadata(summary, project_root=project_root)
    return summary


def _is_oled_local_demo_job(job: dict[str, Any]) -> bool:
    task = job.get("task")
    return isinstance(task, dict) and str(task.get("task_id") or "").strip() == TASK_ID


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


def _add_project_storage_metadata(summary: dict[str, Any], *, project_root: Path | str) -> None:
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
